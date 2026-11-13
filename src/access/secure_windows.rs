//! Native Windows ownership and DACL validation for private access state.

use std::ffi::c_void;
use std::fs::File;
use std::io;
use std::mem::{offset_of, size_of};
use std::os::windows::io::{AsRawHandle, FromRawHandle};
use std::ptr;

use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE, LocalFree};
use windows_sys::Win32::Security::Authorization::{
    ConvertSidToStringSidW, ConvertStringSecurityDescriptorToSecurityDescriptorW, GetSecurityInfo,
    SE_FILE_OBJECT, SetSecurityInfo,
};
use windows_sys::Win32::Security::{
    ACCESS_ALLOWED_ACE, ACE_HEADER, ACL, ACL_SIZE_INFORMATION, AclSizeInformation,
    CONTAINER_INHERIT_ACE, DACL_SECURITY_INFORMATION, EqualSid, GetAce, GetAclInformation,
    GetSecurityDescriptorControl, GetSecurityDescriptorDacl, GetTokenInformation, INHERIT_ONLY_ACE,
    IsWellKnownSid, OBJECT_INHERIT_ACE, OWNER_SECURITY_INFORMATION,
    PROTECTED_DACL_SECURITY_INFORMATION, PSECURITY_DESCRIPTOR, PSID, SE_DACL_PROTECTED,
    TOKEN_QUERY, TOKEN_USER, TokenUser, WinBuiltinAdministratorsSid, WinCreatorOwnerSid,
    WinLocalSystemSid,
};
use windows_sys::Win32::Storage::FileSystem::{
    BY_HANDLE_FILE_INFORMATION, DELETE, FILE_ATTRIBUTE_REPARSE_POINT, FILE_DELETE_CHILD,
    FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT, FILE_READ_ATTRIBUTES,
    FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE, FILE_WRITE_ATTRIBUTES,
    GetFileInformationByHandle, READ_CONTROL, ReOpenFile, WRITE_DAC, WRITE_OWNER,
};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

use super::super::{AccessError, AccessResult};

const ALLOWED_ACE: u8 = 0;
const DENIED_ACE: u8 = 1;
const GENERIC_WRITE: u32 = 0x4000_0000;
const GENERIC_ALL: u32 = 0x1000_0000;

struct LocalAllocation(PSECURITY_DESCRIPTOR);
impl Drop for LocalAllocation {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { LocalFree(self.0) };
        }
    }
}

struct TokenHandle(windows_sys::Win32::Foundation::HANDLE);
impl Drop for TokenHandle {
    fn drop(&mut self) {
        unsafe { CloseHandle(self.0) };
    }
}

fn user_token() -> AccessResult<Vec<usize>> {
    let mut token = ptr::null_mut();
    if unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) } == 0 {
        return Err(io::Error::last_os_error().into());
    }
    let token = TokenHandle(token);
    let mut bytes = 0;
    unsafe { GetTokenInformation(token.0, TokenUser, ptr::null_mut(), 0, &mut bytes) };
    if bytes == 0 {
        return Err(io::Error::last_os_error().into());
    }
    let mut buffer = vec![0usize; (bytes as usize).div_ceil(size_of::<usize>())];
    if unsafe {
        GetTokenInformation(
            token.0,
            TokenUser,
            buffer.as_mut_ptr().cast(),
            bytes,
            &mut bytes,
        )
    } == 0
    {
        return Err(io::Error::last_os_error().into());
    }
    Ok(buffer)
}

fn user_sid(buffer: &[usize]) -> PSID {
    // The aligned allocation is filled by GetTokenInformation with TOKEN_USER.
    unsafe { (*(buffer.as_ptr().cast::<TOKEN_USER>())).User.Sid }
}

fn trusted(sid: PSID, owner: PSID) -> bool {
    unsafe {
        EqualSid(sid, owner) != 0
            || IsWellKnownSid(sid, WinLocalSystemSid) != 0
            || IsWellKnownSid(sid, WinBuiltinAdministratorsSid) != 0
    }
}

pub(super) fn protect(file: &File, directory: bool) -> AccessResult<()> {
    let user = user_token()?;
    let mut sid_text = ptr::null_mut();
    if unsafe { ConvertSidToStringSidW(user_sid(&user), &mut sid_text) } == 0 {
        return Err(io::Error::last_os_error().into());
    }
    let mut length = 0;
    unsafe {
        while *sid_text.add(length) != 0 {
            length += 1;
        }
    }
    let sid = String::from_utf16(unsafe { std::slice::from_raw_parts(sid_text, length) })
        .map_err(|_| AccessError::Configuration("invalid owner SID".into()));
    unsafe { LocalFree(sid_text.cast()) };
    let inheritance = if directory { "OICI" } else { "" };
    let descriptor = format!(
        "D:P(A;{inheritance};FA;;;SY)(A;{inheritance};FA;;;BA)(A;{inheritance};FA;;;{})",
        sid?
    );
    let descriptor: Vec<u16> = descriptor.encode_utf16().chain(Some(0)).collect();
    let mut security = ptr::null_mut();
    if unsafe {
        ConvertStringSecurityDescriptorToSecurityDescriptorW(
            descriptor.as_ptr(),
            1,
            &mut security,
            ptr::null_mut(),
        )
    } == 0
    {
        return Err(io::Error::last_os_error().into());
    }
    let security = LocalAllocation(security);
    let mut present = 0;
    let mut defaulted = 0;
    let mut acl = ptr::null_mut();
    if unsafe { GetSecurityDescriptorDacl(security.0, &mut present, &mut acl, &mut defaulted) } == 0
        || present == 0
        || acl.is_null()
    {
        return Err(io::Error::last_os_error().into());
    }
    // ReOpenFile preserves object identity and obtains WRITE_DAC without an
    // ambient pathname reopen. The object owner may set its protected DACL.
    let handle = unsafe {
        ReOpenFile(
            file.as_raw_handle().cast(),
            READ_CONTROL | WRITE_DAC | FILE_READ_ATTRIBUTES,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT,
        )
    };
    if handle == INVALID_HANDLE_VALUE {
        return Err(io::Error::last_os_error().into());
    }
    let handle = unsafe { File::from_raw_handle(handle.cast()) };
    let result = unsafe {
        SetSecurityInfo(
            handle.as_raw_handle().cast(),
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
            ptr::null_mut(),
            ptr::null_mut(),
            acl,
            ptr::null(),
        )
    };
    if result != 0 {
        return Err(io::Error::from_raw_os_error(result as i32).into());
    }
    Ok(())
}

pub(super) fn validate(file: &File, private: bool, directory: bool) -> AccessResult<()> {
    validate_acl(file, private, directory, false)
}

pub(super) fn validate_creation_parent(file: &File) -> AccessResult<()> {
    validate_acl(file, false, true, true)
}

fn validate_acl(file: &File, private: bool, directory: bool, creating: bool) -> AccessResult<()> {
    let info = information(file)?;
    if info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT != 0
        || (!directory && info.nNumberOfLinks != 1)
    {
        return Err(AccessError::Configuration(
            "access state cannot use reparse points or hard links".into(),
        ));
    }
    let user = user_token()?;
    let user_sid = user_sid(&user);
    let mut owner = ptr::null_mut();
    let mut acl: *mut ACL = ptr::null_mut();
    let mut descriptor = ptr::null_mut();
    let result = unsafe {
        GetSecurityInfo(
            file.as_raw_handle().cast(),
            SE_FILE_OBJECT,
            OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
            &mut owner,
            ptr::null_mut(),
            &mut acl,
            ptr::null_mut(),
            &mut descriptor,
        )
    };
    if result != 0 {
        return Err(io::Error::from_raw_os_error(result as i32).into());
    }
    let descriptor = LocalAllocation(descriptor);
    if owner.is_null()
        || acl.is_null()
        || !trusted(owner, user_sid)
        || (private && unsafe { EqualSid(owner, user_sid) } == 0)
    {
        return Err(AccessError::Configuration(
            "access state must be owned by the service owner with a private DACL".into(),
        ));
    }
    let mut control = 0;
    let mut revision = 0;
    if unsafe { GetSecurityDescriptorControl(descriptor.0, &mut control, &mut revision) } == 0 {
        return Err(io::Error::last_os_error().into());
    }
    if private && control & SE_DACL_PROTECTED == 0 {
        return Err(AccessError::Configuration(
            "access state requires a protected DACL".into(),
        ));
    }
    let mut size = ACL_SIZE_INFORMATION::default();
    if unsafe {
        GetAclInformation(
            acl,
            (&mut size as *mut ACL_SIZE_INFORMATION).cast(),
            size_of::<ACL_SIZE_INFORMATION>() as u32,
            AclSizeInformation,
        )
    } == 0
    {
        return Err(io::Error::last_os_error().into());
    }
    // Ancestors may permit new children (for example C:\), but cannot let
    // another principal replace an existing child or change its authority.
    let write_mask = GENERIC_ALL
        | GENERIC_WRITE
        | WRITE_DAC
        | WRITE_OWNER
        | DELETE
        | FILE_WRITE_ATTRIBUTES
        | FILE_DELETE_CHILD;
    for index in 0..size.AceCount {
        let mut raw: *mut c_void = ptr::null_mut();
        if unsafe { GetAce(acl, index, &mut raw) } == 0 {
            return Err(io::Error::last_os_error().into());
        }
        let header = unsafe { &*raw.cast::<ACE_HEADER>() };
        if header.AceType == DENIED_ACE {
            continue;
        }
        if header.AceType != ALLOWED_ACE {
            return Err(AccessError::Configuration(
                "unsupported access-state DACL entry".into(),
            ));
        }
        if usize::from(header.AceSize) < size_of::<ACCESS_ALLOWED_ACE>() {
            return Err(AccessError::Configuration(
                "invalid access-state DACL entry".into(),
            ));
        }
        let ace = unsafe { &*raw.cast::<ACCESS_ALLOWED_ACE>() };
        let sid = unsafe {
            raw.cast::<u8>()
                .add(offset_of!(ACCESS_ALLOWED_ACE, SidStart))
                .cast::<c_void>()
        };
        let flags = u32::from(header.AceFlags);
        let inherit_only = flags & INHERIT_ONLY_ACE != 0;
        let inheritable = flags & (OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE) != 0;
        // CREATOR OWNER is resolved to our newly created child's owner. It is
        // trusted only as an inherit-only placeholder, never as a group grant.
        let inherited_owner =
            inherit_only && inheritable && unsafe { IsWellKnownSid(sid, WinCreatorOwnerSid) } != 0;
        if ace.Mask != 0
            && !trusted(sid, user_sid)
            && !inherited_owner
            && (private
                || (creating && inheritable)
                || (!inherit_only && ace.Mask & write_mask != 0))
        {
            return Err(AccessError::Configuration("access state grants another account access; use a private owner-controlled directory".into()));
        }
    }
    Ok(())
}

fn information(file: &File) -> AccessResult<BY_HANDLE_FILE_INFORMATION> {
    let mut info = BY_HANDLE_FILE_INFORMATION::default();
    if unsafe { GetFileInformationByHandle(file.as_raw_handle().cast(), &mut info) } == 0 {
        return Err(io::Error::last_os_error().into());
    }
    Ok(info)
}

pub(super) fn object_identity(file: &File) -> AccessResult<String> {
    let info = information(file)?;
    Ok(format!(
        "windows:{}:{}:{}",
        info.dwVolumeSerialNumber, info.nFileIndexHigh, info.nFileIndexLow
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::process::Command;

    fn directory_handle(path: &Path) -> File {
        cap_std::fs::Dir::open_ambient_dir(path, cap_std::ambient_authority())
            .unwrap()
            .into_std_file()
    }

    fn grant_untrusted_inheritance(path: &Path) {
        let output = Command::new("icacls.exe")
            .arg(path)
            .args(["/grant", "*S-1-1-0:(OI)(CI)(IO)(F)"])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "icacls failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn windows_acl_rejects_inherit_only_untrusted_grants_on_private_parent() {
        let temporary = tempfile::tempdir().unwrap();
        let directory = directory_handle(temporary.path());
        protect(&directory, true).unwrap();
        grant_untrusted_inheritance(temporary.path());
        assert!(validate(&directory, true, true).is_err());
        // It does not authorize this existing ancestor, but must prevent new
        // children from inheriting a transient broadly accessible handle.
        validate(&directory, false, true).unwrap();
        assert!(validate_creation_parent(&directory).is_err());
    }

    #[test]
    fn windows_acl_unsafe_inheritance_prevents_file_and_directory_creation() {
        use super::super::super::{create_protected_directory, write_protected_file};
        let temporary = tempfile::tempdir().unwrap();
        let directory = directory_handle(temporary.path());
        protect(&directory, true).unwrap();
        grant_untrusted_inheritance(temporary.path());
        let token = temporary.path().join("token");
        assert!(write_protected_file(&token, b"synthetic-secret").is_err());
        assert!(!token.exists());
        let child = temporary.path().join("private");
        assert!(create_protected_directory(&child).is_err());
        assert!(!child.exists());
    }
}
