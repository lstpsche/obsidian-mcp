//! Darwin extended ACL and ownership enforcement for protected access state.
//!
//! The native declarations and values follow the SDK's sys/acl.h and
//! membership.h. BSD mode bits alone do not constrain an extended allow ACE.

use std::ffi::{c_int, c_uint, c_void};
use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::ptr;

use super::super::{AccessError, AccessResult};

type Acl = *mut c_void;
type Entry = *mut c_void;
type Flags = *mut c_void;

const ACL_TYPE_EXTENDED: c_uint = 0x0000_0100;
const ACL_EXTENDED_ALLOW: c_uint = 1;
const ACL_EXTENDED_DENY: c_uint = 2;
const ACL_FIRST_ENTRY: c_int = 0;
const ACL_NEXT_ENTRY: c_int = -1;
const ACL_ENTRY_FILE_INHERIT: c_uint = 1 << 5;
const ACL_ENTRY_DIRECTORY_INHERIT: c_uint = 1 << 6;
const ACL_ENTRY_ONLY_INHERIT: c_uint = 1 << 8;
const ACL_FLAG_NO_INHERIT: c_uint = 1 << 17;
const ID_TYPE_UID: c_int = 0;

const ACL_WRITE_DATA: u64 = 1 << 2;
const ACL_DELETE: u64 = 1 << 4;
const ACL_APPEND_DATA: u64 = 1 << 5;
const ACL_DELETE_CHILD: u64 = 1 << 6;
const ACL_WRITE_ATTRIBUTES: u64 = 1 << 8;
const ACL_WRITE_EXTATTRIBUTES: u64 = 1 << 10;
const ACL_WRITE_SECURITY: u64 = 1 << 12;
const ACL_CHANGE_OWNER: u64 = 1 << 13;
const MUTATION_RIGHTS: u64 = ACL_WRITE_DATA
    | ACL_DELETE
    | ACL_APPEND_DATA
    | ACL_DELETE_CHILD
    | ACL_WRITE_ATTRIBUTES
    | ACL_WRITE_EXTATTRIBUTES
    | ACL_WRITE_SECURITY
    | ACL_CHANGE_OWNER;

unsafe extern "C" {
    fn acl_get_fd_np(fd: c_int, acl_type: c_uint) -> Acl;
    fn acl_set_fd_np(fd: c_int, acl: Acl, acl_type: c_uint) -> c_int;
    fn acl_init(count: c_int) -> Acl;
    fn acl_free(object: *mut c_void) -> c_int;
    fn acl_valid(acl: Acl) -> c_int;
    fn acl_get_entry(acl: Acl, entry_id: c_int, entry: *mut Entry) -> c_int;
    fn acl_get_tag_type(entry: Entry, tag: *mut c_uint) -> c_int;
    fn acl_get_qualifier(entry: Entry) -> *mut c_void;
    fn acl_get_permset_mask_np(entry: Entry, mask: *mut u64) -> c_int;
    fn acl_get_flagset_np(object: *mut c_void, flags: *mut Flags) -> c_int;
    fn acl_get_flag_np(flags: Flags, flag: c_uint) -> c_int;
    fn acl_add_flag_np(flags: Flags, flag: c_uint) -> c_int;
    fn mbr_uuid_to_id(uuid: *const u8, id: *mut libc::id_t, id_type: *mut c_int) -> c_int;
}

struct NativeAllocation(*mut c_void);

impl Drop for NativeAllocation {
    fn drop(&mut self) {
        // SAFETY: acl_get_fd_np/acl_init/acl_get_qualifier allocate objects whose
        // ownership is transferred to the caller and released with acl_free.
        unsafe { acl_free(self.0) };
    }
}

/// Clear inherited entries on a newly created empty object before secret bytes
/// are written. Existing owner files and ancestor ACLs are never rewritten.
pub(super) fn protect(file: &File) -> AccessResult<()> {
    validate_volume(file)?;
    // SAFETY: acl_init returns one owned ACL allocation or NULL on error.
    let acl = unsafe { acl_init(0) };
    if acl.is_null() {
        return Err(io::Error::last_os_error().into());
    }
    let acl = NativeAllocation(acl);
    let mut flags = ptr::null_mut();
    // SAFETY: the allocation remains live until the fd update completes; flags
    // is a borrowed view into that ACL and is not freed separately.
    if unsafe { acl_get_flagset_np(acl.0, &mut flags) } != 0
        || unsafe { acl_add_flag_np(flags, ACL_FLAG_NO_INHERIT) } != 0
        || unsafe { acl_set_fd_np(file.as_raw_fd(), acl.0, ACL_TYPE_EXTENDED) } != 0
    {
        return Err(io::Error::last_os_error().into());
    }
    Ok(())
}

pub(super) fn validate(file: &File, private: bool) -> AccessResult<()> {
    validate_acl(file, private, false)
}

pub(super) fn validate_creation_parent(file: &File) -> AccessResult<()> {
    validate_acl(file, false, true)
}

fn validate_acl(file: &File, private: bool, creating: bool) -> AccessResult<()> {
    validate_volume(file)?;
    let Some(acl) = opened_acl(file)? else {
        return Ok(());
    };
    // SAFETY: the ACL is an owned allocation from the opened descriptor.
    if unsafe { acl_valid(acl.0) } != 0 {
        return Err(io::Error::last_os_error().into());
    }
    let mut position = ACL_FIRST_ENTRY;
    loop {
        let mut entry = ptr::null_mut();
        // Darwin returns 0 for an entry and -1/EINVAL after the last entry.
        // acl_valid above distinguishes a malformed ACL from normal exhaustion.
        if unsafe { acl_get_entry(acl.0, position, &mut entry) } != 0 {
            let error = io::Error::last_os_error();
            if error.raw_os_error() == Some(libc::EINVAL) {
                break;
            }
            return Err(error.into());
        }
        position = ACL_NEXT_ENTRY;
        let mut tag = 0;
        // SAFETY: entry is borrowed from this still-live validated ACL.
        if unsafe { acl_get_tag_type(entry, &mut tag) } != 0 {
            return Err(io::Error::last_os_error().into());
        }
        if tag == ACL_EXTENDED_DENY {
            continue;
        }
        if tag != ACL_EXTENDED_ALLOW {
            return Err(AccessError::Configuration(
                "access state has an unsupported extended ACL entry".into(),
            ));
        }
        let mut permissions = 0;
        if unsafe { acl_get_permset_mask_np(entry, &mut permissions) } != 0 {
            return Err(io::Error::last_os_error().into());
        }
        if permissions == 0 {
            continue;
        }
        if !private {
            let mut flags = ptr::null_mut();
            if unsafe { acl_get_flagset_np(entry, &mut flags) } != 0 {
                return Err(io::Error::last_os_error().into());
            }
            let only_inherited = entry_flag(flags, ACL_ENTRY_ONLY_INHERIT)?;
            let inheritable = entry_flag(flags, ACL_ENTRY_FILE_INHERIT)?
                || entry_flag(flags, ACL_ENTRY_DIRECTORY_INHERIT)?;
            // A broad inherited grant could let another process retain a handle
            // during child creation, before its empty ACL can be tightened.
            if !(creating && inheritable) && (only_inherited || permissions & MUTATION_RIGHTS == 0)
            {
                continue;
            }
        }
        let qualifier = unsafe { acl_get_qualifier(entry) };
        if qualifier.is_null() {
            return Err(io::Error::last_os_error().into());
        }
        let qualifier = NativeAllocation(qualifier);
        let mut id = 0;
        let mut id_type = -1;
        // Darwin qualifiers are guid_t (16 bytes). Membership resolves both
        // record UUIDs and synthesized UID UUIDs without treating a group with
        // a matching numeric ID as the owner or root user.
        let result = unsafe { mbr_uuid_to_id(qualifier.0.cast(), &mut id, &mut id_type) };
        if result != 0 {
            return Err(io::Error::from_raw_os_error(result).into());
        }
        let owner = unsafe { libc::geteuid() };
        if id_type != ID_TYPE_UID || (id != owner && id != 0) {
            return Err(AccessError::Configuration(
                "access state has an extended ACL granting another principal access; use a private owner-controlled directory".into(),
            ));
        }
    }
    Ok(())
}

fn entry_flag(flags: Flags, flag: c_uint) -> AccessResult<bool> {
    let result = unsafe { acl_get_flag_np(flags, flag) };
    if result < 0 {
        return Err(io::Error::last_os_error().into());
    }
    Ok(result != 0)
}

fn opened_acl(file: &File) -> AccessResult<Option<NativeAllocation>> {
    // SAFETY: the descriptor belongs to a retained, validated file/directory.
    let acl = unsafe { acl_get_fd_np(file.as_raw_fd(), ACL_TYPE_EXTENDED) };
    if !acl.is_null() {
        return Ok(Some(NativeAllocation(acl)));
    }
    let error = io::Error::last_os_error();
    // An opened Darwin object without an extended ACL returns ENOENT. This is
    // the absence of that ACL, not an unavailable path or an I/O fallback.
    if error.raw_os_error() == Some(libc::ENOENT) {
        Ok(None)
    } else {
        Err(error.into())
    }
}

fn validate_volume(file: &File) -> AccessResult<()> {
    let mut information = std::mem::MaybeUninit::<libc::statfs>::uninit();
    // SAFETY: fstatfs initializes this SDK-defined struct on a successful call.
    if unsafe { libc::fstatfs(file.as_raw_fd(), information.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error().into());
    }
    let information = unsafe { information.assume_init() };
    validate_mount_flags(information.f_flags)
}

fn validate_mount_flags(flags: u32) -> AccessResult<()> {
    if flags & libc::MNT_IGNORE_OWNERSHIP as u32 != 0 {
        return Err(AccessError::Configuration(
            "access state requires a volume that enforces ownership; choose a protected data directory or enable ownership on that volume".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    use std::path::Path;
    use std::process::Command;

    fn add_acl(path: &Path, entry: &str) {
        let output = Command::new("/bin/chmod")
            .args(["+a", entry])
            .arg(path)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "chmod failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn owner_name() -> String {
        let output = Command::new("/usr/bin/id").args(["-un"]).output().unwrap();
        assert!(output.status.success());
        String::from_utf8(output.stdout).unwrap().trim().to_owned()
    }

    fn private_file(path: &Path) -> File {
        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)
            .unwrap()
    }

    #[test]
    fn macos_acl_rejects_private_file_allow_even_with_owner_only_mode() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("token");
        let file = private_file(&path);
        add_acl(&path, "everyone allow read");
        assert_eq!(file.metadata().unwrap().permissions().mode() & 0o777, 0o600);
        assert!(validate(&file, true).is_err());
    }

    #[test]
    fn macos_acl_rejects_private_directory_allow_even_with_owner_only_mode() {
        let temporary = tempfile::tempdir().unwrap();
        std::fs::set_permissions(temporary.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        add_acl(temporary.path(), "everyone allow list,search");
        let directory = File::open(temporary.path()).unwrap();
        assert_eq!(
            directory.metadata().unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert!(validate(&directory, true).is_err());
    }

    #[test]
    fn macos_acl_rejects_ancestor_mutation_grants_but_preserves_deny_delete() {
        let temporary = tempfile::tempdir().unwrap();
        let directory = File::open(temporary.path()).unwrap();
        add_acl(temporary.path(), "everyone deny delete");
        assert!(validate(&directory, false).is_ok());
        assert!(validate(&directory, true).is_ok());
        add_acl(temporary.path(), "everyone allow read,search");
        assert!(validate(&directory, false).is_ok());
        add_acl(
            temporary.path(),
            "everyone allow delete_child,writesecurity",
        );
        assert!(validate(&directory, false).is_err());
    }

    #[test]
    fn macos_acl_trusts_owner_and_root_users_but_not_same_numeric_group() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("token");
        let file = private_file(&path);
        add_acl(&path, &format!("user:{} allow read", owner_name()));
        add_acl(&path, "user:root allow read");
        assert!(validate(&file, true).is_ok());
        add_acl(&path, "group:wheel allow read");
        assert!(validate(&file, true).is_err());
    }

    #[test]
    fn macos_acl_protect_clears_inherited_allow_before_private_content() {
        let temporary = tempfile::tempdir().unwrap();
        add_acl(
            temporary.path(),
            "everyone allow read,file_inherit,directory_inherit,only_inherit",
        );
        let path = temporary.path().join("token");
        let file = private_file(&path);
        assert!(validate(&file, true).is_err());
        protect(&file).unwrap();
        assert!(validate(&file, true).is_ok());
        assert!(file.metadata().unwrap().len() == 0);
    }

    #[test]
    fn macos_acl_creation_protects_new_directories_and_files() {
        use super::super::super::create_protected_directory;
        use super::super::ProtectedDirectory;
        let temporary = tempfile::tempdir().unwrap();
        add_acl(
            temporary.path(),
            &format!(
                "user:{} allow read,search,file_inherit,directory_inherit,only_inherit",
                owner_name()
            ),
        );
        let private = temporary.path().join("private");
        create_protected_directory(&private).unwrap();
        let directory = ProtectedDirectory::open(&private, false).unwrap();
        // A trusted inherited owner entry verifies that create also clears the
        // file ACL, even though that entry does not overgrant the parent.
        add_acl(
            &private,
            &format!("user:{} allow read,file_inherit,only_inherit", owner_name()),
        );
        directory
            .create(std::ffi::OsStr::new("token"), b"synthetic-secret")
            .unwrap();
        let file = File::open(private.join("token")).unwrap();
        validate(&file, true).unwrap();
        if let Some(acl) = opened_acl(&file).unwrap() {
            let mut entry = ptr::null_mut();
            assert_ne!(
                unsafe { acl_get_entry(acl.0, ACL_FIRST_ENTRY, &mut entry) },
                0
            );
        }
    }

    #[test]
    fn macos_acl_rejects_unsafe_inheritance_before_directory_creation() {
        use super::super::super::create_protected_directory;
        let temporary = tempfile::tempdir().unwrap();
        add_acl(
            temporary.path(),
            "everyone allow read,search,file_inherit,directory_inherit,only_inherit",
        );
        let parent = File::open(temporary.path()).unwrap();
        validate(&parent, false).unwrap();
        assert!(validate_creation_parent(&parent).is_err());
        let child = temporary.path().join("private");
        assert!(create_protected_directory(&child).is_err());
        assert!(!child.exists());
    }

    #[test]
    fn macos_acl_rejects_ownership_ignoring_mount_flags() {
        assert!(validate_mount_flags(libc::MNT_IGNORE_OWNERSHIP as u32).is_err());
        let temporary = tempfile::tempfile().unwrap();
        validate_volume(&temporary).unwrap();
    }
}
