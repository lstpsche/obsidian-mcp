//! Windows directory operations that retain the kernel directory handle.
//! This also preserves extended UNC roots without translating them to DOS paths.

use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::io;
use std::mem::{offset_of, size_of, size_of_val};
use std::os::windows::ffi::{OsStrExt, OsStringExt};
use std::os::windows::fs::MetadataExt;
use std::os::windows::io::{AsRawHandle, FromRawHandle};
use std::ptr;

use cap_std::fs::Dir;
use windows_sys::Wdk::Foundation::OBJECT_ATTRIBUTES;
use windows_sys::Wdk::Storage::FileSystem::{
    FILE_CREATE, FILE_DIRECTORY_FILE, FILE_OPEN, FILE_OPEN_REPARSE_POINT,
    FILE_RENAME_POSIX_SEMANTICS, FILE_RENAME_REPLACE_IF_EXISTS, FILE_SYNCHRONOUS_IO_NONALERT,
    NtCreateFile,
};
use windows_sys::Win32::Foundation::{
    ERROR_NO_MORE_FILES, OBJ_CASE_INSENSITIVE, RtlNtStatusToDosError, UNICODE_STRING,
};
use windows_sys::Win32::Storage::FileSystem::{
    DELETE, FILE_ATTRIBUTE_DIRECTORY, FILE_ATTRIBUTE_NORMAL, FILE_ATTRIBUTE_REPARSE_POINT,
    FILE_DISPOSITION_INFO, FILE_FULL_DIR_INFO, FILE_LIST_DIRECTORY, FILE_READ_ATTRIBUTES,
    FILE_RENAME_INFO, FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE, FileDispositionInfo,
    FileFullDirectoryInfo, FileFullDirectoryRestartInfo, FileRenameInfo, FileRenameInfoEx,
    GetFileInformationByHandleEx, SYNCHRONIZE, SetFileInformationByHandle,
};
use windows_sys::Win32::System::IO::IO_STATUS_BLOCK;

/// Open a single native name against its pinned parent. An empty name reopens
/// the directory itself with an independent enumeration cursor.
fn open_relative(
    parent: &Dir,
    name: &OsStr,
    access: u32,
    disposition: u32,
    options: u32,
) -> io::Result<File> {
    let mut wide: Vec<u16> = name.encode_wide().collect();
    let length = u16::try_from(size_of_val(wide.as_slice())).map_err(io::Error::other)?;
    wide.push(0);
    let string = UNICODE_STRING {
        Length: length,
        MaximumLength: length,
        Buffer: wide.as_mut_ptr(),
    };
    let attributes = OBJECT_ATTRIBUTES {
        Length: size_of::<OBJECT_ATTRIBUTES>() as u32,
        RootDirectory: parent.as_raw_handle(),
        ObjectName: &string,
        Attributes: OBJ_CASE_INSENSITIVE,
        ..Default::default()
    };
    let mut handle = ptr::null_mut();
    let mut result = IO_STATUS_BLOCK::default();
    // Synchronous directory queries require SYNCHRONIZE on the new handle.
    // OPEN_REPARSE_POINT makes deletion and rename apply to the entry itself.
    let status = unsafe {
        NtCreateFile(
            &mut handle,
            access | SYNCHRONIZE,
            &attributes,
            &mut result,
            ptr::null(),
            FILE_ATTRIBUTE_NORMAL,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            disposition,
            options | FILE_OPEN_REPARSE_POINT | FILE_SYNCHRONOUS_IO_NONALERT,
            ptr::null(),
            0,
        )
    };
    if status < 0 {
        return Err(io::Error::from_raw_os_error(
            unsafe { RtlNtStatusToDosError(status) } as i32,
        ));
    }
    // SAFETY: a successful synchronous NtCreateFile returned one owned handle.
    Ok(unsafe { File::from_raw_handle(handle) })
}

pub(super) fn create_child_dir(parent: &Dir, name: &OsStr) -> io::Result<()> {
    open_relative(
        parent,
        name,
        FILE_READ_ATTRIBUTES,
        FILE_CREATE,
        FILE_DIRECTORY_FILE,
    )
    .map(drop)
}

fn open_entry_for_delete(parent: &Dir, name: &OsStr) -> io::Result<File> {
    open_relative(parent, name, DELETE | FILE_READ_ATTRIBUTES, FILE_OPEN, 0)
}

pub(super) fn remove_entry(dir: &Dir, name: &OsStr, directory: bool) -> io::Result<()> {
    let entry = open_entry_for_delete(dir, name)?;
    let attrs = entry.metadata()?.file_attributes();
    if (attrs & FILE_ATTRIBUTE_DIRECTORY != 0 && attrs & FILE_ATTRIBUTE_REPARSE_POINT == 0)
        != directory
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "directory entry changed type before removal",
        ));
    }
    let info = FILE_DISPOSITION_INFO { DeleteFile: true };
    // Delete the opened entry itself, including junctions. The kernel refuses
    // a nonempty directory, and no leaf pathname is followed a second time.
    if unsafe {
        SetFileInformationByHandle(
            entry.as_raw_handle(),
            FileDispositionInfo,
            ptr::from_ref(&info).cast(),
            size_of_val(&info) as u32,
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

pub(super) fn rename(
    source: &Dir,
    old: &OsStr,
    destination: &Dir,
    new: &OsStr,
    replace: bool,
) -> io::Result<()> {
    let file = open_entry_for_delete(source, old)?;
    let attrs = file.metadata()?.file_attributes();
    if attrs & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT) == FILE_ATTRIBUTE_DIRECTORY
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "note rename requires a file",
        ));
    }
    let name: Vec<u16> = new.encode_wide().collect();
    let name_bytes = u32::try_from(size_of_val(name.as_slice())).map_err(io::Error::other)?;
    let bytes = size_of::<FILE_RENAME_INFO>() + name_bytes as usize;
    let size = u32::try_from(bytes).map_err(io::Error::other)?;
    let mut words = vec![0usize; bytes.div_ceil(size_of::<usize>())];
    let info = words.as_mut_ptr().cast::<FILE_RENAME_INFO>();
    // The aligned allocation includes the variable UTF-16 tail. For checkpoint
    // replacement the old readers retain their old file handle.
    unsafe {
        (*info).RootDirectory = destination.as_raw_handle();
        (*info).FileNameLength = name_bytes;
        if replace {
            (*info).Anonymous.Flags = FILE_RENAME_REPLACE_IF_EXISTS | FILE_RENAME_POSIX_SEMANTICS;
        }
        ptr::copy_nonoverlapping(
            name.as_ptr(),
            ptr::addr_of_mut!((*info).FileName).cast(),
            name.len(),
        );
        let class = if replace {
            FileRenameInfoEx
        } else {
            FileRenameInfo
        };
        if SetFileInformationByHandle(file.as_raw_handle(), class, info.cast(), size) == 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

pub(crate) struct DirectoryEntry {
    name: OsString,
    attributes: u32,
}

impl DirectoryEntry {
    pub(crate) fn file_name(&self) -> OsString {
        self.name.clone()
    }

    pub(crate) fn file_type(&self) -> io::Result<EntryType> {
        Ok(EntryType(self.attributes))
    }
}

pub(crate) struct EntryType(u32);

impl EntryType {
    pub(crate) fn is_dir(&self) -> bool {
        self.0 & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)
            == FILE_ATTRIBUTE_DIRECTORY
    }

    pub(crate) fn is_file(&self) -> bool {
        self.0 & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT) == 0
    }
}

pub(crate) fn dir_entries(
    dir: &Dir,
) -> io::Result<impl Iterator<Item = io::Result<DirectoryEntry>>> {
    let file = open_relative(
        dir,
        OsStr::new(""),
        FILE_LIST_DIRECTORY,
        FILE_OPEN,
        FILE_DIRECTORY_FILE,
    )
    .map_err(|e| io::Error::new(e.kind(), format!("open directory for enumeration: {e}")))?;
    let mut buffer = vec![0u64; 8192];
    let mut entries = Vec::new().into_iter();
    let mut restart = true;
    let mut done = false;
    Ok(std::iter::from_fn(move || {
        loop {
            if let Some(entry) = entries.next() {
                return Some(Ok(entry));
            }
            if done {
                return None;
            }
            let class = if restart {
                FileFullDirectoryRestartInfo
            } else {
                FileFullDirectoryInfo
            };
            restart = false;
            // FILE_FULL_DIR_INFO records require an 8-byte-aligned output buffer.
            if unsafe {
                GetFileInformationByHandleEx(
                    file.as_raw_handle(),
                    class,
                    buffer.as_mut_ptr().cast(),
                    size_of_val(buffer.as_slice()) as u32,
                )
            } == 0
            {
                done = true;
                let err = io::Error::last_os_error();
                return (err.raw_os_error() != Some(ERROR_NO_MORE_FILES as i32)).then(|| {
                    Err(io::Error::new(
                        err.kind(),
                        format!("read directory entries: {err}"),
                    ))
                });
            }
            match directory_page(&buffer) {
                Ok(page) => entries = page.into_iter(),
                Err(err) => {
                    done = true;
                    return Some(Err(err));
                }
            }
        }
    }))
}

fn directory_page(buffer: &[u64]) -> io::Result<Vec<DirectoryEntry>> {
    let bad_record = || {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid native directory record",
        )
    };
    // Only the validated, initialized-length fields and filename of each
    // record are consumed. The OS returns byte offsets for chained records.
    let mut bytes =
        unsafe { std::slice::from_raw_parts(buffer.as_ptr().cast::<u8>(), size_of_val(buffer)) };
    let mut entries = Vec::new();
    loop {
        let field = |offset: usize| -> io::Result<u32> {
            let value = bytes.get(offset..offset + 4).ok_or_else(bad_record)?;
            Ok(u32::from_ne_bytes(
                value.try_into().map_err(|_| bad_record())?,
            ))
        };
        let next = field(offset_of!(FILE_FULL_DIR_INFO, NextEntryOffset))? as usize;
        let attributes = field(offset_of!(FILE_FULL_DIR_INFO, FileAttributes))?;
        let length = field(offset_of!(FILE_FULL_DIR_INFO, FileNameLength))? as usize;
        let start = offset_of!(FILE_FULL_DIR_INFO, FileName);
        let end = start.checked_add(length).ok_or_else(bad_record)?;
        if length == 0 || !length.is_multiple_of(2) || next != 0 && next < end {
            return Err(bad_record());
        }
        let wide: Vec<_> = bytes
            .get(start..end)
            .ok_or_else(bad_record)?
            .as_chunks::<2>()
            .0
            .iter()
            .copied()
            .map(u16::from_ne_bytes)
            .collect();
        let name = OsString::from_wide(&wide);
        if name != "." && name != ".." {
            entries.push(DirectoryEntry { name, attributes });
        }
        if next == 0 {
            return Ok(entries);
        }
        bytes = bytes.get(next..).ok_or_else(bad_record)?;
    }
}
