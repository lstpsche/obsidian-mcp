#include <errno.h>
#include <membership.h>
#include <sys/acl.h>
#include <sys/types.h>

// Darwin exposes extended grants separately from the POSIX mode bits. A
// writable grant inherited by future children is relevant to an IPC path too.
static int foreign_write_grant(acl_t acl, uid_t owner) {
    const acl_permset_mask_t write = ACL_WRITE_DATA | ACL_APPEND_DATA |
        ACL_DELETE | ACL_DELETE_CHILD | ACL_WRITE_ATTRIBUTES |
        ACL_WRITE_EXTATTRIBUTES | ACL_WRITE_SECURITY | ACL_CHANGE_OWNER;
    int which = ACL_FIRST_ENTRY;
    acl_entry_t entry;
    while (acl_get_entry(acl, which, &entry) == 0) {
        which = ACL_NEXT_ENTRY;
        acl_tag_t tag;
        acl_permset_mask_t mask;
        if (acl_get_tag_type(entry, &tag) || acl_get_permset_mask_np(entry, &mask))
            return -1;
        if (tag != ACL_EXTENDED_ALLOW || !(mask & write))
            continue;

        uuid_t *grantee = acl_get_qualifier(entry);
        if (!grantee)
            return -1;
        id_t id;
        int type;
        int error = mbr_uuid_to_id(*grantee, &id, &type);
        acl_free(grantee);
        if (error) {
            errno = error;
            return -1;
        }
        if (type != ID_TYPE_UID || (id != 0 && id != owner))
            return 1;
    }
    // acl_get_entry reports EINVAL at the end of a valid Darwin ACL.
    return errno == EINVAL ? 0 : -1;
}

int obsidian_ipc_foreign_write_acl(int fd, uid_t owner) {
    acl_t acl = acl_get_fd_np(fd, ACL_TYPE_EXTENDED);
    if (!acl)
        return errno == ENOENT ? 0 : -1;
    int result = foreign_write_grant(acl, owner);
    int error = errno;
    acl_free(acl);
    errno = error;
    return result;
}
