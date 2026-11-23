/*--------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  The syncs that order a file's bytes before its rename, the rename, and the
  syncs that order the rename before later changes to the file system.
  --------------------------------------------------------------------------*/

#define _GNU_SOURCE

#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/osdeps.h>
#include <caml/signals.h>
#include <caml/threads.h>
#include <caml/unixsupport.h>

/* caml/unixsupport.h declares the unix library's functions without
   CAMLextern: a stub DLL loaded at run time on Windows would call them
   directly, which flexdll links within 2 GiB only. Declared again with
   CAMLextern, each call goes through a pointer. */
#ifdef _WIN32
CAMLnoret CAMLextern void caml_unix_error(int errcode, const char *cmdname,
                                         value arg);
CAMLextern void caml_unix_check_path(value path, const char *cmdname);
CAMLextern void caml_win32_maperr(DWORD errcode);
#endif

#include <errno.h>

#if defined(_WIN32)
/* windows.h alone pulls in winsock.h, which clashes with the winsock2.h of
   caml/unixsupport.h. */
#include <winsock2.h>
#include <windows.h>
#include <stddef.h>
#include <stdlib.h>
#else
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#endif

/* Syncs

   The file's sync orders its bytes before later changes to the file system,
   such as its rename, as rig.disk's barrier does. macOS's fsync reaches the
   drive, which may still reorder the writes (fsync(2)); F_BARRIERFSYNC orders
   them. A file system without barriers may take F_FULLFSYNC, which orders
   too, and one with neither only fsync. Only a refusal of a call falls
   through to the next: another failure is the sync's. */

#if !defined(_WIN32)
static int refused(int code) {
  return code == ENOTSUP || code == ENOTTY || code == EINVAL;
}

static int fsync_retrying(int fd) {
  int r;
  do
    r = fsync(fd);
  while (r != 0 && errno == EINTR);
  return r == 0 ? 0 : errno;
}

#if defined(__APPLE__)
static int fcntl_retrying(int fd, int cmd) {
  int r;
  do
    r = fcntl(fd, cmd);
  while (r == -1 && errno == EINTR);
  return r == -1 ? errno : 0;
}

static int sync_fd(int fd) {
  int code = fcntl_retrying(fd, F_BARRIERFSYNC);
  if (code == 0 || !refused(code))
    return code;
  code = fcntl_retrying(fd, F_FULLFSYNC);
  if (code == 0 || !refused(code))
    return code;
  return fsync_retrying(fd);
}
#else
static int sync_fd(int fd) { return fsync_retrying(fd); }
#endif
#endif

CAMLprim value caml_nx_io_sync(value vfd) {
#if defined(_WIN32)
  HANDLE fd = Handle_val(vfd);
  caml_release_runtime_system();
  int error = FlushFileBuffers(fd) ? 0 : (int)GetLastError();
#else
  int fd = Int_val(vfd);
  caml_release_runtime_system();
  int error = sync_fd(fd);
#endif
  caml_acquire_runtime_system();
  if (error != 0) {
#if defined(_WIN32)
    caml_win32_maperr((DWORD)error);
    error = errno;
#endif
    unix_error(error, "fsync", Nothing);
  }
  return Val_unit;
}

/* Renames

   A rename puts a file at a path, replacing the file there in one step, as
   POSIX's rename does. On Windows, a rename of POSIX semantics
   (FileRenameInfoEx), which NTFS takes on Windows 10 and later, unlinks the
   old file as POSIX's rename does: its handles and views keep reading it.
   Where it fails, such as on a file system without such renames or an older
   Windows, MoveFileExW renames, and its error is the rename's: it refuses to
   replace a file that a handle holds open, even one shared for deletion, or
   that a view maps. */

#if defined(_WIN32)
/* FILE_RENAME_INFO with the flags that older headers lack in place of its
   ReplaceIfExists, and the values winbase.h gives FileRenameInfoEx and
   FILE_RENAME_FLAG_REPLACE_IF_EXISTS and _POSIX_SEMANTICS. */
struct rename_info {
  DWORD flags;
  HANDLE root;
  DWORD length;
  WCHAR name[1];
};

#define RENAME_INFO_EX ((FILE_INFO_BY_HANDLE_CLASS)22)
#define REPLACE_IF_EXISTS 0x1
#define POSIX_SEMANTICS 0x2

/* Renames [src] to [dst] with POSIX semantics: [0], or the system's error.
   The name is [dst]'s full path, which SetFileInformationByHandle takes. A
   directory at [dst] is left to MoveFileExW, which refuses it as POSIX's
   rename of a file does, and which such a rename might replace if empty. */
static DWORD rename_posix(const wchar_t *src, const wchar_t *dst) {
  DWORD attributes = GetFileAttributesW(dst);
  if (attributes != INVALID_FILE_ATTRIBUTES &&
      (attributes & FILE_ATTRIBUTE_DIRECTORY))
    return ERROR_ACCESS_DENIED;
  DWORD n = GetFullPathNameW(dst, 0, NULL, NULL);
  if (n == 0)
    return GetLastError();
  DWORD size = (DWORD)(offsetof(struct rename_info, name) + n * sizeof(WCHAR));
  struct rename_info *info = malloc(size);
  if (info == NULL)
    return ERROR_NOT_ENOUGH_MEMORY;
  /* The length without the NUL, or the room needed if the current directory
     changed since the first call. */
  DWORD len = GetFullPathNameW(dst, n, info->name, NULL);
  DWORD code = len == 0 ? GetLastError() : 0;
  if (len >= n)
    code = ERROR_FILENAME_EXCED_RANGE;
  if (code == 0) {
    info->flags = REPLACE_IF_EXISTS | POSIX_SEMANTICS;
    info->root = NULL;
    info->length = len * sizeof(WCHAR);
    HANDLE h = CreateFileW(src, DELETE | SYNCHRONIZE,
                           FILE_SHARE_READ | FILE_SHARE_WRITE |
                               FILE_SHARE_DELETE,
                           NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE)
      code = GetLastError();
    else {
      if (!SetFileInformationByHandle(h, RENAME_INFO_EX, info, size))
        code = GetLastError();
      CloseHandle(h);
    }
  }
  free(info);
  return code;
}
#endif

CAMLprim value caml_nx_io_rename(value vsrc, value vdst) {
  CAMLparam2(vsrc, vdst);
  caml_unix_check_path(vsrc, "rename");
  caml_unix_check_path(vdst, "rename");
  /* Signal handlers run here, before the paths are held. The release below
     does not run them, so nothing raises while the paths are held. */
  caml_process_pending_actions();
#if defined(_WIN32)
  /* A conversion raises only for a path that is not UTF-8, and [vsrc] is
     [vdst] with an ASCII suffix: either both convert or the first raises,
     holding nothing. */
  wchar_t *src = caml_stat_strdup_noexc_to_utf16(String_val(vsrc));
  wchar_t *dst = caml_stat_strdup_noexc_to_utf16(String_val(vdst));
  if (src == NULL || dst == NULL) {
    caml_stat_free(src);
    caml_stat_free(dst);
    caml_raise_out_of_memory();
  }
  caml_enter_blocking_section_no_pending();
  DWORD code = 0;
  if (rename_posix(src, dst) != 0 &&
      !MoveFileExW(src, dst,
                   MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
    code = GetLastError();
  caml_leave_blocking_section();
  caml_stat_free(src);
  caml_stat_free(dst);
  if (code != 0) {
    caml_win32_maperr(code);
    unix_error(errno, "rename", vsrc);
  }
#else
  char *src = caml_stat_strdup_noexc(String_val(vsrc));
  char *dst = caml_stat_strdup_noexc(String_val(vdst));
  if (src == NULL || dst == NULL) {
    caml_stat_free(src);
    caml_stat_free(dst);
    caml_raise_out_of_memory();
  }
  caml_enter_blocking_section_no_pending();
  int error = rename(src, dst) == 0 ? 0 : errno;
  caml_leave_blocking_section();
  caml_stat_free(src);
  caml_stat_free(dst);
  if (error != 0)
    unix_error(error, "rename", vsrc);
#endif
  CAMLreturn(Val_unit);
}

/* The directory's sync orders the names a rename or link put in it before
   later changes to the file system, as the file's sync orders its bytes; on
   macOS the barrier promises no more. A directory that cannot be opened, or
   whose file system refuses to sync one, as some do, is left as it is, as
   SQLite does for its journal's. Windows opens no directory as a file, so a
   rename there is ordered by the file system alone. The file's bytes were
   flushed before it, so a crash may lose the rename but never shows it
   without them. */
CAMLprim value caml_nx_io_sync_dir(value vpath) {
#if defined(_WIN32)
  (void)vpath;
  return Val_unit;
#else
  CAMLparam1(vpath);
  caml_unix_check_path(vpath, "open");
  /* Signal handlers run here, before [path] is held. The release below does
     not run them, so nothing raises while [path] is held. */
  caml_process_pending_actions();
  char *path = caml_stat_strdup(String_val(vpath));
  caml_enter_blocking_section_no_pending();
  int error = 0;
  int fd;
  do
    fd = open(path, O_RDONLY | O_CLOEXEC);
  while (fd < 0 && errno == EINTR);
  if (fd >= 0) {
    error = sync_fd(fd);
    if (refused(error))
      error = 0;
    close(fd);
  }
  caml_acquire_runtime_system();
  caml_stat_free(path);
  if (error != 0)
    unix_error(error, "fsync", vpath);
  CAMLreturn(Val_unit);
#endif
}
