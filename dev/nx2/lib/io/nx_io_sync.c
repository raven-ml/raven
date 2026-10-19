/*--------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  The syncs that order a file's bytes before its rename, and the rename
  before later changes to the file system.
  --------------------------------------------------------------------------*/

#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
#include <caml/unixsupport.h>

#include <errno.h>

#if defined(_WIN32)
/* windows.h alone pulls in winsock.h, which clashes with the winsock2.h of
   caml/unixsupport.h. */
#include <winsock2.h>
#include <windows.h>
#else
#include <fcntl.h>
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

/* The directory's sync makes the names a rename or link put in it outlast a
   crash. A directory that cannot be opened, or whose file system refuses to
   sync one, as some do, is left as it is, as SQLite does for its journal's.
   Windows opens no directory as a file: a rename there reaches the disk
   before it returns (MOVEFILE_WRITE_THROUGH), and a link may be lost to a
   crash but is never seen without its file's bytes. */
CAMLprim value caml_nx_io_sync_dir(value vpath) {
#if defined(_WIN32)
  (void)vpath;
  return Val_unit;
#else
  CAMLparam1(vpath);
  CAMLlocal1(vname);
  caml_unix_check_path(vpath, "open");
  char *path = caml_stat_strdup(String_val(vpath));
  caml_release_runtime_system();
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
  if (error != 0) {
    vname = caml_copy_string(path);
    caml_stat_free(path);
    unix_error(error, "fsync", vname);
  }
  caml_stat_free(path);
  CAMLreturn(Val_unit);
#endif
}
