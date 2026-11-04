/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Setting a file's times through a descriptor, which OCaml's Unix library
   does only by path ([Unix.utimes]), renaming a file over one that is open,
   which OCaml's rename does not do on Windows, starting a process with no
   signal blocked, which [Unix.create_process] does not do, and the cores the
   process may occupy, which rig_pool.h counts. */

#define _GNU_SOURCE

/* SetFileInformationByHandle is Windows Vista's. */
#if defined(_WIN32) && (!defined(_WIN32_WINNT) || _WIN32_WINNT < 0x0600)
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif

#define CAML_NAME_SPACE
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/osdeps.h>
#include <caml/signals.h>
#include <caml/unixsupport.h>

#include "rig_pool.h"

/* caml/unixsupport.h declares the unix library's functions without
   CAMLextern: a stub DLL loaded at run time on Windows would call them
   directly, which flexdll links within 2 GiB only. Declared again with
   CAMLextern, each call goes through a pointer. */
#ifdef _WIN32
CAMLnoret CAMLextern void caml_uerror(const char *cmdname, value arg);
CAMLextern void caml_unix_check_path(value path, const char *cmdname);
CAMLextern void caml_win32_maperr(DWORD errcode);
#endif

#ifdef _WIN32
#include <stddef.h>
#include <stdlib.h>
#include <windows.h>
#else
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
extern char **environ;
#endif

/* [tolk_touch fd] sets the access and modification times of the file [fd]
   is open on to now. On Windows the handle needs FILE_WRITE_ATTRIBUTES,
   which a descriptor opened for writing has. Raises [Unix.Unix_error]. */
CAMLprim value tolk_touch(value fd)
{
#ifdef _WIN32
  HANDLE h = Handle_val(fd);
  FILETIME now;
  BOOL ok;
  GetSystemTimeAsFileTime(&now);
  caml_enter_blocking_section();
  ok = SetFileTime(h, NULL, &now, &now);
  caml_leave_blocking_section();
  if (!ok) {
    caml_win32_maperr(GetLastError());
    caml_uerror("SetFileTime", Nothing);
  }
#else
  int d = Int_val(fd);
  int r;
  caml_enter_blocking_section();
  r = futimens(d, NULL);
  caml_leave_blocking_section();
  if (r == -1) caml_uerror("futimens", Nothing);
#endif
  return Val_unit;
}

#ifdef _WIN32
/* FILE_RENAME_INFO whose first member is the flags of FileRenameInfoEx
   (Windows 10 1607), which older headers declare as a BOOLEAN only. */
typedef struct {
  DWORD flags;
  HANDLE root;
  DWORD length; /* of [name] in bytes, without its NUL */
  WCHAR name[1];
} rename_info;

/* FileRenameInfoEx's class and flags, spelled out for the same headers. */
#define FILE_RENAME_INFO_EX 22
#define RENAME_REPLACE_IF_EXISTS 0x1
#define RENAME_POSIX_SEMANTICS 0x2

/* [posix_rename src dst] renames [src] to [dst], replacing the file [dst]
   names even while another handle has it open shared for deletion: POSIX
   semantics take the old file's name away and leave its handles valid. The
   new name is made a full path, so that it names the file a relative [dst]
   names for the rest of the program, whatever the rename makes of a
   relative name. Is [0] or the error. */
static DWORD posix_rename(const wchar_t *src, const wchar_t *dst)
{
  DWORD n = GetFullPathNameW(dst, 0, NULL, NULL);
  if (n == 0) return GetLastError();
  size_t size = offsetof(rename_info, name) + (size_t)n * sizeof(WCHAR);
  rename_info *info = malloc(size);
  if (info == NULL) return ERROR_NOT_ENOUGH_MEMORY;
  DWORD len = GetFullPathNameW(dst, n, info->name, NULL);
  if (len == 0 || len >= n) {
    DWORD err = len == 0 ? GetLastError() : ERROR_BUFFER_OVERFLOW;
    free(info);
    return err;
  }
  info->flags = RENAME_REPLACE_IF_EXISTS | RENAME_POSIX_SEMANTICS;
  info->root = NULL;
  info->length = len * sizeof(WCHAR);
  HANDLE h = CreateFileW(src, DELETE,
                         FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                         NULL, OPEN_EXISTING, 0, NULL);
  DWORD err = 0;
  if (h == INVALID_HANDLE_VALUE) {
    err = GetLastError();
  } else {
    FILE_INFO_BY_HANDLE_CLASS class =
        (FILE_INFO_BY_HANDLE_CLASS)FILE_RENAME_INFO_EX;
    if (!SetFileInformationByHandle(h, class, info, (DWORD)size))
      err = GetLastError();
    CloseHandle(h);
  }
  free(info);
  return err;
}

/* [rename_over src dst] is [posix_rename src dst], or on any failure of it,
   as on systems without POSIX semantics, the classic rename that MoveFileEx
   makes, which fails while any handle has the old file open. Is [0] or the
   classic rename's error. */
static DWORD rename_over(const wchar_t *src, const wchar_t *dst)
{
  if (posix_rename(src, dst) == 0) return 0;
  return MoveFileExW(src, dst, MOVEFILE_REPLACE_EXISTING) ? 0 : GetLastError();
}
#endif

/* [tolk_rename src dst] renames the file [src] to [dst], replacing the file
   [dst] names, on Windows even while another handle has it open shared for
   deletion. Raises [Unix.Unix_error]. */
CAMLprim value tolk_rename(value src, value dst)
{
  CAMLparam2(src, dst);
  caml_unix_check_path(src, "rename");
  caml_unix_check_path(dst, "rename");
  char_os *s = caml_stat_strdup_to_os(String_val(src));
  char_os *d = caml_stat_strdup_to_os(String_val(dst));
#ifdef _WIN32
  DWORD err;
  caml_enter_blocking_section();
  err = rename_over(s, d);
  caml_leave_blocking_section();
  caml_stat_free(d);
  caml_stat_free(s);
  if (err != 0) {
    caml_win32_maperr(err);
    caml_uerror("rename", src);
  }
#else
  int r;
  caml_enter_blocking_section();
  r = rename(s, d);
  caml_leave_blocking_section();
  caml_stat_free(d);
  caml_stat_free(s);
  if (r == -1) caml_uerror("rename", src);
#endif
  CAMLreturn(Val_unit);
}

/* [tolk_spawn prog argv fds] starts [prog], found in PATH as by execvp, with
   the arguments [argv] and the descriptors [fds.(0)], [fds.(1)] and [fds.(2)]
   as its standard input, output and error, and is its pid. The process starts
   with no signal blocked: [Unix.create_process] gives it the calling thread's
   mask, which on a domain Worker spawns blocks every signal. A descriptor of
   [fds] below 3 that is not its own is first copied above 2, so that no
   redirection overwrites one still to come. Every other descriptor that is
   not close-on-exec, a source of the three included, stays open in it.
   Raises Unix.Unix_error if the process does not start. Unix only. */
CAMLprim value tolk_spawn(value v_prog, value v_argv, value v_fds)
{
#ifdef _WIN32
  (void)v_prog;
  (void)v_argv;
  (void)v_fds;
  caml_failwith("tolk_spawn: Windows uses Unix.create_process");
#else
  CAMLparam3(v_prog, v_argv, v_fds);
  int fds[3], copies[3] = {-1, -1, -1}, r = 0;
  pid_t pid = 0;
  posix_spawn_file_actions_t actions;
  posix_spawnattr_t attributes;
  sigset_t none;
  caml_unix_check_path(v_prog, "create_process");
  char **argv = caml_unix_cstringvect(v_argv, "create_process");
  for (int i = 0; i < 3; i++) {
    fds[i] = Int_val(Field(v_fds, i));
    if (fds[i] < 3 && fds[i] != i && r == 0) {
      copies[i] = fcntl(fds[i], F_DUPFD_CLOEXEC, 3);
      if (copies[i] == -1) r = errno;
      else fds[i] = copies[i];
    }
  }
  if (r == 0) r = posix_spawn_file_actions_init(&actions);
  if (r == 0) {
    for (int i = 0; i < 3 && r == 0; i++)
      if (fds[i] != i)
        r = posix_spawn_file_actions_adddup2(&actions, fds[i], i);
    if (r == 0) r = posix_spawnattr_init(&attributes);
    if (r == 0) {
      sigemptyset(&none);
      r = posix_spawnattr_setsigmask(&attributes, &none);
      if (r == 0)
        r = posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETSIGMASK);
      if (r == 0)
        r = posix_spawnp(&pid, String_val(v_prog), &actions, &attributes, argv,
                         environ);
      posix_spawnattr_destroy(&attributes);
    }
    posix_spawn_file_actions_destroy(&actions);
  }
  caml_unix_cstringvect_free(argv);
  for (int i = 0; i < 3; i++)
    if (copies[i] != -1) close(copies[i]);
  if (r != 0) caml_unix_error(r, "create_process", v_prog);
  CAMLreturn(Val_long(pid));
#endif
}

CAMLprim value tolk_cores(value unit)
{
  (void)unit;
  return Val_int(rig_pool_cores());
}
