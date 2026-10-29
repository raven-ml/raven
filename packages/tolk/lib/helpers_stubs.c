/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Setting a file's times through a descriptor, which OCaml's Unix library
   does only by path ([Unix.utimes]), and renaming a file over one that is
   open, which OCaml's rename does not do on Windows. */

#define _GNU_SOURCE

/* SetFileInformationByHandle is Windows Vista's. */
#if defined(_WIN32) && (!defined(_WIN32_WINNT) || _WIN32_WINNT < 0x0600)
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif

#define CAML_NAME_SPACE
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/osdeps.h>
#include <caml/signals.h>
#include <caml/unixsupport.h>

#ifdef _WIN32
#include <stddef.h>
#include <stdlib.h>
#include <windows.h>
#else
#include <stdio.h>
#include <sys/stat.h>
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
