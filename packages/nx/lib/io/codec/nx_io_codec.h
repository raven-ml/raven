/*--------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  --------------------------------------------------------------------------*/

#ifndef NX_IO_CODEC_H
#define NX_IO_CODEC_H

#include <errno.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

/* File output. An nx_io_fd is a descriptor on POSIX and the HANDLE behind an
   OCaml Unix.file_descr on Windows; NX_IO_NO_FD means no file. */
typedef intptr_t nx_io_fd;
#define NX_IO_NO_FD ((nx_io_fd)-1)

#ifndef NX_IO_CODEC_NO_OCAML
#include <caml/mlvalues.h>
#include <caml/unixsupport.h>
#if defined(_WIN32)
#define Nx_io_fd_val(v) ((nx_io_fd)Handle_val(v))
#else
#define Nx_io_fd_val(v) ((nx_io_fd)Int_val(v))
#endif
#endif

/* Writes all [len] bytes to [fd]. Returns 0, or the errno it leaves set (EIO
   when the platform reports nothing finer). */
static inline int nx_io_write_all(nx_io_fd fd, const uint8_t *src, size_t len) {
  size_t off = 0;
  while (off < len) {
#if defined(_WIN32)
    size_t want = len - off;
    DWORD chunk = want > 0x7fffffffu ? 0x7fffffffu : (DWORD)want;
    DWORD written = 0;
    if (!WriteFile((HANDLE)fd, src + off, chunk, &written, NULL) ||
        written == 0) {
      errno = EIO;
      return EIO;
    }
#else
    ssize_t written = write((int)fd, src + off, len - off);
    if (written < 0 && errno == EINTR)
      continue;
    if (written <= 0) {
      if (written == 0)
        errno = EIO;
      return errno;
    }
#endif
    off += (size_t)written;
  }
  return 0;
}

#endif
