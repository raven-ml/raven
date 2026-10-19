/*--------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  --------------------------------------------------------------------------*/

#ifndef NX_IMAGE_CODEC_H
#define NX_IMAGE_CODEC_H

#include <errno.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
/* windows.h alone pulls in winsock.h, which clashes with the winsock2.h of
   caml/unixsupport.h. */
#include <winsock2.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

/* File output. An nx_image_fd is a descriptor on POSIX and the HANDLE behind
   an OCaml Unix.file_descr on Windows. */
typedef intptr_t nx_image_fd;

#include <caml/mlvalues.h>
#include <caml/unixsupport.h>
#if defined(_WIN32)
#define Nx_image_fd_val(v) ((nx_image_fd)Handle_val(v))
#else
#define Nx_image_fd_val(v) ((nx_image_fd)Int_val(v))
#endif

/* Writes all [len] bytes to [fd]. Returns 0, or the errno it leaves set (EIO
   when the platform reports nothing finer). */
static inline int nx_image_write_all(nx_image_fd fd, const uint8_t *src, size_t len) {
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
