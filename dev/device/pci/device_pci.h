/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Windows from C: the accesses a driver's submission makes without calling
   OCaml.

   A window (Window.t) is a range of a machine's addresses. On this machine
   it is mapped into the process. On another machine it is reached through a
   transport, whose functions the accesses call; they may block, so a driver
   whose windows may be another machine's declares that its submission may
   block and runs it without the OCaml runtime.

   Every access returns 0, or -1 once the transport failed: a read then
   gives all ones and a write is dropped, as for a function that left the
   bus, and the transport's [failed] gives the reason. A submission checks
   once, after its last access.
   Values are little-endian; so is every host this library builds for. */

#ifndef DEVICE_PCI_H
#define DEVICE_PCI_H

#include <stddef.h>
#include <stdint.h>

#ifndef CAML_NAME_SPACE
#define CAML_NAME_SPACE
#endif
#include <caml/mlvalues.h>

/* The accesses to another machine's addresses, which the library that
   reaches it implements. [read] and [write] move [n] bytes at [address] of
   the machine, as one access where [n] is 4 or 8 and [address] is aligned
   to it, and return 0, or -1 once the transport failed. They may block, and
   may be called from several threads at once, never holding the OCaml
   runtime; the accesses one thread makes complete in the order it makes
   them. [failed] is the reason the transport failed, or NULL; once it is not
   NULL it stays. */
struct device_pci_transport {
  void *ctx;
  int (*read)(void *ctx, uint64_t address, void *dst, size_t n);
  int (*write)(void *ctx, uint64_t address, const void *src, size_t n);
  const char *(*failed)(void *ctx);
};

struct device_pci_window {
  uint64_t address;                             /* its first byte on its machine */
  size_t length;                                /* its bytes */
  volatile uint8_t *mapped;                     /* in this process, or NULL */
  const struct device_pci_transport *transport; /* when not mapped */
};

/* Reads the window [w], a Window.t, into [out]. It reads the OCaml heap, so
   the caller holds the OCaml runtime; [out] holds no OCaml value and serves
   without it. Bounds are the caller's: no access below checks them. */
void device_pci_window_of(value w, struct device_pci_window *out);

/* Orders every access before it before every access after it, as devices
   see them; on arm64 the full-system barrier stores to a BAR need. */
static inline void device_pci_barrier(void) {
#if defined(__aarch64__)
  __asm__ __volatile__("dsb sy" ::: "memory");
#else
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

/* Register accesses: each is one access of exactly its width, at an
   address aligned to it. */

static inline int device_pci_store32(const struct device_pci_window *w,
                                     size_t off, uint32_t x) {
  if (w->mapped) {
    *(volatile uint32_t *)(w->mapped + off) = x;
    return 0;
  }
  return w->transport->write(w->transport->ctx, w->address + off, &x, 4);
}

static inline int device_pci_store64(const struct device_pci_window *w,
                                     size_t off, uint64_t x) {
  if (w->mapped) {
    *(volatile uint64_t *)(w->mapped + off) = x;
    return 0;
  }
  return w->transport->write(w->transport->ctx, w->address + off, &x, 8);
}

static inline int device_pci_load32(const struct device_pci_window *w,
                                    size_t off, uint32_t *x) {
  if (w->mapped) {
    *x = *(volatile uint32_t *)(w->mapped + off);
    return 0;
  }
  if (w->transport->read(w->transport->ctx, w->address + off, x, 4) == 0)
    return 0;
  *x = UINT32_MAX;
  return -1;
}

static inline int device_pci_load64(const struct device_pci_window *w,
                                    size_t off, uint64_t *x) {
  if (w->mapped) {
    *x = *(volatile uint64_t *)(w->mapped + off);
    return 0;
  }
  if (w->transport->read(w->transport->ctx, w->address + off, x, 8) == 0)
    return 0;
  *x = UINT64_MAX;
  return -1;
}

/* Copies the [n] bytes at [src] to byte [off] of [w], at widths it chooses:
   memory, never registers. */
int device_pci_write(const struct device_pci_window *w, size_t off,
                     const void *src, size_t n);

#endif
