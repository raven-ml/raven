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

   An access reports nothing. Once the transport failed, a read gives all
   ones and a write is dropped, as for a function that left the bus. A
   submission asks rig_pci_failed once, after its last access: NULL
   there means the transport had not failed by then. A store may return
   before it reaches the machine, so only a NULL after a load, such as
   rig_pci_flush's, also means the stores before the load reached it.
   Values are little-endian; so is every host this library builds for. */

#ifndef RIG_PCI_H
#define RIG_PCI_H

#include <stddef.h>
#include <stdint.h>

#ifndef CAML_NAME_SPACE
#define CAML_NAME_SPACE
#endif
#include <caml/mlvalues.h>

/* The accesses to another machine's addresses, which the library that
   reaches it implements. [read] and [write] move [n] bytes at [address] of
   the machine, as one access where [n] is 4 or 8 and [address] is aligned
   to it, and return 0, or -1 once the transport failed. A write may return
   before it reaches the machine; its failure then shows at a later access.
   They may block, and may be called from several threads at once, never
   holding the OCaml runtime; the accesses one thread makes complete in the
   order it makes them. [failed] is the reason the transport failed,
   starting with the machine's name, or NULL; once it is not NULL it
   stays. */
struct rig_pci_transport {
  void *ctx;
  int (*read)(void *ctx, uint64_t address, void *dst, size_t n);
  int (*write)(void *ctx, uint64_t address, const void *src, size_t n);
  const char *(*failed)(void *ctx);
};

struct rig_pci_window {
  uint64_t address;                             /* its first byte on its machine */
  size_t length;                                /* its bytes */
  volatile uint8_t *mapped;                     /* in this process, or NULL */
  const struct rig_pci_transport *transport; /* when not mapped */
  int combines;                                 /* stores may merge until a barrier */
};

/* Reads the window [w], a Window.t, into [out]. It reads the OCaml heap, so
   the caller holds the OCaml runtime; [out] holds no OCaml value and serves
   without it. Bounds are the caller's: no access below checks them. */
void rig_pci_window_of(value w, struct rig_pci_window *out);

/* Orders every access before it before every access after it, as devices
   see them: on arm64 the full-system barrier stores to a BAR need; on
   x86_64 mfence, as Linux's mb(), which Intel's manual names to order
   streaming loads from write-combined memory (MOVNTDQA). A C11 fence is
   not that: gcc compiles it to a locked instruction. */
static inline void rig_pci_barrier(void) {
#if defined(__aarch64__)
  __asm__ __volatile__("dsb sy" ::: "memory");
#elif defined(__x86_64__)
  __asm__ __volatile__("mfence" ::: "memory");
#else
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

/* The reason the machine of [w] failed, or NULL. A mapped window's machine
   is this one, which never fails. */
static inline const char *
rig_pci_failed(const struct rig_pci_window *w) {
  if (w->mapped) return NULL;
  return w->transport->failed(w->transport->ctx);
}

/* Register accesses: each is one access of exactly its width, at an
   address aligned to it. */

static inline void rig_pci_store32(const struct rig_pci_window *w,
                                      size_t off, uint32_t x) {
  if (w->mapped)
    *(volatile uint32_t *)(w->mapped + off) = x;
  else
    (void)w->transport->write(w->transport->ctx, w->address + off, &x, 4);
}

static inline void rig_pci_store64(const struct rig_pci_window *w,
                                      size_t off, uint64_t x) {
  if (w->mapped)
    *(volatile uint64_t *)(w->mapped + off) = x;
  else
    (void)w->transport->write(w->transport->ctx, w->address + off, &x, 8);
}

static inline uint32_t rig_pci_load32(const struct rig_pci_window *w,
                                         size_t off) {
  uint32_t x;
  if (w->mapped) return *(volatile uint32_t *)(w->mapped + off);
  if (w->transport->read(w->transport->ctx, w->address + off, &x, 4) != 0)
    return UINT32_MAX;
  return x;
}

static inline uint64_t rig_pci_load64(const struct rig_pci_window *w,
                                         size_t off) {
  uint64_t x;
  if (w->mapped) return *(volatile uint64_t *)(w->mapped + off);
  if (w->transport->read(w->transport->ctx, w->address + off, &x, 8) != 0)
    return UINT64_MAX;
  return x;
}

/* Makes the stores made through [w] before it reach the function before
   any access after it, through any window: a barrier, then a load of [w]'s
   first word, which the bus answers only after them. A driver flushes the
   window it wrote before the store that has the function read what it
   wrote, such as a doorbell. [w] holds at least 4 bytes from a 4-byte
   boundary. */
static inline void rig_pci_flush(const struct rig_pci_window *w) {
  rig_pci_barrier();
  (void)rig_pci_load32(w, 0);
}

/* Copies the [n] bytes at [src] to byte [off] of [w], at widths it chooses:
   memory, never registers. */
void rig_pci_write(const struct rig_pci_window *w, size_t off,
                      const void *src, size_t n);

#endif
