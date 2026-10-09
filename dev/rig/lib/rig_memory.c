/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Memory devices, whose memory is the host's and whose work runs in the
   submitting thread, and the readers of buffers rig.h declares. Nothing
   here blocks but rig_buffer_wait, which runs Rig.Buffer.wait. */

#define _GNU_SOURCE

#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/callback.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "rig.h"
#include "rig_stubs.h"

/* Memory devices

   A memory device's state is its driver and its word, alone in a page,
   which other devices may map. */

struct memory_device {
  const struct rig_driver *driver;
  _Atomic uint64_t word;
};

static void *aligned(size_t align, size_t n) {
#ifdef _WIN32
  return _aligned_malloc(n, align);
#else
  void *p = NULL;
  return posix_memalign(&p, align, n) == 0 ? p : NULL;
#endif
}

static void aligned_free(void *p) {
#ifdef _WIN32
  _aligned_free(p);
#else
  free(p);
#endif
}


/* [v_n] bytes of host memory, on a page from 64 KiB on, or 0. */
value caml_rig_memory_alloc(value v_n) {
  size_t n = (size_t)Long_val(v_n);
  void *p = aligned(n >= (1 << 16) ? rig_page_bytes() : 64, n);
  return Val_long((intnat)p);
}

value caml_rig_memory_free(value v_p) {
  aligned_free((void *)Long_val(v_p));
  return Val_unit;
}

static int memory_room(void *self, const struct rig_part *parts, int n,
                       const uint8_t *args) {
  (void)args;
  (void)self;
  for (int i = 0; i < n; i++) {
    const struct rig_part *p = &parts[i];
    if (p->kind == RIG_FILL ? p->fill.ring_units != 0 ||
                                  p->fill.segment_bytes != 0
                            : p->kind != RIG_COPY)
      return RIG_NEVER;
  }
  return RIG_FITS;
}

/* Runs the parts in order, the fills with no queue context, then makes [v]
   observable. Its handles are host addresses. */
static int memory_submit(void *self, uint64_t v, const struct rig_wait *waits,
                         int nwaits, const struct rig_part *parts, int nparts,
                         const uint8_t *args, const uint64_t *slots, int nslots,
                         const uint64_t *handles, int nhandles,
                         const char **failure) {
  (void)args;
  (void)slots;
  (void)nslots;
  struct memory_device *m = self;
  (void)waits;
  (void)nwaits;
  (void)handles;
  (void)nhandles;
  for (int i = 0; i < nparts; i++) {
    const struct rig_part *p = &parts[i];
    if (p->kind == RIG_FILL) {
      if (p->fill.fn(NULL, p->fill.arg, v) != 0) {
        *failure = "a fill failed";
        return RIG_FAILED;
      }
    } else if (p->copy.bytes != 0)
      memmove((char *)(uintptr_t)p->copy.dst + p->copy.dst_offset,
              (const char *)(uintptr_t)p->copy.src + p->copy.src_offset,
              (size_t)p->copy.bytes);
  }
  atomic_store_explicit(&m->word, v, memory_order_release);
  return RIG_COMMITTED;
}

/* The work ran at its hand-over: nothing is left to commit. */
static int memory_commit(void *self, uint64_t v, const char **failure) {
  (void)self;
  (void)v;
  (void)failure;
  return RIG_OK;
}

static const struct rig_driver memory_driver = {memory_room, memory_submit,
                                                memory_commit};

/* A memory device's state, which the free of its word gives back. */
value caml_rig_memory_new(value unit) {
  (void)unit;
  size_t page = rig_page_bytes();
  struct memory_device *m = aligned(page, page);
  if (m == NULL) caml_raise_out_of_memory();
  memset(m, 0, page);
  m->driver = &memory_driver;
  return caml_copy_nativeint((intnat)m);
}

/* The host address of the word of the memory device [v_m]. */
value caml_rig_memory_word(value v_m) {
  struct memory_device *m = (struct memory_device *)Nativeint_val(v_m);
  return Val_long((intnat)&m->word);
}

value caml_rig_load64(value v_addr) {
  return Val_long((intnat)atomic_load_explicit(
      (_Atomic uint64_t *)Long_val(v_addr), memory_order_acquire));
}

/* Copies the [v_n] bytes of the string [v_s] from [v_i] to the host
   address [v_dst]. With the runtime held: the string is in the OCaml
   heap. */
value caml_rig_blit_string(value v_s, value v_i, value v_dst, value v_n) {
  size_t n = (size_t)Long_val(v_n);
  if (n > 0)
    memcpy((void *)Long_val(v_dst), String_val(v_s) + Long_val(v_i), n);
  return Val_unit;
}

/* Copies [v_n] bytes from the host address [v_src] into the bytes [v_b]
   from [v_j]. With the runtime held: the bytes are in the OCaml heap. */
value caml_rig_blit_bytes(value v_src, value v_b, value v_j, value v_n) {
  size_t n = (size_t)Long_val(v_n);
  if (n > 0)
    memcpy(Bytes_val(v_b) + Long_val(v_j), (void *)Long_val(v_src), n);
  return Val_unit;
}

/* Reading buffers

   A buffer is the record { mem; offset; length; generation }, its
   memory { dev; bytes; host; address; handle; claim; entry; root; ... },
   its claim { count; generation; why } and the entry of its root
   { owner; memory; bytes; region; io_region; access; stamps; ... },
   as rig's Def module lays them out. */

enum { DEVICE_INDEX, DEVICE_NAME, DEVICE_MACHINE, DEVICE_KIND, DEVICE_C };
enum { BUFFER_MEM, BUFFER_OFFSET, BUFFER_LENGTH, BUFFER_GEN };
enum { MEMORY_DEV, MEMORY_BYTES, MEMORY_HOST, MEMORY_ADDRESS, MEMORY_HANDLE,
       MEMORY_CLAIM, MEMORY_ENTRY, MEMORY_ROOT };
enum { CLAIM_COUNT, CLAIM_GEN, CLAIM_WHY };
enum { ENTRY_OWNER, ENTRY_MEMORY, ENTRY_BYTES, ENTRY_REGION, ENTRY_IO_REGION,
       ENTRY_ACCESS, ENTRY_STAMPS };

/* The claim word's bit for memory that admits only reads, and its step
   per claim, as rig's Memory module lays the word out. */
#define CLAIM_READ_ONLY 2
#define CLAIM_ONE 4

/* A field another domain writes. Read after the claim's compare-and-set,
   which acquires, so it shows what the release of any earlier claim
   published. */
static value load_field(value v, int i) {
  return atomic_load_explicit((_Atomic value *)&Field(v, i),
                              memory_order_relaxed);
}

void *rig_buffer_host(value b) {
  value mem = Field(b, BUFFER_MEM);
  intnat host = Long_val(Field(mem, MEMORY_HOST));
  if (host < 0) return NULL;
  return (char *)host + Long_val(Field(b, BUFFER_OFFSET));
}

size_t rig_buffer_bytes(value b) {
  return (size_t)Long_val(Field(b, BUFFER_LENGTH));
}

const char *rig_buffer_why(value b) {
  value claim = Field(Field(b, BUFFER_MEM), MEMORY_CLAIM);
  value g = atomic_load_explicit((_Atomic value *)&Field(claim, CLAIM_GEN),
                                 memory_order_acquire);
  if (Long_val(g) == Long_val(Field(b, BUFFER_GEN))) return NULL;
  return String_val(Field(claim, CLAIM_WHY));
}

/* Claims */

/* Whether the device of the memory record [mem] is lost: a borrow's
   device, or the device whose memory a root is. */
static int dev_lost(value mem) {
  struct rig_device *d =
      (struct rig_device *)Long_val(Field(Field(mem, MEMORY_DEV), DEVICE_C));
  return atomic_load_explicit(&d->state, memory_order_acquire) != RIG_LIVE;
}

static int uses_done(struct rig_stamps *s) {
  for (; s != NULL; s = atomic_load_explicit(&s->next, memory_order_acquire))
    for (int i = 0; i < RIG_USES; i++) {
      uint64_t p = atomic_load_explicit(&s->use[i], memory_order_acquire);
      if (RIG_VALUE(p) != 0 && !rig_point_done(p)) return 0;
    }
  return 1;
}

/* Whether the work of the stamps [s] that an access must follow is done:
   that of the last write, or of every use if [every], and of every use of
   their hold's. A use word that a submission reserved holds no value until
   its first raise. */
static int stamps_done(struct rig_stamps *s, int every) {
  uint64_t w = atomic_load_explicit(&s->write, memory_order_acquire);
  if (w != 0 && !rig_point_done(w)) return 0;
  if (every && !uses_done(s)) return 0;
  struct rig_stamps *hold = held(s);
  return hold == NULL || uses_done(hold);
}

/* Claims first, then checks [b] under the claim: the compare-and-set
   follows the release of any donation that consumed the memory, so the
   checks see its consumption and its stamps. Work that is not done keeps
   the claim: the caller waits under it, so no donation consumes the memory
   between the wait and the access. */
enum rig_claim rig_buffer_claim(value b, enum rig_access access) {
  value mem = Field(b, BUFFER_MEM);
  value claim = Field(mem, MEMORY_CLAIM);
  _Atomic value *word = (_Atomic value *)&Field(claim, CLAIM_COUNT);
  value w = atomic_load_explicit(word, memory_order_relaxed);
  do {
    if (Long_val(w) < 0) return RIG_EXCLUSIVE;
    if (access == RIG_READ_WRITE && (Long_val(w) & CLAIM_READ_ONLY))
      return RIG_READ_ONLY;
  } while (!atomic_compare_exchange_weak(word, &w,
                                         Val_long(Long_val(w) + CLAIM_ONE)));
  if (Long_val(load_field(claim, CLAIM_GEN)) != Long_val(Field(b, BUFFER_GEN))) {
    rig_buffer_release(b);
    return RIG_DEAD;
  }
  value root = Field(mem, MEMORY_ROOT);
  value entry = load_field(root, MEMORY_ENTRY);
  struct rig_stamps *s =
      (struct rig_stamps *)Long_val(load_field(entry, ENTRY_STAMPS));
  if (dev_lost(mem) || (root != mem && dev_lost(root)) ||
      (s != NULL && !stamps_done(s, access == RIG_READ_WRITE)))
    return RIG_WAIT;
  return RIG_CLAIMED;
}

/* Rig.Buffer.wait, which buffer.ml registers, found once. */
static _Atomic(const value *) buffer_wait;

caml_result rig_buffer_wait(value b, enum rig_access access) {
  const value *f = atomic_load_explicit(&buffer_wait, memory_order_acquire);
  if (f == NULL) {
    f = caml_named_value("rig.buffer.wait");
    if (f == NULL)
      caml_fatal_error("rig_buffer_wait: Rig.Buffer.wait is not registered");
    atomic_store_explicit(&buffer_wait, f, memory_order_release);
  }
  return caml_callback2_res(*f, b, Val_int(access));
}

/* A claim keeps the word at CLAIM_ONE or more, so ending it is one
   subtraction of the tagged step. */
void rig_buffer_release(value b) {
  value claim = Field(Field(b, BUFFER_MEM), MEMORY_CLAIM);
  atomic_fetch_sub((_Atomic value *)&Field(claim, CLAIM_COUNT),
                   Val_long(CLAIM_ONE) - Val_long(0));
}
