/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Memory devices, whose memory is the host's and whose work runs in the
   submitting thread, and the readers of buffers rig.h declares. Nothing
   here blocks but rig_buffer_wait, which runs Rig.Buffer.wait, and the end
   of a with_'s claims, which waits out a consumption under way. */

#define _GNU_SOURCE

#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sched.h>
#endif

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/callback.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "rig.h"
#include "rig_stubs.h"

/* Memory devices

   A memory device's state is its driver and the address of its word, the
   word rig gave its open. */

struct memory_device {
  const struct rig_driver *driver;
  _Atomic uint64_t *word;
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

static int memory_room(void *self, int nwaits, const struct rig_part *parts,
                       int n, void *scratch, const uint8_t *args) {
  (void)nwaits;
  (void)scratch;
  (void)args;
  (void)self;
  for (int i = 0; i < n; i++) {
    const struct rig_part *p = &parts[i];
    if (p->kind != RIG_FILL && p->kind != RIG_COPY)
      return RIG_NEVER;
  }
  return RIG_FITS;
}

/* Runs the copy part [p] over host addresses: a fill sets its row, and a
   box moves its rows, in one move where both sides' rows are contiguous. */
static void memory_copy(const struct rig_part *p) {
  size_t width = (size_t)p->copy.width, rows = (size_t)p->copy.rows;
  if (width == 0 || rows == 0) return;
  char *dst = (char *)(uintptr_t)p->copy.dst + p->copy.dst_offset;
  if (p->copy.fill) {
    memset(dst, p->copy.byte, width);
    return;
  }
  const char *src = (const char *)(uintptr_t)p->copy.src + p->copy.src_offset;
  if (p->copy.src_pitch == width && p->copy.dst_pitch == width) {
    memmove(dst, src, width * rows);
    return;
  }
  for (size_t i = 0; i < rows; i++)
    memmove(dst + i * p->copy.dst_pitch, src + i * p->copy.src_pitch, width);
}

/* Runs the parts in order, the fills with no queue context, then makes [v]
   observable. Its handles are host addresses. */
static int memory_submit(void *self, uint64_t v, const struct rig_wait *waits,
                         int nwaits, const struct rig_part *parts, int nparts,
                         void *scratch, uint8_t *args,
                         const uint64_t *slots, int nslots, uint64_t *times,
                         const char **failure) {
  (void)scratch;
  (void)times;
  (void)nslots;
  struct memory_device *m = self;
  (void)waits;
  (void)nwaits;
  for (int i = 0; i < nparts; i++) {
    const struct rig_part *p = &parts[i];
    if (p->kind == RIG_FILL) {
      if (rig_fill(NULL, p, args, slots, v) != 0) {
        *failure = "a fill failed";
        return RIG_FAILED;
      }
    } else
      memory_copy(p);
  }
  atomic_store_explicit(m->word, v, memory_order_release);
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
                                                memory_commit, NULL};

/* A new memory device's state, whose word lies at [v_word]. */
value caml_rig_memory_new(value v_word) {
  struct memory_device *m = malloc(sizeof *m);
  if (m == NULL) caml_raise_out_of_memory();
  m->driver = &memory_driver;
  m->word = (_Atomic uint64_t *)Nativeint_val(v_word);
  return caml_copy_nativeint((intnat)m);
}

/* Frees the state [v_m] at its device's stop, after which rig calls none of
   its functions. */
value caml_rig_memory_drop(value v_m) {
  free((void *)Nativeint_val(v_m));
  return Val_unit;
}

value caml_rig_load64(value v_addr) {
  return Val_long((intnat)atomic_load_explicit(
      (_Atomic uint64_t *)Long_val(v_addr), memory_order_acquire));
}

/* Copies the [v_n] bytes of the string [v_s] from [v_i] into the host
   buffer [v_b] from [v_j]. With the runtime held: the string is in the
   OCaml heap, and the buffer, which holds its memory, is reachable. */
value caml_rig_blit_string(value v_s, value v_i, value v_b, value v_j,
                           value v_n) {
  size_t n = (size_t)Long_val(v_n);
  if (n > 0)
    memcpy((char *)rig_buffer_host(v_b) + Long_val(v_j),
           String_val(v_s) + Long_val(v_i), n);
  return Val_unit;
}

/* Copies [v_n] bytes of the host buffer [v_src] from [v_i] into the bytes
   [v_b] from [v_j]. With the runtime held, as above. */
value caml_rig_blit_bytes(value v_src, value v_i, value v_b, value v_j,
                          value v_n) {
  size_t n = (size_t)Long_val(v_n);
  if (n > 0)
    memcpy(Bytes_val(v_b) + Long_val(v_j),
           (char *)rig_buffer_host(v_src) + Long_val(v_i), n);
  return Val_unit;
}

/* Sets the [v_n] bytes of the host buffer [v_b] from [v_j] to the byte
   [v_c]. With the runtime held, as above. */
value caml_rig_set_bytes(value v_b, value v_j, value v_n, value v_c) {
  size_t n = (size_t)Long_val(v_n);
  if (n > 0)
    memset((char *)rig_buffer_host(v_b) + Long_val(v_j), Int_val(v_c), n);
  return Val_unit;
}

/* Reading buffers

   A buffer is the record { mem; offset; length; generation }, its
   memory { dev; host; address; handle; alloc }, its allocation
   { owner; host; count; generation; why; entry; ... }, the allocation's
   entry { owner; backing; bytes; access; stamps; ... } and a device
   { index; name; machine; ...; opened }, as rig's Def module lays them
   out. */

enum { DEVICE_INDEX, DEVICE_NAME, DEVICE_MACHINE, DEVICE_OPENED = 16 };
enum { BUFFER_MEM, BUFFER_OFFSET, BUFFER_LENGTH, BUFFER_GEN };
enum { MEMORY_DEV, MEMORY_HOST, MEMORY_ADDRESS, MEMORY_HANDLE, MEMORY_ALLOC };
enum { ALLOC_OWNER, ALLOC_HOST, ALLOC_COUNT, ALLOC_GEN, ALLOC_WHY,
       ALLOC_ENTRY };

/* A device's machine when it is this process's: [Here], the machine type's
   first constant constructor. */
#define MACHINE_HERE Val_int(0)
enum { ENTRY_OWNER, ENTRY_BACKING, ENTRY_BYTES, ENTRY_ACCESS, ENTRY_STAMPS,
       ENTRY_MAPS, ENTRY_UNMAPS, ENTRY_PAGES };

/* The claim word as rig's Memory module lays it out: from 0 up, bits for
   memory outside the claims, memory that admits only reads and a
   consumption under way, the step per read claim from bit 3 and the step
   per hold from bit 32; -1, held exclusive. */
#define CLAIM_LOCKED 4
#define CLAIM_ONE 8
#define CLAIM_HOLD ((intnat)1 << 32)
#define CLAIM_EXCLUSIVE (-1)

/* A spin's pause while another domain's step holds CLAIM_LOCKED, a few
   instructions: the core's hint that it spins, as rig's pool spins. */
static inline void relax(void) {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  __asm__ __volatile__("yield");
#endif
}

/* A field another domain writes. Read after the claim's compare-and-set,
   which acquires, so it shows what the release of any earlier claim
   published. */
static value load_field(value v, int i) {
  return atomic_load_explicit((_Atomic value *)&Field(v, i),
                              memory_order_relaxed);
}

/* The allocation [b] lies in, and its claim word. */
static value alloc_of(value b) {
  return Field(Field(b, BUFFER_MEM), MEMORY_ALLOC);
}

static _Atomic value *claim_word(value b) {
  return (_Atomic value *)&Field(alloc_of(b), ALLOC_COUNT);
}

void *rig_buffer_host(value b) {
  value mem = Field(b, BUFFER_MEM);
  intnat host = Long_val(Field(mem, MEMORY_HOST));
  if (host < 0) return NULL;
  return (char *)host + Long_val(Field(b, BUFFER_OFFSET));
}

/* An io memory's published pages add a local host alias to its stable span.
   Looking for that alias never asks its device to map any pages. */
static intnat cached_host(value b) {
  value entry = load_field(alloc_of(b), ALLOC_ENTRY);
  /* [pages] publishes its bigarray with caml_modify's release store. There
     is no claim or device lock here to acquire that publication for us. */
  value pages = atomic_load_explicit((_Atomic value *)&Field(entry, ENTRY_PAGES),
                                     memory_order_acquire);
  if (Is_block(pages))
    return (intnat)Caml_ba_data_val(Field(pages, 0)) +
           Long_val(Field(b, BUFFER_OFFSET));
  return 0;
}

size_t rig_buffer_bytes(value b) {
  return (size_t)Long_val(Field(b, BUFFER_LENGTH));
}

/* Memory of this process's host lies at its host address. Other memory is
   named by its entry's stamps, which no other memory shares: a device's
   addresses or handles would not do, as memories of a handle-named device
   lie at handles a few bytes apart. An io memory retains these coordinates
   when its pages become host-addressable through cached_host. */
void rig_buffer_span(value b, intnat *space, intnat *first) {
  value alloc = alloc_of(b);
  intnat host = Long_val(Field(alloc, ALLOC_HOST));
  intnat offset = Long_val(Field(b, BUFFER_OFFSET));
  value machine = Field(Field(alloc, ALLOC_OWNER), DEVICE_MACHINE);
  if (host >= 0 && machine == MACHINE_HERE) {
    *space = 0;
    *first = host + offset;
    return;
  }
  *space = Long_val(load_field(load_field(alloc, ALLOC_ENTRY), ENTRY_STAMPS));
  *first = offset;
}

/* [b]'s bytes and where they lie, as struct rig_side holds them. */
static void side_span(value b, struct rig_side *s) {
  intnat space, first;
  rig_buffer_span(b, &space, &first);
  s->space = space;
  s->first = first;
  s->alias = space == 0 ? 0 : cached_host(b);
  s->bytes = rig_buffer_bytes(b);
}

int rig_sides_overlap(const struct rig_side *a, const struct rig_side *b) {
  if (a->bytes == 0 || b->bytes == 0) return 0;
  intptr_t x = a->first, y = b->first;
  if (a->space != b->space) {
    if (a->space != 0) x = a->alias;
    if (b->space != 0) y = b->alias;
    if (x == 0 || y == 0) return 0;
  }
  return x < y + (intptr_t)b->bytes && y < x + (intptr_t)a->bytes;
}

void rig_buffer_side(value b, int local, struct rig_side *s) {
  value mem = Field(b, BUFFER_MEM);
  side_span(b, s);
  s->handle = local ? (uint64_t)Long_val(Field(mem, MEMORY_HOST))
                    : (uint64_t)Nativeint_val(Field(mem, MEMORY_HANDLE));
  s->offset = (uint64_t)Long_val(Field(b, BUFFER_OFFSET));
  s->local = local;
}

/* [b]'s side narrowed to its [bytes] bytes from its byte [first]. A range
   outside [b] is the caller's bug, which aborts as a release of no claim
   does. */
static void side_range(value b, intnat first, intnat bytes,
                       struct rig_side *s) {
  intnat n = (intnat)rig_buffer_bytes(b);
  if (first < 0 || bytes < 0 || first > n || bytes > n - first)
    caml_fatal_error("rig_buffers_overlap: %" ARCH_INTNAT_PRINTF_FORMAT
                     "d bytes from byte %" ARCH_INTNAT_PRINTF_FORMAT
                     "d lie outside the buffer's %" ARCH_INTNAT_PRINTF_FORMAT
                     "d",
                     bytes, first, n);
  side_span(b, s);
  s->first += first;
  if (s->alias != 0) s->alias += first;
  s->bytes = (size_t)bytes;
}

int rig_buffers_overlap(value a, intnat a_first, intnat a_bytes, value b,
                        intnat b_first, intnat b_bytes) {
  struct rig_side x, y;
  side_range(a, a_first, a_bytes, &x);
  side_range(b, b_first, b_bytes, &y);
  return rig_sides_overlap(&x, &y);
}

value caml_rig_buffer_overlaps(value b, value other) {
  intnat n = (intnat)rig_buffer_bytes(b), m = (intnat)rig_buffer_bytes(other);
  return Val_bool(rig_buffers_overlap(b, 0, n, other, 0, m));
}

intnat caml_rig_cached_host(value b) { return cached_host(b); }

value caml_rig_cached_host_byte(value b) {
  return Val_long(caml_rig_cached_host(b));
}

/* rig_buffer_span for claim.ml, whose overlap check sorts by it. */
intnat caml_rig_span_space(value b) {
  intnat space, first;
  rig_buffer_span(b, &space, &first);
  return space;
}

intnat caml_rig_span_first(value b) {
  intnat space, first;
  rig_buffer_span(b, &space, &first);
  return first;
}

value caml_rig_span_space_byte(value b) {
  return Val_long(caml_rig_span_space(b));
}

value caml_rig_span_first_byte(value b) {
  return Val_long(caml_rig_span_first(b));
}

const char *rig_buffer_why(value b) {
  value alloc = alloc_of(b);
  value g = atomic_load_explicit((_Atomic value *)&Field(alloc, ALLOC_GEN),
                                 memory_order_acquire);
  if (Long_val(g) == Long_val(Field(b, BUFFER_GEN))) return NULL;
  return String_val(Field(alloc, ALLOC_WHY));
}

/* Claims */

/* Whether the device [d] is lost. Its slot's line answers while it names
   the device ([struct rig_line]): a device whose slot passed on has
   ended. */
static int dev_lost(value d) {
  struct rig_line *l = &rig_lines[Int_val(Field(d, DEVICE_INDEX))];
  int s = atomic_load_explicit(&l->state, memory_order_acquire);
  return s != RIG_LIVE ||
         atomic_load_explicit(&l->owner, memory_order_acquire) !=
             Long_val(Field(d, DEVICE_OPENED));
}

/* What [a], the answer for the points before [p], becomes with [p]: a
   point a lost device did not reach answers RIG_LOST whatever else does,
   so a wait gets past none; one not done answers RIG_WAIT. */
static enum rig_claim follow(enum rig_claim a, uint64_t p) {
  if (a == RIG_LOST || rig_point_done(p)) return a;
  return rig_point_lost(p) ? RIG_LOST : RIG_WAIT;
}

/* The answer for the work of the stamps [s] an access must follow: that of
   each device's last write, or of every use if [every], and of the point
   folded into the head, read after the words. Out of line, so that a claim
   of memory with no stamps saves no registers for it. */
static __attribute__((noinline)) enum rig_claim follow_stamps(struct rig_stamps *s, int every) {
  struct rig_stamps *head = s;
  enum rig_claim a = RIG_CLAIMED;
  for (int end = 0; s != NULL && a != RIG_LOST && !end;
       s = atomic_load_explicit(&s->next, memory_order_acquire))
    for (int i = 0; i < RIG_USES && !end; i++) {
      int64_t p = rig_stamps_followed(s, i, every);
      if (p < 0) end = 1;
      else if (p != 0) a = follow(a, (uint64_t)p);
    }
  uint64_t folded = rig_stamps_folded(head, every);
  return folded != 0 ? follow(a, folded) : a;
}

/* The first point of the stamps [v_st] a lost device did not reach: of
   their uses if [v_every], of their writes otherwise, then the point folded
   into their head; 0 for none. A wait that would block reads it first, so
   it waits out no work ahead of a loss. */
value caml_rig_stamps_lost(value v_st, value v_every) {
  int every = Bool_val(v_every);
  struct rig_stamps *head = (struct rig_stamps *)Long_val(v_st);
  int end = 0;
  for (struct rig_stamps *s = head; s != NULL && !end;
       s = atomic_load_explicit(&s->next, memory_order_acquire))
    for (int i = 0; i < RIG_USES && !end; i++) {
      int64_t p = rig_stamps_followed(s, i, every);
      if (p < 0) end = 1;
      else if (p != 0 && rig_point_lost((uint64_t)p))
        return Val_long((intnat)p);
    }
  uint64_t p = rig_stamps_folded(head, every);
  return Val_long(p != 0 && rig_point_lost(p) ? (intnat)p : 0);
}

/* The point folded into the head of the stamps [v_st] that an access
   follows, its uses' if [v_every], its writes' otherwise; 0 for none. */
value caml_rig_stamps_folded(value v_st, value v_every) {
  return Val_long((intnat)rig_stamps_folded(
      (struct rig_stamps *)Long_val(v_st), Bool_val(v_every)));
}

/* Claims first, then checks [b] under the claim: the compare-and-set
   follows the release of any donation that consumed the memory, so the
   checks see its consumption and its stamps, those of every use if
   [every], else of every device's last write. Work that is not done keeps
   the claim: the caller waits under it, so no donation consumes the memory
   between the wait and the access, and a loss's wait raises it. Inlined
   into each claim of a list's buffers. */
static inline __attribute__((always_inline)) enum rig_claim
claim_checked(value b, enum rig_access access, int every) {
  value dev = Field(Field(b, BUFFER_MEM), MEMORY_DEV);
  value alloc = alloc_of(b);
  _Atomic value *word = (_Atomic value *)&Field(alloc, ALLOC_COUNT);
  /* A buffer admits only reads iff its generation is odd. */
  if (access == RIG_READ_WRITE && (Long_val(Field(b, BUFFER_GEN)) & 1))
    return RIG_READ_ONLY;
  value w = atomic_load_explicit(word, memory_order_relaxed);
  do {
    if (Long_val(w) < 0) return RIG_EXCLUSIVE;
  } while (!atomic_compare_exchange_weak(word, &w,
                                         Val_long(Long_val(w) + CLAIM_ONE)));
  if (Long_val(load_field(alloc, ALLOC_GEN)) != Long_val(Field(b, BUFFER_GEN))) {
    rig_buffer_release(b);
    return RIG_DEAD;
  }
  value owner = Field(alloc, ALLOC_OWNER);
  if (dev_lost(dev) || (owner != dev && dev_lost(owner))) return RIG_LOST;
  value entry = load_field(alloc, ALLOC_ENTRY);
  struct rig_stamps *s =
      (struct rig_stamps *)Long_val(load_field(entry, ENTRY_STAMPS));
  return s == NULL ? RIG_CLAIMED : follow_stamps(s, every);
}

enum rig_claim rig_buffer_claim(value b, enum rig_access access) {
  return claim_checked(b, access, access == RIG_READ_WRITE);
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

/* A read claim on [v_b]'s memory, with no check of [v_b], for
   Memory.claim_add once another domain's claim beat its compare-and-set:
   false, claiming nothing, if the memory is held exclusive. The loop is in
   C so nothing polls between its read and its claim. */
value caml_rig_claim_add(value v_b) {
  _Atomic value *word = claim_word(v_b);
  value w = atomic_load_explicit(word, memory_order_relaxed);
  do {
    if (Long_val(w) < 0) return Val_false;
  } while (!atomic_compare_exchange_weak_explicit(
      word, &w, Val_long(Long_val(w) + CLAIM_ONE), memory_order_acquire,
      memory_order_relaxed));
  return Val_true;
}

/* Ends a read claim on [b]'s memory: one subtraction of the tagged step.
   If the word's read claims were none, the word the subtraction left matches
   none of the claims' states, so the process aborts. */
static void claim_end(value b, const char *who) {
  value w = atomic_fetch_sub_explicit(
      claim_word(b), Val_long(CLAIM_ONE) - Val_long(0), memory_order_release);
  if (Long_val(w) < 0 || (Long_val(w) & (CLAIM_HOLD - 1)) < CLAIM_ONE)
    caml_fatal_error("%s: no read claim to end", who);
}

/* Ends a read claim of Memory.claim_add. */
value caml_rig_claim_drop(value v_b) {
  claim_end(v_b, "caml_rig_claim_drop");
  return Val_unit;
}

void rig_buffer_release(value b) { claim_end(b, "rig_buffer_release"); }

void rig_buffers_release(int n, const value *buffers) {
  for (int k = 0; k < n; k++) claim_end(buffers[k], "rig_buffers_release");
}

int rig_buffers_claim(int n, const value *buffers,
                      const enum rig_access *access,
                      enum rig_claim *answer) {
  for (int k = 0; k < n; k++) {
    enum rig_claim a = rig_buffer_claim(buffers[k], access[k]);
    answer[k] = a;
    if (a == RIG_DEAD || a == RIG_EXCLUSIVE || a == RIG_READ_ONLY) {
      rig_buffers_release(k, buffers);
      return k;
    }
  }
  return -1;
}

/* Buffer.free's step, as a donation's: a read claim, a check of [v_b] under
   it, then the memory exclusive if that claim is its only one and no hold
   or bit is on its word. It then raises the generation from [v_b]'s own
   with the reason [v_why], and leaves the word empty. Exclusive meanwhile,
   the memory refuses every claim, so a claim that follows finds [v_b]
   dead. It is 1 if it consumed the memory, 0 if the word refused it, and 2
   if [v_b] is dead, holding nothing. */
value caml_rig_buffer_free(value v_b, value v_why) {
  _Atomic value *word = claim_word(v_b);
  value w = atomic_load_explicit(word, memory_order_relaxed);
  do {
    if (Long_val(w) < 0) return Val_long(0);
  } while (!atomic_compare_exchange_weak(word, &w,
                                         Val_long(Long_val(w) + CLAIM_ONE)));
  value alloc = alloc_of(v_b);
  _Atomic value *gen = (_Atomic value *)&Field(alloc, ALLOC_GEN);
  value g = Field(v_b, BUFFER_GEN);
  if (atomic_load(gen) != g) {
    claim_end(v_b, "Rig.Buffer.free");
    return Val_long(2);
  }
  value one = Val_long(CLAIM_ONE);
  if (!atomic_compare_exchange_strong(word, &one,
                                      Val_long(CLAIM_EXCLUSIVE))) {
    claim_end(v_b, "Rig.Buffer.free");
    return Val_long(0);
  }
  caml_modify(&Field(alloc, ALLOC_WHY), v_why);
  atomic_store(gen, Val_long(Long_val(g) + 1));
  atomic_store_explicit(word, Val_long(0), memory_order_release);
  return Val_long(1);
}

/* Holds

   A hold adds CLAIM_HOLD to the word unless the memory is held exclusive,
   and waits out a consumption under way, which checks for holds and raises
   the generation with CLAIM_LOCKED set: so either the consumption finds the
   hold, or the hold's check finds the buffer dead. A hold's record marks each
   hold taken with a byte, which the steps below set and clear with the word,
   so each hold is released once. */

static void unhold_word(_Atomic value *word) {
  value w = atomic_fetch_sub_explicit(
      word, Val_long(CLAIM_HOLD) - Val_long(0), memory_order_release);
  if (Long_val(w) < CLAIM_HOLD)
    caml_fatal_error("Rig.Claim.hold: no hold to end");
}

/* A hold of [b]'s memory, checked under it: false, holding nothing, if the
   memory is held exclusive or [b] is dead. */
static int hold_one(value b) {
  _Atomic value *word = claim_word(b);
  value w = atomic_load_explicit(word, memory_order_relaxed);
  for (;;) {
    if (Long_val(w) < 0) return 0;
    if (Long_val(w) & CLAIM_LOCKED) {
      relax();
      w = atomic_load_explicit(word, memory_order_relaxed);
      continue;
    }
    if (atomic_compare_exchange_weak(word, &w,
                                     Val_long(Long_val(w) + CLAIM_HOLD)))
      break;
  }
  if (Long_val(load_field(alloc_of(b), ALLOC_GEN)) !=
      Long_val(Field(b, BUFFER_GEN))) {
    unhold_word(word);
    return 0;
  }
  return 1;
}

/* Holds the memory of each buffer [i] of [v_bs], and marks byte [i] of
   [v_taken]: -1, or the index of the buffer refused, holding nothing. */
value caml_rig_claim_holds(value v_bs, value v_taken) {
  unsigned char *taken = Bytes_val(v_taken);
  intnat i = 0;
  for (value l = v_bs; l != Val_emptylist; l = Field(l, 1), i++) {
    if (!hold_one(Field(l, 0))) {
      intnat n = 0;
      for (value k = v_bs; n < i; k = Field(k, 1), n++) {
        unhold_word(claim_word(Field(k, 0)));
        taken[n] = 0;
      }
      return Val_long(i);
    }
    taken[i] = 1;
  }
  return Val_long(-1);
}

/* Ends the hold [v_i] of [v_taken] on the allocation [v_alloc]'s memory, if
   it is taken. The mark is cleared in one exchange, so of two releases on two
   domains one ends the hold. */
value caml_rig_claim_unhold(value v_taken, value v_i, value v_alloc) {
  _Atomic unsigned char *taken =
      (_Atomic unsigned char *)(Bytes_val(v_taken) + Long_val(v_i));
  if (atomic_exchange_explicit(taken, 0, memory_order_relaxed) == 0)
    return Val_unit;
  unhold_word((_Atomic value *)&Field(v_alloc, ALLOC_COUNT));
  return Val_unit;
}

/* Claim.with_'s claims, taken in one call and released in one, neither
   running OCaml. */

/* Ends the read claims of the first [n] buffers of [read] followed by
   [donate]. */
static void end_claims(value read, value donate, intnat n) {
  value lists[2] = {read, donate};
  for (int k = 0; k < 2; k++)
    for (value l = lists[k]; l != Val_emptylist && n > 0;
         l = Field(l, 1), n--)
      claim_end(Field(l, 0), "Rig.Claim.with_");
}

/* A read claim on [b]'s memory, checked under the claim as Memory.check
   checks it: false, claiming nothing, if the memory is held exclusive, [b]
   is dead, or a lost device owns it or did not reach work on it. */
static int claim_live(value b) {
  enum rig_claim a = claim_checked(b, RIG_READ, 1);
  if (a == RIG_LOST) rig_buffer_release(b);
  return a == RIG_CLAIMED || a == RIG_WAIT;
}

/* Claims every buffer of [v_read], then of [v_donate], for reading: -1, or
   the index across both lists of the buffer that refused its claim, every
   claim undone. */
value caml_rig_claim_all(value v_read, value v_donate) {
  value lists[2] = {v_read, v_donate};
  intnat i = 0;
  for (int k = 0; k < 2; k++)
    for (value l = lists[k]; l != Val_emptylist; l = Field(l, 1), i++)
      if (!claim_live(Field(l, 0))) {
        end_claims(v_read, v_donate, i);
        return Val_long(i);
      }
  return Val_long(-1);
}

/* The fields of Claim.t. Its state is twice the consumptions under way,
   plus 1 once its with_ has ended. A consumption adds 2, and backs out if
   the end came first; the end adds 1, and waits if a consumption came
   first. */
enum { CLAIMS_READ, CLAIMS_DONATE, CLAIMS_STATE };
#define CLAIMS_CONSUMING (Val_long(2) - Val_long(0))

static _Atomic value *claims_state(value v_c) {
  return (_Atomic value *)&Field(v_c, CLAIMS_STATE);
}

/* The generation a consumption raises [g] to: the next even one for memory
   held exclusive, whose new buffer admits writes, and the next odd one
   otherwise, whose new buffer admits only reads (Memory.admits). */
static intnat next_generation(intnat g, int exclusive) {
  return ((g >> 1) + 1) * 2 + (exclusive ? 0 : 1);
}

/* What the step answers when the claims have ended, when [v_b] is dead
   (another consumption of its memory came first), or when a hold holds the
   memory. */
#define CONSUME_ENDED (-1)
#define CONSUME_DEAD (-2)
#define CONSUME_HELD (-3)

/* Claim.consume's effects, as one step against the end of the claims
   [v_c]: raises the generation of [v_b]'s memory from [v_b]'s own,
   records [v_why] and gives an exclusive word back to the claims' one read
   claim. Memory not held exclusive is locked against holds meanwhile. The
   new generation, or CONSUME_ENDED, CONSUME_DEAD or CONSUME_HELD, doing
   nothing. */
value caml_rig_claim_consume(value v_c, value v_b, value v_why) {
  _Atomic value *state = claims_state(v_c);
  if (Long_val(atomic_fetch_add(state, CLAIMS_CONSUMING)) & 1) {
    atomic_fetch_sub(state, CLAIMS_CONSUMING);
    return Val_long(CONSUME_ENDED);
  }
  value alloc = alloc_of(v_b);
  _Atomic value *word = claim_word(v_b);
  _Atomic value *gen = (_Atomic value *)&Field(alloc, ALLOC_GEN);
  value g = Field(v_b, BUFFER_GEN);
  int locked = 0;
  value w = atomic_load(word);
  while (!locked && Long_val(w) >= 0) {
    if (Long_val(w) >= CLAIM_HOLD) {
      atomic_fetch_sub(state, CLAIMS_CONSUMING);
      return Val_long(CONSUME_HELD);
    }
    if (Long_val(w) & CLAIM_LOCKED) {
      relax();
      w = atomic_load(word);
    } else
      locked = atomic_compare_exchange_weak(
          word, &w, Val_long(Long_val(w) | CLAIM_LOCKED));
  }
  /* The reason before the generation, so that a buffer found dead shows
     it; a consumption that loses the race below wrote it too. The
     generation before the word: a claim whose compare-and-set reads the
     word given back follows the raise, so its check finds every earlier
     buffer dead. Word first, such a claim would pass its check and read
     beside the consumer's writes. */
  if (atomic_load(gen) == g) caml_modify(&Field(alloc, ALLOC_WHY), v_why);
  int raised = atomic_compare_exchange_strong(
      gen, &g, Val_long(next_generation(Long_val(g), !locked)));
  if (locked)
    atomic_fetch_sub(word, Val_long(CLAIM_LOCKED) - Val_long(0));
  else {
    w = Val_long(CLAIM_EXCLUSIVE);
    if (raised) atomic_compare_exchange_strong(word, &w, Val_long(CLAIM_ONE));
  }
  atomic_fetch_sub(state, CLAIMS_CONSUMING);
  return raised ? Val_long(next_generation(Long_val(g), !locked))
                : Val_long(CONSUME_DEAD);
}

/* Waits until the claims [v_c], which have ended, have no consumption
   under way. */
value caml_rig_claim_await(value v_c) {
  _Atomic value *state = claims_state(v_c);
  while (atomic_load(state) != Val_long(1)) {
#ifdef _WIN32
    SwitchToThread();
#else
    sched_yield();
#endif
  }
  return Val_unit;
}

/* Gives each memory of [v_donate] still held exclusive back to its read
   claim, then ends the read claims of [v_read] and [v_donate]. A word is
   still exclusive where the claims did not consume the memory. */
value caml_rig_claim_release_all(value v_read, value v_donate) {
  for (value l = v_donate; l != Val_emptylist; l = Field(l, 1)) {
    value w = Val_long(CLAIM_EXCLUSIVE);
    atomic_compare_exchange_strong(claim_word(Field(l, 0)), &w,
                                   Val_long(CLAIM_ONE));
  }
  end_claims(v_read, v_donate, Max_long);
  return Val_unit;
}
