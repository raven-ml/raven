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

#ifndef _WIN32
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

static int memory_room(void *self, const struct rig_part *parts, int n,
                       void *scratch, const uint8_t *args) {
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

/* Runs the parts in order, the fills with no queue context, then makes [v]
   observable. Its handles are host addresses. */
static int memory_submit(void *self, uint64_t v, const struct rig_wait *waits,
                         int nwaits, const struct rig_part *parts, int nparts,
                         void *scratch, const uint8_t *args,
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
    } else if (p->copy.bytes != 0)
      memmove((char *)(uintptr_t)p->copy.dst + p->copy.dst_offset,
              (const char *)(uintptr_t)p->copy.src + p->copy.src_offset,
              (size_t)p->copy.bytes);
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
   memory { dev; bytes; host; address; handle; claim; entry; root; ... },
   its claim { count; generation; why } and the entry of its root
   { owner; backing; bytes; access; stamps; ... },
   as rig's Def module lays them out. */

enum { DEVICE_INDEX, DEVICE_NAME, DEVICE_MACHINE };
enum { BUFFER_MEM, BUFFER_OFFSET, BUFFER_LENGTH, BUFFER_GEN };
enum { MEMORY_DEV, MEMORY_BYTES, MEMORY_HOST, MEMORY_ADDRESS, MEMORY_HANDLE,
       MEMORY_CLAIM, MEMORY_ENTRY, MEMORY_ROOT };
enum { CLAIM_COUNT, CLAIM_GEN, CLAIM_WHY };

/* A device's machine when it is this process's: [Here], the machine type's
   first constant constructor. */
#define MACHINE_HERE Val_int(0)
enum { ENTRY_OWNER, ENTRY_BACKING, ENTRY_BYTES, ENTRY_ACCESS, ENTRY_STAMPS,
       ENTRY_MAPS, ENTRY_UNMAPS, ENTRY_PAGES };

/* The claim word as rig's Memory module lays it out: from 0 up, bits for
   memory outside the claims and memory that admits only reads, and the
   step per read claim; -1, held exclusive. */
#define CLAIM_READ_ONLY 2
#define CLAIM_ONE 4
#define CLAIM_EXCLUSIVE (-1)

/* A field another domain writes. Read after the claim's compare-and-set,
   which acquires, so it shows what the release of any earlier claim
   published. */
static value load_field(value v, int i) {
  return atomic_load_explicit((_Atomic value *)&Field(v, i),
                              memory_order_relaxed);
}

/* The root of the memory record [mem], as rig's Memory.root reads it: an
   owner's record holds Memory.itself, the one record whose root is itself,
   in place of a pointer back to it. */
static value root_of(value mem) {
  value root = Field(mem, MEMORY_ROOT);
  return Field(root, MEMORY_ROOT) == root ? mem : root;
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
  value root = root_of(Field(b, BUFFER_MEM));
  value entry = load_field(root, MEMORY_ENTRY);
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
   named by its root's stamps, which no other memory shares: a device's
   addresses or handles would not do, as memories of a handle-named device
   lie at handles a few bytes apart. An io memory retains these coordinates
   when its pages become host-addressable through cached_host. */
void rig_buffer_span(value b, intnat *space, intnat *first) {
  value root = root_of(Field(b, BUFFER_MEM));
  intnat host = Long_val(Field(root, MEMORY_HOST));
  intnat offset = Long_val(Field(b, BUFFER_OFFSET));
  value machine = Field(Field(root, MEMORY_DEV), DEVICE_MACHINE);
  if (host >= 0 && machine == MACHINE_HERE) {
    *space = 0;
    *first = host + offset;
    return;
  }
  *space = Long_val(load_field(load_field(root, MEMORY_ENTRY), ENTRY_STAMPS));
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

value caml_rig_buffer_overlaps(value b, value other) {
  struct rig_side x, y;
  side_span(b, &x);
  side_span(other, &y);
  return Val_bool(rig_sides_overlap(&x, &y));
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
  int i = Int_val(Field(Field(mem, MEMORY_DEV), DEVICE_INDEX));
  return atomic_load_explicit(&rig_lines[i].state, memory_order_acquire) !=
         RIG_LIVE;
}

/* What [a], the answer for the points before [p], becomes with [p]: a
   point a lost device did not reach answers RIG_LOST whatever else does,
   so a wait gets past none; one not done answers RIG_WAIT. */
static enum rig_claim follow(enum rig_claim a, uint64_t p) {
  if (a == RIG_LOST || rig_point_done(p)) return a;
  return rig_point_lost(p) ? RIG_LOST : RIG_WAIT;
}

/* The answer for the work of the stamps [s] an access must follow: that of
   each device's last write, or of every use if [every]. Out of line, so
   that a claim of memory with no stamps saves no registers for it. */
static __attribute__((noinline)) enum rig_claim follow_stamps(struct rig_stamps *s, int every) {
  enum rig_claim a = RIG_CLAIMED;
  for (; s != NULL && a != RIG_LOST;
       s = atomic_load_explicit(&s->next, memory_order_acquire))
    for (int i = 0; i < RIG_USES; i++) {
      int64_t p = rig_stamps_followed(s, i, every);
      if (p < 0) return a;
      if (p != 0) a = follow(a, (uint64_t)p);
    }
  return a;
}

/* The first point of the stamps [v_st] a lost device did not reach: of
   their uses if [v_every], of their writes otherwise; 0 for none. A wait
   that would block reads it first, so it waits out no work ahead of a
   loss. */
value caml_rig_stamps_lost(value v_st, value v_every) {
  int every = Bool_val(v_every);
  for (struct rig_stamps *s = (struct rig_stamps *)Long_val(v_st); s != NULL;
       s = atomic_load_explicit(&s->next, memory_order_acquire))
    for (int i = 0; i < RIG_USES; i++) {
      int64_t p = rig_stamps_followed(s, i, every);
      if (p < 0) return Val_long(0);
      if (p != 0 && rig_point_lost((uint64_t)p)) return Val_long((intnat)p);
    }
  return Val_long(0);
}

/* Claims first, then checks [b] under the claim: the compare-and-set
   follows the release of any donation that consumed the memory, so the
   checks see its consumption and its stamps, those of every use if
   [every], else of every device's last write. Work that is not done keeps
   the claim: the caller waits under it, so no donation consumes the memory
   between the wait and the access, and a loss's wait raises it. */
static inline enum rig_claim claim_checked(value b, enum rig_access access,
                                           int every) {
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
  value root = root_of(mem);
  if (dev_lost(mem) || (root != mem && dev_lost(root))) return RIG_LOST;
  value entry = load_field(root, MEMORY_ENTRY);
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

/* A read claim on [v_b]'s memory, with no check of [v_b], for a submit,
   which checks its buffers under the claims: false, claiming nothing, if
   the memory is held exclusive. */
value caml_rig_claim_hold(value v_b) {
  value claim = Field(Field(v_b, BUFFER_MEM), MEMORY_CLAIM);
  _Atomic value *word = (_Atomic value *)&Field(claim, CLAIM_COUNT);
  value w = atomic_load_explicit(word, memory_order_relaxed);
  do {
    if (Long_val(w) < 0) return Val_false;
  } while (!atomic_compare_exchange_weak_explicit(
      word, &w, Val_long(Long_val(w) + CLAIM_ONE), memory_order_acquire,
      memory_order_relaxed));
  return Val_true;
}

/* Ends a read claim on [b]'s memory: one subtraction of the tagged step.
   If the word was below CLAIM_ONE, it held no read claim, and the word the
   subtraction left matches none of the claims' states, so the process
   aborts. */
static void claim_end(value b, const char *who) {
  value claim = Field(Field(b, BUFFER_MEM), MEMORY_CLAIM);
  value w = atomic_fetch_sub_explicit(
      (_Atomic value *)&Field(claim, CLAIM_COUNT),
      Val_long(CLAIM_ONE) - Val_long(0), memory_order_release);
  if (Long_val(w) < CLAIM_ONE)
    caml_fatal_error("%s: no read claim to end", who);
}

/* Ends a claim of [caml_rig_claim_hold]. */
value caml_rig_claim_drop(value v_b) {
  claim_end(v_b, "caml_rig_claim_drop");
  return Val_unit;
}

/* End the acquired prefix without an OCaml loop's interruptible poll. */
value caml_rig_claim_drop_prefix(value v_buffers, value v_n) {
  intnat n = Long_val(v_n);
  for (intnat i = 0; i < n; i++)
    caml_rig_claim_drop(Field(v_buffers, i));
  return Val_unit;
}

void rig_buffer_release(value b) { claim_end(b, "rig_buffer_release"); }

/* Claim.with_'s claims, taken in one call and released in one, neither
   running OCaml. */

static _Atomic value *claim_word(value b) {
  return (_Atomic value *)&Field(Field(Field(b, BUFFER_MEM), MEMORY_CLAIM),
                                 CLAIM_COUNT);
}

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

/* What the step answers when the claims have ended, or when [v_b] is
   dead: another consumption of its memory came first. */
#define CONSUME_ENDED (-1)
#define CONSUME_DEAD (-2)

/* Claim.consume's effects, as one step against the end of the claims
   [v_c]: raises the generation of [v_b]'s memory from [v_b]'s own,
   records [v_why] and gives an exclusive word back to the claims' one read
   claim. The new generation, or CONSUME_ENDED or CONSUME_DEAD, doing
   nothing. */
value caml_rig_claim_consume(value v_c, value v_b, value v_why) {
  _Atomic value *state = claims_state(v_c);
  if (Long_val(atomic_fetch_add(state, CLAIMS_CONSUMING)) & 1) {
    atomic_fetch_sub(state, CLAIMS_CONSUMING);
    return Val_long(CONSUME_ENDED);
  }
  value claim = Field(Field(v_b, BUFFER_MEM), MEMORY_CLAIM);
  _Atomic value *gen = (_Atomic value *)&Field(claim, CLAIM_GEN);
  value g = Field(v_b, BUFFER_GEN);
  /* The reason before the generation, so that a buffer found dead shows
     it; a consumption that loses the race below wrote it too. The
     generation before the word: a claim whose compare-and-set reads the
     word given back follows the raise, so its check finds every earlier
     buffer dead. Word first, such a claim would pass its check and read
     beside the consumer's writes. */
  if (atomic_load(gen) == g) caml_modify(&Field(claim, CLAIM_WHY), v_why);
  if (!atomic_compare_exchange_strong(gen, &g, Val_long(Long_val(g) + 1))) {
    atomic_fetch_sub(state, CLAIMS_CONSUMING);
    return Val_long(CONSUME_DEAD);
  }
  value w = Val_long(CLAIM_EXCLUSIVE);
  atomic_compare_exchange_strong(claim_word(v_b), &w, Val_long(CLAIM_ONE));
  atomic_fetch_sub(state, CLAIMS_CONSUMING);
  return Val_long(Long_val(g) + 1);
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
