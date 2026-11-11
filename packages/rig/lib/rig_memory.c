/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The readers of buffers rig.h declares. Nothing here blocks but
   rig_buffer_wait, which runs Rig.Buffer.wait. A claim that meets a take or
   a free under way spins until that step ends. */

#define _GNU_SOURCE

#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/callback.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "rig.h"
#include "rig_stubs.h"

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
   memory { dev; host; address; handle; alloc; lease }, its lease, an option,
   { count; why; ... }, its allocation
   { owner; host; count; generation; why; entry; ... }, the allocation's
   entry { owner; backing; bytes; access; stamps; ... } and a device
   { index; name; machine; opened; ... }, as rig's Def module lays them
   out. */

enum { DEVICE_INDEX, DEVICE_NAME, DEVICE_MACHINE, DEVICE_OPENED };
enum { BUFFER_MEM, BUFFER_OFFSET, BUFFER_LENGTH, BUFFER_GEN };
enum { MEMORY_DEV, MEMORY_HOST, MEMORY_ADDRESS, MEMORY_HANDLE, MEMORY_ALLOC,
       MEMORY_LEASE };
enum { LEASE_COUNT, LEASE_WHY };

/* What a lease's end adds to its count, past every count of its claims, as
   Memory.lease_end. */
#define LEASE_END ((intnat)1 << 40)
enum { ALLOC_OWNER, ALLOC_HOST, ALLOC_COUNT, ALLOC_GEN, ALLOC_WHY,
       ALLOC_ENTRY };

/* A device's machine when it is this process's: [Here], the machine type's
   first constant constructor. */
#define MACHINE_HERE Val_int(0)
enum { ENTRY_OWNER, ENTRY_BACKING, ENTRY_BYTES, ENTRY_ACCESS, ENTRY_STAMPS,
       ENTRY_MAPS, ENTRY_UNMAPS, ENTRY_PAGES };

/* The claim word as rig's Memory module lays it out: from 0 up, a bit for
   memory outside the claims, the step per read claim from bit 1 and the step
   per hold from bit 32; CLAIM_TAKING while a take or a free holds the
   memory, which only an empty word becomes. */
#define CLAIM_ONE 2
#define CLAIM_HOLD ((intnat)1 << 32)
#define CLAIM_TAKING (-1)

/* A spin's pause while another domain's take or free holds a word at
   CLAIM_TAKING: the core's hint that it spins, as rig's pool spins. */
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

/* The lease of [b]'s memory record, an option: [Some] of a borrow's. */
static value lease_opt(value b) {
  return Field(Field(b, BUFFER_MEM), MEMORY_LEASE);
}

/* The lease of a borrow's buffer [b]. */
static value lease_of(value b) { return Field(lease_opt(b), 0); }

static _Atomic value *lease_count(value b) {
  return (_Atomic value *)&Field(lease_of(b), LEASE_COUNT);
}

/* Whether an unborrow ended [b]'s lease. It stores the reason first. */
static int lease_ended(value b) {
  return Is_block(lease_opt(b)) &&
         Long_val(atomic_load_explicit(lease_count(b), memory_order_acquire)) >=
             LEASE_END;
}

/* Counts a claim or a hold through a borrow's [b] on its lease, which an
   unborrow refuses while it holds, and ends one. A count past an end only
   marks claims that will find the borrow dead. Other buffers have none. */
static void lease_take(value b) {
  if (Is_block(lease_opt(b)))
    atomic_fetch_add_explicit(lease_count(b), Val_long(1) - Val_long(0),
                              memory_order_acquire);
}

static void lease_give(value b) {
  if (Is_block(lease_opt(b)))
    atomic_fetch_sub_explicit(lease_count(b), Val_long(1) - Val_long(0),
                              memory_order_release);
}

static _Atomic value *claim_word(value b) {
  return (_Atomic value *)&Field(alloc_of(b), ALLOC_COUNT);
}

/* Adds [step] to [word] once no take or free holds it at CLAIM_TAKING,
   spinning while one does. The compare-and-set acquires, so the checks
   after it see what such a step published. The loop is in C, so nothing
   polls between its read and its add. */
static void add_settled(_Atomic value *word, intnat step) {
  value w = atomic_load_explicit(word, memory_order_relaxed);
  for (;;) {
    if (Long_val(w) < 0) {
      relax();
      w = atomic_load_explicit(word, memory_order_relaxed);
    } else if (atomic_compare_exchange_weak_explicit(
                   word, &w, Val_long(Long_val(w) + step),
                   memory_order_acquire, memory_order_relaxed))
      return;
  }
}

/* Whether [b] is dead: a take or free raised its memory's generation from
   [b]'s own, or an unborrow ended its lease. */
static int dead(value b) {
  return Long_val(load_field(alloc_of(b), ALLOC_GEN)) !=
             Long_val(Field(b, BUFFER_GEN)) ||
         lease_ended(b);
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

/* Whether [a] and [b] share a byte, or with [words] a word: each range
   rounded out to whole words of its space, by address or by offset. */
static int sides_meet(const struct rig_side *a, const struct rig_side *b,
                      int words) {
  if (a->bytes == 0 || b->bytes == 0) return 0;
  intptr_t x = a->first, y = b->first;
  if (a->space != b->space) {
    if (a->space != 0) x = a->alias;
    if (b->space != 0) y = b->alias;
    if (x == 0 || y == 0) return 0;
  }
  intptr_t x_end = x + (intptr_t)a->bytes, y_end = y + (intptr_t)b->bytes;
  if (words) {
    intptr_t w = RIG_WORD_BYTES;
    x &= -w;
    y &= -w;
    x_end = (x_end + w - 1) & -w;
    y_end = (y_end + w - 1) & -w;
  }
  return x < y_end && y < x_end;
}

int rig_sides_overlap(const struct rig_side *a, const struct rig_side *b) {
  return sides_meet(a, b, 0);
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

/* Whether the two whole buffers share a word, as Rig.Buffer.overlaps
   answers, or with [v_bytes] a byte; their ranges need no check. */
value caml_rig_buffer_meet(value b, value other, value v_bytes) {
  struct rig_side x, y;
  side_span(b, &x);
  side_span(other, &y);
  return Val_bool(sides_meet(&x, &y, !Bool_val(v_bytes)));
}

intnat caml_rig_cached_host(value b) { return cached_host(b); }

value caml_rig_cached_host_byte(value b) {
  return Val_long(caml_rig_cached_host(b));
}

/* rig_buffer_span for Buffer.places. */
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
  if (Long_val(g) != Long_val(Field(b, BUFFER_GEN)))
    return String_val(Field(alloc, ALLOC_WHY));
  if (lease_ended(b)) return String_val(Field(lease_of(b), LEASE_WHY));
  return NULL;
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
  uint64_t folded = rig_stamps_folded(head);
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
  uint64_t p = rig_stamps_folded(head);
  return Val_long(p != 0 && rig_point_lost(p) ? (intnat)p : 0);
}

/* The point folded into the head of the stamps [v_st] that every access
   follows; 0 for none. */
value caml_rig_stamps_folded(value v_st) {
  return Val_long(
      (intnat)rig_stamps_folded((struct rig_stamps *)Long_val(v_st)));
}

/* The work an access of the live [b] must follow, its every use if
   [every], else each device's last write: RIG_LOST if [b]'s device or its
   memory's owner is lost, else as follow_stamps answers. The caller holds
   [b]'s word, so no free returns the stamps meanwhile. */
static inline __attribute__((always_inline)) enum rig_claim
followed(value b, int every) {
  value dev = Field(Field(b, BUFFER_MEM), MEMORY_DEV);
  value alloc = alloc_of(b);
  value owner = Field(alloc, ALLOC_OWNER);
  if (dev_lost(dev) || (owner != dev && dev_lost(owner))) return RIG_LOST;
  value entry = load_field(alloc, ALLOC_ENTRY);
  struct rig_stamps *s =
      (struct rig_stamps *)Long_val(load_field(entry, ENTRY_STAMPS));
  return s == NULL ? RIG_CLAIMED : follow_stamps(s, every);
}

/* Claims first, then checks [b] under the claim: the compare-and-set
   follows the release of any take that raised the generation, so the
   checks see it and the stamps. Work that is not done keeps the claim: the
   caller waits under it, so no take writes the memory between the wait and
   the access, and a loss's wait raises it. Inlined into each claim of a
   list's buffers. */
static inline __attribute__((always_inline)) enum rig_claim
claim_checked(value b, enum rig_access access, int every) {
  /* A buffer admits only reads iff its generation is odd. */
  if (access == RIG_READ_WRITE && (Long_val(Field(b, BUFFER_GEN)) & 1))
    return RIG_READ_ONLY;
  add_settled(claim_word(b), CLAIM_ONE);
  lease_take(b);
  if (dead(b)) {
    rig_buffer_release(b);
    return RIG_DEAD;
  }
  return followed(b, every);
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
   Memory.claim_add once its own compare-and-set failed, to another
   domain's claim or to a take or free under way. */
value caml_rig_claim_add(value v_b) {
  add_settled(claim_word(v_b), CLAIM_ONE);
  return Val_unit;
}

/* Ends a read claim on [b]'s memory's word: one subtraction of the tagged
   step. If the word's read claims were none, the word the subtraction left
   matches none of the claims' states, so the process aborts. */
static void word_end(value b, const char *who) {
  value w = atomic_fetch_sub_explicit(
      claim_word(b), Val_long(CLAIM_ONE) - Val_long(0), memory_order_release);
  if (Long_val(w) < 0 || (Long_val(w) & (CLAIM_HOLD - 1)) < CLAIM_ONE)
    caml_fatal_error("%s: no read claim to end", who);
}

/* Ends a read claim through [b]: its lease's count, then the word's. */
static void claim_end(value b, const char *who) {
  lease_give(b);
  word_end(b, who);
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
    if (a == RIG_DEAD || a == RIG_READ_ONLY) {
      rig_buffers_release(k, buffers);
      return k;
    }
  }
  return -1;
}

/* Takes and frees

   A take or a free holds each word it needs at CLAIM_TAKING, which only an
   empty word becomes: no claim, hold or bit. Under it, the step checks its
   buffer, records the reason, raises the generation by 2 from the buffer's
   own, and gives the word back empty, generation first. A claim, hold or
   mark that meets CLAIM_TAKING spins until the word comes back, so its
   compare-and-set follows the raise and its check finds every earlier
   buffer dead. A step never waits while it holds a word, so whatever a
   spinner holds, the step it waits for ends. */

/* Raises [b]'s memory's generation by 2 from [b]'s own, with the reason
   [why] first, so that a buffer found dead shows it. Its buffers then admit
   what [b] admits (Memory.admits). The caller holds the word. */
static void raise_generation(value b, value why) {
  value alloc = alloc_of(b);
  caml_modify(&Field(alloc, ALLOC_WHY), why);
  atomic_store_explicit((_Atomic value *)&Field(alloc, ALLOC_GEN),
                        Val_long(Long_val(Field(b, BUFFER_GEN)) + 2),
                        memory_order_release);
}

/* Holds the empty [word] at CLAIM_TAKING: whether it was empty. Otherwise
   [*seen] is the word it found. */
static int hold_empty(_Atomic value *word, value *seen) {
  *seen = Val_long(0);
  return atomic_compare_exchange_strong_explicit(
      word, seen, Val_long(CLAIM_TAKING), memory_order_acquire,
      memory_order_relaxed);
}

static void give_back(_Atomic value *word) {
  atomic_store_explicit(word, Val_long(0), memory_order_release);
}

/* Spins until another step gives [word] back. */
static void await_given(_Atomic value *word) {
  while (Long_val(atomic_load_explicit(word, memory_order_relaxed)) < 0)
    relax();
}

/* Buffer.free's step: holds [v_b]'s word if it is empty, waiting out a take
   or free under way, checks [v_b] live under it, and raises the generation
   with the reason [v_why], so a claim that follows finds [v_b] dead. It is
   1 if it freed the memory, 0 if a claim, hold or bit kept it, and 2 if
   [v_b] is dead. */
value caml_rig_buffer_free(value v_b, value v_why) {
  _Atomic value *word = claim_word(v_b);
  value seen;
  while (!hold_empty(word, &seen)) {
    if (Long_val(seen) >= 0) return Val_long(dead(v_b) ? 2 : 0);
    await_given(word);
  }
  if (dead(v_b)) {
    give_back(word);
    return Val_long(2);
  }
  raise_generation(v_b, v_why);
  give_back(word);
  return Val_long(1);
}

/* What the take step answers when it took every buffer, and when a word
   held a claim, hold or bit; otherwise, the index of a buffer it found dead
   or lost. TAKE_BUSY is a try's answer alone: another step held a word. */
#define TAKE_ALL (-1)
#define TAKE_USED (-2)
#define TAKE_BUSY (-3)

/* Whether [word] is that of one of the first [n] buffers of the list
   [bs]. */
static int among(value bs, intnat n, _Atomic value *word) {
  for (; n > 0; bs = Field(bs, 1), n--)
    if (claim_word(Field(bs, 0)) == word) return 1;
  return 0;
}

/* One try of the take step: holds each word of the buffers of [bs] in turn,
   and stops at the first that is not empty. [*busy] is the word another
   step held, for TAKE_BUSY. */
static intnat take_try(value bs, value why, _Atomic value **busy) {
  intnat held = 0, answer = TAKE_ALL;
  for (value l = bs; l != Val_emptylist; l = Field(l, 1)) {
    value b = Field(l, 0);
    _Atomic value *word = claim_word(b);
    value seen;
    if (!hold_empty(word, &seen)) {
      if (dead(b)) answer = held;
      else if (Long_val(seen) >= 0 || among(bs, held, word)) answer = TAKE_USED;
      else {
        answer = TAKE_BUSY;
        *busy = word;
      }
      break;
    }
    held++;
    if (answer == TAKE_ALL && (dead(b) || followed(b, 1) == RIG_LOST))
      answer = held - 1;
  }
  if (answer == TAKE_ALL)
    for (value l = bs; l != Val_emptylist; l = Field(l, 1))
      raise_generation(Field(l, 0), why);
  for (value l = bs; held > 0; l = Field(l, 1), held--)
    give_back(claim_word(Field(l, 0)));
  return answer;
}

/* Buffer.take's step over the buffers of the list [v_bs]: holds each one's
   word if it is empty and checks the buffer under it, every use of the
   memory meanwhile waiting. If every word was empty and every buffer is
   live and its work no lost device's, it raises each generation with the
   reason [v_why], then gives every word back; otherwise it gives back the
   words it held, changing nothing. A word that is not empty answers before
   a buffer found lost. A memory twice in [v_bs] finds its second word held
   by the step itself: TAKE_USED. A word another step holds is waited for
   with no word held, then the step tries again, so its answer is the one
   it would give after that step. It allocates nothing, and a claim that
   meets one of its words spins for at most one try: for [n] buffers, [n]
   compare-and-sets, checks and walks of stamps. */
value caml_rig_buffers_take(value v_bs, value v_why) {
  _Atomic value *busy = NULL;
  intnat answer;
  while ((answer = take_try(v_bs, v_why, &busy)) == TAKE_BUSY)
    await_given(busy);
  return Val_long(answer);
}

/* Buffer.unborrow's step: ends [v_b]'s lease with the reason [v_why] where
   no claim or hold through the borrow keeps it, in one compare-and-set of its count
   from none to its end, so that the borrow's buffers are dead and a claim
   through it that follows finds them so. The memory's claims, of its other
   buffers, are not its. It is 1 if it ended the lease, 0 if a claim through
   the borrow kept it, and 2 if [v_b] is dead. */
value caml_rig_lease_free(value v_b, value v_why) {
  value alloc = alloc_of(v_b);
  if (Long_val(load_field(alloc, ALLOC_GEN)) !=
          Long_val(Field(v_b, BUFFER_GEN)) ||
      lease_ended(v_b))
    return Val_long(2);
  caml_modify(&Field(lease_of(v_b), LEASE_WHY), v_why);
  value none = Val_long(0);
  if (atomic_compare_exchange_strong_explicit(
          lease_count(v_b), &none, Val_long(LEASE_END), memory_order_acq_rel,
          memory_order_acquire))
    return Val_long(1);
  return Val_long(Long_val(none) >= LEASE_END ? 2 : 0);
}

/* Holds

   A keep (Rig.Buffer.keep) holds its memory: it adds CLAIM_HOLD to the
   word, waiting out a take or free under way, then checks its buffer live.
   So either a take finds the hold and refuses, or the hold's check finds the
   buffer dead. A keep's record marks each hold taken with a byte, which the
   steps below set and clear with the word, so each hold is released once. */

/* A hold's mark: taken, or being ended by a release; 0 once it has ended or
   was never taken. */
#define HOLD_TAKEN 1
#define HOLD_ENDING 2

static void unhold_word(_Atomic value *word) {
  value w = atomic_fetch_sub_explicit(
      word, Val_long(CLAIM_HOLD) - Val_long(0), memory_order_release);
  if (Long_val(w) < CLAIM_HOLD)
    caml_fatal_error("Rig.Buffer.release: no keep to end");
}

/* A hold of [b]'s memory, checked under it: false, holding nothing, if [b]
   is dead. */
static int hold_one(value b) {
  _Atomic value *word = claim_word(b);
  add_settled(word, CLAIM_HOLD);
  lease_take(b);
  if (dead(b)) {
    lease_give(b);
    unhold_word(word);
    return 0;
  }
  return 1;
}

/* Holds the memory of each buffer [i] of [v_bs], and marks byte [i] of
   [v_taken]: -1, or the index of the buffer refused, holding nothing. */
value caml_rig_claim_holds(value v_bs, value v_taken) {
  unsigned char *taken = Bytes_val(v_taken);
  intnat n = (intnat)Wosize_val(v_bs);
  for (intnat i = 0; i < n; i++) {
    if (!hold_one(Field(v_bs, i))) {
      for (intnat k = 0; k < i; k++) {
        lease_give(Field(v_bs, k));
        unhold_word(claim_word(Field(v_bs, k)));
        taken[k] = 0;
      }
      return Val_long(i);
    }
    taken[i] = HOLD_TAKEN;
  }
  return Val_long(-1);
}

/* Ends the hold [v_i] of [v_taken] on the allocation [v_alloc]'s memory, if
   it is taken, and its count on the lease it went through, [v_lease] an
   option. A release moves the mark from HOLD_TAKEN to HOLD_ENDING, ends the
   hold, then clears the mark, so of two releases on two domains one ends
   the hold. The other, finding HOLD_ENDING, spins until the mark clears:
   the first is inside this call, which runs no OCaml code. Each returns
   once the hold has ended. */
value caml_rig_claim_unhold(value v_taken, value v_i, value v_alloc,
                            value v_lease) {
  _Atomic unsigned char *mark =
      (_Atomic unsigned char *)(Bytes_val(v_taken) + Long_val(v_i));
  unsigned char seen = HOLD_TAKEN;
  if (!atomic_compare_exchange_strong_explicit(mark, &seen, HOLD_ENDING,
                                               memory_order_relaxed,
                                               memory_order_acquire)) {
    while (atomic_load_explicit(mark, memory_order_acquire) == HOLD_ENDING)
      relax();
    return Val_unit;
  }
  if (Is_block(v_lease))
    atomic_fetch_sub_explicit(
        (_Atomic value *)&Field(Field(v_lease, 0), LEASE_COUNT),
        Val_long(1) - Val_long(0), memory_order_release);
  unhold_word((_Atomic value *)&Field(v_alloc, ALLOC_COUNT));
  atomic_store_explicit(mark, 0, memory_order_release);
  return Val_unit;
}
