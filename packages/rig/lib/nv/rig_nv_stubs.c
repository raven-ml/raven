/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A device's C state, for the OCaml side: made at open from memory the
   path gave, and read by the wait loop.

   A device is named by the address of its state, an OCaml int. Every stub
   holds the runtime, except those whose comment says they release it,
   which do so without running pending signal handlers, so no OCaml code
   runs between the call's answer and its caller. */

#define _GNU_SOURCE

#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/callback.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/signals.h>

#include "rig_nv_stubs.h"

#define Edge_val(v) ((struct rig_nv_edge *)Long_val(v))
#define Device_val(v) (Edge_val(v)->d)
#define Pointer_val(v) ((void *)Long_val(v))

/* Guards */

/* A guard: a mutex of its own memory, which a custom block holds and
   destroys once collected, and its rank. A thread takes guards in rising
   rank, so no two threads wait for each other. [below] is the rank its
   holder held before it took the guard. */
struct guard {
  pthread_mutex_t mutex;
  int rank;
  int below;
};

#define Guard_val(v) (*(struct guard **)Data_custom_val(v))

/* The rank of the last guard this thread took, -1 for none. A take of a
   guard of no higher rank raises: a signal handler, finaliser or
   Gc.Memprof callback that runs while its thread holds a guard and takes
   one would wait for good on its own, or take two in no fixed order. */
static _Thread_local int held = -1;

static void guard_finalize(value v) {
  struct guard *g = Guard_val(v);
  if (g == NULL) return;
  pthread_mutex_destroy(&g->mutex);
  free(g);
}

static struct custom_operations guard_ops = {
    "rig.nv.guard",             guard_finalize,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default};

value caml_rig_nv_guard(value v_rank) {
  CAMLparam1(v_rank);
  CAMLlocal1(v);
  v = caml_alloc_custom(&guard_ops, sizeof(struct guard *), 0, 1);
  Guard_val(v) = NULL;
  struct guard *g = malloc(sizeof *g);
  if (g == NULL) caml_raise_out_of_memory();
  int err = pthread_mutex_init(&g->mutex, NULL);
  if (err != 0) {
    free(g);
    if (err == ENOMEM || err == EAGAIN) caml_raise_out_of_memory();
    char why[96];
    snprintf(why, sizeof why, "Rig_nv: a guard: %s", strerror(err));
    caml_failwith(why);
  }
  g->rank = Int_val(v_rank);
  g->below = -1;
  Guard_val(v) = g;
  CAMLreturn(v);
}

static void taken(struct guard *g) {
  g->below = held;
  held = g->rank;
}

/* Takes the guard [v_g]: by try-lock, and otherwise after releasing the
   runtime without running pending signal handlers, which run at the
   caller's next poll point. Raises Rig_nv.Refused, which rig_nv.ml
   registers, before it takes anything, if this thread holds a guard of the
   same or a higher rank; no other exception can end the take, so a caller
   that takes it as the last step before its handler gives it back on every
   exit. */
value caml_rig_nv_guard_take(value v_g) {
  struct guard *g = Guard_val(v_g);
  if (held >= g->rank)
    caml_raise_with_string(*caml_named_value("Rig_nv.Refused"),
                           "Rig_nv: called from a signal handler, "
                           "finaliser or Gc.Memprof callback while this "
                           "thread holds a lock the call takes, or one "
                           "taken after it");
  if (pthread_mutex_trylock(&g->mutex) != 0) {
    caml_enter_blocking_section_no_pending();
    int err = pthread_mutex_lock(&g->mutex);
    caml_leave_blocking_section();
    if (err != 0) {
      fprintf(stderr, "Rig_nv: a guard: %s\n", strerror(err));
      abort();
    }
  }
  taken(g);
  return Val_unit;
}

/* Takes the guard [v_g] if no thread holds it and this thread holds none of
   the same or a higher rank: whether it took it. */
value caml_rig_nv_guard_try(value v_g) {
  struct guard *g = Guard_val(v_g);
  if (held >= g->rank || pthread_mutex_trylock(&g->mutex) != 0)
    return Val_false;
  taken(g);
  return Val_true;
}

value caml_rig_nv_guard_give(value v_g) {
  struct guard *g = Guard_val(v_g);
  held = g->below;
  pthread_mutex_unlock(&g->mutex);
  return Val_unit;
}

/* Making a device */

/* Each value's hand-over writes its release: a commit does nothing. */
static int rig_nv_commit(void *self, uint64_t v, const char **failure) {
  (void)self;
  (void)v;
  (void)failure;
  return RIG_OK;
}

static const struct rig_driver driver = {rig_nv_room, rig_nv_submit,
                                         rig_nv_commit, rig_nv_settle};

/* A device whose timeline word is the host's 64-bit word at [v_word], the
   GPU's at [v_word_gpu], whose join words follow the GPU's word at
   [v_join_gpu], and whose notifications hold their error at [v_info32] and
   its status at [v_status]; or [0] without memory. Its values start after
   the value its word holds. Its edge lives for the process, its state until
   its stop. */
value caml_rig_nv_create(value v_word, value v_word_gpu, value v_join_gpu,
                         value v_info32, value v_status) {
  struct rig_nv_edge *e = calloc(1, sizeof *e);
  struct device *d = calloc(1, sizeof *d);
  if (e == NULL || d == NULL || pthread_mutex_init(&e->flush_mu, NULL) != 0) {
    free(e);
    free(d);
    return Val_long(0);
  }
  e->driver = &driver;
  e->word = Pointer_val(v_word);
  e->d = d;
  d->word = Pointer_val(v_word);
  d->word_gpu = (uint64_t)Long_val(v_word_gpu);
  d->join_gpu = (uint64_t)Long_val(v_join_gpu);
  d->info32_at = (uint32_t)Long_val(v_info32);
  d->status_at = (uint32_t)Long_val(v_status);
  uint64_t base = atomic_load_explicit(d->word, memory_order_acquire);
  atomic_store_explicit(&d->last, base, memory_order_relaxed);
  for (int q = 0; q < CHANNELS; q++) {
    d->ch[q].owes_setup = 1;
    d->ch[q].released = base;
  }
  return Val_long((intnat)e);
}

/* Frees the state of a device whose open failed: nothing holds it. */
value caml_rig_nv_destroy(value v_self) {
  struct device *d = Device_val(v_self);
  for (int q = 0; q < CHANNELS; q++) free(d->ch[q].marks);
  free(d->copy_launch);
  free(d->timed);
  free(d);
  pthread_mutex_destroy(&Edge_val(v_self)->flush_mu);
  free(Edge_val(v_self));
  return Val_unit;
}

/* The fields of [caml_rig_nv_channel]'s ints. */
enum {
  ch_ring,
  ch_entries,
  ch_gp_put,
  ch_token,
  ch_segments,
  ch_segments_gpu,
  ch_size,
  ch_notifier,
  ch_fields
};

/* Sets channel [v_q] of [v_self] from [v_ints], host addresses and sizes
   as above; [entries] and [size] are powers of two. [true], or [false]
   without memory for its marks. */
value caml_rig_nv_channel(value v_self, value v_q, value v_ints) {
  struct channel *c = &Device_val(v_self)->ch[Int_val(v_q)];
#define F(i) Long_val(Field(v_ints, i))
  c->ring = Pointer_val(Field(v_ints, ch_ring));
  c->entries = (uint64_t)F(ch_entries);
  c->gp_put = Pointer_val(Field(v_ints, ch_gp_put));
  c->token = (uint32_t)F(ch_token);
  c->segments = Pointer_val(Field(v_ints, ch_segments));
  c->segments_gpu = (uint64_t)F(ch_segments_gpu);
  c->size = (uint64_t)F(ch_size);
  c->notifier = Pointer_val(Field(v_ints, ch_notifier));
#undef F
  c->marks = malloc(c->entries * sizeof *c->marks);
  return Val_bool(c->marks != NULL);
}

/* Sets COMPUTE's launch ring to the [v_size] bytes, a power of two, at the
   host address [v_host] and the GPU's [v_gpu], in the GPU's memory through
   its BAR if [v_bar]. */
value caml_rig_nv_launches(value v_self, value v_host, value v_gpu,
                           value v_size, value v_bar) {
  struct channel *c = &Device_val(v_self)->ch[COMPUTE];
  c->launches = Pointer_val(v_host);
  c->launches_gpu = (uint64_t)Long_val(v_gpu);
  c->launches_size = (uint64_t)Long_val(v_size);
  c->bar = Bool_val(v_bar);
  return Val_unit;
}

/* Sets the stamp ring to the [v_slots] slots, a power of two, at the host
   address [v_host] and the GPU's [v_gpu], zeroed: [true], or [false]
   without memory for their host side. Raises [Invalid_argument] if the
   [v_bytes] bytes there hold fewer slots. */
value caml_rig_nv_stamps(value v_self, value v_host, value v_gpu,
                         value v_bytes, value v_slots) {
  struct device *d = Device_val(v_self);
  uint64_t slots = (uint64_t)Long_val(v_slots);
  if ((uint64_t)Long_val(v_bytes) / sizeof(struct stamps) < slots)
    caml_invalid_argument("Rig_nv: the stamp ring is too small");
  d->timed = calloc(slots, sizeof *d->timed);
  if (d->timed == NULL) return Val_false;
  d->stamps = Pointer_val(v_host);
  d->stamps_gpu = (uint64_t)Long_val(v_gpu);
  d->slots = slots;
  return Val_true;
}

/* The reads the first measure may take. */
#define FIRST_READS 16

/* Sets the timer's source to the path's [struct rig_nv_clock] at
   [v_source], or, if it is [0], to the registers of the GPU's BAR 0 mapped
   at [v_bar], and takes the first measure: the first read within
   READ_BOUND_NS of FIRST_READS, or the narrowest. Whether the source could
   read the timer. */
value caml_rig_nv_clock(value v_self, value v_source, value v_bar) {
  struct device *d = Device_val(v_self);
  struct measure m;
  uint64_t width, best = UINT64_MAX;
  d->source = Pointer_val(v_source);
  d->bar = Pointer_val(v_bar);
  for (int i = 0; i < FIRST_READS && best > READ_BOUND_NS; i++) {
    if (rig_nv_measure(d, &m, &width) == MEASURE_FAILED) return Val_false;
    if (width >= best) continue;
    best = width;
    d->first = d->latest = m;
  }
  d->first_wide = best > READ_BOUND_NS;
  return Val_true;
}

value caml_rig_nv_doorbell(value v_self, value v_at) {
  Device_val(v_self)->doorbell = Pointer_val(v_at);
  return Val_unit;
}

/* Templates

   A template crosses as its bytes and its patches, a record of five
   little-endian 64-bit words each: at, bits, slot, add, shift. C checks
   what keeps its reads and the writer's writes in their arrays. */

#define PATCH_WORDS 5
#define OUT_OF_BOUNDS "Rig_nv: a template is out of bounds"

/* Sets [t] to the bytes [v_bytes], at most [bytes] of them, and the patches
   [v_patches], each of a slot below [slots]: whether they lie in [t]'s
   arrays. */
static int read_template(struct template *t, value v_bytes, value v_patches,
                    size_t bytes, uint64_t slots) {
  size_t n = caml_string_length(v_bytes);
  size_t np = caml_string_length(v_patches) / (8 * PATCH_WORDS);
  if (n > bytes || np > PATCHES) return 0;
  t->nbytes = (uint32_t)n;
  t->npatches = (uint32_t)np;
  memcpy(t->bytes, String_val(v_bytes), n);
  for (size_t i = 0; i < np; i++) {
    uint64_t f[PATCH_WORDS];
    memcpy(f, String_val(v_patches) + 8 * PATCH_WORDS * i, sizeof f);
    uint64_t at = f[0], bits = f[1], slot = f[2], add = f[3], shift = f[4];
    unsigned width = bits <= 8 ? 1 : bits <= 16 ? 2 : bits <= 32 ? 4 : 8;
    if (bits < 1 || bits > 64 || slot >= slots || shift > 63 || at > n ||
        width > n - at)
      return 0;
    t->patches[i] = (struct patch){(uint16_t)at, (uint8_t)width,
                                   (uint8_t)bits, (uint8_t)slot,
                                   (uint8_t)shift, add};
  }
  return 1;
}

/* Sets template [v_k] of [v_self] to [v_bytes] and [v_patches], as above:
   a refused template leaves the one before. */
value caml_rig_nv_template(value v_self, value v_k, value v_bytes,
                           value v_patches) {
  struct template t;
  if (caml_string_length(v_bytes) % 4 != 0 ||
      !read_template(&t, v_bytes, v_patches, 4 * TEMPLATE_WORDS,
                     TEMPLATE_VALUES))
    caml_invalid_argument(OUT_OF_BOUNDS);
  Device_val(v_self)->t[Int_val(v_k)] = t;
  return Val_unit;
}

/* Launches */

/* The fields of [caml_rig_nv_launch]'s ints. */
enum { l_params_at, l_bank0_bytes, l_max, l_max_threads = l_max + 6,
       l_max_shared, l_fields };

/* A function set up for launch: [v_structures] holds the bytes and patches
   of qmd[0][0], qmd[0][1], qmd[1][0], qmd[1][1] and bank0, in turn, and
   [v_ints] its numbers, as above. The caller frees it with
   [caml_rig_nv_launch_free]. */
value caml_rig_nv_launch(value v_structures, value v_ints) {
  CAMLparam2(v_structures, v_ints);
  CAMLlocal1(result);
  result = caml_copy_nativeint(0);
  struct launch *l = malloc(sizeof *l);
  if (l == NULL) caml_raise_out_of_memory();
  struct template *s[5] = {&l->qmd[0][0], &l->qmd[0][1], &l->qmd[1][0],
                           &l->qmd[1][1], &l->bank0};
  int ok = 1;
  for (int i = 0; i < 5; i++)
    ok = ok && read_template(s[i], Field(v_structures, 2 * i),
                        Field(v_structures, 2 * i + 1), STRUCTURE_BYTES,
                        VALUES);
#define F(i) ((uint32_t)Long_val(Field(v_ints, i)))
  l->params_at = F(l_params_at);
  l->bank0_bytes = F(l_bank0_bytes);
  for (int i = 0; i < 6; i++) l->max[i] = F(l_max + i);
  l->max_threads = F(l_max_threads);
  l->max_shared = F(l_max_shared);
#undef F
  if (!ok || l->params_at > STRUCTURE_BYTES) {
    free(l);
    caml_invalid_argument(OUT_OF_BOUNDS);
  }
  Nativeint_val(result) = (intnat)l;
  CAMLreturn(result);
}

value caml_rig_nv_bank_max(value v_unit) {
  (void)v_unit;
  return Val_long(BANK_MAX);
}

value caml_rig_nv_most_parts(value v_self) {
  return Val_long(rig_nv_most_parts(Device_val(v_self)));
}

value caml_rig_nv_launch_free(value v_launch) {
  free((void *)Nativeint_val(v_launch));
  return Val_unit;
}

/* Frees each launch of the list [v_launches]. */
value caml_rig_nv_launches_free(value v_launches) {
  for (value l = v_launches; Is_block(l); l = Field(l, 1))
    free((void *)Nativeint_val(Field(l, 0)));
  return Val_unit;
}

/* Installs the private kernel's image and transfers its launch to the
   device. No submission can use the image before make returns. */
value caml_rig_nv_copy_kernel(value v_self, value v_launch, value v_host,
                              value v_image) {
  struct device *d = Device_val(v_self);
  memcpy(Pointer_val(v_host), String_val(v_image), caml_string_length(v_image));
  d->copy_launch = (struct launch *)Nativeint_val(v_launch);
  atomic_store_explicit(&d->invalidate, 1, memory_order_release);
  return Val_unit;
}

/* An entry is its segment's address plus [v_base] plus its words times
   [v_word]. */
value caml_rig_nv_entry(value v_self, value v_base, value v_word) {
  struct device *d = Device_val(v_self);
  d->entry_base = (uint64_t)Long_val(v_base);
  d->entry_word = (uint64_t)Long_val(v_word);
  return Val_unit;
}

/* The word of the device's own BAR a submission reads (struct flush), the
   path's flush word. */
value caml_rig_nv_bar(value v_self, value v_at) {
  atomic_store_explicit(&Device_val(v_self)->flushes[0].word,
                        Pointer_val(v_at), memory_order_release);
  return Val_unit;
}

/* Adds [delta] to the regions written through the BAR whose word is at
   [word], taking a free entry for a new word: [0] if none is free. The
   caller holds [flush_mu]. */
static int count_flush(struct device *d, volatile const uint32_t *word,
                       intnat delta) {
  struct flush *free = NULL;
  for (int i = 0; i < FLUSHES; i++) {
    struct flush *f = &d->flushes[i];
    if (atomic_load(&f->word) == word) {
      atomic_fetch_add_explicit(&f->count, delta, memory_order_acq_rel);
      return 1;
    }
    if (free == NULL && i > 0 &&
        (atomic_load(&f->word) == NULL || atomic_load(&f->count) == 0))
      free = f;
  }
  if (free == NULL) return 0;
  atomic_store_explicit(&free->word, word, memory_order_release);
  atomic_store_explicit(&free->count, delta, memory_order_release);
  return 1;
}

/* [count_flush] for [v_self], whose stop may run on another domain:
   [true] once the device has stopped. */
value caml_rig_nv_flush(value v_self, value v_word, value v_delta) {
  struct rig_nv_edge *e = Edge_val(v_self);
  pthread_mutex_lock(&e->flush_mu);
  int counted = e->d == NULL ||
                count_flush(e->d, Pointer_val(v_word), Long_val(v_delta));
  pthread_mutex_unlock(&e->flush_mu);
  return Val_bool(counted);
}

/* Zeroes the [v_n] bytes of host memory at [v_at]. */
value caml_rig_nv_zero(value v_at, value v_n) {
  memset(Pointer_val(v_at), 0, (size_t)Long_val(v_n));
  return Val_unit;
}

/* Copies the 64-bit word at [v_from] to [v_to], with acquire and release
   order. */
value caml_rig_nv_copy_word(value v_to, value v_from) {
  uint64_t w = atomic_load_explicit((_Atomic uint64_t *)Pointer_val(v_from),
                                    memory_order_acquire);
  atomic_store_explicit((_Atomic uint64_t *)Pointer_val(v_to), w,
                        memory_order_release);
  return Val_unit;
}

/* Owed words */

/* The pending local memory word (struct device's [local]). */
value caml_rig_nv_pending_local(value v_self) {
  return Val_long(
      atomic_load_explicit(&Device_val(v_self)->local, memory_order_acquire));
}

/* Makes [v_packed] the pending local memory: the word it replaced. */
value caml_rig_nv_exchange_local(value v_self, value v_packed) {
  return Val_long(atomic_exchange(&Device_val(v_self)->local,
                                  (uint64_t)Long_val(v_packed)));
}

value caml_rig_nv_owe_invalidate(value v_self) {
  atomic_store(&Device_val(v_self)->invalidate, 1);
  return Val_unit;
}

/* Timeline */

value caml_rig_nv_signaled(value v_self) {
  return Val_long(
      atomic_load_explicit(Edge_val(v_self)->word, memory_order_acquire));
}

/* The error the RM wrote into channel [q]'s notifier, its code and its
   status: [code lsl 16 lor status], or [0] if none. */
static uint64_t notification(const struct device *d, int q) {
  volatile const uint8_t *n = d->ch[q].notifier;
  uint32_t code = *(volatile const uint32_t *)(n + d->info32_at);
  uint16_t status = *(volatile const uint16_t *)(n + d->status_at);
  return (uint64_t)code << 16 | status;
}

value caml_rig_nv_notification(value v_self, value v_q) {
  return Val_long(notification(Device_val(v_self), Int_val(v_q)));
}

/* Raises the word to the last value submitted, by compare-and-set: it
   never moves backwards. */
value caml_rig_nv_raise(value v_self) {
  struct device *d = Device_val(v_self);
  rig_raise(d->word, atomic_load_explicit(&d->last, memory_order_relaxed));
  return Val_unit;
}

/* Frees the state of a stopped device, which the channels never read: its
   edge and word stay. The copy kernel's launch goes only if the channels
   stopped ([v_stopped]). */
value caml_rig_nv_end(value v_self, value v_stopped) {
  struct rig_nv_edge *e = Edge_val(v_self);
  pthread_mutex_lock(&e->flush_mu);
  struct device *d = e->d;
  e->d = NULL;
  pthread_mutex_unlock(&e->flush_mu);
  if (d == NULL) return Val_unit;
  if (Bool_val(v_stopped)) free(d->copy_launch);
  for (int q = 0; q < CHANNELS; q++) free(d->ch[q].marks);
  free(d->timed);
  free(d);
  return Val_unit;
}

/* The edge */

