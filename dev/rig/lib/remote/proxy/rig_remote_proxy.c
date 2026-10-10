/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Proxies: the C edge of a device of another machine, whose hand-overs are
   frames on its link (rig_remote_link.c), and its word, a shadow the link's
   receiving thread advances. A wait between proxies of one link reaches the
   edge with [at] the producer's id on the agent. */

#define _GNU_SOURCE

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
#include <stdlib.h>
#include <string.h>

#include "rig_edge.h"
#include "rig_remote_proxy_stubs.h"

/* The bytes of words and copies a proxy has handed over and not reached
   past which its room check answers RIG_LATER. */
#define WINDOW_BYTES ((uint64_t)64 << 20)

#define Dev_val(v) ((struct rig_remote_dev *)Nativeint_val(v))

static const char *why(struct rig_remote_link *l) {
  const struct rig_remote_why *w = atomic_load(&l->job->why);
  return w != NULL ? w->s : "the job failed";
}

/* Room: only words and copies fit, words only on the host, and no copy
   from this process's memory after a copy into it in one submission, whose
   bytes the hand-over could not wait for. */
static int proxy_room(void *self, const struct rig_part *parts, int n,
                      const uint8_t *args) {
  (void)args;
  struct rig_remote_dev *d = self;
  int wrote_local = 0;
  for (int i = 0; i < n; i++) {
    const struct rig_part *p = &parts[i];
    if (p->kind == RIG_WORDS) {
      if (d->id != 0) return RIG_NEVER;
      continue;
    }
    if (p->kind != RIG_COPY) return RIG_NEVER;
    if (p->copy.local == RIG_LOCAL_SRC && wrote_local) return RIG_NEVER;
    if (p->copy.local == RIG_LOCAL_DST) wrote_local = 1;
  }
  struct rig_remote_link *l = d->link;
  pthread_mutex_lock(&l->mu);
  int later = d->flying > WINDOW_BYTES && d->handed > atomic_load(&d->word);
  pthread_mutex_unlock(&l->mu);
  return later ? RIG_LATER : RIG_FITS;
}

/* Whether the work a hand-over follows is done here: its waits on proxies
   of its link, and its device's last value that copies into this process's
   memory. Holds the link's lock. */
static int followed(struct rig_remote_dev *d, const struct rig_wait *waits,
                    int nwaits) {
  struct rig_remote_link *l = d->link;
  if (atomic_load(&d->word) < d->written) return 0;
  for (int i = 0; i < nwaits; i++) {
    uint64_t id = waits[i].at;
    struct rig_remote_dev *p = id < l->ndevs ? l->devs[id] : NULL;
    if (p != NULL && atomic_load(&p->word) < waits[i].value) return 0;
  }
  return 1;
}

static unsigned char *put_u32(unsigned char *b, uint32_t v) {
  for (int i = 0; i < 4; i++) b[i] = (unsigned char)(v >> (8 * i));
  return b + 4;
}

static unsigned char *put_u64(unsigned char *b, uint64_t v) {
  rig_remote_put_u64(b, v);
  return b + 8;
}

static unsigned char *put_side(unsigned char *b, int local, uint64_t id,
                               uint64_t offset) {
  if (local) {
    *b = 1;
    return b + 1;
  }
  *b = 0;
  return put_u64(put_u64(b + 1, id), offset);
}

/* The hand-over: the frame wire.mli lays out, then the bytes of its copies
   from this process's memory, read in place once the work they follow is
   done. It returns once the frame is sent. */
static int proxy_submit(void *self, uint64_t v, const struct rig_wait *waits,
                        int nwaits, const struct rig_part *parts, int nparts,
                        const uint8_t *args, const uint64_t *slots, int nslots,
                        const uint64_t *handles, int nhandles,
                        uint64_t *times, const char **failure) {
  (void)times;
  (void)args;
  (void)slots;
  (void)nslots;
  struct rig_remote_dev *d = self;
  struct rig_remote_link *l = d->link;
  (void)handles;
  (void)nhandles;
  size_t n = 8 + 8 + 4 + 16 * (size_t)nwaits + 4;
  uint64_t bytes = 0;
  int nreads = 0, nlocals = 0;
  for (int i = 0; i < nparts; i++) {
    const struct rig_part *p = &parts[i];
    if (p->kind == RIG_WORDS) {
      n += 1 + 4 + 4 * p->words.n;
      bytes += 4 * p->words.n;
      continue;
    }
    n += 1 + 8 + (p->copy.local == RIG_LOCAL_SRC ? 1 : 17) +
         (p->copy.local == RIG_LOCAL_DST ? 1 : 17);
    bytes += p->copy.bytes;
    if (p->copy.local == RIG_LOCAL_SRC) nreads++;
    if (p->copy.local == RIG_LOCAL_DST) nlocals++;
  }
  struct rig_remote_local **locals =
      nlocals > 0 ? calloc((size_t)nlocals, sizeof *locals) : NULL;
  struct rig_remote_flight *f = malloc(sizeof *f);
  struct rig_remote_frame *frame = rig_remote_frame(K_HANDOVER, n, nreads);
  int ok = f != NULL && frame != NULL && (nlocals == 0 || locals != NULL);
  for (int i = 0; ok && i < nlocals; i++)
    ok = (locals[i] = malloc(sizeof **locals)) != NULL;
  if (!ok) {
    for (int i = 0; locals != NULL && i < nlocals; i++) free(locals[i]);
    free(locals);
    free(f);
    free(frame);
    *failure = "out of memory for a hand-over";
    return RIG_FAILED;
  }

  if (nreads > 0) {
    pthread_mutex_lock(&l->mu);
    while (!atomic_load(&l->failed) && !followed(d, waits, nwaits))
      pthread_cond_wait(&l->cv, &l->mu);
    pthread_mutex_unlock(&l->mu);
  }

  unsigned char *b = put_u64(put_u64(frame->buf + HEADER, d->id), v);
  b = put_u32(b, (uint32_t)nwaits);
  for (int i = 0; i < nwaits; i++)
    b = put_u64(put_u64(b, waits[i].at), waits[i].value);
  b = put_u32(b, (uint32_t)nparts);
  for (int i = 0; i < nparts; i++) {
    const struct rig_part *p = &parts[i];
    if (p->kind == RIG_WORDS) {
      /* Little-endian, as wire.mli lays words out, whatever this host's
         order. */
      *b++ = 0;
      b = put_u32(b, (uint32_t)p->words.n);
      for (size_t w = 0; w < p->words.n; w++) b = put_u32(b, p->words.at[w]);
      continue;
    }
    *b++ = 1;
    b = put_u64(b, p->copy.bytes);
    b = put_side(b, p->copy.local == RIG_LOCAL_SRC, p->copy.src,
                 p->copy.src_offset);
    b = put_side(b, p->copy.local == RIG_LOCAL_DST, p->copy.dst,
                 p->copy.dst_offset);
  }
  int j = 0;
  for (int i = 0; i < nparts; i++) {
    const struct rig_part *p = &parts[i];
    if (p->kind != RIG_COPY || p->copy.local != RIG_LOCAL_SRC) continue;
    frame->spans[j].p =
        (const unsigned char *)(uintptr_t)p->copy.src + p->copy.src_offset;
    frame->spans[j++].n = p->copy.bytes;
  }

  /* The receiving thread finds the copies into this process's memory and the
     value before the agent can answer them. */
  pthread_mutex_lock(&l->mu);
  int k = 0;
  for (int i = 0; i < nparts; i++) {
    const struct rig_part *p = &parts[i];
    if (p->kind != RIG_COPY || p->copy.local != RIG_LOCAL_DST) continue;
    struct rig_remote_local *c = locals[k++];
    c->next = NULL;
    c->value = v;
    c->at = (unsigned char *)(uintptr_t)p->copy.dst + p->copy.dst_offset;
    c->bytes = p->copy.bytes;
    if (d->locals_last != NULL)
      d->locals_last->next = c;
    else
      d->locals = c;
    d->locals_last = c;
  }
  f->next = NULL;
  f->value = v;
  f->bytes = bytes;
  if (d->flights_last != NULL)
    d->flights_last->next = f;
  else
    d->flights = f;
  d->flights_last = f;
  d->flying += bytes;
  d->handed = v;
  if (nlocals > 0) d->written = v;
  pthread_mutex_unlock(&l->mu);
  free(locals);

  int r = rig_remote_send(l, frame, NULL);
  if (r == 0) return RIG_COMMITTED;
  *failure = r == -3 ? "the job is closed" : why(l);
  return RIG_FAILED;
}

/* Each value's hand-over sends its message: a commit does nothing. */
static int proxy_commit(void *self, uint64_t v, const char **failure) {
  (void)self;
  (void)v;
  (void)failure;
  return RIG_OK;
}

static const struct rig_driver proxy_driver = {proxy_room, proxy_submit,
                                               proxy_commit};

/* Stubs */

/* The C state of the proxy of the agent's device [id] on the link. Does not
   release the runtime. */
value caml_rig_remote_proxy_make(value link, value id) {
  struct rig_remote_dev *d = calloc(1, sizeof *d);
  if (d == NULL) caml_raise_out_of_memory();
  d->driver = &proxy_driver;
  d->link = Link_c(link);
  d->id = (uint64_t)Long_val(id);
  int r = rig_remote_add_dev(d);
  if (r != 0) free(d);
  if (r == -2) caml_raise_out_of_memory();
  if (r == -1)
    caml_invalid_argument(
        "Rig_remote_proxy.make: the link has a proxy of that device");
  return caml_copy_nativeint((intnat)d);
}

/* The host address of the proxy's shadow. */
value caml_rig_remote_proxy_word(value vd) {
  return Val_long((intnat)(uintptr_t)&Dev_val(vd)->word);
}

value caml_rig_remote_proxy_signaled(value vd) {
  return Val_long((intnat)atomic_load(&Dev_val(vd)->word));
}

/* Waits at most [ms] for the shadow to differ from [seen]: [false], or
   [true] if the job failed. Releases the runtime. */
value caml_rig_remote_proxy_sleep(value vd, value vseen, value vms) {
  struct rig_remote_dev *d = Dev_val(vd);
  struct rig_remote_link *l = d->link;
  uint64_t seen = (uint64_t)Long_val(vseen);
  long ms = Long_val(vms);
  if (rig_remote_forked(l->job)) return Val_true;
  caml_release_runtime_system();
  pthread_mutex_lock(&l->mu);
  if (!atomic_load(&l->failed) && atomic_load(&d->word) == seen) {
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    t.tv_sec += ms / 1000;
    t.tv_nsec += (ms % 1000) * 1000000L;
    if (t.tv_nsec >= 1000000000L) {
      t.tv_sec++;
      t.tv_nsec -= 1000000000L;
    }
    while (!atomic_load(&l->failed) && atomic_load(&d->word) == seen)
      if (pthread_cond_timedwait(&l->cv, &l->mu, &t) != 0) break;
  }
  int failed = atomic_load(&l->failed);
  pthread_mutex_unlock(&l->mu);
  caml_acquire_runtime_system();
  return Val_bool(failed);
}

/* Stops the proxy: the last value handed over goes into the shadow at once,
   or, while a copy into this process's memory may still land, once none
   may. Does not release the runtime: it waits for nothing. */
value caml_rig_remote_proxy_stop(value vd) {
  struct rig_remote_dev *d = Dev_val(vd);
  struct rig_remote_link *l = d->link;
  if (rig_remote_forked(l->job)) {
    atomic_store(&d->word, d->handed);
    return Val_unit;
  }
  pthread_mutex_lock(&l->mu);
  d->stopped = 1;
  rig_remote_settle(d);
  pthread_mutex_unlock(&l->mu);
  return Val_unit;
}
