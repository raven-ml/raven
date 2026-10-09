/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The floors of the AMD bench: the packets each row's work needs and no
   other, on a compute and a copy queue the bench makes through KFD, in a
   process that opened no device. The packets are templates the bench
   encodes with the ABI, as the driver does; this file copies them into a
   ring, patches their values, stores the write position, rings the
   doorbell, and spins until a word of host memory holds the last value
   released. Memory floors are the KFD calls behind each memory row.

   Also, for the driver's rows: a fill that places a launch, taking its
   kernel's argument from the device's argument segment where it has one,
   and a submission with waits on host memory through the driver's C
   entries. A failing call raises Failure with its step and errno. Every
   stub holds the runtime. */

#define _GNU_SOURCE

#include <errno.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include <rig_edge.h>

#define Ptr_val(v) ((void *)Nativeint_val(v))
#define Addr_val(v) ((void *)Long_val(v))

/* Templates: packet words and the words that hold a computation on an
   argument, the word's index, one or two words, the argument, then each
   operation (add, shift right, or) and its constant, all as int64s. */

#define WORDS 256
#define HOLES 4
#define OPS 3

enum { OP_ADD, OP_SHIFT, OP_OR };

struct hole {
  int at, wide, arg, nops;
  int op[OPS];
  uint64_t k[OPS];
};

struct template {
  int n, nholes;
  struct hole holes[HOLES];
  uint32_t words[];
};

/* The bytes of a template of the words [v_words]. */
static size_t template_bytes(value v_words) {
  return sizeof(struct template) + caml_string_length(v_words);
}

/* Reads a template into [t], of template_bytes([v_words]) bytes. A
   template with holes is patched on the stack, so it has at most WORDS
   words. */
static void read_template(struct template *t, value v_words, value v_holes) {
  size_t bytes = caml_string_length(v_words);
  if (Wosize_val(v_holes) > 0 && bytes > WORDS * 4)
    caml_failwith("template: too many words");
  memcpy(t->words, String_val(v_words), bytes);
  t->n = (int)(bytes / 4);
  t->nholes = 0;
  for (mlsize_t i = 0; i < Wosize_val(v_holes);) {
    if (t->nholes == HOLES) caml_failwith("template: too many holes");
    struct hole *h = &t->holes[t->nholes++];
#define NEXT Int64_val(Field(v_holes, i++))
    h->at = (int)NEXT;
    h->wide = (int)NEXT;
    h->arg = (int)NEXT;
    h->nops = (int)NEXT;
    if (h->nops > OPS) caml_failwith("template: too many operations");
    for (int j = 0; j < h->nops; j++) {
      h->op[j] = (int)NEXT;
      h->k[j] = (uint64_t)NEXT;
    }
#undef NEXT
  }
}

/* [t]'s words with its holes filled from [args], into [w]. */
static void patch(const struct template *t, const uint64_t *args,
                  uint32_t *w) {
  memcpy(w, t->words, (size_t)t->n * 4);
  for (int i = 0; i < t->nholes; i++) {
    const struct hole *h = &t->holes[i];
    uint64_t v = args[h->arg];
    for (int j = 0; j < h->nops; j++) switch (h->op[j]) {
        case OP_ADD: v += h->k[j]; break;
        case OP_SHIFT: v >>= h->k[j]; break;
        default: v |= h->k[j]; break;
      }
    w[h->at] = (uint32_t)v;
    if (h->wide == 2) w[h->at + 1] = (uint32_t)(v >> 32);
  }
}

/* The driver rows' fill: places the template; a template with holes first
   takes 8 bytes of the argument segment, writes the kernel's argument
   there, and fills argument 0 with the bytes' GPU address. */

struct fill {
  int (*place)(void *queue, const uint32_t *words, size_t n);
  int (*segment)(void *queue, size_t n, void **host, uint64_t *address);
  uint64_t arg;
};

/* A fill's template, which follows it in its argument. */
static struct template *fill_template(struct fill *f) {
  return (struct template *)(f + 1);
}

static int fill(void *queue, void *arg, uint64_t v) {
  (void)v;
  struct fill *f = arg;
  struct template *t = fill_template(f);
  if (t->nholes == 0) return f->place(queue, t->words, (size_t)t->n);
  void *host;
  uint64_t at;
  int e = f->segment(queue, sizeof f->arg, &host, &at);
  if (e) return e;
  memcpy(host, &f->arg, sizeof f->arg);
  uint64_t args[1] = {at};
  uint32_t w[WORDS];
  patch(t, args, w);
  return f->place(queue, w, (size_t)t->n);
}

value rig_amd_bench_fill_entry(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)fill);
}

/* A fill's argument, as a bigarray the bench hands rig as a host
   buffer. */
value rig_amd_bench_fill_arg(value v_place, value v_segment, value v_arg,
                                value v_words, value v_holes) {
  CAMLparam5(v_place, v_segment, v_arg, v_words, v_holes);
  CAMLlocal1(r);
  size_t bytes = sizeof(struct fill) + template_bytes(v_words);
  r = caml_ba_alloc_dims(CAML_BA_UINT8 | CAML_BA_C_LAYOUT, 1, NULL,
                         (intnat)bytes);
  struct fill *f = Caml_ba_data_val(r);
  memset(f, 0, bytes);
  f->place = (int (*)(void *, const uint32_t *, size_t))Ptr_val(v_place);
  f->segment = (int (*)(void *, size_t, void **, uint64_t *))Ptr_val(v_segment);
  f->arg = (uint64_t)Long_val(v_arg);
  read_template(fill_template(f), v_words, v_holes);
  CAMLreturn(r);
}

/* A submission of no parts as value [v_v] through the driver whose C state
   is [v_edge], after [v_n] waits for the word at GPU address [v_at] to hold
   at least [v_value]: for the rows rig cannot express, waits on a word of
   host memory. */
value rig_amd_bench_submit(value v_edge, value v_v, value v_at, value v_value,
                              value v_n) {
  void *self = Ptr_val(v_edge);
  const struct rig_driver *driver = rig_driver_of(self);
  struct rig_wait w[16];
  int n = Int_val(v_n);
  if (n > 16) caml_invalid_argument("submit: more than 16 waits");
  for (int i = 0; i < n; i++)
    w[i] = (struct rig_wait){(uint64_t)Long_val(v_at),
                            (uint64_t)Long_val(v_value), RIG_WORD};
  if (driver->room(self, NULL, 0) != RIG_FITS)
    caml_failwith("submit: no room");
  const char *why = NULL;
  if (driver->submit(self, (uint64_t)Long_val(v_v), w, n, NULL, 0, NULL, 0,
                     &why) == RIG_FAILED)
    caml_failwith(why);
  return Val_unit;
}

/* Host memory */

/* Reads the [v_n] bytes at [v_p], 8 at a time: their sum. */
value rig_amd_bench_read(value v_p, value v_n) {
  const volatile uint64_t *p = Addr_val(v_p);
  size_t n = (size_t)Long_val(v_n) / 8;
  uint64_t s = 0;
  for (size_t i = 0; i < n; i++) s += p[i];
  return Val_long((intnat)s);
}

value rig_amd_bench_write(value v_p, value v_s) {
  memcpy(Addr_val(v_p), String_val(v_s), caml_string_length(v_s));
  return Val_unit;
}

/* Stores [v_v] at [v_p], a 64-bit word of host memory. */
value rig_amd_bench_set64(value v_p, value v_v) {
  *(volatile uint64_t *)Addr_val(v_p) = (uint64_t)Long_val(v_v);
  return Val_unit;
}

#ifdef __linux__

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <linux/kfd_ioctl.h>

/* The memory the floors allocate, as the amdgpu path allocates it: GPU
   memory, and system memory the kernel driver owns. */
#define FLAGS(f) KFD_IOC_ALLOC_MEM_FLAGS_##f
#define ALL (FLAGS(WRITABLE) | FLAGS(NO_SUBSTITUTE) | FLAGS(EXECUTABLE))
#define HOST (FLAGS(COHERENT) | FLAGS(UNCACHED) | FLAGS(PUBLIC) | ALL)

static const uint32_t kinds[] = {FLAGS(VRAM) | ALL, FLAGS(GTT) | HOST};

#define USERPTR_FLAGS (FLAGS(USERPTR) | HOST)

static void fail(const char *step, int e) {
  char msg[128];
  snprintf(msg, sizeof msg, "%s: %s", step, strerror(e));
  caml_failwith(msg);
}

static int kfd = -1, drm = -1;
static uint32_t gpu_id;

static void request(unsigned long req, void *arg, const char *step) {
  int r;
  do r = ioctl(kfd, req, arg);
  while (r < 0 && (errno == EINTR || errno == EAGAIN));
  if (r < 0) fail(step, errno);
}

/* Memory */

struct mem {
  uint64_t handle, at, n;
};

/* [n] bytes of [kind] (0 GPU, 1 system) at addresses reserved in the
   process, mapped for the GPU, and for the host if system memory. */
static struct mem alloc(int kind, uint64_t n) {
  void *at = mmap(NULL, n, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                  -1, 0);
  if (at == MAP_FAILED) fail("reserving GPU addresses", errno);
  struct kfd_ioctl_alloc_memory_of_gpu_args a = {
      .va_addr = (uint64_t)(uintptr_t)at, .size = n, .gpu_id = gpu_id,
      .flags = kinds[kind]};
  request(AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &a, "allocating GPU memory");
  if (kind == 1 && mmap(at, n, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED,
                        drm, (off_t)a.mmap_offset) == MAP_FAILED)
    fail("mapping GPU memory", errno);
  uint32_t ids[1] = {gpu_id};
  struct kfd_ioctl_map_memory_to_gpu_args m = {
      .handle = a.handle, .device_ids_array_ptr = (uint64_t)(uintptr_t)ids,
      .n_devices = 1};
  request(AMDKFD_IOC_MAP_MEMORY_TO_GPU, &m, "mapping memory for the GPU");
  return (struct mem){a.handle, (uint64_t)(uintptr_t)at, n};
}

static void unmap_free(uint64_t handle) {
  uint32_t ids[1] = {gpu_id};
  struct kfd_ioctl_map_memory_to_gpu_args m = {
      .handle = handle, .device_ids_array_ptr = (uint64_t)(uintptr_t)ids,
      .n_devices = 1};
  request(AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &m,
          "unmapping memory from the GPU");
  struct kfd_ioctl_free_memory_of_gpu_args f = {.handle = handle};
  request(AMDKFD_IOC_FREE_MEMORY_OF_GPU, &f, "freeing GPU memory");
}

static void free_mem(struct mem m) {
  unmap_free(m.handle);
  munmap((void *)(uintptr_t)m.at, m.n);
}

/* Queues */

#define RING_BYTES (1u << 20)
#define EOP_BYTES 4096
#define EVENT_PAGE_BYTES 0x8000

struct ring {
  volatile uint32_t *words;
  uint64_t size, put; /* words; put never wraps */
  volatile uint64_t *write, *doorbell;
  int sdma;
};

enum { COMPUTE, COPY };

static struct ring rings[2];
static _Atomic uint64_t *word;
static uint64_t word_gpu, last, max_copy;
static uint32_t interrupt;
static volatile uint64_t *doorbells;
static uint64_t doorbells_base;

/* The templates, in the bench's order. */
enum {
  F_RELEASE,
  F_AGENT,
  F_WAIT,
  F_WAIT64,
  S_POLL,
  S_FENCE,
  S_COPY,
  S_TRAP,
  TEMPLATES
};
static struct template *templates[TEMPLATES];

static volatile uint64_t *doorbell(uint64_t off) {
  uint64_t base = off & ~(uint64_t)0x1fff;
  if (doorbells == NULL) {
    void *p = mmap(NULL, 0x2000, PROT_READ | PROT_WRITE, MAP_SHARED, kfd,
                   (off_t)base);
    if (p == MAP_FAILED) fail("mapping the doorbell page", errno);
    doorbells = p;
    doorbells_base = base;
  }
  if (base != doorbells_base) caml_failwith("two doorbell pages");
  return (volatile uint64_t *)((char *)doorbells + (off - base));
}

/* A queue of [type] (0 PM4, 1 SDMA) on a ring of system memory, its
   positions in the page [pos]; a compute queue also takes an end-of-pipe
   buffer and a context save area of [q] bytes of GPU memory. */
static void make_queue(struct ring *r, int sdma, uint64_t pos, uint64_t *pos_host,
                       const intnat *q) {
  struct mem ring = alloc(1, RING_BYTES);
  struct kfd_ioctl_create_queue_args a = {
      .ring_base_address = ring.at, .ring_size = RING_BYTES,
      .read_pointer_address = pos, .write_pointer_address = pos + 8,
      .gpu_id = gpu_id, .queue_type = sdma ? 1 : 0,
      .queue_percentage = 100, .queue_priority = 7};
  if (!sdma) {
    a.eop_buffer_address = alloc(0, EOP_BYTES).at;
    a.eop_buffer_size = EOP_BYTES;
    a.ctx_save_restore_address = alloc(0, (uint64_t)q[4]).at;
    a.ctx_save_restore_size = (uint32_t)q[2];
    a.ctl_stack_size = (uint32_t)q[3];
  }
  request(AMDKFD_IOC_CREATE_QUEUE, &a, "making a KFD queue");
  r->words = (volatile uint32_t *)(uintptr_t)ring.at;
  r->size = RING_BYTES / 4;
  r->put = 0;
  r->write = pos_host + 1;
  r->doorbell = doorbell(a.doorbell_offset);
  r->sdma = sdma;
}

/* A signal event for the compute floors' releases to interrupt with, as the
   amdgpu path makes its own: an event page made with a first signal event, so
   that this one's id is above 0, then this one, its slot armed with its id. An
   interrupt that names it makes KFD look the event up and read its slot, the
   work a driver's release asks of KFD for a host that may sleep. */
static void make_interrupt(void) {
  struct mem page = alloc(1, EVENT_PAGE_BYTES);
  struct kfd_ioctl_create_event_args first = {.event_page_offset = page.handle,
                                              .auto_reset = 1};
  request(AMDKFD_IOC_CREATE_EVENT, &first, "making the event page");
  struct kfd_ioctl_create_event_args e = {.auto_reset = 1};
  request(AMDKFD_IOC_CREATE_EVENT, &e, "making a signal event");
  interrupt = e.event_id;
  ((volatile uint64_t *)(uintptr_t)page.at)[interrupt] = interrupt;
}

/* Opens GPU [q]: its KFD id, render node minor, a die's context save area
   and control stack bytes, a compute queue's save area bytes, and the
   bytes one SDMA copy moves; then makes the queues and the word. Once per
   process. */
value rig_amd_bench_start(value v_q) {
  intnat q[6];
  char path[64];
  if (kfd >= 0) return Val_unit;
  for (int i = 0; i < 6; i++) q[i] = Long_val(Field(v_q, i));
  gpu_id = (uint32_t)q[0];
  max_copy = (uint64_t)q[5];
  kfd = open("/dev/kfd", O_RDWR | O_CLOEXEC);
  if (kfd < 0) fail("opening /dev/kfd", errno);
  snprintf(path, sizeof path, "/dev/dri/renderD%d", (int)q[1]);
  drm = open(path, O_RDWR | O_CLOEXEC);
  if (drm < 0) fail("opening the render node", errno);
  struct kfd_ioctl_acquire_vm_args vm = {.drm_fd = (uint32_t)drm,
                                         .gpu_id = gpu_id};
  request(AMDKFD_IOC_ACQUIRE_VM, &vm, "acquiring the GPU's address space");
  struct kfd_ioctl_runtime_enable_args rt = {0};
  if (ioctl(kfd, AMDKFD_IOC_RUNTIME_ENABLE, &rt) < 0 && errno != EBUSY)
    fail("enabling the runtime", errno);
  struct mem w = alloc(1, 4096), pos = alloc(1, 4096);
  word = (_Atomic uint64_t *)(uintptr_t)w.at;
  word_gpu = w.at;
  atomic_store(word, 0);
  uint64_t *pos_host = (uint64_t *)(uintptr_t)pos.at;
  memset(pos_host, 0, 4096);
  make_queue(&rings[COMPUTE], 0, pos.at, pos_host, q);
  make_queue(&rings[COPY], 1, pos.at + 64, pos_host + 8, q);
  make_interrupt();
  return Val_unit;
}

/* The interrupt context of the compute floors' releases. */
value rig_amd_bench_interrupt(value unit) {
  (void)unit;
  return Val_long(interrupt);
}

/* Sets floor template [v_i], kept for the process. */
value rig_amd_bench_template(value v_i, value v_words, value v_holes) {
  struct template **t = &templates[Int_val(v_i)];
  if (caml_string_length(v_words) > WORDS * 4)
    caml_failwith("template: too many words");
  free(*t);
  *t = calloc(1, template_bytes(v_words));
  if (*t == NULL) caml_raise_out_of_memory();
  read_template(*t, v_words, v_holes);
  return Val_unit;
}

/* Places [n] words on [r]: on a PM4 ring they wrap; on an SDMA ring they
   never do, and the ring's end is zeroed when they do not fit before it. */
static void put(struct ring *r, const uint32_t *w, size_t n) {
  uint64_t at = r->put & (r->size - 1);
  if (r->sdma && at + n > r->size) {
    for (uint64_t i = at; i < r->size; i++) r->words[i] = 0;
    r->put += r->size - at;
    at = 0;
  }
  for (size_t i = 0; i < n; i++) r->words[(at + i) & (r->size - 1)] = w[i];
  r->put += n;
}

static void emit(int q, int t, uint64_t a0, uint64_t a1, uint64_t a2) {
  uint64_t args[3] = {a0, a1, a2};
  uint32_t w[WORDS];
  patch(templates[t], args, w);
  put(&rings[q], w, (size_t)templates[t]->n);
}

static void barrier(void) {
#if defined(__aarch64__)
  __asm__ __volatile__("dsb sy" ::: "memory");
#elif defined(__x86_64__)
  __asm__ __volatile__("mfence" ::: "memory");
#else
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

static void ring(int q) {
  struct ring *r = &rings[q];
  uint64_t at = r->sdma ? 4 * r->put : r->put;
  barrier();
  *r->write = at;
  barrier();
  *r->doorbell = at;
}

/* The next value, released into the word on [q] once the work before it
   completed. */
static void release(int q) {
  last++;
  if (q == COMPUTE) emit(q, F_RELEASE, word_gpu, last, 0);
  else {
    emit(q, S_FENCE, word_gpu, last, 0);
    emit(q, S_TRAP, 0, 0, 0);
  }
}

static void spin(void) {
  while (atomic_load_explicit(word, memory_order_acquire) < last) {
  }
}

/* [k] releases on the compute queue, each rung alone, then the wait for
   the last: at system scope, or at agent scope if [agent], which writes no
   cache back. */
static void releases(int agent, long k) {
  for (long i = 0; i < k; i++) {
    if (agent) emit(COMPUTE, F_AGENT, word_gpu, ++last, 0);
    else release(COMPUTE);
    ring(COMPUTE);
  }
  spin();
}

value rig_amd_bench_release(value v_k) {
  releases(0, Long_val(v_k));
  return Val_unit;
}

value rig_amd_bench_release_agent(value v_k) {
  releases(1, Long_val(v_k));
  return Val_unit;
}

/* A release on the queue the last one did not use, after a wait for the
   last. */
value rig_amd_bench_switch(value unit) {
  (void)unit;
  int q = (int)(last & 1);
  emit(q, q == COMPUTE ? F_WAIT : S_POLL, word_gpu, last, 0);
  release(q);
  ring(q);
  spin();
  return Val_unit;
}

/* [v_n] satisfied 64-bit waits on the word at GPU address [v_at], then a
   release, on the compute queue. */
value rig_amd_bench_waits(value v_at, value v_n) {
  for (long i = 0; i < Long_val(v_n); i++)
    emit(COMPUTE, F_WAIT64, (uint64_t)Long_val(v_at), 1, 0);
  release(COMPUTE);
  ring(COMPUTE);
  spin();
  return Val_unit;
}

/* [v_k] times the words of [v_w], then a release, on the compute queue. */
value rig_amd_bench_launch(value v_w, value v_k) {
  const uint32_t *w = (const uint32_t *)String_val(v_w);
  size_t n = caml_string_length(v_w) / 4;
  for (long i = 0; i < Long_val(v_k); i++) put(&rings[COMPUTE], w, n);
  release(COMPUTE);
  ring(COMPUTE);
  spin();
  return Val_unit;
}

/* [v_k] times the words of [v_w] on the compute queue, each rung as a
   submission of its own: with [v_each], each with its release; otherwise
   one release after the last. Then the wait for the last. */
value rig_amd_bench_submits(value v_w, value v_k, value v_each) {
  const uint32_t *w = (const uint32_t *)String_val(v_w);
  size_t n = caml_string_length(v_w) / 4;
  for (long i = 0; i < Long_val(v_k); i++) {
    put(&rings[COMPUTE], w, n);
    if (Bool_val(v_each)) release(COMPUTE);
    ring(COMPUTE);
  }
  if (!Bool_val(v_each)) {
    release(COMPUTE);
    ring(COMPUTE);
  }
  spin();
  return Val_unit;
}

/* A copy of [v_n] bytes between GPU addresses, then a release, on the copy
   queue. */
value rig_amd_bench_copy(value v_dst, value v_src, value v_n) {
  uint64_t dst = (uint64_t)Long_val(v_dst);
  uint64_t src = (uint64_t)Long_val(v_src);
  uint64_t n = (uint64_t)Long_val(v_n);
  for (uint64_t off = 0; off < n; off += max_copy) {
    uint64_t k = n - off < max_copy ? n - off : max_copy;
    emit(COPY, S_COPY, dst + off, src + off, k);
  }
  release(COPY);
  ring(COPY);
  spin();
  return Val_unit;
}

/* [v_n] bytes of [v_kind] memory (0 GPU, 1 system), never freed: their
   address, the same for the GPU and the host. */
value rig_amd_bench_buffer(value v_kind, value v_n) {
  struct mem m = alloc(Int_val(v_kind), (uint64_t)Long_val(v_n));
  return Val_long((intnat)m.at);
}

value rig_amd_bench_alloc(value v_kind, value v_n) {
  free_mem(alloc(Int_val(v_kind), (uint64_t)Long_val(v_n)));
  return Val_unit;
}

/* [v_n] bytes of page-aligned host memory, written once, never freed, in
   the host's base pages. Mapping memory for the GPU walks its pages: 256
   MiB took about 6 ms in 4 KiB pages and 1.4 to 3 ms in the huge pages the
   kernel had free for the process, so a row over huge pages read the
   machine's fragmentation. */
value rig_amd_bench_pages(value v_n) {
  void *p;
  size_t n = (size_t)Long_val(v_n);
  if (posix_memalign(&p, (size_t)sysconf(_SC_PAGESIZE), n))
    caml_raise_out_of_memory();
  if (madvise(p, n, MADV_NOHUGEPAGE)) fail("keeping huge pages out", errno);
  memset(p, 1, n);
  return Val_long((intnat)p);
}

/* Maps the [v_n] bytes of host memory at the page [v_p] for the GPU, then
   unmaps them. */
value rig_amd_bench_map_host(value v_p, value v_n) {
  uint64_t p = (uint64_t)Long_val(v_p);
  struct kfd_ioctl_alloc_memory_of_gpu_args a = {
      .va_addr = p, .size = (uint64_t)Long_val(v_n), .mmap_offset = p,
      .gpu_id = gpu_id, .flags = USERPTR_FLAGS};
  request(AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &a, "registering host memory");
  uint32_t ids[1] = {gpu_id};
  struct kfd_ioctl_map_memory_to_gpu_args m = {
      .handle = a.handle, .device_ids_array_ptr = (uint64_t)(uintptr_t)ids,
      .n_devices = 1};
  request(AMDKFD_IOC_MAP_MEMORY_TO_GPU, &m,
          "mapping host memory for the GPU");
  unmap_free(a.handle);
  return Val_unit;
}

#else

#define LINUX(name, ...)                                                       \
  value rig_amd_bench_##name(__VA_ARGS__) {                                 \
    caml_failwith("the AMD floors run on Linux");                              \
  }

#define UNUSED __attribute__((unused))
LINUX(start, value a UNUSED)
LINUX(template, value a UNUSED, value b UNUSED, value c UNUSED)
LINUX(release, value a UNUSED)
LINUX(release_agent, value a UNUSED)
LINUX(interrupt, value a UNUSED)
LINUX(switch, value a UNUSED)
LINUX(waits, value a UNUSED, value b UNUSED)
LINUX(launch, value a UNUSED, value b UNUSED)
LINUX(submits, value a UNUSED, value b UNUSED, value c UNUSED)
LINUX(copy, value a UNUSED, value b UNUSED, value c UNUSED)
LINUX(buffer, value a UNUSED, value b UNUSED)
LINUX(alloc, value a UNUSED, value b UNUSED)
LINUX(map_host, value a UNUSED, value b UNUSED)
LINUX(pages, value a UNUSED)

#endif
