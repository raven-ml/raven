/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The system calls the kernel path makes: NVIDIA's device files, their
   ioctls, and the process's mappings at the addresses the GPU uses.

   A descriptor is an int, as the RM's parameters hold it. A failed call
   raises Unix.Unix_error; the OCaml side says what it was doing. Linux
   only: elsewhere every call raises ENOSYS. A stub whose comment says it
   releases the runtime does so; the others hold it. */

#define _GNU_SOURCE

#include <errno.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
#include <caml/fail.h>
#include <caml/unixsupport.h>
#include <stdint.h>
#include <string.h>

#include "rig_nv.h"

#define PTR(v) ((void *)Long_val(v))

value caml_rig_nv_nvidia_address(value v_params) {
  return Val_long((intnat)Caml_ba_data_val(v_params));
}

#ifdef __linux__
#include <fcntl.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif

/* Opens the device file [v_path]: its descriptor. A GPU's first open
   initialises it, so this releases the runtime. */
value caml_rig_nv_nvidia_open(value v_path) {
  CAMLparam1(v_path);
  char *path = caml_stat_strdup(String_val(v_path));
  caml_release_runtime_system();
  int fd = open(path, O_RDWR | O_CLOEXEC);
  int e = errno;
  caml_acquire_runtime_system();
  caml_stat_free(path);
  if (fd < 0) caml_unix_error(e, "open", v_path);
  CAMLreturn(Val_long(fd));
}

value caml_rig_nv_nvidia_close(value v_fd) {
  close(Int_val(v_fd));
  return Val_unit;
}

/* Runs the ioctl [v_request] on [v_fd] with [v_params]' bytes, which may
   point to other bytes the caller keeps alive, retrying an interrupted
   call. Releases the runtime: the RM may take time to answer, and
   [v_params] stays rooted, so another domain's collection cannot free the
   bytes the kernel writes. */
value caml_rig_nv_nvidia_ioctl(value v_fd, value v_request, value v_params) {
  CAMLparam1(v_params);
  int fd = Int_val(v_fd);
  unsigned long request = (unsigned long)Long_val(v_request);
  void *arg = Caml_ba_data_val(v_params);
  int r;
  caml_release_runtime_system();
  do r = ioctl(fd, request, arg);
  while (r < 0 && errno == EINTR);
  int e = errno;
  caml_acquire_runtime_system();
  if (r < 0) caml_unix_error(e, "ioctl", Nothing);
  CAMLreturn(Val_unit);
}

/* Maps [v_n] bytes at [v_at], readable, writable and shared: of [v_fd] from
   its start, or anonymous memory if [v_fd] is negative. Releases the
   runtime. */
value caml_rig_nv_nvidia_map(value v_fd, value v_at, value v_n) {
  int fd = Int_val(v_fd);
  void *at = PTR(v_at);
  size_t n = (size_t)Long_val(v_n);
  int flags = MAP_SHARED | MAP_FIXED | (fd < 0 ? MAP_ANONYMOUS : 0);
  caml_release_runtime_system();
  void *p = mmap(at, n, PROT_READ | PROT_WRITE, flags, fd, 0);
  int e = errno;
  caml_acquire_runtime_system();
  if (p == MAP_FAILED) caml_unix_error(e, "mmap", Nothing);
  return Val_unit;
}

/* Reserves the [v_n] bytes at [v_at], inaccessible, unless something is
   mapped there. */
value caml_rig_nv_nvidia_reserve(value v_at, value v_n) {
  void *at = PTR(v_at);
  size_t n = (size_t)Long_val(v_n);
  void *p = mmap(at, n, PROT_NONE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE |
                     MAP_FIXED_NOREPLACE,
                 -1, 0);
  if (p == MAP_FAILED) caml_unix_error(errno, "mmap", Nothing);
  /* A kernel before 4.17 reads the flag as a hint. */
  if (p != at) {
    munmap(p, n);
    caml_unix_error(EEXIST, "mmap", Nothing);
  }
  return Val_unit;
}

/* Gives back the reservation of the [v_n] bytes at [v_at]. */
value caml_rig_nv_nvidia_unreserve(value v_at, value v_n) {
  if (munmap(PTR(v_at), (size_t)Long_val(v_n)) != 0)
    caml_unix_error(errno, "munmap", Nothing);
  return Val_unit;
}

/* Returns the [v_n] bytes at [v_at] to the reservation they came from. */
value caml_rig_nv_nvidia_unmap(value v_at, value v_n) {
  void *p = mmap(PTR(v_at), (size_t)Long_val(v_n), PROT_NONE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1,
                 0);
  if (p == MAP_FAILED) caml_unix_error(errno, "mmap", Nothing);
  return Val_unit;
}

/* The host's page size, which bounds the ranges the process maps. */
value caml_rig_nv_nvidia_page_size(value unit) {
  (void)unit;
  return Val_long(sysconf(_SC_PAGESIZE));
}

/* The GPU's timer

   NV2080_CTRL_CMD_TIMER_GET_GPU_CPU_TIME_CORRELATION_INFO, with the CPU
   source PLATFORM_API on the CPU, reads the GPU's PTIMER through BAR 0 in
   the kernel, with no call to the GSP (ctrl2080tmr.h,
   subdevice_ctrl_timer_kernel.c). Its CPU time is CLOCK_MONOTONIC_RAW, so
   the reader takes the host's time as the middle of two readings of rig's
   clock around the call. */

#define TIMER_CORRELATION 0x20800406
#define CPU_CLK_PLATFORM_API 0x3

/* NVOS54_PARAMETERS (nvos.h), a control's arguments. */
struct control {
  uint32_t client, object, cmd, flags;
  uint64_t params;
  uint32_t params_size, status;
};
_Static_assert(sizeof(struct control) == 32, "NVOS54_PARAMETERS");

/* NV2080_CTRL_TIMER_GET_GPU_CPU_TIME_CORRELATION_INFO_PARAMS, with room for
   one sample of its sixteen. */
struct correlation_params {
  uint8_t cpu_clk_id, sample_count;
  uint8_t pad[6];
  struct {
    uint64_t cpu, gpu;
  } samples[16];
};
_Static_assert(sizeof(struct correlation_params) == 264, "its layout");

/* A GPU's reader: the control file, the ioctl of a control, and the
   client and subdevice the control runs on. */
struct correlation {
  struct rig_nv_clock clock;
  int fd;
  unsigned long request;
  uint32_t client, subdevice;
};

static int correlate(struct rig_nv_clock *c, uint64_t *gpu, uint64_t *host) {
  struct correlation *k = (struct correlation *)c;
  struct correlation_params p;
  memset(&p, 0, sizeof p);
  p.cpu_clk_id = CPU_CLK_PLATFORM_API;
  p.sample_count = 1;
  struct control a = {k->client, k->subdevice, TIMER_CORRELATION, 0,
                      (uint64_t)(uintptr_t)&p, sizeof p, 0};
  uint64_t before = rig_now_ns();
  int r;
  do r = ioctl(k->fd, k->request, &a);
  while (r < 0 && errno == EINTR);
  uint64_t after = rig_now_ns();
  if (r < 0 || a.status != 0) return -1;
  *gpu = p.samples[0].gpu;
  *host = before + (after - before) / 2;
  return 0;
}

/* A reader of the GPU's timer through the control file [v_fd], whose
   control ioctl is [v_request], on the subdevice [v_subdevice] of the
   client [v_client]: the address of its struct rig_nv_clock, which lives
   for the process. Raises [Out_of_memory] without memory for it. */
value caml_rig_nv_nvidia_clock(value v_fd, value v_request, value v_client,
                               value v_subdevice) {
  struct correlation *k = malloc(sizeof *k);
  if (k == NULL) caml_raise_out_of_memory();
  k->clock.read = correlate;
  k->fd = Int_val(v_fd);
  k->request = (unsigned long)Long_val(v_request);
  k->client = (uint32_t)Long_val(v_client);
  k->subdevice = (uint32_t)Long_val(v_subdevice);
  return caml_copy_nativeint((intnat)k);
}

#else

value caml_rig_nv_nvidia_page_size(value unit) {
  (void)unit;
  return Val_long(-ENOSYS);
}

value caml_rig_nv_nvidia_open(value v_path) {
  caml_unix_error(ENOSYS, "open", v_path);
}

value caml_rig_nv_nvidia_close(value v_fd) {
  (void)v_fd;
  return Val_unit;
}

static value nosys(void) { caml_unix_error(ENOSYS, "ioctl", Nothing); }

value caml_rig_nv_nvidia_clock(value v_fd, value v_request, value v_client,
                               value v_subdevice) {
  (void)v_fd;
  (void)v_request;
  (void)v_client;
  (void)v_subdevice;
  return nosys();
}

value caml_rig_nv_nvidia_ioctl(value v_fd, value v_request, value v_params) {
  (void)v_fd;
  (void)v_request;
  (void)v_params;
  return nosys();
}

value caml_rig_nv_nvidia_map(value v_fd, value v_at, value v_n) {
  (void)v_fd;
  (void)v_at;
  (void)v_n;
  return nosys();
}

value caml_rig_nv_nvidia_reserve(value v_at, value v_n) {
  (void)v_at;
  (void)v_n;
  return nosys();
}

value caml_rig_nv_nvidia_unmap(value v_at, value v_n) {
  (void)v_at;
  (void)v_n;
  return nosys();
}

value caml_rig_nv_nvidia_unreserve(value v_at, value v_n) {
  (void)v_at;
  (void)v_n;
  return nosys();
}

#endif
