/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* VFIO's requests, with the kernel's own structures from <linux/vfio.h>,
   which a Linux build needs (the kernel headers, Linux 5.4 or later), over
   descriptors that are Unix.file_descr, an int on Linux. A refused request
   raises Unix.Unix_error with its errno, so the caller names what to
   change. A request may wait, as opening a function or resetting it does,
   so each runs with the runtime released; the version and extension checks
   answer at once and hold it. Elsewhere every request raises Unix_error
   ENOSYS. */

#define _GNU_SOURCE
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
#include <caml/unixsupport.h>

#ifdef __linux__
#include <linux/vfio.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#include <unistd.h>

/* The API version Vfio.api_version states: keep the two in sync. */
_Static_assert(VFIO_API_VERSION == 0, "Vfio.api_version is 0");

/* The IOMMU models, the constructors of Vfio.model in their order: keep the
   two in sync. Type 1 v2 translates through mappings; no-IOMMU mode does
   not translate. */
enum model { MODEL_TYPE1V2, MODEL_NO_IOMMU };

static const unsigned long models[] = {
    [MODEL_TYPE1V2] = VFIO_TYPE1v2_IOMMU,
    [MODEL_NO_IOMMU] = VFIO_NOIOMMU_IOMMU,
};

/* The ioctl [req] on [fd] with the runtime released: its result, with
   errno set where it is -1. [arg] is C memory. */
static int request(int fd, unsigned long req, void *arg) {
  caml_release_runtime_system();
  int r = ioctl(fd, req, arg);
  int e = errno;
  caml_acquire_runtime_system();
  errno = e;
  return r;
}

/* Asks [fd] for the structure [*info] of [size] bytes, then again at the size
   the kernel says its capabilities need. [*info] is malloc'd; NULL with errno
   set if a request fails. */
static void *ask(int fd, unsigned long req, void *first, size_t size) {
  void *info = malloc(size);
  if (info == NULL) caml_raise_out_of_memory();
  memcpy(info, first, size);
  if (request(fd, req, info) != 0) {
    int e = errno;
    free(info);
    errno = e;
    return NULL;
  }
  uint32_t need = *(uint32_t *)info;
  if (need <= size) return info;
  void *more = realloc(info, need);
  if (more == NULL) {
    free(info);
    caml_raise_out_of_memory();
  }
  memset((char *)more + size, 0, need - size);
  memcpy(more, first, size);
  *(uint32_t *)more = need;
  if (request(fd, req, more) != 0) {
    int e = errno;
    free(more);
    errno = e;
    return NULL;
  }
  return more;
}

/* The capability [id] of the structure [info] of [size] bytes whose chain
   starts at [first], or NULL. Offsets only grow, so the walk ends. */
static struct vfio_info_cap_header *cap(void *info, uint32_t size,
                                        uint32_t first, uint16_t id) {
  uint32_t off = first;
  while (off != 0 && off + sizeof(struct vfio_info_cap_header) <= size) {
    struct vfio_info_cap_header *h =
        (struct vfio_info_cap_header *)((char *)info + off);
    if (h->id == id) return h;
    if (h->next <= off) return NULL;
    off = h->next;
  }
  return NULL;
}

static value pair(intnat a, intnat b) {
  value p = caml_alloc_tuple(2);
  Store_field(p, 0, Val_long(a));
  Store_field(p, 1, Val_long(b));
  return p;
}

static value cons(value hd, value tl) {
  CAMLparam2(hd, tl);
  value c = caml_alloc_small(2, 0);
  Field(c, 0) = hd;
  Field(c, 1) = tl;
  CAMLreturn(c);
}

/* A 64-bit field saturated at the largest int: a range may end at 2^64-1. */
static intnat saturate(uint64_t x) {
  return x > (uint64_t)Max_long ? Max_long : (intnat)x;
}

/* The API version the container [fd] speaks. */
value caml_rig_pci_vfio_version(value fd) {
  int v = ioctl(Int_val(fd), VFIO_GET_API_VERSION);
  if (v < 0) caml_uerror("ioctl", Nothing);
  return Val_int(v);
}

/* Whether the container [fd] has the IOMMU model [m]. */
value caml_rig_pci_vfio_supports(value fd, value m) {
  int r = ioctl(Int_val(fd), VFIO_CHECK_EXTENSION, models[Int_val(m)]);
  if (r < 0) caml_uerror("ioctl", Nothing);
  return Val_bool(r > 0);
}

value caml_rig_pci_vfio_viable(value group) {
  struct vfio_group_status s = {.argsz = sizeof s};
  if (request(Int_val(group), VFIO_GROUP_GET_STATUS, &s) != 0)
    caml_uerror("ioctl", Nothing);
  return Val_bool(s.flags & VFIO_GROUP_FLAGS_VIABLE);
}

value caml_rig_pci_vfio_set_container(value group, value container) {
  int c = Int_val(container);
  if (request(Int_val(group), VFIO_GROUP_SET_CONTAINER, &c) != 0)
    caml_uerror("ioctl", Nothing);
  return Val_unit;
}

value caml_rig_pci_vfio_set_iommu(value container, value m) {
  if (request(Int_val(container), VFIO_SET_IOMMU,
              (void *)(uintptr_t)models[Int_val(m)]) != 0)
    caml_uerror("ioctl", Nothing);
  return Val_unit;
}

/* Opening the function may reset it, which takes up to seconds. A name
   with a NUL byte names no function: the kernel would read a shorter one. */
value caml_rig_pci_vfio_device(value group, value bus) {
  CAMLparam2(group, bus);
  if (!caml_string_is_c_safe(bus)) caml_unix_error(ENODEV, "ioctl", bus);
  char *name = caml_stat_strdup(String_val(bus));
  int fd = request(Int_val(group), VFIO_GROUP_GET_DEVICE_FD, name);
  caml_stat_free(name);
  if (fd < 0) caml_uerror("ioctl", bus);
  CAMLreturn(Val_int(fd));
}

/* Region [i] of the function [device]: (size, offset in the device's file,
   mappable, Some areas if only those (offset, bytes) parts map). An
   allocation that raises Out_of_memory leaks [info]. */
value caml_rig_pci_vfio_region(value device, value i) {
  CAMLparam2(device, i);
  CAMLlocal4(r, areas, some, p);
  struct vfio_region_info first = {.argsz = sizeof first, .index = Int_val(i)};
  struct vfio_region_info *info =
      ask(Int_val(device), VFIO_DEVICE_GET_REGION_INFO, &first, sizeof first);
  if (info == NULL) caml_uerror("ioctl", Nothing);
  some = Val_none;
  struct vfio_info_cap_header *h =
      (info->flags & VFIO_REGION_INFO_FLAG_CAPS)
          ? cap(info, info->argsz, info->cap_offset,
                VFIO_REGION_INFO_CAP_SPARSE_MMAP)
          : NULL;
  if (h != NULL) {
    struct vfio_region_info_cap_sparse_mmap *s =
        (struct vfio_region_info_cap_sparse_mmap *)h;
    areas = Val_emptylist;
    for (uint32_t k = s->nr_areas; k > 0; k--) {
      p = pair(saturate(s->areas[k - 1].offset), saturate(s->areas[k - 1].size));
      areas = cons(p, areas);
    }
    some = caml_alloc_some(areas);
  }
  r = caml_alloc_tuple(4);
  Store_field(r, 0, Val_long(saturate(info->size)));
  Store_field(r, 1, Val_long(saturate(info->offset)));
  Store_field(r, 2, Val_bool(info->flags & VFIO_REGION_INFO_FLAG_MMAP));
  Store_field(r, 3, some);
  free(info);
  CAMLreturn(r);
}

/* Routes the first MSI vector of [device] to the eventfd [efd]. */
value caml_rig_pci_vfio_msi(value device, value efd) {
  char buf[sizeof(struct vfio_irq_set) + sizeof(int32_t)];
  struct vfio_irq_set *s = (struct vfio_irq_set *)buf;
  s->argsz = sizeof buf;
  s->flags = VFIO_IRQ_SET_DATA_EVENTFD | VFIO_IRQ_SET_ACTION_TRIGGER;
  s->index = VFIO_PCI_MSI_IRQ_INDEX;
  s->start = 0;
  s->count = 1;
  int32_t fd = Int_val(efd);
  memcpy(s->data, &fd, sizeof fd);
  if (request(Int_val(device), VFIO_DEVICE_SET_IRQS, s) != 0)
    caml_uerror("ioctl", Nothing);
  return Val_unit;
}

value caml_rig_pci_vfio_reset(value device) {
  if (request(Int_val(device), VFIO_DEVICE_RESET, NULL) != 0)
    caml_uerror("ioctl", Nothing);
  return Val_unit;
}

/* What the container's IOMMU maps: (page sizes as a bitmap, the (first,
   last) device address ranges, [] if it does not say). An allocation that
   raises Out_of_memory leaks [info]. */
value caml_rig_pci_vfio_iommu(value container) {
  CAMLparam1(container);
  CAMLlocal3(r, ranges, p);
  struct vfio_iommu_type1_info first = {.argsz = sizeof first};
  struct vfio_iommu_type1_info *info =
      ask(Int_val(container), VFIO_IOMMU_GET_INFO, &first, sizeof first);
  if (info == NULL) caml_uerror("ioctl", Nothing);
  int caps = info->flags & VFIO_IOMMU_INFO_CAPS;
  struct vfio_info_cap_header *h =
      caps ? cap(info, info->argsz, info->cap_offset,
                 VFIO_IOMMU_TYPE1_INFO_CAP_IOVA_RANGE)
           : NULL;
  ranges = Val_emptylist;
  if (h != NULL) {
    struct vfio_iommu_type1_info_cap_iova_range *s =
        (struct vfio_iommu_type1_info_cap_iova_range *)h;
    for (uint32_t k = s->nr_iovas; k > 0; k--) {
      p = pair(saturate(s->iova_ranges[k - 1].start),
               saturate(s->iova_ranges[k - 1].end));
      ranges = cons(p, ranges);
    }
  }
  r = caml_alloc_tuple(2);
  Store_field(r, 0,
              Val_long((info->flags & VFIO_IOMMU_INFO_PGSIZES)
                           ? saturate(info->iova_pgsizes)
                           : 0));
  Store_field(r, 1, ranges);
  free(info);
  CAMLreturn(r);
}

/* Maps the [n] bytes at [va] of the process at the device address [iova].
   The kernel pins them. */
value caml_rig_pci_vfio_map(value container, value va, value iova,
                               value n) {
  struct vfio_iommu_type1_dma_map m = {
      .argsz = sizeof m,
      .flags = VFIO_DMA_MAP_FLAG_READ | VFIO_DMA_MAP_FLAG_WRITE,
      .vaddr = (uint64_t)Long_val(va),
      .iova = (uint64_t)Long_val(iova),
      .size = (uint64_t)Long_val(n)};
  if (request(Int_val(container), VFIO_IOMMU_MAP_DMA, &m) != 0)
    caml_uerror("ioctl", Nothing);
  return Val_unit;
}

value caml_rig_pci_vfio_unmap(value container, value iova, value n) {
  struct vfio_iommu_type1_dma_unmap m = {.argsz = sizeof m,
                                         .iova = (uint64_t)Long_val(iova),
                                         .size = (uint64_t)Long_val(n)};
  if (request(Int_val(container), VFIO_IOMMU_UNMAP_DMA, &m) != 0)
    caml_uerror("ioctl", Nothing);
  return Val_unit;
}

/* An eventfd an interrupt signals. */
value caml_rig_pci_eventfd(value unit) {
  (void)unit;
  int fd = eventfd(0, EFD_CLOEXEC);
  if (fd < 0) caml_uerror("eventfd", Nothing);
  return Val_int(fd);
}

/* Waits at most [ms] for the eventfd [efd], with the runtime released. A
   poll a signal interrupts reports no interrupt: callers poll in a loop. */
value caml_rig_pci_wait(value efd, value ms) {
  struct pollfd p = {.fd = Int_val(efd), .events = POLLIN};
  int timeout = Int_val(ms);
  caml_release_runtime_system();
  int r = poll(&p, 1, timeout);
  uint64_t count;
  if (r > 0 && read(p.fd, &count, sizeof count) < 0) r = 0;
  caml_acquire_runtime_system();
  return Val_bool(r > 0);
}

#else

/* Without Linux every request raises Unix_error ENOSYS, and no interrupt
   comes. */

#define FAILS1(f, call)                     \
  value f(value a) {                        \
    (void)a;                                \
    caml_unix_error(ENOSYS, call, Nothing); \
  }

#define FAILS2(f, call)                     \
  value f(value a, value b) {               \
    (void)a;                                \
    (void)b;                                \
    caml_unix_error(ENOSYS, call, Nothing); \
  }

FAILS1(caml_rig_pci_vfio_version, "ioctl")
FAILS2(caml_rig_pci_vfio_supports, "ioctl")
FAILS1(caml_rig_pci_vfio_viable, "ioctl")
FAILS2(caml_rig_pci_vfio_set_container, "ioctl")
FAILS2(caml_rig_pci_vfio_set_iommu, "ioctl")
FAILS2(caml_rig_pci_vfio_device, "ioctl")
FAILS2(caml_rig_pci_vfio_region, "ioctl")
FAILS2(caml_rig_pci_vfio_msi, "ioctl")
FAILS1(caml_rig_pci_vfio_reset, "ioctl")
FAILS1(caml_rig_pci_vfio_iommu, "ioctl")
FAILS1(caml_rig_pci_eventfd, "eventfd")

value caml_rig_pci_vfio_map(value container, value va, value iova,
                               value n) {
  (void)container;
  (void)va;
  (void)iova;
  (void)n;
  caml_unix_error(ENOSYS, "ioctl", Nothing);
}

value caml_rig_pci_vfio_unmap(value container, value iova, value n) {
  (void)container;
  (void)iova;
  (void)n;
  caml_unix_error(ENOSYS, "ioctl", Nothing);
}

value caml_rig_pci_wait(value efd, value ms) {
  (void)efd;
  (void)ms;
  return Val_false;
}
#endif
