/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* VFIO's requests, packed as <linux/vfio.h> lays them out, over descriptors
   that are Unix.file_descr, an int on Linux, and the eventfd its interrupts
   signal. A refused request raises Unix.Unix_error with its errno, so the
   caller names what to change. A request may wait, as opening a function or
   resetting it does, so each runs with the runtime released. Elsewhere
   every request raises Unix_error ENOSYS. */

#define _GNU_SOURCE
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
#include <caml/unixsupport.h>

#ifdef __linux__
#include <linux/vfio.h>
#include <poll.h>
#include <string.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#include <unistd.h>

/* The IOMMU models, in the order of Vfio.model's constructors. */
static const unsigned long models[] = {VFIO_TYPE1v2_IOMMU, VFIO_NOIOMMU_IOMMU};

/* The ioctl [request] on [fd] with the argument [arg], an integer or the
   address of memory that is not the OCaml heap's: its result, or -errno. */
static int call(int fd, unsigned long request, unsigned long arg) {
  caml_release_runtime_system();
  int r = ioctl(fd, request, arg);
  int e = errno;
  caml_acquire_runtime_system();
  return r < 0 ? -e : r;
}

/* [r], or raises Unix_error for the -errno [r]. */
static int check(int r) {
  if (r < 0) caml_unix_error(-r, "ioctl", Nothing);
  return r;
}

/* A 64-bit value saturated at the largest int, since a range may end at
   2^64-1. */
static value u64(uint64_t x) {
  return Val_long(x > (uint64_t)Max_long ? Max_long : (intnat)x);
}

/* The structure [request] answers on [fd], whose first word is argsz and
   whose [bytes] first bytes are [head], in a buffer of [*size] bytes the
   caller frees: with the structure's own bytes, then again with those the
   kernel says its capabilities need. Is NULL with errno set if refused. */
static char *query(int fd, unsigned long request, const void *head,
                   uint32_t bytes, uint32_t *size) {
  *size = bytes;
  for (;;) {
    char *p = calloc(1, *size);
    if (p == NULL) {
      errno = ENOMEM;
      return NULL;
    }
    memcpy(p, head, bytes);
    memcpy(p, size, sizeof *size);
    int r = call(fd, request, (unsigned long)p);
    if (r < 0) {
      free(p);
      errno = -r;
      return NULL;
    }
    uint32_t need;
    memcpy(&need, p, sizeof need);
    if (need <= *size || *size > bytes) return p;
    free(p);
    *size = need;
  }
}

/* The capability [id] in the [size] bytes of [p], whose chain starts at
   [first], or NULL. Offsets only grow, so a walk ends. */
static const struct vfio_info_cap_header *capability(const char *p,
                                                     uint32_t size,
                                                     uint32_t first,
                                                     uint16_t id) {
  uint32_t off = first;
  while (off != 0 && off + sizeof(struct vfio_info_cap_header) <= size) {
    const struct vfio_info_cap_header *h = (const void *)(p + off);
    if (h->id == id) return h;
    if (h->next <= off) return NULL;
    off = h->next;
  }
  return NULL;
}

/* The list of the [n] pairs of 64-bit values at [at], as many as end
   within [size] bytes of [p]. */
static value pairs(const char *p, uint32_t size, uint32_t at, uint32_t n) {
  CAMLparam0();
  CAMLlocal3(list, pair, cell);
  uint32_t most = at <= size ? (size - at) / 16 : 0;
  if (n > most) n = most;
  list = Val_emptylist;
  for (uint32_t i = n; i > 0; i--) {
    uint64_t a, b;
    memcpy(&a, p + at + 16 * (i - 1), 8);
    memcpy(&b, p + at + 16 * (i - 1) + 8, 8);
    pair = caml_alloc_tuple(2);
    Store_field(pair, 0, u64(a));
    Store_field(pair, 1, u64(b));
    cell = caml_alloc_small(2, Tag_cons);
    Field(cell, 0) = pair;
    Field(cell, 1) = list;
    list = cell;
  }
  CAMLreturn(list);
}

value caml_rig_pci_vfio_api_version(value unit) {
  (void)unit;
  return Val_int(VFIO_API_VERSION);
}

value caml_rig_pci_vfio_version(value v_fd) {
  return Val_int(check(call(Int_val(v_fd), VFIO_GET_API_VERSION, 0)));
}

value caml_rig_pci_vfio_supports(value v_fd, value v_model) {
  unsigned long m = models[Int_val(v_model)];
  return Val_bool(check(call(Int_val(v_fd), VFIO_CHECK_EXTENSION, m)) > 0);
}

value caml_rig_pci_vfio_set_iommu(value v_fd, value v_model) {
  unsigned long m = models[Int_val(v_model)];
  check(call(Int_val(v_fd), VFIO_SET_IOMMU, m));
  return Val_unit;
}

value caml_rig_pci_vfio_viable(value v_group) {
  struct vfio_group_status s = {.argsz = sizeof s};
  check(call(Int_val(v_group), VFIO_GROUP_GET_STATUS, (unsigned long)&s));
  return Val_bool(s.flags & VFIO_GROUP_FLAGS_VIABLE);
}

value caml_rig_pci_vfio_set_container(value v_group, value v_container) {
  int container = Int_val(v_container);
  check(call(Int_val(v_group), VFIO_GROUP_SET_CONTAINER,
             (unsigned long)&container));
  return Val_unit;
}

value caml_rig_pci_vfio_open_device(value v_group, value v_bus) {
  char *bus = caml_stat_strdup(String_val(v_bus));
  int r = call(Int_val(v_group), VFIO_GROUP_GET_DEVICE_FD, (unsigned long)bus);
  caml_stat_free(bus);
  return Val_int(check(r));
}

/* A region of Vfio.region: [Bar i] is a block, [Config] an immediate. */
value caml_rig_pci_vfio_region(value v_fd, value v_region) {
  CAMLparam2(v_fd, v_region);
  CAMLlocal3(areas, some, r);
  uint32_t index = Is_block(v_region) ? VFIO_PCI_BAR0_REGION_INDEX +
                                            Int_val(Field(v_region, 0))
                                      : VFIO_PCI_CONFIG_REGION_INDEX;
  struct vfio_region_info head = {.index = index};
  uint32_t size;
  char *p = query(Int_val(v_fd), VFIO_DEVICE_GET_REGION_INFO, &head,
                  sizeof head, &size);
  if (p == NULL) caml_unix_error(errno, "ioctl", Nothing);
  const struct vfio_region_info *info = (const void *)p;
  const struct vfio_info_cap_header *sparse =
      info->flags & VFIO_REGION_INFO_FLAG_CAPS
          ? capability(p, size, info->cap_offset,
                       VFIO_REGION_INFO_CAP_SPARSE_MMAP)
          : NULL;
  areas = Val_none;
  if (sparse != NULL) {
    uint32_t at = (const char *)sparse - p;
    uint32_t n = at + sizeof(struct vfio_region_info_cap_sparse_mmap) <= size
                     ? ((const struct vfio_region_info_cap_sparse_mmap *)sparse)
                           ->nr_areas
                     : 0;
    some =
        pairs(p, size,
              at + offsetof(struct vfio_region_info_cap_sparse_mmap, areas), n);
    areas = caml_alloc_some(some);
  }
  r = caml_alloc_tuple(4);
  Store_field(r, 0, u64(info->size));
  Store_field(r, 1, u64(info->offset));
  Store_field(r, 2, Val_bool(info->flags & VFIO_REGION_INFO_FLAG_MMAP));
  Store_field(r, 3, areas);
  free(p);
  CAMLreturn(r);
}

value caml_rig_pci_vfio_msi(value v_device, value v_efd) {
  char buf[sizeof(struct vfio_irq_set) + sizeof(int32_t)];
  struct vfio_irq_set *s = (struct vfio_irq_set *)buf;
  int32_t efd = Int_val(v_efd);
  *s = (struct vfio_irq_set){
      .argsz = sizeof buf,
      .flags = VFIO_IRQ_SET_DATA_EVENTFD | VFIO_IRQ_SET_ACTION_TRIGGER,
      .index = VFIO_PCI_MSI_IRQ_INDEX,
      .start = 0,
      .count = 1};
  memcpy(s->data, &efd, sizeof efd);
  check(call(Int_val(v_device), VFIO_DEVICE_SET_IRQS, (unsigned long)buf));
  return Val_unit;
}

value caml_rig_pci_vfio_reset(value v_device) {
  check(call(Int_val(v_device), VFIO_DEVICE_RESET, 0));
  return Val_unit;
}

value caml_rig_pci_vfio_iommu(value v_container) {
  CAMLparam1(v_container);
  CAMLlocal2(ranges, r);
  struct vfio_iommu_type1_info head = {0};
  uint32_t size;
  char *p = query(Int_val(v_container), VFIO_IOMMU_GET_INFO, &head, sizeof head,
                  &size);
  if (p == NULL) caml_unix_error(errno, "ioctl", Nothing);
  const struct vfio_iommu_type1_info *info = (const void *)p;
  const struct vfio_info_cap_header *iovas =
      info->flags & VFIO_IOMMU_INFO_CAPS
          ? capability(p, size, info->cap_offset,
                       VFIO_IOMMU_TYPE1_INFO_CAP_IOVA_RANGE)
          : NULL;
  ranges = Val_emptylist;
  if (iovas != NULL) {
    uint32_t at = (const char *)iovas - p;
    uint32_t n =
        at + sizeof(struct vfio_iommu_type1_info_cap_iova_range) <= size
            ? ((const struct vfio_iommu_type1_info_cap_iova_range *)iovas)
                  ->nr_iovas
            : 0;
    ranges = pairs(
        p, size,
        at + offsetof(struct vfio_iommu_type1_info_cap_iova_range, iova_ranges),
        n);
  }
  r = caml_alloc_tuple(2);
  Store_field(r, 0,
              info->flags & VFIO_IOMMU_INFO_PGSIZES ? u64(info->iova_pgsizes)
                                                    : Val_int(0));
  Store_field(r, 1, ranges);
  free(p);
  CAMLreturn(r);
}

/* The kernel pins the mapped bytes. */
value caml_rig_pci_vfio_map(value v_container, value v_va, value v_iova,
                            value v_bytes) {
  struct vfio_iommu_type1_dma_map m = {
      .argsz = sizeof m,
      .flags = VFIO_DMA_MAP_FLAG_READ | VFIO_DMA_MAP_FLAG_WRITE,
      .vaddr = Long_val(v_va),
      .iova = Long_val(v_iova),
      .size = Long_val(v_bytes)};
  check(call(Int_val(v_container), VFIO_IOMMU_MAP_DMA, (unsigned long)&m));
  return Val_unit;
}

value caml_rig_pci_vfio_unmap(value v_container, value v_iova, value v_bytes) {
  struct vfio_iommu_type1_dma_unmap m = {
      .argsz = sizeof m, .iova = Long_val(v_iova), .size = Long_val(v_bytes)};
  check(call(Int_val(v_container), VFIO_IOMMU_UNMAP_DMA, (unsigned long)&m));
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

#define FAILS3(f, call)                     \
  value f(value a, value b, value c) {      \
    (void)a;                                \
    (void)b;                                \
    (void)c;                                \
    caml_unix_error(ENOSYS, call, Nothing); \
  }

#define FAILS4(f, call)                         \
  value f(value a, value b, value c, value d) { \
    (void)a;                                    \
    (void)b;                                    \
    (void)c;                                    \
    (void)d;                                    \
    caml_unix_error(ENOSYS, call, Nothing);     \
  }

FAILS1(caml_rig_pci_vfio_api_version, "ioctl")
FAILS1(caml_rig_pci_vfio_version, "ioctl")
FAILS2(caml_rig_pci_vfio_supports, "ioctl")
FAILS2(caml_rig_pci_vfio_set_iommu, "ioctl")
FAILS1(caml_rig_pci_vfio_viable, "ioctl")
FAILS2(caml_rig_pci_vfio_set_container, "ioctl")
FAILS2(caml_rig_pci_vfio_open_device, "ioctl")
FAILS2(caml_rig_pci_vfio_region, "ioctl")
FAILS2(caml_rig_pci_vfio_msi, "ioctl")
FAILS1(caml_rig_pci_vfio_reset, "ioctl")
FAILS1(caml_rig_pci_vfio_iommu, "ioctl")
FAILS4(caml_rig_pci_vfio_map, "ioctl")
FAILS3(caml_rig_pci_vfio_unmap, "ioctl")
FAILS1(caml_rig_pci_eventfd, "eventfd")

value caml_rig_pci_wait(value efd, value ms) {
  (void)efd;
  (void)ms;
  return Val_false;
}
#endif
