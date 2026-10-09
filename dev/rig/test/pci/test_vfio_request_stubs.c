/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* VFIO's numbers, requests and answers as <linux/vfio.h> lays them out, for
   test_vfio_request's arguments, in its order. Without the header each
   answers an empty array. */

#include <stdint.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#if defined(__linux__) && __has_include(<linux/vfio.h>)
#include <linux/vfio.h>
#include <sys/ioctl.h>
#define HAVE_VFIO 1

/* An array of the [n] byte strings [p[i]] of [len[i]] bytes. */
static value strings(int n, const void *const *p, const size_t *len) {
  CAMLparam0();
  CAMLlocal2(a, s);
  a = caml_alloc(n, 0);
  for (int i = 0; i < n; i++) {
    s = caml_alloc_initialized_string(len[i], p[i]);
    Store_field(a, i, s);
  }
  CAMLreturn(a);
}
#endif

value rig_pci_vfio_c_numbers(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(a);
#ifdef HAVE_VFIO
  const long n[] = {VFIO_GET_API_VERSION,       VFIO_CHECK_EXTENSION,
                    VFIO_SET_IOMMU,             VFIO_GROUP_GET_STATUS,
                    VFIO_GROUP_SET_CONTAINER,   VFIO_GROUP_GET_DEVICE_FD,
                    VFIO_DEVICE_GET_REGION_INFO, VFIO_DEVICE_SET_IRQS,
                    VFIO_DEVICE_RESET,          VFIO_IOMMU_GET_INFO,
                    VFIO_IOMMU_MAP_DMA,         VFIO_IOMMU_UNMAP_DMA,
                    VFIO_TYPE1v2_IOMMU,         VFIO_NOIOMMU_IOMMU,
                    VFIO_API_VERSION,           VFIO_PCI_CONFIG_REGION_INDEX};
  int count = sizeof n / sizeof n[0];
  a = caml_alloc(count, 0);
  for (int i = 0; i < count; i++) Store_field(a, i, Val_long(n[i]));
#else
  a = Atom(0);
#endif
  CAMLreturn(a);
}

value rig_pci_vfio_c_requests(value unit) {
  (void)unit;
#ifdef HAVE_VFIO
  struct vfio_group_status status = {.argsz = sizeof status};
  int container = 5;
  const char *name = "0000:01:00.0";
  struct vfio_region_info region = {.argsz = sizeof region, .index = 2};
  char msi[sizeof(struct vfio_irq_set) + sizeof(int32_t)];
  struct vfio_irq_set *s = (struct vfio_irq_set *)msi;
  s->argsz = sizeof msi;
  s->flags = VFIO_IRQ_SET_DATA_EVENTFD | VFIO_IRQ_SET_ACTION_TRIGGER;
  s->index = VFIO_PCI_MSI_IRQ_INDEX;
  s->start = 0;
  s->count = 1;
  int32_t efd = 9;
  memcpy(s->data, &efd, sizeof efd);
  struct vfio_iommu_type1_info iommu = {.argsz = sizeof iommu};
  struct vfio_iommu_type1_dma_map map = {
      .argsz = sizeof map,
      .flags = VFIO_DMA_MAP_FLAG_READ | VFIO_DMA_MAP_FLAG_WRITE,
      .vaddr = 0x7f1234560000UL,
      .iova = 0x100000000UL,
      .size = 0x200000};
  struct vfio_iommu_type1_dma_unmap unmap = {
      .argsz = sizeof unmap, .iova = 0x100000000UL, .size = 0x200000};
  const void *p[] = {&status, &container, name, &region,
                     msi,     &iommu,     &map, &unmap};
  const size_t len[] = {sizeof status, sizeof container, strlen(name) + 1,
                        sizeof region, sizeof msi,       sizeof iommu,
                        sizeof map,    sizeof unmap};
  return strings(8, p, len);
#else
  return Atom(0);
#endif
}

value rig_pci_vfio_c_answers(value unit) {
  (void)unit;
#ifdef HAVE_VFIO
  struct vfio_group_status viable = {.argsz = sizeof viable,
                                     .flags = VFIO_GROUP_FLAGS_VIABLE};
  struct vfio_group_status not_viable = {
      .argsz = sizeof not_viable, .flags = VFIO_GROUP_FLAGS_CONTAINER_SET};

  /* Region 2 with a type capability, then sparse mmap areas. */
  uint8_t sparse[96];
  memset(sparse, 0, sizeof sparse);
  struct vfio_region_info *r = (struct vfio_region_info *)sparse;
  r->argsz = sizeof sparse;
  r->flags = VFIO_REGION_INFO_FLAG_READ | VFIO_REGION_INFO_FLAG_MMAP |
             VFIO_REGION_INFO_FLAG_CAPS;
  r->index = 2;
  r->cap_offset = 32;
  r->size = 0x1000000;
  r->offset = 0x20000000000UL;
  struct vfio_region_info_cap_type *t =
      (struct vfio_region_info_cap_type *)(sparse + 32);
  t->header.id = VFIO_REGION_INFO_CAP_TYPE;
  t->header.version = 1;
  t->header.next = 48;
  t->type = 3;
  t->subtype = 1;
  struct vfio_region_info_cap_sparse_mmap *m =
      (struct vfio_region_info_cap_sparse_mmap *)(sparse + 48);
  m->header.id = VFIO_REGION_INFO_CAP_SPARSE_MMAP;
  m->header.version = 1;
  m->nr_areas = 2;
  m->areas[0].offset = 0;
  m->areas[0].size = 0x1000;
  m->areas[1].offset = 0x3000;
  m->areas[1].size = 0xffd000;

  struct vfio_region_info plain = {
      .argsz = sizeof plain,
      .flags = VFIO_REGION_INFO_FLAG_READ | VFIO_REGION_INFO_FLAG_WRITE,
      .index = 7,
      .size = 0x1000,
      .offset = 0x70000000000UL};

  /* An IOMMU with a migration capability, then its ranges. */
  uint8_t iommu[104];
  memset(iommu, 0, sizeof iommu);
  struct vfio_iommu_type1_info *i = (struct vfio_iommu_type1_info *)iommu;
  i->argsz = sizeof iommu;
  i->flags = VFIO_IOMMU_INFO_PGSIZES | VFIO_IOMMU_INFO_CAPS;
  i->iova_pgsizes = 0x40201000;
  i->cap_offset = 24;
  struct vfio_iommu_type1_info_cap_migration *g =
      (struct vfio_iommu_type1_info_cap_migration *)(iommu + 24);
  g->header.id = VFIO_IOMMU_TYPE1_INFO_CAP_MIGRATION;
  g->header.version = 1;
  g->header.next = 56;
  g->pgsize_bitmap = 0x1000;
  struct vfio_iommu_type1_info_cap_iova_range *v =
      (struct vfio_iommu_type1_info_cap_iova_range *)(iommu + 56);
  v->header.id = VFIO_IOMMU_TYPE1_INFO_CAP_IOVA_RANGE;
  v->header.version = 1;
  v->nr_iovas = 2;
  v->iova_ranges[0].start = 0;
  v->iova_ranges[0].end = 0xfedfffff;
  v->iova_ranges[1].start = 0xfef00000;
  v->iova_ranges[1].end = 0xffffffffffffffffUL;

  struct vfio_iommu_type1_info bare = {.argsz = sizeof bare};

  const void *p[] = {&viable, &not_viable, sparse, &plain, iommu, &bare};
  const size_t len[] = {sizeof viable, sizeof not_viable, sizeof sparse,
                        sizeof plain,  sizeof iommu,      sizeof bare};
  return strings(6, p, len);
#else
  return Atom(0);
#endif
}
