/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <stdlib.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

/* [n] zeroed bytes the test keeps for good, as mapped memory stands in for. */
value test_support_alloc(value n) {
  void *p = calloc(1, Long_val(n));
  if (p == NULL) caml_raise_out_of_memory();
  return caml_copy_nativeint((intnat)p);
}

/* Linux's own VFIO constants and layouts, from its headers, where the system
   has them: the request numbers in the order of [Vfio.request]'s constructors,
   then the IOMMU models, the configuration region, and the sizes and field
   offsets of the structures. Empty elsewhere. */

#if defined(__linux__) && defined(__has_include)
#if __has_include(<linux/vfio.h>)
#include <linux/vfio.h>
#include <stddef.h>
#define HAVE_VFIO_H
#endif
#endif

value test_support_vfio_header(value unit) {
  (void)unit;
#ifdef HAVE_VFIO_H
  static const long v[] = {
      VFIO_GET_API_VERSION,
      VFIO_CHECK_EXTENSION,
      VFIO_SET_IOMMU,
      VFIO_GROUP_GET_STATUS,
      VFIO_GROUP_SET_CONTAINER,
      VFIO_GROUP_GET_DEVICE_FD,
      VFIO_DEVICE_GET_REGION_INFO,
      VFIO_DEVICE_SET_IRQS,
      VFIO_DEVICE_RESET,
      VFIO_IOMMU_GET_INFO,
      VFIO_IOMMU_MAP_DMA,
      VFIO_IOMMU_UNMAP_DMA,
      VFIO_API_VERSION,
      VFIO_TYPE1v2_IOMMU,
      VFIO_NOIOMMU_IOMMU,
      VFIO_PCI_CONFIG_REGION_INDEX,
      sizeof(struct vfio_group_status),
      sizeof(struct vfio_region_info),
      offsetof(struct vfio_region_info, cap_offset),
      offsetof(struct vfio_region_info, size),
      offsetof(struct vfio_region_info, offset),
      sizeof(struct vfio_irq_set),
      sizeof(struct vfio_iommu_type1_info),
      offsetof(struct vfio_iommu_type1_info, iova_pgsizes),
      offsetof(struct vfio_iommu_type1_info, cap_offset),
      sizeof(struct vfio_iommu_type1_dma_map),
      offsetof(struct vfio_iommu_type1_dma_map, vaddr),
      offsetof(struct vfio_iommu_type1_dma_map, iova),
      offsetof(struct vfio_iommu_type1_dma_map, size),
      sizeof(struct vfio_iommu_type1_dma_unmap),
      offsetof(struct vfio_iommu_type1_dma_unmap, iova),
      offsetof(struct vfio_iommu_type1_dma_unmap, size),
      VFIO_GROUP_FLAGS_VIABLE,
      VFIO_REGION_INFO_FLAG_READ,
      VFIO_REGION_INFO_FLAG_WRITE,
      VFIO_REGION_INFO_FLAG_MMAP,
      VFIO_REGION_INFO_FLAG_CAPS,
      VFIO_REGION_INFO_CAP_SPARSE_MMAP,
      VFIO_IRQ_SET_DATA_EVENTFD | VFIO_IRQ_SET_ACTION_TRIGGER,
      VFIO_PCI_MSI_IRQ_INDEX,
      VFIO_IOMMU_INFO_PGSIZES,
      VFIO_IOMMU_INFO_CAPS,
      VFIO_IOMMU_TYPE1_INFO_CAP_IOVA_RANGE,
      VFIO_IOMMU_TYPE1_INFO_DMA_AVAIL,
      VFIO_DMA_MAP_FLAG_READ | VFIO_DMA_MAP_FLAG_WRITE,
  };
  size_t n = sizeof v / sizeof v[0];
  value a = caml_alloc_tuple(n);
  for (size_t i = 0; i < n; i++) Store_field(a, i, Val_long(v[i]));
  return a;
#else
  return Atom(0);
#endif
}
