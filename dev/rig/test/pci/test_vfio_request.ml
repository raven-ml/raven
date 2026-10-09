(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* VFIO's requests, with no function: each request's number and parameters are
   the bytes a C program compiled against <linux/vfio.h> packs for the same
   arguments, and each reader decodes the bytes such a program lays out for the
   kernel's answer. The expected bytes are that program's, printed on x86_64
   Linux. *)

open Windtrap
module R = Vfio_request

let hex (p : R.params) =
  String.concat ""
    (List.init (Bigarray.Array1.dim p) (fun i ->
         Printf.sprintf "%02x" (Char.code (Bigarray.Array1.get p i))))

let of_hex h =
  let p =
    Bigarray.Array1.create Bigarray.char Bigarray.c_layout (String.length h / 2)
  in
  for i = 0 to Bigarray.Array1.dim p - 1 do
    Bigarray.Array1.set p i
      (Char.chr (int_of_string ("0x" ^ String.sub h (2 * i) 2)))
  done;
  p

let numbers =
  cases ~name:fst "a request's number is C's"
    [
      ("get_api_version", (R.get_api_version, 0x3b64));
      ("check_extension", (R.check_extension, 0x3b65));
      ("set_iommu", (R.set_iommu, 0x3b66));
      ("group_get_status", (R.group_get_status, 0x3b67));
      ("group_set_container", (R.group_set_container, 0x3b68));
      ("group_get_device_fd", (R.group_get_device_fd, 0x3b6a));
      ("device_get_region_info", (R.device_get_region_info, 0x3b6c));
      ("device_set_irqs", (R.device_set_irqs, 0x3b6e));
      ("device_reset", (R.device_reset, 0x3b6f));
      ("iommu_get_info", (R.iommu_get_info, 0x3b70));
      ("iommu_map_dma", (R.iommu_map_dma, 0x3b71));
      ("iommu_unmap_dma", (R.iommu_unmap_dma, 0x3b72));
      ("type1v2_iommu", (R.type1v2_iommu, 3));
      ("noiommu_iommu", (R.noiommu_iommu, 8));
      ("api_version", (R.api_version, 0));
      ("config_region", (R.config_region, 7));
    ]
    (fun (_, (number, c)) -> equal int c number)

let packing =
  cases ~name:fst "a request packs as C does"
    [
      ("group_status", (R.group_status (), "0800000000000000"));
      ("int", (R.int 5, "05000000"));
      ("name", (R.name "0000:01:00.0", "303030303a30313a30302e3000"));
      ( "region_info",
        ( R.region_info 2 ~size:0,
          "2000000000000000020000000000000000000000000000000000000000000000" )
      );
      ("msi", (R.msi 9, "180000002400000001000000000000000100000009000000"));
      ( "iommu_info",
        ( R.iommu_info ~size:0,
          "180000000000000000000000000000000000000000000000" ) );
      ( "dma_map",
        ( R.dma_map ~va:0x7f12_3456_0000 ~iova:0x1_0000_0000 ~bytes:0x20_0000,
          "200000000300000000005634127f000000000000010000000000200000000000" )
      );
      ( "dma_unmap",
        ( R.dma_unmap ~iova:0x1_0000_0000 ~bytes:0x20_0000,
          "180000000000000000000000010000000000200000000000" ) );
    ]
    (fun (_, (p, c)) -> equal string c (hex p))

(* Region 2 as the kernel answers it in 96 bytes: mappable, with a type
   capability, then sparse mmap areas. *)
let sparse_region =
  of_hex
    ("600000000d000000020000002000000000000001000000000000000000020000"
   ^ "0200010030000000030000000100000001000100000000000200000000000000"
   ^ "00000000000000000010000000000000003000000000000000d0ff0000000000")

let answers =
  group "answers"
    [
      test "a viable group" (fun () ->
          equal bool true (R.viable (of_hex "0800000001000000"));
          equal bool false (R.viable (of_hex "0800000002000000")));
      test "a region asked again for its capabilities" (fun () ->
          let p = R.region_info 2 ~size:96 in
          equal int 96 (Bigarray.Array1.dim p);
          equal int 96 (R.needs sparse_region));
      test "a region's sparse areas, past another capability" (fun () ->
          equal
            (quad int int bool (option (list (pair int int))))
            ( 0x100_0000,
              0x200_0000_0000,
              true,
              Some [ (0, 0x1000); (0x3000, 0xff_d000) ] )
            (R.region sparse_region));
      test "a region without capabilities" (fun () ->
          equal
            (quad int int bool (option (list (pair int int))))
            (0x1000, 0x700_0000_0000, false, None)
            (R.region
               (of_hex
                  "2000000003000000070000000000000000100000000000000000000000070000")));
      test "an IOMMU's page sizes and ranges, past another capability"
        (fun () ->
          equal
            (pair int (list (pair int int)))
            (0x4020_1000, [ (0, 0xfedf_ffff); (0xfef0_0000, max_int) ])
            (R.iommu
               (of_hex
                  ("6800000003000000001020400000000018000000000000000200010038000000"
                 ^ "0000000000000000001000000000000000000000000000000100010000000000"
                 ^ "02000000000000000000000000000000ffffdffe000000000000f0fe00000000"
                 ^ "ffffffffffffffff"))));
      test "an IOMMU that says nothing" (fun () ->
          equal
            (pair int (list (pair int int)))
            (0, [])
            (R.iommu
               (of_hex "180000000000000000000000000000000000000000000000")));
    ]

let () = exit (run "rig_pci.vfio_request" [ numbers; packing; answers ])
