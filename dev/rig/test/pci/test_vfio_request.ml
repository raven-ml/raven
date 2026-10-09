(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* VFIO's requests, with no function: each request's number and parameters are
   the bytes <linux/vfio.h> lays out for the same arguments, and each reader
   decodes the bytes it lays out for the kernel's answer. The expected bytes are
   a C program's, printed on x86_64 Linux; where the header is at hand, the
   suite also asks it for them, through test_vfio_request_stubs.c. *)

open Windtrap
module R = Vfio_request

external c_numbers : unit -> int array = "rig_pci_vfio_c_numbers"
external c_requests : unit -> string array = "rig_pci_vfio_c_requests"
external c_answers : unit -> string array = "rig_pci_vfio_c_answers"

let hex_of_string s =
  String.concat ""
    (List.init (String.length s) (fun i ->
         Printf.sprintf "%02x" (Char.code s.[i])))

let hex (p : R.params) =
  hex_of_string
    (String.init (Bigarray.Array1.dim p) (fun i -> Bigarray.Array1.get p i))

let of_hex h =
  let p =
    Bigarray.Array1.create Bigarray.char Bigarray.c_layout (String.length h / 2)
  in
  for i = 0 to Bigarray.Array1.dim p - 1 do
    Bigarray.Array1.set p i
      (Char.chr (int_of_string ("0x" ^ String.sub h (2 * i) 2)))
  done;
  p

(* Expected values, in test_vfio_request_stubs.c's order *)

let numbers =
  [
    ("get_api_version", R.get_api_version, 0x3b64);
    ("check_extension", R.check_extension, 0x3b65);
    ("set_iommu", R.set_iommu, 0x3b66);
    ("group_get_status", R.group_get_status, 0x3b67);
    ("group_set_container", R.group_set_container, 0x3b68);
    ("group_get_device_fd", R.group_get_device_fd, 0x3b6a);
    ("device_get_region_info", R.device_get_region_info, 0x3b6c);
    ("device_set_irqs", R.device_set_irqs, 0x3b6e);
    ("device_reset", R.device_reset, 0x3b6f);
    ("iommu_get_info", R.iommu_get_info, 0x3b70);
    ("iommu_map_dma", R.iommu_map_dma, 0x3b71);
    ("iommu_unmap_dma", R.iommu_unmap_dma, 0x3b72);
    ("type1v2_iommu", R.type1v2_iommu, 3);
    ("noiommu_iommu", R.noiommu_iommu, 8);
    ("api_version", R.api_version, 0);
    ("config_region", R.config_region, 7);
  ]

let requests =
  [
    ("group_status", R.group_status (), "0800000000000000");
    ("int", R.int 5, "05000000");
    ("name", R.name "0000:01:00.0", "303030303a30313a30302e3000");
    ( "region_info",
      R.region_info 2 ~size:0,
      "2000000000000000020000000000000000000000000000000000000000000000" );
    ("msi", R.msi 9, "180000002400000001000000000000000100000009000000");
    ( "iommu_info",
      R.iommu_info ~size:0,
      "180000000000000000000000000000000000000000000000" );
    ( "dma_map",
      R.dma_map ~va:0x7f12_3456_0000 ~iova:0x1_0000_0000 ~bytes:0x20_0000,
      "200000000300000000005634127f000000000000010000000000200000000000" );
    ( "dma_unmap",
      R.dma_unmap ~iova:0x1_0000_0000 ~bytes:0x20_0000,
      "180000000000000000000000010000000000200000000000" );
  ]

let viable = "0800000001000000"
let not_viable = "0800000002000000"

(* Region 2 as the kernel answers it in 96 bytes: mappable, with a type
   capability, then sparse mmap areas. *)
let sparse_region =
  "600000000d000000020000002000000000000001000000000000000000020000"
  ^ "0200010030000000030000000100000001000100000000000200000000000000"
  ^ "00000000000000000010000000000000003000000000000000d0ff0000000000"

let plain_region =
  "2000000003000000070000000000000000100000000000000000000000070000"

(* An IOMMU's answer in 104 bytes: page sizes, a migration capability, then its
   device address ranges. *)
let iommu =
  "6800000003000000001020400000000018000000000000000200010038000000"
  ^ "0000000000000000001000000000000000000000000000000100010000000000"
  ^ "02000000000000000000000000000000ffffdffe000000000000f0fe00000000"
  ^ "ffffffffffffffff"

let bare_iommu = "180000000000000000000000000000000000000000000000"

let answers =
  [
    ("viable", viable);
    ("not viable", not_viable);
    ("sparse region", sparse_region);
    ("plain region", plain_region);
    ("iommu", iommu);
    ("bare iommu", bare_iommu);
  ]

(* Tests *)

let numbered =
  cases
    ~name:(fun (n, _, _) -> n)
    "a request's number is the header's" numbers
    (fun (_, number, c) -> equal int c number)

let packing =
  cases
    ~name:(fun (n, _, _) -> n)
    "a request packs as the header lays it out" requests
    (fun (_, p, c) -> equal string c (hex p))

let reading =
  group "answers"
    [
      test "a viable group" (fun () ->
          equal bool true (R.viable (of_hex viable));
          equal bool false (R.viable (of_hex not_viable)));
      test "a region asked again for its capabilities" (fun () ->
          let p = R.region_info 2 ~size:96 in
          equal int 96 (Bigarray.Array1.dim p);
          equal int 96 (R.needs (of_hex sparse_region)));
      test "a region's sparse areas, past another capability" (fun () ->
          equal
            (quad int int bool (option (list (pair int int))))
            ( 0x100_0000,
              0x200_0000_0000,
              true,
              Some [ (0, 0x1000); (0x3000, 0xff_d000) ] )
            (R.region (of_hex sparse_region)));
      test "a region without capabilities" (fun () ->
          equal
            (quad int int bool (option (list (pair int int))))
            (0x1000, 0x700_0000_0000, false, None)
            (R.region (of_hex plain_region)));
      test "an IOMMU's page sizes and ranges, past another capability"
        (fun () ->
          equal
            (pair int (list (pair int int)))
            (0x4020_1000, [ (0, 0xfedf_ffff); (0xfef0_0000, max_int) ])
            (R.iommu (of_hex iommu)));
      test "an IOMMU that says nothing" (fun () ->
          equal
            (pair int (list (pair int int)))
            (0, [])
            (R.iommu (of_hex bare_iommu)));
    ]

(* The values above, against <linux/vfio.h> where it is at hand. *)
let header =
  let check name expected c =
    test name (fun () ->
        let c = c () in
        if Array.length c = 0 then skip ~reason:"no <linux/vfio.h>" ();
        equal (list string) expected (Array.to_list c))
  in
  group "<linux/vfio.h>"
    [
      check "the numbers are the header's"
        (List.map (fun (_, _, n) -> string_of_int n) numbers)
        (fun () -> Array.map string_of_int (c_numbers ()));
      check "the requests are laid out as the header's"
        (List.map (fun (_, p, _) -> hex p) requests)
        (fun () -> Array.map hex_of_string (c_requests ()));
      check "the answers are laid out as the header's" (List.map snd answers)
        (fun () -> Array.map hex_of_string (c_answers ()));
    ]

let () =
  exit (run "rig_pci.vfio_request" [ numbered; packing; reading; header ])
