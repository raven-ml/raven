(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module System = Tolk_amd.System
module Pci_device = Tolk_amd.System.Pci_device
module File_io = Tolk_amd.Hcq.File_io

let ( // ) = Filename.concat

let uid =
  let n = ref 0 in
  fun () ->
    incr n;
    Printf.sprintf "%d_%d" (Unix.getpid ()) !n

let mkdir_p dir =
  let rec go d =
    if not (Sys.file_exists d) then begin
      go (Filename.dirname d);
      Sys.mkdir d 0o755
    end
  in
  go dir

let rec rm_rf path =
  if Sys.is_directory path then begin
    Array.iter (fun e -> rm_rf (path // e)) (Sys.readdir path);
    Sys.rmdir path
  end
  else Sys.remove path

let write_file path content =
  Out_channel.with_open_bin path (fun oc ->
      Out_channel.output_string oc content)

let read_file path = In_channel.with_open_bin path In_channel.input_all

(* A fake sysfs tree in a temp dir; [Pci_device] and [pci_scan_bus] take its
   root through their [?sysfs] parameter. *)
let with_fake_root f =
  let root = Filename.get_temp_dir_name () // ("tolk_system_test_" ^ uid ()) in
  mkdir_p (root // "bus" // "pci" // "devices");
  Fun.protect ~finally:(fun () -> rm_rf root) (fun () -> f root)

let dev_dir root pcibus = root // "bus" // "pci" // "devices" // pcibus

let add_dev root pcibus ~vendor ~device ~cls =
  let dir = dev_dir root pcibus in
  mkdir_p dir;
  write_file (dir // "vendor") (Printf.sprintf "0x%04x\n" vendor);
  write_file (dir // "device") (Printf.sprintf "0x%04x\n" device);
  write_file (dir // "class") (Printf.sprintf "0x%06x\n" cls)

(* Enough of a device for [Pci_device.create] to succeed against it. *)
let add_full_dev root pcibus =
  add_dev root pcibus ~vendor:0x1002 ~device:0x744c ~cls:0x030000;
  let dir = dev_dir root pcibus in
  write_file (dir // "enable") "0\n";
  Out_channel.with_open_bin (dir // "config") (fun oc ->
      Out_channel.output_bytes oc (Bytes.init 64 (fun i -> Char.chr i)));
  write_file (dir // "resource")
    "0x00000000f0000000 0x00000000f7ffffff 0x0000000000140204\n\
     0x0000000000000000 0x0000000000000000 0x0000000000000000\n\
     0x000000fc00000000 0x000000fdffffffff 0x0000000000140204\n"

(* Device locks are keyed by devpref and pcibus and live for the whole
   process, so every [create] needs a fresh prefix. *)
let create_dev root pcibus =
  Pci_device.create ~sysfs:root ~devpref:("tolktest" ^ uid ()) pcibus

let is_failure = function Failure _ -> true | _ -> false

let () =
  run "System"
    [
      group "constants"
        [
          test "page size is a positive multiple of 4KB" (fun () ->
              is_true (System.page_size > 0);
              equal int 0 (System.page_size mod 4096));
          test "mmap flags match availability" (fun () ->
              let flags =
                [
                  System.map_locked;
                  System.map_populate;
                  System.map_hugetlb;
                  System.map_fixed_noreplace;
                ]
              in
              if System.available then
                is_true (List.for_all (fun f -> f <> 0) flags)
              else is_true (List.for_all (fun f -> f = 0) flags));
        ];
      group "pci_scan_bus"
        [
          test "filters by vendor and device id and sorts" (fun () ->
              with_fake_root (fun root ->
                  add_dev root "0000:03:00.0" ~vendor:0x1002 ~device:0x744c
                    ~cls:0x030000;
                  add_dev root "0000:01:00.0" ~vendor:0x1002 ~device:0x744c
                    ~cls:0x030000;
                  add_dev root "0000:05:00.0" ~vendor:0x10de ~device:0x744c
                    ~cls:0x030000;
                  add_dev root "0000:04:00.0" ~vendor:0x1002 ~device:0x1478
                    ~cls:0x060400;
                  equal (list string)
                    [ "0000:01:00.0"; "0000:03:00.0" ]
                    (System.pci_scan_bus ~sysfs:root ~vendor:0x1002
                       [ (0xFFFF, [ 0x744c ]) ])));
          test "masks the device id before matching" (fun () ->
              with_fake_root (fun root ->
                  add_dev root "0000:03:00.0" ~vendor:0x1002 ~device:0x744c
                    ~cls:0x030000;
                  add_dev root "0000:04:00.0" ~vendor:0x1002 ~device:0x1478
                    ~cls:0x060400;
                  equal (list string) [ "0000:03:00.0" ]
                    (System.pci_scan_bus ~sysfs:root ~vendor:0x1002
                       [ (0xFF00, [ 0x7400 ]) ])));
          test "restricts to the base class when given" (fun () ->
              with_fake_root (fun root ->
                  add_dev root "0000:03:00.0" ~vendor:0x1002 ~device:0x744c
                    ~cls:0x030000;
                  add_dev root "0000:04:00.0" ~vendor:0x1002 ~device:0x1478
                    ~cls:0x060400;
                  equal (list string) [ "0000:04:00.0" ]
                    (System.pci_scan_bus ~sysfs:root ~base_class:0x06
                       ~vendor:0x1002
                       [ (0xFFFF, [ 0x1478 ]) ])));
          test "no PCI bus raises" (fun () ->
              with_fake_root (fun root ->
                  raises (Failure "no pcie") (fun () ->
                      System.pci_scan_bus ~sysfs:(root // "nothing")
                        ~vendor:0x1002
                        [ (0xFFFF, [ 0x744c ]) ])));
        ];
      group "system_paddrs"
        [
          test "decodes page-map entries" (fun () ->
              let ps = System.page_size in
              let path = Filename.temp_file "tolk_pagemap" ".bin" in
              Fun.protect
                ~finally:(fun () -> Sys.remove path)
                (fun () ->
                  (* Two entries at the slot for the third page: physical
                     frame numbers with kernel flag bits (63, 62, 55) set on
                     top, which decoding must mask off. *)
                  let buf = Bytes.make ((3 * 8) + 16) '\000' in
                  Bytes.set_int64_le buf 24 0xC080_0000_0000_1234L;
                  Bytes.set_int64_le buf 32 0x8000_0000_0000_0001L;
                  Out_channel.with_open_bin path (fun oc ->
                      Out_channel.output_bytes oc buf);
                  let fd = File_io.openfile path ~flags:File_io.o_rdonly in
                  Fun.protect
                    ~finally:(fun () -> File_io.close fd)
                    (fun () ->
                      equal (list int)
                        [ 0x1234 * ps; ps ]
                        (System.system_paddrs ~pagemap:fd
                           ~vaddr:(Nativeint.of_int (3 * ps))
                           (2 * ps)))));
          test "truncated page map raises" (fun () ->
              let path = Filename.temp_file "tolk_pagemap" ".bin" in
              Fun.protect
                ~finally:(fun () -> Sys.remove path)
                (fun () ->
                  write_file path "\x01\x00";
                  let fd = File_io.openfile path ~flags:File_io.o_rdonly in
                  Fun.protect
                    ~finally:(fun () -> File_io.close fd)
                    (fun () ->
                      raises_match
                        (Exn.failure ~substring:"unexpected end of file")
                        (fun () ->
                          System.system_paddrs ~pagemap:fd ~vaddr:0n
                            System.page_size))));
        ];
      group "write_sysfs"
        [
          test "matching value is a no-op" (fun () ->
              let path = Filename.temp_file "tolk_sysfs" ".txt" in
              Fun.protect
                ~finally:(fun () -> Sys.remove path)
                (fun () ->
                  write_file path "1\n";
                  System.write_sysfs path ~value:"1" ~msg:"should not happen";
                  equal string "1\n" (read_file path)));
          test "matching expected value is a no-op" (fun () ->
              let path = Filename.temp_file "tolk_sysfs" ".txt" in
              Fun.protect
                ~finally:(fun () -> Sys.remove path)
                (fun () ->
                  write_file path "on\n";
                  System.write_sysfs path ~value:"0" ~expected:"on"
                    ~msg:"should not happen";
                  equal string "on\n" (read_file path)));
        ];
      group "flock_acquire"
        [
          test "acquires and blocks a second holder" (fun () ->
              let name = Printf.sprintf "tolk_test_%s.lock" (uid ()) in
              let fd = System.flock_acquire name in
              is_true (fd >= 0);
              raises_match (Exn.failure ~substring:"Failed to acquire lock")
                (fun () -> System.flock_acquire name));
        ];
      group "Pci_device"
        [
          test "create enables the device" (fun () ->
              with_fake_root (fun root ->
                  add_full_dev root "0000:03:00.0";
                  let t = create_dev root "0000:03:00.0" in
                  equal string "0000:03:00.0" (Pci_device.pcibus t);
                  is_true (Pci_device.lock_fd t >= 0);
                  starts_with ~affix:"1"
                    (read_file (dev_dir root "0000:03:00.0" // "enable"))));
          test "create hot-removes sibling functions" (fun () ->
              with_fake_root (fun root ->
                  add_full_dev root "0000:04:00.0";
                  let sib = dev_dir root "0000:04:00.1" in
                  mkdir_p sib;
                  write_file (sib // "remove") "0\n";
                  let t = create_dev root "0000:04:00.0" in
                  equal string "0000:04:00.0" (Pci_device.pcibus t);
                  starts_with ~affix:"1" (read_file (sib // "remove"))));
          test "create unbinds and reports a stuck driver" (fun () ->
              with_fake_root (fun root ->
                  add_full_dev root "0000:05:00.0";
                  let dir = dev_dir root "0000:05:00.0" in
                  mkdir_p (dir // "driver");
                  write_file (dir // "driver" // "unbind") "";
                  raises (Failure "Driver is bound to 0000:05:00.0") (fun () ->
                      create_dev root "0000:05:00.0");
                  equal string "0000:05:00.0"
                    (read_file (dir // "driver" // "unbind"))));
          test "create tolerates a device with no driver" (fun () ->
              with_fake_root (fun root ->
                  add_full_dev root "0000:06:00.0";
                  let t = create_dev root "0000:06:00.0" in
                  equal string "0000:06:00.0" (Pci_device.pcibus t)));
          test "create reports a missing device" (fun () ->
              with_fake_root (fun root ->
                  add_dev root "0000:07:00.0" ~vendor:0x1002 ~device:0x744c
                    ~cls:0x030000;
                  raises_match (Exn.failure ~substring:"enable") (fun () ->
                      create_dev root "0000:07:00.0")));
          test "create reports permission problems with guidance" (fun () ->
              if Unix.geteuid () = 0 then skip ~reason:"running as root" ()
              else
                with_fake_root (fun root ->
                    add_full_dev root "0000:08:00.0";
                    Unix.chmod (dev_dir root "0000:08:00.0" // "enable") 0o000;
                    raises_match
                      (Exn.failure ~substring:"Cannot access PCI device")
                      (fun () -> create_dev root "0000:08:00.0")));
          test "config space reads and writes little-endian" (fun () ->
              with_fake_root (fun root ->
                  add_full_dev root "0000:09:00.0";
                  let t = create_dev root "0000:09:00.0" in
                  equal int 0x05040302
                    (Pci_device.read_config t ~offset:2 ~size:4);
                  Pci_device.write_config t ~offset:8 ~value:0xAABBCCDD ~size:4;
                  equal int 0xAABBCCDD
                    (Pci_device.read_config t ~offset:8 ~size:4);
                  Pci_device.write_config_flush t ~offset:12 ~value:0x1234
                    ~size:2;
                  equal int 0x1234 (Pci_device.read_config t ~offset:12 ~size:2);
                  equal int 0x0F (Pci_device.read_config t ~offset:15 ~size:1)));
          test "bar_info parses resource lines" (fun () ->
              with_fake_root (fun root ->
                  add_full_dev root "0000:0a:00.0";
                  let t = create_dev root "0000:0a:00.0" in
                  equal (pair int int)
                    (0xf0000000, 0x8000000)
                    (Pci_device.bar_info t 0);
                  equal (pair int int)
                    (0xfc00000000, 0x200000000)
                    (Pci_device.bar_info t 2);
                  raises_match (Exn.failure ~substring:"no line for BAR 7")
                    (fun () -> Pci_device.bar_info t 7)));
          test "resize_bar picks the largest supported size" (fun () ->
              with_fake_root (fun root ->
                  add_full_dev root "0000:0b:00.0";
                  let dir = dev_dir root "0000:0b:00.0" in
                  write_file (dir // "resource0_resize") "00000000000001c0\n";
                  let t = create_dev root "0000:0b:00.0" in
                  Pci_device.resize_bar t 0;
                  starts_with ~affix:"8" (read_file (dir // "resource0_resize"));
                  raises_match (Exn.failure ~substring:"Cannot resize BAR 1")
                    (fun () -> Pci_device.resize_bar t 1)));
        ];
      group "alloc_sysmem"
        [
          test "rejects contiguous allocations over 2MB" (fun () ->
              raises
                (Invalid_argument
                   "Contiguous allocation is only supported for sizes up to \
                    2MB")
                (fun () ->
                  Pci_device.alloc_sysmem ~contiguous:true (3 * (2 lsl 20))));
          test "fails cleanly without a page map" (fun () ->
              if System.available then
                skip ~reason:"page map may be functional on Linux" ()
              else
                raises_match is_failure (fun () ->
                    Pci_device.alloc_sysmem System.page_size));
        ];
    ]
