(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module A = Device_amd
module S = Device_amd_support

let host r = Option.get (A.host r)
let submit g ~v ps = A.submit g ~v ~waits:[||] ~handles:[||] ps

let answer =
  Testable.make
    ~pp:(fun ppf -> function
      | `Ok -> Format.pp_print_string ppf "`Ok"
      | `Failed why -> Format.fprintf ppf "`Failed %S" why)
    ~equal:( = )

let gpus =
  group ~timeout:10. "GPUs"
    [
      test "AMD's display controllers and accelerators are GPUs" (fun () ->
          equal bool ~msg:"display" true
            (A.is_gpu ~vendor:0x1002 ~class_:0x030000);
          equal bool ~msg:"accelerator" true
            (A.is_gpu ~vendor:0x1002 ~class_:0x120000);
          equal bool ~msg:"audio" false
            (A.is_gpu ~vendor:0x1002 ~class_:0x040300);
          equal bool ~msg:"bridge" false
            (A.is_gpu ~vendor:0x1002 ~class_:0x060400);
          equal bool ~msg:"another vendor" false
            (A.is_gpu ~vendor:0x10de ~class_:0x030000));
    ]

let work =
  group ~timeout:60. "work"
    [
      test "an empty submission releases its value" (fun () ->
          S.with_gpu @@ fun g ->
          equal answer ~msg:"submit" `Ok (submit g ~v:1 [||]);
          S.wait g 1;
          equal int ~msg:"the word" 1 (A.signaled g));
      test "a copy through the device's memory comes back" (fun () ->
          S.with_gpu @@ fun g ->
          let n = 4096 in
          let src = Option.get (A.alloc g `Pinned n) in
          let mid = Option.get (A.alloc g `Device n) in
          let dst = Option.get (A.alloc g `Pinned n) in
          let bytes = String.init n (fun i -> Char.chr (i * 7 land 0xff)) in
          S.write (host src) bytes;
          let copy d s = A.part g ~queue:"COPY:0" (`Copy ((d, 0), (s, 0), n)) in
          equal answer ~msg:"submit" `Ok (submit g ~v:1 [| copy mid src |]);
          equal answer ~msg:"submit" `Ok (submit g ~v:2 [| copy dst mid |]);
          S.wait g 2;
          equal string ~msg:"the bytes" bytes (S.read (host dst) n);
          List.iter (A.free g) [ src; mid; dst ]);
      test "an idle device stops with its word at the last value" (fun () ->
          let g = S.gpu () in
          equal answer ~msg:"submit" `Ok (submit g ~v:1 [||]);
          S.wait g 1;
          let stopped =
            match S.stop g with `Stopped -> true | `Unknown -> false
          in
          equal bool ~msg:"stopped" true stopped;
          equal int ~msg:"the word" 1 (A.signaled g));
    ]

let () = exit (run "device_amd" [ gpus; work ])
