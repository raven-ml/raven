(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module A = Device_amd
module S = Device_amd_support
module Abi = Device_amd_abi
module Pm4 = Abi.Pm4

let host r = Option.get (A.host r)
let address r = Option.get (A.address r)
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

(* Kernels *)

let binary =
  lazy
    (In_channel.with_open_bin "fixtures/kernels_gfx1201.hsaco"
       In_channel.input_all)

let code_object =
  lazy (Result.get_ok (Abi.Code_object.of_string (Lazy.force binary)))

let words p =
  let s = Abi.Packet.encode Int64.of_int p in
  Array.init
    (String.length s / 4)
    (fun i -> Int32.to_int (String.get_int32_le s (4 * i)) land 0xffff_ffff)

let le64 n =
  let b = Bytes.create 8 in
  Bytes.set_int64_le b 0 (Int64.of_int n);
  Bytes.to_string b

(* A device's values, numbered as it submits them. *)
type run = { g : A.t; mutable v : int }

let device g = { g; v = 0 }

let go r ps =
  r.v <- r.v + 1;
  equal answer ~msg:"submit" `Ok (submit r.g ~v:r.v ps);
  S.wait r.g r.v

(* [m]'s image, and the part that copies it into its memory from staging memory
   the host wrote. *)
let image r =
  match A.image r.g (Lazy.force binary) with
  | Ok (m, Some (code, bytes)) ->
      let n = String.length bytes in
      let staging = Option.get (A.alloc r.g `Pinned n) in
      S.write (host staging) bytes;
      (m, A.part r.g ~queue:"COPY:0" (`Copy ((code, 0), (staging, 0), n)))
  | Ok (_, None) -> fail "an image without code"
  | Error why -> fail why

(* A dispatch of [name] over [groups] workgroups of 64, its arguments [args]. *)
let dispatch r m name ~args ~groups =
  let gpu = (A.capability r.g).gpu in
  let k = Option.get (Abi.Code_object.kernel (Lazy.force code_object) name) in
  let base = Option.get (A.entry m name) - k.descriptor in
  words
    (Pm4.run gpu
       (Pm4.dispatch gpu k ~program:(base + k.entry) ~scratch:0 ~args ~packet:0
          ~threads:(64, 1, 1) ~groups:(groups, 1, 1) ()))

let arguments r values =
  let args = Option.get (A.alloc r.g `Pinned 4096) in
  S.write (host args) (String.concat "" (List.map le64 values));
  args

let doubled n =
  String.concat ""
    (List.init n (fun i ->
         let b = Bytes.create 4 in
         Bytes.set_int32_le b 0 (Int32.of_int (2 * i));
         Bytes.to_string b))

let kernels =
  group ~timeout:60. "kernels"
    [
      test "a kernel loaded from its image computes" (fun () ->
          S.with_gpu @@ fun g ->
          let r = device g in
          let m, upload = image r in
          go r [| upload |];
          let out = Option.get (A.alloc g `Pinned (4 * 256)) in
          let args = arguments r [ address out ] in
          let ws = dispatch r m "double_index" ~args:(address args) ~groups:4 in
          go r [| A.part g ~queue:"COMPUTE:0" (`Words ws) |];
          equal string ~msg:"out" (doubled 256) (S.read (host out) (4 * 256)));
      test "parts on two queues run in their after order" (fun () ->
          S.with_gpu @@ fun g ->
          let r = device g in
          let m, upload = image r in
          let out = Option.get (A.alloc g `Device (4 * 256)) in
          let back = Option.get (A.alloc g `Pinned (4 * 256)) in
          let args = arguments r [ address out ] in
          let ws = dispatch r m "double_index" ~args:(address args) ~groups:4 in
          go r
            [|
              upload;
              A.part g ~queue:"COMPUTE:0" ~after:[| 0 |] (`Words ws);
              A.part g ~queue:"COPY:0" ~after:[| 1 |]
                (`Copy ((back, 0), (out, 0), 4 * 256));
            |];
          equal string ~msg:"back" (doubled 256) (S.read (host back) (4 * 256)));
      test "a fill places its words" (fun () ->
          S.with_gpu @@ fun g ->
          let r = device g in
          let word = Option.get (A.alloc g `Pinned 8) in
          let ws = words (Pm4.write_data (Memory (address word)) 0xc0ffee) in
          let f, arg = S.fill (A.capability g) ws ~bytes:64 in
          let units = Array.length ws in
          go r [| A.part g ~queue:"COMPUTE:0" (`Fill (f, arg, units, 64)) |];
          equal string ~msg:"word" (le64 0xc0ffee) (S.read (host word) 8));
      test "a fill past its declaration fails, and the word still moves"
        (fun () ->
          S.with_gpu @@ fun g ->
          let word = Option.get (A.alloc g `Pinned 8) in
          let ws = words (Pm4.write_data (Memory (address word)) 1) in
          let f, arg = S.fill (A.capability g) ws ~bytes:0 in
          let p =
            A.part g ~queue:"COMPUTE:0" (`Fill (f, arg, Array.length ws - 1, 0))
          in
          let failed = `Failed "a fill on COMPUTE:0 failed with 1" in
          equal answer ~msg:"submit" failed (submit g ~v:1 [| p |]);
          S.wait g 1;
          equal answer ~msg:"the next" failed (submit g ~v:2 [||]));
      test "long work is no fault" (fun () ->
          S.with_gpu @@ fun g ->
          let r = device g in
          let m, upload = image r in
          go r [| upload |];
          let flag = Option.get (A.alloc g `Pinned 8) in
          let args = arguments r [ address flag; 150_000 ] in
          let ws = dispatch r m "spin" ~args:(address args) ~groups:1 in
          go r [| A.part g ~queue:"COMPUTE:0" (`Words ws) |];
          equal string ~msg:"flag"
            (String.sub (le64 1) 0 4)
            (S.read (host flag) 4));
      test "a device stopped while its work runs stops it" (fun () ->
          let g = S.gpu () in
          let r = device g in
          let m, upload = image r in
          go r [| upload |];
          let flag = Option.get (A.alloc g `Pinned 8) in
          let args = arguments r [ address flag; 1_500_000 ] in
          let ws = dispatch r m "spin" ~args:(address args) ~groups:1 in
          equal answer ~msg:"submit" `Ok
            (submit g ~v:2 [| A.part g ~queue:"COMPUTE:0" (`Words ws) |]);
          let stopped =
            match S.stop g with `Stopped -> true | `Unknown -> false
          in
          equal bool ~msg:"stopped" true stopped;
          equal int ~msg:"the word" 2 (A.signaled g);
          let t0 = Sys.time () in
          while Sys.time () -. t0 < 0.1 do
            equal string ~msg:"flag (sampled)" (String.make 4 '\000')
              (S.read (host flag) 4)
          done);
    ]

let () = exit (run "device_amd" [ gpus; work; kernels ])
