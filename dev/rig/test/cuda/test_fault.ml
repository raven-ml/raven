(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module C = Rig_cuda
module S = Rig_cuda_support
module H = Rig_gpu_support.Host
module B = Rig.Buffer

let host r = Option.get (C.host r)
let still = Rig_gpu_support.still

(* Raises the fault [sleep] reports, waiting at most 10 seconds of the host's
   monotonic clock, which a blocked or descheduled process does not slow. *)
let patience_ns = 10_000_000_000

let rec fault g t0 =
  if Rig.Profile.now () - t0 > patience_ns then fail "no fault within 10 s";
  C.sleep g ~seen:(C.signaled g) ~still_ms:100;
  fault g t0

(* [f ()] raises Fault with the fault's error, which a fault leaves in the
   context for every later call. *)
let sticky name f =
  match f () with
  | () -> failf "%s raised no Fault" name
  | exception C.Fault why ->
      contains ~msg:name ~sub:"CUDA_ERROR_ILLEGAL_ADDRESS" why

(* Value 1 stores to address 0; value 2, queued behind it, would copy into a
   watched host buffer and write the word. Value 3, of no parts, submitted after
   the fault through rig, makes no CUDA call; the wait for it commits, meets the
   fault, which loses the device and stops it. *)
let faults () =
  let ({ S.d; g } as t) = S.open_ () in
  let _, kernel = S.kernels g in
  let watched = B.create ~memory:Pinned d 64 and src = B.create d 64 in
  let zeros = String.make 64 '\000' in
  H.write (B.address watched) zeros;
  S.write_gpu (Nativeint.of_int (B.address src)) (String.make 64 'x');
  let f = S.launch (kernel "fault") ~grid:1 ~block:1 0 0 in
  equal int ~msg:"value 1" 1 (S.submit t [| S.part ~queue:"COMPUTE:0" f |]);
  ignore (S.submit t [| S.copy ~queue:"COMPUTE:0" ~dst:watched src |]);
  raises_match
    (function
      | C.Fault why ->
          String.starts_with
            ~prefix:"the GPU's work failed: CUDA_ERROR_ILLEGAL_ADDRESS" why
      | _ -> false)
    (fun () -> fault g (Rig.Profile.now ()));
  sticky "alloc" (fun () -> ignore (C.alloc g `Device 64));
  sticky "image" (fun () -> ignore (C.image g (S.fixture "kernels.ptx")));
  equal int ~msg:"the word after the fault" 0 (H.get64 (host (C.word g)));
  let v = S.submit t [||] in
  equal int ~msg:"value 3" 3 v;
  (match S.wait t v with
  | () -> fail "a wait after the fault is no loss"
  | exception Rig.Lost (_, why) ->
      contains ~msg:"wait" ~sub:"CUDA_ERROR_ILLEGAL_ADDRESS" why);
  let w = C.signaled g in
  equal int ~msg:"the word after stop" 3 w;
  still ~msg:"the word" int w (fun () -> C.signaled g) ~ms:200;
  still ~msg:"the watched buffer" string zeros
    (fun () -> H.read (B.address watched) 64)
    ~ms:200;
  let e = require_error (C.open_ 0) in
  contains ~sub:"CUDA_ERROR_ILLEGAL_ADDRESS" e

let () =
  S.hold ();
  exit
    (run "rig_cuda fault"
       [
         group ~timeout:60. "fault"
           [ test "a fault of the work is CUDA's report" faults ];
       ])
