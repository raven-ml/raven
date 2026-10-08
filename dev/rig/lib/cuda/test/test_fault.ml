(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module C = Rig_cuda
module S = Rig_cuda_support
module B = Rig.Buffer

let host r = Option.get (C.host r)

(* Raises the fault [sleep] reports, waiting at most 10 seconds. *)
let rec fault g t0 =
  if Sys.time () -. t0 > 10. then fail "no fault within 10 s";
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
   watched host buffer and write the word. Value 3 is submitted after the fault,
   through rig, which loses the device and stops it. *)
let faults () =
  let g = S.gpu () in
  let c = S.core g in
  let _, kernel = S.kernels g in
  let watched = B.create ~memory:Pinned c 64 and src = B.create c 64 in
  let zeros = String.make 64 '\000' in
  S.write (B.address watched) zeros;
  S.write_gpu (Nativeint.of_int (B.address src)) (String.make 64 'x');
  let f = S.launch (kernel "fault") ~grid:1 ~block:1 0 0 in
  equal int ~msg:"value 1" 1 (S.submit g [| S.part ~queue:"COMPUTE:0" f |]);
  ignore (S.submit g [| S.copy ~queue:"COMPUTE:0" ~dst:watched src |]);
  raises_match
    (function
      | C.Fault why ->
          String.starts_with
            ~prefix:"the GPU's work failed: CUDA_ERROR_ILLEGAL_ADDRESS" why
      | _ -> false)
    (fun () -> fault g (Sys.time ()));
  sticky "alloc" (fun () -> ignore (C.alloc g `Device 64));
  sticky "image" (fun () -> ignore (C.image g (S.fixture "kernels.ptx")));
  equal int ~msg:"the word after the fault" 0 (S.get64 (host (C.word g)));
  (match S.submit g [||] with
  | _ -> fail "a submit after the fault is no loss"
  | exception Rig.Lost (_, why) ->
      contains ~msg:"submit" ~sub:"CUDA_ERROR_ILLEGAL_ADDRESS" why);
  let w = C.signaled g in
  equal int ~msg:"the word after stop" 3 w;
  S.still ~msg:"the word" int w (fun () -> C.signaled g) ~ms:200;
  S.still ~msg:"the watched buffer" string zeros
    (fun () -> S.read (B.address watched) 64)
    ~ms:200;
  let e = require_error (C.open_ 0) in
  contains ~sub:"CUDA_ERROR_ILLEGAL_ADDRESS" e

let () =
  S.hold_gpu ();
  exit
    (run "rig_cuda fault"
       [
         group ~timeout:60. "fault"
           [ test "a fault of the work is CUDA's report" faults ];
       ])
