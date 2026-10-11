(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module C = Rig_cuda
module S = Rig_cuda_support
module H = Rig_gpu_support.Host
module B = Rig.Buffer

let host r = Option.get (C.locate r).host
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
   fault, which loses the device and stops it. An image loaded before the fault,
   with one kernel entered, serves the calls that need one after it. *)
let faults () =
  let ({ S.d; g } as t) = S.open_ () in
  let image = S.loaded (require_ok (C.image g (S.fixture "kernels.ptx"))) in
  let empty =
    match C.entry image "empty" with
    | Entry e -> e.code
    | Absent | No_memory _ -> fail "no entry of empty"
  in
  let page = H.pages H.page in
  let watched = B.create ~memory:Pinned d 64 and src = B.create d 64 in
  let zeros = String.make 64 '\000' in
  B.write watched Bigarray.char (fun a -> Bigarray.Array1.fill a '\000');
  let addresses = Rig.Submission.make ~fixed:[ (src, Read_write) ] d [||] in
  S.write_gpu
    (Nativeint.of_int (Rig.Submission.address addresses src))
    (String.make 64 'x');
  equal int ~msg:"value 1" 1
    (Rig.Point.value (S.submit_work t [ S.launch t "fault" 0 0 ]));
  ignore (S.submit t [| Rig.Submission.copy ~src:src ~dst:watched |]);
  raises_match
    (function
      | C.Fault why ->
          String.starts_with
            ~prefix:"the GPU's work failed: CUDA_ERROR_ILLEGAL_ADDRESS" why
      | _ -> false)
    (fun () -> fault g (Rig.Profile.now ()));
  sticky "alloc" (fun () -> ignore (C.alloc g Device 64));
  sticky "image" (fun () -> ignore (C.image g (S.fixture "kernels.ptx")));
  sticky "entry" (fun () -> ignore (C.entry image "double_index"));
  if (C.facts g).maps_host then
    sticky "map_host" (fun () -> ignore (C.map_host g page 64));
  (match (C.capability g).graph [| S.kernel empty 0 0 |] with
  | Ok _ -> fail "a graph after the fault"
  | Error why -> contains ~msg:"graph" ~sub:"CUDA_ERROR_ILLEGAL_ADDRESS" why);
  sticky "unload" (fun () -> C.unload g image);
  H.free_pages page H.page;
  equal int ~msg:"the word after the fault" 0 (H.get64 (host (C.facts g).word));
  let v = S.submit t [||] in
  equal int ~msg:"value 3" 3 (Rig.Point.value v);
  (match Rig.Point.wait v with
  | () -> fail "a wait after the fault is no loss"
  | exception Rig.Lost (_, why) ->
      contains ~msg:"wait" ~sub:"CUDA_ERROR_ILLEGAL_ADDRESS" why);
  let w = C.signaled g in
  equal int ~msg:"the word after stop" 3 w;
  still ~msg:"the word" int w (fun () -> C.signaled g) ~ms:200;
  still ~msg:"the watched buffer" string zeros
    (fun () -> H.read (H.at watched) 64)
    ~ms:200;
  let e = require_error ((C.gpu 0).make ()) in
  contains ~sub:"CUDA_ERROR_ILLEGAL_ADDRESS" e

let () =
  exit
    (run "rig_cuda fault"
       [
         group ~timeout:60. "fault"
           [ test "a fault of the work is CUDA's report" faults ];
       ])
