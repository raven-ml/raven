(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module C = Device_cuda
module S = Device_cuda_support

let host r = Option.get (C.host r)

let answer =
  Testable.make
    ~pp:(fun ppf -> function
      | `Ok -> Format.pp_print_string ppf "`Ok"
      | `Failed why -> Format.fprintf ppf "`Failed %S" why)
    ~equal:( = )

(* Reads [f ()] for about [ms] milliseconds of CPU time, failing if it is ever
   other than [x]. *)
let still ~msg w x f ~ms =
  let t0 = Sys.time () in
  while Sys.time () -. t0 < Float.of_int ms /. 1000. do
    equal w ~msg x (f ())
  done

(* Raises the fault [sleep] reports, waiting at most 10 seconds. *)
let rec fault g t0 =
  if Sys.time () -. t0 > 10. then fail "no fault within 10 s";
  C.sleep g ~seen:(C.signaled g) ~still_ms:100;
  fault g t0

(* Value 1 stores to address 0; value 2, queued behind it, would copy into a
   watched host buffer and write the word. *)
let faults () =
  let g = S.gpu () in
  let _, kernel = S.kernels g in
  let watched = require_some (C.alloc g `Pinned 64) in
  let src = require_some (C.alloc g `Device 64) in
  let zeros = String.make 64 '\000' in
  S.write (host watched) zeros;
  S.write_gpu (C.handle src) (String.make 64 'x');
  let f = S.launch (kernel "fault") ~grid:1 ~block:1 0 0 in
  equal answer `Ok
    (C.submit g ~v:1 ~waits:[||] ~handles:[||]
       [| S.part g ~queue:"COMPUTE:0" f |]);
  let copy = C.part g ~queue:"COMPUTE:0" (`Copy ((watched, 0), (src, 0), 64)) in
  ignore (C.submit g ~v:2 ~waits:[||] ~handles:[||] [| copy |]);
  raises_match
    (function
      | C.Fault why ->
          String.starts_with
            ~prefix:"the GPU's work failed: CUDA_ERROR_ILLEGAL_ADDRESS" why
      | _ -> false)
    (fun () -> fault g (Sys.time ()));
  equal int ~msg:"the word after the fault" 0 (S.get64 (host (C.word g)));
  (match C.stop g with
  | `Stopped -> ()
  | `Unknown -> fail "stop is Unknown after a fault");
  let w = C.signaled g in
  equal int ~msg:"the word after stop" 2 w;
  still ~msg:"the word" int w (fun () -> C.signaled g) ~ms:200;
  still ~msg:"the watched buffer" string zeros
    (fun () -> S.read (host watched) 64)
    ~ms:200;
  let e = require_error (C.open_ 0) in
  contains ~sub:"CUDA_ERROR_ILLEGAL_ADDRESS" e

let () =
  exit
    (run "device_cuda fault"
       [
         group ~timeout:60. "fault"
           [ test "a fault of the work is CUDA's report" faults ];
       ])
