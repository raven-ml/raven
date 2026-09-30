(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Work that never signals fails the device once its timeout passes. It runs in
   a process of its own, since a failed device stays failed. Skips on a machine
   without a CUDA device. *)

open Windtrap
module B = Nx_device.Buffer
module S = Nx_dtype.Scalar

external stall : nativeint -> nativeint -> nativeint -> int -> unit
  = "test_stall"

let test_hang () =
  if Nx_cuda_device.count () = 0 then skip ~reason:"no CUDA device" ();
  let d = Nx_cuda_device.v 0 in
  let c = Option.get (Nx_cuda_device.of_device d) in
  Nx_device.set_timeout d 500;
  Nx_device.submit [ d ] ~touches:[] (fun s ->
      stall (Nx_cuda_device.context c) (Nx_cuda_device.compute c)
        (B.address (Nx_device.signal_word d))
        (Nx_device.Submission.value s d));
  let hung = function
    | Nx_device.Lost (d', why) -> d' == d && why = "hang detected"
    | _ -> false
  in
  let t0 = Unix.gettimeofday () in
  raises_match hung (fun () -> Nx_device.synchronize d);
  is_true ~msg:"after the timeout" (Unix.gettimeofday () -. t0 >= 0.5);
  raises_match hung (fun () -> B.create d S.UInt8 1)

let () = exit (run "nx.cuda.device hang" [ test "hang" test_hang ])
