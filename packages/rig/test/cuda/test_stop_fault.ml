(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module C = Rig_cuda
module S = Rig_cuda_support
module H = Rig_gpu_support.Host

let watchdog = 17 (* CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT *)
let second = 1_000_000_000
let address r = Option.get (C.locate r).address
let host r = Option.get (C.locate r).host

(* How long a reopen may refuse for the work of a stopped device: the 10 s
   bound of the kernel that holds the work running, past which it ends by
   itself. *)
let patience_ns = 10 * second

(* An open of GPU 0 once it no longer refuses for a stopped device's work, or
   that refusal after [patience_ns]. *)
let reopen () =
  let t0 = Rig.Profile.now () in
  let rec go () =
    match Rig_gpu_support.opened (C.gpu 0) with
    | Error e
      when Rig.Profile.now () - t0 < patience_ns
           && String.ends_with ~suffix:"still runs the work of a stopped device"
                e ->
        Domain.cpu_relax ();
        go ()
    | r -> r
  in
  go ()

(* Work a stop leaves running that faults afterwards ends there: CUDA runs
   none of the context's work after a fault, the write of the last value
   included. The GPU is then open to a new device, whose open raises the old
   device's word to its last value and answers the fault's error, as every
   CUDA call does after it. Value 1 runs a kernel until
   [flag] is set, then a store to address 0; a fill that fails in its submit
   loses the device, whose stop finds the kernel running. *)
let fault_after_stop () =
  let t = S.open_ () in
  if S.attribute watchdog <> 0 then begin
    S.close t;
    skip ~reason:"a display watchdog ends long kernels" ()
  end;
  let base = Rig.submitted t.d in
  let flag = require_some (C.alloc t.g Pinned 8) in
  H.set64 (host flag) 0;
  let spin = S.launch t "spin" (address flag) (10 * second) in
  let fault = S.launch t "fault" 0 0 in
  let fails = S.part t.d (S.failing 1) in
  (match S.submit_work t [ spin; fault; fails ] with
  | _ -> fail "the failed fill lost no device"
  | exception Rig.Lost _ -> ());
  equal ~msg:"the word at the stop" int base (C.signaled t.g);
  H.set64 (host flag) 1;
  let reopened = reopen () in
  equal ~msg:"the word, raised by the reopen" int (Rig.submitted t.d)
    (C.signaled t.g);
  match reopened with
  | Ok g ->
      C.stop g ~fault:None;
      fail "the GPU opened after its context faulted"
  | Error e -> contains ~msg:"the reopen" ~sub:"CUDA_ERROR_ILLEGAL_ADDRESS" e

let () =
  exit
    (run "rig_cuda stop fault"
       [
         group ~timeout:60. "stop"
           [
             test "work that faults after its device's stop ends the stop"
               fault_after_stop;
           ];
       ])
