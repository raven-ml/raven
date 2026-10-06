(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A kernel that runs for 35 seconds completes: only the driver declares a
   fault, and a wait lasts until the work signals. It runs in a process of its
   own, since it holds the GPU for that long. Skips on a machine without a CUDA
   device, and on a GPU whose driver ends long kernels (a display's
   watchdog). *)

open Windtrap
module B = Nx_device.Buffer
module S = Nx_dtype.Scalar

external launch :
  nativeint -> nativeint -> nativeint -> nativeint -> nativeint -> int -> unit
  = "test_launch_byte" "test_launch"

external watchdog : int -> bool = "test_watchdog"

let seconds = 35

(* Spins until [seconds] have passed on the GPU's global timer, then writes the
   seconds it spun into its argument. *)
let ptx =
  Printf.sprintf
    {|.version 7.0
.target sm_50
.address_size 64
.visible .entry spin(.param .u64 out) {
  .reg .u32 %%r1;
  .reg .u64 %%rd<7>;
  .reg .pred %%p;
  ld.param.u64 %%rd1, [out];
  cvta.to.global.u64 %%rd2, %%rd1;
  mov.u64 %%rd3, %%globaltimer;
LOOP:
  mov.u64 %%rd4, %%globaltimer;
  sub.u64 %%rd5, %%rd4, %%rd3;
  setp.lt.u64 %%p, %%rd5, %d000000000;
  @%%p bra LOOP;
  div.u64 %%rd6, %%rd5, 1000000000;
  cvt.u32.u64 %%r1, %%rd6;
  st.global.u32 [%%rd2], %%r1;
  ret;
}
|}
    seconds

let test_long () =
  if Nx_cuda_device.count () = 0 then skip ~reason:"no CUDA device" ();
  if watchdog 0 then skip ~reason:"the driver ends long kernels" ();
  let d = match Nx_cuda_device.get 0 with Ok d -> d | Error e -> failwith e in
  let c = Option.get (Nx_cuda_device.of_device d) in
  let f =
    match Nx_device.Program.load d ~binary:ptx ~name:"spin" with
    | Ok f -> f
    | Error why -> failwith why
  in
  let out = B.create d S.Int32 1 in
  let t0 = Unix.gettimeofday () in
  Nx_device.submit [ d ] ~touches:[ out ] (fun s ->
      launch (Nx_cuda_device.context c) (Nx_cuda_device.compute c)
        (B.address (Nx_device.signal_word d))
        (Nx_device.Program.handle f)
        (B.address out)
        (Nx_device.Submission.value s d));
  Nx_device.synchronize d;
  at_least ~msg:"seconds waited" (float 0.1) ~than:(float_of_int seconds)
    (Unix.gettimeofday () -. t0);
  equal ~msg:"not lost" (option string) None (Nx_device.lost d);
  let h = B.create Nx_device.host S.Int32 1 in
  B.copy ~src:out ~dst:h;
  equal ~msg:"the seconds it spun" int32 (Int32.of_int seconds)
    (B.bigarray Bigarray.int32 h).{0}

let () =
  exit
    (run "nx.cuda.device long kernels"
       [
         slow "a kernel that runs 35 s completes, and loses no device" test_long;
       ])
