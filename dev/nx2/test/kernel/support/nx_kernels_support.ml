(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type kind =
  | K0 of Nx_kernel.Prog.op0
  | K1 of Nx_kernel.Prog.op1
  | K2 of Nx_kernel.Prog.op2
  | K3 of Nx_kernel.Prog.op3

type backend = {
  name : string;
  kernels : (module Nx_kernel.S);
  device : Rig.t;
  computes : kind -> Nx_array.Dtype.any -> bool;
  around : 'a. (unit -> 'a) -> 'a;
}

(* nx.cpu's target tables: the host's, and the one its kernels run. *)
external targets : unit -> string list = "nx_kernels_support_targets"
external current : unit -> string = "nx_kernels_support_current"
external use : string -> unit = "nx_kernels_support_use"

let with_target t f =
  let before = current () in
  use t;
  Fun.protect ~finally:(fun () -> use before) f

(* nx_cpu.mli's promise: [Fill] at every dtype of a byte or more, [Iota] at
   the base dtypes, the other kinds at every dtype. *)
let base (Nx_array.Dtype.Any dt) =
  let module D = Nx_array.Dtype in
  match dt with
  | D.Float32 | D.Float64 | D.Int8 | D.Uint8 | D.Int16 | D.Uint16 | D.Int32
  | D.Uint32 | D.Int64 | D.Uint64 | D.Bool ->
      true
  | _ -> false

let cpu_computes k (Nx_array.Dtype.Any dt as d) =
  let module D = Nx_array.Dtype in
  match k with
  | K0 (Fill _) -> D.bits dt >= 8
  | K0 (Iota _) -> base d
  | K1 _ | K2 _ | K3 _ -> true

let cpu target =
  {
    name = "cpu/" ^ target;
    kernels = (module Nx_cpu);
    device = Rig.host;
    computes = cpu_computes;
    around = (fun f -> with_target target f);
  }

let cpus = List.map cpu (targets ())

(* nx.cuda on CUDA's GPU 0, where there is one; the suite then holds the GPU
   lock. nx_cuda.mli states it computes no kind. *)
let cuda () =
  if Rig_cuda.count () = 0 then []
  else begin
    Rig_gpu_lock.hold ();
    let c = Result.get_ok (Rig_cuda.open_ 0) in
    let d =
      Result.get_ok
        (Rig.open_ (module Rig_cuda) ~name:"CUDA:0" (fun () -> Ok c))
    in
    if not (Nx_cuda.computes_on d) then []
    else
      [
        {
          name = "cuda";
          kernels = (module Nx_cuda);
          device = d;
          computes = (fun _ _ -> false);
          around = (fun f -> f ());
        };
      ]
  end

let backends = cpus @ cuda ()

(* nx.cpu's reductions and scans on one thread. *)
external serial_reduce :
  Nx_kernel.Spec.reduce Nx_kernel.Spec.t ->
  dsts:Nx_array.any array ->
  Nx_array.any array ->
  Nx_array.answer = "nx_kernels_support_serial_reduce"

external serial_scan :
  Nx_kernel.Spec.scan Nx_kernel.Spec.t ->
  dsts:Nx_array.any array ->
  Nx_array.any array ->
  Nx_array.answer = "nx_kernels_support_serial_scan"
