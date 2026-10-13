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

(* nx.cuda on GPU 0 of the machine's NVIDIA GPUs, opened by each rig driver
   that reaches it: CUDA's library, and NVIDIA's kernel driver through rig.nv.
   nx_cuda.mli states it computes no kind. *)
let nvidia ~name open_ =
  let d = Result.get_ok (open_ ()) in
  if not (Nx_cuda.computes_on d) then []
  else
    [
      {
        name;
        kernels = (module Nx_cuda);
        device = d;
        computes = (fun _ _ -> false);
        around = (fun f -> f ());
      };
    ]

let gpus () =
  let cuda = Rig_cuda.count () > 0 and nv = Rig_nv_nvidia.count () > 0 in
  let cuda_device () = Rig.open_ { (Rig_cuda.gpu 0) with name = "CUDA:0" } in
  let nv_device () = Rig.open_ { (Rig_nv_nvidia.gpu 0) with name = "NV:0" } in
  (if cuda then nvidia ~name:"cuda" cuda_device else [])
  @ if nv then nvidia ~name:"cuda/nv" nv_device else []

let backends = cpus @ gpus ()

(* Jobs begun from jobs' bodies. *)
external nested : int -> int -> int = "nx_kernels_support_nested"

(* nx.cpu with its reductions, scans and matrices on one thread: the serial
   entries of cpu.h for the matrices. *)
module Serial = struct
  include Nx_cpu

  let reduce = Nx_cpu.reduce_on ~threads:1
  let scan = Nx_cpu.scan_on ~threads:1

  external cholesky :
    Nx_kernel.triangle ->
    dst:('v, 's) Nx_array.t ->
    ('v, 's) Nx_array.t ->
    Nx_array.answer = "nx_kernels_support_serial_cholesky"

  external lu :
    lu:('v, 's) Nx_array.t ->
    pivots:(int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
    perm:(int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
    ('v, 's) Nx_array.t ->
    Nx_array.answer = "nx_kernels_support_serial_lu"

  external qr :
    Nx_kernel.factors ->
    q:('v, 's) Nx_array.t ->
    r:('v, 's) Nx_array.t ->
    ('v, 's) Nx_array.t ->
    Nx_array.answer = "nx_kernels_support_serial_qr"

  external solve_triangular :
    Nx_kernel.triangle ->
    adjoint:bool ->
    unit_diagonal:bool ->
    dst:('v, 's) Nx_array.t ->
    ('v, 's) Nx_array.t ->
    ('v, 's) Nx_array.t ->
    Nx_array.answer
    = "nx_kernels_support_serial_solve_triangular_byte"
      "nx_kernels_support_serial_solve_triangular"
end

let serial = (module Serial : Nx_kernel.S)
