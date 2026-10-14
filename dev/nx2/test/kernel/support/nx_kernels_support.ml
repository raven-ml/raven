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

(* nx_cpu.mli's promise: every kind at every dtype of its domain. *)
let cpu_computes _ _ = true

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

(* Guard pages *)

type side = Before | After

external guarded_bytes :
  bool ->
  int ->
  int ->
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
  * nativeint
  * int = "nx_kernels_support_guarded"

external unguard : nativeint -> int -> unit = "nx_kernels_support_unguard"

let guarded side dt l =
  let lo, hi = Nx_array.Layout.span l in
  let first = lo * Nx_array.Dtype.bits dt / 8 in
  let bytes = Nx_array.Dtype.bytes dt hi in
  let ba, base, total = guarded_bytes (side = After) first bytes in
  Bigarray.Array1.fill (Bigarray.Array1.sub ba first (bytes - first)) '\000';
  let a = Nx_array.v dt l (Rig.Buffer.of_bigarray ba) in
  (a, fun () -> unguard base total)

let guard side x =
  let dt = Nx_array.dtype x and l = Nx_array.layout x in
  let g, free = guarded side dt l in
  let lo, hi = Nx_array.Layout.span l in
  let first = lo * Nx_array.Dtype.bits dt / 8 in
  let bytes a =
    Nx_array.v Nx_array.Dtype.Uint8
      (Nx_array.Layout.v ~offset:first ~strides:[| 1 |]
         [| Nx_array.Dtype.bytes dt hi - first |])
      (Nx_array.buffer a)
  in
  Nx_array.blit ~src:(bytes x) ~dst:(bytes g);
  (g, free)
