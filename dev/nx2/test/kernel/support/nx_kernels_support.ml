(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module P = Nx_kernel.Prog

type op0 = Fill of string | Iota of int

type kind =
  | K0 of op0
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

let backends = cpus

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

(* One-node maps *)

let map (module K : Nx_kernel.S) ~ins n ~dst loads =
  K.map (P.of_node ~ins n) ~dsts:[| Nx_array.Any dst |] loads

let map0 k op ~dst =
  let dt = Nx_array.Dtype.Any (Nx_array.dtype dst) in
  let n = match op with Fill b -> P.Const (dt, b) | Iota i -> P.Coord i in
  map k ~ins:[||] n ~dst [||]

let map1 k op ~dst x =
  let y = Nx_array.Dtype.Any (Nx_array.dtype dst) in
  map k
    ~ins:[| Nx_array.Dtype.Any (Nx_array.dtype x) |]
    (Op1 (op, y, 0)) ~dst [| Nx_kernel.Plain x |]

let map2 k op ~dst x y =
  let d = Nx_array.Dtype.Any (Nx_array.dtype x) in
  map k ~ins:[| d; d |] (Op2 (op, 0, 1)) ~dst [| Nx_kernel.Plain x; Plain y |]

let map3 k op ~dst c x y =
  let d = Nx_array.Dtype.Any (Nx_array.dtype x) in
  map k
    ~ins:[| Nx_array.Dtype.Any (Nx_array.dtype c); d; d |]
    (Op3 (op, 0, 1, 2)) ~dst [| Nx_kernel.Plain c; Plain x; Plain y |]

(* Address space *)

external cap_memory : int -> bool = "nx_kernels_support_cap_memory"
