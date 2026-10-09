(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx's per-call costs above the array layer, each row what a user calls: an
   operation on one-element host values, its kernel called as nx calls it and
   directly, a view, constants, an operation over two devices, reading a value's
   shape, and placing. *)

module A = Nx_array
module D = Nx_array.Dtype

(* A memory device, opened in the measuring worker: a device opened before the
   fork is lost in it. *)
let memory k =
  match Rig.memory_device (Printf.sprintf "bench-m%d" k) with
  | Ok d -> d
  | Error e -> failwith e

let host n =
  Nx.Repr.of_array Nx.host (A.of_array D.Float32 [| n |] (Array.make n 1.))

let x1 = host 1
let x16 = host 16
let x1m = host (1 lsl 20)
let b1 = Nx.less x1 x1
let a1 = Option.get (Nx.Repr.array x1)
let half = Nx.scalar D.Float32 0.5

(* K1's steps over nx.array and nx.cpu, the kernel called statically: the floor
   [dispatch/add-1]'s indirect call is judged against. *)
let add_direct a =
  let l = A.layout a in
  let dst =
    A.v D.Float32 l (Rig.Buffer.create (A.device a) (D.bytes D.Float32 1))
  in
  match Nx_cpu.apply2 (Binary Add) ~dst a a with
  | Done -> Nx.Repr.of_array Nx.host dst
  | refusal -> A.refused "add-1-direct" refusal [ A.Any dst; A.Any a ]

let chain n =
  let rec go k x = if k = 0 then x else go (k - 1) (Nx.add x half) in
  go n (Nx.zeros D.Float32 [| 1 |])

let dispatch_rows =
  Thumper.group "dispatch"
    [
      Thumper.bench "shape" (fun () -> Nx.shape (Thumper.black_box x1));
      Thumper.bench "add-1" (fun () -> Nx.add (Thumper.black_box x1) x1);
      Thumper.bench "add-1-direct" (fun () -> add_direct (Thumper.black_box a1));
      Thumper.bench "less-1" (fun () -> Nx.less (Thumper.black_box x1) x1);
      Thumper.bench "where-1" (fun () -> Nx.where (Thumper.black_box b1) x1 x1);
      Thumper.bench "cast-1" (fun () ->
          Nx.cast D.Float64 (Thumper.black_box x1));
      Thumper.bench "reshape-1" (fun () ->
          Nx.reshape [| 1; 1 |] (Thumper.black_box x1));
      Thumper.bench "zeros_like-1" (fun () ->
          Nx.zeros_like (Thumper.black_box x1));
      Thumper.bench "zeros_like-1M" (fun () ->
          Nx.zeros_like (Thumper.black_box x1m));
      Thumper.bench "zeros-1M" (fun () ->
          Nx.place Nx.Placement.host (Nx.zeros D.Float32 [| 1 lsl 20 |]));
      Thumper.bench "add-scalar-1" (fun () ->
          Nx.add (Thumper.black_box x1) (Nx.scalar D.Float32 1.));
    ]

let constant_rows =
  Thumper.group "constant"
    [
      Thumper.bench "chain-1000" (fun () ->
          Nx.Repr.array (Nx.place Nx.Placement.host (chain 1000)));
    ]

(* A value on a set minted in the worker, of a brand the row does not name. *)
type value = Value : (float, D.float32_elt, 'd) Nx.t -> value

(* One element on each of two memory devices. *)
let split_two () =
  let module Two = (val Nx.devices [ memory 0; memory 1 ]) in
  Value (Nx.place (Two.split ~axis:0) (host 2))

(* Sixteen host elements, and a placement on a memory device. *)
type borrow = Borrow : 'd Nx.Placement.t * (float, D.float32_elt, Nx.host) Nx.t -> borrow

let borrow_16 () =
  let module Mem = (val Nx.devices [ memory 0 ]) in
  Borrow (Mem.on, x16)

let placed_rows =
  Thumper.group "placed"
    [
      Thumper.bench_with_setup "add-1-two-memory-devices" ~setup:split_two
        (fun (Value two) -> Value (Nx.add two two));
    ]

let place_rows =
  Thumper.group "place"
    [
      Thumper.bench "equal-1" (fun () ->
          Nx.place Nx.Placement.host (Thumper.black_box x1));
      Thumper.bench_with_setup "borrow-memory-device-16" ~setup:borrow_16
        (fun (Borrow (on, x)) -> Value (Nx.place on x));
    ]

let () =
  exit
  @@ Thumper.run "nx" [ dispatch_rows; constant_rows; placed_rows; place_rows ]
