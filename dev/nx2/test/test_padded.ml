(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Padded loads and Fold through Nx.Prim: the values Nx_kernel.Spec.load and
   Nx_kernel.Spec.fold state for small operands, their rules before any kernel,
   a split placement, and a library that declines. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module P = Nx_kernel.Prog
module S = Nx_kernel.Spec
module C = Nx_support.Counting

let m = Nx_support.memory

module Count = (val Nx.devices ~kernels:(module C) [ m 0 ])
module Count2 = (val Nx.devices ~kernels:(module C) [ m 0; m 1 ])

(* nx.cpu, declining every fold and every loop with a padded load. *)
module Unpadded = struct
  include Nx_cpu

  let name = "nx.unpadded"

  let padded s =
    Array.exists (function S.Padded _ -> true | S.Plain -> false) (S.loads s)

  let reduce s ~dsts ops =
    if padded s then A.Declined else Nx_cpu.reduce s ~dsts ops

  let fold _ ~dst:_ _ = A.Declined
end

module Dec = (val Nx.devices ~kernels:(module Unpadded) [ m 2 ])

let read x = A.to_array (Option.get (Nx.Repr.array (Nx.place Nx.Host.on x)))
let host a = Nx.Repr.of_array Nx.Host.v a
let identity dt = P.v ~ins:[| D.Any dt |] [| P.In 0 |] ~outs:[| 0 |]

let pad ?(lo = [| 0 |]) ?(hi = [| 0 |]) ?(interior = [| 0 |]) windows =
  { S.lo; hi; interior; windows }

let window size step = { A.Move.axis = 0; size; step; dilation = 1 }

(* Sum over the last axis of [x] loaded through [pad]. *)
let sum_of x ~fill pad =
  let layout =
    match
      S.shapes
        (S.map (identity (Nx.dtype x)) ~loads:[| S.Padded { fill; pad } |])
        [| Nx.shape x |]
    with
    | Ok [| s |] -> L.contiguous s
    | _ -> fail "no loaded shape"
  in
  let r = L.rank layout in
  let y, () =
    Nx.Prim.eval ~by:"t"
      (Reduce
         {
           layout;
           axes = [| r - 1 |];
           prog = identity (Nx.dtype x);
           reductions = [ Monoid (Sum, 0, Nx.dtype x) ];
           loads = [| Padded { x; fill = D.zero (Nx.dtype x); pad } |];
         })
  in
  y

let f32 xs = A.of_array D.Float32 [| Array.length xs |] xs

let test_sum_pool () =
  let x = Nx.place Count.on (host (f32 [| 1.; 2.; 3.; 4. |])) in
  let p = pad ~lo:[| 1 |] ~hi:[| 1 |] [| window 3 1 |] in
  equal ~msg:"a sum over windows of three, padded with zeros"
    (array float_exact) [| 3.; 6.; 9.; 7. |]
    (read (sum_of x ~fill:(P.bits D.Float32 0.) p));
  let p = pad ~lo:[| -1 |] ~interior:[| 1 |] [| window 2 2 |] in
  (* Interior padding [1; 0; 2; 0; 3; 0; 4], cropped by one: [0; 2; 0; 3; 0; 4],
     in windows of two, stepping by two. *)
  equal ~msg:"interior padding, a negative lo crops" (array float_exact)
    [| 2.; 3.; 4. |]
    (read (sum_of x ~fill:(P.bits D.Float32 0.) p))

let test_fold () =
  let p = pad ~lo:[| 1 |] ~hi:[| 1 |] [| window 3 1 |] in
  let ones =
    Nx.place Count.on
      (host (A.of_array D.Float32 [| 4; 3 |] (Array.make 12 1.)))
  in
  let y = Nx.Prim.eval ~by:"t" (Fold { shape = [| 4 |]; pad = p; x = ones }) in
  equal ~msg:"each element, the number of taps that read it" (array float_exact)
    [| 2.; 3.; 3.; 2. |] (read y)

let test_rules () =
  let x = Nx.place Count.on (host (f32 [| 1.; 2.; 3. |])) in
  let raises f = raises_match (Exn.invalid_arg ~substring:"t: ") f in
  C.reset ();
  raises (fun () ->
      Nx.Prim.eval ~by:"t"
        (Map
           {
             layout = L.contiguous [| 3 |];
             prog = identity D.Float32;
             outs = [ Nx.float32 ];
             loads = [| Padded { x; fill = 0.; pad = pad ~lo:[| 1 |] [||] } |];
           }));
  raises (fun () ->
      Nx.Prim.eval ~by:"t"
        (Fold { shape = [| 3 |]; pad = pad ~lo:[| 1 |] [||]; x }));
  raises (fun () ->
      Nx.Prim.eval ~by:"t"
        (Fold
           {
             shape = [| 3 |];
             pad = pad [||];
             x =
               Nx.place Count.on
                 (host (A.of_array D.Bool [| 3 |] [| true; false; true |]));
           }));
  equal ~msg:"kernel calls" int 0 (C.calls ())

let test_split () =
  let a = f32 [| 1.; 2.; 3.; 4. |] in
  let p = pad ~lo:[| 1 |] ~hi:[| 1 |] [| window 3 1 |] in
  let fill = P.bits D.Float32 0. in
  let one = sum_of (Nx.place Count.on (host a)) ~fill p in
  let two = sum_of (Nx.place (Count2.split ~axis:0) (host a)) ~fill p in
  equal ~msg:"as on one device" (array float_exact) (read one) (read two)

let test_declined () =
  let x = Nx.place Dec.on (host (f32 [| 1.; 2.; 3. |])) in
  let p = pad ~lo:[| 1 |] ~hi:[| 1 |] [| window 3 1 |] in
  let check what f =
    match f () with
    | _ -> fail (what ^ ": a decline computed")
    | exception Invalid_argument e ->
        List.iter
          (fun sub -> contains ~msg:(what ^ ": " ^ sub) ~sub e)
          [ "t: "; "nx.unpadded"; "m2"; "not available yet"; "Nx.place" ]
  in
  check "a padded loop" (fun () -> sum_of x ~fill:(P.bits D.Float32 0.) p);
  let ones =
    Nx.place Dec.on (host (A.of_array D.Float32 [| 3; 3 |] (Array.make 9 1.)))
  in
  check "a fold" (fun () ->
      Nx.Prim.eval ~by:"t" (Fold { shape = [| 3 |]; pad = p; x = ones }))

let () =
  exit
    (run "nx padded"
       [
         group "loads and folds"
           [
             test "a padded, windowed load" test_sum_pool;
             test "a fold counts the taps" test_fold;
             test "the rules raise before any kernel" test_rules;
             test "a split value loads as on one device" test_split;
             test "a decline raises naming the move" test_declined;
           ];
       ])
