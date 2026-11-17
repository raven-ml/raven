(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module T = Nx.Ptree

let owned x = Rune.jit T.(tensor @-> returns tensor) Nx.copy x

(* A loop's final carry that takes the memory of a donated argument it does
   not read is computed there, or, of four bits, which the loop holds
   unpacked, stored there packed: either way the undonated call's values,
   and a float carry holds no storage beside the argument. A read of the
   argument's memory where the host computes on the device, unlike a
   {!Nx.Repr.array} of it, leaves it unshared, so that it can still be
   donated. *)
let carry (module S : Nx.Devices) =
  let s = T.(pair tensor tensor @-> returns tensor) in
  let scan step init xs =
    fst (Rune.scan T.tensor T.tensor T.unit step ~init xs)
  in
  let place x = Nx.place S.on x in
  let xs =
    place (Nx.Rng.normal ~key:(Nx.Rng.key 4) Nx.float32 [| 4; 64; 64 |])
  in
  let rig =
    let a = Option.get (Nx.Repr.array ~by:"test" xs) in
    Rig.Buffer.device (Nx_array.buffer a)
  in
  let memory y = Nx.Repr.read ~by:"test" y Nx_array.buffer in
  (* [g] over [w] donated, and whether its result lies in [w]'s memory where
     the host computes on the device. *)
  let in_place g w =
    let before = if Rig.runs_on_host rig then Some (memory w) else None in
    let y = g (Nx.donate w) in
    let within b = Rig.Buffer.overlaps b (memory y) in
    (Option.fold ~none:true ~some:within before, y)
  in
  let host x = Nx.to_array (Nx.place Nx.Host.on x) in
  let f (xs, _) =
    let step c x = (Nx.add (Nx.mul_s c 0.5) x, ()) in
    scan step (Nx.zeros_like (Nx.get [ 0 ] xs)) xs
  in
  let w () =
    owned (place (Nx.Rng.normal ~key:(Nx.Rng.key 5) Nx.float32 [| 64; 64 |]))
  in
  let on_rig (r, b) = if Rig.equal r rig then b else 0 in
  let footprint x =
    List.fold_left (fun n p -> n + on_rig p) 0 (Rune.footprint s f x)
  in
  let bytes = Rig.allocation rig (4 * 64 * 64) in
  at_most ~msg:"the float carry's footprint" Testable.int
    ~than:(footprint (xs, w ()) - bytes)
    (footprint (xs, Nx.donate (w ())));
  let g = Rune.jit s f in
  let expected = host (g (xs, w ())) in
  let took, y = in_place (fun w -> g (xs, w)) (w ()) in
  equal ~msg:"the float carry" Testable.(array float_exact) expected (host y);
  equal ~msg:"the float carry: in place" Testable.bool true took;
  let i4 = Nx.int4 in
  let ys = Nx.cast i4 (Nx.mul_s (Nx.ones Nx.int32 [| 4; 64; 64 |]) 3l) in
  let ys = place ys in
  let f (ys, _) =
    scan (fun c y -> (Nx.maximum c y, ())) (Nx.zeros_like (Nx.get [ 0 ] ys)) ys
  in
  let w () = owned (place (Nx.cast i4 (Nx.ones Nx.int32 [| 64; 64 |]))) in
  let g = Rune.jit T.(pair tensor tensor @-> returns tensor) f in
  let expected = host (g (ys, w ())) in
  let took, y = in_place (fun w -> g (ys, w)) (w ()) in
  equal ~msg:"the four-bit carry" Testable.(array int) expected (host y);
  equal ~msg:"the four-bit carry: in place" Testable.bool true took
