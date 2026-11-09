(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Contractions through every kernel library the host runs: within the
   largest bound nx_kernel.mli allows a library of the exact sum, the same
   bits into a destination identical to init, nothing written where
   declined, and misfits refused. *)

open Windtrap
open Contract_cases
module A = Nx_array
module D = Nx_array.Dtype
module Support = Nx_kernels_support

(* The reference *)

(* The sum of [o]'s products and init, with an error under 2^-100 of the sum
   of their magnitudes, and that sum. Products and the running sum are held
   in two parts. *)
let exact o =
  (* A loop: refs that a closure takes are boxed, a float allocated a
     term. *)
  let start = Option.value ~default:0. o.init in
  let hi = ref start and lo = ref 0. and mag = ref (Float.abs start) in
  for k = 0 to Array.length o.av - 1 do
    let x = o.av.(k) and y = o.bv.(k) in
    let p = x *. y in
    let s, t = two_sum !hi p in
    hi := s;
    lo := !lo +. t +. Float.fma x y (-.p);
    mag := !mag +. Float.abs p
  done;
  (!hi +. !lo, !mag)

let unit_roundoff acc =
  if part_of acc = D.Any D.Float32 then 0x1p-24 else 0x1p-53

let smallest_subnormal acc =
  if part_of acc = D.Any D.Float32 then 0x1p-149 else 0x1p-1074

(* The bound *)

(* [o]'s sum in plain float64 additions, which is not finite where the
   exact one is not. *)
let naive o =
  let s = ref (Option.value ~default:0. o.init) in
  for k = 0 to Array.length o.av - 1 do
    s := !s +. (o.av.(k) *. o.bv.(k))
  done;
  !s

(* |init| + Σ|a||b| of an output whose parts are [parts]: a complex one's
   factors are its real part's, [ar] and [ai] at terms [2p] and [2p + 1]
   against [br] and [-bi]. *)
let magnitude parts =
  let o = parts.(0) in
  if Array.length parts = 1 then begin
    let m = ref (Float.abs (Option.value ~default:0. o.init)) in
    for k = 0 to Array.length o.av - 1 do
      m := !m +. Float.abs (o.av.(k) *. o.bv.(k))
    done;
    !m
  end
  else begin
    let init p = Option.value ~default:0. parts.(p).init in
    let m = ref (Float.hypot (init 0) (init 1)) in
    for p = 0 to (Array.length o.av / 2) - 1 do
      let a = Float.hypot o.av.(2 * p) o.av.((2 * p) + 1) in
      let b = Float.hypot o.bv.(2 * p) o.bv.((2 * p) + 1) in
      m := !m +. (a *. b)
    done;
    !m
  end

(* An output [xs] of [c], one value a part, is within the largest bound
   nx_kernel.mli allows a library, γ(2K + 2) (|init| + Σ|a||b|) of the
   exact sum of its [K] products and init, a complex one in modulus, u
   being the unit roundoff of [acc]'s parts. Where results
   underflow it may err by η more for each term of each part, [η] [acc]'s
   smallest subnormal, twice the most a rounding into the subnormals errs.
   A part whose sum is not finite is matched to the bits of its naive
   sum. *)
let within c parts xs =
  let show x = Printf.sprintf "%h" x in
  let sums = Array.map naive parts in
  if not (Array.for_all Float.is_finite sums) then
    let bad = ref None in
    Array.iteri
      (fun p s ->
        if (not (Float.is_finite s)) && bits xs.(p) <> bits s then
          bad := Some (show xs.(p) ^ ", not " ^ show s))
      sums;
    !bad
  else
    let exact = Array.map exact parts in
    let off = Array.mapi (fun p (want, _) -> xs.(p) -. want) exact in
    let err = Array.fold_left Float.hypot 0. off in
    let terms = Array.length parts.(0).av in
    let k = terms / Array.length parts in
    let u = unit_roundoff c.acc in
    let n = float_of_int ((2 * k) + 2) in
    let g = n *. u /. (1. -. (n *. u)) in
    let parts_n = float_of_int (Array.length parts) in
    let eta = smallest_subnormal c.acc *. float_of_int (terms + 1) *. parts_n in
    let slack = Array.fold_left (fun s (_, m) -> s +. (0x1p-100 *. m)) 0. exact in
    let bound = (g *. magnitude parts) +. slack +. eta in
    if err <= bound then None
    else
      let all f = String.concat ", " (Array.to_list (Array.map f exact)) in
      Some
        (Printf.sprintf "%s is %h from %s, past %h"
           (String.concat ", " (Array.to_list (Array.map show xs)))
           err (all (fun (w, _) -> show w)) bound)

(* The first outputs of [y] past their bound. *)
let outside c y =
  let _, each = outputs c in
  let g = planes c.acc y in
  let bad = ref [] in
  each (fun yi k parts ->
      match within c parts (Array.map (fun part -> part.(k)) g) with
      | Some why when List.length !bad < 8 ->
          bad := Format.asprintf "%a: %s" Nx_array_gen.pp_ints yi why :: !bad
      | _ -> ());
  List.rev !bad

let law_bound b c =
  covers c;
  match run b c with
  | None -> cover "declined" true
  | Some y ->
      cover "computed" true;
      equal (list string) [] (outside c y)

(* In place *)

(* A destination identical to [init], a C-contiguous array of the result's
   dtype, gives the bits a fresh one does: a kernel reads each element of
   [init] before it writes the destination there. *)
let law_in_place (b : Support.backend) (c : case) =
  match c.init with
  | None -> cover "no init" true
  | Some i -> (
      let c = { c with init = Some (cast_to c.out i) } in
      let alias = on b (cast_to c.out i) in
      let answer =
        contract b.kernels c ~dst:alias (on b c.a) (on b c.b) (Some alias)
      in
      match (run b c, answer) with
      | Some y, A.Done ->
          cover "computed" true;
          equal (list string) []
            (differ_bits c.acc ~other:"in place" y (host alias))
      | None, A.Declined -> cover "declined" true
      | y, r ->
          failf "a fresh destination %s, in place %a"
            (if Option.is_some y then "computed" else "declined")
            Nx_array_support.pp_answer r)

(* A kernel that declines writes nothing: [dst] keeps its values. *)
let law_declined (b : Support.backend) c =
  let y, _, _ = sizes c in
  let threes =
    A.Any (A.of_array D.Float64 (Iarray.of_array y) (Array.make (total y) 3.))
  in
  let dst = on b (cast_to c.out threes) in
  let before = planes c.out (host dst) in
  let init = Option.map (on b) c.init in
  match contract b.kernels c ~dst (on b c.a) (on b c.b) init with
  | A.Declined ->
      cover "declined" true;
      equal (array (array float_exact)) before (planes c.out (host dst))
  | A.Done -> cover "computed" true
  | r -> failf "contract answered %a" Nx_array_support.pp_answer r

(* Integers *)

(* A contraction whose operands Shape.contract answers Error for, or whose
   dst's shape is not the result's, does not fit: not Done, and nothing
   written outside dst. Misfits in the pairs' extents, the operands' ranks
   and dst, a result past max_numel, and one behind a group that does not
   merge. *)
let test_misfits (b : Support.backend) () =
  let module K = (val b.kernels) in
  let arr s =
    let ones = Array.make (total s) 1. in
    let x = A.of_array D.Float32 (Iarray.of_array s) ones in
    A.expect D.Float32 (on b (A.Any x))
  in
  let refused ~msg ?(batch : _ iarray = [||]) contracting dst a b' =
    Misfit.check ~msg ~what:"a shape" b [| (D.Any D.Float32, dst) |]
      (fun g ds ->
        K.contract ~batch ~contracting ~segments:None
          ~acc:(D.Any D.Float32) ~dst:(A.expect D.Float32 ds.(0)) ~init:None
          (plain (g.g a)) (plain (g.g b')))
  in
  let mm : _ iarray = [| (1, 0) |] in
  refused ~msg:"contracted extents" mm [| 2; 3 |] (arr [| 2; 4 |])
    (arr [| 5; 3 |]);
  refused ~msg:"dst's shape" mm [| 3; 2 |] (arr [| 2; 4 |]) (arr [| 4; 3 |]);
  refused ~msg:"dst's rank" mm [| 6 |] (arr [| 2; 4 |]) (arr [| 4; 3 |]);
  refused ~msg:"a's rank" mm [| 2; 3 |] (arr [| 4 |]) (arr [| 4; 3 |]);
  let k = 1 lsl 27 in
  refused ~msg:"a result past max_numel" mm [| 1; 1 |] (arr [| k; 0 |])
    (arr [| 0; k |]);
  (* [a]'s batch axes transposed, its contracted extent 4 against [b]'s 5. *)
  let a =
    let x = arr [| 3; 2; 2; 4 |] in
    Option.get (A.move (Nx_array.Move.Permute [| 1; 0; 2; 3 |]) x)
  in
  refused ~msg:"behind an unmerged group"
    ~batch:[| (0, 0); (1, 1) |]
    [| (3, 2) |] [| 2; 3; 2; 2 |] a
    (arr [| 2; 3; 5; 2 |])

(* The suite *)

let laws (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group b.name
    [
      prop ~examples "a contraction is within its bound of the exact sum"
        computed (run (law_bound b));
      prop "a declined contraction writes nothing" any_case
        (run (law_declined b));
      prop ~examples
        "a destination identical to init gives a fresh one's bits" computed
        (run (law_in_place b));
      test
        "a call of shapes that do not fit is not Done and writes nothing \
         outside dst"
        (fun () -> b.around (test_misfits b));
    ]

let () =
  exit (Windtrap.run "nx_kernel.contract" (List.map laws Support.cpus))
