(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Contractions through every kernel library the host runs: within the bound
   Nx_kernel.S.contract states of the exact sum, the same bits into a
   destination identical to init, and nothing written where declined. *)

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

(* The laws of S *)

(* An output [x] of [c] is within γ(K + 1, 2u) (|init| + Σ|a·b|) of the exact
   sum of [o], u being [acc]'s unit roundoff, and (K + 1) η where results
   underflow, η [acc]'s smallest subnormal, twice the most a rounding into
   the subnormals errs; a NaN or infinite sum is matched. *)
let within c o x =
  let naive = ref (Option.value ~default:0. o.init) in
  for k = 0 to Array.length o.av - 1 do
    naive := !naive +. (o.av.(k) *. o.bv.(k))
  done;
  let naive = !naive in
  let show = Printf.sprintf "%h" in
  if not (Float.is_finite naive) then
    if bits x = bits naive then None else Some (show x ^ ", not " ^ show naive)
  else
    let want, mag = exact o in
    let u = unit_roundoff c.acc in
    let n = float_of_int (Array.length o.av + 1) in
    let g = n *. 2. *. u /. (1. -. (n *. 2. *. u)) in
    let eta = smallest_subnormal c.acc in
    let bound = (g *. mag) +. (0x1p-100 *. mag) +. (n *. eta) in
    if Float.abs (x -. want) <= bound then None
    else
      Some
        (Printf.sprintf "%s is %h from %s, past %h" (show x) (x -. want)
           (show want) bound)

let law_bound b c =
  covers c;
  match run b c with
  | None -> cover "declined" true
  | Some y ->
      cover "computed" true;
      equal (list string) [] (wrong c y (within c))

(* An init given as dst itself, an operand identical to the written one,
   gives the bits of the same contraction into a fresh dst. *)
let law_init_as_dst (b : Support.backend) (c : case) =
  match c.init with
  | None -> cover "no init" true
  | Some i ->
      let c' = { c with init = Some (cast_to c.out i) } in
      let donated = on b (cast_to c.out i) in
      let answer =
        contract b.kernels c ~dst:donated (on b c.a) (on b c.b) (Some donated)
      in
      begin match (run b c', answer) with
      | Some y, A.Done ->
          let per = per_batch c in
          cover "fewer than 64 outputs" (per > 0 && per < 64);
          cover "64 outputs or more" (per >= 64);
          equal (list string) []
            (differ_bits c.acc ~other:"in place" y (host donated))
      | None, A.Declined -> cover "declined" true
      | _, r -> failf "in place, contract answered %a" Nx_array_support.pp_answer r
      end

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

(* A contraction answers Shape_mismatch where Shape.contract answers
   Error for its operands, or where dst's shape is not the result's, and
   writes nothing: misfits in the pairs' extents, the operands' ranks and
   dst, a result past max_numel, and one behind a group that does not
   merge. *)
let test_misfits (b : Support.backend) () =
  let module K = (val b.kernels) in
  let arr s =
    let ones = Array.make (total s) 1. in
    let x = A.of_array D.Float32 (Iarray.of_array s) ones in
    A.expect D.Float32 (on b (A.Any x))
  in
  let answer = Testable.make ~pp:Nx_array_support.pp_answer ~equal:( = ) in
  let refused ~msg ?(batch : _ iarray = [||]) contracting dst a b' =
    let before = A.to_array (to64 (host (A.Any dst))) in
    equal ~msg answer A.Shape_mismatch
      (K.contract ~batch ~contracting ~segments:None
         ~acc:(D.Any D.Float32) ~dst ~init:None (plain a) (plain b'));
    equal ~msg:(msg ^ ": untouched") (array float_exact) before
      (A.to_array (to64 (host (A.Any dst))))
  in
  let mm : _ iarray = [| (1, 0) |] in
  refused ~msg:"contracted extents" mm (arr [| 2; 3 |]) (arr [| 2; 4 |])
    (arr [| 5; 3 |]);
  refused ~msg:"dst's shape" mm (arr [| 3; 2 |]) (arr [| 2; 4 |])
    (arr [| 4; 3 |]);
  refused ~msg:"dst's rank" mm (arr [| 6 |]) (arr [| 2; 4 |]) (arr [| 4; 3 |]);
  refused ~msg:"a's rank" mm (arr [| 2; 3 |]) (arr [| 4 |]) (arr [| 4; 3 |]);
  let k = 1 lsl 27 in
  refused ~msg:"a result past max_numel" mm (arr [| 1; 1 |]) (arr [| k; 0 |])
    (arr [| 0; k |]);
  (* [a]'s batch axes transposed, its contracted extent 4 against [b]'s 5. *)
  let a =
    let x = arr [| 3; 2; 2; 4 |] in
    Option.get (A.move (Nx_array.Move.Permute [| 1; 0; 2; 3 |]) x)
  in
  refused ~msg:"behind an unmerged group"
    ~batch:[| (0, 0); (1, 1) |]
    [| (3, 2) |]
    (arr [| 2; 3; 2; 2 |])
    a
    (arr [| 2; 3; 5; 2 |])

(* The suite *)

let laws (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group b.name
    [
      prop ~examples "a contraction is within its bound of the exact sum"
        computed (run (law_bound b));
      prop ~examples "an init given as dst gives the same bits" computed
        (run (law_init_as_dst b));
      prop "a declined contraction writes nothing" any_case
        (run (law_declined b));
      prop ~examples
        "a destination identical to init gives a fresh one's bits" computed (run (law_in_place b));
      test "refuses operands and destinations whose shapes do not fit"
        (fun () -> b.around (test_misfits b));
    ]

let () =
  exit (Windtrap.run "nx_kernel.contract" (List.map laws Support.cpus))
