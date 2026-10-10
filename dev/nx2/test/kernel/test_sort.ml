(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Sorts through every kernel library the host runs: the values are the operand
   at the positions, in order, and keeping k gives the first k of the whole
   sort. nx.cpu's group checks the order Spec states against a stable sort built
   here, at every dtype it computes, through views, with NaN payloads, -0 and
   ties. *)

open Windtrap
open Elements
module S = Nx_kernel.Spec
module Support = Nx_kernels_support

type case = {
  axis : int;
  descending : bool;
  k : int option;
  x : A.any;
  views : string list;
}

let pp_case ppf c =
  let (A.Any x) = c.x in
  Format.fprintf ppf "sort%s along %d%s of %a %a (%s)"
    (if c.descending then " descending" else "")
    c.axis
    (match c.k with None -> "" | Some k -> Printf.sprintf " keeping %d" k)
    D.pp (A.dtype x) pp_ints (shape_of c.x)
    (String.concat ", " c.views)

(* Sub-byte dtypes sort through bytes: nx.cpu declines them. *)
let computed = List.filter (fun (D.Any dt) -> D.bits dt >= 8) D.all

(* Shapes: small ones of any rank, slices at the edges of the insertion sort's,
   and long ones that sort by radix or keep a few. *)
let shape_gen =
  let open Gen in
  let small = array ~size:(int_range 1 4) (int_range 0 6) in
  let edge =
    map (fun n -> [| 2; n |]) (of_list ~pp:Format.pp_print_int [ 63; 64; 65 ])
  in
  let long =
    let* n = int_range 100 3000 in
    let+ before = array ~size:(int_range 0 1) (int_range 1 3) in
    Array.append before [| n |]
  in
  frequency [ (5, small); (1, edge); (2, long) ]

let case_gen ?(dtypes = computed) ?(shapes = shape_gen) () =
  let open Gen in
  let* d = of_list ~pp:pp_dtype dtypes in
  let* s = shapes in
  let r = Array.length s in
  let* axis = int_range 0 (r - 1) in
  let* descending = bool in
  let* k =
    frequency
      [
        (2, constant None);
        (1, map Option.some (int_range 0 (min s.(axis) 4)));
        (1, map Option.some (int_range 0 s.(axis)));
      ]
  in
  let* v = view_of r in
  let+ seed = int in
  let rs = Random.State.make [| seed |] in
  (* Few distinct values, so that ties are common. *)
  let pool = Array.init 6 (fun _ -> element d rs) in
  let x =
    operand d s v (fun _ ->
        if Random.State.int rs 3 = 0 then element d rs
        else pool.(Random.State.int rs 6))
  in
  { axis; descending; k; x; views = view_names v }

let kept c = match c.k with None -> (shape_of c.x).(c.axis) | Some k -> k

let sort_on (b : Support.backend) c =
  let module K = (val b.kernels) in
  let (A.Any x) = c.x in
  let y = Array.copy (shape_of c.x) in
  y.(c.axis) <- kept c;
  let values = A.create Rig.host (A.dtype x) y in
  let positions = A.create Rig.host D.Int64 y in
  let s = S.sort ~axis:c.axis ~descending:c.descending ~k:c.k in
  match K.sort s ~values ~positions x with
  | A.Done -> Some (A.Any values, A.Any positions)
  | A.Declined -> None
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* Each slice's elements with their positions, stably sorted, the first [kept c]
   of them: the values' bits and the positions, in C order of the results. *)
let expected c =
  let d = dtype_of c.x and s = shape_of c.x in
  let es = elements c.x in
  let n = s.(c.axis) and keep = kept c in
  let y = Array.copy s in
  y.(c.axis) <- keep;
  let values = Array.make (total y) "" and ps = Array.make (total y) 0L in
  let cmp (a, _) (b, _) =
    let o = compare_elements d a b in
    if c.descending then -o else o
  in
  Array.iteri
    (fun k _ ->
      let i = index_of y k in
      if i.(c.axis) = 0 then begin
        let slice =
          List.init n (fun j ->
              let ij = Array.copy i in
              ij.(c.axis) <- j;
              (es.(flat s ij), j))
        in
        List.iteri
          (fun j (e, p) ->
            if j < keep then begin
              let ij = Array.copy i in
              ij.(c.axis) <- j;
              values.(flat y ij) <- e;
              ps.(flat y ij) <- Int64.of_int p
            end)
          (List.stable_sort cmp slice)
      end)
    values;
  (values, ps)

let agrees (b : Support.backend) c =
  match sort_on b c with
  | None -> failf "%s declined a sort" b.name
  | Some (v, p) ->
      let ev, ep = expected c in
      equal ~msg:"positions" (array int64) ep (positions p);
      equal ~msg:"values" (array string) ev (elements v)

let law_reference (b : Support.backend) c =
  cover "descending" c.descending;
  cover "keeps some" (c.k <> None && kept c < (shape_of c.x).(c.axis));
  cover "radix" ((shape_of c.x).(c.axis) >= 64 && c.k = None);
  agrees b c

(* Laws for every library *)

(* The values are the operand's elements at the positions along the axis. *)
let law_values_at_positions (b : Support.backend) c =
  match sort_on b c with
  | None -> ()
  | Some (v, p) ->
      let s = shape_of c.x and es = elements c.x and ps = positions p in
      let y = shape_of v in
      let at =
        Array.mapi
          (fun k q ->
            let i = index_of y k in
            i.(c.axis) <- Int64.to_int q;
            es.(flat s i))
          ps
      in
      equal (array string) at (elements v)

(* Keeping k gives the first k of the whole sort. *)
let law_prefix (b : Support.backend) c =
  match (sort_on b { c with k = None }, sort_on b c) with
  | Some (fv, fp), Some (v, p) ->
      let y = shape_of v and s = shape_of fv in
      let prefix a =
        Array.init (total y) (fun k ->
            let i = index_of y k in
            a.(flat s i))
      in
      equal ~msg:"positions" (array int64) (prefix (positions fp)) (positions p);
      equal ~msg:"values" (array string) (prefix (elements fv)) (elements v)
  | None, None -> ()
  | _ -> failf "%s declined one of the two" b.name

let test_refusals (b : Support.backend) () =
  let module K = (val b.kernels) in
  let x = A.of_array D.Float32 [| 2; 3 |] [| 3.; 1.; 2.; 0.; 5.; 4. |] in
  let p = A.create Rig.host D.Int64 [| 2; 3 |] in
  let s k = S.sort ~axis:1 ~descending:false ~k in
  equal ~msg:"values of another shape" answer A.Shape_mismatch
    (K.sort (s None)
       ~values:(A.create Rig.host D.Float32 [| 3; 2 |])
       ~positions:p x);
  equal ~msg:"keeping past the axis" answer A.Shape_mismatch
    (K.sort (s (Some 4))
       ~values:(A.create Rig.host D.Float32 [| 2; 4 |])
       ~positions:(A.create Rig.host D.Int64 [| 2; 4 |])
       x)

(* The suite *)

let cases = Gen.with_pp pp_case (case_gen ())

(* Slices the job's threads share, and one long slice. *)
let large =
  Gen.with_pp pp_case
    (case_gen
       ~shapes:(Gen.of_list [ [| 200_000 |]; [| 400; 500 |]; [| 3; 100_000 |] ])
       ())

let laws (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group b.name
    [
      prop "values are the operand at the positions" cases
        (run (law_values_at_positions b));
      prop "keeping k gives the first k of the whole sort" cases
        (run (law_prefix b));
    ]

let cpu (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group ("nx.cpu " ^ b.name)
    [
      prop "sorts each slice stably in the order" cases (run (law_reference b));
      prop ~count:8 "large sorts on the job's threads" large (run (agrees b));
      test "refuses results of another shape" (fun () ->
          b.around (test_refusals b));
    ]

let () =
  exit
    (Windtrap.run "nx_kernel.sort"
       (List.map laws Support.backends @ List.map cpu Support.cpus))
