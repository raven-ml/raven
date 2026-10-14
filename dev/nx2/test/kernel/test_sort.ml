(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Sorts through every kernel library the host runs: the values are the operand
   at the positions, in order, and keeping k gives the first k of the whole
   sort. nx.cpu's group checks the order Nx_kernel.S.sort states against a
   stable sort built here, at every dtype, through views, with NaN
   payloads, -0 and ties. *)

open Windtrap
open Elements
module Support = Nx_kernels_support

(* How an operand's elements are drawn: from a few values, so that ties are
   common; within a narrow span of keys, a base element's high bytes over its
   given number of random low ones; each at random; or from one value three
   times in four, else at random, so that a long slice's top digit leaves one
   bucket of most elements. *)
type draw = Tied | Narrow of int | Spread | Skewed

type case = {
  axis : int;
  descending : bool;
  k : int option;
  x : A.any;
  draw : draw;
  views : string list;
}

let pp_case ppf c =
  let (A.Any x) = c.x in
  Format.fprintf ppf "sort%s along %d%s of %a %a, %s (%s)"
    (if c.descending then " descending" else "")
    c.axis
    (match c.k with None -> "" | Some k -> Printf.sprintf " keeping %d" k)
    D.pp (A.dtype x) pp_ints (shape_of c.x)
    (match c.draw with
    | Tied -> "tied"
    | Narrow low -> Printf.sprintf "narrow over %d bytes" low
    | Spread -> "spread"
    | Skewed -> "skewed")
    (String.concat ", " c.views)

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

(* An operand of [d] drawn from [seed] as [draw] says. A narrow element keeps
   the base's high bytes, so that integers lie within a few low bytes of each
   other and floats within a few low bytes of their significands. *)
let drawn draw d s v seed =
  let rs = Random.State.make [| seed |] in
  let pool = Array.init 6 (fun _ -> element d rs) in
  let base = element d rs in
  operand d s v (fun _ ->
      match draw with
      | Tied ->
          if Random.State.int rs 3 = 0 then element d rs
          else pool.(Random.State.int rs 6)
      | Skewed -> if Random.State.int rs 4 = 0 then element d rs else base
      | Narrow low ->
          String.mapi
            (fun j c ->
              if j < low then Char.chr (Random.State.int rs 256) else c)
            base
      | Spread ->
          String.init (String.length base) (fun _ ->
              Char.chr (Random.State.int rs 256)))

let case_gen ?(dtypes = D.all) ?(shapes = shape_gen) () =
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
  let (D.Any dt) = d in
  let* draw =
    frequency
      [
        (4, constant Tied);
        (1, map (fun low -> Narrow low) (int_range 1 (max 1 (D.bits dt / 8))));
        (1, constant Spread);
      ]
  in
  let+ seed = int in
  { axis; descending; k; x = drawn draw d s v seed; draw; views = view_names v }

(* A constructed case along the last axis, drawn from seed 1, so that every seed
   reaches its regime. *)
let example ?k ?(descending = false) ?(draw = Tied) ?view d s =
  let r = Array.length s in
  let v = Option.value view ~default:(plain r) in
  {
    axis = r - 1;
    descending;
    k;
    x = drawn draw d s v 1;
    draw;
    views = view_names v;
  }

let kept c = match c.k with None -> (shape_of c.x).(c.axis) | Some k -> k

let sort_on (b : Support.backend) c =
  let module K = (val b.kernels) in
  let (A.Any x) = c.x in
  let y = Array.copy (shape_of c.x) in
  y.(c.axis) <- kept c;
  let values = A.create Rig.host (A.dtype x) y in
  let positions = A.create Rig.host D.Int64 y in
  match K.sort ~axis:c.axis ~descending:c.descending ~values ~positions x with
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
  let n = (shape_of c.x).(c.axis) in
  cover "descending" c.descending;
  cover "keeps some" (c.k <> None && kept c < n);
  cover "radix" (n >= 64 && c.k = None);
  cover "complex128's two keys" (dtype_of c.x = D.Any D.Complex128 && n >= 2);
  cover "sub-byte"
    (let (D.Any d) = dtype_of c.x in
     D.bits d < 8 && n >= 2);
  cover "narrow 8-byte keys"
    (let (D.Any d) = dtype_of c.x in
     (match c.draw with Narrow _ -> true | _ -> false)
     && n >= 64
     && D.bits d = 64);
  agrees b c

(* One slice sorts on the job's threads where it is long and keeps more than a
   few. *)
let law_large (b : Support.backend) c =
  let long =
    Array.length (shape_of c.x) = 1
    && kept c > 64
    && dtype_of c.x <> D.Any D.Complex128
  in
  let wide = long && (let (D.Any d) = dtype_of c.x in D.bits d = 64) in
  cover "one long slice" long;
  cover "long wide keys spread" (wide && c.draw = Spread);
  cover "long wide keys skewed" (wide && c.draw = Skewed);
  cover "long wide keys, keeping all" (wide && c.k = None);
  cover "sub-byte slices on the job's threads"
    (let (D.Any d) = dtype_of c.x in
     D.bits d < 8 && total (shape_of c.x) >= 100_000);
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

(* Refusals *)

(* A sort's shapes, [x], [values] and [positions], along [along], each near
   a fit: [values] keeps up to one past [x]'s extent. *)
type misfit = { along : int; shapes : int array array }

let misfits =
  Gen.with_pp
    (fun ppf c ->
      Format.fprintf ppf "sort along %d of %a into %a and %a" c.along pp_ints
        c.shapes.(0) pp_ints c.shapes.(1) pp_ints c.shapes.(2))
    (let open Gen in
     let* x = Nx_array_gen.shape in
     let* axis = axis_near x in
     let d = if axis >= 0 && axis < Array.length x then x.(axis) else 0 in
     let* k = int_range 0 (d + 1) in
     let* values = near (along x axis k) in
     let+ positions = near values in
     { along = axis; shapes = [| x; values; positions |] })

(* The kernel keeps [values]' extent along the axis: Nx_kernel.Shape.sort's
   [k]. *)
let law_refuses (b : Support.backend) c =
  let module K = (val b.kernels) in
  let x = c.shapes.(0) and values = c.shapes.(1) and positions = c.shapes.(2) in
  let k =
    if c.along >= 0 && c.along < Array.length values then
      Some values.(c.along)
    else None
  in
  let fit =
    Nx_kernel.Shape.sort ~axis:c.along ~k x = Ok values && positions = values
  in
  cover "fits" fit;
  cover "misfits" (not fit);
  equal answer
    (if fit then A.Done else A.Shape_mismatch)
    (K.sort ~axis:c.along ~descending:false
       ~values:(A.create Rig.host D.Float32 values)
       ~positions:(A.create Rig.host D.Int64 positions)
       (A.of_array D.Float32 x (Array.make (total x) 0.)))

let test_refusals (b : Support.backend) () =
  let module K = (val b.kernels) in
  let x = A.of_array D.Float32 [| 2; 3 |] [| 3.; 1.; 2.; 0.; 5.; 4. |] in
  let p = A.create Rig.host D.Int64 [| 2; 3 |] in
  equal ~msg:"values of another shape" answer A.Shape_mismatch
    (K.sort ~axis:1 ~descending:false
       ~values:(A.create Rig.host D.Float32 [| 3; 2 |])
       ~positions:p x);
  equal ~msg:"keeping past the axis" answer A.Shape_mismatch
    (K.sort ~axis:1 ~descending:false
       ~values:(A.create Rig.host D.Float32 [| 2; 4 |])
       ~positions:(A.create Rig.host D.Int64 [| 2; 4 |])
       x)

(* Guard pages *)

(* Slices of one element to a few vectors past the widest, at the edges of
   the insertion sort's, and long enough to sort by radix. *)
let guard_shapes =
  let open Gen in
  let* n =
    frequency
      [
        (6, int_range 1 72);
        (1, of_list ~pp:Format.pp_print_int [ 63; 64; 65; 127; 128; 129 ]);
        (2, int_range 100 3000);
      ]
  in
  let+ outer = array ~size:(int_range 0 2) (int_range 1 4) in
  Array.append outer [| n |]

(* The same values and positions with the operand, through its view, and
   both destinations against an inaccessible page. *)
let law_guards (b : Support.backend) (c, side) =
  let module K = (val b.kernels) in
  let after = side = Support.After in
  let s = shape_of c.x in
  cover "along the innermost axis" (c.axis = Array.length s - 1);
  cover "along an outer axis" (c.axis < Array.length s - 1);
  cover "by radix" (s.(c.axis) >= 100);
  cover "keeping some" (kept c < s.(c.axis));
  cover "sub-byte" (let (D.Any dt) = dtype_of c.x in D.bits dt < 8);
  cover "a stepped operand at the page before"
    ((not after) && List.mem "stepped" c.views);
  cover "a reversed operand at the page after"
    (after && List.mem "reversed" c.views);
  let y = Array.copy s in
  y.(c.axis) <- kept c;
  let call xs ds =
    let (A.Any x) = xs.(0) in
    let values = A.expect (A.dtype x) ds.(0) in
    let positions = A.expect D.Int64 ds.(1) in
    K.sort ~axis:c.axis ~descending:c.descending ~values ~positions x
  in
  equal answer A.Done
    (same_guarded side ~call [| c.x |]
       [| (dtype_of c.x, y); (D.Any D.Int64, y) |])

(* The suite *)

let cases = Gen.with_pp pp_case (case_gen ())

let examples =
  [
    example (D.Any D.Complex128) [| 2; 70 |];
    example (D.Any D.Complex128) [| 9 |] ~descending:true;
    example (D.Any D.Int64) [| 2; 300 |] ~draw:(Narrow 3);
    example (D.Any D.Complex64) [| 300 |] ~draw:(Narrow 5) ~descending:true;
    example (D.Any D.Int4) [| 3; 70 |];
    example (D.Any D.Float4_e2m1fn) [| 9 |] ~descending:true;
  ]

(* Slices the job's threads share, and one long slice. *)
let large =
  Gen.with_pp pp_case
    (case_gen
       ~shapes:(Gen.of_list [ [| 200_000 |]; [| 400; 500 |]; [| 3; 100_000 |] ])
       ())

(* One long slice of 4-byte keys packed with positions; of 8-byte keys beside
   them, keeping some or all, spread, so that they take their top digit first,
   or skewed, so that they fall back to LSD passes; and of 8-byte keys of a
   narrow span, packed. *)
let large_examples =
  [
    example (D.Any D.Float32) [| 200_000 |];
    example (D.Any D.Int64) [| 200_000 |] ~descending:true ~k:150_000;
    example (D.Any D.Int64) [| 200_000 |] ~draw:Spread;
    example (D.Any D.Float64) [| 200_000 |] ~draw:Spread ~descending:true;
    example (D.Any D.Uint64) [| 200_000 |] ~draw:Skewed;
    example (D.Any D.Float64) [| 200_000 |] ~draw:Skewed ~k:150_000;
    example (D.Any D.Int64) [| 200_000 |] ~draw:(Narrow 3);
    (* Sub-byte slices on the job's threads, of an odd length, so that
       neighbouring slices' values share a byte, read through views. *)
    example (D.Any D.Int4) [| 400; 501 |]
      ~view:{ (plain 2) with stepped = true; reversed = Some 1 };
    example (D.Any D.Uint4) [| 400; 501 |] ~descending:true ~k:301
      ~view:{ (plain 2) with perm = [| 1; 0 |] };
    example (D.Any D.Bit) [| 400; 501 |]
      ~view:{ (plain 2) with reversed = Some 0 };
  ]

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
      prop ~examples "sorts each slice stably in the order" cases
        (run (law_reference b));
      prop ~count:8 ~examples:large_examples "large sorts on the job's threads"
        large
        (run (law_large b));
      prop "refuses what Nx_kernel.Shape refuses and other shapes" misfits
        (run (law_refuses b));
      test "refuses results of another shape" (fun () ->
          b.around (test_refusals b));
      prop ~count:300
        "reads and writes no byte past the operand and destinations, \
         whatever pages lie beside them"
        (Gen.pair (case_gen ~shapes:guard_shapes ()) sides)
        (run (law_guards b));
    ]

let () =
  exit
    (Windtrap.run "nx_kernel.sort"
       (List.map laws Support.backends @ List.map cpu Support.cpus))
