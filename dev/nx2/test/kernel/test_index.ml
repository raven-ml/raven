(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Gathers and scatters through every kernel library the host runs: a gather and
   an add-scatter are adjoint, and a scatter into its own operand gives the
   fresh result. nx.cpu's group checks what Spec states, element by element
   against a reference built here from the elements' bits: zero reads and
   dropped writes outside the axis, a target's updates in C order, a narrow
   float's sum in float32 rounded once, the first NaN of an extreme; at every
   dtype and through strided, reversed, permuted and broadcast views. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module S = Nx_kernel.Spec
module P = Nx_kernel.Prog
module Support = Nx_kernels_support
open Elements
(* Cases *)

let all_dtypes = D.all

type gather_case = { axis : int; idx : A.any; x : A.any; gviews : string list }

let pp_gather ppf c =
  let (A.Any x) = c.x in
  Format.fprintf ppf "gather along %d of %a %a by %a (%s)" c.axis D.pp
    (A.dtype x) pp_ints (shape_of c.x) pp_ints (shape_of c.idx)
    (String.concat ", " c.gviews)

(* Shapes of rank 1 to 4, extents 0 to 5, now and then a long one. *)
let shape_gen =
  let open Gen in
  let small = array ~size:(int_range 1 4) (int_range 0 5) in
  let long =
    let* n = int_range 100 3000 in
    let+ before = array ~size:(int_range 0 1) (int_range 1 3) in
    Array.append before [| n |]
  in
  frequency [ (6, small); (1, long) ]

let gather_gen ?(dtypes = all_dtypes) ?(shapes = shape_gen) () =
  let open Gen in
  let* d = of_list ~pp:pp_dtype dtypes in
  let* xs = shapes in
  let r = Array.length xs in
  let* axis = int_range 0 (r - 1) in
  let* n = frequency [ (4, int_range 0 6); (1, int_range 7 40) ] in
  let is = Array.mapi (fun i e -> if i = axis then n else e) xs in
  let* xv = view_of r in
  let* bc =
    array ~size:(constant r)
      (frequency [ (3, constant false); (1, constant true) ])
  in
  let broadcast =
    List.filter (fun i -> bc.(i) && i <> axis) (List.init r Fun.id)
  in
  let* iv = view_of ~broadcast r in
  let+ seed = int in
  let rs = Random.State.make [| seed |] in
  let x = operand d xs xv (fun _ -> element d rs) in

  {
    axis;
    idx = int64s is iv (fun () -> position rs xs.(axis));
    x;
    gviews = view_names xv @ List.map (( ^ ) "idx ") (view_names iv);
  }

(* Gathers *)

let gather_on (b : Support.backend) ~axis idx x =
  let module K = (val b.kernels) in
  let (A.Any x) = x in
  let idx = A.expect D.Int64 idx in
  let dst = A.create Rig.host (A.dtype x) (L.shape (A.layout idx)) in
  match K.gather (S.gather ~axis) ~dst idx x with
  | A.Done -> Some (A.Any dst)
  | A.Declined -> None
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

let gather_expected c =
  let xs = shape_of c.x and is = shape_of c.idx in
  let es = elements c.x and ps = positions c.idx in
  let z = zero (dtype_of c.x) in
  Array.init (total is) (fun k ->
      let p = ps.(k) in
      if p < 0L || p >= Int64.of_int xs.(c.axis) then z
      else
        let i = index_of is k in
        i.(c.axis) <- Int64.to_int p;
        es.(flat xs i))

let gather_agrees (b : Support.backend) c =
  match gather_on b ~axis:c.axis c.idx c.x with
  | None -> failf "%s declined a gather" b.name
  | Some y -> equal (array string) (gather_expected c) (elements y)

let law_gather_reference (b : Support.backend) c =
  let d = Int64.of_int (shape_of c.x).(c.axis) in
  cover "out of range"
    (Array.exists (fun p -> p < 0L || p >= d) (positions c.idx));
  cover "broadcast positions" (List.mem "idx broadcast" c.gviews);
  gather_agrees b c

(* Scatters *)

type scatter_case = {
  combine : S.combine;
  saxis : int;
  into : A.any;
  sidx : A.any;
  updates : A.any;
  sviews : string list;
}

let combine_name = function
  | S.Set -> "Set"
  | Add -> "Add"
  | Max -> "Max"
  | Min -> "Min"

let pp_scatter ppf c =
  let (A.Any x) = c.into in
  Format.fprintf ppf "scatter %s along %d into %a %a of %a (%s)"
    (combine_name c.combine) c.saxis D.pp (A.dtype x) pp_ints (shape_of c.into)
    pp_ints (shape_of c.updates)
    (String.concat ", " c.sviews)

let takes_add (D.Any dt) = not (D.is D.Boolean dt)

let scatter_gen ?(dtypes = all_dtypes) ?(shapes = shape_gen)
    ?(ms = Gen.frequency [ (4, Gen.int_range 0 8); (1, Gen.int_range 9 60) ]) ()
    =
  let open Gen in
  let* d = of_list ~pp:pp_dtype dtypes in
  let* combine =
    of_list
      ~pp:(fun ppf c -> Format.pp_print_string ppf (combine_name c))
      (if takes_add d then S.[ Set; Add; Max; Min ] else S.[ Set; Max; Min ])
  in
  let* ts = shapes in
  let r = Array.length ts in
  let* axis = int_range 0 (r - 1) in
  let* m = ms in
  let us = Array.mapi (fun i e -> if i = axis then m else e) ts in
  let* tv = view_of r in
  let* uv = view_of r in
  let* bc =
    array ~size:(constant r)
      (frequency [ (3, constant false); (1, constant true) ])
  in
  let broadcast =
    List.filter (fun i -> bc.(i) && i <> axis) (List.init r Fun.id)
  in
  let* iv = view_of ~broadcast r in
  let+ seed = int in
  let rs = Random.State.make [| seed |] in
  let into = operand d ts tv (fun _ -> element d rs) in
  let updates = operand d us uv (fun _ -> element d rs) in

  {
    combine;
    saxis = axis;
    into;
    sidx = int64s us iv (fun () -> position rs ts.(axis));
    updates;
    sviews =
      List.map (( ^ ) "into ") (view_names tv)
      @ List.map (( ^ ) "updates ") (view_names uv)
      @ List.map (( ^ ) "idx ") (view_names iv);
  }

let scatter_on (b : Support.backend) ?(in_place = false) c =
  let module K = (val b.kernels) in
  let (A.Any into) = c.into in
  let u = A.expect (A.dtype into) c.updates in
  let idx = A.expect D.Int64 c.sidx in
  let s = S.scatter c.combine ~unique:false ~axis:c.saxis in
  let dst, into =
    if in_place then
      let d = A.copy into in
      (d, d)
    else (A.create Rig.host (A.dtype into) (L.shape (A.layout into)), into)
  in
  match K.scatter s ~dst ~into idx u with
  | A.Done -> Some (A.Any dst)
  | A.Declined -> None
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* Whether an extreme of a and b keeps a: the first NaN, else the greater
   (lesser), a on a tie. *)
let keeps combine ~nan_a ~nan_b cmp =
  if nan_a then true
  else if nan_b then false
  else if combine = S.Max then cmp >= 0
  else cmp <= 0

(* An expected element: its bits, or any NaN of a narrow float. *)
type want = Bits of string | Some_nan

(* The combine of the updates [us] into [x], at [d], in order. *)
let combined (D.Any dt as d) combine x us =
  match us with
  | [] -> Bits x
  | _ -> (
      let w = String.length x in
      match combine with
      | S.Set -> Bits (List.nth us (List.length us - 1))
      | Add when is_narrow d ->
          let s =
            List.fold_left
              (fun s u ->
                let y = decode d u in
                if Float.is_nan s then s
                else if Float.is_nan y then y
                else
                  let t = round32 (s +. y) in
                  if t = 0. then 0. else t)
              (let v = decode d x in
               if Float.is_nan v then v else if v = 0. then 0. else v)
              us
          in
          if Float.is_nan s then Some_nan else Bits (encode d s)
      | Add when D.is D.Float dt ->
          let z = if w = 4 then f32 0. else f64 0. in
          Bits (List.fold_left (add_bits ~w) (add_bits ~w z x) us)
      | Add when D.is D.Complex dt ->
          let h = w / 2 in
          let part e k = String.sub e (k * h) h in
          let z = if h = 4 then f32 0. else f64 0. in
          let sum k =
            List.fold_left
              (fun s u -> add_bits ~w:h s (part u k))
              (add_bits ~w:h z (part x k))
              us
          in
          Bits (sum 0 ^ sum 1)
      | Add ->
          Bits
            (int_bits d w
               (List.fold_left
                  (fun s u -> Int64.add s (int_value d u))
                  (int_value d x) us))
      | (Max | Min) when D.is D.Boolean dt ->
          (* Or and And of truths, stored as 0 or 1. *)
          let op = if combine = S.Max then ( || ) else ( && ) in
          let truth e = e <> "\000" in
          let r = List.fold_left (fun t u -> op t (truth u)) (truth x) us in
          Bits (if r then "\001" else "\000")
      | Max | Min ->
          let pick a b =
            let keep =
              if D.is D.Complex dt then
                let h = w / 2 in
                let get = if h = 4 then get32 else get64 in
                let ra = get a 0
                and ia = get a h
                and rb = get b 0
                and ib = get b h in
                let c = compare_floats ra rb in
                let c = if c = 0 then compare_floats ia ib else c in
                keeps combine
                  ~nan_a:(Float.is_nan ra || Float.is_nan ia)
                  ~nan_b:(Float.is_nan rb || Float.is_nan ib)
                  c
              else if D.is D.Float dt then
                let get e =
                  if is_narrow d then decode d e
                  else if w = 4 then get32 e 0
                  else get64 e 0
                in
                let x = get a and y = get b in
                keeps combine ~nan_a:(Float.is_nan x) ~nan_b:(Float.is_nan y)
                  (compare_floats x y)
              else
                let x = int_value d a and y = int_value d b in
                let c =
                  if D.is D.Signed dt then Int64.compare x y
                  else Int64.unsigned_compare x y
                in
                keeps combine ~nan_a:false ~nan_b:false c
            in
            if keep then a else b
          in
          Bits (List.fold_left pick x us))

let scatter_expected c =
  let d = dtype_of c.into in
  let ts = shape_of c.into and us = shape_of c.updates in
  let xs = elements c.into and ues = elements c.updates in
  let ps = positions c.sidx in
  let lands = Array.make (Array.length xs) [] in
  Array.iteri
    (fun k p ->
      if p >= 0L && p < Int64.of_int ts.(c.saxis) then begin
        let i = index_of us k in
        i.(c.saxis) <- Int64.to_int p;
        let t = flat ts i in
        lands.(t) <- ues.(k) :: lands.(t)
      end)
    ps;
  Array.mapi (fun t x -> combined d c.combine x (List.rev lands.(t))) xs

let is_nan_element d e = is_narrow d && Float.is_nan (decode d e)

let check_scatter c y =
  let d = dtype_of c.into in
  let got = elements y in
  let want = scatter_expected c in
  let wrong =
    List.filter_map Fun.id
      (List.mapi
         (fun t w ->
           match w with
           | Bits b when b = got.(t) -> None
           | Some_nan when is_nan_element d got.(t) -> None
           | _ -> Some t)
         (Array.to_list want))
  in
  let show t =
    Printf.sprintf "%d: %s, not %s" t
      (String.concat ""
         (List.map
            (fun ch -> Printf.sprintf "%02x" (Char.code ch))
            (List.of_seq (String.to_seq got.(t)))))
      (match want.(t) with
      | Some_nan -> "a NaN"
      | Bits b ->
          String.concat ""
            (List.map
               (fun ch -> Printf.sprintf "%02x" (Char.code ch))
               (List.of_seq (String.to_seq b))))
  in
  equal (list string) [] (List.map show (List.filteri (fun i _ -> i < 5) wrong))

let scatter_agrees (b : Support.backend) c =
  match scatter_on b c with
  | None -> failf "%s declined a scatter" b.name
  | Some y -> check_scatter c y

let law_scatter_reference (b : Support.backend) c =
  let ps = positions c.sidx in
  let d = (shape_of c.into).(c.saxis) in
  let inside =
    List.filter (fun p -> p >= 0L && p < Int64.of_int d) (Array.to_list ps)
  in
  cover "shared targets"
    (List.length (List.sort_uniq compare inside) < List.length inside);
  cover "dropped updates" (List.length inside < Array.length ps);
  cover "narrow float add" (c.combine = S.Add && is_narrow (dtype_of c.into));
  scatter_agrees b c

(* Laws for every library *)

(* Σ gather x p · y = Σ x · scatter Add 0 p y, in wrapping int64: a position
   outside the axis reads zero in one and drops its write in the other. *)
let law_adjoint (b : Support.backend) c =
  let d = D.Any D.Int64 in
  let rs = Random.State.make [| 3 |] in
  let ints s =
    operand d s
      (plain (Array.length s))
      (fun _ -> int_bits d 8 (Int64.of_int (Random.State.int rs 2001 - 1000)))
  in
  let xs = shape_of c.x and is = shape_of c.idx in
  let x = ints xs and y = ints is in
  let module K = (val b.kernels) in
  match gather_on b ~axis:c.axis c.idx x with
  | None -> ()
  | Some g -> (
      let into = A.of_array D.Int64 xs (Array.make (total xs) 0L) in
      let s =
        scatter_on b
          {
            combine = Add;
            saxis = c.axis;
            into = A.Any into;
            sidx = c.idx;
            updates = y;
            sviews = [];
          }
      in
      match s with
      | None -> ()
      | Some s ->
          let dot a b' =
            let a = A.to_array (A.expect D.Int64 a)
            and b' = A.to_array (A.expect D.Int64 b') in
            let acc = ref 0L in
            Array.iteri
              (fun i v -> acc := Int64.add !acc (Int64.mul v b'.(i)))
              a;
            !acc
          in
          equal int64 (dot x s) (dot g y))

(* A scatter into its own operand, donated, gives the fresh result. *)
let law_in_place (b : Support.backend) c =
  match (scatter_on b c, scatter_on b ~in_place:true c) with
  | Some fresh, Some own -> equal (array string) (elements fresh) (elements own)
  | None, None -> ()
  | _ -> failf "%s declined one of the two" b.name

let test_refusals (b : Support.backend) () =
  let module K = (val b.kernels) in
  let x = A.of_array D.Float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let idx = A.of_array D.Int64 [| 2; 2 |] [| 0L; 1L; 2L; 0L |] in
  let wrong = A.create Rig.host D.Float32 [| 3; 2 |] in
  equal ~msg:"a gather into another shape" answer A.Shape_mismatch
    (K.gather (S.gather ~axis:1) ~dst:wrong idx x);
  equal ~msg:"positions off the source's shape" answer A.Shape_mismatch
    (K.gather (S.gather ~axis:0)
       ~dst:(A.create Rig.host D.Float32 [| 2; 2 |])
       idx x);
  let u = A.of_array D.Float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let dst = A.create Rig.host D.Float32 [| 2; 3 |] in
  equal ~msg:"a scatter of positions and updates of two shapes" answer
    A.Shape_mismatch
    (K.scatter
       (S.scatter Add ~unique:false ~axis:1)
       ~dst ~into:x
       (A.of_array D.Int64 [| 2; 1 |] [| 0L; 1L |])
       u);
  equal ~msg:"a scatter into another shape" answer A.Shape_mismatch
    (K.scatter (S.scatter Add ~unique:false ~axis:1) ~dst:wrong ~into:x idx u);
  let bools = A.of_array D.Bool [| 2 |] [| true; false |] in
  let bdst = A.create Rig.host D.Bool [| 2 |] in
  equal ~msg:"an Add of booleans" answer A.Wrong_dtype
    (K.scatter
       (S.scatter Add ~unique:false ~axis:0)
       ~dst:bdst ~into:bools
       (A.of_array D.Int64 [| 1 |] [| 0L |])
       (A.of_array D.Bool [| 1 |] [| true |]))

(* nx.cpu's own cases *)

(* A float16 target that 2048 updates of one reach: in float32 the sum is exact,
   where float16 steps would stop at 2048. *)
let test_float16_sum (b : Support.backend) () =
  let n = 4096 in
  let into = A.of_array D.Float16 [| 1 |] [| 0. |] in
  let u = A.of_array D.Float16 [| n |] (Array.make n 1.) in
  let idx = A.of_array D.Int64 [| n |] (Array.make n 0L) in
  let dst = A.create Rig.host D.Float16 [| 1 |] in
  let module K = (val b.kernels) in
  match K.scatter (S.scatter Add ~unique:false ~axis:0) ~dst ~into idx u with
  | A.Done -> equal (array float_exact) [| 4096. |] (A.to_array dst)
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* A boolean is true where its byte is not zero: Max and Min of bytes 2,
   255 and 1 are Or and And of truths, stored as 0 or 1. *)
let test_bool_extremes (b : Support.backend) () =
  let module K = (val b.kernels) in
  let bools s =
    A.v D.Bool (L.contiguous [| 3 |]) (Rig.Buffer.of_string s)
  in
  let idx = A.of_array D.Int64 [| 3 |] [| 0L; 1L; 2L |] in
  let run c =
    let dst = A.create Rig.host D.Bool [| 3 |] in
    match
      K.scatter (S.scatter c ~unique:false ~axis:0) ~dst
        ~into:(bools "\002\000\255") idx (bools "\001\002\000")
    with
    | A.Done -> elements (A.Any dst)
    | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r
  in
  equal ~msg:"Max" (array string) [| "\001"; "\001"; "\001" |] (run S.Max);
  equal ~msg:"Min" (array string) [| "\001"; "\000"; "\000" |] (run S.Min)

let large_shapes =
  Gen.of_list
    [ [| 300_000 |]; [| 600; 500 |]; [| 3; 100_000 |]; [| 100_000; 3 |] ]

(* Constructed cases, one per covered regime, so that every seed reaches
   each: positions in order from [ps], elements drawn from seed 1. *)

let positions_of ps =
  let k = ref 0 in
  fun () ->
    let p = ps.(!k mod Array.length ps) in
    incr k;
    p

let gather_example d ~xs ~axis ~n ?(broadcast = []) ps =
  let r = Array.length xs in
  let rs = Random.State.make [| 1 |] in
  let is = Array.mapi (fun i e -> if i = axis then n else e) xs in
  let iv = { (plain r) with broadcast } in
  {
    axis;
    idx = int64s is iv (positions_of ps);
    x = operand d xs (plain r) (fun _ -> element d rs);
    gviews = List.map (( ^ ) "idx ") (view_names iv);
  }

let scatter_example d combine ~ts ~axis ~m ps =
  let r = Array.length ts in
  let rs = Random.State.make [| 1 |] in
  let us = Array.mapi (fun i e -> if i = axis then m else e) ts in
  {
    combine;
    saxis = axis;
    into = operand d ts (plain r) (fun _ -> element d rs);
    sidx = int64s us (plain r) (positions_of ps);
    updates = operand d us (plain r) (fun _ -> element d rs);
    sviews = [];
  }

let gather_examples =
  [
    (* Positions outside the axis, from below and above. *)
    gather_example (D.Any D.Float32) ~xs:[| 3; 4 |] ~axis:1 ~n:5
      [| 0L; -1L; 4L; Int64.max_int; 3L |];
    (* A row take: positions broadcast along the rows. *)
    gather_example (D.Any D.Int16) ~xs:[| 5; 3 |] ~axis:0 ~n:4 ~broadcast:[ 1 ]
      [| 4L; 0L; 2L; 4L |];
  ]

let scatter_examples =
  [
    (* Shared targets and dropped updates. *)
    scatter_example (D.Any D.Float32) S.Add ~ts:[| 4 |] ~axis:0 ~m:8
      [| 0L; 1L; 0L; -1L; 4L; 1L; 1L; Int64.max_int |];
    (* A narrow float's sum. *)
    scatter_example (D.Any D.Float16) S.Add ~ts:[| 3 |] ~axis:0 ~m:6
      [| 0L; 0L; 2L; 5L; 0L; 1L |];
  ]

let gathers = Gen.with_pp pp_gather (gather_gen ())
let large_gathers = Gen.with_pp pp_gather (gather_gen ~shapes:large_shapes ())
let scatters = Gen.with_pp pp_scatter (scatter_gen ())

let large_scatters =
  Gen.with_pp pp_scatter (scatter_gen ~shapes:large_shapes ())

(* Many updates along the axis of few slices: threads split the targets. *)
let long_scatters =
  Gen.with_pp pp_scatter
    (scatter_gen
       ~shapes:(Gen.of_list [ [| 1000 |]; [| 3; 7 |]; [| 5 |] ])
       ~ms:(Gen.int_range 70_000 100_000)
       ())

let int_gathers =
  Gen.with_pp pp_gather (gather_gen ~dtypes:[ D.Any D.Int64 ] ())

(* The suite *)

let laws (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group b.name
    [
      prop "a gather and an add-scatter are adjoint" int_gathers
        (run (law_adjoint b));
      prop "a scatter into its own operand gives the fresh result" scatters
        (run (law_in_place b));
    ]

let cpu (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group ("nx.cpu " ^ b.name)
    [
      prop ~examples:gather_examples
        "a gather reads each position, zero outside the axis" gathers
        (run (law_gather_reference b));
      prop ~examples:scatter_examples
        "a scatter combines each target's updates in C order" scatters
        (run (law_scatter_reference b));
      prop ~count:10 "large gathers on the job's threads" large_gathers
        (run (gather_agrees b));
      prop ~count:10 "large scatters on the job's threads" large_scatters
        (run (scatter_agrees b));
      prop ~count:10 "long scatters split their targets among threads"
        long_scatters
        (run (scatter_agrees b));
      test "boolean extremes are of truths" (fun () ->
          b.around (test_bool_extremes b));
      test "a float16 sum runs in float32 and rounds once" (fun () ->
          b.around (test_float16_sum b));
      test "refuses shapes that do not fit and an Add of booleans" (fun () ->
          b.around (test_refusals b));
    ]

let () =
  exit
    (Windtrap.run "nx_kernel.index"
       (List.map laws Support.backends @ List.map cpu Support.cpus))
