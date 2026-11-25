(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx_kernel.S's gathers and scatters on every kernel library the host runs, in
   nx_kernel.mli's order: a gather reads at each position, a scatter combines
   each target's updates in C order by Set, Add, Max or Min, both refuse what
   does not fit, and a box of the result has the whole's bits. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module Support = Nx_kernels_support
open Elements
open Index_cases

(* A float32 element of bits [v], kept as they are. *)
let bits32 v =
  let b = Bytes.create 4 in
  Bytes.set_int32_le b 0 v;
  Bytes.to_string b

(* Boxes *)

let whole e = { M.start = 0; count = e; step = 1 }

(* A box of shape [s]: a range of each axis, the whole of [axis] where given. *)
let box_gen ?axis s =
  let open Gen in
  let range i e =
    if Some i = axis then constant (whole e)
    else
      let* start = int_range 0 e in
      let+ count = int_range 0 (e - start) in
      { M.start; count; step = 1 }
  in
  let rec from i =
    if i = Array.length s then constant []
    else
      let* r = range i s.(i) in
      let+ rest = from (i + 1) in
      r :: rest
  in
  map Array.of_list (from 0)

let pp_box ppf box =
  Format.pp_print_list ~pp_sep:Format.pp_print_space
    (fun ppf (r : M.range) -> Format.fprintf ppf "%d+%d" r.start r.count)
    ppf (Array.to_list box)

(* The view of [x] that [box] cuts. *)
let cut box (A.Any x) =
  A.Any (Option.get (A.move (M.Slice (Iarray.of_array box)) x))

(* Whether [box] keeps less than the whole of an axis of [s], and whether it
   starts inside one. *)
let cuts box s = Array.exists2 (fun (r : M.range) e -> r.count < e) box s
let inside box = Array.exists (fun (r : M.range) -> r.start > 0) box

(* Misfits *)

(* A gather's shapes, [x], [idx] and [dst], or a scatter's, [into], [idx],
   [updates] and [dst], along [along], each near a fit. *)
type misfit = { along : int; shapes : int array array }

let pp_misfit what ppf c =
  Format.fprintf ppf "%s along %d of %a" what c.along
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ", ")
       pp_ints)
    (Array.to_list c.shapes)

let zeros32 s =
  A.of_array D.Float32 (Iarray.of_array s) (Array.make (total s) 0.)

let zeros64 s = A.of_array D.Int64 (Iarray.of_array s) (Array.make (total s) 0L)

(* [call] into a float32 [dst] computes where [want] is [dst]'s shape; otherwise
   it does not fit, and is not Done and writes nothing outside [dst]. *)
let check_fit ~msg (b : Support.backend) want dst call =
  let fit = want = Ok dst in
  cover "fits" fit;
  cover "misfits" (not fit);
  Misfit.law ~msg ~what:"a shape" b ~fit
    [| (D.Any D.Float32, dst) |]
    (fun g ds -> call g (A.expect D.Float32 ds.(0)))

(* The view of the rank-1 [x] of its [n] elements from [start]. *)
let run_of x ~start ~n =
  Option.get
    (A.move
       (M.Slice (Iarray.of_array [| { M.start; count = n; step = 1 } |]))
       x)

(* Gathers *)

(* The gather's elements by its sentence: [x]'s element at each index with the
   axis replaced by the position there, the element of zero bits outside the
   axis. *)
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

let law_gather (b : Support.backend) c =
  let (D.Any dt as d) = dtype_of c.x in
  let want = gather_expected c in
  let n = Int64.of_int (shape_of c.x).(c.axis) in
  cover "a position outside the axis"
    (Array.exists (fun p -> p < 0L || p >= n) (positions c.idx));
  cover "a sub-byte element" (D.bits dt < 8);
  cover "a long axis" (Array.exists (fun e -> e >= 100) (shape_of c.x));
  cover "positions broadcast along an axis" (List.mem "idx broadcast" c.gviews);
  cover "a signalling NaN moved" (Array.exists (signalling d) want);
  cover "a source of a negative stride" (List.mem "reversed" c.gviews);
  equal (array string) want (elements (gathered b ~axis:c.axis c.idx c.x))

(* Constructed cases, one per covered regime, so that every seed reaches each:
   positions in order from [ps], elements drawn from seed 1. *)
let gather_example d ~xs ~axis ~n ?(broadcast = []) ps =
  let r = Array.length xs in
  let rs = Random.State.make [| 1 |] in
  let iv = { (plain r) with broadcast } in
  {
    axis;
    idx = int64s (along xs axis n) iv (positions_of ps);
    x = operand d xs (plain r) (fun _ -> element d rs);
    gviews = List.map (( ^ ) "idx ") (view_names iv);
  }

(* A source of 3000 elements, read at 40 positions, int64's extremes among
   them. *)
let long_axis =
  gather_example (D.Any D.Float32) ~xs:[| 3000 |] ~axis:0 ~n:40
    [| -1L; 3000L; Int64.min_int; Int64.max_int; 2999L; 0L; 1500L; 77L |]

let gather_examples =
  (* Rows of 2 to 7 elements, which nx.cpu copies whole, taken at 16 positions
     broadcast along them. *)
  let short w d =
    gather_example d ~xs:[| 6; w |] ~axis:0 ~n:16 ~broadcast:[ 1 ]
      (Array.init 16 (fun i -> Int64.of_int ((i * 5 mod 9) - 1)))
  in
  [
    short 2 (D.Any D.Int64);
    short 3 (D.Any D.Float32);
    short 5 (D.Any D.Uint8);
    short 7 (D.Any D.Complex128);
    short 2 (D.Any D.Int4);
    (* Positions outside the axis, from below and above. *)
    gather_example (D.Any D.Float32) ~xs:[| 3; 4 |] ~axis:1 ~n:5
      [| 0L; -1L; 4L; Int64.max_int; 3L |];
    (* Row takes of rows of 3, 8 and 16 bytes, positions outside the axis among
       them. *)
    gather_example (D.Any D.Int16) ~xs:[| 5; 3 |] ~axis:0 ~n:4 ~broadcast:[ 1 ]
      [| 4L; 0L; 2L; 4L |];
    gather_example (D.Any D.Uint8) ~xs:[| 10; 8 |] ~axis:0 ~n:6 ~broadcast:[ 1 ]
      [| 3L; -1L; 9L; 10L; Int64.max_int; 0L |];
    gather_example (D.Any D.Float64) ~xs:[| 7; 2 |] ~axis:0 ~n:5
      ~broadcast:[ 1 ] [| 6L; 7L; 0L; -3L; 2L |];
    (* Packed sub-byte rows of 16 and 12 positions, the extremes among them,
       whose offsets in the source would overflow. *)
    gather_example (D.Any D.Bit) ~xs:[| 4; 16 |] ~axis:0 ~n:2
      [| Int64.max_int; 1L; Int64.min_int; 3L |];
    gather_example (D.Any D.Int4) ~xs:[| 3; 16 |] ~axis:1 ~n:12
      [| 5L; Int64.max_int; 15L; Int64.min_int; -1L |];
    (* A gather moves bits: signalling NaNs keep theirs. *)
    {
      axis = 0;
      idx =
        int64s [| 6 |] (plain 1) (positions_of [| 0L; 1L; 2L; 3L; -1L; 1L |]);
      x =
        operand (D.Any D.Float32) [| 4 |] (plain 1) (fun k ->
            bits32 [| 0x7FA00001l; 0xFF800002l; 0x7FC00000l; 0x3F800000l |].(k));
      gviews = [];
    };
    long_axis;
  ]

let gathers = Gen.with_pp pp_gather (gather_gen ())

(* Σ gather x p · y = Σ x · scatter Add 0 p y, in wrapping int64: a position
   outside the axis reads zero in one and drops its write in the other. *)
let law_adjoint (b : Support.backend) (c, seed) =
  let rs = Random.State.make [| seed |] in
  let xs = shape_of c.x and is = shape_of c.idx in
  let y =
    A.Any
      (A.of_array D.Int64 (Iarray.of_array is)
         (Array.init (total is) (fun _ ->
              Int64.of_int (Random.State.int rs 2001 - 1000))))
  in
  let into = A.Any (zeros64 xs) in
  let g = gathered b ~axis:c.axis c.idx c.x in
  let s =
    scattered b
      {
        combine = Add;
        saxis = c.axis;
        into;
        sidx = c.idx;
        updates = y;
        sviews = [];
      }
  in
  let dot u v =
    let u = positions u and v = positions v in
    let acc = ref 0L in
    Array.iteri (fun i e -> acc := Int64.add !acc (Int64.mul e v.(i))) u;
    !acc
  in
  equal int64 (dot c.x s) (dot g y)

let int_gathers =
  Gen.pair
    (Gen.with_pp pp_gather (gather_gen ~dtypes:[ D.Any D.Int64 ] ()))
    Gen.int

let gather_misfits =
  Gen.with_pp (pp_misfit "gather")
    (let open Gen in
     let* x = Nx_array_gen.shape in
     let* axis = axis_near x in
     let* n = int_range 0 4 in
     let* idx = near (along x axis n) in
     let+ dst = near idx in
     { along = axis; shapes = [| x; idx; dst |] })

let law_gather_refuses (b : Support.backend) c =
  let module K = (val b.kernels) in
  let x = c.shapes.(0) and idx = c.shapes.(1) and dst = c.shapes.(2) in
  check_fit ~msg:"a gather" b (Nx_kernel.Shape.gather ~axis:c.along idx x) dst
    (fun g dst ->
      K.gather ~axis:c.along ~dst
        (g.Misfit.g (Support.on b (zeros64 idx)))
        (g.Misfit.g (Support.on b (zeros32 x))))

(* A destination over four elements of [x] from its second. *)
let test_gather_overlap (b : Support.backend) =
  let module K = (val b.kernels) in
  let x = Support.on b (zeros32 [| 8 |]) in
  let dst = run_of x ~start:1 ~n:4 in
  equal answer A.Overlapping
    (K.gather ~axis:0 ~dst (Support.on b (zeros64 [| 4 |])) x)

(* A box of a gather's indices, computed from that box of [idx] and the part of
   [x] it reads, all of the axis and the box's range of every other. *)
let law_gather_box (b : Support.backend) (c, box) =
  let xs = shape_of c.x in
  let xbox =
    Array.mapi (fun i r -> if i = c.axis then whole xs.(i) else r) box
  in
  cover "a box short of an axis" (cuts box (shape_of c.idx));
  cover "a box from inside an axis" (inside box);
  let y = gathered b ~axis:c.axis c.idx c.x in
  let part = gathered b ~axis:c.axis (cut box c.idx) (cut xbox c.x) in
  equal (array string) (elements (cut box y)) (elements part)

let gather_boxes =
  Gen.with_pp
    (fun ppf (c, box) -> Format.fprintf ppf "%a, box %a" pp_gather c pp_box box)
    (let open Gen in
     let* c = gather_gen () in
     let+ box = box_gen (shape_of c.idx) in
     (c, box))

(* Scatters *)

(* [c] with its arrays on [b]'s device, through their views. *)
let placed (b : Support.backend) c =
  let on (A.Any x) = A.Any (Support.on b x) in
  { c with into = on c.into; updates = on c.updates; sidx = on c.sidx }

let law_set_last (b : Support.backend) c =
  let shared, dropped = sharing c in
  cover "shared targets" shared;
  cover "dropped updates" dropped;
  check_scatter c (scattered b c)

(* Set: shared targets and positions past both ends, every update on one target,
   no update, and one row. *)
let set_examples =
  let ex d = scatter_example d Nx_kernel.Set in
  [
    ex (D.Any D.Float32) ~ts:[| 4 |] ~axis:0 ~m:6 [| 1L; 3L; 1L; -1L; 4L; 1L |];
    ex (D.Any D.Int8) ~ts:[| 3; 2 |] ~axis:1 ~m:5 [| 0L |];
    ex (D.Any D.Bfloat16) ~ts:[| 3 |] ~axis:0 ~m:0 [| 0L |];
    ex (D.Any D.Complex128) ~ts:[| 5; 4 |] ~axis:0 ~m:1 ~broadcast:[ 1 ]
      [| 3L |];
  ]

let sets = Gen.with_pp pp_scatter (scatter_gen ~combines:Nx_kernel.[ Set ] ())

(* [scatter1 b c dt into idx u] is the scatter by [c] of the rank-1 [u] at [idx]
   into [into], read back. *)
let scatter1 (b : Support.backend) c dt into idx u =
  let module K = (val b.kernels) in
  let n = Array.length into and m = Array.length idx in
  let dst = A.create b.device dt [| n |] in
  let into = Support.on b (A.of_array dt [| n |] into) in
  let idx = Support.on b (A.of_array D.Int64 [| m |] idx) in
  let u = Support.on b (A.of_array dt [| m |] u) in
  match K.scatter c ~unique:false ~axis:0 ~dst ~into idx u with
  | A.Done -> A.to_array (Support.host dst)
  | A.Declined -> failf "%s declined a scatter" b.name
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* Integer Adds *)

let int_dtypes =
  List.filter
    (fun (D.Any dt) -> D.is D.Signed dt || D.is D.Unsigned dt)
    all_dtypes

let law_int_add (b : Support.backend) c =
  let (D.Any dt as d) = dtype_of c.into in
  let w = D.bits dt in
  (* An element's value as a float, exact below 2^53 and near enough above to
     tell a sum past the dtype's range. *)
  let value e =
    let v = Int64.to_float (int_value d e) in
    if v < 0. && not (D.is D.Signed dt) then v +. 0x1p64 else v
  in
  let lo, hi =
    if D.is D.Signed dt then
      (-.Float.ldexp 1. (w - 1), Float.ldexp 1. (w - 1) -. 1.)
    else (0., Float.ldexp 1. w -. 1.)
  in
  let xs = elements c.into in
  let past t us =
    us <> []
    &&
    let s = List.fold_left (fun s u -> s +. value u) (value xs.(t)) us in
    s < lo || s > hi
  in
  cover "a sub-byte integer" (w < 8);
  cover "a sum past its dtype's range"
    (Array.exists Fun.id (Array.mapi past (lands c)));
  check_scatter c (scattered b c)

(* Sums that wrap: int8's greatest plus one and least minus one, and int4's
   greatest plus two. *)
let int_add_examples =
  let ex d into ps us =
    let bits v = int_bits d (String.length (zero d)) v in
    let into = Array.of_list into and us = Array.of_list us in
    {
      combine = Add;
      saxis = 0;
      into =
        operand d [| Array.length into |] (plain 1) (fun k -> bits into.(k));
      sidx =
        int64s [| Array.length us |] (plain 1) (positions_of (Array.of_list ps));
      updates = operand d [| Array.length us |] (plain 1) (fun k -> bits us.(k));
      sviews = [];
    }
  in
  [
    ex (D.Any D.Int8) [ 127L; -128L ] [ 0L; 1L ] [ 1L; -1L ];
    ex (D.Any D.Int4) [ 7L ] [ 0L; 0L ] [ 1L; 1L ];
  ]

let int_adds =
  Gen.with_pp pp_scatter
    (scatter_gen ~dtypes:int_dtypes ~combines:Nx_kernel.[ Add ] ())

(* Float Adds' bound *)

(* The sum of [xs] within 2^-100 of the sum of their magnitudes, and that
   sum: the running sum held in two parts. *)
let exact xs =
  let hi = ref 0. and lo = ref 0. and mag = ref 0. in
  List.iter
    (fun x ->
      let s, t = Exact.two_sum !hi x in
      hi := s;
      lo := !lo +. t;
      mag := !mag +. Float.abs x)
    xs;
  (!hi +. !lo, !mag)

(* float32's largest finite value. *)
let most_float32 = 0x1.FFFFFEp127

(* γ(n) of the unit roundoff [u]. *)
let gamma u n =
  let n = Float.of_int n in
  n *. u /. (1. -. (n *. u))

(* Whether a float32 sum of [xs] left to right from +0 may pass float32's
   largest finite value: a prefix [P_j] of [j] terms with
   [|P_j| + γ(j) Σ_{i ≤ j} |x_i|] past it. *)
let partial_overflows xs =
  let rec go j hi lo mag = function
    | [] -> false
    | x :: rest ->
        let s, t = Exact.two_sum hi x in
        let lo = lo +. t and mag = mag +. Float.abs x in
        Float.abs (s +. lo) +. ((gamma 0x1p-24 j +. 0x1p-100) *. mag)
        > most_float32
        || go (j + 1) s lo mag rest
  in
  go 1 0. 0. 0. xs

(* Whether a float32 sum [x + S] of [x] and updates [us] at increasing positions
   [js] along the axis may pass float32's largest finite value, [S] summed by
   the complete binary tree over the positions, nx.cuda's and nx.amd's order: a
   subtree's sum of [k] terms [P] with [|P| + γ(k) Σ |ui|] past it, or the
   whole's. An aligned subtree of [size] positions from [lo] with more than one
   update adds its halves. *)
let tree_overflows x js us =
  let passes xs =
    let s, mag = exact xs in
    Float.abs s +. ((gamma 0x1p-24 (List.length xs) +. 0x1p-100) *. mag)
    > most_float32
  in
  let rec node lo size = function
    | [] | [ _ ] -> false
    | jus ->
        let h = size / 2 in
        let l, r = List.partition (fun (j, _) -> j < lo + h) jus in
        passes (List.map snd jus) || node lo h l || node (lo + h) h r
  in
  let rec span size =
    if List.exists (fun j -> j >= size) js then span (2 * size) else size
  in
  node 0 (span 1) (List.combine js us) || passes (x :: us)

(* The largest finite value of a narrow float. *)
let most_finite (D.Any dt) =
  match dt with
  | D.Float16 -> 65504.
  | D.Bfloat16 -> 0x1.FEp127
  | D.Float8_e4m3fn -> 448.
  | D.Float8_e5m2 -> 57344.
  | _ -> 6.

(* A finite element of a float or complex dtype: a draw of (-1, 1) times
   2^e, e over 31 binades below 2^30 or a narrow float's largest, so that
   sums of a few round at every width; float32's and a narrow float's
   rounded to their nearest. *)
let finite (D.Any dt as d) r =
  let top =
    if is_narrow d then Float.to_int (Float.log2 (most_finite d)) else 30
  in
  let v () =
    Float.ldexp (Random.State.float r 2. -. 1.) (top - Random.State.int r 31)
  in
  match dt with
  | D.Float32 -> f32 (round32 (v ()))
  | D.Float64 -> f64 (v ())
  | D.Complex64 -> f32 (round32 (v ())) ^ f32 (round32 (v ()))
  | D.Complex128 -> f64 (v ()) ^ f64 (v ())
  | _ -> encode d (v ())

(* The float parts of an element of [d] as values: one, a complex number's two,
   none for an integer or a boolean. *)
let parts (D.Any dt as d) e =
  if is_narrow d then [ decode d e ]
  else if D.is D.Complex dt then
    let h = String.length e / 2 in
    let get = if h = 4 then get32 else get64 in
    [ get e 0; get e h ]
  else if D.is D.Float dt then
    [ (if String.length e = 4 then get32 e 0 else get64 e 0) ]
  else []

(* Whether a narrow float's sum of [x] and its updates [us] at positions [js],
   in C order, cannot overflow: no partial float32 sum passes float32's largest
   finite value, left to right (nx.cpu's order) nor by the position tree, nor
   the sum and its bound the dtype's. The contract's bound holds barring
   overflow in the library's order, and the libraries state these two. *)
let narrow_checked d x js us =
  let xs = x :: us in
  let want, mag = exact xs in
  let bound = (gamma 0x1p-24 (List.length xs) +. 0x1p-100) *. mag in
  not
    (partial_overflows xs || tree_overflows x js us
    || Float.abs want +. bound > most_finite d)

(* Why a target's Add [got] of [x] and its updates [lands], each at its position
   along the axis, is past [γ(n + 1) (|x| + Σ |ui|)] of the exact sum, part by
   part, [u] the unit roundoff of float32 for float32, narrow floats and
   complex64 and of float64 otherwise; a narrow float's sum is rounded once from
   a float32 within it, so [got] lies between the roundings of its ends. Each
   part's magnitudes bound the complex number's moduli. A narrow sum that may
   overflow, a partial float32 sum past float32's largest finite value or the
   sum rounding past the dtype's, is not checked: [None]. *)
let past (D.Any dt as d) x lands got =
  let js = List.map fst lands and us = List.map snd lands in
  let part = if D.is D.Complex dt then D.bits dt / 2 else D.bits dt in
  let u = if part = 64 then 0x1p-53 else 0x1p-24 in
  let g = gamma u (List.length us + 1) in
  let terms = List.map (parts d) (x :: us) in
  let wrong =
    List.concat
      (List.mapi
         (fun k got ->
           let xs = List.map (fun t -> List.nth t k) terms in
           let want, mag = exact xs in
           let bound = (g *. mag) +. (0x1p-100 *. mag) in
           if is_narrow d then
             if not (narrow_checked d (List.hd xs) js (List.tl xs)) then []
             else
               let lo = decode d (encode d (want -. bound))
               and hi = decode d (encode d (want +. bound)) in
               if lo <= got && got <= hi then []
               else [ Printf.sprintf "%h is not in [%h, %h]" got lo hi ]
           else if Float.abs (got -. want) <= bound then []
           else
             [
               Printf.sprintf "%h is %h from %h, past %h" got (got -. want) want
                 bound;
             ])
         (parts d got))
  in
  match wrong with [] -> None | w -> Some (String.concat "; " w)

(* Whether [got] differs from the exact sum of [x] and [us] in a part. *)
let rounds d x us got =
  let terms = List.map (parts d) (x :: us) in
  List.exists Fun.id
    (List.mapi
       (fun k g -> fst (exact (List.map (fun t -> List.nth t k) terms)) <> g)
       (parts d got))

(* At each target an update reaches, a float Add is within its bound of the
   exact sum; at the others it is into's element, its bits unchanged. *)
let law_add_bound (b : Support.backend) c =
  let d = dtype_of c.into in
  let ls = landings c and xs = elements c.into in
  cover "a target several updates reach"
    (Array.exists (fun l -> List.length l > 1) ls);
  cover "a narrow float" (is_narrow d);
  cover "a checked bfloat16 sum of magnitudes past float32's largest"
    (d = D.Any D.Bfloat16
    && Array.exists Fun.id
         (Array.mapi
            (fun t lands ->
              let x = decode d xs.(t) in
              let js = List.map fst lands
              and us = List.map (fun (_, u) -> decode d u) lands in
              lands <> []
              && snd (exact (x :: us)) > most_float32
              && narrow_checked d x js us)
            ls));
  cover "reached" (Array.exists (fun l -> l <> []) ls);
  cover "a signalling NaN no update reaches"
    (Array.exists Fun.id
       (Array.mapi (fun t l -> l = [] && signalling d xs.(t)) ls));
  let got = elements (scattered b c) in
  cover "a wide sum that rounds"
    ((not (is_narrow d))
    && Array.exists Fun.id
         (Array.mapi
            (fun t lands ->
              lands <> [] && rounds d xs.(t) (List.map snd lands) got.(t))
            ls));
  let wrong =
    List.filter_map Fun.id
      (Array.to_list
         (Array.mapi
            (fun t lands ->
              if lands = [] then
                if got.(t) = xs.(t) then None
                else
                  Some
                    (Printf.sprintf "%d: %s, not into's %s" t
                       (hex got.(t))
                       (hex xs.(t)))
              else
                Option.map
                  (Printf.sprintf "%d: %s" t)
                  (past d xs.(t) lands got.(t)))
            ls))
  in
  equal (list string) [] (List.filteri (fun i _ -> i < 5) wrong)

(* A bfloat16 Add into +0 of 40 updates alternating 1.5 · 2^127 and its
   negation: their magnitudes sum past float32's largest finite value, and their
   partial sums stay at or below 1.5 · 2^127, left to right and by the position
   tree. *)
let wide_sum =
  let d = D.Any D.Bfloat16 and m = 40 in
  {
    combine = Nx_kernel.Add;
    saxis = 0;
    into = operand d [| 1 |] (plain 1) (fun _ -> encode d 0.);
    sidx = int64s [| m |] (plain 1) (fun () -> 0L);
    updates =
      operand d [| m |] (plain 1) (fun k ->
          encode d (if k mod 2 = 0 then 0x1.8p127 else -0x1.8p127));
    sviews = [];
  }

(* Two targets no update reaches, -0 and a signalling NaN: the positions lie
   past both ends. *)
let untouched =
  let d = D.Any D.Float32 in
  {
    combine = Nx_kernel.Add;
    saxis = 0;
    into =
      operand d [| 2 |] (plain 1) (fun k ->
          [| f32 (-0.); bits32 0x7FA00001l |].(k));
    sidx = int64s [| 2 |] (plain 1) (positions_of [| 2L; -1L |]);
    updates = operand d [| 2 |] (plain 1) (fun _ -> f32 1.);
    sviews = [];
  }

(* Add's bound: many updates on few targets, an embedding gradient's rows, every
   update on one target, no update, magnitudes past float32's largest, and
   targets no update reaches. *)
let bound_examples =
  let ex d = scatter_example d Nx_kernel.Add ~draw:finite in
  [
    ex (D.Any D.Float32) ~ts:[| 2 |] ~axis:0 ~m:40 [| 0L; 1L |];
    ex (D.Any D.Bfloat16) ~ts:[| 3; 8 |] ~axis:0 ~m:50 ~broadcast:[ 1 ]
      [| 2L; 0L; 2L; 1L |];
    ex (D.Any D.Float16) ~ts:[| 1 |] ~axis:0 ~m:60 [| 0L |];
    ex (D.Any D.Complex64) ~ts:[| 4 |] ~axis:0 ~m:30 [| 3L; 0L; 3L |];
    ex (D.Any D.Float64) ~ts:[| 3 |] ~axis:0 ~m:0 [| 0L |];
    wide_sum;
    untouched;
  ]

let finite_adds =
  Gen.with_pp pp_scatter
    (scatter_gen ~dtypes:float_dtypes
       ~combines:Nx_kernel.[ Add ]
       ~draw:finite ())

(* Two runs of one scatter on fresh destinations agree, a NaN with any NaN: its
   sign and payload may change with the destination's address. *)
let law_add_again (b : Support.backend) c =
  cover "shared targets" (fst (sharing c));
  equal
    (array (same (dtype_of c.into)))
    (elements (scattered b c))
    (elements (scattered b c))

let test_bool_add (b : Support.backend) =
  let module K = (val b.kernels) in
  let on x = Support.on b x in
  let bools = on (A.of_array D.Bool [| 2 |] [| true; false |]) in
  let idx = on (A.of_array D.Int64 [| 1 |] [| 0L |]) in
  let u = on (A.of_array D.Bool [| 1 |] [| true |]) in
  Misfit.check ~msg:"an Add of booleans" ~what:"a dtype" b
    [| (D.Any D.Bool, [| 2 |]) |]
    (fun g ds ->
      K.scatter Add ~unique:false ~axis:0 ~dst:(A.expect D.Bool ds.(0))
        ~into:(g.Misfit.g bools) (g.Misfit.g idx) (g.Misfit.g u))

(* Max and Min *)

(* The least and greatest values of an integer dtype of [w] bytes. *)
let int_extremes (D.Any dt) w =
  let top = (8 * w) - 1 in
  if not (D.is D.Signed dt) then (0L, -1L)
  else if w = 8 then (Int64.min_int, Int64.max_int)
  else
    (Int64.neg (Int64.shift_left 1L top), Int64.pred (Int64.shift_left 1L top))

(* Whether [d] is an integer dtype of whole bytes. *)
let byte_int (D.Any dt) =
  (D.is D.Signed dt || D.is D.Unsigned dt) && D.bits dt >= 8

(* An element of [d] as {!Elements.element} draws it, and a quarter of a
   byte-wide integer's at one of its dtype's extremes. *)
let extreme_element (D.Any dt as d) r =
  if (not (byte_int d)) || Random.State.int r 4 > 0 then element d r
  else
    let w = D.bits dt / 8 in
    let lo, hi = int_extremes d w in
    int_bits d w (if Random.State.bool r then lo else hi)

(* Whether an element of [d] is at one of its byte-wide integer dtype's
   extremes. *)
let at_extreme (D.Any dt as d) e =
  byte_int d
  &&
  let w = D.bits dt / 8 in
  let lo, hi = int_extremes d w in
  e = int_bits d w lo || e = int_bits d w hi

(* The regimes of an extreme scatter's targets that an order can get wrong, each
   over a target that some update reaches. *)
let extreme_covers c =
  let (D.Any dt as d) = dtype_of c.into in
  let reached =
    List.filter
      (fun (_, us) -> us <> [])
      (List.combine (Array.to_list (elements c.into)) (Array.to_list (lands c)))
  in
  let any f = List.exists f reached in
  let tie (x, us) =
    let vs = List.filter (fun e -> not (nan_of d e)) (x :: us) in
    List.exists
      (fun a ->
        List.exists
          (fun b ->
            match (parts d a, parts d b) with
            | [ ra; ia ], [ rb; ib ] ->
                compare_floats ra rb = 0 && compare_floats ia ib <> 0
            | _ -> false)
          vs)
      vs
  in
  let zeros (x, us) =
    let ps = List.concat_map (parts d) (x :: us) in
    List.exists (fun p -> p = 0. && Float.sign_bit p) ps
    && List.exists (fun p -> p = 0. && not (Float.sign_bit p)) ps
  in
  let nan_im e =
    match parts d e with
    | [ re; im ] -> Float.is_nan im && not (Float.is_nan re)
    | _ -> false
  in
  cover "complex64" (D.name dt = "complex64");
  cover "complex128" (D.name dt = "complex128");
  cover "booleans" (D.is D.Boolean dt);
  cover "a tie of real parts, imaginary parts apart"
    (D.is D.Complex dt && any tie);
  cover "-0 and +0 at one target" (any zeros);
  cover "a NaN target" (any (fun (x, _) -> nan_of d x));
  cover "a NaN update" (any (fun (_, us) -> List.exists (nan_of d) us));
  cover "a NaN imaginary part alone"
    (any (fun (x, us) -> List.exists nan_im (x :: us)));
  cover "an infinity"
    (any (fun (x, us) ->
         List.exists
           (fun e -> List.exists (fun p -> Float.abs p = infinity) (parts d e))
           (x :: us)));
  cover "an integer at its dtype's extreme"
    (any (fun (x, us) -> List.exists (at_extreme d) (x :: us)))

(* The left fold of Prog.Maximum or Minimum over [x], u1, …, un, computed on the
   host: -0 below +0, a complex number by its real part then its imaginary part,
   a NaN where one of them is. A NaN's bits are unspecified, and a complex
   number with a NaN part is a NaN (nx_kernel.mli's reductions), so any such
   result stands for one. *)
let law_extreme (b : Support.backend) c =
  extreme_covers c;
  let d = dtype_of c.into in
  check_elements
    (fun w e ->
      match w with
      | Bits w -> w = e
      | Value w -> (nan_of d w && nan_of d e) || Testable.equal (same d) w e)
    (scatter_expected c) (scattered b c)

(* One case for each regime: [ex d combine x us] scatters the updates [us] all
   onto [into]'s first element [x], beside a second element no update reaches.
   Booleans are true where their byte is not zero. *)
let extreme_examples =
  let ex d combine x us =
    let m = List.length us and us = Array.of_list us in
    {
      combine;
      saxis = 0;
      into = operand d [| 2 |] (plain 1) (fun _ -> x);
      sidx = int64s [| m |] (plain 1) (fun () -> 0L);
      updates = operand d [| m |] (plain 1) (fun k -> us.(k));
      sviews = [];
    }
  in
  let c64 re im = f32 re ^ f32 im and c128 re im = f64 re ^ f64 im in
  let ib (D.Any dt as d) v = int_bits d (D.bits dt / 8) v in
  let i8 = D.Any D.Int8 and u64 = D.Any D.Uint64 and i64 = D.Any D.Int64 in
  let bool = D.Any D.Bool in
  [
    ex (D.Any D.Complex64) Nx_kernel.Max (c64 1. 0.)
      [ c64 1. 5.; c64 1. (-2.); c64 0. 9. ];
    ex (D.Any D.Complex128) Nx_kernel.Min (c128 0. 1.)
      [ c128 (-0.) 3.; c128 (-0.) 1.; c128 0. (-7.) ];
    ex (D.Any D.Complex64) Nx_kernel.Max (c64 0. (-0.))
      [ c64 0. 0.; c64 2. nan; c64 (-1.) 1. ];
    ex (D.Any D.Complex128) Nx_kernel.Min (c128 nan 0.)
      [ c128 neg_infinity 0.; c128 1. 1. ];
    ex (D.Any D.Complex64) Nx_kernel.Min (c64 infinity 0.)
      [ c64 neg_infinity 2.; c64 neg_infinity (-0.); c64 neg_infinity 0. ];
    ex (D.Any D.Float32) Nx_kernel.Min (f32 0.) [ f32 (-0.); f32 0. ];
    ex (D.Any D.Float64) Nx_kernel.Max (f64 (-0.))
      [ f64 0.; f64 neg_infinity; f64 nan; f64 infinity ];
    ex (D.Any D.Bfloat16) Nx_kernel.Max
      (encode (D.Any D.Bfloat16) (-0.))
      [ encode (D.Any D.Bfloat16) 0.; encode (D.Any D.Bfloat16) (-1.) ];
    ex i8 Nx_kernel.Max (ib i8 0L) [ ib i8 127L; ib i8 (-128L) ];
    ex u64 Nx_kernel.Max (ib u64 1L) [ ib u64 (-1L); ib u64 0L ];
    ex i64 Nx_kernel.Min (ib i64 0L)
      [ ib i64 Int64.min_int; ib i64 Int64.max_int ];
    (* Max is Or and Min is And of truths, stored as 0 or 1. *)
    ex bool Nx_kernel.Max "\002" [ "\000"; "\255" ];
    ex bool Nx_kernel.Min "\255" [ "\002"; "\000"; "\001" ];
  ]

let extremes =
  Gen.with_pp pp_scatter
    (scatter_gen ~combines:Nx_kernel.[ Max; Min ] ~draw:extreme_element ())

(* Under ~unique *)

(* Scatters whose updates reach distinct targets: along the axis, each slice's
   positions a shuffled prefix of the axis, the rest outside it. *)
let unique_gen =
  let open Gen in
  let* d = of_list ~pp:pp_dtype all_dtypes in
  let* combine = combine_gen d in
  let* ts = shape_gen in
  let r = Array.length ts in
  let* axis = int_range 0 (r - 1) in
  let* m = frequency [ (4, int_range 0 8); (1, int_range 9 60) ] in
  let us = Array.mapi (fun i e -> if i = axis then m else e) ts in
  let* tv = view_of r in
  let* uv = view_of r in
  let+ seed = int in
  let rs = Random.State.make [| seed |] in
  let d_axis = ts.(axis) in
  let slices = Hashtbl.create 16 in
  let shuffled () =
    let p = Array.init d_axis Fun.id in
    for i = d_axis - 1 downto 1 do
      let j = Random.State.int rs (i + 1) in
      let t = p.(i) in
      p.(i) <- p.(j);
      p.(j) <- t
    done;
    p
  in
  let k = ref 0 in
  let next () =
    let i = index_of us !k in
    incr k;
    let j = i.(axis) in
    i.(axis) <- 0;
    let p =
      match Hashtbl.find_opt slices i with
      | Some p -> p
      | None ->
          let p = shuffled () in
          Hashtbl.add slices i p;
          p
    in
    if j < d_axis then Int64.of_int p.(j)
    else if j land 1 = 0 then -1L
    else Int64.of_int d_axis
  in
  {
    combine;
    saxis = axis;
    into = operand d ts tv (fun _ -> element d rs);
    sidx = int64s us (plain r) next;
    updates = operand d us uv (fun _ -> element d rs);
    sviews =
      List.map (( ^ ) "into ") (view_names tv)
      @ List.map (( ^ ) "updates ") (view_names uv);
  }

let unique_scatters = Gen.with_pp pp_scatter unique_gen

(* A scatter whose updates reach distinct targets has, under the caller's
   promise of it, the bits it has without, any NaN for a NaN. *)
let law_unique (b : Support.backend) c =
  cover "Add" (c.combine = Nx_kernel.Add);
  cover "Set" (c.combine = Nx_kernel.Set);
  cover "an extreme" (c.combine = Nx_kernel.Max || c.combine = Nx_kernel.Min);
  cover "dropped updates" (snd (sharing c));
  cover "every target reached"
    ((shape_of c.updates).(c.saxis) >= (shape_of c.into).(c.saxis));
  let c = placed b c in
  let views f = List.exists (fun (A.Any x) -> f (A.layout x)) in
  let arrays = [ c.into; c.updates; c.sidx ] in
  cover "through views" (views (fun l -> not (L.is_contiguous l)) arrays);
  cover "through a view with gaps" (views Support.gapped arrays);
  equal
    (array (same (dtype_of c.into)))
    (elements (scattered b c))
    (elements (scattered ~unique:true b c))

(* A scatter into its own into, which it reads at each target before writing
   there, gives the fresh result, any NaN for a NaN. *)
let law_in_place (b : Support.backend) c =
  List.iter
    (fun m -> cover (combine_name m) (c.combine = m))
    Nx_kernel.[ Set; Add; Max; Min ];
  equal
    (array (same (dtype_of c.into)))
    (elements (scattered b c))
    (elements (scattered b ~in_place:true c))

let scatter_misfits =
  Gen.with_pp (pp_misfit "scatter")
    (let open Gen in
     let* into = Nx_array_gen.shape in
     let* axis = axis_near into in
     let* n = int_range 0 4 in
     let u = along into axis n in
     let* idx = near u in
     let* updates = near u in
     let+ dst = near into in
     { along = axis; shapes = [| into; idx; updates; dst |] })

let law_scatter_refuses (b : Support.backend) c =
  let module K = (val b.kernels) in
  let into = c.shapes.(0) and idx = c.shapes.(1) in
  let u = c.shapes.(2) and dst = c.shapes.(3) in
  check_fit ~msg:"a scatter" b
    (Nx_kernel.Shape.scatter ~axis:c.along into idx u) dst (fun g dst ->
      K.scatter Set ~unique:false ~axis:c.along ~dst
        ~into:(g.Misfit.g (Support.on b (zeros32 into)))
        (g.Misfit.g (Support.on b (zeros64 idx)))
        (g.Misfit.g (Support.on b (zeros32 u))))

(* A destination over four of eight updates, from the second, and one over into,
   one element on. *)
let scatter_overlaps =
  [
    ( "a destination over updates",
      fun (b : Support.backend) ->
        let module K = (val b.kernels) in
        let u = Support.on b (zeros32 [| 8 |]) in
        K.scatter Set ~unique:false ~axis:0 ~dst:(run_of u ~start:1 ~n:4)
          ~into:(Support.on b (zeros32 [| 4 |]))
          (Support.on b (zeros64 [| 8 |]))
          u );
    ( "a destination over into, one element on",
      fun (b : Support.backend) ->
        let module K = (val b.kernels) in
        let x = Support.on b (zeros32 [| 9 |]) in
        K.scatter Set ~unique:false ~axis:0 ~dst:(run_of x ~start:1 ~n:8)
          ~into:(run_of x ~start:0 ~n:8)
          (Support.on b (zeros64 [| 2 |]))
          (Support.on b (zeros32 [| 2 |])) );
  ]

(* A box of the axes off a scatter's axis, computed from that box of [into] and
   of [idx] and [updates], all of the axis, has the bits of that box of the
   whole result: a move's, an integer's and a boolean's exactly, a float
   extreme's any NaN for a NaN. *)
let law_scatter_box (b : Support.backend) (c, box) =
  let d = dtype_of c.into in
  let us = shape_of c.updates in
  let ubox =
    Array.mapi (fun i r -> if i = c.saxis then whole us.(i) else r) box
  in
  cover "Set" (c.combine = Nx_kernel.Set);
  cover "an integer Add" (c.combine = Nx_kernel.Add);
  cover "Max or Min" (c.combine = Nx_kernel.Max || c.combine = Nx_kernel.Min);
  cover "a box short of an axis" (cuts box (shape_of c.into));
  cover "a box from inside an axis" (inside box);
  let part =
    {
      c with
      into = cut box c.into;
      sidx = cut ubox c.sidx;
      updates = cut ubox c.updates;
    }
  in
  let fixed = c.combine = Nx_kernel.Set || c.combine = Nx_kernel.Add in
  equal
    (array (if fixed then string else same d))
    (elements (cut box (scattered b c)))
    (elements (scattered b part))

let scatter_boxes =
  Gen.with_pp
    (fun ppf (c, box) ->
      Format.fprintf ppf "%a, box %a" pp_scatter c pp_box box)
    (let open Gen in
     let* c =
       frequency
         [
           (3, scatter_gen ~combines:Nx_kernel.[ Set; Max; Min ] ());
           (1, scatter_gen ~dtypes:int_dtypes ~combines:Nx_kernel.[ Add ] ());
         ]
     in
     let+ box = box_gen ~axis:c.saxis (shape_of c.into) in
     (c, box))

(* The suite *)

let laws (d : Support.declared) =
  let backend () = Support.opened d in
  let run f x =
    let b = backend () in
    b.around (fun () -> f b x)
  in
  let once f () =
    let b = backend () in
    b.around (fun () -> f b)
  in
  let f32 = D.Float32 in
  group d.label
    [
      group "gather"
        [
          prop ~examples:gather_examples
            "reads x at each position, the element of zero bits outside the \
             axis"
            gathers (run law_gather);
          prop "and a scatter by Add are adjoint" int_gathers (run law_adjoint);
          prop
            "a call Shape.gather does not fit is not Done and writes nothing \
             outside dst"
            gather_misfits (run law_gather_refuses);
          test "into a destination over x is refused as Overlapping"
            (once test_gather_overlap);
          prop
            "of a box of idx, from the part of x it reads, is that box of the \
             whole"
            gather_boxes (run law_gather_box);
        ];
      group "scatter"
        [
          prop ~examples:set_examples
            "Set stores each target's last update in C order, into's bits \
             where none lands"
            sets (run law_set_last);
          test "Set keeps a target's last update"
            (once (fun b ->
                 equal (array float_exact) [| 10.; 4.; 30.; 2. |]
                   (scatter1 b Set f32 [| 10.; 20.; 30.; 40. |]
                      [| 1L; 3L; 1L; 1L |] [| 1.; 2.; 3.; 4. |])));
          test "Set drops positions outside the axis, int64's extremes included"
            (once (fun b ->
                 equal (array float_exact) [| 5.; 20.; 30.; 40. |]
                   (scatter1 b Set f32 [| 10.; 20.; 30.; 40. |]
                      [| -1L; 4L; Int64.min_int; Int64.max_int; 0L |]
                      [| 1.; 2.; 3.; 4.; 5. |])));
          prop ~examples:int_add_examples
            "Add of integers wraps the sum of x and u1 to un at the dtype's \
             width"
            int_adds (run law_int_add);
          prop ~examples:bound_examples
            "Add of floats is within γ(n + 1) (|x| + Σ|ui|) of the exact sum, \
             into's bits where none lands"
            finite_adds (run law_add_bound);
          prop
            "Add of floats gives the same result on every run, a NaN's sign \
             and payload aside"
            float_adds (run law_add_again);
          cases "Add sums from the target, with no +0 before it" ~id:fst
            [
              ( "-0 and -0 updates give -0",
                ([| -0. |], [| -0.; -0. |], [| -0. |]) );
              ("+0 and a -0 update give +0", ([| 0. |], [| -0. |], [| 0. |]));
            ]
            (fun (_, (into, u, want)) ->
              let b = backend () in
              b.around (fun () ->
                  equal (array float_exact) want
                    (scatter1 b Add f32 into (Array.map (fun _ -> 0L) u) u)));
          test "Add of opposite infinities is a NaN"
            (once (fun b ->
                 equal (array float_exact) [| nan; infinity |]
                   (scatter1 b Add f32 [| infinity; 1. |] [| 0L; 1L |]
                      [| neg_infinity; infinity |])));
          (* 1 and 256 updates of 2^-8 sum to 2; rounded to bfloat16 at each
             step the sum would stay at 1, 1 + 2^-8 being a tie that rounds to
             even. *)
          test "Add rounds a bfloat16 sum once, from float32"
            (once (fun b ->
                 equal (array float_exact) [| 2. |]
                   (scatter1 b Add D.Bfloat16 [| 1. |] (Array.make 256 0L)
                      (Array.make 256 0x1p-8))));
          test
            "Add of booleans, which Prog.Add does not take, is not Done and \
             writes nothing outside dst"
            (once test_bool_add);
          prop ~examples:extreme_examples
            "Max and Min are Prog.Maximum and Minimum of x, u1, …, un, a NaN \
             where one is"
            extremes (run law_extreme);
          prop
            "under ~unique, one update per target gives the result without it"
            unique_scatters (run law_unique);
          prop "into its own into gives the fresh result" scatters
            (run law_in_place);
          prop
            "a call Shape.scatter does not fit is not Done and writes nothing \
             outside dst"
            scatter_misfits (run law_scatter_refuses);
          cases
            "into a destination sharing bytes with an operand it is not is \
             refused as Overlapping"
            ~id:fst scatter_overlaps (fun (_, call) ->
              let b = backend () in
              b.around (fun () -> equal answer A.Overlapping (call b)));
          prop "over a box of the axes off its axis is that box of the whole"
            scatter_boxes (run law_scatter_box);
        ];
    ]

(* rig.nv *)

(* rig.nv loads the cubin CUDA loads, so a gather or a scatter through it
   has CUDA's bits: what it can get wrong, its launches, memory and order,
   shows in them. Each case runs through CUDA, then through rig.nv. A
   scatter that combines computes, so a NaN it gives agrees with any NaN,
   as nx_cuda.mli leaves a NaN result's sign and payload unspecified. A
   tiled add, which these cases do not reach, has its own law through each
   driver (test_cuda_moves). *)
let nv_laws (first : Support.declared) (nv : Support.declared) =
  let moved call =
    Elements.agree_bits (Support.opened first) (Support.opened nv) call
  in
  let agree (c : scatter_case) call =
    if c.combine = Nx_kernel.Set then moved call
    else Elements.agree (Support.opened first) (Support.opened nv) call
  in
  group nv.label
    [
      group "bits are cuda's"
        [
          prop ~examples:gather_examples "of gathers" gathers (fun c ->
              let (D.Any dt) = dtype_of c.x in
              cover "a sub-byte element" (D.bits dt < 8);
              cover "a long axis"
                (Array.exists (fun e -> e >= 100) (shape_of c.x));
              moved (fun b -> [| gathered b ~axis:c.axis c.idx c.x |]));
          prop "of scatters" scatters (fun c ->
              List.iter
                (fun m -> cover (combine_name m) (c.combine = m))
                Nx_kernel.[ Set; Add; Max; Min ];
              cover "shared targets" (fst (sharing c));
              agree c (fun b -> [| scattered b c |]));
          prop "of scatters promised distinct targets" unique_scatters (fun c ->
              agree c (fun b -> [| scattered b ~unique:true c |]));
        ];
    ]

let () =
  let nv =
    match (Support.declared_cuda, Support.declared_nv) with
    | Some first, Some nv -> [ nv_laws first nv ]
    | _ -> []
  in
  exit
    (Windtrap.run "nx_kernel.index"
       (List.map laws
          (Support.declare Support.cpu
          :: List.filter_map Fun.id
               [ Support.declared_cuda; Support.declared_amd ])
       @ nv))
