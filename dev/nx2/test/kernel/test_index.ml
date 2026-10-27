(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Gathers and scatters through every kernel library the host runs: a gather and
   an add-scatter are adjoint, and a scatter into its own operand gives the
   fresh result. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module P = Nx_kernel.Prog
module Support = Nx_kernels_support
open Elements
open Index_cases

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
      let into =
        A.of_array D.Int64 (Iarray.of_array xs) (Array.make (total xs) 0L)
      in
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

(* Set keeps each target's last update in C order: its bits. *)
let law_set_last (b : Support.backend) c =
  let shared, dropped = sharing c in
  cover "shared targets" shared;
  cover "dropped updates" dropped;
  match scatter_on b c with None -> () | Some y -> check_scatter c y

(* A second run gives the same bits, any NaN for a NaN. *)
let law_add_again (b : Support.backend) c =
  cover "shared targets" (fst (sharing c));
  match (scatter_on b c, scatter_on b c) with
  | Some y, Some y' ->
      equal (array (same (dtype_of c.into))) (elements y) (elements y')
  | None, None -> ()
  | _ -> failf "%s declined one of two" b.name

(* Scatters whose updates reach distinct targets: along the axis, each
   slice's positions a shuffled prefix of the axis, the rest outside it. *)
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
  cover "through views" (c.sviews <> []);
  match (scatter_on ~unique:true b c, scatter_on b c) with
  | Some y, Some y' ->
      equal (array (same (dtype_of c.into))) (elements y') (elements y)
  | None, None -> ()
  | _ -> failf "%s declined a scatter one way" b.name

(* Bounds of Add *)

(* [x + y] and its rounding error. *)
let two_sum x y =
  let s = x +. y in
  let z = s -. x in
  (s, x -. (s -. z) +. (y -. z))

(* The sum of [xs] within 2^-100 of the sum of their magnitudes, and that
   sum: the running sum held in two parts. *)
let exact xs =
  let hi = ref 0. and lo = ref 0. and mag = ref 0. in
  List.iter
    (fun x ->
      let s, t = two_sum !hi x in
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
        let s, t = two_sum hi x in
        let lo = lo +. t and mag = mag +. Float.abs x in
        Float.abs (s +. lo) +. ((gamma 0x1p-24 j +. 0x1p-100) *. mag)
        > most_float32
        || go (j + 1) s lo mag rest
  in
  go 1 0. 0. 0. xs

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

(* The parts of an element of [d] as values: one, or a complex number's
   two. *)
let parts (D.Any dt as d) e =
  let w = String.length e in
  if is_narrow d then [ decode d e ]
  else if D.is D.Complex dt then
    let h = w / 2 in
    let get = if h = 4 then get32 else get64 in
    [ get e 0; get e h ]
  else if w = 4 then [ get32 e 0 ]
  else [ get64 e 0 ]

(* Whether a narrow float's sum of [xs], [x] and its updates in C order,
   cannot overflow: no partial float32 sum passes float32's largest finite
   value, nor the sum and its bound the dtype's. *)
let narrow_checked d xs =
  let want, mag = exact xs in
  let bound = (gamma 0x1p-24 (List.length xs) +. 0x1p-100) *. mag in
  not (partial_overflows xs || Float.abs want +. bound > most_finite d)

(* Why a target's Add [got] of [x] and [us] is past
   [γ(n + 1) (|x| + Σ |ui|)] of the exact sum, part by part, [u] the unit
   roundoff of float32 for float32, narrow floats and complex64 and of
   float64 otherwise; a narrow float's sum is rounded once from a float32
   within it, so [got] lies between the roundings of its ends. Each part's
   magnitudes bound the complex number's moduli. A narrow sum that may
   overflow, a partial float32 sum past float32's largest finite value or
   the sum rounding past the dtype's, is not checked: [None]. *)
let past (D.Any dt as d) x us got =
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
             if not (narrow_checked d xs) then []
             else
               let lo = decode d (encode d (want -. bound))
               and hi = decode d (encode d (want +. bound)) in
               if lo <= got && got <= hi then []
               else [ Printf.sprintf "%h is not in [%h, %h]" got lo hi ]
           else if Float.abs (got -. want) <= bound then []
           else
             [ Printf.sprintf "%h is %h from %h, past %h" got (got -. want) want bound ])
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

(* A float Add is within its bound of the exact sum at each target an
   update reaches. *)
let law_add_bound (b : Support.backend) c =
  let d = dtype_of c.into in
  let ls = lands c in
  let reached = Array.exists (fun l -> l <> []) ls in
  cover "a target several updates reach" (Array.exists (fun l -> List.length l > 1) ls);
  cover "a narrow float" (is_narrow d);
  (let xs = elements c.into in
   cover "a checked bfloat16 sum of magnitudes past float32's largest"
     (d = D.Any D.Bfloat16
     && Array.exists Fun.id
          (Array.mapi
             (fun t us ->
               let ts = List.map (decode d) (xs.(t) :: us) in
               us <> [] && snd (exact ts) > most_float32 && narrow_checked d ts)
             ls)));
  match scatter_on b c with
  | None -> ()
  | Some y ->
      cover "reached" reached;
      let got = elements y and xs = elements c.into in
      cover "a wide sum that rounds"
        (not (is_narrow d)
        && Array.exists Fun.id
             (Array.mapi
                (fun t us -> us <> [] && rounds d xs.(t) us got.(t))
                ls));
      let wrong =
        List.filter_map Fun.id
          (Array.to_list
             (Array.mapi
                (fun t us ->
                  if us = [] then None
                  else
                    Option.map (Printf.sprintf "%d: %s" t)
                      (past d xs.(t) us got.(t)))
                ls))
      in
      equal (list string) [] (List.filteri (fun i _ -> i < 5) wrong)

(* Constructed cases, one per covered regime, so that every seed reaches
   each, as index_cases.ml's scatter_example makes them. *)

(* Set: shared targets and positions past both ends, every update on one
   target, no update, and one row. *)
let set_examples =
  let ex d = scatter_example d Nx_kernel.Set in
  [
    ex (D.Any D.Float32) ~ts:[| 4 |] ~axis:0 ~m:6 [| 1L; 3L; 1L; -1L; 4L; 1L |];
    ex (D.Any D.Int8) ~ts:[| 3; 2 |] ~axis:1 ~m:5 [| 0L |];
    ex (D.Any D.Bfloat16) ~ts:[| 3 |] ~axis:0 ~m:0 [| 0L |];
    ex (D.Any D.Complex128) ~ts:[| 5; 4 |] ~axis:0 ~m:1 ~broadcast:[ 1 ] [| 3L |];
  ]

(* A bfloat16 Add into +0 of 40 updates alternating 1.5 · 2^127 and its
   negation: their magnitudes sum past float32's largest finite value, and
   their partial sums stay at or below 1.5 · 2^127. *)
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

(* Add's bound: many updates on few targets, an embedding gradient's rows,
   every update on one target, no update, and magnitudes past float32's
   largest. *)
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
  ]

(* Values the contract states: [run c dt into idx u] is the scatter's
   result into a fresh destination. *)
let scatter1 (b : Support.backend) c dt into idx u =
  let module K = (val b.kernels) in
  let n = Array.length into and m = Array.length idx in
  let dst = A.create Rig.host dt [| n |] in
  match
    K.scatter c ~unique:false ~axis:0 ~dst
      ~into:(A.of_array dt [| n |] into)
      (A.of_array D.Int64 [| m |] idx)
      (A.of_array dt [| m |] u)
  with
  | A.Done -> Some (A.to_array dst)
  | A.Declined -> None
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

let stated (b : Support.backend) =
  let f32 = D.Float32 in
  let snan = Int32.float_of_bits 0x7FA00001l in
  let floats name c dt into idx u want =
    ( name,
      fun () ->
        match scatter1 b c dt into idx u with
        | None -> ()
        | Some y -> equal (array float_exact) want y )
  in
  let bits32 a = Array.map Int32.bits_of_float a in
  [
    floats "Set keeps a target's last update" Set f32 [| 10.; 20.; 30.; 40. |]
      [| 1L; 3L; 1L; 1L |] [| 1.; 2.; 3.; 4. |] [| 10.; 4.; 30.; 2. |];
    floats "Set drops positions outside the axis" Set f32 [| 10.; 20.; 30.; 40. |]
      [| -1L; 4L; Int64.min_int; Int64.max_int; 0L |]
      [| 1.; 2.; 3.; 4.; 5. |] [| 5.; 20.; 30.; 40. |];
    floats "Add starts from the target: -0 and -0 updates sum to -0" Add f32
      [| -0. |] [| 0L; 0L |] [| -0.; -0. |] [| -0. |];
    floats "Add of +0 and a -0 update is +0" Add f32 [| 0. |] [| 0L |] [| -0. |]
      [| 0. |];
    floats "Add of opposite infinities is a NaN" Add f32 [| infinity; 1. |]
      [| 0L; 1L |] [| neg_infinity; infinity |] [| nan; infinity |];
    floats "Add rounds a bfloat16 sum once, from float32" Add D.Bfloat16
      [| 1. |] (Array.make 256 0L) (Array.make 256 0x1p-8) [| 2. |];
    ( "a target no update reaches keeps its bits, -0 and a signalling NaN's",
      fun () ->
        match
          scatter1 b Add f32 [| -0.; snan |] [| 2L; -1L |] [| 1.; 1. |]
        with
        | None -> ()
        | Some y ->
            equal (array int32) (bits32 [| -0.; snan |]) (bits32 y) );
    ( "Add of int8 wraps",
      fun () ->
        match scatter1 b Add D.Int8 [| 127; -128 |] [| 0L; 1L |] [| 1; -1 |] with
        | None -> ()
        | Some y -> equal (array int) [| -128; 127 |] y );
  ]

(* Draws *)

let sets =
  Gen.with_pp pp_scatter (scatter_gen ~combines:Nx_kernel.[ Set ] ())

let finite_adds =
  Gen.with_pp pp_scatter
    (scatter_gen ~dtypes:float_dtypes ~combines:Nx_kernel.[ Add ]
       ~draw:finite ())

(* Sub-byte scatters *)

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
      prop ~examples:set_examples
        "a Set scatter keeps each target's last update in C order" sets
        (run (law_set_last b));
      prop "a float Add gives the same result again" float_adds
        (run (law_add_again b));
      prop ~examples:bound_examples
        "a float Add is within γ(n + 1) (|x| + Σ|ui|) of the exact sum"
        finite_adds
        (run (law_add_bound b));
      cases ~name:fst "a scatter's stated values" (stated b) (fun (_, f) ->
          b.around f);
      prop "a scatter promised distinct targets gives the bits of one not"
        unique_scatters (run (law_unique b));
    ]

let () =
  exit (Windtrap.run "nx_kernel.index" (List.map laws Support.backends))
