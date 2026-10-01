(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Reductions, scans and the normalisations built on them, against folds of the
   reference over the same axes. *)

open Windtrap
open Nx_test

let pp_int32 ppf v = Format.fprintf ppf "%ld" v
let pp_float ppf x = Format.fprintf ppf "%.17g" x
let ranked = Gen.array ~size:(Gen.int_range 1 3) (Gen.int_range 0 4)
let nonempty = Gen.array ~size:(Gen.int_range 1 3) (Gen.int_range 1 4)

let small_int32 =
  Gen.frequency
    [
      (6, Gen.map Int32.of_int (Gen.int_range (-3) 3));
      (1, Gen.of_list ~pp:pp_int32 [ Int32.min_int; Int32.max_int ]);
    ]

(* Few distinct values, so that ties are common, and NaN and both zeros. *)
let tied_float =
  Gen.frequency
    [
      (6, Gen.map float_of_int (Gen.int_range (-2) 2));
      (1, Gen.of_list ~pp:pp_float [ Float.nan; -0.; infinity; neg_infinity ]);
    ]

(* A tensor and the axes a reduction takes, some counted from the end, or none
   for every axis. *)
let with_axes tensors =
  let open Gen in
  let* t = tensors in
  let n = Nx.ndim t in
  let* picked = array ~size:(constant n) (pair bool bool) in
  let+ every = frequency [ (1, constant true); (4, constant false) ]
  and+ keepdims = bool in
  let axes =
    List.concat
      (List.mapi
         (fun d (take, from_end) ->
           if take then [ (if from_end then d - n else d) ] else [])
         (Array.to_list picked))
  in
  (t, (if every then None else Some axes), keepdims)

(* A tensor and one of its axes, or none for the flattened tensor. *)
let with_axis tensors =
  let open Gen in
  let* t = tensors in
  let+ axis =
    frequency
      [ (1, constant None); (4, map Option.some (int_range 0 (Nx.ndim t - 1))) ]
  in
  (t, axis)

let int32s shape = viewed ~shape ~pp:pp_int32 Nx.int32 small_int32
let floats shape = viewed ~shape ~pp:pp_float Nx.float64 tied_float
let ints = Ref.witness int32
let positions = Ref.witness int64

(* Whether [axes] of [t] (every axis for [None]) include an empty one. *)
let empty_axis t axes =
  let s = Nx.shape t and n = Nx.ndim t in
  let axes = match axes with None -> List.init n Fun.id | Some l -> l in
  List.exists (fun a -> s.((a + n) mod n) = 0) axes

let flat r = Ref.reshape [| Ref.numel r.Ref.shape |] r

(* Integer reductions are exact, so they equal the reference's fold. *)
let integer_reductions =
  let reduction name nx f init ~empty =
    prop (name ^ " folds its axes")
      (with_axes (int32s ranked))
      (fun (t, axes, keepdims) ->
        if (not empty) && empty_axis t axes then
          raises_invalid_arg (fun () -> nx ?axes ~keepdims t)
        else
          equal ints
            (Ref.reduce ?axes ~keepdims f init (Ref.of_nx t))
            (Ref.of_nx (nx ?axes ~keepdims t)))
  in
  let truth name nx f init =
    prop (name ^ " tests its axes")
      (with_axes (int32s ranked))
      (fun (t, axes, keepdims) ->
        let r = Ref.map (fun v -> v <> 0l) (Ref.of_nx t) in
        equal (Ref.witness bool)
          (Ref.reduce ?axes ~keepdims f init r)
          (Ref.of_nx (nx ?axes ~keepdims t)))
  in
  group "integer reductions"
    [
      reduction "sum, zero over an empty axis,"
        (fun ?axes ~keepdims t -> Nx.sum ?axes ~keepdims t)
        Int32.add 0l ~empty:true;
      reduction "prod, one over an empty axis,"
        (fun ?axes ~keepdims t -> Nx.prod ?axes ~keepdims t)
        Int32.mul 1l ~empty:true;
      reduction "max, refusing an empty axis (nx.mli is silent),"
        (fun ?axes ~keepdims t -> Nx.max ?axes ~keepdims t)
        max Int32.min_int ~empty:false;
      reduction "min, refusing an empty axis (nx.mli is silent),"
        (fun ?axes ~keepdims t -> Nx.min ?axes ~keepdims t)
        min Int32.max_int ~empty:false;
      truth "any"
        (fun ?axes ~keepdims t -> Nx.any ?axes ~keepdims t)
        ( || ) false;
      truth "all"
        (fun ?axes ~keepdims t -> Nx.all ?axes ~keepdims t)
        ( && ) true;
    ]

(* A float sum rounds in an unspecified association: each output is within [n
   eps] of the sum of its terms' magnitudes of the exact sum. *)
let rounding r = Float.of_int (Ref.numel r.Ref.shape + 1) *. epsilon_float

let float_reductions =
  let values =
    viewed ~shape:ranked ~pp:pp_float Nx.float64 (Gen.float_range (-100.) 100.)
  in
  let nanmax a x =
    if Float.is_nan a || Float.is_nan x then Float.nan else Float.max a x
  in
  let nanmin a x =
    if Float.is_nan a || Float.is_nan x then Float.nan else Float.min a x
  in
  group "float reductions"
    [
      prop "sum is within rounding of the exact sum" (with_axes values)
        (fun (t, axes, keepdims) ->
          let r = Ref.of_nx t in
          let exact = Ref.reduce ?axes ~keepdims ( +. ) 0. r in
          let magnitude =
            Ref.reduce ?axes ~keepdims (fun a x -> a +. Float.abs x) 0. r
          in
          let actual = Ref.of_nx (Nx.sum ?axes ~keepdims t) in
          Array.iteri
            (fun i s ->
              let bound = 2. *. rounding r *. magnitude.data.(i) in
              equal (close ~abs:bound ~rel:0. ()) s actual.data.(i))
            exact.data);
      prop "mean is the sum over the count" (with_axes values)
        (fun (t, axes, keepdims) ->
          let count =
            Nx.numel t / Int.max 1 (Nx.numel (Nx.sum ?axes ~keepdims t))
          in
          equal
            (tensor (close ~rel:1e-12 ~abs:1e-12 ()))
            (Nx.div_s (Nx.sum ?axes ~keepdims t) (Float.of_int count))
            (Nx.mean ?axes ~keepdims t));
      prop "max and min propagate NaN and order -0 below +0"
        (with_axes (floats nonempty))
        (fun (t, axes, keepdims) ->
          assume (not (empty_axis t axes));
          let r = Ref.of_nx t in
          let exact = Ref.witness float_exact in
          equal exact
            (Ref.reduce ?axes ~keepdims nanmax neg_infinity r)
            (Ref.of_nx (Nx.max ?axes ~keepdims t));
          equal exact
            (Ref.reduce ?axes ~keepdims nanmin infinity r)
            (Ref.of_nx (Nx.min ?axes ~keepdims t)));
      prop "var is the mean squared deviation over n - ddof, and std its root"
        (Gen.pair (with_axes values) (Gen.int_range 0 1))
        (fun ((t, axes, keepdims), ddof) ->
          let r = Ref.of_nx t in
          let n = Nx.numel t / Int.max 1 (Nx.numel (Nx.sum ?axes t)) in
          assume (n > ddof);
          let mean =
            Ref.map
              (fun s -> s /. Float.of_int n)
              (Ref.reduce ?axes ~keepdims:true ( +. ) 0. r)
          in
          let dev = Ref.map2 (fun x m -> (x -. m) *. (x -. m)) r mean in
          let var =
            Ref.map
              (fun s -> s /. Float.of_int (n - ddof))
              (Ref.reduce ?axes ~keepdims ( +. ) 0. dev)
          in
          let near = Ref.witness (close ~rel:1e-9 ~abs:1e-9 ()) in
          equal near var (Ref.of_nx (Nx.var ?axes ~keepdims ~ddof t));
          equal near (Ref.map Float.sqrt var)
            (Ref.of_nx (Nx.std ?axes ~keepdims ~ddof t)));
      test "var refuses ddof at least the count" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.var ~ddof:3 (Nx.zeros Nx.float64 [| 3 |])));
      test
        "the mean of nothing is NaN, and an integer one refuses (nx.mli is \
         silent)" (fun () ->
          equal (close ~rel:0. ()) Float.nan
            (Nx.item [] (Nx.mean (Nx.zeros Nx.float64 [| 0 |])));
          raises_invalid_arg (fun () -> Nx.mean (Nx.zeros Nx.int32 [| 0 |])));
    ]

(* Whether [x] is greater than [y] in the order of IEEE maximum, where -0 is
   less than +0. *)
let above x y = x > y || (x = y && Float.sign_bit y && not (Float.sign_bit x))

(* The index of the first extreme, a NaN counting as the extreme. *)
let first_extreme better lane =
  let best = ref 0 in
  Array.iteri
    (fun i x ->
      let b = lane.(!best) in
      if (not (Float.is_nan b)) && (Float.is_nan x || better x b) then best := i)
    lane;
  [| Int64.of_int !best |]

let arg_reductions =
  let check name nx better =
    prop
      (name
     ^ " is the index of the first extreme, NaN first of all, of the flattened \
        tensor without an axis")
      (Gen.pair (with_axis (floats nonempty)) Gen.bool)
      (fun ((t, axis), keepdims) ->
        let r = Ref.of_nx t in
        if empty_axis t (Option.map (fun a -> [ a ]) axis) then
          raises_invalid_arg (fun () -> nx ?axis ~keepdims t)
        else
          let r, a = match axis with None -> (flat r, 0) | Some a -> (r, a) in
          let kept = Ref.along ~axis:a ~length:1 (first_extreme better) r in
          let expected =
            if keepdims && axis <> None then kept
            else if axis = None && keepdims then kept
            else Ref.squeeze ~axes:[ a ] kept
          in
          equal positions expected (Ref.of_nx (nx ?axis ~keepdims t)))
  in
  group "argmax and argmin"
    [
      check "argmax"
        (fun ?axis ~keepdims t -> Nx.argmax ?axis ~keepdims t)
        above;
      check "argmin"
        (fun ?axis ~keepdims t -> Nx.argmin ?axis ~keepdims t)
        (fun x y -> above y x);
      test "argmax refuses an axis out of bounds" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.argmax ~axis:2 (Nx.zeros Nx.float64 [| 2; 2 |])));
      test "argmax and argmin refuse an empty axis, or an empty tensor"
        (fun () ->
          let t = Nx.zeros Nx.float64 [| 2; 0 |] in
          raises_invalid_arg (fun () -> Nx.argmax ~axis:1 t);
          raises_invalid_arg (fun () -> Nx.argmin ~axis:1 t);
          raises_invalid_arg (fun () -> Nx.argmax t));
    ]

let scans =
  let scan f lane =
    let acc = ref None in
    Array.map
      (fun x ->
        let v = match !acc with None -> x | Some a -> f a x in
        acc := Some v;
        v)
      lane
  in
  let exact name nx f witness tensors of_nx =
    prop
      (name
     ^ " accumulates along its axis, or the flattened tensor keeping its shape"
      ) (with_axis tensors) (fun (t, axis) ->
        let r = of_nx t in
        let expected =
          match axis with
          | None ->
              let f' = flat r in
              Ref.reshape r.shape
                (Ref.along ~axis:0 ~length:f'.shape.(0) (scan f) f')
          | Some a -> Ref.along ~axis:a ~length:r.shape.(a) (scan f) r
        in
        equal witness expected (Ref.of_nx (nx ?axis t)))
  in
  let nanmax a x =
    if Float.is_nan a || Float.is_nan x then Float.nan else Float.max a x
  in
  let nanmin a x =
    if Float.is_nan a || Float.is_nan x then Float.nan else Float.min a x
  in
  group "scans"
    [
      exact "cumsum"
        (fun ?axis t -> Nx.cumsum ?axis t)
        Int32.add ints (int32s ranked) Ref.of_nx;
      exact "cumprod"
        (fun ?axis t -> Nx.cumprod ?axis t)
        Int32.mul ints (int32s ranked) Ref.of_nx;
      exact "cummax, propagating NaN and ordering -0 below +0,"
        (fun ?axis t -> Nx.cummax ?axis t)
        nanmax (Ref.witness float_exact) (floats ranked) Ref.of_nx;
      exact "cummin, propagating NaN and ordering -0 below +0,"
        (fun ?axis t -> Nx.cummin ?axis t)
        nanmin (Ref.witness float_exact) (floats ranked) Ref.of_nx;
    ]

(* Scans of structures. A scan is checked against the fold of the reference
   along the axis, in order: integer arithmetic is exact, so any association
   gives the fold's value. *)

let running f lane =
  let acc = ref None in
  Array.map
    (fun x ->
      let v = match !acc with None -> x | Some a -> f a x in
      acc := Some v;
      v)
    lane

(* A tensor and one of its axes, counted from either end. *)
let with_some_axis tensors =
  let open Gen in
  let* t = tensors in
  let n = Nx.ndim t in
  let+ a = int_range 0 (n - 1) and+ from_end = bool in
  (t, if from_end then a - n else a)

let matrix_7x2 =
  Gen.map
    (Nx.create Nx.int64 [| 7; 2 |])
    (Gen.array ~size:(Gen.constant 14) Gen.int64)

(* Affine maps [y -> a y + b] composed in order, which does not commute. *)
let compose (a1, b1) (a2, b2) = Nx.(mul a1 a2, add (mul a2 b1) b2)
let affine (a1, b1) (a2, b2) = (Int64.mul a1 a2, Int64.add (Int64.mul a2 b1) b2)

let structure_scans =
  let leaf = Nx.Ptree.tensor in
  group "associative scans"
    [
      prop "the scan of an addition is the running sum along the axis"
        (with_some_axis (int32s ranked))
        (fun (t, axis) ->
          let r = Ref.of_nx t in
          let a = Ref.axis r axis in
          equal ints
            (Ref.along ~axis:a ~length:r.shape.(a) (running Int32.add) r)
            (Ref.of_nx (Nx.associative_scan ~axis leaf Nx.add t)));
      prop "a scan composes in order, each tensor of a structure at its index"
        (Gen.pair matrix_7x2 matrix_7x2) (fun (a, b) ->
          let ra = Ref.of_nx a and rb = Ref.of_nx b in
          let pairs = Ref.map2 (fun x y -> (x, y)) ra rb in
          let expected = Ref.along ~axis:0 ~length:7 (running affine) pairs in
          let sa, sb =
            Nx.associative_scan (Nx.Ptree.pair leaf leaf) compose (a, b)
          in
          equal positions (Ref.map fst expected) (Ref.of_nx sa);
          equal positions (Ref.map snd expected) (Ref.of_nx sb));
      prop "the scan of a prefix is the prefix of the scan, bit for bit"
        (Gen.pair
           (Gen.array ~size:(Gen.int_range 0 40) (Gen.float_range (-1e3) 1e3))
           (Gen.int_range 0 40))
        (fun (xs, m) ->
          let n = Array.length xs in
          let m = Int.min m n in
          let t = Nx.create Nx.float64 [| n |] xs in
          let scan t = Nx.associative_scan leaf Nx.add t in
          equal (tensor float_exact)
            (scan (Nx.slice [ R (0, m) ] t))
            (Nx.slice [ R (0, m) ] (scan t)));
      test "a structure's tensors may have different ranks" (fun () ->
          let v = Nx.create Nx.int32 [| 3 |] [| 1l; 2l; 3l |]
          and m =
            Nx.create Nx.int32 [| 3; 2 |] [| 1l; 10l; 2l; 20l; 3l; 30l |]
          in
          let add (a, b) (c, d) = (Nx.add a c, Nx.add b d) in
          let sv, sm =
            Nx.associative_scan (Nx.Ptree.pair leaf leaf) add (v, m)
          in
          equal (array int32) [| 1l; 3l; 6l |] (Nx.to_array sv);
          equal (array int32) [| 1l; 10l; 3l; 30l; 6l; 60l |] (Nx.to_array sm));
      test "a scan of no element or one is its input" (fun () ->
          let f _ _ = fail "combined" in
          let none = Nx.zeros Nx.float64 [| 0; 3 |]
          and one = Nx.ones Nx.float64 [| 1 |] in
          equal (tensor float_exact) none (Nx.associative_scan leaf f none);
          equal (tensor float_exact) one (Nx.associative_scan leaf f one));
      test "a segmented scan restarts at each segment, as a lift of f"
        (fun () ->
          let start =
            Nx.create Nx.bool [| 6 |]
              [| true; false; false; true; false; true |]
          and v = Nx.create Nx.int32 [| 6 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
          let f (s1, v1) (s2, v2) =
            (Nx.logical_or s1 s2, Nx.where s2 v2 (Nx.add v1 v2))
          in
          let _, sums =
            Nx.associative_scan (Nx.Ptree.pair leaf leaf) f (start, v)
          in
          equal (array int32) [| 1l; 3l; 6l; 4l; 9l; 6l |] (Nx.to_array sums));
      test "tensors of different lengths along the axis are refused" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.associative_scan (Nx.Ptree.pair leaf leaf) compose
                (Nx.zeros Nx.int64 [| 3 |], Nx.zeros Nx.int64 [| 4 |])));
      test "an axis out of a tensor's bounds is refused" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.associative_scan ~axis:1 leaf Nx.add
                (Nx.zeros Nx.int64 [| 3 |])));
    ]

let ewmas =
  let recurrence alpha lane =
    running (fun y x -> ((1. -. alpha) *. y) +. (alpha *. x)) lane
  in
  let alphas = Gen.of_list ~pp:pp_float [ 0.02; 0.25; 0.5; 0.9; 1. ] in
  let values shape =
    viewed ~shape ~pp:pp_float Nx.float64 (Gen.float_range (-10.) 10.)
  in
  let v xs = Nx.create Nx.float64 [| Array.length xs |] xs in
  group "exponentially weighted means"
    [
      prop "ewma follows its recurrence along the axis, or the flattened tensor"
        (Gen.pair (with_axis (values ranked)) alphas)
        (fun ((t, axis), alpha) ->
          let r = Ref.of_nx t in
          let expected =
            match axis with
            | None ->
                let f = flat r in
                Ref.reshape r.shape
                  (Ref.along ~axis:0 ~length:f.shape.(0) (recurrence alpha) f)
            | Some a ->
                Ref.along ~axis:a ~length:r.shape.(a) (recurrence alpha) r
          in
          equal
            (Ref.witness (close ~abs:1e-12 ~rel:1e-12 ()))
            expected
            (Ref.of_nx (Nx.ewma ?axis ~alpha t)));
      test "at alpha 1 the average is the tensor" (fun () ->
          let t = v [| 1.; Float.nan; Float.infinity; -2. |] in
          equal (tensor float_exact) t (Nx.ewma ~alpha:1. t));
      test "a NaN reaches every later element and no earlier one" (fun () ->
          let y =
            Nx.to_array (Nx.ewma ~alpha:0.5 (v [| 2.; 4.; Float.nan; 1.; 3. |]))
          in
          equal (array float_exact)
            [| 2.; 3.; Float.nan; Float.nan; Float.nan |]
            y);
      test "an infinity stays infinite long after its weight underflows"
        (fun () ->
          let x =
            v (Array.init 2048 (fun i -> if i = 0 then Float.infinity else 0.))
          in
          let y = Nx.to_array (Nx.ewma ~alpha:0.9 x) in
          is_true ~msg:"every element is +inf"
            (Array.for_all (fun e -> e = Float.infinity) y));
      prop "the average of a prefix is the prefix of the average, bit for bit"
        (Gen.triple
           (Gen.array ~size:(Gen.int_range 0 40) (Gen.float_range (-1e3) 1e3))
           (Gen.int_range 0 40) alphas)
        (fun (xs, m, alpha) ->
          let m = Int.min m (Array.length xs) in
          let t = v xs in
          equal (tensor float_exact)
            (Nx.ewma ~alpha (Nx.slice [ R (0, m) ] t))
            (Nx.slice [ R (0, m) ] (Nx.ewma ~alpha t)));
      prop "at float16 the average is float32's, rounded once"
        (Gen.pair (values (Gen.constant [| 9 |])) alphas)
        (fun (t, alpha) ->
          let h = Nx.cast Nx.float16 t in
          equal (tensor float_exact)
            (Nx.cast Nx.float16 (Nx.ewma ~alpha (Nx.cast Nx.float32 h)))
            (Nx.ewma ~alpha h));
      test "an alpha outside (0, 1] is refused" (fun () ->
          List.iter
            (fun alpha ->
              raises_invalid_arg (fun () -> Nx.ewma ~alpha (v [| 1. |])))
            [ 0.; -0.5; 1.5; Float.nan ]);
    ]

(* Histograms, against a count of each point into the cells whose bins hold it,
   over ascending edges and points among them, on them, outside them and NaN. *)

let ascending n =
  let open Gen in
  let+ steps = array ~size:(constant n) (float_range 0.25 2.)
  and+ lo = float_range (-3.) 0. in
  Array.of_list
    (List.rev
       (snd
          (Array.fold_left
             (fun (x, acc) d -> (x +. d, x :: acc))
             (lo, []) steps)))

(* The bin of [x] among [edges], or [None]. *)
let bin edges x =
  let k = Array.length edges - 1 in
  let rec find j =
    if j = k then None
    else if edges.(j) <= x && (x < edges.(j + 1) || (j = k - 1 && x = edges.(k)))
    then Some j
    else find (j + 1)
  in
  find 0

let coordinate edges =
  Gen.frequency
    [
      ( 4,
        Gen.float_range (edges.(0) -. 1.) (edges.(Array.length edges - 1) +. 1.)
      );
      (2, Gen.of_list ~pp:pp_float (Array.to_list edges));
      (1, Gen.of_list ~pp:pp_float [ Float.nan; -0.; 0.; Float.infinity ]);
    ]

let histograms =
  let v xs = Nx.create Nx.float64 [| Array.length xs |] xs in
  let points =
    let open Gen in
    let* ka = int_range 1 4 in
    let* kb = int_range 1 3 in
    let* n = int_range 0 30 in
    let* ea = ascending (ka + 1) in
    let* eb = ascending (kb + 1) in
    let+ xa = array ~size:(constant n) (coordinate ea)
    and+ xb = array ~size:(constant n) (coordinate eb)
    and+ w = array ~size:(constant n) (float_range (-2.) 2.) in
    (ea, eb, xa, xb, w)
  in
  let pp ppf (ea, eb, xa, xb, _) =
    let floats ppf a =
      Format.fprintf ppf "[%s]"
        (String.concat "; " (Array.to_list (Array.map (Printf.sprintf "%h") a)))
    in
    Format.fprintf ppf "edges %a and %a, points %a and %a" floats ea floats eb
      floats xa floats xb
  in
  let points = Gen.with_pp pp points in
  let reference ea eb xa xb w =
    let ka = Array.length ea - 1 and kb = Array.length eb - 1 in
    let cells = Array.make (ka * kb) 0. in
    Array.iteri
      (fun i x ->
        match (bin ea x, bin eb xb.(i)) with
        | Some a, Some b -> cells.((a * kb) + b) <- cells.((a * kb) + b) +. w i
        | _ -> ())
      xa;
    Ref.create [| ka; kb |] cells
  in
  group "histograms"
    [
      prop "a point counts in the cell whose bins hold it" points
        (fun (ea, eb, xa, xb, _) ->
          equal (Ref.witness float_exact)
            (reference ea eb xa xb (fun _ -> 1.))
            (Ref.of_nx (Nx.histogram [ (v ea, v xa); (v eb, v xb) ])));
      prop "with weights a cell sums its points' weights" points
        (fun (ea, eb, xa, xb, w) ->
          equal
            (Ref.witness (close ~abs:1e-12 ~rel:1e-12 ()))
            (reference ea eb xa xb (fun i -> w.(i)))
            (Ref.of_nx
               (Nx.histogram ~weights:(v w) [ (v ea, v xa); (v eb, v xb) ])));
      test "a point on the last edge is in the last bin, past it in none"
        (fun () ->
          equal (array float_exact) [| 0.; 2. |]
            (Nx.to_array
               (Nx.histogram [ (v [| 0.; 1.; 2. |], v [| 2.; 2.; 2.5 |]) ])));
      test "coordinates of any shape are points, -0 at the first edge"
        (fun () ->
          let x = Nx.create Nx.float32 [| 2; 2 |] [| -0.; 0.5; 1.; 5. |] in
          equal (array float_exact) [| 2.; 1. |]
            (Nx.to_array
               (Nx.histogram
                  [ (Nx.create Nx.float32 [| 3 |] [| 0.; 1.; 2. |], x) ])));
      test "float16 counts and weights accumulate wider than they store"
        (fun () ->
          let e = Nx.create Nx.float16 [| 2 |] [| 0.; 1. |] in
          let x = Nx.full Nx.float16 [| 4096 |] 0.5 in
          equal (array float_exact) [| 4096. |]
            (Nx.to_array (Nx.histogram [ (e, x) ]));
          equal (array float_exact) [| 4096. |]
            (Nx.to_array
               (Nx.histogram
                  ~weights:(Nx.ones Nx.float16 [| 4096 |])
                  [ (e, x) ])));
      test "no dimension, short edges and mismatched shapes are refused"
        (fun () ->
          let e = v [| 0.; 1. |] and x = v [| 0.5 |] in
          raises_invalid_arg (fun () -> Nx.histogram []);
          raises_invalid_arg (fun () -> Nx.histogram [ (v [| 0. |], x) ]);
          raises_invalid_arg (fun () ->
              Nx.histogram [ (Nx.zeros Nx.float64 [| 2; 2 |], x) ]);
          raises_invalid_arg (fun () ->
              Nx.histogram [ (e, x); (e, v [| 0.; 1. |]) ]);
          raises_invalid_arg (fun () ->
              Nx.histogram ~weights:(v [| 1.; 1. |]) [ (e, x) ]));
    ]

(* A float dtype, to run a case at each width. *)
type float_dtype = F : string * (float, 'b) Nx.dtype -> float_dtype

let float_dtypes =
  [
    F ("float16", Nx.float16);
    F ("bfloat16", Nx.bfloat16);
    F ("float32", Nx.float32);
    F ("float64", Nx.float64);
  ]

(* The 2 x 3 matrix of [xs] in three layouts: contiguous, held transposed, and
   every other column of a wider matrix. Each takes another path through the
   reductions: along the contiguous axis, across it, and strided. *)
let layouts dt xs =
  let n = Array.length xs / 2 in
  let wide =
    Nx.create dt [| 2; 2 * n |]
      (Array.init (4 * n) (fun i -> if i mod 2 = 0 then xs.(i / 2) else 7.))
  in
  [
    ("contiguous", Nx.create dt [| 2; n |] xs);
    ( "held transposed",
      Nx.transpose
        (Nx.create dt [| n; 2 |]
           (Array.init (2 * n) (fun i -> xs.((i mod 2 * n) + (i / 2))))) );
    ( "every other column",
      Nx.squeeze ~axes:[ -1 ] (Nx.sliding_window ~axis:1 ~window:1 ~step:2 wide)
    );
  ]

(* Both signs of zero meet in every row and every column. *)
let mixed_zeros = [| -0.; 0.; -0.; 0.; -0.; 0. |]

(* A check of a matrix at any float dtype, named after its dtype and layout. *)
type check = { run : 'b. string -> (float, 'b) Nx.t -> unit }

(* [on_every_path xs c] runs [c] on the matrix of [xs] at every float dtype
   and in every layout. *)
let on_every_path xs c =
  List.iter
    (fun (F (name, dt)) ->
      List.iter
        (fun (layout, t) -> c.run (name ^ ", " ^ layout) t)
        (layouts dt xs))
    float_dtypes

let signed_zeros =
  group "signed zeros"
    [
      test "max is +0 and min -0 where both zeros meet, on every path"
        (fun () ->
          on_every_path mixed_zeros
            {
              run =
                (fun msg t ->
                  List.iter
                    (fun (axes, shape) ->
                      let filled v = Nx.full (Nx.dtype t) shape v in
                      equal ~msg:(msg ^ ", max") (tensor float_exact)
                        (filled 0.) (Nx.max ?axes t);
                      equal ~msg:(msg ^ ", min") (tensor float_exact)
                        (filled (-0.)) (Nx.min ?axes t))
                    [
                      (None, [||]);
                      (Some [ 0 ], [| 3 |]);
                      (Some [ 1 ], [| 2 |]);
                    ]);
            });
      test "argmax and argmin point at the zero max and min return" (fun () ->
          let ints xs = Nx.create Nx.int64 [| Array.length xs |] xs in
          on_every_path mixed_zeros
            {
              run =
                (fun msg t ->
                  let equal what =
                    equal ~msg:(msg ^ ", " ^ what) (tensor int64)
                  in
                  equal "argmax along rows" (ints [| 1L; 0L |])
                    (Nx.argmax ~axis:1 t);
                  equal "argmin along rows" (ints [| 0L; 1L |])
                    (Nx.argmin ~axis:1 t);
                  equal "argmax along columns" (ints [| 1L; 0L; 1L |])
                    (Nx.argmax ~axis:0 t);
                  equal "argmin along columns" (ints [| 0L; 1L; 0L |])
                    (Nx.argmin ~axis:0 t));
            });
      test "cummax and cummin turn to the extreme zero" (fun () ->
          List.iter
            (fun (F (name, dt)) ->
              let v xs = Nx.create dt [| Array.length xs |] xs in
              equal ~msg:(name ^ ", cummax") (tensor float_exact)
                (v [| -0.; 0.; 0. |])
                (Nx.cummax (v [| -0.; 0.; -0. |]));
              equal ~msg:(name ^ ", cummin") (tensor float_exact)
                (v [| 0.; -0.; -0. |])
                (Nx.cummin (v [| 0.; -0.; 0. |])))
            float_dtypes);
      test "a sum of zeros is +0 on every path, whatever their signs"
        (fun () ->
          List.iter
            (fun xs ->
              on_every_path xs
                {
                  run =
                    (fun msg t ->
                      List.iter
                        (fun (axes, shape) ->
                          let zeros = Nx.zeros (Nx.dtype t) shape in
                          equal ~msg:(msg ^ ", sum") (tensor float_exact) zeros
                            (Nx.sum ?axes t);
                          equal ~msg:(msg ^ ", mean") (tensor float_exact)
                            zeros (Nx.mean ?axes t))
                        [
                          (None, [||]);
                          (Some [ 0 ], [| 3 |]);
                          (Some [ 1 ], [| 2 |]);
                        ]);
                })
            [ Array.make 6 (-0.); mixed_zeros ]);
      test "a sum of nothing is +0" (fun () ->
          let nothing = Nx.zeros Nx.float32 [| 0; 1 |] in
          equal float_exact 0. (Nx.item [ 0 ] (Nx.sum ~axes:[ 0 ] nothing)));
      test "cumsum runs from +0, its first element included" (fun () ->
          on_every_path (Array.make 6 (-0.))
            {
              run =
                (fun msg t ->
                  let zeros = Nx.zeros (Nx.dtype t) (Nx.shape t) in
                  equal ~msg:(msg ^ ", along columns") (tensor float_exact)
                    zeros (Nx.cumsum ~axis:0 t);
                  equal ~msg:(msg ^ ", along rows") (tensor float_exact) zeros
                    (Nx.cumsum ~axis:1 t));
            });
      test "NaN wins over both zeros" (fun () ->
          let v xs = Nx.create Nx.float32 [| Array.length xs |] xs in
          let t = v [| -0.; Float.nan; 0. |] in
          equal ~msg:"max" float_exact Float.nan (Nx.item [] (Nx.max t));
          equal ~msg:"min" float_exact Float.nan (Nx.item [] (Nx.min t));
          equal ~msg:"argmax" int64 1L (Nx.item [] (Nx.argmax t));
          equal ~msg:"argmin" int64 1L (Nx.item [] (Nx.argmin t)));
    ]

(* The normalisations, at float64 on finite values. *)
let normalisations =
  (* An empty lane has no maximum to shift by; nx refuses it, and nx.mli is
     silent, so the lanes drawn here have elements. *)
  let values =
    Gen.such_that
      (fun t -> Nx.numel t > 0)
      (viewed ~shape:nonempty ~pp:pp_float Nx.float64
         (Gen.float_range (-30.) 30.))
  in
  let near = Ref.witness (close ~rel:1e-12 ~abs:1e-13 ()) in
  let lse lane =
    let m = Array.fold_left Float.max neg_infinity lane in
    m
    +. Float.log (Array.fold_left (fun s x -> s +. Float.exp (x -. m)) 0. lane)
  in
  let last t = Nx.ndim t - 1 in
  group "normalisations"
    [
      prop "softmax is exp (scale (x - max)) over its sum, along the last axis"
        (Gen.pair values (Gen.float_range 0.1 3.))
        (fun (t, scale) ->
          let softmax lane =
            let m = Array.fold_left Float.max neg_infinity lane in
            let e = Array.map (fun x -> Float.exp (scale *. (x -. m))) lane in
            let s = Array.fold_left ( +. ) 0. e in
            Array.map (fun x -> x /. s) e
          in
          let r = Ref.of_nx t in
          equal near
            (Ref.along ~axis:(last t) ~length:r.shape.(last t) softmax r)
            (Ref.of_nx (Nx.softmax ~scale t)));
      prop "logsumexp and logmeanexp reduce the last axis" values (fun t ->
          let r = Ref.of_nx t in
          let a = last t and n = Float.of_int r.shape.(last t) in
          let reduced f =
            Ref.squeeze ~axes:[ a ]
              (Ref.along ~axis:a ~length:1 (fun l -> [| f l |]) r)
          in
          equal near (reduced lse) (Ref.of_nx (Nx.logsumexp ~axes:[ a ] t));
          equal near
            (reduced (fun l -> lse l -. Float.log n))
            (Ref.of_nx (Nx.logmeanexp ~axes:[ a ] t)));
      prop "log_softmax is x minus logsumexp along the last axis" values
        (fun t ->
          let r = Ref.of_nx t in
          let a = last t in
          let lsm lane =
            let z = lse lane in
            Array.map (fun x -> x -. z) lane
          in
          equal near
            (Ref.along ~axis:a ~length:r.shape.(a) lsm r)
            (Ref.of_nx (Nx.log_softmax t)));
      prop "standardize is (x - mean) / sqrt (var + epsilon) over all axes"
        values (fun t ->
          let r = Ref.of_nx t in
          let n = Float.of_int (Nx.numel t) in
          let mean = Array.fold_left ( +. ) 0. r.data /. n in
          let var =
            Array.fold_left
              (fun s x -> s +. ((x -. mean) *. (x -. mean)))
              0. r.data
            /. n
          in
          equal
            (Ref.witness (close ~rel:1e-9 ~abs:1e-9 ()))
            (Ref.map (fun x -> (x -. mean) /. Float.sqrt (var +. 1e-5)) r)
            (Ref.of_nx (Nx.standardize t)));
    ]

(* Every integer dtype sums at its width and orders as it is signed. *)
let integer_dtypes =
  group "integer dtypes"
    (List.map
       (fun (Int_dtype d) ->
         prop
           (d.name
          ^ " sum wraps at its width, and max, min and argmax follow its order"
           )
           (Gen.array ~size:(Gen.int_range 1 9)
              (int_value ~bits:d.bits ~signed:d.signed))
           (fun xs ->
             let t =
               Nx.create d.dtype [| Array.length xs |] (Array.map d.of_i64 xs)
             in
             let cmp = int_compare ~signed:d.signed in
             let extreme better =
               Array.fold_left
                 (fun (bi, bv) (i, v) ->
                   if better (cmp v bv) then (i, v) else (bi, bv))
                 (0, xs.(0))
                 (Array.mapi (fun i v -> (i, v)) xs)
             in
             let value v = d.of_i64 (wrap ~bits:d.bits ~signed:d.signed v) in
             equal ~msg:"sum" d.exact
               (value (Array.fold_left Int64.add 0L xs))
               (Nx.item [] (Nx.sum t));
             equal ~msg:"max" d.exact
               (value (snd (extreme (fun c -> c > 0))))
               (Nx.item [] (Nx.max t));
             equal ~msg:"min" d.exact
               (value (snd (extreme (fun c -> c < 0))))
               (Nx.item [] (Nx.min t));
             equal ~msg:"argmax" int64
               (Int64.of_int (fst (extreme (fun c -> c > 0))))
               (Nx.item [] (Nx.argmax t))))
       int_dtypes
    @ [
        slow "a float32 sum of 2^25 ones is exact" (fun () ->
            equal float_exact 0x1p25
              (Nx.item [] (Nx.sum (Nx.ones Nx.float32 [| 1 lsl 25 |]))));
      ])

let at_scale =
  group "reductions at scale"
    [
      slow "argmax and argmin of 2^31 + 1 equal entries are 0" (fun () ->
          let long =
            Nx.broadcast_to [| (1 lsl 31) + 1 |] (Nx.scalar Nx.int8 1)
          in
          equal int64 0L (Nx.item [] (Nx.argmax long));
          equal int64 0L (Nx.item [] (Nx.argmin ~axis:0 long)));
      slow
        "sum along the long axis of a matrix of two columns keeps each column"
        (fun () ->
          let rows = 9_000_000 in
          let t =
            Nx.init Nx.float64 [| rows; 2 |] (fun i ->
                float_of_int ((i.(0) mod 7) + i.(1)))
          in
          let column c =
            let s = ref 0. in
            for i = 0 to rows - 1 do
              s := !s +. float_of_int ((i mod 7) + c)
            done;
            !s
          in
          equal (array float_exact)
            [| column 0; column 1 |]
            (Nx.to_array (Nx.sum ~axes:[ 0 ] t)));
    ]

(* Associations: what nx.mli promises of long reductions and scans, at the
   lengths where the host cuts them into blocks and chunks. *)

type wide = W : string * (float, 'b) Nx.dtype -> wide

let wide = [ W ("float32", Nx.float32); W ("float64", Nx.float64) ]

(* Integers in [-8, 8]: every partial sum of a few thousand of them is exact at
   float32 and float64, so every association gives the integer sum. *)
let small_integers n = Array.init n (fun _ -> float_of_int (Random.int 17 - 8))

(* The [r] x [c] matrix of [m] in three layouts: contiguous, held transposed,
   and flipped along its rows. *)
let matrix_layouts dt r c m =
  let at i j = m.((i * c) + j) in
  [
    ("contiguous", Nx.create dt [| r; c |] m);
    ( "held transposed",
      Nx.transpose
        (Nx.create dt [| c; r |]
           (Array.init (r * c) (fun k -> at (k mod r) (k / r)))) );
    ( "flipped",
      Nx.flip ~axes:[ 0 ]
        (Nx.create dt [| r; c |]
           (Array.init (r * c) (fun k -> at (r - 1 - (k / c)) (k mod c)))) );
  ]

let exact_sums =
  cases
    ~name:(fun (r, c) -> Printf.sprintf "%dx%d" r c)
    "sums of small integers are exact on every path"
    [ (1, 3077); (70, 70); (7, 3077); (3077, 7) ]
    (fun (r, c) ->
      let m = small_integers (r * c) in
      let reference = Ref.create [| r; c |] m in
      List.iter
        (fun (W (name, dt)) ->
          List.iter
            (fun (layout, t) ->
              List.iter
                (fun axes ->
                  let msg = Printf.sprintf "%s, %s" name layout in
                  equal ~msg (Ref.witness float_exact)
                    (Ref.reduce ?axes ~keepdims:false ( +. ) 0. reference)
                    (Ref.of_nx (Nx.sum ?axes t)))
                [ None; Some [ 0 ]; Some [ 1 ] ])
            (matrix_layouts dt r c m))
        wide)

let negative_zeros () =
  List.iter
    (fun (W (name, dt)) ->
      let z = Nx.full dt [| 2049; 3 |] (-0.) in
      equal ~msg:name (array float_exact) [| 0. |]
        (Nx.to_array (Nx.reshape [| 1 |] (Nx.sum z)));
      equal ~msg:name (array float_exact) [| 0.; 0.; 0. |]
        (Nx.to_array (Nx.sum ~axes:[ 0 ] z)))
    wide

(* The bits of [rows] lanes of [len] float32 values, row-major, every other lane
   holding NaNs of distinct payloads at the positions [at], the payloads rotated
   from lane to lane. *)
let nan_payloads = [| 0x7fc0_0001l; 0xffc0_0002l; 0x7fc0_0003l |]

let lanes_with_nans rows len ~at =
  let bits =
    Array.init (rows * len) (fun _ ->
        Int32.bits_of_float (Random.float 2. -. 1.))
  in
  for r = 0 to rows - 1 do
    if r mod 2 = 0 then
      Array.iteri
        (fun i p -> bits.((r * len) + p) <- nan_payloads.((i + r) mod 3))
        at
  done;
  bits

(* The float32 lanes of [bits] along [axis] of the result: the rows of a [rows]
   x [len] matrix along axis 1, its columns along axis 0. *)
let float32_lanes ~axis rows len bits =
  let t = Nx.create Nx.int32 [| rows; len |] bits in
  let t = if axis = 1 then t else Nx.contiguous (Nx.transpose t) in
  Nx.bitcast Nx.float32 t

let extreme_is_arg_element () =
  let rows = 4 and len = 2051 in
  (* one NaN in each of three blocks *)
  let bits = lanes_with_nans rows len ~at:[| 17; 1500; 2049 |] in
  List.iter
    (fun axis ->
      let x = float32_lanes ~axis rows len bits in
      List.iter
        (fun (name, extreme, arg) ->
          let at_arg =
            Nx.squeeze ~axes:[ axis ]
              (Nx.take_along_axis ~axis ~indices:(arg ~axis ~keepdims:true x) x)
          in
          equal
            ~msg:(Printf.sprintf "%s along axis %d" name axis)
            (array int32)
            (Nx.to_array (Nx.bitcast Nx.int32 at_arg))
            (Nx.to_array (Nx.bitcast Nx.int32 (extreme ~axes:[ axis ] x))))
        [
          ( "max",
            (fun ~axes x -> Nx.max ~axes x),
            fun ~axis ~keepdims x -> Nx.argmax ~axis ~keepdims x );
          ( "min",
            (fun ~axes x -> Nx.min ~axes x),
            fun ~axis ~keepdims x -> Nx.argmin ~axis ~keepdims x );
        ])
    [ 1; 0 ]

let running_extremes_keep_the_first_nan () =
  let rows = 3 and len = 9000 in
  (* one NaN in each of three chunks *)
  let bits = lanes_with_nans rows len ~at:[| 17; 5000; 8500 |] in
  let x = float32_lanes ~axis:1 rows len bits in
  List.iter
    (fun (name, scan) ->
      let out = Nx.to_array (Nx.bitcast Nx.int32 (scan x)) in
      for r = 0 to rows - 1 do
        let lane = Array.sub bits (r * len) len in
        match
          Array.find_index (fun b -> Float.is_nan (Int32.float_of_bits b)) lane
        with
        | None -> ()
        | Some first ->
            for i = first to len - 1 do
              equal
                ~msg:(Printf.sprintf "%s, lane %d, element %d" name r i)
                int32 lane.(first)
                out.((r * len) + i)
            done
      done)
    [
      ("cummax", fun x -> Nx.cummax ~axis:1 x);
      ("cummin", fun x -> Nx.cummin ~axis:1 x);
    ]

(* The bits of a float tensor's elements, through float64, which holds every
   float32 exactly. *)
let float_bits t =
  Array.map Int64.bits_of_float (Nx.to_array (Nx.cast Nx.float64 t))

let prefix_law =
  cases ~name:(Printf.sprintf "%d elements")
    "the running values of a prefix are the first running values"
    [ 1; 8193; 12299 ] (fun n ->
      let signed_zero () = if Random.bool () then 0. else -0. in
      let sums =
        Array.init n (fun _ ->
            if Random.int 50 = 0 then signed_zero () else Random.float 2. -. 1.)
      in
      let products =
        Array.init n (fun _ ->
            if Random.int 5000 = 0 then signed_zero ()
            else 1. +. Random.float 0.002 -. 0.001)
      in
      (* each side of the second and third chunks' starts *)
      let prefixes =
        List.filter
          (fun k -> k <= n)
          [ 1; 4096; 4097; 8192; 8193; 12288; 12289; n ]
      in
      List.iter
        (fun (W (name, dt)) ->
          List.iter
            (fun (op, scan, values) ->
              let x = Nx.create dt [| n |] values in
              let whole = float_bits (scan x) in
              List.iter
                (fun k ->
                  equal
                    ~msg:(Printf.sprintf "%s of %s, prefix %d" op name k)
                    (array int64) (Array.sub whole 0 k)
                    (float_bits (scan (Nx.shrink [| (0, k) |] x))))
                prefixes)
            [
              ("cumsum", (fun x -> Nx.cumsum x), sums);
              ("cumprod", (fun x -> Nx.cumprod x), products);
            ])
        wide)

let exact_running_sums () =
  let n = 12299 in
  let xs = small_integers n in
  let running = Array.copy xs in
  for i = 1 to n - 1 do
    running.(i) <- running.(i - 1) +. xs.(i)
  done;
  List.iter
    (fun (W (name, dt)) ->
      equal ~msg:name (array float_exact) running
        (Nx.to_array (Nx.cast Nx.float64 (Nx.cumsum (Nx.create dt [| n |] xs)))))
    wide

(* A running sum is a float sum as sum describes, so each is within rounding of
   the exact running sum. Nothing here assumes the running sums of non-negative
   terms never decrease: rounding can break that. *)
let running_sums_round () =
  let n = 12299 in
  let xs = Array.init n (fun _ -> Random.float 1.) in
  let actual =
    Nx.to_array
      (Nx.cast Nx.float64 (Nx.cumsum (Nx.create Nx.float32 [| n |] xs)))
  in
  let eps =
    epsilon_float *. 0x1p29
    (* float32's *)
  in
  let exact = ref 0. in
  Array.iteri
    (fun i x ->
      (* the float32 terms, summed in float64: exact well within the bound *)
      exact := !exact +. Int32.float_of_bits (Int32.bits_of_float x);
      let bound = 2. *. Float.of_int (i + 2) *. eps *. !exact in
      equal
        ~msg:(Printf.sprintf "element %d" i)
        (close ~abs:bound ~rel:0. ())
        !exact actual.(i))
    xs

let associations =
  group "associations"
    [
      exact_sums;
      test "a long sum of -0s is +0, across rows and along them" negative_zeros;
      test
        "max and min along an axis are the elements argmax and argmin find, \
         NaN payloads included"
        extreme_is_arg_element;
      test "cummax and cummin carry the first NaN past every chunk"
        running_extremes_keep_the_first_nan;
      prefix_law;
      test "cumsum of small integers is exact across chunks" exact_running_sums;
      test "cumsum of non-negative float32 is within rounding of the exact one"
        running_sums_round;
    ]

(* Segment reductions, against a loop that starts each segment at its identity
   and combines its rows into it in row order. Under Max and Min a row replaces
   the segment's value when it wins: it is the greater (Max) or the lesser
   (Min), -0 below +0 and a NaN beyond every number, and the value is not a NaN.
   Floats compare up to NaN payloads, which "a segment holds its first NaN row's
   bits" pins. *)
type segmented =
  | Seg : {
      name : string;
      dtype : ('a, 'b) Nx.dtype;
      value : 'a Gen.t;
      add : ('a -> 'a -> 'a) option;
      wins : greater:bool -> 'a -> 'a -> bool;
      exact : 'a testable;
      pp : Format.formatter -> 'a -> unit;
    }
      -> segmented

let up_to_nan =
  Testable.contramap
    (fun x -> if Float.is_nan x then None else Some (Int64.bits_of_float x))
    (option int64)

let float_wins ~greater a b =
  (not (Float.is_nan a))
  && (Float.is_nan b
     || (if greater then b > a else b < a)
     || (b = a && Float.sign_bit a = greater && Float.sign_bit b <> greater))

let ordered_wins compare ~greater a b =
  if greater then compare b a > 0 else compare b a < 0

let float_seg name dtype =
  Seg
    {
      name;
      dtype;
      value = tied_float;
      add = Some ( +. );
      wins = float_wins;
      exact = up_to_nan;
      pp = pp_float;
    }

let segmented =
  [
    float_seg "float64" Nx.float64;
    float_seg "float32" Nx.float32;
    float_seg "float16" Nx.float16;
    Seg
      {
        name = "int32";
        dtype = Nx.int32;
        value = small_int32;
        add = Some Int32.add;
        wins = ordered_wins Int32.compare;
        exact = int32;
        pp = pp_int32;
      };
    Seg
      {
        name = "uint8";
        dtype = Nx.uint8;
        value = Gen.int_range 0 255;
        add = Some (fun a b -> (a + b) land 255);
        wins = ordered_wins Int.compare;
        exact = int;
        pp = Format.pp_print_int;
      };
    Seg
      {
        name = "bool";
        dtype = Nx.bool;
        value = Gen.bool;
        add = None;
        wins = ordered_wins Bool.compare;
        exact = bool;
        pp = Format.pp_print_bool;
      };
  ]

let int64s xs = Nx.create Nx.int64 [| Array.length xs |] xs

(* [op]'s step: a sum, or the winner under [wins]. *)
let combining add wins op a b =
  match op with
  | `Add -> Option.get add a b
  | `Max -> if wins ~greater:true a b then b else a
  | `Min -> if wins ~greater:false a b then b else a

let identity dtype = function
  | `Add -> Nx_dtype.zero dtype
  | `Max -> Nx_dtype.min_value dtype
  | `Min -> Nx_dtype.max_value dtype

let op_name = function `Add -> "Add" | `Max -> "Max" | `Min -> "Min"

let segment_reductions =
  let combines (Seg s) op =
    let drawn =
      let open Gen in
      let* segments = int_range 0 4 in
      let* n = int_range 0 8 in
      let* width = option (int_range 0 3) in
      let shape = match width with None -> [| n |] | Some w -> [| n; w |] in
      (* A layout may transpose [x]: its rows are counted once it is drawn. *)
      let* x = viewed ~shape:(constant shape) ~pp:s.pp s.dtype s.value in
      let+ ids =
        array ~size:(constant (Nx.dim 0 x)) (int_range (-1) segments)
      in
      (segments, ids, x)
    in
    prop
      (Printf.sprintf "%s reduce_segments %s combines each segment's rows"
         s.name (op_name op))
      drawn
      (fun (segments, ids, x) ->
        let r = Ref.of_nx x in
        let w = Ref.numel (Array.sub r.shape 1 (Ref.ndim r - 1)) in
        let combine = combining s.add s.wins op in
        let expected = Array.make (segments * w) (identity s.dtype op) in
        Array.iteri
          (fun i id ->
            if id >= 0 && id < segments then
              for j = 0 to w - 1 do
                let at = (id * w) + j in
                expected.(at) <- combine expected.(at) r.data.((i * w) + j)
              done)
          ids;
        let shape = Array.copy r.shape in
        shape.(0) <- segments;
        equal (Ref.witness s.exact)
          (Ref.create shape expected)
          (Ref.of_nx
             (Nx.reduce_segments op ~segments
                (int64s (Array.map Int64.of_int ids))
                x)))
  in
  let props =
    List.concat_map
      (fun (Seg s as seg) ->
        (if Option.is_some s.add then [ combines seg `Add ] else [])
        @ [ combines seg `Max; combines seg `Min ])
      segmented
  in
  let empty dtype op =
    Nx.reduce_segments op ~segments:2 (int64s [||]) (Nx.zeros dtype [| 0 |])
  in
  group "segments"
    (props
    @ [
        test "an empty segment holds the identity" (fun () ->
            let floats dtype lo hi =
              equal (array float_exact) [| lo; lo |]
                (Nx.to_array (empty dtype `Max));
              equal (array float_exact) [| hi; hi |]
                (Nx.to_array (empty dtype `Min));
              equal (array float_exact) [| 0.; 0. |]
                (Nx.to_array (empty dtype `Add))
            in
            floats Nx.float64 neg_infinity infinity;
            floats Nx.float16 neg_infinity infinity;
            floats Nx.float8_e4m3 (-448.) 448.;
            equal (array int) [| -128; -128 |]
              (Nx.to_array (empty Nx.int8 `Max));
            equal (array int) [| 127; 127 |] (Nx.to_array (empty Nx.int8 `Min));
            equal (array int) [| 0; 0 |] (Nx.to_array (empty Nx.uint8 `Max));
            equal (array int) [| 255; 255 |] (Nx.to_array (empty Nx.uint8 `Min));
            equal (array int64) [| -1L; -1L |]
              (Nx.to_array (empty Nx.uint64 `Min));
            equal (array bool) [| false; false |]
              (Nx.to_array (empty Nx.bool `Max));
            equal (array bool) [| true; true |]
              (Nx.to_array (empty Nx.bool `Min)));
        test "a segment holds its first NaN row's bits under Max and Min"
          (fun () ->
            let x =
              Nx.bitcast Nx.float64
                (int64s
                   [|
                     Int64.bits_of_float 1.;
                     0x7ff8000000000005L;
                     0xfff8000000000006L;
                   |])
            in
            List.iter
              (fun op ->
                equal (array int64) [| 0x7ff8000000000005L |]
                  (Nx.to_array
                     (Nx.bitcast Nx.uint64
                        (Nx.reduce_segments op ~segments:1
                           (Nx.zeros Nx.int64 [| 3 |])
                           x))))
              [ `Max; `Min ]);
        test "-0 and +0 in one segment are +0 under Max and -0 under Min"
          (fun () ->
            let x = Nx.create Nx.float32 [| 2 |] [| -0.; 0. |] in
            let ids = Nx.zeros Nx.int64 [| 2 |] in
            equal (array float_exact) [| 0. |]
              (Nx.to_array (Nx.reduce_segments `Max ~segments:1 ids x));
            equal (array float_exact) [| -0. |]
              (Nx.to_array (Nx.reduce_segments `Min ~segments:1 ids x)));
        test "reduce_segments counts, sums and takes first rows" (fun () ->
            let ids = int64s [| 0L; 2L; 0L; -1L; 2L |] in
            let x = Nx.create Nx.float64 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
            equal (array float_exact) [| 4.; 0.; 7. |]
              (Nx.to_array (Nx.reduce_segments `Add ~segments:3 ids x));
            equal (array float_exact) [| 3.; neg_infinity; 5. |]
              (Nx.to_array (Nx.reduce_segments `Max ~segments:3 ids x));
            equal (array int64) [| 2L; 0L; 2L |]
              (Nx.to_array
                 (Nx.reduce_segments `Add ~segments:3 ids
                    (Nx.ones Nx.int64 [| 5 |])));
            let first =
              Nx.reduce_segments `Min ~segments:3 ids (Nx.arange Nx.int64 0 5 1)
            in
            equal (array int64) [| 0L; Int64.max_int; 1L |] (Nx.to_array first);
            equal (array float_exact) [| 1.; 0.; 2. |]
              (Nx.to_array (Nx.take ~indices:first x)));
        test
          "reduce_segments refuses negative segments, a scalar, misshapen ids, \
           complex extremes and boolean sums" (fun () ->
            let ids = Nx.zeros Nx.int64 [| 2 |]
            and x = Nx.zeros Nx.float32 [| 2 |] in
            raises_invalid_arg (fun () ->
                Nx.reduce_segments `Add ~segments:(-1) ids x);
            raises_invalid_arg (fun () ->
                Nx.reduce_segments `Add ~segments:1 ids
                  (Nx.scalar Nx.float32 1.));
            raises_invalid_arg (fun () ->
                Nx.reduce_segments `Add ~segments:1
                  (Nx.zeros Nx.int64 [| 3 |])
                  x);
            raises_invalid_arg (fun () ->
                Nx.reduce_segments `Add ~segments:1
                  (Nx.zeros Nx.int64 [| 2; 1 |])
                  x);
            raises_invalid_arg (fun () ->
                Nx.reduce_segments `Max ~segments:1 ids
                  (Nx.zeros Nx.complex64 [| 2 |]));
            raises_invalid_arg (fun () ->
                Nx.reduce_segments `Add ~segments:1 ids
                  (Nx.zeros Nx.bool [| 2 |])));
      ])

(* Range reductions, against the segments' loop over each range's rows after
   clipping, and against max, min and the running scans where they must agree
   bit for bit. Bounds reach past both ends of the rows, and some ranges are
   empty or inverted. *)

let bounds n =
  let open Gen in
  let bound = int_range (-2) (n + 2) in
  array ~size:(int_range 0 6) (pair bound bound)

(* The ranges [(lo, hi)] as the two bound tensors. *)
let ranges b =
  let bound f = int64s (Array.map (fun p -> Int64.of_int (f p)) b) in
  (bound fst, bound snd)

let reduce op b x =
  let lo, hi = ranges b in
  Nx.reduce_ranges op ~lo ~hi x

(* A nonempty range of [n] rows, [n > 0]. *)
let nonempty_range n =
  let open Gen in
  let* lo = int_range 0 (n - 1) in
  let+ hi = int_range (lo + 1) n in
  (lo, hi)

let range_combines (Seg s) op =
  let drawn =
    let open Gen in
    let* n = int_range 0 9 in
    let* width = option (int_range 0 3) in
    let shape = match width with None -> [| n |] | Some w -> [| n; w |] in
    (* A layout may transpose [x]: its rows are counted once it is drawn. *)
    let* x = viewed ~shape:(constant shape) ~pp:s.pp s.dtype s.value in
    let+ b = bounds (Nx.dim 0 x) in
    (b, x)
  in
  prop
    (Printf.sprintf "%s reduce_ranges %s combines each range's rows" s.name
       (op_name op))
    drawn
    (fun (b, x) ->
      let r = Ref.of_nx x in
      let n = r.shape.(0)
      and w = Ref.numel (Array.sub r.shape 1 (Ref.ndim r - 1)) in
      let combine = combining s.add s.wins op in
      let expected = Array.make (Array.length b * w) (identity s.dtype op) in
      Array.iteri
        (fun i (lo, hi) ->
          for row = Int.max lo 0 to Int.min hi n - 1 do
            for j = 0 to w - 1 do
              let at = (i * w) + j in
              expected.(at) <- combine expected.(at) r.data.((row * w) + j)
            done
          done)
        b;
      let shape = Array.copy r.shape in
      shape.(0) <- Array.length b;
      equal (Ref.witness s.exact)
        (Ref.create shape expected)
        (Ref.of_nx (reduce op b x)))

(* float32 bits of small integers, both zeros and NaNs of distinct payloads. *)
let float32_bits =
  Gen.frequency
    [
      ( 5,
        Gen.map
          (fun i -> Int32.bits_of_float (float_of_int i))
          (Gen.int_range (-3) 3) );
      (1, Gen.of_list ~pp:pp_int32 (Array.to_list nan_payloads));
      (1, Gen.of_list ~pp:pp_int32 [ 0l; Int32.min_int ]);
    ]

let extremes_are_max_and_min =
  prop
    "Max and Min of a range are max and min of its rows, NaN payloads included"
    (let open Gen in
     let* n = int_range 1 70 in
     let* bits = array ~size:(constant n) float32_bits in
     let+ b = array ~size:(int_range 1 8) (nonempty_range n) in
     (bits, b))
    (fun (bits, b) ->
      let x =
        Nx.bitcast Nx.float32 (Nx.create Nx.int32 [| Array.length bits |] bits)
      in
      List.iter
        (fun (op, extreme) ->
          let expected =
            Array.map
              (fun (lo, hi) ->
                Nx.item []
                  (Nx.bitcast Nx.int32 (extreme (Nx.shrink [| (lo, hi) |] x))))
              b
          in
          equal ~msg:(op_name op) (array int32) expected
            (Nx.to_array (Nx.bitcast Nx.int32 (reduce op b x))))
        [ (`Max, fun t -> Nx.max t); (`Min, fun t -> Nx.min t) ])

(* Small integers among terms of 1e300, so that a sum that took in a term
   outside its range would lose the small ones, and fractions, whose sums round
   differently in each association. *)
let huge_small_or_fraction =
  Gen.frequency
    [
      (3, Gen.map float_of_int (Gen.int_range (-3) 3));
      (1, Gen.of_list ~pp:pp_float [ 1e300; -1e300 ]);
      (2, Gen.float_range (-1.) 1.);
    ]

let small v = Float.is_integer v && Float.abs v < 4.

let sums_hold_their_own_terms =
  prop "a range's sum is its terms' alone, bit for bit as if it were alone"
    (let open Gen in
     let* xs = array ~size:(int_range 1 80) huge_small_or_fraction in
     let+ b = array ~size:(int_range 1 8) (nonempty_range (Array.length xs)) in
     (xs, b))
    (fun (xs, b) ->
      let x = Nx.create Nx.float64 [| Array.length xs |] xs in
      let together = float_bits (reduce `Add b x) in
      Array.iteri
        (fun i (lo, hi) ->
          let terms = Array.sub xs lo (hi - lo) in
          if Array.for_all small terms then
            equal ~msg:"small terms" float_exact
              (Array.fold_left ( +. ) 0. terms)
              (Int64.float_of_bits together.(i));
          equal ~msg:"alone" int64
            (float_bits (reduce `Add [| (lo, hi) |] x)).(0)
            together.(i))
        b)

(* Unit roundoff of float32. *)
let u32 = ldexp 1. (-24)

let sums_round_little =
  prop
    "a float32 range sum is within a few roundings per level of the exact sum"
    (let open Gen in
     let* xs = array ~size:(int_range 1 600) (float_range (-1.) 1.) in
     let+ b = array ~size:(int_range 1 8) (nonempty_range (Array.length xs)) in
     (xs, b))
    (fun (xs, b) ->
      let x =
        Nx.cast Nx.float32 (Nx.create Nx.float64 [| Array.length xs |] xs)
      in
      let xs = Nx.to_array (Nx.cast Nx.float64 x) in
      let got = Nx.to_array (Nx.cast Nx.float64 (reduce `Add b x)) in
      Array.iteri
        (fun i (lo, hi) ->
          let terms = Array.sub xs lo (hi - lo) in
          let exact = Array.fold_left ( +. ) 0. terms in
          let mass = Array.fold_left (fun a v -> a +. Float.abs v) 0. terms in
          let levels = Float.ceil (Float.log2 (float_of_int (hi - lo))) in
          at_most
            ~msg:(Printf.sprintf "range %d to %d" lo hi)
            float_exact
            ~than:(((2. *. levels) +. 2.) *. u32 *. mass)
            (Float.abs (got.(i) -. exact)))
        b)

let narrow_sums_round_once =
  prop "a float16 range sum is float32's, rounded once"
    (let open Gen in
     let* xs = array ~size:(int_range 1 60) (float_range (-300.) 300.) in
     let+ b = bounds (Array.length xs) in
     (xs, b))
    (fun (xs, b) ->
      let h =
        Nx.cast Nx.float16 (Nx.create Nx.float64 [| Array.length xs |] xs)
      in
      equal (array int64)
        (float_bits (Nx.cast Nx.float16 (reduce `Add b (Nx.cast Nx.float32 h))))
        (float_bits (reduce `Add b h)))

let running =
  prop "the ranges from row 0 to each row are the running sums and maxima"
    (let open Gen in
     let* n = int_range 1 40 in
     let+ ints = array ~size:(constant n) small_int32
     and+ bits = array ~size:(constant n) float32_bits in
     (ints, bits))
    (fun (ints, bits) ->
      let n = Array.length ints in
      let b = Array.init n (fun i -> (0, i + 1)) in
      let ints = Nx.create Nx.int32 [| n |] ints in
      equal ~msg:"sums" (array int32)
        (Nx.to_array (Nx.cumsum ints))
        (Nx.to_array (reduce `Add b ints));
      let x = Nx.bitcast Nx.float32 (Nx.create Nx.int32 [| n |] bits) in
      equal ~msg:"maxima" (array int32)
        (Nx.to_array (Nx.bitcast Nx.int32 (Nx.cummax x)))
        (Nx.to_array (Nx.bitcast Nx.int32 (reduce `Max b x))))

let range_reductions =
  let props =
    List.concat_map
      (fun (Seg s as seg) ->
        (if Option.is_some s.add then [ range_combines seg `Add ] else [])
        @ [ range_combines seg `Max; range_combines seg `Min ])
      segmented
  in
  group "ranges"
    (props
    @ [
        extremes_are_max_and_min;
        sums_hold_their_own_terms;
        sums_round_little;
        narrow_sums_round_once;
        running;
        test "-0 terms sum to +0, and Max and Min order -0 below +0" (fun () ->
            let x = Nx.create Nx.float32 [| 3 |] [| -0.; -0.; 0. |] in
            let b = [| (0, 1); (0, 2); (1, 3); (3, 3) |] in
            equal ~msg:"Add" (array int64)
              (float_bits (Nx.zeros Nx.float32 [| 4 |]))
              (float_bits (reduce `Add b x));
            equal ~msg:"Max" (array int64)
              (float_bits
                 (Nx.create Nx.float32 [| 4 |] [| -0.; -0.; 0.; neg_infinity |]))
              (float_bits (reduce `Max b x));
            equal ~msg:"Min" (array int64)
              (float_bits
                 (Nx.create Nx.float32 [| 4 |] [| -0.; -0.; -0.; infinity |]))
              (float_bits (reduce `Min b x)));
        test "rows of no rows, and no ranges, have the shapes of their rows"
          (fun () ->
            let x = Nx.zeros Nx.float64 [| 0; 2 |] in
            equal (tensor float_exact)
              (Nx.full Nx.float64 [| 2; 2 |] neg_infinity)
              (reduce `Max [| (-1, 3); (0, 0) |] x);
            equal (tensor float_exact)
              (Nx.zeros Nx.float64 [| 0; 3 |])
              (reduce `Add [||] (Nx.ones Nx.float64 [| 4; 3 |])));
        test
          "reduce_ranges refuses a scalar, misshapen bounds, complex extremes \
           and boolean sums" (fun () ->
            let two = Nx.zeros Nx.int64 [| 2 |]
            and x = Nx.zeros Nx.float32 [| 2 |] in
            raises_invalid_arg (fun () ->
                Nx.reduce_ranges `Add ~lo:two ~hi:two (Nx.scalar Nx.float32 1.));
            raises_invalid_arg (fun () ->
                Nx.reduce_ranges `Add ~lo:two ~hi:(Nx.zeros Nx.int64 [| 3 |]) x);
            raises_invalid_arg (fun () ->
                Nx.reduce_ranges `Add
                  ~lo:(Nx.zeros Nx.int64 [| 2; 1 |])
                  ~hi:(Nx.zeros Nx.int64 [| 2; 1 |])
                  x);
            raises_invalid_arg (fun () ->
                Nx.reduce_ranges `Max ~lo:two ~hi:two
                  (Nx.zeros Nx.complex64 [| 2 |]));
            raises_invalid_arg (fun () ->
                Nx.reduce_ranges `Add ~lo:two ~hi:two (Nx.zeros Nx.bool [| 2 |])));
      ])

let () =
  exit
    (run "nx reductions"
       [
         integer_reductions;
         integer_dtypes;
         float_reductions;
         arg_reductions;
         scans;
         signed_zeros;
         normalisations;
         associations;
         segment_reductions;
         range_reductions;
         structure_scans;
         ewmas;
         histograms;
         at_scale;
       ])
