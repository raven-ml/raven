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
      prop "max and min propagate NaN"
        (with_axes (floats nonempty))
        (fun (t, axes, keepdims) ->
          assume (not (empty_axis t axes));
          let r = Ref.of_nx t in
          let exact = Ref.witness (close ~rel:0. ()) in
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

(* The index of the first extreme, a NaN counting as the extreme. *)
let first_extreme better lane =
  let best = ref 0 in
  Array.iteri
    (fun i x ->
      let b = lane.(!best) in
      if (not (Float.is_nan b)) && (Float.is_nan x || better x b) then best := i)
    lane;
  [| Int32.of_int !best |]

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
          equal ints expected (Ref.of_nx (nx ?axis ~keepdims t)))
  in
  group "argmax and argmin"
    [
      check "argmax"
        (fun ?axis ~keepdims t -> Nx.argmax ?axis ~keepdims t)
        ( > );
      check "argmin"
        (fun ?axis ~keepdims t -> Nx.argmin ?axis ~keepdims t)
        ( < );
      test "argmax refuses an axis out of bounds" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.argmax ~axis:2 (Nx.zeros Nx.float64 [| 2; 2 |])));
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
      exact "cummax, propagating NaN,"
        (fun ?axis t -> Nx.cummax ?axis t)
        nanmax
        (Ref.witness (close ~rel:0. ()))
        (floats ranked) Ref.of_nx;
      exact "cummin, propagating NaN,"
        (fun ?axis t -> Nx.cummin ?axis t)
        nanmin
        (Ref.witness (close ~rel:0. ()))
        (floats ranked) Ref.of_nx;
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
             equal ~msg:"argmax" int32
               (Int32.of_int (fst (extreme (fun c -> c > 0))))
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
      test
        "argmax and argmin refuse an axis longer than an int32 index holds \
         (nx.mli is silent)" (fun () ->
          let long = Nx.broadcast_to [| 2147483648 |] (Nx.scalar Nx.int8 1) in
          raises_match Exn.failure (fun () -> Nx.argmax long);
          raises_match Exn.failure (fun () -> Nx.argmin ~axis:0 long));
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

let () =
  exit
    (run "nx reductions"
       [
         integer_reductions;
         integer_dtypes;
         float_reductions;
         arg_reductions;
         scans;
         normalisations;
         at_scale;
       ])
