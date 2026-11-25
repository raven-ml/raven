(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Loss = Kaun.Loss

(* The unit roundoff of float64. *)
let u = Float.ldexp 1. (-53)
let f64 = Nx.float64
let dtype_name dt = Format.asprintf "%a" Nx.Dtype.pp dt
let item t = Nx.item [] t
let sum a = Array.fold_left ( +. ) 0. a

(* A drawn reduction: [None] is the default. *)
let reductions =
  Gen.option
    (Gen.of_list
       ~pp:(fun ppf r ->
         Format.pp_print_string ppf
           (match r with `Mean -> "`Mean" | `Sum -> "`Sum"))
       [ (`Mean : Loss.reduction); `Sum ])

let reduce r terms =
  if r = Some `Sum then sum terms
  else sum terms /. float_of_int (Array.length terms)

(* A sum of [n] non-negative terms, each within a few ulps, is within [(n + 4)
   u] relative of its exact value on either side, and a mean adds one rounding:
   the two sides lie within [2 (n + 5) u] of each other. *)
let sum_bound n = float_rel ~rel:(2. *. float_of_int (n + 5) *. u) ~abs:0.

let raises_prefixed fn f =
  let prefix = "Loss." ^ fn ^ ":" in
  raises_match
    (function Invalid_argument m -> String.starts_with ~prefix m | _ -> false)
    f

(* Losses *)

let scalars_at_dtype () =
  let check (type b) (dt : (float, b) Nx.dtype) =
    let x =
      Nx.cast dt (Nx.create f64 [| 2; 3 |] [| 0.5; -1.; 2.; 0.; 1.5; -0.5 |])
    in
    let y =
      Nx.cast dt (Nx.create f64 [| 2; 3 |] [| 0.; 0.5; 1.; 0.; 1.; 0. |])
    in
    let labels = Nx.create Nx.int64 [| 2 |] [| 0L; 2L |] in
    List.iter
      (fun (name, l) ->
        equal ~msg:name string (dtype_name dt) (dtype_name (Nx.dtype l));
        equal ~msg:name (array int) [||] (Nx.shape l))
      [
        ("mse", Loss.mse x y);
        ("mae", Loss.mae x y);
        ("huber", Loss.huber x y);
        ("sigmoid_bce", Loss.sigmoid_bce x y);
        ("softmax_cross_entropy", Loss.softmax_cross_entropy x y);
        ( "softmax_cross_entropy_sparse",
          Loss.softmax_cross_entropy_sparse x labels );
      ]
  in
  check Nx.float64;
  check Nx.float32;
  check Nx.float16;
  check Nx.bfloat16;
  check Nx.float8_e4m3fn;
  check Nx.float8_e5m2;
  check Nx.float4_e2m1fn

(* Regression *)

(* Predictions and targets of shape [r; c], or one of them of shape [c] or [||]
   broadcast against the other's [r; c], by [kind] 0 to 4; only the first
   entries of the drawn arrays are read. *)
let regression =
  let open Gen in
  let entries = array ~size:(constant 12) (float_range (-10.) 10.) in
  pair
    (triple (int_range 1 3) (int_range 1 4) (int_range 0 4))
    (triple reductions entries entries)

let tiny = Float.ldexp 1. (-1074)
let at0 v = Array.init 12 (fun i -> if i = 0 then v else 0.)

let regression_examples =
  [
    (* A subnormal difference. *)
    ((2, 1, 0), (None, at0 tiny, at0 0.));
    (* A scalar target, the default reduction. *)
    ((3, 4, 2), (None, Array.init 12 float_of_int, at0 2.5));
  ]

(* The shapes of the predictions and of the targets. *)
let shapes (r, c, kind) =
  match kind with
  | 0 -> ([| r; c |], [| r; c |])
  | 1 -> ([| r; c |], [| c |])
  | 2 -> ([| r; c |], [||])
  | 3 -> ([| c |], [| r; c |])
  | _ -> ([||], [| r; c |])

(* The predictions, the targets, and the [r c] differences of the broadcast
   predictions and targets. *)
let regression_case (((r, c, _) as shape), (_, ps, ts)) =
  let p_shape, t_shape = shapes shape in
  let size s = Array.fold_left ( * ) 1 s in
  let p = Array.sub ps 0 (size p_shape) and t = Array.sub ts 0 (size t_shape) in
  let at a i =
    match Array.length a with
    | 1 -> a.(0)
    | n when n = c -> a.(i mod c)
    | _ -> a.(i)
  in
  ( Nx.create f64 p_shape p,
    Nx.create f64 t_shape t,
    Array.init (r * c) (fun i -> at p i -. at t i) )

let is_the_reduction term loss ((shape, (r, _, _)) as case) =
  let _, _, kind = shape in
  cover "targets broadcast" (kind = 1 || kind = 2);
  cover "predictions broadcast" (kind = 3 || kind = 4);
  cover "the default reduction" (r = None);
  let p, t, d = regression_case case in
  equal
    (sum_bound (Array.length d))
    (reduce r (Array.map term d))
    (item (loss ?reduction:r p t))

let squared d = d *. d

let huber_term delta d =
  if Float.abs d <= delta then d *. d /. 2.
  else delta *. (Float.abs d -. (delta /. 2.))

(* Rune multiplies the term's derivative by the reduction's [1 / n]: a few
   roundings on each side, within [8u] relative, or a few subnormals where the
   difference is one. *)
let has_gradient derivative loss ((shape, (r, _, _)) as case) =
  let rows, cols, _ = shape in
  let case = ((rows, cols, 0), snd case) in
  let p, t, d = regression_case case in
  let n = if r = Some `Sum then 1. else float_of_int (Array.length d) in
  let g = Nx.to_array (Rune.grad' (fun p -> loss ?reduction:r p t) p) in
  Array.iteri
    (fun i d ->
      match derivative d with
      | None -> ()
      | Some expected ->
          equal
            ~msg:(Printf.sprintf "at p - t = %h" d)
            (float_rel ~rel:(8. *. u) ~abs:(Float.ldexp 1. (-1072)))
            (expected /. n) g.(i))
    d

let huber_examples =
  [
    (* At [|d| = delta] both branches give [delta^2 / 2]. *)
    (None, ((1, 2, 0), (None, Array.make 12 1., Array.make 12 0.)));
    (Some 2., ((1, 1, 2), (Some `Sum, Array.make 12 3., Array.make 12 0.)));
    (None, ((1, 1, 0), (None, at0 0., at0 tiny)));
  ]

let huber_law (delta, case) =
  cover "the default delta" (delta = None);
  is_the_reduction
    (huber_term (Option.value delta ~default:1.))
    (Loss.huber ?delta) case

let huber_gradient (delta, case) =
  let delta' = Option.value delta ~default:1. in
  has_gradient
    (fun d ->
      Some (if Float.abs d <= delta' then d else Float.copy_sign delta' d))
    (Loss.huber ?delta) case

let deltas = Gen.(option (float_range 0.1 3.))

(* Classification *)

(* [-(y log p + (1 - y) log (1 - p))] at [p = sigmoid z]: [log (1 + exp (-z)) +
   z (1 - y)], two non-negative terms, for [z >= 0], and [log (1 + exp z) - z y]
   below. *)
let bce z y =
  if z >= 0. then Float.log1p (Float.exp (-.z)) +. (z *. (1. -. y))
  else Float.log1p (Float.exp z) -. (z *. y)

let sigmoid z =
  if z >= 0. then 1. /. (1. +. Float.exp (-.z))
  else Float.exp z /. (1. +. Float.exp z)

(* Logits and their targets, or with [scalar] the first target alone, of shape
   [||], broadcast against every logit. *)
let bce_case =
  Gen.(
    triple reductions bool
      (array ~size:(int_range 1 8)
         (pair (float_range (-30.) 30.) (float_range 0. 1.))))

(* The tensors, and each logit with its broadcast target. *)
let bce_tensors scalar zys =
  let n = Array.length zys in
  let y0 = snd zys.(0) in
  ( Nx.create f64 [| n |] (Array.map fst zys),
    (if scalar then Nx.scalar f64 y0
     else Nx.create f64 [| n |] (Array.map snd zys)),
    if scalar then Array.map (fun (z, _) -> (z, y0)) zys else zys )

let bce_examples =
  [
    (None, false, [| (0., 0.); (0., 1.) |]);
    (Some `Sum, false, [| (1000., 1.); (-1000., 0.) |]);
    (Some `Sum, false, [| (1000., 0.); (-1000., 1.) |]);
  ]

let bce_law (r, scalar, zys) =
  cover "soft targets" (Array.exists (fun (_, y) -> y > 0. && y < 1.) zys);
  cover "a scalar target broadcast" (scalar && Array.length zys > 1);
  let z, y, zys = bce_tensors scalar zys in
  equal
    (sum_bound (Array.length zys))
    (reduce r (Array.map (fun (z, y) -> bce z y) zys))
    (item (Loss.sigmoid_bce ?reduction:r z y))

(* [Nx.sigmoid] is within 3 ulps (nx.mli) and the reference's within 2; [sigmoid
   z - y] is then within [16u] of the exact difference on either side. *)
let bce_gradient (r, scalar, zys) =
  let z, y, zys = bce_tensors scalar zys in
  let n = if r = Some `Sum then 1. else float_of_int (Array.length zys) in
  let g = Rune.grad' (fun z -> Loss.sigmoid_bce ?reduction:r z y) z in
  equal
    (array (float (16. *. u /. n)))
    (Array.map (fun (z, y) -> (sigmoid z -. y) /. n) zys)
    (Nx.to_array g)

let bce_gradient_examples =
  [
    (* A logit of 0 and of -0. has gradient 1/2 - y. *)
    (Some `Sum, false, [| (0., 1.); (0., 0.); (-0., 0.25) |]);
    (None, false, [| (1000., 1.); (-1000., 1.) |]);
  ]

(* [log1p (exp (-10))] at float32: [Nx.softplus] is within 3 ulps (nx.mli), and
   [z (1 - y)] adds [0]. *)
let bce_small_loss () =
  let l =
    Loss.sigmoid_bce
      (Nx.create Nx.float32 [| 1 |] [| 10. |])
      (Nx.create Nx.float32 [| 1 |] [| 1. |])
  in
  equal
    (float_rel ~rel:(Float.ldexp 1. (-21)) ~abs:0.)
    (Float.log1p (Float.exp (-10.)))
    (item l)

(* Rows of logits over [classes], [lead] being the leading shape, with soft
   targets (drawn weights, normalized) or one-hot ones (a drawn label). *)
let softmax_case =
  let open Gen in
  pair
    (pair (array ~size:(int_range 0 2) (int_range 1 2)) (int_range 1 5))
    (pair (pair reductions bool)
       (triple
          (array ~size:(constant 20) (float_range (-20.) 20.))
          (array ~size:(constant 20) (float_range 0. 1.))
          (array ~size:(constant 4) (int_range 0 4))))

let softmax_rows ((lead, c), ((_, soft), (zs, ws, labels))) =
  let rows = Array.fold_left ( * ) 1 lead in
  let row a i = Array.sub a (i * c) c in
  let targets i =
    if soft then
      let w = row ws i in
      let total = sum w in
      if total = 0. then Array.make c (1. /. float_of_int c)
      else Array.map (fun w -> w /. total) w
    else Array.init c (fun j -> if j = labels.(i) mod c then 1. else 0.)
  in
  (Array.append lead [| c |], List.init rows (fun i -> (row zs i, targets i)))

let log_sum_exp z =
  let m = Array.fold_left Float.max Float.neg_infinity z in
  m +. Float.log (sum (Array.map (fun z -> Float.exp (z -. m)) z))

let cross_entropy (z, t) =
  let lse = log_sum_exp z in
  sum (Array.mapi (fun j t -> t *. (lse -. z.(j))) t)

let max_abs_logit rows =
  List.fold_left
    (fun m (z, _) -> Array.fold_left (fun m z -> Float.max m (Float.abs z)) m z)
    0. rows

let softmax_tensors (shape, rows) =
  ( Nx.create f64 shape (Array.concat (List.map fst rows)),
    Nx.create f64 shape (Array.concat (List.map snd rows)) )

(* Each log-probability is within [32u (max |z| + ln c + 1)], with room over
   Nx.log_softmax's bound (nx.mli, "Reductions"); a row weighs them by targets
   summing to 1, and reducing the rows' losses adds [u] of their sum per row. *)
let softmax_law (((_, c), ((r, soft), _)) as case) =
  let shape, rows = softmax_rows case in
  cover "soft targets" soft;
  cover "one-hot targets" (not soft);
  cover "logits of rank 1" (Array.length shape = 1);
  let z, t = softmax_tensors (shape, rows) in
  let losses = Array.of_list (List.map cross_entropy rows) in
  let m = max_abs_logit rows in
  let k = float_of_int (List.length rows) in
  let bound =
    k
    *. ((32. *. u *. (m +. Float.log (float_of_int c) +. 1.))
       +. (u *. sum losses))
  in
  let bound = if r = Some `Sum then bound else bound /. k in
  equal (float bound) (reduce r losses)
    (item (Loss.softmax_cross_entropy ?reduction:r z t))

let softmax_examples =
  let case label =
    let zs = Array.make 20 0. in
    zs.(0) <- 1000.;
    let labels = Array.make 4 label in
    (([||], 2), ((Some `Sum, false), (zs, Array.make 20 0., labels)))
  in
  (* Finite at extreme logits: [0] for the extreme class, [1000] for the
     other. *)
  [ case 0; case 1 ]

let softmax_gradient_examples =
  let zs = Array.make 20 0. in
  zs.(0) <- 10.803802917102763;
  zs.(2) <- 19.359263756605877;
  [ (([||], 3), ((None, false), (zs, Array.make 20 0., Array.make 4 0))) ]

let softmax_ref z =
  let lse = log_sum_exp z in
  Array.map (fun z -> Float.exp (z -. lse)) z

(* Each probability's exponent [z - log_sum_exp z] carries an absolute error of
   a few [u (max |z| + ln c)], which the exponential makes relative; with its
   own 3 ulps, each side is within [8u (max |z| + ln c + 1)]. *)
let softmax_gradient (((_, _), ((r, _), _)) as case) =
  let shape, rows = softmax_rows case in
  let z, t = softmax_tensors (shape, rows) in
  let n = if r = Some `Sum then 1. else float_of_int (List.length rows) in
  let expected =
    Array.concat
      (List.map
         (fun (z, t) ->
           Array.mapi (fun j s -> (s -. t.(j)) /. n) (softmax_ref z))
         rows)
  in
  let m = max_abs_logit rows in
  let c = float_of_int (Array.length (fst (List.hd rows))) in
  equal
    (array (float (16. *. u *. (m +. Float.log c +. 1.) /. n)))
    expected
    (Nx.to_array
       (Rune.grad' (fun z -> Loss.softmax_cross_entropy ?reduction:r z t) z))

(* The float32 island *)

(* These rows state the island's result: a narrow loss is its float32 loss cast
   back. Eagerly they pass without the island too, since Nx.log_softmax computes
   a narrow format in float32 and rounds once (nx.mli) and Nx.mean accumulates
   wide: no eager input found tells the two apart.

   4096 classes of short binary fractions. *)
let island_logits dt classes =
  Nx.cast dt
    (Nx.create Nx.float32 [| 2; classes |]
       (Array.init (2 * classes) (fun i -> float_of_int (i mod 17) /. 8.)))

let island_labels = Nx.create Nx.int64 [| 2 |] [| 5L; 4000L |]

let same_scalar dt expected got =
  equal ~msg:(dtype_name dt) string (dtype_name dt) (dtype_name (Nx.dtype got));
  equal ~msg:(dtype_name dt) float_exact (item expected) (item got)

let dense_island () =
  let check (type b) (dt : (float, b) Nx.dtype) =
    let logits = island_logits dt 4096 in
    let targets = Nx.cast dt (Nx.one_hot ~num_classes:4096 island_labels) in
    let up = Nx.cast Nx.float32 in
    same_scalar dt
      (Nx.cast dt (Loss.softmax_cross_entropy (up logits) (up targets)))
      (Loss.softmax_cross_entropy logits targets)
  in
  check Nx.float16;
  check Nx.bfloat16;
  check Nx.float8_e4m3fn;
  check Nx.float8_e5m2;
  check Nx.float4_e2m1fn

let sparse_island_at (type b) (dt : (float, b) Nx.dtype) logits labels =
  same_scalar dt
    (Nx.cast dt
       (Loss.softmax_cross_entropy_sparse (Nx.cast Nx.float32 logits) labels))
    (Loss.softmax_cross_entropy_sparse logits labels)

let sparse_island () =
  let check dt = sparse_island_at dt (island_logits dt 4096) island_labels in
  check Nx.float16;
  check Nx.bfloat16;
  check Nx.float8_e4m3fn;
  check Nx.float8_e5m2;
  check Nx.float4_e2m1fn;
  (* 70000 classes: their sum of exponentials of [0] is past float16's largest
     value, 65504. *)
  sparse_island_at Nx.float16
    (Nx.zeros Nx.float16 [| 1; 70000 |])
    (Nx.create Nx.int64 [| 1 |] [| 3L |])

(* The island's casts differentiate to casts: the gradient at narrow logits is
   the float32 gradient cast to their dtype. *)
let island_gradient () =
  let check (type b) (dt : (float, b) Nx.dtype) =
    let logits = island_logits dt 8 in
    let labels = Nx.create Nx.int64 [| 2 |] [| 5L; 0L |] in
    let grad l =
      Rune.grad' (fun l -> Loss.softmax_cross_entropy_sparse l labels) l
    in
    let got = grad logits in
    equal ~msg:(dtype_name dt) string (dtype_name dt)
      (dtype_name (Nx.dtype got));
    equal ~msg:(dtype_name dt) (array float_exact)
      (Nx.to_array (Nx.cast dt (grad (Nx.cast Nx.float32 logits))))
      (Nx.to_array got)
  in
  check Nx.float16;
  check Nx.bfloat16;
  check Nx.float8_e4m3fn;
  check Nx.float8_e5m2;
  check Nx.float4_e2m1fn

(* Sparse labels *)

(* Logits of shape [lead; classes] and one label per row, drawn in and around
   [0, classes - 1] and over all of int64. *)
let sparse_case =
  let open Gen in
  pair
    (pair (array ~size:(int_range 0 2) (int_range 1 3)) (int_range 1 4))
    (triple reductions
       (array ~size:(constant 36) (float_range (-20.) 20.))
       (array ~size:(constant 9)
          (frequency [ (4, int64_range (-2L) 5L); (1, int64) ])))

(* The logits, the labels, and the dense targets: one-hot rows, a zero row for a
   label outside [0, classes - 1]. *)
let sparse_tensors ((lead, c), (_, zs, labels)) =
  let rows = Array.fold_left ( * ) 1 lead in
  let labels = Array.sub labels 0 rows in
  let one_hot l =
    Array.init c (fun j -> if Int64.equal l (Int64.of_int j) then 1. else 0.)
  in
  let shape = Array.append lead [| c |] in
  ( Nx.create f64 shape (Array.sub zs 0 (rows * c)),
    Nx.create Nx.int64 lead labels,
    Nx.create f64 shape
      (Array.concat (Array.to_list (Array.map one_hot labels))) )

let outside c l =
  Int64.compare l 0L < 0 || Int64.compare l (Int64.of_int c) >= 0

let sparse_covers ((lead, c), (_, _, labels)) =
  let rows = Array.fold_left ( * ) 1 lead in
  let labels = Array.sub labels 0 rows in
  cover "a label outside [0, classes - 1]" (Array.exists (outside c) labels);
  cover "a label inside" (Array.exists (fun l -> not (outside c l)) labels);
  cover "0-d labels" (lead = [||]);
  cover "logits of rank 3" (Array.length lead = 2)

let sparse_is_dense ((_, (r, _, _)) as case) =
  sparse_covers case;
  let z, labels, t = sparse_tensors case in
  equal float_exact
    (item (Loss.softmax_cross_entropy ?reduction:r z t))
    (item (Loss.softmax_cross_entropy_sparse ?reduction:r z labels))

let sparse_gradient_is_dense ((_, (r, _, _)) as case) =
  sparse_covers case;
  let z, labels, t = sparse_tensors case in
  let reduction = r in
  equal (array float_exact)
    (Nx.to_array
       (Rune.grad' (fun z -> Loss.softmax_cross_entropy ?reduction z t) z))
    (Nx.to_array
       (Rune.grad'
          (fun z -> Loss.softmax_cross_entropy_sparse ?reduction z labels)
          z))

let sparse_examples =
  let zs = Array.init 36 (fun i -> float_of_int (i mod 5) -. 2.) in
  let extreme = Array.init 36 (fun i -> if i = 0 then 1000. else 0.) in
  [
    (* Logits [1000; 0]: the loss is 1000 at label 1, 0 at label 0. *)
    ( ([| 2 |], 2),
      (Some `Sum, extreme, Array.init 9 (fun i -> if i = 0 then 1L else 0L)) );
    (* Every label outside: zero loss and zero gradient. *)
    ( ([| 3 |], 2),
      (None, zs, [| -1L; 2L; Int64.max_int; 0L; 0L; 0L; 0L; 0L; 0L |]) );
    (([| 1 |], 2), (Some `Sum, zs, Array.make 9 Int64.min_int));
  ]

let dense_refusals =
  let call z t =
    ignore (Loss.softmax_cross_entropy (Nx.zeros f64 z) (Nx.zeros f64 t))
  in
  [
    ("rank 0 logits", fun () -> call [||] [||]);
    ("an empty class axis", fun () -> call [| 2; 0 |] [| 2; 0 |]);
    ("targets of another shape", fun () -> call [| 2; 3 |] [| 2; 2 |]);
  ]

let sparse_refusals =
  let call z l =
    ignore
      (Loss.softmax_cross_entropy_sparse (Nx.zeros f64 z) (Nx.zeros Nx.int64 l))
  in
  [
    ("rank 0 logits", fun () -> call [||] [||]);
    ("an empty class axis", fun () -> call [| 2; 0 |] [| 2 |]);
    ("labels keeping the class axis", fun () -> call [| 2; 3 |] [| 2; 3 |]);
  ]

let refuses fn (_, f) = raises_prefixed fn f

let () =
  exit
    (run "kaun loss"
       [
         group "losses"
           [ test "are scalars at their predictions' dtype" scalars_at_dtype ];
         group "mse"
           [
             prop "reduces the squared error" ~examples:regression_examples
               regression
               (is_the_reduction squared Loss.mse);
             prop "has gradient 2 (p - t) / n" ~examples:regression_examples
               regression
               (has_gradient (fun d -> Some (2. *. d)) Loss.mse);
           ];
         group "mae"
           [
             prop "reduces the absolute error" ~examples:regression_examples
               regression
               (is_the_reduction Float.abs Loss.mae);
             prop "has gradient sign (p - t) / n where p <> t"
               ~examples:regression_examples regression
               (has_gradient
                  (fun d ->
                    if d = 0. then None else Some (Float.copy_sign 1. d))
                  Loss.mae);
           ];
         group "huber"
           [
             prop "reduces the Huber loss of the error" ~examples:huber_examples
               (Gen.pair deltas regression)
               huber_law;
             prop
               "has gradient d inside delta and delta * sign d beyond, over n"
               ~examples:huber_examples
               (Gen.pair deltas regression)
               huber_gradient;
             cases "raises on a delta that is not positive" ~id:string_of_float
               [ 0.; -0.; -1.; Float.nan ] (fun delta ->
                 raises_prefixed "huber" (fun () ->
                     Loss.huber ~delta (Nx.zeros f64 [| 1 |])
                       (Nx.zeros f64 [| 1 |])));
           ];
         group "sigmoid_bce"
           [
             prop "reduces the binary cross-entropy of sigmoid logits"
               ~examples:bce_examples bce_case bce_law;
             test "keeps its precision where the loss is small" bce_small_loss;
             prop "has gradient (sigmoid z - y) / n"
               ~examples:bce_gradient_examples bce_case bce_gradient;
           ];
         group "softmax_cross_entropy"
           [
             prop "reduces -sum targets * log_softmax logits over examples"
               ~examples:softmax_examples softmax_case softmax_law;
             prop "has gradient (softmax logits - targets) / n"
               ~examples:(softmax_examples @ softmax_gradient_examples)
               softmax_case softmax_gradient;
             test
               "computes narrow logits in a float32 island cast back to their \
                dtype"
               dense_island;
             test
               "differentiates through the island to the float32 gradient cast \
                back"
               island_gradient;
             cases
               "raises on rank 0 logits, an empty class axis or mismatched \
                targets"
               ~id:fst dense_refusals
               (refuses "softmax_cross_entropy");
           ];
         group "softmax_cross_entropy_sparse"
           [
             prop
               "is the dense loss on one-hot targets, a zero row for a label \
                outside [0, classes - 1]"
               ~examples:sparse_examples sparse_case sparse_is_dense;
             prop "has the dense loss's gradient on those targets"
               ~examples:sparse_examples sparse_case sparse_gradient_is_dense;
             test "has the float32 island of softmax_cross_entropy"
               sparse_island;
             cases
               "raises on rank 0 logits, an empty class axis or mismatched \
                labels"
               ~id:fst sparse_refusals
               (refuses "softmax_cross_entropy_sparse");
           ];
       ])
