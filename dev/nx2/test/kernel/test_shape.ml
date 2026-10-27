(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Shapes: each function of Nx_kernel.Shape against the rule its doc states,
   at its bounds and their neighbours; Nx_kernel's dtype rules against their
   tables; the constructors C reads against nx_kernel.h's codes; and the
   grouped contraction view against every element's position by brute
   force. *)

open Windtrap
open Nx_array_gen
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module Shape = Nx_kernel.Shape

(* The view nx.cpu's nx_contract_fill groups, read from its fields
   ({!Nx_kernels_support.view_fields}) by cpu.h's indices. *)
module V = struct
  type operand = A | B | Init | Dst
  type axis = Batch | Row | Column | Contracted

  let operand = function A -> 0 | B -> 1 | Init -> 2 | Dst -> 3
  let axis = function Batch -> 0 | Row -> 1 | Column -> 2 | Contracted -> 3
  let v = Nx_kernels_support.view_fields
  let extent f x = f.(axis x)
  let offset f o = f.(4 + operand o)
  let stride f o x = f.(8 + (4 * operand o) + axis x)
end

(* Shape's results, their messages dropped. *)

let shape_t = result (list int) unit
let shapes_t = result (list (list int)) unit
let misfit = Error ()
let of_shape = function Ok y -> Ok (Array.to_list y) | Error _ -> misfit

let of_shapes = function
  | Ok ys -> Ok (List.map Array.to_list ys)
  | Error _ -> misfit

let fits ~msg want got = equal ~msg shape_t (Ok want) (of_shape got)
let refuses ~msg got = equal ~msg shape_t misfit (of_shape got)

(* Contractions *)

(* A contraction drawn by its groups' extents. [a] is laid out in memory as
   batch, rows, contracted axes and [b] as batch, contracted, columns, each
   C-contiguous or over drawn strides, then permuted: [pa.(i)] is the memory
   axis of [a]'s axis [i]. *)
type case = {
  batch : int array;
  rows : int array;
  contracted : int array;
  columns : int array;
  pa : int array;
  pb : int array;
  la : L.t;
  lb : L.t;
  ly : L.t;
  li : L.t option;
}

let pp_case ppf c =
  Format.fprintf ppf
    "batch %a rows %a contracted %a columns %a pa %a pb %a a %a b %a y %a%s"
    pp_ints c.batch pp_ints c.rows pp_ints c.contracted pp_ints c.columns
    pp_ints c.pa pp_ints c.pb L.pp c.la L.pp c.lb L.pp c.ly
    (match c.li with None -> "" | Some l -> Format.asprintf " init %a" L.pp l)

let inverse p =
  let q = Array.make (Array.length p) 0 in
  Array.iteri (fun i m -> q.(m) <- i) p;
  q

(* The pairs of [c]: each group's memory axis, as an axis of [a] and [b]. *)
let pairs_of c =
  let nb = Array.length c.batch and nr = Array.length c.rows in
  let ia = inverse c.pa and ib = inverse c.pb in
  let batch = Iarray.init nb (fun k -> (ia.(k), ib.(k))) in
  let contracting =
    Iarray.init (Array.length c.contracted) (fun k ->
        (ia.(nb + nr + k), ib.(nb + k)))
  in
  (batch, contracting)

(* The axes no pair names, in axis order. *)
let free r named =
  List.filter (fun ax -> not (List.mem ax named)) (List.init r Fun.id)

(* The result's shape by the rule: the batch extents in pair order, then [a]'s
   free extents, then [b]'s, each in axis order. *)
let result_shape a b batch contracting =
  let named side ps = Array.to_list (Array.map side ps) in
  let fa = free (Array.length a) (named fst batch @ named fst contracting) in
  let fb = free (Array.length b) (named snd batch @ named snd contracting) in
  Array.concat
    [
      Array.map (fun (i, _) -> a.(i)) batch;
      Array.of_list (List.map (fun ax -> a.(ax)) fa);
      Array.of_list (List.map (fun ax -> b.(ax)) fb);
    ]

let extents = Gen.array ~size:(Gen.int_range 0 2) extent
let axis_order n = Gen.map Array.of_list (Gen.permutation (List.init n Fun.id))

(* A layout of shape [s]: C-contiguous, which a view groups, or, unless
   [canonical], over drawn strides. *)
let layout_of ~canonical s =
  if canonical then Gen.constant (L.contiguous s)
  else Gen.one_of [ Gen.constant (L.contiguous s); strided_of s ]

let permuted l p = Option.get (L.move (M.Permute (Iarray.of_array p)) l)

(* [p] with the memory axes [lo] to [lo + n - 1] in increasing order of the axes
   that hold them: a free group's axes keep memory order, so they lie as one run
   in a C-contiguous layout. *)
let in_order p lo n =
  let next = ref lo in
  Array.map
    (fun m ->
      if m >= lo && m < lo + n then begin
        let m' = !next in
        incr next;
        m'
      end
      else m)
    p

let case ~canonical =
  let open Gen in
  let layout_of = layout_of ~canonical in
  let* batch = extents in
  let* rows = extents in
  let* contracted = extents in
  let* columns = extents in
  let ma = Array.concat [ batch; rows; contracted ] in
  let mb = Array.concat [ batch; contracted; columns ] in
  let nb = Array.length batch and nr = Array.length rows in
  let nk = Array.length contracted in
  let order p lo n = if canonical then in_order p lo n else p in
  let* pa = map (fun p -> order p nb nr) (axis_order (Array.length ma)) in
  let* pb =
    map
      (fun p -> order p (nb + nk) (Array.length columns))
      (axis_order (Array.length mb))
  in
  let* la = layout_of ma in
  let* lb = layout_of mb in
  let la = permuted la pa and lb = permuted lb pb in
  let c =
    { batch; rows; contracted; columns; pa; pb; la; lb; ly = la; li = None }
  in
  let b, k = pairs_of c in
  let b = Iarray.to_array b and k = Iarray.to_array k in
  let y = result_shape (L.shape la) (L.shape lb) b k in
  let* ly = layout_of y in
  let+ li = option (layout_of y) in
  { c with ly; li }

let any_case = Gen.with_pp pp_case (case ~canonical:false)
let canonical_case = Gen.with_pp pp_case (case ~canonical:true)

let law_contract c =
  let batch, contracting = pairs_of c in
  fits ~msg:"fits" (Array.to_list (L.shape c.ly))
    (Shape.contract ~batch ~contracting ~segments:None
       (L.shape c.la, None)
       (L.shape c.lb, None))

let test_contract_refuses () =
  let a = [| 2; 3; 4 |] and b = [| 2; 4; 5 |] in
  let contract ?(batch : _ iarray = [| (0, 0) |])
        ?(contracting : _ iarray = [| (2, 1) |]) a b =
    Shape.contract ~batch ~contracting ~segments:None (a, None) (b, None)
  in
  fits ~msg:"fits" [ 2; 3; 5 ] (contract a b);
  refuses ~msg:"a contracting extent" (contract a [| 2; 3; 5 |]);
  refuses ~msg:"a batch extent" (contract a [| 3; 4; 5 |]);
  refuses ~msg:"a pair past b's rank" (contract a [| 2 |]);
  refuses ~msg:"a pair past a's rank" (contract ~contracting:[| (3, 1) |] a b);
  refuses ~msg:"a negative axis" (contract ~batch:[| (-1, 0) |] a b);
  refuses ~msg:"a negative axis of b" (contract ~batch:[| (0, -1) |] a b);
  let sq = [| 4; 4 |] in
  fits ~msg:"distinct axes" [ 4 ]
    (contract ~batch:[| (0, 0) |] ~contracting:[| (1, 1) |] sq sq);
  refuses ~msg:"an axis of a in two pairs"
    (contract ~batch:[| (0, 0) |] ~contracting:[| (0, 1) |] sq sq);
  refuses ~msg:"an axis of b in two pairs"
    (contract ~batch:[| (0, 1); (1, 1) |] ~contracting:[||] sq sq);
  refuses ~msg:"an axis in two contracting pairs"
    (contract ~batch:[||] ~contracting:[| (1, 0); (1, 1) |] sq sq);
  let ones = Array.make L.max_rank 1 in
  fits ~msg:"a result of the most rank" (Array.to_list ones)
    (contract ~batch:[||] ~contracting:[||] ones [||]);
  refuses ~msg:"a result past the most rank"
    (contract ~batch:[||] ~contracting:[||] ones [| 1 |])

(* A scale has its operand's rank, each extent dividing the operand's, 0 only
   along an empty axis. *)
let test_contract_scales () =
  let mm ?sa ?sb a b =
    Shape.contract ~batch:[||] ~contracting:[| (1, 0) |] ~segments:None
      (a, sa) (b, sb)
  in
  let a = [| 4; 8 |] and b = [| 8; 6 |] in
  fits ~msg:"blocks of 4 by 2" [ 4; 6 ] (mm ~sb:[| 2; 3 |] a b);
  fits ~msg:"a block of a whole axis" [ 4; 6 ] (mm ~sb:[| 8; 1 |] a b);
  fits ~msg:"a ratio of 1" [ 4; 6 ] (mm ~sa:a ~sb:b a b);
  refuses ~msg:"an extent that does not divide" (mm ~sb:[| 3; 6 |] a b);
  refuses ~msg:"an extent past the operand's" (mm ~sa:[| 8; 8 |] a b);
  refuses ~msg:"a lower rank" (mm ~sa:[| 4 |] a b);
  refuses ~msg:"a higher rank" (mm ~sa:[| 4; 8; 1 |] a b);
  let e = [| 0; 8 |] in
  fits ~msg:"0 along an empty axis" [ 0; 6 ] (mm ~sa:[| 0; 2 |] e b);
  fits ~msg:"1 along an empty axis" [ 0; 6 ] (mm ~sa:[| 1; 2 |] e b);
  refuses ~msg:"0 along a full axis" (mm ~sa:[| 0; 2 |] a b);
  refuses ~msg:"a negative extent" (mm ~sa:[| -4; 2 |] a b)

(* Rows drops b's group axis from the result; Contracting puts the group axis
   first; offsets have the batch extents, then one more than the groups. *)
let test_contract_segments () =
  let contract ?(batch : _ iarray = [||])
      ?(contracting : _ iarray = [| (1, 1) |]) segments a b =
    Shape.contract ~batch ~contracting ~segments:(Some segments) (a, None)
      (b, None)
  in
  let rows ?(a_axis = 0) ?(b_axis = 0) offsets =
    Nx_kernel.Rows { offsets; a_axis; b_axis }
  in
  let a = [| 10; 8 |] and b = [| 3; 8; 6 |] in
  fits ~msg:"rows" [ 10; 6 ] (contract (rows [| 4 |]) a b);
  fits ~msg:"rows of no group" [ 10; 6 ]
    (contract (rows [| 1 |]) a [| 0; 8; 6 |]);
  refuses ~msg:"offsets of another extent" (contract (rows [| 3 |]) a b);
  refuses ~msg:"offsets of another rank" (contract (rows [| 4; 1 |]) a b);
  refuses ~msg:"no offsets" (contract (rows [||]) a b);
  refuses ~msg:"an a_axis in a pair" (contract (rows ~a_axis:1 [| 4 |]) a b);
  refuses ~msg:"a b_axis in a pair" (contract (rows ~b_axis:1 [| 9 |]) a b);
  refuses ~msg:"an a_axis past a's rank" (contract (rows ~a_axis:2 [| 4 |]) a b);
  refuses ~msg:"a negative b_axis" (contract (rows ~b_axis:(-1) [| 4 |]) a b);
  let a2 = [| 2; 10; 8 |] and b2 = [| 2; 3; 8; 6 |] in
  let batched offsets =
    contract ~batch:[| (0, 0) |] ~contracting:[| (2, 2) |]
      (rows ~a_axis:1 ~b_axis:1 offsets) a2 b2
  in
  fits ~msg:"rows per batch" [ 2; 10; 6 ] (batched [| 2; 4 |]);
  refuses ~msg:"offsets without the batch" (batched [| 4 |]);
  refuses ~msg:"offsets of another batch" (batched [| 3; 4 |]);
  let k offsets pair =
    contract ~contracting:[| (1, 0) |]
      (Nx_kernel.Contracting { offsets; pair })
      [| 4; 8 |] [| 8; 6 |]
  in
  fits ~msg:"contracting" [ 4; 4; 6 ] (k [| 5 |] 0);
  fits ~msg:"contracting into no group" [ 0; 4; 6 ] (k [| 1 |] 0);
  refuses ~msg:"no offsets" (k [| 0 |] 0);
  refuses ~msg:"offsets of another rank" (k [| 2; 5 |] 0);
  refuses ~msg:"a pair past the contracting ones" (k [| 5 |] 1);
  refuses ~msg:"a negative pair" (k [| 5 |] (-1));
  let ones = Array.make (L.max_rank - 1) 1 in
  let deep b =
    contract ~contracting:[| (0, 0) |]
      (Nx_kernel.Contracting { offsets = [| 2 |]; pair = 0 })
      ones b
  in
  fits ~msg:"a result of the most rank"
    (Array.to_list (Array.make L.max_rank 1))
    (deep [| 1; 1 |]);
  refuses ~msg:"a result past the most rank" (deep [| 1; 1; 1 |])

(* The doc's table: a contraction computes in every dtype but booleans and
   floats narrower than float32. *)
let test_accumulates () =
  List.iter
    (fun (D.Any dt) ->
      let want =
        match dt with
        | D.Bool | D.Bit | D.Float16 | D.Bfloat16 | D.Float8_e4m3fn
        | D.Float8_e5m2 | D.Float4_e2m1fn ->
            false
        | _ -> true
      in
      equal ~msg:(D.name dt) bool want (Nx_kernel.accumulates dt))
    D.all

(* Every shape Shape derives, at max_numel and past it. *)
let test_admits () =
  let big = L.max_numel and z : int iarray = [| 0 |] in
  let pad ?(interior : int iarray = [| 0 |]) lo hi =
    { Nx_kernel.lo; hi; interior; windows = [||] }
  in
  fits ~msg:"padded to max_numel" [ big ]
    (Shape.padded (pad [| big - 2 |] z) [| 2 |]);
  refuses ~msg:"padded past max_numel"
    (Shape.padded (pad [| big - 1 |] z) [| 2 |]);
  refuses ~msg:"lo + hi wraps"
    (Shape.padded (pad [| max_int |] [| max_int |]) [| 2 |]);
  refuses ~msg:"lo + hi wraps below"
    (Shape.padded (pad [| min_int |] [| min_int |]) [| 2 |]);
  refuses ~msg:"interior times extent wraps"
    (Shape.padded (pad ~interior:[| max_int |] z z) [| 3 |]);
  refuses ~msg:"interior times extent past max_numel"
    (Shape.padded (pad ~interior:[| big / 2 |] z z) [| 3 |]);
  let pad2 lo hi =
    { Nx_kernel.lo; hi; interior = [| 0; 0 |]; windows = [||] }
  in
  fits ~msg:"no element, other extents at max_numel" [ 0; big ]
    (Shape.padded (pad2 [| 0; 0 |] [| 0; big - 1 |]) [| 0; 1 |]);
  refuses ~msg:"no element, other extents past max_numel"
    (Shape.padded (pad2 [| 0; 0 |] [| 0; big |]) [| 0; 1 |]);
  let k = 1 lsl 27 in
  let mm a b =
    Shape.contract ~batch:[||] ~contracting:[| (1, 0) |] ~segments:None
      (a, None) (b, None)
  in
  fits ~msg:"a contraction to max_numel" [ k / 2; k ]
    (mm [| k / 2; 0 |] [| 0; k |]);
  refuses ~msg:"a contraction past max_numel" (mm [| k; 0 |] [| 0; k |]);
  fits ~msg:"a transform to max_numel points" [ big ]
    (Shape.c2r ~axes:[| 0 |] ~n:big [| (big / 2) + 1 |]);
  refuses ~msg:"a transform past max_numel points"
    (Shape.c2r ~axes:[| 0 |] ~n:(big + 2) [| (big / 2) + 2 |]);
  let svd a =
    of_shapes
      (Result.map (fun (s, u, vh) -> [ s; u; vh ]) (Shape.svd Complete a))
  in
  equal ~msg:"square vectors within max_numel" shapes_t
    (Ok [ [ 0 ]; [ k / 2; k / 2 ]; [ 0; 0 ] ])
    (svd [| k / 2; 0 |]);
  equal ~msg:"square vectors past max_numel" shapes_t misfit
    (svd [| 0; k |]);
  let qr a =
    of_shapes (Result.map (fun (q, r) -> [ q; r ]) (Shape.qr Complete a))
  in
  equal ~msg:"square factors within max_numel" shapes_t
    (Ok [ [ k / 2; k / 2 ]; [ k / 2; 0 ] ])
    (qr [| k / 2; 0 |]);
  equal ~msg:"square factors past max_numel" shapes_t misfit (qr [| k; 0 |])

(* Paddings *)

let pp_pad ppf (pad : Nx_kernel.pad) =
  Format.fprintf ppf "padded lo %a hi %a interior %a, %d windows" pp_ints
    (Iarray.to_array pad.lo) pp_ints (Iarray.to_array pad.hi) pp_ints
    (Iarray.to_array pad.interior)
    (Iarray.length pad.windows)

(* A padding of a rank [r]: crops and pads of a few elements, interiors of
   up to two, and at most one window. *)
let pad_of r =
  let open Gen in
  let small = int_range (-2) 3 in
  let* lo = array ~size:(constant r) small in
  let* hi = array ~size:(constant r) small in
  let* interior = array ~size:(constant r) (int_range 0 2) in
  let+ window = option (pair (int_range 0 (max 0 (r - 1))) (int_range 1 3)) in
  let windows =
    match window with
    | Some (axis, size) when r > 0 ->
        [| { M.axis; size; step = 1 + (size mod 2); dilation = 1 } |]
    | _ -> [||]
  in
  let lo = Iarray.of_array lo and hi = Iarray.of_array hi in
  let interior = Iarray.of_array interior in
  { Nx_kernel.lo; hi; interior; windows = Iarray.of_array windows }

(* Whether the extents of [s] other than 0 multiply to at most max_numel. *)
let admitted s =
  let n =
    Array.fold_left
      (fun n d ->
        match n with
        | Some n when d = 0 -> Some n
        | Some n when d <= L.max_numel / n -> Some (n * d)
        | _ -> None)
      (Some 1) s
  in
  n <> None

(* The shape an operand of shape [x] has once loaded, by the rule: the padded
   extents, computed in floats, exact for the small ones the generators draw
   and past max_numel for the large, then the windows as Move.Window takes
   them. *)
let loaded x = function
  | None -> Some x
  | Some (pad : Nx_kernel.pad) ->
      let extent i d =
        let f = Float.of_int in
        f (Iarray.get pad.lo i) +. f (Iarray.get pad.hi i) +. f d
        +. (f (Iarray.get pad.interior i) *. Float.of_int (max 0 (d - 1)))
      in
      let padded = Array.mapi extent x in
      if Array.exists (fun d -> d < 0. || d > Float.of_int L.max_numel) padded
      then None
      else
        let padded = Array.map Float.to_int padded in
        if not (admitted padded) then None
        else if pad.windows = [||] then Some padded
        else
          match M.shape (Window pad.windows) padded with
          | y -> Some y
          | exception Invalid_argument _ -> None

(* A padding of an operand of any shape. *)
type padded = { x : int array; pad : Nx_kernel.pad }

let padded_case =
  Gen.with_pp
    (fun ppf c -> Format.fprintf ppf "shape %a, %a" pp_ints c.x pp_pad c.pad)
    (let open Gen in
     let* x = Nx_array_gen.shape in
     let+ pad = pad_of (Array.length x) in
     { x; pad })

let law_padded c =
  let want = Option.to_result ~none:() (loaded c.x (Some c.pad)) in
  cover "refused" (Result.is_error want);
  cover "windowed" (Result.is_ok want && c.pad.windows <> [||]);
  equal shape_t (Result.map Array.to_list want)
    (of_shape (Shape.padded c.pad c.x))

let test_padded_refuses () =
  let z = { Nx_kernel.lo = [| 0 |]; hi = [| 0 |]; interior = [| 0 |]; windows = [||] } in
  let window axis size : _ iarray =
    [| { M.axis; size; step = 1; dilation = 1 } |]
  in
  refuses ~msg:"hi of another rank"
    (Shape.padded { z with hi = [| 0; 0 |] } [| 2 |]);
  refuses ~msg:"a pad of another rank" (Shape.padded z [| 2; 2 |]);
  refuses ~msg:"negative interior"
    (Shape.padded { z with interior = [| -1 |] } [| 2 |]);
  fits ~msg:"cropped to nothing" [ 0 ] (Shape.padded { z with lo = [| -2 |] } [| 2 |]);
  refuses ~msg:"a negative padded extent"
    (Shape.padded { z with lo = [| -3 |] } [| 2 |]);
  refuses ~msg:"a negative extent" (Shape.padded z [| -1 |]);
  refuses ~msg:"a window past the rank"
    (Shape.padded { z with windows = window 1 1 } [| 2 |]);
  refuses ~msg:"an empty window"
    (Shape.padded { z with windows = window 0 0 } [| 2 |]);
  fits ~msg:"a window of the padded axis" [ 1; 2 ]
    (Shape.padded { z with windows = window 0 2 } [| 2 |]);
  fits ~msg:"a window past the padded axis is no window" [ 0; 3 ]
    (Shape.padded { z with windows = window 0 3 } [| 2 |])

(* Reductions and scans *)

let reductions_all : Nx_kernel.reduction list =
  [
    Monoid Sum;
    Monoid Prod;
    Monoid Max;
    Monoid Min;
    Logsumexp;
    Moments;
    Arg Max;
    Arg Min;
  ]

let reduction_name : Nx_kernel.reduction -> string = function
  | Monoid Sum -> "Sum"
  | Monoid Prod -> "Prod"
  | Monoid Max -> "Max"
  | Monoid Min -> "Min"
  | Logsumexp -> "Logsumexp"
  | Moments -> "Moments"
  | Arg Max -> "Arg Max"
  | Arg Min -> "Arg Min"

let pp_reduction ppf r = Format.pp_print_string ppf (reduction_name r)

let extreme : Nx_kernel.reduction -> bool = function
  | Monoid (Max | Min) | Arg _ -> true
  | Monoid (Sum | Prod) | Logsumexp | Moments -> false

type reduce_case = {
  y : int array;
  axes : int array;
  rs : Nx_kernel.reduction array;
}

let pp_reduce_case ppf c =
  Format.fprintf ppf "%a along %a of %a"
    (Format.pp_print_list pp_reduction)
    (Array.to_list c.rs) pp_ints c.axes pp_ints c.y

(* Reductions along a subset of a shape's axes. *)
let reduce_case =
  Gen.with_pp pp_reduce_case
    (let open Gen in
     let* y = Nx_array_gen.shape in
     let rank = Array.length y in
     let* keep = array ~size:(constant rank) bool in
     let axes =
       Array.of_list (List.filter (fun i -> keep.(i)) (List.init rank Fun.id))
     in
     let+ rs = array ~size:(int_range 1 2) (of_list ~pp:pp_reduction reductions_all) in
     { y; axes; rs })

(* The destinations' shape by the rule: the loop's without the axes, and no
   shape for an extreme with a destination element and no term. *)
let law_reduce c =
  let kept =
    List.filteri (fun i _ -> not (Array.mem i c.axes)) (Array.to_list c.y)
  in
  let terms = Array.fold_left (fun n a -> n * c.y.(a)) 1 c.axes in
  let no_term = terms = 0 && List.fold_left ( * ) 1 kept > 0 in
  let want = if no_term && Array.exists extreme c.rs then misfit else Ok kept in
  cover "fits" (Result.is_ok want);
  cover "no term" no_term;
  cover "an extreme of no term" (Result.is_error want);
  let axes = Iarray.of_array c.axes in
  equal shape_t want (of_shape (Shape.reduce ~axes c.rs c.y))

let test_reduce_refuses () =
  let y = [| 2; 3 |] and sum = [| Nx_kernel.Monoid Sum |] in
  let reduce ?(rs = sum) ?(y = y) axes = Shape.reduce ~axes rs y in
  fits ~msg:"along no axis" [ 2; 3 ] (reduce [||]);
  fits ~msg:"along every axis" [] (reduce [| 0; 1 |]);
  refuses ~msg:"axes out of order" (reduce [| 1; 0 |]);
  refuses ~msg:"a repeated axis" (reduce [| 0; 0 |]);
  refuses ~msg:"a negative axis" (reduce [| -1 |]);
  fits ~msg:"the last axis" [ 2 ] (reduce [| 1 |]);
  refuses ~msg:"an axis past the rank" (reduce [| 2 |]);
  let empty = [| 2; 0 |] in
  fits ~msg:"a sum of no term" [ 2 ] (reduce ~y:empty [| 1 |]);
  fits ~msg:"moments of no term" [ 2 ] (reduce ~rs:[| Moments |] ~y:empty [| 1 |]);
  refuses ~msg:"a maximum of no term"
    (reduce ~rs:[| Monoid Max |] ~y:empty [| 1 |]);
  refuses ~msg:"a minimum of no term beside a sum"
    (reduce ~rs:[| Monoid Sum; Monoid Min |] ~y:empty [| 1 |]);
  refuses ~msg:"an Arg of no term" (reduce ~rs:[| Arg Min |] ~y:empty [| 1 |]);
  fits ~msg:"a maximum with no output" [ 0 ]
    (reduce ~rs:[| Monoid Max |] ~y:[| 0; 0 |] [| 1 |]);
  fits ~msg:"a scan keeps the shape" [ 2; 3 ] (Shape.scan ~axis:1 y);
  fits ~msg:"a scan of no term" [ 2; 0 ] (Shape.scan ~axis:1 empty);
  refuses ~msg:"a scan along an axis past the rank" (Shape.scan ~axis:2 y);
  refuses ~msg:"a scan along a negative axis" (Shape.scan ~axis:(-1) y);
  refuses ~msg:"a scan of rank 0" (Shape.scan ~axis:0 [||])

(* The doc's table: Sum and Prod take every dtype but booleans, Max, Min and
   Arg every dtype, Logsumexp and Moments floats. *)
let test_accepts () =
  List.iter
    (fun (D.Any dt) ->
      let boolean = match dt with D.Bool | D.Bit -> true | _ -> false in
      let float =
        match dt with
        | D.Float64 | D.Float32 | D.Float16 | D.Bfloat16 | D.Float8_e4m3fn
        | D.Float8_e5m2 | D.Float4_e2m1fn ->
            true
        | _ -> false
      in
      List.iter
        (fun r ->
          let want : bool =
            match (r : Nx_kernel.reduction) with
            | Monoid (Sum | Prod) -> not boolean
            | Monoid (Max | Min) | Arg _ -> true
            | Logsumexp | Moments -> float
          in
          equal
            ~msg:(Format.asprintf "%s of %a" (reduction_name r) D.pp dt)
            bool want (Nx_kernel.accepts r dt))
        reductions_all)
    D.all

(* Gathers, scatters and sorts *)

let test_index_refuses () =
  refuses ~msg:"a gather along a negative axis"
    (Shape.gather ~axis:(-1) [| 2 |] [| 2 |]);
  refuses ~msg:"a scatter along a negative axis"
    (Shape.scatter ~axis:(-1) [| 2 |] [| 2 |] [| 2 |]);
  fits ~msg:"a scatter along the last axis" [ 2 ]
    (Shape.scatter ~axis:0 [| 2 |] [| 3 |] [| 3 |]);
  refuses ~msg:"a scatter along an axis past the rank"
    (Shape.scatter ~axis:1 [| 2 |] [| 2 |] [| 2 |]);
  refuses ~msg:"a sort along a negative axis"
    (Shape.sort ~axis:(-1) ~k:None [| 2 |]);
  refuses ~msg:"a negative k" (Shape.sort ~axis:0 ~k:(Some (-1)) [| 2 |]);
  fits ~msg:"a k of 0" [ 0 ] (Shape.sort ~axis:0 ~k:(Some 0) [| 2 |])

(* Each family's shapes, at a fit and at each way to miss it. *)
let test_index_shapes () =
  let g idx x = Shape.gather ~axis:1 idx x in
  fits ~msg:"gather: positions' shape" [ 2; 5; 4 ]
    (g [| 2; 5; 4 |] [| 2; 3; 4 |]);
  fits ~msg:"gather: from an empty axis" [ 2; 5; 4 ]
    (g [| 2; 5; 4 |] [| 2; 0; 4 |]);
  refuses ~msg:"gather: another extent off the axis"
    (g [| 2; 5; 3 |] [| 2; 3; 4 |]);
  refuses ~msg:"gather: another rank" (g [| 2; 5 |] [| 2; 3; 4 |]);
  fits ~msg:"gather: along the last axis" [ 2; 5 ] (g [| 2; 5 |] [| 2; 3 |]);
  refuses ~msg:"gather: an axis past the rank" (g [| 2 |] [| 2 |]);
  let sc into idx u = Shape.scatter ~axis:0 into idx u in
  fits ~msg:"scatter: into's shape" [ 7; 4 ]
    (sc [| 7; 4 |] [| 9; 4 |] [| 9; 4 |]);
  refuses ~msg:"scatter: positions and updates differ"
    (sc [| 7; 4 |] [| 9; 4 |] [| 8; 4 |]);
  refuses ~msg:"scatter: another extent off the axis"
    (sc [| 7; 4 |] [| 9; 3 |] [| 9; 3 |]);
  refuses ~msg:"scatter: another rank" (sc [| 7; 4 |] [| 9 |] [| 9 |]);
  let so k x = Shape.sort ~axis:1 ~k x in
  fits ~msg:"sort: the operand's shape" [ 2; 5 ] (so None [| 2; 5 |]);
  fits ~msg:"sort: k along the axis" [ 2; 3 ] (so (Some 3) [| 2; 5 |]);
  fits ~msg:"sort: k of the whole axis" [ 2; 5 ] (so (Some 5) [| 2; 5 |]);
  refuses ~msg:"sort: k past the axis" (so (Some 6) [| 2; 5 |]);
  refuses ~msg:"sort: an axis past the rank" (so None [| 2 |])

(* Assemblies *)

type assemble_case = { shape : int array; regions : M.range array array }

let pp_regions ppf rs =
  Array.iter
    (fun r ->
      Format.fprintf ppf "[%s]"
        (String.concat "; "
           (Array.to_list
              (Array.map
                 (fun (x : M.range) ->
                   Printf.sprintf "%d+%d*%d" x.start x.count x.step)
                 r))))
    rs

let assemble_case =
  Gen.with_pp
    (fun ppf c ->
      Format.fprintf ppf "shape %a, regions %a" pp_ints c.shape pp_regions
        c.regions)
    (let open Gen in
     let* shape = Nx_array_gen.shape in
     let region =
       let rec go i acc =
         if i < 0 then constant (Array.of_list acc)
         else
           let* r = Nx_array_gen.range shape.(i) in
           go (i - 1) (r :: acc)
       in
       go (Array.length shape - 1) []
     in
     let+ regions = array ~size:(int_range 0 3) region in
     { shape; regions })

(* Each piece has its region's shape, as Move.Slice keeps it. *)
let law_assemble c =
  let regions = Array.map Iarray.of_array c.regions in
  let pieces =
    Array.map (fun r -> Array.to_list (M.shape (Slice r) c.shape)) regions
  in
  cover "pieces" (c.regions <> [||]);
  equal shapes_t (Ok (Array.to_list pieces))
    (of_shapes (Result.map Array.to_list (Shape.assemble c.shape regions)))

let test_assemble_refuses () =
  let r start count step = { M.start; count; step } in
  let assemble y regions =
    of_shapes (Result.map Array.to_list (Shape.assemble y regions))
  in
  let refuses ~msg got = equal ~msg shapes_t misfit got in
  equal ~msg:"no piece" shapes_t (Ok []) (assemble [| 2 |] [||]);
  equal ~msg:"a region of its axis" shapes_t (Ok [ [ 2 ] ])
    (assemble [| 2 |] [| [| r 0 2 1 |] |]);
  refuses ~msg:"a negative extent" (assemble [| -1 |] [||]);
  refuses ~msg:"a region of another rank"
    (assemble [| 2 |] [| [| r 0 1 1; r 0 1 1 |] |]);
  refuses ~msg:"a region past its axis" (assemble [| 2 |] [| [| r 1 2 1 |] |]);
  refuses ~msg:"a step of 0" (assemble [| 2 |] [| [| r 0 1 0 |] |]);
  equal ~msg:"an assembly of max_numel" shapes_t (Ok [])
    (assemble [| L.max_numel |] [||]);
  refuses ~msg:"an assembly past max_numel"
    (assemble [| L.max_numel + 1 |] [||])

(* Fourier transforms *)

type transform = C2c | R2c | C2r of int

let pp_transform ppf = function
  | C2c -> Format.pp_print_string ppf "c2c"
  | R2c -> Format.pp_print_string ppf "r2c"
  | C2r n -> Format.fprintf ppf "c2r %d" n

type fft_case = { t : transform; fft_axes : int array; x : int array }

let pp_fft_case ppf c =
  Format.fprintf ppf "%a along %a of %a" pp_transform c.t pp_ints c.fft_axes
    pp_ints c.x

(* A transform along a non-empty subset of an operand's axes, whose last
   axis holds, for most C2r, the bins its points need. *)
let fft_case =
  Gen.with_pp pp_fft_case
    (let open Gen in
     let* rank = int_range 1 4 in
     let* x = array ~size:(constant rank) (int_range 0 9) in
     let* keep = array ~size:(constant rank) bool in
     let* last = int_range 0 (rank - 1) in
     let fft_axes =
       Array.of_list
         (List.filter (fun i -> keep.(i) || i = last) (List.init rank Fun.id))
     in
     let* n = int_range 0 17 in
     let* t = of_list ~pp:pp_transform [ C2c; R2c; C2r n ] in
     let+ fits = bool in
     let top = fft_axes.(Array.length fft_axes - 1) in
     (match t with C2r n when fits -> x.(top) <- (n / 2) + 1 | _ -> ());
     { t; fft_axes; x })

let transform t ~axes x =
  let axes = Iarray.of_array axes in
  match t with
  | C2c -> Shape.c2c ~axes x
  | R2c -> Shape.r2c ~axes x
  | C2r n -> Shape.c2r ~axes ~n x

(* The result has the operand's shape, but along the last axis: n/2 + 1
   bins of r2c's n points, c2r's n points of exactly n/2 + 1 bins. *)
let law_fft c =
  let top = c.fft_axes.(Array.length c.fft_axes - 1) in
  let y = Array.copy c.x in
  let want =
    match c.t with
    | C2c -> Ok (Array.to_list y)
    | R2c ->
        y.(top) <- (c.x.(top) / 2) + 1;
        Ok (Array.to_list y)
    | C2r n when c.x.(top) = (n / 2) + 1 ->
        y.(top) <- n;
        Ok (Array.to_list y)
    | C2r _ -> misfit
  in
  cover "fits" (Result.is_ok want);
  cover "c2r" (match c.t with C2r _ -> true | _ -> false);
  equal shape_t want (of_shape (transform c.t ~axes:c.fft_axes c.x))

let test_fft_refuses () =
  let x = [| 2; 4 |] in
  List.iter
    (fun t ->
      let x = match t with C2r _ -> [| 2; 3 |] | _ -> x in
      let msg s = Format.asprintf "%a: %s" pp_transform t s in
      let refuses s axes = refuses ~msg:(msg s) (transform t ~axes x) in
      refuses "no axis" [||];
      refuses "axes out of order" [| 1; 0 |];
      refuses "a repeated axis" [| 1; 1 |];
      refuses "a negative axis" [| -1 |];
      refuses "an axis past the rank" [| 2 |];
      equal ~msg:(msg "both axes") bool true
        (Result.is_ok (transform t ~axes:[| 0; 1 |] x)))
    [ C2c; R2c; C2r 4 ];
  refuses ~msg:"a negative point count" (Shape.c2r ~axes:[| 0 |] ~n:(-1) [| 0 |]);
  refuses ~msg:"a negative point count of one bin"
    (Shape.c2r ~axes:[| 0 |] ~n:(-1) [| 1 |]);
  refuses ~msg:"too few bins" (Shape.c2r ~axes:[| 1 |] ~n:4 [| 2; 2 |]);
  refuses ~msg:"too many bins" (Shape.c2r ~axes:[| 1 |] ~n:4 [| 2; 4 |]);
  fits ~msg:"bins of an odd count" [ 2; 5 ]
    (Shape.c2r ~axes:[| 1 |] ~n:5 [| 2; 3 |]);
  fits ~msg:"no point is one bin" [ 2; 1 ] (Shape.r2c ~axes:[| 1 |] [| 2; 0 |]);
  fits ~msg:"one bin is no point" [ 2; 0 ]
    (Shape.c2r ~axes:[| 1 |] ~n:0 [| 2; 1 |])

(* The doc's pairs: each float32 or complex64, or each float64 or
   complex128. *)
let test_same_precision () =
  let precision (D.Any dt) =
    match dt with
    | D.Float32 | D.Complex64 -> Some 32
    | D.Float64 | D.Complex128 -> Some 64
    | _ -> None
  in
  List.iter
    (fun (D.Any x as dx) ->
      List.iter
        (fun (D.Any y as dy) ->
          let want =
            match (precision dx, precision dy) with
            | Some p, Some q -> p = q
            | _ -> false
          in
          equal
            ~msg:(D.name x ^ " with " ^ D.name y)
            bool want
            (Nx_kernel.same_precision x y))
        D.all)
    D.all

(* Matrices *)

type routine =
  | Cholesky
  | Lu
  | Qr of Nx_kernel.factors
  | Svd of Nx_kernel.factors
  | Eigh
  | Eig
  | Solve

let routines =
  [
    Cholesky;
    Lu;
    Qr Reduced;
    Qr Complete;
    Svd Reduced;
    Svd Complete;
    Eigh;
    Eig;
    Solve;
  ]

let pp_routine ppf r =
  let factors = function
    | Nx_kernel.Reduced -> "reduced"
    | Complete -> "complete"
  in
  Format.pp_print_string ppf
    (match r with
    | Cholesky -> "cholesky"
    | Lu -> "lu"
    | Qr f -> "qr " ^ factors f
    | Svd f -> "svd " ^ factors f
    | Eigh -> "eigh"
    | Eig -> "eig"
    | Solve -> "solve_triangular")

type linalg_case = { r : routine; a : int array; b : int array }

let pp_linalg_case ppf c =
  Format.fprintf ppf "%a of %a%s" pp_routine c.r pp_ints c.a
    (if c.r = Solve then Format.asprintf " and %a" pp_ints c.b else "")

(* A routine on a matrix of a batch of rank 0 to 2, square for most, with
   right-hand sides that fit for most solves, and operands of rank below 2
   now and then. A solve is drawn a quarter of the time: it is one routine
   of many, and the law needs fitting solves in every run. *)
let linalg_case =
  Gen.with_pp pp_linalg_case
    (let open Gen in
     let most = map (fun k -> k > 0) (int_range 0 3) in
     let others = List.filter (fun r -> r <> Solve) routines in
     let* r =
       frequency [ (3, of_list ~pp:pp_routine others); (1, constant ~pp:pp_routine Solve) ]
     in
     let* batch = array ~size:(int_range 0 2) (int_range 0 3) in
     let* m = int_range 0 5 in
     let* n = int_range 0 5 in
     let* square = most in
     let n = if square then m else n in
     let* cols = int_range 0 4 in
     let* fits = most in
     let* rank = int_range 0 1 in
     let a = Array.append batch [| m; n |] in
     let b = Array.append batch [| (if fits then n else n + 1); cols |] in
     let+ odd = int_range 0 9 in
     let a = if odd = 0 then Array.sub a 0 (min rank (Array.length a)) else a in
     { r; a; b })

let routine r a b =
  let one y = [ y ] in
  match r with
  | Cholesky -> Result.map one (Shape.cholesky a)
  | Lu -> Result.map (fun (l, p, q) -> [ l; p; q ]) (Shape.lu a)
  | Qr f -> Result.map (fun (q, r) -> [ q; r ]) (Shape.qr f a)
  | Svd f -> Result.map (fun (s, u, vh) -> [ s; u; vh ]) (Shape.svd f a)
  | Eigh | Eig -> Result.map (fun (w, v) -> [ w; v ]) (Shape.eigh a)
  | Solve -> Result.map one (Shape.solve_triangular a b)

(* The destinations by each routine's rule, for [a] of [batch; m; n]. *)
let law_linalg c =
  let want =
    let ra = Array.length c.a in
    if ra < 2 then misfit
    else
      let batch = Array.to_list (Array.sub c.a 0 (ra - 2)) in
      let m = c.a.(ra - 2) and n = c.a.(ra - 1) in
      let k = min m n in
      let y tail = batch @ tail in
      match c.r with
      | (Cholesky | Eigh | Eig | Solve) when m <> n -> misfit
      | Cholesky -> Ok [ y [ n; n ] ]
      | Eigh | Eig -> Ok [ y [ n ]; y [ n; n ] ]
      | Solve ->
          let b = Array.to_list c.b in
          let rows = List.filteri (fun i _ -> i < ra - 1) b in
          if List.length b = ra && rows = y [ n ] then Ok [ b ] else misfit
      | Lu -> Ok [ y [ m; n ]; y [ k ]; y [ m ] ]
      | Qr Reduced -> Ok [ y [ m; k ]; y [ k; n ] ]
      | Qr Complete -> Ok [ y [ m; m ]; y [ m; n ] ]
      | Svd Reduced -> Ok [ y [ k ]; y [ m; k ]; y [ k; n ] ]
      | Svd Complete -> Ok [ y [ k ]; y [ m; m ]; y [ n; n ] ]
  in
  let wide =
    match List.rev (Array.to_list c.a) with n :: m :: _ -> n > m | _ -> false
  in
  cover "fits" (Result.is_ok want);
  cover "solves" (c.r = Solve && Result.is_ok want);
  cover "wide" (wide && Result.is_ok want);
  equal shapes_t want (of_shapes (routine c.r c.a c.b))

(* The doc's table: float32, float64, complex64 and complex128. *)
let test_full_precision () =
  List.iter
    (fun (D.Any dt) ->
      let want =
        match dt with
        | D.Float32 | D.Float64 | D.Complex64 | D.Complex128 -> true
        | _ -> false
      in
      equal ~msg:(D.name dt) bool want (Nx_kernel.full_precision dt))
    D.all

(* Constructor codes *)

(* Each constructor's name in nx_kernel.h, by a match that a constructor
   added to Nx_kernel leaves short, and the type's constructors, as many as
   nx_kernel.h counts. *)
let load_name = function
  | Nx_kernel.Plain _ -> "NX_LOAD_PLAIN"
  | Padded _ -> "NX_LOAD_PADDED"

let monoid_name : Nx_kernel.monoid -> string = function
  | Sum -> "NX_MONOID_SUM"
  | Prod -> "NX_MONOID_PROD"
  | Max -> "NX_MONOID_MAX"
  | Min -> "NX_MONOID_MIN"

let reduction_name : Nx_kernel.reduction -> string = function
  | Monoid _ -> "NX_REDUCTION_MONOID"
  | Arg _ -> "NX_REDUCTION_ARG"
  | Logsumexp -> "NX_REDUCTION_LOGSUMEXP"
  | Moments -> "NX_REDUCTION_MOMENTS"

let combine_name : Nx_kernel.combine -> string = function
  | Set -> "NX_COMBINE_SET"
  | Add -> "NX_COMBINE_ADD"
  | Max -> "NX_COMBINE_MAX"
  | Min -> "NX_COMBINE_MIN"

let triangle_name : Nx_kernel.triangle -> string = function
  | Lower -> "NX_TRIANGLE_LOWER"
  | Upper -> "NX_TRIANGLE_UPPER"

(* Each constructor C reads by tag has the code nx_kernel.h declares, and
   each type as many as nx_kernel.h counts. *)
let test_codes () =
  let x = A.create Rig.host D.Float32 [| 1 |] in
  let pad =
    { Nx_kernel.lo = [| 0 |]; hi = [| 0 |]; interior = [| 0 |]; windows = [||] }
  in
  let code = Nx_kernel_support.code and tag = Nx_kernel_support.tag in
  let check count name all =
    List.iter (fun v -> equal ~msg:(name v) int (code (name v)) (tag v)) all;
    equal ~msg:count int (code count) (List.length all)
  in
  check "NX_LOAD_COUNT" load_name
    [ Nx_kernel.Plain x; Padded { x; fill = 0.; pad } ];
  check "NX_MONOID_COUNT" monoid_name [ Sum; Prod; Max; Min ];
  check "NX_REDUCTION_BLOCKS" reduction_name [ Monoid Sum; Arg Max ];
  check "NX_REDUCTION_CONSTANTS" reduction_name [ Logsumexp; Moments ];
  check "NX_COMBINE_COUNT" combine_name [ Set; Add; Max; Min ];
  check "NX_TRIANGLE_COUNT" triangle_name [ Lower; Upper ]

(* Contraction views *)

(* An array of float32 over [l], on a host buffer that holds its positions. *)
let over l =
  let hi = snd (L.span l) in
  A.v D.Float32 l (Rig.Buffer.create Rig.host (4 * (hi + 1)))

(* [c]'s operands [a], [b] and [init], and its destination. *)
let operands c = (over c.la, over c.lb, Option.map over c.li, over c.ly)

(* What a view answers, as text: whether it groups, or why not. *)
let outcome = function
  | Ok _ -> "grouped"
  | Error A.Declined -> "declined"
  | Error A.Shape_mismatch -> "shape mismatch"
  | Error _ -> "another answer"

let view c =
  let batch, contracting = pairs_of c in
  let a, b, init, dst = operands c in
  V.v ~batch ~contracting ~dst ~init a b

(* Each operand's axes of each group, by the rule. *)
let groups c =
  let batch, contracting = pairs_of c in
  let ra = L.rank c.la and rb = L.rank c.lb in
  let side f ps = Iarray.to_list (Iarray.map f ps) in
  let fa = free ra (side fst batch @ side fst contracting) in
  let fb = free rb (side snd batch @ side snd contracting) in
  let nb = Iarray.length batch and nfa = List.length fa in
  let y =
    [
      (V.Batch, List.init nb Fun.id);
      (V.Row, List.init nfa (fun k -> nb + k));
      (V.Column, List.init (List.length fb) (fun k -> nb + nfa + k));
    ]
  in
  [
    ( V.A,
      c.la,
      [
        (V.Batch, side fst batch);
        (V.Row, fa);
        (V.Contracted, side fst contracting);
      ] );
    ( V.B,
      c.lb,
      [
        (V.Batch, side snd batch);
        (V.Column, fb);
        (V.Contracted, side snd contracting);
      ] );
    (V.Dst, c.ly, y);
  ]
  @ match c.li with None -> [] | Some l -> [ (V.Init, l, y) ]

let pp_operand ppf o =
  Format.pp_print_string ppf
    (match o with V.A -> "a" | V.B -> "b" | V.Init -> "init" | V.Dst -> "dst")

(* Every element of every operand lies where the view places it: its group's
   index, flat in C order over the group's axes, times the group's stride. An
   axis an operand lacks, and an absent init, read 0. *)
let check_view c v =
  let zero ~msg n = equal ~msg int 0 n in
  zero ~msg:"a's column stride" (V.stride v V.A V.Column);
  zero ~msg:"b's row stride" (V.stride v V.B V.Row);
  zero ~msg:"dst's contracted stride" (V.stride v V.Dst V.Contracted);
  zero ~msg:"init's contracted stride" (V.stride v V.Init V.Contracted);
  if Option.is_none c.li then begin
    zero ~msg:"an absent init's offset" (V.offset v V.Init);
    List.iter
      (fun x -> zero ~msg:"an absent init's stride" (V.stride v V.Init x))
      V.[ Batch; Row; Column; Contracted ]
  end;
  List.iter
    (fun (o, l, gs) ->
      let s = L.shape l in
      List.iter
        (fun (g, axes) ->
          equal ~msg:"extent" int
            (List.fold_left (fun n ax -> n * s.(ax)) 1 axes)
            (V.extent v g))
        gs;
      List.iter
        (fun idx ->
          let at =
            List.fold_left
              (fun p (g, axes) ->
                let flat =
                  List.fold_left (fun f ax -> (f * s.(ax)) + idx.(ax)) 0 axes
                in
                p + (flat * V.stride v o g))
              (V.offset v o) gs
          in
          equal
            ~msg:(Format.asprintf "%a at %a" pp_operand o pp_ints idx)
            int (position l idx) at)
        (indices s))
    (groups c)

let law_view c =
  let v = view c in
  cover "groups" (Result.is_ok v);
  cover "declines" (Result.is_error v);
  not_equal ~msg:"a misfit" string "shape mismatch" (outcome v);
  Result.iter (check_view c) v

let law_canonical c =
  let v = view c in
  equal ~msg:"groups" string "grouped" (outcome v);
  cover "without init" (Option.is_none c.li);
  Result.iter (check_view c) v

(* [l] with two axes of extent 1 and stride [s] after its last. *)
let with_unit_axes s l =
  L.v ~offset:(L.offset l)
    ~strides:(Array.append (L.strides l) [| s; s |])
    (Array.append (L.shape l) [| 1; 1 |])

(* [l]'s shape, C-contiguous, with axis [ax] one longer; [l] where it has no
   such axis. *)
let longer ax l =
  let sh = L.shape l in
  if ax < 0 || ax >= Array.length sh then l
  else L.contiguous (Array.mapi (fun i e -> if i = ax then e + 1 else e) sh)

(* A view of groups of at most one axis each equals the view of the same
   contraction with two unit axes more in its contracted group, which the
   general grouping takes, and refuses where it refuses. [misfit] picks
   what does not fit: nothing, dst's first axis, init's, a's first axis,
   or b's axis in the first batch pair. *)
let law_one_axis (c, (misfit, s)) =
  let batch, contracting = pairs_of c in
  let la, lb, ly, li =
    match misfit with
    | 1 -> (c.la, c.lb, longer 0 c.ly, c.li)
    | 2 -> (c.la, c.lb, c.ly, Option.map (longer 0) c.li)
    | 3 -> (longer 0 c.la, c.lb, c.ly, c.li)
    | 4 when Iarray.length batch > 0 ->
        (c.la, longer (snd (Iarray.get batch 0)) c.lb, c.ly, c.li)
    | _ -> (c.la, c.lb, c.ly, c.li)
  in
  let dst = over ly and init = Option.map over li in
  let v = V.v ~batch ~contracting ~dst ~init (over la) (over lb) in
  let ra = L.rank la and rb = L.rank lb in
  let v' =
    V.v ~batch
      ~contracting:
          (Iarray.append contracting [| (ra, rb); (ra + 1, rb + 1) |])
      ~dst ~init
      (over (with_unit_axes s la))
      (over (with_unit_axes s lb))
  in
  let one =
    List.for_all
      (fun g -> Array.length g <= 1)
      [ c.batch; c.rows; c.columns; c.contracted ]
  in
  cover "one-axis groups, grouped" (one && Result.is_ok v);
  cover "one-axis groups, refused" (one && Result.is_error v);
  cover "one-axis groups, a batch misfit between a and b"
    (one && misfit = 4 && Iarray.length batch > 0);
  cover "one-axis groups, an empty operand"
    (one && (L.numel la = 0 || L.numel lb = 0));
  cover "one-axis groups, a batch axis" (one && Array.length c.batch = 1);
  equal ~msg:"answers" string (outcome v') (outcome v);
  match (v, v') with
  | Ok v, Ok v' -> equal (array int) v' v
  | _ -> ()

(* A view refuses operands that do not fit its pairs, and declines a call
   whose group does not merge. *)
let test_view_declines () =
  let arr s = over (L.contiguous s) in
  let view ?(batch : _ iarray = [||]) ?(contracting : _ iarray = [| (1, 0) |])
      ?init ~dst a b =
    V.v ~batch ~contracting ~dst ~init a b
  in
  let mm ?batch ?contracting ?init ~dst a b =
    outcome (view ?batch ?contracting ?init ~dst a b)
  in
  equal ~msg:"fits" string "grouped"
    (mm ~dst:(arr [| 2; 3 |]) (arr [| 2; 4 |]) (arr [| 4; 3 |]));
  let misfit ~msg f = equal ~msg string "shape mismatch" (f ()) in
  misfit ~msg:"contracted extents" (fun () ->
      mm ~dst:(arr [| 2; 3 |]) (arr [| 2; 4 |]) (arr [| 5; 3 |]));
  misfit ~msg:"dst's shape" (fun () ->
      mm ~dst:(arr [| 3; 2 |]) (arr [| 2; 4 |]) (arr [| 4; 3 |]));
  misfit ~msg:"a's rank" (fun () ->
      mm ~dst:(arr [| 2; 3 |]) (arr [| 4 |]) (arr [| 4; 3 |]));
  misfit ~msg:"init's shape" (fun () ->
      mm ~dst:(arr [| 2; 3 |]) ~init:(arr [| 3; 2 |]) (arr [| 2; 4 |])
        (arr [| 4; 3 |]));
  misfit ~msg:"a pair past a's rank" (fun () ->
      mm ~contracting:[| (2, 0) |] ~dst:(arr [| 2; 3 |]) (arr [| 2; 4 |])
        (arr [| 4; 3 |]));
  misfit ~msg:"a pair past b's rank" (fun () ->
      mm ~contracting:[| (1, 2) |] ~dst:(arr [| 2; 3 |]) (arr [| 2; 4 |])
        (arr [| 4; 3 |]));
  let sq = arr [| 4; 4 |] in
  equal ~msg:"distinct pairs" string "grouped"
    (mm ~batch:[| (0, 0) |] ~contracting:[| (1, 1) |] ~dst:(arr [| 4 |]) sq sq);
  misfit ~msg:"an axis of a in two pairs" (fun () ->
      mm ~batch:[| (0, 0) |] ~contracting:[| (0, 1) |] ~dst:(arr [| 4 |]) sq sq);
  misfit ~msg:"an axis of b in two pairs" (fun () ->
      mm ~batch:[| (0, 0) |] ~contracting:[| (1, 0) |] ~dst:(arr [| 4 |]) sq sq);
  (* A misfit is refused even behind a group that does not merge: [a]'s
     batch axes transposed, its contracted extent 4 against [b]'s 5. *)
  let at =
    Option.get
      (L.move (M.Permute [| 1; 0; 2; 3 |]) (L.contiguous [| 3; 2; 2; 4 |]))
  in
  let a = A.v D.Float32 at (Rig.Buffer.create Rig.host 192) in
  misfit ~msg:"behind an unmerged group" (fun () ->
      mm
        ~batch:[| (0, 0); (1, 1) |]
        ~contracting:[| (3, 2) |]
        ~dst:(arr [| 2; 3; 2; 2 |])
        a
        (arr [| 2; 3; 5; 2 |]));
  (* [b]'s two contracted axes transposed lie as no run. *)
  let bt =
    Option.get (L.move (M.Permute [| 1; 0; 2 |]) (L.contiguous [| 3; 4; 5 |]))
  in
  let b = A.v D.Float32 bt (Rig.Buffer.create Rig.host 240) in
  equal ~msg:"no run" string "declined"
    (mm ~contracting:[| (1, 0); (2, 1) |] ~dst:(arr [| 2; 5 |])
       (arr [| 2; 4; 3 |]) b);
  let a =
    A.v D.Float32 (L.v ~strides:[| 1; 8 |] [| 2; 4 |])
      (Rig.Buffer.create Rig.host 128)
  in
  let v = view ~dst:(arr [| 2; 3 |]) a (arr [| 4; 3 |]) in
  equal ~msg:"strided contraction" string "grouped" (outcome v);
  let v = Result.get_ok v in
  equal int 8 (V.stride v V.A V.Contracted)

let minor_words f =
  let before = Gc.minor_words () in
  f ();
  Gc.minor_words () -. before

(* nx.cpu's nx_contract_fill, which groups each contraction, allocates
   nothing. *)
let test_view_allocates_nothing () =
  let arr s = over (L.contiguous s) in
  let a = arr [| 2; 3; 4 |] and b = arr [| 2; 4; 5 |] in
  let init = Some (arr [| 2; 3; 5 |]) and dst = arr [| 2; 3; 5 |] in
  let batch : _ iarray = [| (0, 0) |] and contracting : _ iarray = [| (2, 1) |] in
  let calls () =
    for _ = 1 to 1000 do
      ignore
        (Nx_kernels_support.view_groups ~batch ~contracting ~dst ~init a b)
    done
  in
  let nothing () = () in
  calls ();
  equal float_exact (minor_words nothing) (minor_words calls)

let tests =
  [
    group "contractions"
      [
        prop "the result is batch, then a's free axes, then b's" any_case
          law_contract;
        test
          "refuses extents that differ, axes past a rank or in two pairs, \
           and results past the most rank"
          test_contract_refuses;
        test
          "refuses a scale of another rank, or extents that do not divide"
          test_contract_scales;
        test
          "shapes segments' results and refuses their offsets of other \
           shapes or axes in pairs"
          test_contract_segments;
        test "admits every shape it derives by max_numel" test_admits;
        test "accumulates states the accumulators" test_accumulates;
      ];
    group "paddings"
      [
        prop "padded is the padded extents, then the windows" padded_case
          law_padded;
        test "refuses paddings that do not fit" test_padded_refuses;
      ];
    group "reductions and scans"
      [
        prop "a reduction drops its axes; an extreme needs a term" reduce_case
          law_reduce;
        test "refuses axes that do not fit and extremes of no term"
          test_reduce_refuses;
        test "accepts states each reduction's dtypes" test_accepts;
      ];
    group "gathers, scatters and sorts"
      [
        test "refuse axes outside the rank and a negative k" test_index_refuses;
        test "state each family's shapes" test_index_shapes;
      ];
    group "assemblies"
      [
        prop "each piece has its region's shape" assemble_case law_assemble;
        test "refuses shapes and regions that do not fit" test_assemble_refuses;
      ];
    group "transforms"
      [
        prop "results keep the shape but along the last axis" fft_case law_fft;
        test "refuses axes and bins that do not fit" test_fft_refuses;
        test "same_precision pairs the dtypes of one precision"
          test_same_precision;
      ];
    group "matrices"
      [
        (* A wide matrix that fits its routine is about 1 case in 26, so
           100 cases missed "wide" about 2% of runs; 1000 miss it about
           once in 10^17. *)
        prop ~count:1000 "results are each routine's, after the batch"
          linalg_case law_linalg;
        test "full_precision states the routines' dtypes" test_full_precision;
      ];
    group "constructor codes"
      [ test "each constructor C reads has nx_kernel.h's code" test_codes ];
    group "contract view"
      [
        prop "places every element where its layout does" any_case law_view;
        prop "groups C-contiguous operands of any axis order" canonical_case
          law_canonical;
        test "refuses what does not fit and declines an unmerged group"
          test_view_declines;
        (* One-axis groups are about 1 draw in 5, a batch misfit among
           them 1 in 10: 1000 draws reach it about 20 times. *)
        prop ~count:1000 "a view of one-axis groups is the general grouping's"
          (Gen.pair any_case (Gen.pair (Gen.int_range 0 4) (Gen.int_range 0 3)))
          law_one_axis;
        test "C's fill allocates nothing" test_view_allocates_nothing;
      ];
  ]

let () = exit (run "nx_kernel.shape" tests)
