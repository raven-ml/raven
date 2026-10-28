(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What the suites of Nx_kernel.S's contractions and of nx.cpu's share: the
   values, the cases, and the reference that reads their operands as float64
   planes. *)

open Windtrap
open Nx_array_gen
module A = Nx_array
module D = Nx_array.Dtype
module Support = Nx_kernels_support
module E = Elements

let pp_dtype ppf (D.Any dt) = D.pp ppf dt

(* An operand without a scale. *)
let plain x : Nx_kernel.operand = { array = A.Any x; scale = None }

let pp_any ppf (A.Any x) =
  Format.fprintf ppf "%a %a" D.pp (A.dtype x) L.pp (A.layout x)

let shape_of (A.Any x) = Iarray.to_array (L.shape (A.layout x))

let total s = Array.fold_left ( * ) 1 s

(* [x] cast into [d], by nx.cpu. *)
let cast_into d (A.Any x) =
  match Support.map1 (module Nx_cpu) Cast ~dst:d x with
  | A.Done -> ()
  | r -> failf "a cast answered %a" Nx_array_support.pp_answer r

(* [x] cast to [dt], C-contiguous. *)
let cast_to (D.Any dt) x =
  let d = A.create Rig.host dt (Iarray.of_array (shape_of x)) in
  cast_into d x;
  A.Any d

(* [x] where the host reads it: [x] itself on the host, a copy elsewhere. *)
let host (A.Any x as a) =
  if Rig.equal (A.device x) Rig.host then a
  else A.Any (A.to_device Rig.host x)

(* [x] on [b]'s device, its layout kept. *)
let on (b : Support.backend) (A.Any x as a) =
  if Rig.equal (A.device x) b.device then a
  else A.Any (A.to_device b.device x)

(* [x] as float64: every value of a dtype a contraction takes is one of
   float64's. *)
let to64 x =
  let d = A.create Rig.host D.Float64 (Iarray.of_array (shape_of x)) in
  cast_into d x;
  d

(* The dtype of [dt]'s parts: a complex dtype's, or [dt] itself. *)
let part_of (D.Any dt as d) =
  match dt with
  | D.Complex64 -> D.Any D.Float32
  | D.Complex128 -> D.Any D.Float64
  | _ -> d

let is_complex acc = part_of acc <> acc

let is_integer (D.Any dt) =
  match D.kind dt with D.Signed | D.Unsigned -> true | _ -> false

(* [x]'s values read in [acc] as float64 planes in C order: one, or the real
   then the imaginary parts for a complex [acc]. *)
let planes acc x =
  if not (is_complex acc) then [| A.to_array (to64 x) |]
  else
    let d = A.create Rig.host D.Complex128 (Iarray.of_array (shape_of x)) in
    cast_into d x;
    let z = A.to_array d in
    [|
      Array.map (fun z -> z.Complex.re) z; Array.map (fun z -> z.Complex.im) z;
    |]

(* Values *)

(* Values every float format holds, float4 included. *)
let grid = [| 0.; -0.; 0.5; -0.5; 1.; -1.; 1.5; -1.5; 2.; -3.; 4.; -6. |]

(* A value of [dt] from [r]: on the grid, an integer in the range of every
   integer dtype of its kind, or, for float32 and float64 and each part of
   complex64 and complex128, one of 24 or 53 significant bits in (-4, 4),
   the same scaled to the smallest normal's exponent, subnormal or near it,
   an infinity or a NaN. *)
let value (D.Any dt) r =
  let pick a = a.(Random.State.int r (Array.length a)) in
  let full p =
    let m = Random.State.int64 r (Int64.shift_left 1L (p + 1)) in
    Float.ldexp (Int64.to_float (Int64.sub m (Int64.shift_left 1L p))) (2 - p)
  in
  let float p =
    match Random.State.int r 40 with
    | 0 -> pick [| infinity; neg_infinity; nan |]
    | 1 -> Float.ldexp (full p) (if p = 23 then -126 else -1022)
    | n when n < 15 -> pick grid
    | _ -> full p
  in
  match (D.kind dt, dt) with
  | D.Boolean, _ -> float_of_int (Random.State.int r 2)
  | _, (D.Complex64 | D.Float32) -> float 23
  | _, D.Complex128 -> float 52
  | D.Unsigned, _ -> float_of_int (Random.State.int r 16)
  | D.Signed, _ -> float_of_int (Random.State.int r 16 - 8)
  | _, D.Float64 -> float 52
  | _ -> pick grid

(* Cases *)

(* A contraction: its pairs, accumulator, operands and result dtype, and
   the views its operands were drawn through. [init] has the result's
   dtype. *)
type case = {
  batch : (int * int) array;
  contracting : (int * int) array;
  acc : D.any;
  a : A.any;
  b : A.any;
  init : A.any option;
  out : D.any;
  views : string list;
}

let pp_case ppf c =
  let pp_init ppf = function
    | None -> ()
    | Some i -> Format.fprintf ppf "; init %a" pp_any i
  in
  Format.fprintf ppf "batch %a contracting %a acc %a out %a; a %a; b %a%a"
    pp_ints (Array.map fst c.batch) pp_ints
    (Array.map fst c.contracting)
    pp_dtype c.acc pp_dtype c.out pp_any c.a pp_any c.b pp_init c.init

type view = E.view = {
  perm : int array;
  stored : int array;
  stepped : bool;
  windowed : (int * int) option;
  reversed : int option;
  broadcast : int list;
}

(* A view of an operand of rank [r] whose first [nb] made axes are a
   batch's: broadcast along the batch at times, windows only where the last
   two made axes are not the batch's. *)
let view r nb =
  let open Gen in
  let* v = E.view_of ~windows:(r - 2 >= nb) ~apart:true r in
  let+ broadcast = if nb > 0 then bool else constant false in
  if not broadcast then v
  else
    {
      v with
      broadcast = List.filter (fun i -> v.perm.(i) < nb) (List.init r Fun.id);
    }

(* An int64 that narrows to the extremes of every integer width and their
   neighbours, or any int64. *)
let wide r =
  let pick a = a.(Random.State.int r (Array.length a)) in
  match Random.State.int r 3 with
  | 0 -> Random.State.bits64 r
  | 1 -> Int64.of_int (Random.State.int r 16 - 8)
  | _ ->
      pick
        [| Int64.min_int; Int64.max_int; 0x7FL; 0x80L; 0xFFL; 0x7FFFL; 0x8000L;
           0xFFFFL; 0x7FFFFFFFL; 0x80000000L; 0xFFFFFFFFL; -1L |]

(* An array of [dt] whose made shape is [s], axis [v.perm.(i)] of the
   made array being axis [i] of the result, through the view [v], its
   values, or each part of them, drawn from [seed]; under [wide] an
   integer's or a bool's from {!wide}, narrowed modulo its width. *)
let operand ?(wide_ints = false) dt s v seed =
  let r = Random.State.make [| seed |] in
  let draw () = value dt r in
  let make made =
    let ms = Iarray.of_array made in
    let xs =
      if
        wide_ints && (is_integer dt || dt = D.Any D.Bool || dt = D.Any D.Bit)
      then
        A.Any
          (A.of_array D.Int64 ms (Array.init (total made) (fun _ -> wide r)))
      else if is_complex dt then
        A.Any
          (A.of_array D.Complex128 ms
             (Array.init (total made) (fun _ ->
                  let re = draw () in
                  { Complex.re; im = draw () })))
      else
        A.Any
          (A.of_array D.Float64 ms (Array.init (total made) (fun _ -> draw ())))
    in
    cast_to dt xs
  in
  E.operand_with make (Array.map (fun m -> s.(m)) v.perm) v

(* Extents of the batch, row, column and contracted axes, in profiles: any
   ranks over a few elements; one output; 64 outputs or more; a depth of
   several blocks of lanes; several thousand columns; few rows over a deep
   contraction; few columns; a batch of several hundred elements; two axes
   in each group; and a batch of two axes over 512 KiB of operands or
   more. *)
let extents =
  let open Gen in
  let one lo hi = map (fun e -> [| e |]) (int_range lo hi) in
  let shaped batch rows cols con =
    let+ batch = batch and+ rows = rows and+ cols = cols and+ con = con in
    (batch, rows, cols, con)
  in
  let axes most lo hi = array ~size:(int_range 0 most) (int_range lo hi) in
  let two lo hi = array ~size:(constant 2) (int_range lo hi) in
  let pair x y = map (fun (x, y) -> [| x; y |]) (Gen.pair x y) in
  let ranked = shaped (axes 2 0 3) (axes 2 0 5) (axes 2 0 5) (axes 2 0 4) in
  let none = constant [||] in
  frequency
    [
      (6, ranked);
      (1, shaped (axes 1 1 1) (axes 2 1 1) (axes 2 1 1) (axes 2 0 6));
      (3, shaped none (one 4 40) (one 8 40) (one 0 40));
      (1, shaped none (one 4 20) (one 8 24) (one 500 1300));
      (1, shaped (one 1 3) (one 1 2) (one 1 2) (one 1000 5000));
      (1, shaped none (one 1 2) (one 3070 3200) (one 1 6));
      (1, shaped none (one 1 4) (one 16 200) (one 400 1100));
      (1, shaped none (one 16 300) (one 1 4) (one 1 600));
      (2, shaped (one 342 420) (constant [| 8 |]) (constant [| 8 |]) (one 1 4));
      (8, shaped (two 2 3) (two 2 3) (two 2 3) (two 2 3));
      (4, shaped (pair (int_range 16 24) (int_range 2 3)) (one 48 48)
            (one 48 48) (one 48 56));
    ]

(* The ranks of [a], [b] and the result for the extents [e]. *)
let ranks (batch, rows, cols, con) =
  let nb = Array.length batch and nk = Array.length con in
  (nb + Array.length rows + nk, nb + nk + Array.length cols,
   nb + Array.length rows + Array.length cols)

(* The case of the extents [e] in [acc] into [out]: [a] and [b]
   of the dtypes and through the views given, an init of [out] through the
   view [init] gives, values drawn from [seed]. *)
let make ~acc ~out (batch, rows, cols, con) (da, va) (db, vb) init seed =
  let nb = Array.length batch and fa = Array.length rows in
  let sa = Array.concat [ batch; rows; con ] in
  let sb = Array.concat [ batch; con; cols ] in
  let ia = E.inverse va.perm and ib = E.inverse vb.perm in
  let pairs = Array.init nb (fun k -> (ia.(k), ib.(k))) in
  let contracting =
    Array.init (Array.length con) (fun k -> (ia.(nb + fa + k), ib.(nb + k)))
  in
  let wide_ints = is_integer acc in
  let a = operand ~wide_ints da sa va seed in
  let b = operand ~wide_ints db sb vb (seed + 1) in
  let y =
    match
      Nx_kernel.Shape.contract ~batch:(Iarray.of_array pairs)
        ~contracting:(Iarray.of_array contracting) ~segments:None
        (shape_of a, None) (shape_of b, None)
    with
    | Ok y -> y
    | Error e -> invalid_arg ("make: " ^ e)
  in
  let made vi =
    operand ~wide_ints out (Array.map (fun i -> y.(i)) (E.inverse vi.perm)) vi
      (seed + 2)
  in
  let names = E.view_names in
  let views = names va @ names vb @ Option.fold ~none:[] ~some:names init in
  { batch = pairs; contracting; acc; a; b; init = Option.map made init; out;
    views }

(* A case in [acc] into [out], each operand of [acc] or of [dts]. *)
let case_of ?same ~acc ~out ~dts () =
  let open Gen in
  let* ((batch, _, _, _) as e) = extents in
  let nb = Array.length batch in
  let ra, rb, ry = ranks e in
  let pick = frequency [ (3, constant ~pp:pp_dtype acc); (2, of_list ~pp:pp_dtype dts) ] in
  let* da = match same with Some d -> d | None -> pick in
  let* db = if same = None then pick else constant ~pp:pp_dtype da in
  let* va = view ra nb in
  let* vb = view rb nb in
  let* vi = view ry 0 in
  let* with_init = bool in
  let+ seed = int in
  make ~acc ~out e (da, va) (db, vb)
    (if with_init then Some vi else None)
    seed

(* The case of the extents [e] in [acc], C-contiguous, with an init where
   [init]. *)
let case_in acc (init, e) =
  let ra, rb, ry = ranks e in
  let ti = if init then Some (E.plain ry) else None in
  make ~acc ~out:acc e (acc, E.plain ra) (acc, E.plain rb) ti 1

(* The view that swaps the last two axes. *)
let swap r =
  { (E.plain r) with
    perm = Array.init r (fun i -> if i >= r - 2 then (2 * r) - 3 - i else i) }

(* {!case_in} with a and b stored transposed. *)
let apart acc (init, e) =
  let copy (A.Any x) = A.Any (A.copy x) in
  let ra, rb, ry = ranks e in
  let ti = if init then Some (E.plain ry) else None in
  let c = make ~acc ~out:acc e (acc, swap ra) (acc, swap rb) ti 1 in
  { c with a = copy c.a; b = copy c.b }

(* Sizes in [acc] the drawn cases reach a few times in a hundred or never:
   no output, one, no product; either side of 4 rows and of 4 columns, of
   64 outputs and of a depth of 128 outputs' worth, where the order turns;
   depths of one and of several blocks of lanes; rows and columns of
   hundreds, a batch of hundreds, and in float64 a batch of 135 million
   products; 2 rows by 1 to 64 columns at depths either side of a block's
   lanes; convolutions, a's windows of an image; each operand stored
   transposed, their axes stored apart, attention's heads; and an a of a
   dtype [acc] does not hold. *)
let regimes acc =
  let plain = E.plain in
  let case = case_in acc and apart = apart acc in
  let copy (A.Any x) = A.Any (A.copy x) in
  (* [case] with a alone stored transposed: few rows pack it, and blocks of
     its rows read the pack from their first row. *)
  let a_apart (init, e) =
    let ra, rb, ry = ranks e in
    let ti = if init then Some (plain ry) else None in
    let c = make ~acc ~out:acc e (acc, swap ra) (acc, plain rb) ti 1 in
    { c with a = copy c.a }
  in
  (* [case] with a and b stored in the orders of their made axes [sa] and
     [sb] name. *)
  let stored (init, e, sa, sb) =
    let _, _, ry = ranks e in
    let ti = if init then Some (plain ry) else None in
    let at s = { (plain (Array.length s)) with stored = s } in
    make ~acc ~out:acc e (acc, at sa) (acc, at sb) ti 1
  in
  (* Contracted axes stored in another order in a than in b, with init and
     without. *)
  let split =
    List.map stored
      [
        (true, ([||], [| 3 |], [| 4 |], [| 5; 6 |]), [| 0; 2; 1 |],
         [| 0; 1; 2 |]);
        (false, ([||], [| 3 |], [| 4 |], [| 5; 6 |]), [| 0; 2; 1 |],
         [| 0; 1; 2 |]);
      ]
  in
  (* b's two axes of columns stored in the other order, with init and
     without: several runs that b alone makes. *)
  let columns =
    List.map stored
      [
        (true, ([||], [| 3 |], [| 4; 5 |], [| 6 |]), [| 0; 1 |],
         [| 0; 2; 1 |]);
        (false, ([||], [| 3 |], [| 4; 5 |], [| 6 |]), [| 0; 1 |],
         [| 0; 2; 1 |]);
      ]
  in
  (* Attention's scores, q and k of [b; s; h; d] read as [b; h; s; d]: a
     batch of two runs, the outer of 16 positions, over 768 KiB of
     operands, which the threads take a position at a time. *)
  let heads =
    stored
      ( false, ([| 16; 4 |], [| 32 |], [| 32 |], [| 32 |]),
        [| 0; 2; 1; 3 |], [| 0; 3; 1; 2 |] )
  in
  (* a of a dtype acc does not hold, cast into it: float64 into float32 or
     an integer, complex128 into complex64, int64 into float64 and
     complex128. *)
  let unheld =
    let wide =
      match acc with
      | D.Any D.Float32 -> D.Any D.Float64
      | D.Any D.Complex64 -> D.Any D.Complex128
      | D.Any (D.Float64 | D.Complex128) -> D.Any D.Int64
      | _ -> D.Any D.Float64
    in
    let e = ([||], [| 3 |], [| 4 |], [| 5 |]) in
    make ~acc ~out:acc e (wide, plain 2) (acc, plain 2) None 1
  in
  (* A convolution's product: a the windows of an image [n; h; w; c] by
     [kh × kw] taps at [stride] and [dilation], [n; oh; ow; c; kh; kw],
     contracted with b, [kh; kw; c; o], over the taps and channels, in the
     order (kh, kw, c) where [taps_first], whose last two make one run of a
     at dilation 1, else (c, kh, kw), each its own run of a and of b. *)
  let conv ?(da = acc) ?(init = false) ~taps_first (n, oh, ow, c, o)
      (kh, kw) (stride, dilation) =
    let extent t k = ((t - 1) * stride) + (dilation * (k - 1)) + 1 in
    let (A.Any x) =
      operand da [| n; extent oh kh; extent ow kw; c |] (plain 4) 3
    in
    let win axis size : M.window = { axis; size; step = stride; dilation } in
    let a = A.Any (Option.get (A.move (M.Window [| win 1 kh; win 2 kw |]) x)) in
    let b = operand acc [| kh; kw; c; o |] (plain 4) 4 in
    let contracting =
      if taps_first then [| (4, 0); (5, 1); (3, 2) |]
      else [| (3, 2); (4, 0); (5, 1) |]
    in
    let init =
      if init then Some (operand acc [| n; oh; ow; o |] (plain 4) 5)
      else None
    in
    { batch = [||]; contracting; acc; a; b; init; out = acc;
      views = [ "windowed" ] }
  in
  (* Many rows by few columns, b's windows streamed; few rows by more
     columns, a's packed whole; dilated taps; the chain path over two
     blocks of the contraction (float32's alone, for its cost); lane
     order, in one pass and in two over 9 rows of several runs, and over
     windows whose last contracted run is shorter than the lanes and as
     long as 48; a of a dtype acc does not hold; and a complex one's a
     real. *)
  let convs =
    [
      conv ~taps_first:true (2, 10, 10, 3, 8) (3, 3) (1, 1);
      conv ~init:true ~taps_first:false (1, 4, 4, 5, 40) (3, 3) (2, 1);
      conv ~taps_first:true (1, 4, 5, 4, 30) (2, 3) (1, 2);
      conv ~init:true ~taps_first:true (1, 3, 3, 4, 3) (3, 3) (1, 1);
      conv ~taps_first:true (1, 3, 3, 4, 16) (3, 3) (1, 1);
      conv ~taps_first:true (1, 3, 3, 16, 3) (3, 3) (1, 1);
    ]
    @ (match acc with
      | D.Any D.Float32 ->
          [
            conv ~taps_first:true (1, 13, 20, 29, 257) (3, 3) (1, 1);
            conv ~da:(D.Any D.Bfloat16) ~taps_first:false
              (2, 6, 6, 3, 16) (3, 3) (1, 1);
          ]
      | D.Any D.Float64 ->
          (* A pack crossing its pitch: a block of 48 rows, windows 12
             apart, whose last run of rows, from row 36, the matrix unit's
             transposes of 16 rows would write up to row 52. *)
          [ conv ~taps_first:true (1, 4, 12, 9, 32) (3, 3) (1, 1) ]
      | D.Any (D.Complex64 | D.Complex128) ->
          [
            conv ~da:(part_of acc) ~taps_first:true (1, 4, 4, 3, 8)
              (3, 3) (1, 1);
          ]
      | _ -> [])
  in
  let rows = 257 in
  let few = (true, ([||], [| 256 |], [| 259 |], [| 3 |])) in
  let complex =
    [
      (true, ([||], [| 3 |], [| 64 |], [| 600 |]));
      (false, ([||], [| 40 |], [| 4 |], [| 600 |]));
      (true, ([||], [| 40 |], [| 17 |], [| 600 |]));
      (true, ([||], [| 3 |], [| 2 |], [| 0 |]));
      (false, ([| 50 |], [| 8 |], [| 8 |], [| 2 |]));
    ]
  in
  let integer = (true, ([||], [| 130 |], [| 130 |], [| 520 |])) :: complex in
  if is_complex acc then
    (apart few :: unheld :: split) @ columns @ convs @ List.map case complex
  else if is_integer acc then
    (apart few :: unheld :: split) @ columns @ convs @ List.map case integer
  else
    [ apart few; heads; unheld;
      a_apart (false, ([| 2 |], [| 20 |], [| 20 |], [| 500 |]));
      (* Lane order's passes over rows of two runs. *)
      stored (true, ([||], [| 3; 3 |], [| 12 |], [| 13900 |]), [| 1; 0; 2 |],
              [| 0; 1 |]) ]
    @ split @ columns @ convs
    @ List.map case
      ([
         (true, ([||], [| 0 |], [| 3 |], [| 2 |]));
         (true, ([| 1 |], [| 1 |], [| 1 |], [| 5 |]));
         (true, ([||], [| 3 |], [| 2 |], [| 0 |]));
         (true, ([||], [| 3 |], [| 64 |], [| 600 |]));
         (true, ([||], [| 64 |], [| 3 |], [| 600 |]));
         (true, ([||], [| 4 |], [| 40 |], [| 1100 |]));
         (true, ([||], [| 5 |], [| 40 |], [| 1100 |]));
         (true, ([||], [| 40 |], [| 4 |], [| 1100 |]));
         (false, ([||], [| 40 |], [| 5 |], [| 1100 |]));
         (true, ([||], [| 40 |], [| 40 |], [| 1100 |]));
         (true, ([||], [| 40 |], [| 17 |], [| 1100 |]));
         (true, ([| 2 |], [| 1 |], [| 2 |], [| 1500 |]));
         (true, ([||], [| 1 |], [| 3100 |], [| 3 |]));
         (true, ([||], [| 3 |], [| 1100 |], [| 1500 |]));
         (true, ([| 350 |], [| 8 |], [| 8 |], [| 2 |]));
         (false, ([| 2 |], [| 20 |], [| 20 |], [| 500 |]));
         few;
         (true, ([||], [| rows |], [| rows + 3 |], [| 1030 |]));
         (false, ([||], [| rows |], [| 3100 |], [| 3 |]));
         (false, ([||], [| rows |], [| rows |], [| 0 |]));
         (true, ([||], [| 300 |], [| 16 |], [| 600 |]));
         (true, ([||], [| 1024 |], [| 16 |], [| 16 |]));
         (* Lane order's passes: more than 8 rows and columns, of 128
            products each or more. *)
         (true, ([||], [| 9 |], [| 12 |], [| 13900 |]));
       ]
      @ List.map
          (fun (n, k) -> (n mod 2 = 0, ([||], [| 2 |], [| n |], [| k |])))
          [
            (1, 17); (2, 3); (3, 33); (4, 16); (5, 47); (6, 1); (7, 15);
            (8, 64); (9, 100); (15, 31); (16, 1024); (17, 5); (64, 64);
            (64, 255);
          ])

(* The dtypes each accumulator holds, as nx_cpu.mli lists them: a complex
   one its parts' and the complex numbers of at most its width. *)
let rec held = function
  | D.Any D.Complex64 -> D.Any D.Complex64 :: held (D.Any D.Float32)
  | D.Any D.Complex128 ->
      D.[ Any Complex128; Any Complex64 ] @ held (D.Any D.Float64)
  | D.Any D.Float32 ->
      D.
        [
          Any Float32;
          Any Float16;
          Any Bfloat16;
          Any Float8_e4m3fn;
          Any Float8_e5m2;
          Any Float4_e2m1fn;
          Any Int8;
          Any Uint8;
          Any Int16;
          Any Uint16;
          Any Int4;
          Any Uint4;
          Any Bool;
          Any Bit;
        ]
  | _ -> D.[ Any Float64; Any Int32; Any Uint32 ] @ held (D.Any D.Float32)

let accs = D.[ Any Float32; Any Float64; Any Complex64; Any Complex128 ]

(* The operands an integer accumulator takes, as nx_cpu.mli lists them:
   every integer, and the booleans. *)
let integers =
  D.
    [
      Any Int64; Any Uint64; Any Int32; Any Uint32; Any Int16; Any Uint16;
      Any Int8; Any Uint8; Any Int4; Any Uint4; Any Bool; Any Bit;
    ]

(* The dtypes of [D.all] not in [dts]. *)
let others dts = List.filter (fun d -> not (List.mem d dts)) D.all

(* Cases of each float and complex acc, of operands it holds, or at times
   of dtypes it does not. *)
let computed =
  Gen.with_pp pp_case
    (Gen.bind (Gen.of_list ~pp:pp_dtype accs) (fun acc ->
         Gen.bind
           (Gen.frequency
              [
                (2, Gen.constant (held acc));
                (1, Gen.constant (others (held acc)));
              ])
           (fun dts -> case_of ~acc ~out:acc ~dts ())))

let examples = List.concat_map regimes accs

(* A batch of chain order's contractions over some 135 million products, a
   and b stored transposed, in float64: a run of it takes seconds, so one
   law takes it. *)
let large_examples =
  [ apart (D.Any D.Float64) (true, ([| 8 |], [| 257 |], [| 257 |], [| 256 |])) ]

(* Cases nx.cpu may decline: accumulators it lacks, other results, operands an
   accumulator does not hold. *)
let any_case =
  let open Gen in
  let dts = D.[ Any Float64; Any Float32; Any Int32; Any Int64; Any Bfloat16; Any Complex64 ] in
  Gen.with_pp pp_case
    (let* acc =
       of_list ~pp:pp_dtype
         D.
           [
             Any Float32; Any Float64; Any Int4; Any Int8; Any Int32;
             Any Complex64; Any Complex128;
           ]
     in
     let* out = frequency [ (2, constant ~pp:pp_dtype acc); (1, of_list ~pp:pp_dtype D.all) ] in
     case_of ~acc ~out ~dts ())

(* The reference *)

(* An output's init and the factors of its products, in increasing order of
   the contracted index: the contracting pairs' indices in C order. A
   complex output is two, its parts', each of the 2K terms nx_kernel.mli's
   orders give them. *)
type output = { init : float option; av : float array; bv : float array }

(* The result's shape, each output's offsets in C-contiguous [a] and [b]
   along the contracted index, in C order, and [walk f], which calls
   [f yi k ra rb] on each output in C order: [yi] its index, [k] its
   position in C order, [ra] and [rb] where its factors start, the
   offsets computed once. *)
let index c =
  let strides s =
    let st = Array.make (Array.length s) 1 in
    for i = Array.length s - 2 downto 0 do
      st.(i) <- st.(i + 1) * s.(i + 1)
    done;
    st
  in
  let sa = shape_of c.a and sb = shape_of c.b in
  let ta = strides sa and tb = strides sb in
  let bp = c.batch and cp = c.contracting in
  let free side r =
    List.filter
      (fun ax ->
        not
          (Array.exists (fun p -> side p = ax) bp
          || Array.exists (fun p -> side p = ax) cp))
      (List.init r Fun.id)
    |> Array.of_list
  in
  let fa = free fst (Array.length sa) and fb = free snd (Array.length sb) in
  let none n = Array.make (Array.length n) 0 in
  let y =
    Array.concat
      [
        Array.map (fun (i, _) -> sa.(i)) bp;
        Array.map (fun ax -> sa.(ax)) fa;
        Array.map (fun ax -> sb.(ax)) fb;
      ]
  in
  (* Each axis of the result's step in a and in b. *)
  let ya =
    Array.concat
      [ Array.map (fun (i, _) -> ta.(i)) bp; Array.map (fun ax -> ta.(ax)) fa; none fb ]
  in
  let yb =
    Array.concat
      [ Array.map (fun (_, j) -> tb.(j)) bp; none fa; Array.map (fun ax -> tb.(ax)) fb ]
  in
  (* The contracted index's offsets in a and in b, in C order. *)
  let kd = Array.map (fun (i, _) -> sa.(i)) cp in
  let offsets steps =
    let at = Array.make (Array.length kd) 0 in
    Array.init (total kd) (fun _ ->
        let off = ref 0 in
        Array.iteri (fun q i -> off := !off + (i * steps.(q))) at;
        let q = ref (Array.length kd - 1) in
        let carry () =
          at.(!q) <- at.(!q) + 1;
          at.(!q) = kd.(!q)
        in
        while !q >= 0 && carry () do
          at.(!q) <- 0;
          decr q
        done;
        !off)
  in
  let oa = offsets (Array.map (fun (i, _) -> ta.(i)) cp) in
  let ob = offsets (Array.map (fun (_, j) -> tb.(j)) cp) in
  let ty = strides y in
  let walk f =
    for k = 0 to total y - 1 do
      let yi = Array.make (Array.length y) 0 in
      let ra = ref 0 and rb = ref 0 and r = ref k in
      for d = 0 to Array.length y - 1 do
        let i = !r / ty.(d) in
        r := !r - (i * ty.(d));
        yi.(d) <- i;
        ra := !ra + (i * ya.(d));
        rb := !rb + (i * yb.(d))
      done;
      f yi k !ra !rb
    done
  in
  (y, oa, ob, walk)

(* The result's shape, and [each f], which calls [f yi k parts] on each
   output in C order: [yi] its index, [k] its position in C order, [parts]
   its init and factors, one for a real accumulator, the real then the
   imaginary part's for a complex one. The operands are read as
   C-contiguous float64 planes. The factors of a row of a, of which C
   order gives every column in turn, and of each column of b are gathered
   once and shared by the outputs that take them. Gathered per output,
   into arrays of up to 16 KiB that the major heap took, they were most of
   the bound law's time: 8.1 to 9.2 s a table on the M1 against 1.2 to
   2.1. *)
let outputs c =
  let flat x = planes c.acc (cast_to c.acc x) in
  let a = flat c.a and b = flat c.b and init = Option.map flat c.init in
  let y, oa, ob, walk = index c in
  let gather x r o = Array.map (fun t -> x.(r + t)) o in
  let pairs x y =
    Array.init (2 * Array.length x) (fun t ->
        (if t land 1 = 0 then x else y).(t / 2))
  in
  (* a's parts in turn against b's (br, −bi) for the real part and (bi, br)
     for the imaginary part. *)
  let row ra =
    if Array.length a = 1 then [| gather a.(0) ra oa |]
    else
      let ar = pairs (gather a.(0) ra oa) (gather a.(1) ra oa) in
      [| ar; ar |]
  in
  let column rb =
    if Array.length b = 1 then [| gather b.(0) rb ob |]
    else
      let br = gather b.(0) rb ob and bi = gather b.(1) rb ob in
      [| pairs br (Array.map Float.neg bi); pairs bi br |]
  in
  let each f =
    let last = ref (-1, [||]) and columns = Hashtbl.create 256 in
    walk (fun yi k ra rb ->
        if fst !last <> ra then last := (ra, row ra);
        let av = snd !last in
        let bv =
          match Hashtbl.find_opt columns rb with
          | Some bv -> bv
          | None ->
              let bv = column rb in
              Hashtbl.add columns rb bv;
              bv
        in
        let part p =
          { init = Option.map (fun x -> x.(p).(k)) init; av = av.(p);
            bv = bv.(p) }
        in
        f yi k (Array.init (Array.length a) part))
  in
  (y, each)

(* [x + y] as a double and its error, exactly (Knuth's TwoSum). *)
let two_sum x y =
  let s = x +. y in
  let z = s -. x in
  (s, x -. (s -. z) +. (y -. z))

(* The extents of [c]'s result, its contracted index and its batch. *)
let sizes c =
  let sa = shape_of c.a in
  let y =
    match
      Nx_kernel.Shape.contract ~batch:(Iarray.of_array c.batch)
        ~contracting:(Iarray.of_array c.contracting) ~segments:None (sa, None)
        (shape_of c.b, None)
    with
    | Ok y -> y
    | Error e -> failf "the case's shapes do not fit its pairs: %s" e
  in
  let along pairs = total (Array.map (fun (i, _) -> sa.(i)) pairs) in
  (y, along c.contracting, along c.batch)

let operands c = Array.of_list ([ c.a; c.b ] @ Option.to_list c.init)

(* [k]'s contraction of [c] into [dst], with the operands [a], [b] and
   [init] in place of [c]'s. *)
let contract ?sa ?sb (module K : Nx_kernel.S) c ~dst:(A.Any d) a b init =
  let init = Option.map (A.expect (A.dtype d)) init in
  K.contract ~batch:(Iarray.of_array c.batch)
    ~contracting:(Iarray.of_array c.contracting) ~segments:None ~acc:c.acc
    ~dst:d ~init
    { array = a; scale = sa }
    { array = b; scale = sb }

(* Whether [c] has an operand of a dtype its acc does not hold, which it
   reads cast to acc. *)
let converted c =
  let takes = if is_integer c.acc then integers else held c.acc in
  Array.exists
    (fun (A.Any x) -> not (List.mem (D.Any (A.dtype x)) takes))
    (operands c)

(* Outputs per batch element. *)
let per_batch c =
  let y, _, batch = sizes c in
  if batch = 0 then 0 else total y / batch

(* The extents of [c]'s rows and columns per batch element, as a
   contraction view's row and column axes give them. *)
let rows_cols c =
  let y, _, _ = sizes c in
  let nb = Array.length c.batch in
  let fa = Array.length (shape_of c.a) - nb - Array.length c.contracting in
  (total (Array.sub y nb fa), total (Array.sub y (nb + fa) (Array.length y - nb - fa)))

(* nx.cpu's order in [acc] for [m] rows, [n] columns and a depth [k]
   (nx_cpu.mli): chain where the rows and columns are both above 4 and
   multiply to at least 64 and the depth is below 128 times their product;
   lane otherwise. An integer sum is exact whatever the order, counted as
   chain. *)
type order = Chain | Lane

let order_of acc m n k =
  let chain = m > 4 && n > 4 && m * n >= 64 && k < 128 * m * n in
  if is_integer acc || chain then Chain else Lane

(* nx.cpu's order for [c]. *)
let order c =
  let m, n = rows_cols c and _, k, _ = sizes c in
  order_of c.acc m n k

(* The cases a run reaches, by the examples where the drawn cases reach
   one rarely: an operand of a dtype [acc] does not hold, no output, one,
   no product, each order of nx_cpu.mli's rule, lane order over several
   blocks of lanes, either side of 64 outputs, an init, a complex
   accumulator with a real operand, and each kind of view. *)
let covers c =
  let y, k, _ = sizes c in
  let per = per_batch c in
  let real (A.Any x) = not (is_complex (D.Any (A.dtype x))) in
  cover "an operand acc does not hold" (converted c);
  cover "no output" (total y = 0);
  cover "one output" (total y = 1);
  cover "no product" (k = 0 && total y > 0);
  cover "lane order" (per > 0 && order c = Lane);
  cover "chain order" (per > 0 && order c = Chain);
  cover "lane order over several blocks of lanes"
    (per > 0 && order c = Lane && k > 1024);
  cover "fewer than 64 outputs" (per > 0 && per < 64);
  cover "64 outputs or more" (per >= 64);
  cover "init" (c.init <> None);
  cover "complex with a real operand"
    (is_complex c.acc && (real c.a || real c.b));
  List.iter
    (fun v -> cover v (List.mem v c.views))
    [ "stepped"; "windowed"; "reversed"; "broadcast"; "stored apart"; "permuted" ]

(* A scale on [b]'s device. *)
let scale_on b : Nx_kernel.scale -> Nx_kernel.scale = function
  | Exponent x -> Exponent (A.expect D.Uint8 (on b (A.Any x)))
  | Value x ->
      let (A.Any y) = on b (A.Any x) in
      Value y

(* The kernels' result on [b]'s device, read back where the host reads it;
   [None] if they declined. *)
let run ?sa ?sb (b : Support.backend) c =
  let y, _, _ = sizes c in
  let (D.Any out) = c.out in
  let dst = A.Any (A.create b.device out (Iarray.of_array y)) in
  let init = Option.map (on b) c.init in
  let sa = Option.map (scale_on b) sa and sb = Option.map (scale_on b) sb in
  match contract ?sa ?sb b.kernels c ~dst (on b c.a) (on b c.b) init with
  | A.Done -> Some (host dst)
  | A.Declined -> None
  | r -> failf "contract answered %a" Nx_array_support.pp_answer r

(* A float's bits, every NaN's as one: a NaN result is some NaN. *)
let bits x =
  if Float.is_nan x then Int64.bits_of_float nan else Int64.bits_of_float x

(* The first outputs of [y] at which [check] answers why a part is wrong. *)
let wrong c y check =
  let _, each = outputs c in
  let g = planes c.acc y in
  let bad = ref [] in
  each (fun yi k parts ->
      let name p =
        if Array.length parts = 1 then "" else [| " re"; " im" |].(p)
      in
      Array.iteri
        (fun p o ->
          match check o g.(p).(k) with
          | Some why when List.length !bad < 8 ->
              bad := Format.asprintf "%a%s: %s" pp_ints yi (name p) why :: !bad
          | _ -> ())
        parts);
  List.rev !bad

(* The laws of S *)

(* The first outputs whose bits, or a part's, differ between [y] and [y'] of
   [acc], the second named [other]. *)
let differ_bits acc ~other y y' =
  let g = planes acc y and g' = planes acc y' in
  let bad = ref [] in
  List.iteri
    (fun k i ->
      Array.iteri
        (fun p x ->
          let x' = g'.(p).(k) in
          if bits x.(k) <> bits x' && List.length !bad < 8 then
            bad :=
              Format.asprintf "%a: %h, %s %h" pp_ints i x.(k) other x' :: !bad)
        g)
    (indices (shape_of y));
  List.rev !bad
