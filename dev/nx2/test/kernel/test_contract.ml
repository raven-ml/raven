(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Contractions through every kernel library the host runs: within the bound
   Nx_kernel.S.contract states of the exact sum, the same bits under every
   layout of the same values and into a destination identical to init, and
   nothing written where declined. nx.cpu's
   group checks what nx_cpu.mli states: the cases it computes, and its order,
   bit for bit, against a reference built here. *)

open Windtrap
open Nx_array_gen
module A = Nx_array
module D = Nx_array.Dtype
module Support = Nx_kernels_support

let pp_dtype ppf (D.Any dt) = D.pp ppf dt

let pp_any ppf (A.Any x) =
  Format.fprintf ppf "%a %a" D.pp (A.dtype x) L.pp (A.layout x)

let shape_of (A.Any x) = L.shape (A.layout x)
let total s = Array.fold_left ( * ) 1 s

(* [x] cast into [d], by nx.cpu. *)
let cast_into d (A.Any x) =
  match Nx_cpu.apply1 Nx_kernel.Prog.Cast ~dst:d x with
  | A.Done -> ()
  | r -> failf "a cast answered %a" Nx_array_support.pp_answer r

(* [x] cast to [dt], C-contiguous. *)
let cast_to (D.Any dt) x =
  let d = A.create Rig.host dt (shape_of x) in
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
  let d = A.create Rig.host D.Float64 (shape_of x) in
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
    let d = A.create Rig.host D.Complex128 (shape_of x) in
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

(* A contraction: its order, pairs, accumulator, operands and result dtype,
   and the views its operands were drawn through. [init] has the result's
   dtype. *)
type case = {
  order : Nx_kernel.order;
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
  let order = match c.order with Nx_kernel.Chain -> "chain" | Lane -> "lane" in
  Format.fprintf ppf
    "%s order, batch %a contracting %a acc %a out %a; a %a; b %a%a" order
    pp_ints (Array.map fst c.batch) pp_ints
    (Array.map fst c.contracting)
    pp_dtype c.acc pp_dtype c.out pp_any c.a pp_any c.b pp_init c.init

let inverse p =
  let q = Array.make (Array.length p) 0 in
  Array.iteri (fun i m -> q.(m) <- i) p;
  q

(* How an operand of rank [r] is viewed: made with extent 1 along its [nb]
   batch axes and broadcast, made with its last axis twice as long and
   stepped by two, then permuted, [perm.(i)] being the made axis of axis
   [i]. *)
type view = { stepped : bool; broadcast : bool; perm : int array }

let view r nb =
  let open Gen in
  let* broadcast = if nb > 0 then bool else constant false in
  let* stepped = bool in
  let stepped = stepped && r > 0 && (r > nb || not broadcast) in
  let+ p = permutation ~pp:Format.pp_print_int (List.init r Fun.id) in
  { stepped; broadcast; perm = Array.of_list p }

let names v =
  (if v.stepped then [ "stepped" ] else [])
  @ (if v.broadcast then [ "broadcast" ] else [])
  @
  if v.perm <> Array.init (Array.length v.perm) Fun.id then [ "permuted" ]
  else []

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

(* An array of [dt] of shape [s] through the view [v], its values, or each
   part of them, drawn from [seed]; under [wide] an integer's or a bool's
   from {!wide}, narrowed modulo its width. *)
let operand ?(wide_ints = false) dt s nb v seed =
  let r = Random.State.make [| seed |] in
  let last = Array.length s - 1 in
  let made =
    Array.mapi
      (fun i e ->
        if v.broadcast && i < nb then 1
        else if v.stepped && i = last then 2 * e
        else e)
      s
  in
  let draw () = value dt r in
  let xs =
    if wide_ints && (is_integer dt || dt = D.Any D.Bool || dt = D.Any D.Bit) then
      A.Any (A.of_array D.Int64 made (Array.init (total made) (fun _ -> wide r)))
    else if is_complex dt then
      A.Any
        (A.of_array D.Complex128 made
           (Array.init (total made) (fun _ ->
                let re = draw () in
                { Complex.re; im = draw () })))
    else
      A.Any
        (A.of_array D.Float64 made (Array.init (total made) (fun _ -> draw ())))
  in
  let (A.Any x) = cast_to dt xs in
  let move m x = Option.get (A.move m x) in
  let step i e =
    if i = last then { M.start = 1; count = s.(i); step = 2 }
    else { M.start = 0; count = e; step = 1 }
  in
  let x = if v.stepped then move (M.Slice (Array.mapi step made)) x else x in
  let x = if v.broadcast then move (M.Broadcast s) x else x in
  A.Any (move (M.Permute v.perm) x)

(* The bounds gemm.c's paths turn on, on every table: a product of more
   rows and columns than [few_rows_most] (AVX2's few, the largest) takes
   the chain path; [kc_least] and [nc_least] are the smallest blocks of the
   contraction and of b's columns, KC and NC, and [panels] the bytes of
   packed operands a group of batch elements shares. *)
let few_rows_most = 256
let kc_least = 256
let nc_least = 1024
let panels = 8 * 1024 * 1024

(* The most rows gemm.c's quads' dots and its axpys take, and the most
   bytes of a unit's sums in the axpys, AXPY_SUMS: more columns than fit
   take several units whatever the threads. The axpys take more than
   [axpy_few] rows only by k of [axpy_deep] or more and columns enough for
   every thread, 64 a thread: [axpy_wide] at the 16 threads of the most
   threaded host here; and more than [axpy_many] only by a b of more than
   an eighth of the host's largest cache: [axpy_cached] bytes pass it on
   every host here, [axpy_held] on none. *)
let quad_rows = 32
let axpy_rows = 128
let axpy_sums = 32 * 1024
let axpy_few = 16
let axpy_deep = 512
let axpy_wide = 64 * 16
let axpy_many = 32
let axpy_cached = 4 * 1024 * 1024
let axpy_held = 1024 * 1024

(* The rows of NEON's quads kernel's tile, MR, from which its axpys run it
   over b's rows zipped; x86's axpys never zip. *)
let quad_mr = 8

(* x86's axpys of several rows: the columns of b they keep at once,
   widened into pairs by AVX2 and zipped into words by AVX-VNNI-INT8, and
   the most rows of a group besides the first. *)
let axpy_cols = 256
let axpy_group = 6

(* Extents of the batch, row, column and contracted axes, in profiles: any
   ranks over a few elements; one output; 64 outputs or more with edge
   tiles; several blocks of the contraction on several threads; dots past a
   block of lanes; several panels of columns; few rows over several blocks
   of the contraction; few columns; several groups of batch elements. *)
let extents =
  let open Gen in
  let one lo hi = map (fun e -> [| e |]) (int_range lo hi) in
  let shaped batch rows cols con =
    let+ batch = batch and+ rows = rows and+ cols = cols and+ con = con in
    (batch, rows, cols, con)
  in
  let axes most lo hi = array ~size:(int_range 0 most) (int_range lo hi) in
  let ranked = shaped (axes 1 0 3) (axes 2 0 5) (axes 2 0 5) (axes 2 0 4) in
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
    ]

(* The ranks of [a], [b] and the result for the extents [e]. *)
let ranks (batch, rows, cols, con) =
  let nb = Array.length batch and nk = Array.length con in
  (nb + Array.length rows + nk, nb + nk + Array.length cols,
   nb + Array.length rows + Array.length cols)

(* The case of the extents [e] in [order] and [acc] into [out]: [a] and [b]
   of the dtypes and through the views given, an init of [out] through the
   view [init] gives, values drawn from [seed]. *)
let make ~order ~acc ~out (batch, rows, cols, con) (da, va) (db, vb) init seed
    =
  let nb = Array.length batch and fa = Array.length rows in
  let sa = Array.concat [ batch; rows; con ] in
  let sb = Array.concat [ batch; con; cols ] in
  let ia = inverse va.perm and ib = inverse vb.perm in
  let pairs = Array.init nb (fun k -> (ia.(k), ib.(k))) in
  let contracting =
    Array.init (Array.length con) (fun k -> (ia.(nb + fa + k), ib.(nb + k)))
  in
  let wide_ints = is_integer acc in
  let a = operand ~wide_ints da sa nb va seed in
  let b = operand ~wide_ints db sb nb vb (seed + 1) in
  let y =
    match
      Nx_kernel.Shape.contract ~batch:pairs ~contracting (shape_of a)
        (shape_of b)
    with
    | Ok y -> y
    | Error e -> invalid_arg ("make: " ^ e)
  in
  let made vi =
    operand ~wide_ints out (Array.map (fun i -> y.(i)) (inverse vi.perm)) 0 vi
      (seed + 2)
  in
  let views = names va @ names vb @ Option.fold ~none:[] ~some:names init in
  { order; batch = pairs; contracting; acc; a; b; init = Option.map made init;
    out; views }

(* A case in either order, in [acc] into [out], each operand of [acc] or of
   [dts]. *)
let case_of ?same ~acc ~out ~dts () =
  let open Gen in
  let* order = of_list Nx_kernel.[ Chain; Lane ] in
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
  make ~order ~acc ~out e (da, va) (db, vb)
    (if with_init then Some vi else None)
    seed

(* A case of each regime the covers name, in [acc], C-contiguous: no output;
   one; no product; 64 outputs or more in 4 rows or fewer, and in 4 columns or
   fewer; either side of 4 rows or columns over several blocks of lanes;
   fewer than 64 outputs over several blocks of lanes; few rows over several
   groups of columns, on several threads; several panels and several groups
   of few rows; 40 rows by 40 and by 17 columns over several blocks of the
   contraction, a tile's partial rows and columns; AVX2's most few rows
   against an edge of b's columns; the chain path over several blocks of the
   contraction, over several panels, without init, and with no product and
   no init; few columns by many rows with a read in place, over several
   blocks of the contraction and over one. The few rows' case again, and in float64 the chain path over
   several groups, with a and b stored transposed. Each in the order whose
   path it names, then cases in the other order: chain order in few rows or
   columns, lane order in many rows and columns over several passes and
   blocks of lanes; the lane tile in 2 rows by 1 to 9 columns, 15 to 17 and
   64, either side of each width's one and two vectors, by k off a step of
   the lanes, one step and a whole block. The drawn cases reach each a few
   times in a hundred or never, so a run of a hundred can miss one. A
   complex contraction is a real one of twice the columns and products, and
   an integer one takes chain order's paths, which the floats' cases reach:
   their own are few, each order over several blocks of products, no
   product, a batch, and the few rows' case stored transposed; an integer's
   also the chain path over several blocks of the contraction. *)
let regimes acc =
  let plain r = { stepped = false; broadcast = false; perm = Array.init r Fun.id } in
  let case (order, init, e) =
    let ra, rb, ry = ranks e in
    let ti = if init then Some (plain ry) else None in
    make ~order ~acc ~out:acc e (acc, plain ra) (acc, plain rb) ti 1
  in
  (* [case] with a and b stored transposed, which every table packs. *)
  let apart (order, init, e) =
    let ra, rb, ry = ranks e in
    let swap r =
      { (plain r) with
        perm = Array.init r (fun i -> if i >= r - 2 then (2 * r) - 3 - i else i) }
    in
    let ti = if init then Some (plain ry) else None in
    let c = make ~order ~acc ~out:acc e (acc, swap ra) (acc, swap rb) ti 1 in
    let copy (A.Any x) = A.Any (A.copy x) in
    { c with a = copy c.a; b = copy c.b }
  in
  let chain = Nx_kernel.Chain and lane = Nx_kernel.Lane in
  let rows = few_rows_most + 1 in
  let few =
    (chain, true, ([||], [| few_rows_most |], [| few_rows_most + 3 |], [| 3 |]))
  in
  (* Several groups of the chain path, both operands packed, take some 135
     million products in float64; float32's packs are half as large, and
     its groups need twice the products: float64 alone draws them. *)
  let groups =
    match acc with
    | D.Any D.Float64 ->
        let batch = (panels / (2 * rows * kc_least * 8)) + 1 in
        let e = ([| batch |], [| rows |], [| rows |], [| kc_least |]) in
        [ apart (chain, true, e) ]
    | _ -> []
  in
  let complex =
    [
      (lane, true, ([||], [| 3 |], [| 64 |], [| 600 |]));
      (lane, false, ([||], [| 40 |], [| 4 |], [| 600 |]));
      (chain, true, ([||], [| 40 |], [| 17 |], [| 600 |]));
      (chain, true, ([||], [| 3 |], [| 2 |], [| 0 |]));
      (chain, false, ([| 50 |], [| 8 |], [| 8 |], [| 2 |]));
    ]
  in
  let integer = (chain, true, ([||], [| 130 |], [| 130 |], [| 520 |])) :: complex in
  if is_complex acc then apart few :: List.map case complex
  else if is_integer acc then apart few :: List.map case integer
  else
    [ apart few ] @ groups
    @ List.map case
      ([
         (lane, true, ([||], [| 0 |], [| 3 |], [| 2 |]));
         (lane, true, ([| 1 |], [| 1 |], [| 1 |], [| 5 |]));
         (lane, true, ([||], [| 3 |], [| 2 |], [| 0 |]));
         (lane, true, ([||], [| 3 |], [| 64 |], [| 600 |]));
         (lane, true, ([||], [| 64 |], [| 3 |], [| 600 |]));
         (lane, true, ([||], [| 4 |], [| 40 |], [| 1100 |]));
         (chain, true, ([||], [| 5 |], [| 40 |], [| 1100 |]));
         (lane, true, ([||], [| 40 |], [| 4 |], [| 1100 |]));
         (chain, false, ([||], [| 40 |], [| 5 |], [| 1100 |]));
         (chain, true, ([||], [| 40 |], [| 40 |], [| 1100 |]));
         (chain, true, ([||], [| 40 |], [| 17 |], [| 1100 |]));
         (lane, true, ([| 2 |], [| 1 |], [| 2 |], [| 1500 |]));
         (lane, true, ([||], [| 1 |], [| 3100 |], [| 3 |]));
         (lane, true, ([||], [| 3 |], [| 1100 |], [| 1500 |]));
         (chain, true, ([| 350 |], [| 8 |], [| 8 |], [| 2 |]));
         few;
         (chain, true, ([||], [| rows |], [| rows + 3 |], [| 1030 |]));
         (chain, false, ([||], [| rows |], [| 3100 |], [| 3 |]));
         (chain, false, ([||], [| rows |], [| rows |], [| 0 |]));
         (chain, true, ([||], [| 300 |], [| 16 |], [| 600 |]));
         (chain, true, ([||], [| 1024 |], [| 16 |], [| 16 |]));
         (* The other order. *)
         (chain, true, ([| 1 |], [| 1 |], [| 1 |], [| 5 |]));
         (chain, true, ([||], [| 3 |], [| 64 |], [| 600 |]));
         (chain, false, ([||], [| 64 |], [| 3 |], [| 600 |]));
         (chain, true, ([||], [| 4 |], [| 40 |], [| 1100 |]));
         (chain, true, ([||], [| 1 |], [| 3100 |], [| 3 |]));
         (lane, true, ([||], [| 5 |], [| 40 |], [| 1100 |]));
         (lane, false, ([||], [| 40 |], [| 40 |], [| 1100 |]));
         (lane, true, ([||], [| 40 |], [| 17 |], [| 2100 |]));
         (lane, true, ([| 350 |], [| 8 |], [| 8 |], [| 2 |]));
         (lane, true, ([||], [| rows |], [| rows + 3 |], [| 30 |]));
         (lane, false, ([||], [| rows |], [| rows |], [| 0 |]));
       ]
      @ List.map
          (fun (n, k) -> (lane, n mod 2 = 0, ([||], [| 2 |], [| n |], [| k |])))
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
  | _ ->
      D.
        [
          Any Float64;
          Any Float32;
          Any Float16;
          Any Bfloat16;
          Any Int32;
          Any Uint32;
          Any Int8;
          Any Bool;
        ]

let accs = D.[ Any Float32; Any Float64; Any Complex64; Any Complex128 ]

let integer_accs =
  D.[ Any Int8; Any Uint8; Any Int16; Any Uint16; Any Int32; Any Uint32; Any Int64; Any Uint64 ]

(* The operands an integer accumulator takes, as nx_cpu.mli lists them:
   every integer, and the booleans. *)
let integers =
  D.
    [
      Any Int64; Any Uint64; Any Int32; Any Uint32; Any Int16; Any Uint16;
      Any Int8; Any Uint8; Any Int4; Any Uint4; Any Bool; Any Bit;
    ]

let computed =
  Gen.with_pp pp_case
    (Gen.bind (Gen.of_list ~pp:pp_dtype accs) (fun acc ->
         case_of ~acc ~out:acc ~dts:(held acc) ()))

let examples = List.concat_map regimes accs

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
   C-contiguous float64 planes, each output's factors at offsets computed
   once. *)
let outputs c =
  let flat = planes c.acc in
  let a = flat c.a and b = flat c.b and init = Option.map flat c.init in
  let y, oa, ob, walk = index c in
  let each f =
    walk (fun yi k ra rb ->
      let pick x r = Array.map (fun t -> x.(r + t)) in
      let part p av bv =
        { init = Option.map (fun x -> x.(p).(k)) init; av; bv }
      in
      let pairs x y =
        Array.init
          (2 * Array.length x)
          (fun t -> (if t land 1 = 0 then x else y).(t / 2))
      in
      let parts =
        if Array.length a = 1 then
          [| part 0 (pick a.(0) ra oa) (pick b.(0) rb ob) |]
        else
          (* a's parts in turn against b's (br, −bi) for the real part and
             (bi, br) for the imaginary part. *)
          let ar = pick a.(0) ra oa and ai = pick a.(1) ra oa in
          let br = pick b.(0) rb ob and bi = pick b.(1) rb ob in
          [|
            part 0 (pairs ar ai) (pairs br (Array.map Float.neg bi));
            part 1 (pairs ar ai) (pairs bi br);
          |]
      in
      f yi k parts)
  in
  (y, each)

(* [x + y] as a double and its error, exactly (Knuth's TwoSum). *)
let two_sum x y =
  let s = x +. y in
  let z = s -. x in
  (s, x -. (s -. z) +. (y -. z))

(* The sum of [o]'s products and init, with an error under 2^-100 of the sum
   of their magnitudes, and that sum. Products and the running sum are held
   in two parts. *)
let exact o =
  let start = Option.value ~default:0. o.init in
  let hi = ref start and lo = ref 0. and mag = ref (Float.abs start) in
  Array.iteri
    (fun k x ->
      let y = o.bv.(k) in
      let p = x *. y in
      let s, t = two_sum !hi p in
      hi := s;
      lo := !lo +. t +. Float.fma x y (-.p);
      mag := !mag +. Float.abs p)
    o.av;
  (!hi +. !lo, !mag)

let unit_roundoff acc =
  if part_of acc = D.Any D.Float32 then 0x1p-24 else 0x1p-53

let smallest_subnormal acc =
  if part_of acc = D.Any D.Float32 then 0x1p-149 else 0x1p-1074

(* The extents of [c]'s result, its contracted index and its batch. *)
let sizes c =
  let sa = shape_of c.a in
  let y =
    match
      Nx_kernel.Shape.contract ~batch:c.batch ~contracting:c.contracting sa
        (shape_of c.b)
    with
    | Ok y -> y
    | Error e -> failf "the case's shapes do not fit its pairs: %s" e
  in
  let along pairs = total (Array.map (fun (i, _) -> sa.(i)) pairs) in
  (y, along c.contracting, along c.batch)

let operands c = Array.of_list ([ c.a; c.b ] @ Option.to_list c.init)

(* [k]'s contraction of [c] into [dst], with the operands [a], [b] and
   [init] in place of [c]'s. *)
let contract (module K : Nx_kernel.S) c ~dst:(A.Any d) (A.Any a) (A.Any b)
    init =
  let init = Option.map (A.expect (A.dtype d)) init in
  K.contract ~order:c.order ~batch:c.batch ~contracting:c.contracting
    ~acc:c.acc ~dst:d ~init a b

(* Whether nx_cpu.mli says nx.cpu computes [c]: its dtypes, and a layout
   Contract_view groups. *)
let cpu_computes c =
  let acc = c.acc in
  let y, _, _ = sizes c in
  let (D.Any out) = c.out in
  let dst = A.create Rig.host out y in
  let (A.Any a) = c.a in
  let (A.Any b) = c.b in
  let init = Option.map (A.expect out) c.init in
  let (D.Any dt) = acc in
  let integer = is_integer acc && D.bits dt >= 8 in
  let takes = if integer then integers else held acc in
  (List.mem acc accs || integer)
  && c.out = acc
  && Array.for_all
       (fun (A.Any x) -> List.mem (D.Any (A.dtype x)) takes)
       (operands c)
  && Result.is_ok
       (Nx_kernel.Contract_view.v ~batch:c.batch ~contracting:c.contracting
          ~dst ~init a b)

(* Outputs per batch element. *)
let per_batch c =
  let y, _, batch = sizes c in
  if batch = 0 then 0 else total y / batch

(* The extents of [c]'s rows and columns per batch element, as
   Contract_view's [Row] and [Column] give them. *)
let rows_cols c =
  let y, _, _ = sizes c in
  let nb = Array.length c.batch in
  let fa = Array.length (shape_of c.a) - nb - Array.length c.contracting in
  (total (Array.sub y nb fa), total (Array.sub y (nb + fa) (Array.length y - nb - fa)))

(* gemm.c's complex paths: a copied where its numbers lie apart along k or
   are of another dtype, b's blocks once where it is broadcast along the
   batch, and one contraction per part where init's columns lie apart. *)
let complex_covers c =
  let complex = is_complex c.acc and _, k, batch = sizes c in
  let strides (A.Any x) = L.strides (A.layout x) in
  let real (A.Any x) = not (is_complex (D.Any (A.dtype x))) in
  let along x (i, _) = (strides x).(i) in
  let last x = Array.length (shape_of x) - 1 in
  cover "complex in chain order" (complex && c.order = Chain);
  cover "complex in lane order" (complex && c.order = Lane);
  cover "complex with a real operand" (complex && (real c.a || real c.b));
  cover "complex with a copied"
    (complex && k > 1
    && (real c.a
       || along c.a c.contracting.(Array.length c.contracting - 1) <> 1));
  cover "complex with b broadcast along the batch"
    (complex && batch > 1
    && Array.exists (fun (_, j) -> (strides c.b).(j) = 0) c.batch);
  cover "complex with init's columns apart"
    (complex
    && Option.fold ~none:false
         ~some:(fun i ->
           last i >= 0
           && (shape_of i).(last i) > 1
           && (strides i).(last i) <> 1)
         c.init)

let covers c =
  let y, k, batch = sizes c in
  let per = per_batch c in
  cover "no output" (total y = 0);
  cover "one output" (total y = 1);
  cover "no product" (k = 0 && total y > 0);
  let m, n = rows_cols c in
  let lanes = per > 0 && c.order = Lane in
  cover "chain order in 4 rows or fewer" (per > 0 && c.order = Chain && m <= 4);
  cover "chain order in 4 columns or fewer"
    (per > 0 && c.order = Chain && n <= 4);
  cover "lane order in 64 outputs or more, rows and columns above 4"
    (lanes && m > 4 && n > 4 && per >= 64);
  cover "lane order in passes: more than 8 rows and columns"
    (lanes && m > 8 && n > 8);
  cover "fewer than 64 outputs" (per > 0 && per < 64);
  cover "64 outputs or more" (per >= 64);
  cover "init" (c.init <> None);
  complex_covers c;
  cover "several blocks of the contraction" (per >= 64 && k > 512);
  cover "several blocks of lanes" (lanes && k > 1024);
  cover "4 rows or fewer" (per >= 64 && m <= 4);
  cover "4 columns or fewer" (per >= 64 && n <= 4);
  cover "4 rows over several blocks of lanes" (m = 4 && n > 16 && k > 1024);
  cover "5 rows over several blocks of lanes" (m = 5 && n > 16 && k > 1024);
  cover "4 columns over several blocks of lanes" (n = 4 && m > 16 && k > 1024);
  cover "5 columns over several blocks of lanes" (n = 5 && m > 16 && k > 1024);
  cover "4 rows over several groups of columns" (m <= 4 && n > 1024 && k > 1024);
  let chain =
    c.order = Chain && per >= 64 && m > few_rows_most && n > few_rows_most
  in
  let w = if part_of c.acc = D.Any D.Float32 then 4 else 8 in
  cover "the chain path" chain;
  cover "chain without init" (chain && c.init = None);
  cover "chain with no product" (chain && k = 0);
  cover "chain over several blocks of the contraction" (chain && k > 1024);
  cover "chain over several panels of columns" (chain && n > 3072);
  cover "chain over several groups of batch elements"
    (chain && batch * (m + min n nc_least) * min k kc_least * w > panels);
  List.iter
    (fun v -> cover v (List.mem v c.views))
    [ "stepped"; "broadcast"; "permuted" ]

(* The kernels' result on [b]'s device, read back where the host reads it;
   [None] if they declined. *)
let run (b : Support.backend) c =
  let y, _, _ = sizes c in
  let (D.Any out) = c.out in
  let dst = A.Any (A.create b.device out y) in
  let init = Option.map (on b) c.init in
  match contract b.kernels c ~dst (on b c.a) (on b c.b) init with
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

(* An output [x] of [c] is within γ(K + 1, 2u) (|init| + Σ|a·b|) of the exact
   sum of [o], u being [acc]'s unit roundoff, and (K + 1) η where results
   underflow, η [acc]'s smallest subnormal, twice the most a rounding into
   the subnormals errs; a NaN or infinite sum is matched. *)
let within c o x =
  let naive =
    Array.fold_left ( +. )
      (Option.value ~default:0. o.init)
      (Array.mapi (fun k a -> a *. o.bv.(k)) o.av)
  in
  let show = Printf.sprintf "%h" in
  if not (Float.is_finite naive) then
    if bits x = bits naive then None else Some (show x ^ ", not " ^ show naive)
  else
    let want, mag = exact o in
    let u = unit_roundoff c.acc in
    let n = float_of_int (Array.length o.av + 1) in
    let g = n *. 2. *. u /. (1. -. (n *. 2. *. u)) in
    let eta = smallest_subnormal c.acc in
    let bound = (g *. mag) +. (0x1p-100 *. mag) +. (n *. eta) in
    if Float.abs (x -. want) <= bound then None
    else
      Some
        (Printf.sprintf "%s is %h from %s, past %h" (show x) (x -. want)
           (show want) bound)

let law_bound b c =
  covers c;
  match run b c with
  | None -> cover "declined" true
  | Some y ->
      cover "computed" true;
      equal (list string) [] (wrong c y (within c))

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

(* Each operand as a C-contiguous copy of its view gives the same bits,
   where both layouts are computed. *)
let law_layouts b c =
  let plain (A.Any x) = A.Any (A.copy x) in
  let c' =
    { c with a = plain c.a; b = plain c.b; init = Option.map plain c.init }
  in
  match (run b c, run b c') with
  | Some y, Some y' ->
      cover "computed" true;
      cover "computed from views" (c.views <> []);
      equal (list string) [] (differ_bits c.acc ~other:"contiguous" y y')
  | _ -> cover "declined" true

(* An init given as dst itself, an operand identical to the written one,
   gives the bits of the same contraction into a fresh dst. *)
let law_init_as_dst (b : Support.backend) (c : case) =
  match c.init with
  | None -> cover "no init" true
  | Some i ->
      let c' = { c with init = Some (cast_to c.out i) } in
      let donated = on b (cast_to c.out i) in
      let answer =
        contract b.kernels c ~dst:donated (on b c.a) (on b c.b) (Some donated)
      in
      begin match (run b c', answer) with
      | Some y, A.Done ->
          let per = per_batch c in
          cover "fewer than 64 outputs" (per > 0 && per < 64);
          cover "64 outputs or more" (per >= 64);
          equal (list string) []
            (differ_bits c.acc ~other:"in place" y (host donated))
      | None, A.Declined -> cover "declined" true
      | _, r -> failf "in place, contract answered %a" Nx_array_support.pp_answer r
      end

(* A destination identical to [init], a C-contiguous array of the result's
   dtype, gives the bits a fresh one does: a kernel reads each element of
   [init] before it writes the destination there. *)
let law_in_place (b : Support.backend) (c : case) =
  match c.init with
  | None -> cover "no init" true
  | Some i -> (
      let c = { c with init = Some (cast_to c.out i) } in
      let alias = on b (cast_to c.out i) in
      let answer =
        contract b.kernels c ~dst:alias (on b c.a) (on b c.b) (Some alias)
      in
      match (run b c, answer) with
      | Some y, A.Done ->
          cover "computed" true;
          equal (list string) []
            (differ_bits c.acc ~other:"in place" y (host alias))
      | None, A.Declined -> cover "declined" true
      | y, r ->
          failf "a fresh destination %s, in place %a"
            (if Option.is_some y then "computed" else "declined")
            Nx_array_support.pp_answer r)

(* A kernel that declines writes nothing: [dst] keeps its values. *)
let law_declined (b : Support.backend) c =
  let y, _, _ = sizes c in
  let threes = A.Any (A.of_array D.Float64 y (Array.make (total y) 3.)) in
  let dst = on b (cast_to c.out threes) in
  let before = planes c.out (host dst) in
  let init = Option.map (on b) c.init in
  match contract b.kernels c ~dst (on b c.a) (on b c.b) init with
  | A.Declined ->
      cover "declined" true;
      equal (array (array float_exact)) before (planes c.out (host dst))
  | A.Done -> cover "computed" true
  | r -> failf "contract answered %a" Nx_array_support.pp_answer r

(* The order given *)

(* [x] rounded to float32. *)
let round32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* float32's fused multiply-add: a · b, exact in a double for float32 [a] and
   [b], plus [c], rounded to odd in a double and then to float32, which
   rounds once since 53 >= 2·24 + 2. *)
let fma32 a b c =
  let p = a *. b in
  let s, e = two_sum c p in
  if e = 0. || not (Float.is_finite s) then round32 s
  else if Int64.logand (Int64.bits_of_float s) 1L = 1L then round32 s
  else round32 (if e > 0. then Float.succ s else Float.pred s)

(* float32's addition, rounded once through a double for the same reason. *)
let add32 x y = round32 (x +. y)

(* [o]'s result in the order given, with [fma] and [add] of the accumulator:
   chain order, one fused chain from init or +0; or lane order, blocks of
   1024 terms in 16 lanes, summed by trees, then init, with no term init as
   it is. *)
let ordered ~chain ~fma ~add o =
  let k = Array.length o.av in
  if chain then begin
    let acc = ref (Option.value ~default:0. o.init) in
    for t = 0 to k - 1 do
      acc := fma o.av.(t) o.bv.(t) !acc
    done;
    !acc
  end
  else begin
    let block x =
      let l = Array.make 16 0. in
      for t = x * 1024 to min k ((x + 1) * 1024) - 1 do
        l.(t mod 16) <- fma o.av.(t) o.bv.(t) l.(t mod 16)
      done;
      List.iter
        (fun w ->
          for i = 0 to w - 1 do
            l.(i) <- add l.(i) l.(i + w)
          done)
        [ 8; 4; 2; 1 ];
      l.(0)
    in
    let rec tree s n =
      if n = 1 then s.(0)
      else
        let h = ref 1 in
        while 2 * !h < n do
          h := 2 * !h
        done;
        add (tree s !h) (tree (Array.sub s !h (n - !h)) (n - !h))
    in
    let n = (k + 1023) / 1024 in
    let sum = if n = 0 then 0. else tree (Array.init n block) n in
    match o.init with None -> sum | Some i -> if n = 0 then i else add i sum
  end

let law_order b c =
  covers c;
  match run b c with
  | None when cpu_computes c ->
      failf "nx.cpu declined a case nx_cpu.mli says it computes"
  | None -> cover "a layout the view does not group" true
  | Some y ->
      let chain = c.order = Chain in
      let fma, add =
        if part_of c.acc = D.Any D.Float32 then (fma32, add32)
        else (Float.fma, ( +. ))
      in
      let want = ordered ~chain ~fma ~add in
      equal (list string) []
        (wrong c y (fun o x ->
             let w = want o in
             if bits x = bits w then None
             else Some (Printf.sprintf "%h, expected %h" x w)))

(* Integers *)

let integer_cases =
  Gen.with_pp pp_case
    (Gen.bind (Gen.of_list ~pp:pp_dtype integer_accs) (fun acc ->
         case_of ~acc ~out:acc ~dts:integers ()))

(* [x]'s values in an array one element longer along its last axis, read
   from its second element: adjacent along that axis, at an odd offset. *)
let shifted (A.Any x as a) =
  let s = shape_of a in
  let last = Array.length s - 1 in
  let made = Array.mapi (fun i e -> if i = last then e + 1 else e) s in
  let at i e = { M.start = (if i = last then 1 else 0); count = e; step = 1 } in
  let y = Option.get (A.move (M.Slice (Array.mapi at s)) (A.create Rig.host (A.dtype x) made)) in
  cast_into y a;
  A.Any y

(* Quads in place: a as made, and b stored [n × k], adjacent along k, over
   several of chain order's blocks of words, past the dots' rows and
   columns; b interleaved past the axpys' rows; a copied tail;
   operands adjacent along k at an odd offset; an empty batch of operands
   adjacent along k. Quads' dots: a batch of few rows by b stored [n × k],
   over columns and words that do not fill the dot's 4 columns and 4
   words; 11 words, a step of 8 and a tail, by odd rows and columns; the
   most rows; one column of b, copied; few columns, with a and b swapped,
   alone and in a batch. Quads' axpys, b stored [k × n]: a batch over
   partial blocks of columns and of b's rows, and the same by NEON's tile
   at its rows; rows past few and past many over a column a thread, by the
   least k and b they take them by, and by a b held in cache; few rows
   over several groups of columns, past the most sums a unit holds; 64
   rows by 64 columns and 64 along k, which chain order takes; few rows by
   few columns, in bands of rows; either side of NEON's tile; two rows
   over more columns than x86's axpys keep at once, by an odd k; a row and
   a full group; a row and three groups. Each with every element of a and
   b at the dtype's extreme and init at the accumulator's largest value,
   so that two products pass 16 bits and every sum wraps. Copies whose
   rows would lie 4 KiB apart: b's of 1024 columns, zipped by NEON's
   tile's rows and copied past the axpys' rows, and a's of 4096 elements
   at an odd offset, by b stored [n × k]. *)
let quad_examples =
  let plain r = { stepped = false; broadcast = false; perm = Array.init r Fun.id } in
  let by_rows r =
    { (plain r) with perm = Array.init r (fun i -> if i < r - 2 then i else (2 * r) - 3 - i) }
  in
  List.concat_map
    (fun (acc, d, low, top) ->
      let case order e vb =
        let ra, _, ry = ranks e in
        make ~order ~acc ~out:acc e (d, plain ra) (d, vb) (Some (plain ry)) 3
      in
      (* [case] with a and b all [low], init all [top]. *)
      let extreme order e vb =
        let c = case order e vb in
        let fill v (A.Any x as a) =
          let s = shape_of a in
          cast_into x (A.Any (A.of_array D.Int64 s (Array.make (total s) v)));
          a
        in
        { c with a = fill low c.a; b = fill low c.b; init = Option.map (fill top) c.init }
      in
      (* [case] with b stored as viewed. *)
      let stored order e vb =
        let c = case order e vb in
        let (A.Any b) = c.b in
        { c with b = A.Any (A.copy b) }
      in
      [
        stored Chain ([||], [| 130 |], [| 40 |], [| 2052 |]) (by_rows 2);
        case Chain ([||], [| axpy_rows + 1 |], [| 50 |], [| 64 |]) (plain 2);
        case Lane ([||], [| 9 |], [| 33 |], [| 7 |]) (by_rows 2);
        (let c = case Chain ([||], [| 5 |], [| 9 |], [| 64 |]) (by_rows 2) in
         { c with a = shifted c.a; b = shifted c.b });
        stored Chain ([| 0 |], [| 3 |], [| 5 |], [| 4 |]) (by_rows 3);
        stored Chain ([| 2 |], [| 3 |], [| 7 |], [| 20 |]) (by_rows 3);
        stored Chain ([||], [| 3 |], [| 6 |], [| 44 |]) (by_rows 2);
        stored Lane ([||], [| quad_rows |], [| 33 |], [| 64 |]) (by_rows 2);
        case Chain ([||], [| 2 |], [| 1 |], [| 7 |]) (plain 2);
        stored Chain ([||], [| 40 |], [| 3 |], [| 20 |]) (by_rows 2);
        stored Lane ([| 2 |], [| 5 |], [| 2 |], [| 8 |]) (by_rows 3);
        case Lane ([| 2 |], [| 3 |], [| 70 |], [| 130 |]) (plain 3);
        case Chain ([||], [| quad_mr |], [| 70 |], [| 130 |]) (plain 2);
        case Chain ([||], [| axpy_many |], [| axpy_wide |], [| axpy_deep |]) (plain 2);
        case Chain ([||], [| axpy_many + 1 |], [| axpy_wide |],
                    [| axpy_cached / axpy_wide |]) (plain 2);
        case Chain ([||], [| 64 |], [| axpy_wide |], [| axpy_deep |]) (plain 2);
        case Chain ([||], [| axpy_few |], [| 600 |], [| 20 |]) (plain 2);
        case Chain ([||], [| 64 |], [| 64 |], [| 64 |]) (plain 2);
        extreme Chain ([||], [| axpy_few |], [| 40 |], [| 2048 |]) (plain 2);
        extreme Chain ([||], [| quad_mr - 1 |], [| 100 |], [| 70 |]) (plain 2);
        extreme Chain ([||], [| quad_mr |], [| 100 |], [| 70 |]) (plain 2);
        case Chain ([||], [| quad_mr |], [| 1024 |], [| 12 |]) (plain 2);
        case Chain ([||], [| axpy_rows + 1 |], [| 1024 |], [| 12 |]) (plain 2);
        (let c = stored Chain ([||], [| 2 |], [| 3 |], [| 4096 |]) (by_rows 2) in
         { c with a = shifted c.a });
        extreme Chain ([||], [| 2 |], [| axpy_cols + 44 |], [| 37 |]) (plain 2);
        extreme Chain ([||], [| axpy_group + 1 |], [| 40 |], [| 33 |]) (plain 2);
        extreme Chain ([||], [| (2 * axpy_group) + 3 |], [| 20 |], [| 18 |]) (plain 2);
      ])
    D.[ (Any Int32, Any Int8, -128L, 0x7FFFFFFFL);
        (Any Uint32, Any Uint8, 255L, 0xFFFFFFFFL) ]

let quad_cases =
  Gen.with_pp pp_case
    (Gen.bind (Gen.of_list ~pp:pp_dtype D.[ Any Int32; Any Uint32 ]) (fun acc ->
         let same = Gen.of_list ~pp:pp_dtype D.[ Any Int8; Any Uint8 ] in
         case_of ~same ~acc ~out:acc ~dts:[] ()))

(* [x]'s values as int64s: an integer's own value, a bool's 0 or 1. *)
let ints x =
  let d = A.create Rig.host D.Int64 (shape_of x) in
  cast_into d x;
  A.to_array d

(* [v] modulo [acc]'s width, read as [acc] reads its bits. *)
let wrap (D.Any dt) v =
  let w = D.bits dt in
  let signed = match D.kind dt with D.Signed -> true | _ -> false in
  if w = 64 then v
  else
    let m = Int64.logand v (Int64.pred (Int64.shift_left 1L w)) in
    if signed && Int64.compare m (Int64.shift_left 1L (w - 1)) >= 0 then
      Int64.sub m (Int64.shift_left 1L w)
    else m

(* Each output is init, or 0, plus the products of its operands' values
   modulo the accumulator's width: int64 arithmetic wraps modulo 2^64, a
   multiple of every width. *)
let integer_covers (c : case) =
  let wider (A.Any x) = D.Any (A.dtype x) <> c.acc in
  let m, n = rows_cols c and _, k, _ = sizes c in
  cover "an operand of another dtype" (wider c.a || wider c.b);
  let (D.Any acc) = c.acc in
  cover "a 64-bit accumulator" (D.bits acc = 64);
  cover "more than 128 rows and columns over 513 products or more"
    (m > 128 && n > 128 && k > 512)

(* The quads path reads an operand in place where k is a multiple of 4,
   the operand is adjacent along k and its offset and other steps are
   multiples of 4. Where b's columns are adjacent instead, it reads b in
   place by axpys for few rows and interleaves b's rows for more; it
   copies any other operand element by element, padded with zeros. *)
let quad_covers (c : case) =
  let _, k, _ = sizes c in
  let strides (A.Any x) = L.strides (A.layout x) in
  let offset (A.Any x) = L.offset (A.layout x) in
  let pairs = Array.append c.batch c.contracting in
  (* [x]'s axes of more than one element, those [side] of the contracting
     pairs names, or the others. *)
  let axes x side ~con =
    List.filter
      (fun ax ->
        (shape_of x).(ax) > 1
        && con = Array.exists (fun p -> side p = ax) c.contracting
        && (con || not (Array.exists (fun p -> side p = ax) pairs)))
      (List.init (Array.length (shape_of x)) Fun.id)
  in
  let adjacent x side =
    let rec steps want = function
      | [] -> true
      | ax :: r -> (strides x).(ax) = want && steps (want * (shape_of x).(ax)) r
    in
    let along = Array.to_list (Array.map side c.contracting) in
    steps 1 (List.rev (List.filter (fun ax -> List.mem ax (axes x side ~con:true)) along))
  in
  let aligned x side =
    offset x mod 4 = 0
    && List.for_all (fun ax -> (strides x).(ax) mod 4 = 0) (axes x side ~con:false)
    && Array.for_all (fun p -> (strides x).(side p) mod 4 = 0) c.batch
  in
  let fours = k > 0 && k mod 4 = 0 in
  let in_place x side = fours && adjacent x side && aligned x side in
  let columns_adjacent =
    match List.rev (axes c.b snd ~con:false) with
    | ax :: _ -> (strides c.b).(ax) = 1
    | [] -> false
  in
  cover "products a multiple of 4" fours;
  cover "a read in place" (in_place c.a fst);
  cover "b read in place" (in_place c.b snd);
  cover "an operand adjacent along k copied for its offset or steps"
    (fours
    && ((adjacent c.a fst && not (aligned c.a fst))
       || (adjacent c.b snd && not (aligned c.b snd))));
  let m, n = rows_cols c in
  let by_rows = k > 1 && not (adjacent c.b snd) && columns_adjacent in
  cover "b's rows interleaved" (by_rows && k >= 4 && m > axpy_rows);
  let b_bytes = Array.fold_left ( * ) 1 (shape_of c.b) in
  let axpys =
    by_rows && m > 0
    && (m <= axpy_few
       || m <= axpy_rows && k >= axpy_deep && n >= axpy_wide
          && (m <= axpy_many || b_bytes >= axpy_cached))
  in
  cover "few rows by quads' axpys" axpys;
  cover "more rows than few by quads' axpys" (axpys && m > axpy_few);
  cover "quads' axpys in bands of rows" (axpys && m > 1 && n <= 64 && k >= 1024);
  cover "more rows than few by few columns in chain order"
    (by_rows && m > axpy_few && m <= axpy_rows && n <= 64);
  cover "more rows than many by quads' axpys" (axpys && m > axpy_many);
  cover "more rows than many by a b held in cache, in chain order"
    (by_rows && m > axpy_many && m <= axpy_rows && k >= axpy_deep
    && n >= axpy_wide && b_bytes <= axpy_held);
  cover "quads' axpys over several groups of columns"
    (axpys && n > axpy_sums / 4 / m / 64 * 64);
  cover "quads' axpys over k not a multiple of 4" (axpys && k mod 4 <> 0);
  cover "quads' axpys below the tile" (axpys && m < quad_mr);
  cover "quads' axpys by the tile" (axpys && m >= quad_mr);
  cover "quads' axpys of two rows" (axpys && m = 2);
  cover "quads' axpys over more rows than a group holds"
    (axpys && m - 1 > axpy_group);
  cover "quads' axpys of several rows over more columns than they keep"
    (axpys && m > 1 && n > axpy_cols);
  let dots = m > 0 && n > 0 && (in_place c.b snd || n = 1) in
  cover "few rows by quads' dots" (dots && m <= n && m <= quad_rows);
  cover "few columns by quads' dots" (dots && n < m && n <= quad_rows);
  let words = (k + 3) / 4 in
  cover "quads' dots over 8 words and a tail"
    (dots && min m n <= quad_rows && words > 8 && words mod 8 <> 0);
  cover "an odd count of rows by quads' dots, the last alone"
    (dots && min m n <= quad_rows && min m n mod 2 = 1);
  cover "b read in place by chain order" (in_place c.b snd && min m n > quad_rows);
  cover "a copy's rows a line past 4 KiB"
    ((m > 0 && n > 0 && k > 0)
    && ((not (in_place c.b snd)) && (not axpys) && n * 4 mod 4096 = 0
       || ((not (in_place c.a fst)) && words * 4 mod 4096 = 0)));
  cover "quads' axpys zipping 1024 columns by the tile"
    (axpys && m >= quad_mr && n mod 1024 = 0 && k > 4);
  cover "fewer than 4 products" (k < 4);
  cover "products not a multiple of 4" (k mod 4 <> 0)

let law_wraps ~covers b (c : case) =
  covers c;
  cover "init" (c.init <> None);
  List.iter
    (fun v -> cover v (List.mem v c.views))
    [ "stepped"; "broadcast"; "permuted" ];
  match run b c with
  | None when cpu_computes c ->
      failf "nx.cpu declined a case nx_cpu.mli says it computes"
  | None -> cover "a layout the view does not group" true
  | Some y ->
      let _, oa, ob, walk = index c in
      let a = ints c.a and bv = ints c.b and init = Option.map ints c.init in
      let got = ints y and bad = ref [] in
      (* An accumulator narrower than OCaml's 63-bit ints sums in them: their
         arithmetic wraps modulo 2^63, a multiple of its width. *)
      let (D.Any acc) = c.acc in
      let sum =
        if D.bits acc < 63 then (
          let a = Array.map Int64.to_int a and bv = Array.map Int64.to_int bv in
          fun s0 ra rb ->
            let s = ref (Int64.to_int s0) in
            for t = 0 to Array.length oa - 1 do
              s := !s + (a.(ra + oa.(t)) * bv.(rb + ob.(t)))
            done;
            Int64.of_int !s)
        else fun s0 ra rb ->
          let s = ref s0 in
          for t = 0 to Array.length oa - 1 do
            s := Int64.add !s (Int64.mul a.(ra + oa.(t)) bv.(rb + ob.(t)))
          done;
          !s
      in
      walk (fun yi k ra rb ->
          let s0 = match init with Some i -> i.(k) | None -> 0L in
          let want = wrap c.acc (sum s0 ra rb) in
          if want <> got.(k) && List.length !bad < 8 then
            bad := Format.asprintf "%a: %Ld, expected %Ld" pp_ints yi got.(k) want :: !bad);
      equal (list string) [] (List.rev !bad)

let law_computes b c =
  let computes = cpu_computes c in
  cover "computed" computes;
  cover "declined" (not computes);
  equal ~msg:"computed" bool computes (Option.is_some (run b c))

(* With no products nx.cpu gives each output init's bits, -0 and a
   signalling NaN included, below 64 outputs and from 64, in either order:
   [ud] is an unsigned dtype of [fd]'s width, [bits] init's values in it. *)
let empty (type u ue f fe) (b : Support.backend) (ud : (u, ue) D.t)
    (fd : (f, fe) D.t) ty (bits : u array) () =
  let module K = (val b.kernels) in
  let at order (m, n) =
    let want = Array.init (m * n) (fun i -> bits.(i mod Array.length bits)) in
    let init = Option.get (A.bitcast fd (A.of_array ud [| m; n |] want)) in
    let a = A.create Rig.host fd [| m; 0 |] and b = A.create Rig.host fd [| 0; n |] in
    let dst = A.create Rig.host fd [| m; n |] in
    (match
       K.contract ~order ~batch:[||] ~contracting:[| (1, 0) |] ~acc:(D.Any fd)
         ~dst ~init:(Some init) a b
     with
    | A.Done -> ()
    | r -> failf "contract answered %a" Nx_array_support.pp_answer r);
    equal (array ty) want (A.to_array (Option.get (A.bitcast ud dst)))
  in
  List.iter
    (fun o -> List.iter (at o) [ (1, 3); (8, 24) ])
    Nx_kernel.[ Chain; Lane ]

(* On a thread that has run no product, an empty product gives +0, or
   init's bits, in chain order on the few-rows path and the chain path, and
   in lane order: a thread's first product makes the buffers nx.cpu keeps
   for it. *)
let fresh (b : Support.backend) () =
  let module K = (val b.kernels) in
  let at order (m, n) init =
    let a = A.create Rig.host D.Float32 [| m; 0 |] in
    let b = A.create Rig.host D.Float32 [| 0; n |] in
    let dst = A.of_array D.Float32 [| m; n |] (Array.make (m * n) 7.) in
    let i = A.of_array D.Float32 [| m; n |] (Array.make (m * n) (-0.)) in
    let init = if init then Some i else None in
    let call () =
      try
        Ok
          (K.contract ~order ~batch:[||] ~contracting:[| (1, 0) |]
             ~acc:(D.Any D.Float32) ~dst ~init a b)
      with e -> Error e
    in
    (match Domain.join (Domain.spawn call) with
    | Ok A.Done -> ()
    | Ok r -> failf "contract answered %a" Nx_array_support.pp_answer r
    | Error e -> failf "contract raised %s" (Printexc.to_string e));
    let want = if Option.is_some init then 0x80000000l else 0l in
    equal (array int32) (Array.make (m * n) want)
      (Array.map Int32.bits_of_float (A.to_array dst))
  in
  List.iter
    (fun o ->
      List.iter (fun mn -> at o mn false; at o mn true) [ (8, 8); (200, 200) ])
    Nx_kernel.[ Chain; Lane ]

(* The suite *)

let laws (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group b.name
    [
      prop ~examples "a contraction is within its bound of the exact sum"
        computed (run (law_bound b));
      prop ~examples "a contraction's bits do not depend on layouts" computed
        (run (law_layouts b));
      prop ~examples "an init given as dst gives the same bits" computed
        (run (law_init_as_dst b));
      prop "a declined contraction writes nothing" any_case
        (run (law_declined b));
      prop ~examples
        "a destination identical to init gives a fresh one's bits" computed (run (law_in_place b));
    ]

(* Guard pages *)

(* A [m × k] by [k × n] product whose operands and dst each lie against an
   inaccessible page on [side]: b a window of columns [c0, c0 + n) of a
   [k × cols] array, so that its rows lie off any alignment, and the
   window ends at the page where [c0 + n = cols]. Slivers of a matrix
   unit's half and whole tiles (16 and 32 float32 columns, 8 and 16
   float64) and rows past one tile are drawn often. *)
type guard_case = {
  f64 : bool;
  side : Support.side;
  m : int;
  n : int;
  k : int;
  cols : int;
  c0 : int;
  seed : int;
}

let pp_guard ppf g =
  Format.fprintf ppf "%s %s m %d n %d k %d cols %d c0 %d seed %d"
    (if g.f64 then "float64" else "float32")
    (match g.side with Support.Before -> "before" | After -> "after")
    g.m g.n g.k g.cols g.c0 g.seed

let guard_cases =
  let open Gen in
  Gen.with_pp pp_guard
    (let* f64 = bool in
     let half = if f64 then 8 else 16 in
     let* after = bool in
     let side = if after then Support.After else Support.Before in
     let* m = frequency [ (1, int_range 1 32); (2, int_range 33 72) ] in
     let* k = int_range 1 140 in
     let* tiles = int_range 0 2 in
     let* edge =
       frequency
         [ (2, constant half); (1, int_range 1 (2 * half - 1)); (1, constant 0) ]
     in
     let n = max 1 ((tiles * 2 * half) + edge) in
     let* extra = int_range 0 8 in
     let* c0 = frequency [ (1, constant extra); (2, int_range 0 extra) ] in
     let+ seed = int in
     { f64; side; m; n; k; cols = n + extra; c0; seed })

let guarded_product (type s) (b : Support.backend) (dt : (float, s) D.t) g =
  let (module K : Nx_kernel.S) = b.kernels in
  let st = Random.State.make [| g.seed |] in
  let values s =
    A.of_array dt s
      (Array.init (total s) (fun _ -> Random.State.float st 2. -. 1.))
  in
  let window x =
    let all c = { Nx_array.Move.start = 0; count = c; step = 1 } in
    Option.get
      (A.move
         (Nx_array.Move.Slice
            [| all g.k; { Nx_array.Move.start = g.c0; count = g.n; step = 1 } |])
         x)
  in
  let product ~dst a x =
    match
      K.contract ~order:Nx_kernel.Chain ~batch:[||] ~contracting:[| (1, 0) |]
        ~acc:(D.Any dt) ~dst ~init:None a (window x)
    with
    | A.Done -> ()
    | r -> failf "contract answered %a" Nx_array_support.pp_answer r
  in
  let guard x =
    let g', free = Support.guarded g.side dt (A.layout x) in
    (match Nx_cpu.apply1 Nx_kernel.Prog.Copy ~dst:g' x with
    | A.Done -> ()
    | r -> failf "a copy answered %a" Nx_array_support.pp_answer r);
    (g', free)
  in
  let a = values [| g.m; g.k |] and x = values [| g.k; g.cols |] in
  let want = A.create Rig.host dt [| g.m; g.n |] in
  product ~dst:want a x;
  let ga, free_a = guard a and gx, free_x = guard x in
  let gd, free_d = guard want in
  Fun.protect
    ~finally:(fun () ->
      free_a ();
      free_x ();
      free_d ())
    (fun () ->
      product ~dst:gd ga gx;
      let bits y = Array.map Int64.bits_of_float (A.to_array y) in
      equal (array int64) (bits want) (bits gd))

let law_guards (b : Support.backend) g =
  let half = if g.f64 then 8 else 16 in
  cover "a sliver of half a matrix unit's tile" (g.n mod (2 * half) = half);
  cover "rows past one tile" (g.m > 2 * half);
  cover "b's window ending at the page"
    (g.side = Support.After && g.c0 + g.n = g.cols);
  cover "b's window starting at the page"
    (g.side = Support.Before && g.c0 = 0);
  cover "a half-tile sliver of b ending at the page, rows past one tile"
    (g.n mod (2 * half) = half
    && g.side = Support.After
    && g.c0 + g.n = g.cols
    && g.m > 2 * half);
  if g.f64 then guarded_product b D.Float64 g
  else guarded_product b D.Float32 g

(* nx.cpu under each table the host runs. *)
let cpu (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group ("nx.cpu " ^ b.name)
    [
      prop ~examples "each output adds its products in the order given"
        computed (run (law_order b));
      prop ~examples:(List.concat_map regimes integer_accs)
        "an integer output is its products' sum modulo the accumulator"
        integer_cases
        (run (law_wraps ~covers:integer_covers b));
      prop ~examples:quad_examples
        "8-bit operands into 32 bits wrap as integers do" quad_cases
        (run (law_wraps ~covers:quad_covers b));
      prop "computes the cases nx_cpu.mli lists, declines others"
        any_case (run (law_computes b));
      prop ~count:500
        "reads and writes no byte past its operands and dst, whatever pages lie \
         beside them"
        guard_cases (run (law_guards b));
      test "with no products, a float32 output is init's bits"
        (run (empty b D.Uint32 D.Float32 int32
                [| 0x80000000l; 0x7F800001l; 0x3FC00000l |]));
      test "with no products, a float64 output is init's bits"
        (run (empty b D.Uint64 D.Float64 int64
                [| 0x8000000000000000L; 0x7FF0000000000001L; 0x3FF8000000000000L |]));
      test "on a fresh thread, an empty product is +0 or init"
        (run (fresh b));
    ]

let order =
  Testable.make
    ~pp:(fun ppf o ->
      Format.pp_print_string ppf
        (match o with Nx_kernel.Chain -> "Chain" | Lane -> "Lane"))
    ~equal:( = )

let orders =
  cases "nx.cpu chooses chain order at rows and columns above 4 and 64 outputs"
    ~name:(fun (m, n, _) -> Printf.sprintf "%d by %d" m n)
    Nx_kernel.
      [
        (0, 1000, Lane);
        (1000, 0, Lane);
        (1, 1, Lane);
        (4, 1000, Lane);
        (5, 1000, Chain);
        (1000, 4, Lane);
        (1000, 5, Chain);
        (5, 5, Lane);
        (5, 12, Lane);
        (5, 13, Chain);
        (12, 5, Lane);
        (13, 5, Chain);
        (7, 9, Lane);
        (9, 7, Lane);
        (8, 8, Chain);
        (16, 4, Lane);
        (max_int, max_int, Chain);
        (max_int, 5, Chain);
        (5, max_int, Chain);
        (max_int, 4, Lane);
        (4, max_int, Lane);
        (max_int, 0, Lane);
      ]
    (fun (rows, columns, want) ->
      equal order want (Nx_cpu.contract_order ~rows ~columns))

let () =
  exit
    (Windtrap.run "nx_kernel.contract"
       ((orders :: List.map laws Support.backends) @ List.map cpu Support.cpus))
