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

(* An operand without a scale. *)
let plain x : Nx_kernel.operand = { array = A.Any x; scale = None }

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
   stepped by two, its elements stored in the order of the made axes
   [stored] names, outermost first, then permuted, [perm.(i)] being the
   made axis of axis [i]. *)
type view = {
  stepped : bool;
  broadcast : bool;
  stored : int array;
  perm : int array;
}

let view r nb =
  let open Gen in
  let axes = List.init r Fun.id in
  let* broadcast = if nb > 0 then bool else constant false in
  let* stepped = bool in
  let stepped = stepped && r > 0 && (r > nb || not broadcast) in
  let* stored =
    frequency
      [ (2, constant axes); (1, permutation ~pp:Format.pp_print_int axes) ]
  in
  let+ p = permutation ~pp:Format.pp_print_int axes in
  { stepped; broadcast; stored = Array.of_list stored; perm = Array.of_list p }

let identity p = p = Array.init (Array.length p) Fun.id

let names v =
  (if v.stepped then [ "stepped" ] else [])
  @ (if v.broadcast then [ "broadcast" ] else [])
  @ (if identity v.stored then [] else [ "stored apart" ])
  @ if identity v.perm then [] else [ "permuted" ]

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
  let move m x = Option.get (A.move m x) in
  let (A.Any x) = cast_to dt xs in
  (* The values copied into an array stored in [v.stored]'s order, read
     back in the made axes' order. *)
  let x =
    if identity v.stored then x
    else
      let t = A.copy (move (M.Permute (Iarray.of_array v.stored)) x) in
      move (M.Permute (Iarray.of_array (inverse v.stored))) t
  in
  let step i e =
    if i = last then { M.start = 1; count = s.(i); step = 2 }
    else { M.start = 0; count = e; step = 1 }
  in
  let x =
    if v.stepped then move (M.Slice (Iarray.of_array (Array.mapi step made))) x
    else x
  in
  let x = if v.broadcast then move (M.Broadcast (Iarray.of_array s)) x else x in
  A.Any (move (M.Permute (Iarray.of_array v.perm)) x)

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
   of the contraction; few columns; several groups of batch elements; two
   axes in each group, which views split into runs; and a batch of two
   axes over 512 KiB of operands or more, whose outer run takes several
   threads. *)
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
      Nx_kernel.Shape.contract ~batch:(Iarray.of_array pairs)
        ~contracting:(Iarray.of_array contracting) ~segments:None
        (shape_of a, None) (shape_of b, None)
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
   blocks of the contraction and over one; few rows of a batch whose
   slivers of b are fewer than the threads, in blocks of rows, with a in
   place and with a alone stored transposed. The few
   rows' case again, and in float64 the chain path over
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
   also the chain path over several blocks of the contraction. Every
   kind's contracted axes stored apart, and the floats' attention heads,
   whose groups make several runs; every kind's a of a dtype it does not
   hold. *)
let regimes acc =
  let plain r =
    let axes = Array.init r Fun.id in
    { stepped = false; broadcast = false; stored = axes; perm = axes }
  in
  let case (order, init, e) =
    let ra, rb, ry = ranks e in
    let ti = if init then Some (plain ry) else None in
    make ~order ~acc ~out:acc e (acc, plain ra) (acc, plain rb) ti 1
  in
  let swap r =
    { (plain r) with
      perm = Array.init r (fun i -> if i >= r - 2 then (2 * r) - 3 - i else i) }
  in
  let copy (A.Any x) = A.Any (A.copy x) in
  (* [case] with a and b stored transposed, which every table packs. *)
  let apart (order, init, e) =
    let ra, rb, ry = ranks e in
    let ti = if init then Some (plain ry) else None in
    let c = make ~order ~acc ~out:acc e (acc, swap ra) (acc, swap rb) ti 1 in
    { c with a = copy c.a; b = copy c.b }
  in
  (* [case] with a alone stored transposed: few rows pack it, and blocks of
     its rows read the pack from their first row. *)
  let a_apart (order, init, e) =
    let ra, rb, ry = ranks e in
    let ti = if init then Some (plain ry) else None in
    let c = make ~order ~acc ~out:acc e (acc, swap ra) (acc, plain rb) ti 1 in
    { c with a = copy c.a }
  in
  (* [case] with a and b stored in the orders of their made axes [sa] and
     [sb] name. *)
  let stored (order, init, e, sa, sb) =
    let _, _, ry = ranks e in
    let ti = if init then Some (plain ry) else None in
    let at s = { (plain (Array.length s)) with stored = s } in
    make ~order ~acc ~out:acc e (acc, at sa) (acc, at sb) ti 1
  in
  let chain = Nx_kernel.Chain and lane = Nx_kernel.Lane in
  (* Contracted axes stored in another order in a than in b, in either
     order. *)
  let split =
    List.map stored
      [
        (chain, true, ([||], [| 3 |], [| 4 |], [| 5; 6 |]), [| 0; 2; 1 |],
         [| 0; 1; 2 |]);
        (lane, false, ([||], [| 3 |], [| 4 |], [| 5; 6 |]), [| 0; 2; 1 |],
         [| 0; 1; 2 |]);
      ]
  in
  (* Attention's scores, q and k of [b; s; h; d] read as [b; h; s; d]: a
     batch of two runs, the outer of 16 positions, over 768 KiB of
     operands, which the threads take a position at a time. *)
  let heads =
    stored
      ( chain, false, ([| 16; 4 |], [| 32 |], [| 32 |], [| 32 |]),
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
    make ~order:chain ~acc ~out:acc e (wide, plain 2) (acc, plain 2) None 1
  in
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
  if is_complex acc then (apart few :: unheld :: split) @ List.map case complex
  else if is_integer acc then
    (apart few :: unheld :: split) @ List.map case integer
  else
    [ apart few; heads; unheld;
      a_apart (chain, false, ([| 2 |], [| 20 |], [| 20 |], [| 500 |])) ]
    @ split @ groups
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
         (chain, false, ([| 2 |], [| 20 |], [| 20 |], [| 500 |]));
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
  | _ -> D.[ Any Float64; Any Int32; Any Uint32 ] @ held (D.Any D.Float32)

let accs = D.[ Any Float32; Any Float64; Any Complex64; Any Complex128 ]

let integer_accs =
  D.
    [
      Any Int4; Any Uint4; Any Int8; Any Uint8; Any Int16; Any Uint16;
      Any Int32; Any Uint32; Any Int64; Any Uint64;
    ]

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

(* The floats narrower than float32, which a float32 sum is rounded to. *)
let narrow =
  D.
    [
      Any Float16; Any Bfloat16; Any Float8_e4m3fn; Any Float8_e5m2;
      Any Float4_e2m1fn;
    ]

(* Cases into a result of another dtype than acc: mostly a narrow float, at
   times any dtype. *)
let rounded =
  let pick = Gen.of_list ~pp:pp_dtype in
  Gen.with_pp pp_case
    (Gen.bind (pick (accs @ integer_accs)) (fun acc ->
         Gen.bind
           (Gen.frequency [ (3, pick narrow); (1, pick D.all) ])
           (fun out ->
             let dts = if is_integer acc then integers else held acc in
             case_of ~acc ~out ~dts ())))

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

(* The sum of [o]'s products and init, with an error under 2^-100 of the sum
   of their magnitudes, and that sum. Products and the running sum are held
   in two parts. *)
let exact o =
  (* A loop: refs that a closure takes are boxed, a float allocated a
     term. *)
  let start = Option.value ~default:0. o.init in
  let hi = ref start and lo = ref 0. and mag = ref (Float.abs start) in
  for k = 0 to Array.length o.av - 1 do
    let x = o.av.(k) and y = o.bv.(k) in
    let p = x *. y in
    let s, t = two_sum !hi p in
    hi := s;
    lo := !lo +. t +. Float.fma x y (-.p);
    mag := !mag +. Float.abs p
  done;
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
  K.contract ~order:c.order ~batch:(Iarray.of_array c.batch)
    ~contracting:(Iarray.of_array c.contracting) ~segments:None ~acc:c.acc
    ~dst:d ~init
    { array = a; scale = sa }
    { array = b; scale = sb }

(* Whether nx_cpu.mli says nx.cpu computes [c]: an acc it computes in, of
   operands of any dtype, into any result, at any layout. *)
let cpu_computes c = List.mem c.acc accs || is_integer c.acc

(* Whether [c] has an operand of a dtype its acc does not hold, which it
   reads cast to acc. *)
let converted c =
  let takes = if is_integer c.acc then integers else held c.acc in
  Array.exists
    (fun (A.Any x) -> not (List.mem (D.Any (A.dtype x)) takes))
    (operands c)

(* The runs each group of [c] makes, batch, rows, columns and contracted,
   as nx_coalesce_dims makes them over the group's operands (gemm.c): its
   axes but those of extent 1, an axis merging into the one before where
   every operand steps there its step times its extent; one run where
   there is no axis or an axis of no element. dst is C-contiguous, as every
   destination is. With [~init_alone], the runs over init alone. *)
let runs ?(init_alone = false) c =
  let y, _, _ = sizes c in
  let nb = Array.length c.batch in
  let strides (A.Any x) = L.strides (A.layout x) in
  let dst = L.strides (L.contiguous y) in
  let along x axes =
    if init_alone then [] else [ Array.map (fun i -> (strides x).(i)) axes ]
  in
  let ys axes =
    if init_alone then [] else [ Array.map (fun i -> dst.(i)) axes ]
  in
  let inits axes =
    match c.init with
    | Some (A.Any i) ->
        [ Array.map (fun k -> (L.strides (A.layout i)).(k)) axes ]
    | None -> []
  in
  (* The axes of [x] no pair names, in order. *)
  let free x side =
    let paired i =
      Array.exists (fun p -> side p = i) (Array.append c.batch c.contracting)
    in
    let r = Array.length (shape_of x) in
    Array.of_list (List.filter (fun i -> not (paired i)) (List.init r Fun.id))
  in
  let fa = free c.a fst and fb = free c.b snd in
  let count exts steps =
    let step i = List.map (fun s -> s.(i)) steps in
    let merges cur i =
      List.for_all2 (fun c s -> c = s.(i) * exts.(i)) cur steps
    in
    let join (runs, cur) i =
      ((if merges cur i then runs else runs + 1), step i)
    in
    let n = Array.length exts in
    match List.filter (fun i -> exts.(i) <> 1) (List.init n Fun.id) with
    | _ when Array.exists (( = ) 0) exts -> 1
    | [] -> 1
    | first :: rest -> fst (List.fold_left join (1, step first) rest)
  in
  let on side pairs = Array.map side pairs in
  let yb = Array.init nb Fun.id in
  let yr = Array.init (Array.length fa) (fun i -> nb + i) in
  let yc = Array.mapi (fun i _ -> nb + Array.length fa + i) fb in
  let exts axes = Array.map (fun i -> y.(i)) axes in
  let ka = on fst c.contracting in
  ( count (exts yb)
      (along c.a (on fst c.batch) @ along c.b (on snd c.batch) @ ys yb
     @ inits yb),
    count (exts yr) (along c.a fa @ ys yr @ inits yr),
    count (exts yc) (along c.b fb @ ys yc @ inits yc),
    count
      (Array.map (fun i -> (shape_of c.a).(i)) ka)
      (along c.a ka @ along c.b (on snd c.contracting)) )

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
  cover "an operand acc does not hold" (converted c);
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
    [ "stepped"; "broadcast"; "stored apart"; "permuted" ];
  let rb, rr, rc, rk = runs c in
  cover "a batch of several runs" (rb > 1);
  cover "rows of several runs" (rr > 1);
  cover "columns of several runs" (rc > 1);
  cover "contracted axes of several runs" (rk > 1);
  let ib, ir, ic, _ = runs ~init_alone:true c in
  cover "an init whose group makes several runs" (ib > 1 || ir > 1 || ic > 1);
  cover "a batch of several runs over 512 KiB of operands or more"
    (rb > 1 && batch * ((m * k) + (k * n) + (m * n)) * 4 >= 512 * 1024)

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
  let dst = A.Any (A.create b.device out y) in
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

(* An output [x] of [c] is within γ(K + 1, 2u) (|init| + Σ|a·b|) of the exact
   sum of [o], u being [acc]'s unit roundoff, and (K + 1) η where results
   underflow, η [acc]'s smallest subnormal, twice the most a rounding into
   the subnormals errs; a NaN or infinite sum is matched. *)
let within c o x =
  let naive = ref (Option.value ~default:0. o.init) in
  for k = 0 to Array.length o.av - 1 do
    naive := !naive +. (o.av.(k) *. o.bv.(k))
  done;
  let naive = !naive in
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
  | None -> cover "declined" true
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

(* A result of another dtype than acc is the result in acc, from init cast
   to acc, cast to it: the sum rounds once. *)
let law_rounds_once b c =
  cover "a narrow float result" (List.mem c.out narrow);
  cover "an integer acc" (is_integer c.acc);
  cover "a 4-bit acc"
    (let (D.Any acc) = c.acc in
     is_integer c.acc && D.bits acc = 4);
  cover "a complex acc" (is_complex c.acc);
  cover "init" (c.init <> None);
  cover "a group of several runs"
    (let rb, rr, rc, rk = runs c in
     rb > 1 || rr > 1 || rc > 1 || rk > 1);
  match run b c with
  | None when cpu_computes c ->
      failf "nx.cpu declined a case nx_cpu.mli says it computes"
  | None -> cover "declined" true
  | Some y -> (
      let c' =
        { c with out = c.acc; init = Option.map (cast_to c.acc) c.init }
      in
      match run b c' with
      | None -> failf "nx.cpu declined the case into acc"
      | Some ya ->
          cover "computed" true;
          let want = cast_to c.out ya in
          equal (list string) []
            (differ_bits c.out ~other:"the cast of acc's" y want))

(* Integers *)

let integer_cases =
  Gen.with_pp pp_case
    (Gen.bind (Gen.of_list ~pp:pp_dtype integer_accs) (fun acc ->
         Gen.bind
           (Gen.frequency
              [
                (2, Gen.constant integers); (1, Gen.constant (others integers));
              ])
           (fun dts -> case_of ~acc ~out:acc ~dts ())))

(* [x]'s values in an array one element longer along its last axis, read
   from its second element: adjacent along that axis, at an odd offset. *)
let shifted (A.Any x as a) =
  let s = shape_of a in
  let last = Array.length s - 1 in
  let made = Array.mapi (fun i e -> if i = last then e + 1 else e) s in
  let at i e = { M.start = (if i = last then 1 else 0); count = e; step = 1 } in
  let y =
    Option.get
      (A.move
         (M.Slice (Iarray.of_array (Array.mapi at s)))
         (A.create Rig.host (A.dtype x) made))
  in
  A.blit ~src:x ~dst:y;
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
  let plain r =
    let axes = Array.init r Fun.id in
    { stepped = false; broadcast = false; stored = axes; perm = axes }
  in
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
          let vs = A.Any (A.of_array D.Int64 s (Array.make (total s) v)) in
          let src = A.expect (A.dtype x) (cast_to (D.Any (A.dtype x)) vs) in
          A.blit ~src ~dst:x;
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
  cover "a 4-bit accumulator" (D.bits acc = 4);
  cover "an operand acc does not hold" (converted c);
  cover "a float or complex operand into a 4-bit accumulator"
    (D.bits acc = 4
    && Array.exists
         (fun (A.Any x) ->
           match D.kind (A.dtype x) with
           | D.Float | D.Complex -> true
           | _ -> false)
         (operands c));
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
  | None -> cover "declined" true
  | Some y ->
      let _, oa, ob, walk = index c in
      let ints x = ints (cast_to c.acc x) in
      let a = ints c.a and bv = ints c.b and init = Option.map ints c.init in
      let got = ints y and bad = ref [] in
      (* Each column of b gathered once, its values mapped by [f], as
         outputs gathers them: read across b's rows for every output, 33 ×
         1024 outputs of 4096 products took 1.3 s, 0.5 gathered. *)
      let gathered f =
        let columns = Hashtbl.create 256 in
        fun rb ->
          match Hashtbl.find_opt columns rb with
          | Some col -> col
          | None ->
              let col = Array.map (fun t -> f bv.(rb + t)) ob in
              Hashtbl.add columns rb col;
              col
      in
      (* An accumulator narrower than OCaml's 63-bit ints sums in them: their
         arithmetic wraps modulo 2^63, a multiple of its width. *)
      let (D.Any acc) = c.acc in
      let sum =
        if D.bits acc < 63 then (
          let a = Array.map Int64.to_int a and column = gathered Int64.to_int in
          fun s0 ra rb ->
            let col = column rb in
            let s = ref (Int64.to_int s0) in
            for t = 0 to Array.length oa - 1 do
              s := !s + (a.(ra + oa.(t)) * col.(t))
            done;
            Int64.of_int !s)
        else
          let column = gathered Fun.id in
          fun s0 ra rb ->
            let col = column rb in
            let s = ref s0 in
            for t = 0 to Array.length oa - 1 do
              s := Int64.add !s (Int64.mul a.(ra + oa.(t)) col.(t))
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
  cover "an operand acc does not hold" (converted c);
  cover "a result of another dtype" (c.out <> c.acc);
  equal ~msg:"computed" bool (cpu_computes c) (Option.is_some (run b c))

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
       K.contract ~order ~batch:[||] ~contracting:[| (1, 0) |] ~segments:None
         ~acc:(D.Any fd) ~dst ~init:(Some init) (plain a) (plain b)
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
             ~segments:None ~acc:(D.Any D.Float32) ~dst ~init (plain a)
             (plain b))
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

(* A contraction answers Shape_mismatch where Shape.contract answers
   Error for its operands, or where dst's shape is not the result's, and
   writes nothing: misfits in the pairs' extents, the operands' ranks and
   dst, a result past max_numel, and one behind a group that does not
   merge. *)
let test_misfits (b : Support.backend) () =
  let module K = (val b.kernels) in
  let arr s =
    A.expect D.Float32
      (on b (A.Any (A.of_array D.Float32 s (Array.make (total s) 1.))))
  in
  let answer = Testable.make ~pp:Nx_array_support.pp_answer ~equal:( = ) in
  let refused ~msg ?(batch : _ iarray = [||]) contracting dst a b' =
    let before = A.to_array (to64 (host (A.Any dst))) in
    equal ~msg answer A.Shape_mismatch
      (K.contract ~order:Chain ~batch ~contracting ~segments:None
         ~acc:(D.Any D.Float32) ~dst ~init:None (plain a) (plain b'));
    equal ~msg:(msg ^ ": untouched") (array float_exact) before
      (A.to_array (to64 (host (A.Any dst))))
  in
  let mm : _ iarray = [| (1, 0) |] in
  refused ~msg:"contracted extents" mm (arr [| 2; 3 |]) (arr [| 2; 4 |])
    (arr [| 5; 3 |]);
  refused ~msg:"dst's shape" mm (arr [| 3; 2 |]) (arr [| 2; 4 |])
    (arr [| 4; 3 |]);
  refused ~msg:"dst's rank" mm (arr [| 6 |]) (arr [| 2; 4 |]) (arr [| 4; 3 |]);
  refused ~msg:"a's rank" mm (arr [| 2; 3 |]) (arr [| 4 |]) (arr [| 4; 3 |]);
  let k = 1 lsl 27 in
  refused ~msg:"a result past max_numel" mm (arr [| 1; 1 |]) (arr [| k; 0 |])
    (arr [| 0; k |]);
  (* [a]'s batch axes transposed, its contracted extent 4 against [b]'s 5. *)
  let a =
    let x = arr [| 3; 2; 2; 4 |] in
    Option.get (A.move (Nx_array.Move.Permute [| 1; 0; 2; 3 |]) x)
  in
  refused ~msg:"behind an unmerged group"
    ~batch:[| (0, 0); (1, 1) |]
    [| (3, 2) |]
    (arr [| 2; 3; 2; 2 |])
    a
    (arr [| 2; 3; 5; 2 |])

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
      test "refuses operands and destinations whose shapes do not fit"
        (fun () -> b.around (test_misfits b));
    ]

(* Scales *)

(* How a scale's array lies: C-contiguous, stepped along its last axis, or
   broadcast along its first. *)
type lies = Contiguous | Stepped | Broadcast

(* A scale drawn for an operand: [Exponent] codes, or [Value]s of [kind];
   its extents; how it lies; its values from [seed], led by the specials
   where [specials]. *)
type spec = {
  kind : D.any option;
  extents : int array;
  lies : lies;
  specials : bool;
  seed : int;
}

let pp_spec ppf s =
  let kind ppf = function
    | None -> Format.pp_print_string ppf "Exponent"
    | Some d -> Format.fprintf ppf "Value %a" pp_dtype d
  in
  let lies = function
    | Contiguous -> "contiguous"
    | Stepped -> "stepped"
    | Broadcast -> "broadcast"
  in
  Format.fprintf ppf "%a %a %s%s (seed %d)" kind s.kind pp_ints s.extents
    (lies s.lies)
    (if s.specials then " with specials" else "")
    s.seed

let pp_scaled ppf (c, sa, sb) =
  let spec ppf = function
    | None -> Format.pp_print_string ppf "none"
    | Some s -> pp_spec ppf s
  in
  Format.fprintf ppf "%a@ a scale %a@ b scale %a" pp_case c spec sa spec sb

(* The codes at a format's edges: the smallest scale, 1, the largest two,
   NaN. *)
let codes = [| 0; 127; 253; 254; 255 |]

(* [s]'s scale. Codes are near 1, values as [value] draws them; with
   [specials], led by each code of [codes] or by NaN. *)
let scale_of s : Nx_kernel.scale =
  let r = Random.State.make [| s.seed |] in
  let e = s.extents and rank = Array.length s.extents in
  let stored =
    match s.lies with
    | Stepped when rank > 0 ->
        Array.mapi (fun i x -> if i = rank - 1 then 2 * x else x) e
    | Broadcast when rank > 0 -> Array.mapi (fun i x -> if i = 0 then 1 else x) e
    | _ -> e
  in
  let view (A.Any x) =
    let move m x = Option.get (A.move m x) in
    let step i x =
      if i = rank - 1 then { M.start = 1; count = x; step = 2 }
      else { M.start = 0; count = x; step = 1 }
    in
    match s.lies with
    | Stepped when rank > 0 ->
        A.Any (move (M.Slice (Iarray.of_array (Array.mapi step e))) x)
    | Broadcast when rank > 0 -> A.Any (move (M.Broadcast (Iarray.of_array e)) x)
    | _ -> A.Any x
  in
  let n = total stored in
  match s.kind with
  | None ->
      let code i =
        if s.specials && i < Array.length codes then codes.(i)
        else if Random.State.int r 4 = 0 then
          codes.(Random.State.int r (Array.length codes))
        else 117 + Random.State.int r 21
      in
      Exponent
        (A.expect D.Uint8 (view (A.Any (A.of_array D.Uint8 stored (Array.init n code)))))
  | Some dt ->
      let v i = if s.specials && i = 0 then nan else value dt r in
      let (A.Any x) =
        view (cast_to dt (A.Any (A.of_array D.Float64 stored (Array.init n v))))
      in
      Value x

let scale_extents : Nx_kernel.scale -> int array = function
  | Exponent x -> L.shape (A.layout x)
  | Value x -> L.shape (A.layout x)

(* A scale's values in C order of its indices, read into [acc]. *)
let decoded acc : Nx_kernel.scale -> float array = function
  | Exponent x ->
      Array.map
        (fun e -> if e = 255 then nan else Float.ldexp 1. (e - 127))
        (A.to_array x)
  | Value x -> A.to_array (to64 (cast_to acc (A.Any x)))

(* [x] read into [acc], each element times its block's scale of [s] rounded
   once in [acc], C-contiguous in [acc]: what a scaled contraction reads;
   and whether a product of finite factors overflowed. *)
let host_scaled acc x s =
  let shape = shape_of x and se = scale_extents s in
  let xs = A.to_array (to64 (cast_to acc x)) and ss = decoded acc s in
  let r = Array.length shape in
  let ratio = Array.init r (fun i -> if se.(i) = 0 then 1 else shape.(i) / se.(i)) in
  let stride = Array.make r 1 in
  for i = r - 2 downto 0 do
    stride.(i) <- stride.(i + 1) * se.(i + 1)
  done;
  let f32 = acc = D.Any D.Float32 and over = ref false in
  let scaled flat v =
    let q = ref flat and at = ref 0 in
    for i = r - 1 downto 0 do
      at := !at + (!q mod shape.(i) / ratio.(i) * stride.(i));
      q := !q / shape.(i)
    done;
    let y = v *. ss.(!at) in
    let y = if f32 then round32 y else y in
    if Float.is_finite v && Float.is_finite ss.(!at) && not (Float.is_finite y)
    then over := true;
    y
  in
  let ys = Array.mapi scaled xs in
  (cast_to acc (A.Any (A.of_array D.Float64 shape ys)), !over)

(* A scale for an operand of shape [s]: along each axis a ratio of 1, the
   whole axis, or a divisor of it. *)
let spec_for s =
  let open Gen in
  let axis e =
    let ds = List.filter (fun d -> d < e && e mod d = 0) [ 2; 3; 4; 32 ] in
    if e = 0 then of_list [ 0; 1 ]
    else
      frequency
        ([ (3, constant e); (2, constant 1) ]
        @ if ds = [] then [] else [ (3, map (fun d -> e / d) (of_list ds)) ])
  in
  let rec axes = function
    | [] -> constant []
    | e :: es ->
        let* x = axis e in
        let+ xs = axes es in
        x :: xs
  in
  let* extents = axes (Array.to_list s) in
  let* kind =
    frequency
      [
        (1, constant None);
        (1, map Option.some (of_list ~pp:pp_dtype D.[ Any Float32; Any Float64; Any Float16; Any Bfloat16 ]));
      ]
  in
  let* lies = of_list [ Contiguous; Contiguous; Stepped; Broadcast ] in
  let* specials = bool in
  let+ seed = int in
  { kind; extents = Array.of_list extents; lies; specials; seed }

(* [c] with a scale on [a], on [b] or on both. *)
let with_scales c =
  let open Gen in
  let* sa = spec_for (shape_of c.a) and+ sb = spec_for (shape_of c.b) in
  let+ which = of_list [ `A; `B; `Both; `Both ] in
  match which with
  | `A -> (c, Some sa, None)
  | `B -> (c, None, Some sb)
  | `Both -> (c, Some sa, Some sb)

(* Float cases of operands of any dtype, into acc or a narrow float. *)
let scaled_cases =
  let open Gen in
  with_pp pp_scaled
    (let* acc = of_list ~pp:pp_dtype D.[ Any Float32; Any Float64 ] in
     let* dts =
       frequency
         [ (2, constant (held acc)); (1, constant (others (held acc))) ]
     in
     let* out =
       frequency [ (3, constant ~pp:pp_dtype acc); (1, of_list ~pp:pp_dtype narrow) ]
     in
     let* c = case_of ~acc ~out ~dts () in
     with_scales c)

(* The ratio along a contraction of [k]: 32 where it divides [k], else
   [k]'s largest divisor up to 64. *)
let ratio_of k =
  if k mod 32 = 0 then 32
  else
    let rec down d = if k mod d = 0 then d else down (d - 1) in
    down (min k 64)

(* Products either side of lane order's blocks of 1024 and nx's cut at
   4096, in both orders and accs: a float32 [2; k] times bfloat16 [k; 3],
   each scaled in blocks along k; and every regime of both accs, which
   reaches each of gemm.c's paths, its operands scaled in turn in 2-D
   blocks, stepped, broadcast or contiguous: a and b, a alone, b alone. *)
let scaled_examples =
  let plain r =
    let axes = Array.init r Fun.id in
    { stepped = false; broadcast = false; stored = axes; perm = axes }
  in
  let spec ?(lies = Contiguous) kind extents seed =
    Some { kind; extents; lies; specials = true; seed }
  in
  let along (order, acc) k =
    let c =
      make ~order ~acc ~out:acc ([||], [| 2 |], [| 3 |], [| k |])
        (D.Any D.Float32, plain 2) (D.Any D.Bfloat16, plain 2) None k
    in
    let q = k / ratio_of k in
    (c, spec (Some (D.Any D.Float16)) [| 2; q |] 1, spec None [| q; 3 |] 2)
  in
  let ks = [ 1; 1023; 1024; 1025; 4095; 4096; 4097 ] in
  let pairs =
    Nx_kernel.[ (Chain, D.Any D.Float32); (Lane, D.Any D.Float64);
                (Lane, D.Any D.Float32); (Chain, D.Any D.Float64) ]
  in
  let blocks x =
    Array.map
      (fun e ->
        match List.find_opt (fun d -> d < e && e mod d = 0) [ 2; 3; 4 ] with
        | Some d -> e / d
        | None -> e)
      (shape_of x)
  in
  let regime i c =
    let lies = [| Stepped; Broadcast; Contiguous |].(i / 3 mod 3) in
    let sa = spec ~lies None (blocks c.a) i in
    let sb = spec ~lies (Some c.acc) (blocks c.b) i in
    match i mod 3 with
    | 0 -> (c, sa, sb)
    | 1 -> (c, sa, None)
    | _ -> (c, None, sb)
  in
  List.concat_map (fun p -> List.map (along p) ks) pairs
  @ List.mapi regime (regimes (D.Any D.Float32) @ regimes (D.Any D.Float64))

let scaled_covers (c, sa, sb) =
  cover "float32" (c.acc = D.Any D.Float32);
  cover "float64" (c.acc = D.Any D.Float64);
  cover "chain order" (c.order = Chain);
  cover "lane order" (c.order = Lane);
  let specs =
    List.filter_map
      (fun (x, s) -> Option.map (fun s -> (shape_of x, s)) s)
      [ (c.a, sa); (c.b, sb) ]
  in
  let any f = List.exists f specs in
  let axes f (x, s) = List.length (List.filter Fun.id (Array.to_list (Array.map2 f x s.extents))) in
  cover "an Exponent scale" (any (fun (_, s) -> s.kind = None));
  cover "a Value scale" (any (fun (_, s) -> s.kind <> None));
  cover "a ratio of 1" (any (fun xs -> axes (fun e d -> e > 1 && d = e) xs > 0));
  cover "a whole axis" (any (fun xs -> axes (fun e d -> e > 1 && d = 1) xs > 0));
  cover "2-D blocks" (any (fun xs -> axes (fun e d -> e > 1 && d < e) xs >= 2));
  cover "a stepped scale" (any (fun (_, s) -> s.lies = Stepped));
  cover "a broadcast scale" (any (fun (_, s) -> s.lies = Broadcast));
  cover "a transposed operand" (List.mem "permuted" c.views);
  cover "a broadcast operand" (List.mem "broadcast" c.views);
  let rb, rr, rc, rk = runs c in
  cover "groups of several runs" (rb > 1 || rr > 1 || rc > 1 || rk > 1);
  let m, n = rows_cols c in
  cover "lane order in passes: more than 8 rows and columns"
    (per_batch c > 0 && c.order = Lane && m > 8 && n > 8);
  (* gemm.c reads an operand of acc's dtype in place along a's last
     contracted axis or b's last column or contracted axis where it steps
     by 1 there, unless it is scaled. *)
  let unit (A.Any x) i = (L.strides (A.layout x)).(i) = 1 in
  let last l = Array.length l > 0 in
  let of_acc (A.Any x) = D.Any (A.dtype x) = c.acc in
  let con side = Array.map side c.contracting in
  let a_streams =
    let ks = con fst in
    of_acc c.a && last ks && unit c.a ks.(Array.length ks - 1)
  in
  let b_streams =
    let ks = con snd and rank = Array.length (shape_of c.b) in
    let paired j =
      Array.exists (fun (_, x) -> x = j) c.batch
      || Array.exists (fun (_, x) -> x = j) c.contracting
    in
    let free = List.filter (fun j -> not (paired j)) (List.init rank Fun.id) in
    of_acc c.b
    && ((free <> [] && unit c.b (List.nth free (List.length free - 1)))
       || (last ks && unit c.b ks.(Array.length ks - 1)))
  in
  cover "a scaled operand of acc's dtype adjacent along its streamed axis"
    ((sa <> None && a_streams) || (sb <> None && b_streams));
  let _, k, _ = sizes c in
  List.iter
    (fun n -> cover (Printf.sprintf "%d products" n) (k = n))
    [ 1; 1023; 1024; 1025; 4095; 4096; 4097 ];
  let made = List.map (fun (_, s) -> (s, scale_of s)) specs in
  let has_code e =
    List.exists
      (fun (_, sc) ->
        match sc with
        | Nx_kernel.Exponent x -> Array.mem e (A.to_array x)
        | Value _ -> false)
      made
  in
  Array.iter (fun e -> cover (Printf.sprintf "code %d" e) (has_code e)) codes;
  cover "a NaN value"
    (List.exists
       (fun (_, sc) ->
         match sc with
         | Nx_kernel.Value _ -> Array.exists Float.is_nan (decoded c.acc sc)
         | Exponent _ -> false)
       made)

(* A scaled contraction has the bits of the contraction of its operands
   scaled on the host, each element rounded once into acc. *)
let scaled_bits b (c, sa, sb) =
  let sa = Option.map scale_of sa and sb = Option.map scale_of sb in
  let scaled x = function None -> (x, false) | Some s -> host_scaled c.acc x s in
  let a, oa = scaled c.a sa and b', ob = scaled c.b sb in
  cover "a product that overflows" (oa || ob);
  match (run ?sa ?sb b c, run b { c with a; b = b' }) with
  | Some y, Some y' ->
      equal (list string) [] (differ_bits c.acc ~other:"host-scaled" y y')
  | _ -> failf "nx.cpu declined a float contraction"

let law_scaled b x =
  scaled_covers x;
  scaled_bits b x

let plain_view r =
  let axes = Array.init r Fun.id in
  { stepped = false; broadcast = false; stored = axes; perm = axes }

(* Block-scaled float4

   A [m × k] by float4 [k × n] product, b scaled in blocks of [mx_ratio]
   along k and of each column or all of them ([mx_whole]), as MXFP4's
   weights are: b stored [n; k] and read along k ([mx_along]), as a
   weight is, or stored [k; n]; its codes from the second of a byte where
   [mx_odd]; a float4 too where [mx_a], scaled likewise along k. *)
type mx = {
  mx_order : Nx_kernel.order;
  mx_acc : D.any;
  mx_m : int;
  mx_n : int;
  mx_k : int;
  mx_ratio : int;
  mx_whole : bool;
  mx_along : bool;
  mx_odd : bool;
  mx_a : bool;
  mx_kind : D.any option;
  mx_seed : int;
}

(* float4's values by code, e2m1: sign, two exponent bits, one of
   significand. *)
let e2m1 =
  [| 0.; 0.5; 1.; 1.5; 2.; 3.; 4.; 6.; -0.; -0.5; -1.; -1.5; -2.; -3.; -4.; -6. |]

(* A float4 array of shape [made], its codes drawn uniformly from
   [seed]. *)
let float4 made seed =
  let r = Random.State.make [| seed |] in
  let xs = Array.init (total made) (fun _ -> e2m1.(Random.State.int r 16)) in
  A.expect D.Float4_e2m1fn
    (cast_to (D.Any D.Float4_e2m1fn) (A.Any (A.of_array D.Float64 made xs)))

(* The distinct float4 codes of [x]. *)
let f4_codes x =
  let v = A.to_array (to64 x) in
  List.length
    (List.sort_uniq compare
       (Array.to_list (Array.map (fun f -> (f, Float.sign_bit f)) v)))

(* A float4 [k × n] array drawn from [seed]: made [n; k + odd] and
   permuted where [along], else [k; n + odd], then sliced from [odd] along
   the axis whose codes are adjacent. *)
let float4_b ~along ~odd k n seed =
  let made = if along then [| n; k + odd |] else [| k; n + odd |] in
  let x = float4 made seed in
  let move m x = Option.get (A.move m x) in
  let window = Iarray.of_list
      [ { M.start = 0; count = made.(0); step = 1 };
        { M.start = odd; count = made.(1) - odd; step = 1 } ]
  in
  let x = move (M.Slice window) x in
  A.Any (if along then move (M.Permute [| 1; 0 |]) x else x)

let mx_case x =
  let k = x.mx_k and n = x.mx_n and m = x.mx_m in
  let plain2 = { stepped = false; broadcast = false; stored = [| 0; 1 |]; perm = [| 0; 1 |] } in
  let da = if x.mx_a then D.Any D.Float4_e2m1fn else D.Any D.Bfloat16 in
  let a = operand da [| m; k |] 0 plain2 x.mx_seed in
  let odd = if x.mx_odd then 1 else 0 in
  let b = float4_b ~along:x.mx_along ~odd k n (x.mx_seed + 1) in
  let c =
    { order = x.mx_order; batch = [||]; contracting = [| (1, 0) |];
      acc = x.mx_acc; a; b; init = None; out = x.mx_acc; views = [] }
  in
  let blocks = if k = 0 then 0 else k / x.mx_ratio in
  let spec extents i =
    { kind = x.mx_kind; extents; lies = Contiguous; specials = true;
      seed = x.mx_seed + i }
  in
  let cols = if x.mx_whole && n > 0 then 1 else n in
  let sa = if x.mx_a then Some (spec [| m; blocks |] 2) else None in
  (c, sa, Some (spec [| blocks; cols |] 3))

let pp_mx ppf x = pp_scaled ppf (mx_case x)

let mx_of ?(order = Nx_kernel.Lane) ?(acc = D.Any D.Float32) ?(ratio = 32)
    ?(whole = false) ?(along = true) ?(odd = false) ?(f4_a = false) ?kind
    ?(seed = 1) m n k =
  { mx_order = order; mx_acc = acc; mx_m = m; mx_n = n; mx_k = k;
    mx_ratio = ratio; mx_whole = whole; mx_along = along; mx_odd = odd;
    mx_a = f4_a; mx_kind = kind; mx_seed = seed }

let mx_draws =
  let open Gen in
  with_pp pp_mx
    (let* order = of_list Nx_kernel.[ Chain; Lane ] in
     let* acc = of_list ~pp:pp_dtype D.[ Any Float32; Any Float64 ] in
     let* k = of_list [ 0; 32; 64; 96; 1024; 1056; 2048; 4096; 4128 ] in
     let* ratio =
       if k = 0 then constant 32
       else of_list (List.filter (fun r -> k mod r = 0) [ 32; 64; 96 ])
     in
     let* m = frequency [ (3, int_range 1 5); (2, int_range 6 40); (1, constant 0) ] in
     let* n = frequency [ (3, int_range 1 40); (1, of_list [ 0; 1; 300 ]) ] in
     let* whole = bool and+ along = frequency [ (3, constant true); (1, constant false) ] in
     let* odd = bool and+ f4_a = frequency [ (4, constant false); (1, constant true) ] in
     let* kind =
       frequency
         [ (3, constant None);
           (1, map Option.some (of_list ~pp:pp_dtype D.[ Any Float16; Any Float32 ])) ]
     in
     let+ seed = int in
     mx_of ~order ~acc ~ratio ~whole ~along ~odd ~f4_a ?kind ~seed m n k)

(* The product of a token by a weight, either side of lane order's blocks
   and nx's cut, in both orders and accs; a weight read along its columns;
   codes from inside a byte; float4 a; Value scales; the chain path, past
   the few rows; and the empty and one-element products. *)
let mx_examples =
  let both =
    Nx_kernel.[ (Lane, D.Any D.Float32); (Chain, D.Any D.Float64);
                (Chain, D.Any D.Float32); (Lane, D.Any D.Float64) ]
  in
  List.concat_map
    (fun (order, acc) ->
      List.map (fun k -> mx_of ~order ~acc 1 24 k) [ 32; 1024; 1056; 4096; 4128 ]
      @ [ mx_of ~order ~acc ~along:false ~odd:true 3 17 1056;
          mx_of ~order ~acc ~odd:true ~ratio:64 2 9 4096;
          mx_of ~order ~acc ~f4_a:true ~whole:true 5 7 2048;
          mx_of ~order ~acc ~kind:(D.Any D.Float16) 4 33 1024;
          mx_of ~order ~acc 1 1 32; mx_of ~order ~acc 0 5 64;
          mx_of ~order ~acc 3 0 64; mx_of ~order ~acc 2 3 0 ])
    both
  @ [ mx_of ~order:Chain ~acc:(D.Any D.Float32) 260 264 1056;
      mx_of ~order:Chain ~acc:(D.Any D.Float64) ~along:false 260 264 1056 ]

let mx_covers x (c, _, _) =
  cover "every float4 code" (f4_codes c.b = 16);
  let lane = x.mx_order = Lane and products = x.mx_m * x.mx_n > 0 && x.mx_k > 0 in
  cover "float32" (x.mx_acc = D.Any D.Float32);
  cover "float64" (x.mx_acc = D.Any D.Float64);
  cover "lane order, b read along k" (lane && products && x.mx_along);
  cover "lane order, b read along its columns" (lane && products && not x.mx_along);
  cover "chain order, b read along k" ((not lane) && products && x.mx_along);
  cover "chain order, b read along its columns"
    ((not lane) && products && not x.mx_along);
  cover "the chain path" ((not lane) && x.mx_m > few_rows_most && x.mx_n > few_rows_most);
  cover "k past a block of lanes" (x.mx_k > 1024);
  cover "k past nx's block" (x.mx_k > 4096);
  cover "blocks of 64 or more" (x.mx_k > 0 && x.mx_ratio > 32);
  cover "codes from inside a byte" (products && x.mx_odd);
  cover "a float4 a" (products && x.mx_a);
  cover "a Value scale" (products && x.mx_kind <> None);
  cover "a scale per column" (products && x.mx_n > 1 && not x.mx_whole);
  cover "no output" (x.mx_m * x.mx_n = 0);
  cover "one output" (x.mx_m * x.mx_n = 1);
  cover "no product" (x.mx_k = 0 && x.mx_m * x.mx_n > 0)

let law_mx b x =
  let c = mx_case x in
  mx_covers x c;
  scaled_bits b c

(* Block-scaled float4 weights through other layouts, in both orders and
   accs: two contracting pairs that merge, one apart, and pairs in
   another order than b's axes; b stepped along k, broadcast across its
   columns; a batch axis. *)
let mx_layouts =
  let move m x = A.Any (Option.get (A.move m x)) in
  let bf16 s seed = operand (D.Any D.Bfloat16) s 0 (plain_view (Array.length s)) seed in
  let codes extents = Some { kind = None; extents; lies = Contiguous; specials = true; seed = 5 } in
  let range start count step = { M.start; count; step } in
  let layouts =
    [
      ("two pairs that merge", [||], [| (1, 0); (2, 1) |], bf16 [| 3; 2; 64 |] 1,
       move (M.Permute [| 1; 2; 0 |]) (float4 [| 5; 2; 64 |] 2), [| 2; 2; 5 |]);
      ("two pairs apart, k1 adjacent", [||], [| (1, 0); (2, 1) |], bf16 [| 3; 2; 64 |] 1,
       move (M.Permute [| 2; 1; 0 |]) (float4 [| 5; 64; 2 |] 2), [| 2; 2; 5 |]);
      ("pairs in the reverse of b's axes", [||], [| (2, 1); (1, 0) |],
       bf16 [| 3; 2; 64 |] 1, move (M.Permute [| 1; 2; 0 |]) (float4 [| 5; 2; 64 |] 2),
       [| 2; 2; 5 |]);
      ("b stepped along k", [||], [| (1, 0) |], bf16 [| 2; 1056 |] 1,
       move (M.Permute [| 1; 0 |])
         (Option.get
            (A.move (M.Slice [| range 0 7 1; range 1 1056 2 |]) (float4 [| 7; 2112 |] 2))),
       [| 33; 7 |]);
      ("b broadcast across its columns", [||], [| (1, 0) |], bf16 [| 3; 1056 |] 1,
       move (M.Broadcast [| 1056; 9 |]) (float4 [| 1056; 1 |] 2), [| 33; 9 |]);
      ("a batch axis", [| (0, 0) |], [| (2, 1) |], bf16 [| 2; 3; 128 |] 1,
       move (M.Permute [| 0; 2; 1 |]) (float4 [| 2; 6; 128 |] 2), [| 2; 4; 6 |]);
    ]
  in
  List.concat_map
    (fun (name, batch, contracting, a, b, extents) ->
      List.map
        (fun (order, acc) ->
          ( { order; batch; contracting; acc; a; b; init = None; out = acc;
              views = [ name ] },
            None, codes extents ))
        Nx_kernel.[ (Lane, D.Any D.Float32); (Chain, D.Any D.Float64) ])
    layouts

let law_mx_layouts b ((c, _, _) as x) =
  let (A.Any w) = c.b in
  let st = L.strides (A.layout w) in
  cover "two contracting pairs" (Array.length c.contracting = 2);
  cover "b stepped along k" (st.(snd c.contracting.(0)) = 2);
  cover "b broadcast across its columns"
    (Array.length st = 2 && st.(1) = 0);
  cover "a batch axis" (c.batch <> [||]);
  scaled_bits b x

(* Integer and complex cases with scales. *)
let unscalable =
  let open Gen in
  with_pp pp_scaled
    (let* acc =
       of_list ~pp:pp_dtype
         D.[ Any Int4; Any Int8; Any Int32; Any Int64; Any Complex64; Any Complex128 ]
     in
     let dts = if is_integer acc then integers else held acc in
     let* c = case_of ~acc ~out:acc ~dts () in
     with_scales c)

(* nx.cpu declines a scale with an integer or complex acc, and writes
   nothing. *)
let law_unscalable b (c, sa, sb) =
  cover "integer" (is_integer c.acc);
  cover "complex" (is_complex c.acc);
  let y, _, _ = sizes c in
  let threes = A.Any (A.of_array D.Float64 y (Array.make (total y) 3.)) in
  let dst = on b (cast_to c.out threes) in
  let before = planes c.out (host dst) in
  let init = Option.map (on b) c.init in
  let sa = Option.map (fun s -> scale_on b (scale_of s)) sa in
  let sb = Option.map (fun s -> scale_on b (scale_of s)) sb in
  match contract ?sa ?sb b.kernels c ~dst (on b c.a) (on b c.b) init with
  | A.Declined ->
      equal (array (array float_exact)) before (planes c.out (host dst))
  | r -> failf "contract answered %a" Nx_array_support.pp_answer r

(* nx.cpu answers Shape_mismatch for a scale that does not fit its
   operand, a [2 × 4] by [4 × 3] product's, and writes nothing: of
   another rank, with an extent that does not divide the operand's, or of
   0 along an axis with elements. *)
let test_scale_misfits (b : Support.backend) () =
  let module K = (val b.kernels) in
  let arr s =
    A.expect D.Float32
      (on b (A.Any (A.of_array D.Float32 s (Array.make (total s) 1.))))
  in
  let codes s =
    Nx_kernel.Exponent
      (A.expect D.Uint8
         (on b (A.Any (A.of_array D.Uint8 s (Array.make (total s) 127)))))
  in
  let answer = Testable.make ~pp:Nx_array_support.pp_answer ~equal:( = ) in
  let refused ~msg ?sa ?sb () =
    let dst = arr [| 2; 3 |] in
    let before = A.to_array (to64 (host (A.Any dst))) in
    let op x scale : Nx_kernel.operand = { array = A.Any x; scale } in
    List.iter
      (fun acc ->
        equal ~msg answer A.Shape_mismatch
          (K.contract ~order:Chain ~batch:[||] ~contracting:[| (1, 0) |]
             ~segments:None ~acc ~dst ~init:None
             (op (arr [| 2; 4 |]) sa)
             (op (arr [| 4; 3 |]) sb));
        equal ~msg:(msg ^ ": untouched") (array float_exact) before
          (A.to_array (to64 (host (A.Any dst)))))
      D.[ Any Float32; Any Float64 ]
  in
  refused ~msg:"a's scale of rank 1" ~sa:(codes [| 2 |]) ();
  refused ~msg:"b's scale of rank 3" ~sb:(codes [| 1; 1; 1 |]) ();
  refused ~msg:"a's 3 blocks along 4" ~sa:(codes [| 2; 3 |]) ();
  refused ~msg:"b's 2 blocks along 3" ~sb:(codes [| 4; 2 |]) ();
  refused ~msg:"a's 4 blocks along 2" ~sa:(codes [| 4; 4 |]) ();
  refused ~msg:"b's 0 blocks along 4" ~sb:(codes [| 0; 3 |]) ();
  refused ~msg:"a fitting scale beside a misfit" ~sa:(codes [| 2; 1 |])
    ~sb:(codes [| 3; 3 |]) ()

(* A scaled contraction whose contracting extents multiply to 0 has no
   products, whatever the extents before the empty one: a [2; 2^28; 0] by
   [2^28; 0; 3], scaled along a, is [+0], or init, at every output. *)
let test_scaled_empty (b : Support.backend) () =
  let module K = (val b.kernels) in
  let big = 1 lsl 28 in
  let arr s x =
    A.expect D.Float64 (on b (A.Any (A.of_array D.Float64 s (Array.make (total s) x))))
  in
  let codes =
    A.expect D.Uint8 (on b (A.Any (A.of_array D.Uint8 [| 2; 1; 1 |] [| 127; 127 |])))
  in
  let at order init =
    let dst = arr [| 2; 3 |] 7. in
    (match
       K.contract ~order ~batch:[||] ~contracting:[| (1, 0); (2, 1) |]
         ~segments:None ~acc:(D.Any D.Float64) ~dst
         ~init:(Option.map (arr [| 2; 3 |]) init)
         { array = A.Any (arr [| 2; big; 0 |] 1.); scale = Some (Exponent codes) }
         (plain (arr [| big; 0; 3 |] 1.))
     with
    | A.Done -> ()
    | r -> failf "contract answered %a" Nx_array_support.pp_answer r);
    equal (array float_exact) (Array.make 6 (Option.value init ~default:0.))
      (A.to_array (to64 (host (A.Any dst))))
  in
  List.iter (fun o -> at o None; at o (Some (-2.5))) Nx_kernel.[ Chain; Lane ]

(* Segments

   A contraction cut into groups by offsets (Nx_kernel.segments): [Rows]
   cuts a free axis of a, each row taking b's group along a free axis of
   b; [Contracting] cuts a contracting pair, each group's window summed
   alone into its slice of a leading axis. Each is checked against
   unsegmented contractions of views, one a row or a group, whose bits
   nx.cpu states: a group sums as the call of its extents does. *)

(* What a segmented contraction cuts: a's free axis and b's, or a
   contracting pair. *)
type cut = Cut_rows of { a_axis : int; b_axis : int } | Cut_pair of int

(* A segmented contraction: [sg]'s order, pairs, dtypes and operands, its
   init of the segmented result's shape; [g] groups cut as [cut] says by
   the offsets [offs], [g + 1] for each batch element in C order, read
   through a stepped view where [stepped]; its operands' scales. [min_int]
   and [max_int] among [offs] stand for int64's (wide). *)
type seg = {
  sg : case;
  cut : cut;
  g : int;
  offs : int array array;
  stepped : bool;
  sa : spec option;
  sb : spec option;
}

(* The int64 offset [x] stands for. *)
let wide x =
  if x = min_int then Int64.min_int
  else if x = max_int then Int64.max_int
  else Int64.of_int x

let pp_seg ppf s =
  let cut ppf = function
    | Cut_rows { a_axis; b_axis } ->
        Format.fprintf ppf "rows of a's axis %d by b's axis %d" a_axis b_axis
    | Cut_pair p -> Format.fprintf ppf "contracting pair %d" p
  in
  let sep ppf () = Format.pp_print_string ppf " " in
  let offs ppf o =
    Format.fprintf ppf "[%s]"
      (String.concat "; " (Array.to_list (Array.map (fun x -> Int64.to_string (wide x)) o)))
  in
  Format.fprintf ppf "%a@ %a, %d groups, offsets %a%s" pp_scaled
    (s.sg, s.sa, s.sb) cut s.cut s.g
    (Format.pp_print_list ~pp_sep:sep offs)
    (Array.to_list s.offs)
    (if s.stepped then " stepped" else "")

(* The batch extents: a's along the batch pairs. *)
let batch_extents c = Array.map (fun (i, _) -> (shape_of c.a).(i)) c.batch

(* The extent of the axis [s] cuts. *)
let cut_extent s =
  let a = shape_of s.sg.a in
  match s.cut with
  | Cut_rows { a_axis; _ } -> a.(a_axis)
  | Cut_pair p -> a.(fst s.sg.contracting.(p))

let segments_of s offsets : _ Nx_kernel.segments =
  match s.cut with
  | Cut_rows { a_axis; b_axis } -> Rows { offsets; a_axis; b_axis }
  | Cut_pair pair -> Contracting { offsets; pair }

(* [s]'s offsets, int64 of the batch extents then [g + 1]: where
   [stepped], every other element of an array twice as long along its
   last axis. *)
let offsets_of s =
  let shape = Array.append (batch_extents s.sg) [| s.g + 1 |] in
  let xs = Array.concat (Array.to_list s.offs) in
  if not s.stepped then A.of_array D.Int64 shape (Array.map wide xs)
  else
    let r = Array.length shape and w = s.g + 1 in
    let made = Array.mapi (fun i e -> if i = r - 1 then 2 * e else e) shape in
    let at q = if q mod 2 = 0 then 999L else wide xs.((q / (2 * w) * w) + (q mod (2 * w) / 2)) in
    let x = A.of_array D.Int64 made (Array.init (total made) at) in
    let step i e =
      if i = r - 1 then { M.start = 1; count = e; step = 2 }
      else { M.start = 0; count = e; step = 1 }
    in
    Option.get (A.move (M.Slice (Iarray.of_array (Array.mapi step shape))) x)

let seg_shape s =
  let c = s.sg in
  let o = Array.append (batch_extents c) [| s.g + 1 |] in
  match
    Nx_kernel.Shape.contract ~batch:(Iarray.of_array c.batch)
      ~contracting:(Iarray.of_array c.contracting)
      ~segments:(Some (segments_of s o)) (shape_of c.a, None)
      (shape_of c.b, None)
  with
  | Ok y -> y
  | Error e -> failf "the segments do not fit: %s" e

(* The group of index [r] under the offsets [o] of [g] groups, as
   Nx_kernel.segments states it: the largest [e < g] with [o.(e) <= r]
   when [o.(0) <= r < o.(g)]. *)
let group_of o g r =
  if g = 0 || r < o.(0) || r >= o.(g) then None
  else
    let e = ref (g - 1) in
    while o.(!e) > r do
      decr e
    done;
    Some !e

(* The indices of the cut axis in group [e] under [o]. *)
let members s o e =
  List.filter (fun r -> group_of o s.g r = Some e) (List.init (cut_extent s) Fun.id)

(* [x] cut along each axis [i] of [cuts] to [count] indices from
   [start]. *)
let restrict (A.Any x as a) cuts =
  let r = Array.map (fun e -> { M.start = 0; count = e; step = 1 }) (shape_of a) in
  List.iter (fun (i, start, count) -> r.(i) <- { M.start; count; step = 1 }) cuts;
  A.Any (Option.get (A.move (M.Slice (Iarray.of_array r)) x))

let reshaped (A.Any x) s =
  A.Any (Option.get (A.move (M.Reshape (Iarray.of_array s)) x))

(* The axes of a side of rank [rank] that no pair of [c] names. *)
let free side c rank =
  let paired i =
    Array.exists (fun p -> side p = i) c.batch
    || Array.exists (fun p -> side p = i) c.contracting
  in
  List.filter (fun i -> not (paired i)) (List.init rank Fun.id)

let before l x = List.length (List.filter (fun i -> i < x) l)

(* [s]'s result by unsegmented contractions of views: each row in a group
   alone with its group of b, or each group's window alone, of the
   operands scaled on the host; init as acc holds it, or +0, at every
   other output. *)
let seg_reference (b : Support.backend) s sa sb =
  let module K = (val b.kernels) in
  let c = s.sg in
  let scaled x = function None -> x | Some sc -> fst (host_scaled c.acc x sc) in
  let a = scaled c.a sa and b' = scaled c.b sb in
  let y = seg_shape s in
  let start =
    match c.init with
    | Some i -> cast_to c.acc i
    | None -> A.Any (A.of_array D.Float64 y (Array.make (total y) 0.))
  in
  let dst = cast_to c.out start in
  let nb = Array.length c.batch in
  let call (A.Any d) a b init =
    let init = Option.map (A.expect (A.dtype d)) init in
    match
      K.contract ~order:c.order ~batch:(Iarray.of_array c.batch)
        ~contracting:(Iarray.of_array c.contracting) ~segments:None ~acc:c.acc
        ~dst:d ~init
        { array = a; scale = None }
        { array = b; scale = None }
    with
    | A.Done -> ()
    | r -> failf "an unsegmented contraction answered %a" Nx_array_support.pp_answer r
  in
  let along side beta = List.init nb (fun k -> (side c.batch.(k), beta.(k), 1)) in
  let at shift beta = List.init nb (fun k -> (k + shift, beta.(k), 1)) in
  let row beta o =
    match s.cut with
    | Cut_rows { a_axis; b_axis } ->
        let fa = free fst c (Array.length (shape_of c.a)) in
        let fb = free snd c (Array.length (shape_of c.b)) in
        let ya = nb + before fa a_axis in
        let yb = nb + List.length fa + before fb b_axis in
        (* dst's row [r], with an axis of 1 where b's group was. *)
        let lift r x =
          let x = restrict x ((ya, r, 1) :: at 0 beta) in
          let sh = shape_of x in
          reshaped x
            (Array.concat
               [ Array.sub sh 0 yb; [| 1 |]; Array.sub sh yb (Array.length sh - yb) ])
        in
        for r = 0 to cut_extent s - 1 do
          match group_of o s.g r with
          | None -> ()
          | Some e ->
              call (lift r dst)
                (restrict a ((a_axis, r, 1) :: along fst beta))
                (restrict b' ((b_axis, e, 1) :: along snd beta))
                (Option.map (lift r) c.init)
        done
    | Cut_pair p ->
        let ia, ib = c.contracting.(p) in
        (* dst's slice [e], without the groups' axis. *)
        let lift e x =
          let x = restrict x ((0, e, 1) :: at 1 beta) in
          let sh = shape_of x in
          reshaped x (Array.sub sh 1 (Array.length sh - 1))
        in
        for e = 0 to s.g - 1 do
          let lo, count =
            match members s o e with [] -> (0, 0) | r :: _ as l -> (r, List.length l)
          in
          call (lift e dst)
            (restrict a ((ia, lo, count) :: along fst beta))
            (restrict b' ((ib, lo, count) :: along snd beta))
            (Option.map (lift e) c.init)
        done
  in
  List.iteri (fun q beta -> row beta s.offs.(q)) (indices (batch_extents c));
  dst

(* Offsets of [g] groups along [n] indices, from [r]: a partition of the
   axis; increasing within it; every index in one group; or any values
   from -2 to [n + 2] and int64's extremes, unordered. *)
let draw_offsets r g n =
  let sorted () =
    let o = Array.init (g + 1) (fun _ -> Random.State.int r (n + 1)) in
    Array.sort compare o;
    o
  in
  match Random.State.int r 4 with
  | 0 ->
      let o = sorted () in
      o.(0) <- 0;
      o.(g) <- n;
      o
  | 1 -> sorted ()
  | 2 ->
      let e = Random.State.int r (max g 1) in
      Array.init (g + 1) (fun i -> if i <= e then 0 else n)
  | _ ->
      let any _ =
        match Random.State.int r (n + 7) with
        | x when x = n + 5 -> min_int
        | x when x = n + 6 -> max_int
        | x -> x - 2
      in
      Array.init (g + 1) any

(* The segmented contraction of the extents [e] as [make] makes it, cut by
   [cut], positions among [e]'s rows and columns or its contracted axes,
   with an init of the segmented result's shape through [vi]. *)
let seg_make ~order ~acc ~out ((batch, _, _, con) as e) (da, va) (db, vb) vi
    ~cut ~g ?(stepped = false) offs seed =
  let c = make ~order ~acc ~out e (da, va) (db, vb) None seed in
  let nb = Array.length batch and nk = Array.length con in
  let ia = inverse va.perm and ib = inverse vb.perm in
  let cut =
    match cut with
    | `Rows (ri, ci) -> Cut_rows { a_axis = ia.(nb + ri); b_axis = ib.(nb + nk + ci) }
    | `Pair p -> Cut_pair p
  in
  let s = { sg = c; cut; g; offs; stepped; sa = None; sb = None } in
  match vi with
  | None -> s
  | Some vi ->
      let y = seg_shape s in
      let init = operand c.out (Array.map (fun i -> y.(i)) (inverse vi.perm)) 0 vi (seed + 2) in
      { s with sg = { c with init = Some init; views = c.views @ names vi } }

(* Draws: rows or a pair cut into up to 6 groups along up to 40 indices,
   or a pair of 4090 to 4200 products, over a batch of up to two axes
   whose elements draw their own offsets, of float accs mostly, with
   scales where acc is a float. *)
let seg_draws =
  let open Gen in
  let axes most lo hi = array ~size:(int_range 0 most) (int_range lo hi) in
  let put x i v =
    Array.concat [ Array.sub x 0 i; [| v |]; Array.sub x i (Array.length x - i) ]
  in
  with_pp pp_seg
    (let* order = of_list Nx_kernel.[ Chain; Lane ] in
     let* acc =
       frequency
         [ (4, of_list ~pp:pp_dtype D.[ Any Float32; Any Float64 ]);
           (1, of_list ~pp:pp_dtype D.[ Any Complex64; Any Int32 ]) ]
     in
     let float = not (is_integer acc || is_complex acc) in
     let dts = if is_integer acc then integers else held acc in
     let pick = frequency [ (3, constant ~pp:pp_dtype acc); (1, of_list ~pp:pp_dtype dts) ] in
     let* da = pick and+ db = pick in
     let* out =
       if float then frequency [ (3, constant ~pp:pp_dtype acc); (1, of_list ~pp:pp_dtype narrow) ]
       else constant ~pp:pp_dtype acc
     in
     let* g = frequency [ (1, constant 0); (2, constant 1); (6, int_range 2 6) ] in
     let* n = frequency [ (5, int_range 0 12); (2, int_range 13 40) ] in
     let* batch = axes 2 1 3 and+ rows_cut = bool in
     let* rows, cols, con, cut =
       if rows_cut then
         let* rows = axes 1 1 3 and+ cols = axes 1 0 4 in
         let* con =
           frequency [ (4, axes 2 0 5); (1, map (fun k -> [| k |]) (int_range 1000 1100)) ]
         in
         let+ ri = int_range 0 (Array.length rows)
         and+ ci = int_range 0 (Array.length cols) in
         (put rows ri n, put cols ci g, con, `Rows (ri, ci))
       else
         let* rows = axes 2 1 4 and+ cols = axes 2 1 4 and+ con = axes 1 1 3 in
         let* k = frequency [ (5, constant n); (1, int_range 4090 4200) ] in
         let+ p = int_range 0 (Array.length con) in
         (rows, cols, put con p k, `Pair p)
     in
     let e = (batch, rows, cols, con) in
     let nb = Array.length batch in
     let ra, rb, ry = ranks e in
     let ry = match cut with `Rows _ -> ry - 1 | `Pair _ -> ry + 1 in
     let* va = view ra nb and+ vb = view rb nb and+ vi = view ry 0 in
     let* with_init = bool and+ stepped = bool and+ seed = int and+ oseed = int in
     let r = Random.State.make [| oseed |] in
     let along = match cut with `Rows (ri, _) -> rows.(ri) | `Pair p -> con.(p) in
     let offs = Array.init (total batch) (fun _ -> draw_offsets r g along) in
     let s =
       seg_make ~order ~acc ~out e (da, va) (db, vb)
         (if with_init then Some vi else None)
         ~cut ~g ~stepped offs seed
     in
     if not float then constant s
     else
       let* sa = spec_for (shape_of s.sg.a) and+ sb = spec_for (shape_of s.sg.b) in
       let+ which = of_list [ `None; `None; `A; `B; `Both ] in
       match which with
       | `None -> s
       | `A -> { s with sa = Some sa }
       | `B -> { s with sb = Some sb }
       | `Both -> { s with sa = Some sa; sb = Some sb })

(* A scale of codes of [extents]. *)
let codes_of extents seed =
  Some { kind = None; extents; lies = Contiguous; specials = true; seed }

(* Experts: [t] bfloat16 tokens of [k] by float4 weights [g; n; k] of [g]
   experts, stored along k, scaled in blocks of 32 along it as MXFP4's
   are, the tokens cut into the experts' groups by [o]. *)
let experts ?(order = Nx_kernel.Lane) ?(acc = D.Any D.Float32) t g n k o =
  let a = operand (D.Any D.Bfloat16) [| t; k |] 0 (plain_view 2) 7 in
  let b = operand (D.Any D.Float4_e2m1fn) [| g; n; k |] 0 (plain_view 3) 8 in
  let c =
    { order; batch = [||]; contracting = [| (1, 2) |]; acc; a; b; init = None;
      out = acc; views = [] }
  in
  { sg = c; cut = Cut_rows { a_axis = 0; b_axis = 0 }; g; offs = [| o |];
    stepped = false; sa = None; sb = codes_of [| g; n; k / 32 |] 9 }

(* [m] bfloat16 rows of [k] by float4 [k; n] stored along k, scaled in
   blocks of 32 along it, k cut into windows by [o]. *)
let windows ?(order = Nx_kernel.Lane) ?(acc = D.Any D.Float32) m n k o =
  let a = operand (D.Any D.Bfloat16) [| m; k |] 0 (plain_view 2) 3 in
  let along = { (plain_view 2) with stored = [| 1; 0 |] } in
  let b = operand (D.Any D.Float4_e2m1fn) [| k; n |] 0 along 4 in
  let c =
    { order; batch = [||]; contracting = [| (1, 0) |]; acc; a; b; init = None;
      out = acc; views = [] }
  in
  { sg = c; cut = Cut_pair 0; g = Array.length o - 1; offs = [| o |];
    stepped = false; sa = None; sb = codes_of [| k / 32; n |] 5 }

(* Cases each cover names, in both orders and accs, with the MXFP4 weights
   of experts and windows of k either side of nx's block, within and on
   the scales' blocks. *)
let seg_examples =
  let open Nx_kernel in
  let f32 = D.Any D.Float32 and f64 = D.Any D.Float64 in
  let ex ?(order = Lane) ?(acc = f32) ?out ?(da = acc) ?(db = acc) ?(view = false)
      ?(init = false) ?stepped ?sa ?sb ~cut ~g e offs =
    let (batch, _, _, _) = e in
    let nb = Array.length batch in
    let ra, rb, ry = ranks e in
    let ry = match cut with `Rows _ -> ry - 1 | `Pair _ -> ry + 1 in
    let v r =
      if not view then plain_view r
      else
        let p = Array.init r (fun i -> r - 1 - i) in
        { stepped = true; broadcast = nb > 0; stored = Array.init r Fun.id; perm = p }
    in
    let out = Option.value out ~default:acc in
    let s =
      seg_make ~order ~acc ~out e (da, v ra) (db, v rb)
        (if init then Some (v ry) else None)
        ~cut ~g ?stepped offs 11
    in
    let blocks x d = Array.map (fun e -> if e mod d = 0 then e / d else e) (shape_of x) in
    let spec x = function
      | None -> None
      | Some d -> Some { kind = Some (D.Any D.Float16); extents = blocks x d; lies = Stepped; specials = true; seed = d }
    in
    { s with sa = spec s.sg.a sa; sb = spec s.sg.b sb }
  in
  [
    ex ~init:true ~cut:(`Rows (0, 0)) ~g:3 ([||], [| 20 |], [| 3; 4 |], [| 7 |])
      [| [| 2; 5; 5; 18 |] |];
    ex ~order:Chain ~acc:f64 ~view:true ~stepped:true ~cut:(`Rows (0, 0)) ~g:2
      ([| 2 |], [| 40 |], [| 2 |], [| 5; 3 |])
      [| [| 0; 3; 40 |]; [| 0; 40; 40 |] |];
    ex ~order:Chain ~out:(D.Any D.Float16) ~init:true ~sa:2 ~cut:(`Rows (1, 0)) ~g:4
      ([||], [| 2; 9 |], [| 4 |], [| 1030 |])
      [| [| 5; 2; 9; 7; 3 |] |];
    ex ~acc:f64 ~sb:2 ~cut:(`Rows (0, 1)) ~g:5 ([||], [| 6 |], [| 2; 5 |], [| 4 |])
      [| [| -2; 4; 11; 4; 12; 3 |] |];
    ex ~init:true ~cut:(`Rows (0, 0)) ~g:3 ([||], [| 6 |], [| 3; 2 |], [| 5 |])
      [| [| min_int; 2; max_int; 5 |] |];
    (* a's blocks of 4 rows and b's of 2 groups, cut at rows 2, 7 and 9. *)
    (let s =
       ex ~init:true ~cut:(`Rows (0, 0)) ~g:4 ([||], [| 12 |], [| 4 |], [| 64 |])
         [| [| 0; 2; 7; 9; 12 |] |]
     in
     let spec extents seed =
       Some { kind = Some (D.Any D.Float16); extents; lies = Stepped; specials = true; seed }
     in
     { s with sa = spec [| 3; 2 |] 4; sb = spec [| 2; 2 |] 2 });
    ex ~init:true ~cut:(`Rows (0, 0)) ~g:0 ([||], [| 3 |], [| 0 |], [| 2 |]) [| [| 1 |] |];
    ex ~acc:(D.Any D.Complex64) ~init:true ~cut:(`Rows (0, 0)) ~g:2
      ([| 2 |], [| 5 |], [| 2 |], [| 3 |])
      [| [| 1; 1; 4 |]; [| 0; 2; 5 |] |];
    ex ~cut:(`Pair 0) ~g:3 ([||], [| 2 |], [| 3 |], [| 4200 |])
      [| [| 10; 1000; 4097; 4200 |] |];
    ex ~order:Chain ~acc:f64 ~init:true ~view:true ~sa:3 ~sb:3 ~cut:(`Pair 1) ~g:2
      ([| 3 |], [| 2 |], [| 2; 2 |], [| 3; 30 |])
      [| [| 0; 0; 30 |]; [| 0; 1; 30 |]; [| 31; -1; 2 |] |];
    ex ~order:Chain ~out:(D.Any D.Bfloat16) ~init:true ~cut:(`Pair 0) ~g:0
      ([||], [| 2 |], [| 2 |], [| 5 |]) [| [| 5 |] |];
    ex ~cut:(`Pair 0) ~g:3 ([||], [| 2 |], [| 3 |], [| 6 |])
      [| [| 1; max_int; 3; max_int |] |];
    ex ~acc:(D.Any D.Int32) ~da:(D.Any D.Int8) ~init:true ~cut:(`Pair 0) ~g:3
      ([||], [| 3 |], [| 2 |], [| 9 |])
      [| [| 0; 4; 4; 9 |] |];
    experts 37 5 9 1056 [| 0; 7; 7; 20; 36; 37 |];
    experts ~order:Chain ~acc:f64 60 4 6 64 [| 0; 3; 43; 43; 50 |];
    experts ~order:Chain 1 4 300 4096 [| 0; 0; 1; 1; 1 |];
    windows 3 5 4160 [| 0; 16; 1040; 4100; 4160 |];
    windows ~order:Chain ~acc:f64 3 5 4160 [| 0; 32; 1056; 4128; 4160 |];
    windows ~order:Chain 2 4 4160 [| 0; 16; 1040; 4100; 4160 |];
    windows ~acc:f64 2 4 4160 [| 0; 32; 1056; 4128; 4160 |];
  ]

let seg_covers s =
  let c = s.sg and n = cut_extent s in
  let rows = match s.cut with Cut_rows _ -> true | Cut_pair _ -> false in
  cover "rows" rows;
  cover "a contracting pair" (not rows);
  cover "float32" (c.acc = D.Any D.Float32);
  cover "float64" (c.acc = D.Any D.Float64);
  cover "a complex accumulator" (is_complex c.acc);
  cover "an integer accumulator" (is_integer c.acc);
  cover "chain order" (c.order = Chain);
  cover "lane order" (c.order = Lane);
  let some f = Array.exists f s.offs in
  let groups o f = List.exists (fun e -> f e (members s o e)) (List.init s.g Fun.id) in
  cover "no group" (s.g = 0);
  cover "an empty group" (some (fun o -> groups o (fun _ m -> m = [])));
  cover "a group of one" (some (fun o -> groups o (fun _ m -> List.length m = 1)));
  cover "every index in one group"
    (n > 1 && some (fun o -> groups o (fun _ m -> List.length m = n)));
  cover "offsets at the axis's bounds" (s.g > 0 && some (fun o -> o.(0) = 0 && o.(s.g) = n));
  cover "indices before the first group" (n > 0 && some (fun o -> o.(0) > 0));
  cover "indices past the last group" (some (fun o -> o.(s.g) < n && o.(s.g) > 0));
  cover "offsets that decrease"
    (some (fun o -> Array.exists Fun.id (Array.init s.g (fun e -> o.(e + 1) < o.(e)))));
  cover "offsets outside the axis" (some (Array.exists (fun x -> x < 0 || x > n)));
  cover "offsets at int64's extremes"
    (some (Array.exists (fun x -> x = min_int || x = max_int)));
  cover "offsets differing across the batch"
    (Array.exists (fun o -> o <> s.offs.(0)) s.offs);
  cover "stepped offsets" s.stepped;
  cover "an init" (c.init <> None);
  cover "a result of another dtype" (c.out <> c.acc);
  cover "an operand through a view" (c.views <> []);
  cover "a scaled operand" (s.sa <> None || s.sb <> None);
  let float4 = function
    | A.Any x -> D.Any (A.dtype x) = D.Any D.Float4_e2m1fn
  in
  cover "a block-scaled float4 operand" (s.sb <> None && float4 c.b);
  let starts f = some (fun o -> groups o (fun _ m -> match m with [] -> false | lo :: _ -> f lo (List.length m))) in
  cover "a group past a tile of 16 rows" (rows && starts (fun lo k -> lo mod 16 <> 0 && k > 16));
  cover "a window across nx's block of 4096 products"
    ((not rows) && starts (fun lo k -> lo < 4096 && lo + k > 4096));
  cover "a window across lane order's block of 1024"
    ((not rows) && starts (fun lo k -> lo < 1024 && lo + k > 1024));
  let ratio = match s.sb with
    | Some sp when not rows -> (
        let ib = snd c.contracting.(match s.cut with Cut_pair p -> p | _ -> 0) in
        let e = (shape_of c.b).(ib) and d = sp.extents.(ib) in
        if d = 0 then 1 else e / d)
    | _ -> 1
  in
  cover "a window from within a scale's block"
    (ratio > 1 && starts (fun lo _ -> lo mod ratio <> 0));
  let row_ratio = match (s.sa, s.cut) with
    | Some sp, Cut_rows { a_axis; _ } ->
        let d = sp.extents.(a_axis) in
        if d = 0 then 1 else (shape_of c.a).(a_axis) / d
    | _ -> 1
  in
  cover "a group of rows from within a scale's block"
    (row_ratio > 1 && starts (fun lo _ -> lo mod row_ratio <> 0));
  cover "windows on a scale's blocks" (ratio >= 32 && starts (fun lo _ -> lo > 0 && lo mod ratio = 0))

(* A segmented contraction has the bits of its unsegmented contractions of
   views, a row or a group at a time (seg_reference). *)
let law_segments (b : Support.backend) s =
  seg_covers s;
  let c = s.sg in
  let sa = Option.map scale_of s.sa and sb = Option.map scale_of s.sb in
  let (D.Any out) = c.out in
  let dst = A.create b.device out (seg_shape s) in
  let module K = (val b.kernels) in
  let init = Option.map (fun i -> A.expect out (on b i)) c.init in
  let offsets = A.expect D.Int64 (on b (A.Any (offsets_of s))) in
  let sa' = Option.map (scale_on b) sa and sb' = Option.map (scale_on b) sb in
  match
    K.contract ~order:c.order ~batch:(Iarray.of_array c.batch)
      ~contracting:(Iarray.of_array c.contracting)
      ~segments:(Some (segments_of s offsets)) ~acc:c.acc ~dst ~init
      { array = on b c.a; scale = sa' }
      { array = on b c.b; scale = sb' }
  with
  | A.Done ->
      equal (list string) []
        (differ_bits c.acc ~other:"unsegmented" (host (A.Any dst))
           (seg_reference b s sa sb))
  | r -> failf "a segmented contraction answered %a" Nx_array_support.pp_answer r

(* nx.cpu answers Shape_mismatch for segments that do not fit, and writes
   nothing: offsets of another shape than the batch extents then [g + 1],
   a cut axis in a pair or past its operand's rank, a pair past the pairs,
   a result of another shape than the segmented one. No offsets' values
   are refused: any values name their groups. *)
let test_segment_misfits (b : Support.backend) () =
  let module K = (val b.kernels) in
  let arr s =
    A.expect D.Float32
      (on b (A.Any (A.of_array D.Float32 s (Array.make (total s) 1.))))
  in
  let offs s =
    A.expect D.Int64
      (on b (A.Any (A.of_array D.Int64 s (Array.init (total s) Int64.of_int))))
  in
  let answer = Testable.make ~pp:Nx_array_support.pp_answer ~equal:( = ) in
  let refused ~msg ?(batch : _ iarray = [||])
      ?(contracting : _ iarray = [| (1, 0) |]) segments dst a b' =
    let before = A.to_array (to64 (host (A.Any dst))) in
    equal ~msg answer A.Shape_mismatch
      (K.contract ~order:Chain ~batch ~contracting ~segments:(Some segments)
         ~acc:(D.Any D.Float32) ~dst ~init:None (plain a) (plain b'));
    equal ~msg:(msg ^ ": untouched") (array float_exact) before
      (A.to_array (to64 (host (A.Any dst))))
  in
  let rows ?(a_axis = 0) ?(b_axis = 1) offsets =
    Nx_kernel.Rows { offsets; a_axis; b_axis }
  in
  let pair ?(pair = 0) offsets = Nx_kernel.Contracting { offsets; pair } in
  (* [2; 4] by [4; 3; 3] cut along b's axis 1 into 3 groups: [2; 3]. *)
  let a = arr [| 2; 4 |] and b3 = arr [| 4; 3; 3 |] and y = arr [| 2; 3 |] in
  refused ~msg:"offsets of rank 2" (rows (offs [| 1; 4 |])) y a b3;
  refused ~msg:"3 offsets for 3 groups" (rows (offs [| 3 |])) y a b3;
  refused ~msg:"a's axis in a pair" (rows ~a_axis:1 (offs [| 4 |])) y a b3;
  refused ~msg:"a's axis past its rank" (rows ~a_axis:2 (offs [| 4 |])) y a b3;
  refused ~msg:"a's axis -1" (rows ~a_axis:(-1) (offs [| 4 |])) y a b3;
  refused ~msg:"b's axis in a pair" (rows ~b_axis:0 (offs [| 4 |])) y a b3;
  refused ~msg:"b's axis past its rank" (rows ~b_axis:3 (offs [| 4 |])) y a b3;
  refused ~msg:"a result keeping b's cut axis" (rows (offs [| 4 |]))
    (arr [| 2; 3; 3 |]) a b3;
  (* [2; 4] by [4; 3] cut along the pair into 2 groups: [2; 2; 3]. *)
  let b2 = arr [| 4; 3 |] in
  refused ~msg:"no offset" (pair (offs [| 0 |])) (arr [| 0; 2; 3 |]) a b2;
  refused ~msg:"a pair past the pairs" (pair ~pair:1 (offs [| 3 |]))
    (arr [| 2; 2; 3 |]) a b2;
  refused ~msg:"pair -1" (pair ~pair:(-1) (offs [| 3 |])) (arr [| 2; 2; 3 |]) a b2;
  refused ~msg:"a result without the groups' axis" (pair (offs [| 3 |])) y a b2;
  refused ~msg:"a groups' axis of 3 for 2 groups" (pair (offs [| 3 |]))
    (arr [| 3; 2; 3 |]) a b2;
  (* [2; 2; 4] by [2; 4; 3] over a batch of 2: [2; 2; 2; 3]. *)
  let batch : _ iarray = [| (0, 0) |] and contracting : _ iarray = [| (2, 1) |] in
  refused ~msg:"offsets without the batch's extents" ~batch ~contracting
    (pair (offs [| 3 |])) (arr [| 2; 2; 2; 3 |]) (arr [| 2; 2; 4 |]) (arr [| 2; 4; 3 |]);
  refused ~msg:"offsets of another batch extent" ~batch ~contracting
    (pair (offs [| 3; 3 |])) (arr [| 2; 2; 2; 3 |]) (arr [| 2; 2; 4 |]) (arr [| 2; 4; 3 |])

(* A row of [Rows] in no group has no products: [init] as [acc] holds it,
   rounded into [dst], or [+0]. A [4; 2] by [1; 2] of zeros, cut by
   [1; 3]: rows 0 and 3 in no group, rows 1 and 2 adding zeros to init. *)
let test_rows_in_none (b : Support.backend) () =
  let module K = (val b.kernels) in
  let run (type s) (dt : (float, s) D.t) acc init =
    let arr s xs = A.expect dt (on b (A.Any (A.of_array dt s xs))) in
    let dst = arr [| 4 |] (Array.make 4 7.) in
    let offsets =
      A.expect D.Int64 (on b (A.Any (A.of_array D.Int64 [| 2 |] [| 1L; 3L |])))
    in
    (match
       K.contract ~order:Chain ~batch:[||] ~contracting:[| (1, 1) |]
         ~segments:(Some (Nx_kernel.Rows { offsets; a_axis = 0; b_axis = 0 }))
         ~acc ~dst ~init:(Option.map (arr [| 4 |]) init)
         (plain (arr [| 4; 2 |] (Array.make 8 0.)))
         (plain (arr [| 1; 2 |] [| 0.; 0. |]))
     with
    | A.Done -> ()
    | r -> failf "contract answered %a" Nx_array_support.pp_answer r);
    A.to_array (to64 (host (A.Any dst)))
  in
  let xs = [| 300.; -300.5; 7.9; 0.1 |] in
  let int8 x = Float.of_int (D.of_float D.Int8 (D.of_float D.Float32 x)) in
  equal ~msg:"int8 acc into float32" (array float_exact) (Array.map int8 xs)
    (run D.Float32 (D.Any D.Int8) (Some xs));
  let xs = [| 0.1; 1e40; 3.; -1e-50 |] in
  equal ~msg:"float32 acc into float64" (array float_exact)
    (Array.map (D.of_float D.Float32) xs)
    (run D.Float64 (D.Any D.Float32) (Some xs));
  equal ~msg:"no init" (array float_exact) (Array.make 4 0.)
    (run D.Float32 (D.Any D.Float32) None)

(* nx.cpu reads the offsets whole before any write, so they may be [dst]
   itself: a [3; 2] by [2; 2] cut by dst's [0; 1; 2], row 0 in group 0,
   row 1 in group 1, row 2 in none. *)
let test_offsets_in_dst (b : Support.backend) () =
  let module K = (val b.kernels) in
  let arr s xs = A.expect D.Int64 (on b (A.Any (A.of_array D.Int64 s xs))) in
  let dst = arr [| 3 |] [| 0L; 1L; 2L |] in
  (match
     K.contract ~order:Chain ~batch:[||] ~contracting:[| (1, 1) |]
       ~segments:(Some (Nx_kernel.Rows { offsets = dst; a_axis = 0; b_axis = 0 }))
       ~acc:(D.Any D.Int64) ~dst ~init:None
       (plain (arr [| 3; 2 |] [| 1L; 2L; 3L; 4L; 5L; 6L |]))
       (plain (arr [| 2; 2 |] [| 10L; 20L; 100L; 200L |]))
   with
  | A.Done -> ()
  | r -> failf "contract answered %a" Nx_array_support.pp_answer r);
  equal (array int64) [| 50L; 1100L; 0L |]
    (A.to_array (A.expect D.Int64 (host (A.Any dst))))

(* Guard pages *)

(* A [m × k] by [k × n] product whose operands and dst each lie against an
   inaccessible page on [side]: b a window of columns [c0, c0 + n) of a
   [k × cols] array, so that its rows lie off any alignment, and the
   window ends at the page where [c0 + n = cols]. Slivers of a matrix
   unit's half and whole tiles (16 and 32 float32 columns, 8 and 16
   float64) and rows past one tile are drawn often. With [heads] 2, a
   batch of 2 × 2 products whose operands are stored [2; m; 2; k] and
   [2; k; 2; cols] and read [2; 2; m; k] and [2; 2; k; cols], as
   attention's heads are, into a C-contiguous [2; 2; m; n]: the batch
   makes two runs, and the product loops over the outer one. *)
type guard_case = {
  f64 : bool;
  heads : int;
  side : Support.side;
  m : int;
  n : int;
  k : int;
  cols : int;
  c0 : int;
  seed : int;
}

let pp_guard ppf g =
  Format.fprintf ppf "%s %s heads %d m %d n %d k %d cols %d c0 %d seed %d"
    (if g.f64 then "float64" else "float32")
    (match g.side with Support.Before -> "before" | After -> "after")
    g.heads g.m g.n g.k g.cols g.c0 g.seed

let guard_cases =
  let open Gen in
  Gen.with_pp pp_guard
    (let* f64 = bool in
     let* heads = frequency [ (2, constant 1); (1, constant 2) ] in
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
     { f64; heads; side; m; n; k; cols = n + extra; c0; seed })

let guarded_product (type s) (b : Support.backend) (dt : (float, s) D.t) g =
  let (module K : Nx_kernel.S) = b.kernels in
  let st = Random.State.make [| g.seed |] in
  let values s =
    A.of_array dt s
      (Array.init (total s) (fun _ -> Random.State.float st 2. -. 1.))
  in
  let move m x = Option.get (A.move m x) in
  (* An [r × c] matrix, or with heads [2; r; heads; c] read
     [2; heads; r; c]. *)
  let matrix make r c =
    if g.heads = 1 then make [| r; c |]
    else move (M.Permute [| 0; 2; 1; 3 |]) (make [| 2; r; g.heads; c |])
  in
  let (batch : _ iarray), (contracting : _ iarray) =
    if g.heads = 1 then ([||], [| (1, 0) |])
    else ([| (0, 0); (1, 1) |], [| (3, 2) |])
  in
  let window x =
    let all c = { M.start = 0; count = c; step = 1 } in
    let cols = { M.start = g.c0; count = g.n; step = 1 } in
    let s = L.shape (A.layout x) in
    let r = Array.length s in
    move
      (M.Slice (Iarray.init r (fun i -> if i = r - 1 then cols else all s.(i))))
      x
  in
  let product ~dst a x =
    match
      K.contract ~order:Nx_kernel.Chain ~batch ~contracting ~segments:None
        ~acc:(D.Any dt) ~dst ~init:None (plain a)
        (plain (window x))
    with
    | A.Done -> ()
    | r -> failf "contract answered %a" Nx_array_support.pp_answer r
  in
  let guard x =
    let g', free = Support.guarded g.side dt (A.layout x) in
    A.blit ~src:x ~dst:g';
    (g', free)
  in
  let a = matrix values g.m g.k and x = matrix values g.k g.cols in
  let want =
    if g.heads = 1 then A.create Rig.host dt [| g.m; g.n |]
    else A.create Rig.host dt [| 2; g.heads; g.m; g.n |]
  in
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
  cover "two heads stored apart" (g.heads = 2);
  cover "two heads stored apart, b's window ending at the page"
    (g.heads = 2 && g.side = Support.After && g.c0 + g.n = g.cols);
  if g.f64 then guarded_product b D.Float64 g
  else guarded_product b D.Float32 g

(* nx.cpu under each table the host runs. *)
(* An [acc] Nx_kernel.accumulates refuses, a narrow float's or a boolean's, is
   refused with Wrong_dtype: the caller's error, which nx.cpu never declines. *)
let test_acc_refused (b : Support.backend) () =
  let module K = (val b.kernels) in
  let a = A.create Rig.host D.Float32 [| 2; 3 |] in
  let x = A.create Rig.host D.Float32 [| 3; 2 |] in
  let dst = A.create Rig.host D.Float32 [| 2; 2 |] in
  List.iter
    (fun acc ->
      equal ~msg:(Format.asprintf "%a" pp_dtype acc) Elements.answer
        A.Wrong_dtype
        (K.contract ~order:Lane ~batch:[||] ~contracting:[| (1, 0) |]
           ~segments:None ~acc ~dst ~init:None (plain a) (plain x)))
    D.[ Any Float16; Any Bfloat16; Any Bool ]

let cpu (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group ("nx.cpu " ^ b.name)
    [
      test "refuses an acc a contraction does not compute in" (fun () ->
          b.around (test_acc_refused b));
      prop ~examples "each output adds its products in the order given"
        computed (run (law_order b));
      prop ~examples:(List.concat_map regimes integer_accs)
        "an integer output is its products' sum modulo the accumulator"
        integer_cases
        (run (law_wraps ~covers:integer_covers b));
      prop ~examples:quad_examples
        "8-bit operands into 32 bits wrap as integers do" quad_cases
        (run (law_wraps ~covers:quad_covers b));
      prop "a result of another dtype rounds the sum in acc once" rounded
        (run (law_rounds_once b));
      prop "computes every contraction nx_cpu.mli lists"
        any_case (run (law_computes b));
      prop ~examples:scaled_examples
        "a scaled contraction has the bits of its operands scaled on the host"
        scaled_cases (run (law_scaled b));
      prop "declines scales in an integer or complex accumulator" unscalable
        (run (law_unscalable b));
      prop ~examples:mx_examples
        "a block-scaled float4 operand has the bits of its elements scaled \
         on the host"
        mx_draws (run (law_mx b));
      prop ~count:0 ~examples:mx_layouts
        "block-scaled float4 weights through other layouts have their \
         elements' bits scaled on the host"
        (Gen.with_pp pp_scaled (Gen.constant (List.hd mx_layouts)))
        (run (law_mx_layouts b));
      test "refuses a scale that does not fit its operand" (fun () ->
          b.around (test_scale_misfits b));
      test "a scaled contraction over an empty extent has no products"
        (fun () -> b.around (test_scaled_empty b));
      prop ~examples:seg_examples
        "each group of a segmented contraction has the bits of its \
         unsegmented contraction"
        seg_draws (run (law_segments b));
      test "refuses segments that do not fit" (fun () ->
          b.around (test_segment_misfits b));
      test "a row in no group is init as acc holds it, or +0" (fun () ->
          b.around (test_rows_in_none b));
      test "reads the offsets before writing dst, which may be them" (fun () ->
          b.around (test_offsets_in_dst b));
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
  cases
    "nx.cpu chooses chain order at rows and columns above 4, 64 outputs and \
     fewer than 128 products per output"
    ~name:(fun (m, n, k, _) -> Printf.sprintf "%d by %d over %d" m n k)
    Nx_kernel.
      [
        (0, 1000, 4096, Lane);
        (1000, 0, 4096, Lane);
        (1, 1, 4096, Lane);
        (4, 1000, 4096, Lane);
        (5, 1000, 4096, Chain);
        (1000, 4, 4096, Lane);
        (1000, 5, 4096, Chain);
        (5, 5, 4096, Lane);
        (5, 12, 4096, Lane);
        (5, 13, 4096, Chain);
        (12, 5, 4096, Lane);
        (13, 5, 4096, Chain);
        (7, 9, 4096, Lane);
        (9, 7, 4096, Lane);
        (8, 8, 4096, Chain);
        (16, 4, 4096, Lane);
        (8, 8, 0, Chain);
        (8, 8, 8191, Chain);
        (8, 8, 8192, Lane);
        (16, 16, 32767, Chain);
        (16, 16, 32768, Lane);
        (64, 64, 524287, Chain);
        (64, 64, 524288, Lane);
        (8, 8, max_int, Lane);
        (max_int, max_int, 4096, Chain);
        (max_int, max_int, max_int, Chain);
        (max_int, 5, max_int, Chain);
        (5, max_int, max_int, Chain);
        (max_int, 4, 4096, Lane);
        (4, max_int, 4096, Lane);
        (max_int, 0, 4096, Lane);
      ]
    (fun (rows, columns, depth, want) ->
      equal order want (Nx_cpu.contract_order ~rows ~columns ~depth))

let () =
  exit
    (Windtrap.run "nx_kernel.contract"
       ((orders :: List.map laws Support.backends) @ List.map cpu Support.cpus))
