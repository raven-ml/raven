(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Segments and ranges: each fold against a fold of the rows on the host, in
   row order, and map_segments against the per-position map its doc defines.
   Sums are of small integers held in float64, exact in any order, or of
   integers that wrap, so the folds compare bit for bit. *)

open Windtrap
open Nx_support
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout

let invalid ~by f = raises_match (Exn.invalid_arg ~sub:(by ^ ": ")) f
let m = Nx_support.memory

module S2 = (val Nx.devices [ m 0; m 1 ])

let numel s = Array.fold_left ( * ) 1 s

(* A host value of [dt] and shape [s] over drawn bytes: NaNs, signed zeros,
   infinities and extremes among its elements. *)
let drawn (type v s) (dt : (v, s) D.t) s : (v, s, Nx.host) Nx.t Gen.t =
  let open Gen in
  let+ data = string_of ~size:(constant (D.bytes dt (numel s))) char in
  let data =
    match dt with
    | D.Bool -> String.map (fun c -> Char.chr (Char.code c land 1)) data
    | _ -> data
  in
  Nx.Repr.of_array Nx.Host.v
    (A.v dt (L.contiguous (Iarray.of_array s))
       (Rig.Buffer.of_string (if data = "" then "\000" else data)))

(* Small integers in float64: their sums are exact in any order. *)
let small s =
  Gen.map
    (fun vs -> Nx.create Nx.float64 s (Array.map Float.of_int vs))
    (Gen.array ~size:(Gen.constant (numel s)) (Gen.int_range (-4) 4))

let int64s n lo hi =
  Gen.map
    (fun vs -> Nx.create Nx.int64 [| n |] (Array.map Int64.of_int vs))
    (Gen.array ~size:(Gen.constant n) (Gen.int_range lo hi))

let pp_combine ppf c =
  Format.pp_print_string ppf
    (match c with Nx.Set -> "Set" | Add -> "Add" | Max -> "Max" | Min -> "Min")

(* The fold a combine states *)

(* [a] above [b] in the dtype's order: a NaN above all, [-0.] below [+0.]. *)
let above (type v s) (dt : (v, s) D.t) (a : v) (b : v) =
  match D.kind dt with
  | D.Float ->
      (Float.is_nan a && not (Float.is_nan b))
      || (not (Float.is_nan b))
         && (a > b || (a = b && Float.sign_bit b && not (Float.sign_bit a)))
  | D.Signed | D.Unsigned | D.Boolean -> a > b
  | D.Complex -> assert false

let is_nan (type v s) (dt : (v, s) D.t) (v : v) =
  match D.kind dt with D.Float -> Float.is_nan v | _ -> false

let identity (type v s) (c : Nx.combine) (dt : (v, s) D.t) : v =
  match c with
  | Set | Add -> D.zero dt
  | Max -> D.min_value dt
  | Min -> D.max_value dt

(* Rows [vs] folded by [c] from the identity, in order: a NaN, else the
   first extreme; the last row; a sum by [add]. *)
let fold (type v s) (c : Nx.combine) (dt : (v, s) D.t) ~add (vs : v list) : v =
  match c with
  | Set -> List.fold_left (fun _ v -> v) (D.zero dt) vs
  | Add -> List.fold_left add (D.zero dt) vs
  | Max | Min ->
      let wins v acc =
        if is_nan dt acc then false
        else if is_nan dt v then true
        else if c = Max then above dt v acc
        else above dt acc v
      in
      List.fold_left (fun acc v -> if wins v acc then v else acc) (identity c dt) vs

(* A sum from [+0]: [-0.] terms give [+0.]. *)
let plus_zero (type v s) (c : Nx.combine) (dt : (v, s) D.t) (v : v) : v =
  match c with
  | Add -> ( match D.kind dt with D.Float -> v +. 0. | _ -> v)
  | Set | Max | Min -> v

(* Element [e] of each row of [x], for rows [rows]: [x]'s elements grouped. *)
let row_elements x =
  let w = numel (Array.sub (Nx.shape x) 1 (Nx.ndim x - 1)) in
  let vs = Nx.to_array x in
  (w, fun i e -> vs.((i * w) + e))

(* Values to fold *)

type rows = Rows : ('v, 's) D.t * Nx.combine * ('v -> 'v -> 'v) * ('v, 's, Nx.host) Nx.t -> rows

let pp_rows ppf (Rows (dt, c, _, x)) =
  Format.fprintf ppf "%a %a %a" pp_combine c D.pp dt pp_ints (Nx.shape x)

let wrap8 a b = ((a + b + 128) land 255) - 128

(* Rows of every kind the folds take: extremes and the last row over drawn
   bytes, sums over small integers and over int8, which wraps. *)
let rows_of ?(kinds = [ 0; 1; 2; 3; 4; 5; 6 ]) s =
  let open Gen in
  let* kind = of_list kinds in
  match kind with
  | 0 -> let+ x = small s in Rows (D.Float64, Add, ( +. ), x)
  | 1 -> let+ x = drawn D.Int8 s in Rows (D.Int8, Add, wrap8, x)
  | 2 -> let+ x = drawn D.Float32 s in Rows (D.Float32, Max, ( +. ), x)
  | 3 -> let+ x = drawn D.Float16 s in Rows (D.Float16, Min, ( +. ), x)
  | 4 -> let+ x = drawn D.Uint8 s in Rows (D.Uint8, Max, ( + ), x)
  | 5 -> let+ x = drawn D.Bool s in Rows (D.Bool, Min, ( || ), x)
  | _ -> let+ x = drawn D.Float32 s in Rows (D.Float32, Set, ( +. ), x)

let rows n =
  Gen.(
    let* row = array ~size:(int_range 0 1) (int_range 0 2) in
    rows_of (Array.append [| n |] row))

(* reduce_segments *)

let segmented =
  Gen.with_pp
    (fun ppf (r, k, _) -> Format.fprintf ppf "%a, %d segments" pp_rows r k)
    (let open Gen in
     let* n = int_range 0 6 in
     let* k = int_range 0 4 in
     let* r = rows n in
     let+ ids = int64s n (-1) (k + 1) in
     (r, k, ids))

let law_segments (Rows (dt, c, add, x), k, ids) =
  let id = Array.map Int64.to_int (Nx.to_array ids) in
  cover "an id out of range" (Array.exists (fun i -> i < 0 || i >= k) id);
  cover "a segment of no row" (List.exists (fun s -> not (Array.mem s id)) (List.init k Fun.id));
  let w, at = row_elements x in
  let expected =
    Array.init (k * w) (fun j ->
        let s = j / w and e = j mod w in
        let mine = List.filter (fun i -> id.(i) = s) (List.init (Array.length id) Fun.id) in
        let v = fold c dt ~add (List.map (fun i -> at i e) mine) in
        plus_zero c dt v)
  in
  let y = Nx.reduce_segments c ~segments:k ids x in
  equal (array int) (Array.append [| k |] (Array.sub (Nx.shape x) 1 (Nx.ndim x - 1))) (Nx.shape y);
  equal_elements dt expected (Nx.to_array y)

let test_segment_doc () =
  let ids = Nx.create Nx.int64 [| 5 |] [| 0L; 2L; 0L; -1L; 2L |] in
  let x = Nx.create Nx.float64 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  equal (array float_exact) [| 4.; 0.; 7. |]
    (Nx.to_array (Nx.reduce_segments Add ~segments:3 ids x));
  equal (array float_exact) [| 3.; Float.neg_infinity; 5. |]
    (Nx.to_array (Nx.reduce_segments Max ~segments:3 ids x))

let test_segment_narrow () =
  let ids = Nx.zeros Nx.int64 [| 3 |] in
  let x = Nx.create Nx.float16 [| 3 |] [| 2048.; 1.; 1. |] in
  equal (array float_exact) [| 2050. |]
    (Nx.to_array (Nx.reduce_segments Add ~segments:1 ids x))

let test_segment_identities () =
  let ids = Nx.zeros Nx.int64 [| 0 |] in
  let x = Nx.zeros Nx.float8_e4m3fn [| 0 |] in
  equal (array float_exact) [| -448. |]
    (Nx.to_array (Nx.reduce_segments Max ~segments:1 ids x));
  let z = Nx.zeros Nx.complex64 [| 0 |] in
  equal_elements D.Complex64
    [| { Complex.re = Float.infinity; im = Float.infinity } |]
    (Nx.to_array (Nx.reduce_segments Min ~segments:1 ids z))

let test_segment_errors () =
  let by = "Nx.reduce_segments" in
  let ids = Nx.zeros Nx.int64 [| 2 |] in
  invalid ~by (fun () -> Nx.reduce_segments Add ~segments:(-1) ids (Nx.zeros Nx.float32 [| 2 |]));
  invalid ~by (fun () -> Nx.reduce_segments Add ~segments:1 ids (Nx.zeros Nx.float32 [||]));
  invalid ~by (fun () -> Nx.reduce_segments Add ~segments:1 ids (Nx.zeros Nx.float32 [| 3 |]));
  invalid ~by (fun () ->
      Nx.reduce_segments Add ~segments:1 (Nx.zeros Nx.int64 [| 2; 1 |]) (Nx.zeros Nx.float32 [| 2 |]));
  invalid ~by (fun () -> Nx.reduce_segments Add ~segments:1 ids (Nx.zeros Nx.bool [| 2 |]))

(* Segments over two devices: [ids] whole on each or split along its
   positions, [x] whole on each or split along its rows or its columns.
   Each case the law covers is drawn on its own a sixth of the time, so
   every run reaches it: [Add] beside [ids] split with [x] whole, with [x]'s
   columns cut, and with [x]'s rows cut in uneven windows, any other combine
   with its rows cut, and any combine with the columns alone cut. *)
type scenario = Ids_x_whole | Ids_x_cols | Add_x_rows | Other_rows | Cols | Any

let add = [ 0; 1 ] and others = [ 2; 3; 4; 5; 6 ]

let cut_segments =
  Gen.with_pp
    (fun ppf (r, k, _, ids_rows, x_cut) ->
      Format.fprintf ppf "%a, %d segments, ids %s, x %s" pp_rows r k
        (if ids_rows then "split" else "whole")
        (match x_cut with
        | None -> "whole"
        | Some a -> Printf.sprintf "split along axis %d" a))
    (let open Gen in
     let* scenario =
       of_list [ Ids_x_whole; Ids_x_cols; Add_x_rows; Other_rows; Cols; Any ]
     in
     let* kinds, n, ids_rows, x_cut =
       let n = of_list [ 0; 1; 2; 3; 5; 8 ] in
       let x_cut = of_list [ None; Some 0; Some 1 ] in
       match scenario with
       | Ids_x_whole -> map (fun n -> (add, n, true, None)) n
       | Ids_x_cols -> map (fun n -> (add, n, true, Some 1)) n
       | Add_x_rows ->
           let+ n = of_list [ 1; 3; 5 ] and+ ids_rows = bool in
           (add, n, ids_rows, Some 0)
       | Other_rows ->
           let+ n = n
           and+ ids_rows, x_cut =
             of_list [ (true, None); (true, Some 1); (false, Some 0); (true, Some 0) ]
           in
           (others, n, ids_rows, x_cut)
       | Cols -> map (fun n -> (add @ others, n, false, Some 1)) n
       | Any ->
           let+ n = n and+ ids_rows = bool and+ x_cut = x_cut in
           (add @ others, n, ids_rows, x_cut)
     in
     let* cols = int_range 1 3 in
     let* k = int_range 0 4 in
     let* r = rows_of ~kinds [| n; cols |] in
     let+ ids = int64s n (-1) (k + 1) in
     (r, k, ids, ids_rows, x_cut))

(* Each combine is a scatter's, which folds rows cut along the positions
   on each device and closes the devices' partial results by partials: it
   gives the host's bits, its rows whole on each device and its columns cut
   as [x]'s. [ids] split move where the rows are read. *)
let law_cut_segments (Rows (dt, c, _, x), k, ids, ids_rows, x_cut) =
  let px = match x_cut with None -> S2.on | Some axis -> S2.split ~axis in
  let pi = if ids_rows then S2.split ~axis:0 else S2.on in
  let n = Nx.dim 0 x in
  let rows_cut = ids_rows || x_cut = Some 0 in
  cover "Add with its rows cut" (c = Nx.Add && rows_cut && n > 0);
  cover "Add with x's rows cut in uneven windows"
    (c = Nx.Add && x_cut = Some 0 && n mod 2 = 1);
  cover "Add with ids split beside x whole"
    (c = Nx.Add && ids_rows && x_cut = None);
  cover "Add with ids split beside x's columns cut"
    (c = Nx.Add && ids_rows && x_cut = Some 1);
  cover "Max, Min or Set with x's rows cut"
    (c <> Nx.Add && x_cut = Some 0 && n > 1 && k > 0);
  cover "a cut of the columns alone" ((not ids_rows) && x_cut = Some 1);
  let host = Nx.to_array (Nx.reduce_segments c ~segments:k ids x) in
  let whole = Nx.Placement.(with_leading_axis (without_leading_axis px)) in
  let y = Nx.reduce_segments c ~segments:k (Nx.place pi ids) (Nx.place px x) in
  equal_elements dt host (Nx.to_array y);
  equal ~msg:"where it lies" (placement ()) whole (Nx.placement y)

let segments =
  group "reduce_segments"
    [
      prop "each segment folds its rows in order" segmented law_segments;
      prop
        "over devices, each combine folds rows split along the positions as \
         one device does"
        cut_segments law_cut_segments;
      test "the doc's sums and maxima" test_segment_doc;
      test "narrow floats sum at float32" test_segment_narrow;
      test "a segment of no row holds the identity" test_segment_identities;
      test "misuse raises" test_segment_errors;
    ]

(* reduce_ranges *)

let ranged =
  Gen.with_pp
    (fun ppf (r, _, _) -> pp_rows ppf r)
    (let open Gen in
     let* n = int_range 0 9 in
     let* k = int_range 0 5 in
     let* r = rows n in
     let* lo = int64s k (-2) (n + 1) in
     let+ hi = int64s k (-1) (n + 2) in
     (r, lo, hi))

let clip n v = max 0 (min n v)

let law_ranges (Rows (dt, c, add, x), lo, hi) =
  let n = Nx.dim 0 x in
  let lo = Nx.to_array lo and hi = Nx.to_array hi in
  let bounds = Array.map2 (fun l h -> (clip n (Int64.to_int l), clip n (Int64.to_int h))) lo hi in
  cover "a range of several rows" (Array.exists (fun (l, h) -> h - l >= 3) bounds);
  cover "an empty range" (Array.exists (fun (l, h) -> l >= h) bounds);
  let w, at = row_elements x in
  let expected =
    Array.init (Array.length bounds * w) (fun j ->
        let l, h = bounds.(j / w) and e = j mod w in
        let v = fold c dt ~add (List.init (max 0 (h - l)) (fun i -> at (l + i) e)) in
        plus_zero c dt v)
  in
  let y =
    Nx.reduce_ranges c ~lo:(Nx.create Nx.int64 [| Array.length lo |] lo)
      ~hi:(Nx.create Nx.int64 [| Array.length hi |] hi) x
  in
  equal (array int) (Array.append [| Array.length lo |] (Array.sub (Nx.shape x) 1 (Nx.ndim x - 1))) (Nx.shape y);
  equal_elements dt expected (Nx.to_array y)

(* Float32 terms whose sums depend on their association. *)
let term =
  Gen.of_list ~pp:Format.pp_print_float [ 1e7; -1e7; 1.; 0.1; 3.3; -2.5e-3; -0. ]

let local =
  Gen.with_pp
    (fun ppf (x, _, _, _) -> pp_ints ppf (Nx.shape x))
    (let open Gen in
     let* n = int_range 1 12 in
     let terms = map (Nx.create Nx.float32 [| n |]) (array ~size:(constant n) term) in
     let* x = terms in
     let* y = terms in
     let* l = int_range 0 (n - 1) in
     let+ h = int_range (l + 1) n in
     (x, y, l, h))

(* A range's sum reads its rows alone: the other rows changed, its bits
   stay. *)
let law_local (x, y, l, h) =
  let n = Nx.dim 0 x in
  let xs = Nx.to_array x and ys = Nx.to_array y in
  let z = Nx.create Nx.float32 [| n |] (Array.init n (fun i -> if i >= l && i < h then xs.(i) else ys.(i))) in
  let b v = Nx.create Nx.int64 [| 1 |] [| Int64.of_int v |] in
  let sum t = Nx.to_array (Nx.reduce_ranges Add ~lo:(b l) ~hi:(b h) t) in
  equal_elements D.Float32 (sum x) (sum z)

(* Every range with bounds from two before the rows to two past them, over
   rows of one sign: a row of zeros, or any value from outside the rows,
   that reached a range would show as its maximum of negative rows or its
   minimum of positive ones. *)
let test_range_rows_only () =
  List.iter
    (fun n ->
      List.iter
        (fun (c, sign) ->
          let xs = Array.init n (fun i -> sign *. Float.of_int (i + 1)) in
          let x = Nx.create Nx.float64 [| n |] xs in
          let bounds =
            List.concat_map
              (fun l -> List.init (n + 5) (fun h -> (l - 2, h - 2)))
              (List.init (n + 5) Fun.id)
          in
          let b f =
            Nx.create Nx.int64 [| List.length bounds |]
              (Array.of_list (List.map (fun p -> Int64.of_int (f p)) bounds))
          in
          let expected =
            List.map
              (fun (l, h) ->
                let l = clip n l and h = clip n h in
                fold c D.Float64 ~add:( +. )
                  (List.init (max 0 (h - l)) (fun i -> xs.(l + i))))
              bounds
          in
          equal
            ~msg:(Format.asprintf "%a of %d rows" pp_combine c n)
            (array float_exact) (Array.of_list expected)
            (Nx.to_array (Nx.reduce_ranges c ~lo:(b fst) ~hi:(b snd) x)))
        [ (Nx.Max, -1.); (Min, 1.) ])
    [ 0; 1; 2; 3; 4; 5; 7; 8; 9; 11; 16; 17 ]

let test_range_doc () =
  let x = Nx.create Nx.float64 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let lo = Nx.create Nx.int64 [| 4 |] [| -2L; 0L; 3L; 4L |]
  and hi = Nx.create Nx.int64 [| 4 |] [| 1L; 3L; 3L; 9L |] in
  equal (array float_exact) [| 1.; 6.; 0.; 5. |] (Nx.to_array (Nx.reduce_ranges Add ~lo ~hi x));
  equal (array float_exact) [| 1.; 3.; Float.neg_infinity; 5. |]
    (Nx.to_array (Nx.reduce_ranges Max ~lo ~hi x))

let test_range_narrow () =
  let x = Nx.create Nx.float16 [| 3 |] [| 2048.; 1.; 1. |] in
  let b v = Nx.create Nx.int64 [| 1 |] [| v |] in
  equal (array float_exact) [| 2050. |] (Nx.to_array (Nx.reduce_ranges Add ~lo:(b 0L) ~hi:(b 3L) x))

(* Complex numbers order by real part, then imaginary part. *)
let test_range_complex () =
  let c re im = { Complex.re; im } in
  let x = Nx.create Nx.complex64 [| 3 |] [| c 1. 5.; c 2. (-1.); c 2. 3. |] in
  let b v = Nx.create Nx.int64 [| 1 |] [| v |] in
  let fold f = Nx.to_array (f ~lo:(b 0L) ~hi:(b 3L) x) in
  equal_elements D.Complex64 [| c 2. 3. |] (fold (Nx.reduce_ranges Max));
  equal_elements D.Complex64 [| c 1. 5. |] (fold (Nx.reduce_ranges Min))

let test_range_errors () =
  let by = "Nx.reduce_ranges" in
  let b = Nx.zeros Nx.int64 [| 2 |] in
  invalid ~by (fun () -> Nx.reduce_ranges Add ~lo:b ~hi:b (Nx.zeros Nx.float32 [||]));
  invalid ~by (fun () -> Nx.reduce_ranges Add ~lo:b ~hi:(Nx.zeros Nx.int64 [| 3 |]) (Nx.zeros Nx.float32 [| 2 |]));
  invalid ~by (fun () ->
      let b = Nx.zeros Nx.int64 [| 2; 1 |] in
      Nx.reduce_ranges Add ~lo:b ~hi:b (Nx.zeros Nx.float32 [| 2 |]));
  invalid ~by (fun () -> Nx.reduce_ranges Add ~lo:b ~hi:b (Nx.zeros Nx.bool [| 2 |]))

(* Signalling NaNs. A Max or Min range is a NaN where one of its rows is,
   as {!Nx.max} gives it, a range of one row too. Drawn at every float
   format with a NaN, from elements whose NaNs {!Nx_support.signalled} makes
   signalling. *)

(* Floats that tie often: small integers, [-0], infinities, and NaNs whose
   payload keeps them NaN once signalling in every float format, the two
   bits below the quiet bit. *)
let nan_payloads =
  [| Int64.float_of_bits 0x7FFC000000000000L; Int64.float_of_bits 0xFFFE000000000000L |]

let draw_float rs =
  match Random.State.int rs 8 with
  | 0 | 1 | 2 -> Float.of_int (Random.State.int rs 5 - 2)
  | 3 -> -0.
  | 4 -> if Random.State.bool rs then Float.infinity else Float.neg_infinity
  | _ -> nan_payloads.(Random.State.int rs 2)

(* The order of numbers with [-0] below [+0]. *)
let order x y =
  let c = Float.compare x y in
  if c <> 0 then c else Bool.compare (Float.sign_bit y) (Float.sign_bit x)

type nan_dtype =
  | Nd :
      ('v, 's) D.t * (Random.State.t -> 'v) * ('v -> bool) * ('v -> 'v -> int)
      -> nan_dtype

let real dt = Nd (dt, draw_float, Float.is_nan, order)

let complex dt =
  Nd
    ( dt,
      (fun rs -> { Complex.re = draw_float rs; im = draw_float rs }),
      (fun z -> Float.is_nan z.Complex.re || Float.is_nan z.im),
      fun a b ->
        let c = order a.Complex.re b.Complex.re in
        if c <> 0 then c else order a.im b.im )

type nan_ranges =
  | Nan_ranges : {
      dt : ('v, 's) D.t;
      nan : 'v -> bool;
      order : 'v -> 'v -> int;
      s : int array;
      bounds : (int * int) array;
      max : bool;
      x : 'v array;
    }
      -> nan_ranges

(* Rows of one or two elements, and ranges of at least one row. *)
let nan_ranges =
  Gen.(
    let* (Nd (dt, draw, nan, order)) =
      of_list
        [
          real D.Float16;
          real D.Bfloat16;
          real D.Float8_e4m3fn;
          real D.Float8_e5m2;
          real D.Float32;
          real D.Float64;
          complex D.Complex64;
          complex D.Complex128;
        ]
    in
    let* n = int_range 1 6 in
    let* w = int_range 1 2 in
    let* k = int_range 1 5 in
    let* max = bool in
    let+ seed = int in
    let rs = Random.State.make [| seed |] in
    let bounds =
      Array.init k (fun _ ->
          let l = Random.State.int rs n in
          (l, l + 1 + Random.State.int rs (n - l)))
    in
    let x = Array.init (n * w) (fun _ -> draw rs) in
    Nan_ranges { dt; nan; order; s = [| n; w |]; bounds; max; x })
  |> Gen.with_pp (fun ppf (Nan_ranges c) ->
      Format.fprintf ppf "%s of %s %a over %s"
        (if c.max then "Max" else "Min")
        (D.name c.dt) pp_ints c.s
        (String.concat "; "
           (Array.to_list
              (Array.map (fun (l, h) -> Printf.sprintf "[%d, %d)" l h) c.bounds))))

let law_nan_ranges (Nan_ranges c) =
  let x = Nx_support.signalled (A.of_array c.dt (Iarray.of_array c.s) c.x) in
  let bytes a = A.to_array (Option.get (A.bitcast D.Uint8 (A.copy a))) in
  let w = D.bits c.dt / 8 and row = c.s.(1) in
  let elt b k = Array.sub b (k * w) w in
  let xb = bytes x and xq = bytes (Nx_support.quieted x) in
  let xs = A.to_array x in
  let signals k = elt xb k <> elt xq k in
  (* The position of each range's extreme of element [e]: a NaN, else the
     first greatest or least. *)
  let pick (l, h) e =
    let best = ref (l * row + e) in
    for i = l + 1 to h - 1 do
      let a = xs.(!best) and b = xs.(i * row + e) in
      let keep =
        if c.nan a then true
        else if c.nan b then false
        else
          let o = c.order a b in
          if c.max then o >= 0 else o <= 0
      in
      if not keep then best := i * row + e
    done;
    !best
  in
  let picks =
    Array.concat
      (Array.to_list
         (Array.map (fun b -> Array.init row (fun e -> (b, pick b e))) c.bounds))
  in
  cover "one row's signalling NaN"
    (Array.exists (fun ((l, h), k) -> h - l = 1 && signals k) picks);
  cover "a signalling NaN first of several rows"
    (Array.exists (fun ((l, h), k) -> h - l > 1 && signals k) picks);
  (* Each result's bytes, or [None] for a NaN. *)
  let expected =
    Array.map
      (fun (_, k) -> if c.nan xs.(k) then None else Some (elt xb k))
      picks
  in
  let bound f =
    Nx.create Nx.int64
      [| Array.length c.bounds |]
      (Array.map (fun b -> Int64.of_int (f b)) c.bounds)
  in
  let y =
    Nx.reduce_ranges
      (if c.max then Max else Min)
      ~lo:(bound fst) ~hi:(bound snd)
      (Nx.Repr.of_array Nx.Host.v x)
  in
  let y = Option.get (Nx.Repr.array y) in
  let yb = bytes y in
  let actual =
    Array.mapi
      (fun i v -> if c.nan v then None else Some (elt yb i))
      (A.to_array y)
  in
  equal (array (option (array int))) expected actual

(* A donated operand: the ranges compute as from the undonated one, and the
   donor lives, since reduce_ranges reads it in many operations. *)
let test_range_donated () =
  let lo = Nx.create Nx.int64 [| 2 |] [| 0L; 1L |] in
  let hi = Nx.create Nx.int64 [| 2 |] [| 3L; 4L |] in
  let xs = [| 1.; 5.; -2.; 4. |] in
  List.iter
    (fun c ->
      let fresh () =
        let x = Nx.create Nx.float32 [| 4 |] xs in
        Nx.add x (Nx.zeros_like x)
      in
      let plain = Nx.to_array (Nx.reduce_ranges c ~lo ~hi (fresh ())) in
      let x = fresh () in
      let donated = Nx.to_array (Nx.reduce_ranges c ~lo ~hi (Nx.donate x)) in
      equal (array float_exact) plain donated;
      equal ~msg:"the donor lives" (array float_exact) xs (Nx.to_array x))
    [ Nx.Add; Max; Min; Set ]

(* Ranges over devices. [x], of three or four columns, lies whole on each
   of four devices, cut along its rows or its columns over two of them or
   four, or over a 2×2 mesh along both. *)

module S4 = (val Nx.devices [ m 0; m 1; m 2; m 3 ])

let mesh = Nx.Mesh.v S4.v [ ("a", 2); ("b", 2) ]

type cut = Whole | Cut of { axis : int; tiles : int } | Both

let at = function
  | Whole -> S4.on
  | Cut { axis; tiles = 4 } -> S4.split ~axis
  | Cut { axis; _ } -> Nx.Placement.mesh mesh [ (axis, [ "b" ]) ]
  | Both -> Nx.Placement.mesh mesh [ (0, [ "a" ]); (1, [ "b" ]) ]

let cuts_rows = function
  | Cut { axis = 0; _ } | Both -> true
  | Whole | Cut _ -> false

let tiles = function Cut { tiles; _ } -> tiles | Both -> 2 | Whole -> 1

let pp_cut ppf = function
  | Whole -> Format.pp_print_string ppf "whole"
  | Cut { axis; tiles } -> Format.fprintf ppf "axis %d in %d" axis tiles
  | Both -> Format.pp_print_string ppf "both axes over a mesh"

let cut_ranges =
  Gen.with_pp
    (fun ppf (c, x, _, _, cut) ->
      Format.fprintf ppf "%a %a at %a" pp_combine c pp_ints (Nx.shape x)
        pp_cut cut)
    (let open Gen in
     let* c = of_list [ Nx.Add; Max; Min; Set ] in
     let* n = of_list [ 0; 1; 2; 3; 5; 8; 16 ] in
     let* k = int_range 0 4 in
     let* cols = of_list [ 3; 4 ] in
     let* x = drawn D.Float32 [| n; cols |] in
     let* lo = int64s k (-1) (n + 1) in
     let* hi = int64s k (-1) (n + 1) in
     let+ cut =
       of_list
         [
           Whole;
           Cut { axis = 0; tiles = 2 };
           Cut { axis = 0; tiles = 4 };
           Cut { axis = 1; tiles = 2 };
           Cut { axis = 1; tiles = 4 };
           Both;
         ]
     in
     (c, x, lo, hi, cut))

(* [Add], [Max] and [Min] read [x]'s rows whole: cut along them, the call
   raises naming the placement with the rows whole and the other cuts kept,
   and placed there it gives the host's bits, lying there. [Set] gives the
   host's bits at any placement, lying as [x] does where its rows are
   whole. *)
let law_cut_ranges (c, x, lo, hi, cut) =
  let n = Nx.dim 0 x and p = at cut in
  let refused = cuts_rows cut && c <> Nx.Set in
  cover "Add, Max or Min with its rows cut" refused;
  cover "Add, Max or Min with fewer rows than tiles"
    (refused && n < tiles cut);
  cover "Add, Max or Min with its rows and columns cut" (refused && cut = Both);
  cover "Set with its rows cut" (cuts_rows cut && c = Nx.Set);
  cover "a cut of the columns alone"
    (match cut with
    | Cut { axis = 1; _ } -> true
    | Whole | Cut _ | Both -> false);
  cover "uneven windows of the columns"
    (match cut with
    | Cut { axis = 1; tiles } -> Nx.dim 1 x mod tiles <> 0
    | Both -> Nx.dim 1 x mod 2 <> 0
    | Whole | Cut _ -> false);
  let host = Nx.to_array (Nx.reduce_ranges c ~lo ~hi x) in
  let on v = Nx.place S4.on v in
  let run p = Nx.reduce_ranges c ~lo:(on lo) ~hi:(on hi) (Nx.place p x) in
  let computes p =
    let y = run p in
    equal_elements D.Float32 host (Nx.to_array y);
    y
  in
  if refused then (
    let whole =
      Nx.Placement.with_leading_axis (Nx.Placement.without_leading_axis p)
    in
    let sub =
      Format.asprintf
        "Nx.reduce_ranges: x lies at %a and is read at %a; Nx.place ("
        Nx.Placement.pp p Nx.Placement.pp whole
    in
    raises_match (Exn.invalid_arg ~sub) (fun () -> run p);
    let y = computes whole in
    equal ~msg:"where it lies" (placement ()) whole (Nx.placement y))
  else
    let y = computes p in
    if not (cuts_rows cut) then
      equal ~msg:"where it lies" (placement ()) p (Nx.placement y)

(* Eight rows in two tiles, which every level's chunks divide, raise as any
   other cut of the rows does. *)
let test_range_rows_cut () =
  let x = Nx.zeros_at (S2.split ~axis:0) Nx.float32 [| 8; 3 |] in
  let b = Nx.place S2.on (Nx.create Nx.int64 [| 2 |] [| 0L; 3L |]) in
  List.iter
    (fun c ->
      raises
        (Invalid_argument
           "Nx.reduce_ranges: x lies at split ~axis:0 [m0; m1] and is read \
            at on [m0; m1]; Nx.place (Placement.on s) x")
        (fun () -> Nx.reduce_ranges c ~lo:b ~hi:b x))
    [ Nx.Add; Max; Min ]

(* The bounds cut the result's first axis, and [x]'s other cuts its other
   axes: split over the devices that cut [x]'s columns, a bound raises
   naming a bound or [x] and the [place] that reads it beside the other,
   and over other devices of a mesh the call gives the host's bits, lying
   at both cuts. *)
let test_range_bounds_cut () =
  let x = Nx.create Nx.float64 [| 5; 4 |] (Array.init 20 Float.of_int) in
  let lo = Nx.create Nx.int64 [| 4 |] [| 0L; 1L; -1L; 3L |] in
  let hi = Nx.create Nx.int64 [| 4 |] [| 9L; 3L; 1L; 3L |] in
  let refused f =
    let named = function
      | Invalid_argument m ->
          List.exists
            (fun name ->
              let prefix = "Nx.reduce_ranges: " ^ name ^ " lies at " in
              String.starts_with ~prefix m
              && String.ends_with ~suffix:(") " ^ name) m)
            [ "lo"; "hi"; "x" ]
      | _ -> false
    in
    raises_match named f
  in
  List.iter
    (fun c ->
      let host = Nx.to_array (Nx.reduce_ranges c ~lo ~hi x) in
      let run plo phi px =
        let y =
          Nx.reduce_ranges c ~lo:(Nx.place plo lo) ~hi:(Nx.place phi hi)
            (Nx.place px x)
        in
        equal_elements D.Float64 host (Nx.to_array y);
        y
      in
      let rows = S2.split ~axis:0 and cols = S2.split ~axis:1 in
      refused (fun () -> run rows rows cols);
      refused (fun () -> run S2.on rows cols);
      let y = run S2.on S2.on cols in
      equal ~msg:"where it lies" (placement ()) cols (Nx.placement y);
      let a = Nx.Placement.mesh mesh [ (0, [ "a" ]) ]
      and b = Nx.Placement.mesh mesh [ (0, [ "b" ]) ] in
      let x_cols = Nx.Placement.mesh mesh [ (1, [ "b" ]) ] in
      refused (fun () -> run b b x_cols);
      let y = run a a x_cols in
      equal ~msg:"where it lies" (placement ())
        (Nx.Placement.mesh mesh [ (0, [ "a" ]); (1, [ "b" ]) ])
        (Nx.placement y))
    [ Nx.Add; Max; Min; Set ]

let ranges =
  group "reduce_ranges"
    [
      prop "a Max or Min range is a NaN where a row is, signalling too"
        nan_ranges law_nan_ranges;
      prop "each range folds its clipped rows in order" ranged law_ranges;
      prop "a range's bits read its rows alone" local law_local;
      test "a range's extreme is one of its rows, at either end"
        test_range_rows_only;
      test "the doc's sums and maxima" test_range_doc;
      test "narrow floats sum at float32" test_range_narrow;
      test "complex extremes in the dtype's order" test_range_complex;
      test "misuse raises" test_range_errors;
      test "a donated operand computes and lives" test_range_donated;
      prop
        "Add, Max and Min raise with their rows cut, naming the placement \
         that keeps the other cuts, and Set gives the host's bits anywhere"
        cut_ranges law_cut_ranges;
      test "eight rows in two tiles raise" test_range_rows_cut;
      test
        "bounds split over the devices that cut x's columns raise naming \
         the place that puts them whole"
        test_range_bounds_cut;
    ]

(* map_segments *)

type routed = { ids : Nx.host Nx.int64_t; x : Nx.host Nx.float64_t; k : int }

let pp_routed ppf r =
  Format.fprintf ppf "ids %a, x %a, %d segments" pp_ints (Nx.shape r.ids)
    pp_ints (Nx.shape r.x) r.k

(* Ids from [-1] to [k], one past the segments, over up to 16 positions; [x]
   of the ids' leading shape, an axis of 1 where positions share a row. *)
let routed =
  Gen.with_pp pp_routed
    (let open Gen in
     let* s = array ~size:(int_range 0 2) (int_range 0 4) in
     let* k = frequency [ (1, int_range 0 5); (1, int_range 1 2) ] in
     let* shared = array ~size:(constant (Array.length s)) bool in
     let lead = Array.mapi (fun i d -> if shared.(i) then 1 else d) s in
     let* row = array ~size:(int_range 0 1) (int_range 1 2) in
     let* vs = array ~size:(constant (numel s)) (int_range (-1) k) in
     let+ x = small (Array.append lead row) in
     { ids = Nx.create Nx.int64 s (Array.map Int64.of_int vs); x; k })

(* A map that reads its owner: each row times [owner + 1], plus [owner]. *)
let by_owner owners rows =
  let g = Nx.dim 0 owners in
  let column t =
    Nx.reshape (Array.append [| g |] (Array.make (Nx.ndim rows - 1) 1)) t
  in
  let o = column (Nx.cast Nx.float64 owners) in
  Nx.add (Nx.mul rows (Nx.add_s o 1.)) o

let law_map r =
  let s = Nx.shape r.ids and xs = Nx.shape r.x in
  let ns = Array.length s in
  let row = Array.sub xs ns (Array.length xs - ns) in
  let w = numel row in
  let p = numel s in
  let ids = Array.map Int64.to_int (Nx.to_array r.ids) in
  cover "few segments for the positions" (r.k > 0 && 2 * r.k < p);
  cover "a position out of range" (Array.exists (fun i -> i < 0 || i >= r.k) ids);
  cover "a shared row" (p > 1 && numel (Array.sub xs 0 ns) < p);
  let full = Nx.to_array (Nx.broadcast_to (Array.append s row) r.x) in
  let expected =
    Array.init (p * w) (fun j ->
        let id = ids.(j / w) in
        if id < 0 || id >= r.k then 0.
        else (full.(j) *. Float.of_int (id + 1)) +. Float.of_int id)
  in
  (* The pairs of an owner and a row that the per-position map meets. *)
  let named = List.filter (fun q -> ids.(q) >= 0 && ids.(q) < r.k) (List.init p Fun.id) in
  cover "no position in range" (p > 0 && r.k > 0 && named = []);
  let met = List.map (fun q -> (ids.(q), Array.sub full (q * w) w)) named in
  let calls = ref 0 in
  let f owners rows =
    incr calls;
    let g = Nx.dim 0 owners and c = Nx.dim 1 rows in
    equal (array int) (Array.append [| g; c |] row) (Nx.shape rows);
    let os = Array.map Int64.to_int (Nx.to_array owners) and rs = Nx.to_array rows in
    Array.iter (fun o -> if o < 0 || o >= r.k then failf "owner %d" o) os;
    (* Each row of a block is a row of an in-range position of the block's
       owner, or zeros where no position is in range, and each in-range
       position fills a row: a pair appears at least as often as positions
       hold it. *)
    let given = List.init (g * c) (fun i -> (os.(i / c), Array.sub rs (i * w) w)) in
    List.iteri
      (fun i ((o, v) as pair) ->
        let ok = if met = [] then Array.for_all (fun e -> e = 0.) v else List.mem pair met in
        if not ok then
          failf "block %d, row %d: owner %d with a row no position of it holds" (i / c) (i mod c) o)
      given;
    let count pair l = List.length (List.filter (( = ) pair) l) in
    List.iter2
      (fun q pair ->
        if count pair given < count pair met then failf "position %d fills no row of its own" q)
      named met;
    by_owner owners rows
  in
  let y = Nx.map_segments ~segments:r.k r.ids f r.x in
  equal int 1 !calls;
  equal (array int) (Array.append s row) (Nx.shape y);
  equal (array float_exact) expected (Nx.to_array y)

(* Ids and rows split along their first axis over two or four devices, both
   or one of them, give the whole result: blocks of one row, then blocks of
   four, in windows of equal or unequal extents. *)
let test_map_split (_, (module S : Nx.Devices), s, segments) =
  let p = numel s in
  let ids =
    Nx.create Nx.int64 s
      (Array.init p (fun i -> Int64.of_int ((i * 5 mod (segments + 2)) - 1)))
  in
  let x =
    Nx.create Nx.float64 (Array.append s [| 2 |])
      (Array.init (2 * p) Float.of_int)
  in
  let whole = Nx.to_array (Nx.map_segments ~segments ids by_owner x) in
  List.iter
    (fun (pi, px) ->
      let y =
        Nx.map_segments ~segments (Nx.place pi ids) by_owner (Nx.place px x)
      in
      equal ~msg:(Format.asprintf "ids at %a, x at %a" Nx.Placement.pp pi
             Nx.Placement.pp px)
        (array float_exact) whole (Nx.to_array y))
    [ (S.split ~axis:0, S.split ~axis:0); (S.split ~axis:0, S.on);
      (S.on, S.split ~axis:0) ]

(* [f]'s blocks are cut as the positions' groups, which in unequal windows
   are the indices of the first axis: with several positions at an index,
   a block would hold positions of several devices. The call raises naming
   the place that puts ids and x whole, where they compute. *)
let test_map_unequal () =
  let ids = Nx.zeros Nx.int64 [| 5; 8 |] in
  let x = Nx.zeros Nx.float64 [| 5; 8; 2 |] in
  let split t = Nx.place (S2.split ~axis:0) t in
  List.iter
    (fun (ids, x) ->
      raises
        (Invalid_argument
           "Nx.map_segments: ids [5; 8] are split in 2 windows of unequal \
            extents, with several positions at each index of their first \
            axis, and are read whole with x; Nx.place (Placement.on s) ids \
            and Nx.place (Placement.on s) x")
        (fun () -> Nx.map_segments ~segments:1 ids by_owner x))
    [ (split ids, split x); (split ids, Nx.place S2.on x);
      (Nx.place S2.on ids, split x) ];
  let whole = Nx.to_array (Nx.map_segments ~segments:1 ids by_owner x) in
  equal (array float_exact) whole
    (Nx.to_array
       (Nx.map_segments ~segments:1 (Nx.place S2.on ids) by_owner
          (Nx.place S2.on x)))

(* Segments far above the positions: each position in range is still its
   row mapped by its owner, the last segment included, and one out of range
   is zeros. *)
let test_map_sparse () =
  let ids =
    Nx.create Nx.int64 [| 6 |]
      [| 3L; Int64.shift_left 1L 40; 0L; -1L; Int64.of_int (max_int - 1); 3L |]
  in
  let x = Nx.create Nx.float64 [| 6; 1 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  equal (array float_exact)
    [| 7.; (3. *. 0x1p40) +. 2.; 3.; 0.; 6. *. 0x1p62; 27. |]
    (Nx.to_array (Nx.map_segments ~segments:max_int ids by_owner x))

let test_map_doc () =
  let ids = Nx.create Nx.int64 [| 4 |] [| 1L; -1L; 0L; 1L |] in
  let x = Nx.create Nx.float64 [| 4; 1 |] [| 1.; 2.; 3.; 4. |] in
  let scale = Nx.create Nx.float64 [| 2; 1; 1 |] [| 10.; 100. |] in
  equal (array float_exact) [| 100.; 0.; 30.; 400. |]
    (Nx.to_array
       (Nx.map_segments ~segments:2 ids
          (fun owners rows -> Nx.mul rows (Nx.take ~axis:0 owners scale))
          x))

let test_map_errors () =
  let by = "Nx.map_segments" in
  let ids = Nx.zeros Nx.int64 [| 3 |] in
  let x = Nx.zeros Nx.float32 [| 3; 2 |] in
  invalid ~by (fun () -> Nx.map_segments ~segments:(-1) ids (fun _ r -> r) x);
  invalid ~by (fun () -> Nx.map_segments ~segments:2 ids (fun _ r -> r) (Nx.zeros Nx.float32 [| 2; 2 |]));
  invalid ~by (fun () ->
      Nx.map_segments ~segments:2 (Nx.zeros Nx.int64 [| 3; 2 |]) (fun _ r -> r) (Nx.zeros Nx.float32 [| 3 |]));
  invalid ~by (fun () -> Nx.map_segments ~segments:2 ids (fun _ r -> Nx.reshape [| -1 |] r) x);
  invalid ~by (fun () ->
      let split = Nx.place (S2.split ~axis:1) in
      Nx.map_segments ~segments:2
        (Nx.zeros_at S2.on Nx.int64 [| 2; 2 |])
        (fun _ r -> r)
        (split (Nx.zeros Nx.float32 [| 2; 2 |])))

(* [f] is the user's function: a refusal of an nx function it calls reaches
   the caller as raised, and what it donates dies naming the function it
   called. *)
let test_map_callback () =
  let ids = Nx.create Nx.int64 [| 2 |] [| 0L; 1L |] in
  let x = Nx.zeros Nx.float32 [| 2; 2 |] in
  invalid ~by:"Nx.reshape" (fun () ->
      Nx.map_segments ~segments:2 ids (fun _ r -> Nx.reshape [| 3 |] r) x);
  let t = Nx.create Nx.float32 [| 2 |] [| 1.; 2. |] in
  let f _ r =
    ignore (Nx.neg (Nx.donate t));
    r
  in
  ignore (Nx.map_segments ~segments:2 ids f x);
  raises_match
    (Exn.invalid_arg ~sub:"Nx.copy: float32 [2] was donated to Nx.neg")
    (fun () -> Nx.copy t)

let maps =
  group "map_segments"
    [
      prop "the per-position map, f called once" routed law_map;
      cases "split ids and rows give the whole result"
        ~id:(fun (t, _, s, k) ->
          Format.asprintf "ids %a over %d devices, %d segments" pp_ints s t k)
        [
          (2, (module S2 : Nx.Devices), [| 8 |], 2);
          (2, (module S2), [| 16 |], 1);
          (2, (module S2), [| 5 |], 2);
          (2, (module S2), [| 1 |], 1);
          (2, (module S2), [| 5; 1 |], 1);
          (4, (module S4), [| 5 |], 2);
          (4, (module S4), [| 6 |], 1);
        ]
        test_map_split;
      test "ids of several axes in unequal windows raise"
        test_map_unequal;
      test "the doc's scaled rows" test_map_doc;
      test "segments far above the positions" test_map_sparse;
      test "misuse raises" test_map_errors;
      test "f keeps the names of the functions it calls" test_map_callback;
    ]

(* matmul_segments: each position's row by the matrix its id names, against
   that product computed in OCaml. Entries are small integers, whose sums
   every float dtype holds exactly, or, at int8, ones that wrap: the result
   compares bit for bit. *)

type mdt = F32 | F64 | F16 | Bf16 | I8

let mdt_name = function
  | F32 -> "float32"
  | F64 -> "float64"
  | F16 -> "float16"
  | Bf16 -> "bfloat16"
  | I8 -> "int8"

(* A case: [w] [[e; n; k]] read transposed to [[e; k; n]] where [w_t], else
   stored [[e; k; n]]; [x]'s leading axes each [s]'s or [1], flipped along
   its last axis where [x_flip]; NaN rows at positions out of range where
   [x] is not broadcast. *)
type mcase = {
  dt : mdt;
  s : int array;
  ids : int64 array;
  lead : int array;
  k : int;
  e : int;
  n : int;
  xs : float array;
  ws : float array;
  w_t : bool;
  x_flip : bool;
}

let pp_mcase ppf c =
  Format.fprintf ppf "%s, ids %a [%s], x lead %a k %d%s, w e %d n %d%s"
    (mdt_name c.dt) pp_ints c.s
    (String.concat "; " (Array.to_list (Array.map Int64.to_string c.ids)))
    pp_ints c.lead c.k
    (if c.x_flip then " flipped" else "")
    c.e c.n
    (if c.w_t then " transposed" else "")

let inside c i =
  Int64.compare i 0L >= 0 && Int64.compare i (Int64.of_int c.e) < 0

let mcase =
  let open Gen in
  let* dt = of_list [ F32; F64; F16; Bf16; I8 ] in
  let* e = frequency [ (1, constant 0); (6, int_range 1 4) ] in
  let* k = of_list [ 0; 1; 2; 3; 5 ] in
  let* n = of_list [ 0; 1; 2; 3 ] in
  let* many = frequency [ (6, constant false); (1, constant true) ] in
  let* s =
    if many then map (fun p -> [| p |]) (int_range 65 130)
    else
      let* r = int_range 1 2 in
      map Array.of_list (list ~size:(constant r) (int_range 0 4))
  in
  let p = numel s in
  let id =
    frequency
      [
        (10, map Int64.of_int (int_range 0 (max 0 (e - 1))));
        (2, of_list [ -1L; Int64.of_int e ]);
        (1, of_list [ Int64.min_int; Int64.max_int; -2L ]);
      ]
  in
  let* one = frequency [ (4, constant false); (1, constant true) ] in
  let* ids = array ~size:(constant p) id in
  let ids = if one && p > 0 then Array.make p ids.(0) else ids in
  let* lead =
    let+ keep = list ~size:(constant (Array.length s)) bool in
    Array.of_list (List.mapi (fun i k -> if k then s.(i) else 1) keep)
  in
  let lo, hi = if dt = I8 then (-100, 100) else (-4, 4) in
  let entries m =
    map (Array.map Float.of_int) (array ~size:(constant m) (int_range lo hi))
  in
  let* xs = entries (numel lead * k) in
  let* ws = entries (e * k * n) in
  let+ w_t = bool and+ x_flip = bool in
  let c = { dt; s; ids; lead; k; e; n; xs; ws; w_t; x_flip } in
  (* A row out of range holds NaN, which it never reaches the result. *)
  if dt <> I8 && lead = s then
    let xs =
      Array.mapi (fun j v -> if inside c ids.(j / k) then v else Float.nan) xs
    in
    { c with xs }
  else c

(* [x] and [w] of a case, at float64, as their views lay them out. *)
let x_of c =
  let x = Nx.create Nx.float64 (Array.append c.lead [| c.k |]) c.xs in
  if c.x_flip then Nx.flip ~axes:[ -1 ] x else x

let w_of c =
  if c.w_t then
    Nx.matrix_transpose (Nx.create Nx.float64 [| c.e; c.n; c.k |] c.ws)
  else Nx.create Nx.float64 [| c.e; c.k; c.n |] c.ws

(* Where a case's operands lie: [put role x] places the host value [x]. *)
type mrole = Mx | Mids | Mw

type 'e mput = {
  put : 'v 's. mrole -> ('v, 's, Nx.host) Nx.t -> ('v, 's, 'e) Nx.t;
}

let on_host = { put = (fun _ x -> x) }

(* The result at float64, the operands cast to the case's dtype, then laid
   out as the case's views, where [p] puts them. *)
let matmul_at (type e) ?sum (p : e mput) c =
  let go (type v s) (d : (v, s) Nx.dtype) =
    let x =
      Nx.cast d (Nx.create Nx.float64 (Array.append c.lead [| c.k |]) c.xs)
    in
    let x = p.put Mx x in
    let x = if c.x_flip then Nx.flip ~axes:[ -1 ] x else x in
    let w =
      if c.w_t then
        Nx.matrix_transpose
          (p.put Mw (Nx.cast d (Nx.create Nx.float64 [| c.e; c.n; c.k |] c.ws)))
      else p.put Mw (Nx.cast d (Nx.create Nx.float64 [| c.e; c.k; c.n |] c.ws))
    in
    let ids = p.put Mids (Nx.create Nx.int64 c.s c.ids) in
    let y = Nx.matmul_segments ?sum ids x w in
    (* The devices' terms close where the result lies uncut. *)
    let y = Nx.place (Nx.Placement.uncut (Nx.placement y)) y in
    Nx.to_array (Nx.cast Nx.float64 y)
  in
  match c.dt with
  | F32 -> go Nx.float32
  | F64 -> go Nx.float64
  | F16 -> go Nx.float16
  | Bf16 -> go Nx.bfloat16
  | I8 -> go Nx.int8

(* Each position's row by its matrix, exact, wrapped at int8. *)
let matmul_reference c =
  let x = Nx.to_array (x_of c) and w = Nx.to_array (w_of c) in
  let p = numel c.s and r = Array.length c.s in
  let row q =
    let at = ref 0 and rest = ref q in
    let coord = Array.make r 0 in
    for a = r - 1 downto 0 do
      coord.(a) <- !rest mod c.s.(a);
      rest := !rest / c.s.(a)
    done;
    for a = 0 to r - 1 do
      at := (!at * c.lead.(a)) + if c.lead.(a) = 1 then 0 else coord.(a)
    done;
    !at
  in
  Array.init (p * c.n) (fun o ->
      let q = o / c.n and j = o mod c.n in
      let id = c.ids.(q) in
      if not (inside c id) then 0.
      else
        let id = Int64.to_int id and sum = ref 0. in
        for t = 0 to c.k - 1 do
          sum :=
            !sum +. (x.((row q * c.k) + t) *. w.((((id * c.k) + t) * c.n) + j))
        done;
        if c.dt = I8 then Float.of_int (wrap8 (Float.to_int !sum) 0) else !sum)

let law_matmul c =
  cover "a position out of range"
    (Array.exists (fun i -> not (inside c i)) c.ids);
  cover "an id at int64's extremes"
    (Array.exists (fun i -> i = Int64.min_int || i = Int64.max_int) c.ids);
  cover "a NaN row out of range" (Array.exists Float.is_nan c.xs);
  cover "a shared row" (c.lead <> c.s && numel c.s > 1);
  cover "no position" (numel c.s = 0);
  cover "no matrix" (c.e = 0);
  cover "a strided weight" c.w_t;
  cover "int8 that wraps" (c.dt = I8 && c.k > 1);
  cover "a matrix of more than 64 rows"
    (Array.exists
       (fun i ->
         inside c i
         && Array.fold_left (fun m j -> if j = i then m + 1 else m) 0 c.ids
            > 64)
       c.ids);
  let y = matmul_at on_host c in
  equal (array float_exact) (matmul_reference c) y

(* A row's bits do not depend on the other positions' ids: float32 rows of
   random values, each position's id kept or redrawn. *)
let test_matmul_bits () =
  let rng = Random.State.make [| 7 |] in
  let e = 3 and k = 200 and n = 3 and p = 40 in
  let rand m = Array.init m (fun _ -> Random.State.float rng 2. -. 1.) in
  let x = Nx.create Nx.float32 [| p; k |] (rand (p * k)) in
  let w = Nx.create Nx.float32 [| e; k; n |] (rand (e * k * n)) in
  let ids () =
    Array.init p (fun _ -> Int64.of_int (Random.State.int rng (e + 1)))
  in
  let bits y = Array.map Int32.bits_of_float (Nx.to_array y) in
  for _ = 1 to 10 do
    let a = ids () and b = ids () in
    let keep = Array.init p (fun q -> q mod 3 = 0) in
    let b = Array.mapi (fun q v -> if keep.(q) then a.(q) else v) b in
    let ya = bits (Nx.matmul_segments (Nx.create Nx.int64 [| p |] a) x w)
    and yb = bits (Nx.matmul_segments (Nx.create Nx.int64 [| p |] b) x w) in
    let kept y = Array.mapi (fun o v -> if keep.(o / n) then v else 0l) y in
    equal (array int32) (kept ya) (kept yb)
  done

(* Over two devices: [w] cut along its matrices, closed or kept as the
   devices' terms with [~sum:true] and then placed whole; the ids and [x]'s
   rows cut together. Each is the whole's result. *)
let test_matmul_cut () =
  let c =
    {
      dt = F32;
      s = [| 6 |];
      ids = [| 3L; 0L; -1L; 1L; 3L; 2L |];
      lead = [| 6 |];
      k = 4;
      e = 4;
      n = 2;
      xs = Array.init 24 (fun i -> Float.of_int ((i mod 7) - 3));
      ws = Array.init 32 (fun i -> Float.of_int ((i mod 5) - 2));
      w_t = false;
      x_flip = false;
    }
  in
  let want = matmul_reference c in
  let at wx wi ww =
    {
      put =
        (fun role x ->
          let cut = match role with Mx -> wx | Mids -> wi | Mw -> ww in
          Nx.place (if cut then S2.split ~axis:0 else S2.on) x);
    }
  in
  equal ~msg:"w cut along its matrices" (array float_exact) want
    (matmul_at (at false false true) c);
  equal ~msg:"w cut along its matrices, summed" (array float_exact) want
    (matmul_at ~sum:true (at false false true) c);
  equal ~msg:"ids and rows cut" (array float_exact) want
    (matmul_at (at true true false) c);
  (* Two axes of ids, four positions of two ids each cut in windows of 2,
     one row of [x] for each position's ids: flattened where they lie. In
     windows of 3 and 2, the ids' flattening reads them whole, which
     raises. *)
  let two s n =
    {
      c with
      s = [| s; 2 |];
      ids =
        Array.init (2 * s) (fun i ->
            [| 3L; 0L; -1L; 1L; 2L; Int64.max_int |].(i mod 6));
      lead = [| s; 1 |];
      xs = Array.init (s * n) (fun i -> Float.of_int ((i mod 7) - 3));
    }
  in
  let c = two 4 4 in
  equal ~msg:"two axes of ids and rows cut" (array float_exact)
    (matmul_reference c)
    (matmul_at (at true true false) c);
  invalid ~by:"Nx.matmul_segments" (fun () ->
      ignore (matmul_at (at true true false) (two 5 4)))

let test_matmul_errors () =
  let by = "Nx.matmul_segments" in
  let ids = Nx.zeros Nx.int64 [| 3 |] in
  let x = Nx.zeros Nx.float32 [| 3; 2 |]
  and w = Nx.zeros Nx.float32 [| 2; 2; 4 |] in
  invalid ~by (fun () ->
      Nx.matmul_segments ids x (Nx.zeros Nx.float32 [| 2; 4 |]));
  invalid ~by (fun () ->
      Nx.matmul_segments ids (Nx.zeros Nx.float32 [| 2 |]) w);
  invalid ~by (fun () ->
      Nx.matmul_segments ids (Nx.zeros Nx.float32 [| 2; 2 |]) w);
  invalid ~by (fun () ->
      Nx.matmul_segments ids (Nx.zeros Nx.float32 [| 3; 3 |]) w);
  invalid ~by (fun () ->
      Nx.matmul_segments ids
        (Nx.zeros Nx.bool [| 3; 2 |])
        (Nx.zeros Nx.bool [| 2; 2; 4 |]))

let matmuls =
  group "matmul_segments"
    [
      prop ~count:300 "each position's row by the matrix its id names, exact"
        (Gen.with_pp pp_mcase mcase) law_matmul;
      test "a row's bits do not depend on the other positions' ids"
        test_matmul_bits;
      test "cuts over devices give the whole's result" test_matmul_cut;
      test "misuse raises" test_matmul_errors;
    ]

(* A donated operand that another operation consumed is dead: each function
   names itself, the one the user called, whatever it computes through. *)
let dead () =
  let x = Nx.create Nx.float32 [| 2 |] [| 1.; 2. |] in
  let _ = Nx.neg (Nx.donate x) in
  x

let test_dead_names () =
  let ids = Nx.create Nx.int64 [| 2 |] [| 0L; 1L |] in
  invalid ~by:"Nx.reduce_segments" (fun () ->
      Nx.reduce_segments Max ~segments:2 ids (dead ()));
  List.iter
    (fun c ->
      invalid ~by:"Nx.reduce_ranges" (fun () ->
          Nx.reduce_ranges c ~lo:ids ~hi:ids (dead ())))
    [ Nx.Add; Max; Set ];
  invalid ~by:"Nx.map_segments" (fun () ->
      Nx.map_segments ~segments:2 ids (fun _ r -> r) (dead ()))

let names = group "errors" [ test "a dead operand names the function called" test_dead_names ]

let () = exit (run "nx segments" [ segments; ranges; maps; matmuls; names ])
