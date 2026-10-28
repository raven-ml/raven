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

let invalid ~by f = raises_match (Exn.invalid_arg ~substring:(by ^ ": ")) f
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
let rows n =
  let open Gen in
  let* row = array ~size:(int_range 0 1) (int_range 0 2) in
  let s = Array.append [| n |] row in
  let* kind = int_range 0 6 in
  match kind with
  | 0 -> let+ x = small s in Rows (D.Float64, Add, ( +. ), x)
  | 1 -> let+ x = drawn D.Int8 s in Rows (D.Int8, Add, wrap8, x)
  | 2 -> let+ x = drawn D.Float32 s in Rows (D.Float32, Max, ( +. ), x)
  | 3 -> let+ x = drawn D.Float16 s in Rows (D.Float16, Min, ( +. ), x)
  | 4 -> let+ x = drawn D.Uint8 s in Rows (D.Uint8, Max, ( + ), x)
  | 5 -> let+ x = drawn D.Bool s in Rows (D.Bool, Min, ( || ), x)
  | _ -> let+ x = drawn D.Float32 s in Rows (D.Float32, Set, ( +. ), x)

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

let segments =
  group "reduce_segments"
    [
      prop "each segment folds its rows in order" segmented law_segments;
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

let ranges =
  group "reduce_ranges"
    [
      prop "a Max or Min range is a NaN where a row is, signalling too"
        nan_ranges law_nan_ranges;
      prop "each range folds its clipped rows in order" ranged law_ranges;
      prop "a range's bits read its rows alone" local law_local;
      test "the doc's sums and maxima" test_range_doc;
      test "narrow floats sum at float32" test_range_narrow;
      test "complex extremes in the dtype's order" test_range_complex;
      test "misuse raises" test_range_errors;
      test "a donated operand computes and lives" test_range_donated;
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
  let calls = ref 0 in
  let f owners rows =
    incr calls;
    let g = Nx.dim 0 owners in
    equal (array int) (Array.append [| g; Nx.dim 1 rows |] row) (Nx.shape rows);
    Array.iter
      (fun o -> if Int64.compare o 0L < 0 || Int64.to_int o >= r.k then failf "owner %Ld" o)
      (Nx.to_array owners);
    by_owner owners rows
  in
  let y = Nx.map_segments ~segments:r.k r.ids f r.x in
  equal int 1 !calls;
  equal (array int) (Array.append s row) (Nx.shape y);
  equal (array float_exact) expected (Nx.to_array y)

(* Ids and rows split over two devices give the whole result, each device's
   positions a group of its own: blocks of one row, then blocks of four. *)
let test_map_split (n, segments) =
  let ids =
    Nx.create Nx.int64 [| n |]
      (Array.init n (fun i -> Int64.of_int ((i * 5 mod (segments + 2)) - 1)))
  in
  let x = Nx.create Nx.float64 [| n; 2 |] (Array.init (2 * n) Float.of_int) in
  let whole = Nx.to_array (Nx.map_segments ~segments ids by_owner x) in
  let split t = Nx.place (S2.split ~axis:0) t in
  let y = Nx.map_segments ~segments (split ids) by_owner (split x) in
  equal (array float_exact) whole (Nx.to_array y)

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
    (Exn.invalid_arg ~substring:"Nx.copy: float32 [2] was donated to Nx.neg")
    (fun () -> Nx.copy t)

let maps =
  group "map_segments"
    [
      prop "the per-position map, f called once" routed law_map;
      cases
        ~name:(fun (n, k) -> Printf.sprintf "%d positions, %d segments" n k)
        "split ids and rows give the whole result" [ (8, 2); (16, 1) ]
        test_map_split;
      test "the doc's scaled rows" test_map_doc;
      test "segments far above the positions" test_map_sparse;
      test "misuse raises" test_map_errors;
      test "f keeps the names of the functions it calls" test_map_callback;
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

let () = exit (run "nx segments" [ segments; ranges; maps; names ])
