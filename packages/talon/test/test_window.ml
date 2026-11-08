(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap

(* Frames

   A drawn frame holds a partition [g], a distinct order [o], a key [k] of one
   kind (none for places), bounds, and values [x] (float64) and [y] (int32). A
   coordinate is a key or a bound as the window's predicate reads it: an
   integer, a temporal value in ticks (a date's in seconds), or a float. *)

type coord = I of int | F of float
type kind =
  | Places | Int8 | Uint8 | Int | Float32 | Float | Date | Clock | Datetime
  | Coarse  (* A datetime[ns] key with bounds of duration[s]. *)
  | Duration
type bound = Absent | Lit of coord | Per_row of coord option array

type frame = {
  kind : kind;
  g : int option array;
  o : int array;
  k : coord option array;
  before : bound;
  after : bound;
  exclude : bool;
  partitioned : bool;
  ordered : bool;
  x : float option array;
  y : int option array;
}

let kinds =
  [
    (3, Places, "places"); (1, Int8, "int8"); (1, Uint8, "uint8");
    (2, Int, "int64"); (1, Float32, "float32"); (2, Float, "float64");
    (1, Date, "date"); (1, Clock, "clock"); (1, Datetime, "datetime");
    (1, Coarse, "datetime by seconds"); (1, Duration, "duration");
  ]

let kind_name k =
  let name (_, k', n) = if k = k' then Some n else None in
  Option.get (List.find_map name kinds)
let to_int = function I i -> i | F _ -> assert false
let to_float = function F x -> x | I _ -> assert false
let specials = [ neg_infinity; -1.5; -0.; 0.; 1.; 2.5; infinity; nan ]

let pp_bound ppf = function
  | Lit (I i) -> Format.fprintf ppf "%d" i
  | Lit (F x) -> Format.fprintf ppf "%h" x
  | Absent | Per_row _ -> Format.pp_print_string ppf "-"

(* Keys and bounds draw from few values, so that ties, empty windows and the
   extremes of int8 come often. *)
let coords kind =
  let ints l = Gen.map (fun i -> I i) (Gen.of_list l) in
  let floats l = Gen.map (fun x -> F x) (Gen.of_list l) in
  let day = 86_400 in
  match kind with
  | Places -> (ints [ 0 ], ints [ min_int; -2; -1; 0; 1; 2; 3; 5; max_int ])
  | Int8 ->
      (ints [ -128; -127; -1; 0; 1; 126; 127 ], ints [ -128; -1; 0; 1; 2; 127 ])
  | Uint8 -> (ints [ 0; 1; 2; 254; 255 ], ints [ 0; 1; 2; 255 ])
  | Int -> (ints [ -3; -1; 0; 1; 2; 4 ], ints [ -1; 0; 1; 2; 3; max_int ])
  | Float | Float32 ->
      (floats specials, floats [ -1.; -0.; 0.; 0.5; 1.; 2.5; infinity; nan ])
  | Coarse ->
      let e = 1_000_000_000_000_000_000 in
      ( ints [ -2 * e; -e; 0; 1_000; e + (e / 2); 2 * e ],
        ints
          [ -20_000_000_000; -10_000_000_000; -3_000_000_000; 0; 1;
            1_000_000_000; 3_000_000_000; 10_000_000_000; 20_000_000_000 ] )
  | Date ->
      ( ints [ 0; day; 2 * day; 5 * day ],
        ints [ 0; day; 3 * day / 2; 3 * day; -day ] )
  | Datetime | Duration ->
      (ints [ -2000; 0; 1000; 1500; 4000 ], ints [ -500; 0; 1; 1000; 2500 ])
  | Clock -> (ints [ 0; 1000; 1500; 4000 ], ints [ -500; 0; 1; 1000; 2500 ])

let frame_of only =
  let open Gen in
  let* kind =
    frequency
      (List.filter_map
         (fun (w, k, _) -> if List.mem k only then Some (w, constant k) else None)
         kinds)
  in
  let key, distance = coords kind in
  let* n = frequency [ (1, int_range 0 1); (6, int_range 2 24) ] in
  let rows g = array ~size:(constant n) g in
  let nullable g = frequency [ (1, constant None); (5, map Option.some g) ] in
  let* g = rows (nullable (int_range 0 1)) in
  let* o = map Array.of_list (permutation (List.init n Fun.id)) in
  let* null_keys = bool in
  let key = if null_keys then nullable key else map Option.some key in
  let* k = rows (if kind = Places then constant None else key) in
  let bound =
    frequency
      [
        (2, constant Absent);
        (* Literal spans hold at most 292 years: coarse bounds are columns. *)
        ((if kind = Coarse then 0 else 3), map (fun c -> Lit c) distance);
        (2, map (fun cs -> Per_row cs) (rows (nullable distance)));
      ]
  in
  let* before, after = pair bound bound in
  let* exclude, (partitioned, ordered) = pair bool (pair bool bool) in
  let value = one_of [ of_list specials; float_range (-8.) 8. ] in
  let* x = rows (nullable value) in
  let+ y = rows (nullable (int_range (-1000) 1000)) in
  { kind; g; o; k; before; after; exclude; partitioned; ordered; x; y }

(* Tables and windows *)

let span i = Time.Span.of_ns (Int64.of_int i)

(* [column kind ~bound cs] is the coordinates [cs] as a key of [kind] or, with
   [~bound], as its bounds. *)
let column kind ~bound cs =
  let v ty f = Column.of_options ty (Array.map (Option.map f) cs) in
  match kind with
  | Places | Int -> v Type.int64 to_int
  | Int8 -> v Type.int8 to_int
  | Uint8 -> v Type.uint8 to_int
  | Float -> v Type.float64 to_float
  | Float32 -> v Type.float32 to_float
  | Clock when not bound -> v (Type.clock Type.Ns) (fun c -> span (to_int c))
  | Coarse when bound ->
      let x = Array.map (function Some c -> Int64.of_int (to_int c) | None -> 0L) cs in
      let validity = Nx.create Nx.bool [| Array.length cs |] (Array.map Option.is_some cs) in
      let values = Nx.P (Nx.create Nx.int64 [| Array.length cs |] x) in
      let validity = Some (Nx.cast Nx.bit validity) in
      Result.get_ok
        (Column.of_layout (Type.Any (Type.duration Type.S))
           (Fixed { validity; values }))
  | Date when bound ->
      v (Type.duration Type.S) (fun c -> span (to_int c * 1_000_000_000))
  | Date ->
      v Type.date (fun c -> Option.get (Time.Date.of_days (to_int c / 86_400)))
  | (Datetime | Coarse) when not bound ->
      v (Type.datetime ~zone:"UTC" Type.Ns) (fun c ->
          Time.of_ns (Int64.of_int (to_int c)))
  | Datetime | Coarse | Duration | Clock ->
      v (Type.duration Type.Ns) (fun c -> span (to_int c))

let table f =
  let bound name = function
    | Per_row cs -> [ (name, column f.kind ~bound:true cs) ]
    | _ -> []
  in
  Talon.v ~rows:(Array.length f.o)
    ([
       ("g", Column.of_options Type.int64 f.g); ("o", Column.v Type.int64 f.o);
       ("x", Column.of_options Type.float64 f.x);
       ("y", Column.of_options Type.int32 f.y);
       ( "z",
         Column.of_options Type.float16
           (Array.map (Option.map (fun y -> Float.of_int y /. 4.)) f.y) );
       ( "d",
         Column.of_options (Type.duration Type.Ns) (Array.map (Option.map span) f.y) );
       ("c", Column.of_options Type.int64 (Array.map (Option.map (fun _ -> 5)) f.g));
       ("t", Column.of_options Type.bool (Array.map (Option.map (fun x -> x > 0.)) f.x));
       (* uint64 values each side of 2^63, which order unsigned. *)
       ( "u",
         let bits y = Int64.(add min_int (of_int (2 * y))) in
         let x = Array.map (function Some y -> bits y | None -> 0L) f.y in
         let n = Array.length x in
         let valid = Nx.create Nx.bool [| n |] (Array.map Option.is_some f.y) in
         Column.of_tensor ~validity:(Nx.cast Nx.bit valid)
           (Nx.bitcast Nx.uint64 (Nx.create Nx.int64 [| n |] x)) );
     ]
    @ (if f.kind = Places then []
       else [ ("k", column f.kind ~bound:false f.k) ])
    @ bound "b" f.before @ bound "a" f.after)

let frames =
  let pp ppf f =
    Format.fprintf ppf "%s before %a after %a exclude=%b by=%b ordered=%b@.%a"
      (kind_name f.kind) pp_bound f.before pp_bound f.after f.exclude
      f.partitioned f.ordered Talon.pp (table f)
  in
  Gen.with_pp pp (frame_of (List.map (fun (_, k, _) -> k) kinds))

type 's win = { w : 'a. ('a, Expr.agg) Expr.t -> ('a, 's) Expr.t }

(* [window f] is [f]'s window, partitioned and ordered as [f] says. *)
let window f =
  let seconds = f.kind = Date in
  let open Expr in
  let by = if f.partitioned then [ "g" ] else [] in
  let order = if f.ordered then [ Order.asc "o" ] else [] in
  let along (type k d) (on : (k, row) t option) (lit : coord -> (d, row) t)
      col =
    let bound name = function
      | Absent -> None
      | Lit c -> Some (lit c)
      | Per_row _ -> Some (col name)
    in
    let before = bound "b" f.before and after = bound "a" f.after in
    let exclude_key = f.exclude in
    let w r = over ~by ~order (window ?on ?before ?after ~exclude_key r) in
    { w }
  in
  let span c =
    span (if seconds then Time.Span.s (to_int c) else Time.Span.ns (to_int c))
  in
  let int c = int (to_int c) in
  match f.kind with
  | Places -> along None int Col.int
  | Int8 | Uint8 | Int -> along (Some (Col.int "k")) int Col.int
  | Float | Float32 ->
      along (Some (Col.float "k")) (fun c -> float (to_float c)) Col.float
  | Date -> along (Some (Col.date "k")) span Col.span
  | Datetime | Coarse -> along (Some (Col.instant "k")) span Col.span
  | Duration | Clock -> along (Some (Col.span "k")) span Col.span

let floats =
  [ "sum"; "mean"; "min"; "max"; "first"; "last"; "var"; "std"; "median" ]
  @ [ "q25"; "imean"; "hsum" ]

let ints =
  [ "count"; "rows"; "n_unique"; "arg_min"; "arg_max"; "isum"; "only" ]
  @ [ "dsum"; "tmin"; "tmax"; "ymin"; "ymax"; "umin"; "umax" ]

let outs (w : 's win) : 's Expr.out list =
  let open Expr in
  let x = Col.float "x" and y = Col.int "y" and z = Col.float "z" in
  [
    "sum" := w.w (sum x); "mean" := w.w (mean x); "min" := w.w (min x);
    "max" := w.w (max x); "first" := w.w (first x); "last" := w.w (last x);
    "var" := w.w (var x); "std" := w.w (std x); "median" := w.w (median x);
    "imean" := w.w (mean y); "count" := w.w (count x); "rows" := w.w rows;
    "n_unique" := w.w (n_unique x); "arg_min" := w.w (arg_min x);
    "arg_max" := w.w (arg_max x); "isum" := w.w (sum y);
    "q25" := w.w (quantile 0.25 x); "hsum" := w.w (sum z);
    "only" := w.w (only (Col.int "c")); "dsum" := w.w (sum (Col.span "d"));
    "tmin" := w.w (min (Col.bool "t")); "tmax" := w.w (max (Col.bool "t"));
    "ymin" := w.w (min y); "ymax" := w.w (max y);
    "umin" := w.w (min (Col.int "u")); "umax" := w.w (max (Col.int "u"));
  ]

(* The reference *)

(* Talon's total order: NaN after every number, [-0.] equal to [0.]. *)
let compare_coord a b =
  match (a, b) with
  | I a, I b -> Int.compare a b
  | F a, F b when Float.is_nan a || Float.is_nan b ->
      Bool.compare (Float.is_nan a) (Float.is_nan b)
  | F a, F b -> Float.compare (a +. 0.) (b +. 0.)
  | _ -> assert false

(* [members f i] is the rows of row [i]'s window in its order, or [None] if [i]
   has none: the rows of its partition whose key [kj] has
   [ki - before < kj <= ki + after], computed by IEEE 754 for floats and, for
   integers, on the lag [ki - kj], which is exact. *)
let members f i =
  let bound = function
    | Absent -> Some None
    | Lit c -> Some (Some c)
    | Per_row cs -> Option.map Option.some cs.(i)
  in
  let rows = List.init (Array.length f.o) Fun.id in
  let mine j = (not f.partitioned) || f.g.(j) = f.g.(i) in
  let frame = List.filter mine rows in
  let by_o a b = Int.compare f.o.(a) f.o.(b) in
  let frame = if f.ordered then List.sort by_o frame else frame in
  let place j =
    Option.map (fun p -> I p) (List.find_index (Int.equal j) frame)
  in
  let key j = if f.kind = Places then place j else f.k.(j) in
  match (key i, bound f.before, bound f.after) with
  | Some ki, Some before, Some after ->
      let inside kj =
        match (ki, kj, before, after) with
        | I ki, I kj, b, a ->
            (* Bounds of [tick] key units: [lag < b * tick] iff
               [floor (lag / tick) < b], with no product to overflow. *)
            let tick = if f.kind = Coarse then 1_000_000_000 else 1 in
            let lag = ki - kj and a = to_int (Option.value a ~default:(I 0)) in
            let q = (if lag >= 0 then lag else lag - tick + 1) / tick in
            Option.fold ~none:true ~some:(fun b -> q < to_int b) b && a >= -q
        | F ki, kj, b, a ->
            let at x = if f.kind = Float32 then F (Int32.float_of_bits (Int32.bits_of_float x)) else F x in
            let lo b = compare_coord (at (ki -. to_float b)) kj < 0 in
            let a = to_float (Option.value a ~default:(F 0.)) in
            Option.fold ~none:true ~some:lo b
            && compare_coord kj (at (ki +. a)) <= 0
        | _ -> assert false
      in
      let kept j =
        match key j with
        | Some kj -> inside kj && not (f.exclude && compare_coord kj ki = 0)
        | None -> false
      in
      let by_key a b = compare_coord (Option.get (key a)) (Option.get (key b)) in
      Some (List.stable_sort by_key (List.filter kept frame))
  | _ -> None

let floats_w ~exact =
  let same a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b) in
  let close a b =
    let scale = Float.max 1. (Float.max (Float.abs a) (Float.abs b)) in
    a = b || Float.abs (a -. b) <= 1e-9 *. scale
  in
  let equal a b =
    (Float.is_nan a && Float.is_nan b) || if exact then same a b else close a b
  in
  Testable.make ~pp:(fun ppf -> Format.fprintf ppf "%h") ~equal

let exact_w = floats_w ~exact:true and float_w = floats_w ~exact:false
let witness c =
  if List.mem c [ "min"; "max"; "first"; "last"; "hsum" ] then exact_w
  else float_w

let exec q = Error.get_ok (Query.run q)
let floats_of t c = Column.options Kind.float (Talon.column t c)
let ints_of t c =
  match Column.type_ (Talon.column t c) with
  | Any Uint64 -> (
      (* The bit patterns 2^63 + 2y decode exactly to y. *)
      match Column.layout (Talon.column t c) with
      | Fixed { values; validity } ->
          let x = Nx.to_array (Nx.bitcast Nx.int64 (Nx.unpack Nx.uint64 values)) in
          let held = Option.map (fun v -> Nx.to_array (Nx.cast Nx.bool v)) validity in
          let y x = Int64.(to_int (shift_right (add x min_int) 1)) in
          Array.mapi
            (fun i x ->
              match held with
              | Some h when not h.(i) -> None
              | _ -> Some (y x))
            x
      | _ -> assert false)
  | Any Bool ->
      let b = Column.options Kind.bool (Talon.column t c) in
      Array.map (Option.map Bool.to_int) b
  | Any (Duration _) ->
      let ns d = Int64.to_int (Time.Span.to_ns d) in
      Array.map (Option.map ns) (Column.options Kind.span (Talon.column t c))
  | _ -> Column.options Kind.int (Talon.column t c)
let indices l =
  Nx.create Nx.int64 [| List.length l |]
    (Array.of_list (List.map Int64.of_int l))

(* [same_outputs ~exact got want] compares the outputs of {!outs}, the floats
   by their bits if [exact]. *)
let same_outputs ?(at = "") ~exact got want =
  let w c = if exact then exact_w else witness c in
  let check w read c =
    equal ~msg:(c ^ at) (array (option w)) (read want c) (read got c)
  in
  List.iter (fun c -> check (w c) floats_of c) floats;
  List.iter (check int ints_of) ints

let filter_law f =
  let t = table f and n = Array.length f.o in
  let got = exec (Query.select (outs (window f)) (Query.of_table t)) in
  let windows = List.init n (members f) in
  let has p = List.exists (function Some r -> p r | None -> false) windows in
  let odd = function
    | F x -> Float.is_nan x || Float.abs x = infinity
    | I _ -> false
  in
  let per_row = function Per_row _ -> true | _ -> false in
  cover (kind_name f.kind) true;
  cover "running" (f.before = Absent);
  cover "excluding the row's key" f.exclude;
  cover "per-row bounds" (per_row f.before || per_row f.after);
  cover "a row without a window" (List.mem None windows);
  cover "an empty window" (has (( = ) []));
  cover "a window of two rows or more" (has (fun r -> List.length r >= 2));
  let distinct = List.length (List.sort_uniq compare (Array.to_list f.k)) in
  cover "tied keys" (f.kind <> Places && distinct < n);
  cover "a NaN or infinite key"
    (Array.exists (Option.fold ~none:false ~some:odd) f.k);
  cover "partitions" (f.partitioned && n > 2);
  cover "no row or one" (n <= 1);
  cover "one partition along keys without nulls"
    (f.kind <> Places && (not f.partitioned) && Array.for_all Option.is_some f.k);
  List.iteri
    (fun i w ->
      let row = Talon.take (indices [ i ]) got in
      let at = Printf.sprintf " at row %d" i in
      let null c = equal ~msg:(c ^ at) int 1 (Column.null_count (Talon.column row c)) in
      match w with
      | None -> List.iter null (floats @ ints)
      | Some rows ->
          let rows = Query.of_table (Talon.take (indices rows) t) in
          let want = exec (Query.aggregate ~by:[] (outs { w = Fun.id }) rows) in
          same_outputs ~at ~exact:false row want)
    windows;
  (* Batches change no window: a window needs the whole frame. *)
  let half k =
    let m = n / 2 in
    let rows = List.init (if k = 0 then m else n - m) (( + ) (k * m)) in
    Talon.take (indices rows) t
  in
  let halves = Query.of_table (Talon.of_batches [ half 0; half 1 ]) in
  same_outputs ~exact:true (exec (Query.select (outs (window f)) halves)) got

(* Places are keys spaced equally: along keys [c + s * o], bounds [s * b] and
   [s * a] hold the rows that [b] and [a] hold over places in [o]'s order, in
   that order, so they compute the same bits. *)
let grid =
  let open Gen in
  let* f = frame_of [ Places ] in
  let+ s, c = pair (int_range 1 4) (int_range (-5) 5) in
  let lit = function Per_row _ -> Absent | b -> b in
  let k = Array.map (fun o -> Some (I (c + (s * o)))) f.o in
  let before = lit f.before and after = lit f.after in
  ({ f with k; before; after; partitioned = false }, s)

let grid_law (f, s) =
  let scaled = function
    | Lit (I v) when v > -(1 lsl 40) && v < 1 lsl 40 -> Lit (I (s * v))
    | b -> b
  in
  let before = scaled f.before and after = scaled f.after in
  let keys = { f with kind = Int; ordered = false; before; after } in
  let q = Query.of_table (table keys) in
  let outputs f = exec (Query.select (outs (window f)) q) in
  cover "running" (f.before = Absent);
  cover "excluding" f.exclude;
  same_outputs ~exact:true (outputs keys) (outputs { f with ordered = true })

(* A partition's windows are those of the frame of its rows alone. *)
let partition_law f =
  let f = { f with partitioned = true } in
  let t = table f in
  let whole = exec (Query.select (outs (window f)) (Query.of_table t)) in
  let g = Col.int "g" in
  List.iter
    (fun v ->
      let rows = List.init (Array.length f.g) Fun.id in
      let rows = List.filter (fun i -> f.g.(i) = v) rows in
      let p = match v with None -> Expr.is_null g | Some v -> Expr.(g = int v) in
      let part = Query.filter p (Query.of_table t) in
      let alone = window { f with partitioned = false } in
      let got = exec (Query.select (outs alone) part) in
      same_outputs ~exact:false got (Talon.take (indices rows) whole))
    [ None; Some 0; Some 1 ]

(* Excluding the row's key leaves out the rows of [~before:(int 1)
   ~after:(int 0)], which along integers are the rows of its key. *)
let exclusion =
  let open Gen in
  let+ f = frame_of [ Places; Int ] in
  let at_least m = function
    | Lit (I b) when b < m -> Lit (I m)
    | Per_row _ -> Absent
    | b -> b
  in
  let before = at_least 1 f.before and after = at_least 0 f.after in
  { f with before; after; exclude = false }

let exclusion_law f =
  let q = Query.of_table (table f) and y = Col.int "y" in
  let values e = Error.get_ok (Query.values (Expr.option e) q) in
  let own = { f with before = Lit (I 1); after = Lit (I 0) } in
  let excluded = { f with exclude = true } in
  let total f = (window f).w (Expr.sum y) in
  let counted f = (window f).w (Expr.count y) in
  cover "running" (f.before = Absent);
  cover "places" (f.kind = Places);
  equal (array (option int)) (values (total f))
    (values Expr.(total excluded + total own));
  equal (array (option int)) (values (counted f))
    (values Expr.(counted excluded + counted own))

(* Fills *)

let series =
  let open Gen in
  let* n = int_range 0 30 in
  let rows g = array ~size:(constant n) g in
  let* k = rows (option (int_range 0 9)) in
  let value = map Option.some (float_range (-8.) 8.) in
  let+ x = rows (frequency [ (2, constant None); (3, value) ]) in
  (k, x)

let filled e (k, x) =
  let k = Column.of_options Type.int64 k in
  let t = Talon.v [ ("k", k); ("x", Column.of_options Type.float64 x) ] in
  Error.get_ok (Query.values (Expr.option e) (Query.of_table t))

(* [interpolated key x] is [x], or at a null the line through the last value
   at or before its key and the first after it, in key order, ties in row
   order. With [~fill], one of them alone. *)
let interpolated ?fill key x =
  let rows = List.init (Array.length x) Fun.id in
  let rows = List.filter (fun j -> key.(j) <> None && x.(j) <> None) rows in
  let rows = List.stable_sort (fun a b -> compare key.(a) key.(b)) rows in
  let at j = Float.of_int (Option.get key.(j)) in
  Array.mapi
    (fun i v ->
      let before = List.filter (fun j -> key.(j) <= key.(i)) rows in
      let j0 = List.nth_opt (List.rev before) 0 and j1 = List.find_opt (fun j -> key.(j) > key.(i)) rows in
      match (v, key.(i), fill, j0, j1) with
      | Some _, _, _, _, _ | _, None, _, _, _ -> v
      | None, _, Some `Forward, j, _ | None, _, Some `Backward, _, j -> Option.bind j (Array.get x)
      | None, _, None, Some j0, _ when key.(j0) = key.(i) -> x.(j0)
      | None, _, None, Some j0, Some j1 ->
          let x0 = Option.get x.(j0) and x1 = Option.get x.(j1) in
          Some (x0 +. ((x1 -. x0) *. (at i -. at j0) /. (at j1 -. at j0)))
      | _ -> None)
    x

let fill_law (k, x) =
  let places = Array.mapi (fun i _ -> Some i) x and c = Col.float "x" in
  let distinct = List.length (List.sort_uniq compare (Array.to_list k)) in
  cover "a null between values"
    (Array.exists Option.is_none x && Array.exists Option.is_some x);
  cover "tied keys" (distinct < Array.length k - 1);
  let check w want e = equal (array (option w)) want (filled e (k, x)) in
  check exact_w (interpolated ~fill:`Forward places x) (Kit.fill_forward c);
  check exact_w (interpolated ~fill:`Backward places x) (Kit.fill_backward c);
  check float_w (interpolated places x) (Kit.interpolate c);
  check float_w (interpolated k x) (Kit.interpolate ~on:(Col.int "k") c)

(* Over values on a line, [interpolate] is the line from the first value to the
   last. *)
let linear_law (alpha, beta, (k, x)) =
  let k = Array.mapi (fun i _ -> Some (3 * i)) k in
  let at k = alpha +. (beta *. Float.of_int k) in
  let line = Array.map (Option.map at) k in
  let x = Array.mapi (fun i v -> Option.bind v (fun _ -> line.(i))) x in
  let held = List.init (Array.length x) Fun.id in
  let held = List.filter (fun i -> x.(i) <> None) held in
  let within i =
    held <> [] && List.hd held <= i && i <= List.hd (List.rev held)
  in
  let expected = Array.mapi (fun i v -> if within i then v else None) line in
  let check e = equal (array (option float_w)) expected (filled e (k, x)) in
  check (Kit.interpolate ~on:(Col.int "k") (Col.float "x"));
  check (Kit.interpolate (Col.float "x"))

(* Oracles

   golden/window.txt holds Polars' rolling windows over the keys in (k - w, k],
   which take in the row's key, and in (k - w, k), which leave it out. *)

let golden =
  let rec split = function
    | [] -> []
    | l :: ls ->
        let rec body acc = function
          | l :: ls when not (String.starts_with ~prefix:"# " l) ->
              body (l :: acc) ls
          | ls -> (List.rev acc, ls)
        in
        let rows, rest = body [] ls in
        (String.sub l 2 (String.length l - 2), rows) :: split rest
  in
  split (In_channel.with_open_text "golden/window.txt" In_channel.input_lines)

let oracle (name, rows) =
  let split l = Array.of_list (String.split_on_char ' ' l) in
  let fields = List.map split rows in
  let col i = Array.of_list (List.map (fun r -> r.(i)) fields) in
  let keys = Array.map int_of_string (col 0) in
  let x = Col.float "x" in
  let k, w =
    if name = "datetime" then
      let at us = Time.of_ns (Int64.mul 1000L (Int64.of_int us)) in
      ( Column.v (Type.datetime ~zone:"UTC" Type.Us) (Array.map at keys),
        fun ?exclude_key r ->
          let before = Expr.span (Time.Span.days 7) in
          Expr.window ~on:(Col.instant "k") ~before ?exclude_key r )
    else
      ( Column.v Type.int64 keys,
        fun ?exclude_key r ->
          Expr.(window ~on:(Col.int "k") ~before:(int 3) ?exclude_key r) )
  in
  let xs = Column.v Type.float64 (Array.map float_of_string (col 1)) in
  let q = Query.of_table (Talon.v [ ("k", k); ("x", xs) ]) in
  let values e = Error.get_ok (Query.values (Expr.option e) q) in
  let number = function "null" -> None | s -> Some (float_of_string s) in
  let expect i = Array.map number (col i) in
  equal (array (option float_w)) (expect 2) (values (w (Expr.sum x)));
  equal (array (option exact_w)) (expect 3) (values (w (Expr.max x)));
  equal (array (option float_w)) (expect 4)
    (values (w ~exclude_key:true (Expr.mean x)))

(* A row whose key is null has no window, so that its reduction does not
   run there. *)
let no_window () =
  let k = Column.of_options Type.int64 [| None; Some 1 |] in
  let t = Talon.v [ ("k", k); ("x", Column.v Type.float64 [| 1.; 2. |]) ] in
  let e = Expr.(window ~on:(Col.int "k") (cast Type.uint8 (count (Col.float "x") - int 1))) in
  equal (result (array (option int)) pass) (Ok [| None; Some 0 |])
    (Query.values (Expr.option e) (Query.of_table t))

(* A running median gathers about n{^ 2} / 2 rows, more than one chunk of
   them. *)
let gathered () =
  let n = 1600 in
  let x = Array.init n (fun i -> Float.of_int ((i * 7919) mod 1009)) in
  let median i =
    let a = Array.sub x 0 (i + 1) in
    Array.sort Float.compare a;
    let m = Array.length a in
    if m mod 2 = 1 then a.(m / 2) else (a.((m / 2) - 1) +. a.(m / 2)) /. 2.
  in
  let t = Talon.v [ ("x", Column.v Type.float64 x) ] in
  let e = Expr.(window (median (Col.float "x"))) in
  equal (array float_w) (Array.init n median)
    (Error.get_ok (Query.values e (Query.of_table t)))

(* Problems *)

let problems () =
  let ts = Array.map (fun s -> Option.get (Time.of_s s)) [| 0L; 1L; 2L |] in
  let q =
    Query.of_table
      (Talon.v
         [
           ("x", Column.v Type.float64 [| 1.; 1.; 2. |]);
           ("s", Column.v Type.string [| "a"; "b"; "c" |]);
           ("ts", Column.v (Type.datetime Type.S) ts);
           ( "d",
             Column.v (Type.duration Type.Ns)
               (Array.map Time.Span.ns [| 1; 2; 3 |]) );
         ])
  in
  let x = Col.float "x" in
  let derive e =
    match Query.run (Query.derive Expr.[ "w" := e ] q) with
    | Ok _ -> "no problem"
    | Error e -> Format.asprintf "%a" Error.pp e
    | exception Invalid_argument m -> m
  in
  expect
    (String.concat "\n\n"
       Expr.
         [
           derive (window ~on:(Col.string "s") (count x));
           derive (window ~on:x ~before:(int 1) (count x));
           derive
             (window ~on:(Col.instant "ts") ~before:(Col.span "d") (count x));
           derive (window ~before:(float 1.) (count x));
           derive (window ~before:null (count x));
           derive (window (sum (shift 1 x)));
           derive (window (sum (over (mean x) +. x)));
           derive (window (count (rank x)));
           derive (window (sum (window (sum x))));
           derive (Kit.fill_forward (x -. shift 1 x));
           derive (window ~before:(int 2) (only x));
         ])
  @@ __POS_OF__ {|
    derive: 1 problem
      "w" := window ~on:s (count x)
        window ~on takes integers, floats, dates, datetimes, durations or clocks, not string.
      input (4 columns): x float64, s string, ts datetime[s], d duration[ns]

    derive: 1 problem
      "w" := window ~on:x ~before:1 (count x)
        window along float64 takes bounds of its kind, not int64.
      input (4 columns): x float64, s string, ts datetime[s], d duration[ns]

    derive: 1 problem
      "w" := window ~on:ts ~before:d (count x)
        window along datetime[s] takes bounds of durations of its unit or a coarser one, not duration[ns].
      input (4 columns): x float64, s string, ts datetime[s], d duration[ns]

    derive: 1 problem
      "w" := window ~before:1. (count x)
        window of places takes integer bounds, not float64.
      input (4 columns): x float64, s string, ts datetime[s], d duration[ns]

    derive: 1 problem
      "w" := window ~before:null (count x)
        null has no type: give it one with store.
      input (4 columns): x float64, s string, ts datetime[s], d duration[ns]

    derive: 1 problem
      "w" := window (sum (shift 1 x))
        the reduction of a window holds shift: compute it outside the window.
      input (4 columns): x float64, s string, ts datetime[s], d duration[ns]

    derive: 1 problem
      "w" := window (sum (over (mean x) +. x))
        the reduction of a window holds over: compute it outside the window.
      input (4 columns): x float64, s string, ts datetime[s], d duration[ns]

    derive: 1 problem
      "w" := window (count (rank x))
        the reduction of a window holds rank: compute it outside the window.
      input (4 columns): x float64, s string, ts datetime[s], d duration[ns]

    derive: 1 problem
      "w" := window (sum (window (sum x)))
        the reduction of a window holds window: compute it outside the window.
      input (4 columns): x float64, s string, ts datetime[s], d duration[ns]

    derive: 1 problem
      "w" := window (last (x -. shift 1 x))
        the reduction of a window holds shift: compute it outside the window.
      input (4 columns): x float64, s string, ts datetime[s], d duration[ns]

    derive ["w" := window ~before:2 (only x)]: row 2: only finds several values.
    |}

let () =
  exit
    (run "window"
       [
         group "laws"
           [
             prop ~count:300
               "window r is r over the rows that its predicate keeps" frames
               filter_law;
             prop "places are keys on a grid" grid grid_law;
             prop "a partition's windows are its frame's" frames partition_law;
             prop "excluding the key leaves out its rows" exclusion
               exclusion_law;
           ];
         group "kit"
           [
             prop "fills and interpolation, against their reference" series
               fill_law;
             prop "interpolate recovers a line"
               Gen.(triple (float_range (-5.) 5.) (float_range (-5.) 5.) series)
               linear_law;
           ];
         cases "Polars' rolling windows" ~id:fst golden oracle;
         test "a row without a window computes nothing" no_window;
         test "a gather in chunks" gathered;
         test "a window's problems" problems;
       ])
