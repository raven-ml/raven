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
  | Places
  | Int8
  | Uint8
  | Int
  | Float32
  | Float
  | Date
  | Clock
  | Datetime
  | Coarse (* A datetime[ns] key with bounds of duration[s]. *)
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
    (1, Places, "places");
    (1, Int8, "int8");
    (1, Uint8, "uint8");
    (1, Int, "int64");
    (1, Float32, "float32");
    (1, Float, "float64");
    (1, Date, "date");
    (1, Clock, "clock");
    (1, Datetime, "datetime");
    (1, Coarse, "datetime by seconds");
    (1, Duration, "duration");
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
          [
            -20_000_000_000;
            -10_000_000_000;
            -3_000_000_000;
            0;
            1;
            1_000_000_000;
            3_000_000_000;
            10_000_000_000;
            20_000_000_000;
          ] )
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
         (fun (w, k, _) ->
           if List.mem k only then Some (w, constant k) else None)
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
      let x =
        Array.map (function Some c -> Int64.of_int (to_int c) | None -> 0L) cs
      in
      let validity =
        Nx.create Nx.bool [| Array.length cs |] (Array.map Option.is_some cs)
      in
      let values = Nx.P (Nx.create Nx.int64 [| Array.length cs |] x) in
      let validity = Some (Nx.cast Nx.bit validity) in
      Result.get_ok
        (Column.of_layout
           (Type.Any (Type.duration Type.S))
           (Fixed { validity; values }))
  | Date when bound ->
      v (Type.duration Type.S) (fun c -> span (to_int c * 1_000_000_000))
  | Date ->
      v Type.date (fun c -> Option.get (Time.Date.of_days (to_int c / 86_400)))
  | (Datetime | Coarse) when not bound ->
      v (Type.instant Type.Ns) (fun c -> Time.of_ns (Int64.of_int (to_int c)))
  | Datetime | Coarse | Duration | Clock ->
      v (Type.duration Type.Ns) (fun c -> span (to_int c))

let table f =
  let bound name = function
    | Per_row cs -> [ (name, column f.kind ~bound:true cs) ]
    | _ -> []
  in
  Talon.v ~rows:(Array.length f.o)
    ([
       ("g", Column.of_options Type.int64 f.g);
       ("o", Column.v Type.int64 f.o);
       ("x", Column.of_options Type.float64 f.x);
       ("y", Column.of_options Type.int32 f.y);
       ( "z",
         Column.of_options Type.float16
           (Array.map (Option.map (fun y -> Float.of_int y /. 4.)) f.y) );
       ( "d",
         Column.of_options (Type.duration Type.Ns)
           (Array.map (Option.map span) f.y) );
       ( "c",
         Column.of_options Type.int64 (Array.map (Option.map (fun _ -> 5)) f.g)
       );
       ( "t",
         Column.of_options Type.bool
           (Array.map (Option.map (fun x -> x > 0.)) f.x) );
       (* uint64 values each side of 2^63, which order unsigned. *)
       ( "u",
         let bits y = Int64.(add min_int (of_int (2 * y))) in
         let x = Array.map (function Some y -> bits y | None -> 0L) f.y in
         let n = Array.length x in
         let valid = Nx.create Nx.bool [| n |] (Array.map Option.is_some f.y) in
         Column.of_tensor ~validity:(Nx.cast Nx.bit valid)
           (Nx.bitcast Nx.uint64 (Nx.create Nx.int64 [| n |] x)) );
     ]
    @ (if f.kind = Places then [] else [ ("k", column f.kind ~bound:false f.k) ])
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
  let along (type k d) (on : (k, row) t option) (lit : coord -> (d, row) t) col
      =
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
    "sum" := w.w (sum x);
    "mean" := w.w (mean x);
    "min" := w.w (min x);
    "max" := w.w (max x);
    "first" := w.w (first x);
    "last" := w.w (last x);
    "var" := w.w (var x);
    "std" := w.w (std x);
    "median" := w.w (median x);
    "imean" := w.w (mean y);
    "count" := w.w (count x);
    "rows" := w.w rows;
    "n_unique" := w.w (n_unique x);
    "arg_min" := w.w (arg_min x);
    "arg_max" := w.w (arg_max x);
    "isum" := w.w (sum y);
    "q25" := w.w (quantile 0.25 x);
    "hsum" := w.w (sum z);
    "only" := w.w (only (Col.int "c"));
    "dsum" := w.w (sum (Col.span "d"));
    "tmin" := w.w (min (Col.bool "t"));
    "tmax" := w.w (max (Col.bool "t"));
    "ymin" := w.w (min y);
    "ymax" := w.w (max y);
    "umin" := w.w (min (Col.int "u"));
    "umax" := w.w (max (Col.int "u"));
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
   has none: the rows of its partition whose key [kj] has [ki - before < kj <=
   ki + after], computed by IEEE 754 for floats and, for integers, on the lag
   [ki - kj], which is exact. *)
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
            (* Bounds of [tick] key units: [lag < b * tick] iff [floor (lag /
               tick) < b], with no product to overflow. *)
            let tick = if f.kind = Coarse then 1_000_000_000 else 1 in
            let lag = ki - kj and a = to_int (Option.value a ~default:(I 0)) in
            let q = (if lag >= 0 then lag else lag - tick + 1) / tick in
            Option.fold ~none:true ~some:(fun b -> q < to_int b) b && a >= -q
        | F ki, kj, b, a ->
            let at x =
              if f.kind = Float32 then
                F (Int32.float_of_bits (Int32.bits_of_float x))
              else F x
            in
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
      let by_key a b =
        compare_coord (Option.get (key a)) (Option.get (key b))
      in
      Some (List.stable_sort by_key (List.filter kept frame))
  | _ -> None

(* Bits, every NaN equal to every NaN. *)
let exact_w =
  let same a b =
    Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)
    || (Float.is_nan a && Float.is_nan b)
  in
  Testable.make ~pp:(fun ppf -> Format.fprintf ppf "%h") ~equal:same

(* Within [abs], every NaN equal to every NaN and an infinity to itself. *)
let within abs =
  let close a b =
    (Float.is_nan a && Float.is_nan b) || a = b || Float.abs (a -. b) <= abs
  in
  Testable.make ~pp:(fun ppf -> Format.fprintf ppf "%h") ~equal:close

(* How a window's float output meets the aggregate over the window's rows. The
   values of a frame lie in [-8, 8] (or are infinite or NaN), so k ≤ 24 values
   span R ≤ 16 and have a variance V ≤ 64. - min, max, first and last pick a
   value, and the int32 mean ("imean") sums exactly and divides once (mli
   Expr.mean): equal bits. So is the float16 sum ("hsum"): it folds in float32,
   where every partial sum of y/4 for |y| ≤ 1000 over 24 rows is exact, and
   rounds once. - With exclude_key and after, var and std are the aggregate's
   (mli window floats): equal bits. - Sums, means and quantiles: any order of
   addition errs by at most (k-1) u Σ|x| ≤ 23 · 2^-53 · 192 ≈ 4.9e-13 on each
   side: 1e-12. - var: with before, nx.mli's moments_ranges bound (2D+5)uV +
   (7D²-3D)uR√V, D = 6, gives 3.4e-12, doubled for the sample's k/(k-1); without
   it, the running scan of window.ml (Chan's update over values less the
   segment's first, depth ≤ 10) errs by at most 40·u·k·R² ≈ 2.7e-11: 3e-11. -
   std: a window of range R has std ≥ R/√(2(k-1)) ≥ R/7, so its error is at most
   Δvar · 7/(2R) ≈ 6e-12: 1e-11. *)
let witness f c =
  let exact_moments = f.exclude && f.after <> Absent in
  match c with
  | "min" | "max" | "first" | "last" | "hsum" | "imean" -> exact_w
  | ("var" | "std") when exact_moments -> exact_w
  | "var" -> within 3e-11
  | "std" -> within 1e-11
  | _ -> within 1e-12

let exec q = Error.get_ok (Query.run q)
let floats_of t c = Column.options Kind.float (Talon.column t c)

let ints_of t c =
  match Column.type_ (Talon.column t c) with
  | Any Uint64 -> (
      (* The bit patterns 2^63 + 2y decode exactly to y. *)
      match Column.layout (Talon.column t c) with
      | Fixed { values; validity } ->
          let x =
            Nx.to_array (Nx.bitcast Nx.int64 (Nx.unpack Nx.uint64 values))
          in
          let held =
            Option.map (fun v -> Nx.to_array (Nx.cast Nx.bool v)) validity
          in
          let y x = Int64.(to_int (shift_right (add x min_int) 1)) in
          Array.mapi
            (fun i x ->
              match held with Some h when not h.(i) -> None | _ -> Some (y x))
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
  Nx.create Nx.int64
    [| List.length l |]
    (Array.of_list (List.map Int64.of_int l))

(* [same_bits got want] states that two runs of {!outs} hold the same values,
   floats by their bits. *)
let same_bits got want =
  List.iter
    (fun c ->
      equal ~msg:c (array (option exact_w)) (floats_of want c) (floats_of got c))
    floats;
  List.iter
    (fun c ->
      equal ~msg:c (array (option int)) (ints_of want c) (ints_of got c))
    ints

(* [expected f t windows] is, for every row, the outputs of {!outs} over its
   window's rows by one grouped aggregate: null where a row has no window, and
   over no rows where its window is empty. *)
let expected f t windows =
  let n = Array.length f.o in
  let rows = List.concat (List.map (Option.value ~default:[]) windows) in
  let tags =
    List.concat
      (List.mapi
         (fun i w -> List.map (fun _ -> i) (Option.value ~default:[] w))
         windows)
  in
  let taken = Talon.take (indices rows) t in
  let tagged =
    Talon.v ~rows:(List.length rows)
      (("win", Column.v Type.int64 (Array.of_list tags))
      :: List.map
           (fun (c, _) -> (c, Talon.column taken c))
           (Schema.columns (Talon.schema t)))
  in
  let grouped =
    exec
      (Query.aggregate ~by:[ "win" ]
         (outs { w = Fun.id })
         (Query.of_table tagged))
  in
  let none =
    exec
      (Query.aggregate ~by:[]
         (outs { w = Fun.id })
         (Query.of_table (Talon.take (indices []) t)))
  in
  let group = Array.make n (-1) in
  Array.iteri
    (fun k w -> Option.iter (fun w -> group.(w) <- k) w)
    (ints_of grouped "win");
  (* One row per frame row: its group's, the row of no rows, or null. *)
  let pick i =
    match List.nth windows i with
    | None -> None
    | Some [] -> Some (none, 0)
    | Some _ -> Some (grouped, group.(i))
  in
  let floats c =
    Array.init n (fun i ->
        Option.bind (pick i) (fun (t, k) -> (floats_of t c).(k)))
  in
  let ints c =
    Array.init n (fun i ->
        Option.bind (pick i) (fun (t, k) -> (ints_of t c).(k)))
  in
  (floats, ints)

(* Whether a small integer key and a bound reach past the key type's range,
   where the window's limit lies below or above every key. *)
let past_range f =
  let lo, hi =
    match f.kind with Int8 -> (-128, 127) | Uint8 -> (0, 255) | _ -> (0, -1)
  in
  let bound i = function
    | Absent -> None
    | Lit (I b) -> Some b
    | Per_row cs -> ( match cs.(i) with Some (I b) -> Some b | _ -> None)
    | Lit (F _) -> None
  in
  lo <= hi
  && List.exists
       (fun i ->
         match f.k.(i) with
         | Some (I k) ->
             Option.fold ~none:false
               ~some:(fun b -> k - b < lo)
               (bound i f.before)
             || Option.fold ~none:false
                  ~some:(fun a -> k + a > hi)
                  (bound i f.after)
         | _ -> false)
       (List.init (Array.length f.o) Fun.id)

(* Two frames of two rows: int8 keys with a bound past -128, and float keys with
   a NaN before and an infinite after. *)
let examples =
  let base =
    {
      kind = Int8;
      g = [| Some 0; Some 0 |];
      o = [| 0; 1 |];
      k = [| Some (I (-128)); Some (I 127) |];
      before = Lit (I 127);
      after = Absent;
      exclude = false;
      partitioned = false;
      ordered = false;
      x = [| Some 1.; Some 2. |];
      y = [| Some 1; Some 2 |];
    }
  in
  [
    base;
    {
      base with
      kind = Float;
      k = [| Some (F 1.); Some (F 2.) |];
      before = Lit (F nan);
      after = Lit (F infinity);
    };
  ]

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
  let bound p = function
    | Absent -> false
    | Lit c -> p c
    | Per_row cs -> Array.exists (Option.fold ~none:false ~some:p) cs
  in
  let either p = bound p f.before || bound p f.after in
  let float p = function F x -> p x | I _ -> false in
  cover "a NaN bound" (either (float Float.is_nan));
  cover "an infinite bound" (either (float (fun x -> Float.abs x = infinity)));
  cover "a bound past the key type's range" (past_range f);
  let want_floats, want_ints = expected f t windows in
  List.iter
    (fun c ->
      equal ~msg:c
        (array (option (witness f c)))
        (want_floats c) (floats_of got c))
    floats;
  List.iter
    (fun c -> equal ~msg:c (array (option int)) (want_ints c) (ints_of got c))
    ints

(* A window needs the whole frame, so batches change none of its values. *)
let batch_law f =
  let t = table f and n = Array.length f.o in
  let half k =
    let m = n / 2 in
    Talon.take
      (indices (List.init (if k = 0 then m else n - m) (( + ) (k * m))))
      t
  in
  cover "two batches of rows" (n >= 2);
  let run t = exec (Query.select (outs (window f)) (Query.of_table t)) in
  same_bits (run (Talon.of_batches [ half 0; half 1 ])) (run t)

(* Places are keys spaced equally: along keys [c + s * o], bounds [s * b] and [s
   * a] hold the rows that [b] and [a] hold over places in [o]'s order, in that
   order, so they compute the same bits. *)
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
  same_bits (outputs keys) (outputs { f with ordered = true })

(* Moments at large magnitudes *)

(* [two_pass xs] is [(v, k)]: the sample variance [v] of [xs *. 2^-k] by two
   passes, 2^k the least power of two above every finite magnitude of [xs], so
   that no intermediate overflows; NaN over a NaN or an infinity, [None] under
   two values. The variance of [xs] is [v] times 4^k, their standard deviation
   its root times 2^k. *)
let two_pass xs =
  let n = List.length xs in
  if n < 2 then None
  else if List.exists (fun x -> not (Float.is_finite x)) xs then Some (nan, 0)
  else
    let k = List.fold_left (fun k x -> Int.max k (snd (Float.frexp x))) 0 xs in
    let xs = List.map (fun x -> Float.ldexp x (-k)) xs in
    let m = List.fold_left ( +. ) 0. xs /. Float.of_int n in
    let q = List.fold_left (fun q x -> q +. ((x -. m) *. (x -. m))) 0. xs in
    Some (q /. Float.of_int (n - 1), k)

(* Frames whose values are [x *. 2^s], [x] within 1 of a centre and [s] 0 or
   512, of one [s], or mixed with 1018: at 512 their squares and the sums of
   their squared deviations pass max_float, while most of their variances stay
   below it, and values of [s = 0] come before and after values near
   max_float. *)
let large =
  let open Gen in
  let gen =
    let* f = frame_of (List.map (fun (_, k, _) -> k) kinds) in
    let* running = bool in
    let f =
      if running then { f with before = Absent; exclude = false } else f
    in
    let* s =
      frequency
        [
          (1, constant (constant 0));
          (2, constant (constant 512));
          ( 3,
            constant
              (frequency
                 [ (6, constant 0); (1, constant 512); (3, constant 1018) ]) );
        ]
    in
    let* c = of_list [ 0.; 1.5; -6. ] in
    let near = map (fun d -> c +. d) (float_range (-1.) 1.) in
    let value = frequency [ (1, of_list specials); (8, near) ] in
    let value =
      let* v = value in
      map (Float.ldexp v) s
    in
    let size = constant (Array.length f.o) in
    let held = map Option.some value in
    let+ x = array ~size (frequency [ (1, constant None); (4, held) ]) in
    { f with x }
  in
  let pp ppf f =
    Format.fprintf ppf "%s before %a after %a exclude=%b by=%b@.%a"
      (kind_name f.kind) pp_bound f.before pp_bound f.after f.exclude
      f.partitioned Talon.pp (table f)
  in
  with_pp pp gen

(* A running frame of 1 and 2 beside 1.5 · 2^1018 in its partition. *)
let beside_huge =
  {
    kind = Places;
    g = [| Some 0; Some 0; Some 0 |];
    o = [| 0; 1; 2 |];
    k = [| None; None; None |];
    before = Absent;
    after = Absent;
    exclude = false;
    partitioned = false;
    ordered = false;
    x = [| Some 1.; Some 2.; Some (Float.ldexp 1.5 1018) |];
    y = [| Some 1; Some 2; Some 3 |];
  }

let large_law f =
  let t = table f and x = Col.float "x" and w = window f in
  let values e =
    Error.get_ok (Query.values (Expr.option (w.w e)) (Query.of_table t))
  in
  let windows = List.init (Array.length f.o) (members f) in
  let held rows = List.filter_map (Array.get f.x) rows in
  let moments =
    List.map (fun r -> Option.bind r (fun r -> two_pass (held r))) windows
  in
  let beyond k x = Float.is_finite x && Float.abs x >= Float.ldexp 1. k in
  let has p = List.exists (function Some r -> p r | None -> false) windows in
  let running = f.before = Absent && ((not f.exclude) || f.after = Absent) in
  let nulls r = List.exists (fun j -> f.x.(j) = None) r in
  (* The sum of squared deviations passes max_float, the variance not. *)
  let overflows r =
    match two_pass (held r) with
    | Some (v, k) when Float.is_finite v ->
        let q = v *. Float.of_int (List.length (held r) - 1) in
        Float.ldexp q (2 * k) = infinity && Float.ldexp v (2 * k) < infinity
    | _ -> false
  in
  (* A window of values below 2^480 whose partition holds one past 2^1000. *)
  let beside i = function
    | Some r ->
        let mine j = (not f.partitioned) || f.g.(j) = f.g.(i) in
        let part = List.filter mine (List.init (Array.length f.x) Fun.id) in
        List.length (held r) >= 2
        && List.for_all (fun x -> Float.abs x < Float.ldexp 1. 480) (held r)
        && List.exists (beyond 1000) (held part)
    | None -> false
  in
  let some_big r = List.exists (beyond 480) (held r) in
  cover "a running window holding a null and values past 2^480"
    (running
    && has (fun r -> nulls r && List.length (held r) >= 2 && some_big r));
  cover "a running window of values below 2^480 beside one past 2^1000"
    (running && List.exists Fun.id (List.mapi beside windows));
  cover "a finite variance whose squared deviations sum past max_float"
    (has overflows);
  cover "an empty window" (has (( = ) []));
  let scaled op = function Some (v, k) -> Some (op v k) | None -> None in
  let var = List.map (scaled (fun v k -> Float.ldexp v (2 * k))) moments in
  let std = List.map (scaled (fun v k -> Float.ldexp (sqrt v) k)) moments in
  let past v d = v = Some infinity && Option.is_some d && d <> Some infinity in
  cover "a finite standard deviation of a variance past max_float"
    (List.exists2 past var std);
  (* Both sides err by a few rounding errors of the squares, at most about n eps
     4^k over a window's 2^k with n <= 24: within [1e-12 4^k] and 1e-9 of the
     variance, and within the root of that of the deviation. *)
  let close abs =
    let equal a b =
      let scale = Float.max (Float.abs a) (Float.abs b) in
      (Float.is_nan a && Float.is_nan b)
      || a = b
      || Float.abs (a -. b) <= abs +. (1e-9 *. scale)
    in
    Testable.make ~pp:(fun ppf -> Format.fprintf ppf "%h") ~equal
  in
  let powers = Array.of_list moments in
  let check msg tol want got =
    List.iteri
      (fun i (want, got) ->
        let abs = match powers.(i) with Some (_, k) -> tol k | None -> 0. in
        let msg = Printf.sprintf "%s at row %d" msg i in
        equal ~msg (option (close abs)) want got)
      (List.combine want (Array.to_list got))
  in
  check "var" (fun k -> Float.ldexp 1e-12 (2 * k)) var (values (Expr.var x));
  check "std" (fun k -> Float.ldexp 1e-6 k) std (values (Expr.std x))

(* The variances and standard deviations of values whose squares or sums pass
   max_float, within 1e-9: the decimal values round. *)
let large_cases =
  let near =
    [| Some 2e155; Some 2.0001e155; None; Some 2.0002e155; Some 2.0003e155 |]
  in
  let apart = [| Some 1e154; Some (-1e154); Some 0. |] in
  let top = [| Some 1e308; Some 1e308; Some 1e308 |] in
  let var = Expr.var and std = Expr.std in
  [
    ( "a null among values near 2e155",
      var,
      None,
      near,
      [| None; Some 5e301; Some 5e301; Some 1e302; Some (1e302 *. 5. /. 3.) |]
    );
    ( "a null among values near 2e155, two at a time",
      var,
      Some 2,
      near,
      [| None; Some 5e301; None; None; Some 5e301 |] );
    ( "a null before values near 2e155",
      var,
      None,
      [| None; Some 2e155; Some 2.0001e155 |],
      [| None; None; Some 5e301 |] );
    ( "squared deviations past max_float",
      var,
      None,
      apart,
      [| None; Some infinity; Some 1e308 |] );
    ( "squared deviations past max_float, three at a time",
      var,
      Some 3,
      apart,
      [| None; Some infinity; Some 1e308 |] );
    ("a sum past max_float", var, None, top, [| None; Some 0.; Some 0. |]);
    ( "a sum past max_float, three at a time",
      var,
      Some 3,
      top,
      [| None; Some 0.; Some 0. |] );
    ( "small values before one near max_float",
      var,
      None,
      [| Some 1.; Some 2.; Some 3.; Some 1e308 |],
      [| None; Some 0.5; Some 1.; Some infinity |] );
    ( "the deviation of small values before one near max_float",
      std,
      None,
      [| Some 1.; Some 2.; Some 3.; Some 1e308 |],
      [| None; Some (sqrt 0.5); Some 1.; Some 5e307 |] );
    ( "the deviation of tiny values",
      std,
      None,
      [| Some 1e-170; Some (-1e-170) |],
      [| None; Some (sqrt 2. *. 1e-170) |] );
    ( "the deviation of tiny values, two at a time",
      std,
      Some 2,
      [| Some 1e-170; Some (-1e-170); Some 1e-170 |],
      [| None; Some (sqrt 2. *. 1e-170); Some (sqrt 2. *. 1e-170) |] );
    ( "the variance of tiny values, below the range",
      var,
      None,
      [| Some 1e-170; Some (-1e-170) |],
      [| None; Some 0. |] );
    ( "the deviation of equal tiny values, seven at a time",
      std,
      Some 7,
      Array.make 12 (Some 1.3e-170),
      Array.append [| None |] (Array.make 11 (Some 0.)) );
    ( "the variance of equal values near max_float, seven at a time",
      var,
      Some 7,
      Array.make 12 (Some 1.3e300),
      Array.append [| None |] (Array.make 11 (Some 0.)) );
    ( "the deviation of a variance past max_float",
      std,
      None,
      [| Some 1e200; Some (-1e200) |],
      [| None; Some (sqrt 2. *. 1e200) |] );
    ( "the deviation of a variance past max_float, two at a time",
      std,
      Some 2,
      [| Some 1e200; Some (-1e200) |],
      [| None; Some (sqrt 2. *. 1e200) |] );
    ( "the deviation of a variance past max_float, a null among three at a time",
      std,
      Some 3,
      [| Some 1e200; None; Some (-1e200) |],
      [| None; None; Some (sqrt 2. *. 1e200) |] );
    ( "the deviation of tiny values beside a large one",
      std,
      None,
      [| Some 1e-170; Some (-1e-170); Some 1e300 |],
      [| None; Some (sqrt 2. *. 1e-170); Some (1e300 /. sqrt 3.) |] );
    ( "the deviation of tiny values beside a large one, two at a time",
      std,
      Some 2,
      [| Some 1e-170; Some (-1e-170); Some 1e300 |],
      [| None; Some (sqrt 2. *. 1e-170); Some (1e300 /. sqrt 2.) |] );
    ( "the deviation of values whose difference overflows",
      std,
      None,
      [| Some 1.2e308; Some (-1.2e308) |],
      [| None; Some (sqrt 2. *. 1.2e308) |] );
    ( "the deviation of three values whose differences overflow",
      std,
      None,
      [| Some 1.2e308; Some (-1.2e308); Some 0. |],
      [| None; Some (sqrt 2. *. 1.2e308); Some 1.2e308 |] );
    ("no rows, two at a time", var, Some 2, [||], [||]);
    ("no rows, from the start", var, None, [||], [||]);
    ( "no values, from the start",
      var,
      None,
      [| None; None; None |],
      [| None; None; None |] );
    ( "no values, two at a time",
      var,
      Some 2,
      [| None; None; None |],
      [| None; None; None |] );
    ( "one value among nulls, two at a time",
      std,
      Some 2,
      [| None; Some 1.; None |],
      [| None; None; None |] );
  ]

let large_case (_, r, before, x, want) =
  let t = Talon.v [ ("x", Column.of_options Type.float64 x) ] in
  let before = Option.map Expr.int before in
  let e = Expr.window ?before (r (Col.float "x")) in
  equal
    (array (option (float_rel ~rel:1e-9 ~abs:0.)))
    want
    (Error.get_ok (Query.values (Expr.option e) (Query.of_table t)))

(* Values far from zero *)

(* [exact_var ks] is the sample variance of the integers [ks] divided by 2^8,
   from integer sums that do not round, or [None] under two values. *)
let exact_var ks =
  let n = List.length ks in
  if n < 2 then None
  else
    let s = List.fold_left ( + ) 0 ks in
    let q = List.fold_left (fun q k -> q + (k * k)) 0 ks in
    let num = (n * q) - (s * s) in
    Some (Float.of_int num /. Float.of_int (n * (n - 1)) /. 65536.)

(* Frames whose values are [c +. d], [c] far past [d]'s spread and [d] a
   multiple of 2^-8, so that each value is exact and the variance of the values
   is that of the [d]. *)
let offset =
  let open Gen in
  let gen =
    let* f = frame_of (List.map (fun (_, k, _) -> k) kinds) in
    let* shape = of_list [ `Running; `Excluding; `Any; `Any ] in
    let f =
      match shape with
      | `Running -> { f with before = Absent; exclude = false }
      | `Excluding -> { f with before = Absent; after = Absent; exclude = true }
      | `Any -> f
    in
    let* c = of_list [ 1e9; 1e12; -1e12 ] in
    let d = map (fun k -> Float.of_int k /. 256.) (int_range (-256) 256) in
    let held = map (fun d -> Some (c +. d)) d in
    let size = constant (Array.length f.o) in
    let+ x = array ~size (frequency [ (1, constant None); (4, held) ]) in
    ({ f with x }, c)
  in
  let pp ppf (f, c) =
    Format.fprintf ppf "%s c=%h before %a after %a exclude=%b by=%b@.%a"
      (kind_name f.kind) c pp_bound f.before pp_bound f.after f.exclude
      f.partitioned Talon.pp (table f)
  in
  with_pp pp gen

let offset_law (f, c) =
  let t = table f and w = window f and x = Col.float "x" in
  let got =
    Error.get_ok
      (Query.values (Expr.option (w.w (Expr.var x))) (Query.of_table t))
  in
  let windows = List.init (Array.length f.o) (members f) in
  let held rows = List.filter_map (Array.get f.x) rows in
  let enough = function Some r -> List.length (held r) >= 2 | None -> false in
  let some = List.exists enough windows in
  cover "a window of two values or more, without before"
    (f.before = Absent && some);
  cover "a window of two values or more, with before"
    (f.before <> Absent && some);
  cover "a window of two values or more, excluding its key, with after"
    (f.exclude && f.after <> Absent && some);
  cover "a window of two values or more, excluding its key, from the start"
    (f.exclude && f.before = Absent && f.after = Absent && some);
  let variance xs =
    let ks = List.map (fun x -> Int.of_float ((x -. c) *. 256.)) xs in
    exact_var ks
  in
  let want =
    List.map (fun r -> Option.bind r (fun r -> variance (held r))) windows
  in
  equal (array (option (float_rel ~rel:1e-12 ~abs:0.))) (Array.of_list want) got

(* Fills

   A series is rows of a value [x] and a key [k]: none (places), integers, or
   floats among which infinities, NaN and both zeros. *)

type keys = Places_keys | Int_keys | Float_keys

let series =
  let open Gen in
  let* n = int_range 0 30 in
  let rows g = array ~size:(constant n) g in
  let* keys = of_list [ Places_keys; Int_keys; Float_keys ] in
  let key =
    match keys with
    | Places_keys -> constant None
    | Int_keys -> option (map Float.of_int (int_range 0 9))
    | Float_keys ->
        option
          (frequency
             [
               (2, of_list [ neg_infinity; -0.; 0.; infinity; nan ]);
               (3, map (fun i -> Float.of_int i /. 2.) (int_range (-6) 6));
             ])
  in
  let* k = rows key in
  let value = map Option.some (float_range (-8.) 8.) in
  let+ x = rows (frequency [ (2, constant None); (3, value) ]) in
  (keys, k, x)

let pp_series ppf (keys, k, x) =
  let o f = function Some v -> f v | None -> "null" in
  Format.fprintf ppf "%s@."
    (match keys with
    | Places_keys -> "places"
    | Int_keys -> "int keys"
    | Float_keys -> "float keys");
  Array.iteri
    (fun i x ->
      Format.fprintf ppf "%s %s@."
        (o (Printf.sprintf "%h") k.(i))
        (o (Printf.sprintf "%h") x))
    x

let drawn_series = Gen.with_pp pp_series series

(* [filled e series] is [e] on each row of the series' table. *)
let filled e (keys, k, x) =
  let key =
    match keys with
    | Places_keys -> []
    | Int_keys ->
        [
          ( "k",
            Column.of_options Type.int64 (Array.map (Option.map Float.to_int) k)
          );
        ]
    | Float_keys -> [ ("k", Column.of_options Type.float64 k) ]
  in
  let t =
    Talon.v ~rows:(Array.length x)
      (key @ [ ("x", Column.of_options Type.float64 x) ])
  in
  Error.get_ok (Query.values (Expr.option e) (Query.of_table t))

let cover_series (_, _, x) =
  cover "a null between values"
    (Array.exists Option.is_none x && Array.exists Option.is_some x)

let held x =
  List.filter (fun j -> x.(j) <> None) (List.init (Array.length x) Fun.id)

let cumulative_law ((_, _, x) as s) =
  cover_series s;
  (* The count of values up to each row. *)
  let counts =
    Array.mapi
      (fun i _ -> Some (List.length (List.filter (fun j -> j <= i) (held x))))
      x
  in
  equal
    (array (option int))
    counts
    (filled (Kit.cumulative (Expr.count (Col.float "x"))) s)

let fill_forward_law ((_, _, x) as s) =
  cover_series s;
  let last i =
    List.fold_left (fun acc j -> if j <= i then x.(j) else acc) None (held x)
  in
  equal
    (array (option exact_w))
    (Array.mapi (fun i _ -> last i) x)
    (filled (Kit.fill_forward (Col.float "x")) s)

let fill_backward_law ((_, _, x) as s) =
  cover_series s;
  let next i =
    List.find_opt (fun j -> j >= i) (held x)
    |> Option.map (fun j -> Option.get x.(j))
  in
  equal
    (array (option exact_w))
    (Array.mapi (fun i _ -> next i) x)
    (filled (Kit.fill_backward (Col.float "x")) s)

(* Talon's total order on floats: NaN after every number, [-0.] equal to
   [0.]. *)
let order a b = compare_coord (F a) (F b)

(* [interpolated keys x] is Kit.interpolate's definition: at a null, the line
   through the last value of the window of keys up to the row's, [k <= ki + 0],
   and the first of the window [ki - 0 < k <= ki + inf], each window in key
   order, ties in row order, its bounds computed by IEEE 754 and compared in
   talon's order. *)
let interpolated keys x =
  let n = Array.length x in
  let rows =
    List.stable_sort
      (fun a b -> order (Option.get keys.(a)) (Option.get keys.(b)))
      (List.filter (fun j -> keys.(j) <> None) (List.init n Fun.id))
  in
  let key j = Option.get keys.(j) in
  Array.init n (fun i ->
      match (x.(i), keys.(i)) with
      | Some v, _ -> Some v
      | None, None -> None
      | None, Some ki -> (
          let upto =
            List.filter (fun j -> order (key j) (ki +. 0.) <= 0) rows
          in
          let lo = ki -. 0. and hi = ki +. infinity in
          let past =
            List.filter
              (fun j -> order lo (key j) < 0 && order (key j) hi <= 0)
              rows
          in
          let last =
            List.fold_left
              (fun acc j -> if x.(j) <> None then Some j else acc)
              None upto
          in
          let first = List.find_opt (fun j -> x.(j) <> None) past in
          match last with
          | None -> None
          | Some j0 -> (
              let x0 = Option.get x.(j0) and k0 = key j0 in
              if order ki k0 = 0 then Some x0
              else
                match first with
                | None -> None
                | Some j1 ->
                    let x1 = Option.get x.(j1) and k1 = key j1 in
                    Some (x0 +. ((x1 -. x0) *. (ki -. k0) /. (k1 -. k0))))))

let interpolate_law ((keys, k, x) as s) =
  cover_series s;
  cover "places" (keys = Places_keys);
  cover "integer keys" (keys = Int_keys);
  let odd v = Float.is_nan v || Float.abs v = infinity in
  cover "an infinite or NaN key"
    (Array.exists (Option.fold ~none:false ~some:odd) k);
  let present = List.filter_map Fun.id (Array.to_list k) in
  cover "tied keys"
    (List.length (List.sort_uniq order present) < List.length present);
  let x' = Col.float "x" in
  let e, keys_of =
    match keys with
    | Places_keys ->
        (Kit.interpolate x', Array.mapi (fun i _ -> Some (Float.of_int i)) x)
    | Int_keys -> (Kit.interpolate ~on:(Col.int "k") x', k)
    | Float_keys -> (Kit.interpolate ~on:(Col.float "k") x', k)
  in
  equal (array (option exact_w)) (interpolated keys_of x) (filled e s)

(* Oracles

   golden/window.txt holds Polars' rolling windows over the keys in (k - w, k],
   which take in the row's key, and in (k - w, k), which leave it out. *)

(* [golden name] is the rows of the series [name] of golden/window.txt. *)
let golden name =
  let lines =
    In_channel.with_open_text "golden/window.txt" In_channel.input_lines
  in
  let rec rows = function
    | [] -> []
    | l :: ls when l = "# " ^ name ->
        let rec body = function
          | l :: ls when not (String.starts_with ~prefix:"# " l) -> l :: body ls
          | _ -> []
        in
        body ls
    | _ :: ls -> rows ls
  in
  rows lines

(* Polars sums in its own order: over at most 160 values of magnitude at most
   25, any two orders differ by at most 160 · 2^-53 · 4000 ≈ 7e-11 on each side.
   Its max picks a value. *)
let oracle name =
  let rows = golden name in
  let split l = Array.of_list (String.split_on_char ' ' l) in
  let fields = List.map split rows in
  let col i = Array.of_list (List.map (fun r -> r.(i)) fields) in
  let keys = Array.map int_of_string (col 0) in
  let x = Col.float "x" in
  let k, w =
    if name = "datetime" then
      let at us = Time.of_ns (Int64.mul 1000L (Int64.of_int us)) in
      ( Column.v (Type.instant Type.Us) (Array.map at keys),
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
  equal (array (option (within 1e-10))) (expect 2) (values (w (Expr.sum x)));
  equal (array (option exact_w)) (expect 3) (values (w (Expr.max x)));
  equal
    (array (option (within 1e-10)))
    (expect 4)
    (values (w ~exclude_key:true (Expr.mean x)))

(* A row whose key is null has no window, so that its reduction does not run
   there. *)
let no_window () =
  let k = Column.of_options Type.int64 [| None; Some 1 |] in
  let t = Talon.v [ ("k", k); ("x", Column.v Type.float64 [| 1.; 2. |]) ] in
  let e =
    Expr.(
      window ~on:(Col.int "k") (cast Type.uint8 (count (Col.float "x") - int 1)))
  in
  equal
    (result (array (option int)) pass)
    (Ok [| None; Some 0 |])
    (Query.values (Expr.option e) (Query.of_table t))

(* A running median gathers n (n + 1) / 2 rows, past 2^20 from n = 1448 on, so
   that it holds them in more than one chunk. *)
let gathered () =
  let n = 1448 in
  let x = Array.init n (fun i -> Float.of_int (i * 7919 mod 1009)) in
  let median i =
    let a = Array.sub x 0 (i + 1) in
    Array.sort Float.compare a;
    let m = Array.length a in
    if m mod 2 = 1 then a.(m / 2) else (a.((m / 2) - 1) +. a.(m / 2)) /. 2.
  in
  let t = Talon.v [ ("x", Column.v Type.float64 x) ] in
  let e = Expr.(window (median (Col.float "x"))) in
  equal (array exact_w) (Array.init n median)
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
           ("ts", Column.v (Type.wall Type.S) ts);
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
           derive (window ~on:(Col.wall "ts") ~before:(Col.span "d") (count x));
           derive (window ~before:(float 1.) (count x));
           derive (window ~before:null (count x));
           derive (window (sum (shift 1 x)));
           derive (window (sum (over (mean x) +. x)));
           derive (window (count (rank x)));
           derive (window (sum (window (sum x))));
           derive (Kit.fill_forward (x -. shift 1 x));
         ])
  @@ __POS_OF__
       {|
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
    |}

(* A data error of a window's reduction names its step and row. *)
let data_error () =
  let t = Talon.v [ ("x", Column.v Type.float64 [| 1.; 1.; 2. |]) ] in
  let e = Expr.(window ~before:(int 2) (only (Col.float "x"))) in
  let q = Query.derive Expr.[ "w" := e ] (Query.of_table t) in
  contains ~sub:"row 2:"
    (Format.asprintf "%a" Error.pp (require_error (Query.run q)))

let () =
  exit
    (run "Window"
       [
         group "Expr.window"
           [
             prop ~count:300 ~examples "is r over the rows its predicate keeps"
               frames filter_law;
             prop "its values do not depend on batches" frames batch_law;
             prop "a reduction's bits are those of keys spaced as places" grid
               grid_law;
             prop ~count:300 ~examples:[ beside_huge ]
               "var and std at large magnitudes agree with two passes" large
               large_law;
             prop ~count:300 "var far from zero is the variance of the spread"
               offset offset_law;
             cases "var and std at large magnitudes"
               ~id:(fun (n, _, _, _, _) -> n)
               large_cases large_case;
             test "agrees with Polars' rolling windows over datetimes"
               (fun () -> oracle "datetime");
             test "agrees with Polars' rolling windows over integers" (fun () ->
                 oracle "int");
             test "a row whose key is null runs no reduction" no_window;
             test "median holds its gathered rows in chunks of 2^20" gathered;
             test "its problems" problems;
             test "a reduction's data error names its row" data_error;
           ];
         group "Kit windows"
           [
             prop "cumulative r is r over the rows up to each" drawn_series
               cumulative_law;
             prop "fill_forward x is x or its last earlier value" drawn_series
               fill_forward_law;
             prop "fill_backward x is x or its next value" drawn_series
               fill_backward_law;
             prop "interpolate x is the line through its neighbours"
               drawn_series interpolate_law;
           ];
       ])
