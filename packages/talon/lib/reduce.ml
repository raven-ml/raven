(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rows [i] of segment [ids.{i}], in the order [order] lists them, row order
   where it is [None]. [first] is each segment's first row, and [sizes] each
   segment's number of rows, [None] for one segment of every row. *)
type segments = {
  ids : Nx.host Nx.int64_t;
  count : int;
  first : Nx.host Nx.int64_t;
  sizes : Nx.host Nx.int64_t option;
  order : Nx.host Nx.int64_t option;
}

let positions n = Nx.arange Nx.int64 0 n 1
let none n = Nx.full Nx.int64 [| n |] (-1L)
let int64 = Type.Any Type.int64
let float64 = Type.Any Type.float64

let one n =
  {
    ids = Nx.zeros Nx.int64 [| n |];
    count = 1;
    first = Nx.zeros Nx.int64 [| 1 |];
    sizes = None;
    order = None;
  }

let of_groups (g : Nx.host Nx.groups) order =
  {
    ids = g.ids;
    count = Nx.dim 0 g.first;
    first = g.first;
    sizes = Some g.counts;
    order;
  }

let group cs = of_groups (Key.groups cs) None

let of_ids ~segments ~sizes ids =
  let first = Nx.sub (Nx.cumsum sizes) sizes in
  { ids; count = segments; first; sizes = Some sizes; order = None }

let count s = s.count
let first s = s.first
let ids s = s.ids

(* [keyed s ws] is the [[n; w]] words [ws] after each row's segment. *)
let keyed s ws =
  let n = Nx.dim 0 s.ids in
  Nx.concatenate ~axis:1 [ Nx.reshape [| n; 1 |] (Nx.cast Nx.uint64 s.ids); ws ]

let refine s ~by ~order =
  let s =
    match by with
    | [] -> s
    | cs -> of_groups (Nx.unique (keyed s (Key.identity cs))) s.order
  in
  match order with
  | [] -> s
  | ks ->
      let k = Key.order ks in
      let order =
        match s.order with
        | None -> Nx.lexsort k
        | Some p -> Nx.take (Nx.lexsort (Nx.take ~axis:0 p k)) p
      in
      { s with order = Some order }

let broadcast s c = Column.gather s.ids c

(* [view s] is the rows of each segment in its order. *)
let view s =
  let p = Option.value s.order ~default:(positions (Nx.dim 0 s.ids)) in
  Nx_ragged.of_ids ~segments:s.count (Nx.take p s.ids) p

(* [fixed ty ?valid x] is the column of [ty] stored as [x], null where [valid]
   is [false]. *)
let fixed ty ?valid x =
  let validity = Option.map (Nx.cast Nx.bit) valid in
  Column.make ty ?validity ~length:(Nx.dim 0 x) (Fixed (P x))

(* [rows_at rows c] is [c]'s rows at [rows], null where a row is [-1]. *)
let rows_at rows c =
  let found = Column.mask (Nx.cast Nx.bit (Nx.greater_equal_s rows 0L)) in
  Column.restrict found (Column.gather rows c)

let tensor dt c =
  match Column.data c with Fixed (P x) -> Nx.cast dt x | _ -> assert false

let segment_sum s ids x = Nx.reduce_segments Nx.Add ~segments:s.count ids x

(* Whether sums of [c]'s values or of their squares can pass max_float, as
   float64's alone do: a 32-bit float squares below 2^256, an int64 below
   2^128. *)
let overflows c =
  match Column.data c with
  | Fixed (P x) ->
      let dt = Nx.dtype x in
      Nx.Dtype.is Float dt && Nx.Dtype.bits dt = 64
  | _ -> false

(* Whether [c]'s values are integers of at most 32 bits, whose int64 sum over
   fewer than [exact_values] of them is exact. *)
let small_ints c =
  match Column.data c with
  | Fixed (P x) -> (
      let dt = Nx.dtype x in
      match Nx.Dtype.kind dt with
      | Signed | Unsigned -> Nx.Dtype.bits dt <= 32
      | _ -> false)
  | _ -> false

let exact_values = 1 lsl 31

let sizes s =
  match s.sizes with
  | Some c -> c
  | None -> Nx.full Nx.int64 [| 1 |] (Int64.of_int (Nx.dim 0 s.ids))

(* [at s] is each row's place in the order of the rows, which keeps each
   segment's rows in its order. *)
let at s =
  let n = Nx.dim 0 s.ids in
  match s.order with
  | None -> positions n
  | Some p -> Nx.scatter ~axis:0 p (positions n) (none n)

(* [pick s op hit] is, in each segment, the row where [hit] holds, or any row
   when it is [None], that comes first ([Nx.Min]) or last ([Nx.Max]) in the
   segment's order, [-1] where no row does: the row of least or greatest
   {!at}. *)
let pick s op hit =
  let n = Nx.dim 0 s.ids in
  let ids =
    match hit with Some h -> Nx.where h s.ids (none n) | None -> s.ids
  in
  let j = Nx.reduce_segments op ~segments:s.count ids (at s) in
  let found =
    Nx.logical_and (Nx.greater_equal_s j 0L) (Nx.less_s j (Int64.of_int n))
  in
  let row = match s.order with None -> j | Some p -> Nx.take j p in
  Nx.where found row (none s.count)

(* [place s rows] is the place of each segment's row [rows], or [-1], among the
   segment's rows in its order: the number of them before it. *)
let place s rows =
  let at = at s in
  let mark = Nx.take rows at in
  let before = Nx.less at (Nx.take s.ids mark) in
  let p = segment_sum s s.ids (Nx.cast Nx.int64 before) in
  Nx.where (Nx.greater_equal_s rows 0L) p (none s.count)

(* [first_where m why] is the first segment where [m] holds, failing for the
   reason [why]. *)
let first_where m why =
  let gs = Nx.positions m in
  if Nx.dim 0 gs = 0 then None else Some (Int64.to_int (Nx.item [ 0 ] gs), why)

(* [exact_sum ty sum x] sums by [sum] the [int64] values [x] exactly, and is
   null where a sum leaves [int64]. A value is three limbs of 21 bits, the top
   one signed, whose sums are exact over fewer than 2^42 values: two limbs of
   32 bits sum exactly over fewer than 2^31 only, which a frame can exceed. *)
let limb = 21

let exact_sum ty sum x =
  let split x =
    let hi = Nx.rshift x limb in
    (hi, Nx.sub x (Nx.lshift hi limb))
  in
  let mid, c = split x in
  let a, b = split mid in
  let carry, c = split (sum c) in
  let carry, b = split (Nx.add (sum b) carry) in
  let a = Nx.add (sum a) carry in
  let top = Int64.shift_left 1L (63 - (2 * limb)) in
  let fits =
    Nx.logical_and (Nx.greater_equal_s a (Int64.neg top)) (Nx.less_s a top)
  in
  let sum = Nx.add (Nx.lshift a (2 * limb)) (Nx.add (Nx.lshift b limb) c) in
  let why = Format.asprintf "the sum overflows %a" Type.pp ty in
  (fixed (Any ty) ~valid:fits sum, first_where (Nx.logical_not fits) why)

(* Frames and parts

   A reduction reads its frames through a [frame], so that one table states
   each reduction for segments and for windows alike, and a column's
   reductions over one frame share its [parts]: the frame's primitives over
   the rows that hold values, the values' counts, int64 and float64 sums, mean
   and variance, order words and their extremes' holders, each computed when
   a reduction first reads it. Parts are fields, where lazy values would
   allocate a closure each. *)

type frame = {
  sizes : unit -> Nx.host Nx.int64_t;
  values : Nx.host Nx.bool_t option -> values;
  place : Nx.host Nx.int64_t -> Nx.host Nx.int64_t;
}

and values = {
  fold :
    'v 'e. Nx.combine -> ('v, 'e, Nx.host) Nx.t -> ('v, 'e, Nx.host) Nx.t;
  pick : Nx.combine -> Nx.host Nx.int64_t;
  extreme : desc:bool -> Nx.host Nx.uint64_t -> Nx.host Nx.int64_t;
  variance : (parts -> moments) option;
  holistic :
    'a 'b.
    ('a, 'b) Expr.reduction ->
    Type.any ->
    parts ->
    Column.t * (int * string) option;
}

and moments = {
  var : Nx.host Nx.float64_t;
  up : Nx.host Nx.float64_t option;
  n : Nx.host Nx.int64_t;
}

and parts = {
  frame : frame;
  c : Column.t;
  held : Nx.host Nx.bool_t option;
  v : values;
  mutable counts : Nx.host Nx.int64_t option;
  mutable sum : Nx.host Nx.int64_t option;
  mutable fsum : Nx.host Nx.float64_t option;
  mutable mean : Nx.host Nx.float64_t option;
  mutable moments : moments option;
  mutable words : Nx.host Nx.uint64_t option;
  mutable least : Nx.host Nx.int64_t option;
  mutable greatest : Nx.host Nx.int64_t option;
}

let parts frame c =
  let held = Option.map (Nx.cast Nx.bool) (Column.validity c) in
  {
    frame;
    c;
    held;
    v = frame.values held;
    counts = None;
    sum = None;
    fsum = None;
    mean = None;
    moments = None;
    words = None;
    least = None;
    greatest = None;
  }

let column p = p.c

(* [memo get set f p] is [f p], computed once and kept in [p]'s field. *)
let memo get set f p =
  match get p with
  | Some x -> x
  | None ->
      let x = f p in
      set p (Some x);
      x

let counts =
  memo (fun p -> p.counts) (fun p x -> p.counts <- x) (fun p ->
      match p.held with
      | None -> p.frame.sizes ()
      | Some _ -> p.v.fold Add (Nx.ones Nx.int64 [| Column.length p.c |]))

let sum =
  memo (fun p -> p.sum) (fun p x -> p.sum <- x) (fun p ->
      p.v.fold Add (tensor Nx.int64 p.c))

let fsum =
  memo (fun p -> p.fsum) (fun p x -> p.fsum <- x) (fun p ->
      p.v.fold Add (tensor Nx.float64 p.c))

(* Integers of at most 32 bits sum exactly in int64 over a frame of fewer than
   2^31 values, then round once to float64, as a float64 sum does wherever its
   partial sums are exact. The test is each frame's, since a batch holds many
   frames: a frame of more values sums in float64. *)
let mean =
  memo (fun p -> p.mean) (fun p x -> p.mean <- x) (fun p ->
      let n = counts p in
      let sum =
        if not (small_ints p.c) then fsum p
        else
          let wide = Nx.greater_equal_s n (Int64.of_int exact_values) in
          let exact = Nx.cast Nx.float64 (sum p) in
          if Nx.item [] (Nx.any wide) then Nx.where wide (fsum p) exact
          else exact
      in
      Nx.div sum (Nx.cast Nx.float64 n))

let moments variance =
  memo (fun p -> p.moments) (fun p x -> p.moments <- x) variance

(* Scales

   A frame whose greatest finite magnitude lies in [\[2^t, 2^(t + 1))] takes
   [down = 2^-e]: [e] the exponent [t] past [below - 1], so that its values
   times [down] are below [2^below], or [t + above] for a frame below
   [2^-above], so that their squares are far above the range's bottom, or
   [0]. *)

let below = 480
let above = 432
let bias = 1023L
let tiny = 0x1p-969

let scales max x =
  let a = Nx.where (Nx.isfinite x) (Nx.abs x) (Nx.zeros_like x) in
  let top = Nx.maximum_s (max a) 0. in
  let t = Nx.sub_s (Nx.rshift (Nx.bitcast Nx.int64 top) 52) bias in
  let high = Nx.maximum_s (Nx.sub_s t (Int64.of_int (below - 1))) 0L in
  (* A zero or subnormal greatest magnitude has exponent field 0, [t = -1023],
     and takes the least normal's scale. *)
  let low =
    Nx.minimum_s (Nx.add_s (Nx.maximum_s t (-1022L)) (Int64.of_int above)) 0L
  in
  let e = Nx.add high low in
  let power k = Nx.bitcast Nx.float64 (Nx.lshift (Nx.add_s k bias) 52) in
  (power e, power (Nx.neg e))

let helped var down =
  Nx.logical_or
    (Nx.logical_and (Nx.logical_not (Nx.isfinite var)) (Nx.less_s down 1.))
    (Nx.logical_and (Nx.less_s var tiny) (Nx.greater_s down 1.))

let words =
  memo (fun p -> p.words) (fun p x -> p.words <- x) (fun p ->
      Key.value Order p.c)

(* The holders of each frame's first least and greatest words. *)
let least =
  memo (fun p -> p.least) (fun p x -> p.least <- x) (fun p ->
      p.v.extreme ~desc:false (words p))

let greatest =
  memo (fun p -> p.greatest) (fun p x -> p.greatest <- x) (fun p ->
      p.v.extreme ~desc:true (words p))

(* Whether [c]'s extremes are its values': an integer's or a boolean's, one
   value to an order word, ordered as its storage. *)
let direct c =
  let (Type.Any ty) = Column.type_ c in
  match Type.kind ty with Kind.Int | Kind.Bool -> true | _ -> false

let reduce : type a b.
    (a, b) Expr.reduction ->
    Type.any ->
    parts ->
    Column.t * (int * string) option =
 fun r (Any rty as ty) p ->
  let { frame; c; v; _ } = p in
  let at_least k = Nx.greater_equal_s (counts p) (Int64.of_int k) in
  let extreme desc = if desc then greatest p else least p in
  (* One fold of the values, where it is an extreme's value. *)
  let value_at desc =
    match Column.data c with
    | Fixed (P x) when direct c ->
        fixed ty ~valid:(at_least 1) (v.fold (if desc then Max else Min) x)
    | _ -> rows_at (extreme desc) c
  in
  let place row =
    let p = frame.place row in
    fixed int64 ~valid:(Nx.greater_equal_s p 0L) p
  in
  let value row = rows_at row c in
  let ok c = (c, None) in
  match r with
  | Count -> ok (fixed int64 (counts p))
  | Sum -> (
      match (rty, Column.data c) with
      | Duration _, _ -> exact_sum rty (v.fold Add) (tensor Nx.int64 c)
      | Int64, _ -> ok (fixed ty (sum p))
      | Float64, _ -> ok (fixed ty (fsum p))
      | _, Fixed (P x) -> ok (fixed ty (v.fold Add x))
      | _ -> invalid_arg "Reduce.reduce: no sum")
  | Mean -> ok (fixed ty ~valid:(at_least 1) (mean p))
  | Var | Std -> (
      match v.variance with
      | None -> v.holistic r ty p
      | Some variance ->
          let { var = x; up; n } = moments variance p in
          let x =
            match (r, up) with
            | Std, None -> Nx.sqrt x
            | Std, Some up -> Nx.mul (Nx.sqrt x) up
            | _, None -> x
            | _, Some up -> Nx.mul (Nx.mul x up) up
          in
          ok (fixed ty ~valid:(Nx.greater_equal_s n 2L) x))
  | Median | Quantile _ | N_unique | Collect -> v.holistic r ty p
  | Min -> ok (value_at false)
  | Max -> ok (value_at true)
  | Arg_min -> ok (place (extreme false))
  | Arg_max -> ok (place (extreme true))
  | First -> ok (value (v.pick Min))
  | Last -> ok (value (v.pick Max))
  | Only ->
      let w = words p in
      let differ = Nx.not_equal (v.fold Min w) (v.fold Max w) in
      let several = Nx.logical_and (at_least 1) differ in
      (value (extreme false), first_where several "only finds several values")

(* [frame s] reads the segments [s]: a row without a value leaves its
   segment, and var and std take the deviations from each segment's mean. A
   variance that overflows is taken again of the values scaled ({!scales}),
   since a sum of the values or of their squared deviations can pass
   max_float where the variance does not. *)
let frame s =
  let n = Nx.dim 0 s.ids in
  let values held =
    let ids =
      match held with Some v -> Nx.where v s.ids (none n) | None -> s.ids
    in
    let fold op x = Nx.reduce_segments op ~segments:s.count ids x in
    let extreme ~desc words =
      let e = fold (if desc then Max else Min) words in
      let hit = Nx.equal words (Nx.take s.ids e) in
      let hit = match held with Some v -> Nx.logical_and v hit | None -> hit in
      pick s Nx.Min (Some hit)
    in
    let variance p =
      let n = counts p and x = tensor Nx.float64 p.c in
      let k = Nx.cast Nx.float64 (Nx.sub_s n 1L) in
      let deviations x m =
        let d = Nx.sub x (Nx.take s.ids m) in
        Nx.div (fold Add (Nx.mul d d)) k
      in
      let var = deviations x (mean p) in
      if not (overflows p.c) then { var; up = None; n }
      else
        let odd =
          Nx.logical_or (Nx.equal_s var Float.infinity) (Nx.less_s var tiny)
        in
        if not (Nx.item [] (Nx.any odd)) then { var; up = None; n }
        else
          let up, down = scales (fold Max) x in
          let redo = helped var down in
          if not (Nx.item [] (Nx.any redo)) then { var; up = None; n }
          else
            (* Equal values have variance 0: a rounded mean's one deviation,
               scaled into range, must not show. *)
            let equal = Nx.equal (fold Min x) (fold Max x) in
            let same = Nx.logical_and redo equal in
            let redo = Nx.logical_and redo (Nx.logical_not same) in
            let x = Nx.mul x (Nx.take s.ids down) in
            let m = Nx.div (fold Add x) (Nx.cast Nx.float64 n) in
            let var = Nx.where redo (deviations x m) var in
            let var = Nx.where same (Nx.zeros_like var) var in
            { var; up = Some (Nx.where redo up (Nx.ones_like up)); n }
    in
    let holistic : type a b.
        (a, b) Expr.reduction ->
        Type.any ->
        parts ->
        Column.t * (int * string) option =
     fun r ty p ->
      let c = p.c in
      let quantile q =
        let r = Nx_ragged.of_ids ~segments:s.count ids (tensor Nx.float64 c) in
        let q = Nx.reshape [| s.count |] (Nx_ragged.quantile [| q |] r) in
        (fixed float64 ~valid:(Nx.greater_equal_s (counts p) 1L) q, None)
      in
      match r with
      | Median -> quantile 0.5
      | Quantile q -> quantile q
      | Collect ->
          let r = view s in
          ( Column.make ty ~length:s.count
              (List
                 {
                   offsets = Nx_ragged.offsets r;
                   child = Column.gather (Nx_ragged.values r) c;
                 }),
            None )
      | N_unique ->
          (* [n] groups at most, so that no count of them is read: a group
             past the last has no key, and counts [0] in segment [0]. *)
          let g = Nx.unique ~size:n (keyed s (Key.identity [ c ])) in
          let real = Nx.cast Nx.int64 (Nx.greater_s g.counts 0L) in
          (fixed int64 (segment_sum s (Nx.take g.first s.ids) real), None)
      | _ -> invalid_arg "Reduce: not a holistic reduction"
    in
    {
      fold;
      pick = (fun op -> pick s op held);
      extreme;
      variance = Some variance;
      holistic;
    }
  in
  { sizes = (fun () -> sizes s); values; place = place s }

let shift s k c =
  let r = view s and n = Column.length c in
  let v = Nx_ragged.values r and offsets = Nx_ragged.offsets r in
  let segment = Nx.take v s.ids in
  let start = Nx.take segment offsets in
  let stop = Nx.take (Nx.add_s segment 1L) offsets in
  let j = Nx.sub_s (positions n) (Int64.of_int k) in
  let inside = Nx.logical_and (Nx.greater_equal j start) (Nx.less j stop) in
  let from = Nx.where inside (Nx.take j v) (none n) in
  rows_at (Nx.scatter ~axis:0 v from (none n)) c

(* A row's rank is its place in its segment's rows sorted by value, at the start
   of its run of equal values: [1] plus the number of rows before the run, which
   are values, nulls sorting last. *)
let rank s c =
  let n = Column.length c in
  if n = 0 then fixed int64 (Nx.zeros Nx.int64 [| 0 |])
  else
    let k = keyed s (Key.order [ (c, Order.asc "") ]) in
    let p = Nx.lexsort k in
    let k = Nx.take ~axis:0 p k in
    let w = Nx.dim 1 k and i = positions n in
    let differs =
      Nx.not_equal
        (Nx.slice [ Nx.R (1, n); Nx.R (0, w) ] k)
        (Nx.slice [ Nx.R (0, n - 1); Nx.R (0, w) ] k)
    in
    let start changed =
      let b = Nx.concatenate ~axis:0 [ Nx.ones Nx.bool [| 1 |]; changed ] in
      Nx.cummax (Nx.where b i (Nx.zeros_like i))
    in
    let run = start (Nx.any ~axes:[ 1 ] differs) in
    let segment =
      start
        (Nx.reshape
           [| n - 1 |]
           (Nx.slice [ Nx.R (0, n - 1); Nx.R (0, 1) ] differs))
    in
    let r = Nx.add_s (Nx.sub run segment) 1L in
    let ranks = Nx.scatter ~axis:0 p r (Nx.zeros Nx.int64 [| n |]) in
    Column.with_data int64 (Fixed (P ranks)) c
