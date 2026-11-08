(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rows sit in {e sorted} order: by segment, then the rows of a key by its
   word, ties in their segment's order, then the rows of a null key. Each row
   with a window has one, numbered in row order, and window [w] is two ranges
   of sorted positions, [a] then [b]. When it excludes its key, the rows of
   the key lie between them; otherwise [b] is empty. Without [before], [a]
   starts its segment, and where [b] is then empty for every window, the
   windows are [running]: a fold over one is a prefix fold, which one scan
   computes for all of them.

   {v
   sorted:  | segment 0 ............................ | segment 1 ...
                 a_lo       a_hi  key i  b_lo   b_hi
   v} *)

type t = {
  sorted : Nx.host Nx.int64_t;  (* The row at each sorted position. *)
  position : Nx.host Nx.int64_t Lazy.t;  (* The sorted position of each row. *)
  starts : Nx.host Nx.bool_t;  (* Where a segment starts, by position. *)
  rows : Nx.host Nx.int64_t;  (* The row of each window. *)
  origins : int64 array Lazy.t;  (* [rows], read when a failure needs it. *)
  window : Nx.host Nx.int64_t;  (* The window of each row, [-1] for none. *)
  a_lo : Nx.host Nx.int64_t;
  a_hi : Nx.host Nx.int64_t;
  b_lo : Nx.host Nx.int64_t;
  b_hi : Nx.host Nx.int64_t;
  split : bool;  (* Whether [b] may hold rows. *)
  running : bool;
}

let int64 = Type.Any Type.int64

let tensor dt c =
  match Column.data c with Fixed (P x) -> Nx.cast dt x | _ -> assert false

let held c =
  match Column.validity c with
  | Some v -> Nx.cast Nx.bool v
  | None -> Nx.ones Nx.bool [| Column.length c |]

let positions n = Nx.arange Nx.int64 0 n 1
let full n v = Nx.full Nx.int64 [| n |] v

(* Keys and bounds

   A bound places a row's window between the words of [k - before] and
   [k + after], computed in the type where the key and its bounds meet. An
   integer result that wraps falls [below] or [above] every key, so that the
   bound is exact wherever it lands; a float rounds once, and an infinity or
   NaN orders where it falls. *)

type edge = {
  word : Nx.host Nx.uint64_t;
  below : Nx.host Nx.bool_t;
  above : Nx.host Nx.bool_t;
}

let edges (type a b) ty (k : (a, b, Nx.host) Nx.t) before after =
  let word x = Key.value Order (Reduce.fixed ty x) in
  let edge ~down (d : (a, b, Nx.host) Nx.t) =
    let r = if down then Nx.sub k d else Nx.add k d in
    let z = Nx.zeros_like d in
    let lower, higher =
      if down then (Nx.greater d z, Nx.less d z)
      else (Nx.less d z, Nx.greater d z)
    in
    {
      word = word r;
      below = Nx.logical_and lower (Nx.greater r k);
      above = Nx.logical_and higher (Nx.less r k);
    }
  in
  let lo = Option.map (edge ~down:true) before in
  (word k, lo, Option.map (edge ~down:false) after)

let unit_of : type a. a Type.t -> Type.unit_ option = function
  | Datetime { unit_; _ } | Duration unit_ | Clock unit_ -> Some unit_
  | _ -> None

(* A duration of a coarser unit than the key's may span 2^63 ticks or more.
   [ticks per b] is the duration [b] in ticks of [per] nanoseconds, modulo
   2^64, with where it spans 2^64 ticks or more ([far]) or from 2^63 ([wide]),
   its sign and its ticks' magnitude. *)
let ticks per b =
  let (Type.Any ty) = Column.type_ b in
  let f = Int64.div (Type.ns_per_unit (Option.get (unit_of ty))) per in
  let x = tensor Nx.int64 b in
  let size = Nx.bitcast Nx.uint64 (Nx.where (Nx.less_s x 0L) (Nx.neg x) x) in
  let far = Nx.greater_s size (Int64.unsigned_div (-1L) f) in
  let wide = Nx.greater_s size (Int64.div Int64.max_int f) in
  let d = Nx.mul_s x f in
  let mag = Nx.bitcast Nx.uint64 (Nx.where (Nx.less_s x 0L) (Nx.neg d) d) in
  (d, (far, wide, Nx.greater_s x 0L, mag))

(* [beyond ~down k b e] is the edge [e] of the int64 keys [k] moved by the
   bound [b] of {!ticks}: a move of 2^63 ticks or more fits where the key has
   that room below it, [k - min_int], or above it, [max_int - k]. *)
let beyond ~down k (far, wide, positive, mag) e =
  let sign = Nx.full_like k Int64.min_int in
  let room = Nx.bitcast Nx.uint64 (Nx.bitwise_xor k sign) in
  let falls = if down then positive else Nx.logical_not positive in
  let short room = Nx.logical_or far (Nx.less room mag) in
  let past = Nx.logical_or far wide in
  let below = Nx.logical_and falls (short room) in
  let rises = Nx.logical_not falls in
  let above = Nx.logical_and rises (short (Nx.bitwise_not room)) in
  let below = Nx.where past below e.below in
  { e with below; above = Nx.where past above e.above }

(* [key on before after] is each row's key word and the edges of its bounds:
   along a temporal [on] in int64 ticks (a date's in seconds), along a number at
   the type that it and its bounds meet. *)
let key on before after =
  let (Type.Any kty as t) = Column.type_ on in
  match kty with
  | Date | Datetime _ | Duration _ | Clock _ ->
      let per, k =
        match kty with
        | Date -> (1_000_000_000L, Nx.mul_s (tensor Nx.int64 on) 86_400L)
        | _ ->
            (Type.ns_per_unit (Option.get (unit_of kty)), tensor Nx.int64 on)
      in
      let before = Option.map (ticks per) before in
      let after = Option.map (ticks per) after in
      let word, lo, hi =
        edges int64 k (Option.map fst before) (Option.map fst after)
      in
      let moved ~down b e = beyond ~down k (snd (Option.get b)) e in
      let lo = Option.map (moved ~down:true before) lo in
      (word, lo, Option.map (moved ~down:false after) hi)
  | _ -> (
      let bounds = List.filter_map Fun.id [ before; after ] in
      let ty =
        Option.get (Type.common_any (t :: List.map Column.type_ bounds))
      in
      let at c = fst (Kernels.cast (Column.type_ c) ty c) in
      match Column.data (at on) with
      | Fixed (P k) ->
          let bound c =
            match Column.data (at c) with
            | Fixed p -> Nx.unpack (Nx.dtype k) p
            | _ -> assert false
          in
          edges ty k (Option.map bound before) (Option.map bound after)
      | _ -> assert false)

(* Windows

   Each way of placing rows gives them in sorted order, each row's range
   [\[lo, hi)] of the rows of its window's predicate, and the range of the
   rows of its key, which [~exclude] takes out. *)

(* Places compute their ranges: row [i] at sorted position [q] has the place
   [q - s0] in its segment [\[s0, s1)], and a bound clipped to [n + 1] places
   past the segment reaches its end, exactly. *)
let placed view ids before after =
  let rows = Nx_ragged.values view and offsets = Nx_ragged.offsets view in
  let n = Nx.dim 0 rows in
  let q = Nx.scatter ~axis:0 rows (positions n) (full n 0L) in
  let s0 = Nx.take ids offsets and s1 = Nx.take (Nx.add_s ids 1L) offsets in
  let most = Int64.of_int (n + 1) in
  let bound c = Nx.clamp ~min:(Int64.neg most) ~max:most (tensor Nx.int64 c) in
  let past d = Nx.minimum (Nx.maximum (Nx.add_s (Nx.add q d) 1L) s0) s1 in
  let next = Nx.add_s q 1L in
  let lo b = past (Nx.neg (bound b)) in
  let lo = Option.fold ~none:s0 ~some:lo before in
  let hi = Option.fold ~none:next ~some:(fun a -> past (bound a)) after in
  (rows, lo, hi, lazy (q, next))

(* Keys search their ranges in the rows sorted by segment, then [1] for a key
   or [2] for a null key, then the key's word, a column of the sort key kept
   only where it varies. A key's rows of segment [\[s0, s1)] end at [e1], the
   null keys after them. *)
let searched view ~segments ids on before after =
  let rows = Nx_ragged.values view and offsets = Nx_ragged.offsets view in
  let n = Nx.dim 0 rows in
  let word, lo, hi = key on before after in
  let held = held on in
  let s0 = Nx.take ids offsets in
  let keys = Nx.reduce_segments Add ~segments ids (Nx.cast Nx.int64 held) in
  let e1 = Nx.add s0 (Nx.take ids keys) in
  let one = Nx.ones Nx.uint64 [| n |] in
  let flag = Nx.where held one (Nx.full Nx.uint64 [| n |] 2L) in
  let sort_key flag w =
    let seg = if segments > 1 then [ Nx.cast Nx.uint64 ids ] else [] in
    let null = if Option.is_some (Column.validity on) then [ flag ] else [] in
    match seg @ null @ [ w ] with [ w ] -> w | cs -> Nx.stack ~axis:1 cs
  in
  let rank = sort_key flag word in
  let sorted = Nx.take (Nx.lexsort (Nx.take ~axis:0 rows rank)) rows in
  let s = Nx.take ~axis:0 sorted rank in
  let find side w = Nx.searchsorted ~side s (sort_key one w) in
  let edge e = Nx.where e.below s0 (Nx.where e.above e1 (find `Right e.word)) in
  let lo = Option.fold ~none:s0 ~some:edge lo in
  let hi = Option.fold ~none:(find `Right word) ~some:edge hi in
  (sorted, lo, hi, lazy (find `Left word, find `Right word))

let v s ~on ~before ~after ~exclude =
  let view = Reduce.view s and ids = Reduce.ids s in
  let n = Nx.dim 0 ids in
  let nullable =
    List.filter
      (fun c -> Option.is_some (Column.validity c))
      (List.filter_map Fun.id [ on; before; after ])
  in
  let valid =
    let both v c = Nx.logical_and v (held c) in
    List.fold_left both (Nx.ones Nx.bool [| n |]) nullable
  in
  let sorted, a_lo, hi, own =
    match on with
    | None -> placed view ids before after
    | Some on -> searched view ~segments:(Reduce.count s) ids on before after
  in
  let a_hi = Nx.maximum a_lo hi in
  let a_hi, b_lo =
    if not exclude then (a_hi, a_hi)
    else
      let k_lo, k_hi = Lazy.force own in
      ( Nx.maximum a_lo (Nx.minimum a_hi k_lo),
        Nx.minimum a_hi (Nx.maximum a_lo k_hi) )
  and b_hi = a_hi in
  (* The windows of the rows that have one, in row order. *)
  let rows = if nullable = [] then positions n else Nx.positions valid in
  let windows = Nx.dim 0 rows in
  let compact x = if nullable = [] then x else Nx.take rows x in
  let starts =
    if n = 0 then Nx.zeros Nx.bool [| 0 |]
    else
      let g = Nx.take sorted ids in
      Nx.concatenate ~axis:0
        [
          Nx.ones Nx.bool [| 1 |];
          Nx.not_equal
            (Nx.slice [ Nx.R (1, n) ] g)
            (Nx.slice [ Nx.R (0, n - 1) ] g);
        ]
  in
  let split = exclude && Option.is_some after in
  {
    sorted;
    position = lazy (Nx.scatter ~axis:0 sorted (positions n) (full n 0L));
    starts;
    rows;
    origins = lazy (Nx.to_array rows);
    window =
      (if nullable = [] then rows
       else Nx.scatter ~axis:0 rows (positions windows) (full n (-1L)));
    a_lo = compact a_lo;
    a_hi = compact a_hi;
    b_lo = compact b_lo;
    b_hi = compact b_hi;
    split;
    running = Option.is_none before && not split;
  }

let windows w = Nx.dim 0 w.rows
let row w i = Int64.to_int (Lazy.force w.origins).(i)

let sizes w =
  Nx.add (Nx.sub w.a_hi w.a_lo) (Nx.sub w.b_hi w.b_lo)

let to_rows w c = Reduce.rows_at w.window c

(* Folds

   [fold w op x] folds the values [x], one per sorted position, over each
   window: running windows read one segmented scan at each window's last
   position, others fold their ranges with [Nx.reduce_ranges], with its bits:
   a float format narrower than 32 bits sums at [float32] and rounds once, and
   a sum of [-0.] is [+0.]. *)

let combine : Nx.combine -> _ = function
  | Add -> Nx.add
  | Min -> Nx.minimum
  | Max -> Nx.maximum
  | Set -> assert false

(* [scan w s f x] is the scan of [f] over each segment, [x] a structure that
   [s] describes, by the segmented lift of [Nx.associative_scan]. *)
let scan w s f x =
  let lift (s1, a) (s2, b) =
    let restart _ a b = Nx.where s2 b a in
    (Nx.logical_or s1 s2, Nx.Ptree.map2 s restart (f a b) b)
  in
  snd (Nx.associative_scan Nx.Ptree.(pair tensor s) lift (w.starts, x))

(* [last w empty y] is, for each window, the scan [y] at its last position,
   and [empty] where it is empty. *)
let last w empty y =
  let at = Nx.maximum_s (Nx.sub_s w.a_hi 1L) 0L in
  Nx.where (Nx.greater w.a_hi w.a_lo) (Nx.take at y) empty

(* [fold w op ~held x] folds the values [x], in sorted order, at the sorted
   positions where [held]. *)
let fold : type v e.
    t ->
    Nx.combine ->
    held:Nx.host Nx.bool_t option ->
    (v, e, Nx.host) Nx.t ->
    (v, e, Nx.host) Nx.t =
 fun w op ~held x ->
  let empty = Nx.zeros Nx.int64 [| 1 |] in
  let fold : type a b. (a, b, Nx.host) Nx.t -> (a, b, Nx.host) Nx.t =
   fun x ->
    (* The identity of [op], from an empty range of one row. *)
    let head = Nx.slice [ Nx.R (0, Int.min 1 (Nx.dim 0 x)) ] x in
    let none = Nx.reduce_ranges op ~lo:empty ~hi:empty head in
    let x = Option.fold ~none:x ~some:(fun v -> Nx.where v x none) held in
    if Nx.dim 0 x = 0 then x
    else if w.running then
      let y = last w none (scan w Nx.Ptree.tensor (combine op) x) in
      match op with Add -> Nx.add y (Nx.zeros_like y) | _ -> y
    else
      let range lo hi = Nx.reduce_ranges op ~lo ~hi x in
      let a = range w.a_lo w.a_hi in
      if w.split then combine op a (range w.b_lo w.b_hi) else a
  in
  let dt = Nx.dtype x in
  if Nx.Dtype.is Float dt && Nx.Dtype.bits dt < 32 then
    Nx.cast dt (fold (Nx.cast Nx.float32 x))
  else fold x

(* Moments

   A window's values are one range of the held values, the range's bounds the
   counts of held positions before the window's. [Nx.moments_ranges] gives
   each range's mean and variance over [k] values; the sample variance is that
   times [k / (k - 1)].

   A variance past max_float is infinite, where its root need not be: the
   spread of float64 values is below 2^1025, so that a variance is below
   2^2049 and the values times [down] have one below 2^1009. A value that
   [down] takes below 2^-1022 rounds by at most 2^-1075, invisible beside the
   root of those windows, at least 2^-8 at that scale.

   A variance below [Reduce.tiny] can be a square's underflow. Unequal values
   with so small a variance have a spread below 2^-484 sqrt (2 k), and being
   apart by at least their ulp, lie below 2^-431 sqrt (2 k), under [low] for
   any [k] up to 2^53. Those values times [lift] have squares far above the
   range's bottom, even the least subnormal's at 2^-968. Windows of zeros and
   of equal large values keep their variance of exactly 0. *)

let down = Float.ldexp 1. (-520)
let lift = Float.ldexp 1. 590
let low = Float.ldexp 1. (-400)
let nonzero_low top = Nx.logical_and (Nx.greater_s top 0.) (Nx.less_s top low)

(* [ranged w held x] is each window's sample variance of the values [x] at the
   sorted positions where [held]. *)
let ranged w held x : Reduce.moments =
  let x, lo, hi =
    match held with
    | None -> (x, w.a_lo, w.a_hi)
    | Some h ->
        let before =
          Nx.concatenate ~axis:0
            [ Nx.zeros Nx.int64 [| 1 |]; Nx.cumsum (Nx.cast Nx.int64 h) ]
        in
        let bound b = Nx.take b before in
        (Nx.take (Nx.positions h) x, bound w.a_lo, bound w.a_hi)
  in
  let n = Nx.sub hi lo in
  let sample x =
    let k = Nx.cast Nx.float64 n in
    let _, var = Nx.moments_ranges ~lo ~hi x in
    Nx.mul var (Nx.div k (Nx.sub_s k 1.))
  in
  let var = sample x in
  let past = Nx.equal_s var Float.infinity in
  let small = Nx.less_s var Reduce.tiny in
  let any b = Nx.item [] (Nx.any b) in
  let small =
    if not (any small) then small
    else
      let top = Nx.reduce_ranges Max ~lo ~hi (Nx.abs x) in
      Nx.logical_and small (nonzero_low top)
  in
  if not (any past || any small) then { var; up = None; n }
  else
    let again which scale (var, up) =
      if not (any which) then (var, up)
      else
        let v = sample (Nx.mul_s x scale) in
        (Nx.where which v var, Nx.where which (Nx.full_like up (1. /. scale)) up)
    in
    let var, up =
      (var, Nx.ones_like var) |> again past down |> again small lift
    in
    { var; up = Some up; n }

(* [running w held x] is each running window's sample variance of the values
   [x] at the sorted positions where [held], and their number, from their
   count, mean and sum of squared deviations, combined by Chan, Golub and
   LeVeque's update of partial moments in one scan. A part without values
   leaves the other as it is: its mean, [0], is no value of the window, and no
   deviation is taken from it. An infinite or NaN value makes the mean so, and
   the variance NaN, as the deviations from the mean are then. *)
let running w held x =
  let len = Nx.dim 0 x in
  let held =
    match held with Some h -> h | None -> Nx.ones Nx.bool [| len |]
  in
  let zero = Nx.zeros_like x in
  let chan (n1, m1, q1) (n2, m2, q2) =
    let n = Nx.add n1 n2 in
    let r = Nx.div n2 n in
    let d = Nx.sub m2 m1 in
    let q = Nx.add (Nx.add q1 q2) (Nx.mul (Nx.mul d d) (Nx.mul n1 r)) in
    let m = Nx.add m1 (Nx.mul d r) in
    let keep_a = Nx.equal_s n2 0. and keep_b = Nx.equal_s n1 0. in
    let pick a b c = Nx.where keep_a a (Nx.where keep_b b c) in
    (n, pick m1 m2 m, pick q1 q2 q)
  in
  let s = Nx.Ptree.(triple tensor tensor tensor) in
  let at empty y = last w (Nx.full Nx.float64 [| 1 |] empty) y in
  let x = (Nx.cast Nx.float64 held, Nx.where held x zero, zero) in
  let n, m, q = scan w s chan x in
  let n = at 0. n in
  let v = Nx.div (at 0. q) (Nx.sub_s n 1.) in
  (Nx.where (Nx.isfinite (at 0. m)) v (Nx.full_like v Float.nan), n)

(* [variance w held c] is each window's sample variance of [c]'s values at the
   sorted positions where [held]: one scan for running windows, the ranges
   otherwise. The windows of two values or more, all finite, whose scan
   overflowed where their variance need not, or whose variance is below
   [Reduce.tiny] with values as small as [low], take the ranges too: the scan
   costs less, and the ranges overflow only where the variance does. *)
let variance w held c =
  let x = Nx.take w.sorted (tensor Nx.float64 c) in
  if not w.running then ranged w held x
  else
    let var, n = running w held x in
    let n = Nx.cast Nx.int64 n in
    let infinite = Nx.logical_not (Nx.isfinite var) in
    let suspect = Nx.logical_or infinite (Nx.less_s var Reduce.tiny) in
    let odd = Nx.logical_and (Nx.greater_equal_s n 2L) suspect in
    let unscaled = { Reduce.var; up = None; n } in
    if not (Reduce.overflows c && Nx.item [] (Nx.any odd)) then unscaled
    else
      let bad = Nx.cast Nx.int64 (Nx.logical_not (Nx.isfinite x)) in
      let clean = Nx.equal_s (fold w Add ~held bad) 0L in
      let top = fold w Max ~held (Nx.abs x) in
      let reachable = Nx.logical_or infinite (nonzero_low top) in
      let redo = Nx.logical_and (Nx.logical_and odd clean) reachable in
      if not (Nx.item [] (Nx.any redo)) then unscaled
      else
        let r = ranged w held x in
        let only up = Nx.where redo up (Nx.ones_like up) in
        let up = Option.map only r.up in
        { var = Nx.where redo r.var var; up; n }

(* Gathers

   A holistic reduction reduces each window's rows gathered, in its order. The
   windows go in chunks of at most [chunk] gathered rows, or one window, so
   that the rows gathered at once are at most [chunk] or one window's, which
   bounds the memory of a gather to tens of megabytes. *)

let chunk = 1 lsl 20

let gather (type a b) (r : (a, b) Expr.reduction) ty w p =
  let c = Reduce.column p in
  let gathered i0 i1 =
    let part x = Nx.slice [ Nx.R (i0, i1) ] x in
    let a_lo = part w.a_lo and b_lo = part w.b_lo in
    let la = Nx.sub (part w.a_hi) a_lo in
    let sizes = Nx.add la (Nx.sub (part w.b_hi) b_lo) in
    let ids = Nx.positions sizes in
    let start = Nx.sub (Nx.cumsum sizes) sizes in
    let k = Nx.sub (positions (Nx.dim 0 ids)) (Nx.take ids start) in
    let la = Nx.take ids la in
    let p =
      Nx.where (Nx.less k la)
        (Nx.add (Nx.take ids a_lo) k)
        (Nx.add (Nx.take ids b_lo) (Nx.sub k la))
    in
    let s = Reduce.of_ids ~segments:(i1 - i0) ~sizes ids in
    let c = Column.gather (Nx.take p w.sorted) c in
    let c, f = Reduce.reduce r ty (Reduce.parts (Reduce.frame s) c) in
    (c, Option.map (fun (i, why) -> (i0 + i, why)) f)
  in
  (* A chunk from window [i0] ends at the last window whose rows end within
     [chunk] rows of its first, or at [i0 + 1]. *)
  let ends = Nx.cumsum (sizes w) and m = windows w in
  let rec chunks i0 =
    let base = if i0 = 0 then 0L else Nx.item [ i0 - 1 ] ends in
    let limit = Int64.add base (Int64.of_int chunk) in
    let limit = Nx.create Nx.int64 [| 1 |] [| limit |] in
    let i1 = Nx.item [ 0 ] (Nx.searchsorted ~side:`Right ends limit) in
    let i1 = Int64.to_int i1 in
    let i1 = Int.min m (Int.max (i0 + 1) i1) in
    gathered i0 i1 :: (if i1 >= m then [] else chunks i1)
  in
  let parts = chunks 0 in
  (Column.concat (List.map fst parts), List.find_map snd parts)

(* [take_at h x] is [x] at each index [h], [-1] where [h] is outside [x]. *)
let take_at h x =
  let n = Nx.dim 0 x in
  let found =
    Nx.logical_and (Nx.greater_equal_s h 0L) (Nx.less_s h (Int64.of_int n))
  in
  let at = Nx.minimum_s (Nx.maximum_s h 0L) (Int64.of_int (n - 1)) in
  Nx.where found (Nx.take at x) (Nx.full_like h (-1L))

(* A window's holders come from sorted positions; a value's rank orders it by
   word, ties by sorted position, so that the least rank holds the first least
   word. *)
let frame w : Reduce.frame =
  let n = Nx.dim 0 w.sorted in
  let values held : Reduce.values =
    let held = Option.map (Nx.take w.sorted) held in
    let pick op = take_at (fold w op ~held (positions n)) w.sorted in
    let extreme ~desc words =
      let ws = Nx.take w.sorted words in
      let by = Nx.lexsort (if desc then Nx.bitwise_not ws else ws) in
      let rank = Nx.scatter ~axis:0 by (positions n) (full n 0L) in
      take_at (take_at (fold w Min ~held rank) by) w.sorted
    in
    let variance p = variance w held (Reduce.column p) in
    {
      fold = (fun op x -> fold w op ~held (Nx.take w.sorted x));
      pick;
      extreme;
      variance = (if w.split then None else Some variance);
      holistic = (fun r ty p -> gather r ty w p);
    }
  in
  let place rows =
    let q = Nx.take (Nx.maximum_s rows 0L) (Lazy.force w.position) in
    let p =
      Nx.where (Nx.less q w.a_hi) (Nx.sub q w.a_lo)
        (Nx.add (Nx.sub w.a_hi w.a_lo) (Nx.sub q w.b_lo))
    in
    Nx.where (Nx.greater_equal_s rows 0L) p rows
  in
  { sizes = (fun () -> sizes w); values; place }
