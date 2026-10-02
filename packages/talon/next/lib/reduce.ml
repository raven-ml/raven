(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rows [i] of segment [ids.{i}], in the order [order] lists them, row order
   where it is [None]. [first] is each segment's first row. *)
type segments = {
  ids : Nx.int64_t;
  count : int;
  first : Nx.int64_t;
  order : Nx.int64_t option;
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
    order = None;
  }

let of_groups (g : Nx.groups) order =
  { ids = g.ids; count = Nx.dim 0 g.first; first = g.first; order }

let group cs = of_groups (Nx.unique (Key.identity cs)) None
let count s = s.count
let first s = s.first

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
        | Some p ->
            Nx.take ~indices:(Nx.lexsort (Nx.take ~axis:0 ~indices:p k)) p
      in
      { s with order = Some order }

let broadcast s c = Column.take s.ids c

(* [view s] is the rows of each segment in its order. *)
let view s =
  let p = Option.value s.order ~default:(positions (Nx.dim 0 s.ids)) in
  Nx_ragged.of_ids ~segments:s.count (Nx.take ~indices:p s.ids) p

(* [fixed ty ?valid x] is the column of [ty] stored as [x], with zeros under its
   nulls. *)
let fixed ty ?valid x =
  let x =
    match valid with Some v -> Nx.where v x (Nx.zeros_like x) | None -> x
  in
  Column.make ty ?valid ~length:(Nx.dim 0 x) (Fixed (P x))

let tensor dt c =
  match Column.data c with Fixed (P x) -> Nx.cast dt x | _ -> assert false

let segment_sum s ids x = Nx.reduce_segments `Add ~segments:s.count ids x

let rows s =
  fixed int64 (segment_sum s s.ids (Nx.ones Nx.int64 [| Nx.dim 0 s.ids |]))

(* [pick s op hit] is, in each segment, the row where [hit] holds that comes
   first ([`Min]) or last ([`Max]) in the segment's order, and its place in the
   segment; [-1] for both where no row does. *)
let pick s op hit =
  let r = view s in
  let v = Nx_ragged.values r and n = Nx.dim 0 s.ids in
  let ids =
    Nx.where (Nx.take ~indices:v hit) (Nx.take ~indices:v s.ids) (none n)
  in
  let j = Nx.reduce_segments op ~segments:s.count ids (positions n) in
  let found =
    Nx.logical_and (Nx.greater_equal_s j 0L) (Nx.less_s j (Int64.of_int n))
  in
  let start = Nx.shrink [| (0, s.count) |] (Nx_ragged.offsets r) in
  ( Nx.where found (Nx.take ~indices:j v) (none s.count),
    Nx.where found (Nx.sub j start) (none s.count) )

(* [first_where m why] is the first segment where [m] holds, failing for the
   reason [why]. *)
let first_where m why =
  let gs = Nx.positions m in
  if Nx.dim 0 gs = 0 then None else Some (Int64.to_int (Nx.item [ 0 ] gs), why)

(* [exact_sum ty s ids x] sums the [int64] values [x] exactly, as high and low
   32 bits apart, and is null where a sum leaves [int64]. *)
let exact_sum ty s ids x =
  let hi = Nx.rshift x 32 in
  let lo = Nx.sub x (Nx.lshift hi 32) in
  let lo = segment_sum s ids lo in
  let carry = Nx.rshift lo 32 in
  let hi = Nx.add (segment_sum s ids hi) carry in
  let fits =
    Nx.logical_and
      (Nx.greater_equal_s hi (-0x8000_0000L))
      (Nx.less_s hi 0x8000_0000L)
  in
  let sum = Nx.add (Nx.lshift hi 32) (Nx.sub lo (Nx.lshift carry 32)) in
  let why = Format.asprintf "the sum overflows %a" Type.pp ty in
  (fixed (Any ty) ~valid:fits sum, first_where (Nx.logical_not fits) why)

let reduce : type a b.
    (a, b) Expr.reduction ->
    Type.any ->
    segments ->
    Column.t ->
    Column.t * (int * string) option =
 fun r (Any rty as ty) s c ->
  let n = Column.length c in
  let ids =
    match Column.valid c with
    | Some v -> Nx.where v s.ids (none n)
    | None -> s.ids
  in
  let valid =
    Option.value (Column.valid c) ~default:(Nx.ones Nx.bool [| n |])
  in
  let count = segment_sum s ids (Nx.ones Nx.int64 [| n |]) in
  let at_least k = Nx.greater_equal_s count (Int64.of_int k) in
  let sum dt = segment_sum s ids (tensor dt c) in
  let mean () = Nx.div (sum Nx.float64) (Nx.cast Nx.float64 count) in
  let var () =
    let d = Nx.sub (tensor Nx.float64 c) (Nx.take ~indices:s.ids (mean ())) in
    let k = Nx.cast Nx.float64 (Nx.sub_s count 1L) in
    Nx.div (segment_sum s ids (Nx.mul d d)) k
  in
  let quantile p =
    let r = Nx_ragged.of_ids ~segments:s.count ids (tensor Nx.float64 c) in
    let q = Nx.reshape [| s.count |] (Nx_ragged.quantile [| p |] r) in
    fixed float64 ~valid:(at_least 1) q
  in
  let words = lazy (Key.value Order c) in
  let extreme op =
    Nx.reduce_segments op ~segments:s.count ids (Lazy.force words)
  in
  let first_at e =
    let hit = Nx.equal (Lazy.force words) (Nx.take ~indices:s.ids e) in
    pick s `Min (Nx.logical_and valid hit)
  in
  let place (_, p) = fixed int64 ~valid:(Nx.greater_equal_s p 0L) p in
  let value (row, _) = Column.take row c in
  let ok c = (c, None) in
  match r with
  | Count -> ok (fixed int64 count)
  | Sum -> (
      match (rty, Column.data c) with
      | Duration _, _ -> exact_sum rty s ids (tensor Nx.int64 c)
      | Int64, _ -> ok (fixed ty (sum Nx.int64))
      | _, Fixed (P x) -> ok (fixed ty (segment_sum s ids x))
      | _ -> invalid_arg "Reduce.reduce: no sum")
  | Mean -> ok (fixed ty ~valid:(at_least 1) (mean ()))
  | Var -> ok (fixed ty ~valid:(at_least 2) (var ()))
  | Std -> ok (fixed ty ~valid:(at_least 2) (Nx.sqrt (var ())))
  | Median -> ok (quantile 0.5)
  | Quantile p -> ok (quantile p)
  | Min -> ok (value (first_at (extreme `Min)))
  | Max -> ok (value (first_at (extreme `Max)))
  | Arg_min -> ok (place (first_at (extreme `Min)))
  | Arg_max -> ok (place (first_at (extreme `Max)))
  | First -> ok (value (pick s `Min valid))
  | Last -> ok (value (pick s `Max valid))
  | Only ->
      let lo = extreme `Min in
      let several =
        Nx.logical_and (at_least 1) (Nx.not_equal lo (extreme `Max))
      in
      (value (first_at lo), first_where several "only finds several values")
  | N_unique ->
      let g = Nx.unique (keyed s (Key.identity [ c ])) in
      let ones = Nx.ones Nx.int64 [| Nx.dim 0 g.first |] in
      ok (fixed int64 (segment_sum s (Nx.take ~indices:g.first s.ids) ones))
  | Ewm _ | Collect ->
      invalid_arg "Reduce.reduce: ewm and collect are not lowered"

let shift s k c =
  let r = view s and n = Column.length c in
  let v = Nx_ragged.values r and offsets = Nx_ragged.offsets r in
  let segment = Nx.take ~indices:v s.ids in
  let start = Nx.take ~indices:segment offsets in
  let stop = Nx.take ~indices:(Nx.add_s segment 1L) offsets in
  let j = Nx.sub_s (positions n) (Int64.of_int k) in
  let inside = Nx.logical_and (Nx.greater_equal j start) (Nx.less j stop) in
  let from = Nx.where inside (Nx.take ~indices:j v) (none n) in
  Column.take (Nx.scatter ~axis:0 ~indices:v ~values:from (none n)) c

(* A row's rank is its place in its segment's rows sorted by value, at the start
   of its run of equal values: [1] plus the number of rows before the run, which
   are values, nulls sorting last. *)
let rank s c =
  let n = Column.length c in
  if n = 0 then fixed int64 (Nx.zeros Nx.int64 [| 0 |])
  else
    let k = keyed s (Key.order [ (c, Order.asc "") ]) in
    let p = Nx.lexsort k in
    let k = Nx.take ~axis:0 ~indices:p k in
    let w = Nx.dim 1 k and i = positions n in
    let differs =
      Nx.not_equal
        (Nx.shrink [| (1, n); (0, w) |] k)
        (Nx.shrink [| (0, n - 1); (0, w) |] k)
    in
    let start changed =
      let b = Nx.concatenate ~axis:0 [ Nx.ones Nx.bool [| 1 |]; changed ] in
      Nx.cummax (Nx.where b i (Nx.zeros_like i))
    in
    let run = start (Nx.any ~axes:[ 1 ] differs) in
    let segment =
      start
        (Nx.reshape [| n - 1 |] (Nx.shrink [| (0, n - 1); (0, 1) |] differs))
    in
    let r = Nx.add_s (Nx.sub run segment) 1L in
    fixed int64 ?valid:(Column.valid c)
      (Nx.scatter ~axis:0 ~indices:p ~values:r (Nx.zeros Nx.int64 [| n |]))
