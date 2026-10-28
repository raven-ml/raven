(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let err fn fmt =
  Printf.ksprintf (fun m -> invalid_arg ("Nx_ragged." ^ fn ^ ": " ^ m)) fmt

let pp_shape = Format.asprintf "%a" Nx.pp_shape

(* [offsets] is 1-D with at least one entry, never decreases, and lies in [0,
   dim 0 values]; row [r] is [values] from [offsets.{r}] to [offsets.{r + 1}]
   along axis 0. *)
type ('v, 's, 'd) t = { offsets : 'd Nx.int64_t; values : ('v, 's, 'd) Nx.t }

(* Reads *)

(* [host fn k x] is [x]'s elements in C order, read on the host by
   [Nx_ragged.fn]. A read refused, as of a traced value, raises naming [fn]; a
   pending check's failure, which the read's wait raises, passes as the check
   raised it. *)
let host fn k x =
  try Nx.to_bigarray k x |> Bigarray.array1_of_genarray
  with
  | Invalid_argument m when String.starts_with ~prefix:"Nx.to_bigarray: " m ->
    err fn "%s" m

let read fn x = host fn Bigarray.int64 x

(* [ints fn xs] reads the 0-D int64 values [xs] in one read. *)
let ints fn xs =
  let a = read fn (Nx.concatenate ~axis:0 (List.map (Nx.reshape [| 1 |]) xs)) in
  Array.init (Bigarray.Array1.dim a) (Bigarray.Array1.get a)

let first o = Nx.slice [ R (0, 1) ] o
let last o = Nx.slice [ R (-1, Nx.dim 0 o) ] o
let starts o = Nx.slice [ R (0, -1) ] o
let ends o = Nx.slice [ R (1, Nx.dim 0 o) ] o

(* Making *)

let check_values fn x =
  if Nx.ndim x = 0 then err fn "0-D values, not at least 1-D"

let pair () = Nx.Ptree.(pair tensor tensor)

(* Monotone offsets lie in [0, rows] when the first is at least 0 and the last
   at most [rows], so each entry is checked against both bounds and its
   predecessor, [0] before the first. *)
let v ~offsets values =
  check_values "v" values;
  if Nx.ndim offsets <> 1 || Nx.dim 0 offsets = 0 then
    err "v" "offsets of shape %s, not 1-D with an entry"
      (pp_shape (Nx.shape offsets));
  let rows = Nx.dim 0 values in
  let before = Nx.pad [| (1, 0) |] 0L (starts offsets) in
  let ok =
    Nx.logical_and
      (Nx.greater_equal offsets before)
      (Nx.less_equal_s offsets (Int64.of_int rows))
  in
  Nx.check (pair ()) ok (offsets, before) (fun i (o, p) ->
      let o = Nx.item [] o and p = Nx.item [] p and i = i.(0) in
      let m =
        if o > Int64.of_int rows then
          Printf.sprintf "offset %d is %Ld, past the %d rows of the values" i o
            rows
        else if i = 0 then Printf.sprintf "offsets start at %Ld, below 0" o
        else Printf.sprintf "offset %d is %Ld, below the %Ld before it" i o p
      in
      Invalid_argument ("Nx_ragged.v: " ^ m));
  { offsets; values }

(* With every length in [0, 2^63), the first running total past int64's range is
   negative. *)
let of_lengths lengths values =
  check_values "of_lengths" values;
  if Nx.ndim lengths <> 1 then
    err "of_lengths" "lengths of shape %s, not 1-D"
      (pp_shape (Nx.shape lengths));
  let rows = Nx.dim 0 values in
  let sums = Nx.cumsum lengths in
  let ok =
    Nx.logical_and
      (Nx.logical_and
         (Nx.greater_equal_s lengths 0L)
         (Nx.greater_equal_s sums 0L))
      (Nx.less_equal_s sums (Int64.of_int rows))
  in
  Nx.check (pair ()) ok (lengths, sums) (fun i (l, s) ->
      let l = Nx.item [] l and s = Nx.item [] s and i = i.(0) in
      let m =
        if l < 0L then Printf.sprintf "length %d is %Ld, negative" i l
        else if s < 0L then
          Printf.sprintf "the lengths to %d sum past int64's range" i
        else
          Printf.sprintf
            "the lengths to %d sum to %Ld, past the %d rows of the values" i s
            rows
      in
      Invalid_argument ("Nx_ragged.of_lengths: " ^ m));
  { offsets = Nx.pad [| (1, 0) |] 0L sums; values }

(* Every id out of range becomes [segments], which the counts drop and the
   stable sort places last. *)
let of_ids ~segments ids x =
  check_values "of_ids" x;
  if segments < 0 then err "of_ids" "%d segments, not at least 0" segments;
  if Nx.ndim ids <> 1 || Nx.dim 0 ids <> Nx.dim 0 x then
    err "of_ids" "ids of shape %s for values of shape %s"
      (pp_shape (Nx.shape ids))
      (pp_shape (Nx.shape x));
  let s = Int64.of_int segments in
  let inside = Nx.logical_and (Nx.greater_equal_s ids 0L) (Nx.less_s ids s) in
  let ids = Nx.where inside ids (Nx.scalar_like ids s) in
  let counts = Nx.reduce_segments Add ~segments ids (Nx.ones_like ids) in
  {
    offsets = Nx.pad [| (1, 0) |] 0L (Nx.cumsum counts);
    values = Nx.take ~axis:0 (Nx.argsort ids) x;
  }

let of_strings ss =
  let n = Array.length ss in
  let offsets = Bigarray.(Array1.create int64 c_layout (n + 1)) in
  offsets.{0} <- 0L;
  for i = 0 to n - 1 do
    offsets.{i + 1} <-
      Int64.add offsets.{i} (Int64.of_int (String.length ss.(i)))
  done;
  let bytes =
    Bigarray.(Array1.create int8_unsigned c_layout (Int64.to_int offsets.{n}))
  in
  for i = 0 to n - 1 do
    let s = ss.(i) and o = Int64.to_int offsets.{i} in
    for k = 0 to String.length s - 1 do
      Bigarray.Array1.unsafe_set bytes (o + k)
        (Char.code (String.unsafe_get s k))
    done
  done;
  let host a = Nx.of_bigarray (Bigarray.genarray_of_array1 a) in
  { offsets = host offsets; values = host bytes }

(* Observing *)

let offsets r = r.offsets
let values r = r.values
let length r = Nx.dim 0 r.offsets - 1
let lengths r = Nx.sub (ends r.offsets) (starts r.offsets)

(* Only the bytes from the first offset to the last are read, so the strings of
   a [sub] of a long array copy only its own bytes. *)
let to_strings r =
  if Nx.ndim r.values <> 1 then
    err "to_strings" "values of shape %s, not 1-D"
      (pp_shape (Nx.shape r.values));
  let o = read "to_strings" r.offsets in
  let at i = Int64.to_int (Bigarray.Array1.unsafe_get o i) in
  let n = length r and lo = at 0 in
  let bytes =
    host "to_strings" Bigarray.int8_unsigned
      (Nx.slice [ R (lo, at n) ] r.values)
  in
  Array.init n (fun i ->
      let start = at i - lo in
      let s = Bytes.create (at (i + 1) - at i) in
      for k = 0 to Bytes.length s - 1 do
        Bytes.unsafe_set s k
          (Char.unsafe_chr (Bigarray.Array1.unsafe_get bytes (start + k)))
      done;
      Bytes.unsafe_to_string s)

(* Transforming *)

let sub r ~offset ~length:k =
  let l = length r in
  if offset < 0 || k < 0 || offset + k > l then
    err "sub" "rows %d to %d of %d rows" offset (offset + k) l;
  { r with offsets = Nx.slice [ R (offset, offset + k + 1) ] r.offsets }

(* Element [k] of new row [j] is the old value at [k] plus [shift.{j}], the
   row's old start less its new one. *)
let take ~indices r =
  if Nx.ndim indices <> 1 then
    err "take" "indices of shape %s, not 1-D" (pp_shape (Nx.shape indices));
  let lens = Nx.take indices (lengths r) in
  let sums = Nx.cumsum lens in
  let offsets = Nx.pad [| (1, 0) |] 0L sums in
  let total =
    if Nx.dim 0 indices = 0 then 0
    else
      (* With every length in [0, 2^63), the first running total past int64's
         range is negative. *)
      match ints "take" [ Nx.reshape [||] (last sums); Nx.min sums ] with
      | [| _; least |] when least < 0L ->
          err "take" "the rows' lengths sum past int64's range"
      | read when read.(0) > Int64.of_int Int.max_int ->
          err "take" "the rows' lengths sum to %Ld, past OCaml's int range"
            read.(0)
      | read -> Int64.to_int read.(0)
  in
  if total = 0 then { offsets; values = Nx.slice [ R (0, 0) ] r.values }
  else
    let row = Nx.positions ~size:total lens in
    let shift = Nx.sub (Nx.take indices (starts r.offsets)) (starts offsets) in
    let here = Nx.Placement.uncut (Nx.placement row) in
    let at =
      Nx.add (Nx.arange_at here Nx.int64 0 total 1) (Nx.take row shift)
    in
    { offsets; values = Nx.take ~axis:0 at r.values }

let cells r = Array.sub (Nx.shape r.values) 1 (Nx.ndim r.values - 1)

(* Part [i] keeps its rows' values, [lo] to [hi], and its offsets move by [base
   - lo], [base] being the values of the parts before it. *)
let concat = function
  | [] -> err "concat" "no ragged array"
  | [ r ] -> r
  | r :: _ as rs ->
      List.iter
        (fun r' ->
          if cells r' <> cells r then
            err "concat" "cells of shape %s and %s"
              (pp_shape (cells r))
              (pp_shape (cells r')))
        rs;
      let bound o = Nx.reshape [||] o in
      let bounds =
        Array.map Int64.to_int
          (ints "concat"
             (List.concat_map
                (fun r -> [ bound (first r.offsets); bound (last r.offsets) ])
                rs))
      in
      let rebase (i, base, parts) r =
        let lo = bounds.(2 * i) and hi = bounds.((2 * i) + 1) in
        let o = Nx.add_s r.offsets (Int64.of_int (base - lo)) in
        let o = if i = 0 then o else ends o in
        let v = Nx.slice [ R (lo, hi) ] r.values in
        (i + 1, base + hi - lo, (o, v) :: parts)
      in
      let _, _, parts = List.fold_left rebase (0, 0, []) rs in
      let parts = List.rev parts in
      {
        offsets = Nx.concatenate ~axis:0 (List.map fst parts);
        values = Nx.concatenate ~axis:0 (List.map snd parts);
      }

let map f r =
  let values = f r.values in
  if Nx.ndim values = 0 || Nx.dim 0 values <> Nx.dim 0 r.values then
    err "map" "f maps values of shape %s to shape %s"
      (pp_shape (Nx.shape r.values))
      (pp_shape (Nx.shape values));
  { offsets = r.offsets; values }

(* Summarizing rows *)

(* The elements of [r]'s values in C order, and its offsets in elements: a row
   of cells of [c] elements is [c] times as long. *)
let elements r =
  let n = Nx.dim 0 r.values in
  let c = if n = 0 then 1 else Nx.numel r.values / n in
  let flat = Nx.reshape [| Nx.numel r.values |] r.values in
  if c = 1 then (flat, r.offsets)
  else (flat, Nx.mul_s r.offsets (Int64.of_int c))

(* [interpolate a b f] is [a + f (b - a)] between the order statistics [a] and
   [b], and [a] itself where [f], rounded to their format, is zero or [a] equals
   [b]: [Nx.quantile]'s interpolation, the narrow floats at float32 and rounded
   once. *)
let interpolate (type s d) (a : (float, s, d) Nx.t) b (f : d Nx.float64_t) =
  let lerp a b =
    let f = Nx.cast (Nx.dtype a) f in
    let exact = Nx.logical_or (Nx.equal_s f 0.) (Nx.equal a b) in
    Nx.where exact a (Nx.add a (Nx.mul f (Nx.sub b a)))
  in
  match Nx.dtype a with
  | Float32 | Float64 -> lerp a b
  | Float16 | Bfloat16 | Float8_e4m3fn | Float8_e5m2 | Float4_e2m1fn ->
      let wide x = Nx.cast Nx.float32 x in
      Nx.cast (Nx.dtype a) (lerp (wide a) (wide b))

(* One sort of the values by row, then value, puts row [r]'s values at
   [offsets.{r}] onwards, least first. A value's row is [-1] before the first
   offset and [n] from the last on. *)
let quantile qs r =
  Array.iter
    (fun q ->
      if not (q >= 0. && q <= 1.) then
        err "quantile" "probability %h is outside [0, 1]" q)
    qs;
  let values, offsets = elements r in
  let n = length r and k = Array.length qs in
  let here = Nx.Placement.uncut (Nx.placement values) in
  if k = 0 then Nx.zeros_at here (Nx.dtype values) [| 0; n |]
  else
    let marks =
      Nx.scatter ~combine:Add ~axis:0 offsets (Nx.ones_like offsets)
        (Nx.zeros_at here Nx.int64 [| Nx.dim 0 values |])
    in
    let row = Nx.sub_s (Nx.cumsum marks) 1L in
    let key x = Nx.order_key Nx.uint64 x in
    let sorted =
      Nx.take (Nx.lexsort (Nx.stack ~axis:1 [ key row; key values ])) values
    in
    let lens = lengths { offsets; values } in
    let last = Nx.sub_s (Nx.cast Nx.float64 lens) 1. in
    let at = Nx.stack ~axis:0 (List.map (Nx.mul_s last) (Array.to_list qs)) in
    let below = Nx.floor at in
    let statistic i =
      Nx.reshape [| k; n |]
        (Nx.take (Nx.reshape [| k * n |] (Nx.add (starts offsets) i)) sorted)
    in
    let lo = Nx.cast Nx.int64 below in
    let hi = Nx.minimum (Nx.add_s lo 1L) (Nx.sub_s lens 1L) in
    let y = interpolate (statistic lo) (statistic hi) (Nx.sub at below) in
    Nx.where (Nx.equal_s lens 0L) (Nx.full_like y Float.nan) y

(* Ids and ranks of rows *)

(* [keys fn x] is the order keys of [x]'s elements at their own width. *)
let keys (type v s d) fn (x : (v, s, d) Nx.t) : d Nx.packed =
  match Nx.dtype x with
  | Complex128 | Complex64 -> err fn "complex values have no order"
  | dt -> (
      match Nx.Dtype.bits dt with
      | 64 -> P (Nx.order_key Nx.uint64 x)
      | 32 -> P (Nx.order_key Nx.uint32 x)
      | 16 -> P (Nx.order_key Nx.uint16 x)
      | _ -> P (Nx.order_key Nx.uint8 x))

let column m x = Nx.reshape [| m; 1 |] (Nx.bitcast Nx.uint64 x)

(* [codes fn r] is a code for each row of [r]: the number of rows that order
   before it, so equal rows share one.

   Rows are told apart in rounds of up to 64 bytes of keys. Each code is the
   first position of the row's class in the order of every row, and a class
   holds the positions from its code on: before the first round, the one class
   of every row, at [0]. A round sorts the rows of the classes still to tell
   apart by their code, their next keys and, where rows end within the round,
   the number of keys each holds there, which puts a row before the rows it is a
   prefix of. A class then splits into runs of equal rows, and the run that
   starts [j] positions into its class has the code [j] more than the class. A
   row is done once it is alone in its run or has no keys left; a run's rows end
   in the same round, so a class still to tell apart is whole in the next. A
   round that leaves a row with keys reads, through [fn], the count and lengths
   of the rows left. *)
let codes fn r =
  let n = length r in
  let values, offsets = elements r in
  let (P keys) = keys fn values in
  let w = Nx.Dtype.bits (Nx.dtype keys) / 8 in
  (* Every window of up to 64 bytes from an element of a row lies in [keys]. *)
  let keys = Nx.pad [| (0, 64 / w) |] (Nx.Dtype.zero (Nx.dtype keys)) keys in
  let start = starts offsets and len = lengths { offsets; values } in
  let rec round ~seen ~active ~m ~longest ~shortest code =
    if m = 0 || longest = 0 then code
    else
      let width =
        Option.value ~default:64
          (List.find_opt (fun b -> b >= longest * w) [ 8; 16; 32 ])
      in
      let e = width / w and q = 8 / w and words = width / 8 in
      let rem = Nx.sub_s (Nx.take active len) (Int64.of_int seen) in
      let count = Nx.minimum_s rem (Int64.of_int (e + 1)) in
      let chunk =
        Nx.take ~axis:0
          (Nx.add_s (Nx.take active start) (Int64.of_int seen))
          (Nx.sliding_window ~window:e keys)
      in
      (* Keys past a row's end are zero. A word's first key is its most
         significant, and an element wider than a byte is stored little-endian,
         so each word's keys are reversed before they join into it. *)
      let reversed x = Nx.flip ~axes:[ -1 ] x in
      let position =
        let here = Nx.Placement.uncut (Nx.placement chunk) in
        reversed
          (Nx.reshape [| 1; words; q |] (Nx.arange_at here Nx.int64 0 e 1))
      in
      let chunk = reversed (Nx.reshape [| m; words; q |] chunk) in
      let chunk =
        Nx.where
          (Nx.less position (Nx.reshape [| m; 1; 1 |] count))
          chunk (Nx.zeros_like chunk)
      in
      let own = Nx.take active code in
      let key =
        Nx.concatenate ~axis:1
          ((if seen = 0 then [] else [ column m own ])
          @ [ Nx.reshape [| m; words |] (Nx.bitcast Nx.uint64 chunk) ]
          @
          if shortest > e || shortest = longest then [] else [ column m count ]
          )
      in
      let perm = Nx.lexsort key in
      let sorted = Nx.take ~axis:0 perm key and own = Nx.take perm own in
      (* Whether each sorted entry starts a run: the first, and each that
         differs from the one before it in some column. *)
      let differs across x =
        Nx.pad
          [| (1, 0) |]
          true
          (across
             (Nx.not_equal (Nx.slice [ R (1, m) ] x) (Nx.slice [ R (0, -1) ] x)))
      in
      let runs = differs (Nx.any ~axes:[ 1 ]) sorted in
      let at =
        Nx.arange_at (Nx.Placement.uncut (Nx.placement sorted)) Nx.int64 0 m 1
      in
      let first_of starts = Nx.cummax (Nx.where starts at (Nx.zeros_like at)) in
      let fresh =
        Nx.add own (Nx.sub (first_of runs) (first_of (differs Fun.id own)))
      in
      let rows = Nx.take perm active in
      let code = Nx.scatter ~axis:0 rows fresh code in
      if longest <= e then code
      else
        (* The rows left: with keys past the round, in a run of two or more. *)
        let alone =
          Nx.logical_and runs
            (Nx.pad [| (0, 1) |] true (Nx.slice [ R (1, m) ] runs))
        in
        let left = Nx.sub_s (Nx.take perm rem) (Int64.of_int e) in
        let keep =
          Nx.logical_and (Nx.greater_s left 0L) (Nx.logical_not alone)
        in
        let read =
          ints fn
            [
              Nx.count keep;
              Nx.max (Nx.where keep left (Nx.zeros_like left));
              Nx.min (Nx.where keep left (Nx.full_like left Int64.max_int));
            ]
        in
        let m = Int64.to_int read.(0) in
        round ~seen:(seen + e)
          ~active:(Nx.take (Nx.positions ~size:m keep) rows)
          ~m
          ~longest:(Int64.to_int read.(1))
          ~shortest:(Int64.to_int read.(2))
          code
  in
  let here = Nx.Placement.uncut (Nx.placement len) in
  if n = 0 then Nx.zeros_at here Nx.int64 [| 0 |]
  else
    let read = ints fn [ Nx.max len; Nx.min len ] in
    round ~seen:0 ~active:(Nx.arange_at here Nx.int64 0 n 1) ~m:n
      ~longest:(Int64.to_int read.(0))
      ~shortest:(Int64.to_int read.(1))
      (Nx.zeros_at here Nx.int64 [| n |])

(* The codes of distinct rows, marked at their positions, count the distinct
   rows before each. *)
let rank r =
  let code = codes "rank" r in
  let marks =
    Nx.scatter ~axis:0 code (Nx.ones_like code) (Nx.zeros_like code)
  in
  Nx.sub_s (Nx.take code (Nx.cumsum marks)) 1L

let ids r =
  let code = codes "ids" r in
  (Nx.unique ~size:(Nx.dim 0 code) code).ids
