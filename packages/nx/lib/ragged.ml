(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Frontend

type ('a, 'b) tensor = ('a, 'b) Nx_effect.t

(* [offsets] is 1-D with at least one entry, never decreases, and lies in [0,
   dim 0 values]; row [r] is [values] from [offsets.{r}] to [offsets.{r + 1}]
   along axis 0. *)
type ('a, 'b) t = { offsets : int64_t; values : ('a, 'b) tensor }

let shape_string x = Nx_array.Shape.to_string (shape x)

(* [require op checks] reads the flags of [checks] through ["Nx." ^ op], once,
   and raises the message of the first that is false. *)
let require op checks =
  let flags =
    read_array ~by:("Nx." ^ op)
      (concatenate ~axis:0 (List.map (fun (c, _) -> reshape [| 1 |] c) checks))
  in
  List.iteri (fun i (_, msg) -> if not flags.(i) then err op "%s" msg) checks

let check_values op values =
  if ndim values = 0 then err op "values of shape [], not at least 1-D"

let first offsets = shrink [| (0, 1) |] offsets

let last offsets =
  let n = dim 0 offsets in
  shrink [| (n - 1, n) |] offsets

let v ~offsets values =
  check_values "Ragged.v" values;
  if ndim offsets <> 1 || dim 0 offsets = 0 then
    err "Ragged.v" "offsets of shape %s, not 1-D with an entry"
      (shape_string offsets);
  let n = dim 0 offsets and rows = dim 0 values in
  let increase =
    greater_equal
      (shrink [| (1, n) |] offsets)
      (shrink [| (0, n - 1) |] offsets)
  in
  require "Ragged.v"
    [
      (greater_equal_s (first offsets) 0L, "offsets start below 0");
      (all increase, "offsets decrease");
      ( less_equal_s (last offsets) (Int64.of_int rows),
        Printf.sprintf "offsets end past the %d rows of the values" rows );
    ];
  { offsets; values }

let of_lengths lengths values =
  check_values "Ragged.of_lengths" values;
  if ndim lengths <> 1 then
    err "Ragged.of_lengths" "lengths of shape %s, not 1-D"
      (shape_string lengths);
  let rows = dim 0 values in
  let ends = cumsum lengths in
  let offsets = pad [| (1, 0) |] 0L ends in
  if dim 0 lengths > 0 then
    (* With every length in [0, 2^63), the first running total past int64's
       range is negative. *)
    require "Ragged.of_lengths"
      [
        (greater_equal_s (min lengths) 0L, "a length is negative");
        (greater_equal_s (min ends) 0L, "the lengths sum past int64's range");
        ( less_equal_s (last ends) (Int64.of_int rows),
          Printf.sprintf "the lengths sum past the %d rows of the values" rows
        );
      ];
  { offsets; values }

let of_ids ~segments ids x =
  check_values "Ragged.of_ids" x;
  if segments < 0 then
    err "Ragged.of_ids" "%d segments, not at least 0" segments;
  if ndim ids <> 1 || dim 0 ids <> dim 0 x then
    err "Ragged.of_ids" "ids of shape %s for values of shape %s"
      (shape_string ids) (shape_string x);
  let ctx = Nx_effect.context ids and n = dim 0 ids in
  let dropped = Int64.of_int segments in
  let ids =
    where
      (logical_and (greater_equal_s ids 0L) (less_s ids dropped))
      ids (scalar_like ids dropped)
  in
  let counts =
    scatter ~mode:`Add ~axis:0 ~indices:ids
      ~values:(broadcast_to [| n |] (scalar ctx Int64 1L))
      (zeros ctx Int64 [| segments + 1 |])
  in
  let offsets =
    pad [| (1, 0) |] 0L (cumsum (shrink [| (0, segments) |] counts))
  in
  { offsets; values = take ~axis:0 ~indices:(argsort ids) x }

let offsets r = r.offsets
let values r = r.values
let length r = dim 0 r.offsets - 1

let lengths r =
  let n = dim 0 r.offsets in
  sub (shrink [| (1, n) |] r.offsets) (shrink [| (0, n - 1) |] r.offsets)

(* The elements of [r]'s values in row-major order, and its offsets in elements:
   a row of cells of [c] elements is [c] times as long. *)
let elements r =
  let n = dim 0 r.values in
  let c = if n = 0 then 1 else numel r.values / n in
  let flat = reshape [| numel r.values |] r.values in
  if c = 1 then (flat, r.offsets) else (flat, mul_s r.offsets (Int64.of_int c))

let quantile (type b) qs (r : (float, b) t) : (float, b) tensor =
  check_probabilities "Ragged.quantile" qs;
  let values, offsets = elements r in
  let n = length r and k = Array.length qs in
  let ctx = Nx_effect.context values in
  (* Each value's row: -1 before the first offset, [n] from the last on. *)
  let row =
    sub_s
      (cumsum
         (scatter ~mode:`Add ~axis:0 ~indices:offsets
            ~values:(broadcast_to [| n + 1 |] (scalar ctx Int64 1L))
            (zeros ctx Int64 [| dim 0 values |])))
      1L
  in
  (* Row [r]'s values are at [offsets.{r}] onwards, least first. *)
  let sorted =
    take
      ~indices:(lexsort (stack ~axis:1 [ order_key row; order_key values ]))
      values
  in
  let lens = reshape [| 1; n |] (lengths { offsets; values }) in
  let at =
    mul (create ctx Float64 [| k; 1 |] qs) (sub_s (cast Float64 lens) 1.)
  in
  let lo = floor at in
  let statistic i =
    let i = add (reshape [| 1; n |] (shrink [| (0, n) |] offsets)) i in
    reshape [| k; n |] (take ~indices:(reshape [| k * n |] i) sorted)
  in
  let lo_i = cast Int64 lo in
  let hi_i = minimum (add_s lo_i 1L) (sub_s lens 1L) in
  where (equal_s lens 0L)
    (full ctx (dtype values) [| k; n |] Float.nan)
    (interpolate (statistic lo_i) (statistic hi_i) (sub at lo))

(* Ids and ranks of rows *)

(* The keys of elements: unsigned integers of the elements' width whose order is
   the sort order. *)
type keys = Keys : ('a, 'b) Nx_dtype.t * ('a, 'b) tensor -> keys

let keys (type a b) op (x : (a, b) tensor) : keys =
  match dtype x with
  | Bool -> Keys (UInt8, cast UInt8 x)
  | UInt4 -> Keys (UInt8, cast UInt8 x)
  | UInt8 -> Keys (UInt8, x)
  | UInt16 -> Keys (UInt16, x)
  | UInt32 -> Keys (UInt32, x)
  | UInt64 -> Keys (UInt64, x)
  | Int4 -> Keys (UInt8, signed_key UInt8 ~sign:(-0x80) (cast Int8 x))
  | Int8 -> Keys (UInt8, signed_key UInt8 ~sign:(-0x80) x)
  | Int16 -> Keys (UInt16, signed_key UInt16 ~sign:(-0x8000) x)
  | Int32 -> Keys (UInt32, signed_key UInt32 ~sign:Int32.min_int x)
  | Int64 -> Keys (UInt64, signed_key UInt64 ~sign:Int64.min_int x)
  | Float8_e4m3 | Float8_e5m2 ->
      Keys (UInt16, float_key Int16 UInt16 ~sign:(-0x8000) (cast Float16 x))
  | Float16 | BFloat16 -> Keys (UInt16, float_key Int16 UInt16 ~sign:(-0x8000) x)
  | Float32 -> Keys (UInt32, float_key Int32 UInt32 ~sign:Int32.min_int x)
  | Float64 -> Keys (UInt64, float_key Int64 UInt64 ~sign:Int64.min_int x)
  | Complex64 | Complex128 -> err op "complex numbers have no order"

(* [ranks op r] is the dense rank of each row of [r] in the sort order, a prefix
   first, and the number of distinct rows. Rows are told apart in rounds of up
   to 64 bytes of keys: a round reads the rows still to tell apart as uint64
   words, most significant key first, and ranks them by their rank so far, their
   words and the number of keys of the round they hold, which separates a row
   from its prefixes. A row is done once it is alone in its class or has no keys
   left. Each round reads once, through ["Nx." ^ op]. *)
let ranks op r =
  let by = "Nx." ^ op in
  let n = length r in
  let ctx = Nx_effect.context r.offsets in
  if n = 0 then (empty ctx Int64 [| 0 |], 0)
  else
    let values, offsets = elements r in
    let (Keys (kt, keys)) = keys op values in
    let w = Nx_dtype.itemsize kt in
    (* Every window of up to 64 bytes from an element of a row lies in
       [keys]. *)
    let keys = pad [| (0, 64 / w) |] (Nx_dtype.zero kt) keys in
    let start = shrink [| (0, n) |] offsets in
    let len = sub (shrink [| (1, n + 1) |] offsets) start in
    let rec round ~seen ~classes ~active ~m ~longest ~shortest rank =
      if m = 0 || longest = 0 then (rank, classes)
      else
        let width =
          List.find_opt (fun b -> b >= longest * w) [ 8; 16; 32 ]
          |> Option.value ~default:64
        in
        let e = width / w and q = 8 / w and words = width / 8 in
        let rem = sub_s (take ~indices:active len) (Int64.of_int seen) in
        (* The keys of the round a row holds, and one more if it holds more. *)
        let count = minimum_s rem (Int64.of_int (e + 1)) in
        let chunk =
          take ~axis:0
            ~indices:(add_s (take ~indices:active start) (Int64.of_int seen))
            (sliding_window ~window:e keys)
        in
        (* Keys past a row's end are zero. A word's first key is its most
           significant, so a little-endian machine reverses each word's keys. *)
        let position =
          create ctx Int64 [| words; q |] (Array.init e Int64.of_int)
        in
        let chunk = reshape [| m; words; q |] chunk in
        let chunk, position =
          if Sys.big_endian then (chunk, position)
          else (flip ~axes:[ 2 ] chunk, flip ~axes:[ 1 ] position)
        in
        let chunk =
          where
            (less
               (reshape [| 1; words; q |] position)
               (reshape [| m; 1; 1 |] count))
            chunk (zeros_like chunk)
        in
        let column x = reshape [| m; 1 |] (bitcast UInt64 x) in
        let row_rank = take ~indices:active rank in
        let key =
          concatenate ~axis:1
            ((if classes > 1 then [ column row_rank ] else [])
            @ [ reshape [| m; words |] (bitcast UInt64 chunk) ]
            @
            if shortest > e || shortest = longest then [] else [ column count ]
            )
        in
        let perm = lexsort key in
        let sorted = take ~axis:0 ~indices:perm key in
        let sorted_rank = take ~indices:perm row_rank in
        (* Whether each sorted row starts a class, and starts one inside its
           class so far. *)
        let starts =
          pad
            [| (1, 0) |]
            true
            (any ~axes:[ 1 ]
               (not_equal
                  (shrink [| (1, m); (0, dim 1 key) |] sorted)
                  (shrink [| (0, m - 1); (0, dim 1 key) |] sorted)))
        in
        let splits =
          logical_and starts
            (pad
               [| (1, 0) |]
               false
               (equal
                  (shrink [| (1, m) |] sorted_rank)
                  (shrink [| (0, m - 1) |] sorted_rank)))
        in
        let splits = cast Int64 splits in
        let within = cumsum splits in
        (* A class moves up by the classes split before it. *)
        let moved =
          if classes = 1 then rank
          else
            let before =
              pad
                [| (1, 0) |]
                0L
                (cumsum
                   (shrink
                      [| (0, classes - 1) |]
                      (reduce_segments `Add ~segments:classes sorted_rank splits)))
            in
            add rank (take ~indices:rank before)
        in
        let rank =
          scatter ~unique_indices:true ~axis:0
            ~indices:(take ~indices:perm active)
            ~values:(add sorted_rank within) moved
        in
        (* Rows still to tell apart: with keys past the round, in a class of two
           rows or more. *)
        let alone =
          logical_and starts
            (pad [| (0, 1) |] true (shrink [| (1, m) |] starts))
        in
        let left = sub_s (take ~indices:perm rem) (Int64.of_int e) in
        let keep = logical_and (greater_s left 0L) (logical_not alone) in
        let kept = cast Int64 keep in
        let read =
          read_array ~by
            (concatenate ~axis:0
               (List.map
                  (fun x -> reshape [| 1 |] x)
                  [
                    sum kept;
                    max (where keep left (zeros_like left));
                    min (where keep left (full_like left Int64.max_int));
                    shrink [| (m - 1, m) |] within;
                  ]))
        in
        let m' = Int64.to_int read.(0) in
        let place =
          where keep (sub (cumsum kept) kept) (scalar ctx Int64 read.(0))
        in
        let active =
          scatter ~unique_indices:true ~axis:0 ~indices:place
            ~values:(take ~indices:perm active)
            (zeros ctx Int64 [| m' |])
        in
        round ~seen:(seen + e)
          ~classes:(classes + Int64.to_int read.(3))
          ~active ~m:m'
          ~longest:(Int64.to_int read.(1))
          ~shortest:(Int64.to_int read.(2))
          rank
    in
    let read =
      read_array ~by
        (concatenate ~axis:0
           [ reshape [| 1 |] (max len); reshape [| 1 |] (min len) ])
    in
    round ~seen:0 ~classes:1 ~active:(arange ctx Int64 0 n 1) ~m:n
      ~longest:(Int64.to_int read.(0))
      ~shortest:(Int64.to_int read.(1))
      (zeros ctx Int64 [| n |])

let rank r = fst (ranks "Ragged.rank" r)

let ids r =
  let rank, classes = ranks "Ragged.ids" r in
  let n = length r in
  let row = arange (Nx_effect.context rank) Int64 0 n 1 in
  (* Each row's class's first row, and the number of classes that start before
     it. *)
  let first =
    take ~indices:rank (reduce_segments `Min ~segments:classes rank row)
  in
  let firsts = cast Int64 (equal first row) in
  take ~indices:first (sub (cumsum firsts) firsts)

let take ~indices r =
  if ndim indices <> 1 then
    err "Ragged.take" "indices of shape %s, not 1-D" (shape_string indices);
  let k = dim 0 indices and l = length r in
  let ctx = Nx_effect.context indices in
  let lens = take ~indices (lengths r) in
  let ends = cumsum lens in
  let offsets = pad [| (1, 0) |] 0L ends in
  let total =
    if k = 0 then 0L
    else
      (* With every length in [0, 2^63), the first running total past int64's
         range is negative. *)
      let read =
        read_array ~by:"Nx.Ragged.take"
          (concatenate ~axis:0 [ last ends; reshape [| 1 |] (min ends) ])
      in
      if read.(1) < 0L then
        err "Ragged.take" "the rows' lengths sum past int64's range";
      read.(0)
  in
  if total = 0L then
    {
      offsets;
      values = take ~axis:0 ~indices:(empty ctx Int64 [| 0 |]) r.values;
    }
  else
    (* Element [j] of new row [i] is value [j + shift.{i}], where [shift.{i}] is
       the row's old start less its new one. That is a running sum of ones with
       each row's change of shift added at its first element, a row that starts
       at [total] being empty. *)
    let firsts = shrink [| (0, k) |] offsets in
    let shift = sub (take ~indices (shrink [| (0, l) |] r.offsets)) firsts in
    let change =
      sub shift (pad [| (1, 0) |] 1L (shrink [| (0, k - 1) |] shift))
    in
    let positions =
      cumsum
        (scatter ~mode:`Add ~axis:0 ~indices:firsts ~values:change
           (full ctx Int64 [| Int64.to_int total |] 1L))
    in
    { offsets; values = take ~axis:0 ~indices:positions r.values }

let sub r ~offset ~length:k =
  let l = length r in
  if offset < 0 || k < 0 || offset + k > l then
    err "Ragged.sub" "rows %d to %d of %d rows" offset (offset + k) l;
  { r with offsets = shrink [| (offset, offset + k + 1) |] r.offsets }

let concat = function
  | [] -> invalid_arg "Ragged.concat: no ragged array"
  | [ r ] -> r
  | r :: _ as rs ->
      let cell r = Array.sub (shape r.values) 1 (ndim r.values - 1) in
      List.iter
        (fun r' ->
          if cell r' <> cell r then
            err "Ragged.concat" "cells of shape %s and %s"
              (Nx_array.Shape.to_string (cell r))
              (Nx_array.Shape.to_string (cell r')))
        rs;
      let bounds =
        read_array ~by:"Nx.Ragged.concat"
          (concatenate ~axis:0
             (List.concat_map (fun r -> [ first r.offsets; last r.offsets ]) rs))
      in
      (* Part [i] keeps its rows' values, [lo] to [hi], and its offsets move by
         [base - lo], [base] being the values of the parts before it. *)
      let rec rebase i base = function
        | [] -> []
        | r :: rs ->
            let lo = bounds.(2 * i) and hi = bounds.((2 * i) + 1) in
            let offsets = add_s r.offsets (Int64.sub base lo) in
            let offsets =
              if i = 0 then offsets else shrink [| (1, dim 0 offsets) |] offsets
            in
            let values =
              slice [ R (Int64.to_int lo, Int64.to_int hi) ] r.values
            in
            (offsets, values)
            :: rebase (i + 1) (Int64.add base (Int64.sub hi lo)) rs
      in
      let parts = rebase 0 0L rs in
      {
        offsets = concatenate ~axis:0 (List.map fst parts);
        values = concatenate ~axis:0 (List.map snd parts);
      }

let map f r =
  let values = f r.values in
  if ndim values = 0 || dim 0 values <> dim 0 r.values then
    err "Ragged.map" "f maps values of shape %s to shape %s"
      (shape_string r.values) (shape_string values);
  { r with values }
