(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx

let err op fmt = Printf.ksprintf (fun msg -> invalid_arg (op ^ ": " ^ msg)) fmt

(* [float_text x] is the text [Nx.pp] gives the float64 [x]: the fewest
   significant digits that read back to it, in full for decimal exponents from
   -4 to 15 and with an exponent beyond; [nan] whatever its sign. *)
let float_text x =
  let sign = if Float.sign_bit x then "-" else "" in
  let a = Float.abs x in
  let reads (m, e) = float_of_string (Printf.sprintf "%de%d" m e) = a in
  let rec pow10 p = if p = 0 then 1 else 10 * pow10 (p - 1) in
  (* The [p] digits [m] and the exponent [e] of [m 10^e] nearest [a], or the
     next decimal above or below it: the values that read back to [a] are an
     interval around it, which holds a [p]-digit decimal only if it holds one of
     the two around [a]. Seventeen digits read back to any float. *)
  let rec shortest p =
    let t = Printf.sprintf "%.*e" (p - 1) a in
    let i = String.index t 'e' in
    let m =
      int_of_string
        (String.concat "" (String.split_on_char '.' (String.sub t 0 i)))
    in
    let e =
      int_of_string (String.sub t (i + 1) (String.length t - i - 1)) - (p - 1)
    in
    let below =
      if m - 1 < pow10 (p - 1) then ((10 * m) - 1, e - 1) else (m - 1, e)
    in
    match List.find_opt reads [ (m, e); (m + 1, e); below ] with
    | Some c -> c
    | None -> if p >= 17 then (m, e) else shortest (p + 1)
  in
  if Float.is_nan x then "nan"
  else if a = Float.infinity then sign ^ "inf"
  else if a = 0. then sign ^ "0"
  else
    let m, e = shortest 1 in
    let digits = string_of_int m in
    let exp = e + String.length digits - 1 in
    (* [m] ends in zeros after a carry, as [m + 1] at [99] gives [100]. *)
    let p = ref (String.length digits) in
    while digits.[!p - 1] = '0' do
      decr p
    done;
    let p = !p in
    let d = String.sub digits 0 p in
    sign
    ^
    if exp < -4 || exp >= 16 then
      let fraction = if p > 1 then "." ^ String.sub d 1 (p - 1) else "" in
      Printf.sprintf "%c%se%c%02d" d.[0] fraction
        (if exp < 0 then '-' else '+')
        (Int.abs exp)
    else if exp >= p - 1 then d ^ String.make (exp - p + 1) '0'
    else if exp >= 0 then
      String.sub d 0 (exp + 1) ^ "." ^ String.sub d (exp + 1) (p - exp - 1)
    else "0." ^ String.make (-exp - 1) '0' ^ d

(* [offsets] is 1-D with at least one entry, never decreases, and lies in [0,
   dim 0 values]; row [r] is [values] from [offsets.{r}] to [offsets.{r + 1}]
   along axis 0. *)
type ('a, 'b) t = { offsets : int64_t; values : ('a, 'b) Nx.t }

let shape_string x = Format.asprintf "%a" pp_shape (shape x)

(* [read ~by x] is the elements of the int64 tensor [x] in C order, read by the
   function [by]. *)
let read ~by x =
  let b = Op.eval (Read { by; x }) in
  Nx_device.Buffer.Claim.read b;
  Fun.protect
    ~finally:(fun () -> Nx_device.Buffer.Claim.release b)
    (fun () ->
      let a = Nx_device.Buffer.bigarray Bigarray.int64 b in
      Array.init (numel x) (Bigarray.Array1.get a))

(* [full_as x shape v] is a tensor of [shape] filled with [v], of [x]'s dtype
   and placed as [x] is. *)
let full_as x shape v = contiguous (broadcast_to shape (scalar_like x v))

(* [require op checks] reads the flags of [checks] through [op], once, and
   raises the message of the first that is false. *)
let require op checks =
  let flags =
    read ~by:op
      (cast Int64
         (concatenate ~axis:0
            (List.map (fun (c, _) -> reshape [| 1 |] c) checks)))
  in
  List.iteri (fun i (_, msg) -> if flags.(i) = 0L then err op "%s" msg) checks

let check_values op values =
  if ndim values = 0 then err op "values of shape [], not at least 1-D"

let first offsets = shrink [| (0, 1) |] offsets

let last offsets =
  let n = dim 0 offsets in
  shrink [| (n - 1, n) |] offsets

let v ~offsets values =
  check_values "Nx_ragged.v" values;
  if ndim offsets <> 1 || dim 0 offsets = 0 then
    err "Nx_ragged.v" "offsets of shape %s, not 1-D with an entry"
      (shape_string offsets);
  let n = dim 0 offsets and rows = dim 0 values in
  let increase =
    greater_equal
      (shrink [| (1, n) |] offsets)
      (shrink [| (0, n - 1) |] offsets)
  in
  require "Nx_ragged.v"
    [
      (greater_equal_s (first offsets) 0L, "offsets start below 0");
      (all increase, "offsets decrease");
      ( less_equal_s (last offsets) (Int64.of_int rows),
        Printf.sprintf "offsets end past the %d rows of the values" rows );
    ];
  { offsets; values }

let of_lengths lengths values =
  check_values "Nx_ragged.of_lengths" values;
  if ndim lengths <> 1 then
    err "Nx_ragged.of_lengths" "lengths of shape %s, not 1-D"
      (shape_string lengths);
  let rows = dim 0 values in
  let ends = cumsum lengths in
  let offsets = pad [| (1, 0) |] 0L ends in
  if dim 0 lengths > 0 then
    (* With every length in [0, 2^63), the first running total past int64's
       range is negative. *)
    require "Nx_ragged.of_lengths"
      [
        (greater_equal_s (min lengths) 0L, "a length is negative");
        (greater_equal_s (min ends) 0L, "the lengths sum past int64's range");
        ( less_equal_s (last ends) (Int64.of_int rows),
          Printf.sprintf "the lengths sum past the %d rows of the values" rows
        );
      ];
  { offsets; values }

let of_ids ~segments ids x =
  check_values "Nx_ragged.of_ids" x;
  if segments < 0 then
    err "Nx_ragged.of_ids" "%d segments, not at least 0" segments;
  if ndim ids <> 1 || dim 0 ids <> dim 0 x then
    err "Nx_ragged.of_ids" "ids of shape %s for values of shape %s"
      (shape_string ids) (shape_string x);
  let n = dim 0 ids in
  let dropped = Int64.of_int segments in
  let ids =
    where
      (logical_and (greater_equal_s ids 0L) (less_s ids dropped))
      ids (scalar_like ids dropped)
  in
  let counts =
    scatter ~mode:`Add ~axis:0 ~indices:ids
      ~values:(broadcast_to [| n |] (scalar_like ids 1L))
      (full_as ids [| segments + 1 |] 0L)
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

let check_probabilities op qs =
  Array.iter
    (fun q ->
      if not (q >= 0. && q <= 1.) then
        err op "probability %s is outside [0, 1]" (float_text q))
    qs

(* [interpolate a b f] is [a + f * (b - a)] between the order statistics [a] and
   [b], and [a] itself where [f] is zero or [a] equals [b]: [Nx.quantile]'s
   interpolation. Narrow floats interpolate in float32 and round once. *)
let interpolate (type b) (a : (float, b) Nx.t) (b : (float, b) Nx.t)
    (f : float64_t) : (float, b) Nx.t =
  let exact = equal_s f 0. in
  let lerp (type c) (a : (float, c) Nx.t) (b : (float, c) Nx.t) =
    let f = cast (dtype a) f in
    where (logical_or exact (equal a b)) a (add a (mul f (sub b a)))
  in
  match dtype a with
  | Float32 | Float64 -> lerp a b
  | Float16 | BFloat16 | Float8_e4m3 | Float8_e5m2 ->
      cast (dtype a) (lerp (cast Float32 a) (cast Float32 b))

let quantile (type b) qs (r : (float, b) t) : (float, b) Nx.t =
  check_probabilities "Nx_ragged.quantile" qs;
  let values, offsets = elements r in
  let n = length r and k = Array.length qs in
  (* Each value's row: -1 before the first offset, [n] from the last on. *)
  let row =
    sub_s
      (cumsum
         (scatter ~mode:`Add ~axis:0 ~indices:offsets
            ~values:(broadcast_to [| n + 1 |] (scalar_like offsets 1L))
            (full_as offsets [| dim 0 values |] 0L)))
      1L
  in
  (* Row [r]'s values are at [offsets.{r}] onwards, least first. *)
  let sorted =
    take
      ~indices:
        (lexsort
           (stack ~axis:1 [ order_key uint64 row; order_key uint64 values ]))
      values
  in
  let lens = reshape [| 1; n |] (lengths { offsets; values }) in
  let at = mul (create Float64 [| k; 1 |] qs) (sub_s (cast Float64 lens) 1.) in
  let lo = floor at in
  let statistic i =
    let i = add (reshape [| 1; n |] (shrink [| (0, n) |] offsets)) i in
    reshape [| k; n |] (take ~indices:(reshape [| k * n |] i) sorted)
  in
  let lo_i = cast Int64 lo in
  let hi_i = minimum (add_s lo_i 1L) (sub_s lens 1L) in
  where (equal_s lens 0L)
    (full_as values [| k; n |] Float.nan)
    (interpolate (statistic lo_i) (statistic hi_i) (sub at lo))

(* Ids and ranks of rows *)

(* The order keys of elements at their own width. *)
let keys (type a b) op (x : (a, b) Nx.t) =
  match dtype x with
  | Bool | Bit | UInt4 | UInt8 | Int4 | Int8 | Float8_e4m3 | Float8_e5m2 ->
      P (order_key uint8 x)
  | UInt16 | Int16 | Float16 | BFloat16 -> P (order_key uint16 x)
  | UInt32 | Int32 | Float32 -> P (order_key uint32 x)
  | UInt64 | Int64 | Float64 -> P (order_key uint64 x)
  | Complex64 | Complex128 -> err op "complex numbers have no order"

(* [rows op r] is the rows of [r] prepared to be told apart in rounds of up to
   64 bytes of keys, through [op]: [round ~ordered ~seen ~active ~m ~longest]
   is the rows [active], [m] of them, every row in order for [None], from
   their [seen]th key, as [e] keys in uint64 words, keys past a row's end zero,
   then [count], the number of keys of the round each row holds, plus one if it
   holds more, which separates a row from its prefixes, and [rem], the keys
   each holds from the [seen]th. [longest] is the most keys any active row
   holds from there. Words compare as their rows only when [ordered], which
   makes a word's first key its most significant. *)
let rows op r =
  let n = length r in
  let values, offsets = elements r in
  let (P keys) = keys op values in
  let w = Nx_dtype.itemsize (dtype keys) in
  (* Every window of up to 64 bytes from an element of a row lies in
     [keys]. *)
  let keys = pad [| (0, 64 / w) |] (Nx_dtype.zero (dtype keys)) keys in
  let start = shrink [| (0, n) |] offsets in
  let len = sub (shrink [| (1, n + 1) |] offsets) start in
  let round ~ordered ~seen ~active ~m ~longest =
    let width =
      List.find_opt (fun b -> b >= longest * w) [ 8; 16; 32 ]
      |> Option.value ~default:64
    in
    let e = width / w and q = 8 / w and words = width / 8 in
    let of_active x =
      match active with None -> x | Some active -> take ~indices:active x
    in
    let rem = sub_s (of_active len) (Int64.of_int seen) in
    let count = minimum_s rem (Int64.of_int (e + 1)) in
    let chunk =
      take ~axis:0
        ~indices:(add_s (of_active start) (Int64.of_int seen))
        (sliding_window ~window:e keys)
    in
    (* Keys past a row's end are zero. Ordered, a word's first key is its most
       significant, so a little-endian machine reverses each word's keys. *)
    let position = create Int64 [| words; q |] (Array.init e Int64.of_int) in
    let chunk = reshape [| m; words; q |] chunk in
    let chunk, position =
      if Sys.big_endian || not ordered then (chunk, position)
      else (flip ~axes:[ 2 ] chunk, flip ~axes:[ 1 ] position)
    in
    let chunk =
      where
        (less
           (reshape [| 1; words; q |] position)
           (reshape [| m; 1; 1 |] count))
        chunk (zeros_like chunk)
    in
    (e, reshape [| m; words |] (bitcast UInt64 chunk), count, rem)
  in
  let read =
    read ~by:op
      (concatenate ~axis:0
         [ reshape [| 1 |] (max len); reshape [| 1 |] (min len) ])
  in
  (round, Int64.to_int read.(0), Int64.to_int read.(1))

let column m x = reshape [| m; 1 |] (bitcast UInt64 x)

(* [ranks op r] is the dense rank of each row of [r] in the sort order, a prefix
   first, and the number of distinct rows. A round ranks the rows still to tell
   apart by their rank so far, their words and their count. A row is done once
   it is alone in its class or has no keys left. Each round reads once, through
   [op]. *)
let ranks op r =
  let n = length r in
  if n = 0 then (full_as r.offsets [| 0 |] 0L, 0)
  else
    let round_keys, longest, shortest = rows op r in
    let rec round ~seen ~classes ~active ~m ~longest ~shortest rank =
      if m = 0 || longest = 0 then (rank, classes)
      else
        let e, words, count, rem =
          round_keys ~ordered:true ~seen ~active:(Some active) ~m ~longest
        in
        let row_rank = take ~indices:active rank in
        let key =
          concatenate ~axis:1
            ((if classes > 1 then [ column m row_rank ] else [])
            @ [ words ]
            @
            if shortest > e || shortest = longest then []
            else [ column m count ])
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
          read ~by:op
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
          where keep (sub (cumsum kept) kept) (scalar_like kept read.(0))
        in
        let active =
          scatter ~unique_indices:true ~axis:0 ~indices:place
            ~values:(take ~indices:perm active)
            (full_as kept [| m' |] 0L)
        in
        round ~seen:(seen + e)
          ~classes:(classes + Int64.to_int read.(3))
          ~active ~m:m'
          ~longest:(Int64.to_int read.(1))
          ~shortest:(Int64.to_int read.(2))
          rank
    in
    round ~seen:0 ~classes:1 ~active:(arange Int64 0 n 1) ~m:n ~longest
      ~shortest
      (full_as r.offsets [| n |] 0L)

let rank r = fst (ranks "Nx_ragged.rank" r)

(* A round groups the rows still to tell apart by their code so far, their
   words and their count, and codes each by its group, past every earlier
   round's codes. A row is done once no other shares its code or it has no
   keys left. Codes tell rows apart but number them in no order, so after
   rounds that refine codes, the rows are grouped once more by their code. *)
let ids r =
  let op = "Nx_ragged.ids" in
  let n = length r in
  let group x = Op.eval (Group { by = op; x }) in
  if n = 0 then full_as r.offsets [| 0 |] 0L
  else
    let round_keys, longest, shortest = rows op r in
    let rec round ~rounds ~seen ~base ~active ~m ~longest ~shortest code =
      if m = 0 || longest = 0 then (rounds, code)
      else
        let e, words, count, rem =
          round_keys ~ordered:false ~seen ~active ~m ~longest
        in
        let key =
          match
            (match active with
            | Some active -> [ column m (take ~indices:active code) ]
            | None -> [])
            @ [ words ]
            @
            if shortest > e || shortest = longest then []
            else [ column m count ]
          with
          | [ words ] -> words
          | columns -> concatenate ~axis:1 columns
        in
        let g = group key in
        let code =
          match active with
          | None -> g
          | Some active ->
              scatter ~unique_indices:true ~axis:0 ~indices:active
                ~values:(add_s g (Int64.of_int base))
                code
        in
        if longest <= e then (rounds + 1, code)
        else
          (* Rows still to tell apart: with keys past the round, sharing their
             code. *)
          let shared =
            greater_s
              (take ~indices:g
                 (reduce_segments `Add ~segments:m g (full_as g [| m |] 1L)))
              1L
          in
          let left = sub_s rem (Int64.of_int e) in
          let keep = logical_and (greater_s left 0L) shared in
          let kept = cast Int64 keep in
          let read =
            read ~by:op
              (concatenate ~axis:0
                 (List.map
                    (fun x -> reshape [| 1 |] x)
                    [
                      sum kept;
                      max (where keep left (zeros_like left));
                      min (where keep left (full_like left Int64.max_int));
                    ]))
          in
          let m' = Int64.to_int read.(0) in
          let place =
            where keep (sub (cumsum kept) kept) (scalar_like kept read.(0))
          in
          let active =
            scatter ~unique_indices:true ~axis:0 ~indices:place
              ~values:
                (match active with
                | Some active -> active
                | None -> arange Int64 0 n 1)
              (full_as kept [| m' |] 0L)
          in
          round ~rounds:(rounds + 1) ~seen:(seen + e) ~base:(base + m)
            ~active:(Some active) ~m:m'
            ~longest:(Int64.to_int read.(1))
            ~shortest:(Int64.to_int read.(2))
            code
    in
    let rounds, code =
      round ~rounds:0 ~seen:0 ~base:0 ~active:None ~m:n
        ~longest ~shortest
        (full_as r.offsets [| n |] 0L)
    in
    if rounds <= 1 then code else group (column n code)

let take ~indices r =
  if ndim indices <> 1 then
    err "Nx_ragged.take" "indices of shape %s, not 1-D" (shape_string indices);
  let k = dim 0 indices and l = length r in
  let lens = take ~indices (lengths r) in
  let ends = cumsum lens in
  let offsets = pad [| (1, 0) |] 0L ends in
  let total =
    if k = 0 then 0L
    else
      (* With every length in [0, 2^63), the first running total past int64's
         range is negative. *)
      let read =
        read ~by:"Nx_ragged.take"
          (concatenate ~axis:0 [ last ends; reshape [| 1 |] (min ends) ])
      in
      if read.(1) < 0L then
        err "Nx_ragged.take" "the rows' lengths sum past int64's range";
      read.(0)
  in
  if total = 0L then
    { offsets; values = take ~axis:0 ~indices:(zeros Int64 [| 0 |]) r.values }
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
           (full_as indices [| Int64.to_int total |] 1L))
    in
    { offsets; values = take ~axis:0 ~indices:positions r.values }

let sub r ~offset ~length:k =
  let l = length r in
  if offset < 0 || k < 0 || offset + k > l then
    err "Nx_ragged.sub" "rows %d to %d of %d rows" offset (offset + k) l;
  { r with offsets = shrink [| (offset, offset + k + 1) |] r.offsets }

let concat = function
  | [] -> invalid_arg "Nx_ragged.concat: no ragged array"
  | [ r ] -> r
  | r :: _ as rs ->
      let cell r = Array.sub (shape r.values) 1 (ndim r.values - 1) in
      List.iter
        (fun r' ->
          if cell r' <> cell r then
            err "Nx_ragged.concat" "cells of shape %s and %s"
              (Format.asprintf "%a" pp_shape (cell r))
              (Format.asprintf "%a" pp_shape (cell r')))
        rs;
      let bounds =
        read ~by:"Nx_ragged.concat"
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
    err "Nx_ragged.map" "f maps values of shape %s to shape %s"
      (shape_string r.values) (shape_string values);
  { r with values }
