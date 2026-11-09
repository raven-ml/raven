(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Ragged arrays, against a model of their rows: each row is the cells of its
   values from one offset to the next, flattened. Values hold distinct elements,
   and rows before the first offset and after the last. *)

open Windtrap
open Nx_support

let invalid ~by f = raises_match (Exn.invalid_arg ~sub:(by ^ ": ")) f

let pp_int64s ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map Int64.to_string a)))

(* [pre] rows before the first offset, rows of [lengths], [post] rows after;
   each row of [width] elements, or a scalar cell when [width] is 0. *)
type drawn = { pre : int; lengths : int array; post : int; width : int }

let pp_drawn ppf d =
  Format.fprintf ppf "rows %a after %d, before %d, cells of %d" pp_ints
    d.lengths d.pre d.post d.width

let rows_of_values d = d.pre + Array.fold_left ( + ) 0 d.lengths + d.post
let cell d = Int.max 1 d.width

let drawn =
  Gen.with_pp pp_drawn
    Gen.(
      let* pre = int_range 0 2 in
      let* lengths =
        array ~size:(int_range 0 6)
          (frequency [ (1, constant 0); (1, constant 1); (3, int_range 2 4) ])
      in
      let* post = int_range 0 2 in
      let+ width = int_range 0 2 in
      { pre; lengths; post; width })

(* What a law over drawn rows needs to reach. *)
let cover_drawn d =
  cover "no row" (Array.length d.lengths = 0);
  cover "an empty row" (Array.mem 0 d.lengths);
  cover "a row of one cell" (Array.mem 1 d.lengths);
  cover "values before the first offset" (d.pre > 0);
  cover "values after the last offset" (d.post > 0);
  cover "cells of several elements" (d.width > 1)

let values d =
  let n = rows_of_values d in
  let flat =
    Nx.create Nx.int32 [| n * cell d |] (Array.init (n * cell d) Int32.of_int)
  in
  if d.width = 0 then flat else Nx.reshape [| n; d.width |] flat

let offsets d =
  let o = Array.make (Array.length d.lengths + 1) (Int64.of_int d.pre) in
  Array.iteri
    (fun i l -> o.(i + 1) <- Int64.add o.(i) (Int64.of_int l))
    d.lengths;
  o

let int64s a = Nx.create Nx.int64 [| Array.length a |] a
let ragged d = Nx_ragged.v ~offsets:(int64s (offsets d)) (values d)

(* The rows of the model: row [r] is elements [offsets.{r} * cell] to
   [offsets.{r + 1} * cell] of the values. *)
let model d =
  let o = offsets d in
  Array.init (Array.length d.lengths) (fun r ->
      let lo = Int64.to_int o.(r) * cell d in
      Array.init (d.lengths.(r) * cell d) (fun k -> Int32.of_int (lo + k)))

(* The rows of [r], read through its offsets and values. *)
let rows r =
  let o = Nx.to_array (Nx_ragged.offsets r) and v = Nx_ragged.values r in
  let w = if Nx.ndim v = 1 then 1 else Nx.numel v / Int.max 1 (Nx.dim 0 v) in
  let flat = Nx.to_array v in
  Array.init (Nx_ragged.length r) (fun i ->
      let lo = Int64.to_int o.(i) and hi = Int64.to_int o.(i + 1) in
      Array.sub flat (lo * w) ((hi - lo) * w))

let same_rows = array (array int32)

(* The values hold exactly the rows, from offset 0. *)
let exactly_rows r =
  let o = Nx.to_array (Nx_ragged.offsets r) in
  equal ~msg:"first offset" int64 0L o.(0);
  equal ~msg:"last offset" int64
    (Int64.of_int (Nx.dim 0 (Nx_ragged.values r)))
    o.(Array.length o - 1)

(* Making *)

let making =
  group "making"
    [
      prop "v cuts the values at the offsets" drawn (fun d ->
          cover_drawn d;
          let r = ragged d in
          equal same_rows (model d) (rows r);
          equal int (Array.length d.lengths) (Nx_ragged.length r);
          equal (array int64)
            (Array.map Int64.of_int d.lengths)
            (Nx.to_array (Nx_ragged.lengths r));
          equal (array int64) (offsets d) (Nx.to_array (Nx_ragged.offsets r));
          equal (array int32)
            (Nx.to_array (values d))
            (Nx.to_array (Nx_ragged.values r)));
      test "v takes offsets from 0 to the last row of the values" (fun () ->
          equal int 2
            (Nx_ragged.length
               (Nx_ragged.v
                  ~offsets:(int64s [| 0L; 0L; 3L |])
                  (Nx.zeros Nx.int32 [| 3 |]))));
      cases "v refuses offsets that do not cut the values" ~id:fst
        [
          ("offsets of two axes", (Nx.zeros Nx.int64 [| 1; 2 |], [| 3 |]));
          ("no offset", (Nx.zeros Nx.int64 [| 0 |], [| 3 |]));
          ("an offset below 0", (int64s [| -1L; 2L |], [| 3 |]));
          ("offsets that decrease", (int64s [| 0L; 2L; 1L |], [| 3 |]));
          ("an offset past the values", (int64s [| 0L; 4L |], [| 3 |]));
          ( "an offset of int64's least",
            (int64s [| 0L; Int64.min_int |], [| 3 |]) );
          ( "an offset of int64's greatest",
            (int64s [| Int64.max_int |], [| 3 |]) );
          ("0-D values", (int64s [| 0L |], [||]));
        ]
        (fun (_, (offsets, shape)) ->
          invalid ~by:"Nx_ragged.v" (fun () ->
              Nx_ragged.v ~offsets (Nx.zeros Nx.int32 shape)));
      prop "of_lengths cuts rows of the lengths from the first value" drawn
        (fun d ->
          cover "no row" (Array.length d.lengths = 0);
          cover "an empty row" (Array.mem 0 d.lengths);
          let d = { d with pre = 0 } in
          let r =
            Nx_ragged.of_lengths
              (int64s (Array.map Int64.of_int d.lengths))
              (values d)
          in
          equal same_rows (model d) (rows r));
      cases "of_lengths refuses lengths that do not cut the values" ~id:fst
        [
          ("lengths of two axes", (Nx.zeros Nx.int64 [| 1; 1 |], [| 3 |]));
          ("a negative length", (int64s [| 2L; -1L; 1L |], [| 3 |]));
          ("lengths past the values", (int64s [| 2L; 2L |], [| 3 |]));
          ( "lengths that sum past int64",
            (int64s [| Int64.max_int; Int64.max_int; 2L |], [| 3 |]) );
          ("a length of int64's least", (int64s [| Int64.min_int |], [| 3 |]));
          ("0-D values", (int64s [| 0L |], [||]));
        ]
        (fun (_, (lengths, shape)) ->
          invalid ~by:"Nx_ragged.of_lengths" (fun () ->
              Nx_ragged.of_lengths lengths (Nx.zeros Nx.int32 shape)));
    ]

(* Strings *)

let every_byte = String.init 256 Char.chr

(* Arrays of strings, some holding a NUL byte, which few strings of random bytes
   do. *)
let strings =
  let open Gen in
  let with_nul =
    let+ a = string and+ b = string in
    a ^ "\000" ^ b
  in
  array ~size:(int_range 0 6)
    (frequency [ (3, string); (1, with_nul); (1, constant "") ])

let strings_of_bytes =
  group "strings"
    [
      prop "to_strings reads back the strings of_strings makes"
        ~examples:[ [||]; [| "" |]; [| ""; "a"; "" |]; [| every_byte |] ]
        strings
        (fun ss ->
          cover "no string" (Array.length ss = 0);
          cover "an empty string" (Array.exists (String.equal "") ss);
          cover "a NUL byte"
            (Array.exists (fun s -> String.contains s '\000') ss);
          cover "a byte above 127"
            (Array.exists (String.exists (fun c -> Char.code c > 127)) ss);
          equal (array string) ss
            (Nx_ragged.to_strings (Nx_ragged.of_strings ss)));
      prop "of_strings holds each string's bytes as a row, from offset 0"
        strings (fun ss ->
          let r = Nx_ragged.of_strings ss in
          exactly_rows r;
          equal (array int64)
            (Array.map (fun s -> Int64.of_int (String.length s)) ss)
            (Nx.to_array (Nx_ragged.lengths r));
          equal (array int)
            (List.concat_map
               (fun s -> List.init (String.length s) (fun k -> Char.code s.[k]))
               (Array.to_list ss)
            |> Array.of_list)
            (Nx.to_array (Nx_ragged.values r)));
      prop "to_strings reads only the rows of a sub" strings (fun ss ->
          let n = Array.length ss in
          let offset = n / 3 and length = n / 2 in
          equal (array string)
            (Array.sub ss offset length)
            (Nx_ragged.to_strings
               (Nx_ragged.sub (Nx_ragged.of_strings ss) ~offset ~length)));
      test "to_strings refuses values with cells" (fun () ->
          invalid ~by:"Nx_ragged.to_strings" (fun () ->
              Nx_ragged.to_strings
                (Nx_ragged.v
                   ~offsets:(int64s [| 0L; 1L |])
                   (Nx.zeros Nx.uint8 [| 1; 2 |]))));
    ]

(* Grouping by ids *)

let grouped =
  Gen.with_pp
    (fun ppf (segments, ids, width) ->
      Format.fprintf ppf "%d segments, ids %a, cells of %d" segments pp_int64s
        ids width)
    Gen.(
      let* segments = int_range 0 4 in
      let* ids =
        array ~size:(int_range 0 10)
          (frequency
             [
               (6, map Int64.of_int (int_range (-2) (segments + 2)));
               ( 1,
                 of_list
                   ~pp:(fun ppf -> Format.fprintf ppf "%Ld")
                   [ Int64.min_int; Int64.max_int ] );
             ])
      in
      let+ width = int_range 0 2 in
      (segments, ids, width))

let in_range segments id = id >= 0L && id < Int64.of_int segments

let by_ids =
  let x ids width =
    values { pre = Array.length ids; lengths = [||]; post = 0; width }
  in
  group "of_ids"
    [
      prop "of_ids gathers the rows of each id in their order" grouped
        (fun (segments, ids, width) ->
          cover "no segment" (segments = 0);
          cover "no id" (Array.length ids = 0);
          cover "an id out of range"
            (Array.exists (fun i -> not (in_range segments i)) ids);
          cover "an id of int64's extremes"
            (Array.exists (fun i -> i = Int64.min_int || i = Int64.max_int) ids);
          let w = Int.max 1 width in
          let row i = Array.init w (fun k -> Int32.of_int ((i * w) + k)) in
          let expected =
            Array.init segments (fun s ->
                Array.concat
                  (List.filter_map
                     (fun (i, id) ->
                       if id = Int64.of_int s then Some (row i) else None)
                     (List.mapi (fun i id -> (i, id)) (Array.to_list ids))))
          in
          let r = Nx_ragged.of_ids ~segments (int64s ids) (x ids width) in
          equal int segments (Nx_ragged.length r);
          equal same_rows expected (rows r));
      prop "of_ids keeps the dropped rows after the last offset" grouped
        (fun (segments, ids, width) ->
          let w = Int.max 1 width in
          let r = Nx_ragged.of_ids ~segments (int64s ids) (x ids width) in
          let v = Nx_ragged.values r in
          equal int (Array.length ids) (Nx.dim 0 v);
          let kept = Nx.to_array (Nx_ragged.offsets r) in
          let kept = Int64.to_int kept.(Array.length kept - 1) in
          let dropped =
            List.filter_map
              (fun (i, id) -> if in_range segments id then None else Some i)
              (List.mapi (fun i id -> (i, id)) (Array.to_list ids))
          in
          let tail =
            Array.sub (Nx.to_array v) (kept * w) ((Array.length ids - kept) * w)
          in
          equal (list int) dropped
            (List.init
               (Array.length tail / w)
               (fun j -> Int32.to_int tail.(j * w) / w)));
      cases "of_ids refuses ids that do not match the rows" ~id:fst
        [
          ( "negative segments",
            fun () ->
              Nx_ragged.of_ids ~segments:(-1) (int64s [| 0L |])
                (Nx.zeros Nx.int32 [| 1 |]) );
          ( "ids of two axes",
            fun () ->
              Nx_ragged.of_ids ~segments:1
                (Nx.zeros Nx.int64 [| 1; 1 |])
                (Nx.zeros Nx.int32 [| 1 |]) );
          ( "an id per row and one more",
            fun () ->
              Nx_ragged.of_ids ~segments:1
                (int64s [| 0L; 0L |])
                (Nx.zeros Nx.int32 [| 1 |]) );
          ( "0-D values",
            fun () ->
              Nx_ragged.of_ids ~segments:1 (int64s [||])
                (Nx.zeros Nx.int32 [||]) );
        ]
        (fun (_, f) -> invalid ~by:"Nx_ragged.of_ids" f);
    ]

(* Transforming *)

let range =
  Gen.with_pp
    (fun ppf (d, o, n) ->
      Format.fprintf ppf "rows %d to %d of %a" o (o + n) pp_drawn d)
    Gen.(
      let* d = drawn in
      let rows = Array.length d.lengths in
      let* o = int_range 0 rows in
      let+ n = int_range 0 (rows - o) in
      (d, o, n))

(* Indices of any row, outside them, and at the extremes; a quarter of the
   draws put two empty rows together between non-empty ones, where both
   rows start at one offset, which few draws of four or more indices reach
   otherwise. *)
let taken =
  Gen.with_pp
    (fun ppf (d, i) -> Format.fprintf ppf "%a at %a" pp_drawn d pp_int64s i)
    Gen.(
      let* d = drawn in
      let n = Array.length d.lengths in
      let index =
        frequency
          [
            (6, map Int64.of_int (int_range (-2) (n + 1)));
            ( 1,
              of_list
                ~pp:(fun ppf -> Format.fprintf ppf "%Ld")
                [ Int64.min_int; Int64.max_int ] );
          ]
      in
      let rows p = List.filter (fun k -> p d.lengths.(k)) (List.init n Fun.id) in
      let full = rows (( < ) 0) in
      let empty = -1 :: n :: rows (( = ) 0) in
      let pick l = map Int64.of_int (of_list l) in
      let between =
        let* a = pick full in
        let* e = pick empty in
        let* e' = pick empty in
        let+ b = pick full in
        [| a; e; e'; b |]
      in
      let+ i =
        frequency
          ((3, array ~size:(int_range 0 8) index)
          :: (if full = [] then [] else [ (1, between) ]))
      in
      (d, i))

(* Values of a dtype, cells and bits a take must keep: bytes, int4, and
   float64 with -0 and NaNs of distinct payloads, in cells of none or
   [[2; 3]]; their elements as int64 bits in C order. *)
type elt = Bytes | Int4 | Float64

type elts = Elt : ('v, 's) Nx.dtype * ('v -> int64) * (int -> 'v) -> elts

let elts = function
  | Bytes -> Elt (Nx.uint8, Int64.of_int, fun i -> (i * 7) land 255)
  | Int4 -> Elt (Nx.int4, Int64.of_int, fun i -> (i mod 16) - 8)
  | Float64 ->
      let f i =
        match i mod 4 with
        | 0 -> -0.
        | 1 ->
            Int64.float_of_bits (Int64.add 0x7FF8000000000001L (Int64.of_int i))
        | _ -> Float.of_int i
      in
      Elt (Nx.float64, Int64.bits_of_float, f)

(* Rows of [d] over values of [e] in cells [cells], a [sub] of them from
   row [o], taken at [i]. *)
let typed_take =
  Gen.with_pp
    (fun ppf (d, i, e, cells, o) ->
      Format.fprintf ppf "%a at %a, %s cells %a from row %d" pp_drawn d
        pp_int64s i
        (match e with Bytes -> "bytes" | Int4 -> "int4" | Float64 -> "float64")
        pp_ints cells o)
    Gen.(
      let* d, i = taken in
      let* e = of_list [ Bytes; Int4; Float64 ] in
      let* cells = of_list [ [||]; [| 2; 3 |] ] in
      let+ o = int_range 0 (Array.length d.lengths) in
      (d, i, e, cells, o))

let law_typed_take ((d : drawn), i, e, cells, o) =
  let (Elt (dt, bits, make)) = elts e in
  let per = Array.fold_left ( * ) 1 cells in
  let n = rows_of_values d in
  let flat = Array.init (n * per) make in
  let values = Nx.create dt (Array.append [| n |] cells) flat in
  let all = Nx_ragged.v ~offsets:(int64s (offsets d)) values in
  let rows = Array.length d.lengths - o in
  let r = Nx_ragged.sub all ~offset:o ~length:rows in
  cover "a sub from a later row" (o > 0 && rows > 0);
  cover "an index at the sub's last row"
    (Array.exists (fun k -> k = Int64.of_int (rows - 1)) i);
  cover "an index past the sub's rows"
    (Array.exists (fun k -> k >= Int64.of_int rows) i);
  let off = offsets d in
  let row k =
    if k < 0L || k >= Int64.of_int rows then [||]
    else
      let j = Int64.to_int k + o in
      let lo = Int64.to_int off.(j) * per
      and hi = Int64.to_int off.(j + 1) * per in
      Array.map bits (Array.sub flat lo (hi - lo))
  in
  let expected = Array.concat (Array.to_list (Array.map row i)) in
  let t = Nx_ragged.take ~indices:(int64s i) r in
  let lengths = Array.map (fun k -> Array.length (row k) / per) i in
  equal ~msg:"lengths" (array int)
    lengths
    (Array.map Int64.to_int (Nx.to_array (Nx_ragged.lengths t)));
  equal ~msg:"bits" (array int64) expected
    (Array.map bits (Nx.to_array (Nx_ragged.values t)));
  exactly_rows t

(* A take by a permutation, then by its inverse, gives the rows back, their
   values from offset 0. *)
let permuted =
  Gen.with_pp
    (fun ppf (d, p) -> Format.fprintf ppf "%a by %a" pp_drawn d pp_ints p)
    Gen.(
      let* d = drawn in
      let+ p =
        permutation ~pp:Format.pp_print_int
          (List.init (Array.length d.lengths) Fun.id)
      in
      (d, Array.of_list p))

let law_round_trip (d, p) =
  let inv = Array.make (Array.length p) 0 in
  Array.iteri (fun i k -> inv.(k) <- i) p;
  let at a = int64s (Array.map Int64.of_int a) in
  let r = ragged d in
  let back =
    Nx_ragged.take ~indices:(at inv) (Nx_ragged.take ~indices:(at p) r)
  in
  cover "values before the first offset" (d.pre > 0);
  equal same_rows (rows r) (rows back);
  exactly_rows back

let transforming =
  group "transforming"
    [
      prop "sub is the rows of the range" range (fun (d, o, n) ->
          cover "no row" (n = 0);
          cover "a range from a later row" (o > 0);
          equal same_rows
            (Array.sub (model d) o n)
            (rows (Nx_ragged.sub (ragged d) ~offset:o ~length:n)));
      prop "sub keeps the values" range (fun (d, o, n) ->
          let r = ragged d in
          let s = Nx_ragged.sub r ~offset:o ~length:n in
          equal bool true (Nx_ragged.values s == Nx_ragged.values r));
      cases "sub refuses a range outside the rows" ~id:fst
        [
          ("a negative offset", (-1, 1));
          ("a negative length", (0, -1));
          ("one row past the end", (1, 2));
        ]
        (fun (_, (offset, length)) ->
          invalid ~by:"Nx_ragged.sub" (fun () ->
              Nx_ragged.sub
                (Nx_ragged.of_lengths
                   (int64s [| 1L; 1L |])
                   (Nx.zeros Nx.int32 [| 2 |]))
                ~offset ~length));
      prop "take is the row at each index, and an empty row outside" taken
        (fun (d, i) ->
          let n = Int64.of_int (Array.length d.lengths) in
          cover "no index" (Array.length i = 0);
          cover "an index outside the rows"
            (Array.exists (fun k -> k < 0L || k >= n) i);
          cover "an index of a non-empty row"
            (Array.exists
               (fun k -> k >= 0L && k < n && d.lengths.(Int64.to_int k) > 0)
               i);
          let len k =
            if k >= 0L && k < n then d.lengths.(Int64.to_int k) else 0
          in
          let count k =
            Array.fold_left (fun c j -> c + Bool.to_int (j = k)) 0 i
          in
          cover "an index repeated" (Array.exists (fun k -> count k > 1) i);
          cover "an empty row before a non-empty one"
            (Array.length i > 1 && len i.(0) = 0
            && Array.exists (fun k -> len k > 0) i);
          (* Rows that start at the total write their change nowhere, and
             empty rows that start together add their changes up. *)
          let lens = Array.map len i in
          let total = Array.fold_left ( + ) 0 lens in
          let last = Array.length lens - 1 in
          cover "empty rows last, after a non-empty one"
            (total > 0 && last >= 0 && lens.(last) = 0);
          cover "two empty rows together between non-empty ones"
            (List.exists
               (fun j ->
                 lens.(j) = 0
                 && lens.(j + 1) = 0
                 && Array.exists (( < ) 0) (Array.sub lens 0 j)
                 && Array.exists (( < ) 0)
                      (Array.sub lens (j + 2) (last - j - 1)))
               (List.init (Int.max 0 last) Fun.id));
          let m = model d in
          let expected =
            Array.map
              (fun k ->
                if k >= 0L && k < Int64.of_int (Array.length m) then
                  m.(Int64.to_int k)
                else [||])
              i
          in
          let r = Nx_ragged.take ~indices:(int64s i) (ragged d) in
          equal same_rows expected (rows r);
          exactly_rows r);
      prop
        "take keeps each row's bits over any dtype, cells and sub, and an \
         empty row outside"
        typed_take law_typed_take;
      prop "a take by a permutation then its inverse gives the rows back"
        permuted law_round_trip;
      cases "take refuses what it cannot take" ~id:fst
        [
          ( "indices of two axes",
            fun () ->
              Nx_ragged.take
                ~indices:(Nx.zeros Nx.int64 [| 1; 1 |])
                (Nx_ragged.of_lengths (int64s [| 1L |])
                   (Nx.zeros Nx.int32 [| 1 |])) );
          ( "rows whose lengths sum past int64",
            fun () ->
              (* 1025 rows of length 2^53, the most elements an array holds. *)
              let n = Int.shift_left 1 53 in
              Nx_ragged.take
                ~indices:(Nx.zeros Nx.int64 [| 1025 |])
                (Nx_ragged.v
                   ~offsets:(int64s [| 0L; Int64.of_int n |])
                   (Nx.zeros Nx.int32 [| n; 0 |])) );
        ]
        (fun (_, f) -> invalid ~by:"Nx_ragged.take" f);
      prop "concat is the rows one after the other"
        (Gen.with_pp
           (Format.pp_print_list pp_drawn)
           Gen.(
             let* width = int_range 0 2 in
             list ~size:(int_range 2 4) (map (fun d -> { d with width }) drawn)))
        (fun ds ->
          cover "a part with values outside its offsets"
            (List.exists (fun d -> d.pre > 0 || d.post > 0) ds);
          cover "a part of no row"
            (List.exists (fun d -> Array.length d.lengths = 0) ds);
          let r = Nx_ragged.concat (List.map ragged ds) in
          equal same_rows (Array.concat (List.map model ds)) (rows r);
          exactly_rows r);
      test "concat returns a single ragged array as it is" (fun () ->
          let r =
            Nx_ragged.of_lengths (int64s [| 1L |]) (Nx.zeros Nx.int32 [| 2 |])
          in
          equal bool true (Nx_ragged.concat [ r ] == r));
      cases "concat refuses what has no rows in common" ~id:fst
        [
          ("no ragged array", fun () -> Nx_ragged.concat []);
          ( "cells of different shapes",
            fun () ->
              Nx_ragged.concat
                [
                  Nx_ragged.of_lengths (int64s [| 1L |])
                    (Nx.zeros Nx.int32 [| 1 |]);
                  Nx_ragged.of_lengths (int64s [| 1L |])
                    (Nx.zeros Nx.int32 [| 1; 2 |]);
                ] );
        ]
        (fun (_, f) -> invalid ~by:"Nx_ragged.concat" f);
      prop "map keeps the offsets and maps the values" drawn (fun d ->
          let r = ragged d in
          let m = Nx_ragged.map (fun v -> Nx.cast Nx.float64 (Nx.neg v)) r in
          equal bool true (Nx_ragged.offsets m == Nx_ragged.offsets r);
          equal (array float_exact)
            (Nx.to_array (Nx.cast Nx.float64 (Nx.neg (values d))))
            (Nx.to_array (Nx_ragged.values m)));
      cases "map refuses a function that changes the rows of the values" ~id:fst
        [
          ("one row fewer", fun v -> Nx.slice [ R (1, Nx.dim 0 v) ] v);
          ("a 0-D result", fun v -> Nx.sum v);
        ]
        (fun (_, f) ->
          invalid ~by:"Nx_ragged.map" (fun () ->
              Nx_ragged.map f
                (Nx_ragged.of_lengths (int64s [| 1L |])
                   (Nx.zeros Nx.int32 [| 2 |]))));
    ]

(* Quantiles of rows *)

let pp_float ppf x = Format.fprintf ppf "%.17g" x

let float_values =
  Gen.frequency
    [
      (6, Gen.map float_of_int (Gen.int_range (-3) 3));
      (2, Gen.float_range (-1e3) 1e3);
      ( 1,
        Gen.of_list ~pp:pp_float
          [ Float.nan; -0.; 0.; Float.infinity; Float.neg_infinity ] );
    ]

let probabilities =
  Gen.array ~size:(Gen.int_range 0 4)
    (Gen.frequency
       [
         (3, Gen.float_range 0. 1.);
         (1, Gen.of_list ~pp:pp_float [ 0.; 0.25; 0.5; 1.; 1. /. 3. ]);
       ])

(* Float rows: their offsets, and values that hold rows outside them. *)
let float_rows =
  let pp_floats ppf a =
    Format.fprintf ppf "[%s]"
      (String.concat "; "
         (Array.to_list (Array.map (Format.asprintf "%a" pp_float) a)))
  in
  Gen.with_pp
    (fun ppf (d, xs, qs) ->
      Format.fprintf ppf "%a of %a at %a" pp_drawn d pp_floats xs pp_floats qs)
    Gen.(
      let* d = drawn in
      let* xs =
        array ~size:(constant (rows_of_values d * cell d)) float_values
      in
      let+ qs = probabilities in
      (d, xs, qs))

type float_dtype = F : (float, 's) Nx.dtype -> float_dtype

let float_dtypes =
  Nx.
    [
      F float64;
      F float32;
      F float16;
      F bfloat16;
      F float8_e4m3fn;
      F float8_e5m2;
      F float4_e2m1fn;
    ]

let row_quantiles =
  group "quantile"
    [
      prop "each row's quantiles are Nx.quantile's, and an empty row's NaN"
        float_rows (fun (d, xs, qs) ->
          cover_drawn d;
          cover "no probability" (Array.length qs = 0);
          cover "a NaN" (Array.exists Float.is_nan xs);
          List.iter
            (fun (F dt) ->
              let values =
                Nx.cast dt (Nx.create Nx.float64 [| Array.length xs |] xs)
              in
              let values =
                if d.width = 0 then values
                else Nx.reshape [| rows_of_values d; d.width |] values
              in
              let r = Nx_ragged.v ~offsets:(int64s (offsets d)) values in
              let o = offsets d in
              let k = Array.length qs and n = Array.length d.lengths in
              let got = Nx_ragged.quantile qs r in
              equal ~msg:"shape" (array int) [| k; n |] (Nx.shape got);
              Array.iteri
                (fun i len ->
                  let expected =
                    if len = 0 then Nx.to_array (Nx.full dt [| k |] Float.nan)
                    else
                      Nx.to_array
                        (Nx.quantile qs
                           (Nx.slice
                              [ R (Int64.to_int o.(i), Int64.to_int o.(i + 1)) ]
                              values))
                  in
                  equal
                    ~msg:(Format.asprintf "%a row %d" Nx.pp_dtype dt i)
                    (array float_exact) expected
                    (Nx.to_array (Nx.slice [ A; I i ] got)))
                d.lengths)
            float_dtypes);
      cases
        "a fraction that rounds to zero beside an infinity gives the lower \
         statistic"
        ~id:fst
        [
          ("float32 [1; inf]", (Nx.float32, [| 1.; Float.infinity |], 1.));
          ( "float32 [-inf; 1]",
            (Nx.float32, [| Float.neg_infinity; 1. |], Float.neg_infinity) );
        ]
        (fun (_, (dt, xs, expected)) ->
          (* At float32, [2^-1074] rounds to zero, so the quantile is the first
             statistic, as Nx.quantile gives it. *)
          let q = Float.ldexp 1. (-1074) in
          let r =
            Nx_ragged.of_lengths (int64s [| 2L |]) (Nx.create dt [| 2 |] xs)
          in
          equal (array float_exact) [| expected |]
            (Nx.to_array (Nx_ragged.quantile [| q |] r));
          equal ~msg:"Nx.quantile" (array float_exact) [| expected |]
            (Nx.to_array (Nx.quantile [| q |] (Nx.create dt [| 2 |] xs))));
      cases "quantile refuses a probability outside [0, 1]" ~id:fst
        [ ("1.5", 1.5); ("-0.5", -0.5); ("NaN", Float.nan) ]
        (fun (_, q) ->
          invalid ~by:"Nx_ragged.quantile" (fun () ->
              Nx_ragged.quantile [| q |]
                (Nx_ragged.of_lengths (int64s [| 1L |])
                   (Nx.zeros Nx.float64 [| 1 |]))));
    ]

(* Ids and ranks of rows *)

(* Rows that share long prefixes: prefixes of three rows of up to 140 elements,
   each with a tail of up to two elements, over few distinct elements, then
   copies of some of them. *)
let shared_rows value =
  let open Gen in
  let* bases = list ~size:(constant 3) (array ~size:(int_range 0 140) value) in
  let bases = Array.of_list bases in
  (* No rows one draw in five: one in thirteen left the cover of no row
     unreached over some seeds' 100 cases. *)
  let* rows =
    array
      ~size:(frequency [ (1, constant 0); (4, int_range 1 12) ])
      (let* b = int_range 0 2 in
       let base = bases.(b) in
       let* cut =
         frequency
           [
             (3, int_range 0 (Array.length base));
             ( 1,
               map
                 (fun k -> Stdlib.min k (Array.length base))
                 (of_list ~pp:Format.pp_print_int
                    [ 7; 8; 9; 15; 16; 17; 63; 64; 65; 129 ]) );
           ]
       in
       let+ tail = array ~size:(int_range 0 2) value in
       Array.append (Array.sub base 0 cut) tail)
  in
  (* Copies of drawn rows: independent draws are rarely equal. *)
  let n = Array.length rows in
  if n = 0 then constant rows
  else
    let+ copies = list ~size:(int_range 0 2) (int_range 0 (n - 1)) in
    Array.append rows
      (Array.of_list (List.map (fun i -> Array.copy rows.(i)) copies))

(* The model: rows compared element by element in the sort order, a prefix
   first. *)
let compare_rows compare a b =
  let rec go i =
    if i = Array.length a then if i = Array.length b then 0 else -1
    else if i = Array.length b then 1
    else match compare a.(i) b.(i) with 0 -> go (i + 1) | c -> c
  in
  go 0

let model_ranks compare rows =
  let distinct = List.sort_uniq (compare_rows compare) (Array.to_list rows) in
  Array.map
    (fun r ->
      let rec index i = function
        | [] -> assert false
        | x :: xs ->
            if compare_rows compare r x = 0 then i else index (i + 1) xs
      in
      Int64.of_int (index 0 distinct))
    rows

(* [same_groups ids r] checks that [Nx_ragged.unique ~size r] groups the rows
   of [r] as [Nx.unique ~size] groups their [ids], numbered in order of first
   appearance: unsized, and sized to no group, to all but one, to all, and to
   two more. *)
let same_groups ids r =
  let k = Array.fold_left (fun k i -> Int.max k (Int64.to_int i + 1)) 0 ids in
  List.iter
    (fun size ->
      let at =
        match size with None -> "unsized" | Some l -> Printf.sprintf "size %d" l
      in
      let want : Nx.host Nx.groups = Nx.unique ?size (int64s ids) in
      let got : Nx.host Nx.groups = Nx_ragged.unique ?size r in
      let field name f =
        equal ~msg:(at ^ ", " ^ name) (array int64) (Nx.to_array (f want))
          (Nx.to_array (f got))
      in
      field "ids" (fun g -> g.ids);
      field "first" (fun g -> g.first);
      field "counts" (fun g -> g.counts);
      field "count" (fun g -> Nx.reshape [| 1 |] g.count))
    [ None; Some 0; Some (Int.max 0 (k - 1)); Some k; Some (k + 2) ]

let model_ids compare rows =
  let seen = ref [] in
  Array.map
    (fun r ->
      match
        List.find_opt (fun (x, _) -> compare_rows compare r x = 0) !seen
      with
      | Some (_, id) -> id
      | None ->
          let id = Int64.of_int (List.length !seen) in
          seen := (r, id) :: !seen;
          id)
    rows

(* [rows] cut from values that hold the elements [pre] before them and [post]
   after. *)
let ragged_of dtype (pre, rows, post) =
  let offsets = Array.make (Array.length rows + 1) (Array.length pre) in
  Array.iteri (fun i r -> offsets.(i + 1) <- offsets.(i) + Array.length r) rows;
  let offsets = Array.map Int64.of_int offsets in
  let values = Array.concat ((pre :: Array.to_list rows) @ [ post ]) in
  Nx_ragged.v ~offsets:(int64s offsets)
    (Nx.create dtype [| Array.length values |] values)

(* The sort order of floats: -0 before +0, every NaN equal and last. *)
let compare_float a b =
  match (Float.is_nan a, Float.is_nan b) with
  | true, true -> 0
  | true, false -> 1
  | false, true -> -1
  | false, false ->
      if a < b then -1
      else if a > b then 1
      else Bool.compare (Float.sign_bit b) (Float.sign_bit a)

type element =
  | E : {
      dtype : ('v, 's) Nx.dtype;
      values : 'v list;
      compare : 'v -> 'v -> int;
      pp : Format.formatter -> 'v -> unit;
    }
      -> element

let floats dtype values =
  E { dtype; values; compare = compare_float; pp = pp_float }

let ints dtype values =
  E { dtype; values; compare = Int.compare; pp = Format.pp_print_int }

let bools dtype =
  E
    {
      dtype;
      values = [ false; true ];
      compare = Bool.compare;
      pp = Format.pp_print_bool;
    }

(* Every dtype with an order, each with values at its extremes and around zero,
   signed zeros, NaN and the infinities where the format has them. *)
let elements =
  let wide =
    [ Float.nan; -0.; 0.; 1.; -1.; Float.infinity; Float.neg_infinity ]
  in
  Nx.
    [
      floats float64 wide;
      floats float32 wide;
      floats float16 wide;
      floats bfloat16 wide;
      floats float8_e4m3fn [ Float.nan; -0.; 0.; 1.; -1.; 448.; -448. ];
      floats float8_e5m2 wide;
      floats float4_e2m1fn [ -0.; 0.; 0.5; 1.; -1.; 6.; -6. ];
      E
        {
          dtype = int64;
          values = [ Int64.min_int; -1L; 0L; 1L; Int64.max_int ];
          compare = Int64.compare;
          pp = (fun ppf -> Format.fprintf ppf "%Ld");
        };
      E
        {
          dtype = uint64;
          values = [ 0L; 1L; Int64.max_int; Int64.min_int; -1L ];
          compare = Int64.unsigned_compare;
          pp = (fun ppf -> Format.fprintf ppf "%Lu");
        };
      E
        {
          dtype = int32;
          values = [ Int32.min_int; -1l; 0l; 1l; Int32.max_int ];
          compare = Int32.compare;
          pp = (fun ppf -> Format.fprintf ppf "%ld");
        };
      E
        {
          dtype = uint32;
          values = [ 0l; 1l; Int32.max_int; Int32.min_int; -1l ];
          compare = Int32.unsigned_compare;
          pp = (fun ppf -> Format.fprintf ppf "%lu");
        };
      ints int16 [ -32768; -1; 0; 1; 32767 ];
      ints uint16 [ 0; 1; 32768; 65535 ];
      ints int8 [ -128; -1; 0; 1; 127 ];
      ints uint8 [ 0; 1; 97; 255 ];
      ints int4 [ -8; -1; 0; 1; 7 ];
      ints uint4 [ 0; 1; 8; 15 ];
      bools bool;
      bools bit;
    ]

let identifying (E e) =
  let name = Nx.Dtype.name e.dtype in
  let value = Gen.of_list ~pp:e.pp e.values in
  let junk = Gen.array ~size:(Gen.int_range 0 3) value in
  let pp_row ppf r =
    Format.fprintf ppf "[%s]"
      (String.concat "; "
         (Array.to_list (Array.map (Format.asprintf "%a" e.pp) r)))
  in
  let drawn =
    Gen.with_pp
      (fun ppf (pre, rows, post) ->
        Format.fprintf ppf "@[<v>rows:@,%a@,after %a, before %a@]"
          (Format.pp_print_seq pp_row)
          (Array.to_seq rows) pp_row pre pp_row post)
      (Gen.triple junk (shared_rows value) junk)
  in
  let covered (_, rows, _) =
    let n = Array.length rows in
    let pairs p =
      List.exists
        (fun i ->
          List.exists
            (fun j -> i <> j && p rows.(i) rows.(j))
            (List.init n Fun.id))
        (List.init n Fun.id)
    in
    let prefix a b =
      Array.length a < Array.length b
      && compare_rows e.compare a (Array.sub b 0 (Array.length a)) = 0
    in
    cover "no row" (n = 0);
    cover "an empty row" (Array.exists (fun r -> Array.length r = 0) rows);
    cover "two equal rows" (pairs (fun a b -> compare_rows e.compare a b = 0));
    cover "a row that is another's proper prefix" (pairs prefix);
    cover "a row of more than 64 bytes of keys"
      (Array.exists
         (fun r -> Array.length r * Int.max 1 (Nx.Dtype.bits e.dtype / 8) > 64)
         rows)
  in
  (* The rows and their ragged array, and all but the first row through
     [sub]. *)
  let both check ((_, rows, _) as d) =
    covered d;
    let r = ragged_of e.dtype d in
    check rows r;
    let n = Array.length rows in
    if n > 0 then
      check (Array.sub rows 1 (n - 1)) (Nx_ragged.sub r ~offset:1 ~length:(n - 1))
  in
  [
    prop
      (name ^ " equal rows group as their ids, in order of first appearance")
      drawn
      (both (fun rows r -> same_groups (model_ids e.compare rows) r));
    prop
      (name ^ " rows' ranks are dense in the sort order, a prefix first")
      drawn
      (both (fun rows r ->
           equal (array int64) (model_ranks e.compare rows)
             (Nx.to_array (Nx_ragged.rank r))));
  ]

(* Texts packed into words, drawn three ways over bytes that hold NUL and
   invalid UTF-8, then with copies of some of them:
   - short texts alone, of at most 7 bytes, which take one word each, of one
     length or several;
   - families of texts of one length that share a prefix of three bases of 150
     bytes and differ in their last byte, cut around the words a text fills
     (7, 8 and 9 bytes, multiples of 8) and around the rounds of 64 bytes,
     so that rows split within the first, second and third round;
   - texts of at most 64 bytes with one of exactly 64, the most one round
     covers. *)
let packed_texts =
  let open Gen in
  let byte = of_list [ '\000'; 'a'; 'b'; '\x80'; '\xff' ] in
  let short =
    let* one = bool in
    let* l = int_range 0 7 in
    let size = if one then constant l else int_range 0 7 in
    array ~size:(int_range 2 8) (string_of ~size byte)
  in
  let families =
    let* bases = list ~size:(constant 3) (string_of ~size:(constant 150) byte) in
    let bases = Array.of_list bases in
    let family =
      let* base = map (Array.get bases) (int_range 0 2) in
      let* cut =
        of_list ~pp:Format.pp_print_int
          [ 0; 1; 6; 7; 8; 15; 16; 23; 31; 32; 47; 63; 64; 65; 80; 127; 128; 140 ]
      in
      let+ tails =
        list ~size:(int_range 1 3) (string_of ~size:(constant 1) byte)
      in
      List.map (fun t -> String.sub base 0 cut ^ t) tails
    in
    let+ fs = list ~size:(int_range 1 5) family in
    Array.of_list (List.concat fs)
  in
  let upto_64 =
    let+ texts =
      array ~size:(int_range 0 6) (string_of ~size:(int_range 0 63) byte)
    and+ full = string_of ~size:(constant 64) byte in
    Array.append texts [| full |]
  in
  let* texts = frequency [ (2, short); (3, families); (1, upto_64) ] in
  let n = Array.length texts in
  let+ copies = list ~size:(int_range 0 3) (int_range 0 (n - 1)) in
  Array.append texts (Array.of_list (List.map (Array.get texts) copies))

(* [first_appearance xs] numbers [xs] in order of first appearance. *)
let first_appearance (type a) (xs : a array) =
  let seen = Hashtbl.create 16 in
  Array.map
    (fun x ->
      match Hashtbl.find_opt seen x with
      | Some id -> id
      | None ->
          let id = Int64.of_int (Hashtbl.length seen) in
          Hashtbl.add seen x id;
          id)
    xs

let text_ids =
  prop "texts group in order of first appearance, numbered as their ranks"
    packed_texts (fun ss ->
      let has p = Array.exists p ss in
      let lengths p = has (fun s -> p (String.length s)) in
      let pairs p =
        let is = List.init (Array.length ss) Fun.id in
        List.exists
          (fun i -> List.exists (fun j -> i <> j && p ss.(i) ss.(j)) is)
          is
      in
      cover "an empty text" (lengths (Int.equal 0));
      cover "a text of 7 bytes" (lengths (Int.equal 7));
      cover "a text of 8 bytes" (lengths (Int.equal 8));
      cover "a text of 9 bytes" (lengths (Int.equal 9));
      cover "a text of a multiple of 8 bytes past 8"
        (lengths (fun l -> l > 8 && l mod 8 = 0));
      cover "a text past 64 bytes" (lengths (fun l -> l > 64));
      cover "two equal texts past 64 bytes"
        (pairs (fun a b -> String.length a > 64 && String.equal a b));
      cover "a text a proper prefix of another"
        (pairs (fun a b ->
             String.length a < String.length b
             && String.starts_with ~prefix:a b));
      cover "a NUL byte" (has (fun s -> String.contains s '\000'));
      cover "invalid UTF-8" (has (fun s -> not (String.is_valid_utf_8 s)));
      let longest =
        Array.fold_left (fun m s -> Int.max m (String.length s)) 0 ss
      in
      let one_word = longest <= 7 in
      cover "one word each, two equal rows" (one_word && pairs String.equal);
      cover "one word each, two distinct rows"
        (one_word && pairs (fun a b -> not (String.equal a b)));
      cover "two rows or more, all of one length"
        (Array.length ss >= 2
        && Array.for_all (fun s -> String.length s = String.length ss.(0)) ss);
      cover "a longest row of exactly 64 bytes" (longest = 64);
      (* Rows of one length that agree on their first [k] bytes and differ
         after them. *)
      let split k =
        pairs (fun a b ->
            String.length a = String.length b
            && String.length a > k
            && String.sub a 0 k = String.sub b 0 k
            && not (String.equal a b))
      in
      cover "rows split past 64 bytes" (split 64);
      cover "rows split in the third round" (split 128);
      let r = Nx_ragged.of_strings ss in
      same_groups (first_appearance ss) r;
      equal ~msg:"ranks numbered in order of first appearance" (array int64)
        (first_appearance (Nx.to_array (Nx_ragged.rank r)))
        (first_appearance ss))

(* NaNs of each payload and sign, as order keys make them one key. *)
let nan_ids =
  test "NaNs of any payload are one key in a row" (fun () ->
      let nans =
        Nx.bitcast Nx.float64
          (int64s
             [|
               0x7FF8000000000000L; 0xFFF8000000000001L; 0x7FF0000000000001L;
             |])
      in
      let nan i = Nx.slice [ R (i, i + 1) ] nans in
      let one = Nx.ones Nx.float64 [| 1 |] in
      (* Rows of one key, of two, and of ten, past a round's 64 bytes. *)
      let ten a b = List.init 9 (fun i -> nan (if i mod 2 = 0 then a else b)) in
      let rows =
        [
          [ nan 0 ];
          [ nan 1 ];
          [ nan 2; one ];
          [ nan 0; one ];
          ten 1 2 @ [ one ];
          ten 0 1 @ [ one ];
          [ one ];
        ]
      in
      let r =
        Nx_ragged.of_lengths
          (int64s
             (Array.of_list
                (List.map (fun r -> Int64.of_int (List.length r)) rows)))
          (Nx.concatenate ~axis:0 (List.concat rows))
      in
      same_groups [| 0L; 0L; 1L; 1L; 2L; 2L; 3L |] r)

let identities =
  group "unique and rank"
    (List.concat_map identifying elements
    @ [
        text_ids;
        nan_ids;
        test "cells of several elements compare element by element" (fun () ->
            let r =
              Nx_ragged.of_lengths
                (int64s [| 1L; 1L; 2L; 1L |])
                (Nx.create Nx.int32 [| 5; 2 |]
                   [| 1l; 2l; 1l; 3l; 1l; 2l; 0l; 0l; 1l; 2l |])
            in
            same_groups [| 0L; 1L; 2L; 0L |] r;
            equal (array int64) [| 0L; 2L; 1L; 0L |]
              (Nx.to_array (Nx_ragged.rank r)));
        test "unique and rank refuse complex values" (fun () ->
            let r =
              Nx_ragged.of_lengths (int64s [| 1L |])
                (Nx.zeros Nx.complex64 [| 1 |])
            in
            invalid ~by:"Nx_ragged.unique" (fun () -> Nx_ragged.unique r);
            invalid ~by:"Nx_ragged.rank" (fun () -> Nx_ragged.rank r));
        test "unique refuses a negative size" (fun () ->
            let r = Nx_ragged.of_strings [| "a" |] in
            invalid ~by:"Nx_ragged.unique" (fun () ->
                Nx_ragged.unique ~size:(-1) r));
      ])

(* Sets *)

let sets =
  let module M = (val Nx.devices [ Nx_support.memory 0 ]) in
  let on_set x = Nx.place M.on x in
  let placed r =
    Nx_ragged.v
      ~offsets:(on_set (Nx_ragged.offsets r))
      (on_set (Nx_ragged.values r))
  in
  let lies_on_set x =
    equal ~msg:"placement"
      (Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal)
      M.on (Nx.placement x)
  in
  group "sets"
    [
      test "a ragged array on a set computes there" (fun () ->
          let r = placed (Nx_ragged.of_strings [| "b"; "a"; ""; "b" |]) in
          let t = Nx_ragged.take ~indices:(on_set (int64s [| 3L; 1L |])) r in
          lies_on_set (Nx_ragged.values t);
          equal ~msg:"take" (array string) [| "b"; "a" |]
            (Nx_ragged.to_strings t);
          let rank = Nx_ragged.rank r in
          lies_on_set rank;
          equal ~msg:"rank" (array int64) [| 2L; 1L; 0L; 2L |]
            (Nx.to_array rank);
          let g = Nx_ragged.unique r in
          lies_on_set g.ids;
          lies_on_set g.first;
          equal ~msg:"ids" (array int64) [| 0L; 1L; 2L; 0L |]
            (Nx.to_array g.ids);
          equal ~msg:"first" (array int64) [| 0L; 1L; 2L |]
            (Nx.to_array g.first);
          equal ~msg:"counts" (array int64) [| 2L; 1L; 1L |]
            (Nx.to_array g.counts));
      test "a failed check on a set raises at the set's next wait" (fun () ->
          let offsets = on_set (int64s [| 0L; 2L; 1L |]) in
          let r = Nx_ragged.v ~offsets (on_set (Nx.zeros Nx.int32 [| 3 |])) in
          invalid ~by:"Nx_ragged.v" (fun () -> Nx.sync (Nx_ragged.offsets r)));
      test "a read raises a pending check's failure as the check raised it"
        (fun () ->
          let offsets = on_set (int64s [| 0L; 2L; 1L |]) in
          let r = Nx_ragged.v ~offsets (on_set (Nx.zeros Nx.int32 [| 3 |])) in
          let from_v = function
            | Invalid_argument m -> String.starts_with ~prefix:"Nx_ragged.v: " m
            | _ -> false
          in
          raises_match from_v (fun () ->
              Nx_ragged.take ~indices:(on_set (int64s [| 0L |])) r));
    ]

(* Reads of traced values *)

type ('v, 's, 'd) Nx.Prim.payload += Tag : ('v, 's, 'd) Nx.Prim.payload

(* [traced f] runs [f t] in an interpretation whose rule gives every operation
   traced results, [t.trace x] a traced value of [x]'s form. *)
type tracer = {
  trace : 'v 's. ('v, 's, Nx.host) Nx.t -> ('v, 's, Nx.host) Nx.t;
}

let traced f =
  Nx.Prim.interpret ~name:"test.trace" Values
    (fun i ~by op ->
      Nx.Prim.results ~by (fun _ form -> Nx.Prim.traced i form Tag) op)
    (fun i -> f { trace = (fun x -> Nx.Prim.traced i (Nx.Prim.form x) Tag) })

let reads =
  let r = Nx_ragged.of_strings [| "ab"; "c"; "ab" |] in
  let o = Nx_ragged.offsets r and values = Nx_ragged.values r in
  cases "a read of a traced value raises naming its function" ~id:fst
    [
      ( "Nx_ragged.take",
        fun { trace } ->
          ignore
            (Nx_ragged.take
               ~indices:(trace (int64s [| 0L |]))
               (Nx_ragged.of_lengths (int64s [| 1L |])
                  (Nx.zeros Nx.int32 [| 1 |]))) );
      ( "Nx_ragged.concat",
        fun { trace } ->
          let t = Nx_ragged.v ~offsets:(trace o) values in
          ignore (Nx_ragged.concat [ t; r ]) );
      ( "Nx_ragged.to_strings",
        fun { trace } -> ignore (Nx_ragged.to_strings (Nx_ragged.map trace r))
      );
      ( "Nx_ragged.unique",
        fun { trace } ->
          ignore (Nx_ragged.unique (Nx_ragged.v ~offsets:(trace o) values)) );
      ( "Nx_ragged.rank",
        fun { trace } ->
          ignore (Nx_ragged.rank (Nx_ragged.v ~offsets:(trace o) values)) );
    ]
    (fun (by, f) -> invalid ~by (fun () -> traced f))

let no_reads =
  test "the functions that read nothing take traced values" (fun () ->
      traced (fun { trace } ->
          let x = trace (Nx.create Nx.float64 [| 3 |] [| 1.; 2.; 3. |]) in
          let r = Nx_ragged.v ~offsets:(trace (int64s [| 0L; 1L; 3L |])) x in
          let l = Nx_ragged.of_lengths (trace (int64s [| 2L; 1L |])) x in
          let g =
            Nx_ragged.of_ids ~segments:2 (trace (int64s [| 1L; 0L; 1L |])) x
          in
          ignore (Nx_ragged.lengths (Nx_ragged.sub r ~offset:1 ~length:1));
          ignore (Nx_ragged.map Nx.neg l);
          ignore (Nx_ragged.quantile [| 0.5 |] g)))

(* Memory *)

(* [counted f] is what [f ()] gives the collector, counted from a compaction:
   the major cycles a pacing law bounds, and what else drives them, for its
   failure to print. *)
type counted = {
  majors : int;
  minors : int;
  promoted : float;
  major_words : float;
}

let counted f =
  Gc.compact ();
  let s0 = Gc.quick_stat () in
  f ();
  let s1 = Gc.quick_stat () in
  {
    majors = s1.major_collections - s0.major_collections;
    minors = s1.minor_collections - s0.minor_collections;
    promoted = s1.promoted_words -. s0.promoted_words;
    major_words = s1.major_words -. s0.major_words;
  }

let pp_counted c =
  Printf.sprintf
    "%d major cycles, %d minor collections, %.0f words promoted, %.0f words \
     allocated in the major heap"
    c.majors c.minors c.promoted c.major_words

(* [with_ratio r f] is [f ()] with the collector's [custom_major_ratio] at
   [r]. *)
let with_ratio r f =
  let was = (Gc.get ()).custom_major_ratio in
  Gc.set { (Gc.get ()) with custom_major_ratio = r };
  Fun.protect f ~finally:(fun () ->
      Gc.set { (Gc.get ()) with custom_major_ratio = was })

(* [of_strings] makes its offsets and bytes in host buffers, which pace the
   collector by the program's memory, whatever [custom_major_ratio] says of
   the heap: the bytes of host buffers that reach the major heap bring a
   cycle's work per bound of at least 4 MiB, and a buffer that starts young
   with more than [custom_minor_max_size] bytes brings an eighth of its
   bytes at birth as well. At a ratio of 1, which paces a bigarray of
   Bigarray's own at a cycle's work each, 64 calls over 2^14 strings of 8
   bytes, 128 KiB of offsets and of bytes each, each held across a minor
   collection that promotes it, bring at most a cycle's work per 4 MiB of
   them and of an eighth of them more than at the default ratio. The count
   starts from a compaction, which leaves the heap about its live words: a
   cycle's work, which the runtime sizes as sweeping the heap and marking
   part of it, then completes at most a cycle, and two more are under way
   at either end. *)
let memory =
  test "of_strings paces the collector by the program's memory" (fun () ->
      let n = 1 lsl 14 and calls = 64 in
      let ss = Array.make n "12345678" in
      let cycles r =
        with_ratio r @@ fun () ->
        ignore (Nx_ragged.of_strings ss);
        counted (fun () ->
            for _ = 1 to calls do
              let held = Nx_ragged.of_strings ss in
              Gc.minor ();
              ignore (Sys.opaque_identity held)
            done)
      in
      let usual = cycles (Gc.get ()).custom_major_ratio in
      let at_1 = cycles 1 in
      let bytes = (8 * n) + (8 * (n + 1)) in
      let made = calls * (bytes + (bytes / 8)) in
      at_most
        ~msg:
          (Printf.sprintf "major cycles at a ratio of 1: %s; at the default: %s"
             (pp_counted at_1) (pp_counted usual))
        int
        ~than:(usual.majors + (made / (4 lsl 20)) + 2)
        at_1.majors)

let () =
  exit
    (run "nx ragged"
       [
         making;
         strings_of_bytes;
         by_ids;
         transforming;
         row_quantiles;
         identities;
         sets;
         group "reads" [ reads; no_reads ];
         memory;
       ])
