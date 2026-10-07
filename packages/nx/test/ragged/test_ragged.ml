(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Ragged arrays, against a model of their rows: each row is the cells of its
   values from one offset to the next, flattened. Values hold distinct elements,
   and rows before the first offset and after the last. *)

open Windtrap
open Nx_test

(* [pre] rows before the first offset, rows of [lengths], [post] rows after;
   each row of [width] elements, or a scalar cell when [width] is 0. *)
type drawn = { pre : int; lengths : int array; post : int; width : int }

let pp_ints ppf a =
  Format.fprintf ppf "[%a]"
    (Format.pp_print_seq
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       Format.pp_print_int)
    (Array.to_seq a)

let pp_drawn ppf d =
  Format.fprintf ppf "rows %a after %d, before %d, cells of %d" pp_ints
    d.lengths d.pre d.post d.width

let rows_of_values d = d.pre + Array.fold_left ( + ) 0 d.lengths + d.post
let cell d = Int.max 1 d.width

let drawn =
  Gen.with_pp pp_drawn
    Gen.(
      let* pre = int_range 0 2 in
      let* lengths = array ~size:(int_range 0 6) (int_range 0 4) in
      let* post = int_range 0 2 in
      let+ width = int_range 0 2 in
      { pre; lengths; post; width })

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
  let flat = Nx.to_array (Nx.flatten v) in
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
          let r = ragged d in
          equal same_rows (model d) (rows r);
          equal int (Array.length d.lengths) (Nx_ragged.length r);
          equal (array int64)
            (Array.map Int64.of_int d.lengths)
            (Nx.to_array (Nx_ragged.lengths r));
          equal (array int64) (offsets d) (Nx.to_array (Nx_ragged.offsets r));
          equal (tensor int32) (values d) (Nx_ragged.values r));
      test "v takes offsets from 0 to the last row of the values" (fun () ->
          equal int 2
            (Nx_ragged.length
               (Nx_ragged.v
                  ~offsets:(int64s [| 0L; 0L; 3L |])
                  (Nx.zeros Nx.int32 [| 3 |]))));
      cases "v refuses offsets that do not cut the values" ~name:fst
        [
          ("offsets of two axes", (Nx.zeros Nx.int64 [| 1; 2 |], [| 3 |]));
          ("no offset", (int64s [||], [| 3 |]));
          ("an offset below 0", (int64s [| -1L; 2L |], [| 3 |]));
          ("offsets that decrease", (int64s [| 0L; 2L; 1L |], [| 3 |]));
          ("an offset past the values", (int64s [| 0L; 4L |], [| 3 |]));
          ("scalar values", (int64s [| 0L |], [||]));
        ]
        (fun (_, (offsets, shape)) ->
          raises_invalid_arg (fun () ->
              Nx_ragged.v ~offsets (Nx.zeros Nx.int32 shape)));
      prop "of_lengths cuts rows of the lengths from the first value" drawn
        (fun d ->
          let d = { d with pre = 0 } in
          let r =
            Nx_ragged.of_lengths
              (int64s (Array.map Int64.of_int d.lengths))
              (values d)
          in
          equal same_rows (model d) (rows r));
      cases "of_lengths refuses lengths that do not cut the values" ~name:fst
        [
          ("lengths of two axes", (Nx.zeros Nx.int64 [| 1; 1 |], [| 3 |]));
          ("a negative length", (int64s [| 2L; -1L; 1L |], [| 3 |]));
          ("lengths past the values", (int64s [| 2L; 2L |], [| 3 |]));
          ( "lengths that sum past int64",
            (int64s [| Int64.max_int; Int64.max_int; 2L |], [| 3 |]) );
          ("scalar values", (int64s [| 0L |], [||]));
        ]
        (fun (_, (lengths, shape)) ->
          raises_invalid_arg (fun () ->
              Nx_ragged.of_lengths lengths (Nx.zeros Nx.int32 shape)));
    ]

(* Strings *)

let every_byte = String.init 256 Char.chr
let strings = Gen.(array ~size:(int_range 0 6) string)

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
          raises_invalid_arg (fun () ->
              Nx_ragged.to_strings
                (Nx_ragged.v
                   ~offsets:(int64s [| 0L; 1L |])
                   (Nx.zeros Nx.uint8 [| 1; 2 |]))));
    ]

(* Grouping by ids *)

let grouped =
  Gen.with_pp
    (fun ppf (segments, ids, width) ->
      Format.fprintf ppf "%d segments, ids [%a], cells of %d" segments
        (Format.pp_print_seq
           ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
           (fun ppf i -> Format.fprintf ppf "%Ld" i))
        (Array.to_seq ids) width)
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

let by_ids =
  let x ids width =
    values { pre = Array.length ids; lengths = [||]; post = 0; width }
  in
  group "of_ids"
    [
      prop "of_ids gathers the rows of each id in their order" grouped
        (fun (segments, ids, width) ->
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
              (fun (i, id) ->
                if id >= 0L && id < Int64.of_int segments then None else Some i)
              (List.mapi (fun i id -> (i, id)) (Array.to_list ids))
          in
          let tail =
            Array.sub
              (Nx.to_array (Nx.flatten v))
              (kept * w)
              ((Array.length ids - kept) * w)
          in
          equal (list int) dropped
            (List.sort compare
               (List.init
                  (Array.length tail / w)
                  (fun j -> Int32.to_int tail.(j * w) / w))));
      cases "of_ids refuses ids that do not match the rows" ~name:fst
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
          ( "scalar values",
            fun () ->
              Nx_ragged.of_ids ~segments:1 (int64s [||])
                (Nx.zeros Nx.int32 [||]) );
        ]
        (fun (_, f) -> raises_invalid_arg f);
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

let taken =
  Gen.with_pp
    (fun ppf (d, i) ->
      Format.fprintf ppf "%a at [%a]" pp_drawn d
        (Format.pp_print_seq
           ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
           (fun ppf i -> Format.fprintf ppf "%Ld" i))
        (Array.to_seq i))
    Gen.(
      let* d = drawn in
      let n = Array.length d.lengths in
      let+ i =
        array ~size:(int_range 0 8)
          (frequency
             [
               (6, map Int64.of_int (int_range (-2) (n + 1)));
               ( 1,
                 of_list
                   ~pp:(fun ppf -> Format.fprintf ppf "%Ld")
                   [ Int64.min_int; Int64.max_int ] );
             ])
      in
      (d, i))

let transforming =
  group "transforming"
    [
      prop "sub is the rows of the range" range (fun (d, o, n) ->
          equal same_rows
            (Array.sub (model d) o n)
            (rows (Nx_ragged.sub (ragged d) ~offset:o ~length:n)));
      prop "sub shares the offsets and the values" range (fun (d, o, n) ->
          let r = ragged d in
          let s = Nx_ragged.sub r ~offset:o ~length:n in
          is_true ~msg:"values" (Nx_ragged.values s == Nx_ragged.values r);
          is_true ~msg:"offsets"
            (storage (Nx_ragged.offsets s) == storage (Nx_ragged.offsets r)));
      cases "sub refuses a range outside the rows" ~name:fst
        [
          ("a negative offset", (-1, 1));
          ("a negative length", (0, -1));
          ("one row past the end", (1, 2));
        ]
        (fun (_, (offset, length)) ->
          raises_invalid_arg (fun () ->
              Nx_ragged.sub
                (Nx_ragged.of_lengths
                   (int64s [| 1L; 1L |])
                   (Nx.zeros Nx.int32 [| 2 |]))
                ~offset ~length));
      prop "take is the row at each index, and an empty row outside" taken
        (fun (d, i) ->
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
      test "take refuses indices of two axes" (fun () ->
          raises_invalid_arg (fun () ->
              Nx_ragged.take
                ~indices:(Nx.zeros Nx.int64 [| 1; 1 |])
                (Nx_ragged.of_lengths (int64s [| 1L |])
                   (Nx.zeros Nx.int32 [| 1 |]))));
      prop "concat is the rows one after the other"
        (Gen.with_pp
           (Format.pp_print_list pp_drawn)
           Gen.(
             let* width = int_range 0 2 in
             list ~size:(int_range 2 4) (map (fun d -> { d with width }) drawn)))
        (fun ds ->
          let r = Nx_ragged.concat (List.map ragged ds) in
          equal same_rows (Array.concat (List.map model ds)) (rows r);
          exactly_rows r);
      test "concat returns a single ragged array as it is" (fun () ->
          let r =
            Nx_ragged.of_lengths (int64s [| 1L |]) (Nx.zeros Nx.int32 [| 2 |])
          in
          is_true (Nx_ragged.concat [ r ] == r));
      cases "concat refuses what has no rows in common" ~name:fst
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
        (fun (_, f) -> raises_invalid_arg f);
      prop "map keeps the offsets and maps the values" drawn (fun d ->
          let r = ragged d in
          let m = Nx_ragged.map (fun v -> Nx.cast Nx.float64 (Nx.neg v)) r in
          is_true ~msg:"offsets" (Nx_ragged.offsets m == Nx_ragged.offsets r);
          equal (tensor float_exact)
            (Nx.cast Nx.float64 (Nx.neg (values d)))
            (Nx_ragged.values m));
      cases "map refuses a function that changes the rows of the values"
        ~name:fst
        [
          ("one row fewer", fun v -> Nx.slice [ R (1, Nx.dim 0 v) ] v);
          ("a scalar", fun v -> Nx.sum v);
        ]
        (fun (_, f) ->
          raises_invalid_arg (fun () ->
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
  Gen.with_pp
    (fun ppf (d, xs, qs) ->
      Format.fprintf ppf "%a of [%a] at [%a]" pp_drawn d
        (Format.pp_print_seq
           ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
           pp_float)
        (Array.to_seq xs)
        (Format.pp_print_seq
           ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
           pp_float)
        (Array.to_seq qs))
    Gen.(
      let* d = drawn in
      let* xs =
        array ~size:(constant (rows_of_values d * cell d)) float_values
      in
      let+ qs = probabilities in
      (d, xs, qs))

let row_quantiles =
  group "quantile"
    [
      prop "each row's quantiles are Nx.quantile's, and an empty row's NaN"
        float_rows (fun (d, xs, qs) ->
          let check (type b) (dt : (float, b) Nx.dtype) =
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
            let expected =
              Array.init n (fun i ->
                  if d.lengths.(i) = 0 then Nx.full dt [| k |] Float.nan
                  else
                    Nx.quantile qs
                      (Nx.slice
                         [ R (Int64.to_int o.(i), Int64.to_int o.(i + 1)) ]
                         values))
            in
            let got = Nx_ragged.quantile qs r in
            equal (array int) [| k; n |] (Nx.shape got);
            Array.iteri
              (fun i e ->
                equal
                  ~msg:(Printf.sprintf "row %d" i)
                  (tensor float_exact) e
                  (Nx.slice [ A; I i ] got))
              expected
          in
          check Nx.float64;
          check Nx.float32;
          check Nx.float16);
      test "Nx_ragged.quantile refuses a probability outside [0, 1]" (fun () ->
          raises_invalid_arg (fun () ->
              Nx_ragged.quantile [| 1.5 |]
                (Nx_ragged.of_lengths (int64s [| 1L |])
                   (Nx.zeros Nx.float64 [| 1 |]))));
    ]

(* Ids and ranks of rows *)

(* Rows that share long prefixes: prefixes of three strings of up to 140
   elements, each with a tail of up to two elements, over few distinct
   elements. *)
let shared_rows value =
  let open Gen in
  let* bases = list ~size:(constant 3) (array ~size:(int_range 0 140) value) in
  let bases = Array.of_list bases in
  array ~size:(int_range 0 12)
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
      name : string;
      dtype : ('a, 'b) Nx.dtype;
      values : 'a list;
      compare : 'a -> 'a -> int;
      pp : Format.formatter -> 'a -> unit;
    }
      -> element

let floats_of name dtype =
  E
    {
      name;
      dtype;
      values =
        [ Float.nan; -0.; 0.; 1.; -1.; Float.infinity; Float.neg_infinity ];
      compare = compare_float;
      pp = pp_float;
    }

let ints_of name dtype values =
  E { name; dtype; values; compare = Int.compare; pp = Format.pp_print_int }

let elements_of =
  [
    ints_of "bytes" Nx.uint8 [ 0; 1; 97; 255 ];
    ints_of "int8" Nx.int8 [ -128; -1; 0; 1; 127 ];
    ints_of "int16" Nx.int16 [ -32768; -1; 0; 1; 32767 ];
    floats_of "float32" Nx.float32;
    floats_of "float64" Nx.float64;
    E
      {
        name = "int64";
        dtype = Nx.int64;
        values = [ Int64.min_int; -1L; 0L; 1L; Int64.max_int ];
        compare = Int64.compare;
        pp = (fun ppf -> Format.fprintf ppf "%Ld");
      };
    E
      {
        name = "bool";
        dtype = Nx.bool;
        values = [ false; true ];
        compare = Bool.compare;
        pp = Format.pp_print_bool;
      };
  ]

let identifying (E e) =
  let value = Gen.of_list ~pp:e.pp e.values in
  let junk = Gen.array ~size:(Gen.int_range 0 3) value in
  let pp_row ppf r =
    Format.fprintf ppf "[%a]"
      (Format.pp_print_seq
         ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
         e.pp)
      (Array.to_seq r)
  in
  let drawn =
    Gen.with_pp
      (fun ppf (pre, rows, post) ->
        Format.fprintf ppf "@[<v>rows:@,%a@,after %a, before %a@]"
          (Format.pp_print_seq pp_row)
          (Array.to_seq rows) pp_row pre pp_row post)
      (Gen.triple junk (shared_rows value) junk)
  in
  (* The rows and their ragged array, and all but the first row through
     [sub]. *)
  let both f model ((_, rows, _) as d) =
    let r = ragged_of e.dtype d in
    equal (array int64) (model e.compare rows) (Nx.to_array (f r));
    let n = Array.length rows in
    if n > 0 then
      equal ~msg:"all but the first row" (array int64)
        (model e.compare (Array.sub rows 1 (n - 1)))
        (Nx.to_array (f (Nx_ragged.sub r ~offset:1 ~length:(n - 1))))
  in
  [
    prop
      (e.name ^ " rows' ids number them in order of first appearance")
      drawn
      (both Nx_ragged.ids model_ids);
    prop
      (e.name ^ " rows' ranks are dense in the sort order, a prefix first")
      drawn
      (both Nx_ragged.rank model_ranks);
  ]

let identities =
  group "ids and rank"
    (List.concat_map identifying elements_of
    @ [
        test "cells of several elements compare element by element" (fun () ->
            let r =
              Nx_ragged.of_lengths
                (int64s [| 1L; 1L; 2L; 1L |])
                (Nx.create Nx.int32 [| 5; 2 |]
                   [| 1l; 2l; 1l; 3l; 1l; 2l; 0l; 0l; 1l; 2l |])
            in
            equal (array int64) [| 0L; 1L; 2L; 0L |]
              (Nx.to_array (Nx_ragged.ids r));
            equal (array int64) [| 0L; 2L; 1L; 0L |]
              (Nx.to_array (Nx_ragged.rank r)));
        test "ids and rank refuse complex values" (fun () ->
            let r =
              Nx_ragged.of_lengths (int64s [| 1L |])
                (Nx.zeros Nx.complex64 [| 1 |])
            in
            raises_invalid_arg (fun () -> Nx_ragged.ids r);
            raises_invalid_arg (fun () -> Nx_ragged.rank r));
      ])

(* Reads *)

(* An interpreter that claims only reads and records the name each carries. *)
let naming () =
  let seen = ref [] in
  let run : type r. r Nx.Op.t -> r =
   fun op ->
    (match op with Read { by; _ } -> seen := by :: !seen | _ -> ());
    Nx.Op.eval op
  in
  let claims : type r. r Nx.Op.t -> bool = function
    | Read _ -> true
    | _ -> false
  in
  ({ Nx.Op.run; claims }, seen)

let reads =
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; -2.; 3. |] in
  let ids = Nx.create Nx.int64 [| 3 |] [| 1L; 0L; 1L |] in
  let lengths = Nx.create Nx.int64 [| 2 |] [| 1L; 2L |] in
  let grouped () = Nx_ragged.of_ids ~segments:2 ids x in
  let discard f () = ignore (f ()) in
  let names = list string in
  group "reads"
    [
      cases ~name:fst "a read names its function and reads once"
        [
          ( "Nx_ragged.v",
            discard (fun () ->
                Nx_ragged.v
                  ~offsets:(Nx.create Nx.int64 [| 3 |] [| 0L; 1L; 3L |])
                  x) );
          ( "Nx_ragged.of_lengths",
            discard (fun () -> Nx_ragged.of_lengths lengths x) );
          ( "Nx_ragged.take",
            discard (fun () -> Nx_ragged.take ~indices:lengths (grouped ())) );
          ( "Nx_ragged.concat",
            discard (fun () ->
                let r = grouped () in
                Nx_ragged.concat [ r; r; r ]) );
        ]
        (fun (expected, f) ->
          let i, seen = naming () in
          Nx.Op.intercept i f;
          equal names [ expected ] !seen);
      test "to_strings reads its offsets, then its bytes" (fun () ->
          let r = Nx_ragged.of_strings [| "ab"; "c" |] in
          let i, seen = naming () in
          Nx.Op.intercept i (fun () -> ignore (Nx_ragged.to_strings r));
          equal names [ "Nx_ragged.to_strings"; "Nx_ragged.to_strings" ] !seen);
      cases ~name:fst "ids and rank name every round's read"
        [
          ("Nx_ragged.ids", discard (fun () -> Nx_ragged.ids (grouped ())));
          ("Nx_ragged.rank", discard (fun () -> Nx_ragged.rank (grouped ())));
        ]
        (fun (expected, f) ->
          let i, seen = naming () in
          Nx.Op.intercept i f;
          equal names [ expected ] (List.sort_uniq String.compare !seen));
    ]

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
         reads;
       ])
