(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_kit

let invalid f = raises_match (Exn.invalid_arg ?substring:None) f
let floats = array float_exact
let near = float 1e-9

(* Floats within [1e-9], [nan] equal to [nan]. *)
let close =
  array
    (Testable.make
       ~pp:(fun ppf x -> Format.fprintf ppf "%.17g" x)
       ~equal:(fun a b ->
         (Float.is_nan a && Float.is_nan b) || Float.abs (a -. b) <= 1e-9))

(* A stack given as rows [(column, series, length)]. *)
let stack ?offset rows =
  let rows = Array.of_list rows in
  Stack.intervals ?offset
    ~columns:(Array.map (fun (c, _, _) -> c) rows)
    ~series:(Array.map (fun (_, s, _) -> s) rows)
    (Array.map (fun (_, _, l) -> l) rows)

let pp_rows ppf (columns, series, lengths) =
  Format.fprintf ppf "@[<1>[%a]@]"
    (Format.pp_print_list ~pp_sep:Format.pp_print_space (fun ppf (c, s, l) ->
         Format.fprintf ppf "(%d, %d, %h)" c s l))
    (List.init (Array.length lengths) (fun r ->
         (columns.(r), series.(r), lengths.(r))))

(* Generators *)

(* Rows over few columns and series, so that cells hold several rows, with
   lengths of both signs, zeros and missing ones. *)
let gen_rows =
  let length =
    Gen.frequency
      [
        (6, Gen.float_range (-100.) 100.);
        (1, Gen.of_list [ 0.; -0.; 1.; -1. ]);
        (1, Gen.of_list [ nan; infinity; neg_infinity ]);
      ]
  in
  Gen.with_pp pp_rows
    (Gen.map
       (fun rows ->
         ( Array.map (fun (c, _, _) -> c) rows,
           Array.map (fun (_, s, _) -> s) rows,
           Array.map (fun (_, _, l) -> l) rows ))
       (Gen.array ~size:(Gen.int_range 0 30)
          (Gen.triple (Gen.int_range 0 5) (Gen.int_range 0 4) length)))

let gen_offset =
  Gen.of_list
    ~pp:(fun ppf o ->
      Format.pp_print_string ppf
        (match o with
        | `Zero -> "`Zero"
        | `Expand -> "`Expand"
        | `Center -> "`Center"))
    [ `Zero; `Expand; `Center ]

let missing l = not (Float.is_finite l)

(* The rows of column [c] that hold a length not missing. *)
let held columns lengths c =
  List.filter
    (fun r -> columns.(r) = c && not (missing lengths.(r)))
    (List.init (Array.length lengths) Fun.id)

let column_numbers columns = List.sort_uniq Int.compare (Array.to_list columns)

(* Piles *)

(* The model of [`Zero]: each column's lengths laid in the order of their
   series, then of their rows, by sign from [0.]. *)
let model (columns, series, lengths) =
  let n = Array.length lengths in
  let starts = Array.make n nan and ends = Array.make n nan in
  List.iter
    (fun c ->
      let rows =
        List.stable_sort
          (fun r r' -> Int.compare series.(r) series.(r'))
          (held columns lengths c)
      in
      ignore
        (List.fold_left
           (fun (lo, hi) r ->
             let l = lengths.(r) in
             if l < 0. then begin
               starts.(r) <- lo;
               ends.(r) <- lo +. l;
               (lo +. l, hi)
             end
             else begin
               starts.(r) <- hi;
               ends.(r) <- hi +. l;
               (lo, hi +. l)
             end)
           (0., 0.) rows))
    (column_numbers columns);
  (starts, ends)

let zero_model (columns, series, lengths) =
  let expected = model (columns, series, lengths) in
  equal (pair floats floats) expected (Stack.intervals ~columns ~series lengths)

let missing_rows (offset, (columns, series, lengths)) =
  let starts, ends = Stack.intervals ~offset ~columns ~series lengths in
  Array.iteri
    (fun r l ->
      let msg = Printf.sprintf "row %d" r in
      equal ~msg bool (missing l) (Float.is_nan starts.(r));
      equal ~msg bool (missing l) (Float.is_nan ends.(r)))
    lengths;
  cover "a missing length" (Array.exists missing lengths)

(* Making a length missing leaves the other intervals as they are without it,
   its column still counting among the columns. *)
let missing_is_absent (offset, (columns, series, lengths)) =
  let n = Array.length lengths in
  assume (n > 1);
  let keep = List.filter (fun r -> r <> n - 1) (List.init n Fun.id) in
  assume (List.exists (fun r -> columns.(r) >= columns.(n - 1)) keep);
  let sub a = Array.of_list (List.map (fun r -> a.(r)) keep) in
  let lengths' = Array.copy lengths in
  lengths'.(n - 1) <- nan;
  let s, e = Stack.intervals ~offset ~columns ~series lengths' in
  let s', e' =
    Stack.intervals ~offset ~columns:(sub columns) ~series:(sub series)
      (sub lengths)
  in
  equal (pair floats floats) (s', e') (sub s, sub e)

let row_order () =
  let starts, ends =
    stack [ (0, 1, 2.); (0, 0, 1.); (0, 1, 3.); (0, 0, -1.) ]
  in
  equal floats [| 1.; 0.; 3.; 0. |] starts;
  equal floats [| 3.; 1.; 6.; -1. |] ends

let piles =
  group "piles"
    [
      prop "lay each column by series and row, by sign from zero" gen_rows
        zero_model;
      test "series in order, then rows in order, a pile per sign" row_order;
      prop "a missing length covers no interval"
        (Gen.pair gen_offset gen_rows)
        missing_rows;
      prop "a missing length is laid as if absent"
        (Gen.pair gen_offset gen_rows)
        missing_is_absent;
      test "zero lengths are laid upward" (fun () ->
          equal (pair floats floats)
            ([| 0.; 2.; 0.; 2. |], [| 2.; 2.; -1.; 2. |])
            (stack [ (0, 0, 2.); (0, 1, -0.); (0, 2, -1.); (0, 3, 0.) ]));
      test "no rows" (fun () ->
          equal (pair floats floats) ([||], [||]) (stack []));
      test "columns and series far apart" (fun () ->
          equal (pair floats floats)
            ([| -0.5; -1. |], [| 0.5; 1. |])
            (stack ~offset:`Center
               [ (0, 0, 1.); (1_000_000_000, 1_000_000_000, 2.) ]));
    ]

(* Offsets *)

let extent starts ends rows =
  List.fold_left
    (fun (lo, hi) r ->
      ( Float.min lo (Float.min starts.(r) ends.(r)),
        Float.max hi (Float.max starts.(r) ends.(r)) ))
    (infinity, neg_infinity) rows

(* [`Expand] maps each column's extent onto [0, 1]: its lengths sum, in
   magnitude, to 1. *)
let expand_law (columns, series, lengths) =
  let starts, ends = Stack.intervals ~offset:`Expand ~columns ~series lengths in
  List.iter
    (fun c ->
      match held columns lengths c with
      | [] -> ()
      | rows ->
          let msg = Printf.sprintf "column %d" c in
          if List.for_all (fun r -> lengths.(r) = 0.) rows then
            List.iter
              (fun r ->
                equal ~msg float_exact 0. starts.(r);
                equal ~msg float_exact 0. ends.(r))
              rows
          else begin
            let lo, hi = extent starts ends rows in
            equal ~msg float_exact 0. lo;
            equal ~msg float_exact 1. hi;
            let total =
              List.fold_left
                (fun t r -> t +. Float.abs (ends.(r) -. starts.(r)))
                0. rows
            in
            equal ~msg near 1. total
          end)
    (column_numbers columns)

(* [`Center] centres each column's extent on 0, keeping its lengths. *)
let center_law (columns, series, lengths) =
  let starts, ends = Stack.intervals ~offset:`Center ~columns ~series lengths in
  let s0, e0 = Stack.intervals ~columns ~series lengths in
  List.iter
    (fun c ->
      match held columns lengths c with
      | [] -> ()
      | r0 :: _ as rows ->
          let msg = Printf.sprintf "column %d" c in
          let lo, hi = extent starts ends rows in
          equal ~msg
            (float (1e-12 *. (1. +. Float.abs lo +. Float.abs hi)))
            0. (lo +. hi);
          let shift = starts.(r0) -. s0.(r0) in
          List.iter
            (fun r ->
              equal ~msg near shift (starts.(r) -. s0.(r));
              equal ~msg near shift (ends.(r) -. e0.(r)))
            rows)
    (column_numbers columns)

let expand_cases () =
  let starts, ends =
    stack ~offset:`Expand
      [ (0, 0, 1.); (0, 1, 3.); (1, 0, -1.); (1, 1, 1.); (2, 0, 0.) ]
  in
  equal close [| 0.; 0.25; 0.5; 0.5; 0. |] starts;
  equal close [| 0.25; 1.; 0.; 1.; 0. |] ends

let center_cases () =
  let starts, ends =
    stack ~offset:`Center [ (0, 0, 1.); (0, 1, 3.); (1, 0, -1.); (1, 1, 3.) ]
  in
  equal close [| -2.; -1.; -1.; -1. |] starts;
  equal close [| -1.; 2.; -2.; 2. |] ends

let offsets =
  group "offsets"
    [
      prop "expand maps each column onto [0, 1]" gen_rows expand_law;
      prop "center centres each column" gen_rows center_law;
      test "expand on hand cases" expand_cases;
      test "center on hand cases" center_cases;
    ]

(* Arguments *)

let errors =
  group "arguments"
    [
      test "lengths that differ raise" (fun () ->
          invalid (fun () ->
              Stack.intervals ~columns:[| 0 |] ~series:[| 0; 0 |] [| 1.; 1. |]);
          invalid (fun () ->
              Stack.intervals ~columns:[| 0; 0 |] ~series:[| 0 |] [| 1.; 1. |]);
          invalid (fun () ->
              Stack.intervals ~columns:[| 0; 0 |] ~series:[| 0; 0 |] [| 1. |]));
      test "a negative column raises" (fun () ->
          invalid (fun () -> stack [ (0, 0, 1.); (-1, 0, 1.) ]));
      test "a negative series raises" (fun () ->
          invalid (fun () -> stack [ (0, 0, 1.); (0, min_int, 1.) ]));
    ]

let () = exit (run "Stack" [ piles; offsets; errors ])
