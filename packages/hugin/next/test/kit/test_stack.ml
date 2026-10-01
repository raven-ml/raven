(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_next_kit

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
let stack ?offset ?order rows =
  let rows = Array.of_list rows in
  Stack.intervals ?offset ?order
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
        | `Center -> "`Center"
        | `Wiggle -> "`Wiggle"))
    [ `Zero; `Expand; `Center; `Wiggle ]

let gen_order =
  Gen.of_list
    ~pp:(fun ppf o ->
      Format.pp_print_string ppf
        (match o with
        | `Given -> "`Given"
        | `Reverse -> "`Reverse"
        | `Ascending -> "`Ascending"
        | `Descending -> "`Descending"
        | `Appearance -> "`Appearance"
        | `Inside_out -> "`Inside_out"))
    [ `Given; `Reverse; `Ascending; `Descending; `Appearance; `Inside_out ]

let missing l = not (Float.is_finite l)

(* The rows of column [c] that hold a length not missing. *)
let held columns lengths c =
  List.filter
    (fun r -> columns.(r) = c && not (missing lengths.(r)))
    (List.init (Array.length lengths) Fun.id)

let column_numbers columns = List.sort_uniq Int.compare (Array.to_list columns)

(* Piles *)

(* The model of [`Zero]: each column's lengths laid in the order of their series
   under [rank], then of their rows, by sign from [0.]. *)
let model rank (columns, series, lengths) =
  let n = Array.length lengths in
  let starts = Array.make n nan and ends = Array.make n nan in
  List.iter
    (fun c ->
      let rows =
        List.stable_sort
          (fun r r' -> Int.compare (rank series.(r)) (rank series.(r')))
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

let zero_model (order, rank) (columns, series, lengths) =
  let expected = model rank (columns, series, lengths) in
  equal (pair floats floats) expected
    (Stack.intervals ~order ~columns ~series lengths)

let missing_rows (offset, order, (columns, series, lengths)) =
  let starts, ends = Stack.intervals ~offset ~order ~columns ~series lengths in
  Array.iteri
    (fun r l ->
      let msg = Printf.sprintf "row %d" r in
      equal ~msg bool (missing l) (Float.is_nan starts.(r));
      equal ~msg bool (missing l) (Float.is_nan ends.(r)))
    lengths;
  cover "a missing length" (Array.exists missing lengths)

(* Making a length missing leaves the other intervals as they are without it,
   its column still counting among the columns. *)
let missing_is_absent (offset, order, (columns, series, lengths)) =
  let n = Array.length lengths in
  assume (n > 1);
  let keep = List.filter (fun r -> r <> n - 1) (List.init n Fun.id) in
  assume (List.exists (fun r -> columns.(r) >= columns.(n - 1)) keep);
  let sub a = Array.of_list (List.map (fun r -> a.(r)) keep) in
  let lengths' = Array.copy lengths in
  lengths'.(n - 1) <- nan;
  let s, e = Stack.intervals ~offset ~order ~columns ~series lengths' in
  let s', e' =
    Stack.intervals ~offset ~order ~columns:(sub columns) ~series:(sub series)
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
      prop "lay each column by series and row, by sign from zero"
        (Gen.pair
           (Gen.of_list
              ~pp:(fun ppf (o, _) ->
                Format.pp_print_string ppf
                  (match o with `Given -> "`Given" | _ -> "`Reverse"))
              [ (`Given, Fun.id); (`Reverse, fun s -> -s) ])
           gen_rows)
        (fun (order, rows) -> zero_model order rows);
      test "series in order, then rows in order, a pile per sign" row_order;
      prop "a missing length covers no interval"
        (Gen.triple gen_offset gen_order gen_rows)
        missing_rows;
      prop "a missing length is laid as if absent"
        (Gen.triple gen_offset gen_order gen_rows)
        missing_is_absent;
      test "zero lengths are laid upward" (fun () ->
          equal (pair floats floats)
            ([| 0.; 2.; 0.; 2. |], [| 2.; 2.; -1.; 2. |])
            (stack [ (0, 0, 2.); (0, 1, -0.); (0, 2, -1.); (0, 3, 0.) ]));
      test "no rows" (fun () ->
          equal (pair floats floats) ([||], [||]) (stack []));
      test "columns and series far apart" (fun () ->
          equal (pair floats floats)
            ([| 0.; -1. |], [| 1.; 1. |])
            (stack ~offset:`Wiggle ~order:`Appearance
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
let expand_law (order, (columns, series, lengths)) =
  let starts, ends =
    Stack.intervals ~offset:`Expand ~order ~columns ~series lengths
  in
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
let center_law (order, (columns, series, lengths)) =
  let starts, ends =
    Stack.intervals ~offset:`Center ~order ~columns ~series lengths
  in
  let s0, e0 = Stack.intervals ~order ~columns ~series lengths in
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

(* [`Wiggle] moves each column as a whole and centres the whole stack. *)
let wiggle_law (order, (columns, series, lengths)) =
  let starts, ends =
    Stack.intervals ~offset:`Wiggle ~order ~columns ~series lengths
  in
  let s0, e0 = Stack.intervals ~order ~columns ~series lengths in
  let all = List.concat_map (held columns lengths) (column_numbers columns) in
  (match all with
  | [] -> ()
  | _ ->
      let lo, hi = extent starts ends all in
      equal (float (1e-9 *. (1. +. Float.abs lo +. Float.abs hi))) 0. (lo +. hi));
  List.iter
    (fun c ->
      match held columns lengths c with
      | [] -> ()
      | r0 :: _ as rows ->
          let shift = starts.(r0) -. s0.(r0) in
          List.iter
            (fun r ->
              let msg = Printf.sprintf "column %d, row %d" c r in
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

let rows_of series =
  List.concat
    (List.mapi (fun s ls -> List.mapi (fun c l -> (c, s, l)) ls) series)

(* The series of d3-shape 3.2's tests of its wiggle offset, which leaves column
   0 at 0 and centres nothing: here every interval is d3's moved by the one
   constant that centres the stack. *)
let d3_series = [ [ 1.; 2.; 1. ]; [ 3.; 4.; 2. ]; [ 5.; 2.; 4. ] ]

let wiggle_d3 () =
  let starts, ends = stack ~offset:`Wiggle (rows_of d3_series) in
  let m = 0.7857142857142857 in
  equal close
    (Array.map
       (fun v -> v -. 4.)
       [| 0.; -1.; m; 1.; 1.; 1. +. m; 4.; 5.; 3. +. m |])
    starts;
  equal close
    (Array.map
       (fun v -> v -. 4.)
       [| 1.; 1.; 1. +. m; 4.; 5.; 3. +. m; 9.; 7.; 7. +. m |])
    ends

let wiggle_d3_reverse () =
  let starts, ends =
    stack ~offset:`Wiggle ~order:`Reverse (rows_of d3_series)
  in
  let m = 0.21428571428571427 in
  equal close
    (Array.map
       (fun v -> v -. 5.)
       [| 8.; 8.; 7. +. m; 5.; 4.; 5. +. m; 0.; 2.; 1. +. m |])
    starts;
  equal close
    (Array.map
       (fun v -> v -. 5.)
       [| 9.; 10.; 8. +. m; 8.; 8.; 7. +. m; 5.; 4.; 5. +. m |])
    ends

let wiggle_d3_missing () =
  let series =
    [ [ 1.; 2.; 1. ]; [ nan; nan; nan ]; [ 3.; 4.; 2. ]; [ 5.; 2.; 4. ] ]
  in
  let starts, _ = stack ~offset:`Wiggle (rows_of series) in
  let m = 0.7857142857142857 in
  equal close
    (Array.map
       (fun v -> v -. 4.)
       [| 0.; -1.; m; nan; nan; nan; 1.; 1.; 1. +. m; 4.; 5.; 3. +. m |])
    starts

(* The recurrence on hand cases: weights are magnitudes, a series that leaves
   still counts in the sums before the next, an empty column keeps [g]. *)
let wiggle_cases =
  [
    ( "one series is centred",
      [ (0, 0, 1.); (1, 0, 3.); (2, 0, 2.) ],
      [| -0.5; -1.5; -1. |],
      [| 0.5; 1.5; 1. |] );
    ( "negative lengths weigh by magnitude",
      [ (0, 0, -1.); (1, 0, -3.) ],
      [| 0.5; 1.5 |],
      [| -0.5; -1.5 |] );
    ( "both signs weigh by magnitude",
      [ (0, 0, 1.); (0, 1, -1.); (1, 0, 2.); (1, 1, -3.) ],
      [| 0.7; 0.7; 0.5; 0.5 |],
      [| 1.7; -0.3; 2.5; -2.5 |] );
    ( "two series",
      [ (0, 0, 1.); (1, 0, 1.); (0, 1, 1.); (1, 1, 3.) ],
      [| -1.25; -2.; -0.25; -1. |],
      [| -0.25; -1.; 0.75; 2. |] );
    ( "an empty column keeps the baseline",
      [ (0, 0, 1.); (2, 0, 3.) ],
      [| 0.; -1.5 |],
      [| 1.; 1.5 |] );
    ( "a series that leaves still counts",
      [ (0, 0, 2.); (0, 1, 1.); (1, 1, 1.) ],
      [| -1.5; 0.5; 0.5 |],
      [| 0.5; 1.5; 1.5 |] );
  ]

let wiggle_case (_, rows, starts, ends) =
  equal (pair close close) (starts, ends) (stack ~offset:`Wiggle rows)

let offsets =
  group "offsets"
    [
      prop "expand maps each column onto [0, 1]"
        (Gen.pair gen_order gen_rows)
        expand_law;
      prop "center centres each column" (Gen.pair gen_order gen_rows) center_law;
      prop "wiggle moves columns and centres the stack"
        (Gen.pair gen_order gen_rows)
        wiggle_law;
      test "expand on hand cases" expand_cases;
      test "center on hand cases" center_cases;
      test "wiggle on d3's example" wiggle_d3;
      test "wiggle on d3's example in reverse order" wiggle_d3_reverse;
      test "wiggle on d3's example with a missing series" wiggle_d3_missing;
      cases "wiggle on hand cases"
        ~name:(fun (n, _, _, _) -> n)
        wiggle_cases wiggle_case;
    ]

(* Orders *)

(* The series of column [c] from the bottom of its pile, every one holding a
   positive length there. *)
let laid starts columns series c =
  List.filter
    (fun r -> columns.(r) = c)
    (List.init (Array.length columns) Fun.id)
  |> List.sort (fun r r' -> Float.compare starts.(r) starts.(r'))
  |> List.map (fun r -> series.(r))

(* Seven series, series [i] peaking in column [6 - i] at [i + 1], as d3's test
   of its inside-out order, and a column 7 where each holds [1.] to show the
   order they are laid in. *)
let d3_inside_out =
  List.concat
    (List.init 7 (fun s ->
         (7, s, 1.)
         :: List.init 7 (fun c ->
             (c, s, if c = 6 - s then Float.of_int (s + 1) else 0.))))

(* Each case shows its order in a probe column where every series holds [1.]. *)
let order_cases =
  let probe c = [ (c, 0, 1.); (c, 1, 1.); (c, 2, 1.) ] in
  [
    ("given", `Given, probe 0, 0, [ 0; 1; 2 ]);
    ("reverse", `Reverse, probe 0, 0, [ 2; 1; 0 ]);
    ( "ascending, ties in numbering order",
      `Ascending,
      [ (0, 0, 3.); (0, 1, 1.); (0, 2, 1.); (2, 2, nan) ] @ probe 1,
      1,
      [ 1; 2; 0 ] );
    ( "descending, ties in numbering order",
      `Descending,
      [ (0, 0, 1.); (0, 1, 3.); (0, 2, 3.) ] @ probe 1,
      1,
      [ 1; 2; 0 ] );
    ( "appearance, by first peak",
      `Appearance,
      [ (0, 0, 0.); (2, 0, 1.); (0, 1, 3.); (1, 1, 2.); (1, 2, 4.) ] @ probe 3,
      3,
      [ 1; 2; 0 ] );
    ( "appearance, a tied peak in its first column",
      `Appearance,
      [ (0, 0, 2.); (2, 0, 2.); (1, 1, 3.); (1, 2, 1.); (2, 2, 3.) ] @ probe 4,
      4,
      [ 0; 1; 2 ] );
  ]

let order_case (_, order, rows, c, expected) =
  let rows = Array.of_list rows in
  let columns = Array.map (fun (c, _, _) -> c) rows in
  let series = Array.map (fun (_, s, _) -> s) rows in
  let lengths = Array.map (fun (_, _, l) -> l) rows in
  let starts, _ = Stack.intervals ~order ~columns ~series lengths in
  equal (list int) expected (laid starts columns series c)

let inside_out () =
  let rows = Array.of_list d3_inside_out in
  let columns = Array.map (fun (c, _, _) -> c) rows in
  let series = Array.map (fun (_, s, _) -> s) rows in
  let starts, _ =
    Stack.intervals ~order:`Inside_out ~columns ~series
      (Array.map (fun (_, _, l) -> l) rows)
  in
  equal (list int) [ 2; 3; 6; 5; 4; 1; 0 ] (laid starts columns series 7)

(* A series of negative lengths peaks in the first column where it has none,
   whose sum is 0: zero lengths, laid upward, show where it falls. *)
let appearance_negative () =
  let starts, _ =
    stack ~order:`Appearance
      [
        (0, 0, -1.);
        (1, 0, -2.);
        (2, 0, -3.);
        (1, 0, 0.);
        (1, 1, 5.);
        (0, 2, 1.);
        (1, 2, 0.);
        (3, 1, 1.);
      ]
  in
  equal ~msg:"series 0, after series 1" float_exact 5. starts.(3);
  equal ~msg:"series 2, before series 1" float_exact 0. starts.(6)

(* Series 0 is absent from column 0 and holds at most [0.], so it peaks there,
   ties series 1, which peaks there at [1.], and is laid first in column 2's
   negative pile. *)
let appearance_absent () =
  let starts, _ =
    Stack.intervals ~order:`Appearance ~columns:[| 1; 2; 0; 2 |]
      ~series:[| 0; 0; 1; 1 |] [| 0.; -1.; 1.; -1. |]
  in
  equal ~msg:"series 0" float_exact 0. starts.(1);
  equal ~msg:"series 1" float_exact (-1.) starts.(3)

let orders =
  group "orders"
    [
      cases "hand cases" ~name:(fun (n, _, _, _, _) -> n) order_cases order_case;
      test "inside_out on d3's example" inside_out;
      test "appearance of a series of negative lengths" appearance_negative;
      test "appearance counts an absent column as 0." appearance_absent;
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

let () = exit (run "Stack" [ piles; offsets; orders; errors ])
