(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_kit

let floats = array float_exact
let intervals = pair floats floats

(* Rows [(column, series, length)] as the three arrays [Stack.intervals]
   takes. *)
let unzip rows =
  let rows = Array.of_list rows in
  ( Array.map (fun (c, _, _) -> c) rows,
    Array.map (fun (_, s, _) -> s) rows,
    Array.map (fun (_, _, l) -> l) rows )

let stack ?offset rows =
  let columns, series, lengths = unzip rows in
  Stack.intervals ?offset ~columns ~series lengths

let pp_rows ppf (columns, series, lengths) =
  Format.fprintf ppf "@[<1>[%a]@]"
    (Format.pp_print_list ~pp_sep:Format.pp_print_space (fun ppf (c, s, l) ->
         Format.fprintf ppf "(%d, %d, %h)" c s l))
    (List.init (Array.length lengths) (fun r ->
         (columns.(r), series.(r), lengths.(r))))

let pp_offset ppf o =
  Format.pp_print_string ppf
    (match o with
    | `Zero -> "`Zero"
    | `Expand -> "`Expand"
    | `Center -> "`Center")

(* Generators *)

let gen_missing = Gen.of_list [ nan; infinity; neg_infinity ]

(* [size] rows over few columns and series, so that a pile holds several rows,
   with lengths of both signs, zeros and missing ones. *)
let gen_rows size =
  let length =
    Gen.frequency
      [
        (6, Gen.float_range (-100.) 100.);
        (1, Gen.of_list [ 0.; -0.; 1.; -1. ]);
        (1, gen_missing);
      ]
  in
  Gen.with_pp pp_rows
    (Gen.map unzip
       (Gen.list ~size
          (Gen.triple (Gen.int_range 0 5) (Gen.int_range 0 4) length)))

let gen_stack = gen_rows (Gen.int_range 0 30)
let gen_offset = Gen.of_list ~pp:pp_offset [ `Zero; `Expand; `Center ]
let missing l = not (Float.is_finite l)

(* The rows of column [c] that hold a length not missing. *)
let held columns lengths c =
  List.filter
    (fun r -> columns.(r) = c && not (missing lengths.(r)))
    (List.init (Array.length lengths) Fun.id)

let column_numbers columns = List.sort_uniq Int.compare (Array.to_list columns)

(* The piles of the header: each column's lengths laid in the order of their
   series, then of their rows, from [0.], those not negative upward and the
   others downward. *)
let piles (columns, series, lengths) =
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

(* [moved f stack piles] maps each end of a column's [piles] through [f lo hi],
   [lo] and [hi] the column's extent. *)
let moved f (columns, _, lengths) (starts, ends) =
  let starts = Array.copy starts and ends = Array.copy ends in
  List.iter
    (fun c ->
      let rows = held columns lengths c in
      let lo = List.fold_left (fun lo r -> Float.min lo ends.(r)) 0. rows in
      let hi = List.fold_left (fun hi r -> Float.max hi ends.(r)) 0. rows in
      List.iter
        (fun r ->
          starts.(r) <- f lo hi starts.(r);
          ends.(r) <- f lo hi ends.(r))
        rows)
    (column_numbers columns);
  (starts, ends)

(* Piles *)

let pile_order stack =
  let columns, series, lengths = stack in
  let both =
    List.exists
      (fun c ->
        let signs =
          List.map (fun r -> lengths.(r) < 0.) (held columns lengths c)
        in
        List.mem true signs && List.mem false signs)
      (column_numbers columns)
  in
  cover "both signs in a column" both;
  cover "a length of zero" (Array.exists (fun l -> l = 0.) lengths);
  cover "a length of -0."
    (Array.exists (fun l -> 1. /. l = neg_infinity) lengths);
  equal intervals (piles stack) (Stack.intervals ~columns ~series lengths)

(* A missing length makes no interval, whatever the offset. *)
let no_interval (offset, (columns, series, lengths)) =
  let starts, ends = Stack.intervals ~offset ~columns ~series lengths in
  cover "a missing length" (Array.exists missing lengths);
  Array.iteri
    (fun r l ->
      if missing l then begin
        let msg = Printf.sprintf "row %d" r in
        equal ~msg float_exact nan starts.(r);
        equal ~msg float_exact nan ends.(r)
      end)
    lengths

(* Making the last length missing leaves the other intervals as they are without
   it. *)
let as_if_absent (offset, (columns, series, lengths), l) =
  let n = Array.length lengths in
  let keep a = Array.sub a 0 (n - 1) in
  cover "its column holds other lengths"
    (List.exists (fun r -> r < n - 1) (held columns lengths columns.(n - 1)));
  let lengths' = Array.copy lengths in
  lengths'.(n - 1) <- l;
  let starts, ends = Stack.intervals ~offset ~columns ~series lengths' in
  equal intervals
    (Stack.intervals ~offset ~columns:(keep columns) ~series:(keep series)
       (keep lengths))
    (keep starts, keep ends)

let pile_examples =
  [
    unzip [];
    (* Series in order, then rows in order, a pile per sign. *)
    unzip [ (0, 1, 2.); (0, 0, 1.); (0, 1, 3.); (0, 0, -1.) ];
    (* Lengths of zero, [-0.] among them, laid upward. *)
    unzip [ (0, 0, 2.); (0, 1, -0.); (0, 2, -1.); (0, 3, 0.) ];
  ]

let piles_group =
  group "piles"
    [
      prop ~examples:pile_examples
        "lay each column by series, then row, from zero by sign" gen_stack
        pile_order;
      prop "a missing length covers no interval"
        (Gen.pair gen_offset gen_stack)
        no_interval;
      prop "a missing length leaves its column laid as if it were absent"
        (Gen.triple gen_offset (gen_rows (Gen.int_range 1 30)) gen_missing)
        as_if_absent;
    ]

(* Offset *)

(* Columns of each sign, of both, of zero extent, and far apart. *)
let offset_examples =
  [
    unzip [ (0, 0, 1.); (0, 1, 3.); (1, 0, -1.); (1, 1, 1.); (2, 0, 0.) ];
    unzip [ (0, 0, 1.); (0, 1, 3.); (1, 0, -1.); (1, 1, 3.) ];
    unzip [ (0, 0, 1.); (1_000_000_000, 1_000_000_000, 2.) ];
  ]

let zero stack =
  let columns, series, lengths = stack in
  equal intervals (piles stack)
    (Stack.intervals ~offset:`Zero ~columns ~series lengths)

let expand_extent stack =
  let columns, series, lengths = stack in
  let starts, ends = Stack.intervals ~offset:`Expand ~columns ~series lengths in
  List.iter
    (fun c ->
      let rows = held columns lengths c in
      let points = List.concat_map (fun r -> [ starts.(r); ends.(r) ]) rows in
      if List.exists (fun r -> lengths.(r) <> 0.) rows then
        equal
          ~msg:(Printf.sprintf "column %d" c)
          (pair float_exact float_exact)
          (0., 1.)
          ( List.fold_left Float.min 0. points,
            List.fold_left Float.max 0. points ))
    (column_numbers columns)

(* Floats within [4 epsilon], [nan] equal to [nan], as a missing length's
   interval is. *)
let near_or_nan =
  Testable.make ~pp:(Testable.pp float_exact) ~equal:(fun a b ->
      (Float.is_nan a && Float.is_nan b)
      || Float.abs (a -. b) <= 4. *. epsilon_float)

(* The intervals of [`Expand] are those of [`Zero] under the affine map from
   each column's extent onto the unit interval. There, any form of the map in
   two rounded operations, as [(v - lo) / (hi - lo)], errs by at most twice
   [epsilon], so two forms agree within [4 epsilon]. *)
let expand stack =
  let columns, series, lengths = stack in
  let zero_extent =
    List.exists
      (fun c ->
        let rows = held columns lengths c in
        rows <> [] && List.for_all (fun r -> lengths.(r) = 0.) rows)
      (column_numbers columns)
  in
  cover "a column of zero extent" zero_extent;
  let onto lo hi v = if lo = hi then v else (v -. lo) /. (hi -. lo) in
  let within = array near_or_nan in
  equal (pair within within)
    (moved onto stack (piles stack))
    (Stack.intervals ~offset:`Expand ~columns ~series lengths)

let center stack =
  let columns, series, lengths = stack in
  equal intervals
    (moved (fun lo hi v -> v -. ((lo +. hi) /. 2.)) stack (piles stack))
    (Stack.intervals ~offset:`Center ~columns ~series lengths)

let offset =
  group "offset"
    [
      prop ~examples:offset_examples "`Zero starts each column's piles at 0"
        gen_stack zero;
      prop ~examples:offset_examples
        "`Expand puts each column's extent at exactly 0 and 1" gen_stack
        expand_extent;
      prop ~examples:offset_examples
        "`Expand maps each column onto [0, 1] and leaves a zero extent"
        gen_stack expand;
      prop ~examples:offset_examples
        "`Center moves each column by -. (lo +. hi) /. 2." gen_stack center;
    ]

(* Intervals *)

(* Two lengths of [max_float] whose sum exceeds [max_float], under each offset,
   and a length of each sign whose extent does, which [`Expand] scales by. *)
let past_max =
  [
    (`Zero, [ (0, 0, max_float); (0, 1, max_float) ]);
    (`Expand, [ (0, 0, max_float); (0, 1, max_float) ]);
    (`Center, [ (0, 0, max_float); (0, 1, max_float) ]);
    (`Expand, [ (0, 0, max_float); (0, 1, -.max_float) ]);
  ]

let non_finite (offset, rows) =
  let starts, ends = stack ~offset rows in
  satisfies ~claim:"an interval that is not finite" intervals
    (fun (s, e) -> not (Array.for_all Float.is_finite (Array.append s e)))
    (starts, ends)

let raises f = raises_match Exn.invalid_arg f

let intervals_group =
  group "intervals"
    [
      cases "piles past max_float give non-finite intervals"
        ~id:(fun (o, rows) ->
          Format.asprintf "%a %a" pp_offset o pp_rows (unzip rows))
        past_max non_finite;
      cases "raises if its arrays differ in length" ~id:fst
        [
          ("one column short", ([| 0 |], [| 0; 0 |], [| 1.; 1. |]));
          ("one series short", ([| 0; 0 |], [| 0 |], [| 1.; 1. |]));
          ("one length short", ([| 0; 0 |], [| 0; 0 |], [| 1. |]));
        ]
        (fun (_, (columns, series, lengths)) ->
          raises (fun () -> Stack.intervals ~columns ~series lengths));
      test "raises on a negative column" (fun () ->
          raises (fun () -> stack [ (0, 0, 1.); (-1, 0, 1.) ]));
      test "raises on a negative series" (fun () ->
          raises (fun () -> stack [ (0, 0, 1.); (0, min_int, 1.) ]));
    ]

let () = exit (run "Stack" [ piles_group; offset; intervals_group ])
