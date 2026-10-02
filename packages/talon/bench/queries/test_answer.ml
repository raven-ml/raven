(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap

let table_w =
  let pp ppf t =
    Format.fprintf ppf "%a: %d rows" Schema.pp (schema t) (rows t)
  in
  Testable.make ~pp ~equal:Talon.equal

let answer ?(k = [ "id001"; "id002"; "id003" ]) ?(x = [ 1.5; 2.5; 3.5 ]) () =
  v
    [
      ("k", Column.v Type.string (Array.of_list k));
      ("v", Column.v Type.float64 (Array.of_list x));
    ]

let floats vs = v [ ("v", Column.of_options Type.float64 (Array.of_list vs)) ]

(* Comparing *)

let agrees name expected got =
  test name (fun () -> equal (list string) [] (Answer.compare expected got))

let differs name expected got problems =
  test name (fun () ->
      equal (list string) problems (Answer.compare expected got))

let comparing =
  group "compare"
    [
      agrees "equal answers agree" (answer ()) (answer ());
      differs "a missing row is a row count" (answer ())
        (answer ~k:[ "id001"; "id002" ] ~x:[ 1.5; 2.5 ] ())
        [ "2 rows, expected 3" ];
      differs "a float past the tolerance names its column and first row"
        (answer ())
        (answer ~x:[ 1.5; 2.5 +. 1e-8; 3.5 ] ())
        [
          "column v: 1 rows differ, first at row 1: \"2.50000001\", expected \
           \"2.5\"";
        ];
      agrees "a float within a relative 1e-9 agrees" (answer ())
        (answer ~x:[ 1.5; 2.5 *. (1. +. 5e-10); 3.5 ] ());
      agrees "floats within an absolute 1e-12 agree" (floats [ Some 0. ])
        (floats [ Some 5e-13 ]);
      agrees "NaN agrees with NaN, and null with null"
        (floats [ Some Float.nan; None ])
        (floats [ Some Float.nan; None ]);
      agrees "infinities agree with themselves"
        (floats [ Some Float.infinity; Some Float.neg_infinity ])
        (floats [ Some Float.infinity; Some Float.neg_infinity ]);
      differs "a null disagrees with a value"
        (floats [ Some 1.; None ])
        (floats [ None; Some 1. ])
        [ "column v: 2 rows differ, first at row 0: null, expected \"1\"" ];
      differs "a NaN disagrees with a number" (floats [ Some 1. ])
        (floats [ Some Float.nan ])
        [ "column v: 1 rows differ, first at row 0: \"nan\", expected \"1\"" ];
      differs "a string differs exactly" (answer ())
        (answer ~k:[ "id001"; "id002"; "id004" ] ())
        [
          "column k: 1 rows differ, first at row 2: \"id004\", expected \
           \"id003\"";
        ];
      differs "schemas differ before rows" (answer ()) (floats [ Some 1. ])
        [ "schema v float64, expected k string, v float64" ];
    ]

(* Committed answers *)

let checking =
  group "check"
    [
      test "compares the rows a stride keeps" (fun () ->
          let got = floats (List.init 5 (fun i -> Some (Float.of_int i))) in
          let expected = floats [ Some 0.; Some 2.; Some 4. ] in
          equal (list string) [] (Answer.check ~expected ~rows:5 ~stride:2 got));
      test "a missing row is the full row count" (fun () ->
          let got = floats [ Some 0.; Some 2. ] in
          let expected = floats [ Some 0.; Some 2. ] in
          equal (list string) [ "2 rows, expected 3" ]
            (Answer.check ~expected ~rows:3 ~stride:1 got));
    ]

(* The stride is the smallest that keeps at most [limit] rows. *)
let stride =
  let kept rows s = (rows + s - 1) / s in
  cases ~name:string_of_int "stride"
    [ 0; 1; 199; 200; 201; 400; 401; 1_000_000 ] (fun rows ->
      let s = Answer.stride rows in
      at_most ~msg:"kept" int ~than:Answer.limit (kept rows s);
      if s > 1 then
        greater ~msg:"kept by a smaller stride" int ~than:Answer.limit
          (kept rows (s - 1)))

let canonical =
  group "canonical"
    [
      test "widens integers and floats and sorts by every column, nulls last"
        (fun () ->
          let t =
            v
              [
                ( "a",
                  Column.of_options Type.int32
                    [| Some 2; None; Some 1; Some 1 |] );
                ("b", Column.v Type.float32 [| 1.; 2.; 4.; 3. |]);
              ]
          in
          let expected =
            v
              [
                ( "a",
                  Column.of_options Type.int64
                    [| Some 1; Some 1; Some 2; None |] );
                ("b", Column.v Type.float64 [| 3.; 4.; 1.; 2. |]);
              ]
          in
          equal table_w expected (Answer.canonical ~ordered:false t));
      test "keeps the order of an ordered answer" (fun () ->
          let t = v [ ("a", Column.v Type.int64 [| 3; 1; 2 |]) ] in
          equal table_w t (Answer.canonical ~ordered:true t));
      test "reads categoricals as strings" (fun () ->
          let t =
            v
              [
                ("c", Column.v (Type.categorical [| "b"; "a" |]) [| "a"; "b" |]);
              ]
          in
          let expected = v [ ("c", Column.v Type.string [| "a"; "b" |]) ] in
          equal table_w expected (Answer.canonical ~ordered:true t));
    ]

let files =
  group "files"
    [
      test "path names the workload's directory" (fun () ->
          equal string "a/groupby-1e6/q01.csv"
            (Answer.path "a" "groupby/1e6/q01");
          equal string "a/tpch-sf0.1/q22.csv" (Answer.path "a" "tpch/sf0.1/q22"));
      test "read reads back what write writes" (fun () ->
          let t =
            v
              [
                ( "s",
                  Column.of_options Type.string
                    [| Some ""; Some "a,b"; Some "\"q\""; None; Some "x\ny" |]
                );
                ( "f",
                  Column.of_options Type.float64
                    [|
                      Some (-0.);
                      Some Float.nan;
                      Some Float.infinity;
                      None;
                      Some 0.1;
                    |] );
                ( "i",
                  Column.of_options Type.int64
                    [| Some min_int; None; Some 0; Some max_int; Some (-1) |] );
              ]
          in
          let file = Filename.concat (temp_dir ()) "a.csv" in
          Answer.write file t;
          equal table_w t (Answer.read (schema t) file));
    ]

let () = exit (run "answer" [ comparing; checking; stride; canonical; files ])
