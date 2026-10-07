(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Values and headers: the value grammars of FITS 4.0 §4.2, records kept as
   written, keyword rules, duplicates, and printing in the fixed and free
   formats. *)

open Windtrap
open Ymir_fits
module H = Fits.Header
module V = Fits.Value

let pad r = r ^ String.make (80 - String.length r) ' '

let header records =
  require_ok
    (H.of_string (String.concat "" (List.map pad records @ [ pad "END" ])))

let result_w w = result w string

let starts_error ~sub = function
  | Ok _ -> fail "expected an Error"
  | Error e -> contains ~sub e

(* Reading values *)

let ints =
  cases
    ~name:(fun (field, _) -> field)
    "int reads FITS integers"
    [
      ("42", Some 42);
      ("+7", Some 7);
      ("-0", Some 0);
      ("  -12  ", Some (-12));
      ("1.5", None);
      ("1E3", None);
      ("0x10", None);
      ("1_000", None);
      ("T", None);
    ]
    (fun (field, expected) ->
      let h = header [ "K       = " ^ field ] in
      match expected with
      | Some n -> equal (result_w int) (Ok n) (H.get V.int "K" h)
      | None -> is_error (H.get V.int "K" h))

let floats =
  cases
    ~name:(fun (field, _) -> field)
    "float reads integer and real text, correctly rounded"
    [
      ("1.5", Some 1.5);
      ("1288.4", Some 1288.4);
      ("42", Some 42.);
      ("1.5D3", Some 1500.);
      ("1.5E-3", Some 0.0015);
      ("1.5e3", Some 1500.);
      (".5", Some 0.5);
      ("5.", Some 5.);
      ("-0.0", Some (-0.));
      ("2.2250738585072011E-308", Some 0x0.fffffffffffffp-1022);
      ("NAN", None);
      ("NaN", None);
      ("INF", None);
      ("-INF", None);
      ("0x1p3", None);
      ("1_000.0", None);
      ("1E400", None);
      ("1.5E", None);
      ("E3", None);
      ("'1.5'", None);
    ]
    (fun (field, expected) ->
      let h = header [ "K       = " ^ field ] in
      match expected with
      | Some x -> equal (result_w float_exact) (Ok x) (H.get V.float "K" h)
      | None -> is_error (H.get V.float "K" h))

let nan_names_card () =
  let h =
    require_ok
      (H.of_string ~name:"sp.fits"
         (String.concat ""
            [ pad "A       = 1"; pad "DUSTA   = NAN"; pad "END" ]))
  in
  equal (result_w float_exact)
    (Error
       "sp.fits: card 2 (DUSTA): NAN is not a FITS number; Value.text reads \
        the field as written")
    (H.get V.float "DUSTA" h)

let strings =
  cases
    ~name:(fun (field, _) -> field)
    "string reads quoted text, trailing spaces dropped"
    [
      ("'NGC 346 '", Some "NGC 346");
      ("'O''Hara'", Some "O'Hara");
      ("''", Some "");
      ("'   '", Some " ");
      ("'  lead'", Some "  lead");
      ("'a / b' / comment", Some "a / b");
      ("'unterminated", None);
      ("42", None);
    ]
    (fun (field, expected) ->
      let h = header [ "K       = " ^ field ] in
      match expected with
      | Some s -> equal (result_w string) (Ok s) (H.get V.string "K" h)
      | None -> is_error (H.get V.string "K" h))

let bools =
  cases
    ~name:(fun (field, _) -> field)
    "bool reads T and F"
    [
      ("T", Some true);
      ("                   F", Some false);
      ("1", None);
      ("'T'", None);
    ]
    (fun (field, expected) ->
      let h = header [ "K       = " ^ field ] in
      match expected with
      | Some b -> equal (result_w bool) (Ok b) (H.get V.bool "K" h)
      | None -> is_error (H.get V.bool "K" h))

let text () =
  let h =
    header
      [
        "BIG     = 123456789012345678901234567890 / past int";
        "Z       = (1.0, 2.0)";
      ]
  in
  equal (result_w string) (Ok "123456789012345678901234567890")
    (H.get V.text "BIG" h);
  is_error (H.get V.int "BIG" h);
  equal (result_w string) (Ok "(1.0, 2.0)") (H.get V.text "Z" h)

let continue_ () =
  (* §4.2.1.2's example, a string continued over three records. *)
  let h =
    header
      [
        "WEIGHTS = 'Long string &'      / comment";
        "CONTINUE  'continued &'";
        "CONTINUE  'to the end'";
        "NEXT    = 1";
      ]
  in
  equal (result_w string) (Ok "Long string continued to the end")
    (H.get V.string "WEIGHTS" h);
  equal (result_w int) (Ok 1) (H.get V.int "NEXT" h)

let undefined () =
  let h =
    header [ "U       =                      / no value"; "D       = 1" ]
  in
  equal (result_w (option int)) (Ok None) (H.find V.int "U" h);
  equal (result_w (option int)) (Ok None) (H.find V.int "ABSENT" h);
  is_error (H.get V.int "U" h);
  is_error (H.get V.int "ABSENT" h)

let hierarch () =
  let h =
    header
      [
        "HIERARCH ESO DET DIT = 1.5 / exposure";
        "HIERARCH ESO  INS   MODE = 'IMG'";
        "HIERARCH X = 1";
      ]
  in
  equal (result_w float_exact) (Ok 1.5) (H.get V.float "ESO DET DIT" h);
  equal (result_w string) (Ok "IMG") (H.get V.string "ESO INS MODE" h);
  (* One token of at most 8 bytes forms no hierarchical keyword: the record
     is commentary. *)
  equal (list string) [ " X = 1" ] (H.commentary "HIERARCH" h)

let commentary () =
  let h =
    header
      [
        "COMMENT a comment";
        "HISTORY first";
        "        blank keyword";
        "HISTORY second";
        "NOTE    free text";
      ]
  in
  equal (list string) [ "first"; "second" ] (H.commentary "HISTORY" h);
  equal (list string) [ "a comment" ] (H.commentary "COMMENT" h);
  equal (list string) [ "blank keyword" ] (H.commentary "" h);
  equal (list string) [ "free text" ] (H.commentary "NOTE" h);
  let long = String.make 100 'h' in
  let h' = H.add_commentary "HISTORY" long H.empty in
  equal (list string)
    [ String.make 72 'h'; String.make 28 'h' ]
    (H.commentary "HISTORY" h')

let duplicates () =
  let agree = header [ "A       = 1"; "B       = 2"; "A       = 1.0E0" ] in
  equal (result_w int) (Ok 1)
    (H.get V.int "A" (header [ "A       = 1"; "A       = 1" ]));
  equal (result_w float_exact) (Ok 1.) (H.get V.float "A" agree);
  let differ = header [ "A       = 1"; "B       = 2"; "A       = 3" ] in
  match H.get V.int "A" differ with
  | Ok _ -> fail "disagreeing cards read"
  | Error e ->
      contains ~sub:"cards 1 and 3" e;
      contains ~sub:"1 and 3" e

let keywords () =
  let h = H.empty in
  raises_match (Exn.invalid_arg ~substring:"naxis") (fun () ->
      H.get V.int "naxis" h);
  raises_match (Exn.invalid_arg ~substring:"COMMENT") (fun () ->
      H.set V.int "COMMENT" 1 h);
  raises_match (Exn.invalid_arg ~substring:"naxis1") (fun () ->
      H.find V.int "naxis1" h |> ignore);
  equal (result_w (option int)) (Ok None) (H.find V.int "NINECHARS" h);
  raises_match Exn.invalid_arg (fun () -> H.remove "A B=" h);
  equal (result_w (option int)) (Ok None) (H.find V.int "ESO DET DIT" h)

let map () =
  let frame =
    V.map
      (function
        | "ICRS" -> Ok `Icrs | "FK5" -> Ok `Fk5 | s -> Error (s ^ " is no frame"))
      (function `Icrs -> "ICRS" | `Fk5 -> "FK5")
      V.string
  in
  let h = H.set frame "RADESYS" `Fk5 H.empty in
  equal (result_w string) (Ok "FK5") (H.get V.string "RADESYS" h);
  equal (result_w string) (Ok "FK5")
    (Result.map
       (function `Icrs -> "ICRS" | `Fk5 -> "FK5")
       (H.get frame "RADESYS" h));
  starts_error ~sub:"card 1 (RADESYS): GAL is no frame"
    (H.get frame "RADESYS" (H.set V.string "RADESYS" "GAL" H.empty))

(* Printing *)

let printing () =
  let h =
    H.empty
    |> H.set ~comment:"[s] exposure" V.float "EXPTIME" 1288.4
    |> H.set V.string "OBJECT" "NGC 346"
    |> H.set V.bool "FLAG" true |> H.set V.int "COUNT" (-42)
    |> H.set V.float "BIG" 6.02214076e23
    |> H.set V.float "TWO" 2.
    |> H.set V.text "Z" "(1.0, 2.0)"
    |> H.set V.float "ESO DET DIT" 1.5
    |> H.set V.string "LONGSTR"
         (String.make 50 'x'
        ^ " and a string that runs past one record's sixty-eight bytes")
  in
  expect (Format.asprintf "%a" H.pp h)
  @@ __POS_OF__
       {|
    EXPTIME =               1288.4 / [s] exposure
    OBJECT  = 'NGC 346 '
    FLAG    =                    T
    COUNT   =                  -42
    BIG     =       6.02214076E+23
    TWO     =                  2.0
    Z       =           (1.0, 2.0)
    HIERARCH ESO DET DIT = 1.5
    LONGSTR = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx and a string tha&'
    CONTINUE  't runs past one record''s sixty-eight bytes'
    |}

let set_in_place () =
  let h =
    header
      [ "A       = 1 / first"; "B       = 2"; "A       = 1"; "C       = 3" ]
  in
  let h' = H.set V.int "A" 5 h in
  equal (list string)
    (List.map pad
       [
         "A       =                    5 / first"; "B       = 2"; "C       = 3";
       ])
    (H.records h');
  let h'' = H.set ~comment:"new" V.int "D" 4 h in
  equal int 5 (List.length (H.records h''));
  equal (list string) (H.records h)
    (List.filteri (fun i _ -> i < 4) (H.records h''))

let set_raises () =
  raises_match Exn.invalid_arg (fun () -> H.set V.float "X" Float.nan H.empty);
  raises_match Exn.invalid_arg (fun () ->
      H.set V.float "X" Float.infinity H.empty);
  raises_match Exn.invalid_arg (fun () ->
      H.set V.string "X" "caf\xc3\xa9" H.empty);
  raises_match Exn.invalid_arg (fun () -> H.set V.text "X" "1 2" H.empty);
  raises_match Exn.invalid_arg (fun () ->
      H.set ~comment:(String.make 60 'c') V.int "X" 1 H.empty);
  raises_match Exn.invalid_arg (fun () ->
      H.set V.float
        (String.concat " " (List.init 20 (fun _ -> "ABC")))
        1. H.empty)

let remove_continued () =
  let h = header [ "A       = 'one &'"; "CONTINUE  'two'"; "B       = 1" ] in
  equal (list string) [ pad "B       = 1" ] (H.records (H.remove "A" h))

(* Records as written *)

let kept_bytes () =
  (* Bytes outside 32-126, blank records and duplicates survive a round
     trip. *)
  let odd = "ODD     = 'caf\xc3\xa9'" in
  let h = header [ odd; ""; "A       = 1"; "A       = 1" ] in
  equal string (H.to_string h)
    (H.to_string (require_ok (H.of_string (H.to_string h))));
  equal int 4 (List.length (H.records h));
  is_error (H.get V.string "ODD" h);
  equal int 2880 (String.length (H.to_string h))

let head_lines () =
  let lines =
    "SIMPLE  =                    T\n\
     BITPIX  =                    8\r\n\
     NAXIS   =                    0\n\
     END\n"
  in
  let h = require_ok (H.of_string lines) in
  equal (result_w int) (Ok 8) (H.get V.int "BITPIX" h);
  is_error (H.of_string "SIMPLE  =                    T\n");
  is_error (H.of_string (String.make 81 'A' ^ "\nEND\n"))

(* Laws *)

let printable = Gen.char_range ' ' '~'
let key = "KEY"

let float_round_trip =
  let subnormal =
    Gen.map
      (fun (m, neg) ->
        Int64.float_of_bits (Int64.logor m (if neg then Int64.min_int else 0L)))
      Gen.(pair (int64_range 1L 0xFFFFFFFFFFFFFL) bool)
  in
  prop "get float (set float x) is x for every finite x"
    (Gen.frequency [ (8, Gen.float); (1, subnormal); (1, Gen.constant (-0.)) ])
    (fun x ->
      cover "subnormal" (x <> 0. && Float.abs x < 0x1p-1022);
      cover "negative zero" (1. /. x = Float.neg_infinity);
      equal (result_w float_exact) (Ok x)
        (H.get V.float key (H.set V.float key x H.empty)))

let trim_right s =
  let n = ref (String.length s) in
  while !n > 0 && s.[!n - 1] = ' ' do
    decr n
  done;
  String.sub s 0 !n

let string_round_trip =
  let gen =
    Gen.(
      frequency
        [
          (1, constant "");
          ( 9,
            map trim_right
              (string_of ~size:(int_range 0 300)
                 (frequency [ (8, printable); (1, constant '\'') ])) );
        ])
  in
  prop "get string (set string s) is s for every string without trailing spaces"
    gen (fun s ->
      cover "continued" (String.length s > 68);
      cover "quotes" (String.contains s '\'');
      cover "empty" (s = "");
      equal (result_w string) (Ok s)
        (H.get V.string key (H.set V.string key s H.empty));
      List.iter
        (fun r -> equal int 80 (String.length r))
        (H.records (H.set V.string key s H.empty)))

let int_round_trip =
  prop "get int (set int n) is n" Gen.int (fun n ->
      equal (result_w int) (Ok n) (H.get V.int key (H.set V.int key n H.empty)))

let untouched =
  prop "set leaves every other record byte for byte"
    Gen.(
      pair
        (list ~size:(int_range 0 8)
           (string_of ~size:(int_range 0 80) printable))
        float)
    (fun (others, x) ->
      let h =
        H.of_string
          (String.concat ""
             (List.map (fun r -> "COMMENT " ^ String.sub (pad r) 0 72) others)
          ^ pad "END")
        |> require_ok
      in
      let h' = H.set V.float "NEW" x h in
      equal (list string) (H.records h)
        (List.filteri (fun i _ -> i < List.length others) (H.records h')))

let () =
  exit
  @@ run "Fits.Header"
       [
         group "values"
           [
             ints;
             floats;
             test "NAN names its card" nan_names_card;
             strings;
             bools;
             test "text reads the field as written" text;
             test "CONTINUE joins records" continue_;
             test "undefined values" undefined;
             test "HIERARCH keywords" hierarch;
             test "commentary" commentary;
             test "duplicates" duplicates;
             test "keyword names" keywords;
             test "map" map;
           ];
         group "printing"
           [
             test "fixed and free formats" printing;
             test "set replaces in place" set_in_place;
             test "set raises on what FITS cannot hold" set_raises;
             test "remove takes CONTINUE records" remove_continued;
           ];
         group "records"
           [ test "bytes are kept" kept_bytes; test ".head lines" head_lines ];
         group "laws"
           [ float_round_trip; string_round_trip; int_round_trip; untouched ];
       ]
