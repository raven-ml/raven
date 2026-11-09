(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap

let s = Col.string "s"
let texts vs = v [ ("s", Column.of_options Type.string vs) ]

let result e t =
  let run = Query.run (Query.select Expr.[ "out" := e ] (Query.of_table t)) in
  column (require_ok ~pp:Error.pp run) "out"

let lower vs = Column.options Kind.string (result (Expr.Str.lower s) (texts vs))
let upper vs = Column.options Kind.string (result (Expr.Str.upper s) (texts vs))
let same = array (option string)

let utf_8 u =
  let b = Buffer.create 4 in
  Buffer.add_utf_8_uchar b (Uchar.of_int u);
  Buffer.contents b

(* The data of the test generator, read from Unicode's files directly. *)
let scan_hex field =
  List.map (fun f -> int_of_string ("0x" ^ f)) (String.split_on_char ' ' field)

let to_text us = String.concat "" (List.map utf_8 us)

let lowers, uppers, cased_ranges, ignorable_ranges, white_space_ranges =
  let lower = Hashtbl.create 3000 and upper = Hashtbl.create 3000 in
  let cased = ref [] and ignorable = ref [] and white = ref [] in
  In_channel.with_open_text "golden/case.txt" @@ fun ic ->
  In_channel.fold_lines
    (fun () line ->
      if line <> "" && line.[0] <> '#' then
        match line.[0] with
        | 'S' -> (
            let body = String.sub line 2 (String.length line - 2) in
            match String.split_on_char ';' body with
            | [ u; l; up ] ->
                let u = int_of_string ("0x" ^ u) in
                Hashtbl.replace lower u (to_text (scan_hex l));
                Hashtbl.replace upper u (to_text (scan_hex up))
            | _ -> failwith line)
        | ('C' | 'I' | 'W') as kind -> (
            let body = String.sub line 2 (String.length line - 2) in
            match String.index_opt body '.' with
            | None -> failwith line
            | Some d ->
                let first = int_of_string ("0x" ^ String.sub body 0 d) in
                let n = String.length body - d - 2 in
                let last = int_of_string ("0x" ^ String.sub body (d + 2) n) in
                let into =
                  match kind with
                  | 'C' -> cased
                  | 'I' -> ignorable
                  | _ -> white
                in
                into := (first, last) :: !into)
        | _ -> failwith line)
    () ic;
  (lower, upper, !cased, !ignorable, !white)

let member ranges u = List.exists (fun (a, b) -> a <= u && u <= b) ranges
let is_cased u = member cased_ranges u
let is_ignorable u = member ignorable_ranges u
let is_white_space u = member white_space_ranges u

(* [scalars t] are the scalar values of [t] with the byte each starts at. *)
let scalars t =
  let rec go i acc =
    if i >= String.length t then List.rev acc
    else
      let d = String.get_utf_8_uchar t i in
      let n = Uchar.utf_decode_length d in
      go (i + n) (Uchar.to_int (Uchar.utf_decode_uchar d) :: acc)
  in
  go 0 []

(* [final_sigma us i] is Table 3-17's Final_Sigma for [us.(i)]: scanning past
   case-ignorable scalars, a cased one precedes it and none follows it. *)
let final_sigma us i =
  let n = Array.length us in
  let rec before j =
    if j < 0 then false
    else if is_ignorable us.(j) then before (j - 1)
    else is_cased us.(j)
  in
  let rec after j =
    if j >= n then true
    else if is_ignorable us.(j) then after (j + 1)
    else not (is_cased us.(j))
  in
  before (i - 1) && after (i + 1)

let map_lower u = Option.value (Hashtbl.find_opt lowers u) ~default:(utf_8 u)
let map_upper u = Option.value (Hashtbl.find_opt uppers u) ~default:(utf_8 u)

let expect_lower t =
  let us = Array.of_list (scalars t) in
  String.concat ""
    (Array.to_list
       (Array.mapi
          (fun i u ->
            if u = 0x3A3 && final_sigma us i then utf_8 0x3C2 else map_lower u)
          us))

let expect_upper t = String.concat "" (List.map map_upper (scalars t))

(* The tests *)

(* Every scalar value alone in its row, where final sigma has no effect: the
   scalars with a mapping and the far more numerous ones that have none. *)
let every_scalar =
  test "every scalar's full mapping, from the Unicode files" (fun () ->
      let surrogates = 0xE000 - 0xD800 in
      let scalar i = if i < 0xD800 then i else i + surrogates in
      let us = Array.init (0x110000 - surrogates) scalar in
      let inputs = Array.map (fun u -> Some (utf_8 u)) us in
      let wrong name map actual =
        List.filteri
          (fun i _ -> actual.(i) <> Some (map us.(i)))
          (Array.to_list us)
        |> List.map (Printf.sprintf "%s of U+%04X" name)
      in
      equal (list string) [] (wrong "lower" map_lower (lower inputs));
      equal (list string) [] (wrong "upper" map_upper (upper inputs)))

let every_white_space =
  test "trim drops exactly the White_Space scalars, from the Unicode files"
    (fun () ->
      let surrogates = 0xE000 - 0xD800 in
      let scalar i = if i < 0xD800 then i else i + surrogates in
      let us = Array.init (0x110000 - surrogates) scalar in
      let row u = utf_8 u ^ "x" ^ utf_8 u in
      let trim vs =
        Column.options Kind.string (result (Expr.Str.trim s) (texts vs))
      in
      let trimmed = trim (Array.map (fun u -> Some (row u)) us) in
      let expected u = if is_white_space u then "x" else row u in
      let wrong =
        List.filteri
          (fun i _ -> trimmed.(i) <> Some (expected us.(i)))
          (Array.to_list us)
      in
      equal (list string) [] (List.map (Printf.sprintf "U+%04X") wrong))

(* Scalars at the edges of a range decide whether a boundary test or a range
   lookup is off by one. Each is tried next to a sigma, in both sides' positions
   of the context. *)
let edges =
  let bound (a, b) = [ a - 1; a; b; b + 1 ] in
  List.sort_uniq compare
    (List.concat_map bound (cased_ranges @ ignorable_ranges))
  |> List.filter (fun u ->
         u >= 0 && u <= 0x10FFFF && not (u >= 0xD800 && u <= 0xDFFF))

let sigma_edges =
  test "final sigma at every edge of Cased and Case_Ignorable" (fun () ->
      let contexts u =
        let c = utf_8 u in
        [ "\xce\x91" ^ c ^ "\xce\xa3"; "\xce\x91\xce\xa3" ^ c; c ^ "\xce\xa3" ]
      in
      let inputs = List.concat_map contexts edges in
      let actual = lower (Array.of_list (List.map Option.some inputs)) in
      List.iteri
        (fun i t ->
          equal ~msg:t (option string) (Some (expect_lower t)) actual.(i))
        inputs)

let token =
  let open Gen in
  let of_list vs = of_list ~pp:(fun ppf -> Format.fprintf ppf "%S") vs in
  frequency
    [
      ( 3,
        of_list
          [
            "\xce\xa3" (* U+03A3 *); "\xce\x91" (* U+0391 *);
            "\xcd\x85" (* U+0345, cased and case-ignorable *);
            "\xc3\x9f" (* U+00DF, upper to two scalars *);
            "\xc4\xb0" (* U+0130, lower to two scalars *);
            "\xce\xb0" (* U+03B0, upper to three *);
            "\xe1\xba\x9e" (* U+1E9E, shorter lowercase *);
            "\xc8\xba" (* U+023A, longer lowercase *);
            "\xef\xac\x83" (* U+FB03 *);
            "\xf0\x90\x90\x80" (* U+10400, four bytes *);
            "\xf0\x9f\x98\x80" (* U+1F600, four bytes, unmapped *);
            "\xe6\x97\xa5" (* CJK, unmapped *);
          ] );
      (3, of_list [ "a"; "Z"; "z"; "A"; " "; "'"; "."; "1"; "\xc2\xad" ]);
      ( 1,
        map (fun n -> String.make n 'q') (int_range 0 20)
        |> map (fun s -> s ^ "Q") );
    ]

let text =
  Gen.map (String.concat "")
    (Gen.list ~size:(Gen.int_range 0 12) token)

let pp_rows ppf rows =
  Format.fprintf ppf "%a"
    (Format.pp_print_seq ~pp_sep:Format.pp_print_space (fun ppf -> function
       | None -> Format.pp_print_string ppf "null"
       | Some t -> Format.fprintf ppf "%S" t))
    (Array.to_seq rows)

let rows =
  Gen.with_pp pp_rows (Gen.array ~size:(Gen.int_range 0 8) (Gen.option text))

let law =
  prop "lower and upper agree with Unicode's Default Case Conversion" rows
    (fun vs ->
      let has p = Array.exists (function Some t -> p t | None -> false) vs in
      let sigma t = List.mem 0x3A3 (scalars t) in
      cover "a text with a sigma" (has sigma);
      let grows t = String.length (expect_lower t) > String.length t in
      cover "a text that grows in lowercase" (has grows);
      let shrinks t = String.length (expect_lower t) < String.length t in
      cover "a text that shrinks in lowercase" (has shrinks);
      cover "a null row" (Array.exists Option.is_none vs);
      equal same (Array.map (Option.map expect_lower) vs) (lower vs);
      equal same (Array.map (Option.map expect_upper) vs) (upper vs))

(* Spec and data: Table 3-17's examples, then single rows named by their bytes
   or by the SpecialCasing file. *)
let final_sigma_cases =
  let sigma = "\xce\xa3" and alpha = "\xce\x91" and ypog = "\xcd\x85" in
  let small = "\xcf\x83" and final = "\xcf\x82" and a = "\xce\xb1" in
  cases "Final_Sigma, as Table 3-17 and its text state"
    ~id:(fun (t, _) -> String.escaped t)
    [
      (* Alone: nothing cased comes before. *)
      (sigma, small);
      (alpha ^ sigma, a ^ final);
      (alpha ^ sigma ^ alpha, a ^ small ^ a);
      (* The case-ignorable U+0345 is passed over, so the standard's examples
         of a scalar in both sets read as it says. *)
      (ypog ^ sigma, ypog ^ small);
      (alpha ^ ypog ^ sigma, a ^ ypog ^ final);
      (alpha ^ sigma ^ ypog, a ^ final ^ ypog);
      (alpha ^ sigma ^ ypog ^ alpha, a ^ small ^ ypog ^ a);
      (alpha ^ sigma ^ " " ^ alpha, a ^ final ^ " " ^ a);
      (alpha ^ "'" ^ sigma, a ^ "'" ^ final);
      ("1" ^ sigma, "1" ^ small);
      ("a" ^ sigma, "a" ^ final);
    ]
    (fun (t, expected) -> equal same [| Some expected |] (lower [| Some t |]))

let mapping_cases =
  let u_upper u expected =
    test (Printf.sprintf "upper of U+%04X" u) (fun () ->
        equal same [| Some expected |] (upper [| Some (utf_8 u) |]))
  in
  let u_lower u expected =
    test (Printf.sprintf "lower of U+%04X" u) (fun () ->
        equal same [| Some expected |] (lower [| Some (utf_8 u) |]))
  in
  group "full mappings named by the data files"
    [
      (* SpecialCasing.txt, unconditional entries. *)
      u_upper 0xDF "SS";
      u_lower 0x130 "i\xcc\x87";
      u_upper 0xFB03 "FFI";
      u_upper 0xFB00 "FF";
      u_upper 0x149 "\xca\xbcN";
      u_upper 0x390 "\xce\x99\xcc\x88\xcc\x81";
      (* UnicodeData.txt, simple mappings of different byte widths. *)
      u_lower 0x212A "k";
      u_lower 0x2126 "\xcf\x89";
      u_lower 0x1E9E "\xc3\x9f";
      u_lower 0x23A "\xe2\xb1\xa5";
      u_upper 0x1C5 "\xc7\x84";
      u_lower 0x1C5 "\xc7\x86";
      u_lower 0x10400 "\xf0\x90\x90\xa8";
      (* No language's tailoring: dotted and dotless I as anywhere else. *)
      u_lower 0x49 "i";
      u_upper 0x69 "I";
      u_upper 0x131 "I";
      u_lower 0x307 (utf_8 0x307);
      (* No mapping at all. *)
      u_lower 0x65E5 (utf_8 0x65E5);
      u_upper 0x10FFFF (utf_8 0x10FFFF);
    ]

let lengths =
  test "a row of every length around the fast path's word" (fun () ->
      (* Rows of length 0 to 33, all ASCII and with one non-ASCII scalar
         placed at each position, in a column with and without a null. *)
      let ascii n = String.init n (fun i -> if i mod 3 = 0 then 'B' else 'x') in
      let vs =
        Array.of_list
          (List.concat_map
             (fun n ->
               Some (ascii n)
               :: List.init (n + 1) (fun p ->
                      Some (String.sub (ascii n) 0 p ^ "\xce\xa3"
                           ^ String.sub (ascii n) p (n - p))))
             (List.init 34 Fun.id))
      in
      let expected = Array.map (Option.map expect_lower) vs in
      equal same expected (lower vs);
      let with_null = Array.append [| None |] vs in
      equal same
        (Array.append [| None |] expected)
        (lower with_null);
      equal same (Array.map (Option.map expect_upper) vs) (upper vs))

let shapes =
  group "shapes and nulls"
    [
      test "no rows" (fun () ->
          equal same [||] (lower [||]);
          equal same [||] (upper [||]));
      test "an all-empty column" (fun () ->
          equal same [| Some ""; Some "" |] (lower [| Some ""; Some "" |]));
      test "nulls stay null, in an ASCII column" (fun () ->
          equal same
            [| None; Some "ab"; None; Some "" |]
            (lower [| None; Some "aB"; None; Some "" |]));
      test "rows after a longer mapped row sit at their offsets" (fun () ->
          equal same
            [| Some "i\xcc\x87"; Some "a"; Some "ss"; Some "\xc3\x9f" |]
            (lower
               [|
                 Some "\xc4\xb0"; Some "A"; Some "SS"; Some "\xe1\xba\x9e";
               |]));
      test "a slice of a column, its offsets not from zero" (fun () ->
          let sliced vs =
            let q = Query.(of_table (texts vs) |> slice ~offset:1 ~length:2) in
            let e = Expr.[ "out" := Str.lower s ] in
            let t = require_ok ~pp:Error.pp (Query.run (Query.select e q)) in
            Column.options Kind.string (column t "out")
          in
          equal same
            [| Some "b"; Some "c" |]
            (sliced [| Some "A"; Some "B"; Some "C"; Some "D" |]);
          let e_acute = Some "\xc3\x89" in
          equal same
            [| Some "\xc3\xa9"; None |]
            (sliced [| e_acute; e_acute; None; e_acute |]));
      test "several batches lower as one" (fun () ->
          let t =
            of_batches
              [
                texts [| Some "\xce\x91\xce\xa3" |];
                texts [| Some "\xc3\x9f"; None |];
              ]
          in
          equal same
            [| Some "\xce\xb1\xcf\x82"; Some "\xc3\x9f"; None |]
            (Column.options Kind.string (result (Expr.Str.lower s) t)));
      test "a categorical lowers as its text" (fun () ->
          let ty = Type.categorical [| "aB"; "Ab"; "\xce\xa3" |] in
          let rows = [| Some "aB"; None; Some "\xce\xa3" |] in
          let t = v [ ("s", Column.of_options ty rows) ] in
          equal same
            [| Some "ab"; None; Some "\xcf\x83" |]
            (Column.options Kind.string (result (Expr.Str.lower s) t)));
    ]

let invalid_utf_8 =
  (* A string column holds valid UTF-8; talon refuses other bytes where they
     enter, so no case mapping ever reads them. *)
  test "a string column cannot hold invalid UTF-8" (fun () ->
      raises_match Exn.invalid_arg (fun () ->
          Column.of_options Type.string [| Some "\xc3" |]))

let () =
  exit @@ run "Case"
    [
      group "Str.lower and Str.upper"
        [
          every_scalar;
          sigma_edges;
          law;
          final_sigma_cases;
          mapping_cases;
          lengths;
          shapes;
          invalid_utf_8;
        ];
      group "Str.trim" [ every_white_space ];
    ]
