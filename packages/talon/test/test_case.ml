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

let strings e t = Column.options Kind.string (result e t)
let lower vs = strings (Expr.Str.lower s) (texts vs)
let upper vs = strings (Expr.Str.upper s) (texts vs)
let trim vs = strings (Expr.Str.trim s) (texts vs)
let same = array (option string)

let utf_8 u =
  let b = Buffer.create 4 in
  Buffer.add_utf_8_uchar b (Uchar.of_int u);
  Buffer.contents b

(* Every scalar value, in order. *)
let every_scalar () =
  let surrogates = 0xE000 - 0xD800 in
  Array.init (0x110000 - surrogates) (fun i ->
      if i < 0xD800 then i else i + surrogates)

(* [scalars t] are the scalar values of [t]. *)
let scalars t =
  let rec go i acc =
    if i >= String.length t then List.rev acc
    else
      let d = String.get_utf_8_uchar t i in
      go
        (i + Uchar.utf_decode_length d)
        (Uchar.to_int (Uchar.utf_decode_uchar d) :: acc)
  in
  go 0 []

(* Unicode 18.0.0's data, as gen/case.py prints it from UnicodeData.txt,
   SpecialCasing.txt, DerivedCoreProperties.txt and PropList.txt. *)

type ucd = {
  lowers : (int, string) Hashtbl.t; (* full mappings that are not the scalar *)
  uppers : (int, string) Hashtbl.t;
  cased : (int * int) list;
  ignorable : (int * int) list; (* Case_Ignorable *)
  white_space : (int * int) list;
}

let ucd =
  lazy
    (let hex f = int_of_string ("0x" ^ f) in
     let text f =
       String.concat ""
         (List.map (fun h -> utf_8 (hex h)) (String.split_on_char ' ' f))
     in
     let range body =
       match String.split_on_char '.' body with
       | [ a; ""; b ] -> (hex a, hex b)
       | _ -> failwith body
     in
     let d =
       {
         lowers = Hashtbl.create 3000;
         uppers = Hashtbl.create 3000;
         cased = [];
         ignorable = [];
         white_space = [];
       }
     in
     In_channel.with_open_text "golden/case.txt" In_channel.input_lines
     |> List.filter (fun l -> l <> "" && l.[0] <> '#')
     |> List.fold_left
          (fun d line ->
            let body = String.sub line 2 (String.length line - 2) in
            match (line.[0], String.split_on_char ';' body) with
            | 'S', [ u; l; up ] ->
                Hashtbl.replace d.lowers (hex u) (text l);
                Hashtbl.replace d.uppers (hex u) (text up);
                d
            | 'C', _ -> { d with cased = range body :: d.cased }
            | 'I', _ -> { d with ignorable = range body :: d.ignorable }
            | 'W', _ -> { d with white_space = range body :: d.white_space }
            | _ -> failwith line)
          d)

let member ranges u = List.exists (fun (a, b) -> a <= u && u <= b) ranges

let map_lower u =
  Option.value (Hashtbl.find_opt (Lazy.force ucd).lowers u) ~default:(utf_8 u)

let map_upper u =
  Option.value (Hashtbl.find_opt (Lazy.force ucd).uppers u) ~default:(utf_8 u)

(* [final_sigma us i] is Table 3-17's Final_Sigma for [us.(i)]: scanning past
   case-ignorable scalars, a cased one precedes it and none follows it. *)
let final_sigma us i =
  let { cased; ignorable; _ } = Lazy.force ucd in
  let n = Array.length us in
  let rec before j =
    if j < 0 then false
    else if member ignorable us.(j) then before (j - 1)
    else member cased us.(j)
  in
  let rec after j =
    if j >= n then true
    else if member ignorable us.(j) then after (j + 1)
    else not (member cased us.(j))
  in
  before (i - 1) && after (i + 1)

(* Default Case Conversion (Unicode 18.0.0, 3.13), the references of the
   laws. *)
let expect_lower t =
  let us = Array.of_list (scalars t) in
  let map i u =
    if u = 0x3A3 && final_sigma us i then utf_8 0x3C2 else map_lower u
  in
  String.concat "" (Array.to_list (Array.mapi map us))

let expect_upper t = String.concat "" (List.map map_upper (scalars t))

(* [mismatches name map actual us] names the scalars of [us] whose row in
   [actual] is not [map u]. *)
let mismatches name map actual us =
  List.filteri (fun i _ -> actual.(i) <> Some (map us.(i))) (Array.to_list us)
  |> List.map (Printf.sprintf "%s of U+%04X" name)

(* Generators *)

let token =
  let open Gen in
  let of_list vs = of_list ~pp:(fun ppf -> Format.fprintf ppf "%S") vs in
  frequency
    [
      ( 3,
        of_list
          [
            "\xce\xa3" (* U+03A3 *);
            "\xce\x91" (* U+0391 *);
            "\xcd\x85" (* U+0345, cased and case-ignorable *);
            "\xc3\x9f" (* U+00DF, upper to two scalars *);
            "\xc4\xb0" (* U+0130, lower to two scalars *);
            "\xce\xb0" (* U+03B0, upper to three *);
            "\xe1\xba\x9e" (* U+1E9E, shorter lowercase *);
            "\xc8\xba" (* U+023A, longer lowercase *);
            "\xc5\xbf" (* U+017F, shorter uppercase *);
            "\xef\xac\x83" (* U+FB03 *);
            "\xf0\x90\x90\x80" (* U+10400, four bytes *);
            "\xf0\x9f\x98\x80" (* U+1F600, four bytes, unmapped *);
            "\xe6\x97\xa5" (* CJK, unmapped *);
          ] );
      (3, of_list [ "a"; "Z"; "z"; "A"; " "; "'"; "."; "1"; "\xc2\xad" ]);
      (1, map (fun n -> String.make n 'q' ^ "Q") (int_range 0 20));
    ]

let pp_rows ppf rows =
  Format.pp_print_seq ~pp_sep:Format.pp_print_space
    (fun ppf -> function
      | None -> Format.pp_print_string ppf "null"
      | Some t -> Format.fprintf ppf "%S" t)
    ppf (Array.to_seq rows)

let rows =
  let text =
    Gen.map (String.concat "") (Gen.list ~size:(Gen.int_range 0 12) token)
  in
  Gen.with_pp pp_rows (Gen.array ~size:(Gen.int_range 0 8) (Gen.option text))

(* No rows, empty texts, nulls, and rows after one whose mapping is longer. *)
let shapes =
  [
    [||];
    [| Some ""; Some "" |];
    [| None; Some "aB"; None; Some "" |];
    [| Some "\xc4\xb0"; Some "A"; Some "SS"; Some "\xe1\xba\x9e" |];
  ]

let agrees name map case vs =
  let has p = Array.exists (function Some t -> p t | None -> false) vs in
  let grows t = String.length (map t) > String.length t in
  let shrinks t = String.length (map t) < String.length t in
  cover "a text with a sigma" (has (fun t -> List.mem 0x3A3 (scalars t)));
  cover "a text that the mapping lengthens" (has grows);
  cover "a text that the mapping shortens" (has shrinks);
  cover "a null row" (Array.exists Option.is_none vs);
  cover "no rows" (vs = [||]);
  equal ~msg:name same (Array.map (Option.map map) vs) (case vs)

(* A null, then rows of 0 to 33 bytes, all ASCII and with U+03A3 at each place,
   past a word of eight bytes. *)
let every_length () =
  let ascii n = String.init n (fun i -> if i mod 3 = 0 then 'B' else 'x') in
  let with_sigma n p =
    String.sub (ascii n) 0 p ^ "\xce\xa3" ^ String.sub (ascii n) p (n - p)
  in
  Array.of_list
    (None
    :: List.concat_map
         (fun n ->
           Some (ascii n) :: List.init (n + 1) (fun p -> Some (with_sigma n p)))
         (List.init 34 Fun.id))

(* Str.trim *)

let trimming =
  group "Str.trim"
    [
      test
        "removes exactly the White_Space scalars of PropList.txt at both ends"
        (fun () ->
          let white_space = (Lazy.force ucd).white_space in
          let us = every_scalar () in
          let row u = utf_8 u ^ "x" ^ utf_8 u in
          let trimmed u = if member white_space u then "x" else row u in
          let got = trim (Array.map (fun u -> Some (row u)) us) in
          equal (list string) [] (mismatches "trim" trimmed got us));
    ]

(* Str.lower *)

let lowering =
  let sigma = "\xce\xa3" and alpha = "\xce\x91" and ypog = "\xcd\x85" in
  let small = "\xcf\x83" and final = "\xcf\x82" and a = "\xce\xb1" in
  group "Str.lower"
    [
      test "maps each scalar alone to its full lowercase" (fun () ->
          let us = every_scalar () in
          let got = lower (Array.map (fun u -> Some (utf_8 u)) us) in
          equal (list string) [] (mismatches "lower" map_lower got us));
      prop "agrees with Default Case Conversion" ~examples:shapes rows
        (agrees "lower" expect_lower lower);
      cases "reads Final_Sigma as Table 3-17 states"
        ~id:(fun (t, _) -> String.escaped t)
        [
          (* Nothing cased comes before. *)
          (sigma, small);
          (alpha ^ sigma, a ^ final);
          (alpha ^ sigma ^ alpha, a ^ small ^ a);
          (* U+0345 is case-ignorable, and passed over. *)
          (ypog ^ sigma, ypog ^ small);
          (alpha ^ ypog ^ sigma, a ^ ypog ^ final);
          (alpha ^ sigma ^ ypog, a ^ final ^ ypog);
          (alpha ^ sigma ^ ypog ^ alpha, a ^ small ^ ypog ^ a);
          (alpha ^ sigma ^ " " ^ alpha, a ^ final ^ " " ^ a);
          (alpha ^ "'" ^ sigma, a ^ "'" ^ final);
          ("1" ^ sigma, "1" ^ small);
          ("a" ^ sigma, "a" ^ final);
        ]
        (fun (t, expected) ->
          equal same [| Some expected |] (lower [| Some t |]));
      test "reads Final_Sigma at every edge of Cased and Case_Ignorable"
        (fun () ->
          let { cased; ignorable; _ } = Lazy.force ucd in
          let edges =
            List.concat_map
              (fun (a, b) -> [ a - 1; a; b; b + 1 ])
              (cased @ ignorable)
            |> List.filter (fun u ->
                u >= 0 && u <= 0x10FFFF && (u < 0xD800 || u > 0xDFFF))
            |> List.sort_uniq compare
          in
          let contexts u =
            let c = utf_8 u in
            [ alpha ^ c ^ sigma; alpha ^ sigma ^ c; c ^ sigma ]
          in
          let vs =
            Array.of_list
              (List.map Option.some (List.concat_map contexts edges))
          in
          equal same (Array.map (Option.map expect_lower) vs) (lower vs));
      cases "maps the scalars the .mli names, with no language's tailoring"
        ~id:(fun (t, _) -> String.escaped t)
        [ ("\xc4\xb0", "i\xcc\x87"); ("I", "i"); ("\xcc\x87", "\xcc\x87") ]
        (fun (t, expected) ->
          equal same [| Some expected |] (lower [| Some t |]));
      test "maps rows of 0 to 33 bytes with a non-ASCII scalar at each place"
        (fun () ->
          let vs = every_length () in
          equal same (Array.map (Option.map expect_lower) vs) (lower vs));
      test "maps the rows of a slice, whose offsets do not start at zero"
        (fun () ->
          let sliced vs =
            let q = Query.(of_table (texts vs) |> slice ~offset:1 ~length:2) in
            let t =
              require_ok ~pp:Error.pp
                (Query.run (Query.select Expr.[ "out" := Str.lower s ] q))
            in
            Column.options Kind.string (column t "out")
          in
          let e_acute = Some "\xc3\x89" in
          equal same [| Some "b"; Some "c" |]
            (sliced [| Some "A"; Some "B"; Some "C"; Some "D" |]);
          equal same
            [| Some "\xc3\xa9"; None |]
            (sliced [| e_acute; e_acute; None; e_acute |]));
      test "maps several batches as one" (fun () ->
          let t =
            of_batches
              [
                texts [| Some (alpha ^ sigma) |];
                texts [| Some "\xc3\x9f"; None |];
              ]
          in
          equal same
            [| Some (a ^ final); Some "\xc3\x9f"; None |]
            (strings (Expr.Str.lower s) t));
      test "maps a categorical as its text" (fun () ->
          let ty = Type.categorical [| "aB"; "Ab"; sigma |] in
          let t =
            v [ ("s", Column.of_options ty [| Some "aB"; None; Some sigma |]) ]
          in
          equal same
            [| Some "ab"; None; Some small |]
            (strings (Expr.Str.lower s) t));
    ]

(* Str.upper *)

let uppering =
  group "Str.upper"
    [
      test "maps each scalar alone to its full uppercase" (fun () ->
          let us = every_scalar () in
          let got = upper (Array.map (fun u -> Some (utf_8 u)) us) in
          equal (list string) [] (mismatches "upper" map_upper got us));
      prop "agrees with Default Case Conversion" ~examples:shapes rows
        (agrees "upper" expect_upper upper);
      cases "maps the scalars the .mli names, with no language's tailoring"
        ~id:(fun (t, _) -> String.escaped t)
        [
          ("\xc3\x9f", "SS");
          ("\xef\xac\x83", "FFI");
          ("i", "I");
          ("\xc4\xb1", "I");
        ]
        (fun (t, expected) ->
          equal same [| Some expected |] (upper [| Some t |]));
      test "maps rows of 0 to 33 bytes with a non-ASCII scalar at each place"
        (fun () ->
          let vs = every_length () in
          equal same (Array.map (Option.map expect_upper) vs) (upper vs));
    ]

let () = exit @@ run "Case" [ trimming; lowering; uppering ]
