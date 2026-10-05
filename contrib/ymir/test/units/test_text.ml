(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The canonical text: every unit's text reads back as the unit, every other
   spelling of a unit is refused naming the canonical text, text that denotes no
   unit is refused naming the reason, and the bounds of the reader. Expected
   texts follow the grammar and the coefficient rule of the interface. *)

open Windtrap
open Ymir_units
open Ymir_units_test

let text = Unit.to_string
let of_string = Unit.of_string
let result_w = result unit string

let not_canonical s t =
  Error
    (Printf.sprintf "%S is not canonical: the unit's canonical text is %S" s t)

let not_a_unit s = Printf.sprintf "%S is not a unit's canonical text: " s

(* Round trip *)

let test_pp u = equal string (text u) (Format.asprintf "%a" Unit.pp u)

let round_trip =
  group "round trip"
    [
      prop "of_string reads every canonical text back" units
        (Law.round_trip unit string text (fun s -> require_ok (of_string s)));
      prop "pp formats the canonical text" units test_pp;
    ]

(* Respellings. Each rewrites a canonical text into another text whose items
   denote the same unit. *)

let items t = String.split_on_char ' ' t
let unwords = String.concat " "
let is_digit c = c >= '0' && c <= '9'
let all_digits s = s <> "" && String.for_all is_digit s

let is_name_start c =
  (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c = '_'

(* An item with a base that takes an exponent: a name, pi, or digits below 2^62
   with no coefficient exponent or fraction. *)
let is_power item =
  (is_name_start item.[0] || String.contains item '^')
  || (all_digits item && String.length item <= 18)

let split_power item =
  match String.index_opt item '^' with
  | None -> (item, 1, 1)
  | Some i ->
      let base = String.sub item 0 i in
      let e = String.sub item (i + 1) (String.length item - i - 1) in
      let n, d =
        match String.index_opt e '/' with
        | None -> (int_of_string e, 1)
        | Some j ->
            ( int_of_string (String.sub e 0 j),
              int_of_string (String.sub e (j + 1) (String.length e - j - 1)) )
      in
      (base, n, d)

let replace_nth l k x = List.mapi (fun i y -> if i = k then x else y) l

let swap l k =
  List.mapi
    (fun i y ->
      if i = k then List.nth l (k + 1)
      else if i = k + 1 then List.nth l k
      else y)
    l

let hex_upper = "0123456789ABCDEF"

(* Per item, the respellings of that item alone. *)
let item_respellings ~first item =
  let r = ref [] in
  let add kind s = r := (kind, s) :: !r in
  if is_digit item.[0] then add "a leading zero" ("0" ^ item);
  if is_power item then begin
    let base, n, d = split_power item in
    if (n, d) = (1, 1) then add "an exponent of 1" (base ^ "^1");
    if abs n < 1 lsl 60 && d < 1 lsl 60 then
      add "a doubled exponent" (Printf.sprintf "%s^%d/%d" base (2 * n) (2 * d));
    if d = 1 && n >= 2 then
      add "a split power" (Printf.sprintf "%s %s^%d" base base (n - 1));
    if d = 1 && n <= -2 then
      add "a split power" (Printf.sprintf "%s^-1 %s^%d" base base (n + 1));
    if String.contains item '^' then begin
      let i = String.index item '^' + 1 in
      let i = if item.[i] = '-' then i + 1 else i in
      add "a zero in the exponent"
        (String.sub item 0 i ^ "0" ^ String.sub item i (String.length item - i))
    end
  end;
  if first && is_digit item.[0] && not (String.contains item '^') then
    begin match (String.index_opt item 'e', String.index_opt item '/') with
    | Some i, _ ->
        let m = String.sub item 0 i in
        let k =
          int_of_string (String.sub item (i + 1) (String.length item - i - 1))
        in
        add "a shifted mantissa" (Printf.sprintf "%s0e%d" m (k - 1))
    | None, Some i ->
        let n = String.sub item 0 i
        and d = String.sub item (i + 1) (String.length item - i - 1) in
        add "an unreduced fraction" (n ^ "0/" ^ d ^ "0")
    | None, None -> add "a shifted mantissa" (item ^ "0e-1")
    end;
  (match String.index_opt item '{' with
  | None -> ()
  | Some i ->
      let close = String.rindex item '}' in
      (* The first kept byte of the scope, encoded. *)
      let rec kept j =
        if j >= close then None
        else if item.[j] = '%' then kept (j + 3)
        else Some j
      in
      (match kept (i + 1) with
      | Some j ->
          let c = Char.code item.[j] in
          let enc =
            Printf.sprintf "%%%c%c" hex_upper.[c lsr 4] hex_upper.[c land 15]
          in
          add "an encoded kept byte"
            (String.sub item 0 j ^ enc
            ^ String.sub item (j + 1) (String.length item - j - 1))
      | None -> ());
      let lower = Bytes.of_string item in
      let rec down j =
        if j < close then
          if item.[j] <> '%' then down (j + 1)
          else begin
            Bytes.set lower (j + 1) (Char.lowercase_ascii item.[j + 1]);
            Bytes.set lower (j + 2) (Char.lowercase_ascii item.[j + 2]);
            down (j + 3)
          end
      in
      down (i + 1);
      let lower = Bytes.to_string lower in
      if lower <> item then add "lower-case hexadecimal" lower);
  List.rev !r

let respellings t =
  let its = items t in
  let n = List.length its in
  let per_item =
    List.concat
      (List.mapi
         (fun k item ->
           List.map
             (fun (kind, s) -> (kind, unwords (replace_nth its k s)))
             (item_respellings ~first:(k = 0) item))
         its)
  in
  let swaps =
    List.filter_map
      (fun k ->
        if List.nth its k = List.nth its (k + 1) then None
        else Some ("swapped items", unwords (swap its k)))
      (List.init (Int.max 0 (n - 1)) Fun.id)
  in
  (("a leading 1", "1 " ^ t) :: per_item) @ swaps

let kinds =
  [
    "a leading 1";
    "a leading zero";
    "an exponent of 1";
    "a doubled exponent";
    "a split power";
    "a zero in the exponent";
    "a shifted mantissa";
    "an unreduced fraction";
    "an encoded kept byte";
    "lower-case hexadecimal";
    "swapped items";
  ]

let test_respellings u =
  let t = text u in
  let rs = respellings t in
  List.iter (fun k -> cover k (List.exists (fun (k', _) -> k = k') rs)) kinds;
  List.iter
    (fun (kind, r) ->
      equal ~msg:kind result_w (not_canonical r t) (of_string r))
    rs

(* Units whose texts hold every kind of item. *)
let rich_units =
  let open Gen in
  let extra =
    of_list ~pp:Unit.pp
      Unit.
        [
          int 6 / int 7;
          int 1000;
          int 3;
          pi;
          root 2 pi;
          int 16777259;
          scoped ~scope:"AB z" "x";
          metre ** -3;
          second ** 2;
        ]
  in
  let+ u = units and+ e = extra in
  match Unit.(u * e) with v -> v | exception _ -> u

let respelt =
  group "respellings"
    [
      prop "every other spelling of a unit is refused naming its canonical text"
        (Gen.with_pp Unit.pp rich_units)
        test_respellings;
    ]

(* Canonical texts the coefficient rule states: with k = min (v2 Q) (v5 Q), Q
   10^-k written m or mek when an integer, n/d otherwise. *)

let canonical =
  Unit.
    [
      ("40", int 40, "4e1");
      ("65520", int 65520, "6552e1");
      ("7/5", int 7 / int 5, "14e-1");
      ("4/25", int 4 / int 25, "16e-2");
      ("3/2", int 3 / int 2, "15e-1");
      ("2/3", int 2 / int 3, "2/3");
      ("10/3", int 10 / int 3, "10/3");
      ("1/3", one / int 3, "1/3");
      ("1/6", one / int 6, "1/6");
      ("1/8", one / int 8, "125e-3");
      ("3/20", int 3 / int 20, "15e-2");
      ("astronomical_unit", astronomical_unit, "1495978707e2 m");
      ("hbar", hbar, "3313035075e-43 pi^-1 kg m^2 rad^-1 s^-1");
      ("10^1233", int 10 ** 1233, "1e1233");
      ("2^62", int 2 ** 62, "4611686018427387904");
      ("1e3 16777259", int 1000 * int 16777259, "1e3 16777259");
      ( "a prime past 2^24 squared",
        int 2305843009213693951 ** 2,
        "2305843009213693951^2" );
      ( "root 2 max_int",
        root 2 (int max_int),
        "3^1/2 715827883^1/2 2147483647^1/2" );
      ("pi^-1", pi ** -1, "pi^-1");
      ("root 2 pi", root 2 pi, "pi^1/2");
      ("m^-1/2", root 2 (metre ** -1), "m^-1/2");
      ("m^0", metre ** 0, "1");
      ( "every kept scope byte",
        scoped ~scope:"AZaz09._~:#/@+-" "x",
        "x{AZaz09._~:#/@+-}" );
      ( "encoded scope bytes",
        scoped ~scope:"\000 \"%^{}\xff" "x",
        "x{%00%20%22%25%5E%7B%7D%FF}" );
      ( "coefficient, pi, primes, symbols",
        int 6 / pi * root 3 (int 2) * symbol "s" * (symbol "m" ** 2),
        "3 pi^-1 2^4/3 m^2 s" );
      ( "a scoped symbol after its unscoped name",
        scoped ~scope:"a" "m" / symbol "m",
        "m^-1 m{a}" );
    ]

let texts =
  group "canonical text"
    [
      cases
        ~name:(fun (n, _, _) -> n)
        "to_string" canonical
        (fun (_, u, t) -> equal string t (text u));
      cases
        ~name:(fun (n, _, _) -> n)
        "of_string" canonical
        (fun (_, u, t) -> equal result_w (Ok u) (of_string t));
    ]

(* Texts that denote a unit and are not its canonical text. *)

let denoting =
  [
    ("1000 m", "1e3 m");
    ("s m", "m s");
    ("1 m", "m");
    ("m^0", "1");
    ("m^0/5", "1");
    ("m m^-1", "1");
    ("m^2/2", "m");
    ("m^1", "m");
    ("pi^1", "pi");
    ("1e0", "1");
    ("10e-1", "1");
    ("16777259e3", "1e3 16777259");
    ("50331777", "3 16777259");
    ("16777259 3", "3 16777259");
    ("2/4", "5e-1");
    ("3/1", "3");
    ("x{%61}", "x{a}");
    ("x{%7b}", "x{%7B}");
    ("4611686018427387903^1/2", "3^1/2 715827883^1/2 2147483647^1/2");
    (* (16777259^2)^((2^62 - 1)/2): the exponent's product leaves int before it
       is reduced. *)
    ("281476419553081^4611686018427387903/2", "16777259^4611686018427387903");
    ("1" ^ String.make 1233 '0', "1e1233");
    (* Items past a bound whose product is within it. *)
    ("2^4096 2^-1", text Unit.(int 2 ** 4095));
  ]

(* Text that denotes no unit, by syntax or past a bound. *)

let refused =
  [
    "";
    " ";
    "m ";
    " m";
    "m  s";
    "m^";
    "m^-";
    "m^1/";
    "m^/2";
    "m^+1";
    "m^1.5";
    "m^--1";
    "m{";
    "m{}";
    "m{a";
    "m{%4}";
    "m{%G0}";
    "m}";
    "1e";
    "1e-";
    "1.5";
    "-1";
    "+1";
    "1E3";
    "1e+3";
    "^2";
    "m\n";
    "m\t";
    "m\000";
    "m^1/0";
    "0";
    "0 m";
    "0/1";
    "1/0";
    "pi{a}";
    "\xc3\xa9";
    "m,s";
    "m*s";
    "m/s";
    "1/2/3";
    "1e3e3";
    "m^1^2";
    (* Bounds read before any arithmetic, and their neighbours past them. *)
    "1e4097";
    "1e-4097";
    "1e4096";
    "1e1234";
    "1" ^ String.make 1234 '0';
    "m^4611686018427387904";
    "m^-4611686018427387904";
    "m^1/4611686018427387904";
    "4611686018427387904^1/2";
    "1e99999999999999999999";
    "m^99999999999999999999";
    (* Items within the bounds whose product is past them. *)
    "m^4611686018427387903 m";
    "2^4095 2";
  ]

let accepted =
  [
    "1e1233";
    "1e-1233";
    "m^4611686018427387903";
    "m^-4611686018427387903";
    "m^1/4611686018427387903";
    "4611686018427387904";
    "3 16777259";
    "2305843009213693951^2";
    "16777259^4611686018427387903";
  ]

let short s = if String.length s > 40 then String.sub s 0 37 ^ "..." else s

let strictness =
  group "of_string"
    [
      cases
        ~name:(fun (s, _) -> short s)
        "names the canonical text" denoting
        (fun (s, t) -> equal result_w (not_canonical s t) (of_string s));
      cases
        ~name:(fun s -> Printf.sprintf "refuses %S" (short s))
        "refuses" refused
        (fun s ->
          starts_with ~affix:(not_a_unit s) (require_error (of_string s)));
      cases ~name:short "accepts" accepted (fun s ->
          equal string s (text (require_ok (of_string s))));
      test "states the byte of a syntax error" (fun () ->
          equal result_w
            (Error
               {|"m^" is not a unit's canonical text: expected digits at byte 2|})
            (of_string "m^"));
      test
        "refuses a coefficient with a cofactor past 2^62 without naming a text"
        (fun () ->
          (* (2^61 - 1)^2 *)
          let s = "5316911983139663487003542222693990401" in
          match of_string s with
          | Ok u -> failf "%S is accepted as the unit %S" s (text u)
          | Error e -> not_contains ~sub:"canonical text is" e);
    ]

(* Structure *)

let structure =
  group "ptree"
    [
      test "a unit is one case holding its text" (fun () ->
          equal (list string)
            [ {|the root: case "1e3 m"|} ]
            (List.map
               (Format.asprintf "%a" Nx.Ptree.pp_visit)
               (Nx.Ptree.visits Unit.ptree Unit.(kilo metre))));
    ]

(* Any string *)

let alphabet =
  Gen.of_list ~pp:Format.pp_print_char
    (List.of_seq (String.to_seq "ms pi0123^-/{}e%a1"))

let test_any s =
  match of_string s with
  | Ok u ->
      cover "accepted" true;
      equal string s (text u)
  | Error _ -> cover "accepted" false

let any_string =
  group "any string"
    [
      prop "of_string accepts only canonical texts and never raises"
        (Gen.string_of ~size:(Gen.int_range 0 8) alphabet)
        test_any;
      prop "of_string never raises on bytes" Gen.string (fun s ->
          ignore (of_string s));
    ]

let () =
  exit
    (run "Unit text"
       [ round_trip; respelt; texts; strictness; structure; any_string ])
