(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

exception Rejected of string

let fail fmt = Printf.ksprintf (fun msg -> raise (Rejected msg)) fmt

(* Sets of code points, as sorted disjoint inclusive ranges. *)

type set = (int * int) list

let max_scalar = 0x10FFFF

let normalize ranges =
  let rec merge acc = function
    | [] -> List.rev acc
    | (lo, hi) :: rest -> (
        match acc with
        | (plo, phi) :: acc when lo <= phi + 1 ->
            merge ((plo, max phi hi) :: acc) rest
        | acc -> merge ((lo, hi) :: acc) rest)
  in
  merge [] (List.sort compare ranges)

let union a b = normalize (List.rev_append a b)

let complement s =
  let rec fill from = function
    | [] -> if from <= max_scalar then [ (from, max_scalar) ] else []
    | (lo, hi) :: rest ->
        if from < lo then (from, lo - 1) :: fill (hi + 1) rest
        else fill (hi + 1) rest
  in
  fill 0 s

(* Unicode properties *)

let index_of_category : Uucp.Gc.t -> int = function
  | `Cc -> 0 | `Cf -> 1 | `Cn -> 2 | `Co -> 3 | `Cs -> 4 | `Ll -> 5 | `Lm -> 6
  | `Lo -> 7 | `Lt -> 8 | `Lu -> 9 | `Mc -> 10 | `Me -> 11 | `Mn -> 12
  | `Nd -> 13 | `Nl -> 14 | `No -> 15 | `Pc -> 16 | `Pd -> 17 | `Pe -> 18
  | `Pf -> 19 | `Pi -> 20 | `Po -> 21 | `Ps -> 22 | `Sc -> 23 | `Sk -> 24
  | `Sm -> 25 | `So -> 26 | `Zl -> 27 | `Zp -> 28 | `Zs -> 29
[@@ocamlformat "disable"]

let alphabetic = 30

(* A value computed on first use and kept. A domain that finds the cell empty
   while another is filling it computes the same value again, harmlessly. *)
let once f =
  let cell = ref None in
  fun () ->
    match !cell with
    | Some v -> v
    | None ->
        let v = f () in
        cell := Some v;
        v

(* The general categories and the alphabetic property as range sets, computed in
   one pass over the code points the first time a class needs them. *)
let properties =
  once (fun () ->
      let n = alphabetic + 1 in
      let sets = Array.make n [] in
      let starts = Array.make n (-1) and lasts = Array.make n (-2) in
      let add i cp =
        if lasts.(i) = cp - 1 then lasts.(i) <- cp
        else begin
          if starts.(i) >= 0 then
            sets.(i) <- (starts.(i), lasts.(i)) :: sets.(i);
          starts.(i) <- cp;
          lasts.(i) <- cp
        end
      in
      for cp = 0 to max_scalar do
        if cp < 0xD800 || cp > 0xDFFF then begin
          let u = Uchar.unsafe_of_int cp in
          add (index_of_category (Uucp.Gc.general_category u)) cp;
          if Uucp.Alpha.is_alphabetic u then add alphabetic cp
        end
      done;
      Array.mapi
        (fun i ranges ->
          List.rev
            (if starts.(i) >= 0 then (starts.(i), lasts.(i)) :: ranges
             else ranges))
        sets)

let property i = (properties ()).(i)

let categories gcs =
  List.fold_left (fun s gc -> union s (property (index_of_category gc))) [] gcs

(* Property names, short and long, matched ignoring case, spaces, hyphens and
   underscores. *)
let category_names =
  [
    ("c", "other", [ `Cc; `Cf; `Cn; `Co; `Cs ]);
    ("cc", "control", [ `Cc ]);
    ("cf", "format", [ `Cf ]);
    ("cn", "unassigned", [ `Cn ]);
    ("co", "privateuse", [ `Co ]);
    ("cs", "surrogate", [ `Cs ]);
    ("l", "letter", [ `Ll; `Lm; `Lo; `Lt; `Lu ]);
    ("lc", "casedletter", [ `Ll; `Lt; `Lu ]);
    ("ll", "lowercaseletter", [ `Ll ]);
    ("lm", "modifierletter", [ `Lm ]);
    ("lo", "otherletter", [ `Lo ]);
    ("lt", "titlecaseletter", [ `Lt ]);
    ("lu", "uppercaseletter", [ `Lu ]);
    ("m", "mark", [ `Mc; `Me; `Mn ]);
    ("m", "combiningmark", [ `Mc; `Me; `Mn ]);
    ("mc", "spacingmark", [ `Mc ]);
    ("me", "enclosingmark", [ `Me ]);
    ("mn", "nonspacingmark", [ `Mn ]);
    ("n", "number", [ `Nd; `Nl; `No ]);
    ("nd", "decimalnumber", [ `Nd ]);
    ("nl", "letternumber", [ `Nl ]);
    ("no", "othernumber", [ `No ]);
    ("p", "punctuation", [ `Pc; `Pd; `Pe; `Pf; `Pi; `Po; `Ps ]);
    ("pc", "connectorpunctuation", [ `Pc ]);
    ("pd", "dashpunctuation", [ `Pd ]);
    ("pe", "closepunctuation", [ `Pe ]);
    ("pf", "finalpunctuation", [ `Pf ]);
    ("pi", "initialpunctuation", [ `Pi ]);
    ("po", "otherpunctuation", [ `Po ]);
    ("ps", "openpunctuation", [ `Ps ]);
    ("s", "symbol", [ `Sc; `Sk; `Sm; `So ]);
    ("sc", "currencysymbol", [ `Sc ]);
    ("sk", "modifiersymbol", [ `Sk ]);
    ("sm", "mathsymbol", [ `Sm ]);
    ("so", "othersymbol", [ `So ]);
    ("z", "separator", [ `Zl; `Zp; `Zs ]);
    ("zl", "lineseparator", [ `Zl ]);
    ("zp", "paragraphseparator", [ `Zp ]);
    ("zs", "spaceseparator", [ `Zs ]);
  ]

let category_of_name name =
  if name = "any" then Some [ (0, max_scalar) ]
  else
    List.find_map
      (fun (short, long, gcs) ->
        if name = short || name = long then Some (categories gcs) else None)
      category_names

let space =
  once (fun () ->
      union [ (0x09, 0x0D); (0x85, 0x85) ] (categories [ `Zl; `Zp; `Zs ]))

let digit = once (fun () -> categories [ `Nd ])

(* Word characters are the alphabetic characters, the marks, the decimal digits
   and the connector punctuation. A [\w] outside a bracket class is matched
   below U+0100 by the Latin-1 table of the engine tokenizer files are written
   for, which also counts the superscript digits and the vulgar fractions as
   word characters. *)
let word_class =
  once (fun () ->
      union (property alphabetic) (categories [ `Mc; `Me; `Mn; `Nd; `Pc ]))

let word =
  once (fun () ->
      union (word_class ()) [ (0xB2, 0xB3); (0xB9, 0xB9); (0xBC, 0xBE) ])

(* UTF-8 *)

let encode cp =
  let b = Buffer.create 4 in
  Buffer.add_utf_8_uchar b (Uchar.of_int cp);
  Buffer.contents b

(* The byte range sequences that together match exactly the UTF-8 encodings of a
   range of scalar values: the range is split at the encoding length boundaries,
   then wherever a continuation byte would not span a whole range, so that every
   sequence is a product of byte ranges. *)
let rec sequences lo hi acc =
  if lo > hi then acc
  else if lo < 0xE000 && hi > 0xD7FF then
    let acc = if hi >= 0xE000 then sequences 0xE000 hi acc else acc in
    if lo <= 0xD7FF then sequences lo 0xD7FF acc else acc
  else if lo <= 0x7F && hi > 0x7F then sequences lo 0x7F (sequences 0x80 hi acc)
  else if lo <= 0x7FF && hi > 0x7FF then
    sequences lo 0x7FF (sequences 0x800 hi acc)
  else if lo <= 0xFFFF && hi > 0xFFFF then
    sequences lo 0xFFFF (sequences 0x10000 hi acc)
  else if hi <= 0x7F then [ (lo, hi) ] :: acc
  else
    let rec split i =
      if i > 3 then None
      else
        let m = (1 lsl (6 * i)) - 1 in
        if lo land lnot m = hi land lnot m then split (i + 1)
        else if lo land m <> 0 then Some (lo lor m)
        else if hi land m <> m then Some ((hi land lnot m) - 1)
        else split (i + 1)
    in
    match split 1 with
    | Some mid -> sequences lo mid (sequences (mid + 1) hi acc)
    | None ->
        let a = encode lo and b = encode hi in
        List.init (String.length a) (fun i ->
            (Char.code a.[i], Char.code b.[i]))
        :: acc

let re_of_set set =
  let byte_range (lo, hi) = Re.rg (Char.chr lo) (Char.chr hi) in
  match List.fold_right (fun (lo, hi) acc -> sequences lo hi acc) set [] with
  | [] -> Re.empty
  | seqs -> Re.alt (List.map (fun seq -> Re.seq (List.map byte_range seq)) seqs)

(* Parsing *)

(* Whether a node can match the empty string, and whether a [^] in it can be the
   last thing a match goes through: [^] matches after a newline except at the
   very end of the text, which {!Re.bol} cannot express, so such a [^] is
   refused. *)
type node = { re : Re.t; nullable : bool; open_bol : bool }

let atom re = { re; nullable = false; open_bol = false }
let assertion re = { re; nullable = true; open_bol = false }
let epsilon = assertion Re.epsilon

let seq nodes =
  let rec open_bol = function
    | [] -> false
    | n :: rest ->
        (n.open_bol && List.for_all (fun m -> m.nullable) rest) || open_bol rest
  in
  {
    re = Re.seq (List.map (fun n -> n.re) nodes);
    nullable = List.for_all (fun n -> n.nullable) nodes;
    open_bol = open_bol nodes;
  }

let alt nodes =
  {
    re = Re.alt (List.map (fun n -> n.re) nodes);
    nullable = List.exists (fun n -> n.nullable) nodes;
    open_bol = List.exists (fun n -> n.open_bol) nodes;
  }

type parser = { s : string; mutable i : int }

let eos p = p.i >= String.length p.s
let peek p = p.s.[p.i]
let peek_next p c = p.i + 1 < String.length p.s && p.s.[p.i + 1] = c
let skip p = p.i <- p.i + 1

let accept p c =
  if (not (eos p)) && peek p = c then begin
    skip p;
    true
  end
  else false

(* The character at the current position, consumed. *)
let scalar p =
  let d = String.get_utf_8_uchar p.s p.i in
  if not (Uchar.utf_decode_is_valid d) then fail "pattern is not valid UTF-8";
  p.i <- p.i + Uchar.utf_decode_length d;
  Uchar.to_int (Uchar.utf_decode_uchar d)

let decimal_digit c = if c >= '0' && c <= '9' then Char.code c - 48 else -1
let octal_digit c = if c >= '0' && c <= '7' then Char.code c - 48 else -1

let hex_digit c =
  match c with
  | '0' .. '9' -> Char.code c - 48
  | 'a' .. 'f' -> Char.code c - 87
  | 'A' .. 'F' -> Char.code c - 55
  | _ -> -1

let digits p ~base ~max ~digit =
  let value = ref 0 and count = ref 0 in
  while (not (eos p)) && !count < max && digit (peek p) >= 0 do
    value := (!value * base) + digit (peek p);
    skip p;
    incr count
  done;
  if !count = 0 then None else Some !value

let is_scalar cp = cp <= max_scalar && (cp < 0xD800 || cp > 0xDFFF)

(* [\x{H..}] is a code point, [\xHH] a byte: only ASCII bytes are characters. *)
let hex_escape p =
  if accept p '{' then
    match digits p ~base:16 ~max:8 ~digit:hex_digit with
    | Some cp when accept p '}' && is_scalar cp -> cp
    | _ -> fail "invalid code point escape"
  else
    match digits p ~base:16 ~max:2 ~digit:hex_digit with
    | Some cp when cp < 0x80 -> cp
    | Some _ -> fail "byte escapes above \\x7F are not supported, use \\x{..}"
    | None -> fail "invalid hexadecimal escape"

(* Property names are matched ignoring case, spaces, hyphens and underscores. *)
let property_escape p ~negate =
  if not (accept p '{') then fail "invalid property escape";
  let negate = if accept p '^' then not negate else negate in
  let start = p.i in
  while (not (eos p)) && peek p <> '}' do
    skip p
  done;
  if eos p then fail "unterminated property escape";
  let name = String.sub p.s start (p.i - start) in
  skip p;
  let significant c = c <> ' ' && c <> '-' && c <> '_' in
  match
    category_of_name
      (String.lowercase_ascii
         (String.of_seq (Seq.filter significant (String.to_seq name))))
  with
  | Some set -> if negate then complement set else set
  | None -> fail "unsupported property \\p{%s}" name

(* An escape common to both contexts, after the backslash: a class or a single
   character. *)
let escape p ~in_class =
  match peek p with
  | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' -> (
      let c = peek p in
      skip p;
      let word = if in_class then word_class () else word () in
      match c with
      | 'd' -> `Set (digit ())
      | 'D' -> `Set (complement (digit ()))
      | 's' -> `Set (space ())
      | 'S' -> `Set (complement (space ()))
      | 'w' -> `Set word
      | 'W' -> `Set (complement word)
      | 'p' -> `Set (property_escape p ~negate:false)
      | 'P' -> `Set (property_escape p ~negate:true)
      | 't' -> `Char 0x09
      | 'n' -> `Char 0x0A
      | 'r' -> `Char 0x0D
      | 'f' -> `Char 0x0C
      | 'v' -> `Char 0x0B
      | 'e' -> `Char 0x1B
      | 'a' -> `Char 0x07
      | 'x' -> `Char (hex_escape p)
      | '0' ->
          `Char
            (Option.value
               (digits p ~base:8 ~max:2 ~digit:octal_digit)
               ~default:0)
      | '1' .. '9' -> fail "backreferences are not supported"
      | c -> fail "unsupported escape \\%c" c)
  | _ -> `Char (scalar p)

let class_item p =
  match peek p with
  | '[' when peek_next p ':' ->
      fail "POSIX bracket expressions are not supported"
  | '[' -> fail "nested character classes are not supported"
  | '&' when peek_next p '&' ->
      fail "character class intersection is not supported"
  | '\\' ->
      skip p;
      if eos p then fail "pattern ends with a backslash";
      if accept p 'b' then `Char 0x08 else escape p ~in_class:true
  | _ -> `Char (scalar p)

(* A leading []] is a literal, and so is a [-] that does not sit between two
   characters. *)
let bracket_class p =
  let negate = accept p '^' in
  let range_follows p =
    (not (eos p))
    && peek p = '-'
    && p.i + 1 < String.length p.s
    && not (peek_next p ']')
  in
  let rec items acc ~first =
    if eos p then fail "unterminated character class";
    if (not first) && accept p ']' then acc
    else
      match class_item p with
      | `Set s ->
          if range_follows p then fail "invalid range in character class";
          items (List.rev_append s acc) ~first:false
      | `Char lo when range_follows p -> (
          skip p;
          match class_item p with
          | `Char hi when lo <= hi -> items ((lo, hi) :: acc) ~first:false
          | `Char _ -> fail "empty range in character class"
          | `Set _ -> fail "invalid range in character class")
      | `Char c -> items ((c, c) :: acc) ~first:false
  in
  let set = normalize (items [] ~first:true) in
  atom (re_of_set (if negate then complement set else set))

let dot = once (fun () -> re_of_set (complement [ (0x0A, 0x0A) ]))

let rec regexp p =
  let rec branches acc =
    if accept p '|' then branches (branch p :: acc) else acc
  in
  match branches [ branch p ] with [ b ] -> b | bs -> alt (List.rev bs)

and branch p =
  let rec pieces acc =
    if eos p || peek p = '|' || peek p = ')' then acc
    else pieces (piece p :: acc)
  in
  match pieces [] with [] -> epsilon | [ n ] -> n | ns -> seq (List.rev ns)

and piece p = quantifiers p (atom_ p)

(* [{n}?] is an optional repetition where [{n,m}?] is a lazy one. Greediness is
   marked on both sides: {!Re.non_greedy} applies to the whole subtree, so a
   greedy quantifier under a lazy one must say so. *)
and quantifiers p node =
  match quantifier p with
  | None -> node
  | Some (min, max) ->
      let re = Re.repn node.re min max in
      let re, min =
        if accept p '?' then
          if Some min = max then (Re.greedy (Re.opt (Re.greedy re)), 0)
          else (Re.non_greedy re, min)
        else if accept p '+' then
          fail "possessive quantifiers are not supported"
        else (Re.greedy re, min)
      in
      quantifiers p
        { re; nullable = min = 0 || node.nullable; open_bol = node.open_bol }

and quantifier p =
  if eos p then None
  else
    match peek p with
    | '*' ->
        skip p;
        Some (0, None)
    | '+' ->
        skip p;
        Some (1, None)
    | '?' ->
        skip p;
        Some (0, Some 1)
    | '{' -> interval p
    | _ -> None

(* [{n}], [{n,}], [{,m}] and [{n,m}]; a brace that forms none of them is a
   literal. *)
and interval p =
  let start = p.i in
  skip p;
  let bound () = digits p ~base:10 ~max:9 ~digit:decimal_digit in
  let bounds =
    match bound () with
    | Some n when accept p '}' -> Some (n, Some n)
    | lo when accept p ',' -> (
        match (lo, bound ()) with
        | None, None -> None
        | lo, hi when accept p '}' -> Some (Option.value lo ~default:0, hi)
        | _ -> None)
    | _ -> None
  in
  match bounds with
  | Some (lo, Some hi) when lo > hi -> fail "invalid interval {%d,%d}" lo hi
  | Some (lo, hi) when lo > 100_000 || Option.value hi ~default:0 > 100_000 ->
      fail "repeat count above 100000"
  | Some _ -> bounds
  | None ->
      p.i <- start;
      None

and atom_ p =
  match peek p with
  | '(' ->
      skip p;
      group p
  | '[' ->
      skip p;
      bracket_class p
  | '.' ->
      skip p;
      atom (dot ())
  | '^' ->
      skip p;
      { re = Re.bol; nullable = true; open_bol = true }
  | '$' ->
      skip p;
      assertion Re.eol
  | '*' | '+' | '?' -> fail "nothing to repeat"
  | '{' -> (
      match interval p with
      | Some _ -> fail "nothing to repeat"
      | None ->
          skip p;
          atom (Re.char '{'))
  | '\\' -> (
      skip p;
      if eos p then fail "pattern ends with a backslash";
      match peek p with
      | 'A' ->
          skip p;
          assertion Re.bos
      | 'z' ->
          skip p;
          assertion Re.eos
      | 'Z' ->
          skip p;
          assertion Re.leol
      | 'G' ->
          skip p;
          assertion Re.start
      | 'b' | 'B' -> fail "word boundaries are not supported"
      | _ -> (
          match escape p ~in_class:false with
          | `Set s -> atom (re_of_set s)
          | `Char cp -> atom (Re.str (encode cp))))
  | _ -> atom (Re.str (encode (scalar p)))

(* Groups do not capture: nothing reads the captures. *)
and group p =
  let body () =
    let n = regexp p in
    if not (accept p ')') then fail "unmatched (";
    n
  in
  if not (accept p '?') then body ()
  else if accept p ':' then body ()
  else if accept p '#' then begin
    while (not (eos p)) && peek p <> ')' do
      skip p
    done;
    if eos p then fail "unterminated comment";
    skip p;
    epsilon
  end
  else if accept p '<' then begin
    let name_char c =
      (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c = '_'
    in
    if eos p then fail "unterminated group";
    if peek p = '=' || peek p = '!' then fail "lookaround is not supported";
    if not (name_char (peek p)) then fail "invalid group name";
    while
      (not (eos p)) && (name_char (peek p) || decimal_digit (peek p) >= 0)
    do
      skip p
    done;
    if not (accept p '>') then fail "invalid group name";
    body ()
  end
  else if eos p then fail "unterminated group"
  else
    match peek p with
    | '=' | '!' -> fail "lookaround is not supported"
    | '>' -> fail "atomic groups are not supported"
    | _ -> fail "group options are not supported"

let compile pattern =
  let p = { s = pattern; i = 0 } in
  match regexp p with
  | exception Rejected msg -> Error msg
  | node ->
      if not (eos p) then Error "unmatched )"
      else if node.open_bol then
        Error "a ^ that can end a match is not supported"
      else Ok (Re.compile node.re)
