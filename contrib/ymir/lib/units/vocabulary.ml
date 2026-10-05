(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

module Smap = Map.Make (String)

type prefixing = Prefixable | Bare

(* What an entry can be in a spelling, read from its unit: [Number] when its
   number is not a power of ten (au, eV, °); [Power] when it is one symbol to
   the power [power] times a power of ten (m, g, L, sr); [Named] for two or more
   symbols times a power of ten (N, Jy); [Decimal] for a power of ten alone. *)
type kind =
  | Number
  | Power of { term : Unit.term; power : int * int }
  | Named
  | Decimal

(* An entry's number is 10^[ten], [ten] 0 for a [Number]. *)
type entry = {
  index : int;
  symbol : string;
  prefixing : prefixing;
  unit : Unit.t;
  kind : kind;
  ten : int * int;
}

(* The entries in the author's order, and each by its symbol. *)
type t = { entries : entry list; symbols : entry Smap.t }

(* SI prefixes *)

(* Each prefix's power of ten and its spellings, the first the one a spelling
   writes. Micro reads as u, as µ (U+00B5) or as μ (U+03BC), and writes μ. *)
let prefixes =
  [
    (-30, [ "q" ]);
    (-27, [ "r" ]);
    (-24, [ "y" ]);
    (-21, [ "z" ]);
    (-18, [ "a" ]);
    (-15, [ "f" ]);
    (-12, [ "p" ]);
    (-9, [ "n" ]);
    (-6, [ "\xce\xbc"; "\xc2\xb5"; "u" ]);
    (-3, [ "m" ]);
    (-2, [ "c" ]);
    (-1, [ "d" ]);
    (1, [ "da" ]);
    (2, [ "h" ]);
    (3, [ "k" ]);
    (6, [ "M" ]);
    (9, [ "G" ]);
    (12, [ "T" ]);
    (15, [ "P" ]);
    (18, [ "E" ]);
    (21, [ "Z" ]);
    (24, [ "Y" ]);
    (27, [ "R" ]);
    (30, [ "Q" ]);
  ]

(* Every spelling with its power, longest first, so da precedes d. *)
let spellings =
  List.concat_map (fun (k, ps) -> List.map (fun p -> (p, k)) ps) prefixes
  |> List.stable_sort (fun (p, _) (q, _) ->
      Int.compare (String.length q) (String.length p))

let prefix_text k = List.hd (List.assoc k prefixes)
let ten = List.map (fun (k, _) -> (k, Unit.decimal (strf "1e%d" k))) prefixes

(* Exponents *)

(* Exponents are reduced fractions [(num, den)] with [den >= 1], added and
   multiplied as the unit algebra does. Arithmetic raises [No_spelling] when a
   result leaves [int]. *)

exception No_spelling

let zero = (0, 1)
let exact = function Some q -> q | None -> raise_notrace No_spelling
let plus a b = exact (Unit.exponent_sum a b)
let times a b = exact (Unit.exponent_product a b)
let over x (c, d) = if c < 0 then times x (-d, -c) else times x (d, c)
let minus x (c, d) = plus x (-c, d)
let magnitude (n, d) = (abs n, d)

(* [order x y] compares the fractions [x] and [y] exactly, through their
   continued fractions, so it never overflows. *)
let rec order (a, b) (c, d) =
  let floor n m = if n mod m < 0 then (n / m) - 1 else n / m in
  let rest n m = if n mod m < 0 then (n mod m) + m else n mod m in
  match Int.compare (floor a b) (floor c d) with
  | 0 -> (
      match (rest a b, rest c d) with
      | 0, 0 -> 0
      | 0, _ -> -1
      | _, 0 -> 1
      | r, r' -> order (d, r') (b, r))
  | k -> k

(* Axes *)

(* The axes a power of ten leaves alone: each symbol, π, each prime other than
   2 and 5, and [Excess], 2's exponent less 5's. A unit is a power of ten times
   its exponents on these axes. *)
type axis = Term of Unit.term | Excess

let equal_term (a : Unit.term) (b : Unit.term) =
  match (a, b) with
  | Prime p, Prime q -> p = q
  | Pi, Pi -> true
  | Symbol a, Symbol b ->
      String.equal a.name b.name && Option.equal String.equal a.scope b.scope
  | (Prime _ | Pi | Symbol _), _ -> false

let exponent t u =
  match List.find_opt (fun (t', _, _) -> equal_term t t') (Unit.terms u) with
  | Some (_, n, d) -> (n, d)
  | None -> zero

let on axis u =
  match axis with
  | Term t -> exponent t u
  | Excess -> minus (exponent (Prime 2) u) (exponent (Prime 5) u)

let symbol_terms u =
  List.filter_map
    (function
      | (Unit.Symbol _ as t), n, d -> Some (t, (n, d))
      | (Prime _ | Pi), _, _ -> None)
    (Unit.terms u)

(* [number_axes u] is the axes other than symbols on which [u]'s exponent is
   not 0. It is empty iff [u]'s number is a power of ten. *)
let number_axes u =
  let number = function
    | (Unit.Pi as t), _, _ -> Some (Term t)
    | (Prime p as t), _, _ when p <> 2 && p <> 5 -> Some (Term t)
    | (Prime _ | Symbol _), _, _ -> None
  in
  let axes = List.filter_map number (Unit.terms u) in
  if exponent (Prime 2) u = exponent (Prime 5) u then axes
  else axes @ [ Excess ]

let axes u = List.map (fun (t, _) -> Term t) (symbol_terms u) @ number_axes u

(* [decimal u] is [Some t] iff [u] is 10^t. *)
let decimal u =
  match Unit.terms u with
  | [] -> Some zero
  | [ (Prime 2, n, d); (Prime 5, n', d') ] when n = n' && d = d' -> Some (n, d)
  | _ -> None

let kind u =
  if number_axes u <> [] then Number
  else
    match symbol_terms u with
    | [] -> Decimal
    | [ (term, power) ] -> Power { term; power }
    | _ -> Named

(* Constructors *)

let make fn triples =
  let entry index (symbol, prefixing, unit) =
    let kind = kind unit in
    let ten = if kind = Number then zero else exponent (Prime 5) unit in
    { index; symbol; prefixing; unit; kind; ten }
  in
  let entries = List.mapi entry triples in
  let add symbols e =
    if e.symbol = "" then invalid_arg (strf "%s: a symbol is empty" fn);
    if Smap.mem e.symbol symbols then
      invalid_arg (strf "%s: %S is given twice" fn e.symbol);
    Smap.add e.symbol e symbols
  in
  let symbols = List.fold_left add Smap.empty entries in
  (* Each string a prefix and a prefixable symbol spell must read one way, or be
     a symbol, which [lookup] reads first. *)
  let read readings e =
    let reading readings (p, _) =
      let s = p ^ e.symbol in
      if Smap.mem s symbols then readings
      else
        match Smap.find_opt s readings with
        | Some (p', symbol') ->
            invalid_arg
              (strf "%s: %S reads as %S on %S and as %S on %S" fn s p' symbol' p
                 e.symbol)
        | None -> Smap.add s (p, e.symbol) readings
    in
    match e.prefixing with
    | Bare -> readings
    | Prefixable -> List.fold_left reading readings spellings
  in
  ignore (List.fold_left read Smap.empty entries);
  { entries; symbols }

let v entries = make "Vocabulary.v" entries

let union a b =
  let triple e = (e.symbol, e.prefixing, e.unit) in
  make "Vocabulary.union" (List.map triple (a.entries @ b.entries))

(* Lookup *)

let lookup voc s =
  match Smap.find_opt s voc.symbols with
  | Some e -> Some e.unit
  | None ->
      let prefixed (p, k) =
        if not (String.starts_with ~prefix:p s) then None
        else
          let n = String.length p in
          match
            Smap.find_opt (String.sub s n (String.length s - n)) voc.symbols
          with
          | Some { prefixing = Prefixable; unit; _ } ->
              Some Unit.(List.assoc k ten * unit)
          | Some { prefixing = Bare; _ } | None -> None
      in
      List.find_map prefixed spellings

(* Spelling *)

type word = { prefix : int; symbol : string; num : int; den : int }
type spelling = { decade : int; words : word list }

let word prefix (e : entry) (num, den) = { prefix; symbol = e.symbol; num; den }
let is_power e = match e.kind with Power _ -> true | _ -> false
let of_kind voc k = List.filter (fun e -> e.kind = k) voc.entries

(* [powers_of voc s] is [voc]'s entries of the one symbol [s], in order. *)
let powers_of voc s =
  let of_s e =
    match e.kind with Power p -> equal_term p.term s | _ -> false
  in
  List.filter of_s voc.entries

(* [attempt f] is [f ()], or [None] when it raises [No_spelling]. *)
let attempt f = match f () with v -> v | exception No_spelling -> None

(* [div_power r e q] is [r / e^q]. It raises [No_spelling] when the power or the
   quotient leaves the unit algebra's bounds, the only failure of [Unit.power]
   and [Unit.( / )]. *)
let div_power r e (num, den) =
  match Unit.(r / power "Vocabulary.spell" e num den) with
  | r -> r
  | exception Invalid_argument _ -> raise_notrace No_spelling

let divide r words =
  List.fold_left (fun r (e, q) -> div_power r e.unit q) r words

(* [lex cs] is the first of the comparisons [cs] that is not 0. *)
let lex cs = Option.value ~default:0 (List.find_opt (fun c -> c <> 0) cs)

(* [best compare xs] is the least of [xs], the first of equals. *)
let best compare xs =
  let keep acc x =
    match acc with
    | Some y when compare x y >= 0 -> acc
    | Some _ | None -> Some x
  in
  List.fold_left keep None xs

let rec pairs = function
  | [] -> []
  | x :: xs -> List.map (fun y -> (x, y)) xs @ pairs xs

(* [takes voc e p] is [true] iff [e]'s word can carry the SI prefix of 10^p: a
   prefix names [p], [e] takes prefixes, and no spelling of that prefix on [e]'s
   symbol is a symbol of [voc], which [lookup] would read instead. *)
let takes voc e p =
  e.prefixing = Prefixable
  &&
  match List.assoc_opt p prefixes with
  | None -> false
  | Some ps ->
      not (List.exists (fun s -> Smap.mem (s ^ e.symbol) voc.symbols) ps)

(* [allowed u] is [true] on the exponents a word of [u]'s spelling may have: a
   word's exponent is an integer when all of [u]'s exponents are, so a
   fractional exponent comes only from [u] itself. *)
let allowed u =
  let whole = List.for_all (fun (_, _, d) -> d = 1) (Unit.terms u) in
  fun (_, d) -> d = 1 || not whole

(* [one_word voc u] is the best word (10^p e)^q equal to [u]: without a prefix,
   then of an entry whose unit has no number, then with its prefix on a positive
   exponent, then with the least |q|, then of the earliest entry. *)
let one_word voc u =
  let allowed = allowed u in
  let candidate e =
    match axes e.unit with
    | [] -> None
    | a :: _ -> (
        let x = on a u in
        if x = zero then None
        else
          let q = over x (on a e.unit) in
          if not (allowed q) then None
          else
            match decimal (div_power u e.unit q) with
            | None -> None
            | Some t when t = zero -> Some (0, e, q)
            | Some t -> (
                match over t q with
                | p, 1 when takes voc e p -> Some (p, e, q)
                | _ -> None))
  in
  let has_number e = e.kind = Number || e.ten <> zero in
  let compare (p, e, q) (p', e', q') =
    lex
      [
        Bool.compare (p <> 0) (p' <> 0);
        Bool.compare (has_number e) (has_number e');
        Bool.compare (p <> 0 && fst q < 0) (p' <> 0 && fst q' < 0);
        order (magnitude q) (magnitude q');
        Int.compare e.index e'.index;
      ]
  in
  List.filter_map (fun e -> attempt (fun () -> candidate e)) voc.entries
  |> best compare
  |> Option.map (fun (p, e, q) -> word p e q)

(* [number_words voc u] is each choice of the fewest entries of [voc], at most
   two, whose numbers are not powers of ten, with the exponents that give [u]'s
   number up to a power of ten; only the empty choice when [u]'s number is a
   power of ten. *)
let number_words voc u =
  let fits ws = number_axes (divide u ws) = [] in
  let single e =
    match number_axes e.unit with
    | [] -> None
    | a :: _ ->
        let q = over (on a u) (on a e.unit) in
        if q <> zero && fits [ (e, q) ] then Some [ (e, q) ] else None
  in
  (* Two entries' exponents solve the system of two axes on which the entries
     are independent; when a solution fits every axis, it is the only one. A
     pair of equal axes is dependent. *)
  let pair (a, b) =
    let solve (k, k') =
      let a1 = on k a.unit and b1 = on k b.unit and c1 = on k u in
      let a2 = on k' a.unit and b2 = on k' b.unit and c2 = on k' u in
      let det = minus (times a1 b2) (times a2 b1) in
      if det = zero then None
      else
        let qa = over (minus (times c1 b2) (times c2 b1)) det in
        let qb = over (minus (times a1 c2) (times a2 c1)) det in
        let ws = [ (a, qa); (b, qb) ] in
        if qa <> zero && qb <> zero && fits ws then Some ws else None
    in
    pairs (number_axes a.unit @ number_axes b.unit)
    |> List.find_map (fun ks -> attempt (fun () -> solve ks))
  in
  if number_axes u = [] then [ [] ]
  else
    match
      List.filter_map
        (fun e -> attempt (fun () -> single e))
        (of_kind voc Number)
    with
    | _ :: _ as singles -> singles
    | [] -> List.filter_map pair (pairs (of_kind voc Number))

(* [symbol_word voc s x] is the word for the symbol [s] to the power [x], of an
   entry whose unit is s^a times a power of ten, with exponent x/a: preferring
   no power of ten, an integer exponent, the least exponent, then the earliest
   entry. *)
let symbol_word voc s x =
  let candidate e =
    match e.kind with
    | Power { power; _ } -> attempt (fun () -> Some (e, over x power))
    | Number | Named | Decimal -> None
  in
  let compare (e, q) (e', q') =
    lex
      [
        Bool.compare (e.ten <> zero) (e'.ten <> zero);
        Bool.compare (snd q <> 1) (snd q' <> 1);
        order (magnitude q) (magnitude q');
        Int.compare e.index e'.index;
      ]
  in
  best compare (List.filter_map candidate (powers_of voc s))

(* [place voc t words] writes [words] with 10^t as the SI prefix of the first
   word, in written order, that takes it, and as the decade when none does. A
   word of one symbol takes it through any entry of that symbol to the same
   power, one without a power of ten first, then the earliest: (10^p d)^q is
   10^t e^q when p = t/q + ten e - ten d. *)
let place voc t words =
  let plain (e, q) = word 0 e q in
  let alternatives e =
    match e.kind with
    | Power { term; power; _ } ->
        let same d =
          match d.kind with Power p -> p.power = power | _ -> false
        in
        List.filter same (powers_of voc term)
        |> List.stable_sort (fun d d' ->
            Bool.compare (d.ten <> zero) (d'.ten <> zero))
    | Number | Named | Decimal -> [ e ]
  in
  let prefixed (e, q) =
    let prefix d =
      attempt (fun () ->
          match plus (over (t, 1) q) (minus e.ten d.ten) with
          | 0, 1 -> Some (word 0 d q)
          | p, 1 when takes voc d p -> Some (word p d q)
          | _ -> None)
    in
    List.find_map prefix (alternatives e)
  in
  let rec first = function
    | [] -> None
    | w :: ws -> (
        match prefixed w with
        | Some w -> Some (w :: List.map plain ws)
        | None -> Option.map (fun ws -> plain w :: ws) (first ws))
  in
  if t = 0 then { decade = 0; words = List.map plain words }
  else
    match first words with
    | Some words -> { decade = 0; words }
    | None -> { decade = t; words = List.map plain words }

(* A cost is a spelling's items, a decade counting one; whether it has a named
   word; the sum of its exponents' magnitudes, a decade counting one; whether
   it has a decade; its named word's index, -1 for none; and its number words'
   indices. The cheapest is the least, in that order. *)
let compare_cost ((n, h, s, d, i, ns), _) ((n', h', s', d', i', ns'), _) =
  lex
    [
      Int.compare n n';
      Bool.compare h h';
      order s s';
      Bool.compare d d';
      Int.compare i i';
      List.compare Int.compare ns ns';
    ]

(* Words are written with positive exponents first, then negative ones; within
   each, number and named words before symbol words, in entry order. *)
let written (e, q) (e', q') =
  lex
    [
      Bool.compare (fst q < 0) (fst q' < 0);
      Bool.compare (is_power e) (is_power e');
      Int.compare e.index e'.index;
    ]

(* [product voc allowed numbers r head] is the spelling of [r] times the
   [numbers] words: the [head] word if any, one word per symbol left, and the
   power of ten left, with its cost, when every word's exponent is [allowed]. *)
let product voc allowed numbers r head =
  let r, named =
    match head with
    | None -> (r, [])
    | Some (h, q) -> (div_power r h.unit q, [ (h, q) ])
  in
  let symbol (s, x) =
    match symbol_word voc s x with
    | Some w -> w
    | None -> raise_notrace No_spelling
  in
  let symbols = List.map symbol (symbol_terms r) in
  let words = numbers @ named @ symbols in
  match decimal (divide r symbols) with
  | Some (t, 1) when List.for_all (fun (_, q) -> allowed q) words ->
      let words = List.stable_sort written words in
      let s = place voc t words in
      let decade = s.decade <> 0 in
      let magnitudes acc (_, q) = plus acc (magnitude q) in
      let size =
        List.fold_left magnitudes (if decade then (1, 1) else zero) words
      in
      let index (e, _) = e.index in
      let cost =
        ( List.length words + Bool.to_int decade,
          Option.is_some head,
          size,
          decade,
          Option.fold ~none:(-1) ~some:index head,
          List.map index numbers )
      in
      Some (cost, s)
  | Some _ | None -> None

(* [compound voc u] is the cheapest product of number words, at most one named
   word with an exponent that cancels one of its symbols, symbol words and a
   power of ten that is [u]. *)
let compound voc u =
  let allowed = allowed u in
  let spellings numbers =
    match divide u numbers with
    | exception No_spelling -> []
    | r ->
        let cancels h (s, a) =
          let x = exponent s r in
          if x = zero then None else attempt (fun () -> Some (h, over x a))
        in
        let heads h = List.filter_map (cancels h) (symbol_terms h.unit) in
        let heads =
          None
          :: List.map Option.some (List.concat_map heads (of_kind voc Named))
        in
        List.filter_map
          (fun head -> attempt (fun () -> product voc allowed numbers r head))
          heads
  in
  List.concat_map spellings (number_words voc u)
  |> best compare_cost |> Option.map snd

let spell voc u =
  match List.find_opt (fun e -> Unit.equal e.unit u) voc.entries with
  | Some e -> Some { decade = 0; words = [ word 0 e (1, 1) ] }
  | None -> (
      match one_word voc u with
      | Some w -> Some { decade = 0; words = [ w ] }
      | None -> compound voc u)

(* Formatting *)

let word_text w =
  let base =
    if w.prefix = 0 then w.symbol else prefix_text w.prefix ^ w.symbol
  in
  if w.num = 1 && w.den = 1 then base
  else if w.den = 1 then strf "%s^%d" base w.num
  else strf "%s^%d/%d" base w.num w.den

let spelling_text s =
  let decade = if s.decade = 0 then [] else [ strf "1e%d" s.decade ] in
  match decade @ List.map word_text s.words with
  | [] -> "1"
  | items -> String.concat " " items

let pp voc ppf u =
  match spell voc u with
  | Some s -> Format.pp_print_string ppf (spelling_text s)
  | None -> Unit.pp ppf u

(* The SI *)

let si =
  let open Unit in
  v
    [
      ("kg", Bare, kilogram);
      ("A", Prefixable, ampere);
      ("m", Prefixable, metre);
      ("s", Prefixable, second);
      ("K", Prefixable, kelvin);
      ("mol", Prefixable, mole);
      ("cd", Prefixable, candela);
      ("rad", Prefixable, radian);
      ("sr", Prefixable, steradian);
      ("Hz", Prefixable, hertz);
      ("N", Prefixable, newton);
      ("Pa", Prefixable, pascal);
      ("J", Prefixable, joule);
      ("W", Prefixable, watt);
      ("C", Prefixable, coulomb);
      ("V", Prefixable, volt);
      ("F", Prefixable, farad);
      ("\xce\xa9", Prefixable, ohm);
      ("S", Prefixable, siemens);
      ("Wb", Prefixable, weber);
      ("T", Prefixable, tesla);
      ("H", Prefixable, henry);
      ("lm", Prefixable, lumen);
      ("lx", Prefixable, lux);
      ("g", Prefixable, gram);
      ("t", Prefixable, tonne);
      ("L", Prefixable, litre);
      ("l", Prefixable, litre);
      ("min", Bare, minute);
      ("h", Bare, hour);
      ("d", Bare, day);
      ("ha", Bare, hectare);
      ("au", Bare, astronomical_unit);
      ("\xc2\xb0", Bare, degree);
      ("\xe2\x80\xb2", Bare, arcminute);
      ("\xe2\x80\xb3", Bare, arcsecond);
      ("eV", Prefixable, electronvolt);
    ]
