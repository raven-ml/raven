(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Scalars
module A = Bigarray.Array1

(* Sets of scalars

   A class is its ranges of scalars: ascending, disjoint, inclusive. The
   surrogates are ordinary numbers here (a negation holds them), and compiling
   a class to UTF-8 is what drops them. *)

let scalar_max = 0x10FFFF

let normalize ranges =
  let ordered = List.sort compare ranges in
  let rec merge = function
    | (a0, b0) :: (a1, b1) :: rest when a1 <= b0 + 1 ->
        merge ((a0, Int.max b0 b1) :: rest)
    | r :: rest -> r :: merge rest
    | [] -> []
  in
  merge ordered

let negate ranges =
  let rec go lo = function
    | [] -> if lo <= scalar_max then [ (lo, scalar_max) ] else []
    | (a, b) :: rest ->
        if a > lo then (lo, a - 1) :: go (b + 1) rest else go (b + 1) rest
  in
  go 0 ranges

(* [of_bounds b] is the ranges a table's ascending bounds delimit. *)
let of_bounds b =
  List.init
    (Array.length b / 2)
    (fun i -> (b.(2 * i), b.((2 * i) + 1) - 1))

let member sorted u =
  let rec go lo hi =
    if lo >= hi then false
    else
      let mid = (lo + hi) / 2 and a, b = sorted.((lo + hi) / 2) in
      if u < a then go lo mid else if u > b then go (mid + 1) hi else true
  in
  go 0 (Array.length sorted)

(* [closure ranges] adds to [ranges] every scalar related by simple case
   folding to one of them: the set stays closed, so a negated class closes
   before it negates. *)
let closure ranges =
  let sorted = Array.of_list ranges and orbits = Unicode_tables.case_orbits in
  let extra = ref [] in
  let i = ref 0 in
  while !i < Array.length orbits do
    let start = !i and hit = ref false in
    while orbits.(!i) >= 0 do
      if member sorted orbits.(!i) then hit := true;
      incr i
    done;
    if !hit then
      for j = start to !i - 1 do
        extra := (orbits.(j), orbits.(j)) :: !extra
      done;
    incr i
  done;
  normalize (List.rev_append !extra ranges)

(* Parsing *)

type assertion =
  | Start_text
  | End_text
  | Start_line
  | End_line
  | Word_boundary
  | Not_word_boundary

type node =
  | Empty
  | Set of (int * int) list
  | Cat of node list
  | Alt of node list
  | Repeat of { body : node; min : int; max : int option; greedy : bool }
  | Capture of int * node
  | Assertion of assertion

type flags = { case : bool; multi_line : bool; dot_all : bool }

type parser = {
  src : string;
  n : int;
  mutable pos : int;
  mutable flags : flags;
  mutable count : int;
  mutable names : (string * int) list;
}

let fail p at fmt =
  Printf.ksprintf
    (fun m ->
      invalid_arg
        (Printf.sprintf "Expr.Str.regex: %s, at byte %d of %S" m at p.src))
    fmt

let peek p = if p.pos < p.n then Some p.src.[p.pos] else None
let peek2 p = if p.pos + 1 < p.n then Some p.src.[p.pos + 1] else None

let advance p k = p.pos <- p.pos + k

let eat p c =
  if p.pos < p.n && Char.equal p.src.[p.pos] c then (
    advance p 1;
    true)
  else false

let expect p c what =
  if not (eat p c) then fail p p.pos "%s" what

(* [scalar_of p] reads one scalar of the pattern, which is valid UTF-8. *)
let scalar_of p =
  let d = String.get_utf_8_uchar p.src p.pos in
  advance p (Uchar.utf_decode_length d);
  Uchar.to_int (Uchar.utf_decode_uchar d)

(* [braced p at what] reads the text up to the next [}], past the [{] already
   eaten, and steps over the [}]. *)
let braced p at what =
  match String.index_from_opt p.src p.pos '}' with
  | None -> fail p at "unclosed %s" what
  | Some stop ->
      let body = String.sub p.src p.pos (stop - p.pos) in
      advance p (stop - p.pos + 1);
      body

let digit_value c = Char.code c - Char.code '0'
let is_digit c = c >= '0' && c <= '9'
let is_alpha c = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c = '_'
let is_alnum c = is_alpha c || is_digit c

(* [count p] reads a repetition count; an overflow cannot be a count. *)
let count p =
  let start = p.pos in
  let rec go n =
    match peek p with
    | Some c when is_digit c ->
        advance p 1;
        if n > (max_int - digit_value c) / 10 then
          fail p start "repetition count too large";
        go ((n * 10) + digit_value c)
    | _ -> n
  in
  go 0

(* A property name is read loosely: case, spaces, [_] and [-] do not count
   (UAX44-LM3, without its optional prefix "is"). *)
let loose s =
  let b = Buffer.create (String.length s) in
  String.iter
    (fun c ->
      match Char.lowercase_ascii c with
      | ' ' | '_' | '-' -> ()
      | c -> Buffer.add_char b c)
    s;
  Buffer.contents b

let property p at name =
  let lookup table key =
    List.find_map
      (fun (names, data) ->
        if List.mem key names then Some data else None)
      table
  in
  let scripts =
    List.map (fun (names, sc, _) -> (names, sc)) Unicode_tables.script
  and extensions =
    List.map (fun (names, _, scx) -> (names, scx)) Unicode_tables.script
  and categories = Unicode_tables.general_category
  and binaries = Unicode_tables.binary in
  let key = loose name in
  let found =
    match String.index_opt key '=' with
    | Some i -> (
        let value = String.sub key (i + 1) (String.length key - i - 1) in
        match String.sub key 0 i with
        | "gc" | "generalcategory" -> lookup categories value
        | "sc" | "script" -> lookup scripts value
        | "scx" | "scriptextensions" -> lookup extensions value
        | _ -> None)
    | None -> (
        match lookup categories key with
        | Some d -> Some d
        | None -> (
            match lookup scripts key with
            | Some d -> Some d
            | None -> lookup binaries key))
  in
  match found with
  | Some data -> of_bounds (Unicode_tables.bounds data)
  | None -> fail p at "unknown Unicode property %S" name

let close_if p set = if p.flags.case then closure set else set

let named p at name = close_if p (property p at name)

(* Escapes: [`Set] for a class, [`Scalar] for one code point, [`Assert]
   outside a class only. *)
let hex_value p at =
  let start = p.pos in
  let braces = eat p '{' in
  let body =
    if braces then braced p at "\\x{"
    else begin
      let body = String.sub p.src p.pos (Int.min 2 (p.n - p.pos)) in
      advance p (String.length body);
      body
    end
  in
  let is_hex c =
    is_digit c || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')
  in
  if
    body = ""
    || ((not braces) && String.length body < 2)
    || not (String.for_all is_hex body)
  then fail p start "\\x takes two hex digits or {hex digits}";
  match int_of_string_opt ("0x" ^ body) with
  | Some v when v >= 0 && v <= scalar_max && (v < 0xD800 || v > 0xDFFF) -> v
  | _ -> fail p start "\\x{%s} is not a Unicode scalar value" body

let unsupported_escape p at c =
  match c with
  | '1' .. '9' | 'k' | 'g' -> fail p at "backreference \\%c is not supported" c
  | 'C' -> fail p at "\\C, a single byte, is not supported"
  | 'Q' | 'E' -> fail p at "\\Q and \\E are not supported; use Str.quote"
  | 'R' -> fail p at "\\R is not supported; write (?:\\r\\n|[\\n-\\r])"
  | 'X' -> fail p at "\\X, a grapheme cluster, is not supported"
  | 'K' -> fail p at "\\K is not supported"
  | 'G' -> fail p at "\\G is not supported"
  | 'h' | 'H' -> fail p at "\\h and \\H are not supported; use [ \\t]"
  | 'N' -> fail p at "\\N is not supported"
  | '0' -> fail p at "\\0, an octal escape, is not supported"
  | 'Z' -> fail p at "\\Z is not supported; use \\z or $"
  | c -> fail p at "unknown escape \\%c" c

let escape p ~in_class =
  let at = p.pos - 1 in
  match peek p with
  | None -> fail p at "escape at the end of the pattern"
  | Some c -> (
      advance p 1;
      let set name negated =
        let s = named p at name in
        `Set (if negated then negate s else s)
      in
      match c with
      | 'd' -> set "digit" false
      | 'D' -> set "digit" true
      | 'w' -> set "word" false
      | 'W' -> set "word" true
      | 's' -> set "space" false
      | 'S' -> set "space" true
      | 'p' | 'P' ->
          let start = p.pos in
          let name =
            if eat p '{' then braced p at "\\p{"
            else
              match peek p with
              | Some n when is_alpha n ->
                  advance p 1;
                  String.make 1 n
              | _ -> fail p start "\\p takes a {property} name"
          in
          let s = named p at name in
          `Set (if c = 'P' then negate s else s)
      | 'x' -> `Scalar (hex_value p at)
      | 'n' -> `Scalar 0x0A
      | 't' -> `Scalar 0x09
      | 'r' -> `Scalar 0x0D
      | 'f' -> `Scalar 0x0C
      | 'v' -> `Scalar 0x0B
      | 'a' -> `Scalar 0x07
      | 'e' -> `Scalar 0x1B
      | 'A' when not in_class -> `Assert Start_text
      | 'z' when not in_class -> `Assert End_text
      | 'b' when not in_class -> `Assert Word_boundary
      | 'B' when not in_class -> `Assert Not_word_boundary
      | 'A' | 'z' | 'b' | 'B' ->
          fail p at "\\%c is not allowed in a character class" c
      | c when Char.code c < 0x80 && ((not (is_alnum c)) || c = '_') ->
          `Scalar (Char.code c)
      | c when Char.code c >= 0x80 ->
          fail p at "only ASCII punctuation escapes to itself"
      | c -> unsupported_escape p at c)

(* Character classes *)

let parse_class p =
  let at = p.pos - 1 in
  let negated = eat p '^' in
  let item () =
    let at_item = p.pos in
    match peek p with
    | None -> fail p at "unclosed character class"
    | Some '[' ->
        if peek2 p = Some ':' then
          fail p at_item "POSIX classes are not supported; use \\p{alpha}"
        else fail p at_item "nested classes are not supported; RL1.3 set \
                             operations are not part of this language"
    | Some '&' when peek2 p = Some '&' ->
        fail p at_item "class intersection (&&) is not supported"
    | Some '-' when peek2 p = Some '-' ->
        fail p at_item "class difference (--) is not supported"
    | Some '\\' ->
        advance p 1;
        escape p ~in_class:true
    | Some _ -> `Scalar (scalar_of p)
  in
  let rec items acc =
    match peek p with
    | None -> fail p at "unclosed character class"
    | Some ']' when acc <> [] ->
        advance p 1;
        acc
    | _ -> (
        match item () with
        | `Set s -> items (s :: acc)
        | `Assert _ -> assert false (* [escape ~in_class:true] refuses them. *)
        | `Scalar lo ->
            if peek p = Some '-' && peek2 p = Some '-' then
              fail p p.pos "class difference (--) is not supported";
            if peek p = Some '-' && peek2 p <> None && peek2 p <> Some ']'
            then begin
              let at_range = p.pos in
              advance p 1;
              let hi =
                match item () with
                | `Scalar hi -> hi
                | `Set _ | `Assert _ ->
                    fail p at_range "a range ends with a single character"
              in
              if hi < lo then fail p at_range "reversed character range";
              items (close_if p [ (lo, hi) ] :: acc)
            end
            else items (close_if p [ (lo, lo) ] :: acc))
  in
  let sets = items [] in
  let whole = normalize (List.concat sets) in
  Set (if negated then negate whole else whole)

(* Groups and flags *)

let group_name p at =
  let start = p.pos in
  let rec go () =
    match peek p with
    | Some c when is_alnum c ->
        advance p 1;
        go ()
    | _ -> ()
  in
  go ();
  let name = String.sub p.src start (p.pos - start) in
  if name = "" || not (is_alpha name.[0]) then
    fail p at "a group name is an ASCII identifier";
  name

(* [apply_flags p at flags text] is [flags] changed by the flag letters of a
   group: the ones before a [-] turn on, the ones after it off. *)
let apply_flags p at flags text =
  let on = ref true in
  let set flags c =
    match c with
    | '-' ->
        if not !on then fail p at "a flag group has only one -";
        on := false;
        flags
    | 'i' -> { flags with case = !on }
    | 'm' -> { flags with multi_line = !on }
    | 's' -> { flags with dot_all = !on }
    | 'x' | 'u' | 'U' | 'n' -> fail p at "flag %c is not supported" c
    | c -> fail p at "unknown flag %c" c
  in
  String.fold_left set flags text

let parse_group p at_paren parse_alt =
  (* The flags set inside a group end with it. *)
  let body () =
    let saved = p.flags in
    let node = parse_alt () in
    p.flags <- saved;
    expect p ')' "unclosed group";
    node
  in
  let open_group ?name () =
    let index =
      p.count <- p.count + 1;
      p.count
    in
    (match name with
    | Some n ->
        if List.mem_assoc n p.names then
          fail p at_paren "duplicate group name %s" n;
        p.names <- (n, index) :: p.names
    | None -> ());
    Capture (index, body ())
  in
  let named_group at =
    let name = group_name p at in
    expect p '>' "unclosed group name";
    open_group ~name ()
  in
  if not (eat p '?') then open_group ()
  else
    let at = p.pos in
    let fail_at = fail p at_paren in
    match peek p with
    | Some ('=' | '!') -> fail_at "lookahead is not supported"
    | Some '#' -> fail_at "comments are not supported"
    | Some '>' -> fail_at "atomic groups are not supported"
    | Some '|' -> fail_at "branch reset groups are not supported"
    | Some '(' -> fail_at "conditionals are not supported"
    | Some ('R' | '0' .. '9' | '&' | '+') ->
        fail_at "recursion is not supported"
    | Some 'P' when peek2 p = Some '=' ->
        fail_at "backreference (?P=name) is not supported"
    | Some 'P' when peek2 p = Some '>' ->
        fail_at "recursion is not supported"
    | Some 'P' when peek2 p = Some '<' ->
        advance p 2;
        named_group at
    | Some '<' -> (
        match peek2 p with
        | Some ('=' | '!') -> fail_at "lookbehind is not supported"
        | _ ->
            advance p 1;
            named_group at)
    | _ -> (
        let start = p.pos in
        let rec span () =
          match peek p with
          | Some (':' | ')') | None -> ()
          | Some _ ->
              advance p 1;
              span ()
        in
        span ();
        let text = String.sub p.src start (p.pos - start) in
        match peek p with
        | Some ')' ->
            if text = "" then fail p at "empty flag group";
            advance p 1;
            p.flags <- apply_flags p at p.flags text;
            Empty
        | Some ':' ->
            advance p 1;
            let saved = p.flags in
            p.flags <- apply_flags p at p.flags text;
            let node = body () in
            p.flags <- saved;
            node
        | _ -> fail p at "unknown group construct (?%s" text)

(* The grammar *)

let rec parse_alt p =
  let first = parse_cat p in
  if peek p = Some '|' then begin
    let rec branches acc =
      if eat p '|' then branches (parse_cat p :: acc) else List.rev acc
    in
    Alt (first :: branches [])
  end
  else first

and parse_cat p =
  let rec go acc =
    match peek p with
    | None | Some ('|' | ')') -> (
        match acc with
        | [] -> Empty
        | [ x ] -> x
        | xs -> Cat (List.rev xs))
    | Some _ -> go (parse_postfix p (parse_atom p) :: acc)
  in
  go []

and parse_atom p =
  let at = p.pos in
  match peek p with
  | None -> Empty
  | Some c -> (
      match c with
      | '(' ->
          advance p 1;
          parse_group p at (fun () -> parse_alt p)
      | '[' ->
          advance p 1;
          parse_class p
      | '.' ->
          advance p 1;
          if p.flags.dot_all then Set [ (0, scalar_max) ]
          else Set (negate [ (0x0A, 0x0A) ])
      | '^' ->
          advance p 1;
          Assertion (if p.flags.multi_line then Start_line else Start_text)
      | '$' ->
          advance p 1;
          Assertion (if p.flags.multi_line then End_line else End_text)
      | '\\' -> (
          advance p 1;
          match escape p ~in_class:false with
          | `Set s -> Set s
          | `Scalar u -> Set (close_if p [ (u, u) ])
          | `Assert a -> Assertion a)
      | '*' | '+' | '?' | '{' ->
          fail p at "quantifier %c has nothing to repeat" c
      | _ ->
          let u = scalar_of p in
          Set (close_if p [ (u, u) ]))

and parse_postfix p atom =
  let at = p.pos in
  let quantified =
    match peek p with
    | Some '*' ->
        advance p 1;
        Some (0, None)
    | Some '+' ->
        advance p 1;
        Some (1, None)
    | Some '?' ->
        advance p 1;
        Some (0, Some 1)
    | Some '{' ->
        advance p 1;
        let at_count = p.pos in
        let min =
          match peek p with
          | Some c when is_digit c -> count p
          | _ -> fail p at_count "expected a number after {"
        in
        let max =
          if eat p ',' then
            match peek p with
            | Some '}' -> None
            | Some c when is_digit c -> Some (count p)
            | _ -> fail p at_count "malformed repetition"
          else Some min
        in
        expect p '}' "unclosed repetition";
        (match max with
        | Some m when m < min ->
            fail p at "repetition {%d,%d} has min above max" min m
        | _ -> ());
        Some (min, max)
    | _ -> None
  in
  match quantified with
  | None -> atom
  | Some (min, max) -> (
      match peek p with
      | Some '+' -> fail p p.pos "possessive quantifiers are not supported"
      | Some ('*' | '{') ->
          fail p p.pos
            "a quantifier cannot follow a quantifier; write a group: (?:x{2}){3}"
      | _ ->
          let greedy = not (eat p '?') in
          Repeat { body = atom; min; max; greedy })

let parse src =
  if not (String.is_valid_utf_8 src) then
    invalid_arg (Printf.sprintf "Expr.Str.regex: %S is not valid UTF-8" src);
  let p =
    {
      src;
      n = String.length src;
      pos = 0;
      flags = { case = false; multi_line = false; dot_all = false };
      count = 0;
      names = [];
    }
  in
  let node = parse_alt p in
  if p.pos < p.n then fail p p.pos "unmatched )";
  (node, p.count, List.rev p.names)

(* Programs *)

type inst =
  | Sparse of int array
  | Split of int * int
  | Save of int * int
  | Assert of assertion * int
  | Accept

type t = {
  source_text : string;
  prog : inst array;
  entry : int;
  group_count : int;
  names : (string * int) list;
  required_literal : string;
}

type builder = {
  mutable code : inst array;
  mutable len : int;
  sparse : (string, int) Hashtbl.t;  (* hash-consed Sparse instructions *)
}

let emit b i =
  if b.len = Array.length b.code then
    b.code <- Array.append b.code (Array.make (Int.max 16 b.len) Accept);
  b.code.(b.len) <- i;
  b.len <- b.len + 1;
  b.len - 1

(* UTF-8 of a class

   A scalar range [lo, hi] splits into sequences of byte ranges, each encoding
   a block whose continuation bytes span all of [0x80, 0xBF]. Sorted sequences
   then form a trie with one instruction per distinct node, so a class of
   hundreds of ranges costs a small table instead of a large alternation. *)

let encoded_length u =
  if u < 0x80 then 1 else if u < 0x800 then 2 else if u < 0x10000 then 3 else 4

let utf_8_of u =
  let b = Buffer.create 4 in
  Buffer.add_utf_8_uchar b (Uchar.of_int u);
  Buffer.contents b

let bytes_of u = List.map Char.code (List.of_seq (String.to_seq (utf_8_of u)))

(* [same_length lo hi] cuts [lo, hi] at the encoded-length boundaries and around
   the surrogates, which it drops, so each part encodes one length. *)
let same_length lo hi =
  let rec cut lo hi bounds =
    match bounds with
    | [] -> [ (lo, hi) ]
    | b :: rest ->
        if lo <= b && hi > b then (lo, b) :: cut (b + 1) hi rest
        else cut lo hi rest
  in
  List.filter
    (fun (lo, _) -> lo < 0xD800 || lo > 0xDFFF)
    (cut lo hi [ 0x7F; 0x7FF; 0xD7FF; 0xDFFF; 0xFFFF ])

(* [sequences lo hi] is the runs of byte ranges of the scalars of [lo, hi] of
   one encoded length, split until each byte is a whole range. *)
let rec sequences lo hi =
  let n = encoded_length lo in
  if n = 1 then [ [ (lo, hi) ] ]
  else
    let rec split i =
      if i >= n then None
      else
        let m = (1 lsl (6 * i)) - 1 in
        if lo land lnot m <> hi land lnot m then
          if lo land m <> 0 then
            Some (sequences lo (lo lor m) @ sequences ((lo lor m) + 1) hi)
          else if hi land m <> m then
            Some
              (sequences lo ((hi land lnot m) - 1)
              @ sequences (hi land lnot m) hi)
          else split (i + 1)
        else split (i + 1)
    in
    match split 1 with
    | Some parts -> parts
    | None -> [ List.map2 (fun a b -> (a, b)) (bytes_of lo) (bytes_of hi) ]

type trie = { mutable kids : ((int * int) * trie) list }

let insert root seq =
  let rec go t = function
    | [] -> ()
    | range :: rest -> (
        match t.kids with
        | (r, child) :: _ when r = range -> go child rest
        | _ ->
            let child = { kids = [] } in
            t.kids <- (range, child) :: t.kids;
            go child rest)
  in
  go root seq

(* [emit_trie b t next] is the instruction of [t], whose leaves go to [next]. *)
let rec emit_trie b t next =
  match t.kids with
  | [] -> next
  | kids ->
      let triples =
        List.rev_map
          (fun ((lo, hi), child) -> (lo, hi, emit_trie b child next))
          kids
      in
      let key =
        String.concat ";"
          (List.map (fun (l, h, x) -> Printf.sprintf "%d,%d,%d" l h x) triples)
      in
      (match Hashtbl.find_opt b.sparse key with
      | Some pc -> pc
      | None ->
          let a = Array.make (3 * List.length triples) 0 in
          List.iteri
            (fun i (l, h, x) ->
              a.(3 * i) <- l;
              a.((3 * i) + 1) <- h;
              a.((3 * i) + 2) <- x)
            triples;
          let pc = emit b (Sparse a) in
          Hashtbl.add b.sparse key pc;
          pc)

let compile_set b ranges next =
  let root = { kids = [] } in
  List.iter
    (fun (lo, hi) ->
      List.iter
        (fun (a, b) -> List.iter (insert root) (sequences a b))
        (same_length lo hi))
    ranges;
  (* An empty class compiles to a Sparse with no ranges, so it matches
     nothing: [emit_trie] would give [next], making it match the empty text. *)
  if root.kids = [] then emit b (Sparse [||]) else emit_trie b root next

(* [nullable n] is [true] iff [n] can match the empty text. *)
let rec nullable = function
  | Empty | Assertion _ -> true
  | Set _ -> false
  | Cat ns -> List.for_all nullable ns
  | Alt ns -> List.exists nullable ns
  | Capture (_, n) -> nullable n
  | Repeat { body; min; _ } -> min = 0 || nullable body

let rec compile_node b node next =
  match node with
  | Empty -> next
  | Set ranges -> compile_set b ranges next
  | Cat nodes -> List.fold_right (compile_node b) nodes next
  | Alt nodes ->
      let rec branches = function
        | [] -> next
        | [ x ] -> compile_node b x next
        | x :: rest ->
            let r = branches rest in
            let l = compile_node b x next in
            emit b (Split (l, r))
      in
      branches nodes
  | Capture (g, body) ->
      let close = emit b (Save ((2 * g) + 1, next)) in
      let inner = compile_node b body close in
      emit b (Save (2 * g, inner))
  | Assertion a -> emit b (Assert (a, next))
  | Repeat { body; min; max; greedy } -> (
      let pair x y = if greedy then Split (x, y) else Split (y, x) in
      let copies n next =
        let e = ref next in
        for _ = 1 to n do
          e := compile_node b body !e
        done;
        !e
      in
      (* [plus ()] compiles [x+] over one copy of the body: its entry, and the
         Split after the body that loops back or leaves. *)
      let plus () =
        let split = emit b (Split (-1, -1)) in
        let entry = compile_node b body split in
        b.code.(split) <- pair entry next;
        (entry, split)
      in
      match max with
      | None when min = 0 && nullable body ->
          (* [x*] of a body that can match empty is [(?:x+)?]: an empty first
             iteration leaves by the Split after the body. The plain loop would
             reach its own Split again, drop that thread, and let a
             lower-priority alternative win. Later iterations still reach a
             visited Split and drop an empty path, the rule the .mli states. *)
          let entry, _ = plus () in
          emit b (pair entry next)
      | None when min = 0 -> snd (plus ())
      | None -> copies (min - 1) (fst (plus ()))
      | Some m ->
          let e = ref next in
          for _ = 1 to m - min do
            e := emit b (pair (compile_node b body !e) next)
          done;
          copies min !e)

(* [starts_at_text_start n] is [true] iff every match of [n] begins with [\A]
   or a [^] that is not multi-line. *)
let rec starts_at_text_start = function
  | Assertion Start_text -> true
  | Cat (n :: _) | Capture (_, n) -> starts_at_text_start n
  | Alt ns -> List.for_all starts_at_text_start ns
  | Repeat { body; min; _ } when min > 0 -> starts_at_text_start body
  | _ -> false

(* A required literal

   Text every match of the pattern contains: a row without it has no match,
   found by the kernels' byte search. A literal of valid UTF-8 occurs only on
   scalar boundaries in valid UTF-8, so the search needs no check. *)

(* [exact n] is the text [n] matches, where it matches exactly one text. *)
let rec exact = function
  | Empty -> Some ""
  | Set [ (lo, hi) ] when lo = hi ->
      Some (utf_8_of lo)
  | Cat ns ->
      List.fold_left
        (fun acc n ->
          match (acc, exact n) with
          | Some a, Some b -> Some (a ^ b)
          | _ -> None)
        (Some "") ns
  | Capture (_, n) -> exact n
  | _ -> None

let longest a b = if String.length b > String.length a then b else a

let rec required n =
  match exact n with
  | Some s -> s
  | None -> (
      match n with
      | Cat ns ->
          (* Adjacent exact parts form one literal; each other part offers its
             own. The longest wins, for the search's selectivity. *)
          let best = ref "" and run = Buffer.create 16 in
          let flush () =
            best := longest !best (Buffer.contents run);
            Buffer.clear run
          in
          List.iter
            (fun n ->
              match exact n with
              | Some s -> Buffer.add_string run s
              | None ->
                  flush ();
                  best := longest !best (required n))
            ns;
          flush ();
          !best
      | Capture (_, n) -> required n
      | Repeat { body; min; _ } when min > 0 -> required body
      | _ -> "")

let compile source_text =
  let node, group_count, names = parse source_text in
  let b = { code = [||]; len = 0; sparse = Hashtbl.create 64 } in
  let accept = emit b Accept in
  let close = emit b (Save (1, accept)) in
  let body = compile_node b node close in
  let entry = emit b (Save (0, body)) in
  (* The search starts at the match or skips one whole scalar and returns, at
     lower priority, so every scalar boundary starts a thread. An anchored
     pattern starts only where its assertion holds. *)
  let start =
    if starts_at_text_start node then entry
    else begin
      let split = emit b (Split (-1, -1)) in
      let any = compile_set b [ (0, scalar_max) ] split in
      b.code.(split) <- Split (entry, any);
      split
    end
  in
  {
    source_text;
    prog = Array.sub b.code 0 b.len;
    entry = start;
    group_count;
    names;
    required_literal = required node;
  }

let source r = r.source_text
let groups r = r.group_count
let program r = r.prog
let required r = r.required_literal
let start r = r.entry

let dfa_handles r =
  Array.for_all
    (function
      | Assert ((Start_line | End_line | Word_boundary | Not_word_boundary), _)
        ->
          false
      | _ -> true)
    r.prog

(* A metacharacter is escaped; whitespace and the rest of ASCII stand for
   themselves in this language. *)
let quote s =
  let b = Buffer.create (String.length s + 8) in
  String.iter
    (fun c ->
      (match c with
      | '\\' | '.' | '+' | '*' | '?' | '(' | ')' | '|' | '[' | ']' | '{' | '}'
      | '^' | '$' | '-' | '&' | '#' | '~' ->
          Buffer.add_char b '\\'
      | _ -> ());
      Buffer.add_char b c)
    s;
  Buffer.contents b

(* Word characters

   [\w] is Annex C's class. [\b] compares its neighbours, the row's scalars
   before and after a position, each looked up in this table. *)

let word_bounds =
  Unicode_tables.bounds
    (List.assoc "word"
       (List.concat_map
          (fun (names, data) -> List.map (fun n -> (n, data)) names)
          Unicode_tables.binary))

let is_word_scalar u = within word_bounds u 0 (Array.length word_bounds)

(* The Pike VM

   Threads are program counters in priority order. A step advances every thread
   over one byte, breaking a Sparse choice by binary search; a thread that
   reaches Accept records its slots and cuts every lower-priority thread. A
   position is a scalar boundary whenever a thread runs an assertion or the
   search starts a new thread, so a span never splits a scalar. *)

type run = {
  regex : t;
  nslots : int;
  mutable cur : int array;
  mutable next : int array;
  mutable cur_slots : int array;
  mutable next_slots : int array;
  mutable ncur : int;
  mutable nnext : int;
  seen : int array;
  mutable gen : int;
  work : int array;
  best : int array;
  mutable word_pos : int;  (* the position [before]/[after] describe, or -1 *)
  mutable before : bool;
  mutable after : bool;
}

let run r ~slots =
  let m = Array.length r.prog in
  {
    regex = r;
    nslots = slots;
    cur = Array.make m 0;
    next = Array.make m 0;
    cur_slots = Array.make (m * slots) (-1);
    next_slots = Array.make (m * slots) (-1);
    ncur = 0;
    nnext = 0;
    seen = Array.make m (-1);
    gen = 0;
    work = Array.make (Int.max slots 2) (-1);
    best = Array.make (Int.max slots 2) (-1);
    word_pos = -1;
    before = false;
    after = false;
  }

let holds run (v : buf) first stop at = function
  | Start_text -> at = first
  | End_text -> at = stop
  | Start_line -> at = first || A.unsafe_get v (at - 1) = 0x0A
  | End_line -> at = stop || A.unsafe_get v at = 0x0A
  | (Word_boundary | Not_word_boundary) as a ->
      if run.word_pos <> at then begin
        run.word_pos <- at;
        run.before <- at > first && is_word_scalar (decode_before v first at);
        run.after <- at < stop && is_word_scalar (decode v at stop)
      end;
      if a = Word_boundary then run.before <> run.after
      else Bool.equal run.before run.after

let rec add run v first stop pc at =
  if run.seen.(pc) <> run.gen then begin
    run.seen.(pc) <- run.gen;
    match Array.unsafe_get run.regex.prog pc with
    | Split (a, b) ->
        add run v first stop a at;
        add run v first stop b at
    | Save (slot, nx) ->
        if slot < run.nslots then begin
          let old = run.work.(slot) in
          run.work.(slot) <- at;
          add run v first stop nx at;
          run.work.(slot) <- old
        end
        else add run v first stop nx at
    | Assert (a, nx) ->
        if holds run v first stop at a then add run v first stop nx at
    | Sparse _ | Accept ->
        let i = run.nnext in
        run.next.(i) <- pc;
        run.nnext <- i + 1;
        Array.blit run.work 0 run.next_slots (i * run.nslots) run.nslots
  end

let sparse_next a b =
  let lo = ref 0 and hi = ref ((Array.length a / 3) - 1) and hit = ref (-1) in
  while !hit < 0 && !lo <= !hi do
    let mid = (!lo + !hi) / 2 in
    if b < Array.unsafe_get a (3 * mid) then hi := mid - 1
    else if b > Array.unsafe_get a ((3 * mid) + 1) then lo := mid + 1
    else hit := Array.unsafe_get a ((3 * mid) + 2)
  done;
  !hit

let find run (v : buf) ~first ~from ~stop =
  let t = run.regex and nslots = run.nslots in
  let matched = ref false and at = ref from and live = ref true in
  run.nnext <- 0;
  run.gen <- run.gen + 1;
  run.word_pos <- -1;
  Array.fill run.work 0 (Int.max nslots 2) (-1);
  add run v first stop t.entry from;
  while !live do
    let p = !at in
    (* The list just built becomes the current one; the arrays swap, nothing
       is allocated. *)
    let cur = run.cur and slots = run.cur_slots in
    run.cur <- run.next;
    run.cur_slots <- run.next_slots;
    run.next <- cur;
    run.next_slots <- slots;
    run.ncur <- run.nnext;
    run.nnext <- 0;
    run.gen <- run.gen + 1;
    if run.ncur = 0 then live := false
    else begin
      let b = if p < stop then A.unsafe_get v p else -1 in
      let i = ref 0 and n = run.ncur in
      while !i < n do
        let pc = run.cur.(!i) in
        (match Array.unsafe_get t.prog pc with
        | Sparse a ->
            if b >= 0 then begin
              let nx = sparse_next a b in
              if nx >= 0 then begin
                if nslots > 0 then
                  Array.blit run.cur_slots (!i * nslots) run.work 0 nslots;
                add run v first stop nx (p + 1)
              end
            end
        | Accept ->
            if nslots > 0 then
              Array.blit run.cur_slots (!i * nslots) run.best 0 nslots;
            matched := true;
            i := n
        | Split _ | Save _ | Assert _ -> assert false);
        incr i
      done;
      if p >= stop || (!matched && nslots = 0) then live := false
      else at := p + 1
    end
  done;
  !matched

let group run i = run.best.(i)

(* Replacement templates *)

type part = Text of string | Group_of of int

type template = { parts : part list; slots : int }

let literal_template by = { parts = [ Text by ]; slots = 2 }

let template r by =
  let parts = ref [] and max_group = ref 0 in
  let bad at why =
    invalid_arg
      (Printf.sprintf "Expr.Str.replace: %s, at byte %d of %S" why at by)
  in
  let add_group at g =
    if g > r.group_count then
      bad at (Printf.sprintf "group %d does not exist" g);
    if g > !max_group then max_group := g;
    parts := Group_of g :: !parts
  in
  let named at name =
    if name <> "" && is_digit name.[0] then
      match int_of_string_opt name with
      | Some g when String.for_all is_digit name -> add_group at g
      | _ -> bad at "a group number is a run of digits"
    else
      match List.assoc_opt name r.names with
      | Some g -> add_group at g
      | None -> bad at (Printf.sprintf "group %s does not exist" name)
  in
  let n = String.length by in
  let text = Buffer.create n in
  let flush () =
    if Buffer.length text > 0 then begin
      parts := Text (Buffer.contents text) :: !parts;
      Buffer.clear text
    end
  in
  let i = ref 0 in
  let name_while ok at =
    flush ();
    let start = !i in
    while !i < n && ok by.[!i] do
      incr i
    done;
    named at (String.sub by start (!i - start))
  in
  while !i < n do
    let c = by.[!i] in
    if c <> '$' then (
      Buffer.add_char text c;
      incr i)
    else begin
      let at = !i in
      incr i;
      if !i >= n then bad at "$ starts nothing";
      match by.[!i] with
      | '$' ->
          Buffer.add_char text '$';
          incr i
      | '{' -> (
          match String.index_from_opt by (!i + 1) '}' with
          | None -> bad at "unclosed ${"
          | Some close ->
              flush ();
              named at (String.sub by (!i + 1) (close - !i - 1));
              i := close + 1)
      | c when is_digit c -> name_while is_digit at
      | c when is_alpha c -> name_while is_alnum at
      | _ -> bad at "$ starts nothing"
    end
  done;
  flush ();
  { parts = List.rev !parts; slots = 2 * (!max_group + 1) }

let template_slots t = t.slots

let expand_size t (spans : int array) base =
  List.fold_left
    (fun size -> function
      | Text s -> size + String.length s
      | Group_of g ->
          let lo = spans.(base + (2 * g)) in
          if lo < 0 then size else size + spans.(base + (2 * g) + 1) - lo)
    0 t.parts

let expand t (spans : int array) base (v : buf) (b : buf) at =
  let at = ref at in
  List.iter
    (function
      | Text s ->
          for k = 0 to String.length s - 1 do
            A.unsafe_set b (!at + k) (Char.code (String.unsafe_get s k))
          done;
          at := !at + String.length s
      | Group_of g ->
          let lo = spans.(base + (2 * g)) in
          if lo >= 0 then begin
            let hi = spans.(base + (2 * g) + 1) in
            blit v lo b !at (hi - lo);
            at := !at + hi - lo
          end)
    t.parts;
  !at
