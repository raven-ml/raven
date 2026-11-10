(* OCaml source as HTML, highlighted at build time. One pass over the source
   splits it into the tokens of the OCaml manual's "Lexical conventions" and
   wraps each that has a class in a span:

   - hl-c  comments, nested, with string literals inside them skipped as
           OCaml's lexer skips them;
   - hl-s  string literals, quoted strings {id|...|id} and character
           literals;
   - hl-n  integer and float literals, in every radix, with underscores
           and suffixes;
   - hl-k  keywords;
   - hl-m  capitalized names (modules and constructors) and polymorphic
           variants;
   - hl-l  labels, ~name: and ?name:, and punned ~name and ?name;
   - hl-t  type variables, 'a;
   - hl-b  the names a let, and, val, external or method introduces;
   - hl-o  operators.

   Everything else, lowercase names and punctuation, is plain text. *)

let keywords =
  [ "and"; "as"; "assert"; "asr"; "begin"; "class"; "constraint"; "do"; "done"; "downto"; "else";
    "end"; "exception"; "external"; "false"; "for"; "fun"; "function"; "functor"; "if"; "in";
    "include"; "inherit"; "initializer"; "land"; "lazy"; "let"; "lor"; "lsl"; "lsr"; "lxor";
    "match"; "method"; "mod"; "module"; "mutable"; "new"; "nonrec"; "object"; "of"; "open"; "or";
    "private"; "rec"; "sig"; "struct"; "then"; "to"; "true"; "try"; "type"; "val"; "virtual";
    "when"; "while"; "with" ]

(* Keywords after which a lowercase name is a binding. [rec] passes the
   role on: [let rec f]. *)
let binders = [ "let"; "and"; "val"; "external"; "method"; "rec" ]

let is_lower c = (c >= 'a' && c <= 'z') || c = '_'
let is_upper c = c >= 'A' && c <= 'Z'
let is_digit c = c >= '0' && c <= '9'
let is_hex c = is_digit c || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')
let is_ident c = is_lower c || is_upper c || is_digit c || c = '\''
let is_operator c = String.contains "~!?$&*+-/=>@^|%<:." c

let to_html src =
  let n = String.length src in
  let buf = Buffer.create (2 * n) in
  let at i = if i < n then src.[i] else '\000' in
  let text i j = for k = i to j - 1 do Html.add_escaped buf src.[k] done in
  let span cls i j =
    Buffer.add_string buf "<span class=\"hl-";
    Buffer.add_string buf cls;
    Buffer.add_string buf "\">";
    text i j;
    Buffer.add_string buf "</span>"
  in
  let rec while_ p i = if i < n && p src.[i] then while_ p (i + 1) else i in
  (* The end of a string literal whose opening quote precedes [i]. *)
  let rec string i =
    if i >= n then n else match src.[i] with '\\' -> string (i + 2) | '"' -> i + 1 | _ -> string (i + 1)
  in
  (* The end of {id|...|id} opening at [i], if [i] opens one. *)
  let quoted i =
    let j = while_ is_lower (i + 1) in
    if at j <> '|' then None
    else
      let close = "|" ^ String.sub src (i + 1) (j - i - 1) ^ "}" in
      let m = String.length close in
      let rec find k = if k + m > n then n else if String.sub src k m = close then k + m else find (k + 1) in
      Some (find (j + 1))
  in
  (* The end of a comment that opens before [i], at nesting [depth]. *)
  let rec comment i depth =
    if i >= n then n
    else if at i = '(' && at (i + 1) = '*' then comment (i + 2) (depth + 1)
    else if at i = '*' && at (i + 1) = ')' then if depth = 1 then i + 2 else comment (i + 2) (depth - 1)
    else if at i = '"' then comment (string (i + 1)) depth
    else if at i = '{' then match quoted i with Some j -> comment j depth | None -> comment (i + 1) depth
    else comment (i + 1) depth
  in
  (* The end of a character literal opening at [i], if [i] opens one. *)
  let char_literal i =
    match at (i + 1) with
    | '\\' -> (
        match at (i + 2) with
        | '0' .. '9' -> if at (i + 5) = '\'' then Some (i + 6) else None
        | 'x' | 'o' -> if at (i + 5) = '\'' then Some (i + 6) else None
        | _ -> if at (i + 3) = '\'' then Some (i + 4) else None)
    | '\'' | '\000' -> None
    | _ -> if at (i + 2) = '\'' then Some (i + 3) else None
  in
  (* The end of a number literal starting at [i]. *)
  let number i =
    let radix = if at i = '0' then match at (i + 1) with 'x' | 'X' -> `Hex | 'o' | 'O' | 'b' | 'B' -> `Other | _ -> `Dec else `Dec in
    let digit, exp = match radix with `Hex -> (is_hex, "pP") | `Other -> (is_digit, "") | `Dec -> (is_digit, "eE") in
    let i = match radix with `Dec -> i | _ -> i + 2 in
    let j = while_ (fun c -> digit c || c = '_') i in
    let j = if at j = '.' && exp <> "" then while_ (fun c -> digit c || c = '_') (j + 1) else j in
    let j =
      if exp <> "" && String.contains exp (at j) then
        let k = if at (j + 1) = '+' || at (j + 1) = '-' then j + 2 else j + 1 in
        if is_digit (at k) then while_ (fun c -> is_digit c || c = '_') k else j
      else j
    in
    if String.contains "lLn" (at j) then j + 1 else j
  in
  (* [binder]: the next lowercase name is a binding. *)
  let rec scan i ~binder =
    if i < n then
      let c = src.[i] in
      let token cls j = span cls i j; scan j ~binder:false in
      match c with
      | ' ' | '\t' | '\n' | '\r' -> text i (i + 1); scan (i + 1) ~binder
      | '(' when at (i + 1) = '*' -> token "c" (comment (i + 2) 1)
      | '"' -> token "s" (string (i + 1))
      | '{' -> ( match quoted i with Some j -> token "s" j | None -> text i (i + 1); scan (i + 1) ~binder:false)
      | '\'' -> (
          match char_literal i with
          | Some j -> token "s" j
          | None when is_lower (at (i + 1)) -> token "t" (while_ is_ident (i + 1))
          | None -> text i (i + 1); scan (i + 1) ~binder:false)
      | ('~' | '?') when is_lower (at (i + 1)) ->
          let j = while_ is_ident (i + 1) in
          token "l" (if at j = ':' && at (j + 1) <> ':' && at (j + 1) <> '=' then j + 1 else j)
      | '`' when is_upper (at (i + 1)) -> token "m" (while_ is_ident (i + 1))
      | c when is_digit c -> token "n" (number i)
      | c when is_upper c -> token "m" (while_ is_ident i)
      | c when is_lower c ->
          let j = while_ is_ident i in
          let word = String.sub src i (j - i) in
          if List.mem word keywords then (span "k" i j; scan j ~binder:(List.mem word binders))
          else if binder && word <> "_" then token "b" j
          else (text i j; scan j ~binder:false)
      | c when is_operator c && not (c = '.' && not (is_operator (at (i + 1)))) ->
          token "o" (while_ is_operator i)
      | _ -> text i (i + 1); scan (i + 1) ~binder:false
  in
  scan 0 ~binder:false;
  Buffer.contents buf
