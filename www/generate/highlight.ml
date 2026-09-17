(* Build-time syntax highlighting for code blocks. Each scanner walks its
   source once and emits HTML with [hl-*] spans styled by site/styles.css. *)

let add_char buf = function
  | '&' -> Buffer.add_string buf "&amp;"
  | '<' -> Buffer.add_string buf "&lt;"
  | '>' -> Buffer.add_string buf "&gt;"
  | '"' -> Buffer.add_string buf "&quot;"
  | c -> Buffer.add_char buf c

let add_string buf s = String.iter (add_char buf) s

let span buf cls s =
  Buffer.add_string buf "<span class=\"hl-";
  Buffer.add_string buf cls;
  Buffer.add_string buf "\">";
  add_string buf s;
  Buffer.add_string buf "</span>"

let is_space c = c = ' ' || c = '\t' || c = '\n' || c = '\r'
let is_digit c = c >= '0' && c <= '9'
let is_lower c = (c >= 'a' && c <= 'z') || c = '_'
let is_upper c = c >= 'A' && c <= 'Z'
let is_ident_start c = is_lower c || is_upper c
let is_ident c = is_ident_start c || is_digit c || c = '\''

let member words =
  let tbl = Hashtbl.create 64 in
  List.iter (fun w -> Hashtbl.replace tbl w ()) words;
  Hashtbl.mem tbl

(* OCaml *)

let ocaml_keyword =
  member
    [ "and"; "as"; "assert"; "begin"; "class"; "constraint"; "do"; "done";
      "downto"; "else"; "end"; "exception"; "external"; "false"; "for"; "fun";
      "function"; "functor"; "if"; "in"; "include"; "inherit"; "initializer";
      "lazy"; "let"; "match"; "method"; "module"; "mutable"; "new";
      "nonrec"; "object"; "of"; "open"; "private"; "rec"; "sig"; "struct";
      "then"; "to"; "true"; "try"; "type"; "val"; "virtual"; "when"; "while";
      "with"; "mod"; "land"; "lor"; "lxor"; "lsl"; "lsr"; "asr" ]

(* Keywords after which the next lowercase identifier names a binding. *)
let ocaml_binder = member [ "let"; "rec"; "and"; "val"; "external"; "method" ]

let scan_ocaml buf src =
  let n = String.length src in
  let at i = if i < n then src.[i] else '\000' in
  let sub i j = String.sub src i (j - i) in
  let rec comment i depth =
    if i >= n then n
    else if at i = '(' && at (i + 1) = '*' then comment (i + 2) (depth + 1)
    else if at i = '*' && at (i + 1) = ')' then
      if depth = 1 then i + 2 else comment (i + 2) (depth - 1)
    else comment (i + 1) depth
  in
  let rec string i =
    if i >= n then n
    else
      match src.[i] with
      | '\\' -> string (i + 2)
      | '"' -> i + 1
      | _ -> string (i + 1)
  in
  (* [{id|...|id}], scanned from the opening brace. *)
  let quoted_string i =
    let j = ref (i + 1) in
    while !j < n && is_lower src.[!j] do
      incr j
    done;
    if at !j <> '|' then None
    else
      let close = "|" ^ sub (i + 1) !j ^ "}" in
      match Site.find_sub ~start:(!j + 1) src close with
      | Some k -> Some (k + String.length close)
      | None -> Some n
  in
  (* ['x'], ['\n'], ['\123'] or ['\xhh'], scanned from the opening quote. *)
  let char_literal i =
    if at (i + 1) <> '\\' then
      if at (i + 2) = '\'' && at (i + 1) <> '\'' then Some (i + 3) else None
    else
      match String.index_from_opt src (i + 2) '\'' with
      | Some k when k - i <= 5 -> Some (k + 1)
      | _ -> None
  in
  let rec number j =
    if j < n && (is_ident src.[j] || src.[j] = '.') then number (j + 1)
    else if
      j < n
      && (src.[j] = '+' || src.[j] = '-')
      && (src.[j - 1] = 'e' || src.[j - 1] = 'E')
    then number (j + 1)
    else j
  in
  let rec ident j = if j < n && is_ident src.[j] then ident (j + 1) else j in
  (* [binder] is true where a lowercase identifier names a new binding. *)
  let rec scan i binder =
    if i < n then
      let c = src.[i] in
      let token cls j =
        span buf cls (sub i j);
        scan j false
      in
      let plain () =
        add_char buf c;
        scan (i + 1) (binder && is_space c)
      in
      match c with
      | '(' when at (i + 1) = '*' -> token "comment" (comment (i + 2) 1)
      | '"' -> token "string" (string (i + 1))
      | '{' -> (
          match quoted_string i with
          | Some j -> token "string" j
          | None -> plain ())
      | '\'' -> (
          match char_literal i with
          | Some j -> token "string" j
          | None ->
              if is_ident_start (at (i + 1)) then token "type" (ident (i + 1))
              else plain ())
      | c when is_digit c -> token "number" (number (i + 1))
      | c when is_ident_start c ->
          let j = ident (i + 1) in
          let word = sub i j in
          if ocaml_keyword word then span buf "keyword" word
          else if is_upper c then span buf "type" word
          else if binder then span buf "function" word
          else add_string buf word;
          scan j (ocaml_binder word)
      | _ -> plain ()
  in
  scan 0 false

(* Dune *)

let scan_dune buf src =
  let n = String.length src in
  let sub i j = String.sub src i (j - i) in
  let is_atom c = not (is_space c || String.contains "()\";" c) in
  let rec string i =
    if i >= n then n
    else
      match src.[i] with
      | '\\' -> string (i + 2)
      | '"' -> i + 1
      | _ -> string (i + 1)
  in
  let rec atom j = if j < n && is_atom src.[j] then atom (j + 1) else j in
  let rec line j = if j < n && src.[j] <> '\n' then line (j + 1) else j in
  (* [head] is true for the first atom of a list, which names a stanza or a
     field. *)
  let rec scan i head =
    if i < n then
      let c = src.[i] in
      let token cls j =
        span buf cls (sub i j);
        scan j false
      in
      match c with
      | ';' ->
          let j = line i in
          span buf "comment" (sub i j);
          scan j head
      | '"' -> token "string" (string (i + 1))
      | '(' ->
          Buffer.add_char buf c;
          scan (i + 1) true
      | ')' ->
          Buffer.add_char buf c;
          scan (i + 1) false
      | c when is_space c ->
          Buffer.add_char buf c;
          scan (i + 1) head
      | _ ->
          let j = atom (i + 1) in
          if head then token "keyword" j
          else if c = '%' || c = ':' || is_digit c then token "number" j
          else (
            add_string buf (sub i j);
            scan j false)
  in
  scan 0 false

(* Shell *)

let shell_keyword =
  member
    [ "if"; "then"; "elif"; "else"; "fi"; "for"; "in"; "do"; "done"; "while";
      "until"; "case"; "esac"; "function"; "select"; "time"; "export";
      "local"; "return"; "exit" ]

(* Keywords after which the next word is again a command. *)
let shell_opener =
  member [ "if"; "then"; "elif"; "else"; "do"; "while"; "until"; "time" ]

let scan_shell buf src =
  let n = String.length src in
  let at i = if i < n then src.[i] else '\000' in
  let sub i j = String.sub src i (j - i) in
  let rec until i stop =
    if i >= n then n
    else if src.[i] = '\\' && stop = '"' then until (i + 2) stop
    else if src.[i] = stop then i + 1
    else until (i + 1) stop
  in
  let rec line j = if j < n && src.[j] <> '\n' then line (j + 1) else j in
  let is_word c = not (is_space c || String.contains "\"'#$;|&()<>" c) in
  let rec word j = if j < n && is_word src.[j] then word (j + 1) else j in
  (* [cmd] is true where the next word names a command. *)
  let rec scan i cmd =
    if i < n then
      let c = src.[i] in
      let token cls j =
        span buf cls (sub i j);
        scan j false
      in
      match c with
      | '#' when i = 0 || is_space (at (i - 1)) ->
          let j = line i in
          span buf "comment" (sub i j);
          scan j cmd
      | '"' | '\'' -> token "string" (until (i + 1) c)
      | '$' when at (i + 1) = '{' -> token "number" (until (i + 2) '}')
      | '$' when is_ident_start (at (i + 1)) -> token "number" (word (i + 1))
      | '\n' | ';' | '|' | '&' | '(' ->
          Buffer.add_char buf c;
          scan (i + 1) true
      | c when is_word c ->
          let j = word (i + 1) in
          let w = sub i j in
          if shell_keyword w then span buf "keyword" w
          else if cmd then span buf "function" w
          else add_string buf w;
          scan j (shell_opener w)
      | _ ->
          add_char buf c;
          scan (i + 1) (cmd && is_space c)
  in
  scan 0 true

let block scan src =
  let buf = Buffer.create (2 * String.length src) in
  Buffer.add_string buf "<pre><code>";
  scan buf src;
  Buffer.add_string buf "</code></pre>";
  Buffer.contents buf

let ocaml src = block scan_ocaml src

let to_html ~lang src =
  match lang with
  | "ocaml" | "ml" | "mli" -> Some (ocaml src)
  | "dune" -> Some (block scan_dune src)
  | "sh" | "bash" | "shell" -> Some (block scan_shell src)
  | _ -> None
