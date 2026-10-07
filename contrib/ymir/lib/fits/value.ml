(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Value fields: the grammars of FITS 4.0 §4.2 for reading, and the fixed and
   free formats of §4.2.2-4 for printing. *)

let strf = Printf.sprintf
let is_printable c = c >= ' ' && c <= '~'
let is_digit c = c >= '0' && c <= '9'
let is_space c = c = ' '

(* Fields *)

(* A value field split as written: the value's text, trimmed, and the comment
   after its slash. A string's text runs from its opening quote to its closing
   one. [token] is [""] for an undefined value. *)
type field = { token : string; comment : string option }

let trim_right s =
  let n = ref (String.length s) in
  while !n > 0 && is_space s.[!n - 1] do
    decr n
  done;
  String.sub s 0 !n

let skip_spaces s i =
  let i = ref i in
  while !i < String.length s && is_space s.[!i] do
    incr i
  done;
  !i

(* The end of the quoted string starting at [i]: the index after its closing
   quote, or [None] if it does not close. A doubled quote is a quote. *)
let string_end s i =
  let n = String.length s in
  let rec go j =
    if j >= n then None
    else if s.[j] <> '\'' then go (j + 1)
    else if j + 1 < n && s.[j + 1] = '\'' then go (j + 2)
    else Some (j + 1)
  in
  go (i + 1)

let comment_of s i =
  let i = skip_spaces s i in
  if i < String.length s && s.[i] = '/' then
    let c = String.sub s (i + 1) (String.length s - i - 1) in
    let c =
      if String.length c > 0 && c.[0] = ' ' then
        String.sub c 1 (String.length c - 1)
      else c
    in
    Some (trim_right c)
  else None

(* [split s] reads the value field [s] (the bytes after the value indicator). A
   quote that does not close makes the whole field the token, which the string
   grammar then refuses. *)
let split s =
  let i = skip_spaces s 0 in
  if i < String.length s && s.[i] = '\'' then
    match string_end s i with
    | Some j -> { token = String.sub s i (j - i); comment = comment_of s j }
    | None ->
        {
          token = trim_right (String.sub s i (String.length s - i));
          comment = None;
        }
  else
    match String.index_from_opt s i '/' with
    | Some j ->
        {
          token = String.trim (String.sub s i (j - i));
          comment = comment_of s j;
        }
    | None ->
        {
          token = String.trim (String.sub s i (String.length s - i));
          comment = None;
        }

(* Strings *)

(* The text of the string token [t], quotes removed and doubled quotes made
   single, with its trailing spaces kept. *)
let unquote t =
  let n = String.length t in
  if n < 2 || t.[0] <> '\'' || t.[n - 1] <> '\'' then None
  else
    let b = Buffer.create n in
    let rec go i =
      if i >= n - 1 then Some (Buffer.contents b)
      else if t.[i] = '\'' then
        if i + 1 < n - 1 && t.[i + 1] = '\'' then (
          Buffer.add_char b '\'';
          go (i + 2))
        else None
      else (
        Buffer.add_char b t.[i];
        go (i + 1))
    in
    go 1

(* §4.2.1.1: trailing spaces are not significant, but a string of spaces is
   one space. *)
let string_value raw =
  let s = trim_right raw in
  if s = "" && raw <> "" then " " else s

let quote s =
  let b = Buffer.create (String.length s + 2) in
  Buffer.add_char b '\'';
  String.iter
    (fun c ->
      if c = '\'' then Buffer.add_string b "''" else Buffer.add_char b c)
    s;
  Buffer.add_char b '\'';
  Buffer.contents b

(* Numbers *)

let int_of_token t =
  let n = String.length t in
  let start = if n > 0 && (t.[0] = '+' || t.[0] = '-') then 1 else 0 in
  if start >= n then None
  else if not (String.for_all is_digit (String.sub t start (n - start))) then
    None
  else int_of_string_opt t

(* §4.2.4: [sign] (digits [. [digits]] | . digits) [exponent], the exponent's
   letter E or D, in lower case as archives write it. Integer text is real
   text. *)
let real_grammar t =
  let n = String.length t in
  let i = ref 0 in
  if !i < n && (t.[!i] = '+' || t.[!i] = '-') then incr i;
  let digits () =
    let s = !i in
    while !i < n && is_digit t.[!i] do
      incr i
    done;
    !i - s
  in
  let whole = digits () in
  let frac =
    if !i < n && t.[!i] = '.' then (
      incr i;
      digits ())
    else 0
  in
  if whole + frac = 0 then false
  else if !i = n then true
  else
    match t.[!i] with
    | 'E' | 'D' | 'e' | 'd' ->
        incr i;
        if !i < n && (t.[!i] = '+' || t.[!i] = '-') then incr i;
        digits () > 0 && !i = n
    | _ -> false

let float_of_token t =
  if not (real_grammar t) then None
  else
    let t = String.map (function 'D' | 'd' -> 'E' | c -> c) t in
    match float_of_string_opt t with
    | Some x when Float.is_finite x -> Some x
    | _ -> None

(* The fewest significant digits that read back to [x], with a point or an
   exponent so the text never reads as an integer. *)
let print_float x =
  let rec shortest p =
    if p >= 17 || Float.equal (float_of_string (strf "%.*E" (p - 1) x)) x then p
    else shortest (p + 1)
  in
  let p = shortest 1 in
  let e =
    if x = 0. then 0 else int_of_float (Float.floor (Float.log10 (Float.abs x)))
  in
  (* Positional text when it is as short as twenty bytes, as 1288.4 or
     0.001; otherwise the exponent form, as 6.02214076E+23. *)
  let fixed = strf "%.*f" (Int.max 0 (p - 1 - e)) x in
  let fixed = if String.contains fixed '.' then fixed else fixed ^ ".0" in
  if String.length fixed <= 20 && Float.equal (float_of_string fixed) x then
    fixed
  else strf "%.*E" (p - 1) x

(* A complex value: two reals in parentheses, separated by a comma. *)
let is_complex t =
  let n = String.length t in
  n >= 2
  && t.[0] = '('
  && t.[n - 1] = ')'
  &&
  match String.split_on_char ',' (String.sub t 1 (n - 2)) with
  | [ re; im ] -> real_grammar (String.trim re) && real_grammar (String.trim im)
  | _ -> false

(* Whether [t] is one FITS value as written in a value field. *)
let is_value t =
  t = "T" || t = "F" || real_grammar t || is_complex t
  || t <> ""
     && t.[0] = '\''
     && string_end t 0 = Some (String.length t)
     && Option.is_some (unquote t)

(* Grammars *)

type _ base =
  | Bool : bool base
  | Int : int base
  | Float : float base
  | String : string base
  | Text : string base

type 'a t = V : 'b base * ('b -> ('a, string) result) * ('a -> 'b) -> 'a t

let bool = V (Bool, Result.ok, Fun.id)
let int = V (Int, Result.ok, Fun.id)
let float = V (Float, Result.ok, Fun.id)
let string = V (String, Result.ok, Fun.id)
let text = V (Text, Result.ok, Fun.id)

let map f g (V (b, read, print)) =
  V (b, (fun x -> Result.bind (read x) f), fun y -> print (g y))

let base_equal : type b. b base -> b -> b -> bool =
 fun b x y ->
  match b with
  | Bool -> Bool.equal x y
  | Int -> Int.equal x y
  | Float -> Float.equal x y
  | String -> String.equal x y
  | Text -> String.equal x y

let base_to_string : type b. b base -> b -> string =
 fun b x ->
  match b with
  | Bool -> if x then "T" else "F"
  | Int -> string_of_int x
  | Float -> print_float x
  | String -> quote x
  | Text -> x

(* [read b token joined] reads a defined token; [joined] is the string a
   string token and its CONTINUE records hold. *)
let read : type b. b base -> string -> string option -> (b, string) result =
 fun b token joined ->
  let typed what =
    if String.for_all is_printable token then None
    else Some (strf "%s holds a byte outside ASCII 32-126" what)
  in
  match b with
  | Text -> Ok token
  | Bool -> (
      match typed "the value" with
      | Some e -> Error e
      | None -> (
          match token with
          | "T" -> Ok true
          | "F" -> Ok false
          | _ -> Error (strf "%s is not a FITS logical (T or F)" token)))
  | Int -> (
      match typed "the value" with
      | Some e -> Error e
      | None -> (
          match int_of_token token with
          | Some i -> Ok i
          | None ->
              if
                real_grammar token
                && not
                     (String.exists
                        (fun c ->
                          c = '.' || c = 'E' || c = 'D' || c = 'e' || c = 'd')
                        token)
              then
                Error
                  (strf
                     "%s is past int's range; Value.text reads the field as \
                      written"
                     token)
              else Error (strf "%s is not a FITS integer" token)))
  | Float -> (
      match typed "the value" with
      | Some e -> Error e
      | None -> (
          match float_of_token token with
          | Some x -> Ok x
          | None ->
              if real_grammar token then
                Error (strf "%s is past float64's range" token)
              else
                Error
                  (strf
                     "%s is not a FITS number; Value.text reads the field as \
                      written"
                     token)))
  | String -> (
      match joined with
      | None -> Error (strf "%s is not a FITS string" token)
      | Some s -> (
          match typed "the string" with Some e -> Error e | None -> Ok s))

let is_string_base : type b. b base -> bool = function
  | String -> true
  | _ -> false
