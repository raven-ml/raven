(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t =
  | Object of string * t list
  | Text of string
  | Number of float
  | Word of string

exception Syntax of int

(* A recursive descent over [s] from [!pos]. [close] is the right delimiter the
   text's first left delimiter fixed, so a text mixing brackets and parentheses
   stops at the first delimiter of the other form. *)

let is_space c = c = ' ' || c = '\t' || c = '\n' || c = '\r'
let is_digit c = '0' <= c && c <= '9'
let is_letter c = ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c = '_'

let parse s =
  let n = String.length s in
  let pos = ref 0 in
  let close = ref None in
  let peek () = if !pos < n then Some s.[!pos] else None in
  let fail () = raise (Syntax !pos) in
  let skip () =
    while !pos < n && is_space s.[!pos] do
      incr pos
    done
  in
  let expect c = if peek () = Some c then incr pos else fail () in
  let span ok =
    let start = !pos in
    while !pos < n && ok s.[!pos] do
      incr pos
    done;
    String.sub s start (!pos - start)
  in
  let word () =
    if not (Option.fold ~none:false ~some:is_letter (peek ())) then fail ();
    span (fun c -> is_letter c || is_digit c)
  in
  let text () =
    expect '"';
    let b = Buffer.create 32 in
    let rec go () =
      if !pos >= n then fail ();
      let c = s.[!pos] in
      incr pos;
      if c <> '"' then (
        Buffer.add_char b c;
        go ())
      else if peek () = Some '"' then (
        Buffer.add_char b '"';
        incr pos;
        go ())
    in
    go ();
    Text (Buffer.contents b)
  in
  (* §6.3.2: [sign] (digits [. [digits]] | . digits) [E [sign] digits]. *)
  let sign () =
    match peek () with
    | Some (('+' | '-') as c) ->
        incr pos;
        String.make 1 c
    | _ -> ""
  in
  let number () =
    let sgn = sign () in
    let whole = span is_digit in
    let frac =
      if peek () = Some '.' then (
        incr pos;
        span is_digit)
      else ""
    in
    if whole = "" && frac = "" then fail ();
    let exp =
      if peek () = Some 'e' || peek () = Some 'E' then begin
        incr pos;
        let esgn = sign () in
        let digits = span is_digit in
        if digits = "" then fail ();
        "e" ^ esgn ^ digits
      end
      else ""
    in
    (* Written out in full, since OCaml's float syntax wants digits before the
       period. [float_of_string] is C's strtod, which glibc, Apple's libc and
       musl round correctly, so they agree. *)
    let digits d = if d = "" then "0" else d in
    Number (float_of_string (sgn ^ digits whole ^ "." ^ digits frac ^ exp))
  in
  let rec value () =
    skip ();
    match peek () with
    | Some '"' -> text ()
    | Some c when is_digit c || c = '+' || c = '-' || c = '.' -> number ()
    | _ -> (
        let key = word () in
        skip ();
        match peek () with Some ('[' | '(') -> obj key | _ -> Word key)
  and obj key =
    let right = if s.[!pos] = '[' then ']' else ')' in
    (match !close with
    | None -> close := Some right
    | Some r -> if r <> right then fail ());
    incr pos;
    let rec items acc =
      let acc = value () :: acc in
      skip ();
      match peek () with
      | Some ',' ->
          incr pos;
          items acc
      | Some c when c = right ->
          incr pos;
          List.rev acc
      | _ -> fail ()
    in
    Object (key, items [])
  in
  match
    skip ();
    let key = word () in
    skip ();
    if not (peek () = Some '[' || peek () = Some '(') then fail ();
    let t = obj key in
    skip ();
    if !pos < n then fail ();
    t
  with
  | t -> Ok t
  | exception Syntax at ->
      Error (Printf.sprintf "WKT syntax error at byte %d" at)

let same a b = String.lowercase_ascii a = String.lowercase_ascii b

let rec equal a b =
  match (a, b) with
  | Object (k, xs), Object (l, ys) ->
      same k l && List.length xs = List.length ys && List.for_all2 equal xs ys
  | Text x, Text y | Word x, Word y -> same x y
  | Number x, Number y -> x = y
  | (Object _ | Text _ | Number _ | Word _), _ -> false
