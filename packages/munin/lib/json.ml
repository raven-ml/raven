(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t =
  | Null
  | Bool of bool
  | Number of string
  | String of string
  | Array of t list
  | Object of (string * t) list

let utf8 = String.is_valid_utf_8

(* Decoding *)

let max_depth = 64

exception Bad of string

let fail i fmt =
  Printf.ksprintf (fun m -> raise (Bad (Printf.sprintf "byte %d: %s" i m))) fmt

let add_utf8 b u =
  if u < 0x80 then Buffer.add_char b (Char.chr u)
  else if u < 0x800 then (
    Buffer.add_char b (Char.chr (0xC0 lor (u lsr 6)));
    Buffer.add_char b (Char.chr (0x80 lor (u land 0x3F))))
  else if u < 0x10000 then (
    Buffer.add_char b (Char.chr (0xE0 lor (u lsr 12)));
    Buffer.add_char b (Char.chr (0x80 lor ((u lsr 6) land 0x3F)));
    Buffer.add_char b (Char.chr (0x80 lor (u land 0x3F))))
  else (
    Buffer.add_char b (Char.chr (0xF0 lor (u lsr 18)));
    Buffer.add_char b (Char.chr (0x80 lor ((u lsr 12) land 0x3F)));
    Buffer.add_char b (Char.chr (0x80 lor ((u lsr 6) land 0x3F)));
    Buffer.add_char b (Char.chr (0x80 lor (u land 0x3F))))

let of_string s =
  let n = String.length s in
  let pos = ref 0 in
  let peek () = if !pos < n then s.[!pos] else '\000' in
  let rec skip () =
    if !pos < n then
      match s.[!pos] with
      | ' ' | '\t' | '\n' | '\r' ->
          incr pos;
          skip ()
      | _ -> ()
  in
  let expect c =
    if peek () <> c || !pos >= n then fail !pos "expected %C" c;
    incr pos
  in
  let literal word v =
    let l = String.length word in
    if !pos + l <= n && String.sub s !pos l = word then (
      pos := !pos + l;
      v)
    else fail !pos "invalid literal"
  in
  let hex4 () =
    if !pos + 4 > n then fail !pos "truncated escape";
    let v = ref 0 in
    for k = 0 to 3 do
      let d =
        match s.[!pos + k] with
        | '0' .. '9' as c -> Char.code c - 48
        | 'a' .. 'f' as c -> Char.code c - 87
        | 'A' .. 'F' as c -> Char.code c - 55
        | _ -> fail (!pos + k) "invalid hex digit"
      in
      v := (!v lsl 4) lor d
    done;
    pos := !pos + 4;
    !v
  in
  let string () =
    expect '"';
    let b = Buffer.create 16 in
    let rec loop () =
      if !pos >= n then fail !pos "unterminated string";
      let c = s.[!pos] in
      if c = '"' then incr pos
      else if c = '\\' then begin
        incr pos;
        if !pos >= n then fail !pos "unterminated string";
        let e = s.[!pos] in
        incr pos;
        (match e with
        | '"' -> Buffer.add_char b '"'
        | '\\' -> Buffer.add_char b '\\'
        | '/' -> Buffer.add_char b '/'
        | 'b' -> Buffer.add_char b '\b'
        | 'f' -> Buffer.add_char b '\012'
        | 'n' -> Buffer.add_char b '\n'
        | 'r' -> Buffer.add_char b '\r'
        | 't' -> Buffer.add_char b '\t'
        | 'u' ->
            let u = hex4 () in
            if u >= 0xD800 && u <= 0xDBFF then begin
              if !pos + 6 > n || s.[!pos] <> '\\' || s.[!pos + 1] <> 'u' then
                fail !pos "lone surrogate";
              pos := !pos + 2;
              let lo = hex4 () in
              if lo < 0xDC00 || lo > 0xDFFF then fail !pos "lone surrogate";
              add_utf8 b (0x10000 + ((u - 0xD800) lsl 10) + (lo - 0xDC00))
            end
            else if u >= 0xDC00 && u <= 0xDFFF then fail !pos "lone surrogate"
            else add_utf8 b u
        | _ -> fail !pos "invalid escape");
        loop ()
      end
      else if Char.code c < 0x20 then fail !pos "control character in string"
      else begin
        Buffer.add_char b c;
        incr pos;
        loop ()
      end
    in
    loop ();
    Buffer.contents b
  in
  let number () =
    let start = !pos in
    let stop = Munin_float_text.Float_read.number_end s start n in
    if stop < 0 then fail start "invalid number";
    pos := stop;
    Number (String.sub s start (stop - start))
  in
  (* No file of the format nests deeper than a few levels; a line nested past
     [max_depth] is damage, refused before it can exhaust the stack. *)
  let rec value depth =
    if depth > max_depth then fail !pos "nested deeper than any record";
    skip ();
    match peek () with
    | '{' ->
        incr pos;
        skip ();
        if peek () = '}' then (
          incr pos;
          Object [])
        else
          let rec members acc =
            skip ();
            let at = !pos in
            let name = string () in
            if List.mem_assoc name acc then fail at "member %S repeats" name;
            skip ();
            expect ':';
            let v = value (depth + 1) in
            skip ();
            let acc = (name, v) :: acc in
            match peek () with
            | ',' ->
                incr pos;
                members acc
            | '}' ->
                incr pos;
                Object (List.rev acc)
            | _ -> fail !pos "expected ',' or '}'"
          in
          members []
    | '[' ->
        incr pos;
        skip ();
        if peek () = ']' then (
          incr pos;
          Array [])
        else
          let rec items acc =
            let v = value (depth + 1) in
            skip ();
            match peek () with
            | ',' ->
                incr pos;
                items (v :: acc)
            | ']' ->
                incr pos;
                Array (List.rev (v :: acc))
            | _ -> fail !pos "expected ',' or ']'"
          in
          items []
    | '"' -> String (string ())
    | 't' -> literal "true" (Bool true)
    | 'f' -> literal "false" (Bool false)
    | 'n' -> literal "null" Null
    | '-' | '0' .. '9' -> number ()
    | _ -> fail !pos "expected a value"
  in
  if not (utf8 s) then Error "invalid UTF-8"
  else
    match
      let v = value 0 in
      skip ();
      if !pos <> n then fail !pos "trailing bytes";
      v
    with
    | v -> Ok v
    | exception Bad why -> Error why

(* Encoding *)

let add_string b s =
  Buffer.add_char b '"';
  String.iter
    (function
      | '"' -> Buffer.add_string b "\\\""
      | '\\' -> Buffer.add_string b "\\\\"
      | '\n' -> Buffer.add_string b "\\n"
      | '\r' -> Buffer.add_string b "\\r"
      | '\t' -> Buffer.add_string b "\\t"
      | c when Char.code c < 0x20 || c = '\x7F' ->
          Printf.bprintf b "\\u%04x" (Char.code c)
      | c -> Buffer.add_char b c)
    s;
  Buffer.add_char b '"'

(* The largest magnitude JSON carries exactly in every reader (RFC 8259 §6). *)
let exact_int_bound = 1 lsl 53

let add_float b x =
  if Float.is_nan x then Buffer.add_string b {|{"float":"NaN"}|}
  else if x = Float.infinity then Buffer.add_string b {|{"float":"Infinity"}|}
  else if x = Float.neg_infinity then
    Buffer.add_string b {|{"float":"-Infinity"}|}
  else Buffer.add_string b (Munin_float_text.Float_print.exact_float x)

let add_int b n =
  if n > exact_int_bound || n < -exact_int_bound then
    Printf.bprintf b {|{"int":"%d"}|} n
  else Buffer.add_string b (string_of_int n)

let add_value b = function
  | `Bool v -> Buffer.add_string b (if v then "true" else "false")
  | `Int n -> add_int b n
  | `Float x -> add_float b x
  | `String s -> add_string b s

let add_object b members =
  Buffer.add_char b '{';
  List.iteri
    (fun i (name, add) ->
      if i > 0 then Buffer.add_char b ',';
      add_string b name;
      Buffer.add_char b ':';
      add b)
    members;
  Buffer.add_char b '}'

(* Reading *)

let integral text =
  not (String.exists (fun c -> c = '.' || c = 'e' || c = 'E') text)

let float = function
  | Number t -> Munin_float_text.Float_read.of_string t
  | Object [ ("float", String "NaN") ] -> Some Float.nan
  | Object [ ("float", String "Infinity") ] -> Some Float.infinity
  | Object [ ("float", String "-Infinity") ] -> Some Float.neg_infinity
  | _ -> None

let int = function
  | Number t when integral t -> int_of_string_opt t
  | Object [ ("int", String t) ] -> (
      match int_of_string_opt t with
      | Some n
        when String.for_all (function '-' | '0' .. '9' -> true | _ -> false) t
        ->
          Some n
      | _ -> None)
  | _ -> None

let value j =
  match j with
  | Bool v -> Some (`Bool v)
  | String s -> Some (`String s)
  | Number t when integral t -> Option.map (fun n -> `Int n) (int j)
  | Number _ | Object [ ("float", _) ] ->
      Option.map (fun x -> `Float x) (float j)
  | Object [ ("int", _) ] -> Option.map (fun n -> `Int n) (int j)
  | _ -> None

let values = function
  | Object members ->
      let rec go acc = function
        | [] -> Some (List.rev acc)
        | (k, j) :: rest -> (
            match value j with
            | Some v -> go ((k, v) :: acc) rest
            | None -> None)
      in
      go [] members
  | _ -> None

let mem name = function
  | Object members -> List.assoc_opt name members
  | _ -> None

let string = function String s -> Some s | _ -> None
let field name f j = Option.bind (mem name j) f

let opt name f j =
  match mem name j with
  | None -> Some None
  | Some v -> Option.map Option.some (f v)

let all f l =
  List.fold_right
    (fun x acc ->
      Option.bind acc (fun acc -> Option.map (fun y -> y :: acc) (f x)))
    l (Some [])

let list f = function Array l -> all f l | _ -> None

let assoc f = function
  | Object members ->
      all (fun (k, v) -> Option.map (fun x -> (k, x)) (f v)) members
  | _ -> None
