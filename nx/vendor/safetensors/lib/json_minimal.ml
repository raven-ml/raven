(** Minimal JSON parser/serializer for safetensors Only supports the subset
    needed for safetensors metadata *)

type t =
  [ `Assoc of (string * t) list
  | `String of string
  | `Int of int
  | `List of t list ]

exception Parse_error of string

(** Serialization *)

let rec to_string = function
  | `String s -> Printf.sprintf "\"%s\"" (String.escaped s)
  | `Int i -> string_of_int i
  | `List l -> "[" ^ String.concat ", " (List.map to_string l) ^ "]"
  | `Assoc kv ->
      let pairs =
        List.map
          (fun (k, v) ->
            Printf.sprintf "\"%s\": %s" (String.escaped k) (to_string v))
          kv
      in
      "{" ^ String.concat ", " pairs ^ "}"

(** Parsing *)

type parser = { mutable input : string; mutable pos : int }

let peek p =
  if p.pos < String.length p.input then Some p.input.[p.pos] else None

let advance p = p.pos <- p.pos + 1

let skip_whitespace p =
  while
    p.pos < String.length p.input
    &&
    match p.input.[p.pos] with ' ' | '\t' | '\n' | '\r' -> true | _ -> false
  do
    advance p
  done

let expect p c =
  skip_whitespace p;
  match peek p with
  | Some ch when ch = c -> advance p
  | Some ch ->
      raise (Parse_error (Printf.sprintf "Expected '%c' but got '%c'" c ch))
  | None -> raise (Parse_error (Printf.sprintf "Expected '%c' but got EOF" c))

let parse_string p =
  expect p '"';
  let buf = Buffer.create 16 in
  let rec loop () =
    match peek p with
    | None -> raise (Parse_error "Unterminated string")
    | Some '"' ->
        advance p;
        Buffer.contents buf
    | Some '\\' -> (
        advance p;
        match peek p with
        | None -> raise (Parse_error "Unterminated string escape")
        | Some 'n' ->
            Buffer.add_char buf '\n';
            advance p;
            loop ()
        | Some 'r' ->
            Buffer.add_char buf '\r';
            advance p;
            loop ()
        | Some 't' ->
            Buffer.add_char buf '\t';
            advance p;
            loop ()
        | Some '"' ->
            Buffer.add_char buf '"';
            advance p;
            loop ()
        | Some '\\' ->
            Buffer.add_char buf '\\';
            advance p;
            loop ()
        | Some c ->
            Buffer.add_char buf c;
            advance p;
            loop ())
    | Some c ->
        Buffer.add_char buf c;
        advance p;
        loop ()
  in
  loop ()

let parse_int p =
  skip_whitespace p;
  let start = p.pos in
  let negative =
    match peek p with
    | Some '-' ->
        advance p;
        true
    | _ -> false
  in
  if peek p = None then raise (Parse_error "Expected number");
  while
    p.pos < String.length p.input
    && match p.input.[p.pos] with '0' .. '9' -> true | _ -> false
  do
    advance p
  done;
  let num_str = String.sub p.input start (p.pos - start) in
  try int_of_string num_str
  with _ -> raise (Parse_error ("Invalid number: " ^ num_str))

let rec parse_value p =
  skip_whitespace p;
  match peek p with
  | None -> raise (Parse_error "Unexpected EOF")
  | Some '"' -> `String (parse_string p)
  | Some '{' -> parse_object p
  | Some '[' -> parse_list p
  | Some ('-' | '0' .. '9') -> `Int (parse_int p)
  | Some c ->
      raise (Parse_error (Printf.sprintf "Unexpected character: '%c'" c))

and parse_list p =
  expect p '[';
  skip_whitespace p;
  if peek p = Some ']' then (
    advance p;
    `List [])
  else
    let rec loop acc =
      let v = parse_value p in
      skip_whitespace p;
      match peek p with
      | Some ',' ->
          advance p;
          loop (v :: acc)
      | Some ']' ->
          advance p;
          `List (List.rev (v :: acc))
      | _ -> raise (Parse_error "Expected ',' or ']'")
    in
    loop []

and parse_object p =
  expect p '{';
  skip_whitespace p;
  if peek p = Some '}' then (
    advance p;
    `Assoc [])
  else
    let rec loop acc =
      skip_whitespace p;
      let key = parse_string p in
      skip_whitespace p;
      expect p ':';
      let value = parse_value p in
      skip_whitespace p;
      match peek p with
      | Some ',' ->
          advance p;
          loop ((key, value) :: acc)
      | Some '}' ->
          advance p;
          `Assoc (List.rev ((key, value) :: acc))
      | _ -> raise (Parse_error "Expected ',' or '}'")
    in
    loop []

let from_string s =
  let p = { input = s; pos = 0 } in
  try
    let v = parse_value p in
    skip_whitespace p;
    if p.pos < String.length s then
      raise (Parse_error "Unexpected characters after JSON")
    else v
  with Parse_error msg ->
    raise
      (Parse_error
         (Printf.sprintf "JSON parse error at position %d: %s" p.pos msg))

(** Utility functions for safetensors *)

let to_assoc = function
  | `Assoc kv -> kv
  | _ -> raise (Parse_error "Expected object")

let to_string_exn = function
  | `String s -> s
  | _ -> raise (Parse_error "Expected string")

let to_int_exn = function
  | `Int i -> i
  | _ -> raise (Parse_error "Expected integer")

let to_list_exn = function
  | `List l -> l
  | _ -> raise (Parse_error "Expected array")

let member key json =
  match json with
  | `Assoc kv -> List.assoc key kv
  | _ -> raise (Parse_error "Expected object")
