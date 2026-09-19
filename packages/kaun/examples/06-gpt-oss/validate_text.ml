(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Validation of the text side of the gpt-oss example against the reference
   implementations.

   [fixtures/harmony.json] is recorded by [reference_text.py] from the
   openai-harmony package and from HuggingFace tokenizers reading the
   checkpoint's [tokenizer.json]. Three things are checked: that brot encodes a
   set of strings to the recorded ids and decodes them back, special tokens
   included; that [Harmony.render] gives the recorded ids of a set of
   conversations; and that a [Harmony.parser] fed the recorded ids of a set of
   completions, one at a time, gives the recorded messages and stops where the
   reference does.

   Usage: validate_text.exe [--fixtures DIR] [--tokenizer FILE]. Without the
   tokenizer file it comes from the fixture's repository (27 MB, cached). Not
   part of the test suite: it needs the download. *)

let json_of_file path =
  let text = In_channel.with_open_bin path In_channel.input_all in
  match Jsont_bytesrw.decode_string Jsont.json text with
  | Ok j -> j
  | Error e -> failwith (path ^ ": " ^ e)

let mem name = function
  | Jsont.Object (mems, _) -> (
      match List.find_opt (fun ((n, _), _) -> n = name) mems with
      | Some (_, v) -> v
      | None -> failwith ("fixture: missing " ^ name))
  | _ -> failwith "fixture: not an object"

let list = function
  | Jsont.Array (l, _) -> l
  | _ -> failwith "fixture: not an array"

let string = function
  | Jsont.String (s, _) -> s
  | _ -> failwith "fixture: not a string"

let string_option = function Jsont.Null _ -> None | j -> Some (string j)

let bool = function
  | Jsont.Bool (b, _) -> b
  | _ -> failwith "fixture: not a boolean"

let ids j =
  let id = function
    | Jsont.Number (f, _) -> int_of_float f
    | _ -> failwith "fixture: not a number"
  in
  Array.of_list (List.map id (list j))

let failures = ref 0
let checks = ref 0

let check name ok =
  incr checks;
  if not ok then incr failures;
  Printf.printf "%s %s\n" (if ok then "ok  " else "FAIL") name

let check_strings tokenizer fixture =
  List.iter
    (fun case ->
      let text = string (mem "text" case) and expected = ids (mem "ids" case) in
      let name = Printf.sprintf "%S" text in
      let name =
        if String.length name <= 60 then name else String.sub name 0 57 ^ "..."
      in
      check ("encode " ^ name)
        (Brot.encode_ids tokenizer ~add_special_tokens:false text = expected);
      check ("decode " ^ name) (Brot.decode tokenizer expected = text))
    (list (mem "strings" fixture))

let effort = function
  | "low" -> Harmony.Low
  | "medium" -> Harmony.Medium
  | "high" -> Harmony.High
  | s -> failwith ("fixture: effort " ^ s)

let channel = function
  | "analysis" -> Harmony.Analysis
  | "commentary" -> Harmony.Commentary
  | "final" -> Harmony.Final
  | s -> failwith ("fixture: channel " ^ s)

let turn j =
  let content = string (mem "content" j) in
  match string (mem "role" j) with
  | "user" -> Harmony.User content
  | "assistant" ->
      Harmony.Assistant (channel (string (mem "channel" j)), content)
  | s -> failwith ("fixture: role " ^ s)

let check_conversations harmony fixture =
  List.iter
    (fun case ->
      let rendered =
        Harmony.render harmony
          ?date:(string_option (mem "date" case))
          ?instructions:(string_option (mem "instructions" case))
          ~effort:(effort (string (mem "effort" case)))
          (List.map turn (list (mem "messages" case)))
      in
      check
        ("render: " ^ string (mem "name" case))
        (rendered = ids (mem "ids" case)))
    (list (mem "conversations" fixture))

(* The texts a parser gives, joined per message: a new message starts where the
   channel changes. Every text must be valid UTF-8 on its own, since it is
   printed as it comes. *)
let parse harmony ids =
  let add messages (channel, text) =
    match messages with
    | (c, t) :: rest when c = channel -> (c, t ^ text) :: rest
    | _ -> (channel, text) :: messages
  in
  let parser, messages, valid =
    Array.fold_left
      (fun (parser, messages, valid) id ->
        match Harmony.feed parser id with
        | parser, None -> (parser, messages, valid)
        | parser, Some ((_, text) as piece) ->
            (parser, add messages piece, valid && String.is_valid_utf_8 text))
      (Harmony.parser harmony, [], true)
      ids
  in
  (List.rev messages, Harmony.stopped parser, valid)

let check_completions harmony fixture =
  List.iter
    (fun case ->
      let expected =
        List.map
          (fun m ->
            (channel (string (mem "channel" m)), string (mem "content" m)))
          (list (mem "messages" case))
      in
      let messages, stopped, valid = parse harmony (ids (mem "ids" case)) in
      let name = "parse: " ^ string (mem "name" case) in
      check (name ^ ", messages") (messages = expected);
      check (name ^ ", stop") (stopped = bool (mem "stopped" case));
      check (name ^ ", every text valid UTF-8") valid)
    (list (mem "completions" fixture))

let () =
  let fixtures = ref "fixtures" and tokenizer = ref "" in
  Arg.parse
    [
      ("--fixtures", Arg.Set_string fixtures, "Directory of reference values");
      ("--tokenizer", Arg.Set_string tokenizer, "A gpt-oss tokenizer.json");
    ]
    (fun a -> raise (Arg.Bad ("unexpected argument " ^ a)))
    "validate_text.exe [--fixtures DIR] [--tokenizer FILE]";
  let fixture = json_of_file (Filename.concat !fixtures "harmony.json") in
  let source = mem "tokenizer" fixture in
  let path =
    if !tokenizer <> "" then !tokenizer
    else
      Kaun_hf.download_file ~file:"tokenizer.json" (string (mem "repo" source))
  in
  Printf.printf "tokenizer %s, recorded from sha256 %s\n" path
    (string (mem "sha256" source));
  let brot =
    match Brot.from_file path with
    | Ok t -> t
    | Error e -> failwith (path ^ ": " ^ e)
  in
  check_strings brot fixture;
  let harmony = Harmony.of_file path in
  check_conversations harmony fixture;
  check_completions harmony fixture;
  Printf.printf "%d checks, %d failures\n" !checks !failures;
  if !failures > 0 then exit 1
