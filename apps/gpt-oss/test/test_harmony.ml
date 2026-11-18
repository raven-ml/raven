(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Harmony against openai-harmony: [Harmony.render] gives the recorded ids of
   each conversation, and a parser fed a completion's ids one at a time gives
   the recorded messages and stops where the reference does.
   [gen/reference_text.py] recorded the fixture. *)

open Windtrap
open Fixture

let fx = load "harmony"

let harmony =
  lazy
    (let source = mem "tokenizer" fx in
     let revision = to_string (mem "revision" source) in
     Harmony.of_file
       (Kaun_hf.download_file ~revision ~file:"tokenizer.json"
          (to_string (mem "repo" source))))

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
  let content = to_string (mem "content" j) in
  match to_string (mem "role" j) with
  | "user" -> Harmony.User content
  | "assistant" ->
      Harmony.Assistant (channel (to_string (mem "channel" j)), content)
  | s -> failwith ("fixture: role " ^ s)

let render case () =
  equal ~msg:"ids" (array int)
    (ints (mem "ids" case))
    (Harmony.render (Lazy.force harmony)
       ?date:(to_string_option (mem "date" case))
       ?instructions:(to_string_option (mem "instructions" case))
       ~effort:(effort (to_string (mem "effort" case)))
       (List.map turn (to_list (mem "messages" case))))

(* The texts a parser gives, joined per message: a new message starts where
   the channel changes. Each text is valid UTF-8 on its own, since main prints
   it as it comes. *)
let parse case () =
  let add messages (channel, text) =
    equal ~msg:("valid UTF-8: " ^ String.escaped text) bool true
      (String.is_valid_utf_8 text);
    match messages with
    | (c, t) :: rest when c = channel -> (c, t ^ text) :: rest
    | _ -> (channel, text) :: messages
  in
  let parser, messages =
    Array.fold_left
      (fun (parser, messages) id ->
        match Harmony.feed parser id with
        | parser, None -> (parser, messages)
        | parser, Some piece -> (parser, add messages piece))
      (Harmony.parser (Lazy.force harmony), [])
      (ints (mem "ids" case))
  in
  let message m =
    (channel (to_string (mem "channel" m)), to_string (mem "content" m))
  in
  let name = function
    | Harmony.Analysis -> "analysis"
    | Commentary -> "commentary"
    | Final -> "final"
  in
  let named = List.map (fun (c, t) -> (name c, t)) in
  equal ~msg:"messages" (list (pair string string))
    (named (List.map message (to_list (mem "messages" case))))
    (named (List.rev messages));
  equal ~msg:"stopped" bool
    (to_bool (mem "stopped" case))
    (Harmony.stopped parser)

let cases key f =
  List.map
    (fun case -> test (to_string (mem "name" case)) (f case))
    (to_list (mem key fx))

let () =
  run "gpt-oss harmony against openai-harmony"
    [
      group "render" (cases "conversations" render);
      group "parse" (cases "completions" parse);
    ]
  |> exit
