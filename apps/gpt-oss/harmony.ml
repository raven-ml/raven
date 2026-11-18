(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type effort = Low | Medium | High
type channel = Analysis | Commentary | Final
type turn = User of string | Assistant of channel * string

type t = {
  tokenizer : Brot.t;
  start : int;
  message : int;
  channel : int;
  end_ : int;
  return : int;
  call : int;
  end_of_text : int;
}

let of_file path =
  let fail msg = failwith (path ^ ": " ^ msg) in
  let tokenizer =
    match Brot.from_file path with Ok t -> t | Error e -> fail e
  in
  let id token =
    match Brot.token_to_id tokenizer token with
    | Some id -> id
    | None -> fail ("no token " ^ token)
  in
  {
    tokenizer;
    start = id "<|start|>";
    message = id "<|message|>";
    channel = id "<|channel|>";
    end_ = id "<|end|>";
    return = id "<|return|>";
    call = id "<|call|>";
    end_of_text = id "<|endoftext|>";
  }

(* Rendering *)

let effort_name = function Low -> "low" | Medium -> "medium" | High -> "high"

let channel_name = function
  | Analysis -> "analysis"
  | Commentary -> "commentary"
  | Final -> "final"

let system_text ?date effort =
  let date =
    match date with None -> "" | Some d -> "Current date: " ^ d ^ "\n"
  in
  "You are ChatGPT, a large language model trained by OpenAI.\n\
   Knowledge cutoff: 2024-06\n" ^ date ^ "\nReasoning: " ^ effort_name effort
  ^ "\n\n\
     # Valid channels: analysis, commentary, final. Channel must be included \
     for every message."

let is_final = function Assistant (Final, _) -> true | _ -> false
let is_assistant = function Assistant _ -> true | User _ -> false

(* Once a final message closes the last assistant turn, the reasoning that led
   to the first final message is no longer shown to the model. *)
let drop_stale_analysis turns =
  let answered =
    match List.find_opt is_assistant (List.rev turns) with
    | Some last -> is_final last
    | None -> false
  in
  if not answered then turns
  else
    let rec drop = function
      | Assistant (Analysis, _) :: rest -> drop rest
      | ([] | Assistant (Final, _) :: _) as rest -> rest
      | turn :: rest -> turn :: drop rest
    in
    drop turns

let render t ?date ?instructions ~effort turns =
  let buf = ref [] in
  let id i = buf := i :: !buf in
  let text s =
    Array.iter id (Brot.encode_ids t.tokenizer ~match_special_tokens:false s)
  in
  let message role ?channel content =
    id t.start;
    text role;
    Option.iter
      (fun c ->
        id t.channel;
        text (channel_name c))
      channel;
    id t.message;
    text content;
    id t.end_
  in
  message "system" (system_text ?date effort);
  Option.iter
    (fun i -> message "developer" ("# Instructions\n\n" ^ i))
    instructions;
  List.iter
    (function
      | User content -> message "user" content
      | Assistant (channel, content) -> message "assistant" ~channel content)
    (drop_stale_analysis turns);
  id t.start;
  text "assistant";
  Array.of_list (List.rev !buf)

(* Parsing *)

type state =
  | Header of int list option
  | Content of channel * Brot.stream
  | Stopped

type parser = { encoding : t; state : state }

let parser encoding = { encoding; state = Header None }

let stopped p =
  match p.state with Stopped -> true | Header _ | Content _ -> false

let channel_of_header t rev_ids =
  let name = Brot.decode t.tokenizer (Array.of_list (List.rev rev_ids)) in
  match String.split_on_char ' ' (String.trim name) with
  | "analysis" :: _ -> Analysis
  | "commentary" :: _ -> Commentary
  | _ -> Final

let content t channel = Content (channel, Brot.decode_stream t.tokenizer)
let text channel = function "" -> None | s -> Some (channel, s)

let feed p id =
  let t = p.encoding in
  let closes = id = t.end_ || id = t.start in
  let stops = id = t.return || id = t.call || id = t.end_of_text in
  match p.state with
  | Stopped -> (p, None)
  | (Header _ | Content _) when closes || stops ->
      let state = if stops then Stopped else Header None in
      let held =
        match p.state with
        | Content (channel, s) -> text channel (Brot.decode_finish s)
        | Header _ | Stopped -> None
      in
      ({ p with state }, held)
  | Header None ->
      let state =
        if id = t.channel then Header (Some [])
        else if id = t.message then content t Final
        else p.state
      in
      ({ p with state }, None)
  | Header (Some name) ->
      let state =
        if id = t.message then content t (channel_of_header t name)
        else Header (Some (id :: name))
      in
      ({ p with state }, None)
  | Content (channel, s) ->
      let piece, s = Brot.decode_step s id in
      ({ p with state = Content (channel, s) }, text channel piece)
