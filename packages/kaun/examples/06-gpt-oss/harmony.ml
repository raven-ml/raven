(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type effort = Low | Medium | High
type channel = Analysis | Commentary | Final
type turn = User of string | Assistant of channel * string

type t = {
  full : Brot.t;
  plain : Brot.t;
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
  let json =
    let text = In_channel.with_open_bin path In_channel.input_all in
    match Jsont_bytesrw.decode_string Jsont.json text with
    | Ok json -> json
    | Error e -> fail e
  in
  let tokenizer json =
    match Brot.of_json json with Ok t -> t | Error e -> fail e
  in
  let without_added_tokens =
    match json with
    | Jsont.Object (mems, meta) ->
        let keep ((name, _), _) = name <> "added_tokens" in
        Jsont.Object (List.filter keep mems, meta)
    | _ -> fail "not a JSON object"
  in
  let full = tokenizer json in
  let id token =
    match Brot.token_to_id full token with
    | Some id -> id
    | None -> fail ("no token " ^ token)
  in
  {
    full;
    plain = tokenizer without_added_tokens;
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
  let text s = Array.iter id (Brot.encode_ids t.plain s) in
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
  | Content of channel * int list
  | Stopped

type parser = { encoding : t; state : state }

let parser encoding = { encoding; state = Header None }

let stopped p =
  match p.state with Stopped -> true | Header _ | Content _ -> false

let decode t rev_ids = Brot.decode t.full (Array.of_list (List.rev rev_ids))

let channel_of_header t rev_ids =
  match String.split_on_char ' ' (String.trim (decode t rev_ids)) with
  | "analysis" :: _ -> Analysis
  | "commentary" :: _ -> Commentary
  | _ -> Final

let replacement = "\xEF\xBF\xBD"

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
        | Content (channel, (_ :: _ as held)) -> Some (channel, decode t held)
        | Content (_, []) | Header _ | Stopped -> None
      in
      ({ p with state }, held)
  | Header None ->
      let state =
        if id = t.channel then Header (Some [])
        else if id = t.message then Content (Final, [])
        else p.state
      in
      ({ p with state }, None)
  | Header (Some name) ->
      let state =
        if id = t.message then Content (channel_of_header t name, [])
        else Header (Some (id :: name))
      in
      ({ p with state }, None)
  | Content (channel, held) ->
      let held = id :: held in
      let text = decode t held in
      if String.ends_with ~suffix:replacement text then
        ({ p with state = Content (channel, held) }, None)
      else ({ p with state = Content (channel, []) }, Some (channel, text))
