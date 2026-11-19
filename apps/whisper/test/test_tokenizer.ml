(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The tokenizers against upstream's own Tokenizer over trimmed rank tables.

   upstream/gpt2-1000.tiktoken and upstream/multilingual-1000.tiktoken hold the
   first 1000 ranks of Whisper's two rank files, the multilingual one with the
   empty placeholder appended at rank 1000 as the full file holds it last;
   upstream/tokenizer.json holds what upstream's Tokenizer gives over them with
   99 and 100 languages: specials, start sequences, the non-speech ids, and
   encodings and decodings. The full files are checked against the oracle
   outside runtest (check/check.exe tokenizer). *)

open Windtrap
open Whisper_support
module T = Whisper.Tokenizer

let golden = Json.read "upstream/tokenizer.json"
let configs = Json.list (Json.mem "configs" golden)
let name c = Json.string (Json.mem "name" c)

let tokenizer c =
  let ranks =
    In_channel.with_open_bin
      ("upstream/" ^ Json.string (Json.mem "file" c))
      In_channel.input_all
  in
  match T.of_ranks ~languages:(Json.int (Json.mem "languages" c)) ranks with
  | Ok t -> t
  | Error e -> failwith e

let tokenizers = List.map (fun c -> (c, tokenizer c)) configs

(* The goldens' specials in upstream's order, with their variants. *)
let specials t =
  let langs =
    Array.to_list (Array.map (fun c -> T.Language c) (T.languages t))
  in
  [ T.End_of_text; Start_of_transcript ]
  @ langs
  @ T.
      [
        Task Translate;
        Task Transcribe;
        Start_of_lm;
        Start_of_prev;
        No_speech;
        No_timestamps;
      ]
  @ List.map (fun i -> T.Timestamp i) [ 0; 1; 2; 54; 1499; 1500 ]

let per_config title f =
  cases title ~id:(fun (c, _) -> name c) tokenizers (fun (c, t) -> f c t)

let special_ids =
  per_config "special is upstream's id, and decodes to its text" (fun c t ->
      let want =
        List.map
          (fun p ->
            match Json.list p with
            | [ s; id ] -> (Json.string s, Json.int id)
            | _ -> failwith "specials")
          (Json.list (Json.mem "specials" c))
      in
      let got =
        List.map
          (fun s ->
            let id = T.special t s in
            (T.decode_with_timestamps t [| id |], id))
          (specials t)
      in
      equal (list (pair string int)) want got)

let vocab =
  per_config "vocab_size is upstream's n_vocab" (fun c t ->
      equal int (Json.int (Json.mem "n_vocab" c)) (T.vocab_size t))

let sot =
  per_config "sot_sequence is upstream's" (fun c t ->
      List.iter
        (fun (key, ids) ->
          let prompt =
            match String.split_on_char '/' key with
            | [ "none" ] -> None
            | [ code; "transcribe" ] -> Some (code, T.Transcribe)
            | [ code; "translate" ] -> Some (code, T.Translate)
            | _ -> failwith key
          in
          equal ~msg:key (array int) (Json.ints ids) (T.sot_sequence t prompt))
        (Json.mems (Json.mem "sot_sequences" c)))

let non_speech =
  per_config "non_speech is upstream's rule" (fun c t ->
      equal (array int)
        (Json.ints (Json.mem "non_speech_tokens" c))
        (T.non_speech t))

let encode =
  per_config "encode and decode are upstream's, special text as text"
    (fun c t ->
      List.iter
        (fun e ->
          let text = Json.string (Json.mem "text" e) in
          let ids = T.encode t text in
          equal ~msg:text (array int) (Json.ints (Json.mem "ids" e)) ids;
          equal ~msg:text string
            (Json.string (Json.mem "decode" e))
            (T.decode t ids))
        (Json.list (Json.mem "encode" c)))

let decode =
  per_config "decode drops timestamps, decode_with_timestamps writes them"
    (fun c t ->
      List.iter
        (fun d ->
          let ids = Json.ints (Json.mem "ids" d) in
          let msg =
            String.concat " " (Array.to_list (Array.map string_of_int ids))
          in
          equal ~msg string (Json.string (Json.mem "decode" d)) (T.decode t ids);
          equal ~msg string
            (Json.string (Json.mem "decode_with_timestamps" d))
            (T.decode_with_timestamps t ids))
        (Json.list (Json.mem "decode" c)))

let t99 = snd (List.nth tokenizers 1)
let t100 = snd (List.nth tokenizers 2)

let refused =
  cases "special refuses what the tokenizer has no token for"
    ~id:(fun (n, _, _) -> n)
    [
      ("yue among 99 languages", t99, T.Language "yue");
      ("an unknown code", t100, T.Language "xx");
      ("timestamp -1", t100, T.Timestamp (-1));
      ("timestamp 1501", t100, T.Timestamp 1501);
    ]
    (fun (_, t, s) -> raises_match Exn.invalid_arg (fun () -> T.special t s))

let yue =
  test "the 100th language is yue" (fun () ->
      equal int 99 (Array.length (T.languages t99));
      equal string "yue" (T.languages t100).(99);
      equal int
        (T.special t100 (Task Translate) - 1)
        (T.special t100 (Language "yue")))

let counts =
  cases "of_ranks refuses a language count outside [0, 100]" ~id:string_of_int
    [ -1; 101 ] (fun n ->
      raises_match Exn.invalid_arg (fun () ->
          T.of_ranks ~languages:n "IQ== 0\n"))

let malformed =
  test "of_ranks is an error on a malformed rank file" (fun () ->
      is_error (T.of_ranks ~languages:99 "IQ== 0\nIQ== 1\n"))

let () =
  exit
    (run "whisper.tokenizer"
       [
         group "upstream"
           [ special_ids; vocab; sot; non_speech; encode; decode ];
         group "contract" [ refused; yue; counts; malformed ];
       ])
