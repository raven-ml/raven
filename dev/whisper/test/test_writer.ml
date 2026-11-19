(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The writers against upstream's.

   upstream/writer.json holds transcripts, each window with its seek, size, its
   chosen attempt's temperature, average log probability, compression ratio and
   no-speech probability, and its segments in frames, and what upstream's
   WriteTXT, WriteVTT, WriteSRT, WriteTSV and WriteJSON write for the result
   transcribe would return for them: white space at the ends of texts, Unicode's
   included, "-->" and tabs in texts, times past an hour, non-ASCII text and a
   character past U+FFFF, cleared segments. *)

open Windtrap
open Whisper_support
module Tr = Whisper.Transcript

let golden = Json.read "upstream/writer.json"

(* A window's chosen attempt: its rung at its temperature, the probabilities and
   the ratio the writers read. *)
let window j : Tr.window =
  let num k = Json.float (Json.mem k j) in
  let temperature = num "temperature" in
  let rung : Tr.rung =
    if temperature = 0. then Greedy else Sample { best_of = 1; temperature }
  in
  let choice : Tr.attempt =
    {
      rung;
      result =
        {
          tokens = [||];
          text = "";
          avg_logprob = num "avg_logprob";
          no_speech_prob = num "no_speech_prob";
        };
      compression_ratio = num "compression_ratio";
    }
  in
  let segments =
    List.map
      (fun s : Tr.segment ->
        {
          start = Json.int (Json.mem "start" s);
          stop = Json.int (Json.mem "stop" s);
          tokens = Json.ints (Json.mem "tokens" s);
          text = Json.string (Json.mem "text" s);
        })
      (Json.list (Json.mem "segments" j))
  in
  {
    seek = Json.int (Json.mem "seek" j);
    size = Json.int (Json.mem "size" j);
    choice;
    rejected = [];
    silent = segments = [];
    segments;
  }

let transcript c : Tr.t =
  {
    language = Json.string (Json.mem "language" c);
    text = Json.string (Json.mem "text" c);
    windows = List.map window (Json.list (Json.mem "windows" c));
  }

let formats =
  Whisper.Writer.
    [ ("txt", txt); ("vtt", vtt); ("srt", srt); ("tsv", tsv); ("json", json) ]

let upstream =
  cases "each format is upstream's writer's, byte for byte"
    ~id:(fun (c, (f, _)) -> Json.string (Json.mem "name" c) ^ " " ^ f)
    (List.concat_map
       (fun c -> List.map (fun f -> (c, f)) formats)
       (Json.list (Json.mem "cases" golden)))
    (fun (c, (f, write)) ->
      equal string
        (Json.string (Json.mem f (Json.mem "outputs" c)))
        (write (transcript c)))

let () = exit (run "whisper.writer" [ group "upstream" [ upstream ] ])
