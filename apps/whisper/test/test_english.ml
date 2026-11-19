(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The normaliser against upstream's EnglishTextNormalizer.

   upstream/normalize.json holds, under "strings", texts and what upstream's
   normaliser gives for them: edge strings, the oracle's references of 219
   LibriSpeech utterances and the greedy transcripts of tiny, tiny.en and
   base.en with and without timestamps. *)

open Windtrap
open Whisper_support

let golden = Json.read "upstream/normalize.json"

let strings =
  List.map
    (fun p ->
      match Json.list p with
      | [ text; normalised ] -> (Json.string text, Json.string normalised)
      | _ -> failwith "strings")
    (Json.list (Json.mem "strings" golden))

let upstream =
  cases "normalize is upstream's EnglishTextNormalizer"
    ~id:(fun (text, _) -> Printf.sprintf "%S" text)
    strings
    (fun (text, want) -> equal string want (Whisper.English.normalize text))

(* Texts of words, numbers, contractions, symbols and white space. *)
let text_gen =
  let open Gen in
  let piece =
    of_list
      [
        "one";
        "twenty";
        "hundred";
        "and";
        "a";
        "half";
        "point";
        "five";
        "percent";
        "dollars";
        "cents";
        "Mr.";
        "won't";
        "it's";
        "1,000";
        "3.5";
        "$";
        "%";
        "¢";
        "(";
        ")";
        "[";
        "]";
        "<";
        ">";
        "'";
        ".";
        ",";
        " ";
        "  ";
        "\t";
        "\n";
        "\u{00A0}";
        "\u{3000}";
        "café";
        "Straße";
        "ΣΟΦΟΣ";
        "uh";
        "colour";
        "1";
        "1s";
        "double";
        "oh";
        "minus";
        "-";
        "first";
        "sixties";
      ]
  in
  map (String.concat "") (list ~size:(int_range 0 12) piece)

let spaces =
  prop "normalize leaves single spaces, and no other white space" text_gen
    (fun text ->
      let s = Whisper.English.normalize text in
      not_contains ~msg:"two spaces" ~sub:"  " s;
      String.iter
        (fun c ->
          if c = '\t' || c = '\n' || c = '\r' then failf "white space %C" c)
        s;
      not_contains ~msg:"no-break space" ~sub:"\u{00A0}" s;
      not_contains ~msg:"ideographic space" ~sub:"\u{3000}" s)

let () = exit (run "whisper.english" [ group "normalize" [ upstream; spaces ] ])
