(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Word error rates against jiwer (4.0.0, process_words with its default
   transform).

   upstream/normalize.json holds, under "corpora", the references and greedy
   transcripts of each model, protocol and LibriSpeech set of the oracle, as
   indices into its "strings", with jiwer's counts and rate after upstream's
   normaliser, and under "pairs", random word sequences that tie alignments,
   with their counts and rate. *)

open Windtrap
open Whisper_support
module W = Whisper.Wer

let golden = Json.read "upstream/normalize.json"

let strings =
  Array.of_list
    (List.map
       (fun p -> Json.string (List.hd (Json.list p)))
       (Json.list (Json.mem "strings" golden)))

let counts c =
  match Json.ints (Json.mem "counts" c) with
  | [| hits; substitutions; deletions; insertions |] ->
      { W.hits; substitutions; deletions; insertions }
  | _ -> failwith "counts"

let counts_t =
  Testable.make
    ~pp:(fun ppf (c : W.counts) ->
      Format.fprintf ppf "{H %d; S %d; D %d; I %d}" c.hits c.substitutions
        c.deletions c.insertions)
    ~equal:( = )

let corpora =
  cases "corpus after normalize is jiwer's on the oracle's transcripts"
    ~id:(fun c ->
      String.concat " "
        (List.map
           (fun k -> Json.string (Json.mem k c))
           [ "model"; "protocol"; "set" ]))
    (Json.list (Json.mem "corpora" golden))
    (fun c ->
      let text k = Whisper.English.normalize strings.(k) in
      let texts key =
        Array.to_list (Array.map text (Json.ints (Json.mem key c)))
      in
      let got = W.corpus (List.combine (texts "refs") (texts "hyps")) in
      equal counts_t (counts c) got;
      equal float_exact (Json.float (Json.mem "wer" c)) (W.rate got))

let pairs =
  cases "corpus of one pair is jiwer's"
    ~id:(fun (i, c) ->
      Printf.sprintf "%d: %S / %S" i
        (Json.string (Json.mem "reference" c))
        (Json.string (Json.mem "hypothesis" c)))
    (List.mapi (fun i c -> (i, c)) (Json.list (Json.mem "pairs" golden)))
    (fun (_, c) ->
      let got =
        W.corpus
          [
            ( Json.string (Json.mem "reference" c),
              Json.string (Json.mem "hypothesis" c) );
          ]
      in
      equal counts_t (counts c) got;
      equal float_exact (Json.float (Json.mem "wer" c)) (W.rate got))

(* The Levenshtein distance of two word sequences, by its recurrence. *)
let distance a b =
  let n = Array.length a and m = Array.length b in
  let d = Array.make_matrix (n + 1) (m + 1) 0 in
  for i = 0 to n do
    for j = 0 to m do
      d.(i).(j) <-
        (if i = 0 then j
         else if j = 0 then i
         else
           min
             (min (d.(i - 1).(j) + 1) (d.(i).(j - 1) + 1))
             (d.(i - 1).(j - 1) + if a.(i - 1) = b.(j - 1) then 0 else 1))
    done
  done;
  d.(n).(m)

let sentence =
  Gen.(array ~size:(int_range 0 12) (of_list [ "a"; "b"; "c"; "d" ]))

let least_edits =
  prop "align is an alignment of least edits"
    Gen.(pair sentence sentence)
    (fun (reference, hypothesis) ->
      let c = W.align ~reference ~hypothesis in
      cover "no reference word" (reference = [||]);
      cover "no hypothesis word" (hypothesis = [||]);
      cover "a tie" (distance reference hypothesis > 1);
      equal ~msg:"edits" int
        (distance reference hypothesis)
        (c.substitutions + c.deletions + c.insertions);
      equal ~msg:"reference words" int (Array.length reference)
        (c.hits + c.substitutions + c.deletions);
      equal ~msg:"hypothesis words" int (Array.length hypothesis)
        (c.hits + c.substitutions + c.insertions))

let words =
  cases "words splits as jiwer's default transform"
    ~id:(fun (s, _) -> Printf.sprintf "%S" s)
    [
      ("", [||]);
      ("  ", [||]);
      (" a  b ", [| "a"; "b" |]);
      ("a\tb", [| "a\tb" |]);
      ("a \t b", [| "a"; "b" |]);
      ("a\u{00A0}\u{00A0}b", [| "a"; "b" |]);
      ("\na b\n", [| "a"; "b" |]);
    ]
    (fun (s, want) -> equal (array string) want (W.words s))

let () =
  exit
    (run "whisper.wer"
       [
         group "words" [ words ];
         group "align" [ least_edits; pairs ];
         group "corpus" [ corpora ];
       ])
