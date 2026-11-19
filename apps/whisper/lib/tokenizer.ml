(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type task = Transcribe | Translate

type special =
  | End_of_text
  | Start_of_transcript
  | Language of string
  | Task of task
  | Start_of_lm
  | Start_of_prev
  | No_speech
  | No_timestamps
  | Timestamp of int

(* Whisper's languages in the order their tokens are numbered
   (whisper/tokenizer.py, LANGUAGES). *)
let codes =
  String.split_on_char ' '
    ("en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms "
   ^ "cs ro da hu ta no th ur hr bg lt la mi ml cy sk te fa lv bn sr az sl kn "
   ^ "et mk br eu is hy ne mn bs kk sq sw gl mr pa si km sn yo so af oc ka be "
   ^ "tg sd gu am yi lo uz fo ht ps tk nn mt sa lb my bo tl mg as tt haw ln "
   ^ "ha ba jw su yue")
  |> Array.of_list

let timestamps = 1501

let pattern =
  {|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+|}

(* [<|1.08|>] for [i = 54]. [i * 0.02] lies within an ulp of [2 i / 100], far
   from a rounding boundary, so Python's [f"{i * 0.02:.2f}"] is these digits. *)
let timestamp_text i = Printf.sprintf "<|%d.%02d|>" (2 * i / 100) (2 * i mod 100)

(* The specials are numbered [eot], [eot + 1], ... in [specials]' order. *)
type t = {
  brot : Brot.t;
  languages : string array;
  eot : int;
  non_speech : int array;
}

let languages t = Array.copy t.languages

(* A special's offset from [eot] in the order [of_ranks] adds them. *)
let offset t = function
  | End_of_text -> 0
  | Start_of_transcript -> 1
  | Language code -> (
      let rec find i =
        if i = Array.length t.languages then None
        else if t.languages.(i) = code then Some i
        else find (i + 1)
      in
      match find 0 with
      | Some i -> 2 + i
      | None -> invalid_argf "Whisper.Tokenizer.special: no language %S" code)
  | Task Translate -> 2 + Array.length t.languages
  | Task Transcribe -> 3 + Array.length t.languages
  | Start_of_lm -> 4 + Array.length t.languages
  | Start_of_prev -> 5 + Array.length t.languages
  | No_speech -> 6 + Array.length t.languages
  | No_timestamps -> 7 + Array.length t.languages
  | Timestamp i ->
      if i < 0 || i >= timestamps then
        invalid_argf "Whisper.Tokenizer.special: timestamp %d outside [0, %d]" i
          (timestamps - 1);
      8 + Array.length t.languages + i

let special t s = t.eot + offset t s

let sot_sequence t prompt =
  let sot = special t Start_of_transcript in
  match prompt with
  | None -> [| sot |]
  | Some (code, task) ->
      [| sot; special t (Language code); special t (Task task) |]

let encode t s = Brot.encode_ids t.brot ~match_special_tokens:false s
let decode_with_timestamps t ids = Brot.decode t.brot ids

let decode t ids =
  let first = special t (Timestamp 0) in
  Brot.decode t.brot
    (Array.of_list (List.filter (fun id -> id < first) (Array.to_list ids)))

let non_speech t = Array.copy t.non_speech
let vocab_size t = Brot.vocab_size t.brot

(* Non-speech tokens *)

let symbols =
  let chars = {s|"#()*+/:;<=>@[\]^_`{|}~|s} in
  List.init (String.length chars) (fun i -> String.make 1 chars.[i])
  @ [
      "「";
      "」";
      "『";
      "』";
      "<<";
      ">>";
      "<<<";
      ">>>";
      "--";
      "---";
      "-(";
      "-[";
      "('";
      "(\"";
      "((";
      "))";
      "(((";
      ")))";
      "[[";
      "]]";
      "{{";
      "}}";
      "♪♪";
      "♪♪♪";
    ]

(* Symbols of U+2640 to U+267F whose three UTF-8 bytes share the first two, so
   suppressing the first token of a longer encoding suppresses only them. *)
let miscellaneous = [ "♩"; "♪"; "♫"; "♬"; "♭"; "♮"; "♯" ]

let compute_non_speech t =
  let first s = (encode t s).(0) in
  let single s = match encode t s with [| id |] -> [ id ] | _ -> [] in
  let ids =
    [ first " -"; first " '" ]
    @ List.concat_map (fun s -> single s @ single (" " ^ s)) symbols
    @ List.concat_map (fun s -> [ first s; first (" " ^ s) ]) miscellaneous
  in
  Array.of_list (List.sort_uniq compare ids)

(* Constructors *)

let specials languages =
  [ "<|endoftext|>"; "<|startoftranscript|>" ]
  @ Array.to_list (Array.map (Printf.sprintf "<|%s|>") languages)
  @ [
      "<|translate|>";
      "<|transcribe|>";
      "<|startoflm|>";
      "<|startofprev|>";
      "<|nospeech|>";
      "<|notimestamps|>";
    ]
  @ List.init timestamps timestamp_text

let of_ranks ~languages ranks =
  if languages < 0 || languages > Array.length codes then
    invalid_argf "Whisper.Tokenizer.of_ranks: %d languages outside [0, %d]"
      languages (Array.length codes);
  match Brot.of_tiktoken ~pattern ranks with
  | Error e -> Error e
  | Ok brot -> (
      let languages = Array.sub codes 0 languages in
      let specials = specials languages in
      let brot = Brot.add_tokens brot (List.map Brot.added_token specials) in
      let id s = Option.get (Brot.token_to_id brot s) in
      let eot = id (List.hd specials) in
      (* A rank whose bytes are a special's text would keep its id. *)
      match
        List.find_opt
          (fun (k, s) -> id s <> eot + k)
          (List.mapi (fun k s -> (k, s)) specials)
      with
      | Some (_, s) ->
          Error (Printf.sprintf "the ranks hold the special token %s" s)
      | None ->
          let t = { brot; languages; eot; non_speech = [||] } in
          Ok { t with non_speech = compute_non_speech t })
