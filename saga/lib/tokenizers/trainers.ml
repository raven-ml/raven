(** Training module for tokenization models.

    This module provides a unified interface for training different tokenization
    models. It wraps the actual training implementations in Bpe.Trainer and
    Wordpiece.Trainer. *)

type training_result = { model : Models.t; special_tokens : string list }
(** Training result *)

type bpe_config = {
  vocab_size : int;
  min_frequency : int;
  show_progress : bool;
  special_tokens : string list;
  limit_alphabet : int;
  initial_alphabet : char list;
  continuing_subword_prefix : string option;
  end_of_word_suffix : string option;
}
(** Configuration types for each trainer *)

type wordpiece_config = {
  vocab_size : int;
  min_frequency : int;
  show_progress : bool;
  special_tokens : string list;
  limit_alphabet : int;
  initial_alphabet : char list;
  continuing_subword_prefix : string;
}

type wordlevel_config = {
  vocab_size : int;
  min_frequency : int;
  show_progress : bool;
  special_tokens : string list;
}

type unigram_config = {
  vocab_size : int;
  show_progress : bool;
  special_tokens : string list;
  shrinking_factor : float;
  unk_token : string option;
  max_piece_length : int;
  n_sub_iterations : int;
}

(** Main trainer type *)
type t =
  | BPE of bpe_config
  | WordPiece of wordpiece_config
  | WordLevel of wordlevel_config
  | Unigram of unigram_config

(** Read lines from files *)
let read_files files =
  let lines = ref [] in
  List.iter
    (fun file ->
      let ic = open_in file in
      try
        while true do
          lines := input_line ic :: !lines
        done
      with End_of_file -> close_in ic)
    files;
  List.rev !lines

(** Read lines from iterator *)
let read_iterator iterator =
  let lines = ref [] in
  let rec loop () =
    match iterator () with
    | None -> ()
    | Some line ->
        lines := line :: !lines;
        loop ()
  in
  loop ();
  List.rev !lines

(** Train BPE model *)
let train_bpe (config : bpe_config) lines existing_model =
  let trainer_config =
    Bpe.Trainer.
      {
        min_frequency = config.min_frequency;
        vocab_size = config.vocab_size;
        show_progress = config.show_progress;
        special_tokens = config.special_tokens;
        limit_alphabet = Some config.limit_alphabet;
        initial_alphabet = config.initial_alphabet;
        continuing_subword_prefix = config.continuing_subword_prefix;
        end_of_word_suffix = config.end_of_word_suffix;
        max_token_length = None;
      }
  in

  let trainer = Bpe.Trainer.create trainer_config in
  Bpe.Trainer.feed trainer lines;

  (* Create or use existing BPE model *)
  let base_model =
    match existing_model with
    | Some (Models.BPE bpe) ->
        Bpe.create
          {
            vocab = bpe.vocab;
            merges = bpe.merges;
            cache_capacity = bpe.cache_capacity;
            dropout = bpe.dropout;
            unk_token = bpe.unk_token;
            continuing_subword_prefix = bpe.continuing_subword_prefix;
            end_of_word_suffix = bpe.end_of_word_suffix;
            fuse_unk = bpe.fuse_unk;
            byte_fallback = bpe.byte_fallback;
            ignore_merges = false;
          }
    | _ ->
        (* Create empty model *)
        Bpe.create
          {
            vocab = Hashtbl.create 100;
            merges = [];
            cache_capacity = 10000;
            dropout = None;
            unk_token = None;
            continuing_subword_prefix = config.continuing_subword_prefix;
            end_of_word_suffix = config.end_of_word_suffix;
            fuse_unk = false;
            byte_fallback = false;
            ignore_merges = false;
          }
  in

  let special_tokens = Bpe.Trainer.train trainer base_model in

  (* Convert back to Models.t *)
  let vocab_list = Bpe.get_vocab base_model in
  let vocab_tbl = Hashtbl.create (List.length vocab_list) in
  List.iter (fun (token, id) -> Hashtbl.add vocab_tbl token id) vocab_list;

  let trained_model =
    Models.BPE
      {
        vocab = vocab_tbl;
        merges = [];
        (* TODO: Get merges from trained model *)
        cache_capacity = 10000;
        dropout = None;
        unk_token = None;
        continuing_subword_prefix = config.continuing_subword_prefix;
        end_of_word_suffix = config.end_of_word_suffix;
        fuse_unk = false;
        byte_fallback = false;
      }
  in

  { model = trained_model; special_tokens }

(** Train WordPiece model *)
let train_wordpiece (config : wordpiece_config) lines existing_model =
  let trainer_config =
    Wordpiece.Trainer.
      {
        min_frequency = config.min_frequency;
        vocab_size = config.vocab_size;
        show_progress = config.show_progress;
        special_tokens = config.special_tokens;
        limit_alphabet = Some config.limit_alphabet;
        initial_alphabet = config.initial_alphabet;
        continuing_subword_prefix = config.continuing_subword_prefix;
        end_of_word_suffix = None;
        (* WordPiece doesn't use end_of_word_suffix typically *)
      }
  in

  let trainer = Wordpiece.Trainer.create trainer_config in
  Wordpiece.Trainer.feed trainer lines;

  (* Create or use existing WordPiece model *)
  let base_model =
    match existing_model with
    | Some (Models.WordPiece wp) ->
        Wordpiece.create
          {
            vocab = wp.vocab;
            unk_token = wp.unk_token;
            max_input_chars_per_word = wp.max_input_chars_per_word;
            continuing_subword_prefix = config.continuing_subword_prefix;
          }
    | _ ->
        (* Create empty model *)
        Wordpiece.create
          {
            vocab = Hashtbl.create 100;
            unk_token = "[UNK]";
            max_input_chars_per_word = 100;
            continuing_subword_prefix = config.continuing_subword_prefix;
          }
  in

  let special_tokens = Wordpiece.Trainer.train trainer base_model in

  (* Convert back to Models.t *)
  let vocab_list = Wordpiece.get_vocab base_model in
  let vocab_tbl = Hashtbl.create (List.length vocab_list) in
  List.iter (fun (token, id) -> Hashtbl.add vocab_tbl token id) vocab_list;

  let trained_model =
    Models.WordPiece
      { vocab = vocab_tbl; unk_token = "[UNK]"; max_input_chars_per_word = 100 }
  in

  { model = trained_model; special_tokens }

(** Train WordLevel model *)
let train_wordlevel (config : wordlevel_config) lines _existing_model =
  let vocab_size = config.vocab_size in
  let min_frequency = config.min_frequency in
  let special_tokens = config.special_tokens in

  (* Simple word-level tokenization: just count word frequencies *)
  let word_counts = Hashtbl.create 10000 in

  List.iter
    (fun line ->
      let words = Str.split (Str.regexp "[ \t\n]+") line in
      List.iter
        (fun word ->
          let count = try Hashtbl.find word_counts word with Not_found -> 0 in
          Hashtbl.replace word_counts word (count + 1))
        words)
    lines;

  (* Sort by frequency and take top vocab_size *)
  let sorted =
    Hashtbl.fold (fun word count acc -> (word, count) :: acc) word_counts []
    |> List.sort (fun (_, c1) (_, c2) -> compare c2 c1)
  in

  (* Build vocabulary *)
  let vocab = Hashtbl.create vocab_size in
  let rec build_vocab words id =
    match words with
    | [] -> ()
    | _ when id >= vocab_size -> ()
    | (word, count) :: rest ->
        if count >= min_frequency then (
          Hashtbl.add vocab word id;
          build_vocab rest (id + 1))
        else build_vocab rest id
  in

  (* Add special tokens first *)
  List.iteri (fun i token -> Hashtbl.add vocab token i) special_tokens;

  build_vocab sorted (List.length special_tokens);

  let model = Models.WordLevel { vocab; unk_token = "<unk>" } in

  { model; special_tokens }

(** Train Unigram model *)
let train_unigram (_config : unigram_config) _lines _existing_model =
  (* Minimal unigram-style trainer: build token frequency over whitespace tokens
     and assign probabilities proportional to frequency. *)
  let lines = _lines in
  let freq = Hashtbl.create 10000 in
  List.iter
    (fun line ->
      let words = Str.split (Str.regexp "[ \t\n]+") line in
      List.iter
        (fun w ->
          let c =
            match Hashtbl.find_opt freq w with Some x -> x | None -> 0
          in
          Hashtbl.replace freq w (c + 1))
        words)
    lines;
  let total = float_of_int (Hashtbl.fold (fun _ c acc -> acc + c) freq 0) in
  let vocab =
    Hashtbl.fold
      (fun w c acc -> (w, float_of_int c /. max 1.0 total) :: acc)
      freq []
  in
  let model = Models.Unigram { vocab } in
  { model; special_tokens = [] }

(** Main training function *)
let train trainer ~files ?model () =
  let lines = read_files files in
  match trainer with
  | BPE config -> train_bpe config lines model
  | WordPiece config -> train_wordpiece config lines model
  | WordLevel config -> train_wordlevel config lines model
  | Unigram config -> train_unigram config lines model

(** Train from iterator *)
let train_from_iterator trainer ~iterator ?model () =
  let lines = read_iterator iterator in
  match trainer with
  | BPE config -> train_bpe config lines model
  | WordPiece config -> train_wordpiece config lines model
  | WordLevel config -> train_wordlevel config lines model
  | Unigram config -> train_unigram config lines model

(** Constructors *)
let bpe ?(vocab_size = 30000) ?(min_frequency = 0) ?(special_tokens = [])
    ?(limit_alphabet = 1000) ?(initial_alphabet = [])
    ?(continuing_subword_prefix = "") ?(end_of_word_suffix = "")
    ?(show_progress = true) ?max_token_length () =
  let _ = max_token_length in
  BPE
    {
      vocab_size;
      min_frequency;
      show_progress;
      special_tokens;
      limit_alphabet;
      initial_alphabet =
        List.map
          (fun s -> if String.length s > 0 then s.[0] else ' ')
          initial_alphabet;
      continuing_subword_prefix =
        (if continuing_subword_prefix = "" then None
         else Some continuing_subword_prefix);
      end_of_word_suffix =
        (if end_of_word_suffix = "" then None else Some end_of_word_suffix);
    }

let wordpiece ?(vocab_size = 30000) ?(min_frequency = 0) ?(special_tokens = [])
    ?(limit_alphabet = 1000) ?(initial_alphabet = [])
    ?(continuing_subword_prefix = "##") ?(end_of_word_suffix = "")
    ?(unk_token = "[UNK]") ?(show_progress = true) () =
  let _ = (end_of_word_suffix, unk_token) in
  WordPiece
    {
      vocab_size;
      min_frequency;
      show_progress;
      special_tokens;
      limit_alphabet;
      initial_alphabet =
        List.map
          (fun s -> if String.length s > 0 then s.[0] else ' ')
          initial_alphabet;
      continuing_subword_prefix;
    }

let word_level ?(vocab_size = 30000) ?(min_frequency = 0) ?(special_tokens = [])
    ?(show_progress = true) () =
  WordLevel { vocab_size; min_frequency; show_progress; special_tokens }

let unigram ?(vocab_size = 8000) ?(n_sub_iterations = 2)
    ?(shrinking_factor = 0.75) ?(unk_token = "<unk>") ?(special_tokens = [])
    ?(show_progress = true) ?(initial_alphabet = []) ?(max_piece_length = 16) ()
    =
  let _ = initial_alphabet in
  Unigram
    {
      vocab_size;
      show_progress;
      special_tokens;
      shrinking_factor;
      unk_token = (if unk_token = "" then None else Some unk_token);
      max_piece_length;
      n_sub_iterations;
    }

let chars ?(min_frequency = 0) ?(special_tokens = []) ?(show_progress = true) ()
    =
  (* Character-level tokenization is similar to word-level but with single
     characters *)
  WordLevel
    {
      vocab_size = 256;
      (* ASCII characters typically *)
      min_frequency;
      show_progress;
      special_tokens;
    }

(** Serialization *)
let to_json = function
  | BPE bpe ->
      `Assoc
        [
          ("type", `String "BpeTrainer");
          ("vocab_size", `Int bpe.vocab_size);
          ("min_frequency", `Int bpe.min_frequency);
          ("show_progress", `Bool bpe.show_progress);
          ( "special_tokens",
            `List (List.map (fun s -> `String s) bpe.special_tokens) );
          ("limit_alphabet", `Int bpe.limit_alphabet);
          ( "initial_alphabet",
            `List
              (List.map
                 (fun c -> `String (String.make 1 c))
                 bpe.initial_alphabet) );
          ( "continuing_subword_prefix",
            match bpe.continuing_subword_prefix with
            | None -> `Null
            | Some s -> `String s );
          ( "end_of_word_suffix",
            match bpe.end_of_word_suffix with
            | None -> `Null
            | Some s -> `String s );
        ]
  | WordPiece wp ->
      `Assoc
        [
          ("type", `String "WordPieceTrainer");
          ("vocab_size", `Int wp.vocab_size);
          ("min_frequency", `Int wp.min_frequency);
          ("show_progress", `Bool wp.show_progress);
          ( "special_tokens",
            `List (List.map (fun s -> `String s) wp.special_tokens) );
          ("limit_alphabet", `Int wp.limit_alphabet);
          ( "initial_alphabet",
            `List
              (List.map
                 (fun c -> `String (String.make 1 c))
                 wp.initial_alphabet) );
          ("continuing_subword_prefix", `String wp.continuing_subword_prefix);
        ]
  | WordLevel wl ->
      `Assoc
        [
          ("type", `String "WordLevelTrainer");
          ("vocab_size", `Int wl.vocab_size);
          ("min_frequency", `Int wl.min_frequency);
          ("show_progress", `Bool wl.show_progress);
          ( "special_tokens",
            `List (List.map (fun s -> `String s) wl.special_tokens) );
        ]
  | Unigram ug ->
      `Assoc
        [
          ("type", `String "UnigramTrainer");
          ("vocab_size", `Int ug.vocab_size);
          ("show_progress", `Bool ug.show_progress);
          ( "special_tokens",
            `List (List.map (fun s -> `String s) ug.special_tokens) );
          ("shrinking_factor", `Float ug.shrinking_factor);
          ( "unk_token",
            match ug.unk_token with None -> `Null | Some s -> `String s );
          ("max_piece_length", `Int ug.max_piece_length);
          ("n_sub_iterations", `Int ug.n_sub_iterations);
        ]

let of_json = function
  | `Assoc fields -> (
      match List.assoc_opt "type" fields with
      | Some (`String "BpeTrainer") ->
          let vocab_size =
            match List.assoc_opt "vocab_size" fields with
            | Some (`Int i) -> i
            | _ -> 30000
          in
          let min_frequency =
            match List.assoc_opt "min_frequency" fields with
            | Some (`Int i) -> i
            | _ -> 0
          in
          let show_progress =
            match List.assoc_opt "show_progress" fields with
            | Some (`Bool b) -> b
            | _ -> true
          in
          let special_tokens =
            match List.assoc_opt "special_tokens" fields with
            | Some (`List tokens) ->
                List.map (function `String s -> s | _ -> "") tokens
            | _ -> []
          in
          let limit_alphabet =
            match List.assoc_opt "limit_alphabet" fields with
            | Some (`Int i) -> i
            | _ -> 1000
          in
          let initial_alphabet =
            match List.assoc_opt "initial_alphabet" fields with
            | Some (`List chars) ->
                List.map (function `String s -> s | _ -> "") chars
            | _ -> []
          in
          let continuing_subword_prefix =
            match List.assoc_opt "continuing_subword_prefix" fields with
            | Some (`String s) -> s
            | _ -> ""
          in
          let end_of_word_suffix =
            match List.assoc_opt "end_of_word_suffix" fields with
            | Some (`String s) -> s
            | _ -> ""
          in
          bpe ~vocab_size ~min_frequency ~show_progress ~special_tokens
            ~limit_alphabet ~initial_alphabet ~continuing_subword_prefix
            ~end_of_word_suffix ()
      | Some (`String "WordPieceTrainer") ->
          let vocab_size =
            match List.assoc_opt "vocab_size" fields with
            | Some (`Int i) -> i
            | _ -> 30000
          in
          let min_frequency =
            match List.assoc_opt "min_frequency" fields with
            | Some (`Int i) -> i
            | _ -> 0
          in
          let show_progress =
            match List.assoc_opt "show_progress" fields with
            | Some (`Bool b) -> b
            | _ -> true
          in
          let special_tokens =
            match List.assoc_opt "special_tokens" fields with
            | Some (`List tokens) ->
                List.map (function `String s -> s | _ -> "") tokens
            | _ -> []
          in
          let limit_alphabet =
            match List.assoc_opt "limit_alphabet" fields with
            | Some (`Int i) -> i
            | _ -> 1000
          in
          let initial_alphabet =
            match List.assoc_opt "initial_alphabet" fields with
            | Some (`List chars) ->
                List.map (function `String s -> s | _ -> "") chars
            | _ -> []
          in
          let continuing_subword_prefix =
            match List.assoc_opt "continuing_subword_prefix" fields with
            | Some (`String s) -> s
            | _ -> "##"
          in
          wordpiece ~vocab_size ~min_frequency ~show_progress ~special_tokens
            ~limit_alphabet ~initial_alphabet ~continuing_subword_prefix ()
      | Some (`String "WordLevelTrainer") ->
          let vocab_size =
            match List.assoc_opt "vocab_size" fields with
            | Some (`Int i) -> i
            | _ -> 30000
          in
          let min_frequency =
            match List.assoc_opt "min_frequency" fields with
            | Some (`Int i) -> i
            | _ -> 0
          in
          let show_progress =
            match List.assoc_opt "show_progress" fields with
            | Some (`Bool b) -> b
            | _ -> true
          in
          let special_tokens =
            match List.assoc_opt "special_tokens" fields with
            | Some (`List tokens) ->
                List.map (function `String s -> s | _ -> "") tokens
            | _ -> []
          in
          word_level ~vocab_size ~min_frequency ~show_progress ~special_tokens
            ()
      | Some (`String "UnigramTrainer") ->
          let vocab_size =
            match List.assoc_opt "vocab_size" fields with
            | Some (`Int i) -> i
            | _ -> 8000
          in
          let show_progress =
            match List.assoc_opt "show_progress" fields with
            | Some (`Bool b) -> b
            | _ -> true
          in
          let special_tokens =
            match List.assoc_opt "special_tokens" fields with
            | Some (`List tokens) ->
                List.map (function `String s -> s | _ -> "") tokens
            | _ -> []
          in
          let shrinking_factor =
            match List.assoc_opt "shrinking_factor" fields with
            | Some (`Float f) -> f
            | _ -> 0.75
          in
          let unk_token =
            match List.assoc_opt "unk_token" fields with
            | Some (`String s) -> s
            | _ -> "<unk>"
          in
          let max_piece_length =
            match List.assoc_opt "max_piece_length" fields with
            | Some (`Int i) -> i
            | _ -> 16
          in
          let n_sub_iterations =
            match List.assoc_opt "n_sub_iterations" fields with
            | Some (`Int i) -> i
            | _ -> 2
          in
          unigram ~vocab_size ~show_progress ~special_tokens ~shrinking_factor
            ~unk_token ~max_piece_length ~n_sub_iterations ()
      | _ -> failwith "Unknown trainer type")
  | _ -> failwith "Invalid trainer JSON"
