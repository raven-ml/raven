(** WordPiece tokenization implementation *)

type vocab = (string, int) Hashtbl.t
type vocab_r = (int, string) Hashtbl.t
type token = { id : int; value : string; offsets : int * int }

type config = {
  vocab : vocab;
  unk_token : string;
  continuing_subword_prefix : string;
  max_input_chars_per_word : int;
}

type t = {
  vocab : vocab;
  vocab_r : vocab_r;
  unk_token : string;
  continuing_subword_prefix : string;
  max_input_chars_per_word : int;
}

let create (cfg : config) : t =
  let vocab_r = Hashtbl.create (Hashtbl.length cfg.vocab) in
  Hashtbl.iter (fun k v -> Hashtbl.add vocab_r v k) cfg.vocab;

  (* Ensure unk_token is in vocabulary *)
  if not (Hashtbl.mem cfg.vocab cfg.unk_token) then
    failwith
      (Printf.sprintf "WordPiece error: Missing %s token from the vocabulary"
         cfg.unk_token);

  {
    vocab = cfg.vocab;
    vocab_r;
    unk_token = cfg.unk_token;
    continuing_subword_prefix = cfg.continuing_subword_prefix;
    max_input_chars_per_word = cfg.max_input_chars_per_word;
  }

let tokenize model sequence =
  (* Count characters *)
  let char_count =
    let decoder = Uutf.decoder (`String sequence) in
    let count = ref 0 in
    let rec loop () =
      match Uutf.decode decoder with
      | `Uchar _ ->
          incr count;
          loop ()
      | `End -> !count
      | `Malformed _ ->
          incr count;
          loop ()
      | `Await -> assert false
    in
    loop ()
  in

  (* If word is too long, return unknown token *)
  if char_count > model.max_input_chars_per_word then
    match Hashtbl.find_opt model.vocab model.unk_token with
    | Some id ->
        [
          { id; value = model.unk_token; offsets = (0, String.length sequence) };
        ]
    | None -> []
  else
    (* Greedy longest-match-first algorithm *)
    let rec tokenize_greedy start acc =
      if start >= String.length sequence then List.rev acc
      else
        let rec find_longest_match end_pos =
          if end_pos <= start then None
          else
            (* Extract substring *)
            let substr = String.sub sequence start (end_pos - start) in
            let token_str =
              if start > 0 then
                (* Add continuing subword prefix for non-initial tokens *)
                model.continuing_subword_prefix ^ substr
              else substr
            in

            (* Check if token exists in vocabulary *)
            match Hashtbl.find_opt model.vocab token_str with
            | Some id ->
                Some { id; value = token_str; offsets = (start, end_pos) }
            | None ->
                (* Try shorter substring *)
                (* Move back by one UTF-8 character *)
                let new_end =
                  if end_pos <= start + 1 then start (* Can't go shorter *)
                  else
                    let rec find_char_start pos =
                      if pos <= start then start
                      else
                        let byte = Char.code sequence.[pos] in
                        (* Check if this is the start of a UTF-8 character *)
                        if byte land 0xC0 <> 0x80 then pos
                        else find_char_start (pos - 1)
                    in
                    find_char_start (end_pos - 1)
                in
                if new_end = end_pos || new_end <= start then None
                  (* No shorter match possible *)
                else find_longest_match new_end
        in

        match find_longest_match (String.length sequence) with
        | Some token ->
            let end_pos = snd token.offsets in
            tokenize_greedy end_pos (token :: acc)
        | None -> (
            (* No match found, return unknown token for entire sequence *)
            match Hashtbl.find_opt model.vocab model.unk_token with
            | Some id ->
                [
                  {
                    id;
                    value = model.unk_token;
                    offsets = (0, String.length sequence);
                  };
                ]
            | None -> [])
    in

    tokenize_greedy 0 []

let token_to_id model token = Hashtbl.find_opt model.vocab token
let id_to_token model id = Hashtbl.find_opt model.vocab_r id
let get_vocab model = Hashtbl.fold (fun k v acc -> (k, v) :: acc) model.vocab []
let get_vocab_size model = Hashtbl.length model.vocab

let read_file ~vocab_file =
  let vocab = Hashtbl.create 10000 in
  let ic = open_in vocab_file in
  let index = ref 0 in
  (try
     while true do
       let line = input_line ic in
       let token = String.trim line in
       if String.length token > 0 then (
         Hashtbl.add vocab token !index;
         incr index)
     done
   with End_of_file -> ());
  close_in ic;
  vocab

let from_file ~vocab_file =
  let vocab = read_file ~vocab_file in
  let cfg : config =
    {
      vocab;
      unk_token = "[UNK]";
      continuing_subword_prefix = "##";
      max_input_chars_per_word = 100;
    }
  in
  create cfg

let default () =
  create
    {
      vocab = Hashtbl.create 0;
      unk_token = "[UNK]";
      continuing_subword_prefix = "##";
      max_input_chars_per_word = 100;
    }

let save model ~path ?name () =
  let vocab_file =
    match name with
    | Some n -> Filename.concat path (Printf.sprintf "%s-vocab.txt" n)
    | None -> Filename.concat path "vocab.txt"
  in

  (* Sort vocabulary by ID *)
  let vocab_list =
    Hashtbl.fold (fun k v acc -> (k, v) :: acc) model.vocab []
    |> List.sort (fun (_, a) (_, b) -> compare a b)
  in

  (* Write vocab.txt *)
  let oc = open_out vocab_file in
  List.iter
    (fun (token, _) ->
      output_string oc token;
      output_char oc '\n')
    vocab_list;
  close_out oc

(* Store reference to create function for Builder module *)
let create_internal = create

(** Builder module *)
module Builder = struct
  type builder = {
    mutable vocab : vocab;
    mutable unk_token : string;
    mutable continuing_subword_prefix : string;
    mutable max_input_chars_per_word : int;
  }

  let create () =
    {
      vocab = Hashtbl.create 0;
      unk_token = "[UNK]";
      continuing_subword_prefix = "##";
      max_input_chars_per_word = 100;
    }

  let vocab builder v =
    builder.vocab <- v;
    builder

  let unk_token builder token =
    builder.unk_token <- token;
    builder

  let continuing_subword_prefix builder prefix =
    builder.continuing_subword_prefix <- prefix;
    builder

  let max_input_chars_per_word builder max_chars =
    builder.max_input_chars_per_word <- max_chars;
    builder

  let build builder =
    create_internal
      {
        vocab = builder.vocab;
        unk_token = builder.unk_token;
        continuing_subword_prefix = builder.continuing_subword_prefix;
        max_input_chars_per_word = builder.max_input_chars_per_word;
      }
end

(** Trainer module *)
module Trainer = struct
  type trainer_config = {
    min_frequency : int;
    vocab_size : int;
    show_progress : bool;
    special_tokens : string list;
    limit_alphabet : int option;
    initial_alphabet : char list;
    continuing_subword_prefix : string;
    end_of_word_suffix : string option;
  }

  type trainer = { config : trainer_config; bpe_trainer : Bpe.Trainer.trainer }

  let _ = fun (t : trainer) -> t.config (* Suppress unused field warning *)

  let default_config =
    {
      min_frequency = 0;
      vocab_size = 30000;
      show_progress = true;
      special_tokens = [ "[PAD]"; "[UNK]"; "[CLS]"; "[SEP]"; "[MASK]" ];
      limit_alphabet = None;
      initial_alphabet = [];
      continuing_subword_prefix = "##";
      end_of_word_suffix = None;
    }

  let create config =
    (* Create a BPE trainer with WordPiece-specific settings *)
    let bpe_config : Bpe.Trainer.trainer_config =
      {
        min_frequency = config.min_frequency;
        vocab_size = config.vocab_size;
        show_progress = config.show_progress;
        special_tokens = config.special_tokens;
        limit_alphabet = config.limit_alphabet;
        initial_alphabet = config.initial_alphabet;
        continuing_subword_prefix = Some config.continuing_subword_prefix;
        end_of_word_suffix = config.end_of_word_suffix;
        max_token_length = None;
      }
    in
    { config; bpe_trainer = Bpe.Trainer.create bpe_config }

  let feed trainer texts = Bpe.Trainer.feed trainer.bpe_trainer texts

  let train trainer _model =
    (* Train a BPE model first *)
    let bpe_model = Bpe.default () in
    let special_tokens = Bpe.Trainer.train trainer.bpe_trainer bpe_model in

    (* Convert BPE model to WordPiece *)
    (* Note: In practice, WordPiece training is more complex and involves
       maximizing the likelihood of the training data. This is a simplified version. *)
    special_tokens
end

(** Conversion from BPE *)
let from_bpe bpe_model =
  let vocab =
    let v = Hashtbl.create 10000 in
    List.iter (fun (k, id) -> Hashtbl.add v k id) (Bpe.get_vocab bpe_model);
    v
  in

  let unk_token =
    match Bpe.token_to_id bpe_model "[UNK]" with
    | Some _ -> "[UNK]"
    | None -> "<unk>" (* Fallback if [UNK] not in vocab *)
  in

  create
    {
      vocab;
      unk_token;
      continuing_subword_prefix = "##";
      (* Default WordPiece prefix *)
      max_input_chars_per_word = 100;
    }
