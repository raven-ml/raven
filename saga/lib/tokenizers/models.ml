(** Tokenization models module. *)

type token = { id : int; value : string; offsets : int * int }
(** Tokenization result *)

type bpe_model = {
  vocab : Bpe.vocab;
  merges : Bpe.merges;
  cache_capacity : int;
  dropout : float option;
  unk_token : string option;
  continuing_subword_prefix : string option;
  end_of_word_suffix : string option;
  fuse_unk : bool;
  byte_fallback : bool;
}
(** BPE model configuration *)

type wordpiece_model = {
  vocab : Wordpiece.vocab;
  unk_token : string;
  max_input_chars_per_word : int;
}
(** WordPiece model configuration *)

type wordlevel_model = { vocab : (string, int) Hashtbl.t; unk_token : string }
(** WordLevel model configuration *)

type unigram_model = {
  vocab : (string * float) list;
  token_map : (string, int) Hashtbl.t;
  tokens : string array;
}
(** Unigram model configuration *)

(** Main model type *)
type t =
  | BPE of bpe_model
  | WordPiece of wordpiece_model
  | WordLevel of wordlevel_model
  | Unigram of unigram_model

(** Convert internal tokens to generic tokens *)
let convert_bpe_token (tok : Bpe.token) : token =
  { id = tok.id; value = tok.value; offsets = tok.offsets }

let convert_wordpiece_token (tok : Wordpiece.token) : token =
  { id = tok.id; value = tok.value; offsets = tok.offsets }

(** Tokenize using BPE *)
let tokenize_bpe (model : bpe_model) text =
  let config =
    Bpe.
      {
        vocab = model.vocab;
        merges = model.merges;
        cache_capacity = model.cache_capacity;
        dropout = model.dropout;
        unk_token = model.unk_token;
        continuing_subword_prefix = model.continuing_subword_prefix;
        end_of_word_suffix = model.end_of_word_suffix;
        fuse_unk = model.fuse_unk;
        byte_fallback = model.byte_fallback;
        ignore_merges = false;
      }
  in
  Bpe.tokenize (Bpe.create config) text |> List.map convert_bpe_token

(** Tokenize using WordPiece *)
let tokenize_wordpiece (model : wordpiece_model) text =
  let config =
    Wordpiece.
      {
        vocab = model.vocab;
        unk_token = model.unk_token;
        max_input_chars_per_word = model.max_input_chars_per_word;
        continuing_subword_prefix = "##";
      }
  in
  Wordpiece.tokenize (Wordpiece.create config) text
  |> List.map convert_wordpiece_token

(** Tokenize using WordLevel *)
let tokenize_wordlevel vocab unk_token text =
  if String.length text = 0 then []
  else if Hashtbl.length vocab = 0 then
    (* Check if this is the special chars() model *)
    if unk_token = "" then (
      (* Character-level tokenization *)
      let chars = ref [] in
      let offset = ref 0 in
      String.iter
        (fun c ->
          let char_str = String.make 1 c in
          let id = Char.code c in
          (* Use ASCII code as ID *)
          chars :=
            { id; value = char_str; offsets = (!offset, !offset + 1) } :: !chars;
          incr offset)
        text;
      List.rev !chars)
    else
      (* No vocabulary - cannot tokenize *)
      []
  else
    (* Try to find the text in vocab, otherwise use unk_token *)
    let id =
      try Hashtbl.find vocab text
      with Not_found -> (
        try Hashtbl.find vocab unk_token
        with Not_found -> 0 (* Default to 0 if no unk_token *))
    in
    [ { id; value = text; offsets = (0, String.length text) } ]

(** Tokenize using Unigram *)
let tokenize_unigram (model : unigram_model) text =
  (* Simple greedy longest-match-first over the provided vocab tokens. *)
  let vocab_tbl = Hashtbl.create (List.length model.vocab) in
  List.iteri (fun i (tok, _) -> Hashtbl.add vocab_tbl tok i) model.vocab;
  let len = String.length text in
  let rec consume pos acc =
    if pos >= len then List.rev acc
    else if text.[pos] = ' ' || text.[pos] = '\n' || text.[pos] = '\t' then
      consume (pos + 1) acc
    else
      (* try longest match *)
      let best = ref None in
      for l = len - pos downto 1 do
        let s = String.sub text pos l in
        match Hashtbl.find_opt vocab_tbl s with
        | Some id ->
            best := Some (id, s, (pos, pos + l));
            (* first (longest) match wins *)
            raise Exit
        | None -> ()
      done;
      match !best with
      | Some (id, s, off) ->
          let tok = { id; value = s; offsets = off } in
          consume (snd off) (tok :: acc)
      | None ->
          (* fallback: single char *)
          let s = String.sub text pos 1 in
          let id = Hashtbl.hash s mod max 1 (List.length model.vocab + 1) in
          let tok = { id; value = s; offsets = (pos, pos + 1) } in
          consume (pos + 1) (tok :: acc)
  in
  try consume 0 [] with Exit -> []

(** Main tokenize function *)
let tokenize model text =
  match model with
  | BPE bpe -> tokenize_bpe bpe text
  | WordPiece wp -> tokenize_wordpiece wp text
  | WordLevel wl -> tokenize_wordlevel wl.vocab wl.unk_token text
  | Unigram ug -> tokenize_unigram ug text

(** Get token ID *)
let token_to_id model token =
  match model with
  | BPE { vocab; _ } -> (
      try Some (Hashtbl.find vocab token) with Not_found -> None)
  | WordPiece { vocab; _ } -> (
      try Some (Hashtbl.find vocab token) with Not_found -> None)
  | WordLevel { vocab; _ } -> (
      try Some (Hashtbl.find vocab token) with Not_found -> None)
  | Unigram { token_map; _ } -> (
      try Some (Hashtbl.find token_map token) with Not_found -> None)

(** Get token from ID *)
let id_to_token model id =
  match model with
  | BPE { vocab; _ } ->
      Hashtbl.fold
        (fun token tid acc -> if tid = id then Some token else acc)
        vocab None
  | WordPiece { vocab; _ } ->
      Hashtbl.fold
        (fun token tid acc -> if tid = id then Some token else acc)
        vocab None
  | WordLevel { vocab; unk_token; _ } ->
      (* Check if this is a chars() model *)
      if Hashtbl.length vocab = 0 && unk_token = "" then
        (* Character-level model - convert ID back to char *)
        if id >= 0 && id <= 255 then Some (String.make 1 (Char.chr id))
        else None
      else
        Hashtbl.fold
          (fun token tid acc -> if tid = id then Some token else acc)
          vocab None
  | Unigram { tokens; _ } ->
      if id >= 0 && id < Array.length tokens then Some tokens.(id) else None

(** Get vocabulary *)
let get_vocab model =
  match model with
  | BPE { vocab; _ } ->
      Hashtbl.fold (fun token id acc -> (token, id) :: acc) vocab []
  | WordPiece { vocab; _ } ->
      Hashtbl.fold (fun token id acc -> (token, id) :: acc) vocab []
  | WordLevel { vocab; _ } ->
      Hashtbl.fold (fun token id acc -> (token, id) :: acc) vocab []
  | Unigram { vocab; _ } -> List.mapi (fun i (token, _) -> (token, i)) vocab

(** Get vocabulary size *)
let get_vocab_size model =
  match model with
  | BPE { vocab; _ } -> Hashtbl.length vocab
  | WordPiece { vocab; _ } -> Hashtbl.length vocab
  | WordLevel { vocab; _ } -> Hashtbl.length vocab
  | Unigram { tokens; _ } -> Array.length tokens

(** Save model *)
let save model ~folder ?(prefix = "") () =
  let path = folder in
  (* For compatibility with existing code *)
  let _ = prefix in
  match model with
  | BPE bpe ->
      let config =
        Bpe.
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
      in
      let _ = Bpe.save (Bpe.create config) ~path () in
      [ "vocab.json"; "merges.txt" ]
      (* Return list of created files *)
  | WordPiece wp ->
      let config =
        Wordpiece.
          {
            vocab = wp.vocab;
            unk_token = wp.unk_token;
            max_input_chars_per_word = wp.max_input_chars_per_word;
            continuing_subword_prefix = "##";
          }
      in
      let _ = Wordpiece.save (Wordpiece.create config) ~path () in
      [ "vocab.txt" ]
      (* Return list of created files *)
  | _ -> [] (* TODO: Implement for other models *)

(** Constructors *)
let bpe ?(vocab = []) ?(merges = []) ?(cache_capacity = 10000) ?dropout
    ?unk_token ?continuing_subword_prefix ?end_of_word_suffix
    ?(fuse_unk = false) ?(byte_fallback = false) ?(ignore_merges = false) () =
  let _ = ignore_merges in
  (* Not used for now *)
  let vocab_tbl = Hashtbl.create (max 1000 (List.length vocab)) in
  List.iter (fun (token, id) -> Hashtbl.add vocab_tbl token id) vocab;
  BPE
    {
      vocab = vocab_tbl;
      merges;
      cache_capacity;
      dropout;
      unk_token;
      continuing_subword_prefix;
      end_of_word_suffix;
      fuse_unk;
      byte_fallback;
    }

let wordpiece ?(vocab = []) ?(unk_token = "[UNK]")
    ?(continuing_subword_prefix = "##") ?(max_input_chars_per_word = 100) () =
  let _ = continuing_subword_prefix in
  (* Used by WordPiece internally *)
  let vocab_tbl = Hashtbl.create (max 1000 (List.length vocab)) in
  List.iter (fun (token, id) -> Hashtbl.add vocab_tbl token id) vocab;
  WordPiece { vocab = vocab_tbl; unk_token; max_input_chars_per_word }

let word_level ?(vocab = []) ?(unk_token = "<unk>") () =
  let vocab_tbl = Hashtbl.create (List.length vocab) in
  List.iter (fun (token, id) -> Hashtbl.add vocab_tbl token id) vocab;
  WordLevel { vocab = vocab_tbl; unk_token }

let build_unigram_lookup vocab =
  let token_map = Hashtbl.create (List.length vocab) in
  List.iteri (fun i (token, _score) -> Hashtbl.replace token_map token i) vocab;
  let tokens = Array.of_list (List.map fst vocab) in
  (token_map, tokens)

let unigram ?(vocab = []) ?(unk_token = "<unk>") ?(byte_fallback = false)
    ?(max_piece_length = 16) ?(n_sub_iterations = 2) ?(shrinking_factor = 0.75)
    () =
  let _ =
    ( unk_token,
      byte_fallback,
      max_piece_length,
      n_sub_iterations,
      shrinking_factor )
  in
  let token_map, tokens = build_unigram_lookup vocab in

  Unigram { vocab; token_map; tokens }

let chars () =
  (* Character-level tokenization - create a special marker *)
  (* We'll handle this as a special case in tokenize *)
  WordLevel { vocab = Hashtbl.create 256; unk_token = "" }

let regex _pattern =
  (* Regex-based tokenization not implemented yet *)
  failwith "Regex tokenization not yet implemented"

let from_file ~vocab ?merges () =
  let _ = merges in
  (* Load vocab file and create appropriate model *)
  (* For now, create a simple word-level model *)
  let vocab_list =
    try
      let ic = open_in vocab in
      let rec read_lines acc =
        try
          let line = input_line ic in
          let parts = String.split_on_char '\t' line in
          match parts with
          | [ token; id ] -> read_lines ((token, int_of_string id) :: acc)
          | _ -> read_lines acc
        with End_of_file ->
          close_in ic;
          List.rev acc
      in
      read_lines []
    with _ -> []
  in
  word_level ~vocab:vocab_list ()

(** Add tokens to model's vocabulary *)
let add_tokens model tokens =
  match model with
  | BPE { vocab; _ } ->
      let start_id = Hashtbl.length vocab in
      let count = ref 0 in
      List.iteri
        (fun i token ->
          if not (Hashtbl.mem vocab token) then (
            Hashtbl.add vocab token (start_id + i);
            incr count))
        tokens;
      !count
  | WordPiece { vocab; _ } ->
      let start_id = Hashtbl.length vocab in
      let count = ref 0 in
      List.iteri
        (fun i token ->
          if not (Hashtbl.mem vocab token) then (
            Hashtbl.add vocab token (start_id + i);
            incr count))
        tokens;
      !count
  | WordLevel { vocab; _ } ->
      let start_id = Hashtbl.length vocab in
      let count = ref 0 in
      List.iteri
        (fun i token ->
          if not (Hashtbl.mem vocab token) then (
            Hashtbl.add vocab token (start_id + i);
            incr count))
        tokens;
      !count
  | Unigram _ ->
      (* For Unigram, vocabulary is immutable list of (string * float) *)
      (* Would need to change the model structure to support mutable vocab *)
      (* For now, just return the number of tokens that would be added *)
      List.length tokens

(** Serialization *)
let to_json = function
  | BPE bpe ->
      `Assoc
        [
          ("type", `String "BPE");
          ( "dropout",
            match bpe.dropout with None -> `Null | Some d -> `Float d );
          ( "unk_token",
            match bpe.unk_token with None -> `Null | Some s -> `String s );
          ( "continuing_subword_prefix",
            match bpe.continuing_subword_prefix with
            | None -> `Null
            | Some s -> `String s );
          ( "end_of_word_suffix",
            match bpe.end_of_word_suffix with
            | None -> `Null
            | Some s -> `String s );
          ("fuse_unk", `Bool bpe.fuse_unk);
          ("byte_fallback", `Bool bpe.byte_fallback);
          (* Include vocab *)
          ( "vocab",
            `Assoc
              (Hashtbl.fold
                 (fun token id acc -> (token, `Int id) :: acc)
                 bpe.vocab []) );
          (* merges would be in separate files *)
        ]
  | WordPiece wp ->
      `Assoc
        [
          ("type", `String "WordPiece");
          ("unk_token", `String wp.unk_token);
          ("max_input_chars_per_word", `Int wp.max_input_chars_per_word);
          ( "vocab",
            `Assoc
              (Hashtbl.fold
                 (fun token id acc -> (token, `Int id) :: acc)
                 wp.vocab []) );
        ]
  | WordLevel wl ->
      `Assoc
        [
          ("type", `String "WordLevel");
          ("unk_token", `String wl.unk_token);
          ( "vocab",
            `Assoc
              (Hashtbl.fold
                 (fun token id acc -> (token, `Int id) :: acc)
                 wl.vocab []) );
        ]
  | Unigram _ -> `Assoc [ ("type", `String "Unigram") ]

let of_json = function
  | `Assoc fields -> (
      match List.assoc_opt "type" fields with
      | Some (`String "BPE") ->
          let dropout =
            match List.assoc_opt "dropout" fields with
            | Some (`Float d) -> Some d
            | _ -> None
          in
          let unk_token =
            match List.assoc_opt "unk_token" fields with
            | Some (`String s) -> Some s
            | _ -> None
          in
          let continuing_subword_prefix =
            match List.assoc_opt "continuing_subword_prefix" fields with
            | Some (`String s) -> Some s
            | _ -> None
          in
          let end_of_word_suffix =
            match List.assoc_opt "end_of_word_suffix" fields with
            | Some (`String s) -> Some s
            | _ -> None
          in
          let fuse_unk =
            match List.assoc_opt "fuse_unk" fields with
            | Some (`Bool b) -> b
            | _ -> false
          in
          let byte_fallback =
            match List.assoc_opt "byte_fallback" fields with
            | Some (`Bool b) -> b
            | _ -> false
          in
          let vocab =
            match List.assoc_opt "vocab" fields with
            | Some (`Assoc vocab_list) ->
                List.map
                  (fun (token, id_json) ->
                    match id_json with `Int id -> (token, id) | _ -> (token, 0))
                  vocab_list
            | _ -> []
          in
          bpe ~vocab ?dropout ?unk_token ?continuing_subword_prefix
            ?end_of_word_suffix ~fuse_unk ~byte_fallback ()
      | Some (`String "WordPiece") ->
          let unk_token =
            match List.assoc_opt "unk_token" fields with
            | Some (`String s) -> s
            | _ -> "[UNK]"
          in
          let max_input_chars_per_word =
            match List.assoc_opt "max_input_chars_per_word" fields with
            | Some (`Int i) -> i
            | _ -> 100
          in
          let vocab =
            match List.assoc_opt "vocab" fields with
            | Some (`Assoc vocab_list) ->
                List.map
                  (fun (token, id_json) ->
                    match id_json with `Int id -> (token, id) | _ -> (token, 0))
                  vocab_list
            | _ -> []
          in
          wordpiece ~vocab ~unk_token ~max_input_chars_per_word ()
      | Some (`String "WordLevel") ->
          let unk_token =
            match List.assoc_opt "unk_token" fields with
            | Some (`String s) -> s
            | _ -> "<unk>"
          in
          let vocab =
            match List.assoc_opt "vocab" fields with
            | Some (`Assoc vocab_list) ->
                List.map
                  (fun (token, id_json) ->
                    match id_json with `Int id -> (token, id) | _ -> (token, 0))
                  vocab_list
            | _ -> []
          in
          word_level ~vocab ~unk_token ()
      | Some (`String "Unigram") -> unigram ()
      | _ -> failwith "Unknown model type")
  | _ -> failwith "Invalid model JSON"
