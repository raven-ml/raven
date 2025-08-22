(** Saga implementation *)

module Tokenizers = Saga_tokenizers
module Models = Saga_models

(** {1 Core Types} *)

type vocab = {
  token_to_idx : (string, int) Hashtbl.t;
  idx_to_token : (int, string) Hashtbl.t;
  mutable size : int;
}

type tokenizer_impl =
  | BPE of Tokenizers.Bpe.t * vocab ref
  | WordPiece of Tokenizers.Wordpiece.t * vocab ref
  | Words of vocab ref
  | Chars of vocab ref
  | Regex of string * vocab ref

type tokenizer = {
  impl : tokenizer_impl;
  normalizer : (string -> string) option;
  pre_tokenizer : (string -> string list) option;
}

(** {1 Internal helpers} *)

let tokenize_words text =
  let tokens = ref [] in
  let start = ref 0 in
  let in_token = ref false in
  let len = String.length text in

  let is_word_char c =
    match c with
    | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '\'' | '-' -> true
    | _ -> false
  in

  for i = 0 to len do
    if i = len || not (is_word_char text.[i]) then (
      if !in_token then (
        let token = String.sub text !start (i - !start) in
        tokens := token :: !tokens;
        in_token := false);
      (* Also capture punctuation as separate tokens *)
      if
        i < len
        && not
             (text.[i] = ' '
             || text.[i] = '\t'
             || text.[i] = '\n'
             || text.[i] = '\r')
      then tokens := String.make 1 text.[i] :: !tokens)
    else if not !in_token then (
      start := i;
      in_token := true)
  done;
  List.rev !tokens

let tokenize_chars text =
  let decoder = Uutf.decoder (`String text) in
  let tokens = ref [] in
  let rec loop () =
    match Uutf.decode decoder with
    | `Uchar u ->
        let buf = Buffer.create 4 in
        Uutf.Buffer.add_utf_8 buf u;
        tokens := Buffer.contents buf :: !tokens;
        loop ()
    | `End -> ()
    | `Malformed _ -> loop ()
    | `Await -> assert false
  in
  loop ();
  List.rev !tokens

let tokenize_regex pattern text =
  try
    let re = Re.Perl.re pattern |> Re.compile in
    Re.all re text |> List.map (fun g -> Re.Group.get g 0)
  with Re.Perl.Parse_error ->
    failwith (Printf.sprintf "Invalid regex pattern: %s" pattern)

(** {1 Vocabulary} *)

module Vocab = struct
  type t = vocab

  let pad_token = "<pad>"
  let unk_token = "<unk>"
  let bos_token = "<bos>"
  let eos_token = "<eos>"

  let create () =
    let v =
      {
        token_to_idx = Hashtbl.create 1024;
        idx_to_token = Hashtbl.create 1024;
        size = 0;
      }
    in
    (* Add special tokens *)
    List.iter
      (fun token ->
        Hashtbl.add v.token_to_idx token v.size;
        Hashtbl.add v.idx_to_token v.size token;
        v.size <- v.size + 1)
      [ pad_token; unk_token; bos_token; eos_token ];
    v

  let add t token =
    if not (Hashtbl.mem t.token_to_idx token) then (
      Hashtbl.add t.token_to_idx token t.size;
      Hashtbl.add t.idx_to_token t.size token;
      t.size <- t.size + 1)

  let add_tokens t tokens = List.iter (add t) tokens
  let token_to_id t token = Hashtbl.find_opt t.token_to_idx token
  let id_to_token t idx = Hashtbl.find_opt t.idx_to_token idx
  let size t = t.size
  let pad_id t = Hashtbl.find t.token_to_idx pad_token
  let unk_id t = Hashtbl.find t.token_to_idx unk_token
  let bos_id t = Hashtbl.find t.token_to_idx bos_token
  let eos_id t = Hashtbl.find t.token_to_idx eos_token
end

(** {1 Tokenizer implementation} *)

let get_or_build_vocab tokenizer_impl tokens =
  match tokenizer_impl with
  | BPE (_, vocab_ref)
  | WordPiece (_, vocab_ref)
  | Words vocab_ref
  | Chars vocab_ref
  | Regex (_, vocab_ref) -> (
      match !vocab_ref with
      | v when Vocab.size v > 4 ->
          v (* Already has tokens beyond special ones *)
      | _ ->
          let v = Vocab.create () in
          Vocab.add_tokens v tokens;
          vocab_ref := v;
          v)

let tokenizer config =
  let impl =
    match config with
    | `BPE (vocab_file, merges_file) ->
        let bpe = Tokenizers.Bpe.from_files ~vocab_file ~merges_file in
        BPE (bpe, ref (Vocab.create ()))
    | `WordPiece (vocab_file, unk_token) ->
        let vocab = Tokenizers.Wordpiece.read_file ~vocab_file in
        let wp =
          Tokenizers.Wordpiece.create
            {
              vocab;
              unk_token;
              continuing_subword_prefix = "##";
              max_input_chars_per_word = 100;
            }
        in
        WordPiece (wp, ref (Vocab.create ()))
    | `Words -> Words (ref (Vocab.create ()))
    | `Chars -> Chars (ref (Vocab.create ()))
    | `Regex pattern -> Regex (pattern, ref (Vocab.create ()))
  in
  { impl; normalizer = None; pre_tokenizer = None }

let tokenize_with tok text =
  let text = match tok.normalizer with Some f -> f text | None -> text in
  let texts =
    match tok.pre_tokenizer with Some f -> f text | None -> [ text ]
  in
  let tokens =
    List.concat_map
      (fun text ->
        match tok.impl with
        | BPE (bpe, _) ->
            Tokenizers.Bpe.tokenize bpe text
            |> List.map (fun t -> t.Tokenizers.Bpe.value)
        | WordPiece (wp, _) ->
            Tokenizers.Wordpiece.tokenize wp text
            |> List.map (fun t -> t.Tokenizers.Wordpiece.value)
        | Words _ -> tokenize_words text
        | Chars _ -> tokenize_chars text
        | Regex (pattern, _) -> tokenize_regex pattern text)
      texts
  in
  tokens

let encode tokenizer text =
  let tokens = tokenize_with tokenizer text in
  let vocab = get_or_build_vocab tokenizer.impl tokens in
  let unk_id = Vocab.unk_id vocab in
  Array.of_list
    (List.map
       (fun token ->
         match Vocab.token_to_id vocab token with
         | Some id -> id
         | None -> unk_id)
       tokens)

let decode tokenizer ids =
  let vocab =
    match tokenizer.impl with
    | BPE (_, vref)
    | WordPiece (_, vref)
    | Words vref
    | Chars vref
    | Regex (_, vref) ->
        !vref
  in
  let tokens = Array.to_list ids |> List.filter_map (Vocab.id_to_token vocab) in
  String.concat " " tokens

let encode_batch tokenizer ?(max_length = 512) ?(padding = true)
    ?(truncation = false) texts =
  let encoded = List.map (encode tokenizer) texts in
  let batch_size = List.length texts in

  let actual_max_len =
    if padding then max_length
    else min max_length (List.fold_left max 0 (List.map Array.length encoded))
  in

  let vocab =
    match tokenizer.impl with
    | BPE (_, vref)
    | WordPiece (_, vref)
    | Words vref
    | Chars vref
    | Regex (_, vref) ->
        !vref
  in
  let pad_id = Vocab.pad_id vocab in

  let arr = Nx.zeros Nx.int32 [| batch_size; actual_max_len |] in
  if padding then ignore (Nx.fill (Int32.of_int pad_id) arr);

  List.iteri
    (fun i seq ->
      let seq_len =
        if truncation then min (Array.length seq) actual_max_len
        else Array.length seq
      in
      if seq_len > actual_max_len then
        failwith
          (Printf.sprintf "Sequence length %d exceeds max_length %d" seq_len
             actual_max_len);
      for j = 0 to seq_len - 1 do
        Nx.set_item [ i; j ] (Int32.of_int seq.(j)) arr
      done)
    encoded;
  arr

(** {1 Text Processing} *)

let normalize ?(lowercase = false) ?(strip_accents = false)
    ?(clean_whitespace = false) text =
  let text = if lowercase then Tokenizers.Unicode.case_fold text else text in
  let text =
    if strip_accents then Tokenizers.Unicode.strip_accents text else text
  in
  let text =
    if clean_whitespace then
      Tokenizers.Unicode.clean_text ~remove_control:false
        ~normalize_whitespace:true text
    else text
  in
  text

let tokenize text = tokenize_words text

let split_sentences text =
  (* Simple sentence splitting - could be improved *)
  let sentences = Re.Str.split (Re.Str.regexp "[.!?][ \n\t]+") text in
  List.filter (fun s -> String.length (String.trim s) > 0) sentences

(** {1 Vocabulary management} *)

let build_vocab ?(max_size = 50000) ?(min_freq = 1) tokenizer texts =
  let all_tokens = List.concat_map (tokenize_with tokenizer) texts in
  let freq_table = Hashtbl.create 1024 in
  List.iter
    (fun token ->
      let count = Option.value (Hashtbl.find_opt freq_table token) ~default:0 in
      Hashtbl.replace freq_table token (count + 1))
    all_tokens;

  let vocab = Vocab.create () in

  let sorted_tokens =
    Hashtbl.fold (fun token count acc -> (token, count) :: acc) freq_table []
    |> List.filter (fun (_, count) -> count >= min_freq)
    |> List.sort (fun (_, c1) (_, c2) -> compare c2 c1)
    |> List.map fst
  in

  let rec add_tokens tokens remaining =
    match (tokens, remaining) with
    | _, 0 -> ()
    | [], _ -> ()
    | token :: rest, n ->
        Vocab.add vocab token;
        add_tokens rest (n - 1)
  in

  add_tokens sorted_tokens (max_size - 4);

  (* Reserve 4 for special tokens *)

  (* Update tokenizer's vocab reference *)
  (match tokenizer.impl with
  | BPE (_, vref)
  | WordPiece (_, vref)
  | Words vref
  | Chars vref
  | Regex (_, vref) ->
      vref := vocab);

  vocab

let vocab_size = Vocab.size

let save_vocab vocab path =
  let oc = open_out path in
  for i = 0 to Vocab.size vocab - 1 do
    match Vocab.id_to_token vocab i with
    | Some token -> Printf.fprintf oc "%s\n" token
    | None -> ()
  done;
  close_out oc

let load_vocab path =
  let ic = open_in path in
  let vocab = Vocab.create () in
  try
    while true do
      let token = input_line ic in
      Vocab.add vocab token
    done;
    vocab
  with End_of_file ->
    close_in ic;
    vocab

(** {1 Language Models} *)

module LM = struct
  type t = Ngram of int * Models.Ngram.t * vocab

  let train_ngram ~n ?(smoothing = 1.0) tokenizer texts =
    let vocab = build_vocab tokenizer texts in
    let all_tokens =
      List.concat_map (fun text -> Array.to_list (encode tokenizer text)) texts
    in
    let model = Models.Ngram.create ~n ~smoothing (Array.of_list all_tokens) in
    Ngram (n, model, vocab)

  let generate model ?(max_tokens = 100) ?(temperature = 1.0) ?top_k
      ?(prompt = "") tokenizer =
    match model with
    | Ngram (_n, ngram_model, _vocab) ->
        let prompt_ids =
          if prompt = "" then [||] else encode tokenizer prompt
        in
        let _top_k = top_k in
        (* TODO: implement top_k in ngram *)
        let generated_ids =
          Models.Ngram.generate ngram_model ~max_tokens ~temperature ?seed:None
            ~start:prompt_ids ()
        in
        decode tokenizer generated_ids

  let perplexity model tokenizer text =
    match model with
    | Ngram (_, ngram_model, _) ->
        let ids = encode tokenizer text in
        Models.Ngram.perplexity ngram_model ids

  let save model path =
    match model with
    | Ngram (n, ngram_model, vocab) ->
        (* Save model info *)
        let oc = open_out_bin path in
        output_value oc (n, ngram_model, vocab);
        close_out oc

  let load path =
    let ic = open_in_bin path in
    let n, ngram_model, vocab = input_value ic in
    close_in ic;
    Ngram (n, ngram_model, vocab)
end

(** {1 Advanced Tokenizer} *)

module Tokenizer = struct
  type 'a t = tokenizer

  let create config =
    match config with
    | `BPE bpe ->
        let vocab_ref = ref (Vocab.create ()) in
        { impl = BPE (bpe, vocab_ref); normalizer = None; pre_tokenizer = None }
    | `WordPiece wp ->
        let vocab_ref = ref (Vocab.create ()) in
        {
          impl = WordPiece (wp, vocab_ref);
          normalizer = None;
          pre_tokenizer = None;
        }
    | `Words -> tokenizer `Words
    | `Chars -> tokenizer `Chars
    | `Regex pattern -> tokenizer (`Regex pattern)

  let with_normalizer f tok = { tok with normalizer = Some f }
  let with_pre_tokenizer f tok = { tok with pre_tokenizer = Some f }

  let encode_with_offsets tok text =
    (* Simplified - would need more work for full offset tracking *)
    let _tokens = tokenize_with tok text in
    let ids = encode tok text in
    Array.mapi
      (fun i id -> (id, i * 10, (i + 1) * 10) (* Placeholder offsets *))
      ids
end
