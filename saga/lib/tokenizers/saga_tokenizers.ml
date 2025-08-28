(** Tokenizers library implementation *)

module Bpe = Bpe
module Wordpiece = Wordpiece
module Unicode = Unicode
module Pre_tokenizers = Pre_tokenizers

type vocab = {
  token_to_idx : (string, int) Hashtbl.t;
  idx_to_token : (int, string) Hashtbl.t;
  mutable size : int;
}

type tokenizer_method = [ `Words | `Chars | `Regex of string ]

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
    Nx_core.Error.invalid ~op:"tokenize"
      ~what:(Printf.sprintf "regex pattern '%s'" pattern)
      ~reason:"invalid regex pattern" ()

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

  let add_batch t tokens = List.iter (add t) tokens
  let get_index t token = Hashtbl.find_opt t.token_to_idx token
  let get_token t idx = Hashtbl.find_opt t.idx_to_token idx
  let size t = t.size
  let pad_idx t = Hashtbl.find t.token_to_idx pad_token
  let unk_idx t = Hashtbl.find t.token_to_idx unk_token
  let bos_idx t = Hashtbl.find t.token_to_idx bos_token
  let eos_idx t = Hashtbl.find t.token_to_idx eos_token

  let from_tokens ?(max_size = max_int) ?(min_freq = 1) tokens =
    if min_freq < 1 then
      Nx_core.Error.invalid ~op:"vocab"
        ~what:(Printf.sprintf "min_freq %d" min_freq)
        ~reason:"must be >= 1" ();
    let freq_table = Hashtbl.create 1024 in
    List.iter
      (fun token ->
        let count =
          Option.value (Hashtbl.find_opt freq_table token) ~default:0
        in
        Hashtbl.replace freq_table token (count + 1))
      tokens;

    let vocab = create () in

    (* Sort by frequency *)
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
          add vocab token;
          add_tokens rest (n - 1)
    in

    add_tokens sorted_tokens (max_size - 4);
    (* Reserve 4 for special tokens *)
    if size vocab > max_size then
      Nx_core.Error.invalid ~op:"vocab"
        ~what:(Printf.sprintf "vocab size %d" (size vocab))
        ~reason:(Printf.sprintf "exceeds maximum %d" max_size)
        ();
    vocab
end

let tokenize ?(method_ = `Words) text =
  match method_ with
  | `Words -> tokenize_words text
  | `Chars -> tokenize_chars text
  | `Regex pattern -> tokenize_regex pattern text

let vocab = Vocab.from_tokens
let vocab_size = Vocab.size

let encode ?vocab text =
  let tokens = tokenize text in
  let vocab =
    match vocab with Some v -> v | None -> Vocab.from_tokens tokens
  in
  List.map
    (fun token ->
      match Vocab.get_index vocab token with
      | Some idx -> idx
      | None -> Vocab.unk_idx vocab)
    tokens

let encode_batch ?vocab ?(max_len = 512) ?(pad = true) texts =
  (* Build vocab from all texts if not provided *)
  let vocab =
    match vocab with
    | Some v -> v
    | None ->
        let all_tokens = List.concat_map tokenize texts in
        Vocab.from_tokens all_tokens
  in

  (* Encode each text *)
  let encoded = List.map (encode ~vocab) texts in
  let batch_size = List.length texts in

  let actual_max_len =
    if pad then max_len
    else min max_len (List.fold_left max 0 (List.map List.length encoded))
  in

  (* Check for overflow *)
  List.iter
    (fun seq ->
      if List.length seq > actual_max_len then
        Nx_core.Error.cannot ~op:"encode_batch" ~what:"encode sequence"
          ~from:(Printf.sprintf "length %d" (List.length seq))
          ~to_:(Printf.sprintf "max_length %d" actual_max_len)
          ~hint:"increase max_length or truncate input" ())
    encoded;

  let arr = Nx.zeros Nx.int32 [| batch_size; actual_max_len |] in
  let pad_idx = Vocab.pad_idx vocab in

  (* Fill with padding token if padding enabled *)
  if pad then ignore (Nx.fill (Int32.of_int pad_idx) arr);

  (* Copy encoded sequences *)
  List.iteri
    (fun i seq ->
      let seq_len = min (List.length seq) actual_max_len in
      List.iteri
        (fun j idx ->
          if j < seq_len then Nx.set_item [ i; j ] (Int32.of_int idx) arr)
        seq)
    encoded;

  arr

let decode vocab indices =
  let tokens = List.filter_map (Vocab.get_token vocab) indices in
  String.concat " " tokens

let decode_batch vocab tensor =
  let shape = Nx.shape tensor in
  match Array.to_list shape with
  | [ batch_size; seq_len ] ->
      let results = ref [] in
      for i = 0 to batch_size - 1 do
        let indices = ref [] in
        for j = 0 to seq_len - 1 do
          let idx = Nx.get_item [ i; j ] tensor |> Int32.to_int in
          if idx <> Vocab.pad_idx vocab then indices := idx :: !indices
        done;
        let text = decode vocab (List.rev !indices) in
        results := text :: !results
      done;
      List.rev !results
  | _ ->
      Nx_core.Error.invalid ~op:"decode_batch" ~what:"tensor shape"
        ~reason:"expected 2D tensor [batch_size; seq_len]" ()

let normalize ?(lowercase = false) ?(strip_accents = false)
    ?(collapse_whitespace = false) text =
  let text = if lowercase then Unicode.case_fold text else text in
  let text = if strip_accents then Unicode.strip_accents text else text in
  let text =
    if collapse_whitespace then
      Unicode.clean_text ~remove_control:false ~normalize_whitespace:true text
    else text
  in
  text

let vocab_save vocab path =
  try
    let oc = open_out path in
    for i = 0 to Vocab.size vocab - 1 do
      match Vocab.get_token vocab i with
      | Some token -> Printf.fprintf oc "%s\n" token
      | None -> ()
    done;
    close_out oc
  with Sys_error msg ->
    Nx_core.Error.failed ~op:"vocab_save"
      ~what:(Printf.sprintf "save to '%s'" path)
      ~reason:msg ()

let vocab_load path =
  try
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
  with Sys_error _ ->
    Nx_core.Error.failed ~op:"vocab_load"
      ~what:(Printf.sprintf "load vocab from '%s'" path)
      ~reason:"file not found" ()

module Tokenizer = struct
  type 'a t = {
    tokenize : string -> string list;
    normalizer : (string -> string) option;
    pre_tokenizer : (string -> string list) option;
  }

  let words =
    { tokenize = tokenize_words; normalizer = None; pre_tokenizer = None }

  let chars =
    { tokenize = tokenize_chars; normalizer = None; pre_tokenizer = None }

  let regex pattern =
    {
      tokenize = tokenize_regex pattern;
      normalizer = None;
      pre_tokenizer = None;
    }

  let bpe ~vocab ~merges =
    let bpe_model = Bpe.from_files ~vocab_file:vocab ~merges_file:merges in
    {
      tokenize =
        (fun text ->
          Bpe.tokenize bpe_model text |> List.map (fun t -> t.Bpe.value));
      normalizer = None;
      pre_tokenizer = None;
    }

  let wordpiece ~vocab ~unk_token =
    let wp_model =
      let v = Wordpiece.read_file ~vocab_file:vocab in
      Wordpiece.create
        {
          vocab = v;
          unk_token;
          continuing_subword_prefix = "##";
          max_input_chars_per_word = 100;
        }
    in
    {
      tokenize =
        (fun text ->
          Wordpiece.tokenize wp_model text
          |> List.map (fun t -> t.Wordpiece.value));
      normalizer = None;
      pre_tokenizer = None;
    }

  let run t text =
    let text = match t.normalizer with Some f -> f text | None -> text in
    match t.pre_tokenizer with
    | Some pre -> List.concat_map t.tokenize (pre text)
    | None -> t.tokenize text

  let run_with_offsets t text =
    (* Simplified version - would need more work for full offset tracking *)
    let tokens = run t text in
    let rec find_offsets tokens text pos acc =
      match tokens with
      | [] -> List.rev acc
      | tok :: rest -> (
          match String.index_from_opt text pos tok.[0] with
          | None -> find_offsets rest text pos acc
          | Some start ->
              let end_ = start + String.length tok in
              find_offsets rest text end_ ((tok, start, end_) :: acc))
    in
    find_offsets tokens text 0 []

  let with_normalizer f t = { t with normalizer = Some f }
  let with_pre_tokenizer f t = { t with pre_tokenizer = Some f }
end
