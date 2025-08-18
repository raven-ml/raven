(* Simple tokenizer interface for Gemma *)

type t = {
  vocab : (string, int) Hashtbl.t;
  reverse_vocab : (int, string) Hashtbl.t;
  unk_token_id : int;
  pad_token_id : int;
  bos_token_id : int;
  eos_token_id : int;
}

let create_dummy_tokenizer vocab_size =
  let vocab = Hashtbl.create vocab_size in
  let reverse_vocab = Hashtbl.create vocab_size in

  (* Add special tokens *)
  Hashtbl.add vocab "<pad>" 0;
  Hashtbl.add vocab "<unk>" 1;
  Hashtbl.add vocab "<bos>" 2;
  Hashtbl.add vocab "<eos>" 3;

  Hashtbl.add reverse_vocab 0 "<pad>";
  Hashtbl.add reverse_vocab 1 "<unk>";
  Hashtbl.add reverse_vocab 2 "<bos>";
  Hashtbl.add reverse_vocab 3 "<eos>";

  (* Add dummy tokens *)
  for i = 4 to min 1000 (vocab_size - 1) do
    let token = Printf.sprintf "token_%d" i in
    Hashtbl.add vocab token i;
    Hashtbl.add reverse_vocab i token
  done;

  {
    vocab;
    reverse_vocab;
    unk_token_id = 1;
    pad_token_id = 0;
    bos_token_id = 2;
    eos_token_id = 3;
  }

let encode tokenizer text =
  (* Simple whitespace tokenization for demo *)
  let words = String.split_on_char ' ' text in
  List.map
    (fun word ->
      match Hashtbl.find_opt tokenizer.vocab word with
      | Some id -> id
      | None -> tokenizer.unk_token_id)
    words

let decode tokenizer ids =
  let tokens =
    List.map
      (fun id ->
        match Hashtbl.find_opt tokenizer.reverse_vocab id with
        | Some token -> token
        | None -> "<unk>")
      ids
  in
  String.concat " " tokens

let pad_sequence tokenizer sequence max_length =
  let len = List.length sequence in
  if len >= max_length then List.filteri (fun i _ -> i < max_length) sequence
  else sequence @ List.init (max_length - len) (fun _ -> tokenizer.pad_token_id)
