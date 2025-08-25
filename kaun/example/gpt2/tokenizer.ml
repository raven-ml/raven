(* GPT-2 Tokenizer using BPE *)

(* Download files if not present *)
let download_if_needed url target_path =
  if not (Sys.file_exists target_path) then begin
    Printf.printf "Downloading %s...\n" (Filename.basename target_path);
    flush stdout;
    let cmd = Printf.sprintf "curl -L -o %s '%s'" target_path url in
    let _ = Sys.command cmd in
    Printf.printf "Download complete: %s\n" target_path;
    flush stdout
  end

(* Get GPT-2 tokenizer *)
let get_tokenizer ?(cache_dir="/tmp/gpt2_tokenizer") () =
  (* Ensure cache directory exists *)
  if not (Sys.file_exists cache_dir) then
    Unix.mkdir cache_dir 0o755;
  
  (* File paths *)
  let vocab_file = Filename.concat cache_dir "vocab.json" in
  let merges_file = Filename.concat cache_dir "merges.txt" in
  
  (* Download files if needed *)
  download_if_needed 
    "https://www.dropbox.com/s/s93xkhgcac5nbmn/vocab.json?dl=1" 
    vocab_file;
  download_if_needed 
    "https://www.dropbox.com/s/7f5n1gf348sy1mt/merges.txt?dl=1" 
    merges_file;
  
  (* Create BPE tokenizer *)
  Saga.tokenizer (`BPE (vocab_file, merges_file))

(* Special tokens *)
let eos_token = "<|endoftext|>"
let bos_token = "<|endoftext|>"
let unk_token = "<|endoftext|>"
let pad_token = "<|endoftext|>"

(* Get special token IDs *)
let get_special_token_id tokenizer token =
  let ids = Saga.encode tokenizer token in
  if Array.length ids > 0 then ids.(0) else 50256  (* GPT-2 vocab size *)

(* Encode text to token IDs *)
let encode tokenizer text =
  Saga.encode tokenizer text

(* Decode token IDs back to text *)
let decode tokenizer ids =
  Saga.decode tokenizer ids

(* Batch encoding with padding *)
let encode_batch tokenizer texts ~max_length ~padding =
  let encoded = List.map (encode tokenizer) texts in
  
  (* Find max length or use provided *)
  let actual_max_length = 
    if max_length > 0 then max_length 
    else List.fold_left (fun acc arr -> max acc (Array.length arr)) 0 encoded
  in
  
  (* Pad if needed *)
  if padding then
    let pad_id = get_special_token_id tokenizer pad_token in
    List.map (fun arr ->
      let len = Array.length arr in
      if len < actual_max_length then
        Array.append arr (Array.make (actual_max_length - len) pad_id)
      else if len > actual_max_length then
        Array.sub arr 0 actual_max_length
      else
        arr
    ) encoded
  else
    encoded