(** Byte Pair Encoding implementation *)

module IntPairMap = Map.Make (struct
  type t = int * int

  let compare = compare
end)

module IntPairSet = Set.Make (struct
  type t = int * int

  let compare = compare
end)

module StringMap = Map.Make (String)

type vocab = (string, int) Hashtbl.t
type vocab_r = (int, string) Hashtbl.t
type merges = (string * string) list
type merge_map = (int * int) IntPairMap.t

type symbol = {
  mutable c : int;
  mutable prev : int;
  mutable next : int;
  mutable len : int;
}

type word = { mutable symbols : symbol array; mutable size : int }
type token = { id : int; value : string; offsets : int * int }
type cache_entry = word

type config = {
  vocab : vocab;
  merges : merges;
  cache_capacity : int;
  dropout : float option;
  unk_token : string option;
  continuing_subword_prefix : string option;
  end_of_word_suffix : string option;
  fuse_unk : bool;
  byte_fallback : bool;
  ignore_merges : bool;
}

type t = {
  vocab : vocab;
  vocab_r : vocab_r;
  merges : merge_map;
  cache : (string, cache_entry) Hashtbl.t option;
  dropout : float option;
  unk_token : string option;
  continuing_subword_prefix : string option;
  end_of_word_suffix : string option;
  fuse_unk : bool;
  byte_fallback : bool;
  ignore_merges : bool;
}

let create_word capacity =
  {
    symbols = Array.make capacity { c = -1; prev = -1; next = -1; len = 0 };
    size = 0;
  }

let add_symbol word c byte_len =
  if word.size >= Array.length word.symbols then
    failwith "Word capacity exceeded";

  let prev = if word.size > 0 then word.size - 1 else -1 in
  let symbol = { c; prev; next = -1; len = byte_len } in
  let _ = symbol.prev in
  (* Used for tracking symbol chain *)

  if prev >= 0 then word.symbols.(prev).next <- word.size;

  word.symbols.(word.size) <- symbol;
  word.size <- word.size + 1

let merge_word model text =
  let len = String.length text in
  let word = create_word len in

  let decoder = Uutf.decoder (`String text) in
  let i = ref 0 in
  let pending_unk = ref None in
  (* Track pending unknown token for fusion *)

  let flush_unk () =
    match !pending_unk with
    | Some (unk_id, unk_len) ->
        add_symbol word unk_id unk_len;
        pending_unk := None
    | None -> ()
  in

  let rec process_chars () =
    match Uutf.decode decoder with
    | `Uchar u ->
        let start = !i in
        let char_str =
          let buf = Buffer.create 4 in
          Uutf.Buffer.add_utf_8 buf u;
          Buffer.contents buf
        in
        let byte_len = String.length char_str in
        i := !i + byte_len;

        let is_first = start = 0 in
        let is_last = !i >= len in

        let token_str =
          let s = ref char_str in

          (if not is_first then
             match model.continuing_subword_prefix with
             | Some prefix -> s := prefix ^ !s
             | None -> ());

          (if is_last then
             match model.end_of_word_suffix with
             | Some suffix -> s := !s ^ suffix
             | None -> ());

          !s
        in

        (match Hashtbl.find_opt model.vocab token_str with
        | Some id ->
            flush_unk ();
            (* Flush any pending unknown token *)
            add_symbol word id byte_len
        | None -> (
            if model.byte_fallback then (
              let handled = ref false in
              String.iter
                (fun byte ->
                  let hex = Printf.sprintf "<0x%02X>" (Char.code byte) in
                  match Hashtbl.find_opt model.vocab hex with
                  | Some id ->
                      flush_unk ();
                      add_symbol word id 1;
                      handled := true
                  | None -> ())
                char_str;
              if not !handled then
                match model.unk_token with
                | Some unk -> (
                    match Hashtbl.find_opt model.vocab unk with
                    | Some unk_id ->
                        if model.fuse_unk then
                          (* Fuse with existing unknown token *)
                          pending_unk :=
                            Some
                              (match !pending_unk with
                              | Some (id, len) -> (id, len + byte_len)
                              | None -> (unk_id, byte_len))
                        else (
                          flush_unk ();
                          add_symbol word unk_id byte_len)
                    | None ->
                        failwith
                          (Printf.sprintf "Unknown token '%s' not in vocabulary"
                             unk))
                | None -> ())
            else
              match model.unk_token with
              | Some unk -> (
                  match Hashtbl.find_opt model.vocab unk with
                  | Some unk_id ->
                      if model.fuse_unk then
                        (* Fuse with existing unknown token *)
                        pending_unk :=
                          Some
                            (match !pending_unk with
                            | Some (id, len) -> (id, len + byte_len)
                            | None -> (unk_id, byte_len))
                      else (
                        flush_unk ();
                        add_symbol word unk_id byte_len)
                  | None ->
                      failwith
                        (Printf.sprintf "Unknown token '%s' not in vocabulary"
                           unk))
              | None -> ()));

        process_chars ()
    | `End -> flush_unk () (* Flush any remaining unknown token *)
    | `Malformed _ -> process_chars ()
    | `Await -> assert false
  in

  process_chars ();

  (* Apply merges *)
  (if model.dropout = None || model.dropout = Some 0.0 then
     let rec apply_merges () =
       (* Find the best merge (lowest rank) *)
       let best_merge = ref None in
       let best_rank = ref max_int in
       let best_pos = ref (-1) in

       for i = 0 to word.size - 2 do
         if word.symbols.(i).len > 0 && word.symbols.(i + 1).len > 0 then
           let pair = (word.symbols.(i).c, word.symbols.(i + 1).c) in
           match IntPairMap.find_opt pair model.merges with
           | Some (rank, new_id) when rank < !best_rank ->
               best_merge := Some new_id;
               best_rank := rank;
               best_pos := i
           | _ -> ()
       done;

       match !best_merge with
       | Some new_id ->
           let i = !best_pos in
           word.symbols.(i).c <- new_id;
           word.symbols.(i).len <-
             word.symbols.(i).len + word.symbols.(i + 1).len;
           word.symbols.(i).next <- word.symbols.(i + 1).next;
           word.symbols.(i + 1).len <- 0;
           if word.symbols.(i).next >= 0 && word.symbols.(i).next < word.size
           then word.symbols.(word.symbols.(i).next).prev <- i;

           (* Compact the array *)
           let new_symbols =
             Array.make (Array.length word.symbols) word.symbols.(0)
           in
           let j = ref 0 in
           for k = 0 to word.size - 1 do
             if word.symbols.(k).len > 0 then (
               new_symbols.(!j) <- word.symbols.(k);
               incr j)
           done;
           word.symbols <- new_symbols;
           word.size <- !j;
           apply_merges ()
       | None -> ()
     in
     apply_merges ()
   else
     (* With dropout - randomly skip merges *)
     let apply_merges_with_dropout () =
       (* TODO: Implement dropout version *)
       ()
     in
     apply_merges_with_dropout ());

  word

let word_to_tokens model word =
  let tokens = ref [] in
  let offset = ref 0 in

  for i = 0 to word.size - 1 do
    if word.symbols.(i).len > 0 then (
      let id = word.symbols.(i).c in
      let value =
        match Hashtbl.find_opt model.vocab_r id with
        | Some v -> v
        | None -> "<unk>"
      in
      let start = !offset in
      let end_ = !offset + word.symbols.(i).len in
      tokens := { id; value; offsets = (start, end_) } :: !tokens;
      offset := end_)
  done;

  List.rev !tokens

let tokenize model text =
  if String.length text = 0 then []
  else if model.ignore_merges then
    match Hashtbl.find_opt model.vocab text with
    | Some id -> [ { id; value = text; offsets = (0, String.length text) } ]
    | None -> []
  else
    match model.cache with
    | Some cache when String.length text < 1000 -> (
        match Hashtbl.find_opt cache text with
        | Some word -> word_to_tokens model word
        | None ->
            let word = merge_word model text in
            Hashtbl.add cache text word;
            word_to_tokens model word)
    | _ ->
        let word = merge_word model text in
        word_to_tokens model word

let token_to_id model token = Hashtbl.find_opt model.vocab token
let id_to_token model id = Hashtbl.find_opt model.vocab_r id
let get_vocab model = Hashtbl.fold (fun k v acc -> (k, v) :: acc) model.vocab []
let get_vocab_size model = Hashtbl.length model.vocab

let clear_cache model =
  match model.cache with Some cache -> Hashtbl.clear cache | None -> ()

let resize_cache model _capacity =
  match model.cache with
  | Some cache -> Hashtbl.clear cache
  (* Note: OCaml Hashtbl doesn't have a direct resize, clear is the best we can
     do *)
  | None -> ()

let convert_merges_to_merge_map vocab merges continuing_subword_prefix =
  let prefix_len =
    match continuing_subword_prefix with Some p -> String.length p | None -> 0
  in

  List.mapi
    (fun rank (a, b) ->
      match (Hashtbl.find_opt vocab a, Hashtbl.find_opt vocab b) with
      | Some a_id, Some b_id -> (
          let new_token =
            if prefix_len > 0 && String.length b > prefix_len then
              a ^ String.sub b prefix_len (String.length b - prefix_len)
            else a ^ b
          in
          match Hashtbl.find_opt vocab new_token with
          | Some new_id -> Some ((a_id, b_id), (rank, new_id))
          | None ->
              failwith
                (Printf.sprintf "Merge token '%s' not in vocabulary" new_token))
      | _ -> failwith (Printf.sprintf "Merge tokens not in vocabulary"))
    merges
  |> List.filter_map (fun x -> x)
  |> List.fold_left (fun acc (k, v) -> IntPairMap.add k v acc) IntPairMap.empty

let create (cfg : config) : t =
  let vocab_r = Hashtbl.create (Hashtbl.length cfg.vocab) in
  Hashtbl.iter (fun k v -> Hashtbl.add vocab_r v k) cfg.vocab;

  let cache =
    if cfg.cache_capacity = 0 then None
    else Some (Hashtbl.create cfg.cache_capacity)
  in

  let merges =
    convert_merges_to_merge_map cfg.vocab cfg.merges
      cfg.continuing_subword_prefix
  in

  {
    vocab = cfg.vocab;
    vocab_r;
    merges;
    cache;
    dropout = cfg.dropout;
    unk_token = cfg.unk_token;
    continuing_subword_prefix = cfg.continuing_subword_prefix;
    end_of_word_suffix = cfg.end_of_word_suffix;
    fuse_unk = cfg.fuse_unk;
    byte_fallback = cfg.byte_fallback;
    ignore_merges = cfg.ignore_merges;
  }

let read_files ~vocab_file ~merges_file =
  (* Read vocab.json *)
  let vocab_json =
    let ic = open_in vocab_file in
    let content = really_input_string ic (in_channel_length ic) in
    close_in ic;
    Yojson.Basic.from_string content
  in

  let vocab = Hashtbl.create 1024 in
  (match vocab_json with
  | `Assoc items ->
      List.iter
        (fun (k, v) ->
          match v with
          | `Int id -> Hashtbl.add vocab k id
          | `Float f -> Hashtbl.add vocab k (int_of_float f)
          | _ -> failwith "Invalid vocab format")
        items
  | _ -> failwith "Invalid vocab.json format");

  (* Read merges.txt *)
  let merges =
    let ic = open_in merges_file in
    let merges = ref [] in
    (try
       while true do
         let line = input_line ic in
         if
           (not (String.starts_with ~prefix:"#" line)) && String.length line > 0
         then
           match String.split_on_char ' ' line with
           | [ a; b ] -> merges := (a, b) :: !merges
           | _ -> failwith (Printf.sprintf "Invalid merge line: %s" line)
       done
     with End_of_file -> ());
    close_in ic;
    List.rev !merges
  in

  (vocab, merges)

let from_files ~vocab_file ~merges_file =
  let vocab, merges = read_files ~vocab_file ~merges_file in
  create
    {
      vocab;
      merges;
      cache_capacity = 10000;
      dropout = None;
      unk_token = None;
      continuing_subword_prefix = None;
      end_of_word_suffix = None;
      fuse_unk = false;
      byte_fallback = false;
      ignore_merges = false;
    }

let default () =
  create
    {
      vocab = Hashtbl.create 0;
      merges = [];
      cache_capacity = 10000;
      dropout = None;
      unk_token = None;
      continuing_subword_prefix = None;
      end_of_word_suffix = None;
      fuse_unk = false;
      byte_fallback = false;
      ignore_merges = false;
    }

let save model ~path ?name () =
  let vocab_file =
    match name with
    | Some n -> Filename.concat path (Printf.sprintf "%s-vocab.json" n)
    | None -> Filename.concat path "vocab.json"
  in

  let merges_file =
    match name with
    | Some n -> Filename.concat path (Printf.sprintf "%s-merges.txt" n)
    | None -> Filename.concat path "merges.txt"
  in

  (* Write vocab.json *)
  let vocab_items =
    Hashtbl.fold
      (fun k v acc -> (k, (`Int v : Yojson.Basic.t)) :: acc)
      model.vocab []
    |> List.sort (fun p1 p2 ->
           match (p1, p2) with
           | (_, `Int a), (_, `Int b) -> compare a b
           | _ -> 0)
  in
  let vocab_json : Yojson.Basic.t = `Assoc vocab_items in
  let oc = open_out vocab_file in
  output_string oc (Yojson.Basic.to_string vocab_json);
  close_out oc;

  (* Write merges.txt *)
  let oc = open_out merges_file in
  output_string oc "#version: 0.2\n";

  let merges_list =
    IntPairMap.fold
      (fun (a_id, b_id) (rank, _) acc ->
        match
          ( Hashtbl.find_opt model.vocab_r a_id,
            Hashtbl.find_opt model.vocab_r b_id )
        with
        | Some a, Some b -> (rank, a, b) :: acc
        | _ -> acc)
      model.merges []
    |> List.sort (fun (r1, _, _) (r2, _, _) -> compare r1 r2)
  in

  List.iter (fun (_, a, b) -> Printf.fprintf oc "%s %s\n" a b) merges_list;

  close_out oc

(** Create function needs to be defined before Builder module *)
let create_internal = create

(** Builder module *)
module Builder = struct
  type builder = {
    mutable vocab : vocab;
    mutable merges : merges;
    mutable cache_capacity : int;
    mutable dropout : float option;
    mutable unk_token : string option;
    mutable continuing_subword_prefix : string option;
    mutable end_of_word_suffix : string option;
    mutable fuse_unk : bool;
    mutable byte_fallback : bool;
    mutable ignore_merges : bool;
  }

  let create () =
    {
      vocab = Hashtbl.create 0;
      merges = [];
      cache_capacity = 10000;
      dropout = None;
      unk_token = None;
      continuing_subword_prefix = None;
      end_of_word_suffix = None;
      fuse_unk = false;
      byte_fallback = false;
      ignore_merges = false;
    }

  let vocab_and_merges builder vocab merges =
    builder.vocab <- vocab;
    builder.merges <- merges;
    builder

  let cache_capacity builder capacity =
    builder.cache_capacity <- capacity;
    builder

  let dropout builder p =
    if p < 0.0 || p > 1.0 then failwith "Dropout must be between 0.0 and 1.0";
    builder.dropout <- Some p;
    builder

  let unk_token builder token =
    builder.unk_token <- Some token;
    builder

  let continuing_subword_prefix builder prefix =
    builder.continuing_subword_prefix <- Some prefix;
    builder

  let end_of_word_suffix builder suffix =
    builder.end_of_word_suffix <- Some suffix;
    builder

  let fuse_unk builder fuse =
    builder.fuse_unk <- fuse;
    builder

  let byte_fallback builder fallback =
    builder.byte_fallback <- fallback;
    builder

  let ignore_merges builder ignore =
    builder.ignore_merges <- ignore;
    builder

  let build b =
    create_internal
      {
        vocab = b.vocab;
        merges = b.merges;
        cache_capacity = b.cache_capacity;
        dropout = b.dropout;
        unk_token = b.unk_token;
        continuing_subword_prefix = b.continuing_subword_prefix;
        end_of_word_suffix = b.end_of_word_suffix;
        fuse_unk = b.fuse_unk;
        byte_fallback = b.byte_fallback;
        ignore_merges = b.ignore_merges;
      }
end

(** Trainer module *)
module Trainer = struct
  type word_count = (string, int) Hashtbl.t

  type trainer_config = {
    min_frequency : int;
    vocab_size : int;
    show_progress : bool;
    special_tokens : string list;
    limit_alphabet : int option;
    initial_alphabet : char list;
    continuing_subword_prefix : string option;
    end_of_word_suffix : string option;
    max_token_length : int option;
  }

  type trainer = { config : trainer_config; words : word_count }

  let default_config =
    {
      min_frequency = 0;
      vocab_size = 30000;
      show_progress = true;
      special_tokens = [];
      limit_alphabet = None;
      initial_alphabet = [];
      continuing_subword_prefix = None;
      end_of_word_suffix = None;
      max_token_length = None;
    }

  let create config = { config; words = Hashtbl.create 10000 }

  let feed trainer texts =
    List.iter
      (fun text ->
        (* Simple word tokenization for training *)
        let words = String.split_on_char ' ' text in
        List.iter
          (fun word ->
            if String.length word > 0 then
              Hashtbl.replace trainer.words word
                (1 + try Hashtbl.find trainer.words word with Not_found -> 0))
          words)
      texts

  let compute_pair_counts words =
    let pair_counts = Hashtbl.create 10000 in

    Hashtbl.iter
      (fun word count ->
        let chars =
          let decoder = Uutf.decoder (`String word) in
          let chars = ref [] in
          let rec loop () =
            match Uutf.decode decoder with
            | `Uchar u ->
                let buf = Buffer.create 4 in
                Uutf.Buffer.add_utf_8 buf u;
                chars := Buffer.contents buf :: !chars;
                loop ()
            | `End -> ()
            | _ -> loop ()
          in
          loop ();
          List.rev !chars
        in

        for i = 0 to List.length chars - 2 do
          let a = List.nth chars i in
          let b = List.nth chars (i + 1) in
          let pair = (a, b) in
          Hashtbl.replace pair_counts pair
            (count + try Hashtbl.find pair_counts pair with Not_found -> 0)
        done)
      words;

    pair_counts

  let train trainer _model =
    (* Build initial vocabulary from characters *)
    let vocab = Hashtbl.create 10000 in
    let vocab_size = ref 0 in

    (* Add special tokens *)
    List.iter
      (fun token ->
        Hashtbl.add vocab token !vocab_size;
        incr vocab_size)
      trainer.config.special_tokens;

    (* Add all characters from words *)
    Hashtbl.iter
      (fun word _ ->
        let decoder = Uutf.decoder (`String word) in
        let rec loop () =
          match Uutf.decode decoder with
          | `Uchar u ->
              let buf = Buffer.create 4 in
              Uutf.Buffer.add_utf_8 buf u;
              let char_str = Buffer.contents buf in
              if not (Hashtbl.mem vocab char_str) then (
                Hashtbl.add vocab char_str !vocab_size;
                incr vocab_size);
              loop ()
          | `End -> ()
          | _ -> loop ()
        in
        loop ())
      trainer.words;

    (* Compute merges *)
    let merges = ref [] in
    let words_copy = Hashtbl.copy trainer.words in

    while !vocab_size < trainer.config.vocab_size do
      let pair_counts = compute_pair_counts words_copy in

      (* Find most frequent pair *)
      let best_pair = ref None in
      let best_count = ref 0 in
      Hashtbl.iter
        (fun pair count ->
          if count > !best_count then (
            best_count := count;
            best_pair := Some pair))
        pair_counts;

      match !best_pair with
      | None ->
          vocab_size := trainer.config.vocab_size (* No more merges possible *)
      | Some (a, b) ->
          let new_token = a ^ b in
          if not (Hashtbl.mem vocab new_token) then (
            Hashtbl.add vocab new_token !vocab_size;
            incr vocab_size;
            merges := (a, b) :: !merges;

            (* Update words with the merge *)
            let new_words = Hashtbl.create (Hashtbl.length words_copy) in
            Hashtbl.iter
              (fun word count ->
                let merged =
                  Str.global_replace
                    (Str.regexp_string (a ^ " " ^ b))
                    new_token word
                in
                Hashtbl.add new_words merged count)
              words_copy;
            Hashtbl.clear words_copy;
            Hashtbl.iter (fun k v -> Hashtbl.add words_copy k v) new_words)
    done;

    (* Update model with trained vocab and merges *)
    let bpe_config : config =
      {
        vocab;
        merges = List.rev !merges;
        cache_capacity = 10000;
        dropout = None;
        unk_token = None;
        continuing_subword_prefix = trainer.config.continuing_subword_prefix;
        end_of_word_suffix = trainer.config.end_of_word_suffix;
        fuse_unk = false;
        byte_fallback = false;
        ignore_merges = false;
      }
    in

    let _trained_model = create_internal bpe_config in

    (* Return special tokens *)
    trainer.config.special_tokens
end
