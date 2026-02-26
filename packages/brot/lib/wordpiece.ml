(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type token = { id : int; value : string; offsets : int * int }

(* Compact trie for zero-allocation longest-prefix matching *)

type trie = {
  trie_ids : int array;
  child_starts : int array;
  edge_bytes : bytes;
  edge_targets : int array;
  (* Flat 256-element arrays for dense nodes (>16 children) — O(1) lookup *)
  flat_nodes : int array array;
}

let build_trie vocab =
  if Hashtbl.length vocab = 0 then
    {
      trie_ids = [||];
      child_starts = [| 0 |];
      edge_bytes = Bytes.empty;
      edge_targets = [||];
      flat_nodes = [||];
    }
  else
    let cap = ref 256 in
    let ids = ref (Array.make !cap (-1)) in
    let ch = ref (Array.init !cap (fun _ -> Hashtbl.create 0)) in
    let n = ref 1 in
    !ch.(0) <- Hashtbl.create 64;
    let grow () =
      let new_cap = !cap * 2 in
      let new_ids = Array.make new_cap (-1) in
      Array.blit !ids 0 new_ids 0 !n;
      ids := new_ids;
      let new_ch =
        Array.init new_cap (fun i ->
            if i < !n then !ch.(i) else Hashtbl.create 0)
      in
      ch := new_ch;
      cap := new_cap
    in
    Hashtbl.iter
      (fun key id ->
        let cur = ref 0 in
        for i = 0 to String.length key - 1 do
          let byte = Char.code (String.unsafe_get key i) in
          let child =
            match Hashtbl.find_opt !ch.(!cur) byte with
            | Some c -> c
            | None ->
                if !n >= !cap then grow ();
                let c = !n in
                incr n;
                !ch.(c) <- Hashtbl.create 4;
                Hashtbl.add !ch.(!cur) byte c;
                c
          in
          cur := child
        done;
        !ids.(!cur) <- id)
      vocab;
    let node_count = !n in
    let trie_ids = Array.init node_count (fun i -> !ids.(i)) in
    let child_starts = Array.make (node_count + 1) 0 in
    let total = ref 0 in
    for i = 0 to node_count - 1 do
      child_starts.(i) <- !total;
      total := !total + Hashtbl.length !ch.(i)
    done;
    child_starts.(node_count) <- !total;
    let edge_bytes = Bytes.create !total in
    let edge_targets = Array.make !total 0 in
    let pos = ref 0 in
    for i = 0 to node_count - 1 do
      Hashtbl.iter
        (fun byte child ->
          Bytes.unsafe_set edge_bytes !pos (Char.unsafe_chr byte);
          edge_targets.(!pos) <- child;
          incr pos)
        !ch.(i)
    done;
    (* Sort each node's children by byte value for binary search *)
    for i = 0 to node_count - 1 do
      let start = child_starts.(i) in
      let stop = child_starts.(i + 1) in
      for j = start + 1 to stop - 1 do
        let kb = Bytes.unsafe_get edge_bytes j in
        let kt = edge_targets.(j) in
        let k = ref (j - 1) in
        while !k >= start && Bytes.unsafe_get edge_bytes !k > kb do
          Bytes.unsafe_set edge_bytes (!k + 1) (Bytes.unsafe_get edge_bytes !k);
          edge_targets.(!k + 1) <- edge_targets.(!k);
          decr k
        done;
        Bytes.unsafe_set edge_bytes (!k + 1) kb;
        edge_targets.(!k + 1) <- kt
      done
    done;
    (* Build flat 256-element arrays for dense nodes (>16 children) *)
    let flat_nodes = Array.make node_count [||] in
    for i = 0 to node_count - 1 do
      let start = child_starts.(i) in
      let count = child_starts.(i + 1) - start in
      if count > 16 then begin
        let flat = Array.make 256 (-1) in
        for j = start to start + count - 1 do
          let b = Char.code (Bytes.unsafe_get edge_bytes j) in
          flat.(b) <- Array.unsafe_get edge_targets j
        done;
        flat_nodes.(i) <- flat
      end
    done;
    { trie_ids; child_starts; edge_bytes; edge_targets; flat_nodes }

let[@inline] trie_step trie node byte =
  let flat = Array.unsafe_get trie.flat_nodes node in
  if Array.length flat > 0 then Array.unsafe_get flat byte
  else
    let lo = ref (Array.unsafe_get trie.child_starts node) in
    let hi = ref (Array.unsafe_get trie.child_starts (node + 1) - 1) in
    let result = ref (-1) in
    while !lo <= !hi do
      let mid = !lo + ((!hi - !lo) asr 1) in
      let mid_byte = Char.code (Bytes.unsafe_get trie.edge_bytes mid) in
      if mid_byte = byte then (
        result := Array.unsafe_get trie.edge_targets mid;
        lo := !hi + 1)
      else if mid_byte < byte then lo := mid + 1
      else hi := mid - 1
    done;
    !result

let trie_longest_match trie sequence ~start ~prefix ~prefix_len =
  if Array.length trie.trie_ids = 0 then None
  else
    let seq_len = String.length sequence in
    let last_id = ref (-1) in
    let last_end = ref start in
    let current = ref 0 in
    let stopped = ref false in
    let i = ref 0 in
    while !i < prefix_len && not !stopped do
      let child =
        trie_step trie !current (Char.code (String.unsafe_get prefix !i))
      in
      if child < 0 then stopped := true
      else (
        current := child;
        incr i)
    done;
    (if not !stopped then
       let j = ref start in
       while !j < seq_len && not !stopped do
         let child =
           trie_step trie !current (Char.code (String.unsafe_get sequence !j))
         in
         if child < 0 then stopped := true
         else (
           current := child;
           incr j;
           let tid = Array.unsafe_get trie.trie_ids child in
           if tid >= 0 then (
             last_id := tid;
             last_end := !j))
       done);
    if !last_id >= 0 then Some (!last_id, !last_end) else None

(* Model type *)

type t = {
  vocab : (string, int) Hashtbl.t;
  vocab_r : string array;
  trie : trie;
  unk_token : string;
  continuing_subword_prefix : string;
  max_input_chars_per_word : int;
}

let create ~vocab ?(unk_token = "[UNK]") ?(continuing_subword_prefix = "##")
    ?(max_input_chars_per_word = 100) () =
  let max_id = Hashtbl.fold (fun _ id acc -> max id acc) vocab (-1) in
  let vocab_r = Array.make (max_id + 1) "" in
  Hashtbl.iter (fun k v -> Array.unsafe_set vocab_r v k) vocab;
  if Hashtbl.length vocab > 0 && not (Hashtbl.mem vocab unk_token) then
    invalid_arg "Wordpiece.create: unk_token not in vocab";
  let trie = build_trie vocab in
  {
    vocab;
    vocab_r;
    trie;
    unk_token;
    continuing_subword_prefix;
    max_input_chars_per_word;
  }

let read_file ~vocab_file =
  let vocab = Hashtbl.create 10000 in
  let ic = open_in vocab_file in
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () ->
      let index = ref 0 in
      (try
         while true do
           let line = input_line ic in
           let token = String.trim line in
           if token <> "" then (
             Hashtbl.add vocab token !index;
             incr index)
         done
       with End_of_file -> ());
      vocab)

let from_file ~vocab_file =
  let vocab = read_file ~vocab_file in
  create ~vocab ()

let count_chars s =
  let len = String.length s in
  let n = ref 0 in
  for i = 0 to len - 1 do
    if Char.code (String.unsafe_get s i) land 0xC0 <> 0x80 then incr n
  done;
  !n

let tokenize model sequence =
  if Hashtbl.length model.vocab = 0 then []
  else
    let seq_len = String.length sequence in
    if count_chars sequence > model.max_input_chars_per_word then
      let id = Hashtbl.find model.vocab model.unk_token in
      [ { id; value = model.unk_token; offsets = (0, seq_len) } ]
    else
      let prefix = model.continuing_subword_prefix in
      let prefix_len = String.length prefix in
      let rec greedy start acc =
        if start >= seq_len then List.rev acc
        else
          let p = if start > 0 then prefix else "" in
          let pl = if start > 0 then prefix_len else 0 in
          match
            trie_longest_match model.trie sequence ~start ~prefix:p
              ~prefix_len:pl
          with
          | Some (id, end_byte) ->
              let value = Array.unsafe_get model.vocab_r id in
              greedy end_byte ({ id; value; offsets = (start, end_byte) } :: acc)
          | None ->
              let id = Hashtbl.find model.vocab model.unk_token in
              [ { id; value = model.unk_token; offsets = (0, seq_len) } ]
      in
      greedy 0 []

let tokenize_ids model sequence =
  if Hashtbl.length model.vocab = 0 then [||]
  else
    let seq_len = String.length sequence in
    if count_chars sequence > model.max_input_chars_per_word then
      let id = Hashtbl.find model.vocab model.unk_token in
      [| id |]
    else
      let prefix = model.continuing_subword_prefix in
      let prefix_len = String.length prefix in
      let ids = ref [] in
      let n = ref 0 in
      let rec greedy start =
        if start >= seq_len then ()
        else
          let p = if start > 0 then prefix else "" in
          let pl = if start > 0 then prefix_len else 0 in
          match
            trie_longest_match model.trie sequence ~start ~prefix:p
              ~prefix_len:pl
          with
          | Some (id, end_byte) ->
              ids := id :: !ids;
              incr n;
              greedy end_byte
          | None ->
              let unk_id = Hashtbl.find model.vocab model.unk_token in
              ids := [ unk_id ];
              n := 1
      in
      greedy 0;
      let result = Array.make !n 0 in
      List.iteri (fun i id -> result.(!n - 1 - i) <- id) !ids;
      result

let tokenize_spans_encoding model pre_tokens ~type_id =
  if Hashtbl.length model.vocab = 0 then Encoding.empty
  else
    let trie = model.trie in
    let prefix = model.continuing_subword_prefix in
    let prefix_len = String.length prefix in
    let unk_id = Hashtbl.find model.vocab model.unk_token in
    let max_chars = model.max_input_chars_per_word in
    let vocab_r = model.vocab_r in
    let unk_token_str = model.unk_token in
    (* Single pass: convert pre_tokens to array for direct access (no closure),
       tokenize all fragments and fill growable output arrays directly. *)
    let pre_arr = Array.of_list pre_tokens in
    let n_pre = Array.length pre_arr in
    let cap = ref (max 16 (n_pre * 2)) in
    let ids = ref (Array.make !cap 0) in
    let token_strs = ref (Array.make !cap "") in
    let offsets_arr = ref (Array.make !cap (0, 0)) in
    let n = ref 0 in
    let grow () =
      let new_cap = !cap * 2 in
      let new_ids = Array.make new_cap 0 in
      Array.blit !ids 0 new_ids 0 !n;
      ids := new_ids;
      let new_strs = Array.make new_cap "" in
      Array.blit !token_strs 0 new_strs 0 !n;
      token_strs := new_strs;
      let new_off = Array.make new_cap (0, 0) in
      Array.blit !offsets_arr 0 new_off 0 !n;
      offsets_arr := new_off;
      cap := new_cap
    in
    (* Hoisted mutable state for trie matching — allocated once *)
    let current = ref 0 in
    let stopped = ref false in
    let last_id = ref (-1) in
    let last_end = ref 0 in
    let pos = ref 0 in
    let is_unk = ref false in
    let char_count = ref 0 in
    let i_ref = ref 0 in
    let j_ref = ref 0 in
    for frag_idx = 0 to n_pre - 1 do
      let fragment, _ = Array.unsafe_get pre_arr frag_idx in
      let seq_len = String.length fragment in
      char_count := 0;
      for k = 0 to seq_len - 1 do
        if Char.code (String.unsafe_get fragment k) land 0xC0 <> 0x80 then
          incr char_count
      done;
      if !char_count > max_chars then begin
        if !n >= !cap then grow ();
        Array.unsafe_set !ids !n unk_id;
        Array.unsafe_set !token_strs !n unk_token_str;
        Array.unsafe_set !offsets_arr !n (0, seq_len);
        incr n
      end
      else begin
        pos := 0;
        is_unk := false;
        let start_n = !n in
        while !pos < seq_len && not !is_unk do
          let match_start = !pos in
          current := 0;
          stopped := false;
          last_id := -1;
          last_end := !pos;
          if !pos > 0 then begin
            i_ref := 0;
            while !i_ref < prefix_len && not !stopped do
              let child =
                trie_step trie !current
                  (Char.code (String.unsafe_get prefix !i_ref))
              in
              if child < 0 then stopped := true
              else begin
                current := child;
                incr i_ref
              end
            done
          end;
          if not !stopped then begin
            j_ref := !pos;
            while !j_ref < seq_len && not !stopped do
              let child =
                trie_step trie !current
                  (Char.code (String.unsafe_get fragment !j_ref))
              in
              if child < 0 then stopped := true
              else begin
                current := child;
                incr j_ref;
                let tid = Array.unsafe_get trie.trie_ids child in
                if tid >= 0 then begin
                  last_id := tid;
                  last_end := !j_ref
                end
              end
            done
          end;
          if !last_id >= 0 then begin
            if !n >= !cap then grow ();
            Array.unsafe_set !ids !n !last_id;
            Array.unsafe_set !token_strs !n (Array.unsafe_get vocab_r !last_id);
            Array.unsafe_set !offsets_arr !n (match_start, !last_end);
            incr n;
            pos := !last_end
          end
          else is_unk := true
        done;
        if !is_unk then begin
          n := start_n;
          if !n >= !cap then grow ();
          Array.unsafe_set !ids !n unk_id;
          Array.unsafe_set !token_strs !n unk_token_str;
          Array.unsafe_set !offsets_arr !n (0, seq_len);
          n := start_n + 1
        end
      end
    done;
    let total = !n in
    if total = 0 then Encoding.empty
    else
      let final_ids = if total = !cap then !ids else Array.sub !ids 0 total in
      let final_strs =
        if total = !cap then !token_strs else Array.sub !token_strs 0 total
      in
      let final_off =
        if total = !cap then !offsets_arr else Array.sub !offsets_arr 0 total
      in
      Encoding.create ~ids:final_ids ~type_ids:(Array.make total type_id)
        ~tokens:final_strs ~words:(Array.make total None) ~offsets:final_off
        ~special_tokens_mask:(Array.make total 0)
        ~attention_mask:(Array.make total 1) ()

let token_to_id model token = Hashtbl.find_opt model.vocab token

let id_to_token model id =
  if id >= 0 && id < Array.length model.vocab_r then
    Some (Array.unsafe_get model.vocab_r id)
  else None

let get_vocab model = Hashtbl.fold (fun k v acc -> (k, v) :: acc) model.vocab []
let get_vocab_size model = Hashtbl.length model.vocab
let get_unk_token model = model.unk_token
let get_continuing_subword_prefix model = model.continuing_subword_prefix

let save model ~path ?name () =
  let vocab_file =
    match name with
    | Some n -> Filename.concat path (n ^ "-vocab.txt")
    | None -> Filename.concat path "vocab.txt"
  in
  let vocab_list =
    Hashtbl.fold (fun k v acc -> (v, k) :: acc) model.vocab []
    |> List.sort compare
    |> List.map (fun (_, k) -> k)
  in
  let oc = open_out vocab_file in
  Fun.protect
    ~finally:(fun () -> close_out oc)
    (fun () ->
      List.iter
        (fun token ->
          output_string oc token;
          output_char oc '\n')
        vocab_list);
  vocab_file

let from_bpe bpe =
  let vocab = Hashtbl.create (Bpe.get_vocab_size bpe) in
  List.iter (fun (k, id) -> Hashtbl.add vocab k id) (Bpe.get_vocab bpe);
  let unk_token =
    match Bpe.get_unk_token bpe with Some u -> u | None -> "[UNK]"
  in
  if not (Hashtbl.mem vocab unk_token) then begin
    let max_id = Hashtbl.fold (fun _ id acc -> max id acc) vocab (-1) in
    Hashtbl.add vocab unk_token (max_id + 1)
  end;
  let continuing_subword_prefix =
    match Bpe.get_continuing_subword_prefix bpe with
    | Some p -> p
    | None -> "##"
  in
  create ~vocab ~unk_token ~continuing_subword_prefix ()

(* Trainer *)

let train ~min_frequency ~vocab_size ~show_progress ~special_tokens
    ~limit_alphabet ~initial_alphabet ~continuing_subword_prefix
    ~end_of_word_suffix texts existing =
  let _ = existing in
  (* WordPiece training uses BPE algorithm internally *)
  let bpe_trained, result_tokens =
    Bpe.train ~min_frequency ~vocab_size ~show_progress ~special_tokens
      ~limit_alphabet ~initial_alphabet
      ~continuing_subword_prefix:(Some continuing_subword_prefix)
      ~end_of_word_suffix ~max_token_length:None texts None
  in
  let wordpiece_model = from_bpe bpe_trained in
  (wordpiece_model, result_tokens)
