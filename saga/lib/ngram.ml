(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type smoothing = [ `Add_k of float | `Stupid_backoff of float ]
type stats = { vocab_size : int; total_tokens : int; unique_ngrams : int }

type counts = {
  orders : (int array, (int, int) Hashtbl.t) Hashtbl.t array;
  order_totals : (int array, int) Hashtbl.t array;
}

type t = {
  order : int;
  smoothing : smoothing;
  mutable vocab_size : int;
  mutable total_tokens : int;
  counts : counts;
}

let ensure_order order =
  if order < 1 then invalid_arg "Ngram.empty: order must be >= 1"

let make_counts order =
  {
    orders = Array.init order (fun _ -> Hashtbl.create 1024);
    order_totals = Array.init order (fun _ -> Hashtbl.create 1024);
  }

let empty ~order ?(smoothing = `Add_k 0.01) () =
  ensure_order order;
  {
    order;
    smoothing;
    vocab_size = 0;
    total_tokens = 0;
    counts = make_counts order;
  }

let smoothing t = t.smoothing
let order t = t.order
let is_trained t = t.total_tokens > 0

let update_vocab_size t tokens =
  Array.iter
    (fun token -> if token + 1 > t.vocab_size then t.vocab_size <- token + 1)
    tokens

let add_to_counts t tokens =
  let len = Array.length tokens in
  t.total_tokens <- t.total_tokens + len;
  update_vocab_size t tokens;
  for i = 0 to len - 1 do
    for k = 1 to t.order do
      if i + k - 1 < len then (
        let ctx_len = k - 1 in
        let context =
          if ctx_len = 0 then [||] else Array.sub tokens i ctx_len
        in
        let next = tokens.(i + k - 1) in
        let order_idx = k - 1 in
        let orders_tbl = t.counts.orders.(order_idx) in
        let next_tbl =
          match Hashtbl.find_opt orders_tbl context with
          | Some tbl -> tbl
          | None ->
              let tbl = Hashtbl.create 8 in
              Hashtbl.add orders_tbl context tbl;
              tbl
        in
        let current =
          match Hashtbl.find_opt next_tbl next with Some c -> c | None -> 0
        in
        Hashtbl.replace next_tbl next (current + 1);
        let totals_tbl = t.counts.order_totals.(order_idx) in
        let prev_total =
          match Hashtbl.find_opt totals_tbl context with
          | Some v -> v
          | None -> 0
        in
        Hashtbl.replace totals_tbl context (prev_total + 1))
    done
  done

let add_sequence t tokens =
  add_to_counts t tokens;
  t

let of_sequences ~order ?smoothing sequences =
  let model = empty ~order ?smoothing () in
  List.iter (fun seq -> ignore (add_sequence model seq)) sequences;
  model

let stats t =
  let unique =
    Array.fold_left
      (fun acc tbl ->
        Hashtbl.fold
          (fun _ next_map acc -> acc + Hashtbl.length next_map)
          tbl acc)
      0 t.counts.orders
  in
  {
    vocab_size = t.vocab_size;
    total_tokens = t.total_tokens;
    unique_ngrams = unique;
  }

let normalise_context t context =
  if t.order = 1 then [||]
  else
    let len = Array.length context in
    let need = min (t.order - 1) len in
    if need = len then Array.copy context
    else Array.sub context (len - need) need

let rec backoff_score t context token alpha order_idx =
  if order_idx < 0 then
    if t.vocab_size = 0 then 0.0 else 1.0 /. float_of_int t.vocab_size
  else
    let orders_tbl = t.counts.orders.(order_idx) in
    let totals_tbl = t.counts.order_totals.(order_idx) in
    let ctx_len = Array.length context in
    let used_ctx =
      if ctx_len = order_idx then context
      else if order_idx = 0 then [||]
      else
        let start = max 0 (ctx_len - order_idx) in
        Array.sub context start order_idx
    in
    let next_counts =
      match Hashtbl.find_opt orders_tbl used_ctx with
      | Some tbl -> tbl
      | None -> Hashtbl.create 0
    in
    let total =
      match Hashtbl.find_opt totals_tbl used_ctx with
      | Some value -> float_of_int value
      | None -> 0.0
    in
    let c =
      match Hashtbl.find_opt next_counts token with
      | Some value -> float_of_int value
      | None -> 0.0
    in
    if c > 0.0 && total > 0.0 then c /. total
    else alpha *. backoff_score t context token alpha (order_idx - 1)

let logits t ~context =
  if not (is_trained t) then invalid_arg "Ngram.logits: model not trained";
  let context = normalise_context t context in
  match t.smoothing with
  | `Add_k k ->
      let vocab = max 1 t.vocab_size in
      let logits = Array.make vocab Float.neg_infinity in
      let orders_tbl = t.counts.orders.(t.order - 1) in
      let totals_tbl = t.counts.order_totals.(t.order - 1) in
      let total =
        match Hashtbl.find_opt totals_tbl context with
        | Some value -> float_of_int value +. (k *. float_of_int vocab)
        | None -> k *. float_of_int vocab
      in
      let counts_tbl =
        match Hashtbl.find_opt orders_tbl context with
        | Some tbl -> tbl
        | None -> Hashtbl.create 0
      in
      for token = 0 to vocab - 1 do
        let count =
          match Hashtbl.find_opt counts_tbl token with
          | Some c -> float_of_int c
          | None -> 0.0
        in
        logits.(token) <- log ((count +. k) /. total)
      done;
      logits
  | `Stupid_backoff alpha ->
      let vocab = max 1 t.vocab_size in
      Array.init vocab (fun token ->
          let prob = backoff_score t context token alpha (t.order - 1) in
          if prob <= 0.0 then Float.neg_infinity else log prob)

let log_prob t tokens =
  if not (is_trained t) then invalid_arg "Ngram.log_prob: model not trained";
  let sum = ref 0.0 in
  let len = Array.length tokens in
  for i = t.order - 1 to len - 1 do
    let context =
      if t.order = 1 then [||]
      else
        Array.sub tokens (max 0 (i - t.order + 1)) (min (t.order - 1) (i + 1))
    in
    let logits = logits t ~context in
    let token = tokens.(i) in
    if token >= 0 && token < Array.length logits then
      sum := !sum +. logits.(token)
  done;
  !sum

let perplexity t tokens =
  let len = Array.length tokens in
  if len = 0 then infinity
  else
    let lp = log_prob t tokens in
    let denom = float_of_int (max 1 (len - (t.order - 1))) in
    exp (-.lp /. denom)

let save t path =
  let oc = open_out_bin path in
  Fun.protect
    ~finally:(fun () -> close_out oc)
    (fun () -> Marshal.to_channel oc t [])

let load path =
  let ic = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () ->
      let model : t = Marshal.from_channel ic in
      model)
