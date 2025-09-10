(** Implementation of n-gram language models *)

module Int_map = Map.Make (Int)

module Int_pair_map = Map.Make (struct
  type t = int * int

  let compare = compare
end)

type vocab_stats = { vocab_size : int; total_tokens : int; unique_ngrams : int }
type smoothing = Add_k of float | Stupid_backoff of float

type t = {
  n : int;
  vocab_size : int;
  counts : (int array, (int, int) Hashtbl.t) Hashtbl.t;
      (* context -> (next_token -> count) *)
  context_totals : (int array, int) Hashtbl.t; (* context -> total_count *)
  smoothing : smoothing;
  orders : (int array, (int, int) Hashtbl.t) Hashtbl.t array;
  order_totals : (int array, int) Hashtbl.t array;
  logits_cache : (int array, float array) Hashtbl.t option;
  cache_capacity : int;
}

(** Unigram model implementation *)
(* Removed Unigram submodule in favor of generic API *)

(** Bigram model implementation *)
(* Removed Bigram submodule in favor of generic API *)

(** Trigram model implementation *)
(* Removed Trigram submodule in favor of generic API *)

(** Generic n-gram functions *)

let build_orders ~n tokens =
  let orders = Array.init n (fun _ -> Hashtbl.create 1000) in
  let order_totals = Array.init n (fun _ -> Hashtbl.create 1000) in
  let max_token = ref 0 in
  Array.iter (fun t -> if t > !max_token then max_token := t) tokens;
  let len = Array.length tokens in
  for i = 0 to len - 1 do
    for k = 1 to n do
      if i + k - 1 < len then (
        let ctx_len = k - 1 in
        let context =
          if ctx_len = 0 then [||] else Array.sub tokens i ctx_len
        in
        let next = tokens.(i + k - 1) in
        if next > !max_token then max_token := next;
        let tbl = orders.(k - 1) in
        let next_counts =
          match Hashtbl.find_opt tbl context with
          | Some t -> t
          | None ->
              let t = Hashtbl.create 8 in
              Hashtbl.add tbl context t;
              t
        in
        let c =
          match Hashtbl.find_opt next_counts next with Some x -> x | None -> 0
        in
        Hashtbl.replace next_counts next (c + 1);
        let totals = order_totals.(k - 1) in
        let tot =
          match Hashtbl.find_opt totals context with Some x -> x | None -> 0
        in
        Hashtbl.replace totals context (tot + 1))
    done
  done;
  (!max_token + 1, orders, order_totals)

let create ~n ?(smoothing = Add_k 1.0) ?(cache_capacity = 0) tokens =
  if n < 1 then invalid_arg "n must be >= 1";

  let vocab_size, orders, order_totals = build_orders ~n tokens in

  let counts = orders.(n - 1) in
  let context_totals = order_totals.(n - 1) in
  {
    n;
    vocab_size;
    counts;
    context_totals;
    smoothing;
    orders;
    order_totals;
    logits_cache =
      (if cache_capacity > 0 then Some (Hashtbl.create cache_capacity) else None);
    cache_capacity;
  }

let logits_add_k model ~context k =
  (* Compute log probabilities P(next|context) with add-k smoothing *)
  let vocab = model.vocab_size in
  let logits = Array.make vocab (log 0.0) in

  let context_total_smoothed =
    match Hashtbl.find_opt model.context_totals context with
    | Some total -> float_of_int total +. (k *. float_of_int vocab)
    | None -> k *. float_of_int vocab
  in

  let next_counts =
    match Hashtbl.find_opt model.counts context with
    | Some tbl -> tbl
    | None -> Hashtbl.create 0
  in

  for token = 0 to vocab - 1 do
    let c =
      match Hashtbl.find_opt next_counts token with Some x -> x | None -> 0
    in
    logits.(token) <- log ((float_of_int c +. k) /. context_total_smoothed)
  done;

  logits

let rec backoff_score model context token alpha order_idx =
  if order_idx < 0 then 1.0 /. float_of_int model.vocab_size
  else
    let counts_tbl = model.orders.(order_idx) in
    let totals_tbl = model.order_totals.(order_idx) in
    let ctx_len = Array.length context in
    let used_ctx =
      if ctx_len = order_idx then context
      else if order_idx = 0 then [||]
      else
        let start = max 0 (ctx_len - order_idx) in
        Array.sub context start order_idx
    in
    let next_counts =
      match Hashtbl.find_opt counts_tbl used_ctx with
      | Some t -> t
      | None -> Hashtbl.create 0
    in
    let total =
      match Hashtbl.find_opt totals_tbl used_ctx with
      | Some t -> float_of_int t
      | None -> 0.0
    in
    let c =
      match Hashtbl.find_opt next_counts token with
      | Some x -> float_of_int x
      | None -> 0.0
    in
    if c > 0.0 && total > 0.0 then c /. total
    else alpha *. backoff_score model context token alpha (order_idx - 1)

let logits_backoff model ~context alpha =
  let vocab = model.vocab_size in
  let scores = Array.make vocab 0.0 in
  for token = 0 to vocab - 1 do
    scores.(token) <- backoff_score model context token alpha (model.n - 1)
  done;
  (* Normalize and return log *)
  let sum = Array.fold_left ( +. ) 0.0 scores in
  if sum <= 0.0 then Array.make vocab (log (1.0 /. float_of_int vocab))
  else Array.map (fun p -> log (p /. sum)) scores

let logits model ~context =
  (* Cache lookup *)
  match model.logits_cache with
  | Some cache -> (
      match Hashtbl.find_opt cache context with
      | Some v -> v
      | None ->
          let v =
            match model.smoothing with
            | Add_k k -> logits_add_k model ~context k
            | Stupid_backoff a -> logits_backoff model ~context a
          in
          (* simple capacity policy: clear if over capacity *)
          if Hashtbl.length cache >= model.cache_capacity then
            Hashtbl.clear cache;
          Hashtbl.add cache context v;
          v)
  | None -> (
      match model.smoothing with
      | Add_k k -> logits_add_k model ~context k
      | Stupid_backoff a -> logits_backoff model ~context a)

let perplexity model tokens =
  let log_prob_sum = ref 0.0 in
  let count = ref 0 in

  for i = model.n - 1 to Array.length tokens - 1 do
    let context =
      if model.n = 1 then [||]
      else Array.sub tokens (i - model.n + 1) (model.n - 1)
    in
    let log_probs = logits model ~context in
    let tok = tokens.(i) in
    if tok >= 0 && tok < model.vocab_size then (
      log_prob_sum := !log_prob_sum +. log_probs.(tok);
      incr count)
  done;

  if !count = 0 then infinity else exp (-. !log_prob_sum /. float_of_int !count)

let generate model ?(max_tokens = 100) ?(temperature = 1.0)
    ?(seed = Random.int 1000000) ?(start = [||]) () =
  let rng = Random.State.make [| seed |] in
  (* Maintain generated tokens as a reverse list for O(1) append. *)
  let rev_tokens = List.rev (Array.to_list start) in

  let rec gen_loop rev_tokens remaining =
    if remaining <= 0 then Array.of_list (List.rev rev_tokens)
    else
      let context =
        if model.n = 1 then [||]
        else
          let ctx_len = min (model.n - 1) (List.length rev_tokens) in
          let ctx_rev = List.(rev (rev_tokens |> take ctx_len)) in
          Array.of_list ctx_rev
      in
      let log_probs = logits model ~context in

      (* Sample with temperature *)
      let probs =
        Array.map (fun log_p -> exp (log_p /. temperature)) log_probs
      in
      let sum = Array.fold_left ( +. ) 0.0 probs in
      let probs =
        if sum > 0. then Array.map (fun p -> p /. sum) probs else probs
      in

      let r = Random.State.float rng 1.0 in
      let cumsum = ref 0.0 in
      let next_token = ref (model.vocab_size - 1) in
      for i = 0 to model.vocab_size - 1 do
        cumsum := !cumsum +. probs.(i);
        if !cumsum > r && !next_token = model.vocab_size - 1 then
          next_token := i
      done;

      gen_loop (!next_token :: rev_tokens) (remaining - 1)
  in

  gen_loop rev_tokens max_tokens

let stats model =
  let unique =
    Hashtbl.fold
      (fun _ next_map acc -> acc + Hashtbl.length next_map)
      model.counts 0
  in
  {
    vocab_size = model.vocab_size;
    total_tokens = Hashtbl.fold (fun _ c acc -> acc + c) model.context_totals 0;
    unique_ngrams = unique;
  }

let save model path =
  let oc = open_out_bin path in
  output_value oc model;
  close_out oc

let load path =
  let ic = open_in_bin path in
  let model = input_value ic in
  close_in ic;
  model

let save_text model path =
  let oc = open_out path in
  let smoothing_tag, smoothing_val =
    match model.smoothing with
    | Add_k k -> ("addk", k)
    | Stupid_backoff a -> ("sbo", a)
  in
  Printf.fprintf oc "n %d vocab %d smooth %s %f\n" model.n model.vocab_size
    smoothing_tag smoothing_val;
  Printf.fprintf oc "orders %d\n" (Array.length model.orders);
  for oi = 0 to Array.length model.orders - 1 do
    let tbl = model.orders.(oi) in
    Printf.fprintf oc "order %d contexts %d\n" (oi + 1) (Hashtbl.length tbl);
    Hashtbl.iter
      (fun ctx nexts ->
        (* print context *)
        Printf.fprintf oc "ctx %d" (Array.length ctx);
        Array.iter (fun t -> Printf.fprintf oc " %d" t) ctx;
        (* print next counts *)
        Printf.fprintf oc " next %d" (Hashtbl.length nexts);
        Hashtbl.iter (fun token c -> Printf.fprintf oc " %d:%d" token c) nexts;
        output_string oc "\n")
      tbl
  done;
  close_out oc

let load_text path =
  let ic = open_in path in
  let line = input_line ic in
  let n, vocab_size, smoothing =
    Scanf.sscanf line "n %d vocab %d smooth %s %f" (fun n v tag sval ->
        let s = if tag = "addk" then Add_k sval else Stupid_backoff sval in
        (n, v, s))
  in
  let orders = Array.init n (fun _ -> Hashtbl.create 1000) in
  let order_totals = Array.init n (fun _ -> Hashtbl.create 1000) in
  let _ = input_line ic in
  (* orders line *)
  let current_order = ref 0 in
  let rec loop () =
    match input_line ic with
    | exception End_of_file -> ()
    | l ->
        if l = "" then loop ()
        else if String.length l >= 5 && String.sub l 0 5 = "order" then (
          (* parse order index, 1-based *)
          (try
             Scanf.sscanf l "order %d contexts %d" (fun oi _ ->
                 current_order := oi - 1)
           with _ -> ());
          loop ())
        else if String.length l >= 3 && String.sub l 0 3 = "ctx" then
          (* parse ctx line *)
          try
            let rest = String.sub l 4 (String.length l - 4) in
            (* rest: "<clen> <c0> ... <cN> next <m> <t1:c1> ..." *)
            let parts =
              List.filter (( <> ) "") (String.split_on_char ' ' rest)
            in
            match parts with
            | clen_str :: tl ->
                let clen = int_of_string clen_str in
                let ctx = Array.make clen 0 in
                let rec take_ctx i lst =
                  if i = clen then lst
                  else
                    match lst with
                    | h :: t ->
                        ctx.(i) <- int_of_string h;
                        take_ctx (i + 1) t
                    | [] -> []
                in
                let after_ctx = take_ctx 0 tl in
                let after_next =
                  match after_ctx with
                  | h :: t when h = "next" -> t
                  | _ -> after_ctx
                in
                let _m, pairs =
                  match after_next with
                  | m :: rest -> (int_of_string m, rest)
                  | _ -> (0, [])
                in
                let tbl = Hashtbl.create 8 in
                List.iter
                  (fun p ->
                    match String.split_on_char ':' p with
                    | [ a; b ] ->
                        Hashtbl.replace tbl (int_of_string a) (int_of_string b)
                    | _ -> ())
                  pairs;
                Hashtbl.add orders.(!current_order) ctx tbl;
                let total = Hashtbl.fold (fun _ c acc -> acc + c) tbl 0 in
                Hashtbl.add order_totals.(!current_order) ctx total;
                loop ()
            | _ -> loop ()
          with _ -> loop ()
        else loop ()
  in
  loop ();
  close_in ic;
  let counts = orders.(n - 1) in
  let context_totals = order_totals.(n - 1) in
  {
    n;
    vocab_size;
    counts;
    context_totals;
    smoothing;
    orders;
    order_totals;
    logits_cache = None;
    cache_capacity = 0;
  }
