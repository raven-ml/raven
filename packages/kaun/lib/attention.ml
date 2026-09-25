(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type 'a t = {
  q : 'a Linear.t;
  k : 'a Linear.t;
  v : 'a Linear.t;
  out : 'a Linear.t;
}

let walk c { q; k; v; out } =
  let open Nx.Ptree.Walk in
  let q = field c "q" Linear.walk q in
  let k = field c "k" Linear.walk k in
  let v = field c "v" Linear.walk v in
  let out = field c "out" Linear.walk out in
  { q; k; v; out }

let make ?w_init ?bias_init ?bias ?q_dim ?kv_dim ~embed_dim dtype =
  let q_dim = Option.value q_dim ~default:embed_dim in
  let kv_dim = Option.value kv_dim ~default:embed_dim in
  if embed_dim <= 0 || q_dim <= 0 || kv_dim <= 0 then
    Printf.ksprintf invalid_arg
      "Attention.make: embed_dim, q_dim and kv_dim must be positive, got \
       embed_dim=%d q_dim=%d kv_dim=%d"
      embed_dim q_dim kv_dim;
  let proj ~inputs ~outputs =
    Linear.make ?w_init ?bias_init ?bias ~inputs ~outputs dtype
  in
  {
    q = proj ~inputs:embed_dim ~outputs:q_dim;
    k = proj ~inputs:embed_dim ~outputs:kv_dim;
    v = proj ~inputs:embed_dim ~outputs:kv_dim;
    out = proj ~inputs:q_dim ~outputs:embed_dim;
  }

let init ~embed_dim = make ~embed_dim Nx.float32

(* Half and quarter precision floats are too coarse for the attention scores:
   see [scaled_dot_product_attention]. *)
let low_precision : type b. (float, b) Nx.dtype -> bool = function
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 -> true
  | Nx.Float32 | Nx.Float64 -> false

let shape_to_string s =
  "[" ^ String.concat "; " (Array.to_list (Array.map string_of_int s)) ^ "]"

(* Whether [shape] broadcasts to [onto], aligned on the right. *)
let broadcasts ~onto shape =
  let n = Array.length onto and r = Array.length shape in
  r <= n
  && Array.for_all Fun.id
       (Array.mapi (fun i d -> d = 1 || d = onto.(n - r + i)) shape)

let scaled_dot_product_attention ?mask ?scale ?sinks q k v =
  let qs = Nx.shape q and ks = Nx.shape k and vs = Nx.shape v in
  let qr = Array.length qs and kr = Array.length ks and vr = Array.length vs in
  if qr < 2 || kr < 2 || vr < 2 then
    invalid_arg
      "Attention.scaled_dot_product_attention: q, k and v must have at least 2 \
       axes";
  if ks.(kr - 1) <> qs.(qr - 1) then
    Printf.ksprintf invalid_arg
      "Attention.scaled_dot_product_attention: q has %d features but k has %d"
      qs.(qr - 1)
      ks.(kr - 1);
  if vs.(vr - 2) <> ks.(kr - 2) then
    Printf.ksprintf invalid_arg
      "Attention.scaled_dot_product_attention: k has %d positions but v has %d"
      ks.(kr - 2)
      vs.(vr - 2);
  let scale =
    Option.value scale ~default:(1.0 /. sqrt (float_of_int qs.(qr - 1)))
  in
  (* Scores, masking and softmax, generically in the dtype of [q] and [k]. *)
  let weights q k sinks =
    let scores =
      Nx.mul_s (Nx.matmul q (Nx.swapaxes (kr - 2) (kr - 1) k)) scale
    in
    let neg_inf t = Nx.scalar_like t Float.neg_infinity in
    match (mask, sinks) with
    | None, None -> Fn.softmax scores
    | _, Some sinks ->
        let queries = Array.sub (Nx.shape scores) 0 (Nx.ndim scores - 1) in
        if not (broadcasts ~onto:queries (Nx.shape sinks)) then
          Printf.ksprintf invalid_arg
            "Attention.scaled_dot_product_attention: sinks of shape %s do not \
             broadcast to %s, the scores without their last axis"
            (shape_to_string (Nx.shape sinks))
            (shape_to_string queries);
        (* A sink is one more key of value zero: it enters the normaliser and
           has no column. It is finite, so no query is empty. *)
        let sink = Nx.unsqueeze ~axes:[ -1 ] sinks in
        let scores =
          match mask with
          | None -> scores
          | Some m -> Nx.where m scores (neg_inf scores)
        in
        let top = Nx.maximum (Nx.max ~axes:[ -1 ] ~keepdims:true scores) sink in
        let e = Nx.exp (Nx.sub scores top) in
        let total =
          Nx.add
            (Nx.sum ~axes:[ -1 ] ~keepdims:true e)
            (Nx.exp (Nx.sub sink top))
        in
        Nx.div e total
    | Some m, None ->
        let scores = Nx.where m scores (neg_inf scores) in
        let top = Nx.max ~axes:[ -1 ] ~keepdims:true scores in
        (* A query that sees no key: its weights are zero, not 0 / 0. *)
        let empty = Nx.equal top (neg_inf top) in
        let e =
          Nx.exp (Nx.sub scores (Nx.where empty (Nx.zeros_like top) top))
        in
        let total = Nx.sum ~axes:[ -1 ] ~keepdims:true e in
        Nx.div e (Nx.where empty (Nx.ones_like total) total)
  in
  let dt = Nx.dtype q in
  (* Half and quarter precision floats overflow the scores and starve the
     softmax: the score contraction, masking and softmax run in a float32 island
     and only the probabilities come back down for the value matmul. Wider
     dtypes keep their own arithmetic, so the float32 and float64 graphs are
     exactly the pre-island ones. *)
  let probs =
    if low_precision dt then
      Nx.cast dt
        (weights (Nx.cast Nx.float32 q) (Nx.cast Nx.float32 k)
           (Option.map (Nx.cast Nx.float32) sinks))
    else weights q k sinks
  in
  Nx.matmul probs v

(* Head geometry, read from the projection widths. *)

let check_geometry ~fn ~head_dim p =
  if head_dim <= 0 then
    Printf.ksprintf invalid_arg
      "Attention.%s: head_dim must be positive, got %d" fn head_dim;
  let width (l : _ Linear.t) = (Nx.shape l.Linear.w).(1) in
  let qw = width p.q and kw = width p.k and vw = width p.v in
  if vw <> kw then
    Printf.ksprintf invalid_arg
      "Attention.%s: the key and value projections differ in width (k=%d v=%d)"
      fn kw vw;
  if qw mod head_dim <> 0 || kw mod head_dim <> 0 then
    Printf.ksprintf invalid_arg
      "Attention.%s: head_dim %d does not divide the projection widths (q=%d \
       k=%d)"
      fn head_dim qw kw;
  let heads = qw / head_dim and kv_heads = kw / head_dim in
  if heads mod kv_heads <> 0 then
    Printf.ksprintf invalid_arg
      "Attention.%s: %d key-value heads do not divide %d query heads" fn
      kv_heads heads

(* The pieces of a layer *)

let split ~head_dim l x =
  if head_dim <= 0 then
    Printf.ksprintf invalid_arg
      "Attention.split: head_dim must be positive, got %d" head_dim;
  let width = (Nx.shape l.Linear.w).(1) in
  if width mod head_dim <> 0 then
    Printf.ksprintf invalid_arg
      "Attention.split: head_dim %d does not divide the projection width %d"
      head_dim width;
  (match Nx.shape x with
  | [| _; _; _ |] -> ()
  | _ -> invalid_arg "Attention.split: x must have shape [batch; seq; embed]");
  let y = Linear.apply l x in
  let batch = Nx.dim 0 y and seq = Nx.dim 1 y in
  Nx.swapaxes 1 2 (Nx.reshape [| batch; seq; width / head_dim; head_dim |] y)

let check_mask ~fn ~batch ~n ~m mask =
  match Nx.shape mask with
  | [| n'; m' |] when n' = n && m' = m -> ()
  | [| b; n'; m' |] when b = batch && n' = n && m' = m -> ()
  | _ ->
      Printf.ksprintf invalid_arg
        "Attention.%s: mask must have shape [%d; %d] or [%d; %d; %d]" fn n m
        batch n m

(* Each key-value head serves [heads / kv_heads] query heads: the queries gain a
   group axis the keys broadcast over, so no key is repeated. *)
let attend ?mask ?scale ?sinks q k v =
  let batch, heads, n, d =
    match Nx.shape q with
    | [| batch; heads; n; d |] -> (batch, heads, n, d)
    | _ ->
        invalid_arg "Attention.attend: q must have shape [batch; heads; n; d]"
  in
  let kv_heads, m =
    match (Nx.shape k, Nx.shape v) with
    | [| b; h; m; d' |], [| b'; h'; m'; _ |]
      when b = batch && b' = batch && h = h' && m = m' && d' = d ->
        (h, m)
    | _ ->
        Printf.ksprintf invalid_arg
          "Attention.attend: k must have shape [%d; kv_heads; m; %d] and v \
           [%d; kv_heads; m; dv]"
          batch d batch
  in
  if heads mod kv_heads <> 0 then
    Printf.ksprintf invalid_arg
      "Attention.attend: %d key-value heads do not divide %d query heads"
      kv_heads heads;
  let groups = heads / kv_heads in
  Option.iter (check_mask ~fn:"attend" ~batch ~n ~m) mask;
  let sinks =
    Option.map
      (fun sinks ->
        if Nx.shape sinks <> [| heads |] then
          Printf.ksprintf invalid_arg
            "Attention.attend: sinks must have shape [%d]" heads;
        Nx.reshape [| kv_heads; groups; 1 |] sinks)
      sinks
  in
  let q = Nx.reshape [| batch; kv_heads; groups; n; d |] (Nx.contiguous q) in
  let grouped t = Nx.unsqueeze ~axes:[ 2 ] t in
  let mask =
    Option.map
      (fun mk ->
        match Nx.shape mk with
        | [| _; _ |] -> Nx.reshape [| 1; 1; 1; n; m |] mk
        | _ -> Nx.reshape [| batch; 1; 1; n; m |] mk)
      mask
  in
  let out =
    scaled_dot_product_attention ?mask ?scale ?sinks q (grouped k) (grouped v)
  in
  Nx.reshape [| batch; heads; n; Nx.dim 3 v |] out

let merge l y =
  match Nx.shape y with
  | [| batch; heads; seq; d |] ->
      Linear.apply l
        (Nx.reshape
           [| batch; seq; heads * d |]
           (Nx.contiguous (Nx.swapaxes 1 2 y)))
  | _ -> invalid_arg "Attention.merge: y must have shape [batch; heads; seq; d]"

let causal_mask ~seq ?valid () =
  if seq <= 0 then
    Printf.ksprintf invalid_arg
      "Attention.causal_mask: seq must be positive, got %d" seq;
  let idx = Nx.arange Nx.int32 0 seq 1 in
  let row = Nx.reshape [| 1; seq |] idx and col = Nx.reshape [| seq; 1 |] idx in
  (* [tri.(i).(j)] is [j <= i]: query [i] sees keys up to itself. *)
  let tri = Nx.less_equal row col in
  match valid with
  | None -> tri
  | Some valid ->
      (match Nx.shape valid with
      | [| _; s |] when s = seq -> ()
      | _ ->
          Printf.ksprintf invalid_arg
            "Attention.causal_mask: valid must have shape [batch; %d]" seq);
      let batch = Nx.dim 0 valid in
      Nx.logical_and
        (Nx.reshape [| 1; seq; seq |] tri)
        (Nx.reshape [| batch; 1; seq |] valid)

let apply ~head_dim ?mask ?rope p x =
  let shape = Nx.shape x in
  let rank = Array.length shape in
  if rank < 2 then
    invalid_arg
      "Attention.apply: input must have at least sequence and feature axes";
  let embed = (Nx.shape p.q.Linear.w).(0) in
  if shape.(rank - 1) <> embed then
    Printf.ksprintf invalid_arg
      "Attention.apply: last axis has size %d but the layer attends over %d \
       features"
      shape.(rank - 1)
      embed;
  check_geometry ~fn:"apply" ~head_dim p;
  let seq = shape.(rank - 2) in
  let batch = Array.fold_left ( * ) 1 (Array.sub shape 0 (rank - 2)) in
  Option.iter (check_mask ~fn:"apply" ~batch ~n:seq ~m:seq) mask;
  (* Leading axes fold into one batch axis for the pieces and unfold after. *)
  let x = Nx.reshape [| batch; seq; embed |] (Nx.contiguous x) in
  let q = split ~head_dim p.q x and k = split ~head_dim p.k x in
  let v = split ~head_dim p.v x in
  let q, k =
    match rope with
    | None -> (q, k)
    | Some t ->
        let pos = Nx.reshape [| 1; seq |] (Nx.arange Nx.int32 0 seq 1) in
        (Rope.apply t ~pos q, Rope.apply t ~pos k)
  in
  Nx.reshape shape (merge p.out (attend ?mask q k v))

(* Key-value cache *)

module Cache = struct
  type 'a t = { keys : 'a; values : 'a }

  let make ~slots ~kv_heads ~head_dim dtype =
    if slots < 0 || kv_heads <= 0 || head_dim <= 0 then
      Printf.ksprintf invalid_arg
        "Attention.Cache.make: slots must not be negative and kv_heads and \
         head_dim must be positive, got slots=%d kv_heads=%d head_dim=%d"
        slots kv_heads head_dim;
    let pool () = Nx.zeros dtype [| slots; kv_heads; head_dim |] in
    { keys = pool (); values = pool () }

  (* Pools hold tokens first, [batch; seq; kv_heads; head_dim]; heads come first
     everywhere else. *)
  let extend index cache k v =
    let through pool t =
      let seen, pool = Cache_index.extend index (Nx.swapaxes 1 2 t) pool in
      (Nx.swapaxes 1 2 seen, pool)
    in
    let k, keys = through cache.keys k in
    let v, values = through cache.values v in
    (k, v, { keys; values })

  let walk c { keys; values } =
    let open Nx.Ptree.Walk in
    let keys = field c "keys" leaf keys in
    let values = field c "values" leaf values in
    { keys; values }
end

(* Cached attention *)

let cached ~head_dim ?rope p cache index x =
  let batch = Cache_index.batch index and seq = Cache_index.seq index in
  let embed = (Nx.shape p.q.Linear.w).(0) in
  if Nx.shape x <> [| batch; seq; embed |] then
    Printf.ksprintf invalid_arg
      "Attention.cached: input must have shape [%d; %d; %d]" batch seq embed;
  check_geometry ~fn:"cached" ~head_dim p;
  let kv_heads = (Nx.shape p.k.Linear.w).(1) / head_dim in
  (match (Nx.shape cache.Cache.keys, Nx.shape cache.Cache.values) with
  | [| n; h; d |], [| n'; h'; d' |]
    when n = n' && h = kv_heads && h' = kv_heads && d = head_dim
         && d' = head_dim ->
      ()
  | _ ->
      Printf.ksprintf invalid_arg
        "Attention.cached: the cache must have shape [slots; %d; %d]" kv_heads
        head_dim);
  let q = split ~head_dim p.q x and k = split ~head_dim p.k x in
  let v = split ~head_dim p.v x in
  let q, k =
    match rope with
    | None -> (q, k)
    | Some t ->
        let pos = Cache_index.positions index in
        (Rope.apply t ~pos q, Rope.apply t ~pos k)
  in
  let k, v, cache = Cache.extend index cache k v in
  (merge p.out (attend ~mask:(Cache_index.mask index) q k v), cache)
