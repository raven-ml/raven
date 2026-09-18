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

let map f { q; k; v; out } =
  let q = Linear.map f q in
  let k = Linear.map f k in
  let v = Linear.map f v in
  let out = Linear.map f out in
  { q; k; v; out }

let map2 f p p' =
  let q = Linear.map2 f p.q p'.q in
  let k = Linear.map2 f p.k p'.k in
  let v = Linear.map2 f p.v p'.v in
  let out = Linear.map2 f p.out p'.out in
  { q; k; v; out }

let iter f { q; k; v; out } =
  Linear.iter f q;
  Linear.iter f k;
  Linear.iter f v;
  Linear.iter f out

let join prefix path = if path = "" then prefix else prefix ^ "." ^ path

let fold f acc { q; k; v; out } =
  let under prefix acc l =
    Linear.fold (fun path -> f (join prefix path)) acc l
  in
  under "out" (under "v" (under "k" (under "q" acc q) k) v) out

let fold2 f acc p p' =
  let under prefix acc l l' =
    Linear.fold2 (fun path -> f (join prefix path)) acc l l'
  in
  under "out"
    (under "v" (under "k" (under "q" acc p.q p'.q) p.k p'.k) p.v p'.v)
    p.out p'.out

let names p =
  let sub prefix l = Linear.map (join prefix) (Linear.names l) in
  { q = sub "q" p.q; k = sub "k" p.k; v = sub "v" p.v; out = sub "out" p.out }

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

let scaled_dot_product_attention ?mask q k v =
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
  let scale = 1.0 /. sqrt (float_of_int qs.(qr - 1)) in
  (* Scores, masking and softmax, generically in the dtype of [q] and [k]. *)
  let weights q k =
    let scores =
      Nx.mul_s (Nx.matmul q (Nx.swapaxes (kr - 2) (kr - 1) k)) scale
    in
    match mask with
    | None -> Fn.softmax scores
    | Some m ->
        let neg_inf t = Nx.scalar_like t Float.neg_infinity in
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
      Nx.cast dt (weights (Nx.cast Nx.float32 q) (Nx.cast Nx.float32 k))
    else weights q k
  in
  Nx.matmul probs v

(* Head geometry, read from the projection widths. *)

let geometry ~fn ~head_dim p =
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
      kv_heads heads;
  (heads, kv_heads)

(* [batch; n; heads * head_dim] -> [batch; heads; n; head_dim]: heads become a
   batch axis so the core runs each head independently. *)
let split ~heads ~head_dim t =
  let s = Nx.shape t in
  Nx.swapaxes 1 2 (Nx.reshape [| s.(0); s.(1); heads; head_dim |] t)

(* Attention of [q : [batch; heads; n; d]] over [k], [v : [batch; kv_heads; m;
   d]], merged to [batch; n; heads * d]. Each key-value head serves [heads /
   kv_heads] query heads: the queries gain a group axis the keys broadcast over,
   so no key is repeated. [mask] is [n; m] or [batch; n; m]. *)
let attend ~kv_heads ?mask q k v =
  let qs = Nx.shape q in
  let batch = qs.(0) and heads = qs.(1) and n = qs.(2) and d = qs.(3) in
  let m = (Nx.shape k).(2) in
  let groups = heads / kv_heads in
  let q = Nx.reshape [| batch; kv_heads; groups; n; d |] (Nx.contiguous q) in
  let grouped t = Nx.unsqueeze ~axes:[ 2 ] t in
  let mask =
    Option.map
      (fun mk ->
        match Nx.shape mk with
        | [| _; _ |] -> Nx.reshape [| 1; 1; 1; n; m |] mk
        | [| b; _; _ |] -> Nx.reshape [| b; 1; 1; n; m |] mk
        | _ -> assert false)
      mask
  in
  let out = scaled_dot_product_attention ?mask q (grouped k) (grouped v) in
  let out = Nx.reshape [| batch; heads; n; d |] out in
  Nx.reshape [| batch; n; heads * d |] (Nx.contiguous (Nx.swapaxes 1 2 out))

let check_mask ~fn ~batch ~n ~m mask =
  match Nx.shape mask with
  | [| n'; m' |] when n' = n && m' = m -> ()
  | [| b; n'; m' |] when b = batch && n' = n && m' = m -> ()
  | _ ->
      Printf.ksprintf invalid_arg
        "Attention.%s: mask must have shape [%d; %d] or [%d; %d; %d]" fn n m
        batch n m

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
  let heads, kv_heads = geometry ~fn:"apply" ~head_dim p in
  let seq = shape.(rank - 2) in
  let batch = Array.fold_left ( * ) 1 (Array.sub shape 0 (rank - 2)) in
  Option.iter (check_mask ~fn:"apply" ~batch ~n:seq ~m:seq) mask;
  (* Leading axes fold into one batch axis for the core and unfold after. *)
  let project l ~heads =
    let y = Linear.apply l x in
    split ~heads ~head_dim
      (Nx.reshape [| batch; seq; heads * head_dim |] (Nx.contiguous y))
  in
  let q = project p.q ~heads and k = project p.k ~heads:kv_heads in
  let v = project p.v ~heads:kv_heads in
  let q, k =
    match rope with
    | None -> (q, k)
    | Some t ->
        let pos = Nx.reshape [| 1; seq |] (Nx.arange Nx.int32 0 seq 1) in
        (Rope.apply t ~pos q, Rope.apply t ~pos k)
  in
  let out = Linear.apply p.out (attend ~kv_heads ?mask q k v) in
  Nx.reshape shape out

(* Key-value cache *)

module Cache = struct
  type 'a t = { keys : 'a; values : 'a }

  let make ~slots ~kv_heads ~head_dim dtype =
    if slots < 0 || kv_heads <= 0 || head_dim <= 0 then
      Printf.ksprintf invalid_arg
        "Attention.Cache.make: slots must not be negative and kv_heads and \
         head_dim must be positive, got slots=%d kv_heads=%d head_dim=%d"
        slots kv_heads head_dim;
    (* The last row is scratch: what addresses nothing is written there. *)
    let shape = [| slots + 1; kv_heads; head_dim |] in
    { keys = Nx.zeros dtype shape; values = Nx.zeros dtype shape }

  let map f { keys; values } =
    let keys = f keys in
    let values = f values in
    { keys; values }

  let map2 f c c' =
    let keys = f c.keys c'.keys in
    let values = f c.values c'.values in
    { keys; values }

  let iter f { keys; values } =
    f keys;
    f values

  let fold f acc { keys; values } = f "values" (f "keys" acc keys) values

  let fold2 f acc c c' =
    f "values" (f "keys" acc c.keys c'.keys) c.values c'.values

  let names _ = { keys = "keys"; values = "values" }

  (* One cache per block, in block order: a decoder's carried state. *)
  module List = struct
    type 'a cache = 'a t
    type 'a t = 'a cache list

    module L = Stdlib.List

    let one_map = map
    and one_map2 = map2
    and one_iter = iter
    and one_fold = fold
    and one_fold2 = fold2
    and one_names = names

    let same_length fn l l' =
      if L.compare_lengths l l' <> 0 then
        Printf.ksprintf invalid_arg
          "Attention.Cache.List.%s: lists differ in length" fn

    let map f l = L.map (one_map f) l

    let map2 f l l' =
      same_length "map2" l l';
      L.map2 (one_map2 f) l l'

    let iter f l = L.iter (one_iter f) l
    let under i path = string_of_int i ^ "." ^ path

    let fold f acc l =
      snd
        (L.fold_left
           (fun (i, acc) c ->
             (i + 1, one_fold (fun path -> f (under i path)) acc c))
           (0, acc) l)

    let fold2 f acc l l' =
      same_length "fold2" l l';
      snd
        (L.fold_left2
           (fun (i, acc) c c' ->
             (i + 1, one_fold2 (fun path -> f (under i path)) acc c c'))
           (0, acc) l l')

    let names l = L.mapi (fun i c -> one_map (under i) (one_names c)) l
  end
end

(* Cached attention *)

let cached ~head_dim ?rope p cache index x =
  let batch = Cache_index.batch index and seq = Cache_index.seq index in
  let embed = (Nx.shape p.q.Linear.w).(0) in
  if Nx.shape x <> [| batch; seq; embed |] then
    Printf.ksprintf invalid_arg
      "Attention.cached: input must have shape [%d; %d; %d]" batch seq embed;
  let heads, kv_heads = geometry ~fn:"cached" ~head_dim p in
  (match (Nx.shape cache.Cache.keys, Nx.shape cache.Cache.values) with
  | [| n; h; d |], [| n'; h'; d' |]
    when n = n' && n > 0 && h = kv_heads && h' = kv_heads && d = head_dim
         && d' = head_dim ->
      ()
  | _ ->
      Printf.ksprintf invalid_arg
        "Attention.cached: the cache must have shape [slots + 1; %d; %d]"
        kv_heads head_dim);
  let q = split ~heads ~head_dim (Linear.apply p.q x) in
  (* Keys and values as the index takes them, [batch; seq; kv_heads; head_dim].
     Keys are stored rotated: a slot is valid at the position it was written
     for. *)
  let tokens y = Nx.reshape [| batch; seq; kv_heads; head_dim |] y in
  let q, k =
    match rope with
    | None -> (q, tokens (Linear.apply p.k x))
    | Some t ->
        let pos = Cache_index.positions index in
        let k = split ~heads:kv_heads ~head_dim (Linear.apply p.k x) in
        (Rope.apply t ~pos q, Nx.swapaxes 1 2 (Rope.apply t ~pos k))
  in
  let extend values leaf =
    let seen, leaf = Cache_index.extend index values leaf in
    (Nx.swapaxes 1 2 seen, leaf)
  in
  let k, keys = extend k cache.Cache.keys in
  let v, values = extend (tokens (Linear.apply p.v x)) cache.Cache.values in
  let out = attend ~kv_heads ~mask:(Cache_index.mask index) q k v in
  (Linear.apply p.out out, { Cache.keys; values })
