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
    let scores =
      match mask with
      | None -> scores
      | Some m -> Nx.where m scores (Nx.scalar_like scores Float.neg_infinity)
    in
    Fn.softmax scores
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
  let grouped t = Nx.reshape [| batch; kv_heads; 1; m; d |] (Nx.contiguous t) in
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
      (* The diagonal stays whatever [valid] says, so a padded query still
         admits one key and its row of the softmax is finite. *)
      Nx.logical_or
        (Nx.logical_and
           (Nx.reshape [| 1; seq; seq |] tri)
           (Nx.reshape [| batch; 1; seq |] valid))
        (Nx.reshape [| 1; seq; seq |] (Nx.equal row col))

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

(* Addressing *)

module Span = struct
  type t = { pos : Nx.int32_t; slots : Nx.int32_t }

  let make ~pos ~slots =
    (match (Nx.shape pos, Nx.shape slots) with
    | [| b; s |], [| b'; c |] when b = b' && b > 0 && s > 0 && c > 0 -> ()
    | _ ->
        invalid_arg
          "Attention.Span.make: pos must have shape [batch; seq] and slots \
           [batch; context], none of them empty");
    { pos; slots }

  let rows ~context lens =
    let batch = Array.length lens in
    if batch = 0 then invalid_arg "Attention.Span.rows: no rows";
    if context <= 0 then
      Printf.ksprintf invalid_arg
        "Attention.Span.rows: context must be positive, got %d" context;
    Array.iter
      (fun n ->
        if n < 0 || n > context then
          Printf.ksprintf invalid_arg
            "Attention.Span.rows: a row of %d tokens does not fit a context of \
             %d"
            n context)
      lens;
    let seq = Array.fold_left max 1 lens in
    (* Rows are padded on the left, so the last column is every row's last
       token. *)
    let pos =
      Array.init (batch * seq) (fun t ->
          let b = t / seq and i = t mod seq in
          Int32.of_int (max (-1) (i - (seq - lens.(b)))))
    in
    let slots = Array.init (batch * context) Int32.of_int in
    {
      pos = Nx.create Nx.int32 [| batch; seq |] pos;
      slots = Nx.create Nx.int32 [| batch; context |] slots;
    }

  let positions t =
    let context = Int32.of_int (Nx.dim 1 t.slots) in
    let inside =
      Nx.logical_and (Nx.greater_equal_s t.pos 0l) (Nx.less_s t.pos context)
    in
    Nx.where inside t.pos (Nx.zeros_like t.pos)

  let advance t =
    (* Any negative position is padding: a row of it advances to 0. *)
    let last = Nx.maximum_s (Nx.max ~axes:[ 1 ] ~keepdims:true t.pos) (-1l) in
    { t with pos = Nx.add_s last 1l }

  let map f { pos; slots } =
    let pos = f pos in
    let slots = f slots in
    { pos; slots }

  let map2 f a b =
    let pos = f a.pos b.pos in
    let slots = f a.slots b.slots in
    { pos; slots }

  let iter f { pos; slots } =
    f pos;
    f slots
end

(* Key-value cache *)

module Cache = struct
  type 'a t = { keys : 'a; values : 'a }

  let make ~slots ~kv_heads ~head_dim dtype =
    if slots <= 0 || kv_heads <= 0 || head_dim <= 0 then
      Printf.ksprintf invalid_arg
        "Attention.Cache.make: slots, kv_heads and head_dim must be positive, \
         got slots=%d kv_heads=%d head_dim=%d"
        slots kv_heads head_dim;
    let shape = [| slots; kv_heads; head_dim |] in
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

(* Routes *)

type route = {
  batch : int;
  seq : int;
  context : int;
  slots : int;
  positions : Nx.int32_t; (* [batch; seq], effective *)
  source : Nx.int32_t; (* [slots]: the token written to each slot, clamped *)
  written : (bool, Nx.bool_elt) Nx.t; (* [slots; 1; 1] *)
  read : Nx.int32_t; (* [batch * context]: the slot of each column, clamped *)
  live : (bool, Nx.bool_elt) Nx.t; (* [batch * context; 1; 1] *)
  mask : (bool, Nx.bool_elt) Nx.t; (* [batch; seq; context] *)
}

let route ~slots (span : Span.t) =
  if slots <= 0 then
    Printf.ksprintf invalid_arg
      "Attention.route: slots must be positive, got %d" slots;
  let batch = Nx.dim 0 span.pos and seq = Nx.dim 1 span.pos in
  let context = Nx.dim 1 span.slots in
  let tokens = batch * seq in
  let positions = Span.positions span in
  let inside =
    Nx.logical_and
      (Nx.greater_equal_s span.pos 0l)
      (Nx.less_s span.pos (Int32.of_int context))
  in
  let none = Nx.full Nx.int32 [| 1; 1 |] (-1l) in
  (* The slot each token writes, or none: an address outside its range addresses
     nothing, so no index below leaves its range. *)
  let target =
    Nx.where inside
      (Nx.take_along_axis ~axis:1 ~indices:positions span.slots)
      none
  in
  (* Inverted once per call: the last token, in row-major order, aimed at each
     slot. Every layer's write is then one select over the pool. *)
  let hit =
    Nx.equal
      (Nx.reshape [| 1; tokens |] target)
      (Nx.reshape [| slots; 1 |] (Nx.arange Nx.int32 0 slots 1))
  in
  let token = Nx.reshape [| 1; tokens |] (Nx.arange Nx.int32 0 tokens 1) in
  let writer = Nx.max ~axes:[ 1 ] (Nx.where hit token none) in
  let flat = Nx.reshape [| batch * context |] (Nx.contiguous span.slots) in
  let allocated =
    Nx.logical_and
      (Nx.greater_equal_s flat 0l)
      (Nx.less_s flat (Int32.of_int slots))
  in
  let column = Nx.arange Nx.int32 0 context 1 in
  (* A column past every position of its row is seen by no query of the row. *)
  let horizon = Nx.max ~axes:[ 1 ] ~keepdims:true positions in
  let within = Nx.less_equal (Nx.reshape [| 1; context |] column) horizon in
  {
    batch;
    seq;
    context;
    slots;
    positions;
    source = Nx.maximum_s writer 0l;
    written = Nx.reshape [| slots; 1; 1 |] (Nx.greater_equal_s writer 0l);
    read = Nx.clamp ~min:0l ~max:(Int32.of_int (slots - 1)) flat;
    live =
      Nx.reshape
        [| batch * context; 1; 1 |]
        (Nx.logical_and allocated (Nx.reshape [| batch * context |] within));
    mask =
      Nx.less_equal
        (Nx.reshape [| 1; 1; context |] column)
        (Nx.reshape [| batch; seq; 1 |] positions);
  }

let cached ~head_dim ?rope p cache r x =
  let shape = Nx.shape x in
  let embed = (Nx.shape p.q.Linear.w).(0) in
  (match shape with
  | [| b; s; e |] when b = r.batch && s = r.seq && e = embed -> ()
  | _ ->
      Printf.ksprintf invalid_arg
        "Attention.cached: input must have shape [%d; %d; %d]" r.batch r.seq
        embed);
  let heads, kv_heads = geometry ~fn:"cached" ~head_dim p in
  let pool = [| r.slots; kv_heads; head_dim |] in
  if Nx.shape cache.Cache.keys <> pool || Nx.shape cache.Cache.values <> pool
  then
    Printf.ksprintf invalid_arg
      "Attention.cached: the cache must have shape [%d; %d; %d]" r.slots
      kv_heads head_dim;
  let tokens = r.batch * r.seq in
  let q = split ~heads ~head_dim (Linear.apply p.q x) in
  let k = split ~heads:kv_heads ~head_dim (Linear.apply p.k x) in
  let q, k =
    match rope with
    | None -> (q, k)
    | Some t ->
        (Rope.apply t ~pos:r.positions q, Rope.apply t ~pos:r.positions k)
  in
  (* Keys are stored rotated: a slot is valid at the position it was written
     for. *)
  let k_rows =
    Nx.reshape
      [| tokens; kv_heads; head_dim |]
      (Nx.contiguous (Nx.swapaxes 1 2 k))
  in
  let v_rows =
    Nx.reshape
      [| tokens; kv_heads; head_dim |]
      (Nx.contiguous (Linear.apply p.v x))
  in
  (* Every slot takes its writer's row or keeps its own: one select over the
     pool, which a compiler performs in place on a donated cache. *)
  let write old rows =
    Nx.where r.written (Nx.take ~axis:0 ~indices:r.source rows) old
  in
  let keys = write cache.Cache.keys k_rows in
  let values = write cache.Cache.values v_rows in
  (* Columns no query of the row may see read as zero, so they contribute
     exactly zero and not [0 * v]. *)
  let read leaf =
    let win = Nx.take ~axis:0 ~indices:r.read leaf in
    let win = Nx.where r.live win (Nx.scalar_like win 0.0) in
    Nx.swapaxes 1 2
      (Nx.reshape [| r.batch; r.context; kv_heads; head_dim |] win)
  in
  let out = attend ~kv_heads ~mask:r.mask q (read keys) (read values) in
  (Linear.apply p.out out, { Cache.keys; values })
