(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type 'b params = {
  q : 'b Linear.params;
  k : 'b Linear.params;
  v : 'b Linear.params;
  out : 'b Linear.params;
}

type t = Nx.float32_elt params

let map (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t) { q; k; v; out } =
  {
    q = Linear.map f q;
    k = Linear.map f k;
    v = Linear.map f v;
    out = Linear.map f out;
  }

let map2 (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t) p p' =
  {
    q = Linear.map2 f p.q p'.q;
    k = Linear.map2 f p.k p'.k;
    v = Linear.map2 f p.v p'.v;
    out = Linear.map2 f p.out p'.out;
  }

let iter (f : 'a 'c. ('a, 'c) Nx.t -> unit) { q; k; v; out } =
  Linear.iter f q;
  Linear.iter f k;
  Linear.iter f v;
  Linear.iter f out

let names p =
  let sub prefix l = List.map (fun n -> prefix ^ "." ^ n) (Linear.names l) in
  sub "q" p.q @ sub "k" p.k @ sub "v" p.v @ sub "out" p.out

let make ?w_init ?bias_init ?bias ~embed_dim dtype =
  if embed_dim <= 0 then
    Printf.ksprintf invalid_arg
      "Attention.make: embed_dim must be positive, got %d" embed_dim;
  let proj () =
    Linear.make ?w_init ?bias_init ?bias ~inputs:embed_dim ~outputs:embed_dim
      dtype
  in
  { q = proj (); k = proj (); v = proj (); out = proj () }

let init ~embed_dim = make ~embed_dim Nx.float32

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
  let scores = Nx.mul_s (Nx.matmul q (Nx.swapaxes (kr - 2) (kr - 1) k)) scale in
  let scores =
    match mask with
    | None -> scores
    | Some m -> Nx.where m scores (Nx.scalar_like scores Float.neg_infinity)
  in
  Nx.matmul (Fn.softmax scores) v

let apply ?(num_heads = 1) ?(causal = false) p x =
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
  if num_heads <= 0 then
    Printf.ksprintf invalid_arg
      "Attention.apply: num_heads must be positive, got %d" num_heads;
  if embed mod num_heads <> 0 then
    Printf.ksprintf invalid_arg
      "Attention.apply: num_heads (%d) must divide the embedding dimension (%d)"
      num_heads embed;
  let head_dim = embed / num_heads in
  let seq = shape.(rank - 2) in
  (* [..., seq, embed] -> [..., num_heads, seq, head_dim]: heads become a batch
     axis so the attention core runs each head independently. *)
  let split t =
    let s =
      Array.append (Array.sub shape 0 (rank - 1)) [| num_heads; head_dim |]
    in
    Nx.swapaxes (rank - 2) (rank - 1) (Nx.reshape s t)
  in
  let merge t =
    (* The swap leaves the tensor non-contiguous; reshape needs a copy. *)
    Nx.reshape shape (Nx.contiguous (Nx.swapaxes (rank - 2) (rank - 1) t))
  in
  let q = split (Linear.apply p.q x) in
  let k = split (Linear.apply p.k x) in
  let v = split (Linear.apply p.v x) in
  let mask =
    if not causal then None
    else
      (* [mask.(i).(j)] is [j <= i]: query [i] sees keys up to itself. *)
      let idx = Nx.arange Nx.int32 0 seq 1 in
      Some
        (Nx.less_equal
           (Nx.reshape [| 1; seq |] idx)
           (Nx.reshape [| seq; 1 |] idx))
  in
  Linear.apply p.out (merge (scaled_dot_product_attention ?mask q k v))
