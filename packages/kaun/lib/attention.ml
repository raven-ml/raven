(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type ('a, 'd) t = {
  q : ('a, 'd) Linear.t;
  k : ('a, 'd) Linear.t;
  v : ('a, 'd) Linear.t;
  out : ('a, 'd) Linear.t;
}

let walk c { q; k; v; out } =
  let open Nx.Ptree.Walk in
  let q = field c "q" Linear.walk q in
  let k = field c "k" Linear.walk k in
  let v = field c "v" Linear.walk v in
  let out = field c "out" Linear.walk out in
  { q; k; v; out }

let make ?w_init ?bias_init ?bias ?q_dim ?kv_dim ~key ~embed_dim dtype =
  let q_dim = Option.value q_dim ~default:embed_dim in
  let kv_dim = Option.value kv_dim ~default:embed_dim in
  if embed_dim <= 0 || q_dim <= 0 || kv_dim <= 0 then
    Printf.ksprintf invalid_arg
      "Attention.make: embed_dim, q_dim and kv_dim must be positive, got \
       embed_dim=%d q_dim=%d kv_dim=%d"
      embed_dim q_dim kv_dim;
  let keys = Nx.Rng.split ~n:4 key in
  let proj i ~inputs ~outputs =
    Linear.make ?w_init ?bias_init ?bias ~key:keys.(i) ~inputs ~outputs dtype
  in
  {
    q = proj 0 ~inputs:embed_dim ~outputs:q_dim;
    k = proj 1 ~inputs:embed_dim ~outputs:kv_dim;
    v = proj 2 ~inputs:embed_dim ~outputs:kv_dim;
    out = proj 3 ~inputs:q_dim ~outputs:embed_dim;
  }

let init ~key ~embed_dim = make ~key ~embed_dim Nx.float32

let shape_to_string s =
  "[" ^ String.concat "; " (Array.to_list (Array.map string_of_int s)) ^ "]"

(* Whether [shape] broadcasts to [onto], aligned on the right. *)
let broadcasts ~onto shape =
  let n = Array.length onto and r = Array.length shape in
  r <= n
  && Array.for_all Fun.id
       (Array.mapi (fun i d -> d = 1 || d = onto.(n - r + i)) shape)

(* The dtypes {!Nx.attention} takes. *)
let check_dtype : type b. fn:string -> (float, b) Nx.dtype -> unit =
 fun ~fn dt ->
  match dt with
  | Float16 | Bfloat16 | Float32 | Float64 -> ()
  | Float8_e4m3fn | Float8_e5m2 | Float4_e2m1fn ->
      Printf.ksprintf invalid_arg
        "Attention.%s: %s is not float16, bfloat16, float32 or float64" fn
        (Format.asprintf "%a" Nx.pp_dtype dt)

(* {!Nx.attention} of [q] [[lead; hq; tq; dk]] over [k] and [v], in
   float64 for a float64 [q] and float32 otherwise, with the sinks [sinks],
   which broadcast to the rows [[lead; hq; tq]]. A sink is one more key of
   score [σ] and value zero: it scales a row's softmax over its keys by
   [Σ e^s / (Σ e^s + e^σ)], which is [sigmoid (lse - σ)], [0] for a row that
   sees no key. The product computes in the accumulator and rounds once. *)
let attention (type b) ?mask ?scale ?sinks (q : (float, b, _) Nx.t) k v =
  let within (type a) (acc : (float, a) Nx.dtype) =
    let o, lse = Nx.attention ?scale ?mask ~acc ~kv:(Nx.Dense { k; v }) q in
    match sinks with
    | None -> o
    | Some sinks ->
        let kept = Nx.sigmoid (Nx.sub lse (Nx.cast acc sinks)) in
        Nx.cast (Nx.dtype o)
          (Nx.mul (Nx.cast acc o) (Nx.unsqueeze ~axes:[ -1 ] kept))
  in
  match Nx.dtype q with
  | Float64 -> within Nx.float64
  | _ -> within Nx.float32

let scaled_dot_product_attention ?mask ?scale ?sinks q k v =
  let fn = "scaled_dot_product_attention" in
  check_dtype ~fn (Nx.dtype q);
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
  if qs.(qr - 1) = 0 && scale = None then
    invalid_arg
      "Attention.scaled_dot_product_attention: q and k of no feature need a \
       scale";
  let batch s = Array.sub s 0 (Array.length s - 2) in
  let lead =
    try Nx.broadcast_shapes [ batch qs; batch ks; batch vs ]
    with Invalid_argument _ ->
      Printf.ksprintf invalid_arg
        "Attention.scaled_dot_product_attention: the batch axes of q %s, k %s \
         and v %s do not broadcast"
        (shape_to_string qs) (shape_to_string ks) (shape_to_string vs)
  in
  let n = qs.(qr - 2) and m = ks.(kr - 2) in
  Option.iter
    (fun sinks ->
      let rows = Array.append lead [| n |] in
      if not (broadcasts ~onto:rows (Nx.shape sinks)) then
        Printf.ksprintf invalid_arg
          "Attention.scaled_dot_product_attention: sinks of shape %s do not \
           broadcast to %s, the scores without their last axis"
          (shape_to_string (Nx.shape sinks))
          (shape_to_string rows))
    sinks;
  (* The last batch axis is the heads axis of {!Nx.attention}; without batch
     axes there is one head. *)
  let heads = if lead = [||] then [| 1 |] else lead in
  let stretch x rows = Nx.broadcast_to (Array.append heads rows) x in
  let o =
    attention ?mask ?scale ?sinks
      (stretch q [| n; qs.(qr - 1) |])
      (stretch k [| m; ks.(kr - 1) |])
      (stretch v [| m; vs.(vr - 1) |])
  in
  Nx.reshape (Array.append lead [| n; vs.(vr - 1) |]) o

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

(* Key-value head [h] serves query heads [h * groups] to
   [h * groups + groups - 1], as {!Nx.attention} reads them: no key is
   repeated. *)
let attend ?mask ?scale ?sinks q k v =
  check_dtype ~fn:"attend" (Nx.dtype q);
  let batch, heads, d =
    match Nx.shape q with
    | [| batch; heads; _; d |] -> (batch, heads, d)
    | _ ->
        invalid_arg "Attention.attend: q must have shape [batch; heads; n; d]"
  in
  let kv_heads =
    match (Nx.shape k, Nx.shape v) with
    | [| b; h; m; d' |], [| b'; h'; m'; _ |]
      when b = batch && b' = batch && h = h' && m = m' && d' = d ->
        h
    | _ ->
        Printf.ksprintf invalid_arg
          "Attention.attend: k must have shape [%d; kv_heads; m; %d] and v \
           [%d; kv_heads; m; dv]"
          batch d batch
  in
  if (kv_heads = 0 && heads <> 0) || (kv_heads > 0 && heads mod kv_heads <> 0)
  then
    Printf.ksprintf invalid_arg
      "Attention.attend: %d key-value heads do not divide %d query heads"
      kv_heads heads;
  if d = 0 && scale = None then
    invalid_arg "Attention.attend: q and k of no feature need a scale";
  let sinks =
    Option.map
      (fun sinks ->
        if Nx.shape sinks <> [| heads |] then
          Printf.ksprintf invalid_arg
            "Attention.attend: sinks must have shape [%d]" heads;
        Nx.reshape [| heads; 1 |] sinks)
      sinks
  in
  attention ?mask ?scale ?sinks q k v

let merge ?sum l y =
  match Nx.shape y with
  | [| batch; heads; seq; d |] ->
      Linear.apply ?sum l
        (Nx.reshape [| batch; seq; heads * d |] (Nx.swapaxes 1 2 y))
  | _ -> invalid_arg "Attention.merge: y must have shape [batch; heads; seq; d]"

let apply ~head_dim ?mask ?rope ?sum p x =
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
  check_dtype ~fn:"apply" (Nx.dtype x);
  check_geometry ~fn:"apply" ~head_dim p;
  let seq = shape.(rank - 2) in
  let batch = Array.fold_left ( * ) 1 (Array.sub shape 0 (rank - 2)) in
  (* Leading axes fold into one batch axis for the pieces and unfold after. *)
  let x = Nx.reshape [| batch; seq; embed |] x in
  let q = split ~head_dim p.q x and k = split ~head_dim p.k x in
  let v = split ~head_dim p.v x in
  let q, k =
    match rope with
    | None -> (q, k)
    | Some t ->
        let at = Nx.beside x in
        let pos = Nx.reshape [| 1; seq |] (Nx.arange_at at Nx.int64 0 seq 1) in
        (Rope.apply t ~pos q, Rope.apply t ~pos k)
  in
  Nx.reshape shape (merge ?sum p.out (attend ?mask q k v))

(* Key-value cache *)

module Cache = struct
  type ('a, _) t = { keys : 'a; values : 'a }

  let make ~slots ~kv_heads ~head_dim dtype at =
    if slots < 0 || kv_heads <= 0 || head_dim <= 0 then
      Printf.ksprintf invalid_arg
        "Attention.Cache.make: slots must not be negative and kv_heads and \
         head_dim must be positive, got slots=%d kv_heads=%d head_dim=%d"
        slots kv_heads head_dim;
    let pool () = Nx.zeros_at at dtype [| slots; kv_heads; head_dim |] in
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

let cached ~head_dim ?rope ?sum p cache index x =
  let batch = Cache_index.batch index and seq = Cache_index.seq index in
  let embed = (Nx.shape p.q.Linear.w).(0) in
  if Nx.shape x <> [| batch; seq; embed |] then
    Printf.ksprintf invalid_arg
      "Attention.cached: input must have shape [%d; %d; %d]" batch seq embed;
  check_dtype ~fn:"cached" (Nx.dtype x);
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
  (merge ?sum p.out (attend ~mask:(Cache_index.visible index) q k v), cache)
