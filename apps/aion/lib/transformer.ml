(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The epsilon of every layer norm, the query-key norms included: upstream's FM
   builds them as LayerNorm(eps=1e-6) (aion/fourm/fm.py:1020). *)
let eps = 1e-6

(* Attention with query-key norm: the projections, then a layer norm over each
   head's queries and keys. *)
type ('a, 'd) attention = {
  proj : ('a, 'd) Kaun.Attention.t;
  q_norm : ('a, 'd) Kaun.Layer_norm.t;
  k_norm : ('a, 'd) Kaun.Layer_norm.t;
}
[@@deriving ptree]

(* SwiGLU: [fc2 (silu (fc1 x) * fc3 x)]. *)
type ('a, 'd) mlp = {
  fc1 : ('a, 'd) Kaun.Linear.t;
  fc2 : ('a, 'd) Kaun.Linear.t;
  fc3 : ('a, 'd) Kaun.Linear.t;
}
[@@deriving ptree]

type ('a, 'd) encoder_block = {
  norm1 : ('a, 'd) Kaun.Layer_norm.t;
  attn : ('a, 'd) attention;
  norm2 : ('a, 'd) Kaun.Layer_norm.t;
  mlp : ('a, 'd) mlp;
}
[@@deriving ptree]

type ('a, 'd) decoder_block = {
  d_norm1 : ('a, 'd) Kaun.Layer_norm.t;
  self_attn : ('a, 'd) attention;
  query_norm : ('a, 'd) Kaun.Layer_norm.t;
  context_norm : ('a, 'd) Kaun.Layer_norm.t;
  cross_attn : ('a, 'd) attention;
  d_norm2 : ('a, 'd) Kaun.Layer_norm.t;
  d_mlp : ('a, 'd) mlp;
}
[@@deriving ptree]

(* One modality's embeddings: the encoder's token table [vocab; dim], the
   encoder's and the decoder's positional tables [tokens; dim], the modality
   embedding [dim], which the encoder and the decoder share, and the decoder's
   output projection. *)
type ('a, 'd) embedding = {
  modality : Modality.t;
  table : ('a, 'd) Kaun.Embedding.t;
  enc_pos : 'a;
  dec_pos : 'a;
  mod_emb : 'a;
  logits : ('a, 'd) Kaun.Linear.t;
}
[@@deriving ptree]

type ('a, 'd) t = {
  embeddings : ('a, 'd) embedding list;
  encoder : ('a, 'd) encoder_block list;
  encoder_norm : ('a, 'd) Kaun.Layer_norm.t;
  context : ('a, 'd) Kaun.Linear.t;
  decoder : ('a, 'd) decoder_block list;
  decoder_norm : ('a, 'd) Kaun.Layer_norm.t;
  mask_token : 'a;
}
[@@deriving ptree]

let modalities t = List.map (fun (e : _ embedding) -> e.modality) t.embeddings

(* The arguments of the computations. *)

type ('a, 'd) encoding = { context : 'a; lengths : 'd Nx.int64_t }
[@@deriving ptree]

module Encoding = struct
  type ('a, 'd) t = ('a, 'd) encoding

  let walk = walk_encoding
end

type 'd target = { modality : Modality.t; positions : 'd Nx.int64_t option }
[@@deriving ptree]

let invalid fmt = Format.kasprintf invalid_arg fmt
let refusal fmt = Format.kasprintf (fun m -> Invalid_argument m) fmt

(* Refuses the first modality of [ms] that an earlier one repeats. *)
let distinct fn what ms =
  let rec go seen = function
    | [] -> ()
    | m :: rest ->
        if List.mem m seen then
          invalid "Aion.%s: %s %s given twice" fn what (Modality.key m);
        go (m :: seen) rest
  in
  go [] ms

(* Reading the weights. Upstream's names (aion/fourm/fm.py, fm_utils.py): a
   linear layer is [out; in], as kaun's is; self-attention fuses its query, key
   and value projections in one [qkv] of [3 dim] rows, in that order, and
   cross-attention its key and value in one [kv]. Every layer norm stores a
   shift, which upstream's adds: a zero buffer in AION-1's weights, which train
   none. A modality's embedding holds a positional table [1; tokens; dim] in the
   encoder and another in the decoder, each read where upstream reads it, and
   one modality embedding stored under the decoder's name, which the file's
   metadata aliases to the encoder's. *)

(* Whether [a] holds [name]: a probe through [names], which [import] does not
   note as a read. *)
let has a name = List.mem name (Nx_io.Archive.names a)
let weight a name shape = Nx_io.Archive.tensor ~shape Nx.float32 name a

(* A torch [out; in] weight, kaun's layout. *)
let linear ?b a name ~inputs ~outputs : _ Kaun.Linear.t =
  { w = weight a (name ^ ".weight") [| outputs; inputs |]; b }

let failure fmt = Printf.ksprintf failwith fmt

let norm a name n : _ Kaun.Layer_norm.t =
  let gamma = weight a (name ^ ".weight") [| n |] in
  { gamma; beta = Some (weight a (name ^ ".bias") [| n |]) }

let rows w lo n = Nx.slice [ R (lo, lo + n); A ] w

let attention a name ~dim ~heads ~fused =
  let head_dim = dim / heads in
  let q, k, v =
    if fused then
      let qkv = weight a (name ^ ".qkv.weight") [| 3 * dim; dim |] in
      (rows qkv 0 dim, rows qkv dim dim, rows qkv (2 * dim) dim)
    else
      let kv = weight a (name ^ ".kv.weight") [| 2 * dim; dim |] in
      ( weight a (name ^ ".q.weight") [| dim; dim |],
        rows kv 0 dim,
        rows kv dim dim )
  in
  let lin w : _ Kaun.Linear.t = { w; b = None } in
  {
    proj =
      {
        q = lin q;
        k = lin k;
        v = lin v;
        out = linear a (name ^ ".proj") ~inputs:dim ~outputs:dim;
      };
    q_norm = norm a (name ^ ".q_norm") head_dim;
    k_norm = norm a (name ^ ".k_norm") head_dim;
  }

let mlp a name ~dim ~hidden =
  {
    fc1 = linear a (name ^ ".fc1") ~inputs:dim ~outputs:hidden;
    fc2 = linear a (name ^ ".fc2") ~inputs:hidden ~outputs:dim;
    fc3 = linear a (name ^ ".fc3") ~inputs:dim ~outputs:hidden;
  }

(* The catalog's model tokens have no published map from its codec's tokens, so
   the transformer leaves the catalog out, and its tensors unread. *)
let modelled m = m <> Modality.Catalog

let embedding a ~dim m =
  let key = Modality.key m in
  let n = Modality.tokens m and vocab = Modality.vocab m in
  let enc = "encoder_embeddings." ^ key and dec = "decoder_embeddings." ^ key in
  let pos name = Nx.reshape [| n; dim |] (weight a name [| 1; n; dim |]) in
  {
    modality = m;
    table = { table = weight a (enc ^ ".token_emb.weight") [| vocab; dim |] };
    enc_pos = pos (enc ^ ".pos_emb");
    dec_pos = pos (dec ^ ".pos_emb");
    mod_emb = Nx.reshape [| dim |] (weight a (dec ^ ".mod_emb") [| 1; 1; dim |]);
    logits = linear a (dec ^ ".to_logits") ~inputs:dim ~outputs:vocab;
  }

(* Tensors no path reads: the decoder's token tables, since the decoder's input
   is the mask token for every modality but the catalog, and the catalog's. *)
let unread name =
  let prefix p = String.starts_with ~prefix:p name in
  prefix "encoder_embeddings.catalog."
  || prefix "decoder_embeddings.catalog."
  || prefix "decoder_embeddings."
     && String.ends_with ~suffix:".token_emb.weight" name

let blocks a prefix make =
  let rec go i =
    if has a (Printf.sprintf "%s.%d.norm1.weight" prefix i) then
      make (Printf.sprintf "%s.%d" prefix i) :: go (i + 1)
    else []
  in
  go 0

let read a =
  (* The width, the heads and the feed-forward's width, read from the stored
     shapes of three weights before the others are asked for at theirs. *)
  let stored name = Nx.dim 0 (Nx_io.Archive.tensor Nx.float32 name a) in
  let dim = stored "encoder_norm.weight" in
  let heads = dim / stored "encoder.0.attn.q_norm.weight" in
  let hidden = stored "encoder.0.mlp.fc1.weight" in
  let embeddings =
    List.filter_map
      (fun m ->
        let name =
          "encoder_embeddings." ^ Modality.key m ^ ".token_emb.weight"
        in
        if modelled m && has a name then Some (embedding a ~dim m) else None)
      Modality.all
  in
  if embeddings = [] then failure "Aion.Transformer.of_archive: no modality";
  let encoder =
    blocks a "encoder" (fun p ->
        {
          norm1 = norm a (p ^ ".norm1") dim;
          attn = attention a (p ^ ".attn") ~dim ~heads ~fused:true;
          norm2 = norm a (p ^ ".norm2") dim;
          mlp = mlp a (p ^ ".mlp") ~dim ~hidden;
        })
  in
  let decoder =
    blocks a "decoder" (fun p ->
        {
          d_norm1 = norm a (p ^ ".norm1") dim;
          self_attn = attention a (p ^ ".self_attn") ~dim ~heads ~fused:true;
          query_norm = norm a (p ^ ".query_norm") dim;
          context_norm = norm a (p ^ ".context_norm") dim;
          cross_attn = attention a (p ^ ".cross_attn") ~dim ~heads ~fused:false;
          d_norm2 = norm a (p ^ ".norm2") dim;
          d_mlp = mlp a (p ^ ".mlp") ~dim ~hidden;
        })
  in
  {
    embeddings;
    encoder;
    encoder_norm = norm a "encoder_norm" dim;
    context =
      linear a "decoder_proj_context" ~inputs:dim ~outputs:dim
        ~b:(weight a "decoder_proj_context.bias" [| dim |]);
    decoder;
    decoder_norm = norm a "decoder_norm" dim;
    mask_token = Nx.reshape [| dim |] (weight a "mask_token" [| 1; 1; dim |]);
  }

let of_archive a = Nx_io.Archive.import ~unread read a

(* Initialising *)

type arch = {
  width : int;
  heads : int;
  blocks : int * int;
  hidden : int;
  modalities : Modality.t list;
}

let init ~key a =
  let encoders, decoders = a.blocks in
  if a.width < 1 || a.heads < 1 || a.hidden < 1 || encoders < 0 || decoders < 0
  then
    invalid_arg
      "Aion.Transformer.init: a width or heads not positive, or blocks \
       negative";
  if a.width mod a.heads <> 0 then
    invalid "Aion.Transformer.init: %d heads do not divide the width %d"
      a.heads a.width;
  if a.modalities = [] then invalid_arg "Aion.Transformer.init: no modality";
  distinct "Transformer.init" "modality" a.modalities;
  let at = Nx.Rng.placement key in
  let n = ref 0 in
  let key () =
    incr n;
    Nx.Rng.fold_in key !n
  in
  let d = a.width and f32 = Nx.float32 in
  let linear i o =
    Kaun.Linear.make ~bias:false ~key:(key ()) ~inputs:i ~outputs:o f32
  in
  let norm n = Kaun.Layer_norm.init ~dim:n at in
  let normal shape =
    Kaun.Init.normal ~stddev:0.02 (key ()) ~fan_in:1 ~fan_out:1 f32 shape
  in
  let attention () =
    {
      proj = Kaun.Attention.make ~bias:false ~key:(key ()) ~embed_dim:d f32;
      q_norm = norm (d / a.heads);
      k_norm = norm (d / a.heads);
    }
  in
  let mlp () =
    {
      fc1 = linear d a.hidden;
      fc2 = linear a.hidden d;
      fc3 = linear d a.hidden;
    }
  in
  let embedding m =
    let n = Modality.tokens m and vocab = Modality.vocab m in
    {
      modality = m;
      table = Kaun.Embedding.init ~key:(key ()) ~vocab ~dim:d;
      enc_pos = normal [| n; d |];
      dec_pos = normal [| n; d |];
      mod_emb = normal [| d |];
      logits = linear d vocab;
    }
  in
  let encoder _ =
    { norm1 = norm d; attn = attention (); norm2 = norm d; mlp = mlp () }
  in
  let decoder _ =
    {
      d_norm1 = norm d;
      self_attn = attention ();
      query_norm = norm d;
      context_norm = norm d;
      cross_attn = attention ();
      d_norm2 = norm d;
      d_mlp = mlp ();
    }
  in
  {
    embeddings = List.map embedding a.modalities;
    encoder = List.init encoders encoder;
    encoder_norm = norm d;
    context = Kaun.Linear.init ~key:(key ()) ~inputs:d ~outputs:d;
    decoder = List.init decoders decoder;
    decoder_norm = norm d;
    mask_token = normal [| d |];
  }

(* Computing *)

(* The names of the refusing functions. *)
let encode_fn = "Transformer.encode"
let decode_fn = "Transformer.decode"
let dim t = Nx.dim 0 t.mask_token

let find fn t m =
  match
    List.find_opt (fun (e : _ embedding) -> e.modality = m) t.embeddings
  with
  | Some e -> e
  | None -> invalid "Aion.%s: no weights for %s" fn (Modality.key m)

let attend ?mask ?lengths a x context =
  let head_dim = Nx.dim 0 a.q_norm.gamma in
  let split l y = Kaun.Attention.split ~head_dim l y in
  let q = Kaun.Layer_norm.apply ~eps a.q_norm (split a.proj.q x) in
  let k = Kaun.Layer_norm.apply ~eps a.k_norm (split a.proj.k context) in
  let v = split a.proj.v context in
  Kaun.Attention.merge a.proj.out
    (Kaun.Attention.attend ?mask ?lengths q (Nx.Dense { k; v }))

let swiglu m x =
  Kaun.Linear.apply m.fc2
    (Nx.mul
       (Kaun.Fn.silu (Kaun.Linear.apply m.fc1 x))
       (Kaun.Linear.apply m.fc3 x))

let ln n x = Kaun.Layer_norm.apply ~eps n x

(* Encoding *)

(* An input's embeddings, its tokens and presence checked as [b; n]. *)
let input t (i : _ Tokens.t) =
  let e = find encode_fn t i.modality in
  let key = Modality.key i.modality and n = Modality.tokens i.modality in
  (match Nx.shape i.tokens with
  | [| _; n' |] when n' = n -> ()
  | s ->
      invalid
        "Aion.Transformer.encode: %s tokens of shape %a, [b; %d] expected" key
        Nx.pp_shape s n);
  if Nx.shape i.present <> Nx.shape i.tokens then
    invalid "Aion.Transformer.encode: %s presence of shape %a, %a expected" key
      Nx.pp_shape (Nx.shape i.present) Nx.pp_shape (Nx.shape i.tokens);
  e

(* Every present token of [i] in its modality's vocabulary. *)
let check_vocab (i : _ Tokens.t) =
  let vocab = Modality.vocab i.modality in
  let inside =
    Nx.logical_and
      (Nx.greater_equal_s i.tokens 0L)
      (Nx.less_s i.tokens (Int64.of_int vocab))
  in
  Nx.check Nx.Ptree.unit
    (Nx.logical_or (Nx.logical_not i.present) inside)
    ()
    (fun _ () ->
      refusal "Aion.Transformer.encode: a present %s token is outside [0, %d)"
        (Modality.key i.modality) vocab)

(* The encoder's input: every input's token embeddings and their positional plus
   modality embeddings, concatenated in the inputs' order, then the present
   tokens moved first in that order by a stable sort and the others zeroed;
   [lengths] counts each object's present tokens. *)
let embed t ~batch es (inputs : _ Tokens.t list) present =
  let x =
    Nx.concatenate ~axis:1
      (List.map2
         (fun (e : _ embedding) (i : _ Tokens.t) ->
           Kaun.Embedding.apply e.table i.tokens)
         es inputs)
  in
  let d = dim t in
  let emb =
    Nx.concatenate ~axis:1
      (List.map
         (fun (e : _ embedding) ->
           Nx.broadcast_to
             [| batch; Nx.dim 0 e.enc_pos; d |]
             (Nx.add e.enc_pos (Nx.reshape [| 1; d |] e.mod_emb)))
         es)
  in
  let order = Nx.argsort ~axis:1 (Nx.cast Nx.int32 (Nx.logical_not present)) in
  let lengths = Nx.sum ~axes:[ 1 ] (Nx.cast Nx.int64 present) in
  let n = Nx.dim 1 present in
  let kept =
    Nx.less
      (Nx.reshape [| 1; n; 1 |]
         (Nx.arange_at (Nx.beside lengths) Nx.int64 0 n 1))
      (Nx.reshape [| batch; 1; 1 |] lengths)
  in
  let gather y =
    let y = Nx.take_along_axis ~axis:1 (Nx.reshape [| batch; n; 1 |] order) y in
    Nx.where kept y (Nx.zeros_like y)
  in
  (gather x, gather emb, lengths)

let encode t inputs =
  if inputs = [] then invalid "Aion.Transformer.encode: no input";
  distinct encode_fn "input"
    (List.map (fun (i : _ Tokens.t) -> i.modality) inputs);
  let es = List.map (input t) inputs in
  let batch = Nx.dim 0 (List.hd inputs).tokens in
  List.iter
    (fun (i : _ Tokens.t) ->
      if Nx.dim 0 i.tokens <> batch then
        invalid "Aion.Transformer.encode: %s has %d objects, %d expected"
          (Modality.key i.modality) (Nx.dim 0 i.tokens) batch)
    inputs;
  List.iter check_vocab inputs;
  let present =
    Nx.concatenate ~axis:1 (List.map (fun (i : _ Tokens.t) -> i.present) inputs)
  in
  Nx.check Nx.Ptree.unit
    (Nx.any ~axes:[ 1 ] present)
    ()
    (fun _ () ->
      refusal "Aion.Transformer.encode: an object has no present token");
  let x, emb, lengths = embed t ~batch es inputs present in
  let h =
    List.fold_left
      (fun h b ->
        let a = ln b.norm1 h in
        let h = Nx.add h (attend ~lengths b.attn a a) in
        Nx.add h (swiglu b.mlp (ln b.norm2 h)))
      (Nx.add x emb) t.encoder
  in
  let h = ln t.encoder_norm h in
  { context = Nx.add (Kaun.Linear.apply t.context h) emb; lengths }

(* The mean of each object's context rows before its length. *)
let embedding { context; lengths } =
  let n = Nx.dim 1 context in
  let before =
    Nx.less
      (Nx.reshape [| 1; n; 1 |] (Nx.arange_at (Nx.beside lengths) Nx.int64 0 n 1))
      (Nx.reshape [| -1; 1; 1 |] lengths)
  in
  let sum = Nx.sum ~axes:[ 1 ] (Nx.where before context (Nx.zeros_like context)) in
  Nx.div sum (Nx.reshape [| -1; 1 |] (Nx.cast (Nx.dtype context) lengths))

(* Decoding *)

(* [context] is [b; n; width] and each of the [b] [lengths] in [1, n]. *)
let check_encoding t context lengths =
  match (Nx.shape context, Nx.shape lengths) with
  | [| b; n; w |], [| b' |] when b = b' && w = dim t ->
      let inside =
        Nx.logical_and (Nx.greater_equal_s lengths 1L)
          (Nx.less_equal_s lengths (Int64.of_int n))
      in
      Nx.check Nx.Ptree.unit inside ()
        (fun _ () ->
          refusal "Aion.Transformer.decode: a length outside [1, %d]" n)
  | c, l ->
      invalid
        "Aion.Transformer.decode: context %a and lengths %a, [b; n; %d] and \
         [b] expected"
        Nx.pp_shape c
        Nx.pp_shape l (dim t)

(* A target's positions: [b; k], in [0, n), and distinct within a row, where an
   element repeats when more than one of its row's elements equals it ([k^2]
   comparisons a row). A refusal names the first offending element's
   [row; index] and value. *)
let check_positions ~batch (tg : _ target) =
  let key = Modality.key tg.modality and n = Modality.tokens tg.modality in
  let refuse what i p =
    refusal "Aion.Transformer.decode: %s position %Ld at %a %s" key
      (Nx.item [] p)
      Nx.pp_shape i what
  in
  match tg.positions with
  | None -> ()
  | Some p ->
      (match Nx.shape p with
      | [| b; k |] when b = batch && k >= 1 -> ()
      | s ->
          invalid
            "Aion.Transformer.decode: %s positions of shape %a, [%d; k >= 1] \
             expected"
            key Nx.pp_shape s
            batch);
      let inside =
        Nx.logical_and (Nx.greater_equal_s p 0L) (Nx.less_s p (Int64.of_int n))
      in
      Nx.check Nx.Ptree.tensor inside p
        (refuse (Printf.sprintf "outside [0, %d)" n));
      let k = Nx.dim 1 p in
      let equal =
        Nx.equal
          (Nx.reshape [| batch; k; 1 |] p)
          (Nx.reshape [| batch; 1; k |] p)
      in
      let once = Nx.equal_s (Nx.sum ~axes:[ 2 ] (Nx.cast Nx.int64 equal)) 1L in
      Nx.check Nx.Ptree.tensor once p (refuse "repeats within its row")

(* The decoder's input: for each target, the mask token plus the positional and
   modality embeddings of its positions, [b; k; width], concatenated in the
   targets' order. A token's self-attention sees the tokens of its own target,
   the mask of training for whole targets and the step of random-order
   generation for chosen positions. *)
let decode t { context; lengths } targets =
  if targets = [] then invalid "Aion.Transformer.decode: no target";
  check_encoding t context lengths;
  distinct decode_fn "target"
    (List.map (fun (tg : _ target) -> tg.modality) targets);
  let batch = Nx.dim 0 context and d = dim t in
  let es =
    List.map
      (fun (tg : _ target) ->
        let e = find decode_fn t tg.modality in
        check_positions ~batch tg;
        e)
      targets
  in
  let parts =
    List.map2
      (fun (e : _ embedding) (tg : _ target) ->
        let emb = Nx.add e.dec_pos (Nx.reshape [| 1; d |] e.mod_emb) in
        match tg.positions with
        | None ->
            let n = Nx.dim 0 emb in
            (e, n, Nx.broadcast_to [| batch; n; d |] emb)
        | Some p -> (e, Nx.dim 1 p, Nx.take ~axis:0 p emb))
      es targets
  in
  let y =
    Nx.add t.mask_token
      (Nx.concatenate ~axis:1 (List.map (fun (_, _, x) -> x) parts))
  in
  let m = Nx.dim 1 y in
  let ids =
    List.mapi (fun j (_, k, _) -> Array.make k (Int64.of_int j)) parts
  in
  let ids_q =
    Nx.place (Nx.beside lengths)
      (Nx.create Nx.int64 [| 1; 1; m; 1 |] (Array.concat ids))
  in
  let same = Nx.equal ids_q (Nx.reshape [| 1; 1; 1; m |] ids_q) in
  let mask _ = same in
  let y =
    List.fold_left
      (fun y b ->
        let a = ln b.d_norm1 y in
        let y = Nx.add y (attend ~mask b.self_attn a a) in
        let y =
          Nx.add y
            (attend ~lengths b.cross_attn (ln b.query_norm y)
               (ln b.context_norm context))
        in
        Nx.add y (swiglu b.d_mlp (ln b.d_norm2 y)))
      y t.decoder
  in
  let y = ln t.decoder_norm y in
  let _, logits =
    List.fold_left
      (fun (off, acc) ((e : _ embedding), k, _) ->
        let rows = Nx.slice [ A; R (off, off + k); A ] y in
        (off + k, Kaun.Linear.apply e.logits rows :: acc))
      (0, []) parts
  in
  List.rev logits

let forward t inputs targets = decode t (encode t inputs) targets
