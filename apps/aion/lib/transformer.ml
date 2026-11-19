(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The epsilon of every layer norm, the query-key norms included: upstream's FM
   builds them as LayerNorm(eps=1e-6) (aion/fourm/fm.py:1020). *)
let eps = 1e-6

type ('b, 'd) w = (float, 'b, 'd) Nx.t
type ('b, 'd) linear = (('b, 'd) w, 'd) Kaun.Linear.t
type ('b, 'd) norm = (('b, 'd) w, 'd) Kaun.Layer_norm.t

(* Attention with query-key norm: the projections, then a layer norm over each
   head's queries and keys. *)
type ('b, 'd) attention = {
  proj : (('b, 'd) w, 'd) Kaun.Attention.t;
  q_norm : ('b, 'd) norm;
  k_norm : ('b, 'd) norm;
}

(* SwiGLU: [fc2 (silu (fc1 x) * fc3 x)]. *)
type ('b, 'd) mlp = {
  fc1 : ('b, 'd) linear;
  fc2 : ('b, 'd) linear;
  fc3 : ('b, 'd) linear;
}

type ('b, 'd) encoder_block = {
  norm1 : ('b, 'd) norm;
  attn : ('b, 'd) attention;
  norm2 : ('b, 'd) norm;
  mlp : ('b, 'd) mlp;
}

type ('b, 'd) decoder_block = {
  d_norm1 : ('b, 'd) norm;
  self_attn : ('b, 'd) attention;
  query_norm : ('b, 'd) norm;
  context_norm : ('b, 'd) norm;
  cross_attn : ('b, 'd) attention;
  d_norm2 : ('b, 'd) norm;
  d_mlp : ('b, 'd) mlp;
}

(* The arguments of the computations. *)

type 'd input = {
  modality : Modality.t;
  tokens : 'd Nx.int64_t;
  present : 'd Nx.bool_t option;
}

type ('b, 'd) encoding = { context : ('b, 'd) w; lengths : 'd Nx.int64_t }
type 'd target = { modality : Modality.t; positions : 'd Nx.int64_t option }

(* One modality's embeddings. [pos] is its positional table [tokens; dim] and
   [mod_emb] its modality embedding [dim], both shared by the encoder and the
   decoder; [table] is the encoder's token embedding [vocab; dim] and [logits]
   the decoder's output projection. *)
type ('b, 'd) embedding = {
  modality : Modality.t;
  table : ('b, 'd) w;
  pos : ('b, 'd) w;
  mod_emb : ('b, 'd) w;
  logits : ('b, 'd) linear;
}

(* A compiled transformer's programs ({!jit}): the computations of encode,
   decode and forward, without their checks. *)
type ('b, 'd) programs = {
  p_encode : 'd input list -> ('b, 'd) encoding;
  p_decode : ('b, 'd) encoding -> 'd target list -> ('b, 'd) w list;
  p_forward : 'd input list -> 'd target list -> ('b, 'd) w list;
}

type ('b, 'd) t = {
  embeddings : ('b, 'd) embedding list;
  encoder : ('b, 'd) encoder_block list;
  encoder_norm : ('b, 'd) norm;
  context : ('b, 'd) linear;
  decoder : ('b, 'd) decoder_block list;
  decoder_norm : ('b, 'd) norm;
  mask_token : ('b, 'd) w;
  programs : ('b, 'd) programs option;
}

let modalities t = List.map (fun (e : _ embedding) -> e.modality) t.embeddings
let dim t = (Nx.shape t.mask_token).(0)
let head_dim (a : _ attention) = (Nx.shape a.q_norm.gamma).(0)

(* Mapping every weight *)

let map (type b c d e) (f : (b, d) w -> (c, e) w) (t : (b, d) t) : (c, e) t =
  let linear (l : (b, d) linear) : (c, e) linear =
    { w = f l.w; b = Option.map f l.b }
  in
  let norm (n : (b, d) norm) : (c, e) norm =
    { gamma = f n.gamma; beta = Option.map f n.beta }
  in
  let attention a =
    {
      proj =
        {
          q = linear a.proj.q;
          k = linear a.proj.k;
          v = linear a.proj.v;
          out = linear a.proj.out;
        };
      q_norm = norm a.q_norm;
      k_norm = norm a.k_norm;
    }
  in
  let mlp m = { fc1 = linear m.fc1; fc2 = linear m.fc2; fc3 = linear m.fc3 } in
  {
    embeddings =
      List.map
        (fun e ->
          {
            modality = e.modality;
            table = f e.table;
            pos = f e.pos;
            mod_emb = f e.mod_emb;
            logits = linear e.logits;
          })
        t.embeddings;
    encoder =
      List.map
        (fun b ->
          {
            norm1 = norm b.norm1;
            attn = attention b.attn;
            norm2 = norm b.norm2;
            mlp = mlp b.mlp;
          })
        t.encoder;
    encoder_norm = norm t.encoder_norm;
    context = linear t.context;
    decoder =
      List.map
        (fun b ->
          {
            d_norm1 = norm b.d_norm1;
            self_attn = attention b.self_attn;
            query_norm = norm b.query_norm;
            context_norm = norm b.context_norm;
            cross_attn = attention b.cross_attn;
            d_norm2 = norm b.d_norm2;
            d_mlp = mlp b.d_mlp;
          })
        t.decoder;
    decoder_norm = norm t.decoder_norm;
    mask_token = f t.mask_token;
    programs = None;
  }


(* Reading the weights. Upstream's names (aion/fourm/fm.py, fm_utils.py): a
   linear layer is [out; in], as kaun's is; self-attention fuses its query, key
   and value projections in one [qkv] of [3 dim] rows, in that order, and
   cross-attention its key and value in one [kv]. Every layer norm stores a
   shift, a zero buffer since AION-1 trains none. A modality's embedding holds a
   positional table [1; tokens; dim] in the encoder and another in the decoder,
   and one modality embedding stored under the decoder's name, which the file's
   metadata aliases to the encoder's. *)

let fail fmt =
  Printf.ksprintf (fun m -> failwith ("Aion.Transformer: " ^ m)) fmt

(* An archive and the names read from it, so that a tensor nothing reads is
   refused. *)
type 'd reader = {
  archive : 'd Nx_io.Archive.t;
  read : (string, unit) Hashtbl.t;
}

let has r name = Option.is_some (Nx_io.Archive.find name r.archive)
let pp_shape s = String.concat "; " (Array.to_list (Array.map string_of_int s))

let float32 r name =
  match Nx_io.Archive.find name r.archive with
  | None -> fail "no tensor %s" name
  | Some p -> (
      try Nx.unpack Nx.float32 p
      with Invalid_argument _ -> fail "%s is not float32" name)

(* [extent r name i] is the extent of axis [i] of [name], read before the weight
   itself to size the others. *)
let extent r name i = (Nx.shape (float32 r name)).(i)

let weight r name shape =
  Hashtbl.replace r.read name ();
  let w = float32 r name in
  if Nx.shape w <> shape then
    fail "%s has shape [%s], [%s] expected" name
      (pp_shape (Nx.shape w))
      (pp_shape shape);
  w

(* A torch [out; in] weight, kaun's layout. *)
let linear ?b r name ~inputs ~outputs : _ Kaun.Linear.t =
  { w = weight r (name ^ ".weight") [| outputs; inputs |]; b }

(* A layer norm's shift must be zero: AION-1's weights have none, and a file
   with one would compute something [apply] does not. *)
let norm r name n : _ Kaun.Layer_norm.t =
  let shift = name ^ ".bias" in
  if Nx.item [] (Nx.any (Nx.not_equal_s (weight r shift [| n |]) 0.)) then
    fail "%s is not zero" shift;
  { gamma = weight r (name ^ ".weight") [| n |]; beta = None }

let rows w lo n = Nx.slice [ R (lo, lo + n); A ] w

let attention r name ~dim ~heads ~fused =
  let head_dim = dim / heads in
  let q, k, v =
    if fused then
      let qkv = weight r (name ^ ".qkv.weight") [| 3 * dim; dim |] in
      (rows qkv 0 dim, rows qkv dim dim, rows qkv (2 * dim) dim)
    else
      let kv = weight r (name ^ ".kv.weight") [| 2 * dim; dim |] in
      ( weight r (name ^ ".q.weight") [| dim; dim |],
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
        out = linear r (name ^ ".proj") ~inputs:dim ~outputs:dim;
      };
    q_norm = norm r (name ^ ".q_norm") head_dim;
    k_norm = norm r (name ^ ".k_norm") head_dim;
  }

let mlp r name ~dim ~hidden =
  {
    fc1 = linear r (name ^ ".fc1") ~inputs:dim ~outputs:hidden;
    fc2 = linear r (name ^ ".fc2") ~inputs:hidden ~outputs:dim;
    fc3 = linear r (name ^ ".fc3") ~inputs:dim ~outputs:hidden;
  }

(* The catalog's model tokens have no published map from its codec's tokens, so
   the transformer leaves the catalog out, and its tensors unread. *)
let modelled m = m <> Modality.Catalog

let embedding r ~dim m =
  let key = Modality.key m in
  let n = Modality.tokens m and vocab = Modality.vocab m in
  let enc = "encoder_embeddings." ^ key and dec = "decoder_embeddings." ^ key in
  let pos = weight r (enc ^ ".pos_emb") [| 1; n; dim |] in
  let dec_pos = weight r (dec ^ ".pos_emb") [| 1; n; dim |] in
  if not (Nx.item [] (Nx.array_equal pos dec_pos)) then
    fail "%s.pos_emb and %s.pos_emb differ" enc dec;
  {
    modality = m;
    table = weight r (enc ^ ".token_emb.weight") [| vocab; dim |];
    pos = Nx.reshape [| n; dim |] pos;
    mod_emb = Nx.reshape [| dim |] (weight r (dec ^ ".mod_emb") [| 1; 1; dim |]);
    logits = linear r (dec ^ ".to_logits") ~inputs:dim ~outputs:vocab;
  }

(* Tensors no path reads: the decoder's token tables, since the decoder's input
   is the mask token for every modality but the catalog, and the catalog's. *)
let unread name =
  let prefix p = String.starts_with ~prefix:p name in
  prefix "encoder_embeddings.catalog."
  || prefix "decoder_embeddings.catalog."
  || prefix "decoder_embeddings."
     && String.ends_with ~suffix:".token_emb.weight" name

let blocks r prefix make =
  let rec go i =
    if has r (Printf.sprintf "%s.%d.norm1.weight" prefix i) then
      make (Printf.sprintf "%s.%d" prefix i) :: go (i + 1)
    else []
  in
  go 0

let of_archive archive =
  let r = { archive; read = Hashtbl.create 1024 } in
  let dim = extent r "encoder_norm.weight" 0 in
  let heads = dim / extent r "encoder.0.attn.q_norm.weight" 0 in
  let hidden = extent r "encoder.0.mlp.fc1.weight" 0 in
  let embeddings =
    List.filter_map
      (fun m ->
        let name =
          "encoder_embeddings." ^ Modality.key m ^ ".token_emb.weight"
        in
        if modelled m && has r name then Some (embedding r ~dim m) else None)
      Modality.all
  in
  if embeddings = [] then fail "no modality";
  let encoder =
    blocks r "encoder" (fun p ->
        {
          norm1 = norm r (p ^ ".norm1") dim;
          attn = attention r (p ^ ".attn") ~dim ~heads ~fused:true;
          norm2 = norm r (p ^ ".norm2") dim;
          mlp = mlp r (p ^ ".mlp") ~dim ~hidden;
        })
  in
  let decoder =
    blocks r "decoder" (fun p ->
        {
          d_norm1 = norm r (p ^ ".norm1") dim;
          self_attn = attention r (p ^ ".self_attn") ~dim ~heads ~fused:true;
          query_norm = norm r (p ^ ".query_norm") dim;
          context_norm = norm r (p ^ ".context_norm") dim;
          cross_attn = attention r (p ^ ".cross_attn") ~dim ~heads ~fused:false;
          d_norm2 = norm r (p ^ ".norm2") dim;
          d_mlp = mlp r (p ^ ".mlp") ~dim ~hidden;
        })
  in
  let t =
    {
      embeddings;
      encoder;
      encoder_norm = norm r "encoder_norm" dim;
      context =
        linear r "decoder_proj_context" ~inputs:dim ~outputs:dim
          ~b:(weight r "decoder_proj_context.bias" [| dim |]);
      decoder;
      decoder_norm = norm r "decoder_norm" dim;
      mask_token = Nx.reshape [| dim |] (weight r "mask_token" [| 1; 1; dim |]);
      programs = None;
    }
  in
  List.iter
    (fun name ->
      if not (Hashtbl.mem r.read name || unread name) then
        fail "%s is not read" name)
    (Nx_io.Archive.names archive);
  t

let load ?cache_dir ?offline () =
  of_archive
    (Nx_io.load_safetensors
       (Kaun_hf.download_file ?cache_dir ?offline ~revision:Hub.revision
          ~file:"model.safetensors" Hub.repo))

(* Computing *)

let invalid fmt =
  Printf.ksprintf (fun m -> invalid_arg ("Aion.Transformer." ^ m)) fmt

let find fn t m =
  match
    List.find_opt (fun (e : _ embedding) -> e.modality = m) t.embeddings
  with
  | Some e -> e
  | None -> invalid "%s: no weights for %s" fn (Modality.key m)

let distinct fn what ms =
  List.iteri
    (fun j m ->
      if List.mem m (List.filteri (fun k _ -> k < j) ms) then
        invalid "%s: %s %s given twice" fn what (Modality.key m))
    ms

let attend ?mask ?lengths a x context =
  let head_dim = head_dim a in
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

(* Checks

   They read shapes where the arrays are, and values by {!Nx.check}, which
   raises at the call on the host. A compiled transformer checks copies on the
   host, so that its refusals come before its programs run, and its programs
   check nothing. *)

(* An input's tokens as [b; n], and where they are present, checked by shape. *)
let shaped t (i : _ input) =
  let e = find "encode" t i.modality in
  let key = Modality.key i.modality and n = Modality.tokens i.modality in
  let tokens =
    match Nx.shape i.tokens with
    | [| b |] when n = 1 -> Nx.reshape [| b; 1 |] i.tokens
    | [| _; n' |] when n' = n -> i.tokens
    | s ->
        invalid "encode: %s tokens of shape [%s], [b; %d] expected" key
          (pp_shape s) n
  in
  let present =
    match i.present with
    | None -> Nx.ones_at (Nx.beside tokens) Nx.bool (Nx.shape tokens)
    | Some p when Nx.shape p = Nx.shape tokens -> p
    | Some p when Nx.shape p = Nx.shape i.tokens ->
        Nx.reshape (Nx.shape tokens) p
    | Some p ->
        invalid "encode: %s presence of shape [%s], [%s] expected" key
          (pp_shape (Nx.shape p))
          (pp_shape (Nx.shape tokens))
  in
  (e, tokens, present)

(* The inputs' parts by {!shaped}, of one batch. *)
let parts t inputs =
  if inputs = [] then invalid "encode: no input";
  distinct "encode" "input" (List.map (fun (i : _ input) -> i.modality) inputs);
  let parts = List.map (shaped t) inputs in
  let batch = match parts with (_, k, _) :: _ -> Nx.dim 0 k | [] -> 0 in
  List.iter
    (fun ((e : _ embedding), k, _) ->
      if Nx.dim 0 k <> batch then
        invalid "encode: %s has %d objects, %d expected"
          (Modality.key e.modality) (Nx.dim 0 k) batch)
    parts;
  (batch, parts)

let refusal fmt =
  Printf.ksprintf
    (fun m _ () -> Invalid_argument ("Aion.Transformer." ^ m))
    fmt

(* Every present token in its vocabulary, and every object with one. *)
let inputs_ok t inputs =
  let _, parts = parts t inputs in
  List.iter
    (fun ((e : _ embedding), tokens, present) ->
      let vocab = Modality.vocab e.modality in
      let inside =
        Nx.logical_and (Nx.greater_equal_s tokens 0L)
          (Nx.less_s tokens (Int64.of_int vocab))
      in
      Nx.check Nx.Ptree.unit
        (Nx.logical_or (Nx.logical_not present) inside)
        ()
        (refusal "encode: a present %s token is outside [0, %d)"
           (Modality.key e.modality) vocab))
    parts;
  let present = Nx.concatenate ~axis:1 (List.map (fun (_, _, p) -> p) parts) in
  Nx.check Nx.Ptree.unit
    (Nx.any ~axes:[ 1 ] present)
    ()
    (refusal "encode: an object has no present token")

(* [e]'s context is [b; n; width] and each of its [b] lengths in [1, n]. *)
let check_encoding t context lengths =
  match (context, Nx.shape lengths) with
  | [| b; n; w |], [| b' |] when b = b' && w = dim t ->
      let inside =
        Nx.logical_and (Nx.greater_equal_s lengths 1L)
          (Nx.less_equal_s lengths (Int64.of_int n))
      in
      Nx.check Nx.Ptree.unit inside ()
        (refusal "decode: a length outside [1, %d]" n)
  | c, l ->
      invalid
        "decode: context [%s] and lengths [%s], [b; n; %d] and [b] expected"
        (pp_shape c) (pp_shape l) (dim t)

(* A target's positions: [b; k], in [0, n), and distinct within a row, where an
   element repeats when more than one of its row's elements equals it ([k^2]
   comparisons a row). A refusal names the first offending element's
   [row; index] and value. *)
let check_positions ~batch (tg : _ target) =
  let key = Modality.key tg.modality and n = Modality.tokens tg.modality in
  let refuse what i p =
    Invalid_argument
      (Printf.sprintf "Aion.Transformer.decode: %s position %Ld at [%s] %s" key
         (Nx.item [] p) (pp_shape i) what)
  in
  match tg.positions with
  | None -> ()
  | Some p ->
      (match Nx.shape p with
      | [| b; k |] when b = batch && k >= 1 -> ()
      | s ->
          invalid "decode: %s positions of shape [%s], [%d; k >= 1] expected"
            key (pp_shape s) batch);
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

let check_targets t ~batch targets =
  if targets = [] then invalid "decode: no target";
  distinct "decode" "target"
    (List.map (fun (tg : _ target) -> tg.modality) targets);
  List.iter
    (fun (tg : _ target) ->
      ignore (find "decode" t tg.modality);
      check_positions ~batch tg)
    targets

(* The arguments on the host, for a compiled transformer's checks. *)
let host x = Nx.place Nx.Host.on x

let host_input (i : _ input) : _ input =
  { i with tokens = host i.tokens; present = Option.map host i.present }

let host_target (tg : _ target) : _ target =
  { tg with positions = Option.map host tg.positions }

(* Computations, on checked arguments *)

(* The encoder's input: every input's token embeddings and their positional plus
   modality embeddings, concatenated in the inputs' order, then the present
   tokens moved first in that order by a stable sort and the others zeroed;
   [lengths] counts each object's present tokens. *)
let embed t inputs =
  let batch, parts = parts t inputs in
  let x =
    Nx.concatenate ~axis:1
      (List.map
         (fun (e, k, _) -> Kaun.Embedding.apply { table = e.table } k)
         parts)
  in
  let d = dim t in
  let emb =
    Nx.concatenate ~axis:1
      (List.map
         (fun (e, k, _) ->
           Nx.broadcast_to
             [| batch; Nx.dim 1 k; d |]
             (Nx.add e.pos (Nx.reshape [| 1; d |] e.mod_emb)))
         parts)
  in
  let present = Nx.concatenate ~axis:1 (List.map (fun (_, _, p) -> p) parts) in
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

let encode_raw t inputs =
  let x, emb, lengths = embed t inputs in
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

(* The decoder's input: for each target, the mask token plus the positional and
   modality embeddings of its positions, [b; k; width], concatenated in the
   targets' order. A token's self-attention sees the tokens of its own target,
   the mask of training for whole targets and the step of random-order
   generation for chosen positions. *)
let decode_raw t { context; lengths } targets =
  let batch = Nx.dim 0 context and d = dim t in
  let parts =
    List.map
      (fun (tg : _ target) ->
        let e = find "decode" t tg.modality in
        let emb = Nx.add e.pos (Nx.reshape [| 1; d |] e.mod_emb) in
        match tg.positions with
        | None ->
            let n = Nx.dim 0 emb in
            (e, n, Nx.broadcast_to [| batch; n; d |] emb)
        | Some p -> (e, Nx.dim 1 p, Nx.take ~axis:0 p emb))
      targets
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

let forward_raw t inputs targets = decode_raw t (encode_raw t inputs) targets

(* Entry points: the checks, then the computation, compiled where [t] is. *)

let compiled t = Option.is_some t.programs
let check t inputs = inputs_ok t (List.map host_input inputs)

let check_inputs t inputs =
  if compiled t then check t inputs else inputs_ok t inputs

let checks_decode t (e : _ encoding) targets =
  if targets = [] then invalid "decode: no target";
  let shape = Nx.shape e.context in
  if compiled t then check_encoding t shape (host e.lengths)
  else check_encoding t shape e.lengths;
  let batch = shape.(0) in
  if compiled t then check_targets t ~batch (List.map host_target targets)
  else check_targets t ~batch targets

let encode t inputs =
  check_inputs t inputs;
  match t.programs with
  | None -> encode_raw t inputs
  | Some p -> p.p_encode inputs

let decode t e targets =
  checks_decode t e targets;
  match t.programs with
  | None -> decode_raw t e targets
  | Some p -> p.p_decode e targets

let forward t inputs targets =
  check_inputs t inputs;
  let batch, _ = parts t inputs in
  if compiled t then check_targets t ~batch (List.map host_target targets)
  else check_targets t ~batch targets;
  match t.programs with
  | None -> forward_raw t inputs targets
  | Some p -> p.p_forward inputs targets

(* Compiling *)

(* The structures of the arguments and results, each modality reported as a
   case, so that a program is traced per list of modalities. *)
let modality c m =
  Nx.Ptree.Walk.case c (Modality.key m);
  m

let inputs_s () =
  Nx.Ptree.list
    (Nx.Ptree.v (fun c (i : _ input) ->
         let open Nx.Ptree.Walk in
         let modality = field c "modality" modality i.modality in
         let tokens = field c "tokens" tensor i.tokens in
         let present = field c "present" (option tensor) i.present in
         { modality; tokens; present }))

let targets_s () =
  Nx.Ptree.list
    (Nx.Ptree.v (fun c (tg : _ target) ->
         let open Nx.Ptree.Walk in
         let modality = field c "modality" modality tg.modality in
         let positions = field c "positions" (option tensor) tg.positions in
         { modality; positions }))

let encoding_s () =
  Nx.Ptree.v (fun c (e : _ encoding) ->
      let open Nx.Ptree.Walk in
      let context = field c "context" tensor e.context in
      let lengths = field c "lengths" tensor e.lengths in
      { context; lengths })

let jit t =
  let t = { t with programs = None } in
  let open Nx.Ptree in
  let inputs = inputs_s () and targets = targets_s () in
  let encoding = encoding_s () and logits = list tensor in
  let programs =
    {
      p_encode = Rune.jit (inputs @-> returns encoding) (encode_raw t);
      p_decode =
        Rune.jit (encoding @-> targets @-> returns logits) (decode_raw t);
      p_forward =
        Rune.jit (inputs @-> targets @-> returns logits) (forward_raw t);
    }
  in
  { t with programs = Some programs }

(* A compiled transformer's programs capture its weights, so moving them
   compiles again. *)
let place p t =
  let t' = map (Nx.place p) t in
  if compiled t then jit t' else t'

let cast dt t =
  let t' = map (Nx.cast dt) t in
  if compiled t then jit t' else t'

let eager t = { t with programs = None }
