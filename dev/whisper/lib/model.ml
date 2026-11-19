(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun

(* Dimensions *)

type dims = {
  n_mels : int;
  n_audio_ctx : int;
  n_audio_state : int;
  n_audio_head : int;
  n_audio_layer : int;
  n_vocab : int;
  n_text_ctx : int;
  n_text_state : int;
  n_text_head : int;
  n_text_layer : int;
}

(* whisper/model.py, Whisper.is_multilingual and num_languages: the English-only
   vocabulary has 51864 tokens, and each language past 99 adds one. *)
let multilingual dims = dims.n_vocab >= 51865
let languages dims = dims.n_vocab - 51765 - Bool.to_int (multilingual dims)

(* Parameters *)

type ('a, 'd) encoder_block = {
  attn_ln : ('a, 'd) Layer_norm.t;
  attn : ('a, 'd) Attention.t;
  mlp_ln : ('a, 'd) Layer_norm.t;
  mlp_up : ('a, 'd) Linear.t;
  mlp_down : ('a, 'd) Linear.t;
}

type ('a, 'd) decoder_block = {
  attn_ln : ('a, 'd) Layer_norm.t;
  attn : ('a, 'd) Attention.t;
  cross_attn_ln : ('a, 'd) Layer_norm.t;
  cross_attn : ('a, 'd) Attention.t;
  mlp_ln : ('a, 'd) Layer_norm.t;
  mlp_up : ('a, 'd) Linear.t;
  mlp_down : ('a, 'd) Linear.t;
}

type ('a, 'd) encoder = {
  conv1 : ('a, 'd) Conv.t;
  conv2 : ('a, 'd) Conv.t;
  positions : 'a;
  blocks : ('a, 'd) encoder_block list;
  ln_post : ('a, 'd) Layer_norm.t;
}

type ('a, 'd) decoder = {
  tokens : ('a, 'd) Embedding.t;
  positions : ('a, 'd) Embedding.t;
  blocks : ('a, 'd) decoder_block list;
  ln : ('a, 'd) Layer_norm.t;
}

type ('a, 'd) t = {
  dims : dims;
  encoder : ('a, 'd) encoder;
  decoder : ('a, 'd) decoder;
}

module Params = struct
  type nonrec ('a, 'd) t = ('a, 'd) t

  open Nx.Ptree.Walk

  let walk_dims c d =
    let int name n = field c name int n in
    {
      n_mels = int "n_mels" d.n_mels;
      n_audio_ctx = int "n_audio_ctx" d.n_audio_ctx;
      n_audio_state = int "n_audio_state" d.n_audio_state;
      n_audio_head = int "n_audio_head" d.n_audio_head;
      n_audio_layer = int "n_audio_layer" d.n_audio_layer;
      n_vocab = int "n_vocab" d.n_vocab;
      n_text_ctx = int "n_text_ctx" d.n_text_ctx;
      n_text_state = int "n_text_state" d.n_text_state;
      n_text_head = int "n_text_head" d.n_text_head;
      n_text_layer = int "n_text_layer" d.n_text_layer;
    }

  let walk_encoder_block c (b : _ encoder_block) : _ encoder_block =
    let attn_ln = field c "attn_ln" Layer_norm.walk b.attn_ln in
    let attn = field c "attn" Attention.walk b.attn in
    let mlp_ln = field c "mlp_ln" Layer_norm.walk b.mlp_ln in
    let mlp_up = field c "mlp_up" Linear.walk b.mlp_up in
    let mlp_down = field c "mlp_down" Linear.walk b.mlp_down in
    { attn_ln; attn; mlp_ln; mlp_up; mlp_down }

  let walk_decoder_block c (b : _ decoder_block) : _ decoder_block =
    let attn_ln = field c "attn_ln" Layer_norm.walk b.attn_ln in
    let attn = field c "attn" Attention.walk b.attn in
    let cross_attn_ln =
      field c "cross_attn_ln" Layer_norm.walk b.cross_attn_ln
    in
    let cross_attn = field c "cross_attn" Attention.walk b.cross_attn in
    let mlp_ln = field c "mlp_ln" Layer_norm.walk b.mlp_ln in
    let mlp_up = field c "mlp_up" Linear.walk b.mlp_up in
    let mlp_down = field c "mlp_down" Linear.walk b.mlp_down in
    { attn_ln; attn; cross_attn_ln; cross_attn; mlp_ln; mlp_up; mlp_down }

  let walk_encoder c (e : _ encoder) : _ encoder =
    let conv1 = field c "conv1" Conv.walk e.conv1 in
    let conv2 = field c "conv2" Conv.walk e.conv2 in
    let positions = field c "positions" leaf e.positions in
    let blocks = field c "blocks" (list walk_encoder_block) e.blocks in
    let ln_post = field c "ln_post" Layer_norm.walk e.ln_post in
    { conv1; conv2; positions; blocks; ln_post }

  let walk_decoder c (d : _ decoder) : _ decoder =
    let tokens = field c "tokens" Embedding.walk d.tokens in
    let positions = field c "positions" Embedding.walk d.positions in
    let blocks = field c "blocks" (list walk_decoder_block) d.blocks in
    let ln = field c "ln" Layer_norm.walk d.ln in
    { tokens; positions; blocks; ln }

  let walk c m =
    let dims = field c "dims" walk_dims m.dims in
    let encoder = field c "encoder" walk_encoder m.encoder in
    let decoder = field c "decoder" walk_decoder m.decoder in
    { dims; encoder; decoder }
end

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let shape_string s =
  "[" ^ String.concat "; " (Array.to_list (Array.map string_of_int s)) ^ "]"

(* Forward passes. A block adds each sublayer's output to the residual stream,
   the sublayer reading the stream through its own layer norm (eps 1e-5, as
   upstream's LayerNorm). *)

let mlp up down x = Linear.apply down (Fn.gelu (Linear.apply up x))

(* Encoder *)

let encode m mel =
  let d = m.dims and e = m.encoder in
  let shape = Nx.shape mel in
  if
    Array.length shape <> 3
    || shape.(1) <> 2 * d.n_audio_ctx
    || shape.(2) <> d.n_mels
  then
    invalid_argf "Whisper.Model.encode: mel of shape %s, not [n; %d; %d]"
      (shape_string shape) (2 * d.n_audio_ctx) d.n_mels;
  let same = [| (1, 1) |] in
  let x = Fn.gelu (Conv.apply ~padding:same e.conv1 mel) in
  let x = Fn.gelu (Conv.apply ~stride:[| 2 |] ~padding:same e.conv2 x) in
  let head_dim = d.n_audio_state / d.n_audio_head in
  let block x (b : _ encoder_block) =
    let a = Attention.apply ~head_dim b.attn (Layer_norm.apply b.attn_ln x) in
    let x = Nx.add x a in
    Nx.add x (mlp b.mlp_up b.mlp_down (Layer_norm.apply b.mlp_ln x))
  in
  Layer_norm.apply e.ln_post
    (List.fold_left block (Nx.add x e.positions) e.blocks)

(* Decoder *)

type 'a audio = { keys : 'a; values : 'a }

module Audio = struct
  type ('a, 'd) t = 'a audio

  let walk c a =
    let open Nx.Ptree.Walk in
    let keys = field c "keys" leaf a.keys in
    let values = field c "values" leaf a.values in
    { keys; values }
end

let text_head_dim d = d.n_text_state / d.n_text_head

let audio m features =
  let d = m.dims in
  let shape = Nx.shape features in
  if
    Array.length shape <> 3
    || shape.(1) <> d.n_audio_ctx
    || shape.(2) <> d.n_audio_state
  then
    invalid_argf "Whisper.Model.audio: features of shape %s, not [n; %d; %d]"
      (shape_string shape) d.n_audio_ctx d.n_audio_state;
  let head_dim = text_head_dim d in
  List.map
    (fun (b : _ decoder_block) ->
      {
        keys = Attention.split ~head_dim b.cross_attn.k features;
        values = Attention.split ~head_dim b.cross_attn.v features;
      })
    m.decoder.blocks

let cache m ~slots at =
  let d = m.dims in
  let dtype = Nx.dtype m.decoder.ln.gamma in
  List.init d.n_text_layer (fun _ ->
      Attention.Cache.make ~slots ~kv_heads:d.n_text_head
        ~head_dim:(text_head_dim d) dtype at)

let decode m audio caches index ids =
  let d = m.dims and p = m.decoder in
  let blocks = List.length p.blocks in
  if List.length audio <> blocks || List.length caches <> blocks then
    invalid_argf "Whisper.Model.decode: %d audio and %d caches for %d blocks"
      (List.length audio) (List.length caches) blocks;
  if Cache_index.context index > d.n_text_ctx then
    invalid_argf "Whisper.Model.decode: context %d exceeds n_text_ctx %d"
      (Cache_index.context index)
      d.n_text_ctx;
  let batch = Cache_index.batch index in
  List.iter
    (fun a ->
      let n = Nx.dim 0 a.keys in
      if n = 0 || batch mod n <> 0 then
        invalid_argf "Whisper.Model.decode: audio of %d lanes for %d" n batch)
    audio;
  let head_dim = text_head_dim d in
  (* A group of [g] lanes reads one audio: its queries, [[| n * g; h; s; dh |]],
     fold into that audio's query axis as [[| n; h; g * s; dh |]]. The
     cross-attention has no mask, so each query row attends over its audio's
     keys alone, as it would over a copy of them per lane. *)
  let cross q kv n =
    let g = batch / n in
    if g = 1 then Attention.attend q kv
    else
      let h = Nx.dim 1 q and s = Nx.dim 2 q and dh = Nx.dim 3 q in
      let fold q =
        Nx.reshape
          [| n; h; g * s; dh |]
          (Nx.transpose ~axes:[ 0; 2; 1; 3; 4 ]
             (Nx.reshape [| n; g; h; s; dh |] q))
      in
      let y = Attention.attend (fold q) kv in
      let dv = Nx.dim 3 y in
      Nx.reshape [| batch; h; s; dv |]
        (Nx.transpose ~axes:[ 0; 2; 1; 3; 4 ]
           (Nx.reshape [| n; h; g; s; dv |] y))
  in
  let block (x, caches) (b : _ decoder_block) a cache =
    let y, cache =
      Attention.cached ~head_dim b.attn cache index
        (Layer_norm.apply b.attn_ln x)
    in
    let x = Nx.add x y in
    let q =
      Attention.split ~head_dim b.cross_attn.q
        (Layer_norm.apply b.cross_attn_ln x)
    in
    let kv = Nx.Dense { k = a.keys; v = a.values } in
    let x =
      Nx.add x (Attention.merge b.cross_attn.out (cross q kv (Nx.dim 0 a.keys)))
    in
    let x = Nx.add x (mlp b.mlp_up b.mlp_down (Layer_norm.apply b.mlp_ln x)) in
    (x, cache :: caches)
  in
  let x =
    Nx.add
      (Embedding.apply p.tokens ids)
      (Embedding.apply p.positions (Cache_index.positions index))
  in
  let x, caches =
    List.fold_left2
      (fun acc b (a, c) -> block acc b a c)
      (x, []) p.blocks
      (List.combine audio caches)
  in
  (x, List.rev caches)

let logits m h =
  let d = m.dims and p = m.decoder in
  let shape = Nx.shape h in
  let rank = Array.length shape in
  if rank = 0 || shape.(rank - 1) <> d.n_text_state then
    invalid_argf "Whisper.Model.logits: stream of shape %s, not [...; %d]"
      (shape_string shape) d.n_text_state;
  Nx.cast Nx.float32
    (Nx.matmul (Layer_norm.apply p.ln h) (Nx.transpose p.tokens.table))

(* Checkpoints

   whisper/__init__.py, load_model: the file is [{"dims": dict,
   "model_state_dict": state}], and the state names each weight by its module's
   path in whisper/model.py. *)

let field name = function
  | Nx_io.Pytorch.Dict kvs -> (
      match List.assoc_opt (Nx_io.Pytorch.String name) kvs with
      | Some v -> v
      | None -> failwith ("Whisper.Model: checkpoint: no " ^ name))
  | _ -> failwith ("Whisper.Model: checkpoint: no dict holds " ^ name)

let dims_of v =
  let int name =
    match field name v with
    | Nx_io.Pytorch.Int n -> n
    | _ -> failwith ("Whisper.Model: checkpoint: dims." ^ name ^ " is no int")
  in
  {
    n_mels = int "n_mels";
    n_audio_ctx = int "n_audio_ctx";
    n_audio_state = int "n_audio_state";
    n_audio_head = int "n_audio_head";
    n_audio_layer = int "n_audio_layer";
    n_vocab = int "n_vocab";
    n_text_ctx = int "n_text_ctx";
    n_text_state = int "n_text_state";
    n_text_head = int "n_text_head";
    n_text_layer = int "n_text_layer";
  }

let of_checkpoint dt c =
  let dims = dims_of (field "dims" c) in
  let state = Nx_io.Pytorch.tensors (field "model_state_dict" c) in
  let float ~shape name = Nx_io.Archive.float ~shape dt name state in
  let at path leaf = path ^ "." ^ leaf in
  let linear ?(bias = true) ~inputs ~outputs path =
    {
      Linear.w = float ~shape:[| outputs; inputs |] (at path "weight");
      b =
        (if bias then Some (float ~shape:[| outputs |] (at path "bias"))
         else None);
    }
  in
  (* Stored [out; in; k]: [Conv]'s [out; k; in] moves the input axis last. *)
  let conv ~inputs ~outputs path =
    {
      Conv.w =
        Nx.moveaxis 1 (-1)
          (float ~shape:[| outputs; inputs; 3 |] (at path "weight"));
      b = Some (float ~shape:[| outputs |] (at path "bias"));
    }
  in
  let layer_norm d path =
    {
      Layer_norm.gamma = float ~shape:[| d |] (at path "weight");
      beta = Some (float ~shape:[| d |] (at path "bias"));
    }
  in
  let attention d path =
    let proj ?bias leaf = linear ?bias ~inputs:d ~outputs:d (at path leaf) in
    {
      Attention.q = proj "query";
      k = proj ~bias:false "key";
      v = proj "value";
      out = proj "out";
    }
  in
  let mlp d path =
    ( linear ~inputs:d ~outputs:(4 * d) (at path "mlp.0"),
      linear ~inputs:(4 * d) ~outputs:d (at path "mlp.2") )
  in
  let encoder_block i : _ encoder_block =
    let d = dims.n_audio_state
    and path = Printf.sprintf "encoder.blocks.%d" i in
    let mlp_up, mlp_down = mlp d path in
    {
      attn_ln = layer_norm d (at path "attn_ln");
      attn = attention d (at path "attn");
      mlp_ln = layer_norm d (at path "mlp_ln");
      mlp_up;
      mlp_down;
    }
  in
  let decoder_block i : _ decoder_block =
    let d = dims.n_text_state and path = Printf.sprintf "decoder.blocks.%d" i in
    let mlp_up, mlp_down = mlp d path in
    {
      attn_ln = layer_norm d (at path "attn_ln");
      attn = attention d (at path "attn");
      cross_attn_ln = layer_norm d (at path "cross_attn_ln");
      cross_attn = attention d (at path "cross_attn");
      mlp_ln = layer_norm d (at path "mlp_ln");
      mlp_up;
      mlp_down;
    }
  in
  let d = dims.n_audio_state in
  let encoder =
    {
      conv1 = conv ~inputs:dims.n_mels ~outputs:d "encoder.conv1";
      conv2 = conv ~inputs:d ~outputs:d "encoder.conv2";
      positions =
        float ~shape:[| dims.n_audio_ctx; d |] "encoder.positional_embedding";
      blocks = List.init dims.n_audio_layer encoder_block;
      ln_post = layer_norm d "encoder.ln_post";
    }
  in
  let d = dims.n_text_state in
  let decoder =
    {
      tokens =
        {
          Embedding.table =
            float ~shape:[| dims.n_vocab; d |] "decoder.token_embedding.weight";
        };
      positions =
        {
          Embedding.table =
            float ~shape:[| dims.n_text_ctx; d |] "decoder.positional_embedding";
        };
      blocks = List.init dims.n_text_layer decoder_block;
      ln = layer_norm d "decoder.ln";
    }
  in
  { dims; encoder; decoder }

let load dt path = of_checkpoint dt (Nx_io.load_pytorch path)

(* Releases *)

type release =
  | Tiny_en
  | Tiny
  | Base_en
  | Base
  | Small_en
  | Small
  | Medium_en
  | Medium
  | Large_v1
  | Large_v2
  | Large_v3
  | Large_v3_turbo

let release_of_string = function
  | "tiny.en" -> Some Tiny_en
  | "tiny" -> Some Tiny
  | "base.en" -> Some Base_en
  | "base" -> Some Base
  | "small.en" -> Some Small_en
  | "small" -> Some Small
  | "medium.en" -> Some Medium_en
  | "medium" -> Some Medium
  | "large-v1" -> Some Large_v1
  | "large-v2" -> Some Large_v2
  | "large-v3" | "large" -> Some Large_v3
  | "large-v3-turbo" | "turbo" -> Some Large_v3_turbo
  | _ -> None

(* whisper/__init__.py, _ALIGNMENT_HEADS: the base85 of the gzip of a bool array
   [n_text_layer; n_text_head], one byte per head, with each release's decoder
   layers and heads. *)
let heads_dump = function
  | Tiny_en -> ("ABzY8J1N>@0{>%R00Bk>$p{7v037`oCl~+#00", 4, 6)
  | Tiny -> ("ABzY8bu8Lr0{>%RKn9Fp%m@SkK7Kt=7ytkO", 4, 6)
  | Base_en -> ("ABzY8;40c<0{>%RzzG;p*o+Vo09|#PsxSZm00", 6, 8)
  | Base -> ("ABzY8KQ!870{>%RzyTQH3`Q^yNP!>##QT-<FaQ7m", 6, 8)
  | Small_en -> ("ABzY8>?_)10{>%RpeA61k&I|OI3I$65C{;;pbCHh0B{qLQ;+}v00", 12, 12)
  | Small -> ("ABzY8DmU6=0{>%Rpa?J`kvJ6qF(V^F86#Xh7JUGMK}P<N0000", 12, 12)
  | Medium_en ->
      ("ABzY8usPae0{>%R7<zz_OvQ{)4kMa0BMw6u5rT}kRKX;$NfYBv00*Hl@qhsU00", 24, 16)
  | Medium -> ("ABzY8B0Jh+0{>%R7}kK1fFL7w6%<-Pf*t^=N)Qr&0RR9", 24, 16)
  | Large_v1 -> ("ABzY8r9j$a0{>%R7#4sLmoOs{s)o3~84-RPdcFk!JR<kSfC2yj", 32, 20)
  | Large_v2 ->
      ( "ABzY8zd+h!0{>%R7=D0pU<_bnWW*tkYAhobTNnu$jnkEkXqp)j;w1Tzk)UH3X%SZd&fFZ2fC2yj",
        32,
        20 )
  | Large_v3 ->
      ("ABzY8gWO1E0{>%R7(9S+Kn!D~%ngiGaR?*L!iJG9p-nab0JQ=-{D1-g00", 32, 20)
  | Large_v3_turbo -> ("ABzY8j^C+e0{>%RARaKHP%t(lGR*)0g!tONPyhe`", 4, 20)

(* RFC 1924's alphabet, Python's base64.b85decode: each 5 digits are 4 bytes,
   big-endian; a last group of k < 5 digits is padded with the last digit and
   gives k - 1 bytes. *)
let base85 =
  "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~"

let b85decode s =
  let digit c =
    match String.index_opt base85 c with
    | Some d -> d
    | None -> invalid_arg "Whisper.Model: base85 digit"
  in
  let n = String.length s in
  let out = Buffer.create ((n * 4 / 5) + 4) in
  let rec group i =
    if i < n then begin
      let k = min 5 (n - i) in
      let v = ref 0 in
      for j = 0 to 4 do
        v := (!v * 85) + if j < k then digit s.[i + j] else 84
      done;
      for j = 0 to k - 2 do
        Buffer.add_char out (Char.chr ((!v lsr (8 * (3 - j))) land 0xff))
      done;
      group (i + 5)
    end
  in
  group 0;
  Buffer.contents out

let bigbytes n =
  Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout n

let bigbytes_of_string s =
  let b = bigbytes (String.length s) in
  String.iteri (fun i c -> b.{i} <- Char.code c) s;
  b

let alignment_heads r =
  let dump, layers, heads = heads_dump r in
  let src = bigbytes_of_string (b85decode dump) in
  let dst = bigbytes (layers * heads) in
  (match Lz.Gzip.decode src dst with
  | Ok { produced; _ } when produced = layers * heads -> ()
  | _ -> assert false);
  List.concat
    (List.init layers (fun l ->
         List.filter_map
           (fun h -> if dst.{(l * heads) + h} <> 0 then Some (l, h) else None)
           (List.init heads Fun.id)))
