(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Dtype = Nx_core.Dtype
module Ptree = Ptree
module Initializers = Initializers

let normalize_mask (type a) (type layout) (mask : (a, layout) Rune.t) :
    Rune.bool_t =
  let dtype = Rune.dtype mask in
  match Dtype.equal_witness dtype Rune.bool with
  | Some Type.Equal -> Rune.cast Rune.bool mask
  | None ->
      let zeros = Rune.zeros_like mask in
      Rune.not_equal mask zeros |> Rune.cast Rune.bool

let compute_attention_from_projected ?attention_mask ?(is_causal = false)
    ?dropout_rate ?dropout_rng ?scale ~q ~k ~v ~embed_dim ~num_heads
    ~num_kv_heads ~head_dim () =
  if embed_dim <> num_heads * head_dim then
    invalid_arg
      (Printf.sprintf
         "multi-head attention: embed_dim (%d) must equal num_heads (%d) * \
          head_dim (%d)"
         embed_dim num_heads head_dim);
  let reshape_heads tensor heads =
    let tensor = Rune.contiguous tensor in
    let shape = Rune.shape tensor in
    if Array.length shape <> 3 then
      invalid_arg "multi-head attention expects projected tensors of rank 3";
    let last_dim = shape.(2) in
    if last_dim <> heads * head_dim then
      invalid_arg
        (Printf.sprintf
           "multi-head attention: projected dimension mismatch (got %d, \
            expected %d)"
           last_dim (heads * head_dim));
    let reshaped =
      Rune.reshape [| shape.(0); shape.(1); heads; head_dim |] tensor
    in
    Rune.transpose reshaped ~axes:[ 0; 2; 1; 3 ]
  in
  let q_heads = reshape_heads q num_heads in
  let k_heads = reshape_heads k num_kv_heads in
  let v_heads = reshape_heads v num_kv_heads in
  let repeat_if_needed tensor =
    if num_kv_heads < num_heads then (
      if num_heads mod num_kv_heads <> 0 then
        invalid_arg
          (Printf.sprintf
             "multi-head attention: num_heads (%d) must be a multiple of \
              num_kv_heads (%d)"
             num_heads num_kv_heads);
      let repeat_factor = num_heads / num_kv_heads in
      let shape = Rune.shape tensor in
      let expanded = Rune.expand_dims [ 2 ] tensor in
      let target =
        [| shape.(0); shape.(1); repeat_factor; shape.(2); shape.(3) |]
      in
      let broadcasted = Rune.broadcast_to target expanded in
      Rune.reshape [| shape.(0); num_heads; shape.(2); shape.(3) |] broadcasted)
    else tensor
  in
  let k_heads = repeat_if_needed k_heads in
  let v_heads = repeat_if_needed v_heads in
  let q_shape = Rune.shape q in
  let batch = q_shape.(0) in
  let seq_len_q = q_shape.(1) in
  let seq_len_k = (Rune.shape k).(1) in
  let attention_mask =
    match attention_mask with
    | None -> None
    | Some mask ->
        let mask = normalize_mask mask in
        let shape = Rune.shape mask in
        let prepared =
          match Array.length shape with
          | 2 ->
              let batch_dim = shape.(0) in
              let key_dim = shape.(1) in
              if
                (batch_dim <> batch && batch_dim <> 1)
                || (key_dim <> seq_len_k && key_dim <> 1)
              then
                invalid_arg
                  "attention mask of rank 2 must align with [batch; seq_len_k]";
              Rune.reshape [| batch_dim; 1; 1; key_dim |] mask
          | 3 ->
              let batch_dim = shape.(0) in
              let query_dim = shape.(1) in
              let key_dim = shape.(2) in
              if
                (batch_dim <> batch && batch_dim <> 1)
                || (query_dim <> seq_len_q && query_dim <> 1)
                || (key_dim <> seq_len_k && key_dim <> 1)
              then
                invalid_arg
                  "attention mask of rank 3 must align with [batch; seq_len_q; \
                   seq_len_k]";
              Rune.expand_dims [ 1 ] mask
          | 4 ->
              let batch_dim = shape.(0) in
              let head_dim = shape.(1) in
              let query_dim = shape.(2) in
              let key_dim = shape.(3) in
              if
                (batch_dim <> batch && batch_dim <> 1)
                || (head_dim <> num_heads && head_dim <> 1)
                || (query_dim <> seq_len_q && query_dim <> 1)
                || (key_dim <> seq_len_k && key_dim <> 1)
              then
                invalid_arg
                  "attention mask of rank 4 must align with [batch; num_heads; \
                   seq_len_q; seq_len_k]";
              mask
          | _ ->
              invalid_arg
                "attention mask rank must be 2, 3, or 4 for multi-head \
                 attention"
        in
        let target = [| batch; num_heads; seq_len_q; seq_len_k |] in
        Some (Rune.broadcast_to target prepared)
  in
  let attn =
    let dropout_key =
      match dropout_rng with
      | Some rng -> Some rng
      | None when Option.is_some dropout_rate ->
          invalid_arg "attention dropout requires RNG"
      | None -> None
    in
    Rune.dot_product_attention ?attention_mask ?scale ?dropout_rate ?dropout_key
      ~is_causal q_heads k_heads v_heads
  in
  attn
  |> Rune.transpose ~axes:[ 0; 2; 1; 3 ]
  |> Rune.contiguous
  |> Rune.reshape [| batch; seq_len_q; embed_dim |]

module Multi_head = struct
  type config = {
    embed_dim : int;
    num_heads : int;
    num_kv_heads : int option;
    head_dim : int option;
    dropout : float;
    use_qk_norm : bool;
    attn_logits_soft_cap : float option;
    query_pre_attn_scalar : float option;
  }

  let make_config ~embed_dim ~num_heads ?num_kv_heads ?head_dim ?(dropout = 0.0)
      ?(use_qk_norm = false) ?attn_logits_soft_cap ?query_pre_attn_scalar () =
    {
      embed_dim;
      num_heads;
      num_kv_heads;
      head_dim;
      dropout;
      use_qk_norm;
      attn_logits_soft_cap;
      query_pre_attn_scalar;
    }

  type params = Ptree.t

  let init config ~rngs ~dtype =
    let head_dim =
      Option.value config.head_dim ~default:(config.embed_dim / config.num_heads)
    in
    if head_dim * config.num_heads <> config.embed_dim then
      invalid_arg
        (Printf.sprintf
           "multi-head attention: embed_dim (%d) not divisible by num_heads \
            (%d)"
           config.embed_dim config.num_heads);
    let num_kv_heads =
      Option.value config.num_kv_heads ~default:config.num_heads
    in
    let num_keys = if config.use_qk_norm then 6 else 4 in
    let keys = Rune.Rng.split ~n:num_keys rngs in
    let init_fn = (Initializers.glorot_uniform ()).f in
    let q_proj =
      init_fn keys.(0) [| config.embed_dim; config.num_heads * head_dim |] dtype
    in
    let k_proj =
      init_fn keys.(1) [| config.embed_dim; num_kv_heads * head_dim |] dtype
    in
    let v_proj =
      init_fn keys.(2) [| config.embed_dim; num_kv_heads * head_dim |] dtype
    in
    let out_proj =
      init_fn keys.(3) [| config.num_heads * head_dim; config.embed_dim |] dtype
    in
    let base =
      [
        ("q_proj", Ptree.tensor q_proj);
        ("k_proj", Ptree.tensor k_proj);
        ("v_proj", Ptree.tensor v_proj);
        ("out_proj", Ptree.tensor out_proj);
      ]
    in
    let base =
      if config.use_qk_norm then
        let scale = Rune.ones dtype [| head_dim |] in
        base
        @ [
            ("q_norm_scale", Ptree.tensor scale);
            ("k_norm_scale", Ptree.tensor scale);
          ]
      else base
    in
    Ptree.dict base

  let apply ?rngs ?attention_mask config params ~training ~query ~key ~value =
    let dtype = Rune.dtype query in
    let fields = Ptree.Dict.fields_exn ~ctx:"attention.multi_head" params in
    let get name = Ptree.Dict.get_tensor_exn fields ~name dtype in
    let q_proj = get "q_proj" in
    let k_proj = get "k_proj" in
    let v_proj = get "v_proj" in
    let out_proj = get "out_proj" in
    let head_dim =
      Option.value config.head_dim ~default:(config.embed_dim / config.num_heads)
    in
    let num_kv_heads =
      Option.value config.num_kv_heads ~default:config.num_heads
    in
    let scale : float =
      match config.query_pre_attn_scalar with
      | Some s -> s
      | None -> 1.0 /. Stdlib.sqrt (float_of_int head_dim)
    in
    let effective_dropout = if training then config.dropout else 0.0 in
    let dropout_rng =
      if effective_dropout > 0.0 then
        match rngs with
        | Some key -> Some key
        | None -> failwith "attention dropout requires RNG"
      else None
    in
    let attention_mask = Option.map normalize_mask attention_mask in
    let q_projected = Rune.matmul query q_proj in
    let k_projected = Rune.matmul key k_proj in
    let v_projected = Rune.matmul value v_proj in
    let context =
      compute_attention_from_projected ?attention_mask
        ?dropout_rate:
          (if effective_dropout > 0.0 then Some effective_dropout else None)
        ?dropout_rng ~is_causal:false ~scale ~q:q_projected ~k:k_projected
        ~v:v_projected ~embed_dim:config.embed_dim ~num_heads:config.num_heads
        ~num_kv_heads ~head_dim ()
    in
    let output = Rune.matmul context out_proj in
    match config.attn_logits_soft_cap with
    | None -> output
    | Some cap ->
        let scaled = Rune.div output (Rune.scalar (Rune.dtype output) cap) in
        let capped = Rune.tanh scaled in
        Rune.mul capped (Rune.scalar (Rune.dtype output) cap)
end
