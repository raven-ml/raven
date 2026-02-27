(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

module Dtype = Nx_core.Dtype

let require_same_float_dtype (type p in_elt) ~ctx
    (expected : (float, p) Nx.dtype) (x : (float, in_elt) Nx.t) :
    (float, p) Nx.t =
  match Dtype.equal_witness expected (Nx.dtype x) with
  | Some Type.Equal -> (x : (float, p) Nx.t)
  | None ->
      invalid_argf "%s: input dtype %s does not match model dtype %s" ctx
        (Dtype.to_string (Nx.dtype x))
        (Dtype.to_string expected)

let normalize_axis ~ctx ~ndim axis =
  let normalized = if axis < 0 then ndim + axis else axis in
  if normalized < 0 || normalized >= ndim then
    invalid_argf "%s: axis %d out of bounds for rank %d" ctx axis ndim;
  normalized

(* Rotary position embeddings *)

let rope ?(theta = 10000.0) ?(seq_dim = -2) x =
  let ctx = "Attention.rope" in
  let shape = Nx.shape x in
  let ndim = Array.length shape in
  if ndim < 2 then invalid_argf "%s: expected rank >= 2, got rank %d" ctx ndim;
  let seq_axis = normalize_axis ~ctx ~ndim seq_dim in
  if seq_axis = ndim - 1 then
    invalid_argf
      "%s: seq_dim points to the last axis; last axis is reserved for head_dim"
      ctx;
  let seq_len = shape.(seq_axis) in
  let head_dim = shape.(ndim - 1) in
  if head_dim mod 2 <> 0 then
    invalid_argf "%s: head_dim must be even, got %d" ctx head_dim;
  let half = head_dim / 2 in
  let dtype = Nx.dtype x in
  let inv_freq =
    let exponents = Nx.arange_f dtype 0.0 (float_of_int head_dim) 2.0 in
    let normalized =
      Nx.div exponents (Nx.scalar dtype (float_of_int head_dim))
    in
    Nx.pow (Nx.scalar dtype theta) (Nx.neg normalized)
  in
  let positions = Nx.arange_f dtype 0.0 (float_of_int seq_len) 1.0 in
  let angles =
    Nx.matmul
      (Nx.reshape [| seq_len; 1 |] positions)
      (Nx.reshape [| 1; half |] inv_freq)
  in
  let broadcast_shape = Array.make ndim 1 in
  broadcast_shape.(seq_axis) <- seq_len;
  broadcast_shape.(ndim - 1) <- half;
  let cos_angles = Nx.reshape broadcast_shape (Nx.cos angles) in
  let sin_angles = Nx.reshape broadcast_shape (Nx.sin angles) in
  let last_axis_slice start stop =
    let slices = Array.make ndim Nx.A in
    slices.(ndim - 1) <- Nx.R (start, stop);
    Array.to_list slices
  in
  let x1 = Nx.slice (last_axis_slice 0 half) x in
  let x2 = Nx.slice (last_axis_slice half head_dim) x in
  let r1 = Nx.sub (Nx.mul x1 cos_angles) (Nx.mul x2 sin_angles) in
  let r2 = Nx.add (Nx.mul x1 sin_angles) (Nx.mul x2 cos_angles) in
  Nx.concatenate ~axis:(-1) [ r1; r2 ]

(* Multi-head self-attention *)

let attention_mask_key = "attention_mask"
let apply_rope ~theta t = rope ~theta t

let multi_head_attention ~embed_dim ~num_heads ?(num_kv_heads = num_heads)
    ?(dropout = 0.0) ?(is_causal = false) ?(rope = false)
    ?(rope_theta = 10000.0) () =
  let use_rope = rope in
  let head_dim = embed_dim / num_heads in
  if head_dim * num_heads <> embed_dim then
    invalid_argf
      "Attention.multi_head_attention: embed_dim (%d) not divisible by \
       num_heads (%d)"
      embed_dim num_heads;
  if num_heads mod num_kv_heads <> 0 then
    invalid_argf
      "Attention.multi_head_attention: num_heads (%d) not divisible by \
       num_kv_heads (%d)"
      num_heads num_kv_heads;
  if dropout < 0.0 || dropout >= 1.0 then
    invalid_argf
      "Attention.multi_head_attention: expected 0.0 <= dropout < 1.0, got %g"
      dropout;
  let weight_init = Init.glorot_uniform () in
  {
    Layer.init =
      (fun ~dtype ->
        let q_proj =
          weight_init.f [| embed_dim; num_heads * head_dim |] dtype
        in
        let k_proj =
          weight_init.f [| embed_dim; num_kv_heads * head_dim |] dtype
        in
        let v_proj =
          weight_init.f [| embed_dim; num_kv_heads * head_dim |] dtype
        in
        let out_proj =
          weight_init.f [| num_heads * head_dim; embed_dim |] dtype
        in
        Layer.make_vars
          ~params:
            (Ptree.dict
               [
                 ("q_proj", Ptree.tensor q_proj);
                 ("k_proj", Ptree.tensor k_proj);
                 ("v_proj", Ptree.tensor v_proj);
                 ("out_proj", Ptree.tensor out_proj);
               ])
          ~state:(Ptree.list []) ~dtype);
    apply =
      (fun ~params ~state ~dtype ~training ?ctx x ->
        let x =
          require_same_float_dtype ~ctx:"Attention.multi_head_attention" dtype x
        in
        let shape = Nx.shape x in
        let batch = shape.(0) in
        let seq_len = shape.(1) in
        let fields =
          Ptree.Dict.fields_exn ~ctx:"Attention.multi_head_attention.params"
            params
        in
        let get name = Ptree.Dict.get_tensor_exn fields ~name dtype in
        let q_proj = get "q_proj" in
        let k_proj = get "k_proj" in
        let v_proj = get "v_proj" in
        let out_proj = get "out_proj" in
        let q = Nx.matmul x q_proj in
        let k = Nx.matmul x k_proj in
        let v = Nx.matmul x v_proj in
        let reshape_heads t heads =
          let t = Nx.reshape [| batch; seq_len; heads; head_dim |] t in
          Nx.transpose t ~axes:[ 0; 2; 1; 3 ]
        in
        let q = reshape_heads q num_heads in
        let k = reshape_heads k num_kv_heads in
        let v = reshape_heads v num_kv_heads in
        let repeat_kv t =
          if num_kv_heads < num_heads then
            let repetition = num_heads / num_kv_heads in
            let shape = Nx.shape t in
            let expanded = Nx.expand_dims [ 2 ] t in
            let target =
              [| shape.(0); shape.(1); repetition; shape.(2); shape.(3) |]
            in
            Nx.broadcast_to target expanded
            |> Nx.contiguous
            |> Nx.reshape [| shape.(0); num_heads; shape.(2); shape.(3) |]
          else t
        in
        let k = repeat_kv k in
        let v = repeat_kv v in
        let q, k =
          if use_rope then
            (apply_rope ~theta:rope_theta q, apply_rope ~theta:rope_theta k)
          else (q, k)
        in
        let dropout_rate =
          if training && dropout > 0.0 then Some dropout else None
        in
        (* Read attention mask from context if present. Accepts [batch; seq_k]
           int32 (0/1) or bool, reshapes to [batch; 1; 1; seq_k] for
           broadcasting over heads and queries. *)
        let attention_mask =
          match ctx with
          | None -> None
          | Some ctx -> (
              match Context.find ctx ~name:attention_mask_key with
              | None -> None
              | Some tensor ->
                  let bool_mask =
                    match Ptree.Tensor.to_typed Nx.bool tensor with
                    | Some m -> m
                    | None ->
                        (* int/float mask: cast to int32, nonzero = true *)
                        let int_mask =
                          match Ptree.Tensor.to_typed Nx.int32 tensor with
                          | Some m -> m
                          | None ->
                              let (Ptree.P raw) = tensor in
                              Nx.cast Nx.int32 raw
                        in
                        Nx.not_equal int_mask
                          (Nx.zeros Nx.int32 (Nx.shape int_mask))
                  in
                  let mask_shape = Nx.shape bool_mask in
                  let ndim = Array.length mask_shape in
                  let reshaped =
                    if ndim = 2 then
                      Nx.reshape
                        [| mask_shape.(0); 1; 1; mask_shape.(1) |]
                        bool_mask
                    else bool_mask
                  in
                  Some reshaped)
        in
        let attn =
          Fn.dot_product_attention ?attention_mask ?dropout_rate ~is_causal q k
            v
        in
        let merged =
          Nx.transpose attn ~axes:[ 0; 2; 1; 3 ]
          |> Nx.contiguous
          |> Nx.reshape [| batch; seq_len; embed_dim |]
        in
        let output = Nx.matmul merged out_proj in
        (output, state));
  }
