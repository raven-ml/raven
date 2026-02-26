(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let invalid_argf_fn fn fmt =
  Printf.ksprintf (fun msg -> invalid_argf "Fn.%s: %s" fn msg) fmt

(* Helpers *)

let normalize_axis ~fn ~ndim ax =
  let axis = if ax < 0 then ndim + ax else ax in
  if axis < 0 || axis >= ndim then
    invalid_argf_fn fn "axis %d out of bounds for rank %d" ax ndim;
  axis

let normalize_axes ~fn ~ndim axes =
  match axes with
  | [] -> invalid_argf_fn fn "axes must contain at least one axis"
  | lst -> List.map (normalize_axis ~fn ~ndim) lst

let keep_shape ~axes x_shape =
  Array.mapi
    (fun idx dim -> if List.exists (fun ax -> ax = idx) axes then 1 else dim)
    x_shape

let unaffected_axes ~ndim ~axes =
  Array.init ndim Fun.id |> Array.to_list
  |> List.filter (fun ax -> not (List.exists (( = ) ax) axes))

let core_shape ~axes:unaffected x_shape =
  Array.of_list (List.map (fun ax -> x_shape.(ax)) unaffected)

let broadcast_param ~fn ~name ~x_shape ~keep_shape ~core_shape param =
  let param_shape = Rune.shape param in
  if param_shape = keep_shape then param
  else if param_shape = core_shape then Rune.reshape keep_shape param
  else if param_shape = x_shape then param
  else
    invalid_argf_fn fn "%s: shape must match normalized axes or remaining axes"
      name

(* Normalization *)

let batch_norm ?axes ?(epsilon = 1e-5) ~scale ~bias x =
  let ndim = Rune.ndim x in
  let axes =
    let default =
      match axes with
      | Some ax -> ax
      | None ->
          if ndim = 2 then [ 0 ] else if ndim = 4 then [ 0; 2; 3 ] else [ 0 ]
    in
    normalize_axes ~fn:"batch_norm" ~ndim default
  in
  let x_shape = Rune.shape x in
  let keep = keep_shape ~axes x_shape in
  let unaffected = unaffected_axes ~ndim ~axes in
  let core = core_shape ~axes:unaffected x_shape in
  let broadcast name param =
    let param =
      if Rune.dtype param <> Rune.dtype x then Rune.cast (Rune.dtype x) param
      else param
    in
    broadcast_param ~fn:"batch_norm" ~name ~x_shape ~keep_shape:keep
      ~core_shape:core param
  in
  let mean_x = Rune.mean x ~axes ~keepdims:true in
  let variance = Rune.var x ~axes ~keepdims:true in
  let eps = Rune.scalar_like x epsilon in
  let normalized =
    Rune.mul (Rune.sub x mean_x) (Rune.rsqrt (Rune.add variance eps))
  in
  let scale_b = broadcast "scale" scale in
  let bias_b = broadcast "bias" bias in
  Rune.add (Rune.mul normalized scale_b) bias_b

let rms_norm ?axes ?(epsilon = 1e-5) ?gamma x =
  let ndim = Rune.ndim x in
  let axes =
    let default = match axes with Some ax -> ax | None -> [ -1 ] in
    normalize_axes ~fn:"rms_norm" ~ndim default
  in
  let x_shape = Rune.shape x in
  let keep = keep_shape ~axes x_shape in
  let mean_square = Rune.mean (Rune.mul x x) ~axes ~keepdims:true in
  let eps = Rune.scalar_like x epsilon in
  let normalized = Rune.mul x (Rune.rsqrt (Rune.add mean_square eps)) in
  match gamma with
  | None -> normalized
  | Some gamma ->
      let gamma_shape = Rune.shape gamma in
      let gamma =
        if gamma_shape = keep then gamma
        else
          let unaffected = unaffected_axes ~ndim ~axes in
          let core = core_shape ~axes:unaffected x_shape in
          if gamma_shape = core then Rune.reshape keep gamma
          else if gamma_shape = x_shape then gamma
          else
            invalid_argf_fn "rms_norm"
              "gamma: shape must match normalized axes or remaining axes"
      in
      Rune.mul normalized gamma

let layer_norm ?(axes = [ -1 ]) ?(epsilon = 1e-5) ?gamma ?beta x =
  let ndim = Rune.ndim x in
  let axes =
    List.map
      (fun ax ->
        let axis = if ax < 0 then ndim + ax else ax in
        if axis < 0 || axis >= ndim then
          invalid_argf_fn "layer_norm" "axis %d out of bounds for rank %d" ax
            ndim;
        axis)
      axes
  in
  let x_shape = Rune.shape x in
  let keep =
    Array.mapi (fun idx dim -> if List.mem idx axes then dim else 1) x_shape
  in
  let broadcast_param name param =
    let param_shape = Rune.shape param in
    if param_shape = x_shape then param
    else if param_shape = keep then param
    else
      let axes_shape = Array.of_list (List.map (fun ax -> x_shape.(ax)) axes) in
      if param_shape = axes_shape then Rune.reshape keep param
      else
        invalid_argf_fn "layer_norm" "%s: shape must match normalized axes" name
  in
  let mean_x = Rune.mean x ~axes ~keepdims:true in
  let centered = Rune.sub x mean_x in
  let variance = Rune.mean (Rune.mul centered centered) ~axes ~keepdims:true in
  let eps = Rune.scalar_like x epsilon in
  let inv_std = Rune.rsqrt (Rune.add variance eps) in
  let normalized = Rune.mul centered inv_std in
  let with_scale =
    match gamma with
    | None -> normalized
    | Some gamma ->
        let gamma_broadcast = broadcast_param "gamma" gamma in
        Rune.mul normalized gamma_broadcast
  in
  match beta with
  | None -> with_scale
  | Some beta ->
      let beta_broadcast = broadcast_param "beta" beta in
      Rune.add with_scale beta_broadcast

(* Embedding *)

let embedding ?(scale = true) ~embedding indices =
  let embed_shape = Rune.shape embedding in
  if Array.length embed_shape <> 2 then
    invalid_argf_fn "embedding"
      "embedding matrix must have shape [vocab_size; embed_dim]";
  let embed_dim = embed_shape.(1) in
  let indices_shape = Rune.shape indices in
  let is_scalar = Array.length indices_shape = 0 in
  let vocab_size = embed_shape.(0) in
  if vocab_size <= 0 then
    invalid_argf_fn "embedding" "vocabulary dimension must be positive";
  let flat_size = Array.fold_left ( * ) 1 indices_shape in
  let indices_flat =
    if is_scalar then Rune.reshape [| 1 |] indices
    else Rune.reshape [| flat_size |] indices
  in
  let gathered = Rune.take ~axis:0 indices_flat embedding in
  let output_shape =
    if is_scalar then [| embed_dim |]
    else Array.append indices_shape [| embed_dim |]
  in
  let embedded = Rune.reshape output_shape gathered in
  if not scale then embedded
  else
    let factor =
      Rune.scalar_like embedding (Stdlib.sqrt (float_of_int embed_dim))
    in
    Rune.mul embedded factor

(* Dropout *)

let dropout ~key ~rate x =
  if rate < 0.0 || rate >= 1.0 then
    invalid_argf_fn "dropout" "rate must satisfy 0.0 <= rate < 1.0";
  let tensor_dtype = Rune.dtype x in
  if not (Nx_core.Dtype.is_float tensor_dtype) then
    invalid_argf_fn "dropout" "requires floating point dtype";
  if rate = 0.0 then x
  else
    let keep_prob = 1.0 -. rate in
    let random_vals = Rune.rand tensor_dtype ~key (Rune.shape x) in
    let threshold = Rune.scalar_like x keep_prob in
    let keep_mask = Rune.less random_vals threshold in
    let keep_mask_float = Rune.cast tensor_dtype keep_mask in
    let scale = Rune.scalar_like x (1.0 /. keep_prob) in
    Rune.mul x (Rune.mul keep_mask_float scale)

(* Attention *)

let dot_product_attention (type b) ?attention_mask ?scale ?dropout_rate
    ?dropout_key ?(is_causal = false) (q : (float, b) Rune.t)
    (k : (float, b) Rune.t) (v : (float, b) Rune.t) =
  let check_float name (t : (float, b) Rune.t) =
    match Rune.dtype t with
    | Rune.Float16 -> ()
    | Rune.Float32 -> ()
    | Rune.Float64 -> ()
    | _ ->
        invalid_argf_fn "dot_product_attention"
          "%s: requires floating point dtype" name
  in
  check_float "query" q;
  check_float "key" k;
  check_float "value" v;
  let q_shape = Rune.shape q in
  let k_shape = Rune.shape k in
  let v_shape = Rune.shape v in
  let q_rank = Array.length q_shape in
  if q_rank < 2 then
    invalid_argf_fn "dot_product_attention" "query: must have rank >= 2";
  if Array.length k_shape <> q_rank || Array.length v_shape <> q_rank then
    invalid_argf_fn "dot_product_attention" "key/value: must match query rank";
  let depth = q_shape.(q_rank - 1) in
  if k_shape.(q_rank - 1) <> depth then
    invalid_argf_fn "dot_product_attention"
      "key last dim %d does not match query last dim %d"
      k_shape.(q_rank - 1)
      depth;
  let scale_factor =
    match scale with
    | Some s -> s
    | None -> 1.0 /. Stdlib.sqrt (float_of_int depth)
  in
  let transpose_last_two tensor =
    let nd = Array.length (Rune.shape tensor) in
    if nd < 2 then
      invalid_argf_fn "dot_product_attention" "key/value: must have rank >= 2";
    let axes = Array.init nd Fun.id in
    let tmp = axes.(nd - 1) in
    axes.(nd - 1) <- axes.(nd - 2);
    axes.(nd - 2) <- tmp;
    Rune.transpose tensor ~axes:(Array.to_list axes)
  in
  let k_t = transpose_last_two k in
  let scores = Rune.matmul q k_t in
  let scores =
    if scale_factor = 1.0 then scores
    else Rune.mul scores (Rune.scalar_like scores scale_factor)
  in
  let scores =
    if is_causal then (
      let scores_shape = Rune.shape scores in
      let seq_len_q = scores_shape.(q_rank - 2) in
      let seq_len_k = scores_shape.(q_rank - 1) in
      if seq_len_q <> seq_len_k then
        invalid_argf_fn "dot_product_attention"
          "causal masking requires seq_len_q == seq_len_k";
      let ones_matrix =
        Rune.full (Rune.dtype scores) [| seq_len_q; seq_len_k |] 1.0
      in
      let causal_mask = Rune.tril ones_matrix in
      let causal_mask = Rune.cast Rune.bool causal_mask in
      let causal_mask = Rune.broadcast_to scores_shape causal_mask in
      let neg_inf = Rune.scalar_like scores (-1e9) in
      Rune.where causal_mask scores neg_inf)
    else scores
  in
  let scores =
    match attention_mask with
    | None -> scores
    | Some mask ->
        let neg_inf = Rune.scalar_like scores (-1e9) in
        Rune.where mask scores neg_inf
  in
  let probs = Rune.softmax ~axes:[ -1 ] scores in
  let probs =
    match dropout_rate with
    | None -> probs
    | Some rate ->
        let key =
          match dropout_key with
          | Some k -> k
          | None ->
              invalid_arg
                "Fn.dot_product_attention: dropout_key required when \
                 dropout_rate is set"
        in
        dropout ~key ~rate probs
  in
  Rune.matmul probs v
