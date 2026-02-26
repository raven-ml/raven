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

let dropout ~rate x =
  if rate < 0.0 || rate >= 1.0 then
    invalid_argf_fn "dropout" "rate must satisfy 0.0 <= rate < 1.0";
  let tensor_dtype = Rune.dtype x in
  if not (Nx_core.Dtype.is_float tensor_dtype) then
    invalid_argf_fn "dropout" "requires floating point dtype";
  if rate = 0.0 then x
  else
    let keep_prob = 1.0 -. rate in
    let random_vals = Rune.rand tensor_dtype (Rune.shape x) in
    let threshold = Rune.scalar_like x keep_prob in
    let keep_mask = Rune.less random_vals threshold in
    let keep_mask_float = Rune.cast tensor_dtype keep_mask in
    let scale = Rune.scalar_like x (1.0 /. keep_prob) in
    Rune.mul x (Rune.mul keep_mask_float scale)

(* Attention *)

let dot_product_attention (type b) ?attention_mask ?scale ?dropout_rate
    ?(is_causal = false) (q : (float, b) Rune.t) (k : (float, b) Rune.t)
    (v : (float, b) Rune.t) =
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
    match dropout_rate with None -> probs | Some rate -> dropout ~rate probs
  in
  Rune.matmul probs v

(* Conv / Pool helpers *)

let ceildiv a b = (a + b - 1) / b

let calculate_nn_padding input_spatial ~kernel_size ~stride ~dilation
    ~(padding : [ `Same | `Valid ]) =
  let k = Array.length kernel_size in
  match padding with
  | `Valid -> Array.make k (0, 0)
  | `Same ->
      Array.init k (fun i ->
          let eff_k = (dilation.(i) * (kernel_size.(i) - 1)) + 1 in
          let out = ceildiv input_spatial.(i) stride.(i) in
          let total =
            Stdlib.max 0 (((out - 1) * stride.(i)) + eff_k - input_spatial.(i))
          in
          (total / 2, total - (total / 2)))

let apply_ceil_mode input_spatial ~kernel_size ~stride ~dilation ~padding
    ~ceil_mode =
  if not ceil_mode then padding
  else
    Array.init (Array.length kernel_size) (fun i ->
        let pb, pa = padding.(i) in
        let padded = input_spatial.(i) + pb + pa in
        let eff_k = (dilation.(i) * (kernel_size.(i) - 1)) + 1 in
        let out_floor = ((padded - eff_k) / stride.(i)) + 1 in
        let out_ceil = ceildiv (padded - eff_k) stride.(i) + 1 in
        if out_ceil > out_floor then
          let extra = ((out_ceil - 1) * stride.(i)) + eff_k - padded in
          (pb, pa + extra)
        else (pb, pa))

(* Convolution *)

let conv1d ?(groups = 1) ?(stride = 1) ?(dilation = 1) ?(padding = `Valid) ?bias
    x w =
  let x_shape = Rune.shape x in
  let w_shape = Rune.shape w in
  if Array.length x_shape <> 3 then
    invalid_argf_fn "conv1d" "input must be 3D (N, C_in, L)";
  if Array.length w_shape <> 3 then
    invalid_argf_fn "conv1d" "weight must be 3D (C_out, C_in/groups, K)";
  let n = x_shape.(0) in
  let cin = x_shape.(1) in
  let cout = w_shape.(0) in
  let cin_per_group = w_shape.(1) in
  if cin <> groups * cin_per_group then
    invalid_argf_fn "conv1d" "C_in=%d does not match groups=%d * C_in/g=%d" cin
      groups cin_per_group;
  let kernel_size = [| w_shape.(2) |] in
  let stride_arr = [| stride |] in
  let dilation_arr = [| dilation |] in
  let input_spatial = [| x_shape.(2) |] in
  let pad_pairs =
    calculate_nn_padding input_spatial ~kernel_size ~stride:stride_arr
      ~dilation:dilation_arr ~padding
  in
  let kernel_elements = w_shape.(2) in
  (* unfold: (N, C_in, L_in) -> (N, C_in, K, L_out) *)
  let x_unf =
    Rune.extract_patches ~kernel_size ~stride:stride_arr ~dilation:dilation_arr
      ~padding:pad_pairs x
  in
  let x_unf_shape = Rune.shape x_unf in
  let l_out = x_unf_shape.(3) in
  (* Merge channels and kernel: (N, C_in*K, L_out) *)
  let x_col = Rune.reshape [| n; cin * kernel_elements; l_out |] x_unf in
  let result =
    if groups = 1 then
      let w_flat = Rune.reshape [| cout; cin * kernel_elements |] w in
      Rune.matmul w_flat x_col
    else
      let rcout = cout / groups in
      let x_grouped =
        Rune.reshape
          [| n; groups; cin_per_group * kernel_elements; l_out |]
          x_col
      in
      let w_grouped =
        Rune.reshape [| groups; rcout; cin_per_group * kernel_elements |] w
      in
      let x_batched =
        Rune.reshape
          [| n * groups; cin_per_group * kernel_elements; l_out |]
          x_grouped
      in
      let w_expanded = Rune.unsqueeze ~axes:[ 0 ] w_grouped in
      let w_expanded =
        Rune.expand
          [| n; groups; rcout; cin_per_group * kernel_elements |]
          w_expanded
      in
      let w_expanded =
        Rune.reshape
          [| n * groups; rcout; cin_per_group * kernel_elements |]
          w_expanded
      in
      let result = Rune.matmul w_expanded x_batched in
      let result = Rune.reshape [| n; groups; rcout; l_out |] result in
      Rune.reshape [| n; cout; l_out |] result
  in
  match bias with
  | None -> result
  | Some b -> Rune.add result (Rune.reshape [| 1; cout; 1 |] b)

let conv2d ?(groups = 1) ?(stride = (1, 1)) ?(dilation = (1, 1))
    ?(padding = `Valid) ?bias x w =
  let x_shape = Rune.shape x in
  let w_shape = Rune.shape w in
  if Array.length x_shape <> 4 then
    invalid_argf_fn "conv2d" "input must be 4D (N, C_in, H, W)";
  if Array.length w_shape <> 4 then
    invalid_argf_fn "conv2d" "weight must be 4D (C_out, C_in/groups, kH, kW)";
  let n = x_shape.(0) in
  let cin = x_shape.(1) in
  let cout = w_shape.(0) in
  let cin_per_group = w_shape.(1) in
  if cin <> groups * cin_per_group then
    invalid_argf_fn "conv2d" "C_in=%d does not match groups=%d * C_in/g=%d" cin
      groups cin_per_group;
  let sh, sw = stride in
  let dh, dw = dilation in
  let kernel_size = [| w_shape.(2); w_shape.(3) |] in
  let stride_arr = [| sh; sw |] in
  let dilation_arr = [| dh; dw |] in
  let input_spatial = [| x_shape.(2); x_shape.(3) |] in
  let pad_pairs =
    calculate_nn_padding input_spatial ~kernel_size ~stride:stride_arr
      ~dilation:dilation_arr ~padding
  in
  let kernel_elements = w_shape.(2) * w_shape.(3) in
  (* unfold: (N, C_in, H, W) -> (N, C_in, kH*kW, L) *)
  let x_unf =
    Rune.extract_patches ~kernel_size ~stride:stride_arr ~dilation:dilation_arr
      ~padding:pad_pairs x
  in
  let x_unf_shape = Rune.shape x_unf in
  let l_out = x_unf_shape.(3) in
  (* Merge channels and kernel: (N, C_in*kH*kW, L) *)
  let x_col = Rune.reshape [| n; cin * kernel_elements; l_out |] x_unf in
  let result =
    if groups = 1 then
      let w_flat = Rune.reshape [| cout; cin * kernel_elements |] w in
      Rune.matmul w_flat x_col
    else
      let rcout = cout / groups in
      let x_grouped =
        Rune.reshape
          [| n; groups; cin_per_group * kernel_elements; l_out |]
          x_col
      in
      let w_grouped =
        Rune.reshape [| groups; rcout; cin_per_group * kernel_elements |] w
      in
      let x_batched =
        Rune.reshape
          [| n * groups; cin_per_group * kernel_elements; l_out |]
          x_grouped
      in
      let w_expanded = Rune.unsqueeze ~axes:[ 0 ] w_grouped in
      let w_expanded =
        Rune.expand
          [| n; groups; rcout; cin_per_group * kernel_elements |]
          w_expanded
      in
      let w_expanded =
        Rune.reshape
          [| n * groups; rcout; cin_per_group * kernel_elements |]
          w_expanded
      in
      let result = Rune.matmul w_expanded x_batched in
      let result = Rune.reshape [| n; groups; rcout; l_out |] result in
      Rune.reshape [| n; cout; l_out |] result
  in
  (* Reshape from (N, C_out, L) to (N, C_out, H_out, W_out) *)
  let padded_h = input_spatial.(0) + fst pad_pairs.(0) + snd pad_pairs.(0) in
  let padded_w = input_spatial.(1) + fst pad_pairs.(1) + snd pad_pairs.(1) in
  let eff_kh = ((kernel_size.(0) - 1) * dh) + 1 in
  let eff_kw = ((kernel_size.(1) - 1) * dw) + 1 in
  let h_out = ((padded_h - eff_kh) / sh) + 1 in
  let w_out = ((padded_w - eff_kw) / sw) + 1 in
  let result = Rune.reshape [| n; cout; h_out; w_out |] result in
  match bias with
  | None -> result
  | Some b -> Rune.add result (Rune.reshape [| 1; cout; 1; 1 |] b)

(* Pooling *)

let max_pool1d ~kernel_size ?(stride = 1) ?(dilation = 1) ?(padding = `Valid)
    ?(ceil_mode = false) x =
  let x_shape = Rune.shape x in
  if Array.length x_shape <> 3 then
    invalid_argf_fn "max_pool1d" "input must be 3D (N, C, L)";
  let n = x_shape.(0) in
  let c = x_shape.(1) in
  let kernel_size_arr = [| kernel_size |] in
  let stride_arr = [| stride |] in
  let dilation_arr = [| dilation |] in
  let input_spatial = [| x_shape.(2) |] in
  let pad_pairs =
    calculate_nn_padding input_spatial ~kernel_size:kernel_size_arr
      ~stride:stride_arr ~dilation:dilation_arr ~padding
  in
  let pad_pairs =
    apply_ceil_mode input_spatial ~kernel_size:kernel_size_arr
      ~stride:stride_arr ~dilation:dilation_arr ~padding:pad_pairs ~ceil_mode
  in
  (* unfold: (N, C, L) -> (N, C, K, L_out) *)
  let x_unf =
    Rune.extract_patches ~kernel_size:kernel_size_arr ~stride:stride_arr
      ~dilation:dilation_arr ~padding:pad_pairs x
  in
  let x_unf_ndim = Rune.ndim x_unf in
  let reduced = Rune.max x_unf ~axes:[ x_unf_ndim - 2 ] ~keepdims:false in
  let x_unf_shape = Rune.shape x_unf in
  let l_out = x_unf_shape.(x_unf_ndim - 1) in
  Rune.reshape [| n; c; l_out |] reduced

let max_pool2d ~kernel_size ?(stride = (1, 1)) ?(dilation = (1, 1))
    ?(padding = `Valid) ?(ceil_mode = false) x =
  let x_shape = Rune.shape x in
  if Array.length x_shape <> 4 then
    invalid_argf_fn "max_pool2d" "input must be 4D (N, C, H, W)";
  let n = x_shape.(0) in
  let c = x_shape.(1) in
  let kh, kw = kernel_size in
  let sh, sw = stride in
  let dh, dw = dilation in
  let kernel_size_arr = [| kh; kw |] in
  let stride_arr = [| sh; sw |] in
  let dilation_arr = [| dh; dw |] in
  let input_spatial = [| x_shape.(2); x_shape.(3) |] in
  let pad_pairs =
    calculate_nn_padding input_spatial ~kernel_size:kernel_size_arr
      ~stride:stride_arr ~dilation:dilation_arr ~padding
  in
  let pad_pairs =
    apply_ceil_mode input_spatial ~kernel_size:kernel_size_arr
      ~stride:stride_arr ~dilation:dilation_arr ~padding:pad_pairs ~ceil_mode
  in
  let x_unf =
    Rune.extract_patches ~kernel_size:kernel_size_arr ~stride:stride_arr
      ~dilation:dilation_arr ~padding:pad_pairs x
  in
  let x_unf_ndim = Rune.ndim x_unf in
  let reduced = Rune.max x_unf ~axes:[ x_unf_ndim - 2 ] ~keepdims:false in
  let x_unf_shape = Rune.shape x_unf in
  let l_out = x_unf_shape.(x_unf_ndim - 1) in
  let padded_h = input_spatial.(0) + fst pad_pairs.(0) + snd pad_pairs.(0) in
  let padded_w = input_spatial.(1) + fst pad_pairs.(1) + snd pad_pairs.(1) in
  let eff_kh = ((kh - 1) * dh) + 1 in
  let eff_kw = ((kw - 1) * dw) + 1 in
  let h_out = ((padded_h - eff_kh) / sh) + 1 in
  let w_out = ((padded_w - eff_kw) / sw) + 1 in
  let _ = l_out in
  Rune.reshape [| n; c; h_out; w_out |] reduced

let avg_pool1d ~kernel_size ?(stride = 1) ?(dilation = 1) ?(padding = `Valid)
    ?(ceil_mode = false) ?(count_include_pad = true) x =
  let x_shape = Rune.shape x in
  if Array.length x_shape <> 3 then
    invalid_argf_fn "avg_pool1d" "input must be 3D (N, C, L)";
  let n = x_shape.(0) in
  let c = x_shape.(1) in
  let kernel_size_arr = [| kernel_size |] in
  let stride_arr = [| stride |] in
  let dilation_arr = [| dilation |] in
  let input_spatial = [| x_shape.(2) |] in
  let pad_pairs =
    calculate_nn_padding input_spatial ~kernel_size:kernel_size_arr
      ~stride:stride_arr ~dilation:dilation_arr ~padding
  in
  let pad_pairs =
    apply_ceil_mode input_spatial ~kernel_size:kernel_size_arr
      ~stride:stride_arr ~dilation:dilation_arr ~padding:pad_pairs ~ceil_mode
  in
  let x_unf =
    Rune.extract_patches ~kernel_size:kernel_size_arr ~stride:stride_arr
      ~dilation:dilation_arr ~padding:pad_pairs x
  in
  let x_unf_ndim = Rune.ndim x_unf in
  let x_unf_shape = Rune.shape x_unf in
  let l_out = x_unf_shape.(x_unf_ndim - 1) in
  let summed = Rune.sum x_unf ~axes:[ x_unf_ndim - 2 ] in
  let result = Rune.reshape [| n; c; l_out |] summed in
  if count_include_pad then Rune.div_s result (float_of_int kernel_size)
  else
    let ones = Rune.ones_like x in
    let ones_unf =
      Rune.extract_patches ~kernel_size:kernel_size_arr ~stride:stride_arr
        ~dilation:dilation_arr ~padding:pad_pairs ones
    in
    let count = Rune.sum ones_unf ~axes:[ Rune.ndim ones_unf - 2 ] in
    let count = Rune.reshape [| n; c; l_out |] count in
    Rune.div result count

let avg_pool2d ~kernel_size ?(stride = (1, 1)) ?(dilation = (1, 1))
    ?(padding = `Valid) ?(ceil_mode = false) ?(count_include_pad = true) x =
  let x_shape = Rune.shape x in
  if Array.length x_shape <> 4 then
    invalid_argf_fn "avg_pool2d" "input must be 4D (N, C, H, W)";
  let n = x_shape.(0) in
  let c = x_shape.(1) in
  let kh, kw = kernel_size in
  let sh, sw = stride in
  let dh, dw = dilation in
  let kernel_size_arr = [| kh; kw |] in
  let stride_arr = [| sh; sw |] in
  let dilation_arr = [| dh; dw |] in
  let input_spatial = [| x_shape.(2); x_shape.(3) |] in
  let pad_pairs =
    calculate_nn_padding input_spatial ~kernel_size:kernel_size_arr
      ~stride:stride_arr ~dilation:dilation_arr ~padding
  in
  let pad_pairs =
    apply_ceil_mode input_spatial ~kernel_size:kernel_size_arr
      ~stride:stride_arr ~dilation:dilation_arr ~padding:pad_pairs ~ceil_mode
  in
  let x_unf =
    Rune.extract_patches ~kernel_size:kernel_size_arr ~stride:stride_arr
      ~dilation:dilation_arr ~padding:pad_pairs x
  in
  let x_unf_ndim = Rune.ndim x_unf in
  let x_unf_shape = Rune.shape x_unf in
  let l_out = x_unf_shape.(x_unf_ndim - 1) in
  let summed = Rune.sum x_unf ~axes:[ x_unf_ndim - 2 ] in
  let padded_h = input_spatial.(0) + fst pad_pairs.(0) + snd pad_pairs.(0) in
  let padded_w = input_spatial.(1) + fst pad_pairs.(1) + snd pad_pairs.(1) in
  let eff_kh = ((kh - 1) * dh) + 1 in
  let eff_kw = ((kw - 1) * dw) + 1 in
  let h_out = ((padded_h - eff_kh) / sh) + 1 in
  let w_out = ((padded_w - eff_kw) / sw) + 1 in
  let _ = l_out in
  let result = Rune.reshape [| n; c; h_out; w_out |] summed in
  if count_include_pad then
    let kernel_numel = float_of_int (kh * kw) in
    Rune.div_s result kernel_numel
  else
    let ones = Rune.ones_like x in
    let ones_unf =
      Rune.extract_patches ~kernel_size:kernel_size_arr ~stride:stride_arr
        ~dilation:dilation_arr ~padding:pad_pairs ones
    in
    let count = Rune.sum ones_unf ~axes:[ Rune.ndim ones_unf - 2 ] in
    let count = Rune.reshape [| n; c; h_out; w_out |] count in
    Rune.div result count
