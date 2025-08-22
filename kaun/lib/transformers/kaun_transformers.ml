(** Transformer building blocks implementation *)

open Rune

(** Scaled dot-product attention implementation *)
let scaled_dot_product_attention ?mask ?dropout ?is_causal ?scale q k v =
  let device = device q in
  let dtype = dtype q in
  let shape_q = shape q in

  (* Determine dimensions *)
  let _num_heads = shape_q.(1) in
  let seq_len = shape_q.(2) in
  let head_dim = shape_q.(3) in

  (* Compute scale factor if not provided *)
  let scale =
    match scale with
    | Some s -> s
    | None -> 1.0 /. Stdlib.sqrt (float_of_int head_dim)
  in

  (* Compute attention scores: Q @ K^T *)
  let k_transposed = transpose ~axes:[| 0; 1; 3; 2 |] k in
  let scores = matmul q k_transposed in

  (* Apply scaling *)
  let scores = mul scores (scalar device dtype scale) in

  (* Apply causal mask if requested *)
  let scores =
    if Option.value is_causal ~default:false then
      let causal_mask =
        let mask_shape = [| 1; 1; seq_len; seq_len |] in
        let mask = ones device dtype mask_shape in
        (* Create lower triangular mask *)
        let mask_data = unsafe_data mask in
        for i = 0 to seq_len - 1 do
          for j = i + 1 to seq_len - 1 do
            let idx = (i * seq_len) + j in
            Bigarray_ext.Array1.set mask_data idx 0.0
          done
        done;
        mask
      in
      (* Apply mask: set masked positions to -inf *)
      let neg_inf = scalar device dtype (-1e10) in
      where (cast uint8 causal_mask) scores neg_inf
    else scores
  in

  (* Apply additional mask if provided *)
  let scores =
    match mask with
    | Some m ->
        let neg_inf = scalar device dtype (-1e10) in
        where m scores neg_inf
    | None -> scores
  in

  (* Apply softmax *)
  let attention_weights = softmax ~axes:[| 3 |] scores in

  (* Apply dropout if specified *)
  let attention_weights =
    match dropout with
    | Some p when p > 0.0 ->
        (* Simple dropout implementation - in practice would use RNG *)
        let dropout_mask =
          let mask = Rune.rand device dtype (shape attention_weights) in
          greater mask (scalar device dtype p)
        in
        let scale = scalar device dtype (1.0 /. (1.0 -. p)) in
        mul
          (where dropout_mask attention_weights (zeros_like attention_weights))
          scale
    | _ -> attention_weights
  in

  (* Apply attention to values: attention @ V *)
  matmul attention_weights v

(** Rotary Position Embeddings *)
module Rope = struct
  type t = {
    inv_freq : float array;
        (* Store frequencies instead of precomputed cos/sin *)
    dim : int;
    max_seq_len : int;
    base : float;
  }

  (* Suppress unused field warning *)
  let _ = fun (t : t) -> t.base

  let make ?(base = 10000.0) ~dim ~max_seq_len () =
    if dim mod 2 <> 0 then invalid_arg "RoPE dimension must be even";

    (* Compute and store inverse frequencies *)
    let half_dim = dim / 2 in
    let inv_freq =
      Array.init half_dim (fun i ->
          1.0 /. (base ** (float_of_int (2 * i) /. float_of_int dim)))
    in

    { inv_freq; dim; max_seq_len; base }

  let compute_freqs_for_device t seq_len device dtype =
    (* Create position indices *)
    let positions = arange_f device dtype 0.0 (float_of_int seq_len) 1.0 in
    let positions = reshape [| seq_len; 1 |] positions in

    (* Create frequency tensor from stored array *)
    let half_dim = Array.length t.inv_freq in
    let freqs =
      let freqs_list = Array.to_list t.inv_freq in
      let result = zeros device dtype [| half_dim |] in
      List.iteri
        (fun i v ->
          let v_tensor = scalar device dtype v in
          set [ i ] result v_tensor)
        freqs_list;
      result
    in
    let freqs = reshape [| 1; half_dim |] freqs in

    (* Compute angles: positions @ freqs *)
    let angles = matmul positions freqs in

    (* Compute and return cos and sin *)
    (cos angles, sin angles)

  let _get_cos_sin t seq_len =
    if seq_len > t.max_seq_len then
      invalid_arg
        (Printf.sprintf "Sequence length %d exceeds maximum %d" seq_len
           t.max_seq_len);

    (* Compute on default device/dtype for now *)
    compute_freqs_for_device t seq_len c float32

  let rotate_half x =
    (* Split tensor in half along last dimension *)
    let shape_x = shape x in
    let half_dim = shape_x.(Array.length shape_x - 1) / 2 in
    let x1 = slice [ R []; R []; R []; R [ 0; half_dim ] ] x in
    let x2 = slice [ R []; R []; R []; R [ half_dim; shape_x.(3) ] ] x in
    (* Return concatenation of [-x2, x1] *)
    concatenate ~axis:3 [ neg x2; x1 ]

  let apply t ?seq_len q k =
    let shape_q = shape q in
    let actual_seq_len = Option.value seq_len ~default:shape_q.(2) in
    let batch_size = shape_q.(0) in
    let num_heads = shape_q.(1) in
    let head_dim = shape_q.(3) in

    if head_dim <> t.dim / 2 then
      invalid_arg
        (Printf.sprintf "Head dimension %d doesn't match RoPE dimension %d"
           head_dim (t.dim / 2));

    (* Compute cos and sin on the same device as q/k *)
    let device_q = device q in
    let dtype_q = dtype q in
    let cos, sin = compute_freqs_for_device t actual_seq_len device_q dtype_q in

    (* Reshape and broadcast to match q/k shape *)
    let cos = reshape [| 1; 1; actual_seq_len; head_dim |] cos in
    let sin = reshape [| 1; 1; actual_seq_len; head_dim |] sin in
    let cos =
      broadcast_to [| batch_size; num_heads; actual_seq_len; head_dim |] cos
    in
    let sin =
      broadcast_to [| batch_size; num_heads; actual_seq_len; head_dim |] sin
    in

    (* Apply rotation: q_rot = q * cos + rotate_half(q) * sin *)
    let q_rot = add (mul q cos) (mul (rotate_half q) sin) in
    let k_rot = add (mul k cos) (mul (rotate_half k) sin) in

    (q_rot, k_rot)
end
