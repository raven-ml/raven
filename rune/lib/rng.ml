(* JAX-style splittable random number generation *)

open Tensor_with_debug

type key = int

let key seed = Stdlib.abs seed land 0x7FFFFFFF (* Ensure positive 31-bit int *)

let split ?(n = 2) key =
  (* MurmurHash-inspired integer hash for better distribution *)
  let hash x =
    let open Int32 in
    let x = of_int x in
    let x = logxor x (shift_right_logical x 16) in
    let x = mul x 0x85ebca6bl in
    let x = logxor x (shift_right_logical x 13) in
    let x = mul x 0xc2b2ae35l in
    let x = logxor x (shift_right_logical x 16) in
    to_int (logand x 0x7FFFFFFFl)
  in
  Array.init n (fun i -> hash ((key * (n + 1)) + i + 1))

let fold_in key data =
  let hash x =
    let open Int32 in
    let x = of_int x in
    let x = logxor x (shift_right_logical x 16) in
    let x = mul x 0x85ebca6bl in
    let x = logxor x (shift_right_logical x 13) in
    let x = mul x 0xc2b2ae35l in
    let x = logxor x (shift_right_logical x 16) in
    to_int (logand x 0x7FFFFFFFl)
  in
  hash (key lxor data)

let to_int key = key

(* Random sampling functions *)

let uniform key dtype shape =
  (* Use the key as seed for existing rand function *)
  rand dtype ~seed:key shape

let normal key dtype shape =
  (* Use the key as seed for existing randn function *)
  randn dtype ~seed:key shape

let randint key ~min ~max shape =
  if min >= max then
    invalid_arg
      (Printf.sprintf "randint: min (%d) must be less than max (%d)" min max);
  let range = max - min in
  let uniform_vals = uniform key Tensor.Float32 shape in
  let scaled = mul uniform_vals (scalar Tensor.Float32 (float_of_int range)) in
  let shifted = add scaled (scalar Tensor.Float32 (float_of_int min)) in
  astype Tensor.Int32 shifted

let bernoulli key ~p shape =
  if p < 0. || p > 1. then
    invalid_arg (Printf.sprintf "bernoulli: p (%.2f) must be in [0, 1]" p);
  let uniform_vals = uniform key Tensor.Float32 shape in
  let threshold = scalar Float32 p in
  cmplt uniform_vals threshold

let permutation key n =
  if n <= 0 then
    invalid_arg (Printf.sprintf "permutation: n (%d) must be positive" n);
  (* Generate random values for each index *)
  let random_vals = uniform key Tensor.Float32 [| n |] in
  (* Get argsort to create permutation *)
  argsort random_vals ~axis:0 ~descending:false

let shuffle key x =
  let shape_x = Tensor.shape x in
  if Array.length shape_x = 0 then x
  else
    let n = shape_x.(0) in
    let perm = permutation key n in
    take ~axis:0 perm x

let categorical (type a b) (key : key) ?(axis : int = -1)
    ?(shape : int array = [||]) (logits : (a, b) Tensor.t) : Tensor.int32_t =
  (* Gumbel-max categorical sampler with optional prefix shape (JAX-like) *)

  (* Work in a floating dtype for the Gumbel transform; don't mutate the
     input *)
  let dtype = Tensor.dtype logits in
  let shape_array = Tensor.shape logits in
  let ndim = Array.length shape_array in
  let axis = if axis < 0 then ndim + axis else axis in
  if axis < 0 || axis >= ndim then
    invalid_arg
      (Printf.sprintf
         "categorical: axis (%d) out of bounds for logits ndim (%d)" axis ndim);

  (* Build the full shape for the uniform/Gumbel noise: prefix_shape +
     logits_shape *)
  let full_shape = Array.append shape shape_array in

  let run_float32 eps =
    let u = uniform key Tensor.Float32 full_shape in
    let u_clamped = clip u ~min:eps ~max:(1. -. eps) in
    let neg_one = scalar Tensor.Float32 (-1.0) in
    let log_u = log u_clamped in
    let neg_log_u = mul log_u neg_one in
    let log_neg_log_u = log neg_log_u in
    let gumbel = mul log_neg_log_u neg_one |> astype dtype in
    let noisy = add logits gumbel in
    let prefix_len = Array.length shape in
    let argmax_axis = axis + prefix_len in
    let inds = argmax noisy ~axis:argmax_axis ~keepdims:false in
    astype Tensor.Int32 inds
  in

  let run_float64 eps =
    let u = uniform key Tensor.Float64 full_shape in
    let u_clamped = clip u ~min:eps ~max:(1. -. eps) in
    let neg_one = scalar Tensor.Float64 (-1.0) in
    let log_u = log u_clamped in
    let neg_log_u = mul log_u neg_one in
    let log_neg_log_u = log neg_log_u in
    let gumbel = mul log_neg_log_u neg_one |> astype dtype in
    let noisy = add logits gumbel in
    let prefix_len = Array.length shape in
    let argmax_axis = axis + prefix_len in
    let inds = argmax noisy ~axis:argmax_axis ~keepdims:false in
    astype Tensor.Int32 inds
  in

  match dtype with
  | Tensor.Float64 -> run_float64 1e-12
  | Tensor.Float32 -> run_float32 1e-6
  | Tensor.Float16 -> run_float32 1e-3
  | Tensor.BFloat16 -> run_float32 1e-2
  | Tensor.Float8_e4m3 | Tensor.Float8_e5m2 ->
      invalid_arg "categorical: float8 logits not supported"
  | _ -> invalid_arg "categorical: logits dtype must be floating point"

let truncated_normal (type a b) key (dtype : (a, b) Nx_core.Dtype.t) ~lower
    ~upper shape =
  if lower >= upper then
    invalid_arg
      (Printf.sprintf "truncated_normal: lower must be less than upper");

  let supported =
    match dtype with
    | Tensor.Float16 | Tensor.Float32 | Tensor.Float64 | Tensor.BFloat16 -> true
    | _ -> false
  in
  if not supported then
    invalid_arg "truncated_normal: dtype must be a floating point type";

  let scalar_lower = scalar dtype lower in
  let scalar_upper = scalar dtype upper in

  let split2 key =
    match split ~n:2 key with [| a; b |] -> (a, b) | _ -> assert false
  in

  let has_remaining mask =
    let any_mask = Tensor.any mask in
    let arr = to_array any_mask in
    match arr with [| v |] -> v | _ -> false
  in

  let max_attempts = 1000 in

  let sample_key, next_key = split2 key in
  let initial = normal sample_key dtype shape in
  let within_lower = greater_equal initial scalar_lower in
  let within_upper = less_equal initial scalar_upper in
  let accepted = logical_and within_lower within_upper in
  let remaining = logical_not accepted in

  let rec fill key acc remaining attempt =
    if not (has_remaining remaining) then acc
    else if attempt > max_attempts then
      invalid_arg
        (Printf.sprintf
           "truncated_normal: failed to generate samples within bounds after \
            %d attempts"
           max_attempts)
    else
      let resample_key, next_key = split2 key in
      let candidates = normal resample_key dtype shape in
      let within_lower = greater_equal candidates scalar_lower in
      let within_upper = less_equal candidates scalar_upper in
      let within = logical_and within_lower within_upper in
      let take_new = logical_and remaining within in
      let acc = where take_new candidates acc in
      let still_remaining = logical_and remaining (logical_not within) in
      fill next_key acc still_remaining (attempt + 1)
  in
  fill next_key initial remaining 1
