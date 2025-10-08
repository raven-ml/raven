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
    (* Create shuffled tensor by indexing *)
    let perm_array = to_array perm |> Array.map Int32.to_int in
    let results = Array.map (fun i -> get [ i ] x) perm_array in
    concatenate ~axis:0 (Array.to_list results)

let categorical key ?(axis = -1) logits =
  let shape_array = Tensor.shape logits in
  let ndim = Array.length shape_array in
  let axis = if axis < 0 then ndim + axis else axis in
  
  (* Convert logits to float32 for softmax and cumsum operations *)
  let logits_float = astype Tensor.Float32 logits in
  
  (* Generate uniform random values with same shape *)
  let uniform_vals = uniform key Tensor.Float32 shape_array in
  
  (* Apply softmax to get probabilities *)
  let probs = softmax logits_float ~axes:[axis] in
  
  (* Compute cumulative sum along axis *)
  let cumsum = cumsum probs ~axis in
  
  (* Find where uniform_vals < cumsum for the first time *)
  let comparison = cmplt uniform_vals cumsum in
  
  (* argmax along axis gives us the first True index *)
  argmax comparison ~axis ~keepdims:false

let truncated_normal key dtype ~lower ~upper shape =
  if lower >= upper then
    invalid_arg
      (Printf.sprintf "truncated_normal: lower must be less than upper");

  (* Simple clipping approach for now *)
  let vals = normal key dtype shape in

  (* Clip values to bounds *)
  clip vals ~min:lower ~max:upper
