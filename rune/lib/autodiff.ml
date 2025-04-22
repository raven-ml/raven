open Internal
open Tensor

type arg = L | R

let deriv_neg x =
  let ones = Dispatch.ones_like x.data in
  neg (const ones)

let deriv_sin x = cos x
let deriv_cos x = neg (sin x)
let deriv_exp x = exp x
let deriv_log x = div (const (Dispatch.ones_like x.data)) x

let deriv_add _ x _ =
  let ones = Dispatch.ones_like x.data in
  const ones

let deriv_sub arg x _ =
  let ones = Dispatch.ones_like x.data in
  match arg with L -> const ones | R -> neg (const ones)

let deriv_mul arg x y = match arg with L -> y | R -> x

let deriv_div arg x y =
  let ones = Dispatch.ones_like x.data in
  match arg with L -> div (const ones) y | R -> neg (div x (mul y y))

module Set_int = Set.Make (struct
  type t = int

  let compare = compare
end)

let compute_new_shape input_shape reduced_axes =
  let ndim = Array.length input_shape in
  let reduced_set =
    Array.fold_left
      (fun set ax -> Set_int.add ax set)
      Set_int.empty reduced_axes
  in
  Array.init ndim (fun i ->
      if Set_int.mem i reduced_set then 1 else input_shape.(i))

let deriv_sum x axes =
  let input_shape = Dispatch.shape x.data in
  fun twg_bv ->
    if axes = None then const (Dispatch.broadcast_to input_shape twg_bv.data)
    else
      let reduced_axes = Option.get axes in
      let new_shape = compute_new_shape input_shape reduced_axes in
      let reshaped_grad = Dispatch.reshape new_shape twg_bv.data in
      const (Dispatch.broadcast_to input_shape reshaped_grad)

let deriv_mean x axes =
  let input_shape = Dispatch.shape x.data in
  fun twg_bv ->
    let grad =
      if axes = None then Dispatch.broadcast_to input_shape twg_bv.data
      else
        let reduced_axes = Option.get axes in
        let new_shape = compute_new_shape input_shape reduced_axes in
        let reshaped_grad = Dispatch.reshape new_shape twg_bv.data in
        Dispatch.broadcast_to input_shape reshaped_grad
    in
    if axes = None then
      let n = Array.fold_left ( * ) 1 input_shape in
      const (Dispatch.div_scalar grad (float_of_int n))
    else
      let reduced_axes = Option.get axes in
      let reduced_sizes = Array.map (fun ax -> input_shape.(ax)) reduced_axes in
      let n = Array.fold_left ( * ) 1 reduced_sizes in
      const (Dispatch.div_scalar grad (float_of_int n))

let deriv_max x axes =
  let input_shape = Dispatch.shape x.data in
  let reduced_axes =
    match axes with
    | None -> Array.init (Array.length input_shape) (fun i -> i)
    | Some ax -> ax
  in
  let max_x = Dispatch.max x.data ~axes:reduced_axes in
  let new_shape =
    Array.mapi
      (fun i dim -> if Array.mem i reduced_axes then 1 else dim)
      input_shape
  in
  let reshaped_max = Dispatch.reshape new_shape max_x in
  let broadcasted_max = Dispatch.broadcast_to input_shape reshaped_max in
  let mask = Dispatch.equal x.data broadcasted_max in
  let mask_cast = Dispatch.astype (Dispatch.dtype x.data) mask in
  fun twg_bv ->
    let reshaped_grad = Dispatch.reshape new_shape twg_bv.data in
    let broadcasted_grad = Dispatch.broadcast_to input_shape reshaped_grad in
    let count = Dispatch.sum mask_cast ~axes:reduced_axes in
    let reshaped_count = Dispatch.reshape new_shape count in
    let broadcasted_count = Dispatch.broadcast_to input_shape reshaped_count in
    let grad_x =
      Dispatch.mul (Dispatch.div broadcasted_grad broadcasted_count) mask_cast
    in
    const grad_x

let deriv_min x axes =
  let input_shape = Dispatch.shape x.data in
  let reduced_axes =
    match axes with
    | None -> Array.init (Array.length input_shape) (fun i -> i)
    | Some ax -> ax
  in
  let min_x = Dispatch.min x.data ~axes:reduced_axes in
  let new_shape =
    Array.mapi
      (fun i dim -> if Array.mem i reduced_axes then 1 else dim)
      input_shape
  in
  let reshaped_min = Dispatch.reshape new_shape min_x in
  let broadcasted_min = Dispatch.broadcast_to input_shape reshaped_min in
  let mask = Dispatch.equal x.data broadcasted_min in
  let mask_cast = Dispatch.astype (Dispatch.dtype x.data) mask in
  fun twg_bv ->
    let reshaped_grad = Dispatch.reshape new_shape twg_bv.data in
    let broadcasted_grad = Dispatch.broadcast_to input_shape reshaped_grad in
    let count = Dispatch.sum mask_cast ~axes:reduced_axes in
    let reshaped_count = Dispatch.reshape new_shape count in
    let broadcasted_count = Dispatch.broadcast_to input_shape reshaped_count in
    let grad_x =
      Dispatch.mul (Dispatch.div broadcasted_grad broadcasted_count) mask_cast
    in
    const grad_x

let deriv_maximum arg x y =
  let t_ones = const (Dispatch.ones_like x.data) in
  let t_zeros = const (Dispatch.zeros_like x.data) in
  match arg with
  | L ->
      const
        (Dispatch.where
           (Dispatch.greater_equal x.data y.data)
           t_ones.data t_zeros.data)
  | R ->
      const
        (Dispatch.where
           (Dispatch.greater y.data x.data)
           t_ones.data t_zeros.data)

let deriv_minimum arg x y =
  let t_ones = const (Dispatch.ones_like x.data) in
  let t_zeros = const (Dispatch.zeros_like x.data) in
  match arg with
  | L ->
      const
        (Dispatch.where
           (Dispatch.less_equal x.data y.data)
           t_ones.data t_zeros.data)
  | R ->
      const
        (Dispatch.where (Dispatch.less y.data x.data) t_ones.data t_zeros.data)

type ('a, 'b, 'dev) t_with_grad = {
  v : ('a, 'b, 'dev) t;
  mutable bv : ('a, 'b, 'dev) t;
}

type any_t_with_grad =
  | Any_t_with_grad : ('a, 'b, 'dev) t_with_grad -> any_t_with_grad

let unwrap_t_with_grad (type a b dev) (_ : (a, b, dev) t)
    (any : any_t_with_grad) : (a, b, dev) t_with_grad =
  match any with Any_t_with_grad m -> Obj.magic m

let reduce_gradient grad_output input_shape output_shape =
  let ndim_output = Array.length output_shape in
  let ndim_input = Array.length input_shape in
  let padded_input_shape =
    Array.append (Array.make (ndim_output - ndim_input) 1) input_shape
  in
  let axes_to_sum =
    List.filter
      (fun i -> padded_input_shape.(i) = 1 && output_shape.(i) > 1)
      (List.init ndim_output (fun i -> i))
  in
  if axes_to_sum = [] then grad_output
  else
    let axes_array = Array.of_list axes_to_sum in
    const (Dispatch.sum grad_output.data ~axes:axes_array)

let make_reverse_handler tape =
  let open Effect.Deep in
  let effc : type a. a Effect.t -> ((a, _) continuation -> _) option =
    let handle_ap0 n k =
      let tensor = const n in
      let zeros = Dispatch.zeros_like tensor.data in
      let t_with_grad = { v = tensor; bv = const zeros } in
      Hashtbl.add tape tensor.id (Any_t_with_grad t_with_grad);
      continue k tensor
    in

    let handle_ap1 ~deriv ~op x k =
      let r = op x in
      let zeros = Dispatch.zeros_like r.data in
      let twg = { v = r; bv = const zeros } in
      Hashtbl.add tape twg.v.id (Any_t_with_grad twg);
      (if not (Hashtbl.mem tape x.id) then
         let zeros_x = Dispatch.zeros_like x.data in
         let twg_x = { v = x; bv = const zeros_x } in
         Hashtbl.add tape x.id (Any_t_with_grad twg_x));
      let any_twg_x = Hashtbl.find tape x.id in
      let twg_x = unwrap_t_with_grad x any_twg_x in
      let t = continue k r in
      twg_x.bv <- add twg_x.bv (mul (deriv twg_x.v) twg.bv);
      t
    in

    let handle_ap2 ~deriv ~op x y k =
      let r = op x y in
      let zeros = Dispatch.zeros_like r.data in
      let twg = { v = r; bv = const zeros } in
      Hashtbl.add tape twg.v.id (Any_t_with_grad twg);
      (if not (Hashtbl.mem tape x.id) then
         let zeros_x = Dispatch.zeros_like x.data in
         let twg_x = { v = x; bv = const zeros_x } in
         Hashtbl.add tape x.id (Any_t_with_grad twg_x));
      let any_twg_x = Hashtbl.find tape x.id in
      let twg_x = unwrap_t_with_grad x any_twg_x in
      (if not (Hashtbl.mem tape y.id) then
         let zeros_y = Dispatch.zeros_like y.data in
         let twg_y = { v = y; bv = const zeros_y } in
         Hashtbl.add tape y.id (Any_t_with_grad twg_y));
      let any_twg_y = Hashtbl.find tape y.id in
      let twg_y = unwrap_t_with_grad y any_twg_y in
      let t = continue k r in
      let grad_x = mul (deriv L x y) twg.bv in
      let grad_y = mul (deriv R x y) twg.bv in
      let reduced_grad_x =
        reduce_gradient grad_x (Dispatch.shape x.data) (Dispatch.shape r.data)
      in
      let reduced_grad_y =
        reduce_gradient grad_y (Dispatch.shape y.data) (Dispatch.shape r.data)
      in
      twg_x.bv <- add twg_x.bv reduced_grad_x;
      twg_y.bv <- add twg_y.bv reduced_grad_y;
      t
    in

    function
    | Const v -> Some (handle_ap0 v)
    | Neg x -> Some (handle_ap1 ~deriv:deriv_neg ~op:neg x)
    | Exp x -> Some (handle_ap1 ~deriv:deriv_exp ~op:exp x)
    | Log x -> Some (handle_ap1 ~deriv:deriv_log ~op:log x)
    | Sin x -> Some (handle_ap1 ~deriv:deriv_sin ~op:sin x)
    | Cos x -> Some (handle_ap1 ~deriv:deriv_cos ~op:cos x)
    | Add (x, y) -> Some (handle_ap2 ~deriv:deriv_add ~op:add x y)
    | Sub (x, y) -> Some (handle_ap2 ~deriv:deriv_sub ~op:sub x y)
    | Mul (x, y) -> Some (handle_ap2 ~deriv:deriv_mul ~op:mul x y)
    | Div (x, y) -> Some (handle_ap2 ~deriv:deriv_div ~op:div x y)
    | Maximum (x, y) -> Some (handle_ap2 ~deriv:deriv_maximum ~op:maximum x y)
    | Minimum (x, y) -> Some (handle_ap2 ~deriv:deriv_minimum ~op:minimum x y)
    | Sum (x, axes, _keepdims) ->
        Some
          (fun k ->
            let r = sum ?axes x in
            let zeros = Dispatch.zeros_like r.data in
            let twg = { v = r; bv = const zeros } in
            Hashtbl.add tape twg.v.id (Any_t_with_grad twg);
            (if not (Hashtbl.mem tape x.id) then
               let zeros_x = Dispatch.zeros_like x.data in
               let twg_x = { v = x; bv = const zeros_x } in
               Hashtbl.add tape x.id (Any_t_with_grad twg_x));
            let any_twg_x = Hashtbl.find tape x.id in
            let twg_x = unwrap_t_with_grad x any_twg_x in
            let t = continue k r in
            let deriv_fn = deriv_sum x axes in
            twg_x.bv <- add twg_x.bv (deriv_fn twg.bv);
            t)
    | Mean (x, axes, _keepdims) ->
        Some
          (fun k ->
            let r = mean ?axes x in
            let zeros = Dispatch.zeros_like r.data in
            let twg = { v = r; bv = const zeros } in
            Hashtbl.add tape twg.v.id (Any_t_with_grad twg);
            (if not (Hashtbl.mem tape x.id) then
               let zeros_x = Dispatch.zeros_like x.data in
               let twg_x = { v = x; bv = const zeros_x } in
               Hashtbl.add tape x.id (Any_t_with_grad twg_x));
            let any_twg_x = Hashtbl.find tape x.id in
            let twg_x = unwrap_t_with_grad x any_twg_x in
            let t = continue k r in
            let deriv_fn = deriv_mean x axes in
            twg_x.bv <- add twg_x.bv (deriv_fn twg.bv);
            t)
    | Max (x, axes, _keepdims) ->
        Some
          (fun k ->
            let r = max ?axes x in
            let zeros = Dispatch.zeros_like r.data in
            let twg = { v = r; bv = const zeros } in
            Hashtbl.add tape twg.v.id (Any_t_with_grad twg);
            (if not (Hashtbl.mem tape x.id) then
               let zeros_x = Dispatch.zeros_like x.data in
               let twg_x = { v = x; bv = const zeros_x } in
               Hashtbl.add tape x.id (Any_t_with_grad twg_x));
            let any_twg_x = Hashtbl.find tape x.id in
            let twg_x = unwrap_t_with_grad x any_twg_x in
            let t = continue k r in
            let deriv_fn = deriv_max x axes in
            twg_x.bv <- add twg_x.bv (deriv_fn twg.bv);
            t)
    | Min (x, axes, _keepdims) ->
        Some
          (fun k ->
            let r = min ?axes x in
            let zeros = Dispatch.zeros_like r.data in
            let twg = { v = r; bv = const zeros } in
            Hashtbl.add tape twg.v.id (Any_t_with_grad twg);
            (if not (Hashtbl.mem tape x.id) then
               let zeros_x = Dispatch.zeros_like x.data in
               let twg_x = { v = x; bv = const zeros_x } in
               Hashtbl.add tape x.id (Any_t_with_grad twg_x));
            let any_twg_x = Hashtbl.find tape x.id in
            let twg_x = unwrap_t_with_grad x any_twg_x in
            let t = continue k r in
            let deriv_fn = deriv_min x axes in
            twg_x.bv <- add twg_x.bv (deriv_fn twg.bv);
            t)
    | Matmul (x, y) ->
        Some
          (fun k ->
            let r = matmul x y in
            let zeros = Dispatch.zeros_like r.data in
            let twg = { v = r; bv = const zeros } in
            Hashtbl.add tape twg.v.id (Any_t_with_grad twg);
            (if not (Hashtbl.mem tape x.id) then
               let zeros_x = Dispatch.zeros_like x.data in
               let twg_x = { v = x; bv = const zeros_x } in
               Hashtbl.add tape x.id (Any_t_with_grad twg_x));
            let any_twg_x = Hashtbl.find tape x.id in
            let twg_x = unwrap_t_with_grad x any_twg_x in
            (if not (Hashtbl.mem tape y.id) then
               let zeros_y = Dispatch.zeros_like y.data in
               let twg_y = { v = y; bv = const zeros_y } in
               Hashtbl.add tape y.id (Any_t_with_grad twg_y));
            let any_twg_y = Hashtbl.find tape y.id in
            let twg_y = unwrap_t_with_grad y any_twg_y in
            let t = continue k r in
            twg_x.bv <- add twg_x.bv (matmul twg.bv (transpose twg_y.v));
            twg_y.bv <- add twg_y.bv (matmul (transpose twg_x.v) twg.bv);
            t)
    | Transpose x ->
        Some
          (fun k ->
            let r = transpose x in
            let zeros = Dispatch.zeros_like r.data in
            let twg = { v = r; bv = const zeros } in
            Hashtbl.add tape twg.v.id (Any_t_with_grad twg);
            (if not (Hashtbl.mem tape x.id) then
               let zeros_x = Dispatch.zeros_like x.data in
               let twg_x = { v = x; bv = const zeros_x } in
               Hashtbl.add tape x.id (Any_t_with_grad twg_x));
            let any_twg_x = Hashtbl.find tape x.id in
            let twg_x = unwrap_t_with_grad x any_twg_x in
            let t = continue k r in
            twg_x.bv <- add twg_x.bv (transpose twg.bv);
            t)
    | Reshape (x, shape) ->
        Some
          (fun k ->
            let original_shape = Dispatch.shape x.data in
            let r = reshape shape x in
            let zeros = Dispatch.zeros_like r.data in
            let twg = { v = r; bv = const zeros } in
            Hashtbl.add tape twg.v.id (Any_t_with_grad twg);
            let any_twg_x = Hashtbl.find tape x.id in
            let twg_x = unwrap_t_with_grad x any_twg_x in
            let t = continue k r in
            twg_x.bv <- add twg_x.bv (reshape original_shape twg.bv);
            t)
    | Slice (x, starts, stops, steps) ->
        Some
          (fun k ->
            let r_data = Dispatch.slice ?steps starts stops x.data in
            let r = create_internal r_data in
            let zeros = Dispatch.zeros_like r_data in
            let twg = { v = r; bv = const zeros } in
            Hashtbl.add tape twg.v.id (Any_t_with_grad twg);

            (if not (Hashtbl.mem tape x.id) then
               let zeros_x = Dispatch.zeros_like x.data in
               let twg_x = { v = x; bv = const zeros_x } in
               Hashtbl.add tape x.id (Any_t_with_grad twg_x));
            let any_twg_x = Hashtbl.find tape x.id in
            let twg_x = unwrap_t_with_grad x any_twg_x in

            let t = continue k r in

            let grad_x_data = Dispatch.zeros_like x.data in
            Dispatch.set_slice ?steps starts stops twg.bv.data grad_x_data;
            let grad_x = const grad_x_data in
            twg_x.bv <- add twg_x.bv grad_x;
            t)
    | Cast (dtype, x) ->
        Some
          (fun k ->
            let r = astype dtype x in
            let zeros = Dispatch.zeros_like r.data in
            let twg = { v = r; bv = const zeros } in
            Hashtbl.add tape r.id (Any_t_with_grad twg);
            (if not (Hashtbl.mem tape x.id) then
               let zeros_x = Dispatch.zeros_like x.data in
               let twg_x = { v = x; bv = const zeros_x } in
               Hashtbl.add tape x.id (Any_t_with_grad twg_x));
            let t = continue k r in
            let any_twg_x = Hashtbl.find tape x.id in
            let twg_x = unwrap_t_with_grad x any_twg_x in
            let dtype_x = Dispatch.dtype x.data in
            let casted_grad = astype dtype_x twg.bv in
            twg_x.bv <- add twg_x.bv casted_grad;
            t)
    | Move (device, x) ->
        Some
          (fun k ->
            let r = move device x in
            let zeros = Dispatch.zeros_like r.data in
            let twg = { v = r; bv = const zeros } in
            Hashtbl.add tape r.id (Any_t_with_grad twg);
            (if not (Hashtbl.mem tape x.id) then
               let zeros_x = Dispatch.zeros_like x.data in
               let twg_x = { v = x; bv = const zeros_x } in
               Hashtbl.add tape x.id (Any_t_with_grad twg_x));
            let t = continue k r in
            let any_twg_x = Hashtbl.find tape x.id in
            let twg_x = unwrap_t_with_grad x any_twg_x in
            let dev_from = Tensor.device x in
            let moved_grad = move dev_from twg.bv in
            twg_x.bv <- add twg_x.bv moved_grad;
            t)
    | _ -> None
  in
  {
    retc =
      (fun x ->
        match Hashtbl.find_opt tape x.id with
        | Some any_m ->
            let m = unwrap_t_with_grad x any_m in
            let ones = Dispatch.ones_like x.data in
            m.bv <- const ones;
            x
        | None ->
            Printf.eprintf "Result tensor not found in tape\n";
            x);
    exnc = raise;
    effc;
  }

let grad f input_tensor =
  let tape = Hashtbl.create 10 in
  let zeros = Dispatch.zeros_like input_tensor.data in
  let m_input = { v = input_tensor; bv = const zeros } in
  Hashtbl.add tape input_tensor.id (Any_t_with_grad m_input);
  let handler = make_reverse_handler tape in
  let _ = Effect.Deep.match_with f input_tensor handler in
  let any_final_m_input = Hashtbl.find tape input_tensor.id in
  let final_m_input = unwrap_t_with_grad input_tensor any_final_m_input in
  final_m_input.bv

let grads f input_tensors =
  let tape = Hashtbl.create 10 in
  let input_twgs =
    List.map
      (fun input_tensor ->
        let zeros = Dispatch.zeros_like input_tensor.data in
        let twg = { v = input_tensor; bv = const zeros } in
        Hashtbl.add tape input_tensor.id (Any_t_with_grad twg);
        twg)
      input_tensors
  in
  let handler = make_reverse_handler tape in
  let _ = Effect.Deep.match_with f input_tensors handler in
  List.map (fun twg -> twg.bv) input_twgs

let value_and_grad f input_tensor =
  let tape = Hashtbl.create 10 in
  let zeros = Dispatch.zeros_like input_tensor.data in
  let m_input = { v = input_tensor; bv = const zeros } in
  Hashtbl.add tape input_tensor.id (Any_t_with_grad m_input);
  let handler = make_reverse_handler tape in
  let result_tensor = Effect.Deep.match_with f input_tensor handler in
  let any_final_m_input = Hashtbl.find tape input_tensor.id in
  let final_m_input = unwrap_t_with_grad input_tensor any_final_m_input in
  (result_tensor, final_m_input.bv)

let value_and_grads f input_tensors =
  let tape = Hashtbl.create 10 in
  let input_twgs =
    List.map
      (fun input_tensor ->
        let zeros = Dispatch.zeros_like input_tensor.data in
        let twg = { v = input_tensor; bv = const zeros } in
        Hashtbl.add tape input_tensor.id (Any_t_with_grad twg);
        twg)
      input_tensors
  in
  let handler = make_reverse_handler tape in
  let result_tensor = Effect.Deep.match_with f input_tensors handler in
  let grads = List.map (fun twg -> twg.bv) input_twgs in
  (result_tensor, grads)
