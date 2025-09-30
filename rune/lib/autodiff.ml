open Nx_core
open Nx_rune
module T = Tensor

(* Custom hashtable module that uses physical equality to distinguish tensors *)
module PhysicalTbl = struct
  type ('a, 'b) t = (Obj.t * 'b) list ref

  let create _ = ref []

  let find_opt tbl key =
    let key_repr = Obj.repr key in
    List.find_opt (fun (k, _) -> k == key_repr) !tbl |> Option.map snd

  let add tbl key value =
    let key_repr = Obj.repr key in
    tbl := (key_repr, value) :: !tbl

  let find tbl key =
    match find_opt tbl key with Some v -> v | None -> raise Not_found
end

(* Global ID generator for t_with_grad instances *)
let next_twg_id_counter = ref 0

let fresh_twg_id () =
  incr next_twg_id_counter;
  !next_twg_id_counter

(* Type to store a tensor's forward value and its accumulated gradient *)
type ('a, 'b) t_with_grad = {
  v : ('a, 'b) t;
  mutable bv : ('a, 'b) t;
  id : int;
}

type any_t_with_grad =
  | Any_t_with_grad : ('a, 'b) t_with_grad -> any_t_with_grad

let value_of twg = twg.v
let grad_of twg = twg.bv

let unwrap_twg (type a b) (_dtype : (a, b) Dtype.t) (any : any_t_with_grad) :
    (a, b) t_with_grad =
  match any with Any_t_with_grad m -> Obj.magic m

(* Forward mode AD: dual numbers storing value and tangent *)
type ('a, 'b) dual = { primal : ('a, 'b) t; tangent : ('a, 'b) t }
type any_dual = Any_dual : ('a, 'b) dual -> any_dual

let primal_of dual = dual.primal
let tangent_of dual = dual.tangent

let unwrap_dual (type a b) (_dtype : (a, b) Dtype.t) (any : any_dual) :
    (a, b) dual =
  match any with Any_dual d -> Obj.magic d

(* --- Derivative definitions for UOps --- *)

let ln2 = 0.693147180559945309417
let deriv_neg x = T.neg (T.ones_like x)

let deriv_log2 (type a b) (x : (a, b) T.t) : (a, b) T.t =
  (* d/dx log2(x) = 1 / (x * ln(2)) where ln(2) ≈ 0.6931 *)
  match T.dtype x with
  | Float16 ->
      let ln2_tensor = T.full (context x) (T.dtype x) (T.shape x) ln2 in
      T.div (T.ones_like x) (T.mul x ln2_tensor)
  | Float32 ->
      let ln2_tensor = T.full (context x) (T.dtype x) (T.shape x) ln2 in
      T.div (T.ones_like x) (T.mul x ln2_tensor)
  | Float64 ->
      let ln2_tensor = T.full (context x) (T.dtype x) (T.shape x) ln2 in
      T.div (T.ones_like x) (T.mul x ln2_tensor)
  | _ -> failwith "deriv_log2: unsupported dtype"

let deriv_exp2 (type a b) (exp2_x : (a, b) T.t) (_x : (a, b) T.t) : (a, b) T.t =
  match T.dtype exp2_x with
  | Float16 ->
      let ln2_tensor =
        T.full (context exp2_x) (T.dtype exp2_x) (T.shape exp2_x) ln2
      in
      T.mul exp2_x ln2_tensor
  | Float32 ->
      let ln2_tensor =
        T.full (context exp2_x) (T.dtype exp2_x) (T.shape exp2_x) ln2
      in
      T.mul exp2_x ln2_tensor
  | Float64 ->
      let ln2_tensor =
        T.full (context exp2_x) (T.dtype exp2_x) (T.shape exp2_x) ln2
      in
      T.mul exp2_x ln2_tensor
  | _ -> failwith "deriv_exp2: unsupported dtype"

let deriv_sin (type a b) (x : (a, b) T.t) : (a, b) T.t =
  match T.dtype x with
  | Float16 ->
      let cos_x = T.cos x in
      T.cast (T.dtype x) cos_x
  | Float32 ->
      let cos_x = T.cos x in
      T.cast (T.dtype x) cos_x
  | Float64 ->
      let cos_x = T.cos x in
      T.cast (T.dtype x) cos_x
  | _ -> failwith "deriv_sin: unsupported dtype"

let deriv_sqrt sqrt_x _x =
  let one = T.ones_like sqrt_x in
  let two = T.add one one in
  T.div one (T.mul two sqrt_x)

let deriv_recip x =
  let x_squared = T.mul x x in
  T.neg (T.recip x_squared)

let deriv_fdiv_wrt_op1 _op1 op2 = T.recip op2

let deriv_fdiv_wrt_op2 op1 op2 =
  let op2_sq = T.mul op2 op2 in
  T.div (T.neg op1) op2_sq

let deriv_pow_wrt_op1 op1 op2 =
  let exp_minus_1 = T.sub op2 (T.ones_like op2) in
  let op1_pow_exp_minus_1 = T.pow op1 exp_minus_1 in
  T.mul op2 op1_pow_exp_minus_1

let log_e_float x =
  let ctx = context x in
  let log2_x = T.log2 x in
  let log_2 = T.full ctx (dtype x) (T.shape x) ln2 in
  T.mul log2_x log_2

let deriv_pow_wrt_op2_float result_val op1 =
  let log_op1 = log_e_float op1 in
  T.mul result_val log_op1

let deriv_max_wrt_op1 op1 op2 op1_dtype = T.cast op1_dtype (T.greater op1 op2)

let deriv_max_wrt_op2 op1 op2 op2_dtype =
  T.cast op2_dtype (T.greater_equal op2 op1)

let normalize_axis axis shape =
  let rank = Array.length shape in
  let axis = if axis < 0 then axis + rank else axis in
  if axis < 0 || axis >= rank then
    invalid_arg (Printf.sprintf "axis %d out of bounds for rank %d" axis rank)
  else axis

let reverse_cumsum tensor axis =
  let flipped = T.flip tensor ~axes:[| axis |] in
  let scanned = T.cumsum ~axis flipped in
  T.flip scanned ~axes:[| axis |]

let prefix_exclusive axis tensor =
  let shape = T.shape tensor in
  let pad_config =
    Array.mapi (fun i _ -> if i = axis then (1, 0) else (0, 0)) shape
  in
  let one = Dtype.one (T.dtype tensor) in
  let padded = T.pad pad_config one tensor in
  let cumprod_padded = T.cumprod ~axis padded in
  let slice_specs =
    Array.mapi
      (fun i dim -> if i = axis then T.R (0, dim) else T.R (0, dim))
      shape
  in
  T.slice (Array.to_list slice_specs) cumprod_padded

let suffix_exclusive axis tensor =
  let shape = T.shape tensor in
  let one = Dtype.one (T.dtype tensor) in
  let flipped = T.flip tensor ~axes:[| axis |] in
  let flipped_cumprod = T.cumprod ~axis flipped in
  let suffix_inclusive = T.flip flipped_cumprod ~axes:[| axis |] in
  let pad_config =
    Array.mapi (fun i _ -> if i = axis then (0, 1) else (0, 0)) shape
  in
  let padded = T.pad pad_config one suffix_inclusive in
  let slice_specs =
    Array.mapi
      (fun i dim -> if i = axis then T.R (1, dim + 1) else T.R (0, dim))
      shape
  in
  T.slice (Array.to_list slice_specs) padded

let divide_no_nan num denom =
  let zero_scalar = Dtype.zero (T.dtype denom) in
  let zero_tensor =
    T.full (context denom) (T.dtype denom) (T.shape denom) zero_scalar
  in
  let zero_mask = T.equal denom zero_tensor in
  let safe_denom = T.where zero_mask (T.ones_like denom) denom in
  let base = T.div num safe_denom in
  T.where zero_mask (T.zeros_like base) base

let prepare_grad_for_broadcast grad_output input_tensor_val axes op_keepdims
    reduction_op_for_shape =
  if op_keepdims then grad_output
  else
    let dummy_input_like = T.zeros_like input_tensor_val in
    let reduced_shape_with_kept_dims =
      T.shape (reduction_op_for_shape dummy_input_like ~axes ~keepdims:true)
    in
    T.reshape reduced_shape_with_kept_dims grad_output

(* Helper functions to reduce boilerplate *)
let handle_identity_gradient_op ~op_name ~op get_or_init_twg t_in_val k_continue
    =
  let result_val = op t_in_val in
  let forward_val = Effect.Deep.continue k_continue result_val in
  Debug.with_context ("∇" ^ op_name) (fun () ->
      let twg_in = get_or_init_twg t_in_val in
      let twg_res = get_or_init_twg result_val in
      let d_loss_d_result = grad_of twg_res in
      twg_in.bv <- T.add twg_in.bv d_loss_d_result);
  forward_val

let handle_unary_op ~op_name ~op ~deriv get_or_init_twg t_in_val k_continue =
  let result_val = op t_in_val in
  let forward_val = Effect.Deep.continue k_continue result_val in
  Debug.with_context ("∇" ^ op_name) (fun () ->
      let twg_in = get_or_init_twg t_in_val in
      let twg_res = get_or_init_twg result_val in
      let d_loss_d_result = grad_of twg_res in
      let grad_contrib = T.mul d_loss_d_result (deriv (value_of twg_in)) in
      twg_in.bv <- T.add twg_in.bv grad_contrib);
  forward_val

let handle_binary_op ~op_name ~op ~deriv_wrt_op1 ~deriv_wrt_op2 get_or_init_twg
    op1_val op2_val k_continue =
  let result_val = op op1_val op2_val in
  let forward_val = Effect.Deep.continue k_continue result_val in
  Debug.with_context ("∇" ^ op_name) (fun () ->
      let twg_op1 = get_or_init_twg op1_val in
      let twg_op2 = get_or_init_twg op2_val in
      let twg_res = get_or_init_twg result_val in
      let d_loss_d_result = grad_of twg_res in

      let grad_op1 =
        T.mul d_loss_d_result
          (deriv_wrt_op1 (value_of twg_op1) (value_of twg_op2))
      in
      twg_op1.bv <- T.add twg_op1.bv grad_op1;

      let grad_op2 =
        T.mul d_loss_d_result
          (deriv_wrt_op2 (value_of twg_op1) (value_of twg_op2))
      in
      twg_op2.bv <- T.add twg_op2.bv grad_op2);
  forward_val

(* The main reverse-mode AD effect handler *)
let make_reverse_handler tape_by_twg_id val_to_twg_id_map =
  let open Effect.Deep in
  let get_or_init_twg tensor_val =
    match PhysicalTbl.find_opt val_to_twg_id_map tensor_val with
    | Some twg_id -> (
        match Hashtbl.find_opt tape_by_twg_id twg_id with
        | Some any_twg -> unwrap_twg (dtype tensor_val) any_twg
        | None -> failwith "Rune.Autodiff inconsistency")
    | None ->
        let zero_grad = T.zeros_like tensor_val in
        let new_id = fresh_twg_id () in
        let new_twg = { v = tensor_val; bv = zero_grad; id = new_id } in
        Hashtbl.add tape_by_twg_id new_id (Any_t_with_grad new_twg);
        PhysicalTbl.add val_to_twg_id_map tensor_val new_id;
        new_twg
  in

  let effc : type a. a Effect.t -> ((a, _) continuation -> _) option = function
    | E_buffer { context = effect_ctx; dtype = dt; size_in_elements } ->
        Some
          (fun k ->
            let result_val = op_buffer effect_ctx dt size_in_elements in
            let forward_val = continue k result_val in
            Debug.with_context "∇buffer" (fun () ->
                let _twg_res = get_or_init_twg result_val in
                ());
            forward_val)
    | E_const_scalar { context = effect_ctx; value; dtype = dt } ->
        Some
          (fun k ->
            let result_val = op_const_scalar effect_ctx value dt in
            let forward_val = continue k result_val in
            Debug.with_context "∇const_scalar" (fun () ->
                let _twg_res = get_or_init_twg result_val in
                ());
            forward_val)
    | E_add { a = op1_val; b = op2_val } ->
        Some
          (fun k ->
            let result_val = op_add op1_val op2_val in
            let forward_val = continue k result_val in
            Debug.with_context "∇add" (fun () ->
                let twg_op1 = get_or_init_twg op1_val in
                let twg_op2 = get_or_init_twg op2_val in
                let twg_res = get_or_init_twg result_val in
                let d_loss_d_result = grad_of twg_res in
                twg_op1.bv <- T.add twg_op1.bv d_loss_d_result;
                twg_op2.bv <- T.add twg_op2.bv d_loss_d_result);
            forward_val)
    | E_mul { a = op1_val; b = op2_val } ->
        Some
          (handle_binary_op ~op_name:"mul" ~op:op_mul
             ~deriv_wrt_op1:(fun _ op2 -> op2)
             ~deriv_wrt_op2:(fun op1 _ -> op1)
             get_or_init_twg op1_val op2_val)
    | E_neg { t_in } ->
        Some
          (handle_unary_op ~op_name:"neg" ~op:op_neg ~deriv:deriv_neg
             get_or_init_twg t_in)
    | E_log2 { t_in } ->
        Some
          (handle_unary_op ~op_name:"log2" ~op:op_log2 ~deriv:deriv_log2
             get_or_init_twg t_in)
    | E_exp2 { t_in = t_in_val } ->
        Some
          (fun k ->
            let result_val = op_exp2 t_in_val in
            let forward_val = continue k result_val in
            Debug.with_context "∇exp2" (fun () ->
                let twg_in = get_or_init_twg t_in_val in
                let twg_res = get_or_init_twg result_val in
                let d_loss_d_result = grad_of twg_res in
                let d_result_d_input =
                  deriv_exp2 result_val (value_of twg_in)
                in
                let grad_contrib = T.mul d_loss_d_result d_result_d_input in
                twg_in.bv <- T.add twg_in.bv grad_contrib);
            forward_val)
    | E_sin { t_in } ->
        Some
          (handle_unary_op ~op_name:"sin" ~op:op_sin ~deriv:deriv_sin
             get_or_init_twg t_in)
    | E_sqrt { t_in = t_in_val } ->
        Some
          (fun k ->
            let result_val = T.sqrt t_in_val in
            let forward_val = continue k result_val in
            Debug.with_context "∇sqrt" (fun () ->
                let twg_in = get_or_init_twg t_in_val in
                let twg_res = get_or_init_twg result_val in
                let d_loss_d_result = grad_of twg_res in
                let d_result_d_input =
                  deriv_sqrt result_val (value_of twg_in)
                in
                let grad_contrib = T.mul d_loss_d_result d_result_d_input in
                twg_in.bv <- T.add twg_in.bv grad_contrib);
            forward_val)
    | E_recip { t_in } ->
        Some
          (handle_unary_op ~op_name:"recip" ~op:op_recip ~deriv:deriv_recip
             get_or_init_twg t_in)
    | E_fdiv { a; b } ->
        Some
          (handle_binary_op ~op_name:"fdiv" ~op:op_fdiv
             ~deriv_wrt_op1:deriv_fdiv_wrt_op1 ~deriv_wrt_op2:deriv_fdiv_wrt_op2
             get_or_init_twg a b)
    | E_pow { a = op1_val; b = op2_val } ->
        Some
          (fun k ->
            let result_val = op_pow op1_val op2_val in
            let forward_val = continue k result_val in
            Debug.with_context "∇pow" (fun () ->
                let twg_op1 = get_or_init_twg op1_val in
                let twg_op2 = get_or_init_twg op2_val in
                let twg_res = get_or_init_twg result_val in
                let d_loss_d_result = grad_of twg_res in

                let d_result_d_op1 =
                  deriv_pow_wrt_op1 (value_of twg_op1) (value_of twg_op2)
                in
                let grad_contrib_to_op1 =
                  T.mul d_loss_d_result d_result_d_op1
                in
                twg_op1.bv <- T.add twg_op1.bv grad_contrib_to_op1;

                match dtype (value_of twg_op1) with
                | Dtype.Float32 | Dtype.Float64 ->
                    let op1_float = T.cast Dtype.float32 (value_of twg_op1) in
                    let result_float = T.cast Dtype.float32 result_val in
                    let d_result_d_op2 =
                      deriv_pow_wrt_op2_float result_float op1_float
                    in
                    let d_result_d_op2_orig_dtype =
                      T.cast (dtype (value_of twg_op2)) d_result_d_op2
                    in
                    let grad_contrib_to_op2 =
                      T.mul d_loss_d_result d_result_d_op2_orig_dtype
                    in
                    twg_op2.bv <- T.add twg_op2.bv grad_contrib_to_op2
                | _ -> ());
            forward_val)
    | E_max { a = op1_val; b = op2_val } ->
        Some
          (fun k ->
            let result_val = op_max op1_val op2_val in
            let forward_val = continue k result_val in
            Debug.with_context "∇max" (fun () ->
                let twg_op1 = get_or_init_twg op1_val in
                let twg_op2 = get_or_init_twg op2_val in
                let twg_res = get_or_init_twg result_val in
                let d_loss_d_result = grad_of twg_res in
                let val_op1 = value_of twg_op1 in
                let val_op2 = value_of twg_op2 in
                let d_result_d_op1 =
                  deriv_max_wrt_op1 val_op1 val_op2 (dtype val_op1)
                in
                let grad_contrib_to_op1 =
                  T.mul d_loss_d_result d_result_d_op1
                in
                twg_op1.bv <- T.add twg_op1.bv grad_contrib_to_op1;
                let d_result_d_op2 =
                  deriv_max_wrt_op2 val_op1 val_op2 (dtype val_op2)
                in
                let grad_contrib_to_op2 =
                  T.mul d_loss_d_result d_result_d_op2
                in
                twg_op2.bv <- T.add twg_op2.bv grad_contrib_to_op2);
            forward_val)
    | E_reshape { t_in = t_in_val; new_shape } ->
        Some
          (fun k ->
            let result_val =
              op_reshape t_in_val (Symbolic_shape.of_ints new_shape)
            in
            let forward_val = continue k result_val in
            Debug.with_context "∇reshape" (fun () ->
                let twg_in = get_or_init_twg t_in_val in
                let twg_res = get_or_init_twg result_val in
                let d_loss_d_result = grad_of twg_res in
                let original_shape_in = T.shape (value_of twg_in) in
                let grad_contrib_in =
                  T.reshape original_shape_in d_loss_d_result
                in
                twg_in.bv <- T.add twg_in.bv grad_contrib_in);
            forward_val)
    | E_expand { t_in = t_in_val; new_target_shape } ->
        Some
          (fun k ->
            let result_val =
              op_expand t_in_val (Symbolic_shape.of_ints new_target_shape)
            in
            let forward_val = continue k result_val in
            Debug.with_context "∇expand" (fun () ->
                let twg_in = get_or_init_twg t_in_val in
                let twg_res = get_or_init_twg result_val in
                let d_loss_d_expanded_result = grad_of twg_res in

                let grad_contrib_to_original_input =
                  let original_input_shape = T.shape (value_of twg_in) in
                  let expanded_output_shape = new_target_shape in
                  if original_input_shape = expanded_output_shape then
                    d_loss_d_expanded_result
                  else
                    let rank_orig_in = Array.length original_input_shape in
                    let rank_expanded_out =
                      Array.length expanded_output_shape
                    in
                    let axes_to_sum_list = ref [] in

                    if rank_expanded_out > rank_orig_in then
                      for i = 0 to rank_expanded_out - rank_orig_in - 1 do
                        axes_to_sum_list := i :: !axes_to_sum_list
                      done;

                    for i = 0 to rank_orig_in - 1 do
                      let orig_in_dim_size = original_input_shape.(i) in
                      let expanded_out_dim_idx =
                        i + (rank_expanded_out - rank_orig_in)
                      in
                      let expanded_out_dim_size =
                        expanded_output_shape.(expanded_out_dim_idx)
                      in
                      if orig_in_dim_size = 1 && expanded_out_dim_size > 1 then
                        axes_to_sum_list :=
                          expanded_out_dim_idx :: !axes_to_sum_list
                    done;

                    let summed_grad =
                      if !axes_to_sum_list <> [] then
                        T.sum d_loss_d_expanded_result
                          ~axes:(Array.of_list (List.rev !axes_to_sum_list))
                          ~keepdims:true
                      else d_loss_d_expanded_result
                    in
                    if T.shape summed_grad <> original_input_shape then
                      T.reshape original_input_shape summed_grad
                    else summed_grad
                in
                twg_in.bv <- T.add twg_in.bv grad_contrib_to_original_input);
            forward_val)
    | E_reduce_sum { t_in = t_in_val; axes; keepdims } ->
        Some
          (fun k ->
            let result_val = op_reduce_sum ~axes ~keepdims t_in_val in
            let forward_val = continue k result_val in
            Debug.with_context "∇reduce_sum" (fun () ->
                let twg_in = get_or_init_twg t_in_val in
                let twg_res = get_or_init_twg result_val in
                let d_loss_d_result = grad_of twg_res in
                let original_input_shape = T.shape (value_of twg_in) in
                let grad_prepared_for_broadcast =
                  prepare_grad_for_broadcast d_loss_d_result (value_of twg_in)
                    axes keepdims (fun t ~axes ~keepdims ->
                      T.sum t ~axes ~keepdims)
                in
                let grad_contrib_to_input =
                  T.broadcast_to original_input_shape
                    grad_prepared_for_broadcast
                in
                twg_in.bv <- T.add twg_in.bv grad_contrib_to_input);
            forward_val)
    | E_reduce_max { t_in = t_in_val; axes; keepdims } ->
        Some
          (fun k ->
            let result_val = op_reduce_max ~axes ~keepdims t_in_val in
            let forward_val = continue k result_val in
            Debug.with_context "∇reduce_max" (fun () ->
                let twg_in = get_or_init_twg t_in_val in
                let twg_res = get_or_init_twg result_val in
                let d_loss_d_result = grad_of twg_res in
                let val_in = value_of twg_in in
                let original_input_shape = T.shape val_in in

                let grad_prepared_for_broadcast =
                  prepare_grad_for_broadcast d_loss_d_result val_in axes
                    keepdims (fun t ~axes ~keepdims -> T.max t ~axes ~keepdims)
                in
                let d_loss_d_result_broadcasted =
                  T.broadcast_to original_input_shape
                    grad_prepared_for_broadcast
                in

                let result_val_prepared_for_broadcast =
                  prepare_grad_for_broadcast result_val val_in axes keepdims
                    (fun t ~axes ~keepdims -> T.max t ~axes ~keepdims)
                in
                let result_val_broadcasted_for_compare =
                  T.broadcast_to original_input_shape
                    result_val_prepared_for_broadcast
                in

                let mask = T.equal val_in result_val_broadcasted_for_compare in
                let d_result_d_input_mask_casted =
                  T.cast (dtype d_loss_d_result) mask
                in
                let grad_contrib_to_input =
                  T.mul d_loss_d_result_broadcasted d_result_d_input_mask_casted
                in
                twg_in.bv <- T.add twg_in.bv grad_contrib_to_input);
            forward_val)
    | E_reduce_prod { t_in = t_in_val; axes; keepdims } ->
        Some
          (fun k ->
            let result_val = op_reduce_prod ~axes ~keepdims t_in_val in
            let forward_val = continue k result_val in
            Debug.with_context "reduce_prod" (fun () ->
                let twg_in = get_or_init_twg t_in_val in
                let twg_res = get_or_init_twg result_val in
                let d_loss_d_result = grad_of twg_res in
                let val_in = value_of twg_in in
                let original_input_shape = T.shape val_in in

                let grad_prepared_for_broadcast =
                  prepare_grad_for_broadcast d_loss_d_result val_in axes
                    keepdims (fun t ~axes ~keepdims -> T.prod t ~axes ~keepdims)
                in
                let d_loss_d_result_broadcasted =
                  T.broadcast_to original_input_shape
                    grad_prepared_for_broadcast
                in

                let result_val_prepared_for_broadcast =
                  prepare_grad_for_broadcast result_val val_in axes keepdims
                    (fun t ~axes ~keepdims -> T.prod t ~axes ~keepdims)
                in
                let result_val_broadcasted =
                  T.broadcast_to original_input_shape
                    result_val_prepared_for_broadcast
                in

                let epsilon = T.zeros_like val_in in
                let t_in_val_safe = T.add val_in epsilon in
                let d_result_d_input_term =
                  T.div result_val_broadcasted t_in_val_safe
                in
                let grad_contrib_to_input =
                  T.mul d_loss_d_result_broadcasted d_result_d_input_term
                in
                twg_in.bv <- T.add twg_in.bv grad_contrib_to_input);
            forward_val)
    | E_associative_scan { t_in = t_in_val; axis; op } ->
        Some
          (fun k ->
            let result_val = op_associative_scan ~axis ~op t_in_val in
            let forward_val = continue k result_val in
            let shape_in = T.shape t_in_val in
            let axis_norm = normalize_axis axis shape_in in
            Debug.with_context "∇associative_scan" (fun () ->
                let twg_in = get_or_init_twg t_in_val in
                let twg_res = get_or_init_twg result_val in
                let d_loss_d_result = grad_of twg_res in
                let grad_contrib =
                  match op with
                  | `Sum -> reverse_cumsum d_loss_d_result axis_norm
                  | `Prod ->
                      let prefix = prefix_exclusive axis_norm t_in_val in
                      let suffix = suffix_exclusive axis_norm t_in_val in
                      let h = divide_no_nan d_loss_d_result suffix in
                      let tail_sum = T.sub (reverse_cumsum h axis_norm) h in
                      let inner =
                        T.add d_loss_d_result (T.mul suffix tail_sum)
                      in
                      T.mul prefix inner
                  | `Max | `Min ->
                      failwith "autodiff: cummax/cummin gradients not supported"
                in
                twg_in.bv <- T.add twg_in.bv grad_contrib);
            forward_val)
    | E_permute { t_in = t_in_val; axes = permute_axes } ->
        Some
          (fun k ->
            let result_val = op_permute t_in_val permute_axes in
            let forward_val = continue k result_val in
            Debug.with_context "∇permute" (fun () ->
                let twg_in = get_or_init_twg t_in_val in
                let twg_res = get_or_init_twg result_val in
                let d_loss_d_result = grad_of twg_res in

                let rank = Array.length permute_axes in
                let un_permute_axes = Array.make rank 0 in
                Array.iteri
                  (fun i original_pos -> un_permute_axes.(original_pos) <- i)
                  permute_axes;

                let grad_contrib_to_input =
                  T.transpose d_loss_d_result ~axes:un_permute_axes
                in
                twg_in.bv <- T.add twg_in.bv grad_contrib_to_input);
            forward_val)
    | E_pad { t_in = t_in_val; padding_config; fill_value } ->
        Some
          (fun k ->
            let result_val = op_pad t_in_val padding_config fill_value in
            let forward_val = continue k result_val in
            Debug.with_context "∇pad" (fun () ->
                let twg_in = get_or_init_twg t_in_val in
                let twg_res = get_or_init_twg result_val in
                let d_loss_d_result = grad_of twg_res in
                let original_input_shape = T.shape (value_of twg_in) in

                let shrink_limits =
                  Array.mapi
                    (fun dim_idx (pad_before, _) ->
                      (pad_before, pad_before + original_input_shape.(dim_idx)))
                    padding_config
                in
                let grad_contrib_to_input =
                  T.shrink shrink_limits d_loss_d_result
                in
                twg_in.bv <- T.add twg_in.bv grad_contrib_to_input);
            forward_val)
    | E_shrink { t_in = t_in_val; limits = shrink_limits } ->
        Some
          (fun k ->
            let result_val = op_shrink t_in_val shrink_limits in
            let forward_val = continue k result_val in
            Debug.with_context "∇shrink" (fun () ->
                let twg_in = get_or_init_twg t_in_val in
                let twg_res = get_or_init_twg result_val in
                let d_loss_d_result = grad_of twg_res in
                let original_input_shape = T.shape (value_of twg_in) in

                let padding_config =
                  Array.mapi
                    (fun dim_idx (start, stop_exclusive) ->
                      let original_dim_size = original_input_shape.(dim_idx) in
                      (start, original_dim_size - stop_exclusive))
                    shrink_limits
                in
                let zero_val = Dtype.zero (dtype d_loss_d_result) in
                let grad_contrib_to_input =
                  T.pad padding_config zero_val d_loss_d_result
                in
                twg_in.bv <- T.add twg_in.bv grad_contrib_to_input);
            forward_val)
    | E_as_strided { t_in = t_in_val; new_shape; new_strides; offset } ->
        Some
          (fun k ->
            let result_val =
              op_as_strided t_in_val
                (Nx_core.Symbolic_shape.of_ints new_shape)
                new_strides offset
            in
            let forward_val = continue k result_val in
            Debug.with_context "∇as_strided" (fun () ->
                let _twg_in = get_or_init_twg t_in_val in
                let twg_res = get_or_init_twg result_val in
                let _d_loss_d_result = grad_of twg_res in

                (* For as_strided, gradients need to be accumulated back to the original
                   tensor following the strided access pattern. For now, we'll raise an
                   error for non-contiguous strides during autodiff. *)
                (* TODO: Implement proper gradient accumulation for strided views *)
                let () =
                  failwith
                    "as_strided gradient not yet implemented - use contiguous \
                     tensors in autodiff"
                in
                ());
            forward_val)
    | E_flip { t_in = t_in_val; dims_to_flip } ->
        Some
          (fun k ->
            let axes_to_flip =
              dims_to_flip |> Array.to_list
              |> List.mapi (fun i flip -> if flip then Some i else None)
              |> List.filter_map Fun.id |> Array.of_list
            in
            let result_val = op_flip t_in_val dims_to_flip in
            let forward_val = continue k result_val in
            Debug.with_context "∇flip" (fun () ->
                let twg_in = get_or_init_twg t_in_val in
                let twg_res = get_or_init_twg result_val in
                let d_loss_d_result = grad_of twg_res in
                let grad_contrib_to_input =
                  T.flip d_loss_d_result ~axes:axes_to_flip
                in
                twg_in.bv <- T.add twg_in.bv grad_contrib_to_input);
            forward_val)
    | E_cat { t_list; axis } ->
        Some
          (fun k ->
            let result_val = op_cat t_list axis in
            let forward_val = continue k result_val in
            Debug.with_context "∇cat" (fun () ->
                let twg_inputs = List.map get_or_init_twg t_list in
                let twg_res = get_or_init_twg result_val in
                let d_loss_d_result = grad_of twg_res in
                let d_loss_result_shape = T.shape d_loss_d_result in

                let current_offset = ref 0 in
                List.iter
                  (fun twg_in_current ->
                    let input_val = value_of twg_in_current in
                    let input_shape = T.shape input_val in
                    let size_along_axis = input_shape.(axis) in
                    let shrink_limits =
                      Array.mapi
                        (fun i dim_size ->
                          if i = axis then
                            (!current_offset, !current_offset + size_along_axis)
                          else (0, dim_size))
                        d_loss_result_shape
                    in
                    let grad_slice_for_input =
                      T.shrink shrink_limits d_loss_d_result
                    in
                    twg_in_current.bv <-
                      T.add twg_in_current.bv grad_slice_for_input;
                    current_offset := !current_offset + size_along_axis)
                  twg_inputs);
            forward_val)
    | E_cast { t_in = t_in_val; target_dtype } ->
        Some
          (fun k ->
            let result_val = op_cast t_in_val target_dtype in
            let forward_val = continue k result_val in
            Debug.with_context "∇cast" (fun () ->
                let twg_in = get_or_init_twg t_in_val in
                let twg_res = get_or_init_twg result_val in
                let d_loss_d_result = grad_of twg_res in
                let original_dtype = dtype (value_of twg_in) in
                let grad_contrib_to_input =
                  T.cast original_dtype d_loss_d_result
                in
                twg_in.bv <- T.add twg_in.bv grad_contrib_to_input);
            forward_val)
    | E_contiguous { t_in = t_in_val } ->
        Some
          (handle_identity_gradient_op ~op_name:"contiguous" ~op:op_contiguous
             get_or_init_twg t_in_val)
    | E_copy { t_in = t_in_val } ->
        Some
          (handle_identity_gradient_op ~op_name:"copy" ~op:op_copy
             get_or_init_twg t_in_val)
    | E_where { condition = cond_val; if_true = true_val; if_false = false_val }
      ->
        Some
          (fun k ->
            let result_val = op_where cond_val true_val false_val in
            let forward_val = continue k result_val in
            Debug.with_context "∇where" (fun () ->
                let _twg_cond = get_or_init_twg cond_val in
                let twg_true = get_or_init_twg true_val in
                let twg_false = get_or_init_twg false_val in
                let twg_res = get_or_init_twg result_val in
                let d_loss_d_result = grad_of twg_res in

                let condition_mask_casted =
                  T.cast (dtype d_loss_d_result) cond_val
                in
                let grad_contrib_to_true =
                  T.mul d_loss_d_result condition_mask_casted
                in
                twg_true.bv <- T.add twg_true.bv grad_contrib_to_true;

                let ones_for_mask_dtype = T.ones_like condition_mask_casted in
                let not_condition_mask_casted =
                  T.sub ones_for_mask_dtype condition_mask_casted
                in
                let grad_contrib_to_false =
                  T.mul d_loss_d_result not_condition_mask_casted
                in
                twg_false.bv <- T.add twg_false.bv grad_contrib_to_false);
            forward_val)
    | E_gather { data = data_val; indices = indices_val; axis } ->
        Some
          (fun k ->
            let result_val = op_gather data_val indices_val axis in
            let forward_val = continue k result_val in
            Debug.with_context "∇gather" (fun () ->
                let twg_data = get_or_init_twg data_val in
                let _twg_indices = get_or_init_twg indices_val in
                let twg_res = get_or_init_twg result_val in
                let d_loss_d_result = grad_of twg_res in

                let zeros_data = T.zeros_like (value_of twg_data) in
                let scattered_grads =
                  op_scatter ~mode:`Add zeros_data indices_val d_loss_d_result
                    axis
                in
                twg_data.bv <- T.add twg_data.bv scattered_grads);
            forward_val)
    | E_scatter
        { data_template = dt_val; indices = idx_val; updates = upd_val; axis }
      ->
        Some
          (fun k ->
            let result_val = op_scatter dt_val idx_val upd_val axis in
            let forward_val = continue k result_val in
            Debug.with_context "∇scatter" (fun () ->
                let twg_dt = get_or_init_twg dt_val in
                let twg_upd = get_or_init_twg upd_val in
                let _twg_idx = get_or_init_twg idx_val in
                let twg_res = get_or_init_twg result_val in
                let d_loss_d_result = grad_of twg_res in

                let grad_contrib_to_updates =
                  op_gather d_loss_d_result idx_val axis
                in
                twg_upd.bv <- T.add twg_upd.bv grad_contrib_to_updates;

                let mask_for_dt_grad =
                  op_scatter (T.ones_like dt_val) idx_val (T.zeros_like upd_val)
                    axis
                in
                let grad_contrib_to_dt =
                  T.mul d_loss_d_result mask_for_dt_grad
                in
                twg_dt.bv <- T.add twg_dt.bv grad_contrib_to_dt);
            forward_val)
    | E_assign { dst = dst_val; src = src_val } ->
        Some
          (fun k ->
            let old_dst_val = T.copy dst_val in
            op_assign dst_val src_val;
            let forward_val = continue k () in
            Debug.with_context "∇assign" (fun () ->
                let twg_src = get_or_init_twg src_val in
                let twg_dst = get_or_init_twg dst_val in
                let _twg_old_dst = get_or_init_twg old_dst_val in
                twg_src.bv <- T.add twg_src.bv (grad_of twg_dst));
            forward_val)
    | E_idiv { a; b } ->
        Some
          (fun k ->
            let result_val = op_idiv a b in
            let forward_val = continue k result_val in
            Debug.with_context "∇idiv" (fun () ->
                let _twg_a = get_or_init_twg a in
                let _twg_b = get_or_init_twg b in
                let _twg_res = get_or_init_twg result_val in
                ());
            forward_val)
    | E_mod { a; b } ->
        Some
          (fun k ->
            let result_val = T.mod_ a b in
            let forward_val = continue k result_val in
            Debug.with_context "∇mod" (fun () ->
                let _twg_a = get_or_init_twg a in
                let _twg_b = get_or_init_twg b in
                let _twg_res = get_or_init_twg result_val in
                ());
            forward_val)
    | E_cmplt { a; b } ->
        Some
          (fun k ->
            let result_val = op_cmplt a b in
            let forward_val = continue k result_val in
            Debug.with_context "∇cmplt" (fun () ->
                let _twg_a = get_or_init_twg a in
                let _twg_b = get_or_init_twg b in
                let _twg_res = get_or_init_twg result_val in
                ());
            forward_val)
    | E_cmpne { a; b } ->
        Some
          (fun k ->
            let result_val = op_cmpne a b in
            let forward_val = continue k result_val in
            Debug.with_context "∇cmpne" (fun () ->
                let _twg_a = get_or_init_twg a in
                let _twg_b = get_or_init_twg b in
                let _twg_res = get_or_init_twg result_val in
                ());
            forward_val)
    | E_xor { a; b } ->
        Some
          (fun k ->
            let result_val = op_xor a b in
            let forward_val = continue k result_val in
            Debug.with_context "∇xor" (fun () ->
                let _twg_a = get_or_init_twg a in
                let _twg_b = get_or_init_twg b in
                let _twg_res = get_or_init_twg result_val in
                ());
            forward_val)
    | E_or { a; b } ->
        Some
          (fun k ->
            let result_val = op_or a b in
            let forward_val = continue k result_val in
            Debug.with_context "∇or" (fun () ->
                let _twg_a = get_or_init_twg a in
                let _twg_b = get_or_init_twg b in
                let _twg_res = get_or_init_twg result_val in
                ());
            forward_val)
    | E_and { a; b } ->
        Some
          (fun k ->
            let result_val = op_and a b in
            let forward_val = continue k result_val in
            Debug.with_context "∇and" (fun () ->
                let _twg_a = get_or_init_twg a in
                let _twg_b = get_or_init_twg b in
                let _twg_res = get_or_init_twg result_val in
                ());
            forward_val)
    | E_const_array { context = effect_ctx; array } ->
        Some
          (fun k ->
            let result_val = op_const_array effect_ctx array in
            let forward_val = continue k result_val in
            Debug.with_context "∇const_array" (fun () ->
                let _twg_res = get_or_init_twg result_val in
                ());
            forward_val)
    | E_threefry { key = key_val; ctr = ctr_val } ->
        Some
          (fun k ->
            let result_val = op_threefry key_val ctr_val in
            let forward_val = continue k result_val in
            Debug.with_context "∇threefry" (fun () ->
                let _twg_key = get_or_init_twg key_val in
                let _twg_ctr = get_or_init_twg ctr_val in
                let _twg_res = get_or_init_twg result_val in
                ());
            forward_val)
    | E_unfold { t_in = t_in_val; kernel_size; stride; dilation; padding } ->
        Some
          (fun k ->
            let result_val =
              op_unfold t_in_val ~kernel_size ~stride ~dilation ~padding
            in
            let forward_val = continue k result_val in
            Debug.with_context "∇unfold" (fun () ->
                let twg_in = get_or_init_twg t_in_val in
                let twg_res = get_or_init_twg result_val in
                let d_loss_d_result = grad_of twg_res in
                (* Gradient of unfold is fold operation *)
                let input_shape = T.shape (value_of twg_in) in
                let num_spatial_dims = Array.length kernel_size in
                let output_size =
                  Array.sub input_shape
                    (Array.length input_shape - num_spatial_dims)
                    num_spatial_dims
                in
                let grad_contrib_in =
                  Nx_rune.op_fold d_loss_d_result ~output_size ~kernel_size
                    ~stride ~dilation ~padding
                in
                twg_in.bv <- T.add twg_in.bv grad_contrib_in);
            forward_val)
    | E_fold
        { t_in = t_in_val; output_size; kernel_size; stride; dilation; padding }
      ->
        Some
          (fun k ->
            let result_val =
              op_fold t_in_val ~output_size ~kernel_size ~stride ~dilation
                ~padding
            in
            let forward_val = continue k result_val in
            Debug.with_context "∇fold" (fun () ->
                let twg_in = get_or_init_twg t_in_val in
                let twg_res = get_or_init_twg result_val in
                let d_loss_d_result = grad_of twg_res in
                (* Gradient of fold is unfold operation *)
                let grad_contrib_in =
                  Nx_rune.op_unfold d_loss_d_result ~kernel_size ~stride
                    ~dilation ~padding
                in
                twg_in.bv <- T.add twg_in.bv grad_contrib_in);
            forward_val)
    | E_matmul { a = a_val; b = b_val } ->
        Some
          (fun k ->
            let result_val = op_matmul a_val b_val in
            let forward_val = continue k result_val in
            Debug.with_context "∇matmul" (fun () ->
                let twg_a = get_or_init_twg a_val in
                let twg_b = get_or_init_twg b_val in
                let twg_res = get_or_init_twg result_val in
                let d_loss_d_result = grad_of twg_res in
                (* For C = A @ B: dL/dA = dL/dC @ B^T dL/dB = A^T @ dL/dC *)
                (* Handle broadcasting for matmul gradients *)
                let a_ndim = Array.length (T.shape a_val) in
                let b_ndim = Array.length (T.shape b_val) in
                let grad_contrib_to_a, grad_contrib_to_b =
                  if a_ndim = 2 && b_ndim = 3 then
                    (* Special case: A is 2D, B is 3D - this is a broadcasted matmul *)
                    (* For C = A @ B where A:[m,k] B:[b,k,n] -> C:[b,m,n] *)
                    (* grad_A = sum(grad_C @ B^T, axis=0) *)
                    (* grad_B = A^T @ grad_C *)
                    let b_transposed = T.transpose ~axes:[| 0; 2; 1 |] b_val in
                    let grad_a_3d = T.matmul d_loss_d_result b_transposed in
                    let grad_a = T.sum grad_a_3d ~axes:[| 0 |] in
                    let a_expanded = T.expand_dims [| 0 |] a_val in
                    let a_transposed =
                      T.transpose ~axes:[| 0; 2; 1 |] a_expanded
                    in
                    let grad_b = T.matmul a_transposed d_loss_d_result in
                    (grad_a, grad_b)
                  else if a_ndim = 3 && b_ndim = 2 then
                    (* Special case: A is 3D, B is 2D - this is a broadcasted matmul *)
                    (* For C = A @ B where A:[b,m,k] B:[k,n] -> C:[b,m,n] *)
                    (* grad_A = grad_C @ B^T *)
                    (* grad_B = sum(A^T @ grad_C, axis=0) *)
                    let grad_a = T.matmul d_loss_d_result (T.transpose b_val) in
                    let a_transposed = T.transpose ~axes:[| 0; 2; 1 |] a_val in
                    let grad_b_3d = T.matmul a_transposed d_loss_d_result in
                    let grad_b = T.sum grad_b_3d ~axes:[| 0 |] in
                    (grad_a, grad_b)
                  else
                    (* Standard case - both same dimensionality *)
                    (* For matmul, we need to transpose the last two dimensions *)
                    let ndim = Array.length (T.shape a_val) in
                    let transpose_last_two tensor =
                      if ndim >= 2 then (
                        let axes = Array.init ndim (fun i -> i) in
                        (* Swap last two dimensions *)
                        axes.(ndim - 2) <- ndim - 1;
                        axes.(ndim - 1) <- ndim - 2;
                        T.transpose ~axes tensor)
                      else T.transpose tensor
                    in
                    let grad_a =
                      T.matmul d_loss_d_result (transpose_last_two b_val)
                    in
                    let grad_b =
                      T.matmul (transpose_last_two a_val) d_loss_d_result
                    in
                    (grad_a, grad_b)
                in
                twg_a.bv <- T.add twg_a.bv grad_contrib_to_a;
                twg_b.bv <- T.add twg_b.bv grad_contrib_to_b);
            forward_val)
    | _ -> None
  in

  {
    retc =
      (fun final_result_val ->
        Debug.with_context "∇grad_init" (fun () ->
            let twg_final_result = get_or_init_twg final_result_val in
            twg_final_result.bv <- T.ones_like final_result_val);
        final_result_val);
    exnc = raise;
    effc;
  }

(* --- User-facing grad functions --- *)

let grad (f : ('a, 'b) t -> ('c, 'd) t) (input_val : ('a, 'b) t) : ('a, 'b) t =
  let tape_by_twg_id : (int, any_t_with_grad) Hashtbl.t = Hashtbl.create 16 in
  let val_to_twg_id_map = PhysicalTbl.create 16 in
  let initial_grad_for_input = T.zeros_like input_val in
  let twg_input_id = fresh_twg_id () in
  let twg_input =
    { v = input_val; bv = initial_grad_for_input; id = twg_input_id }
  in
  Hashtbl.add tape_by_twg_id twg_input_id (Any_t_with_grad twg_input);
  PhysicalTbl.add val_to_twg_id_map input_val twg_input_id;
  let ad_handler = make_reverse_handler tape_by_twg_id val_to_twg_id_map in
  let result_value_from_f = Effect.Deep.match_with f input_val ad_handler in

  (* Initialize output gradient to 1.0 *)
  (match PhysicalTbl.find_opt val_to_twg_id_map result_value_from_f with
  | Some twg_id -> (
      match Hashtbl.find_opt tape_by_twg_id twg_id with
      | Some any_twg ->
          let twg_res = unwrap_twg (dtype result_value_from_f) any_twg in
          twg_res.bv <- T.ones_like result_value_from_f
      | None -> ())
  | None -> ());

  let final_twg_input_id = PhysicalTbl.find val_to_twg_id_map input_val in
  let final_twg_input_any = Hashtbl.find tape_by_twg_id final_twg_input_id in
  let final_twg_input = unwrap_twg (dtype input_val) final_twg_input_any in
  final_twg_input.bv

let value_and_grad (f : ('a, 'b) t -> ('c, 'd) t) (input_val : ('a, 'b) t) :
    ('c, 'd) t * ('a, 'b) t =
  let tape_by_twg_id : (int, any_t_with_grad) Hashtbl.t = Hashtbl.create 16 in
  let val_to_twg_id_map = PhysicalTbl.create 16 in
  let initial_grad_for_input = T.zeros_like input_val in
  let twg_input_id = fresh_twg_id () in
  let twg_input =
    { v = input_val; bv = initial_grad_for_input; id = twg_input_id }
  in
  Hashtbl.add tape_by_twg_id twg_input_id (Any_t_with_grad twg_input);
  PhysicalTbl.add val_to_twg_id_map input_val twg_input_id;
  let ad_handler = make_reverse_handler tape_by_twg_id val_to_twg_id_map in
  let result_value_from_f = Effect.Deep.match_with f input_val ad_handler in

  (* Initialize output gradient to 1.0 *)
  (match PhysicalTbl.find_opt val_to_twg_id_map result_value_from_f with
  | Some twg_id -> (
      match Hashtbl.find_opt tape_by_twg_id twg_id with
      | Some any_twg ->
          let twg_res = unwrap_twg (dtype result_value_from_f) any_twg in
          twg_res.bv <- T.ones_like result_value_from_f
      | None -> ())
  | None -> ());

  let final_twg_input_id = PhysicalTbl.find val_to_twg_id_map input_val in
  let final_twg_input_any = Hashtbl.find tape_by_twg_id final_twg_input_id in
  let final_twg_input = unwrap_twg (dtype input_val) final_twg_input_any in
  (result_value_from_f, final_twg_input.bv)

(* New functions for multiple inputs *)

let grads (f : ('a, 'b) t list -> ('c, 'd) t) (input_vals : ('a, 'b) t list) :
    ('a, 'b) t list =
  let tape_by_twg_id : (int, any_t_with_grad) Hashtbl.t = Hashtbl.create 16 in
  let val_to_twg_id_map = PhysicalTbl.create 16 in

  (* Initialize all inputs *)
  let input_twgs =
    List.map
      (fun input_val ->
        let initial_grad = T.zeros_like input_val in
        let twg_id = fresh_twg_id () in
        let twg = { v = input_val; bv = initial_grad; id = twg_id } in
        Hashtbl.add tape_by_twg_id twg_id (Any_t_with_grad twg);
        PhysicalTbl.add val_to_twg_id_map input_val twg_id;
        twg)
      input_vals
  in

  let ad_handler = make_reverse_handler tape_by_twg_id val_to_twg_id_map in
  let result_value_from_f = Effect.Deep.match_with f input_vals ad_handler in

  (* Initialize output gradient to 1.0 *)
  (match PhysicalTbl.find_opt val_to_twg_id_map result_value_from_f with
  | Some twg_id -> (
      match Hashtbl.find_opt tape_by_twg_id twg_id with
      | Some any_twg ->
          let twg_res = unwrap_twg (dtype result_value_from_f) any_twg in
          twg_res.bv <- T.ones_like result_value_from_f
      | None -> ())
  | None -> ());

  (* Extract gradients for all inputs *)
  List.map2
    (fun input_val _ ->
      let twg_id = PhysicalTbl.find val_to_twg_id_map input_val in
      let any_twg = Hashtbl.find tape_by_twg_id twg_id in
      let twg = unwrap_twg (dtype input_val) any_twg in
      twg.bv)
    input_vals input_twgs

let value_and_grads (f : ('a, 'b) t list -> ('c, 'd) t)
    (input_vals : ('a, 'b) t list) : ('c, 'd) t * ('a, 'b) t list =
  let tape_by_twg_id : (int, any_t_with_grad) Hashtbl.t = Hashtbl.create 16 in
  let val_to_twg_id_map = PhysicalTbl.create 16 in

  (* Initialize all inputs *)
  let input_twgs =
    List.map
      (fun input_val ->
        let initial_grad = T.zeros_like input_val in
        let twg_id = fresh_twg_id () in
        let twg = { v = input_val; bv = initial_grad; id = twg_id } in
        Hashtbl.add tape_by_twg_id twg_id (Any_t_with_grad twg);
        PhysicalTbl.add val_to_twg_id_map input_val twg_id;
        twg)
      input_vals
  in

  let ad_handler = make_reverse_handler tape_by_twg_id val_to_twg_id_map in
  let result_value_from_f = Effect.Deep.match_with f input_vals ad_handler in

  (* Initialize output gradient to 1.0 *)
  (match PhysicalTbl.find_opt val_to_twg_id_map result_value_from_f with
  | Some twg_id -> (
      match Hashtbl.find_opt tape_by_twg_id twg_id with
      | Some any_twg ->
          let twg_res = unwrap_twg (dtype result_value_from_f) any_twg in
          twg_res.bv <- T.ones_like result_value_from_f
      | None -> ())
  | None -> ());

  (* Extract gradients for all inputs *)
  let grads =
    List.map2
      (fun input_val _ ->
        let twg_id = PhysicalTbl.find val_to_twg_id_map input_val in
        let any_twg = Hashtbl.find tape_by_twg_id twg_id in
        let twg = unwrap_twg (dtype input_val) any_twg in
        twg.bv)
      input_vals input_twgs
  in
  (result_value_from_f, grads)

(* --- Forward mode AD implementation --- *)

(* The main forward-mode AD effect handler *)
let make_forward_handler primal_to_dual_map =
  let open Effect.Deep in
  let get_dual (type a b) (tensor_val : (a, b) t) : (a, b) dual =
    match PhysicalTbl.find_opt primal_to_dual_map tensor_val with
    | Some (Any_dual d) -> unwrap_dual (dtype tensor_val) (Any_dual d)
    | None ->
        (* Non-differentiable tensors have zero tangent *)
        let zero_tangent = T.zeros_like tensor_val in
        let dual = { primal = tensor_val; tangent = zero_tangent } in
        PhysicalTbl.add primal_to_dual_map tensor_val (Any_dual dual);
        dual
  in

  let effc : type a. a Effect.t -> ((a, _) continuation -> _) option = function
    | E_buffer { context = effect_ctx; dtype = dt; size_in_elements } ->
        Some
          (fun k ->
            let result_val = op_buffer effect_ctx dt size_in_elements in
            let forward_val = continue k result_val in
            (* Buffer creates new tensor - initialize with zero tangent *)
            let zero_tangent = T.zeros_like result_val in
            let dual = { primal = result_val; tangent = zero_tangent } in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual dual);
            forward_val)
    | E_const_scalar { context = effect_ctx; value; dtype = dt } ->
        Some
          (fun k ->
            let result_val = op_const_scalar effect_ctx value dt in
            let forward_val = continue k result_val in
            (* Constants have zero tangent *)
            let zero_tangent = T.zeros_like result_val in
            let dual = { primal = result_val; tangent = zero_tangent } in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual dual);
            forward_val)
    | E_add { a = op1_val; b = op2_val } ->
        Some
          (fun k ->
            let result_val = op_add op1_val op2_val in
            let dual1 = get_dual op1_val in
            let dual2 = get_dual op2_val in
            let result_tangent = T.add dual1.tangent dual2.tangent in
            let result_dual =
              { primal = result_val; tangent = result_tangent }
            in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual result_dual);
            let forward_val = continue k result_val in
            forward_val)
    | E_mul { a = op1_val; b = op2_val } ->
        Some
          (fun k ->
            let result_val = op_mul op1_val op2_val in
            let dual1 = get_dual op1_val in
            let dual2 = get_dual op2_val in
            (* d(a*b) = da*b + a*db *)
            let tangent1 = T.mul dual1.tangent dual2.primal in
            let tangent2 = T.mul dual1.primal dual2.tangent in
            let result_tangent = T.add tangent1 tangent2 in
            let result_dual =
              { primal = result_val; tangent = result_tangent }
            in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual result_dual);
            let forward_val = continue k result_val in
            forward_val)
    | E_neg { t_in } ->
        Some
          (fun k ->
            let result_val = op_neg t_in in
            let dual_in = get_dual t_in in
            let result_tangent = T.neg dual_in.tangent in
            let result_dual =
              { primal = result_val; tangent = result_tangent }
            in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual result_dual);
            let forward_val = continue k result_val in
            forward_val)
    | E_log2 { t_in } ->
        Some
          (fun k ->
            let result_val = op_log2 t_in in
            let dual_in = get_dual t_in in
            let deriv = deriv_log2 dual_in.primal in
            let result_tangent = T.mul dual_in.tangent deriv in
            let result_dual =
              { primal = result_val; tangent = result_tangent }
            in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual result_dual);
            let forward_val = continue k result_val in
            forward_val)
    | E_exp2 { t_in } ->
        Some
          (fun k ->
            let result_val = op_exp2 t_in in
            let dual_in = get_dual t_in in
            let deriv = deriv_exp2 result_val dual_in.primal in
            let result_tangent = T.mul dual_in.tangent deriv in
            let result_dual =
              { primal = result_val; tangent = result_tangent }
            in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual result_dual);
            let forward_val = continue k result_val in
            forward_val)
    | E_sin { t_in } ->
        Some
          (fun k ->
            let result_val = op_sin t_in in
            let dual_in = get_dual t_in in
            let deriv = deriv_sin dual_in.primal in
            let result_tangent = T.mul dual_in.tangent deriv in
            let result_dual =
              { primal = result_val; tangent = result_tangent }
            in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual result_dual);
            let forward_val = continue k result_val in
            forward_val)
    | E_sqrt { t_in } ->
        Some
          (fun k ->
            let result_val = T.sqrt t_in in
            let dual_in = get_dual t_in in
            let deriv = deriv_sqrt result_val dual_in.primal in
            let result_tangent = T.mul dual_in.tangent deriv in
            let result_dual =
              { primal = result_val; tangent = result_tangent }
            in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual result_dual);
            let forward_val = continue k result_val in
            forward_val)
    | E_recip { t_in } ->
        Some
          (fun k ->
            let result_val = op_recip t_in in
            let dual_in = get_dual t_in in
            let deriv = deriv_recip dual_in.primal in
            let result_tangent = T.mul dual_in.tangent deriv in
            let result_dual =
              { primal = result_val; tangent = result_tangent }
            in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual result_dual);
            let forward_val = continue k result_val in
            forward_val)
    | E_fdiv { a; b } ->
        Some
          (fun k ->
            let result_val = op_fdiv a b in
            let dual_a = get_dual a in
            let dual_b = get_dual b in
            (* d(a/b) = da/b - a*db/b^2 *)
            let term1 = T.div dual_a.tangent dual_b.primal in
            let term2_num = T.mul dual_a.primal dual_b.tangent in
            let term2_den = T.mul dual_b.primal dual_b.primal in
            let term2 = T.div term2_num term2_den in
            let result_tangent = T.sub term1 term2 in
            let result_dual =
              { primal = result_val; tangent = result_tangent }
            in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual result_dual);
            let forward_val = continue k result_val in
            forward_val)
    | E_pow { a; b } ->
        Some
          (fun k ->
            let result_val = op_pow a b in
            let dual_a = get_dual a in
            let dual_b = get_dual b in
            (* d(a^b) = b*a^(b-1)*da + a^b*log(a)*db *)
            let deriv_wrt_a = deriv_pow_wrt_op1 dual_a.primal dual_b.primal in
            let term1 = T.mul dual_a.tangent deriv_wrt_a in
            let term2 =
              match dtype dual_a.primal with
              | Dtype.Float32 | Dtype.Float64 ->
                  let a_float = T.cast Dtype.float32 dual_a.primal in
                  let result_float = T.cast Dtype.float32 result_val in
                  let deriv_wrt_b =
                    deriv_pow_wrt_op2_float result_float a_float
                  in
                  let deriv_wrt_b_orig =
                    T.cast (dtype dual_b.primal) deriv_wrt_b
                  in
                  T.mul dual_b.tangent deriv_wrt_b_orig
              | _ -> T.zeros_like result_val
            in
            let result_tangent = T.add term1 term2 in
            let result_dual =
              { primal = result_val; tangent = result_tangent }
            in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual result_dual);
            let forward_val = continue k result_val in
            forward_val)
    | E_max { a; b } ->
        Some
          (fun k ->
            let result_val = op_max a b in
            let dual_a = get_dual a in
            let dual_b = get_dual b in
            let mask_a =
              deriv_max_wrt_op1 dual_a.primal dual_b.primal
                (dtype dual_a.primal)
            in
            let mask_b =
              deriv_max_wrt_op2 dual_a.primal dual_b.primal
                (dtype dual_b.primal)
            in
            let term1 = T.mul mask_a dual_a.tangent in
            let term2 = T.mul mask_b dual_b.tangent in
            let result_tangent = T.add term1 term2 in
            let result_dual =
              { primal = result_val; tangent = result_tangent }
            in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual result_dual);
            let forward_val = continue k result_val in
            forward_val)
    | E_reshape { t_in; new_shape } ->
        Some
          (fun k ->
            let result_val =
              op_reshape t_in (Symbolic_shape.of_ints new_shape)
            in
            let dual_in = get_dual t_in in
            let result_tangent = T.reshape new_shape dual_in.tangent in
            let result_dual =
              { primal = result_val; tangent = result_tangent }
            in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual result_dual);
            let forward_val = continue k result_val in
            forward_val)
    | E_expand { t_in; new_target_shape } ->
        Some
          (fun k ->
            let result_val =
              op_expand t_in (Symbolic_shape.of_ints new_target_shape)
            in
            let dual_in = get_dual t_in in
            let result_tangent =
              T.broadcast_to new_target_shape dual_in.tangent
            in
            let result_dual =
              { primal = result_val; tangent = result_tangent }
            in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual result_dual);
            let forward_val = continue k result_val in
            forward_val)
    | E_reduce_sum { t_in; axes; keepdims } ->
        Some
          (fun k ->
            let result_val = op_reduce_sum ~axes ~keepdims t_in in
            let dual_in = get_dual t_in in
            let result_tangent = T.sum dual_in.tangent ~axes ~keepdims in
            let result_dual =
              { primal = result_val; tangent = result_tangent }
            in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual result_dual);
            let forward_val = continue k result_val in
            forward_val)
    | E_reduce_max { t_in; axes; keepdims } ->
        Some
          (fun k ->
            let result_val = op_reduce_max ~axes ~keepdims t_in in
            let dual_in = get_dual t_in in
            (* For reduce_max, gradient flows only through the max elements *)
            let original_shape = T.shape t_in in
            let result_broadcasted =
              if keepdims then result_val
              else
                let dummy = T.zeros_like t_in in
                let shape_with_dims =
                  T.shape (T.max dummy ~axes ~keepdims:true)
                in
                let reshaped = T.reshape shape_with_dims result_val in
                T.broadcast_to original_shape reshaped
            in
            let mask = T.equal t_in result_broadcasted in
            let mask_float = T.cast (dtype dual_in.tangent) mask in
            let masked_tangent = T.mul dual_in.tangent mask_float in
            let result_tangent = T.sum masked_tangent ~axes ~keepdims in
            let result_dual =
              { primal = result_val; tangent = result_tangent }
            in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual result_dual);
            let forward_val = continue k result_val in
            forward_val)
    | E_reduce_prod { t_in; axes; keepdims } ->
        Some
          (fun k ->
            let result_val = op_reduce_prod ~axes ~keepdims t_in in
            let forward_val = continue k result_val in
            let dual_in = get_dual t_in in
            (* d(prod(x)) = sum(prod(x)/x_i * dx_i) *)
            let original_shape = T.shape t_in in
            let result_broadcasted =
              if keepdims then result_val
              else
                let dummy = T.zeros_like t_in in
                let shape_with_dims =
                  T.shape (T.prod dummy ~axes ~keepdims:true)
                in
                let reshaped = T.reshape shape_with_dims result_val in
                T.broadcast_to original_shape reshaped
            in
            let epsilon = T.zeros_like t_in in
            let safe_input = T.add t_in epsilon in
            let grad_term = T.div result_broadcasted safe_input in
            let result_tangent_full = T.mul dual_in.tangent grad_term in
            let result_tangent = T.sum result_tangent_full ~axes ~keepdims in
            let result_dual =
              { primal = result_val; tangent = result_tangent }
            in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual result_dual);
            forward_val)
    | E_associative_scan { t_in; axis; op } ->
        Some
          (fun k ->
            let result_val = op_associative_scan ~axis ~op t_in in
            let dual_in = get_dual t_in in
            let shape_in = T.shape t_in in
            let axis_norm = normalize_axis axis shape_in in
            let result_tangent =
              match op with
              | `Sum -> T.cumsum ~axis:axis_norm dual_in.tangent
              | `Prod ->
                  let prefix = prefix_exclusive axis_norm t_in in
                  let dx_over_x = divide_no_nan dual_in.tangent t_in in
                  let cum = T.cumsum ~axis:axis_norm dx_over_x in
                  let inner = T.mul result_val cum in
                  let zero_tensor =
                    T.full (context t_in) (T.dtype t_in) shape_in
                      (Dtype.zero (T.dtype t_in))
                  in
                  let zero_mask = T.equal t_in zero_tensor in
                  let fallback = T.mul prefix dual_in.tangent in
                  T.where zero_mask fallback inner
              | `Max | `Min ->
                  failwith "autodiff JVP: cummax/cummin not supported"
            in
            let result_dual =
              { primal = result_val; tangent = result_tangent }
            in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual result_dual);
            let forward_val = continue k result_val in
            forward_val)
    | E_permute { t_in; axes } ->
        Some
          (fun k ->
            let result_val = op_permute t_in axes in
            let dual_in = get_dual t_in in
            let result_tangent = T.transpose dual_in.tangent ~axes in
            let result_dual =
              { primal = result_val; tangent = result_tangent }
            in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual result_dual);
            let forward_val = continue k result_val in
            forward_val)
    | E_pad { t_in; padding_config; fill_value } ->
        Some
          (fun k ->
            let result_val = op_pad t_in padding_config fill_value in
            let forward_val = continue k result_val in
            let dual_in = get_dual t_in in
            let zero_val = Dtype.zero (dtype dual_in.tangent) in
            let result_tangent =
              T.pad padding_config zero_val dual_in.tangent
            in
            let result_dual =
              { primal = result_val; tangent = result_tangent }
            in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual result_dual);
            forward_val)
    | E_shrink { t_in; limits } ->
        Some
          (fun k ->
            let result_val = op_shrink t_in limits in
            let forward_val = continue k result_val in
            let dual_in = get_dual t_in in
            let result_tangent = T.shrink limits dual_in.tangent in
            let result_dual =
              { primal = result_val; tangent = result_tangent }
            in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual result_dual);
            forward_val)
    | E_as_strided { t_in; new_shape; new_strides; offset } ->
        Some
          (fun k ->
            let result_val =
              op_as_strided t_in
                (Nx_core.Symbolic_shape.of_ints new_shape)
                new_strides offset
            in
            let forward_val = continue k result_val in
            (* For JVP, as_strided applies the same transformation to the tangent *)
            (* TODO: This needs proper implementation for strided tangent propagation *)
            let () = failwith "as_strided JVP not yet implemented" in
            forward_val)
    | E_flip { t_in; dims_to_flip } ->
        Some
          (fun k ->
            let result_val = op_flip t_in dims_to_flip in
            let forward_val = continue k result_val in
            let dual_in = get_dual t_in in
            let axes_to_flip =
              dims_to_flip |> Array.to_list
              |> List.mapi (fun i flip -> if flip then Some i else None)
              |> List.filter_map Fun.id |> Array.of_list
            in
            let result_tangent = T.flip dual_in.tangent ~axes:axes_to_flip in
            let result_dual =
              { primal = result_val; tangent = result_tangent }
            in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual result_dual);
            forward_val)
    | E_cat { t_list; axis } ->
        Some
          (fun k ->
            let result_val = op_cat t_list axis in
            let forward_val = continue k result_val in
            let duals = List.map get_dual t_list in
            let tangents = List.map (fun d -> d.tangent) duals in
            let result_tangent = op_cat tangents axis in
            let result_dual =
              { primal = result_val; tangent = result_tangent }
            in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual result_dual);
            forward_val)
    | E_cast { t_in; target_dtype } ->
        Some
          (fun k ->
            let result_val = op_cast t_in target_dtype in
            let forward_val = continue k result_val in
            let dual_in = get_dual t_in in
            let result_tangent = T.cast target_dtype dual_in.tangent in
            let result_dual =
              { primal = result_val; tangent = result_tangent }
            in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual result_dual);
            forward_val)
    | E_contiguous { t_in } ->
        Some
          (fun k ->
            let result_val = op_contiguous t_in in
            let forward_val = continue k result_val in
            let dual_in = get_dual t_in in
            let result_tangent = T.contiguous dual_in.tangent in
            let result_dual =
              { primal = result_val; tangent = result_tangent }
            in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual result_dual);
            forward_val)
    | E_copy { t_in } ->
        Some
          (fun k ->
            let result_val = op_copy t_in in
            let forward_val = continue k result_val in
            let dual_in = get_dual t_in in
            let result_tangent = T.copy dual_in.tangent in
            let result_dual =
              { primal = result_val; tangent = result_tangent }
            in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual result_dual);
            forward_val)
    | E_where { condition; if_true; if_false } ->
        Some
          (fun k ->
            let result_val = op_where condition if_true if_false in
            let forward_val = continue k result_val in
            let dual_true = get_dual if_true in
            let dual_false = get_dual if_false in
            let result_tangent =
              T.where condition dual_true.tangent dual_false.tangent
            in
            let result_dual =
              { primal = result_val; tangent = result_tangent }
            in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual result_dual);
            forward_val)
    | E_gather { data; indices; axis } ->
        Some
          (fun k ->
            let result_val = op_gather data indices axis in
            let forward_val = continue k result_val in
            let dual_data = get_dual data in
            let result_tangent = op_gather dual_data.tangent indices axis in
            let result_dual =
              { primal = result_val; tangent = result_tangent }
            in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual result_dual);
            forward_val)
    | E_scatter { data_template; indices; updates; axis } ->
        Some
          (fun k ->
            let result_val = op_scatter data_template indices updates axis in
            let forward_val = continue k result_val in
            let dual_template = get_dual data_template in
            let dual_updates = get_dual updates in
            let result_tangent =
              op_scatter dual_template.tangent indices dual_updates.tangent axis
            in
            let result_dual =
              { primal = result_val; tangent = result_tangent }
            in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual result_dual);
            forward_val)
    | E_matmul { a; b } ->
        Some
          (fun k ->
            let result_val = op_matmul a b in
            let forward_val = continue k result_val in
            let dual_a = get_dual a in
            let dual_b = get_dual b in
            (* d(A @ B) = dA @ B + A @ dB *)
            let term1 = op_matmul dual_a.tangent dual_b.primal in
            let term2 = op_matmul dual_a.primal dual_b.tangent in
            let result_tangent = op_add term1 term2 in
            let result_dual =
              { primal = result_val; tangent = result_tangent }
            in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual result_dual);
            forward_val)
    | E_unfold { t_in; kernel_size; stride; dilation; padding } ->
        Some
          (fun k ->
            let result_val =
              op_unfold t_in ~kernel_size ~stride ~dilation ~padding
            in
            let forward_val = continue k result_val in
            let dual_in = get_dual t_in in
            let result_tangent =
              op_unfold dual_in.tangent ~kernel_size ~stride ~dilation ~padding
            in
            let result_dual =
              { primal = result_val; tangent = result_tangent }
            in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual result_dual);
            forward_val)
    | E_fold { t_in; output_size; kernel_size; stride; dilation; padding } ->
        Some
          (fun k ->
            let result_val =
              op_fold t_in ~output_size ~kernel_size ~stride ~dilation ~padding
            in
            let forward_val = continue k result_val in
            let dual_in = get_dual t_in in
            let result_tangent =
              op_fold dual_in.tangent ~output_size ~kernel_size ~stride
                ~dilation ~padding
            in
            let result_dual =
              { primal = result_val; tangent = result_tangent }
            in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual result_dual);
            forward_val)
    | E_assign { dst; src } ->
        Some
          (fun k ->
            op_assign dst src;
            let forward_val = continue k () in
            let dual_src = get_dual src in
            let dual_dst = get_dual dst in
            op_assign dual_dst.tangent dual_src.tangent;
            forward_val)
    (* Non-differentiable operations *)
    | E_idiv { a; b } ->
        Some
          (fun k ->
            let result_val = op_idiv a b in
            let forward_val = continue k result_val in
            let _ = get_dual result_val in
            forward_val)
    | E_mod { a; b } ->
        Some
          (fun k ->
            let result_val = T.mod_ a b in
            let forward_val = continue k result_val in
            let _ = get_dual result_val in
            forward_val)
    | E_cmplt { a; b } ->
        Some
          (fun k ->
            let result_val = op_cmplt a b in
            let forward_val = continue k result_val in
            let _ = get_dual result_val in
            forward_val)
    | E_cmpne { a; b } ->
        Some
          (fun k ->
            let result_val = op_cmpne a b in
            let forward_val = continue k result_val in
            let _ = get_dual result_val in
            forward_val)
    | E_xor { a; b } ->
        Some
          (fun k ->
            let result_val = op_xor a b in
            let forward_val = continue k result_val in
            let _ = get_dual result_val in
            forward_val)
    | E_or { a; b } ->
        Some
          (fun k ->
            let result_val = op_or a b in
            let forward_val = continue k result_val in
            let _ = get_dual result_val in
            forward_val)
    | E_and { a; b } ->
        Some
          (fun k ->
            let result_val = op_and a b in
            let forward_val = continue k result_val in
            let _ = get_dual result_val in
            forward_val)
    | E_const_array { context; array } ->
        Some
          (fun k ->
            let result_val = op_const_array context array in
            let forward_val = continue k result_val in
            (* Constants have zero tangent *)
            let zero_tangent = T.zeros_like result_val in
            let dual = { primal = result_val; tangent = zero_tangent } in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual dual);
            forward_val)
    | E_threefry { key; ctr } ->
        Some
          (fun k ->
            let result_val = op_threefry key ctr in
            let forward_val = continue k result_val in
            (* Random generation has zero tangent *)
            let zero_tangent = T.zeros_like result_val in
            let dual = { primal = result_val; tangent = zero_tangent } in
            PhysicalTbl.add primal_to_dual_map result_val (Any_dual dual);
            forward_val)
    | _ -> None
  in

  { retc = (fun x -> x); exnc = raise; effc }

(* JVP function following JAX API *)
let jvp (type a b c d) (f : (a, b) t -> (c, d) t) (primals : (a, b) t)
    (tangents : (a, b) t) : (c, d) t * (c, d) t =
  let primal_to_dual_map = PhysicalTbl.create 16 in
  (* Initialize input dual *)
  let input_dual = { primal = primals; tangent = tangents } in
  PhysicalTbl.add primal_to_dual_map primals (Any_dual input_dual);

  let handler = make_forward_handler primal_to_dual_map in
  let result_primal = Effect.Deep.match_with f primals handler in

  (* Get result tangent *)
  let result_dual =
    match PhysicalTbl.find_opt primal_to_dual_map result_primal with
    | Some (Any_dual d) -> unwrap_dual (dtype result_primal) (Any_dual d)
    | None -> { primal = result_primal; tangent = T.zeros_like result_primal }
  in

  (result_dual.primal, result_dual.tangent)

(* JVP with auxiliary output *)
let jvp_aux (type a b c d e) (f : (a, b) t -> (c, d) t * e) (primals : (a, b) t)
    (tangents : (a, b) t) : (c, d) t * (c, d) t * e =
  let primal_to_dual_map = PhysicalTbl.create 16 in
  (* Initialize input dual *)
  let input_dual = { primal = primals; tangent = tangents } in
  PhysicalTbl.add primal_to_dual_map primals (Any_dual input_dual);

  let handler = make_forward_handler primal_to_dual_map in
  let result_primal, aux = Effect.Deep.match_with f primals handler in

  (* Get result tangent *)
  let result_dual =
    match PhysicalTbl.find_opt primal_to_dual_map result_primal with
    | Some (Any_dual d) -> unwrap_dual (dtype result_primal) (Any_dual d)
    | None -> { primal = result_primal; tangent = T.zeros_like result_primal }
  in

  (result_dual.primal, result_dual.tangent, aux)

(* Multiple inputs version *)
let jvps (type a b c d) (f : (a, b) t list -> (c, d) t)
    (primals : (a, b) t list) (tangents : (a, b) t list) : (c, d) t * (c, d) t =
  if List.length primals <> List.length tangents then
    failwith "jvps: primals and tangents must have the same length";

  let primal_to_dual_map = PhysicalTbl.create 16 in
  (* Initialize input duals *)
  List.iter2
    (fun primal tangent ->
      let dual = { primal; tangent } in
      PhysicalTbl.add primal_to_dual_map primal (Any_dual dual))
    primals tangents;

  let handler = make_forward_handler primal_to_dual_map in
  let result_primal = Effect.Deep.match_with f primals handler in

  (* Get result tangent *)
  let result_dual =
    match PhysicalTbl.find_opt primal_to_dual_map result_primal with
    | Some (Any_dual d) -> unwrap_dual (dtype result_primal) (Any_dual d)
    | None -> { primal = result_primal; tangent = T.zeros_like result_primal }
  in

  (result_dual.primal, result_dual.tangent)
