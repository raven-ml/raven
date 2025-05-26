open Nx_core
open Nx_rune
module T = Tensor

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

(* --- Derivative definitions for UOps --- *)

let deriv_op_add_wrt_op1 op1_val _op2_val = T.ones_like op1_val
let deriv_op_add_wrt_op2 _op1_val op2_val = T.ones_like op2_val
let deriv_neg x = T.neg (T.ones_like x)

let deriv_log2 x =
  let x_squared = T.mul x x in
  T.div x x_squared

let deriv_exp2 exp2_x _x = exp2_x
let deriv_sin _x = failwith "deriv_sin: cos operation not yet implemented"
let deriv_sqrt sqrt_x _x = T.recip sqrt_x

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
  let log_2 = T.full ctx (dtype x) (T.shape x) 0.693147180559945309417 in
  T.mul log2_x log_2

let deriv_pow_wrt_op2_float result_val op1 =
  let log_op1 = log_e_float op1 in
  T.mul result_val log_op1

let deriv_max_wrt_op1 op1 op2 op1_dtype =
  T.cast op1_dtype (T.greater_equal op1 op2)

let deriv_max_wrt_op2 op1 op2 op2_dtype = T.cast op2_dtype (T.greater op2 op1)

let is_uninitialized_grad grad_tensor input_tensor =
  try grad_tensor == T.zeros_like input_tensor with _ -> false

let prepare_grad_for_broadcast grad_output input_tensor_val axes op_keepdims
    reduction_op_for_shape =
  if op_keepdims then grad_output
  else
    let dummy_input_like = T.zeros_like input_tensor_val in
    let reduced_shape_with_kept_dims =
      T.shape (reduction_op_for_shape dummy_input_like ~axes ~keepdims:true)
    in
    T.reshape reduced_shape_with_kept_dims grad_output

(* The main reverse-mode AD effect handler *)
let make_reverse_handler tape_by_twg_id val_to_twg_id_map =
  let open Effect.Deep in
  let get_or_init_twg tensor_val =
    match Hashtbl.find_opt val_to_twg_id_map (Obj.repr tensor_val) with
    | Some twg_id -> (
        match Hashtbl.find_opt tape_by_twg_id twg_id with
        | Some any_twg -> unwrap_twg (dtype tensor_val) any_twg
        | None -> failwith "Rune.Autodiff inconsistency")
    | None ->
        let zero_grad = T.zeros_like tensor_val in
        let new_id = fresh_twg_id () in
        let new_twg = { v = tensor_val; bv = zero_grad; id = new_id } in
        Hashtbl.add tape_by_twg_id new_id (Any_t_with_grad new_twg);
        Hashtbl.add val_to_twg_id_map (Obj.repr tensor_val) new_id;
        new_twg
  in

  let effc : type a. a Effect.t -> ((a, _) continuation -> _) option = function
    | E_buffer { context = effect_ctx; dtype = dt; size_in_elements } ->
        Some
          (fun k_continue ->
            let result_val = op_buffer effect_ctx dt size_in_elements in
            let _twg_res = get_or_init_twg result_val in
            continue k_continue result_val)
    | E_const_scalar { context = effect_ctx; value; dtype = dt } ->
        Some
          (fun k_continue ->
            let result_val = op_const_scalar effect_ctx value dt in
            let _twg_res = get_or_init_twg result_val in
            continue k_continue result_val)
    | E_add { a = op1_val; b = op2_val } ->
        Some
          (fun k_continue ->
            let result_val = T.add op1_val op2_val in
            let twg_op1 = get_or_init_twg op1_val in
            let twg_op2 = get_or_init_twg op2_val in
            let twg_res = get_or_init_twg result_val in
            let original_forward_val = continue k_continue result_val in
            let d_loss_d_result = grad_of twg_res in
            twg_op1.bv <- T.add twg_op1.bv d_loss_d_result;
            twg_op2.bv <- T.add twg_op2.bv d_loss_d_result;
            original_forward_val)
    | E_mul { a = op1_val; b = op2_val } ->
        Some
          (fun k_continue ->
            let result_val = op_mul op1_val op2_val in
            let twg_op1 = get_or_init_twg op1_val in
            let twg_op2 = get_or_init_twg op2_val in
            let twg_res = get_or_init_twg result_val in
            let original_forward_val = continue k_continue result_val in
            let d_loss_d_result = grad_of twg_res in
            let grad_contrib_to_op1 = op_mul d_loss_d_result op2_val in
            twg_op1.bv <- op_add twg_op1.bv grad_contrib_to_op1;
            let grad_contrib_to_op2 = op_mul d_loss_d_result op1_val in
            twg_op2.bv <- op_add twg_op2.bv grad_contrib_to_op2;
            original_forward_val)
    | E_reshape { t_in = t_in_val; new_shape } ->
        Some
          (fun k_continue ->
            let result_val = op_reshape t_in_val new_shape in
            let twg_in = get_or_init_twg t_in_val in
            let twg_res = get_or_init_twg result_val in
            let original_forward_val = continue k_continue result_val in
            let d_loss_d_result = grad_of twg_res in
            let original_shape_in = T.shape (value_of twg_in) in
            let grad_contrib_in = T.reshape original_shape_in d_loss_d_result in
            twg_in.bv <- T.add twg_in.bv grad_contrib_in;
            original_forward_val)
    | E_expand { t_in = t_in_val; new_target_shape } ->
        Some
          (fun k_continue ->
            let result_val = op_expand t_in_val new_target_shape in
            let twg_in = get_or_init_twg t_in_val in
            let twg_res = get_or_init_twg result_val in
            let original_forward_val = continue k_continue result_val in
            let d_loss_d_expanded_result = grad_of twg_res in
            let grad_contrib_to_original_input =
              let original_input_shape = T.shape (value_of twg_in) in
              let expanded_output_shape = new_target_shape in
              if original_input_shape = expanded_output_shape then
                d_loss_d_expanded_result
              else
                let rank_orig_in = Array.length original_input_shape in
                let rank_expanded_out = Array.length expanded_output_shape in
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
            twg_in.bv <- T.add twg_in.bv grad_contrib_to_original_input;
            original_forward_val)
    | E_neg { t_in = t_in_val } ->
        Some
          (fun k_continue ->
            let result_val = op_neg t_in_val in
            let twg_in = get_or_init_twg t_in_val in
            let twg_res = get_or_init_twg result_val in
            let original_forward_val = continue k_continue result_val in
            let d_loss_d_result = grad_of twg_res in
            let d_result_d_input = deriv_neg (value_of twg_in) in
            let grad_contrib_to_input =
              T.mul d_loss_d_result d_result_d_input
            in
            twg_in.bv <- T.add twg_in.bv grad_contrib_to_input;
            original_forward_val)
    | E_log2 { t_in = t_in_val } ->
        Some
          (fun k_continue ->
            let result_val = op_log2 t_in_val in
            let twg_in = get_or_init_twg t_in_val in
            let twg_res = get_or_init_twg result_val in
            let original_forward_val = continue k_continue result_val in
            let d_loss_d_result = grad_of twg_res in
            let d_result_d_input = deriv_log2 (value_of twg_in) in
            let grad_contrib_to_input =
              T.mul d_loss_d_result d_result_d_input
            in
            twg_in.bv <- T.add twg_in.bv grad_contrib_to_input;
            original_forward_val)
    | E_exp2 { t_in = t_in_val } ->
        Some
          (fun k_continue ->
            let result_val = op_exp2 t_in_val in
            let twg_in = get_or_init_twg t_in_val in
            let twg_res = get_or_init_twg result_val in
            let original_forward_val = continue k_continue result_val in
            let d_loss_d_result = grad_of twg_res in
            let d_result_d_input = deriv_exp2 result_val (value_of twg_in) in
            let grad_contrib_to_input =
              T.mul d_loss_d_result d_result_d_input
            in
            twg_in.bv <- T.add twg_in.bv grad_contrib_to_input;
            original_forward_val)
    | E_sin { t_in = t_in_val } ->
        Some
          (fun k_continue ->
            let result_val = op_sin t_in_val in
            let twg_in = get_or_init_twg t_in_val in
            let twg_res = get_or_init_twg result_val in
            let original_forward_val = continue k_continue result_val in
            let d_loss_d_result = grad_of twg_res in
            let d_result_d_input = deriv_sin (value_of twg_in) in
            let grad_contrib_to_input =
              T.mul d_loss_d_result d_result_d_input
            in
            twg_in.bv <- T.add twg_in.bv grad_contrib_to_input;
            original_forward_val)
    | E_sqrt { t_in = t_in_val } ->
        Some
          (fun k_continue ->
            let result_val = T.sqrt t_in_val in
            let twg_in = get_or_init_twg t_in_val in
            let twg_res = get_or_init_twg result_val in
            let original_forward_val = continue k_continue result_val in
            let d_loss_d_result = grad_of twg_res in
            let d_result_d_input = deriv_sqrt result_val (value_of twg_in) in
            let grad_contrib_to_input =
              T.mul d_loss_d_result d_result_d_input
            in
            twg_in.bv <- T.add twg_in.bv grad_contrib_to_input;
            original_forward_val)
    | E_recip { t_in = t_in_val } ->
        Some
          (fun k_continue ->
            let result_val = op_recip t_in_val in
            let twg_in = get_or_init_twg t_in_val in
            let twg_res = get_or_init_twg result_val in
            let original_forward_val = continue k_continue result_val in
            let d_loss_d_result = grad_of twg_res in
            let d_result_d_input = deriv_recip (value_of twg_in) in
            let grad_contrib_to_input =
              T.mul d_loss_d_result d_result_d_input
            in
            twg_in.bv <- T.add twg_in.bv grad_contrib_to_input;
            original_forward_val)
    | E_fdiv { a = op1_val; b = op2_val } ->
        Some
          (fun k_continue ->
            let result_val = op_fdiv op1_val op2_val in
            let twg_op1 = get_or_init_twg op1_val in
            let twg_op2 = get_or_init_twg op2_val in
            let twg_res = get_or_init_twg result_val in
            let original_forward_val = continue k_continue result_val in
            let d_loss_d_result = grad_of twg_res in
            let d_result_d_op1 =
              deriv_fdiv_wrt_op1 (value_of twg_op1) (value_of twg_op2)
            in
            let grad_contrib_to_op1 = T.mul d_loss_d_result d_result_d_op1 in
            twg_op1.bv <- T.add twg_op1.bv grad_contrib_to_op1;
            let d_result_d_op2 =
              deriv_fdiv_wrt_op2 (value_of twg_op1) (value_of twg_op2)
            in
            let grad_contrib_to_op2 = T.mul d_loss_d_result d_result_d_op2 in
            twg_op2.bv <- T.add twg_op2.bv grad_contrib_to_op2;
            original_forward_val)
    | E_pow { a = op1_val; b = op2_val } ->
        Some
          (fun k_continue ->
            let result_val = op_pow op1_val op2_val in
            let twg_op1 = get_or_init_twg op1_val in
            let twg_op2 = get_or_init_twg op2_val in
            let twg_res = get_or_init_twg result_val in
            let original_forward_val = continue k_continue result_val in
            let d_loss_d_result = grad_of twg_res in
            let d_result_d_op1 =
              deriv_pow_wrt_op1 (value_of twg_op1) (value_of twg_op2)
            in
            let grad_contrib_to_op1 = T.mul d_loss_d_result d_result_d_op1 in
            twg_op1.bv <- T.add twg_op1.bv grad_contrib_to_op1;
            (match dtype (value_of twg_op1) with
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
            original_forward_val)
    | E_max { a = op1_val; b = op2_val } ->
        Some
          (fun k_continue ->
            let result_val = op_max op1_val op2_val in
            let twg_op1 = get_or_init_twg op1_val in
            let twg_op2 = get_or_init_twg op2_val in
            let twg_res = get_or_init_twg result_val in
            let original_forward_val = continue k_continue result_val in
            let d_loss_d_result = grad_of twg_res in
            let val_op1 = value_of twg_op1 in
            let val_op2 = value_of twg_op2 in
            let d_result_d_op1 =
              deriv_max_wrt_op1 val_op1 val_op2 (dtype val_op1)
            in
            let grad_contrib_to_op1 = T.mul d_loss_d_result d_result_d_op1 in
            twg_op1.bv <- T.add twg_op1.bv grad_contrib_to_op1;
            let d_result_d_op2 =
              deriv_max_wrt_op2 val_op1 val_op2 (dtype val_op2)
            in
            let grad_contrib_to_op2 = T.mul d_loss_d_result d_result_d_op2 in
            twg_op2.bv <- T.add twg_op2.bv grad_contrib_to_op2;
            original_forward_val)
    | E_idiv { a = op1_val; b = op2_val } ->
        Some
          (fun k_continue ->
            let result_val = op_idiv op1_val op2_val in
            let _twg_op1 = get_or_init_twg op1_val in
            let _twg_op2 = get_or_init_twg op2_val in
            let _twg_res = get_or_init_twg result_val in
            continue k_continue result_val)
    | E_mod { a = op1_val; b = op2_val } ->
        Some
          (fun k_continue ->
            let result_val = T.mod_ op1_val op2_val in
            let _twg_op1 = get_or_init_twg op1_val in
            let _twg_op2 = get_or_init_twg op2_val in
            let _twg_res = get_or_init_twg result_val in
            continue k_continue result_val)
    | E_cmplt { a = op1_val; b = op2_val } ->
        Some
          (fun k_continue ->
            let result_val = op_cmplt op1_val op2_val in
            let _twg_op1 = get_or_init_twg op1_val in
            let _twg_op2 = get_or_init_twg op2_val in
            let _twg_res = get_or_init_twg result_val in
            continue k_continue result_val)
    | E_cmpne { a = op1_val; b = op2_val } ->
        Some
          (fun k_continue ->
            let result_val = op_cmpne op1_val op2_val in
            let _twg_op1 = get_or_init_twg op1_val in
            let _twg_op2 = get_or_init_twg op2_val in
            let _twg_res = get_or_init_twg result_val in
            continue k_continue result_val)
    | E_xor { a = op1_val; b = op2_val } ->
        Some
          (fun k_continue ->
            let result_val = op_xor op1_val op2_val in
            let _twg_op1 = get_or_init_twg op1_val in
            let _twg_op2 = get_or_init_twg op2_val in
            let _twg_res = get_or_init_twg result_val in
            continue k_continue result_val)
    | E_or { a = op1_val; b = op2_val } ->
        Some
          (fun k_continue ->
            let result_val = op_or op1_val op2_val in
            let _twg_op1 = get_or_init_twg op1_val in
            let _twg_op2 = get_or_init_twg op2_val in
            let _twg_res = get_or_init_twg result_val in
            continue k_continue result_val)
    | E_and { a = op1_val; b = op2_val } ->
        Some
          (fun k_continue ->
            let result_val = op_and op1_val op2_val in
            let _twg_op1 = get_or_init_twg op1_val in
            let _twg_op2 = get_or_init_twg op2_val in
            let _twg_res = get_or_init_twg result_val in
            continue k_continue result_val)
    | E_where { condition = cond_val; if_true = true_val; if_false = false_val }
      ->
        Some
          (fun k_continue ->
            let result_val = op_where cond_val true_val false_val in
            let _twg_cond = get_or_init_twg cond_val in
            let twg_true = get_or_init_twg true_val in
            let twg_false = get_or_init_twg false_val in
            let twg_res = get_or_init_twg result_val in
            let original_forward_val = continue k_continue result_val in
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
            twg_false.bv <- T.add twg_false.bv grad_contrib_to_false;
            original_forward_val)
    | E_reduce_sum { t_in = t_in_val; axes; keepdims } ->
        Some
          (fun k_continue ->
            let result_val = op_reduce_sum ~axes ~keepdims t_in_val in
            let twg_in = get_or_init_twg t_in_val in
            let twg_res = get_or_init_twg result_val in
            let original_forward_val = continue k_continue result_val in
            let d_loss_d_result = grad_of twg_res in
            let original_input_shape = T.shape (value_of twg_in) in
            let grad_prepared_for_broadcast =
              prepare_grad_for_broadcast d_loss_d_result (value_of twg_in) axes
                keepdims (fun t ~axes ~keepdims -> T.sum t ~axes ~keepdims)
            in
            let grad_contrib_to_input =
              T.broadcast_to original_input_shape grad_prepared_for_broadcast
            in
            twg_in.bv <- T.add twg_in.bv grad_contrib_to_input;
            original_forward_val)
    | E_reduce_max { t_in = t_in_val; axes; keepdims } ->
        Some
          (fun k_continue ->
            let result_val = op_reduce_max ~axes ~keepdims t_in_val in
            let twg_in = get_or_init_twg t_in_val in
            let twg_res = get_or_init_twg result_val in
            let original_forward_val = continue k_continue result_val in
            let d_loss_d_result = grad_of twg_res in
            let val_in = value_of twg_in in
            let original_input_shape = T.shape val_in in
            let grad_prepared_for_broadcast =
              prepare_grad_for_broadcast d_loss_d_result val_in axes keepdims
                (fun t ~axes ~keepdims -> T.max t ~axes ~keepdims)
            in
            let d_loss_d_result_broadcasted =
              T.broadcast_to original_input_shape grad_prepared_for_broadcast
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
            twg_in.bv <- T.add twg_in.bv grad_contrib_to_input;
            original_forward_val)
    | E_reduce_prod { t_in = t_in_val; axes; keepdims } ->
        Some
          (fun k_continue ->
            let result_val = op_reduce_prod ~axes ~keepdims t_in_val in
            let twg_in = get_or_init_twg t_in_val in
            let twg_res = get_or_init_twg result_val in
            let original_forward_val = continue k_continue result_val in
            let d_loss_d_result = grad_of twg_res in
            let val_in = value_of twg_in in
            let original_input_shape = T.shape val_in in
            let grad_prepared_for_broadcast =
              prepare_grad_for_broadcast d_loss_d_result val_in axes keepdims
                (fun t ~axes ~keepdims -> T.prod t ~axes ~keepdims)
            in
            let d_loss_d_result_broadcasted =
              T.broadcast_to original_input_shape grad_prepared_for_broadcast
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
            twg_in.bv <- T.add twg_in.bv grad_contrib_to_input;
            original_forward_val)
    | E_permute { t_in = t_in_val; axes = permute_axes } ->
        Some
          (fun k_continue ->
            let result_val = op_permute t_in_val permute_axes in
            let twg_in = get_or_init_twg t_in_val in
            let twg_res = get_or_init_twg result_val in
            let original_forward_val = continue k_continue result_val in
            let d_loss_d_result = grad_of twg_res in
            let rank = Array.length permute_axes in
            let un_permute_axes = Array.make rank 0 in
            Array.iteri
              (fun i original_pos -> un_permute_axes.(original_pos) <- i)
              permute_axes;
            let grad_contrib_to_input =
              T.transpose d_loss_d_result ~axes:un_permute_axes
            in
            twg_in.bv <- T.add twg_in.bv grad_contrib_to_input;
            original_forward_val)
    | E_pad { t_in = t_in_val; padding_config; fill_value } ->
        Some
          (fun k_continue ->
            let result_val = op_pad t_in_val padding_config fill_value in
            let twg_in = get_or_init_twg t_in_val in
            let twg_res = get_or_init_twg result_val in
            let original_forward_val = continue k_continue result_val in
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
            twg_in.bv <- T.add twg_in.bv grad_contrib_to_input;
            original_forward_val)
    | E_shrink { t_in = t_in_val; limits = shrink_limits } ->
        Some
          (fun k_continue ->
            let result_val = op_shrink t_in_val shrink_limits in
            let twg_in = get_or_init_twg t_in_val in
            let twg_res = get_or_init_twg result_val in
            let original_forward_val = continue k_continue result_val in
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
            twg_in.bv <- T.add twg_in.bv grad_contrib_to_input;
            original_forward_val)
    | E_flip { t_in = t_in_val; dims_to_flip } ->
        Some
          (fun k_continue ->
            let axes_to_flip =
              dims_to_flip |> Array.to_list
              |> List.mapi (fun i flip -> if flip then Some i else None)
              |> List.filter_map Fun.id |> Array.of_list
            in
            let result_val = op_flip t_in_val dims_to_flip in
            let twg_in = get_or_init_twg t_in_val in
            let twg_res = get_or_init_twg result_val in
            let original_forward_val = continue k_continue result_val in
            let d_loss_d_result = grad_of twg_res in
            let grad_contrib_to_input =
              T.flip d_loss_d_result ~axes:axes_to_flip
            in
            twg_in.bv <- T.add twg_in.bv grad_contrib_to_input;
            original_forward_val)
    | E_cat { t_list; axis } ->
        Some
          (fun k_continue ->
            let result_val = op_cat t_list axis in
            let twg_inputs = List.map get_or_init_twg t_list in
            let twg_res = get_or_init_twg result_val in
            let original_forward_val = continue k_continue result_val in
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
              twg_inputs;
            original_forward_val)
    | E_cast { t_in = t_in_val; target_dtype } ->
        Some
          (fun k_continue ->
            let result_val = op_cast t_in_val target_dtype in
            let twg_in = get_or_init_twg t_in_val in
            let twg_res = get_or_init_twg result_val in
            let original_forward_val = continue k_continue result_val in
            let d_loss_d_result = grad_of twg_res in
            let original_dtype = dtype (value_of twg_in) in
            let grad_contrib_to_input = T.cast original_dtype d_loss_d_result in
            twg_in.bv <- T.add twg_in.bv grad_contrib_to_input;
            original_forward_val)
    | E_contiguous { t_in = t_in_val } ->
        Some
          (fun k_continue ->
            let result_val = op_contiguous t_in_val in
            let twg_in = get_or_init_twg t_in_val in
            let twg_res = get_or_init_twg result_val in
            let original_forward_val = continue k_continue result_val in
            let d_loss_d_result = grad_of twg_res in
            twg_in.bv <- T.add twg_in.bv d_loss_d_result;
            original_forward_val)
    | E_copy { t_in = t_in_val } ->
        Some
          (fun k_continue ->
            let result_val = op_copy t_in_val in
            let twg_in = get_or_init_twg t_in_val in
            let _twg_res = get_or_init_twg result_val in
            let original_forward_val = continue k_continue result_val in
            let d_loss_d_result = grad_of (get_or_init_twg result_val) in
            twg_in.bv <- T.add twg_in.bv d_loss_d_result;
            original_forward_val)
    | E_const_array { context = effect_ctx; array } ->
        Some
          (fun k_continue ->
            let result_val = op_const_array effect_ctx array in
            let _twg_res = get_or_init_twg result_val in
            continue k_continue result_val)
    | E_threefry { key = key_val; ctr = ctr_val } ->
        Some
          (fun k_continue ->
            let result_val = op_threefry key_val ctr_val in
            let _twg_key = get_or_init_twg key_val in
            let _twg_ctr = get_or_init_twg ctr_val in
            let _twg_res = get_or_init_twg result_val in
            continue k_continue result_val)
    | E_gather { data = data_val; indices = indices_val; axis } ->
        Some
          (fun k_continue ->
            let result_val = op_gather data_val indices_val axis in
            let twg_data = get_or_init_twg data_val in
            let _twg_indices = get_or_init_twg indices_val in
            let twg_res = get_or_init_twg result_val in
            let original_forward_val = continue k_continue result_val in
            let d_loss_d_result = grad_of twg_res in
            let zeros_data = T.zeros_like (value_of twg_data) in
            let scattered_grads =
              op_scatter zeros_data indices_val d_loss_d_result axis
            in
            twg_data.bv <- T.add twg_data.bv scattered_grads;
            original_forward_val)
    | E_scatter
        { data_template = dt_val; indices = idx_val; updates = upd_val; axis }
      ->
        Some
          (fun k_continue ->
            let result_val = op_scatter dt_val idx_val upd_val axis in
            let twg_dt = get_or_init_twg dt_val in
            let _twg_idx = get_or_init_twg idx_val in
            let twg_upd = get_or_init_twg upd_val in
            let twg_res = get_or_init_twg result_val in
            let original_forward_val = continue k_continue result_val in
            let d_loss_d_result = grad_of twg_res in
            let grad_contrib_to_updates =
              op_gather d_loss_d_result idx_val axis
            in
            twg_upd.bv <- T.add twg_upd.bv grad_contrib_to_updates;
            let mask_for_dt_grad =
              op_scatter (T.ones_like dt_val) idx_val (T.zeros_like upd_val)
                axis
            in
            let grad_contrib_to_dt = T.mul d_loss_d_result mask_for_dt_grad in
            twg_dt.bv <- T.add twg_dt.bv grad_contrib_to_dt;
            original_forward_val)
    | E_assign { dst = dst_val; src = src_val } ->
        Some
          (fun k_continue ->
            let old_dst_val = T.copy dst_val in
            op_assign dst_val src_val;
            let twg_dst = get_or_init_twg dst_val in
            let twg_src = get_or_init_twg src_val in
            let _twg_old_dst = get_or_init_twg old_dst_val in
            let original_forward_val = continue k_continue () in
            twg_src.bv <- T.add twg_src.bv (grad_of twg_dst);
            original_forward_val)
    | _ -> None
  in

  {
    retc =
      (fun final_result_val ->
        let twg_final_result = get_or_init_twg final_result_val in
        let _ =
          is_uninitialized_grad (grad_of twg_final_result) final_result_val
        in
        twg_final_result.bv <- T.ones_like final_result_val;
        final_result_val);
    exnc = raise;
    effc;
  }

(* --- User-facing grad functions --- *)

let grad (f : ('a, 'b) t -> ('c, 'd) t) (input_val : ('a, 'b) t) : ('a, 'b) t =
  let tape_by_twg_id : (int, any_t_with_grad) Hashtbl.t = Hashtbl.create 16 in
  let val_to_twg_id_map : (Obj.t, int) Hashtbl.t = Hashtbl.create 16 in
  let initial_grad_for_input = T.zeros_like input_val in
  let twg_input_id = fresh_twg_id () in
  let twg_input =
    { v = input_val; bv = initial_grad_for_input; id = twg_input_id }
  in
  Hashtbl.add tape_by_twg_id twg_input_id (Any_t_with_grad twg_input);
  Hashtbl.add val_to_twg_id_map (Obj.repr input_val) twg_input_id;
  let ad_handler = make_reverse_handler tape_by_twg_id val_to_twg_id_map in
  let _result_value_from_f = Effect.Deep.match_with f input_val ad_handler in
  let final_twg_input_id =
    Hashtbl.find val_to_twg_id_map (Obj.repr input_val)
  in
  let final_twg_input_any = Hashtbl.find tape_by_twg_id final_twg_input_id in
  let final_twg_input = unwrap_twg (dtype input_val) final_twg_input_any in
  final_twg_input.bv

let value_and_grad (f : ('a, 'b) t -> ('c, 'd) t) (input_val : ('a, 'b) t) :
    ('c, 'd) t * ('a, 'b) t =
  let tape_by_twg_id : (int, any_t_with_grad) Hashtbl.t = Hashtbl.create 16 in
  let val_to_twg_id_map : (Obj.t, int) Hashtbl.t = Hashtbl.create 16 in
  let initial_grad_for_input = T.zeros_like input_val in
  let twg_input_id = fresh_twg_id () in
  let twg_input =
    { v = input_val; bv = initial_grad_for_input; id = twg_input_id }
  in
  Hashtbl.add tape_by_twg_id twg_input_id (Any_t_with_grad twg_input);
  Hashtbl.add val_to_twg_id_map (Obj.repr input_val) twg_input_id;
  let ad_handler = make_reverse_handler tape_by_twg_id val_to_twg_id_map in
  let result_value_from_f = Effect.Deep.match_with f input_val ad_handler in
  let final_twg_input_id =
    Hashtbl.find val_to_twg_id_map (Obj.repr input_val)
  in
  let final_twg_input_any = Hashtbl.find tape_by_twg_id final_twg_input_id in
  let final_twg_input = unwrap_twg (dtype input_val) final_twg_input_any in
  (result_value_from_f, final_twg_input.bv)
