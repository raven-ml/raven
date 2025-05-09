open Nx_core
open Nx_rune (* Provides E_... effects, t, and context *)
module T = Tensor (* Frontend: Nx_core.Make_frontend(Nx_rune) *)

(* Global ID generator for t_with_grad instances *)
let next_twg_id_counter = ref 0

let fresh_twg_id () =
  incr next_twg_id_counter;
  !next_twg_id_counter

(* Type to store a tensor's forward value and its accumulated gradient *)
type ('a, 'b) t_with_grad = {
  v : ('a, 'b) t; (* The forward value *)
  mutable bv : ('a, 'b) t; (* The backward value (gradient) *)
  id : int; (* Unique identifier for this gradient tracking entry *)
}

(* Existential type to store any t_with_grad in the tape *)
type any_t_with_grad =
  | Any_t_with_grad : ('a, 'b) t_with_grad -> any_t_with_grad

(* Helper to get the underlying t from t_with_grad *)
let value_of (type a b) (twg : (a, b) t_with_grad) : (a, b) t = twg.v
let grad_of (type a b) (twg : (a, b) t_with_grad) : (a, b) t = twg.bv

(* Unwrapper for any_twg. Type safety relies on careful usage. *)
let unwrap_twg (type a b) (_dtype : (a, b) Dtype.t) (any : any_t_with_grad) :
    (a, b) t_with_grad =
  match any with Any_t_with_grad m -> Obj.magic m

(* --- Derivative definitions for UOps --- *)
(* These functions compute d(Op_Output)/d(Input_N). They return a regular t.
   The shape of the derivative tensor is often related to the shape of Input_N.
*)

(* d(add(op1, op2))/d(op1) = 1 (broadcasted to op1's original contribution
   shape) *)
let deriv_op_add_wrt_op1 (ctx : context) (op1_val : ('a, 'b) t)
    (_op2_val : ('c, 'd) t) : ('a, 'b) t =
  T.ones_like ctx op1_val

(* d(add(op1, op2))/d(op2) = 1 (broadcasted to op2's original contribution
   shape) *)
let deriv_op_add_wrt_op2 (ctx : context) (_op1_val : ('a, 'b) t)
    (op2_val : ('c, 'd) t) : ('c, 'd) t =
  T.ones_like ctx op2_val

(* --- Gradient manipulation helpers --- *)

(* Stub for a helper that might be in T eventually, or use a more robust
   check *)
let is_uninitialized_grad (ctx : context) grad_tensor input_tensor =
  (* A simple heuristic: if it's exactly the same object as the initial
     zeros_like. This is fragile and mainly for the retc warning. A more robust
     system might track initialization. *)
  try grad_tensor == T.zeros_like ctx input_tensor with _ -> false

(* The main reverse-mode AD effect handler *)
let make_reverse_handler (grad_ctx : context)
    (tape_by_twg_id : (int, any_t_with_grad) Hashtbl.t)
    (val_to_twg_id_map : (Obj.t, int) Hashtbl.t) =
  let open Effect.Deep in
  (* Helper to get a twg from tape for a given t value. If not found, it means
     this tensor is a constant w.r.t. the differentiated inputs, or it's a new
     tensor resulting from an op. It creates a twg with zero gradient and adds
     to tape. This uses grad_ctx for AD bookkeeping. *)
  let get_or_init_twg (type a b) (tensor_val : (a, b) t) : (a, b) t_with_grad =
    match Hashtbl.find_opt val_to_twg_id_map (Obj.repr tensor_val) with
    | Some twg_id -> (
        match Hashtbl.find_opt tape_by_twg_id twg_id with
        | Some any_twg -> unwrap_twg tensor_val.dtype any_twg
        | None ->
            failwith
              "Rune.Autodiff inconsistency: ID found in val_to_twg_id_map but \
               not in tape_by_twg_id")
    | None ->
        let zero_grad = T.zeros_like grad_ctx tensor_val in
        let new_id = fresh_twg_id () in
        let new_twg = { v = tensor_val; bv = zero_grad; id = new_id } in
        Hashtbl.add tape_by_twg_id new_id (Any_t_with_grad new_twg);
        Hashtbl.add val_to_twg_id_map (Obj.repr tensor_val) new_id;
        new_twg
  in

  let effc : type a. a Effect.t -> ((a, _) continuation -> _) option =
   fun e ->
    match e with
    | E_buffer { context = effect_ctx; dtype = dt; size_in_elements } ->
        Some
          (fun k_continue ->
            let result_val = op_buffer effect_ctx dt size_in_elements in
            let _twg_res = get_or_init_twg result_val in
            (* Leaf node, grad will remain zero unless it's the final output *)
            continue k_continue result_val)
    | E_const_scalar { context = effect_ctx; value; dtype = dt } ->
        Some
          (fun k_continue ->
            let result_val = op_const_scalar effect_ctx value dt in
            let _twg_res = get_or_init_twg result_val in
            (* Leaf node *)
            continue k_continue result_val)
    | E_add { context = effect_ctx; a = op1_val; b = op2_val } ->
        Some
          (fun k_continue ->
            (* 1. Perform the forward computation. op1_val and op2_val are
               assumed to be broadcast-compatible here, as Nx_rune.op_add is
               low-level. The T.add call will trigger Nx_rune.op_add, which
               raises this E_add. If op1_val or op2_val were results of previous
               E_expand operations handled by AD, their `twg` entries are
               already set up. *)
            let result_val = T.add effect_ctx op1_val op2_val in

            (* 2. Get/initialize twg for inputs and result *)
            let twg_op1 = get_or_init_twg op1_val in
            (* op1_val is the (potentially expanded) first input *)
            let twg_op2 = get_or_init_twg op2_val in
            (* op2_val is the (potentially expanded) second input *)
            let twg_res = get_or_init_twg result_val in

            (* 3. Call the original continuation *)
            let original_forward_val_from_continuation =
              continue k_continue result_val
            in

            (* 4. Gradient accumulation *)
            let d_loss_d_result = grad_of twg_res in

            (* For element-wise add: d(Res)/d(op1) = 1, d(Res)/d(op2) = 1. So,
               d(Loss)/d(op1) = d(Loss)/d(Result) * 1 d(Loss)/d(op2) =
               d(Loss)/d(Result) * 1 The shapes match because op1_val, op2_val,
               and result_val all have the same (broadcasted) shape at this
               point of element-wise op_add. *)
            twg_op1.bv <- T.add effect_ctx twg_op1.bv d_loss_d_result;
            twg_op2.bv <- T.add effect_ctx twg_op2.bv d_loss_d_result;

            original_forward_val_from_continuation)
    | E_reshape { context = effect_ctx; t_in = t_in_val; new_shape } ->
        Some
          (fun k_continue ->
            let result_val = T.reshape effect_ctx t_in_val new_shape in

            let twg_in = get_or_init_twg t_in_val in
            let twg_res = get_or_init_twg result_val in

            let original_forward_val = continue k_continue result_val in

            let d_loss_d_result = grad_of twg_res in
            let original_shape_in = T.shape (value_of twg_in) in
            let grad_contrib_in =
              T.reshape effect_ctx d_loss_d_result original_shape_in
            in
            twg_in.bv <- T.add effect_ctx twg_in.bv grad_contrib_in;
            original_forward_val)
    | E_expand { context = effect_ctx; t_in = t_in_val; new_target_shape } ->
        Some
          (fun k_continue ->
            (* 1. Perform the forward computation for E_expand. 'result_val' is
               the expanded tensor. 't_in_val' is the original tensor. *)
            let result_val = T.expand effect_ctx t_in_val new_target_shape in
            (* This calls Nx_rune.op_expand *)

            (* 2. Get/initialize twg for the original input and the expanded
               result *)
            let twg_in = get_or_init_twg t_in_val in
            let twg_res = get_or_init_twg result_val in
            (* result_val is the EXPANDED tensor *)

            (* 3. Call the original continuation to proceed with forward
               execution using the expanded tensor *)
            let original_forward_val_from_continuation =
              continue k_continue result_val
            in

            (* 4. Gradient accumulation (occurs during effect unwinding) *)
            let d_loss_d_expanded_result = grad_of twg_res in
            (* Gradient of the expanded tensor *)

            (* 5. Compute d(Loss)/d(t_in_val) by summing
               d_loss_d_expanded_result. The shape of t_in_val is T.shape
               (value_of twg_in). The shape of result_val (expanded) is T.shape
               (value_of twg_res) or new_target_shape. This function now
               internalizes the logic of reduce_grad_to_input_shape. *)
            let grad_contrib_to_original_input =
              let original_input_shape = T.shape (value_of twg_in) in
              let expanded_output_shape = new_target_shape in
              (* or T.shape result_val *)

              if original_input_shape = expanded_output_shape then
                d_loss_d_expanded_result
                (* No expansion happened, or expanded to same shape *)
              else
                let rank_orig_in = Array.length original_input_shape in
                let rank_expanded_out = Array.length expanded_output_shape in
                let axes_to_sum_list = ref [] in

                (* Identify leading dimensions in expanded_out that are not in
                   orig_in (rank difference) *)
                if rank_expanded_out > rank_orig_in then
                  for i = 0 to rank_expanded_out - rank_orig_in - 1 do
                    axes_to_sum_list := i :: !axes_to_sum_list
                  done;

                (* Identify dimensions that were size 1 in orig_in but >1 in
                   expanded_out *)
                for i = 0 to rank_orig_in - 1 do
                  let orig_in_dim_size = original_input_shape.(i) in
                  (* Align indices for comparison with expanded_output_shape *)
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
                    (* Use effect_ctx for the sum operation *)
                    T.sum effect_ctx d_loss_d_expanded_result
                      ~axes:(Array.of_list (List.rev !axes_to_sum_list))
                        (* Ensure correct order if T.sum cares *)
                      ~keepdims:true
                  else d_loss_d_expanded_result
                in
                (* Reshape to remove summed dimensions (if keepdims=true was
                   used) and match original_input_shape *)
                if T.shape summed_grad <> original_input_shape then
                  T.reshape effect_ctx summed_grad original_input_shape
                else summed_grad
            in
            twg_in.bv <-
              T.add effect_ctx twg_in.bv grad_contrib_to_original_input;
            original_forward_val_from_continuation)
    (* E_load, E_store, E_define_global, E_range, E_special gradients are not
       handled yet. They will pass through to be handled by JIT or eager
       execution. AD for them is more complex. *)
    | E_load _ | E_store _ | E_define_global _ | E_range _ | E_special _ -> None
    | _ -> None (* Unknown effect, pass to other handlers *)
  in

  (* retc: Called when the function `f` (wrapped by this AD handler) completes.
     `final_result_val` is the tensor returned by `f`. We set
     d(Loss)/d(final_result_val) to 1, assuming Loss = final_result_val. Uses
     grad_ctx for AD bookkeeping. *)
  {
    retc =
      (fun (final_result_val : ('final_a, 'final_b) t) ->
        let twg_final_result = get_or_init_twg final_result_val in
        if
          not
            (is_uninitialized_grad grad_ctx (grad_of twg_final_result)
               final_result_val)
        then
          Printf.eprintf
            "Rune.Autodiff Warning: Final result tensor already had a \
             non-zero/non-initial gradient before seeding with ones. This \
             might be unexpected if the output is not also an input.\n";

        twg_final_result.bv <- T.ones_like grad_ctx final_result_val;
        final_result_val);
    exnc = raise;
    (* Standard exception handling *)
    effc;
  }

(* --- User-facing grad functions --- *)

(* Grad of a function f: tensor -> tensor *)
let grad (ctx : context) (f : ('a, 'b) t -> ('c, 'd) t) (input_val : ('a, 'b) t)
    : ('a, 'b) t =
  let tape_by_twg_id : (int, any_t_with_grad) Hashtbl.t = Hashtbl.create 16 in
  let val_to_twg_id_map : (Obj.t, int) Hashtbl.t = Hashtbl.create 16 in

  (* Wrap the primary input_val for AD and place it on tape. Its gradient starts
     at zero. *)
  let initial_grad_for_input = T.zeros_like ctx input_val in
  let twg_input_id = fresh_twg_id () in
  let twg_input =
    { v = input_val; bv = initial_grad_for_input; id = twg_input_id }
  in
  Hashtbl.add tape_by_twg_id twg_input_id (Any_t_with_grad twg_input);
  Hashtbl.add val_to_twg_id_map (Obj.repr input_val) twg_input_id;

  let ad_handler = make_reverse_handler ctx tape_by_twg_id val_to_twg_id_map in

  (* Run f under the AD handler. This populates the tape, computes forward
     values, and during unwinding, computes and accumulates gradients. *)
  let _result_value_from_f = Effect.Deep.match_with f input_val ad_handler in

  (* The gradient for `input_val` is now accumulated in `twg_input.bv`. *)
  let final_twg_input_id =
    Hashtbl.find val_to_twg_id_map (Obj.repr input_val)
  in
  (* Should always exist *)
  let final_twg_input_any = Hashtbl.find tape_by_twg_id final_twg_input_id in
  let final_twg_input = unwrap_twg input_val.dtype final_twg_input_any in
  final_twg_input.bv

(* Value_and_grad of f: tensor -> tensor *)
let value_and_grad (ctx : context) (f : ('a, 'b) t -> ('c, 'd) t)
    (input_val : ('a, 'b) t) : ('c, 'd) t * ('a, 'b) t =
  let tape_by_twg_id : (int, any_t_with_grad) Hashtbl.t = Hashtbl.create 16 in
  let val_to_twg_id_map : (Obj.t, int) Hashtbl.t = Hashtbl.create 16 in

  let initial_grad_for_input = T.zeros_like ctx input_val in
  let twg_input_id = fresh_twg_id () in
  let twg_input =
    { v = input_val; bv = initial_grad_for_input; id = twg_input_id }
  in
  Hashtbl.add tape_by_twg_id twg_input_id (Any_t_with_grad twg_input);
  Hashtbl.add val_to_twg_id_map (Obj.repr input_val) twg_input_id;

  let ad_handler = make_reverse_handler ctx tape_by_twg_id val_to_twg_id_map in
  let result_value_from_f = Effect.Deep.match_with f input_val ad_handler in

  let final_twg_input_id =
    Hashtbl.find val_to_twg_id_map (Obj.repr input_val)
  in
  let final_twg_input_any = Hashtbl.find tape_by_twg_id final_twg_input_id in
  let final_twg_input = unwrap_twg input_val.dtype final_twg_input_any in
  (result_value_from_f, final_twg_input.bv)
