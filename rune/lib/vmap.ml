open Nx_rune
module T = Tensor

(* Type to represent mapping specification for a single axis *)
type axis_spec =
  | Map of int (* Map over this axis index *)
  | NoMap (* Don't map this axis *)

(* Type to represent container mapping specifications *)
type 'a in_axes_spec = Single of axis_spec | Container of 'a

(* Type to represent output axes specification *)
type 'a out_axes_spec = OutSingle of int option | OutContainer of 'a

(* Type to store information about a vmapped tensor *)
type ('a, 'b) vmapped_tensor = {
  base : ('a, 'b) t;
  mapped_axis : int;
  batch_size : int; [@warning "-69"]
  original_shape : int array;
}

type any_vmapped_tensor =
  | Any_vmapped : ('a, 'b) vmapped_tensor -> any_vmapped_tensor

(* Custom hashtable module that uses physical equality to distinguish tensors *)
module PhysicalTbl = struct
  let create _ = ref []

  let find_opt tbl key =
    let key_repr = Obj.repr key in
    List.find_opt (fun (k, _) -> k == key_repr) !tbl |> Option.map snd

  let add tbl key value =
    let key_repr = Obj.repr key in
    tbl := (key_repr, value) :: !tbl
end

(* Helper to extract mapped axis from in_axes specification *)
let extract_axis_spec = function
  | Single spec -> spec
  | Container _ -> failwith "vmap: container in_axes not yet supported"

(* Helper to extract output axis from out_axes specification *)
let extract_out_axis_spec = function
  | OutSingle spec -> spec
  | OutContainer _ -> failwith "vmap: container out_axes not yet supported"

(* Helper to move an axis to the front or back of a tensor *)
let move_axis (tensor : ('a, 'b) t) ~from_axis ~to_axis : ('a, 'b) t =
  let shape = T.shape tensor in
  let ndim = Array.length shape in
  let from_axis = if from_axis < 0 then ndim + from_axis else from_axis in
  let to_axis = if to_axis < 0 then ndim + to_axis + 1 else to_axis in

  if from_axis = to_axis then tensor
  else
    let axes = Array.init ndim (fun i -> i) in
    (* Remove from_axis from its current position *)
    let temp_axes =
      Array.concat
        [
          Array.sub axes 0 from_axis;
          Array.sub axes (from_axis + 1) (ndim - from_axis - 1);
        ]
    in
    (* Insert at to_axis position *)
    let new_axes =
      if to_axis = 0 then Array.concat [ [| from_axis |]; temp_axes ]
      else if to_axis >= ndim then Array.concat [ temp_axes; [| from_axis |] ]
      else
        Array.concat
          [
            Array.sub temp_axes 0 to_axis;
            [| from_axis |];
            Array.sub temp_axes to_axis (Array.length temp_axes - to_axis);
          ]
    in
    T.transpose tensor ~axes:new_axes

(* Helper to add a batch dimension to a tensor *)
let add_batch_dim (tensor : ('a, 'b) t) ~axis ~size : ('a, 'b) t =
  let shape = T.shape tensor in
  let new_shape =
    Array.concat
      [
        Array.sub shape 0 axis;
        [| size |];
        Array.sub shape axis (Array.length shape - axis);
      ]
  in
  T.reshape new_shape (T.expand_dims [| axis |] tensor)

(* The main vmap effect handler *)
let make_vmap_handler mapped_tensors batch_size _in_axis out_axis =
  let open Effect.Deep in
  let get_or_check_vmapped (type a b) (tensor_val : (a, b) t) :
      (a, b) vmapped_tensor =
    match PhysicalTbl.find_opt mapped_tensors tensor_val with
    | Some (Any_vmapped vm) ->
        Obj.magic vm (* Safe because we store with correct type *)
    | None ->
        (* This tensor is not mapped, we need to broadcast it *)
        let original_shape = T.shape tensor_val in
        {
          base = tensor_val;
          mapped_axis = -1;
          (* -1 indicates broadcasting *)
          batch_size;
          original_shape;
        }
  in

  (* Helper to get the actual batched tensor from a possibly unbatched view *)
  let get_batched_tensor (type a b) (tensor_val : (a, b) t) : (a, b) t =
    let vm = get_or_check_vmapped tensor_val in
    if vm.mapped_axis >= 0 then vm.base else tensor_val
  in

  let effc : type c. c Effect.t -> ((c, _) continuation -> _) option = function
    | E_buffer { context = effect_ctx; dtype = dt; size_in_elements } ->
        Some
          (fun k ->
            let result_val = op_buffer effect_ctx dt size_in_elements in
            let forward_val = continue k result_val in
            forward_val)
    | E_const_scalar { context = effect_ctx; value; dtype = dt } ->
        Some
          (fun k ->
            let result_val = op_const_scalar effect_ctx value dt in
            let forward_val = continue k result_val in
            forward_val)
    | E_const_array { context = effect_ctx; array } ->
        Some
          (fun k ->
            let result_val = op_const_array effect_ctx array in
            let forward_val = continue k result_val in
            forward_val)
    | E_add { a = op1_val; b = op2_val } ->
        Some
          (fun k ->
            let vm1 = get_or_check_vmapped op1_val in
            let vm2 = get_or_check_vmapped op2_val in

            (* Get the actual batched tensors *)
            let batched_t1 = get_batched_tensor op1_val in
            let batched_t2 = get_batched_tensor op2_val in

            (* Handle broadcasting if one tensor is not mapped *)
            let t1, t2 =
              if vm1.mapped_axis = -1 && vm2.mapped_axis >= 0 then
                (* Broadcast tensor1 to match tensor2's batch dimension *)
                let t1_broadcast =
                  add_batch_dim vm1.base ~axis:0 ~size:batch_size
                in
                let t1_expanded =
                  T.broadcast_to (T.shape batched_t2) t1_broadcast
                in
                (t1_expanded, batched_t2)
              else if vm2.mapped_axis = -1 && vm1.mapped_axis >= 0 then
                (* Broadcast tensor2 to match tensor1's batch dimension *)
                let t2_broadcast =
                  add_batch_dim vm2.base ~axis:0 ~size:batch_size
                in
                let t2_expanded =
                  T.broadcast_to (T.shape batched_t1) t2_broadcast
                in
                (batched_t1, t2_expanded)
              else (batched_t1, batched_t2)
            in

            let result = op_add t1 t2 in
            let forward_val = continue k result in
            let vmapped_result =
              {
                base = result;
                mapped_axis = 0;
                batch_size;
                original_shape = T.shape result;
              }
            in
            PhysicalTbl.add mapped_tensors result (Any_vmapped vmapped_result);
            forward_val)
    | E_mul { a = op1_val; b = op2_val } ->
        Some
          (fun k ->
            let batched_t1 = get_batched_tensor op1_val in
            let batched_t2 = get_batched_tensor op2_val in
            let result = op_mul batched_t1 batched_t2 in
            let forward_val = continue k result in
            let vmapped_result =
              {
                base = result;
                mapped_axis = 0;
                batch_size;
                original_shape = T.shape result;
              }
            in
            PhysicalTbl.add mapped_tensors result (Any_vmapped vmapped_result);
            forward_val)
    | E_neg { t_in } ->
        Some
          (fun k ->
            let batched_tensor = get_batched_tensor t_in in
            let result = op_neg batched_tensor in
            let forward_val = continue k result in
            let vmapped_result =
              {
                base = result;
                mapped_axis = 0;
                batch_size;
                original_shape = T.shape result;
              }
            in
            PhysicalTbl.add mapped_tensors result (Any_vmapped vmapped_result);
            forward_val)
    | E_reshape { t_in; new_shape } ->
        Some
          (fun k ->
            let vm = get_or_check_vmapped t_in in
            (* Adjust shape to include batch dimension *)
            let batched_shape =
              if vm.mapped_axis >= 0 then
                Array.concat [ [| batch_size |]; new_shape ]
              else new_shape
            in
            (* Use the actual batched tensor, not the unbatched view *)
            let batched_tensor =
              if vm.mapped_axis >= 0 then vm.base else t_in
            in
            let result = op_reshape batched_tensor batched_shape in
            let forward_val = continue k result in
            let vmapped_result =
              {
                base = result;
                mapped_axis = (if vm.mapped_axis >= 0 then 0 else -1);
                batch_size;
                original_shape = new_shape;
              }
            in
            PhysicalTbl.add mapped_tensors result (Any_vmapped vmapped_result);
            forward_val)
    | E_matmul { a; b } ->
        Some
          (fun k ->
            let result = op_matmul a b in
            let forward_val = continue k result in
            let vmapped_result =
              {
                base = result;
                mapped_axis = 0;
                batch_size;
                original_shape = T.shape result;
              }
            in
            PhysicalTbl.add mapped_tensors result (Any_vmapped vmapped_result);
            forward_val)
    | E_fdiv { a = op1_val; b = op2_val } ->
        Some
          (fun k ->
            let result = op_fdiv op1_val op2_val in
            let forward_val = continue k result in
            let vmapped_result =
              {
                base = result;
                mapped_axis = 0;
                batch_size;
                original_shape = T.shape result;
              }
            in
            PhysicalTbl.add mapped_tensors result (Any_vmapped vmapped_result);
            forward_val)
    | E_pow { a = op1_val; b = op2_val } ->
        Some
          (fun k ->
            let result = op_pow op1_val op2_val in
            let forward_val = continue k result in
            let vmapped_result =
              {
                base = result;
                mapped_axis = 0;
                batch_size;
                original_shape = T.shape result;
              }
            in
            PhysicalTbl.add mapped_tensors result (Any_vmapped vmapped_result);
            forward_val)
    | E_max { a = op1_val; b = op2_val } ->
        Some
          (fun k ->
            let result = op_max op1_val op2_val in
            let forward_val = continue k result in
            let vmapped_result =
              {
                base = result;
                mapped_axis = 0;
                batch_size;
                original_shape = T.shape result;
              }
            in
            PhysicalTbl.add mapped_tensors result (Any_vmapped vmapped_result);
            forward_val)
    | E_log2 { t_in } ->
        Some
          (fun k ->
            let result = op_log2 t_in in
            let forward_val = continue k result in
            let vmapped_result =
              {
                base = result;
                mapped_axis = 0;
                batch_size;
                original_shape = T.shape result;
              }
            in
            PhysicalTbl.add mapped_tensors result (Any_vmapped vmapped_result);
            forward_val)
    | E_exp2 { t_in } ->
        Some
          (fun k ->
            let result = op_exp2 t_in in
            let forward_val = continue k result in
            let vmapped_result =
              {
                base = result;
                mapped_axis = 0;
                batch_size;
                original_shape = T.shape result;
              }
            in
            PhysicalTbl.add mapped_tensors result (Any_vmapped vmapped_result);
            forward_val)
    | E_sin { t_in } ->
        Some
          (fun k ->
            let result = op_sin t_in in
            let forward_val = continue k result in
            let vmapped_result =
              {
                base = result;
                mapped_axis = 0;
                batch_size;
                original_shape = T.shape result;
              }
            in
            PhysicalTbl.add mapped_tensors result (Any_vmapped vmapped_result);
            forward_val)
    | E_sqrt { t_in } ->
        Some
          (fun k ->
            let result = op_sqrt t_in in
            let forward_val = continue k result in
            let vmapped_result =
              {
                base = result;
                mapped_axis = 0;
                batch_size;
                original_shape = T.shape result;
              }
            in
            PhysicalTbl.add mapped_tensors result (Any_vmapped vmapped_result);
            forward_val)
    | E_recip { t_in } ->
        Some
          (fun k ->
            let result = op_recip t_in in
            let forward_val = continue k result in
            let vmapped_result =
              {
                base = result;
                mapped_axis = 0;
                batch_size;
                original_shape = T.shape result;
              }
            in
            PhysicalTbl.add mapped_tensors result (Any_vmapped vmapped_result);
            forward_val)
    | E_cast { t_in; target_dtype } ->
        Some
          (fun k ->
            let result = op_cast t_in target_dtype in
            let forward_val = continue k result in
            let vm = get_or_check_vmapped t_in in
            let vmapped_result =
              {
                base = result;
                mapped_axis = vm.mapped_axis;
                batch_size;
                original_shape =
                  (if vm.mapped_axis >= 0 then
                     Array.sub (T.shape result) 1
                       (Array.length (T.shape result) - 1)
                   else T.shape result);
              }
            in
            PhysicalTbl.add mapped_tensors result (Any_vmapped vmapped_result);
            forward_val)
    | E_contiguous { t_in } ->
        Some
          (fun k ->
            let result = op_contiguous t_in in
            let forward_val = continue k result in
            let vm = get_or_check_vmapped t_in in
            let vmapped_result =
              {
                base = result;
                mapped_axis = vm.mapped_axis;
                batch_size;
                original_shape = vm.original_shape;
              }
            in
            PhysicalTbl.add mapped_tensors result (Any_vmapped vmapped_result);
            forward_val)
    | E_copy { t_in } ->
        Some
          (fun k ->
            let result = op_copy t_in in
            let forward_val = continue k result in
            let vm = get_or_check_vmapped t_in in
            let vmapped_result =
              {
                base = result;
                mapped_axis = vm.mapped_axis;
                batch_size;
                original_shape = vm.original_shape;
              }
            in
            PhysicalTbl.add mapped_tensors result (Any_vmapped vmapped_result);
            forward_val)
    | E_reduce_sum { t_in; axes; keepdims } ->
        Some
          (fun k ->
            let vm = get_or_check_vmapped t_in in
            let batched_tensor = get_batched_tensor t_in in
            (* Adjust axes for batch dimension at front *)
            let adjusted_axes =
              if vm.mapped_axis >= 0 then
                (* Check if this is a "reduce all" operation *)
                let tensor_shape = T.shape batched_tensor in
                let num_dims = Array.length tensor_shape in
                if Array.length axes = num_dims - 1 then
                  (* This is reducing all dimensions of the unbatched tensor *)
                  (* We need to reduce all dimensions except the batch dimension *)
                  Array.init (num_dims - 1) (fun i -> i + 1)
                else
                  (* Normal case: adjust each axis by 1 to account for batch
                     dim *)
                  Array.map (fun a -> a + 1) axes
              else axes
            in
            let result =
              op_reduce_sum ~axes:adjusted_axes ~keepdims batched_tensor
            in
            let forward_val = continue k result in
            let vmapped_result =
              {
                base = result;
                mapped_axis = (if vm.mapped_axis >= 0 then 0 else -1);
                batch_size;
                original_shape = T.shape result;
              }
            in
            PhysicalTbl.add mapped_tensors result (Any_vmapped vmapped_result);
            forward_val)
    | E_reduce_max { t_in; axes; keepdims } ->
        Some
          (fun k ->
            let vm = get_or_check_vmapped t_in in
            (* Adjust axes for batch dimension at front *)
            let adjusted_axes =
              if vm.mapped_axis >= 0 then Array.map (fun a -> a + 1) axes
              else axes
            in
            let result = op_reduce_max ~axes:adjusted_axes ~keepdims t_in in
            let forward_val = continue k result in
            let vmapped_result =
              {
                base = result;
                mapped_axis = (if vm.mapped_axis >= 0 then 0 else -1);
                batch_size;
                original_shape = T.shape result;
              }
            in
            PhysicalTbl.add mapped_tensors result (Any_vmapped vmapped_result);
            forward_val)
    | E_reduce_prod { t_in; axes; keepdims } ->
        Some
          (fun k ->
            let vm = get_or_check_vmapped t_in in
            (* Adjust axes for batch dimension at front *)
            let adjusted_axes =
              if vm.mapped_axis >= 0 then Array.map (fun a -> a + 1) axes
              else axes
            in
            let result = op_reduce_prod ~axes:adjusted_axes ~keepdims t_in in
            let forward_val = continue k result in
            let vmapped_result =
              {
                base = result;
                mapped_axis = (if vm.mapped_axis >= 0 then 0 else -1);
                batch_size;
                original_shape = T.shape result;
              }
            in
            PhysicalTbl.add mapped_tensors result (Any_vmapped vmapped_result);
            forward_val)
    | E_permute { t_in; axes } ->
        Some
          (fun k ->
            let vm = get_or_check_vmapped t_in in
            (* Adjust axes for batch dimension at front *)
            let adjusted_axes =
              if vm.mapped_axis >= 0 then
                Array.concat [ [| 0 |]; Array.map (fun a -> a + 1) axes ]
              else axes
            in
            let result = op_permute t_in adjusted_axes in
            let forward_val = continue k result in
            let vmapped_result =
              {
                base = result;
                mapped_axis = (if vm.mapped_axis >= 0 then 0 else -1);
                batch_size;
                original_shape = T.shape result;
              }
            in
            PhysicalTbl.add mapped_tensors result (Any_vmapped vmapped_result);
            forward_val)
    | E_expand { t_in; new_target_shape } ->
        Some
          (fun k ->
            let vm = get_or_check_vmapped t_in in
            (* Adjust shape for batch dimension *)
            let adjusted_shape =
              if vm.mapped_axis >= 0 then
                Array.concat [ [| batch_size |]; new_target_shape ]
              else new_target_shape
            in
            let result = op_expand t_in adjusted_shape in
            let forward_val = continue k result in
            let vmapped_result =
              {
                base = result;
                mapped_axis = (if vm.mapped_axis >= 0 then 0 else -1);
                batch_size;
                original_shape = new_target_shape;
              }
            in
            PhysicalTbl.add mapped_tensors result (Any_vmapped vmapped_result);
            forward_val)
    | E_where { condition; if_true; if_false } ->
        Some
          (fun k ->
            let vm_cond = get_or_check_vmapped condition in
            let vm_true = get_or_check_vmapped if_true in
            let vm_false = get_or_check_vmapped if_false in

            (* Handle broadcasting for where operation *)
            let target_shape =
              if vm_cond.mapped_axis >= 0 then T.shape condition
              else if vm_true.mapped_axis >= 0 then T.shape if_true
              else T.shape if_false
            in

            let cond_broadcast =
              if
                vm_cond.mapped_axis = -1
                && (vm_true.mapped_axis >= 0 || vm_false.mapped_axis >= 0)
              then T.broadcast_to target_shape condition
              else condition
            in

            let true_broadcast =
              if
                vm_true.mapped_axis = -1
                && (vm_cond.mapped_axis >= 0 || vm_false.mapped_axis >= 0)
              then T.broadcast_to target_shape if_true
              else if_true
            in

            let false_broadcast =
              if
                vm_false.mapped_axis = -1
                && (vm_cond.mapped_axis >= 0 || vm_true.mapped_axis >= 0)
              then T.broadcast_to target_shape if_false
              else if_false
            in

            let result =
              op_where cond_broadcast true_broadcast false_broadcast
            in
            let forward_val = continue k result in
            let has_mapped =
              vm_cond.mapped_axis >= 0 || vm_true.mapped_axis >= 0
              || vm_false.mapped_axis >= 0
            in
            let vmapped_result =
              {
                base = result;
                mapped_axis = (if has_mapped then 0 else -1);
                batch_size;
                original_shape = T.shape result;
              }
            in
            PhysicalTbl.add mapped_tensors result (Any_vmapped vmapped_result);
            forward_val)
    | E_gather { data; indices; axis } ->
        Some
          (fun k ->
            let vm_data = get_or_check_vmapped data in
            let vm_indices = get_or_check_vmapped indices in
            (* Adjust axis for batch dimension *)
            let adjusted_axis =
              if vm_data.mapped_axis >= 0 then axis + 1 else axis
            in
            let result = op_gather data indices adjusted_axis in
            let forward_val = continue k result in
            let has_mapped =
              vm_data.mapped_axis >= 0 || vm_indices.mapped_axis >= 0
            in
            let vmapped_result =
              {
                base = result;
                mapped_axis = (if has_mapped then 0 else -1);
                batch_size;
                original_shape = T.shape result;
              }
            in
            PhysicalTbl.add mapped_tensors result (Any_vmapped vmapped_result);
            forward_val)
    | E_scatter { data_template; indices; updates; axis } ->
        Some
          (fun k ->
            let vm_template = get_or_check_vmapped data_template in
            let vm_indices = get_or_check_vmapped indices in
            let vm_updates = get_or_check_vmapped updates in
            (* Adjust axis for batch dimension *)
            let adjusted_axis =
              if vm_template.mapped_axis >= 0 then axis + 1 else axis
            in
            let result =
              op_scatter data_template indices updates adjusted_axis
            in
            let forward_val = continue k result in
            let has_mapped =
              vm_template.mapped_axis >= 0
              || vm_indices.mapped_axis >= 0
              || vm_updates.mapped_axis >= 0
            in
            let vmapped_result =
              {
                base = result;
                mapped_axis = (if has_mapped then 0 else -1);
                batch_size;
                original_shape = T.shape result;
              }
            in
            PhysicalTbl.add mapped_tensors result (Any_vmapped vmapped_result);
            forward_val)
    | E_cat { t_list; axis } ->
        Some
          (fun k ->
            let vms = List.map get_or_check_vmapped t_list in
            let has_mapped = List.exists (fun vm -> vm.mapped_axis >= 0) vms in
            (* Adjust axis for batch dimension *)
            let adjusted_axis = if has_mapped then axis + 1 else axis in
            let result = op_cat t_list adjusted_axis in
            let forward_val = continue k result in
            let vmapped_result =
              {
                base = result;
                mapped_axis = (if has_mapped then 0 else -1);
                batch_size;
                original_shape = T.shape result;
              }
            in
            PhysicalTbl.add mapped_tensors result (Any_vmapped vmapped_result);
            forward_val)
    | E_pad { t_in; padding_config; fill_value } ->
        Some
          (fun k ->
            let vm = get_or_check_vmapped t_in in
            (* Adjust padding config for batch dimension *)
            let adjusted_padding =
              if vm.mapped_axis >= 0 then
                Array.concat [ [| (0, 0) |]; padding_config ]
              else padding_config
            in
            let result = op_pad t_in adjusted_padding fill_value in
            let forward_val = continue k result in
            let vmapped_result =
              {
                base = result;
                mapped_axis = vm.mapped_axis;
                batch_size;
                original_shape = T.shape result;
              }
            in
            PhysicalTbl.add mapped_tensors result (Any_vmapped vmapped_result);
            forward_val)
    | E_shrink { t_in; limits } ->
        Some
          (fun k ->
            let vm = get_or_check_vmapped t_in in
            (* Adjust shrink limits for batch dimension *)
            let adjusted_limits =
              if vm.mapped_axis >= 0 then
                Array.concat [ [| (0, batch_size) |]; limits ]
              else limits
            in
            let result = op_shrink t_in adjusted_limits in
            let forward_val = continue k result in
            let vmapped_result =
              {
                base = result;
                mapped_axis = vm.mapped_axis;
                batch_size;
                original_shape = T.shape result;
              }
            in
            PhysicalTbl.add mapped_tensors result (Any_vmapped vmapped_result);
            forward_val)
    | E_flip { t_in; dims_to_flip } ->
        Some
          (fun k ->
            let vm = get_or_check_vmapped t_in in
            (* Adjust dims_to_flip for batch dimension *)
            let adjusted_dims =
              if vm.mapped_axis >= 0 then
                Array.concat [ [| false |]; dims_to_flip ]
              else dims_to_flip
            in
            let result = op_flip t_in adjusted_dims in
            let forward_val = continue k result in
            let vmapped_result =
              {
                base = result;
                mapped_axis = vm.mapped_axis;
                batch_size;
                original_shape = T.shape result;
              }
            in
            PhysicalTbl.add mapped_tensors result (Any_vmapped vmapped_result);
            forward_val)
    (* Operations that don't have clear vmap semantics yet *)
    | E_assign _ | E_idiv _ | E_mod _ | E_cmplt _ | E_cmpne _ | E_xor _ | E_or _
    | E_and _ | E_threefry _ | E_unfold _ | E_fold _ ->
        None
    | _ -> None
  in

  {
    retc =
      (fun final_result ->
        (* Handle output axis specification *)
        match out_axis with
        | None ->
            (* When out_axes is None, we need to aggregate the batch dimension *)
            (* For now, we'll sum across the batch dimension *)
            (* This might need to be more general in the future *)
            if Array.length (T.shape final_result) > 0 then
              (* Use the backend operation directly to avoid recursion *)
              op_reduce_sum ~axes:[| 0 |] ~keepdims:false final_result
            else final_result
        | Some out_pos ->
            (* Move batch dimension to specified position *)
            if out_pos = 0 then final_result
            else move_axis final_result ~from_axis:0 ~to_axis:out_pos);
    exnc = raise;
    effc;
  }

(* Main vmap function *)
let vmap ?(in_axes = Single (Map 0)) ?(out_axes = OutSingle (Some 0))
    ?axis_name:_ ?axis_size f =
 fun input ->
  let mapped_tensors = PhysicalTbl.create 16 in

  (* Extract axis specification *)
  let axis_spec = extract_axis_spec in_axes in
  let out_axis_spec = extract_out_axis_spec out_axes in

  (* Determine batch size *)
  let batch_size =
    match (axis_spec, axis_size) with
    | Map axis_idx, None ->
        let shape = T.shape input in
        if axis_idx >= Array.length shape || axis_idx < 0 then
          failwith
            (Printf.sprintf
               "vmap: invalid axis %d for tensor with %d dimensions" axis_idx
               (Array.length shape));
        shape.(axis_idx)
    | NoMap, None ->
        failwith "vmap: axis_size must be provided when in_axes is None"
    | _, Some size -> size
  in

  (* Prepare input tensor *)
  let prepared_input =
    match axis_spec with
    | Map axis_idx when axis_idx <> 0 ->
        (* Move mapped axis to front *)
        move_axis input ~from_axis:axis_idx ~to_axis:0
    | Map _ -> input
    | NoMap ->
        (* Add batch dimension at front *)
        add_batch_dim input ~axis:0 ~size:batch_size
  in

  (* Store mapped tensor info *)
  let vmapped_input =
    {
      base = prepared_input;
      mapped_axis = 0;
      batch_size;
      original_shape = T.shape input;
    }
  in
  PhysicalTbl.add mapped_tensors prepared_input (Any_vmapped vmapped_input);

  (* Apply vmap handler *)
  let vmap_handler =
    make_vmap_handler mapped_tensors batch_size axis_spec out_axis_spec
  in
  Effect.Deep.match_with f prepared_input vmap_handler

(* vmaps function for multiple arguments *)
let vmaps ?(in_axes = []) ?(out_axes = OutSingle (Some 0)) ?axis_name:_
    ?axis_size f =
 fun inputs ->
  let mapped_tensors = PhysicalTbl.create 16 in

  (* Default to Map 0 for all inputs if in_axes is empty *)
  let axis_specs =
    if in_axes = [] then List.map (fun _ -> Map 0) inputs
    else if List.length in_axes <> List.length inputs then
      failwith "vmaps: in_axes must have the same length as inputs or be empty"
    else in_axes
  in

  let out_axis_spec = extract_out_axis_spec out_axes in

  (* Determine batch size from first mapped input *)
  let batch_size =
    match axis_size with
    | Some size -> size
    | None ->
        (* Find first mapped input to determine batch size *)
        let rec find_batch_size inputs specs =
          match (inputs, specs) with
          | input :: _, Map axis_idx :: _ ->
              let shape = T.shape input in
              if axis_idx >= Array.length shape || axis_idx < 0 then
                failwith
                  (Printf.sprintf
                     "vmaps: invalid axis %d for tensor with %d dimensions"
                     axis_idx (Array.length shape));
              shape.(axis_idx)
          | _ :: rest_inputs, NoMap :: rest_specs ->
              find_batch_size rest_inputs rest_specs
          | [], [] ->
              failwith
                "vmaps: axis_size must be provided when all in_axes are NoMap"
          | _ -> failwith "vmaps: internal error"
        in
        find_batch_size inputs axis_specs
  in

  (* Prepare each input tensor *)
  let prepared_inputs =
    List.map2
      (fun input axis_spec ->
        match axis_spec with
        | Map axis_idx when axis_idx <> 0 ->
            (* Move mapped axis to front *)
            move_axis input ~from_axis:axis_idx ~to_axis:0
        | Map _ -> input
        | NoMap ->
            (* Add batch dimension at front *)
            add_batch_dim input ~axis:0 ~size:batch_size)
      inputs axis_specs
  in

  (* Store mapped tensor info for each input *)
  List.iter2
    (fun prepared_input axis_spec ->
      let vmapped_input =
        {
          base = prepared_input;
          mapped_axis = (match axis_spec with NoMap -> -1 | Map _ -> 0);
          batch_size;
          original_shape = T.shape prepared_input;
        }
      in
      PhysicalTbl.add mapped_tensors prepared_input (Any_vmapped vmapped_input))
    prepared_inputs axis_specs;

  (* Apply vmap handler *)
  let vmap_handler =
    make_vmap_handler mapped_tensors batch_size (Map 0) out_axis_spec
  in
  Effect.Deep.match_with (fun inputs -> f inputs) prepared_inputs vmap_handler
