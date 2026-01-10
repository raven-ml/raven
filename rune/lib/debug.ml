(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_rune
module T = Tensor

type tensor_ref = Tensor_ref : ('a, 'b) T.t -> tensor_ref

(* Debug context effects *)
type _ Effect.t +=
  | E_push_debug_context : string -> unit Effect.t
  | E_pop_debug_context : unit Effect.t

type tensor_stats = {
  mean : float;
  std : float;
  min_val : float;
  max_val : float;
  nan_count : int;
}

let push_context name =
  try Effect.perform (E_push_debug_context name) with Effect.Unhandled _ -> ()

let pop_context () =
  try Effect.perform E_pop_debug_context with Effect.Unhandled _ -> ()

let with_context name f =
  try
    Effect.perform (E_push_debug_context name);
    Fun.protect f ~finally:(fun () -> Effect.perform E_pop_debug_context)
  with Effect.Unhandled _ -> f ()

let compute_stats (Tensor_ref t) =
  try
    let t_f32 = T.cast T.float32 t in
    let mean = T.item [] (T.mean t_f32) in
    let std = T.item [] (T.std t_f32) in
    let min_val = T.item [] (T.min t_f32) in
    let max_val = T.item [] (T.max t_f32) in
    let is_nan = T.isnan t_f32 in
    let nan_count =
      int_of_float (T.item [] (T.sum (T.cast T.float32 is_nan)))
    in
    { mean; std; min_val; max_val; nan_count }
  with _ ->
    { mean = 0.0; std = 0.0; min_val = 0.0; max_val = 0.0; nan_count = 0 }

let get_debug_indent context_stack =
  let depth = List.length context_stack in
  if depth = 0 then "├─ "
  else
    let rec build_prefix n =
      if n = 0 then "" else "│  " ^ build_prefix (n - 1)
    in
    build_prefix depth ^ "├─ "

let format_number f =
  (* Handle negative zero *)
  let f = if f = -0. then 0. else f in
  (* Use 2 decimal precision for consistency *)
  Printf.sprintf "%.2f" f

let dtype_to_string (type a b) (dtype : (a, b) Nx_core.Dtype.t) =
  match dtype with
  | Float32 -> "f32"
  | Float64 -> "f64"
  | Float16 -> "f16"
  | Int32 -> "i32"
  | Int64 -> "i64"
  | UInt8 -> "u8"
  | Int8 -> "i8"
  | Int16 -> "i16"
  | UInt16 -> "u16"
  | UInt32 -> "u32"
  | UInt64 -> "u64"
  | Complex64 -> "c64"
  | Complex128 -> "c128"
  | BFloat16 -> "bf16"
  | Bool -> "bool"
  | Int4 -> "i4"
  | UInt4 -> "u4"
  | Float8_e4m3 -> "f8e4m3"
  | Float8_e5m2 -> "f8e5m2"

let format_input_shapes input_tensors =
  match input_tensors with
  | [] -> ""
  | tensors ->
      tensors
      |> List.map (function Tensor_ref t -> T.shape_to_string (T.shape t))
      |> String.concat ","

let log_operation context_stack op_name input_tensors output_tensor =
  let indent = get_debug_indent context_stack in

  (* Check if we're in a gradient context *)
  let in_grad_context =
    List.exists
      (fun ctx ->
        String.length ctx > 0 && String.starts_with ~prefix:"\xE2\x88\x87" ctx)
      context_stack
  in

  (* Format input part with arrow *)
  let input_part =
    let input_str = format_input_shapes input_tensors in
    if input_str = "" then "→ " else input_str ^ " → "
  in

  let shape_str, dtype_str =
    match output_tensor with
    | Tensor_ref t ->
        let shape = T.shape_to_string (T.shape t) in
        let dtype = dtype_to_string (T.dtype t) in
        (shape, dtype)
  in

  (* Put dtype inside brackets for output *)
  let output_shape_with_dtype =
    if shape_str = "[]" then "[" ^ dtype_str ^ "]"
    else
      let shape_without_brackets =
        String.sub shape_str 1 (String.length shape_str - 2)
      in
      "[" ^ shape_without_brackets ^ " " ^ dtype_str ^ "]"
  in

  let stats = compute_stats output_tensor in

  (* Check if tensor is all zeros *)
  let stats_str =
    if
      stats.mean = 0. && stats.std = 0. && stats.min_val = 0.
      && stats.max_val = 0.
    then Printf.sprintf " zeros nans=%d" stats.nan_count
    else if
      stats.mean = 1. && stats.std = 0. && stats.min_val = 1.
      && stats.max_val = 1.
    then Printf.sprintf " ones nans=%d" stats.nan_count
    else
      Printf.sprintf " μ=%s σ=%s range=[%s,%s] nans=%d"
        (format_number stats.mean) (format_number stats.std)
        (format_number stats.min_val)
        (format_number stats.max_val)
        stats.nan_count
  in

  (* Add memory usage *)
  let memory_str =
    match output_tensor with
    | Tensor_ref t ->
        let shape = T.shape t in
        let num_elements = Array.fold_left ( * ) 1 shape in
        let bytes_per_element =
          match T.dtype t with
          | Float32 | Int32 | UInt32 -> 4
          | Float64 | Int64 | UInt64 | Complex64 -> 8
          | Float16 | Int16 | UInt16 | BFloat16 -> 2
          | UInt8 | Int8 | Float8_e4m3 | Float8_e5m2 | Bool -> 1
          | Complex128 -> 16
          | Int4 | UInt4 -> 1 (* 2 values packed per byte *)
        in
        let bytes = num_elements * bytes_per_element in
        let memory_mb = float bytes /. (1024. *. 1024.) in
        if memory_mb < 0.01 then Printf.sprintf " %.3fMB" memory_mb
        else Printf.sprintf " %.1fMB" memory_mb
  in

  (* Add NaN warning *)
  let nan_warning = if stats.nan_count > 0 then " ⚠ NaN detected!" else "" in

  (* Check for exploding gradients in gradient operations *)
  let grad_warning =
    if in_grad_context then
      (* This is a gradient operation *)
      let max_abs = max (abs_float stats.max_val) (abs_float stats.min_val) in
      if max_abs > 100. then " ⚠ Exploding gradients!" else ""
    else ""
  in

  Printf.printf "%s%s %s%s%s%s%s%s\n%!" indent op_name input_part
    output_shape_with_dtype stats_str memory_str nan_warning grad_warning

let debug_handler () =
  let context_stack = ref [] in
  let open Effect.Deep in
  {
    retc = (fun x -> x);
    exnc = raise;
    effc =
      (fun (type a) (eff : a Effect.t) ->
        match eff with
        | E_push_debug_context name ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let parent_indent = get_debug_indent !context_stack in
                Printf.printf "%s%s\n%!" parent_indent name;
                context_stack := name :: !context_stack;
                continue k ())
        | E_pop_debug_context ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                (match !context_stack with
                | [] -> failwith "Cannot pop from an empty context stack"
                | _ :: rest -> context_stack := rest);
                continue k ())
        | E_add { out; a; b } ->
            Some
              (fun (k : (unit, _) Effect.Deep.continuation) ->
                op_add ~out a b;
                log_operation !context_stack "add"
                  [ Tensor_ref a; Tensor_ref b ]
                  (Tensor_ref out);
                continue k ())
        | E_sub { out; a; b } ->
            Some
              (fun (k : (unit, _) Effect.Deep.continuation) ->
                op_sub ~out a b;
                log_operation !context_stack "sub"
                  [ Tensor_ref a; Tensor_ref b ]
                  (Tensor_ref out);
                continue k ())
        | E_mul { out; a; b } ->
            Some
              (fun (k : (unit, _) Effect.Deep.continuation) ->
                op_mul ~out a b;
                log_operation !context_stack "mul"
                  [ Tensor_ref a; Tensor_ref b ]
                  (Tensor_ref out);
                continue k ())
        | E_matmul { out; a; b } ->
            Some
              (fun (k : (unit, _) Effect.Deep.continuation) ->
                op_matmul ~out a b;
                log_operation !context_stack "matmul"
                  [ Tensor_ref a; Tensor_ref b ]
                  (Tensor_ref out);
                continue k ())
        | E_neg { out; t_in } ->
            Some
              (fun (k : (unit, _) Effect.Deep.continuation) ->
                op_neg ~out t_in;
                log_operation !context_stack "neg" [ Tensor_ref t_in ]
                  (Tensor_ref out);
                continue k ())
        | E_reduce_sum { out; t_in; axes; keepdims } ->
            Some
              (fun (k : (unit, _) Effect.Deep.continuation) ->
                op_reduce_sum ~out ~axes ~keepdims t_in;
                log_operation !context_stack "sum" [ Tensor_ref t_in ]
                  (Tensor_ref out);
                continue k ())
        | E_reduce_max { out; t_in; axes; keepdims } ->
            Some
              (fun (k : (unit, _) Effect.Deep.continuation) ->
                op_reduce_max ~out ~axes ~keepdims t_in;
                log_operation !context_stack "max" [ Tensor_ref t_in ]
                  (Tensor_ref out);
                continue k ())
        | E_reduce_min { out; t_in; axes; keepdims } ->
            Some
              (fun (k : (unit, _) Effect.Deep.continuation) ->
                op_reduce_min ~out ~axes ~keepdims t_in;
                log_operation !context_stack "min" [ Tensor_ref t_in ]
                  (Tensor_ref out);
                continue k ())
        | E_reshape { t_in; new_shape } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = op_reshape t_in new_shape in
                log_operation !context_stack "reshape" [ Tensor_ref t_in ]
                  (Tensor_ref result);
                continue k result)
        | E_cast { t_in; target_dtype } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = op_cast t_in target_dtype in
                log_operation !context_stack "cast" [ Tensor_ref t_in ]
                  (Tensor_ref result);
                continue k result)
        | E_sqrt { out; t_in } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                op_sqrt ~out t_in;
                log_operation !context_stack "sqrt" [ Tensor_ref t_in ]
                  (Tensor_ref out);
                continue k ())
        | E_sin { out; t_in } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                op_sin ~out t_in;
                log_operation !context_stack "sin" [ Tensor_ref t_in ]
                  (Tensor_ref out);
                continue k ())
        | E_fdiv { out; a; b } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                op_fdiv ~out a b;
                log_operation !context_stack "div"
                  [ Tensor_ref a; Tensor_ref b ]
                  (Tensor_ref out);
                continue k ())
        | E_pow { out; a; b } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                op_pow ~out a b;
                log_operation !context_stack "pow"
                  [ Tensor_ref a; Tensor_ref b ]
                  (Tensor_ref out);
                continue k ())
        | E_max { out; a; b } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                op_max ~out a b;
                log_operation !context_stack "max"
                  [ Tensor_ref a; Tensor_ref b ]
                  (Tensor_ref out);
                continue k ())
        | E_where { out; condition; if_true; if_false } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                op_where ~out condition if_true if_false;
                log_operation !context_stack "where"
                  [
                    Tensor_ref condition;
                    Tensor_ref if_true;
                    Tensor_ref if_false;
                  ]
                  (Tensor_ref out);
                continue k ())
        | E_cat { t_list; axis } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = op_cat t_list axis in
                log_operation !context_stack "cat"
                  (List.map (fun t -> Tensor_ref t) t_list)
                  (Tensor_ref result);
                continue k result)
        | E_gather { data; indices; axis } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = op_gather data indices axis in
                log_operation !context_stack "gather"
                  [ Tensor_ref data; Tensor_ref indices ]
                  (Tensor_ref result);
                continue k result)
        | E_scatter { data_template; indices; updates; axis } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = op_scatter data_template indices updates axis in
                log_operation !context_stack "scatter"
                  [
                    Tensor_ref data_template;
                    Tensor_ref indices;
                    Tensor_ref updates;
                  ]
                  (Tensor_ref result);
                continue k result)
        | E_permute { t_in; axes } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = op_permute t_in axes in
                log_operation !context_stack "permute" [ Tensor_ref t_in ]
                  (Tensor_ref result);
                continue k result)
        | E_expand { t_in; new_target_shape } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = op_expand t_in new_target_shape in
                log_operation !context_stack "expand" [ Tensor_ref t_in ]
                  (Tensor_ref result);
                continue k result)
        | E_pad { t_in; padding_config; fill_value } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = op_pad t_in padding_config fill_value in
                log_operation !context_stack "pad" [ Tensor_ref t_in ]
                  (Tensor_ref result);
                continue k result)
        | E_shrink { t_in; limits } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = op_shrink t_in limits in
                log_operation !context_stack "shrink" [ Tensor_ref t_in ]
                  (Tensor_ref result);
                continue k result)
        | E_flip { t_in; dims_to_flip } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = op_flip t_in dims_to_flip in
                log_operation !context_stack "flip" [ Tensor_ref t_in ]
                  (Tensor_ref result);
                continue k result)
        | E_contiguous { t_in } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = op_contiguous t_in in
                log_operation !context_stack "contiguous" [ Tensor_ref t_in ]
                  (Tensor_ref result);
                continue k result)
        | E_copy { t_in } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = op_copy t_in in
                log_operation !context_stack "copy" [ Tensor_ref t_in ]
                  (Tensor_ref result);
                continue k result)
        | E_buffer { context; dtype; size_in_elements } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = op_buffer context dtype size_in_elements in
                log_operation !context_stack "buffer" [] (Tensor_ref result);
                continue k result)
        | E_const_scalar { context; value; dtype } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = op_const_scalar context value dtype in
                log_operation !context_stack "const_scalar" []
                  (Tensor_ref result);
                continue k result)
        | E_const_array { context; array } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = op_const_array context array in
                log_operation !context_stack "const_array" []
                  (Tensor_ref result);
                continue k result)
        | E_idiv { out; a; b } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                op_idiv ~out a b;
                log_operation !context_stack "idiv"
                  [ Tensor_ref a; Tensor_ref b ]
                  (Tensor_ref out);
                continue k ())
        | E_mod { out; a; b } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                op_mod ~out a b;
                log_operation !context_stack "mod"
                  [ Tensor_ref a; Tensor_ref b ]
                  (Tensor_ref out);
                continue k ())
        | E_cmplt { out; a; b } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                op_cmplt ~out a b;
                log_operation !context_stack "lt"
                  [ Tensor_ref a; Tensor_ref b ]
                  (Tensor_ref out);
                continue k ())
        | E_cmpne { out; a; b } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                op_cmpne ~out a b;
                log_operation !context_stack "ne"
                  [ Tensor_ref a; Tensor_ref b ]
                  (Tensor_ref out);
                continue k ())
        | E_xor { out; a; b } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                op_xor ~out a b;
                log_operation !context_stack "xor"
                  [ Tensor_ref a; Tensor_ref b ]
                  (Tensor_ref out);
                continue k ())
        | E_or { out; a; b } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                op_or ~out a b;
                log_operation !context_stack "or"
                  [ Tensor_ref a; Tensor_ref b ]
                  (Tensor_ref out);
                continue k ())
        | E_and { out; a; b } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                op_and ~out a b;
                log_operation !context_stack "and"
                  [ Tensor_ref a; Tensor_ref b ]
                  (Tensor_ref out);
                continue k ())
        | E_recip { out; t_in } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                op_recip ~out t_in;
                log_operation !context_stack "recip" [ Tensor_ref t_in ]
                  (Tensor_ref out);
                continue k ())
        | E_reduce_prod { out; t_in; axes; keepdims } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                op_reduce_prod ~out ~axes ~keepdims t_in;
                log_operation !context_stack "prod" [ Tensor_ref t_in ]
                  (Tensor_ref out);
                continue k ())
        | E_assign { dst; src } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                op_assign dst src;
                log_operation !context_stack "assign" [ Tensor_ref src ]
                  (Tensor_ref dst);
                continue k ())
        | E_threefry { key; ctr } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = op_threefry key ctr in
                log_operation !context_stack "threefry"
                  [ Tensor_ref key; Tensor_ref ctr ]
                  (Tensor_ref result);
                continue k result)
        | E_to_device { t_in; context } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result = to_device context t_in in
                log_operation !context_stack "to_device" [ Tensor_ref t_in ]
                  (Tensor_ref result);
                continue k result)
        | E_unfold { t_in; kernel_size; stride; dilation; padding } ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result =
                  op_unfold t_in ~kernel_size ~stride ~dilation ~padding
                in
                log_operation !context_stack "unfold" [ Tensor_ref t_in ]
                  (Tensor_ref result);
                continue k result)
        | E_fold { t_in; output_size; kernel_size; stride; dilation; padding }
          ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                let result =
                  op_fold t_in ~output_size ~kernel_size ~stride ~dilation
                    ~padding
                in
                log_operation !context_stack "fold" [ Tensor_ref t_in ]
                  (Tensor_ref result);
                continue k result)
        | _ -> None);
  }

let debug f x =
  let handler = debug_handler () in
  Effect.Deep.match_with f x handler
