(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Operation logging as an effect handler. Purely observational: each
   intercepted operation is re-performed in the enclosing context (so other
   handlers compose as usual) and its name and output shape are printed.
   Operations without a logging arm execute unlogged — falling through is
   harmless here, unlike in the differentiation and batching handlers. *)

open Nx_effect
module T = Nx

let shape_string s =
  "[" ^ String.concat "," (Array.to_list (Array.map string_of_int s)) ^ "]"

let handler ppf =
  let open Effect.Deep in
  (* Logs [name] with the output shape, then continues with the output. *)
  let obs (type a b) k name (out : (a, b) t) =
    Format.fprintf ppf "%s -> %s@." name (shape_string (T.shape out));
    continue k out
  in
  let effc : type c. c Effect.t -> ((c, _) continuation -> _) option =
   fun eff ->
    match eff with
    | E_add { a; b } -> Some (fun k -> obs k "add" (add a b))
    | E_sub { a; b } -> Some (fun k -> obs k "sub" (sub a b))
    | E_mul { a; b } -> Some (fun k -> obs k "mul" (mul a b))
    | E_fdiv { a; b } -> Some (fun k -> obs k "div" (div a b))
    | E_idiv { a; b } -> Some (fun k -> obs k "div" (div a b))
    | E_pow { a; b } -> Some (fun k -> obs k "pow" (pow a b))
    | E_mod { a; b } -> Some (fun k -> obs k "mod" (mod_ a b))
    | E_max { a; b } -> Some (fun k -> obs k "max" (max a b))
    | E_min { a; b } -> Some (fun k -> obs k "min" (min a b))
    | E_atan2 { a; b } -> Some (fun k -> obs k "atan2" (atan2 a b))
    | E_cmpeq { a; b } -> Some (fun k -> obs k "cmpeq" (cmpeq a b))
    | E_cmpne { a; b } -> Some (fun k -> obs k "cmpne" (cmpne a b))
    | E_cmplt { a; b } -> Some (fun k -> obs k "cmplt" (cmplt a b))
    | E_cmple { a; b } -> Some (fun k -> obs k "cmple" (cmple a b))
    | E_neg { t_in } -> Some (fun k -> obs k "neg" (neg t_in))
    | E_sin { t_in } -> Some (fun k -> obs k "sin" (sin t_in))
    | E_cos { t_in } -> Some (fun k -> obs k "cos" (cos t_in))
    | E_tan { t_in } -> Some (fun k -> obs k "tan" (tan t_in))
    | E_asin { t_in } -> Some (fun k -> obs k "asin" (asin t_in))
    | E_acos { t_in } -> Some (fun k -> obs k "acos" (acos t_in))
    | E_atan { t_in } -> Some (fun k -> obs k "atan" (atan t_in))
    | E_sinh { t_in } -> Some (fun k -> obs k "sinh" (sinh t_in))
    | E_cosh { t_in } -> Some (fun k -> obs k "cosh" (cosh t_in))
    | E_tanh { t_in } -> Some (fun k -> obs k "tanh" (tanh t_in))
    | E_exp { t_in } -> Some (fun k -> obs k "exp" (exp t_in))
    | E_log { t_in } -> Some (fun k -> obs k "log" (log t_in))
    | E_sqrt { t_in } -> Some (fun k -> obs k "sqrt" (sqrt t_in))
    | E_recip { t_in } -> Some (fun k -> obs k "recip" (recip t_in))
    | E_abs { t_in } -> Some (fun k -> obs k "abs" (abs t_in))
    | E_sign { t_in } -> Some (fun k -> obs k "sign" (sign t_in))
    | E_erf { t_in } -> Some (fun k -> obs k "erf" (erf t_in))
    | E_trunc { t_in } -> Some (fun k -> obs k "trunc" (trunc t_in))
    | E_ceil { t_in } -> Some (fun k -> obs k "ceil" (ceil t_in))
    | E_floor { t_in } -> Some (fun k -> obs k "floor" (floor t_in))
    | E_round { t_in } -> Some (fun k -> obs k "round" (round t_in))
    | E_where { condition; if_true; if_false } ->
        Some (fun k -> obs k "where" (where condition if_true if_false))
    | E_reshape { t_in; new_shape } ->
        Some (fun k -> obs k "reshape" (reshape t_in new_shape))
    | E_permute { t_in; axes } ->
        Some (fun k -> obs k "permute" (permute t_in axes))
    | E_expand { t_in; new_target_shape } ->
        Some (fun k -> obs k "expand" (expand t_in new_target_shape))
    | E_pad { t_in; padding_config; fill_value } ->
        Some (fun k -> obs k "pad" (pad t_in padding_config fill_value))
    | E_shrink { t_in; limits } ->
        Some (fun k -> obs k "shrink" (shrink t_in limits))
    | E_flip { t_in; dims_to_flip } ->
        Some (fun k -> obs k "flip" (flip t_in dims_to_flip))
    | E_cat { t_list; axis } -> Some (fun k -> obs k "cat" (cat t_list ~axis))
    | E_cast { t_in; target_dtype } ->
        Some (fun k -> obs k "cast" (cast ~dtype:target_dtype t_in))
    | E_contiguous { t_in } ->
        Some (fun k -> obs k "contiguous" (contiguous t_in))
    | E_copy { t_in } -> Some (fun k -> obs k "copy" (copy t_in))
    | E_reduce_sum { t_in; axes; keepdims } ->
        Some (fun k -> obs k "reduce_sum" (reduce_sum ~axes ~keepdims t_in))
    | E_reduce_max { t_in; axes; keepdims } ->
        Some (fun k -> obs k "reduce_max" (reduce_max ~axes ~keepdims t_in))
    | E_reduce_min { t_in; axes; keepdims } ->
        Some (fun k -> obs k "reduce_min" (reduce_min ~axes ~keepdims t_in))
    | E_reduce_prod { t_in; axes; keepdims } ->
        Some (fun k -> obs k "reduce_prod" (reduce_prod ~axes ~keepdims t_in))
    | E_associative_scan { t_in; axis; op } ->
        Some
          (fun k -> obs k "associative_scan" (associative_scan ~axis ~op t_in))
    | E_argmax { t_in; axis; keepdims } ->
        Some (fun k -> obs k "argmax" (argmax ~axis ~keepdims t_in))
    | E_argmin { t_in; axis; keepdims } ->
        Some (fun k -> obs k "argmin" (argmin ~axis ~keepdims t_in))
    | E_sort { t_in; axis; descending } ->
        Some (fun k -> obs k "sort" (sort ~axis ~descending t_in))
    | E_argsort { t_in; axis; descending } ->
        Some (fun k -> obs k "argsort" (argsort ~axis ~descending t_in))
    | E_gather { data; indices; axis } ->
        Some (fun k -> obs k "gather" (gather data indices ~axis))
    | E_scatter { data_template; indices; updates; axis; mode; unique_indices }
      ->
        Some
          (fun k ->
            obs k "scatter"
              (scatter ~mode ~unique_indices data_template ~indices ~updates
                 ~axis))
    | E_matmul { a; b } -> Some (fun k -> obs k "matmul" (matmul a b))
    | _ -> None
  in
  { retc = Fun.id; exnc = raise; effc }

let with_debug ?(ppf = Format.err_formatter) f =
  Effect.Deep.match_with f () (handler ppf)
