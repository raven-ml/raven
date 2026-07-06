(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Vectorizing maps as an effect handler over Nx operations.

   The mapped function is written for unbatched values. Under the handler, every
   tensor is either batched — it physically carries the batch dimension,
   canonically at axis 0 — or a constant of the map. Two mechanisms keep this
   transparent to the function and to the Nx frontend itself:

   - Shape queries go through the [E_view] effect; for batched tensors the
   handler answers with the unbatched remainder of the view. The frontend
   therefore makes exactly the decisions of the unbatched program (broadcasting,
   promotion, reshapes), and the handler translates each resulting primitive to
   its batched form. - Each primitive's translation inserts the batch dimension:
   shape parameters gain a leading batch entry, axis parameters shift by one,
   and constants meeting batched operands are lifted with a broadcast view.

   The virtual view presents the remainder as contiguous even when the batched
   tensor is not; rules compensate by forcing contiguity before reshapes.
   Operations whose operands are all constants fall through unintercepted.
   Nested vmaps stack: each handler owns its batched set and batch size, and the
   translations one level emits are re-translated by the level above.

   Every Nx effect constructor is matched explicitly; operations without a
   batching rule raise when an operand is batched rather than silently producing
   wrong shapes. *)

open Nx_effect
module T = Nx

let err_no_rule op =
  invalid_arg (Printf.sprintf "Rune_next: vmap has no batching rule for %s" op)

type state = { batch_size : int; batched : Tensor_map.Ids.t }

let create ~batch_size = { batch_size; batched = Tensor_map.Ids.create () }
let mark st x = Tensor_map.Ids.add st.batched x
let batched st x = Tensor_map.Ids.mem st.batched x

(* The unbatched shape a tensor presents to the mapped function. *)
let vshape st x =
  let s = T.shape x in
  if batched st x then Array.sub s 1 (Array.length s - 1) else s

let broadcast_shapes sa sb =
  let ra = Array.length sa and rb = Array.length sb in
  let r = Stdlib.max ra rb in
  Array.init r (fun i ->
      let da = if i < r - ra then 1 else sa.(i - (r - ra)) in
      let db = if i < r - rb then 1 else sb.(i - (r - rb)) in
      if da = db then da
      else if da = 1 then db
      else if db = 1 then da
      else invalid_arg "Rune_next: vmap cannot broadcast operand shapes")

(* [to_batched st x target] is [x] as a physically batched tensor of shape
   [batch_size :: target], where [target] is a virtual shape [x]'s virtual shape
   broadcasts to. Constants are lifted with a broadcast view. *)
let to_batched st x target =
  let s = vshape st x in
  if batched st x && s = target then x
  else begin
    let ones = Array.make (Array.length target - Array.length s) 1 in
    let lead = if batched st x then st.batch_size else 1 in
    let x = reshape (contiguous x) (Array.concat [ [| lead |]; ones; s ]) in
    expand x (Array.append [| st.batch_size |] target)
  end

let ensure_batched st x = to_batched st x (vshape st x)

(* Axis parameters count from the virtual shape; the batch dimension sits at 0,
   so non-negative axes shift by one and negative axes are unchanged. *)
let taxis ax = if ax >= 0 then ax + 1 else ax

let handler (st : state) =
  let open Effect.Deep in
  (* Elementwise operations: broadcast all operands to the common batched shape
     and apply the operation unchanged. *)
  let elt1 (type a b c d) k (op : (a, b) t -> (c, d) t) (x : (a, b) t) =
    let out = op x in
    mark st out;
    continue k out
  in
  let elt2 (type a b c d) k (op : (a, b) t -> (a, b) t -> (c, d) t)
      (a_in : (a, b) t) (b_in : (a, b) t) =
    let target = broadcast_shapes (vshape st a_in) (vshape st b_in) in
    let out = op (to_batched st a_in target) (to_batched st b_in target) in
    mark st out;
    continue k out
  in
  let effc : type c. c Effect.t -> ((c, _) continuation -> _) option =
   fun eff ->
    match eff with
    (* Shape queries: batched tensors present their unbatched remainder, as a
       contiguous view. *)
    | E_view x ->
        if batched st x then
          Some
            (fun k ->
              let s = T.shape x in
              continue k
                (Nx_core.View.create (Array.sub s 1 (Array.length s - 1))))
        else None
    (* Constants: creation and metadata. *)
    | E_buffer _ -> None
    | E_const_scalar _ -> None
    | E_from_host _ -> None
    | E_to_device _ -> None
    (* Mutation is incompatible with identity-keyed tracking. *)
    | E_assign { dst; src } ->
        if batched st dst || batched st src then
          Some
            (fun _k ->
              invalid_arg
                "in-place mutation (set_item, set_slice, blit, assign) cannot \
                 be used inside vmap — use scatter instead")
        else None
    (* Elementwise binary *)
    | E_add { a; b } when batched st a || batched st b ->
        Some (fun k -> elt2 k add a b)
    | E_sub { a; b } when batched st a || batched st b ->
        Some (fun k -> elt2 k sub a b)
    | E_mul { a; b } when batched st a || batched st b ->
        Some (fun k -> elt2 k mul a b)
    | E_fdiv { a; b } when batched st a || batched st b ->
        Some (fun k -> elt2 k div a b)
    | E_idiv { a; b } when batched st a || batched st b ->
        Some (fun k -> elt2 k div a b)
    | E_pow { a; b } when batched st a || batched st b ->
        Some (fun k -> elt2 k pow a b)
    | E_mod { a; b } when batched st a || batched st b ->
        Some (fun k -> elt2 k mod_ a b)
    | E_max { a; b } when batched st a || batched st b ->
        Some (fun k -> elt2 k max a b)
    | E_min { a; b } when batched st a || batched st b ->
        Some (fun k -> elt2 k min a b)
    | E_atan2 { a; b } when batched st a || batched st b ->
        Some (fun k -> elt2 k atan2 a b)
    | E_xor { a; b } when batched st a || batched st b ->
        Some (fun k -> elt2 k xor a b)
    | E_or { a; b } when batched st a || batched st b ->
        Some (fun k -> elt2 k or_ a b)
    | E_and { a; b } when batched st a || batched st b ->
        Some (fun k -> elt2 k and_ a b)
    | E_cmpeq { a; b } when batched st a || batched st b ->
        Some (fun k -> elt2 k cmpeq a b)
    | E_cmpne { a; b } when batched st a || batched st b ->
        Some (fun k -> elt2 k cmpne a b)
    | E_cmplt { a; b } when batched st a || batched st b ->
        Some (fun k -> elt2 k cmplt a b)
    | E_cmple { a; b } when batched st a || batched st b ->
        Some (fun k -> elt2 k cmple a b)
    | E_threefry { key; ctr } when batched st key || batched st ctr ->
        Some (fun k -> elt2 k threefry key ctr)
    (* Elementwise unary *)
    | E_neg { t_in } when batched st t_in -> Some (fun k -> elt1 k neg t_in)
    | E_sin { t_in } when batched st t_in -> Some (fun k -> elt1 k sin t_in)
    | E_cos { t_in } when batched st t_in -> Some (fun k -> elt1 k cos t_in)
    | E_tan { t_in } when batched st t_in -> Some (fun k -> elt1 k tan t_in)
    | E_asin { t_in } when batched st t_in -> Some (fun k -> elt1 k asin t_in)
    | E_acos { t_in } when batched st t_in -> Some (fun k -> elt1 k acos t_in)
    | E_atan { t_in } when batched st t_in -> Some (fun k -> elt1 k atan t_in)
    | E_sinh { t_in } when batched st t_in -> Some (fun k -> elt1 k sinh t_in)
    | E_cosh { t_in } when batched st t_in -> Some (fun k -> elt1 k cosh t_in)
    | E_tanh { t_in } when batched st t_in -> Some (fun k -> elt1 k tanh t_in)
    | E_exp { t_in } when batched st t_in -> Some (fun k -> elt1 k exp t_in)
    | E_log { t_in } when batched st t_in -> Some (fun k -> elt1 k log t_in)
    | E_sqrt { t_in } when batched st t_in -> Some (fun k -> elt1 k sqrt t_in)
    | E_recip { t_in } when batched st t_in -> Some (fun k -> elt1 k recip t_in)
    | E_abs { t_in } when batched st t_in -> Some (fun k -> elt1 k abs t_in)
    | E_sign { t_in } when batched st t_in -> Some (fun k -> elt1 k sign t_in)
    | E_erf { t_in } when batched st t_in -> Some (fun k -> elt1 k erf t_in)
    | E_trunc { t_in } when batched st t_in -> Some (fun k -> elt1 k trunc t_in)
    | E_ceil { t_in } when batched st t_in -> Some (fun k -> elt1 k ceil t_in)
    | E_floor { t_in } when batched st t_in -> Some (fun k -> elt1 k floor t_in)
    | E_round { t_in } when batched st t_in -> Some (fun k -> elt1 k round t_in)
    | E_contiguous { t_in } when batched st t_in ->
        Some (fun k -> elt1 k contiguous t_in)
    | E_copy { t_in } when batched st t_in -> Some (fun k -> elt1 k copy t_in)
    | E_cast { t_in; target_dtype } when batched st t_in ->
        Some (fun k -> elt1 k (cast ~dtype:target_dtype) t_in)
    (* Selection *)
    | E_where { condition; if_true; if_false }
      when batched st condition || batched st if_true || batched st if_false ->
        Some
          (fun k ->
            let target =
              broadcast_shapes
                (broadcast_shapes (vshape st condition) (vshape st if_true))
                (vshape st if_false)
            in
            let out =
              where
                (to_batched st condition target)
                (to_batched st if_true target)
                (to_batched st if_false target)
            in
            mark st out;
            continue k out)
    (* Movement: insert the batch dimension into shape parameters. *)
    | E_reshape { t_in; new_shape } when batched st t_in ->
        Some
          (fun k ->
            let out =
              reshape (contiguous t_in)
                (Array.append [| st.batch_size |] new_shape)
            in
            mark st out;
            continue k out)
    | E_permute { t_in; axes } when batched st t_in ->
        Some
          (fun k ->
            let axes' =
              Array.append [| 0 |] (Array.map (fun d -> d + 1) axes)
            in
            let out = permute t_in axes' in
            mark st out;
            continue k out)
    | E_expand { t_in; new_target_shape } when batched st t_in ->
        Some
          (fun k ->
            let out = to_batched st t_in new_target_shape in
            mark st out;
            continue k out)
    | E_pad { t_in; padding_config; fill_value } when batched st t_in ->
        Some
          (fun k ->
            let out =
              pad t_in (Array.append [| (0, 0) |] padding_config) fill_value
            in
            mark st out;
            continue k out)
    | E_shrink { t_in; limits } when batched st t_in ->
        Some
          (fun k ->
            let out =
              shrink t_in (Array.append [| (0, st.batch_size) |] limits)
            in
            mark st out;
            continue k out)
    | E_flip { t_in; dims_to_flip } when batched st t_in ->
        Some
          (fun k ->
            let out = flip t_in (Array.append [| false |] dims_to_flip) in
            mark st out;
            continue k out)
    | E_cat { t_list; axis } when List.exists (batched st) t_list ->
        Some
          (fun k ->
            let out =
              cat (List.map (ensure_batched st) t_list) ~axis:(taxis axis)
            in
            mark st out;
            continue k out)
    (* Reductions and scans *)
    | E_reduce_sum { t_in; axes; keepdims } when batched st t_in ->
        Some
          (fun k ->
            let out = reduce_sum ~axes:(Array.map taxis axes) ~keepdims t_in in
            mark st out;
            continue k out)
    | E_reduce_max { t_in; axes; keepdims } when batched st t_in ->
        Some
          (fun k ->
            let out = reduce_max ~axes:(Array.map taxis axes) ~keepdims t_in in
            mark st out;
            continue k out)
    | E_reduce_min { t_in; axes; keepdims } when batched st t_in ->
        Some
          (fun k ->
            let out = reduce_min ~axes:(Array.map taxis axes) ~keepdims t_in in
            mark st out;
            continue k out)
    | E_reduce_prod { t_in; axes; keepdims } when batched st t_in ->
        Some
          (fun k ->
            let out = reduce_prod ~axes:(Array.map taxis axes) ~keepdims t_in in
            mark st out;
            continue k out)
    | E_associative_scan { t_in; axis; op } when batched st t_in ->
        Some
          (fun k ->
            let out = associative_scan ~axis:(taxis axis) ~op t_in in
            mark st out;
            continue k out)
    | E_argmax { t_in; axis; keepdims } when batched st t_in ->
        Some
          (fun k ->
            let out = argmax ~axis:(taxis axis) ~keepdims t_in in
            mark st out;
            continue k out)
    | E_argmin { t_in; axis; keepdims } when batched st t_in ->
        Some
          (fun k ->
            let out = argmin ~axis:(taxis axis) ~keepdims t_in in
            mark st out;
            continue k out)
    | E_sort { t_in; axis; descending } when batched st t_in ->
        Some
          (fun k ->
            let out = sort ~axis:(taxis axis) ~descending t_in in
            mark st out;
            continue k out)
    | E_argsort { t_in; axis; descending } when batched st t_in ->
        Some
          (fun k ->
            let out = argsort ~axis:(taxis axis) ~descending t_in in
            mark st out;
            continue k out)
    (* Gather / scatter: operands agree on rank, so all are lifted. *)
    | E_gather { data; indices; axis }
      when batched st data || batched st indices ->
        Some
          (fun k ->
            let out =
              gather (ensure_batched st data)
                (ensure_batched st indices)
                ~axis:(taxis axis)
            in
            mark st out;
            continue k out)
    | E_scatter { data_template; indices; updates; axis; mode; unique_indices }
      when batched st data_template || batched st indices || batched st updates
      ->
        Some
          (fun k ->
            let out =
              scatter ~mode ~unique_indices
                (ensure_batched st data_template)
                ~indices:(ensure_batched st indices)
                ~updates:(ensure_batched st updates)
                ~axis:(taxis axis)
            in
            mark st out;
            continue k out)
    (* Matrix multiplication: the backend broadcasts leading batch dimensions,
       and the frontend promotes vectors to matrices against virtual shapes
       before this effect is performed. *)
    | E_matmul { a; b } when batched st a || batched st b ->
        Some
          (fun k ->
            let out = matmul a b in
            mark st out;
            continue k out)
    (* No batching rule yet. *)
    | E_unfold { t_in; _ } when batched st t_in -> err_no_rule "unfold"
    | E_fold { t_in; _ } when batched st t_in -> err_no_rule "fold"
    | E_fft { t; _ } when batched st t -> err_no_rule "fft"
    | E_ifft { t; _ } when batched st t -> err_no_rule "ifft"
    | E_rfft { t; _ } when batched st t -> err_no_rule "rfft"
    | E_irfft { t; _ } when batched st t -> err_no_rule "irfft"
    | E_psum { t_in } when batched st t_in -> err_no_rule "psum"
    | E_cholesky { t_in; _ } when batched st t_in -> err_no_rule "cholesky"
    | E_qr { t_in; _ } when batched st t_in -> err_no_rule "qr"
    | E_svd { t_in; _ } when batched st t_in -> err_no_rule "svd"
    | E_eig { t_in; _ } when batched st t_in -> err_no_rule "eig"
    | E_eigh { t_in; _ } when batched st t_in -> err_no_rule "eigh"
    | E_triangular_solve { a; b; _ } when batched st a || batched st b ->
        err_no_rule "triangular_solve"
    (* Operations on constants, and effects from other libraries, fall through.
       A new Nx tensor operation must be added to this match: an unmatched
       batched operand would silently produce wrong shapes. *)
    | _ -> None
  in
  { retc = Fun.id; exnc = raise; effc }
