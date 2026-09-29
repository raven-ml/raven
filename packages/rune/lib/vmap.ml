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

let no_rule op () =
  invalid_arg (Printf.sprintf "Rune: vmap has no batching rule for %s" op)

type state = {
  batch_size : int;
  batched : Tensor_map.Ids.t;
  axis : Axis.t option;
}

let create ?axis ~batch_size () =
  { batch_size; batched = Tensor_map.Ids.create (); axis }

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
      else invalid_arg "Rune: vmap cannot broadcast operand shapes")

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

(* The sum over the lanes of a value each lane holds: a batched one summed along
   its batch axis, one every lane shares times their number. *)
let sum_lanes st v =
  if batched st v then T.sum ~axes:[ 0 ] v
  else T.mul_s v (Nx_dtype.of_float (T.dtype v) (Float.of_int st.batch_size))

(* Axis parameters count from the virtual shape; the batch dimension sits at 0,
   so non-negative axes shift by one and negative axes are unchanged. *)
let taxis ax = if ax >= 0 then ax + 1 else ax

(* Quantised products. The lane becomes a leading axis of the weight's parts, of
   [ids] and of [x], a unit axis where one is unbatched, after [pad] unit axes
   that align the operands' batch axes, so that each lane meets its own weight
   with its own ids. *)

let lift st ?shape ~lead ~pad x =
  let s = match shape with Some s -> s | None -> vshape st x in
  let b = if batched st x then st.batch_size else 1 in
  let x = T.reshape (Array.concat [ [| b |]; Array.make pad 1; s ]) x in
  if b >= lead then x
  else T.broadcast_to (Array.concat [ [| lead |]; Array.make pad 1; s ]) x

let lift_weight st ~pad (Nx_quant.Mxfp4 { codes; scales }) =
  let lead =
    if batched st codes || batched st scales then st.batch_size else 1
  in
  Nx_quant.mxfp4 ~scales:(lift st ~lead ~pad scales) (lift st ~lead ~pad codes)

let quant_batched (type a b) st (Nx_quant.Mxfp4 { codes; scales })
    (op : (a, b) Nx_quant.Effect.op) =
  batched st codes || batched st scales
  ||
  match op with
  | Apply { ids; x; _ } ->
      batched st x || Option.fold ~none:false ~some:(batched st) ids
  | Dequant _ -> false

let quant (type a b) st w (op : (a, b) Nx_quant.Effect.op) : (a, b) t =
  match op with
  | Dequant _ -> Nx_quant.Effect.perform (lift_weight st ~pad:0 w) op
  | Apply { ids; x; transpose } ->
      let ws = vshape st (match w with Nx_quant.Mxfp4 { codes; _ } -> codes)
      and xs = vshape st x in
      let vector = Array.length xs = 1 in
      let xb = if vector then [||] else Array.sub xs 0 (Array.length xs - 2) in
      let wb =
        match ids with
        | None -> Array.sub ws 0 (Array.length ws - 2)
        | Some ids -> vshape st ids
      in
      let rank = Stdlib.max (Array.length xb) (Array.length wb) in
      let pad = rank - Array.length wb in
      let x =
        lift st
          ?shape:(if vector then Some (Array.append [| 1 |] xs) else None)
          ~lead:1
          ~pad:(rank - Array.length xb)
          x
      in
      let ids = Option.map (lift st ~lead:1 ~pad) ids in
      let y =
        Nx_quant.Effect.perform (lift_weight st ~pad w)
          (Apply { ids; x; transpose })
      in
      if vector then
        let s = T.shape y in
        let r = Array.length s in
        T.reshape (Array.append (Array.sub s 0 (r - 2)) [| s.(r - 1) |]) y
      else y

(* Polymorphic recursion: the nested fibers spawned for custom rules run at the
   rule's own result type. *)
let rec handler : type r. state -> (r, r) Effect.Deep.handler =
 fun (st : state) ->
  let open Effect.Deep in
  (* Elementwise operations: broadcast all operands to the common batched shape
     and apply the operation unchanged. *)
  let elt1 (type a b c d) (op : (a, b) t -> (c, d) t) (x : (a, b) t) =
    let out = op x in
    mark st out;
    out
  in
  let elt2 (type a b c d) (op : (a, b) t -> (a, b) t -> (c, d) t)
      (a_in : (a, b) t) (b_in : (a, b) t) =
    let target = broadcast_shapes (vshape st a_in) (vshape st b_in) in
    let out = op (to_batched st a_in target) (to_batched st b_in target) in
    mark st out;
    out
  in
  let rule : type c. c Effect.t -> (unit -> c) option =
   fun eff ->
    match eff with
    (* Shape queries: batched tensors present their unbatched remainder, as a
       contiguous view. *)
    | E_view x ->
        if batched st x then
          Some
            (fun () ->
              let s = T.shape x in
              Nx_array.View.create (Array.sub s 1 (Array.length s - 1)))
        else None
    (* Reading the value of a batched tensor would expose the physical, batched
       buffer to code that believes it is unbatched. *)
    | E_to_host x ->
        if batched st x then
          Some
            (fun () ->
              invalid_arg
                "Rune: cannot read the value of a batched tensor inside vmap; \
                 return it from the mapped function instead")
        else None
    (* Constants: creation and metadata. *)
    | E_buffer _ -> None
    | E_const_scalar _ -> None
    | E_from_host _ -> None
    (* Placement: the batch axis sits in front of a split axis. A lane of a map
       over the split axis has no placement of its own. *)
    | E_place { placement = p; t_in } when batched st t_in ->
        let p = Nx_effect.Placement.with_leading_axis p in
        Some (fun () -> elt1 (place p) t_in)
    | E_place _ -> None
    | E_placement x when batched st x ->
        Some
          (fun () ->
            match Nx_effect.Placement.without_leading_axis (placement x) with
            | None ->
                invalid_arg
                  "Rune: a lane of vmap over a split axis has no placement"
            | Some p -> p)
    | E_placement _ -> None
    (* Elementwise binary *)
    | E_add { a; b } when batched st a || batched st b ->
        Some (fun () -> elt2 add a b)
    | E_sub { a; b } when batched st a || batched st b ->
        Some (fun () -> elt2 sub a b)
    | E_mul { a; b } when batched st a || batched st b ->
        Some (fun () -> elt2 mul a b)
    | E_fdiv { a; b } when batched st a || batched st b ->
        Some (fun () -> elt2 fdiv a b)
    | E_idiv { a; b } when batched st a || batched st b ->
        Some (fun () -> elt2 idiv a b)
    | E_pow { a; b } when batched st a || batched st b ->
        Some (fun () -> elt2 pow a b)
    | E_mod { a; b } when batched st a || batched st b ->
        Some (fun () -> elt2 mod_ a b)
    | E_max { a; b } when batched st a || batched st b ->
        Some (fun () -> elt2 max a b)
    | E_min { a; b } when batched st a || batched st b ->
        Some (fun () -> elt2 min a b)
    | E_atan2 { a; b } when batched st a || batched st b ->
        Some (fun () -> elt2 atan2 a b)
    | E_xor { a; b } when batched st a || batched st b ->
        Some (fun () -> elt2 xor a b)
    | E_or { a; b } when batched st a || batched st b ->
        Some (fun () -> elt2 or_ a b)
    | E_and { a; b } when batched st a || batched st b ->
        Some (fun () -> elt2 and_ a b)
    | E_cmpeq { a; b } when batched st a || batched st b ->
        Some (fun () -> elt2 cmpeq a b)
    | E_cmpne { a; b } when batched st a || batched st b ->
        Some (fun () -> elt2 cmpne a b)
    | E_cmplt { a; b } when batched st a || batched st b ->
        Some (fun () -> elt2 cmplt a b)
    | E_cmple { a; b } when batched st a || batched st b ->
        Some (fun () -> elt2 cmple a b)
    | E_threefry { key; ctr } when batched st key || batched st ctr ->
        Some (fun () -> elt2 threefry key ctr)
    (* The lane index is the per-lane iota [0 .. batch_size-1], carried as a
       batched scalar, so that a key folded with it decorrelates the lanes. A
       named map answers only the calls that name it. *)
    | Axis.E_lane_index axis when axis = st.axis ->
        Some
          (fun () ->
            let idx = T.arange Nx.int32 0 st.batch_size 1 in
            mark st idx;
            idx)
    (* The map a gather names answers it with the lanes as data: a fresh alias
       of a batched operand's physical tensor, or the broadcast of an operand
       every lane shares, a constant of the map either way. Another map gathers
       its batched operand's physical tensor and keeps its own lanes in front
       of the gathered axis. *)
    | Axis.E_lanes { axis; t_in } when st.axis = Some axis ->
        Some
          (fun () ->
            if batched st t_in then Structure.alias t_in
            else
              T.broadcast_to
                (Array.append [| st.batch_size |] (T.shape t_in))
                t_in)
    | Axis.E_lanes { axis; t_in } when batched st t_in ->
        Some
          (fun () ->
            let out = T.swapaxes 0 1 (Axis.lanes axis t_in) in
            mark st out;
            out)
    (* Elementwise unary *)
    | E_neg { t_in } when batched st t_in -> Some (fun () -> elt1 neg t_in)
    | E_sin { t_in } when batched st t_in -> Some (fun () -> elt1 sin t_in)
    | E_cos { t_in } when batched st t_in -> Some (fun () -> elt1 cos t_in)
    | E_tan { t_in } when batched st t_in -> Some (fun () -> elt1 tan t_in)
    | E_asin { t_in } when batched st t_in -> Some (fun () -> elt1 asin t_in)
    | E_acos { t_in } when batched st t_in -> Some (fun () -> elt1 acos t_in)
    | E_atan { t_in } when batched st t_in -> Some (fun () -> elt1 atan t_in)
    | E_sinh { t_in } when batched st t_in -> Some (fun () -> elt1 sinh t_in)
    | E_cosh { t_in } when batched st t_in -> Some (fun () -> elt1 cosh t_in)
    | E_tanh { t_in } when batched st t_in -> Some (fun () -> elt1 tanh t_in)
    | E_exp { t_in } when batched st t_in -> Some (fun () -> elt1 exp t_in)
    | E_log { t_in } when batched st t_in -> Some (fun () -> elt1 log t_in)
    | E_sqrt { t_in } when batched st t_in -> Some (fun () -> elt1 sqrt t_in)
    | E_recip { t_in } when batched st t_in -> Some (fun () -> elt1 recip t_in)
    | E_abs { t_in } when batched st t_in -> Some (fun () -> elt1 abs t_in)
    | E_sign { t_in } when batched st t_in -> Some (fun () -> elt1 sign t_in)
    | E_erf { t_in } when batched st t_in -> Some (fun () -> elt1 erf t_in)
    | E_trunc { t_in } when batched st t_in -> Some (fun () -> elt1 trunc t_in)
    | E_ceil { t_in } when batched st t_in -> Some (fun () -> elt1 ceil t_in)
    | E_floor { t_in } when batched st t_in -> Some (fun () -> elt1 floor t_in)
    | E_round { t_in } when batched st t_in -> Some (fun () -> elt1 round t_in)
    | E_contiguous { t_in } when batched st t_in ->
        Some (fun () -> elt1 contiguous t_in)
    | E_copy { t_in } when batched st t_in -> Some (fun () -> elt1 copy t_in)
    | E_cast { t_in; target_dtype } when batched st t_in ->
        Some (fun () -> elt1 (cast ~dtype:target_dtype) t_in)
    | E_bitcast { t_in; target_dtype } when batched st t_in ->
        Some (fun () -> elt1 (bitcast ~dtype:target_dtype) t_in)
    (* Selection *)
    | E_where { condition; if_true; if_false }
      when batched st condition || batched st if_true || batched st if_false ->
        Some
          (fun () ->
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
            out)
    (* Movement: insert the batch dimension into shape parameters. *)
    | E_reshape { t_in; new_shape } when batched st t_in ->
        Some
          (fun () ->
            let out =
              reshape (contiguous t_in)
                (Array.append [| st.batch_size |] new_shape)
            in
            mark st out;
            out)
    | E_permute { t_in; axes } when batched st t_in ->
        Some
          (fun () ->
            let axes' =
              Array.append [| 0 |] (Array.map (fun d -> d + 1) axes)
            in
            let out = permute t_in axes' in
            mark st out;
            out)
    | E_expand { t_in; new_target_shape } when batched st t_in ->
        Some
          (fun () ->
            let out = to_batched st t_in new_target_shape in
            mark st out;
            out)
    | E_pad { t_in; padding_config; fill_value } when batched st t_in ->
        Some
          (fun () ->
            let out =
              pad t_in (Array.append [| (0, 0) |] padding_config) fill_value
            in
            mark st out;
            out)
    | E_shrink { t_in; limits } when batched st t_in ->
        Some
          (fun () ->
            let out =
              shrink t_in (Array.append [| (0, st.batch_size) |] limits)
            in
            mark st out;
            out)
    | E_flip { t_in; dims_to_flip } when batched st t_in ->
        Some
          (fun () ->
            let out = flip t_in (Array.append [| false |] dims_to_flip) in
            mark st out;
            out)
    | E_sliding_window { t_in; axis; window; step } when batched st t_in ->
        Some
          (fun () ->
            (* The trailing window axis lands at the physical end, which is the
               virtual end shifted past the batch dimension. *)
            let out = sliding_window t_in ~axis:(taxis axis) ~window ~step in
            mark st out;
            out)
    | E_cat { t_list; axis } when List.exists (batched st) t_list ->
        Some
          (fun () ->
            let out =
              cat (List.map (ensure_batched st) t_list) ~axis:(taxis axis)
            in
            mark st out;
            out)
    (* Reductions and scans *)
    | E_reduce_sum { t_in; axes } when batched st t_in ->
        Some
          (fun () ->
            let out = reduce ~op:`Sum ~axes:(Array.map taxis axes) t_in in
            mark st out;
            out)
    | E_reduce_max { t_in; axes } when batched st t_in ->
        Some
          (fun () ->
            let out = reduce ~op:`Max ~axes:(Array.map taxis axes) t_in in
            mark st out;
            out)
    | E_reduce_min { t_in; axes } when batched st t_in ->
        Some
          (fun () ->
            let out = reduce ~op:`Min ~axes:(Array.map taxis axes) t_in in
            mark st out;
            out)
    | E_reduce_prod { t_in; axes } when batched st t_in ->
        Some
          (fun () ->
            let out = reduce ~op:`Prod ~axes:(Array.map taxis axes) t_in in
            mark st out;
            out)
    | E_associative_scan { t_in; axis; op } when batched st t_in ->
        Some
          (fun () ->
            let out = associative_scan ~axis:(taxis axis) ~op t_in in
            mark st out;
            out)
    | E_argmax { t_in; axis; keepdims } when batched st t_in ->
        Some
          (fun () ->
            let out = argmax ~axis:(taxis axis) ~keepdims t_in in
            mark st out;
            out)
    | E_argmin { t_in; axis; keepdims } when batched st t_in ->
        Some
          (fun () ->
            let out = argmin ~axis:(taxis axis) ~keepdims t_in in
            mark st out;
            out)
    | E_sort { t_in; axis; descending } when batched st t_in ->
        Some
          (fun () ->
            let out = sort ~axis:(taxis axis) ~descending t_in in
            mark st out;
            out)
    | E_argsort { t_in; axis; descending } when batched st t_in ->
        Some
          (fun () ->
            let out = argsort ~axis:(taxis axis) ~descending t_in in
            mark st out;
            out)
    (* Gather / scatter: operands agree on rank, so all are lifted. *)
    | E_gather { data; indices; axis }
      when batched st data || batched st indices ->
        Some
          (fun () ->
            let out =
              gather (ensure_batched st data)
                (ensure_batched st indices)
                ~axis:(taxis axis)
            in
            mark st out;
            out)
    | E_scatter { data_template; indices; updates; axis; mode; unique_indices }
      when batched st data_template || batched st indices || batched st updates
      ->
        Some
          (fun () ->
            let out =
              scatter ~mode ~unique_indices
                (ensure_batched st data_template)
                ~indices:(ensure_batched st indices)
                ~updates:(ensure_batched st updates)
                ~axis:(taxis axis)
            in
            mark st out;
            out)
    | E_update { t_in; starts; v }
      when batched st t_in || batched st starts || batched st v ->
        Some
          (fun () ->
            let t = ensure_batched st t_in and v = ensure_batched st v in
            let out =
              if batched st starts then begin
                (* A window per example: along each axis the rows are gathered
                   from [v] at the example's offset, clamped, and kept where
                   they fall inside the window. *)
                let starts = ensure_batched st starts in
                let b = st.batch_size in
                let tshape = vshape st t_in and vs = vshape st v in
                let rank = Array.length tshape in
                let win = ref v and mask = ref None in
                for ax = 0 to rank - 1 do
                  let n = tshape.(ax) and len = vs.(ax) in
                  let start =
                    T.reshape [| b; 1 |] (T.slice [ T.A; T.I ax ] starts)
                  in
                  let rel =
                    T.sub (T.reshape [| 1; n |] (T.arange T.int32 0 n 1)) start
                  in
                  let inside =
                    T.logical_and (T.greater_equal_s rel 0l)
                      (T.less_s rel (Int32.of_int len))
                  in
                  let idx = T.clamp ~min:0l ~max:(Int32.of_int (len - 1)) rel in
                  let along =
                    Array.init (rank + 1) (fun d ->
                        if d = 0 then b else if d = ax + 1 then n else 1)
                  in
                  let shp =
                    Array.mapi
                      (fun d s -> if d = ax + 1 then n else s)
                      (T.shape !win)
                  in
                  win :=
                    T.take_along_axis ~axis:(ax + 1)
                      ~indices:(T.broadcast_to shp (T.reshape along idx))
                      !win;
                  let inside = T.reshape along inside in
                  mask :=
                    Some
                      (match !mask with
                      | None -> inside
                      | Some m -> T.logical_and m inside)
                done;
                match !mask with None -> v | Some mask -> T.where mask !win t
              end
              else
                (* The batch axis is never written: its start is 0 and [v]'s
                   batch extent is the whole axis. *)
                update t ~starts:(pad starts [| (1, 0) |] 0l) v
            in
            mark st out;
            out)
    (* Matrix multiplication: the frontend promotes vectors to matrices against
       virtual shapes before this effect is performed, and the backend
       broadcasts leading batch dimensions positionally. Plain matrices need no
       translation, the batch axis being the only leading dimension. When an
       operand carries leading dimensions of its own, both operands are lifted
       to the batched form at a common leading rank first; otherwise the batch
       axis of one would align against the other's first batch dimension. *)
    | E_matmul { a; b } when batched st a || batched st b ->
        Some
          (fun () ->
            let sa = vshape st a and sb = vshape st b in
            let lead s = Array.sub s 0 (Array.length s - 2) in
            let l =
              Stdlib.max (Array.length (lead sa)) (Array.length (lead sb))
            in
            let out =
              if l = 0 then matmul a b
              else
                let padded s =
                  Array.append (Array.make (l - Array.length (lead s)) 1) s
                in
                matmul
                  (to_batched st a (padded sa))
                  (to_batched st b (padded sb))
            in
            mark st out;
            out)
    (* Windowing: both address the last spatial dimensions and pass every
       leading dimension through untouched, so the batch dimension rides along
       as one more leading dimension and no parameter shifts. *)
    | E_unfold { t_in; kernel_size; stride; dilation; padding }
      when batched st t_in ->
        Some
          (fun () ->
            let out = unfold t_in ~kernel_size ~stride ~dilation ~padding in
            mark st out;
            out)
    | E_fold { t_in; output_size; kernel_size; stride; dilation; padding }
      when batched st t_in ->
        Some
          (fun () ->
            let out =
              fold t_in ~output_size ~kernel_size ~stride ~dilation ~padding
            in
            mark st out;
            out)
    (* FFT: the transformed axes shift past the batch dimension; sizes ([s]) are
       per-axis and unchanged. *)
    | E_fft { t; axes } when batched st t ->
        Some
          (fun () ->
            let out = fft t ~axes:(Array.map taxis axes) in
            mark st out;
            out)
    | E_ifft { t; axes } when batched st t ->
        Some
          (fun () ->
            let out = ifft t ~axes:(Array.map taxis axes) in
            mark st out;
            out)
    | E_rfft { t; dtype; axes } when batched st t ->
        Some
          (fun () ->
            let out = rfft t ~dtype ~axes:(Array.map taxis axes) in
            mark st out;
            out)
    | E_irfft { t; dtype; axes; s } when batched st t ->
        Some
          (fun () ->
            let out = irfft t ~axes:(Array.map taxis axes) ?s ~dtype in
            mark st out;
            out)
    | E_cholesky { t_in; _ } when batched st t_in -> Some (no_rule "cholesky")
    | E_qr { t_in; _ } when batched st t_in -> Some (no_rule "qr")
    | E_lu { t_in } when batched st t_in -> Some (no_rule "lu")
    | E_svd { t_in; _ } when batched st t_in -> Some (no_rule "svd")
    | E_eigvals { t_in } when batched st t_in -> Some (no_rule "eigvals")
    | E_eig { t_in } when batched st t_in -> Some (no_rule "eig")
    | E_eigvalsh { t_in } when batched st t_in -> Some (no_rule "eigvalsh")
    | E_eigh { t_in } when batched st t_in -> Some (no_rule "eigh")
    | E_solve_triangular { a; b; _ } when batched st a || batched st b ->
        Some (no_rule "solve_triangular")
    (* Custom rules. A custom call passes on as the custom call of its batched
       functions, as a remat does, whatever its parameters: any of them may read
       a tensor this map batches, which only this handler reads as lanes. Each
       receives the physical tensors, marks those at the batched parameters'
       positions (a parameter's tangent is batched where the parameter is) and
       runs under this handler, and the results the call returns are marked
       where they came out batched. The batched [fwd] of a custom vjp returns
       every result batched, so that each lane receives its own cotangents, and
       its batched [bwd] returns the cotangent of a parameter this map batches
       batched and that of one it does not summed over the lanes. The batched
       rule of a custom jvp returns a primal and its tangent batched together,
       as the claimer's shape check requires. *)
    | Custom.E_custom_vjp
        (Custom.Vjp_call { params_s; result_s; params; fwd; bwd }) ->
        let flags =
          List.map
            (fun (Nx.P p) -> batched st p)
            (fst (Nx.Ptree.flatten params_s params))
        in
        let fwd' ps =
          List.iter2
            (fun (Nx.P p) b -> if b then mark st p)
            (fst (Nx.Ptree.flatten params_s ps))
            flags;
          let y, res = match_with fwd ps (handler st) in
          (Nx.Ptree.map result_s (fun _ l -> ensure_batched st l) y, res)
        in
        let bwd' res cts =
          Nx.Ptree.fold result_s (fun _ c () -> mark st c) cts ();
          let gs = match_with (fun () -> bwd res cts) () (handler st) in
          Structure.map2 "Rune.custom_vjp" params_s ~this:"the parameters"
            ~that:"bwd's gradients"
            (fun _ p g ->
              if batched st p then ensure_batched st g else sum_lanes st g)
            params gs
        in
        Some
          (fun () ->
            let y =
              Custom.custom_vjp params_s result_s ~fwd:fwd' ~bwd:bwd' params
            in
            Nx.Ptree.fold result_s (fun _ l () -> mark st l) y ();
            y)
    | Custom.E_custom_jvp
        (Custom.Jvp_call { params_s; result_s; params; f; jvp }) ->
        let flags =
          List.map
            (fun (Nx.P p) -> batched st p)
            (fst (Nx.Ptree.flatten params_s params))
        in
        let mark_params ps =
          List.iter2
            (fun (Nx.P p) b -> if b then mark st p)
            (fst (Nx.Ptree.flatten params_s ps))
            flags
        in
        let out = ref [] in
        let f' ps =
          mark_params ps;
          let y = match_with f ps (handler st) in
          out :=
            List.map
              (fun (Nx.P l) -> batched st l)
              (fst (Nx.Ptree.flatten result_s y));
          y
        in
        let jvp' ps dps =
          mark_params ps;
          mark_params dps;
          let y, dy = match_with (fun () -> jvp ps dps) () (handler st) in
          let both = ref [] in
          let batch_both (type a b) (y : (a, b) t) (dy : (a, b) t) =
            let b = batched st y || batched st dy in
            both := b :: !both;
            if b then (ensure_batched st y, ensure_batched st dy) else (y, dy)
          in
          let ys = ref [] in
          let dy =
            Structure.map2 "Rune.custom_jvp" result_s ~this:"the result"
              ~that:"jvp's tangents"
              (fun _ yl dyl ->
                let yl, dyl = batch_both yl dyl in
                ys := Nx.P yl :: !ys;
                dyl)
              y dy
          in
          out := List.rev !both;
          (Nx.Ptree.rebuild result_s ~like:y (List.rev !ys), dy)
        in
        Some
          (fun () ->
            let y =
              Custom.custom_jvp params_s result_s ~f:f' ~jvp:jvp' params
            in
            List.iter2
              (fun (Nx.P l) b -> if b then mark st l)
              (fst (Nx.Ptree.flatten result_s y))
              !out;
            y)
    (* Gradient checkpointing: the remat passes on with its function batched, so
       that the enclosing context recomputes the batched computation, the
       tensors [f] captures included. The batched function receives the physical
       tensors of the arguments, those of the call or their aliases in a
       backward pass, marks those at the batched arguments' positions, and
       records which results come out batched; the results the call returns,
       which may be aliases, are marked at those positions. *)
    | Remat.E_remat (Remat.Call { params_s; result_s; params; f; residuals }) ->
        let flags =
          List.map
            (fun (Nx.P p) -> batched st p)
            (fst (Nx.Ptree.flatten params_s params))
        in
        let out = ref [] in
        let f' params =
          List.iter2
            (fun (Nx.P p) b -> if b then mark st p)
            (fst (Nx.Ptree.flatten params_s params))
            flags;
          let y = match_with f params (handler st) in
          out :=
            List.map
              (fun (Nx.P l) -> batched st l)
              (fst (Nx.Ptree.flatten result_s y));
          y
        in
        Some
          (fun () ->
            let y =
              Remat.run
                (Remat.Call { params_s; result_s; params; f = f'; residuals })
            in
            List.iter2
              (fun (Nx.P l) b -> if b then mark st l)
              (fst (Nx.Ptree.flatten result_s y))
              !out;
            y)
    | Remat.E_barrier { values; after } ->
        if not (List.exists (fun (Nx.P v) -> batched st v) values) then None
        else
          Some
            (fun () ->
              let out = Remat.barrier ~after values in
              List.iter2
                (fun (Nx.P v) (Nx.P o) -> if batched st v then mark st o)
                values out;
              out)
    (* Scan. The claim is unconditional: the body may close over batched tensors
       that appear in neither the carry nor the rows. When a stager lies beyond,
       the scan passes on batched: a batched row has the scan axis moved in
       front of the lane, a batched carry stays batched through every step, and
       the step runs the body under a nested instance of this handler. A carry
       unbatched at [init] becomes batched only once a step batches it: the step
       then aborts its run with [Grow] and the scan passes on again, batching
       it. Otherwise, the eager fold runs under a nested instance of this
       handler. *)
    | Scan.E_scan_probe -> Some Scan.probe
    | Scan.E_scan req ->
        Some
          (fun () ->
            let fold () =
              match_with (fun () -> Scan.eager req) () (handler st)
            in
            Scan.pass_on ~fold @@ fun () ->
              let exception Grow of bool list in
              let flags = List.map (fun (Nx.P l) -> batched st l) in
              let lanes carried leaves =
                List.map2
                  (fun b (Nx.P l) ->
                    if b then Nx.P (ensure_batched st l) else Nx.P l)
                  carried leaves
              in
              let rows = flags req.req_xs in
              let xs =
                List.map2
                  (fun b (Nx.P x) ->
                    if b then Nx.P (T.swapaxes 0 1 x) else Nx.P x)
                  rows req.req_xs
              in
              let rec attempt carried =
                let outputs = ref [] in
                let run c x =
                  List.iter2 (fun b (Nx.P c) -> if b then mark st c) carried c;
                  List.iter2 (fun b (Nx.P x) -> if b then mark st x) rows x;
                  let c', y =
                    match_with (fun () -> req.req_step.run c x) () (handler st)
                  in
                  let next = flags c' in
                  if List.exists2 (fun b b' -> b' && not b) carried next then
                    raise (Grow (List.map2 ( || ) carried next));
                  outputs := flags y;
                  (lanes carried c', y)
                in
                match
                  Effect.perform
                    (Scan.E_scan
                       {
                         req with
                         req_carry = lanes carried req.req_carry;
                         req_xs = xs;
                         req_step = { run };
                       })
                with
                | res ->
                    List.iter2
                      (fun b (Nx.P c) -> if b then mark st c)
                      carried res.r_carry;
                    let r_ys =
                      List.map2
                        (fun b (Nx.P y) ->
                          if b then (
                            let y = T.swapaxes 0 1 y in
                            mark st y;
                            Nx.P y)
                          else Nx.P y)
                        !outputs res.r_ys
                    in
                    { res with r_ys }
                (* The aborted run's slot tensors are never reached again. *)
                | exception Grow carried -> attempt carried
              in
              attempt (flags req.req_carry))
    (* An addition to a total is the sum of its lanes' additions. *)
    | Total.E_add (t, v) -> Some (fun () -> Total.add t (sum_lanes st v))
    | Nx_quant.Effect.E_quant { w; op } when quant_batched st w op ->
        Some
          (fun () ->
            let out = quant st w op in
            mark st out;
            out)
    (* Operations on constants, and effects from other libraries, fall through.
       A new Nx tensor operation must be added to this match: an unmatched
       batched operand would silently produce wrong shapes. *)
    | _ -> None
  in
  let effc : type c. c Effect.t -> ((c, _) continuation -> _) option =
   fun eff -> Option.map Gate.deliver (rule eff)
  in
  { retc = Fun.id; exnc = raise; effc }
