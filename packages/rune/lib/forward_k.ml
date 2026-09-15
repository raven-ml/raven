(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Batched forward-mode differentiation as an effect handler over Nx operations.

   [Forward] pushes a single tangent through the computation; this handler
   pushes [k] at once. Every tensor's tangent is a batch — the [k] directions
   stacked on a new leading axis — and every rule computes all [k] directional
   derivatives with one batched pass over the same operations that computed the
   primal: per operation, primal work ×1 and tangent work ×[k], with the primal
   computed once. Batching vmap around [Forward] computes the same mathematical
   object but re-performs the primal once per lane, which for the batched
   matmuls of a recurrence multiplies the dominant cost by [k]; the reference
   composition is therefore the *test* oracle, not the implementation.

   The rules are the ones in forward.ml — they are linear in the tangents, so
   their formulas are unchanged — with every shape and axis parameter translated
   for the extra leading axis: reshapes and expansions gain a leading [k] entry,
   axis parameters shift by one, pads and shrinks gain a leading (0,0)/(0,k)
   entry, and index tensors are lifted with a broadcast lane axis so gather and
   scatter rank contracts hold.

   Two rules need more than parameter translation, because the lane axis is new
   and the backend broadcasts batch dimensions positionally:

   - Elementwise broadcasting combines operands of different ranks (a row
   against a matrix), and a lane-stacked tangent would broadcast its lane axis
   against the other operand's first dimension. Every elementwise rule lifts its
   operands' tangents to the output's broadcast shape first, keeping the lane
   axis ahead of every dimension the primal broadcast may grow. - matmul
   broadcasts leading batch dimensions positionally too, so an operand carrying
   leading dimensions of its own would align its first one against the lanes;
   the rule views every active operand at the output's leading layout [lanes ::
   lead]. Both lifts are no-ops for plain matrix operands — the recurrent [S;M]
   × [M;K] path.

   The store of tangents is keyed on ephemerons (Tangent_store), so the tangents
   of past steps of a long loop are collected together with their primals and
   memory stays proportional to the live working set rather than the whole
   history. [live_entries] exposes the live binding count; the
   [with_active_store] registry makes it reachable from inside the
   differentiated function.

   Excluded relative to forward.ml: [cholesky] and [triangular_solve] (the
   backend solves require exactly-matching leading batch dimensions, so the
   primal operands would need lifting along the lane axis — future work), and
   the operations forward mode has no rule for anyway ([qr], [svd], [eig*]).

   Every Nx effect constructor is matched explicitly, with the same three
   deliberate categories as forward.ml: zero-derivative operations fall through
   untracked; operations with no rule yet raise when an input is active;
   mutation always raises. The rule table parallels forward.ml: a new Nx
   operation needs a rule in both files, in the same category, and the
   jvp_k-versus-vmap∘jvp reference tests must cover it. *)

open Nx_effect
module T = Nx

let err_no_rule op =
  invalid_arg
    (Printf.sprintf
       "Rune: the batched tangent of %s is not implemented; detach its input \
        if differentiation should not flow through it"
       op)

(* Axis parameters address the primal's dimensions, which sit after the lane
   axis: non-negative axes shift by one, negative axes are unchanged (vmap's
   [taxis]; the frontend resolves the negatives it produces before the effect
   fires, so in practice every axis is non-negative here). *)
let taxis ax = if ax >= 0 then ax + 1 else ax
let taxis_list axes = List.map taxis (Array.to_list axes)
let kshape k s = Array.append [| k |] s

(* [view_at out x dx] is the tangent [dx] of [x] viewed at the broadcast shape
   of [out]. [dx] has shape [lanes :: shape x] and [out] the shape the primal
   broadcast produced, so the lift inserts [x]'s leading 1-dims between the lane
   axis and [x]'s own dims. Stored tangents are contiguous, so the reshape is a
   view. *)
let view_at lanes out x dx =
  let target = Tangent_store.shape_of out in
  let s = Tangent_store.shape_of x in
  if s = target then dx
  else begin
    let lead = Array.make (Array.length target - Array.length s) 1 in
    expand
      (reshape dx (Array.concat [ [| lanes |]; lead; s ]))
      (kshape lanes target)
  end

(* Index tensors address the same positions in every lane: give them a broadcast
   lane axis so gather and scatter rank contracts hold. *)
let lift_indices lanes idx =
  let s = Tangent_store.shape_of idx in
  expand (reshape (contiguous idx) (Array.append [| 1 |] s)) (kshape lanes s)

(* The tangent stores of the jvp_k-family calls currently running, innermost
   first. [live_entries] reads the innermost one; it is the memory probe that
   makes the bounded-lifetime property testable from inside the differentiated
   function. *)
let active : Tangent_store.t list ref = ref []

let with_active_store store f =
  active := store :: !active;
  Fun.protect f ~finally:(fun () -> active := List.tl !active)

let live_entries () =
  match !active with store :: _ -> Tangent_store.live_entries store | [] -> 0

let rec handler : type r. Tangent_store.t -> (r, r) Effect.Deep.handler =
 fun tangents ->
  let lanes = Tangent_store.k tangents in
  let open Effect.Deep in
  let tangent x = Tangent_store.find tangents x in
  let active x = Option.is_some (tangent x) in
  (* Inactive operands get zero tangent batches: [lanes] stacked on the
     operand's own shape. *)
  let tan_or_zeros (type a b) (x : (a, b) t) : (a, b) t =
    match tangent x with
    | Some dx -> dx
    | None -> T.zeros (T.dtype x) (kshape lanes (Tangent_store.shape_of x))
  in
  (* Materialize stored tangents — rule outputs can be lazy views (broadcasts,
     transposes), later rules may reshape them — and validate the lane-axis
     invariant, which turns a miscomposed batch dimension into an immediate,
     attributable failure instead of silent garbage. *)
  let set_tangent out v = Tangent_store.set tangents out (T.contiguous v) in

  (* [lift1 k out x dfun] stores [dfun dx] as the tangent batch of [out] when
     [x] has tangent [dx]; unary rules keep [out]'s shape, so [dx] needs no view
     lift. *)
  let lift1 (type a b c d) k (out : (a, b) t) (x : (c, d) t)
      (dfun : (c, d) t -> (a, b) t) =
    (match tangent x with None -> () | Some dx -> set_tangent out (dfun dx));
    continue k out
  in

  (* [lift2 k out a b make] stores [make da db] as the tangent batch of [out]
     when either input is active; both tangents are first lifted to [out]'s
     (broadcast) shape, which keeps the lane axis ahead of every dimension the
     primal broadcast grew. *)
  let lift2 (type a b) k (out : (a, b) t) (a_in : (a, b) t) (b_in : (a, b) t)
      (make : (a, b) t -> (a, b) t -> (a, b) t) =
    if active a_in || active b_in then
      set_tangent out
        (make
           (view_at lanes out a_in (tan_or_zeros a_in))
           (view_at lanes out b_in (tan_or_zeros b_in)));
    continue k out
  in

  let no_rule (type c) k (op : string) (inputs_active : bool) (out : unit -> c)
      =
    if inputs_active then err_no_rule op else continue k (out ())
  in

  let effc : type c. c Effect.t -> ((c, _) continuation -> _) option =
   fun eff ->
    if not !Gate.enabled then None
    else
      match eff with
      (* Constants: creation, RNG, metadata. Fresh outputs are inactive. *)
      | E_view _ -> None
      | E_to_host _ -> None
      | E_buffer _ -> None
      | E_const_scalar _ -> None
      | E_from_host _ -> None
      | E_threefry _ -> None
      | E_to_device _ -> None
      (* Zero derivative: boolean, bitwise and integer results. *)
      | E_cmpeq _ -> None
      | E_cmpne _ -> None
      | E_cmplt _ -> None
      | E_cmple _ -> None
      | E_xor _ -> None
      | E_or _ -> None
      | E_and _ -> None
      | E_idiv _ -> None
      | E_argmax _ -> None
      | E_argmin _ -> None
      | E_argsort _ -> None
      (* Zero derivative: piecewise-constant real functions. *)
      | E_sign _ -> None
      | E_trunc _ -> None
      | E_ceil _ -> None
      | E_floor _ -> None
      | E_round _ -> None
      (* Mutation is incompatible with identity-keyed tracking. *)
      | E_assign _ ->
          Some
            (fun _k ->
              invalid_arg
                "in-place mutation (set_item, set_slice, blit, assign) cannot \
                 be used inside jvp_k — use scatter instead")
      (* The tangent query: answered from this store with the [k]-lane batch,
         so a consumer inside the scope reads exactly the tangents this
         differentiation maintains. A tensor the store does not track is
         answered [None] rather than passed outward: the innermost forward mode
         owns the tangent convention in its extent (a lane batch here, a single
         tangent under [Forward] — see tangent_query.ml). *)
      | Tangent_query.E_tangent x -> Some (fun k -> continue k (tangent x))
      (* [jit] asks whether to step aside; a batched forward mode is observing. *)
      | Gate.E_transforming -> Some (fun k -> continue k true)
      (* Scan: forward mode has no staged rule yet, so run the eager fold under
         a nested instance of this handler — every step's operations flow
         through it and acquire their tangent batches as they always did.
         Claiming [E_scan] here obliges answering the probe with [false]. *)
      | Scan.E_scan_probe -> Some (fun k -> continue k false)
      | Scan.E_scan req ->
          Some
            (fun k ->
              let res : Scan.scan_res =
                Effect.Deep.match_with
                  (fun () -> Scan.eager req)
                  () (handler tangents)
              in
              continue k res)
      (* Binary arithmetic *)
      | E_add { a; b } -> Some (fun k -> lift2 k (add a b) a b T.add)
      | E_sub { a; b } -> Some (fun k -> lift2 k (sub a b) a b T.sub)
      | E_mul { a; b } ->
          Some
            (fun k ->
              lift2 k (mul a b) a b (fun da db ->
                  T.add (T.mul da b) (T.mul a db)))
      | E_fdiv { a; b } ->
          Some
            (fun k ->
              lift2 k (fdiv a b) a b (fun da db ->
                  T.sub (T.div da b) (T.mul (T.div a (T.mul b b)) db)))
      | E_pow { a; b } ->
          Some
            (fun k ->
              let out = pow a b in
              lift2 k out a b (fun da db ->
                  T.add
                    (T.mul da (Derivs.pow_wrt_base a b))
                    (T.mul db (Derivs.pow_wrt_exp a out))))
      | E_max { a; b } ->
          Some
            (fun k ->
              let out = max a b in
              lift2 k out a b (fun da db ->
                  let mask = T.cast (dtype out) (T.greater a b) in
                  T.add (T.mul da mask)
                    (T.mul db (T.rsub_s (Derivs.one_like mask) mask))))
      | E_min { a; b } ->
          Some
            (fun k ->
              let out = min a b in
              lift2 k out a b (fun da db ->
                  let mask = T.cast (dtype out) (T.less a b) in
                  T.add (T.mul da mask)
                    (T.mul db (T.rsub_s (Derivs.one_like mask) mask))))
      | E_atan2 { a; b } ->
          Some
            (fun k ->
              lift2 k (atan2 a b) a b (fun da db ->
                  let denom = T.add (T.mul a a) (T.mul b b) in
                  T.sub (T.mul da (T.div b denom)) (T.mul db (T.div a denom))))
      | E_mod { a; b } ->
          Some
            (fun k ->
              no_rule k "mod" (active a || active b) (fun () -> mod_ a b))
      (* Unary arithmetic *)
      | E_neg { t_in } -> Some (fun k -> lift1 k (neg t_in) t_in T.neg)
      | E_sin { t_in } ->
          Some
            (fun k -> lift1 k (sin t_in) t_in (fun dx -> T.mul dx (T.cos t_in)))
      | E_cos { t_in } ->
          Some
            (fun k ->
              lift1 k (cos t_in) t_in (fun dx -> T.mul dx (T.neg (T.sin t_in))))
      | E_tan { t_in } ->
          Some
            (fun k ->
              lift1 k (tan t_in) t_in (fun dx -> T.mul dx (Derivs.tan' t_in)))
      | E_asin { t_in } ->
          Some
            (fun k ->
              lift1 k (asin t_in) t_in (fun dx -> T.mul dx (Derivs.asin' t_in)))
      | E_acos { t_in } ->
          Some
            (fun k ->
              lift1 k (acos t_in) t_in (fun dx ->
                  T.mul dx (T.neg (Derivs.asin' t_in))))
      | E_atan { t_in } ->
          Some
            (fun k ->
              lift1 k (atan t_in) t_in (fun dx -> T.mul dx (Derivs.atan' t_in)))
      | E_sinh { t_in } ->
          Some
            (fun k ->
              lift1 k (sinh t_in) t_in (fun dx -> T.mul dx (T.cosh t_in)))
      | E_cosh { t_in } ->
          Some
            (fun k ->
              lift1 k (cosh t_in) t_in (fun dx -> T.mul dx (T.sinh t_in)))
      | E_tanh { t_in } ->
          Some
            (fun k ->
              let out = tanh t_in in
              lift1 k out t_in (fun dx -> T.mul dx (Derivs.tanh' out)))
      | E_exp { t_in } ->
          Some
            (fun k ->
              let out = exp t_in in
              lift1 k out t_in (fun dx -> T.mul dx out))
      | E_log { t_in } ->
          Some
            (fun k ->
              lift1 k (log t_in) t_in (fun dx -> T.mul dx (T.recip t_in)))
      | E_sqrt { t_in } ->
          Some
            (fun k ->
              let out = sqrt t_in in
              lift1 k out t_in (fun dx -> T.mul dx (Derivs.sqrt' out)))
      | E_recip { t_in } ->
          Some
            (fun k ->
              lift1 k (recip t_in) t_in (fun dx ->
                  T.mul dx (Derivs.recip' t_in)))
      (* On complex dtypes [abs] is the modulus: real-valued, and not
         holomorphic. Its pushforward conjugates the direction — going through
         [sign z] itself would flip the sign of the imaginary contribution — and
         keeps the real part of the product, since a real-valued output cannot
         move in the imaginary direction. Both are the identity on real
         dtypes. *)
      | E_abs { t_in } ->
          Some
            (fun k ->
              lift1 k (abs t_in) t_in (fun dx ->
                  Derivs.real_part (T.mul dx (T.conjugate (T.sign t_in)))))
      | E_erf { t_in } ->
          Some
            (fun k ->
              lift1 k (erf t_in) t_in (fun dx -> T.mul dx (Derivs.erf' t_in)))
      (* Selection *)
      | E_where { condition; if_true; if_false } ->
          Some
            (fun k ->
              let out = where condition if_true if_false in
              if active if_true || active if_false then begin
                let mask = T.cast (dtype out) condition in
                let dt_ = view_at lanes out if_true (tan_or_zeros if_true) in
                let df = view_at lanes out if_false (tan_or_zeros if_false) in
                set_tangent out
                  (T.add (T.mul dt_ mask)
                     (T.mul df (T.rsub_s (Derivs.one_like mask) mask)))
              end;
              continue k out)
      (* Movement: linear ops apply to the tangent batch unchanged, with the
         shape and axis parameters translated for the lane axis. *)
      | E_reshape { t_in; new_shape } ->
          Some
            (fun k ->
              lift1 k (reshape t_in new_shape) t_in (fun dx ->
                  reshape dx (kshape lanes new_shape)))
      | E_permute { t_in; axes } ->
          Some
            (fun k ->
              lift1 k (permute t_in axes) t_in (fun dx ->
                  permute dx
                    (Array.append [| 0 |] (Array.map (fun d -> d + 1) axes))))
      | E_expand { t_in; new_target_shape } ->
          Some
            (fun k ->
              lift1 k (expand t_in new_target_shape) t_in (fun dx ->
                  expand dx (kshape lanes new_target_shape)))
      | E_pad { t_in; padding_config; fill_value } ->
          Some
            (fun k ->
              (* The fill value is a constant: the tangent pads with zero. *)
              lift1 k (pad t_in padding_config fill_value) t_in (fun dx ->
                  pad dx
                    (Array.append [| (0, 0) |] padding_config)
                    (Nx_core.Dtype.zero (dtype t_in))))
      | E_shrink { t_in; limits } ->
          Some
            (fun k ->
              lift1 k (shrink t_in limits) t_in (fun dx ->
                  shrink dx (Array.append [| (0, lanes) |] limits)))
      | E_flip { t_in; dims_to_flip } ->
          Some
            (fun k ->
              lift1 k (flip t_in dims_to_flip) t_in (fun dx ->
                  flip dx (Array.append [| false |] dims_to_flip)))
      | E_sliding_window { t_in; axis; window; step } ->
          Some
            (fun k ->
              lift1 k (sliding_window t_in ~axis ~window ~step) t_in (fun dx ->
                  sliding_window dx ~axis:(taxis axis) ~window ~step))
      | E_cat { t_list; axis } ->
          Some
            (fun k ->
              let out = cat t_list ~axis in
              if List.exists active t_list then
                set_tangent out
                  (cat (List.map tan_or_zeros t_list) ~axis:(taxis axis));
              continue k out)
      | E_cast { t_in; target_dtype } ->
          Some
            (fun k ->
              lift1 k (cast ~dtype:target_dtype t_in) t_in (fun dx ->
                  T.cast target_dtype dx))
      | E_contiguous { t_in } ->
          Some (fun k -> lift1 k (contiguous t_in) t_in Fun.id)
      | E_copy { t_in } -> Some (fun k -> lift1 k (copy t_in) t_in Fun.id)
      (* Reductions *)
      | E_reduce_sum { t_in; axes } ->
          Some
            (fun k ->
              lift1 k (reduce ~op:`Sum ~axes t_in) t_in (fun dx ->
                  T.sum dx ~axes:(taxis_list axes)))
      | E_reduce_max { t_in; axes } ->
          Some
            (fun k ->
              let out = reduce ~op:`Max ~axes t_in in
              lift1 k out t_in (fun dx ->
                  let shape_in = Tangent_store.shape_of t_in in
                  let out_bc =
                    let kept =
                      T.max t_in ~axes:(Array.to_list axes) ~keepdims:true
                    in
                    T.broadcast_to shape_in kept
                  in
                  let mask = T.cast (dtype out) (T.equal t_in out_bc) in
                  T.sum (T.mul dx mask) ~axes:(taxis_list axes)))
      | E_reduce_min { t_in; axes } ->
          Some
            (fun k ->
              let out = reduce ~op:`Min ~axes t_in in
              lift1 k out t_in (fun dx ->
                  let shape_in = Tangent_store.shape_of t_in in
                  let out_bc =
                    let kept =
                      T.min t_in ~axes:(Array.to_list axes) ~keepdims:true
                    in
                    T.broadcast_to shape_in kept
                  in
                  let mask = T.cast (dtype out) (T.equal t_in out_bc) in
                  T.sum (T.mul dx mask) ~axes:(taxis_list axes)))
      | E_reduce_prod { t_in; axes } ->
          Some
            (fun k ->
              let out = reduce ~op:`Prod ~axes t_in in
              lift1 k out t_in (fun dx ->
                  let shape_in = Tangent_store.shape_of t_in in
                  let out_bc =
                    let kept =
                      T.prod t_in ~axes:(Array.to_list axes) ~keepdims:true
                    in
                    T.broadcast_to shape_in kept
                  in
                  T.sum (T.mul (T.div out_bc t_in) dx) ~axes:(taxis_list axes)))
      (* Sorting: a sort is a gather at the argsort indices. *)
      | E_sort { t_in; axis; descending } ->
          Some
            (fun k ->
              lift1 k (sort ~axis ~descending t_in) t_in (fun dx ->
                  let indices = argsort ~axis ~descending t_in in
                  gather dx (lift_indices lanes indices) ~axis:(taxis axis)))
      (* Scans *)
      | E_associative_scan { t_in; axis; op } ->
          Some
            (fun k ->
              let out = associative_scan ~axis ~op t_in in
              lift1 k out t_in (fun dx ->
                  match op with
                  | `Sum -> associative_scan ~axis:(taxis axis) ~op:`Sum dx
                  | `Prod ->
                      (* d cumprod_k = cumprod_k * sum_{i<=k} dx_i / x_i;
                         requires nonzero inputs, like the reverse rule. *)
                      let ratio = T.div dx t_in in
                      T.mul out
                        (associative_scan ~axis:(taxis axis) ~op:`Sum ratio)
                  | `Max | `Min ->
                      (* The tangent flows from positions where the running
                         extremum strictly improves; the shifted extremum and
                         the mask are primal-shaped, so the lane axis only
                         broadcasts in the final multiply. *)
                      let shape = Tangent_store.shape_of out in
                      let ndim = Array.length shape in
                      let axis_norm = if axis < 0 then axis + ndim else axis in
                      let dt = dtype t_in in
                      let boundary =
                        match op with
                        | `Max -> Nx_core.Dtype.min_value dt
                        | _ -> Nx_core.Dtype.max_value dt
                      in
                      let pad_left =
                        Array.mapi
                          (fun i _ -> if i = axis_norm then (1, 0) else (0, 0))
                          shape
                      in
                      let padded = T.pad pad_left boundary out in
                      let slice_specs =
                        Array.map (fun dim -> T.R (0, dim)) shape
                      in
                      let shifted =
                        T.slice (Array.to_list slice_specs) padded
                      in
                      let active_mask =
                        match op with
                        | `Max -> T.greater out shifted
                        | _ -> T.less out shifted
                      in
                      (* Positions where the extremum does not improve keep a
                         zero tangent rather than carrying the previous
                         extremum's tangent; this matches the reverse rule (they
                         are transposes of each other). *)
                      T.mul dx (T.cast dt active_mask)))
      (* Gather / scatter: indices address the same positions in every lane. *)
      | E_gather { data; indices; axis } ->
          Some
            (fun k ->
              lift1 k (gather data indices ~axis) data (fun dx ->
                  gather dx (lift_indices lanes indices) ~axis:(taxis axis)))
      | E_scatter
          { data_template; indices; updates; axis; mode; unique_indices } ->
          Some
            (fun k ->
              let out =
                scatter ~mode ~unique_indices data_template ~indices ~updates
                  ~axis
              in
              if active data_template || active updates then begin
                let idx = lift_indices lanes indices in
                let d_template =
                  match mode with
                  | `Add -> tan_or_zeros data_template
                  | `Set ->
                      (* The written positions drop the template's tangent. *)
                      let mask =
                        scatter ~mode:`Set ~unique_indices
                          (T.ones_like data_template)
                          ~indices ~updates:(T.zeros_like updates) ~axis
                      in
                      T.mul (tan_or_zeros data_template) mask
                in
                let d_updates =
                  scatter ~mode ~unique_indices
                    (T.zeros (dtype data_template)
                       (kshape lanes (Tangent_store.shape_of data_template)))
                    ~indices:idx ~updates:(tan_or_zeros updates)
                    ~axis:(taxis axis)
                in
                set_tangent out (T.add d_template d_updates)
              end;
              continue k out)
      (* Windowing: unfold and fold are linear and pass leading dimensions
         through untouched, so the lane axis rides along. *)
      | E_unfold { t_in; kernel_size; stride; dilation; padding } ->
          Some
            (fun k ->
              lift1 k (unfold t_in ~kernel_size ~stride ~dilation ~padding) t_in
                (fun dx -> unfold dx ~kernel_size ~stride ~dilation ~padding))
      | E_fold { t_in; output_size; kernel_size; stride; dilation; padding } ->
          Some
            (fun k ->
              lift1 k
                (fold t_in ~output_size ~kernel_size ~stride ~dilation ~padding)
                t_in (fun dx ->
                  fold dx ~output_size ~kernel_size ~stride ~dilation ~padding))
      (* Matrix multiplication. The product rule is [matmul da b + matmul a db]
         as in forward.ml, but the backend broadcasts the operands' leading
         batch dimensions positionally, and the lane axis is new: it must sit
         outside them. When either operand carries leading dimensions of its
         own, a raw [matmul da b] would align the lanes against [b]'s first
         batch dimension. View every active operand at the output's leading
         layout [lanes :: lead] — a primal gains a broadcast lane axis, a
         tangent grows its own leading dimensions up to [lead] — and the product
         rule holds there unchanged. Plain matrix operands ([lead] empty — the
         hot [S;M] × [M;K] of a recurrence) lift to themselves. *)
      | E_matmul { a; b } ->
          Some
            (fun k ->
              let out = matmul a b in
              (match (tangent a, tangent b) with
              | None, None -> ()
              | da, db ->
                  let sa = Tangent_store.shape_of a in
                  let sb = Tangent_store.shape_of b in
                  let ma = Array.sub sa (Array.length sa - 2) 2 in
                  let mb = Array.sub sb (Array.length sb - 2) 2 in
                  let lead_a = Array.sub sa 0 (Array.length sa - 2) in
                  let lead_b = Array.sub sb 0 (Array.length sb - 2) in
                  let out_shape = Tangent_store.shape_of out in
                  let lead =
                    Array.sub out_shape 0 (Array.length out_shape - 2)
                  in
                  let pad n = Array.make (Array.length lead - n) 1 in
                  let lift_tangent t own mat =
                    if Array.length lead = 0 then t
                    else
                      expand
                        (reshape (contiguous t)
                           (Array.concat
                              [ [| lanes |]; pad (Array.length own); own; mat ]))
                        (Array.concat [ [| lanes |]; lead; mat ])
                  in
                  let lift_primal x own mat =
                    if Array.length lead = 0 then x
                    else
                      expand
                        (reshape (contiguous x)
                           (Array.concat
                              [ [| 1 |]; pad (Array.length own); own; mat ]))
                        (Array.concat [ [| lanes |]; lead; mat ])
                  in
                  let terms =
                    List.filter_map Fun.id
                      [
                        Option.map
                          (fun da ->
                            matmul
                              (lift_tangent da lead_a ma)
                              (lift_primal b lead_b mb))
                          da;
                        Option.map
                          (fun db ->
                            matmul (lift_primal a lead_a ma)
                              (lift_tangent db lead_b mb))
                          db;
                      ]
                  in
                  let tan =
                    match terms with
                    | [ t ] -> t
                    | [ t1; t2 ] -> T.add t1 t2
                    | _ -> assert false
                  in
                  set_tangent out tan);
              continue k out)
      (* FFT: linear operations apply to the tangent; the transformed axes shift
         past the lane axis. *)
      | E_fft { t; axes } ->
          Some
            (fun k ->
              lift1 k (fft t ~axes) t (fun dx ->
                  fft dx ~axes:(Array.map taxis axes)))
      | E_ifft { t; axes } ->
          Some
            (fun k ->
              lift1 k (ifft t ~axes) t (fun dx ->
                  ifft dx ~axes:(Array.map taxis axes)))
      | E_rfft { t; dtype; axes } ->
          Some
            (fun k ->
              lift1 k (rfft t ~dtype ~axes) t (fun dx ->
                  rfft dx ~dtype ~axes:(Array.map taxis axes)))
      | E_irfft { t; dtype; axes; s } ->
          Some
            (fun k ->
              lift1 k (irfft t ~axes ?s ~dtype) t (fun dx ->
                  irfft dx ~axes:(Array.map taxis axes) ?s ~dtype))
      | E_psum { t_in } ->
          Some
            (fun k -> no_rule k "psum" (active t_in) (fun () -> op_psum t_in))
      (* Linear algebra. [cholesky] and [triangular_solve] are excluded for now:
         their backends require exactly-matching leading batch dimensions on
         both operands, so the primal factorizations would need lifting along
         the lane axis before the single-tangent rules translate. *)
      | E_cholesky { t_in; upper } ->
          Some
            (fun k ->
              if active t_in then err_no_rule "cholesky"
              else continue k (cholesky ~upper t_in))
      | E_triangular_solve { a; b; upper; transpose; unit_diag } ->
          Some
            (fun k ->
              if active a || active b then err_no_rule "triangular_solve"
              else
                continue k (triangular_solve ~upper ~transpose ~unit_diag a b))
      | E_qr { t_in; reduced } ->
          Some
            (fun k ->
              if active t_in then err_no_rule "qr"
              else continue k (qr ~reduced t_in))
      | E_svd { t_in; full_matrices } ->
          Some
            (fun k ->
              no_rule k "svd" (active t_in) (fun () -> svd ~full_matrices t_in))
      | E_eigvals { t_in } ->
          Some
            (fun k ->
              no_rule k "eigvals" (active t_in) (fun () -> eigvals t_in))
      | E_eig { t_in } ->
          Some (fun k -> no_rule k "eig" (active t_in) (fun () -> eig t_in))
      | E_eigvalsh { t_in } ->
          Some
            (fun k ->
              no_rule k "eigvalsh" (active t_in) (fun () -> eigvalsh t_in))
      | E_eigh { t_in } ->
          Some (fun k -> no_rule k "eigh" (active t_in) (fun () -> eigh t_in))
      (* Custom rules. A custom jvp rule is written for a single tangent, so
         present each lane's tangent to it by vectorizing the rule call over the
         lane axis: the lane-stacked tangent leaves become vmap's batch, the
         parameters are constants of the map, and the rule body runs once as a
         batched translation of itself. The rule's ops flow to the handlers
         outside this one, as any rule computation does; its result gains the
         lane axis back here. *)
      | Custom.E_custom_jvp (Custom.Jvp_call { tree; params; f; jvp }) ->
          Some
            (fun k ->
              let (module Q) = tree in
              let any = ref false in
              Q.iter (fun leaf -> if active leaf then any := true) params;
              if not !any then continue k (f params)
              else begin
                let dparams = Q.map (fun leaf -> tan_or_zeros leaf) params in
                let st = Vmap.create ~batch_size:lanes in
                Q.iter (fun leaf -> Vmap.mark st leaf) dparams;
                let y, dy =
                  match_with (fun () -> jvp params dparams) () (Vmap.handler st)
                in
                let dy =
                  if Vmap.batched st dy then dy
                  else
                    (* The rule ignored its tangents: broadcast its single
                       result over the lanes. *)
                    T.broadcast_to (kshape lanes (Tangent_store.shape_of dy)) dy
                in
                let y =
                  if Vmap.batched st y then
                    (* The rule computed its primal through the batched
                       tangents; every lane computed the same value. *)
                    T.slice [ T.I 0 ] y
                  else y
                in
                set_tangent y dy;
                continue k y
              end)
      | Custom.E_custom_vjp (Custom.Vjp_call { tree; params; fwd; _ }) ->
          Some
            (fun k ->
              let (module Q) = tree in
              let any = ref false in
              Q.iter (fun leaf -> if active leaf then any := true) params;
              if !any then
                invalid_arg
                  "Rune: a custom_vjp function is not forward-differentiable; \
                   define a custom_jvp rule instead"
              else continue k (fst (fwd params)))
      (* Effects from other libraries fall through. A new Nx tensor operation
         must be added to this match: an unmatched tensor effect would be
         differentiated as a constant. *)
      | _ -> None
  in
  { retc = Fun.id; exnc = raise; effc }
