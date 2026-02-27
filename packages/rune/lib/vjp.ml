(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Reverse-mode automatic differentiation (VJP). Runs the forward computation
   under an effect handler that records a tape, then propagates cotangents
   backward through the tape as the continuation stack unwinds. *)

open Nx_core
open Nx_effect
module T = Nx

(* Tape types *)

type any_tensor = Any : ('a, 'b) t -> any_tensor

let unwrap (type a b) (_ : (a, b) Dtype.t) (Any t) : (a, b) t = Obj.magic t

type ('a, 'b) t_with_grad = {
  v : ('a, 'b) t;
  mutable grad : ('a, 'b) t;
  id : int;
}

type any_twg = Any_twg : ('a, 'b) t_with_grad -> any_twg

let unwrap_twg (type a b) (_ : (a, b) Dtype.t) (Any_twg twg) :
    (a, b) t_with_grad =
  Obj.magic twg

let twg_id_counter = ref 0

let fresh_twg_id () =
  incr twg_id_counter;
  !twg_id_counter

(* Effect handler *)

let make_handler tape seed_output =
  let open Effect.Deep in
  let get_or_init (type a b) (t : (a, b) t) : (a, b) t_with_grad =
    match Autodiff.Physical_tbl.find tape t with
    | Some (Any_twg twg) -> unwrap_twg (dtype t) (Any_twg twg)
    | None ->
        let id = fresh_twg_id () in
        let twg = { v = t; grad = T.zeros_like t; id } in
        Autodiff.Physical_tbl.add tape t (Any_twg twg);
        twg
  in

  let effc : type c. c Effect.t -> ((c, _) continuation -> _) option =
   fun eff ->
    if not !Autodiff.autodiff_enabled then None
    else
      match eff with
      (* Sources *)
      | E_const_scalar { context = _; value; dtype = dt } ->
          Some
            (fun k ->
              let res = T.full dt [||] value in
              let fwd = continue k res in
              let _ = get_or_init res in
              fwd)
      | E_from_host { context = ctx; array } ->
          Some
            (fun k ->
              let res = from_host ctx array in
              let fwd = continue k res in
              let _ = get_or_init res in
              fwd)
      | E_buffer { context = ctx; dtype = dt; size_in_elements } ->
          Some
            (fun k ->
              let res = buffer ctx dt [| size_in_elements |] in
              let fwd = continue k res in
              let _ = get_or_init res in
              fwd)
      | E_threefry { key; ctr } ->
          Some
            (fun k ->
              let res =
                let out = buffer (context ctr) (dtype ctr) (T.shape ctr) in
                threefry ~out key ctr;
                out
              in
              let fwd = continue k res in
              let _ = get_or_init res in
              fwd)
      (* Binary Arithmetic *)
      | E_add { out; a; b } ->
          Some
            (fun k ->
              add ~out a b;
              let fwd = continue k () in
              let twg_a = get_or_init a in
              let twg_b = get_or_init b in
              let twg_out = get_or_init out in
              let g = twg_out.grad in
              twg_a.grad <-
                T.add twg_a.grad (Autodiff.unbroadcast_grad g (T.shape a));
              twg_b.grad <-
                T.add twg_b.grad (Autodiff.unbroadcast_grad g (T.shape b));
              fwd)
      | E_sub { out; a; b } ->
          Some
            (fun k ->
              sub ~out a b;
              let fwd = continue k () in
              let twg_a = get_or_init a in
              let twg_b = get_or_init b in
              let twg_out = get_or_init out in
              let g = twg_out.grad in
              twg_a.grad <-
                T.add twg_a.grad (Autodiff.unbroadcast_grad g (T.shape a));
              twg_b.grad <-
                T.add twg_b.grad
                  (Autodiff.unbroadcast_grad (T.neg g) (T.shape b));
              fwd)
      | E_mul { out; a; b } ->
          Some
            (fun k ->
              mul ~out a b;
              let fwd = continue k () in
              let twg_a = get_or_init a in
              let twg_b = get_or_init b in
              let twg_out = get_or_init out in
              let g = twg_out.grad in
              twg_a.grad <-
                T.add twg_a.grad
                  (Autodiff.unbroadcast_grad (T.mul g b) (T.shape a));
              twg_b.grad <-
                T.add twg_b.grad
                  (Autodiff.unbroadcast_grad (T.mul g a) (T.shape b));
              fwd)
      | E_fdiv { out; a; b } ->
          Some
            (fun k ->
              div ~out a b;
              let fwd = continue k () in
              let twg_a = get_or_init a in
              let twg_b = get_or_init b in
              let twg_out = get_or_init out in
              let g = twg_out.grad in
              let ga = T.div g b in
              let gb = T.mul (T.neg g) (T.div a (T.mul b b)) in
              twg_a.grad <-
                T.add twg_a.grad (Autodiff.unbroadcast_grad ga (T.shape a));
              twg_b.grad <-
                T.add twg_b.grad (Autodiff.unbroadcast_grad gb (T.shape b));
              fwd)
      | E_pow { out; a; b } ->
          Some
            (fun k ->
              pow ~out a b;
              let fwd = continue k () in
              let twg_a = get_or_init a in
              let twg_b = get_or_init b in
              let twg_out = get_or_init out in
              let g = twg_out.grad in
              let ga = T.mul g (Autodiff.deriv_pow_wrt_base a b) in
              let gb = T.mul g (Autodiff.deriv_pow_wrt_exp a out) in
              twg_a.grad <-
                T.add twg_a.grad (Autodiff.unbroadcast_grad ga (T.shape a));
              twg_b.grad <-
                T.add twg_b.grad (Autodiff.unbroadcast_grad gb (T.shape b));
              fwd)
      | E_max { out; a; b } ->
          Some
            (fun k ->
              max ~out a b;
              let fwd = continue k () in
              let twg_a = get_or_init a in
              let twg_b = get_or_init b in
              let twg_out = get_or_init out in
              let g = twg_out.grad in
              let mask_a = T.cast (dtype g) (T.cmpgt a b) in
              let mask_b = T.sub (T.ones_like mask_a) mask_a in
              let ga = T.mul g mask_a in
              let gb = T.mul g mask_b in
              twg_a.grad <-
                T.add twg_a.grad (Autodiff.unbroadcast_grad ga (T.shape a));
              twg_b.grad <-
                T.add twg_b.grad (Autodiff.unbroadcast_grad gb (T.shape b));
              fwd)
      | E_min { out; a; b } ->
          Some
            (fun k ->
              min ~out a b;
              let fwd = continue k () in
              let twg_a = get_or_init a in
              let twg_b = get_or_init b in
              let twg_out = get_or_init out in
              let g = twg_out.grad in
              let mask_a = T.cast (dtype g) (T.cmplt a b) in
              let mask_b = T.sub (T.ones_like mask_a) mask_a in
              let ga = T.mul g mask_a in
              let gb = T.mul g mask_b in
              twg_a.grad <-
                T.add twg_a.grad (Autodiff.unbroadcast_grad ga (T.shape a));
              twg_b.grad <-
                T.add twg_b.grad (Autodiff.unbroadcast_grad gb (T.shape b));
              fwd)
      | E_atan2 { out; a; b } ->
          Some
            (fun k ->
              atan2 ~out a b;
              let fwd = continue k () in
              let twg_a = get_or_init a in
              let twg_b = get_or_init b in
              let twg_out = get_or_init out in
              let g = twg_out.grad in
              let denom = T.add (T.mul a a) (T.mul b b) in
              let ga = T.mul g (T.div b denom) in
              let gb = T.mul g (T.neg (T.div a denom)) in
              twg_a.grad <-
                T.add twg_a.grad (Autodiff.unbroadcast_grad ga (T.shape a));
              twg_b.grad <-
                T.add twg_b.grad (Autodiff.unbroadcast_grad gb (T.shape b));
              fwd)
      (* Unary Arithmetic *)
      | E_neg { out; t_in } ->
          Some
            (fun k ->
              neg ~out t_in;
              let fwd = continue k () in
              let twg_in = get_or_init t_in in
              let twg_out = get_or_init out in
              twg_in.grad <- T.add twg_in.grad (T.neg twg_out.grad);
              fwd)
      | E_sin { out; t_in } ->
          Some
            (fun k ->
              sin ~out t_in;
              let fwd = continue k () in
              let twg_in = get_or_init t_in in
              let twg_out = get_or_init out in
              let g = T.mul twg_out.grad (Autodiff.deriv_sin t_in) in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      | E_cos { out; t_in } ->
          Some
            (fun k ->
              cos ~out t_in;
              let fwd = continue k () in
              let twg_in = get_or_init t_in in
              let twg_out = get_or_init out in
              let g =
                T.mul twg_out.grad (T.neg (Obj.magic (T.sin (Obj.magic t_in))))
              in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      | E_log { out; t_in } ->
          Some
            (fun k ->
              log ~out t_in;
              let fwd = continue k () in
              let twg_in = get_or_init t_in in
              let twg_out = get_or_init out in
              let g = T.mul twg_out.grad (T.recip t_in) in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      | E_exp { out; t_in } ->
          Some
            (fun k ->
              exp ~out t_in;
              let fwd = continue k () in
              let twg_in = get_or_init t_in in
              let twg_out = get_or_init out in
              let g = T.mul twg_out.grad out in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      | E_sqrt { out; t_in } ->
          Some
            (fun k ->
              sqrt ~out t_in;
              let fwd = continue k () in
              let twg_in = get_or_init t_in in
              let twg_out = get_or_init out in
              let g = T.mul twg_out.grad (Autodiff.deriv_sqrt out) in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      | E_recip { out; t_in } ->
          Some
            (fun k ->
              recip ~out t_in;
              let fwd = continue k () in
              let twg_in = get_or_init t_in in
              let twg_out = get_or_init out in
              let g = T.mul twg_out.grad (Autodiff.deriv_recip t_in) in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      | E_abs { out; t_in } ->
          Some
            (fun k ->
              abs ~out t_in;
              let fwd = continue k () in
              let twg_in = get_or_init t_in in
              let twg_out = get_or_init out in
              let g = T.mul twg_out.grad (T.sign t_in) in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      | E_sign { out; t_in } ->
          Some
            (fun k ->
              sign ~out t_in;
              let fwd = continue k () in
              let _ = get_or_init out in
              fwd)
      | E_tan { out; t_in } ->
          Some
            (fun k ->
              tan ~out t_in;
              let fwd = continue k () in
              let twg_in = get_or_init t_in in
              let twg_out = get_or_init out in
              let g = T.mul twg_out.grad (Autodiff.deriv_tan t_in) in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      | E_asin { out; t_in } ->
          Some
            (fun k ->
              asin ~out t_in;
              let fwd = continue k () in
              let twg_in = get_or_init t_in in
              let twg_out = get_or_init out in
              let g = T.mul twg_out.grad (Autodiff.deriv_asin t_in) in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      | E_acos { out; t_in } ->
          Some
            (fun k ->
              acos ~out t_in;
              let fwd = continue k () in
              let twg_in = get_or_init t_in in
              let twg_out = get_or_init out in
              let g = T.mul twg_out.grad (Autodiff.deriv_acos t_in) in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      | E_atan { out; t_in } ->
          Some
            (fun k ->
              atan ~out t_in;
              let fwd = continue k () in
              let twg_in = get_or_init t_in in
              let twg_out = get_or_init out in
              let g = T.mul twg_out.grad (Autodiff.deriv_atan t_in) in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      | E_sinh { out; t_in } ->
          Some
            (fun k ->
              sinh ~out t_in;
              let fwd = continue k () in
              let twg_in = get_or_init t_in in
              let twg_out = get_or_init out in
              let g =
                T.mul twg_out.grad (Obj.magic (T.cosh (Obj.magic t_in)))
              in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      | E_cosh { out; t_in } ->
          Some
            (fun k ->
              cosh ~out t_in;
              let fwd = continue k () in
              let twg_in = get_or_init t_in in
              let twg_out = get_or_init out in
              let g =
                T.mul twg_out.grad (Obj.magic (T.sinh (Obj.magic t_in)))
              in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      | E_tanh { out; t_in } ->
          Some
            (fun k ->
              tanh ~out t_in;
              let fwd = continue k () in
              let twg_in = get_or_init t_in in
              let twg_out = get_or_init out in
              let one = T.ones_like out in
              let g = T.mul twg_out.grad (T.sub one (T.mul out out)) in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      | E_trunc { out; t_in } ->
          Some
            (fun k ->
              trunc ~out t_in;
              let fwd = continue k () in
              let _ = get_or_init out in
              fwd)
      | E_ceil { out; t_in } ->
          Some
            (fun k ->
              ceil ~out t_in;
              let fwd = continue k () in
              let _ = get_or_init out in
              fwd)
      | E_floor { out; t_in } ->
          Some
            (fun k ->
              floor ~out t_in;
              let fwd = continue k () in
              let _ = get_or_init out in
              fwd)
      | E_round { out; t_in } ->
          Some
            (fun k ->
              round ~out t_in;
              let fwd = continue k () in
              let _ = get_or_init out in
              fwd)
      | E_erf { out; t_in } ->
          Some
            (fun k ->
              erf ~out t_in;
              let fwd = continue k () in
              let twg_in = get_or_init t_in in
              let twg_out = get_or_init out in
              let g = T.mul twg_out.grad (Autodiff.deriv_erf t_in) in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      (* Shape Operations *)
      | E_reshape { t_in; new_shape } ->
          Some
            (fun k ->
              let res = reshape t_in new_shape in
              let fwd = continue k res in
              let twg_in = get_or_init t_in in
              let twg_res = get_or_init res in
              let g = T.reshape (T.shape t_in) twg_res.grad in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      | E_permute { t_in; axes } ->
          Some
            (fun k ->
              let res = permute t_in axes in
              let fwd = continue k res in
              let twg_in = get_or_init t_in in
              let twg_res = get_or_init res in
              let inv = Array.make (Array.length axes) 0 in
              Array.iteri (fun i d -> inv.(d) <- i) axes;
              let g = T.transpose twg_res.grad ~axes:(Array.to_list inv) in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      | E_expand { t_in; new_target_shape } ->
          Some
            (fun k ->
              let res = expand t_in new_target_shape in
              let fwd = continue k res in
              let twg_in = get_or_init t_in in
              let twg_res = get_or_init res in
              let g = Autodiff.unbroadcast_grad twg_res.grad (T.shape t_in) in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      | E_shrink { t_in; limits } ->
          Some
            (fun k ->
              let res = shrink t_in limits in
              let fwd = continue k res in
              let twg_in = get_or_init t_in in
              let twg_res = get_or_init res in
              let pads =
                Array.mapi
                  (fun i (start, _) ->
                    let total = (T.shape t_in).(i) in
                    let len = (T.shape res).(i) in
                    (start, total - start - len))
                  limits
              in
              let g = pad twg_res.grad pads (Dtype.zero (dtype t_in)) in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      | E_flip { t_in; dims_to_flip } ->
          Some
            (fun k ->
              let res = flip t_in dims_to_flip in
              let fwd = continue k res in
              let twg_in = get_or_init t_in in
              let twg_res = get_or_init res in
              let g = flip twg_res.grad dims_to_flip in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      | E_pad { t_in; padding_config; fill_value = _ } ->
          Some
            (fun k ->
              let res = pad t_in padding_config (Dtype.zero (dtype t_in)) in
              let fwd = continue k res in
              let twg_in = get_or_init t_in in
              let twg_res = get_or_init res in
              let limits =
                Array.mapi
                  (fun i (pre, _) ->
                    let dim = (T.shape t_in).(i) in
                    (pre, pre + dim))
                  padding_config
              in
              let g = T.shrink limits twg_res.grad in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      | E_cat { t_list; axis } ->
          Some
            (fun k ->
              let res =
                match t_list with
                | [] -> failwith "cat: empty tensor list"
                | first :: _ ->
                    let first_shape = T.shape first in
                    let rank = Array.length first_shape in
                    let axis = if axis < 0 then axis + rank else axis in
                    let out_shape = Array.copy first_shape in
                    out_shape.(axis) <-
                      List.fold_left
                        (fun acc t -> acc + (T.shape t).(axis))
                        0 t_list;
                    let out = buffer (context first) (dtype first) out_shape in
                    cat ~out t_list ~axis;
                    out
              in
              let fwd = continue k res in
              let twg_res = get_or_init res in
              let g = twg_res.grad in
              let g_shape = T.shape g in
              let off = ref 0 in
              List.iter
                (fun t ->
                  let twg = get_or_init t in
                  let len = (T.shape t).(axis) in
                  let limits =
                    Array.init (Array.length g_shape) (fun i ->
                        if i = axis then (!off, !off + len) else (0, g_shape.(i)))
                  in
                  off := !off + len;
                  twg.grad <- T.add twg.grad (T.shrink limits g))
                t_list;
              fwd)
      (* Reductions *)
      | E_reduce_sum { out; t_in; axes; keepdims } ->
          Some
            (fun k ->
              reduce_sum ~out ~axes ~keepdims t_in;
              let fwd = continue k () in
              let twg_in = get_or_init t_in in
              let twg_out = get_or_init out in
              let g =
                if keepdims then twg_out.grad
                else
                  let kept_shape =
                    T.shape
                      (T.sum t_in ~axes:(Array.to_list axes) ~keepdims:true)
                  in
                  T.reshape kept_shape twg_out.grad
              in
              let g_bc = T.broadcast_to (T.shape t_in) g in
              twg_in.grad <- T.add twg_in.grad g_bc;
              fwd)
      | E_reduce_max { out; t_in; axes; keepdims } ->
          Some
            (fun k ->
              reduce_max ~out ~axes ~keepdims t_in;
              let fwd = continue k () in
              let twg_in = get_or_init t_in in
              let twg_out = get_or_init out in
              let shape_in = T.shape t_in in
              let out_bc =
                if keepdims then T.broadcast_to shape_in out
                else
                  let kept =
                    T.max t_in ~axes:(Array.to_list axes) ~keepdims:true
                  in
                  T.broadcast_to shape_in kept
              in
              let g_bc =
                if keepdims then T.broadcast_to shape_in twg_out.grad
                else
                  let kept_shape =
                    T.shape
                      (T.max t_in ~axes:(Array.to_list axes) ~keepdims:true)
                  in
                  T.broadcast_to shape_in (T.reshape kept_shape twg_out.grad)
              in
              let mask = T.cast (dtype out) (T.equal t_in out_bc) in
              twg_in.grad <- T.add twg_in.grad (T.mul g_bc mask);
              fwd)
      | E_reduce_min { out; t_in; axes; keepdims } ->
          Some
            (fun k ->
              reduce_min ~out ~axes ~keepdims t_in;
              let fwd = continue k () in
              let twg_in = get_or_init t_in in
              let twg_out = get_or_init out in
              let shape_in = T.shape t_in in
              let out_bc =
                if keepdims then T.broadcast_to shape_in out
                else
                  let kept =
                    T.min t_in ~axes:(Array.to_list axes) ~keepdims:true
                  in
                  T.broadcast_to shape_in kept
              in
              let g_bc =
                if keepdims then T.broadcast_to shape_in twg_out.grad
                else
                  let kept_shape =
                    T.shape
                      (T.min t_in ~axes:(Array.to_list axes) ~keepdims:true)
                  in
                  T.broadcast_to shape_in (T.reshape kept_shape twg_out.grad)
              in
              let mask = T.cast (dtype out) (T.equal t_in out_bc) in
              twg_in.grad <- T.add twg_in.grad (T.mul g_bc mask);
              fwd)
      | E_argmax { out; t_in; axis; keepdims } ->
          Some
            (fun k ->
              argmax ~out ~axis ~keepdims t_in;
              let fwd = continue k () in
              let _ = get_or_init out in
              fwd)
      | E_argmin { out; t_in; axis; keepdims } ->
          Some
            (fun k ->
              argmin ~out ~axis ~keepdims t_in;
              let fwd = continue k () in
              let _ = get_or_init out in
              fwd)
      | E_sort { out; t_in; axis; descending } ->
          Some
            (fun k ->
              sort ~out ~axis ~descending t_in;
              let fwd = continue k () in
              let _ = get_or_init out in
              fwd)
      | E_argsort { out; t_in; axis; descending } ->
          Some
            (fun k ->
              argsort ~out ~axis ~descending t_in;
              let fwd = continue k () in
              let _ = get_or_init out in
              fwd)
      (* Matrix Operations *)
      | E_matmul { out; a; b } ->
          Some
            (fun k ->
              matmul ~out a b;
              let fwd = continue k () in
              let twg_a = get_or_init a in
              let twg_b = get_or_init b in
              let twg_out = get_or_init out in
              let g = twg_out.grad in
              let a_shape = T.shape a in
              let b_shape = T.shape b in
              let g_shape = T.shape g in
              let a_ndim = Array.length a_shape in
              let b_ndim = Array.length b_shape in
              let g_ndim = Array.length g_shape in
              let transpose_last2 t =
                let nd = Array.length (T.shape t) in
                if nd < 2 then t
                else
                  let axes =
                    List.init nd (fun i ->
                        if i = nd - 2 then -1 else if i = nd - 1 then -2 else i)
                  in
                  T.transpose ~axes t
              in
              let grad_a =
                if a_ndim = 2 && b_ndim >= 3 then
                  let b_t = transpose_last2 b in
                  let g_bt = T.matmul g b_t in
                  let batch_dims = List.init (g_ndim - 2) Fun.id in
                  if batch_dims = [] then g_bt
                  else T.sum g_bt ~axes:batch_dims ~keepdims:false
                else if a_ndim >= 3 && b_ndim >= 3 then
                  T.matmul g (transpose_last2 b)
                else T.matmul g (T.transpose b)
              in
              let grad_b =
                if b_ndim = 2 && a_ndim >= 3 then
                  let at_g = T.matmul (transpose_last2 a) g in
                  let batch_dims = List.init (g_ndim - 2) Fun.id in
                  if batch_dims = [] then at_g
                  else T.sum at_g ~axes:batch_dims ~keepdims:false
                else if a_ndim = 2 && b_ndim >= 3 then
                  let a_t = T.transpose a in
                  let batch_shape = Array.sub g_shape 0 (g_ndim - 2) in
                  let a_t_shape = T.shape a_t in
                  let target_shape = Array.concat [ batch_shape; a_t_shape ] in
                  let a_t_expanded =
                    T.broadcast_to target_shape
                      (T.reshape (Array.concat [ [| 1 |]; a_t_shape ]) a_t)
                  in
                  T.matmul a_t_expanded g
                else if a_ndim >= 3 && b_ndim >= 3 then
                  T.matmul (transpose_last2 a) g
                else T.matmul (T.transpose a) g
              in
              twg_a.grad <- T.add twg_a.grad grad_a;
              twg_b.grad <- T.add twg_b.grad grad_b;
              fwd)
      (* Selection *)
      | E_where { out; condition; if_true; if_false } ->
          Some
            (fun k ->
              where ~out condition if_true if_false;
              let fwd = continue k () in
              let twg_t = get_or_init if_true in
              let twg_f = get_or_init if_false in
              let twg_out = get_or_init out in
              let g = twg_out.grad in
              let mask = T.cast (dtype g) condition in
              let inv_mask = T.sub (T.ones_like mask) mask in
              twg_t.grad <- T.add twg_t.grad (T.mul g mask);
              twg_f.grad <- T.add twg_f.grad (T.mul g inv_mask);
              fwd)
      (* Comparisons (no gradient) *)
      | E_cmplt { out; a; b } ->
          Some
            (fun k ->
              cmplt ~out a b;
              continue k ())
      | E_cmpne { out; a; b } ->
          Some
            (fun k ->
              cmpne ~out a b;
              continue k ())
      | E_cmpeq { out; a; b } ->
          Some
            (fun k ->
              cmpeq ~out a b;
              continue k ())
      | E_cmple { out; a; b } ->
          Some
            (fun k ->
              cmple ~out a b;
              continue k ())
      | E_xor { out; a; b } ->
          Some
            (fun k ->
              xor ~out a b;
              continue k ())
      | E_or { out; a; b } ->
          Some
            (fun k ->
              or_ ~out a b;
              continue k ())
      | E_and { out; a; b } ->
          Some
            (fun k ->
              and_ ~out a b;
              continue k ())
      (* Other *)
      | E_copy { t_in } ->
          Some
            (fun k ->
              let res = copy t_in in
              let fwd = continue k res in
              let twg_in = get_or_init t_in in
              let twg_res = get_or_init res in
              twg_in.grad <- T.add twg_in.grad twg_res.grad;
              fwd)
      | E_contiguous { t_in } ->
          Some
            (fun k ->
              let res = contiguous t_in in
              let fwd = continue k res in
              let twg_in = get_or_init t_in in
              let twg_res = get_or_init res in
              twg_in.grad <- T.add twg_in.grad twg_res.grad;
              fwd)
      | E_assign { dst; src } ->
          Some
            (fun k ->
              assign dst src;
              let fwd = continue k () in
              let twg_src = get_or_init src in
              let twg_dst = get_or_init dst in
              twg_src.grad <- T.add twg_src.grad twg_dst.grad;
              fwd)
      | E_cast { t_in; target_dtype } ->
          Some
            (fun k ->
              let res =
                let out = buffer (context t_in) target_dtype (T.shape t_in) in
                cast ~out t_in;
                out
              in
              let fwd = continue k res in
              let twg_in = get_or_init t_in in
              let twg_res = get_or_init res in
              let g = T.cast (dtype t_in) twg_res.grad in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      (* Reduce Prod *)
      | E_reduce_prod { out; t_in; axes; keepdims } ->
          Some
            (fun k ->
              reduce_prod ~out ~axes ~keepdims t_in;
              let fwd = continue k () in
              let twg_in = get_or_init t_in in
              let twg_out = get_or_init out in
              let shape_in = T.shape t_in in
              let g_prepared =
                if keepdims then twg_out.grad
                else
                  let kept_shape =
                    T.shape
                      (T.prod t_in ~axes:(Array.to_list axes) ~keepdims:true)
                  in
                  T.reshape kept_shape twg_out.grad
              in
              let g_bc = T.broadcast_to shape_in g_prepared in
              let out_prepared =
                if keepdims then out
                else
                  let kept_shape =
                    T.shape
                      (T.prod t_in ~axes:(Array.to_list axes) ~keepdims:true)
                  in
                  T.reshape kept_shape out
              in
              let out_bc = T.broadcast_to shape_in out_prepared in
              let grad_contrib = T.mul g_bc (T.div out_bc t_in) in
              twg_in.grad <- T.add twg_in.grad grad_contrib;
              fwd)
      (* Associative Scan *)
      | E_associative_scan { t_in; axis; op } ->
          Some
            (fun k ->
              let res =
                let out = buffer (context t_in) (dtype t_in) (T.shape t_in) in
                associative_scan ~out ~axis ~op t_in;
                out
              in
              let fwd = continue k res in
              let twg_in = get_or_init t_in in
              let twg_res = get_or_init res in
              let g = twg_res.grad in
              let shape_in = T.shape t_in in
              let axis_norm =
                let rank = Array.length shape_in in
                if axis < 0 then axis + rank else axis
              in
              let grad_contrib =
                match op with
                | `Sum ->
                    let flipped = T.flip g ~axes:[ axis_norm ] in
                    let scanned = T.cumsum ~axis:axis_norm flipped in
                    T.flip scanned ~axes:[ axis_norm ]
                | `Prod ->
                    let prefix_exclusive axis tensor =
                      let shape = T.shape tensor in
                      let pad_config =
                        Array.mapi
                          (fun i _ -> if i = axis then (1, 0) else (0, 0))
                          shape
                      in
                      let one = Dtype.one (T.dtype tensor) in
                      let padded = T.pad pad_config one tensor in
                      let cumprod_padded = T.cumprod ~axis padded in
                      let slice_specs =
                        Array.mapi
                          (fun i dim ->
                            if i = axis then T.R (0, dim) else T.R (0, dim))
                          shape
                      in
                      T.slice (Array.to_list slice_specs) cumprod_padded
                    in
                    let suffix_exclusive axis tensor =
                      let shape = T.shape tensor in
                      let one = Dtype.one (T.dtype tensor) in
                      let flipped = T.flip tensor ~axes:[ axis ] in
                      let flipped_cumprod = T.cumprod ~axis flipped in
                      let suffix_inclusive =
                        T.flip flipped_cumprod ~axes:[ axis ]
                      in
                      let pad_config =
                        Array.mapi
                          (fun i _ -> if i = axis then (0, 1) else (0, 0))
                          shape
                      in
                      let padded = T.pad pad_config one suffix_inclusive in
                      let slice_specs =
                        Array.mapi
                          (fun i dim ->
                            if i = axis then T.R (1, dim + 1) else T.R (0, dim))
                          shape
                      in
                      T.slice (Array.to_list slice_specs) padded
                    in
                    let divide_no_nan num denom =
                      let zero_tensor = T.zeros_like denom in
                      let zero_mask = T.equal denom zero_tensor in
                      let safe_denom =
                        T.where zero_mask (T.ones_like denom) denom
                      in
                      let base = T.div num safe_denom in
                      T.where zero_mask (T.zeros_like base) base
                    in
                    let reverse_cumsum tensor axis =
                      let flipped = T.flip tensor ~axes:[ axis ] in
                      let scanned = T.cumsum ~axis flipped in
                      T.flip scanned ~axes:[ axis ]
                    in
                    let prefix = prefix_exclusive axis_norm t_in in
                    let suffix = suffix_exclusive axis_norm t_in in
                    let h = divide_no_nan g suffix in
                    let tail_sum = T.sub (reverse_cumsum h axis_norm) h in
                    let inner = T.add g (T.mul suffix tail_sum) in
                    T.mul prefix inner
                | `Max ->
                    let shape = T.shape res in
                    let dt = dtype t_in in
                    let min_val = Dtype.min_value dt in
                    let pad_left =
                      Array.mapi
                        (fun i _ -> if i = axis_norm then (1, 0) else (0, 0))
                        shape
                    in
                    let padded = T.pad pad_left min_val res in
                    let slice_right =
                      Array.mapi
                        (fun i dim ->
                          if i = axis_norm then T.R (0, dim) else T.R (0, dim))
                        shape
                    in
                    let shifted_res =
                      T.slice (Array.to_list slice_right) padded
                    in
                    let active_mask = T.cast dt (T.cmpgt res shifted_res) in
                    T.mul g active_mask
                | `Min ->
                    let shape = T.shape res in
                    let dt = dtype t_in in
                    let max_val = Dtype.max_value dt in
                    let pad_left =
                      Array.mapi
                        (fun i _ -> if i = axis_norm then (1, 0) else (0, 0))
                        shape
                    in
                    let padded = T.pad pad_left max_val res in
                    let slice_right =
                      Array.mapi
                        (fun i dim ->
                          if i = axis_norm then T.R (0, dim) else T.R (0, dim))
                        shape
                    in
                    let shifted_res =
                      T.slice (Array.to_list slice_right) padded
                    in
                    let active_mask = T.cast dt (T.cmplt res shifted_res) in
                    T.mul g active_mask
              in
              twg_in.grad <- T.add twg_in.grad grad_contrib;
              fwd)
      (* Gather *)
      | E_gather { data; indices; axis } ->
          Some
            (fun k ->
              let res =
                let out =
                  buffer (context data) (dtype data) (T.shape indices)
                in
                gather ~out data indices ~axis;
                out
              in
              let fwd = continue k res in
              let twg_data = get_or_init data in
              let _ = get_or_init indices in
              let twg_res = get_or_init res in
              let g = twg_res.grad in
              let zeros_data = T.zeros_like data in
              let scattered_grads =
                scatter ~mode:`Add zeros_data ~indices ~updates:g ~axis
              in
              twg_data.grad <- T.add twg_data.grad scattered_grads;
              fwd)
      (* Scatter *)
      | E_scatter { data_template; indices; updates; axis } ->
          Some
            (fun k ->
              let res = scatter data_template ~indices ~updates ~axis in
              let fwd = continue k res in
              let twg_dt = get_or_init data_template in
              let twg_upd = get_or_init updates in
              let _ = get_or_init indices in
              let twg_res = get_or_init res in
              let g = twg_res.grad in
              let grad_upd =
                let out = buffer (context g) (dtype g) (T.shape indices) in
                gather ~out g indices ~axis;
                out
              in
              twg_upd.grad <- T.add twg_upd.grad grad_upd;
              let mask =
                scatter
                  (T.ones_like data_template)
                  ~indices ~updates:(T.zeros_like updates) ~axis
              in
              let grad_dt = T.mul g mask in
              twg_dt.grad <- T.add twg_dt.grad grad_dt;
              fwd)
      (* Unfold *)
      | E_unfold { t_in; kernel_size; stride; dilation; padding } ->
          Some
            (fun k ->
              let res = unfold t_in ~kernel_size ~stride ~dilation ~padding in
              let fwd = continue k res in
              let twg_in = get_or_init t_in in
              let twg_res = get_or_init res in
              let g = twg_res.grad in
              let input_shape = T.shape t_in in
              let num_spatial_dims = Array.length kernel_size in
              let output_size =
                Array.sub input_shape
                  (Array.length input_shape - num_spatial_dims)
                  num_spatial_dims
              in
              let grad_contrib =
                fold g ~output_size ~kernel_size ~stride ~dilation ~padding
              in
              twg_in.grad <- T.add twg_in.grad grad_contrib;
              fwd)
      (* Fold *)
      | E_fold { t_in; output_size; kernel_size; stride; dilation; padding } ->
          Some
            (fun k ->
              let res =
                fold t_in ~output_size ~kernel_size ~stride ~dilation ~padding
              in
              let fwd = continue k res in
              let twg_in = get_or_init t_in in
              let twg_res = get_or_init res in
              let g = twg_res.grad in
              let grad_contrib =
                unfold g ~kernel_size ~stride ~dilation ~padding
              in
              twg_in.grad <- T.add twg_in.grad grad_contrib;
              fwd)
      (* Cholesky *)
      | E_cholesky { t_in; upper } ->
          Some
            (fun k ->
              let l = cholesky ~upper t_in in
              let fwd = continue k l in
              let twg_in = get_or_init t_in in
              let twg_l = get_or_init l in
              let dl = twg_l.grad in
              let l_lower, dl_lower =
                if upper then (T.transpose l, T.transpose dl) else (l, dl)
              in
              let c = T.matmul (T.transpose l_lower) dl_lower in
              let p =
                let tril_c = T.tril c in
                let diag_c = T.diagonal c in
                let two = T.add (T.ones_like diag_c) (T.ones_like diag_c) in
                let half_diag = T.div diag_c two in
                T.sub tril_c (T.diag half_diag)
              in
              let z =
                triangular_solve ~upper:false ~transpose:true ~unit_diag:false
                  l_lower p
              in
              let y =
                triangular_solve ~upper:false ~transpose:true ~unit_diag:false
                  l_lower (T.transpose z)
              in
              let s = T.transpose y in
              let s_t = T.transpose s in
              let sum = T.add s s_t in
              let diag_s = T.diagonal s in
              let diag_mat = T.diag diag_s in
              let da_sym = T.sub sum diag_mat in
              let da = T.tril da_sym in
              twg_in.grad <- T.add twg_in.grad da;
              fwd)
      (* Triangular solve *)
      | E_triangular_solve { a; b; upper; transpose; unit_diag } ->
          Some
            (fun k ->
              let res = triangular_solve ~upper ~transpose ~unit_diag a b in
              let fwd = continue k res in
              let twg_a = get_or_init a in
              let twg_b = get_or_init b in
              let twg_res = get_or_init res in
              let g = twg_res.grad in
              let grad_b =
                if transpose then
                  triangular_solve ~upper ~transpose:false ~unit_diag a g
                else triangular_solve ~upper ~transpose:true ~unit_diag a g
              in
              twg_b.grad <- T.add twg_b.grad grad_b;
              let res_2d, grad_b_2d =
                let g_ndim = Array.length (T.shape g) in
                if g_ndim = 1 then
                  (T.expand_dims [ -1 ] res, T.expand_dims [ -1 ] grad_b)
                else (res, grad_b)
              in
              let grad_a_full =
                if transpose then
                  T.neg (T.matmul res_2d (T.transpose grad_b_2d))
                else T.neg (T.matmul grad_b_2d (T.transpose res_2d))
              in
              let grad_a =
                if upper then T.triu grad_a_full else T.tril grad_a_full
              in
              twg_a.grad <- T.add twg_a.grad grad_a;
              fwd)
      (* QR *)
      | E_qr { t_in; reduced } ->
          Some
            (fun k ->
              let q, r = qr ~reduced t_in in
              let fwd = continue k (q, r) in
              let twg_in = get_or_init t_in in
              let twg_q = get_or_init q in
              let twg_r = get_or_init r in
              let gq = twg_q.grad in
              let gr_full = twg_r.grad in
              let gr =
                let rt = T.transpose gr_full in
                T.transpose (T.tril rt)
              in
              let m =
                let term1 = T.matmul r (T.transpose gr) in
                let term2 = T.matmul (T.transpose gq) q in
                T.sub term1 term2
              in
              let lower_strict = T.tril ~k:(-1) m in
              let diag_m = T.contiguous (T.diagonal m) in
              let diag_mat = T.diag diag_m in
              let copyltu =
                T.add (T.add lower_strict (T.transpose lower_strict)) diag_mat
              in
              let rhs = T.add gq (T.matmul q copyltu) in
              let da_t =
                triangular_solve ~upper:true ~transpose:false ~unit_diag:false r
                  (T.transpose rhs)
              in
              let da = T.transpose da_t in
              twg_in.grad <- T.add twg_in.grad da;
              fwd)
      (* FFT Operations *)
      | E_fft { t; axes } ->
          Some
            (fun k ->
              let res = fft t ~axes in
              let fwd = continue k res in
              let twg_in = get_or_init t in
              let twg_res = get_or_init res in
              let g = twg_res.grad in
              let grad_contrib = ifft g ~axes in
              twg_in.grad <- T.add twg_in.grad grad_contrib;
              fwd)
      | E_ifft { t; axes } ->
          Some
            (fun k ->
              let res = ifft t ~axes in
              let fwd = continue k res in
              let twg_in = get_or_init t in
              let twg_res = get_or_init res in
              let g = twg_res.grad in
              let grad_contrib = fft g ~axes in
              twg_in.grad <- T.add twg_in.grad grad_contrib;
              fwd)
      | _ -> None
  in
  {
    retc =
      (fun final_result ->
        let twg_final = get_or_init final_result in
        twg_final.grad <- seed_output final_result;
        final_result);
    exnc = raise;
    effc;
  }

(* Helpers *)

let lookup_grad tape x =
  match Autodiff.Physical_tbl.find tape x with
  | Some (Any_twg twg) -> (unwrap_twg (dtype x) (Any_twg twg)).grad
  | None -> T.zeros_like x

let lookup_grads tape xs = List.map (lookup_grad tape) xs

(* API *)

let vjp (type a b c d) (f : (a, b) t -> (c, d) t) (x : (a, b) t)
    (cotangent : (c, d) t) : (c, d) t * (a, b) t =
  let tape = Autodiff.Physical_tbl.create 32 in
  let handler = make_handler tape (fun _ -> cotangent) in
  let y = Effect.Deep.match_with f x handler in
  (y, lookup_grad tape x)

let vjps (type a b c d) (f : (a, b) t list -> (c, d) t) (xs : (a, b) t list)
    (cotangent : (c, d) t) : (c, d) t * (a, b) t list =
  let tape = Autodiff.Physical_tbl.create 32 in
  let handler = make_handler tape (fun _ -> cotangent) in
  let y = Effect.Deep.match_with f xs handler in
  (y, lookup_grads tape xs)

let grad (type a b c d) (f : (a, b) t -> (c, d) t) (x : (a, b) t) : (a, b) t =
  let tape = Autodiff.Physical_tbl.create 32 in
  let handler = make_handler tape T.ones_like in
  let _ = Effect.Deep.match_with f x handler in
  lookup_grad tape x

let grads (type a b c d) (f : (a, b) t list -> (c, d) t) (xs : (a, b) t list) :
    (a, b) t list =
  let tape = Autodiff.Physical_tbl.create 32 in
  let handler = make_handler tape T.ones_like in
  let _ = Effect.Deep.match_with f xs handler in
  lookup_grads tape xs

let value_and_grad (type a b c d) (f : (a, b) t -> (c, d) t) (x : (a, b) t) :
    (c, d) t * (a, b) t =
  let tape = Autodiff.Physical_tbl.create 32 in
  let handler = make_handler tape T.ones_like in
  let y = Effect.Deep.match_with f x handler in
  (y, lookup_grad tape x)

let value_and_grad_aux (type a b c d e) (f : (a, b) t -> (c, d) t * e)
    (x : (a, b) t) : (c, d) t * (a, b) t * e =
  let tape = Autodiff.Physical_tbl.create 32 in
  let aux = ref None in
  let f' x =
    let y, a = f x in
    aux := Some a;
    y
  in
  let handler = make_handler tape T.ones_like in
  let y = Effect.Deep.match_with f' x handler in
  let aux_value =
    match !aux with
    | Some a -> a
    | None -> failwith "value_and_grad_aux: objective did not produce output"
  in
  (y, lookup_grad tape x, aux_value)

let value_and_grads (type a b c d) (f : (a, b) t list -> (c, d) t)
    (xs : (a, b) t list) : (c, d) t * (a, b) t list =
  let tape = Autodiff.Physical_tbl.create 32 in
  let handler = make_handler tape T.ones_like in
  let y = Effect.Deep.match_with f xs handler in
  (y, lookup_grads tape xs)

let value_and_grads_aux (type a b c d e) (f : (a, b) t list -> (c, d) t * e)
    (xs : (a, b) t list) : (c, d) t * (a, b) t list * e =
  let tape = Autodiff.Physical_tbl.create 32 in
  let aux = ref None in
  let f' xs =
    let y, a = f xs in
    aux := Some a;
    y
  in
  let handler = make_handler tape T.ones_like in
  let y = Effect.Deep.match_with f' xs handler in
  let aux_value =
    match !aux with
    | Some a -> a
    | None -> failwith "value_and_grads_aux: objective did not produce output"
  in
  (y, lookup_grads tape xs, aux_value)

let detach t = Autodiff.without_autodiff (fun () -> T.copy t)
let no_grad f = Autodiff.without_autodiff f
