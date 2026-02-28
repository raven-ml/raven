(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Forward-mode automatic differentiation (JVP). Propagates tangent vectors
   alongside primal values through an effect handler that intercepts every
   tensor operation. *)

open Nx_core
open Nx_effect
module T = Nx

(* Dual numbers *)

type ('a, 'b) dual = { primal : ('a, 'b) t; tangent : ('a, 'b) t }
type any_dual = Any_dual : ('a, 'b) dual -> any_dual

let unwrap_dual (type a b) (_ : (a, b) Dtype.t) (Any_dual d) : (a, b) dual =
  Obj.magic d

(* Effect handler *)

let make_handler dual_map =
  let open Effect.Deep in
  let get_dual (type a b) (t : (a, b) t) : (a, b) dual =
    match Autodiff.Physical_tbl.find dual_map t with
    | Some (Any_dual d) -> unwrap_dual (dtype t) (Any_dual d)
    | None -> { primal = t; tangent = T.zeros_like t }
  in
  let register t dual = Autodiff.Physical_tbl.add dual_map t (Any_dual dual) in

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
              register res { primal = res; tangent = T.zeros_like res };
              continue k res)
      | E_from_host { context = ctx; array } ->
          Some
            (fun k ->
              let res = from_host ctx array in
              register res { primal = res; tangent = T.zeros_like res };
              continue k res)
      | E_buffer { context = ctx; dtype = dt; size_in_elements } ->
          Some
            (fun k ->
              let res = buffer ctx dt [| size_in_elements |] in
              continue k res)
      | E_threefry { key; ctr } ->
          Some
            (fun k ->
              let res =
                let out = buffer (context ctr) (dtype ctr) (T.shape ctr) in
                threefry ~out key ctr;
                out
              in
              register res { primal = res; tangent = T.zeros_like res };
              continue k res)
      (* Binary Arithmetic *)
      | E_add { out; a; b } ->
          Some
            (fun k ->
              add ~out a b;
              let da = get_dual a in
              let db = get_dual b in
              let tan = T.add da.tangent db.tangent in
              register out { primal = out; tangent = tan };
              continue k ())
      | E_sub { out; a; b } ->
          Some
            (fun k ->
              sub ~out a b;
              let da = get_dual a in
              let db = get_dual b in
              let tan = T.sub da.tangent db.tangent in
              register out { primal = out; tangent = tan };
              continue k ())
      | E_mul { out; a; b } ->
          Some
            (fun k ->
              mul ~out a b;
              let da = get_dual a in
              let db = get_dual b in
              (* d(a*b) = da*b + a*db *)
              let tan =
                T.add (T.mul da.tangent db.primal) (T.mul da.primal db.tangent)
              in
              register out { primal = out; tangent = tan };
              continue k ())
      | E_fdiv { out; a; b } ->
          Some
            (fun k ->
              div ~out a b;
              let da = get_dual a in
              let db = get_dual b in
              (* d(a/b) = da/b - a*db/b^2 *)
              let term1 = T.div da.tangent db.primal in
              let term2 =
                T.div (T.mul da.primal db.tangent) (T.mul db.primal db.primal)
              in
              let tan = T.sub term1 term2 in
              register out { primal = out; tangent = tan };
              continue k ())
      | E_pow { out; a; b } ->
          Some
            (fun k ->
              pow ~out a b;
              let da = get_dual a in
              let db = get_dual b in
              let term1 =
                T.mul da.tangent
                  (Autodiff.deriv_pow_wrt_base da.primal db.primal)
              in
              let term2 =
                T.mul db.tangent (Autodiff.deriv_pow_wrt_exp da.primal out)
              in
              let tan = T.add term1 term2 in
              register out { primal = out; tangent = tan };
              continue k ())
      | E_max { out; a; b } ->
          Some
            (fun k ->
              max ~out a b;
              let da = get_dual a in
              let db = get_dual b in
              let mask_a = T.cast (dtype a) (T.cmpgt a b) in
              let mask_b = T.sub (T.ones_like mask_a) mask_a in
              let tan =
                T.add (T.mul da.tangent mask_a) (T.mul db.tangent mask_b)
              in
              register out { primal = out; tangent = tan };
              continue k ())
      | E_min { out; a; b } ->
          Some
            (fun k ->
              min ~out a b;
              let da = get_dual a in
              let db = get_dual b in
              let mask_a = T.cast (dtype a) (T.cmplt a b) in
              let mask_b = T.sub (T.ones_like mask_a) mask_a in
              let tan =
                T.add (T.mul da.tangent mask_a) (T.mul db.tangent mask_b)
              in
              register out { primal = out; tangent = tan };
              continue k ())
      | E_atan2 { out; a; b } ->
          Some
            (fun k ->
              atan2 ~out a b;
              let da = get_dual a in
              let db = get_dual b in
              let denom =
                T.add (T.mul da.primal da.primal) (T.mul db.primal db.primal)
              in
              let tan =
                T.add
                  (T.mul da.tangent (T.div db.primal denom))
                  (T.mul db.tangent (T.neg (T.div da.primal denom)))
              in
              register out { primal = out; tangent = tan };
              continue k ())
      (* Unary Arithmetic *)
      | E_neg { out; t_in } ->
          Some
            (fun k ->
              neg ~out t_in;
              let d = get_dual t_in in
              let tan = T.neg d.tangent in
              register out { primal = out; tangent = tan };
              continue k ())
      | E_sin { out; t_in } ->
          Some
            (fun k ->
              sin ~out t_in;
              let d = get_dual t_in in
              let tan = T.mul d.tangent (Autodiff.deriv_sin d.primal) in
              register out { primal = out; tangent = tan };
              continue k ())
      | E_cos { out; t_in } ->
          Some
            (fun k ->
              cos ~out t_in;
              let d = get_dual t_in in
              (* d/dx cos(x) = -sin(x) *)
              let tan = T.mul d.tangent (T.neg (T.sin d.primal))
              in
              register out { primal = out; tangent = tan };
              continue k ())
      | E_log { out; t_in } ->
          Some
            (fun k ->
              log ~out t_in;
              let d = get_dual t_in in
              let tan = T.mul d.tangent (T.recip d.primal) in
              register out { primal = out; tangent = tan };
              continue k ())
      | E_exp { out; t_in } ->
          Some
            (fun k ->
              exp ~out t_in;
              let d = get_dual t_in in
              let tan = T.mul d.tangent out in
              register out { primal = out; tangent = tan };
              continue k ())
      | E_sqrt { out; t_in } ->
          Some
            (fun k ->
              sqrt ~out t_in;
              let d = get_dual t_in in
              let tan = T.mul d.tangent (Autodiff.deriv_sqrt out) in
              register out { primal = out; tangent = tan };
              continue k ())
      | E_recip { out; t_in } ->
          Some
            (fun k ->
              recip ~out t_in;
              let d = get_dual t_in in
              let tan = T.mul d.tangent (Autodiff.deriv_recip d.primal) in
              register out { primal = out; tangent = tan };
              continue k ())
      | E_abs { out; t_in } ->
          Some
            (fun k ->
              abs ~out t_in;
              let d = get_dual t_in in
              let tan = T.mul d.tangent (T.sign d.primal) in
              register out { primal = out; tangent = tan };
              continue k ())
      | E_sign { out; t_in } ->
          Some
            (fun k ->
              sign ~out t_in;
              register out { primal = out; tangent = T.zeros_like out };
              continue k ())
      | E_tan { out; t_in } ->
          Some
            (fun k ->
              tan ~out t_in;
              let d = get_dual t_in in
              let tanv = T.mul d.tangent (Autodiff.deriv_tan d.primal) in
              register out { primal = out; tangent = tanv };
              continue k ())
      | E_asin { out; t_in } ->
          Some
            (fun k ->
              asin ~out t_in;
              let d = get_dual t_in in
              let tanv = T.mul d.tangent (Autodiff.deriv_asin d.primal) in
              register out { primal = out; tangent = tanv };
              continue k ())
      | E_acos { out; t_in } ->
          Some
            (fun k ->
              acos ~out t_in;
              let d = get_dual t_in in
              let tanv = T.mul d.tangent (Autodiff.deriv_acos d.primal) in
              register out { primal = out; tangent = tanv };
              continue k ())
      | E_atan { out; t_in } ->
          Some
            (fun k ->
              atan ~out t_in;
              let d = get_dual t_in in
              let tanv = T.mul d.tangent (Autodiff.deriv_atan d.primal) in
              register out { primal = out; tangent = tanv };
              continue k ())
      | E_sinh { out; t_in } ->
          Some
            (fun k ->
              sinh ~out t_in;
              let d = get_dual t_in in
              let tanv = T.mul d.tangent (T.cosh d.primal)
              in
              register out { primal = out; tangent = tanv };
              continue k ())
      | E_cosh { out; t_in } ->
          Some
            (fun k ->
              cosh ~out t_in;
              let d = get_dual t_in in
              let tanv = T.mul d.tangent (T.sinh d.primal)
              in
              register out { primal = out; tangent = tanv };
              continue k ())
      | E_tanh { out; t_in } ->
          Some
            (fun k ->
              tanh ~out t_in;
              let d = get_dual t_in in
              let one = T.ones_like out in
              let tanv = T.mul d.tangent (T.sub one (T.mul out out)) in
              register out { primal = out; tangent = tanv };
              continue k ())
      | E_trunc { out; t_in } ->
          Some
            (fun k ->
              trunc ~out t_in;
              register out { primal = out; tangent = T.zeros_like out };
              continue k ())
      | E_ceil { out; t_in } ->
          Some
            (fun k ->
              ceil ~out t_in;
              register out { primal = out; tangent = T.zeros_like out };
              continue k ())
      | E_floor { out; t_in } ->
          Some
            (fun k ->
              floor ~out t_in;
              register out { primal = out; tangent = T.zeros_like out };
              continue k ())
      | E_round { out; t_in } ->
          Some
            (fun k ->
              round ~out t_in;
              register out { primal = out; tangent = T.zeros_like out };
              continue k ())
      | E_erf { out; t_in } ->
          Some
            (fun k ->
              erf ~out t_in;
              let d = get_dual t_in in
              let tanv = T.mul d.tangent (Autodiff.deriv_erf d.primal) in
              register out { primal = out; tangent = tanv };
              continue k ())
      (* Shape Operations *)
      | E_reshape { t_in; new_shape } ->
          Some
            (fun k ->
              let res = reshape t_in new_shape in
              let d = get_dual t_in in
              let tan = reshape d.tangent new_shape in
              register res { primal = res; tangent = tan };
              continue k res)
      | E_permute { t_in; axes } ->
          Some
            (fun k ->
              let res = permute t_in axes in
              let d = get_dual t_in in
              let tan = permute d.tangent axes in
              register res { primal = res; tangent = tan };
              continue k res)
      | E_expand { t_in; new_target_shape } ->
          Some
            (fun k ->
              let res = expand t_in new_target_shape in
              let d = get_dual t_in in
              let tan = expand d.tangent new_target_shape in
              register res { primal = res; tangent = tan };
              continue k res)
      | E_shrink { t_in; limits } ->
          Some
            (fun k ->
              let res = shrink t_in limits in
              let d = get_dual t_in in
              let tan = shrink d.tangent limits in
              register res { primal = res; tangent = tan };
              continue k res)
      | E_flip { t_in; dims_to_flip } ->
          Some
            (fun k ->
              let res = flip t_in dims_to_flip in
              let d = get_dual t_in in
              let tan = flip d.tangent dims_to_flip in
              register res { primal = res; tangent = tan };
              continue k res)
      | E_pad { t_in; padding_config; fill_value } ->
          Some
            (fun k ->
              let res = pad t_in padding_config fill_value in
              let d = get_dual t_in in
              let tan =
                pad d.tangent padding_config (Dtype.zero (dtype t_in))
              in
              register res { primal = res; tangent = tan };
              continue k res)
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
              let tangents = List.map (fun t -> (get_dual t).tangent) t_list in
              let tan =
                match tangents with
                | [] -> failwith "cat: empty tensor list"
                | first :: _ ->
                    let first_shape = T.shape first in
                    let rank = Array.length first_shape in
                    let axis = if axis < 0 then axis + rank else axis in
                    let out_shape = Array.copy first_shape in
                    out_shape.(axis) <-
                      List.fold_left
                        (fun acc t -> acc + (T.shape t).(axis))
                        0 tangents;
                    let out = buffer (context first) (dtype first) out_shape in
                    cat ~out tangents ~axis;
                    out
              in
              register res { primal = res; tangent = tan };
              continue k res)
      (* Reductions *)
      | E_reduce_sum { out; t_in; axes; keepdims } ->
          Some
            (fun k ->
              reduce_sum ~out ~axes ~keepdims t_in;
              let d = get_dual t_in in
              let tan = T.sum d.tangent ~axes:(Array.to_list axes) ~keepdims in
              register out { primal = out; tangent = tan };
              continue k ())
      | E_reduce_max { out; t_in; axes; keepdims } ->
          Some
            (fun k ->
              reduce_max ~out ~axes ~keepdims t_in;
              let d = get_dual t_in in
              let shape_in = T.shape t_in in
              let out_bc =
                if keepdims then T.broadcast_to shape_in out
                else
                  let kept =
                    T.max t_in ~axes:(Array.to_list axes) ~keepdims:true
                  in
                  T.broadcast_to shape_in kept
              in
              let mask = T.cast (dtype out) (T.equal d.primal out_bc) in
              let tan =
                T.sum (T.mul d.tangent mask) ~axes:(Array.to_list axes)
                  ~keepdims
              in
              register out { primal = out; tangent = tan };
              continue k ())
      | E_reduce_min { out; t_in; axes; keepdims } ->
          Some
            (fun k ->
              reduce_min ~out ~axes ~keepdims t_in;
              let d = get_dual t_in in
              let shape_in = T.shape t_in in
              let out_bc =
                if keepdims then T.broadcast_to shape_in out
                else
                  let kept =
                    T.min t_in ~axes:(Array.to_list axes) ~keepdims:true
                  in
                  T.broadcast_to shape_in kept
              in
              let mask = T.cast (dtype out) (T.equal d.primal out_bc) in
              let tan =
                T.sum (T.mul d.tangent mask) ~axes:(Array.to_list axes)
                  ~keepdims
              in
              register out { primal = out; tangent = tan };
              continue k ())
      | E_argmax { out; t_in; axis; keepdims } ->
          Some
            (fun k ->
              argmax ~out ~axis ~keepdims t_in;
              continue k ())
      | E_argmin { out; t_in; axis; keepdims } ->
          Some
            (fun k ->
              argmin ~out ~axis ~keepdims t_in;
              continue k ())
      | E_sort { out; t_in; axis; descending } ->
          Some
            (fun k ->
              sort ~out ~axis ~descending t_in;
              continue k ())
      | E_argsort { out; t_in; axis; descending } ->
          Some
            (fun k ->
              argsort ~out ~axis ~descending t_in;
              continue k ())
      (* Matrix Operations *)
      | E_matmul { out; a; b } ->
          Some
            (fun k ->
              matmul ~out a b;
              let da = get_dual a in
              let db = get_dual b in
              (* d(A@B) = dA@B + A@dB *)
              let tan =
                T.add
                  (T.matmul da.tangent db.primal)
                  (T.matmul da.primal db.tangent)
              in
              register out { primal = out; tangent = tan };
              continue k ())
      (* Selection *)
      | E_where { out; condition; if_true; if_false } ->
          Some
            (fun k ->
              where ~out condition if_true if_false;
              let dt = get_dual if_true in
              let df = get_dual if_false in
              let tan = T.where condition dt.tangent df.tangent in
              register out { primal = out; tangent = tan };
              continue k ())
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
              let d = get_dual t_in in
              let tan = copy d.tangent in
              register res { primal = res; tangent = tan };
              continue k res)
      | E_contiguous { t_in } ->
          Some
            (fun k ->
              let res = contiguous t_in in
              let d = get_dual t_in in
              let tan = contiguous d.tangent in
              register res { primal = res; tangent = tan };
              continue k res)
      | E_assign { dst; src } ->
          Some
            (fun k ->
              assign dst src;
              let d_src = get_dual src in
              register dst { primal = dst; tangent = d_src.tangent };
              continue k ())
      | E_cast { t_in; target_dtype } ->
          Some
            (fun k ->
              let res =
                let out = buffer (context t_in) target_dtype (T.shape t_in) in
                cast ~out t_in;
                out
              in
              let d = get_dual t_in in
              let tan =
                let out =
                  buffer (context d.tangent) target_dtype (T.shape d.tangent)
                in
                cast ~out d.tangent;
                out
              in
              register res { primal = res; tangent = tan };
              continue k res)
      (* Reduce Prod *)
      | E_reduce_prod { out; t_in; axes; keepdims } ->
          Some
            (fun k ->
              reduce_prod ~out ~axes ~keepdims t_in;
              let d = get_dual t_in in
              let shape_in = T.shape t_in in
              let out_bc =
                if keepdims then T.broadcast_to shape_in out
                else
                  let kept =
                    T.prod t_in ~axes:(Array.to_list axes) ~keepdims:true
                  in
                  T.broadcast_to shape_in kept
              in
              (* Gradient contribution: res / x_i * dx_i, summed over axes *)
              let contrib = T.mul (T.div out_bc d.primal) d.tangent in
              let tan = T.sum contrib ~axes:(Array.to_list axes) ~keepdims in
              register out { primal = out; tangent = tan };
              continue k ())
      (* Associative Scan *)
      | E_associative_scan { t_in; axis; op } ->
          Some
            (fun k ->
              let res =
                let out = buffer (context t_in) (dtype t_in) (T.shape t_in) in
                associative_scan ~out ~axis ~op t_in;
                out
              in
              let d = get_dual t_in in
              let tan =
                match op with
                | `Sum ->
                    let out =
                      buffer (context d.tangent) (dtype d.tangent)
                        (T.shape d.tangent)
                    in
                    associative_scan ~out ~axis ~op:`Sum d.tangent;
                    out
                | `Prod ->
                    let ratio = T.div d.tangent d.primal in
                    let cumsum_ratio =
                      let out =
                        buffer (context ratio) (dtype ratio) (T.shape ratio)
                      in
                      associative_scan ~out ~axis ~op:`Sum ratio;
                      out
                    in
                    T.mul res cumsum_ratio
                | `Max ->
                    let ndim = Array.length (T.shape res) in
                    let axis_norm = if axis < 0 then axis + ndim else axis in
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
                    T.mul d.tangent active_mask
                | `Min ->
                    let ndim = Array.length (T.shape res) in
                    let axis_norm = if axis < 0 then axis + ndim else axis in
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
                    T.mul d.tangent active_mask
              in
              register res { primal = res; tangent = tan };
              continue k res)
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
              let d = get_dual data in
              let tan =
                let out =
                  buffer (context d.tangent) (dtype d.tangent) (T.shape indices)
                in
                gather ~out d.tangent indices ~axis;
                out
              in
              register res { primal = res; tangent = tan };
              continue k res)
      (* Scatter *)
      | E_scatter { data_template; indices; updates; axis } ->
          Some
            (fun k ->
              let res = scatter data_template ~indices ~updates ~axis in
              let d_template = get_dual data_template in
              let d_updates = get_dual updates in
              let mask =
                scatter
                  (T.ones_like data_template)
                  ~indices ~updates:(T.zeros_like updates) ~axis
              in
              let tan_template = T.mul d_template.tangent mask in
              let tan_updates =
                scatter
                  (T.zeros_like data_template)
                  ~indices ~updates:d_updates.tangent ~axis
              in
              let tan = T.add tan_template tan_updates in
              register res { primal = res; tangent = tan };
              continue k res)
      (* FFT Operations *)
      | E_fft { t; axes } ->
          Some
            (fun k ->
              let res = fft t ~axes in
              let d = get_dual t in
              let tan = fft d.tangent ~axes in
              register res { primal = res; tangent = tan };
              continue k res)
      | E_ifft { t; axes } ->
          Some
            (fun k ->
              let res = ifft t ~axes in
              let d = get_dual t in
              let tan = ifft d.tangent ~axes in
              register res { primal = res; tangent = tan };
              continue k res)
      | _ -> None
  in
  { retc = Fun.id; exnc = raise; effc }

(* API *)

let lookup_tangent dual_map result =
  match Autodiff.Physical_tbl.find dual_map result with
  | Some (Any_dual d) ->
      let d = unwrap_dual (dtype result) (Any_dual d) in
      (d.primal, d.tangent)
  | None -> (result, T.zeros_like result)

let jvp (type a b c d) (f : (a, b) t -> (c, d) t) (primals : (a, b) t)
    (tangents : (a, b) t) : (c, d) t * (c, d) t =
  let dual_map = Autodiff.Physical_tbl.create 16 in
  Autodiff.Physical_tbl.add dual_map primals
    (Any_dual { primal = primals; tangent = tangents });
  let handler = make_handler dual_map in
  let result = Effect.Deep.match_with f primals handler in
  lookup_tangent dual_map result

let jvps (type a b c d) (f : (a, b) t list -> (c, d) t)
    (primals : (a, b) t list) (tangents : (a, b) t list) : (c, d) t * (c, d) t =
  let dual_map = Autodiff.Physical_tbl.create 16 in
  List.iter2
    (fun p t ->
      Autodiff.Physical_tbl.add dual_map p
        (Any_dual { primal = p; tangent = t }))
    primals tangents;
  let handler = make_handler dual_map in
  let result = Effect.Deep.match_with f primals handler in
  lookup_tangent dual_map result

let jvp_aux (type a b c d e) (f : (a, b) t -> (c, d) t * e) (primals : (a, b) t)
    (tangents : (a, b) t) : (c, d) t * (c, d) t * e =
  let dual_map = Autodiff.Physical_tbl.create 16 in
  Autodiff.Physical_tbl.add dual_map primals
    (Any_dual { primal = primals; tangent = tangents });
  let handler = make_handler dual_map in
  let result, aux = Effect.Deep.match_with f primals handler in
  let primal, tangent = lookup_tangent dual_map result in
  (primal, tangent, aux)
