open Nx_core
open Nx_rune
module T = Tensor

(* ───── Type Definitions & Utils ───── *)

module Physical_tbl = struct
  (* A physical identity-based map using pointer hashing. Hashes on the object's
     address (not contents), so mutations don't affect lookup. Uses physical
     equality (==) for collision resolution. *)

  module Tbl = Hashtbl.Make (struct
    type t = Obj.t

    let equal = ( == )

    let hash obj =
      (* Hash on the object's address, not its contents. Obj.magic to nativeint
         extracts the pointer value. *)
      Hashtbl.hash (Obj.magic obj : nativeint)
  end)

  type ('k, 'v) t = 'v Tbl.t

  let create n = Tbl.create n
  let find t key = Tbl.find_opt t (Obj.repr key)
  let add t key value = Tbl.replace t (Obj.repr key) value
end

let autodiff_enabled = ref true

let without_autodiff f =
  let prev = !autodiff_enabled in
  autodiff_enabled := false;
  Fun.protect f ~finally:(fun () -> autodiff_enabled := prev)

(* ───── Forward Mode (JVP) ───── *)

type ('a, 'b) dual = { primal : ('a, 'b) t; tangent : ('a, 'b) t }
type any_dual = Any_dual : ('a, 'b) dual -> any_dual

let unwrap_dual (type a b) (_ : (a, b) Dtype.t) (Any_dual d) : (a, b) dual =
  Obj.magic d

(* Derivative helpers for transcendental functions. *)
let ln2 = 0.693147180559945309417

let float_scalar_like (type a b) (x : (a, b) t) (v : float) : (a, b) t =
  Obj.magic (T.full (dtype x) [||] (Obj.magic v : a))

let deriv_sin (type a b) (x : (a, b) t) : (a, b) t =
  (* d/dx sin(x) = cos(x) *)
  Obj.magic (T.cos (Obj.magic x))

let deriv_sqrt (type a b) (sqrt_x : (a, b) t) : (a, b) t =
  (* d/dx sqrt(x) = 1 / (2 * sqrt(x)) *)
  T.div (T.ones_like sqrt_x) (T.mul (float_scalar_like sqrt_x 2.0) sqrt_x)

let deriv_recip (type a b) (x : (a, b) t) : (a, b) t =
  (* d/dx (1/x) = -1/x^2 *)
  T.neg (T.recip (T.mul x x))

let deriv_pow_wrt_base (type a b) (base : (a, b) t) (exp : (a, b) t) : (a, b) t
    =
  (* d/da (a^b) = b * a^(b-1) *)
  T.mul exp (T.pow base (T.sub exp (T.ones_like exp)))

let deriv_pow_wrt_exp (type a b) (base : (a, b) t) (result : (a, b) t) :
    (a, b) t =
  (* d/db (a^b) = a^b * ln(a) = a^b * log2(a) * ln(2) *)
  let ln_base =
    T.mul (Obj.magic (T.log2 (Obj.magic base))) (float_scalar_like base ln2)
  in
  T.mul result ln_base

let make_jvp_handler dual_map =
  let open Effect.Deep in
  let get_dual (type a b) (t : (a, b) t) : (a, b) dual =
    match Physical_tbl.find dual_map t with
    | Some (Any_dual d) -> unwrap_dual (dtype t) (Any_dual d)
    | None -> { primal = t; tangent = T.zeros_like t }
  in
  let register t dual = Physical_tbl.add dual_map t (Any_dual dual) in

  let effc : type c. c Effect.t -> ((c, _) continuation -> _) option =
   fun eff ->
    if not !autodiff_enabled then None
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
              let res = op_buffer ctx dt size_in_elements in
              (* Don't register buffer here - the actual operation that fills it
                 will register with the correct tangent *)
              continue k res)
      | E_threefry { key; ctr } ->
          Some
            (fun k ->
              let res = op_threefry key ctr in
              register res { primal = res; tangent = T.zeros_like res };
              continue k res)
      (* Binary Arithmetic *)
      | E_add { out; a; b } ->
          Some
            (fun k ->
              op_add ~out a b;
              let da = get_dual a in
              let db = get_dual b in
              let tan = T.add da.tangent db.tangent in
              register out { primal = out; tangent = tan };
              continue k ())
      | E_sub { out; a; b } ->
          Some
            (fun k ->
              op_sub ~out a b;
              let da = get_dual a in
              let db = get_dual b in
              let tan = T.sub da.tangent db.tangent in
              register out { primal = out; tangent = tan };
              continue k ())
      | E_mul { out; a; b } ->
          Some
            (fun k ->
              op_mul ~out a b;
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
              op_fdiv ~out a b;
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
              op_pow ~out a b;
              let da = get_dual a in
              let db = get_dual b in
              let term1 =
                T.mul da.tangent (deriv_pow_wrt_base da.primal db.primal)
              in
              let term2 = T.mul db.tangent (deriv_pow_wrt_exp da.primal out) in
              let tan = T.add term1 term2 in
              register out { primal = out; tangent = tan };
              continue k ())
      | E_max { out; a; b } ->
          Some
            (fun k ->
              op_max ~out a b;
              let da = get_dual a in
              let db = get_dual b in
              (* Use cmpgt for mask_a: tangent flows from a only when a > b This
                 ensures that when a == b, tangent flows from b (not a) which
                 gives correct behavior for relu(x) = max(x, 0) at x=0 *)
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
              op_min ~out a b;
              let da = get_dual a in
              let db = get_dual b in
              (* Use cmplt for mask_a: tangent flows from a only when a < b *)
              let mask_a = T.cast (dtype a) (T.cmplt a b) in
              let mask_b = T.sub (T.ones_like mask_a) mask_a in
              let tan =
                T.add (T.mul da.tangent mask_a) (T.mul db.tangent mask_b)
              in
              register out { primal = out; tangent = tan };
              continue k ())
      (* Unary Arithmetic *)
      | E_neg { out; t_in } ->
          Some
            (fun k ->
              op_neg ~out t_in;
              let d = get_dual t_in in
              let tan = T.neg d.tangent in
              register out { primal = out; tangent = tan };
              continue k ())
      | E_sin { out; t_in } ->
          Some
            (fun k ->
              op_sin ~out t_in;
              let d = get_dual t_in in
              let tan = T.mul d.tangent (deriv_sin d.primal) in
              register out { primal = out; tangent = tan };
              continue k ())
      | E_cos { out; t_in } ->
          Some
            (fun k ->
              op_cos ~out t_in;
              let d = get_dual t_in in
              (* d/dx cos(x) = -sin(x) *)
              let tan =
                T.mul d.tangent (T.neg (Obj.magic (T.sin (Obj.magic d.primal))))
              in
              register out { primal = out; tangent = tan };
              continue k ())
      | E_log { out; t_in } ->
          Some
            (fun k ->
              op_log ~out t_in;
              let d = get_dual t_in in
              (* d/dx log(x) = 1/x *)
              let tan = T.mul d.tangent (T.recip d.primal) in
              register out { primal = out; tangent = tan };
              continue k ())
      | E_exp { out; t_in } ->
          Some
            (fun k ->
              op_exp ~out t_in;
              let d = get_dual t_in in
              (* d/dx exp(x) = exp(x) *)
              let tan = T.mul d.tangent out in
              register out { primal = out; tangent = tan };
              continue k ())
      | E_sqrt { out; t_in } ->
          Some
            (fun k ->
              op_sqrt ~out t_in;
              let d = get_dual t_in in
              let tan = T.mul d.tangent (deriv_sqrt out) in
              register out { primal = out; tangent = tan };
              continue k ())
      | E_recip { out; t_in } ->
          Some
            (fun k ->
              op_recip ~out t_in;
              let d = get_dual t_in in
              let tan = T.mul d.tangent (deriv_recip d.primal) in
              register out { primal = out; tangent = tan };
              continue k ())
      | E_abs { out; t_in } ->
          Some
            (fun k ->
              op_abs ~out t_in;
              let d = get_dual t_in in
              (* d/dx |x| = sign(x) *)
              let tan = T.mul d.tangent (T.sign d.primal) in
              register out { primal = out; tangent = tan };
              continue k ())
      (* Shape Operations *)
      | E_reshape { t_in; new_shape } ->
          Some
            (fun k ->
              let res = op_reshape t_in new_shape in
              let d = get_dual t_in in
              let tan = op_reshape d.tangent new_shape in
              register res { primal = res; tangent = tan };
              continue k res)
      | E_permute { t_in; axes } ->
          Some
            (fun k ->
              let res = op_permute t_in axes in
              let d = get_dual t_in in
              let tan = op_permute d.tangent axes in
              register res { primal = res; tangent = tan };
              continue k res)
      | E_expand { t_in; new_target_shape } ->
          Some
            (fun k ->
              let res = op_expand t_in new_target_shape in
              let d = get_dual t_in in
              let tan = op_expand d.tangent new_target_shape in
              register res { primal = res; tangent = tan };
              continue k res)
      | E_shrink { t_in; limits } ->
          Some
            (fun k ->
              let res = op_shrink t_in limits in
              let d = get_dual t_in in
              let tan = op_shrink d.tangent limits in
              register res { primal = res; tangent = tan };
              continue k res)
      | E_flip { t_in; dims_to_flip } ->
          Some
            (fun k ->
              let res = op_flip t_in dims_to_flip in
              let d = get_dual t_in in
              let tan = op_flip d.tangent dims_to_flip in
              register res { primal = res; tangent = tan };
              continue k res)
      | E_pad { t_in; padding_config; fill_value } ->
          Some
            (fun k ->
              let res = op_pad t_in padding_config fill_value in
              let d = get_dual t_in in
              let tan =
                op_pad d.tangent padding_config (Dtype.zero (dtype t_in))
              in
              register res { primal = res; tangent = tan };
              continue k res)
      | E_cat { t_list; axis } ->
          Some
            (fun k ->
              let res = op_cat t_list axis in
              let tangents = List.map (fun t -> (get_dual t).tangent) t_list in
              let tan = op_cat tangents axis in
              register res { primal = res; tangent = tan };
              continue k res)
      (* Reductions *)
      | E_reduce_sum { out; t_in; axes; keepdims } ->
          Some
            (fun k ->
              op_reduce_sum ~out ~axes ~keepdims t_in;
              let d = get_dual t_in in
              let tan = T.sum d.tangent ~axes:(Array.to_list axes) ~keepdims in
              register out { primal = out; tangent = tan };
              continue k ())
      | E_reduce_max { out; t_in; axes; keepdims } ->
          Some
            (fun k ->
              op_reduce_max ~out ~axes ~keepdims t_in;
              let d = get_dual t_in in
              let shape_in = T.shape t_in in
              (* Broadcast result back to input shape to create mask *)
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
              op_reduce_min ~out ~axes ~keepdims t_in;
              let d = get_dual t_in in
              let shape_in = T.shape t_in in
              (* Broadcast result back to input shape to create mask *)
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
      (* Matrix Operations *)
      | E_matmul { out; a; b } ->
          Some
            (fun k ->
              op_matmul ~out a b;
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
              op_where ~out condition if_true if_false;
              let dt = get_dual if_true in
              let df = get_dual if_false in
              let tan = T.where condition dt.tangent df.tangent in
              register out { primal = out; tangent = tan };
              continue k ())
      (* Comparisons (no gradient) *)
      | E_cmplt { out; a; b } ->
          Some
            (fun k ->
              op_cmplt ~out a b;
              continue k ())
      | E_cmpne { out; a; b } ->
          Some
            (fun k ->
              op_cmpne ~out a b;
              continue k ())
      | E_cmpeq { out; a; b } ->
          Some
            (fun k ->
              op_cmpeq ~out a b;
              continue k ())
      | E_cmple { out; a; b } ->
          Some
            (fun k ->
              op_cmple ~out a b;
              continue k ())
      | E_xor { out; a; b } ->
          Some
            (fun k ->
              op_xor ~out a b;
              continue k ())
      | E_or { out; a; b } ->
          Some
            (fun k ->
              op_or ~out a b;
              continue k ())
      | E_and { out; a; b } ->
          Some
            (fun k ->
              op_and ~out a b;
              continue k ())
      (* Other *)
      | E_copy { t_in } ->
          Some
            (fun k ->
              let res = op_copy t_in in
              let d = get_dual t_in in
              let tan = op_copy d.tangent in
              register res { primal = res; tangent = tan };
              continue k res)
      | E_contiguous { t_in } ->
          Some
            (fun k ->
              let res = op_contiguous t_in in
              let d = get_dual t_in in
              let tan = op_contiguous d.tangent in
              register res { primal = res; tangent = tan };
              continue k res)
      | E_cast { t_in; target_dtype } ->
          Some
            (fun k ->
              let res = op_cast t_in target_dtype in
              let d = get_dual t_in in
              let tan = op_cast d.tangent target_dtype in
              register res { primal = res; tangent = tan };
              continue k res)
      (* Reduce Prod *)
      | E_reduce_prod { out; t_in; axes; keepdims } ->
          Some
            (fun k ->
              op_reduce_prod ~out ~axes ~keepdims t_in;
              let d = get_dual t_in in
              (* d(prod(x)) = prod(x) * sum(dx/x) over reduction axes *)
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
      (* Associative Scan (cumsum/cumprod/cummax/cummin) *)
      | E_associative_scan { t_in; axis; op } ->
          Some
            (fun k ->
              let res = op_associative_scan ~axis ~op t_in in
              let d = get_dual t_in in
              let tan =
                match op with
                | `Sum ->
                    (* cumsum is linear: d(cumsum(x)) = cumsum(dx) *)
                    op_associative_scan ~axis ~op:`Sum d.tangent
                | `Prod ->
                    (* cumprod tangent: d(cumprod(x))_i = sum_{j<=i}
                       cumprod(x)_i / x_j * dx_j = cumprod(x)_i * cumsum(dx /
                       x)_i *)
                    let ratio = T.div d.tangent d.primal in
                    let cumsum_ratio =
                      op_associative_scan ~axis ~op:`Sum ratio
                    in
                    T.mul res cumsum_ratio
                | `Max ->
                    (* cummax tangent: flows where new max is established *)
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
                    (* cummin tangent: flows where new min is established *)
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
      (* As Strided (for slicing/indexing) *)
      | E_as_strided { t_in; new_shape; new_strides; offset } ->
          Some
            (fun k ->
              let res =
                op_as_strided t_in
                  (Nx_core.Symbolic_shape.of_ints new_shape)
                  new_strides offset
              in
              let d = get_dual t_in in
              (* Apply same striding to tangent *)
              let tan =
                op_as_strided d.tangent
                  (Nx_core.Symbolic_shape.of_ints new_shape)
                  new_strides offset
              in
              register res { primal = res; tangent = tan };
              continue k res)
      (* Gather *)
      | E_gather { data; indices; axis } ->
          Some
            (fun k ->
              let res = op_gather data indices axis in
              let d = get_dual data in
              (* Gather from tangent using same indices *)
              let tan = op_gather d.tangent indices axis in
              register res { primal = res; tangent = tan };
              continue k res)
      (* Scatter *)
      | E_scatter { data_template; indices; updates; axis } ->
          Some
            (fun k ->
              let res = op_scatter data_template indices updates axis in
              let d_template = get_dual data_template in
              let d_updates = get_dual updates in
              (* Scatter tangent: mask template tangent and add scattered update
                 tangent *)
              let mask =
                op_scatter
                  (T.ones_like data_template)
                  indices (T.zeros_like updates) axis
              in
              let tan_template = T.mul d_template.tangent mask in
              let tan_updates =
                op_scatter
                  (T.zeros_like data_template)
                  indices d_updates.tangent axis
              in
              let tan = T.add tan_template tan_updates in
              register res { primal = res; tangent = tan };
              continue k res)
      (* FFT Operations *)
      | E_fft { t; axes } ->
          Some
            (fun k ->
              let res = op_fft t ~axes in
              let d = get_dual t in
              (* d(FFT(x)) = FFT(dx) *)
              let tan = op_fft d.tangent ~axes in
              register res { primal = res; tangent = tan };
              continue k res)
      | E_ifft { t; axes } ->
          Some
            (fun k ->
              let res = op_ifft t ~axes in
              let d = get_dual t in
              (* d(IFFT(x)) = IFFT(dx) *)
              let tan = op_ifft d.tangent ~axes in
              register res { primal = res; tangent = tan };
              continue k res)
      | _ -> None
  in
  { retc = Fun.id; exnc = raise; effc }

(* ───── Reverse Mode (VJP) ───── *)

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

(** Reduce gradient to match source shape (for broadcasting). *)
let unbroadcast_grad (type a b) (g : (a, b) t) (src_shape : int array) :
    (a, b) t =
  let dst_shape = T.shape g in
  if src_shape = dst_shape then g
  else
    let src_rank = Array.length src_shape in
    let dst_rank = Array.length dst_shape in
    let axes = ref [] in
    (* Leading dimensions added by broadcast *)
    for i = 0 to dst_rank - src_rank - 1 do
      axes := i :: !axes
    done;
    (* Dimensions that were 1 in source but expanded *)
    for i = 0 to src_rank - 1 do
      if src_shape.(i) = 1 && dst_shape.(i + (dst_rank - src_rank)) > 1 then
        axes := (i + (dst_rank - src_rank)) :: !axes
    done;
    match !axes with
    | [] -> g
    | ax ->
        let summed = T.sum g ~axes:ax ~keepdims:true in
        if T.shape summed <> src_shape then T.reshape src_shape summed
        else summed

let make_vjp_handler tape seed_output =
  let open Effect.Deep in
  let get_or_init (type a b) (t : (a, b) t) : (a, b) t_with_grad =
    match Physical_tbl.find tape t with
    | Some (Any_twg twg) -> unwrap_twg (dtype t) (Any_twg twg)
    | None ->
        let id = fresh_twg_id () in
        let twg = { v = t; grad = T.zeros_like t; id } in
        Physical_tbl.add tape t (Any_twg twg);
        twg
  in

  let effc : type c. c Effect.t -> ((c, _) continuation -> _) option =
   fun eff ->
    if not !autodiff_enabled then None
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
              let res = op_buffer ctx dt size_in_elements in
              let fwd = continue k res in
              let _ = get_or_init res in
              fwd)
      | E_threefry { key; ctr } ->
          Some
            (fun k ->
              let res = op_threefry key ctr in
              let fwd = continue k res in
              let _ = get_or_init res in
              fwd)
      (* Binary Arithmetic *)
      | E_add { out; a; b } ->
          Some
            (fun k ->
              op_add ~out a b;
              let fwd = continue k () in
              let twg_a = get_or_init a in
              let twg_b = get_or_init b in
              let twg_out = get_or_init out in
              let g = twg_out.grad in
              twg_a.grad <- T.add twg_a.grad (unbroadcast_grad g (T.shape a));
              twg_b.grad <- T.add twg_b.grad (unbroadcast_grad g (T.shape b));
              fwd)
      | E_sub { out; a; b } ->
          Some
            (fun k ->
              op_sub ~out a b;
              let fwd = continue k () in
              let twg_a = get_or_init a in
              let twg_b = get_or_init b in
              let twg_out = get_or_init out in
              let g = twg_out.grad in
              twg_a.grad <- T.add twg_a.grad (unbroadcast_grad g (T.shape a));
              twg_b.grad <-
                T.add twg_b.grad (unbroadcast_grad (T.neg g) (T.shape b));
              fwd)
      | E_mul { out; a; b } ->
          Some
            (fun k ->
              op_mul ~out a b;
              let fwd = continue k () in
              let twg_a = get_or_init a in
              let twg_b = get_or_init b in
              let twg_out = get_or_init out in
              let g = twg_out.grad in
              twg_a.grad <-
                T.add twg_a.grad (unbroadcast_grad (T.mul g b) (T.shape a));
              twg_b.grad <-
                T.add twg_b.grad (unbroadcast_grad (T.mul g a) (T.shape b));
              fwd)
      | E_fdiv { out; a; b } ->
          Some
            (fun k ->
              op_fdiv ~out a b;
              let fwd = continue k () in
              let twg_a = get_or_init a in
              let twg_b = get_or_init b in
              let twg_out = get_or_init out in
              let g = twg_out.grad in
              (* d/da = 1/b, d/db = -a/b^2 *)
              let ga = T.div g b in
              let gb = T.mul (T.neg g) (T.div a (T.mul b b)) in
              twg_a.grad <- T.add twg_a.grad (unbroadcast_grad ga (T.shape a));
              twg_b.grad <- T.add twg_b.grad (unbroadcast_grad gb (T.shape b));
              fwd)
      | E_pow { out; a; b } ->
          Some
            (fun k ->
              op_pow ~out a b;
              let fwd = continue k () in
              let twg_a = get_or_init a in
              let twg_b = get_or_init b in
              let twg_out = get_or_init out in
              let g = twg_out.grad in
              let ga = T.mul g (deriv_pow_wrt_base a b) in
              let gb = T.mul g (deriv_pow_wrt_exp a out) in
              twg_a.grad <- T.add twg_a.grad (unbroadcast_grad ga (T.shape a));
              twg_b.grad <- T.add twg_b.grad (unbroadcast_grad gb (T.shape b));
              fwd)
      | E_max { out; a; b } ->
          Some
            (fun k ->
              op_max ~out a b;
              let fwd = continue k () in
              let twg_a = get_or_init a in
              let twg_b = get_or_init b in
              let twg_out = get_or_init out in
              let g = twg_out.grad in
              (* Use cmpgt for mask_a: gradient flows to a only when a > b This
                 ensures that when a == b, gradient flows to b (not a) which
                 gives correct behavior for relu(x) = max(x, 0) at x=0 *)
              let mask_a = T.cast (dtype g) (T.cmpgt a b) in
              let mask_b = T.sub (T.ones_like mask_a) mask_a in
              let ga = T.mul g mask_a in
              let gb = T.mul g mask_b in
              twg_a.grad <- T.add twg_a.grad (unbroadcast_grad ga (T.shape a));
              twg_b.grad <- T.add twg_b.grad (unbroadcast_grad gb (T.shape b));
              fwd)
      | E_min { out; a; b } ->
          Some
            (fun k ->
              op_min ~out a b;
              let fwd = continue k () in
              let twg_a = get_or_init a in
              let twg_b = get_or_init b in
              let twg_out = get_or_init out in
              let g = twg_out.grad in
              (* Use cmplt for mask_a: gradient flows to a only when a < b *)
              let mask_a = T.cast (dtype g) (T.cmplt a b) in
              let mask_b = T.sub (T.ones_like mask_a) mask_a in
              let ga = T.mul g mask_a in
              let gb = T.mul g mask_b in
              twg_a.grad <- T.add twg_a.grad (unbroadcast_grad ga (T.shape a));
              twg_b.grad <- T.add twg_b.grad (unbroadcast_grad gb (T.shape b));
              fwd)
      (* Unary Arithmetic *)
      | E_neg { out; t_in } ->
          Some
            (fun k ->
              op_neg ~out t_in;
              let fwd = continue k () in
              let twg_in = get_or_init t_in in
              let twg_out = get_or_init out in
              twg_in.grad <- T.add twg_in.grad (T.neg twg_out.grad);
              fwd)
      | E_sin { out; t_in } ->
          Some
            (fun k ->
              op_sin ~out t_in;
              let fwd = continue k () in
              let twg_in = get_or_init t_in in
              let twg_out = get_or_init out in
              let g = T.mul twg_out.grad (deriv_sin t_in) in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      | E_cos { out; t_in } ->
          Some
            (fun k ->
              op_cos ~out t_in;
              let fwd = continue k () in
              let twg_in = get_or_init t_in in
              let twg_out = get_or_init out in
              (* d/dx cos(x) = -sin(x) *)
              let g =
                T.mul twg_out.grad (T.neg (Obj.magic (T.sin (Obj.magic t_in))))
              in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      | E_log { out; t_in } ->
          Some
            (fun k ->
              op_log ~out t_in;
              let fwd = continue k () in
              let twg_in = get_or_init t_in in
              let twg_out = get_or_init out in
              (* d/dx log(x) = 1/x *)
              let g = T.mul twg_out.grad (T.recip t_in) in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      | E_exp { out; t_in } ->
          Some
            (fun k ->
              op_exp ~out t_in;
              let fwd = continue k () in
              let twg_in = get_or_init t_in in
              let twg_out = get_or_init out in
              (* d/dx exp(x) = exp(x) *)
              let g = T.mul twg_out.grad out in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      | E_sqrt { out; t_in } ->
          Some
            (fun k ->
              op_sqrt ~out t_in;
              let fwd = continue k () in
              let twg_in = get_or_init t_in in
              let twg_out = get_or_init out in
              let g = T.mul twg_out.grad (deriv_sqrt out) in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      | E_recip { out; t_in } ->
          Some
            (fun k ->
              op_recip ~out t_in;
              let fwd = continue k () in
              let twg_in = get_or_init t_in in
              let twg_out = get_or_init out in
              let g = T.mul twg_out.grad (deriv_recip t_in) in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      | E_abs { out; t_in } ->
          Some
            (fun k ->
              op_abs ~out t_in;
              let fwd = continue k () in
              let twg_in = get_or_init t_in in
              let twg_out = get_or_init out in
              (* d/dx |x| = sign(x) *)
              let g = T.mul twg_out.grad (T.sign t_in) in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      (* Shape Operations *)
      | E_reshape { t_in; new_shape } ->
          Some
            (fun k ->
              let res = op_reshape t_in new_shape in
              let fwd = continue k res in
              let twg_in = get_or_init t_in in
              let twg_res = get_or_init res in
              let g = T.reshape (T.shape t_in) twg_res.grad in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      | E_permute { t_in; axes } ->
          Some
            (fun k ->
              let res = op_permute t_in axes in
              let fwd = continue k res in
              let twg_in = get_or_init t_in in
              let twg_res = get_or_init res in
              (* Inverse permutation *)
              let inv = Array.make (Array.length axes) 0 in
              Array.iteri (fun i d -> inv.(d) <- i) axes;
              let g = T.transpose twg_res.grad ~axes:(Array.to_list inv) in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      | E_expand { t_in; new_target_shape } ->
          Some
            (fun k ->
              let res = op_expand t_in new_target_shape in
              let fwd = continue k res in
              let twg_in = get_or_init t_in in
              let twg_res = get_or_init res in
              let g = unbroadcast_grad twg_res.grad (T.shape t_in) in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      | E_shrink { t_in; limits } ->
          Some
            (fun k ->
              let res = op_shrink t_in limits in
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
              let g = op_pad twg_res.grad pads (Dtype.zero (dtype t_in)) in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      | E_flip { t_in; dims_to_flip } ->
          Some
            (fun k ->
              let res = op_flip t_in dims_to_flip in
              let fwd = continue k res in
              let twg_in = get_or_init t_in in
              let twg_res = get_or_init res in
              let g = op_flip twg_res.grad dims_to_flip in
              twg_in.grad <- T.add twg_in.grad g;
              fwd)
      | E_pad { t_in; padding_config; fill_value = _ } ->
          Some
            (fun k ->
              let res = op_pad t_in padding_config (Dtype.zero (dtype t_in)) in
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
              let res = op_cat t_list axis in
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
              op_reduce_sum ~out ~axes ~keepdims t_in;
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
              op_reduce_max ~out ~axes ~keepdims t_in;
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
              op_reduce_min ~out ~axes ~keepdims t_in;
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
      (* Matrix Operations *)
      | E_matmul { out; a; b } ->
          Some
            (fun k ->
              op_matmul ~out a b;
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
              (* Helper to transpose last two dimensions (for batched matmul) *)
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
              (* Handle broadcasting in matmul backward: When 2D @ 3D or 3D @
                 2D, we need to properly broadcast/reduce gradients *)
              let grad_a =
                if a_ndim = 2 && b_ndim >= 3 then
                  (* a was broadcast: g @ B^T, then sum over batch dimensions *)
                  let b_t = transpose_last2 b in
                  let g_bt = T.matmul g b_t in
                  (* g_bt has shape [...batch, m, k], reduce to [m, k] *)
                  let batch_dims = List.init (g_ndim - 2) Fun.id in
                  if batch_dims = [] then g_bt
                  else T.sum g_bt ~axes:batch_dims ~keepdims:false
                else if a_ndim >= 3 && b_ndim >= 3 then
                  T.matmul g (transpose_last2 b)
                else T.matmul g (T.transpose b)
              in
              let grad_b =
                if b_ndim = 2 && a_ndim >= 3 then
                  (* b was broadcast: A^T @ g, then sum over batch dimensions *)
                  let at_g = T.matmul (transpose_last2 a) g in
                  (* at_g has shape [...batch, k, n], reduce to [k, n] *)
                  let batch_dims = List.init (g_ndim - 2) Fun.id in
                  if batch_dims = [] then at_g
                  else T.sum at_g ~axes:batch_dims ~keepdims:false
                else if a_ndim = 2 && b_ndim >= 3 then
                  (* a is 2D, b is 3D+: need to expand a for matmul *)
                  (* A^T @ g where A is [m, k] and g is [..., m, n] *)
                  (* We need [..., k, m] @ [..., m, n] = [..., k, n] *)
                  let a_t = T.transpose a in
                  (* [k, m] *)
                  (* Expand a_t to match g's batch dimensions *)
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
              op_where ~out condition if_true if_false;
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
              op_cmplt ~out a b;
              continue k ())
      | E_cmpne { out; a; b } ->
          Some
            (fun k ->
              op_cmpne ~out a b;
              continue k ())
      | E_cmpeq { out; a; b } ->
          Some
            (fun k ->
              op_cmpeq ~out a b;
              continue k ())
      | E_cmple { out; a; b } ->
          Some
            (fun k ->
              op_cmple ~out a b;
              continue k ())
      | E_xor { out; a; b } ->
          Some
            (fun k ->
              op_xor ~out a b;
              continue k ())
      | E_or { out; a; b } ->
          Some
            (fun k ->
              op_or ~out a b;
              continue k ())
      | E_and { out; a; b } ->
          Some
            (fun k ->
              op_and ~out a b;
              continue k ())
      (* Other *)
      | E_copy { t_in } ->
          Some
            (fun k ->
              let res = op_copy t_in in
              let fwd = continue k res in
              let twg_in = get_or_init t_in in
              let twg_res = get_or_init res in
              twg_in.grad <- T.add twg_in.grad twg_res.grad;
              fwd)
      | E_contiguous { t_in } ->
          Some
            (fun k ->
              let res = op_contiguous t_in in
              let fwd = continue k res in
              let twg_in = get_or_init t_in in
              let twg_res = get_or_init res in
              twg_in.grad <- T.add twg_in.grad twg_res.grad;
              fwd)
      | E_cast { t_in; target_dtype } ->
          Some
            (fun k ->
              let res = op_cast t_in target_dtype in
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
              op_reduce_prod ~out ~axes ~keepdims t_in;
              let fwd = continue k () in
              let twg_in = get_or_init t_in in
              let twg_out = get_or_init out in
              let shape_in = T.shape t_in in
              (* Prepare gradient for broadcasting *)
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
              (* Prepare result for broadcasting *)
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
              (* d(prod)/dx_i = prod / x_i *)
              let grad_contrib = T.mul g_bc (T.div out_bc t_in) in
              twg_in.grad <- T.add twg_in.grad grad_contrib;
              fwd)
      (* Associative Scan (cumsum/cumprod) *)
      | E_associative_scan { t_in; axis; op } ->
          Some
            (fun k ->
              let res = op_associative_scan ~axis ~op t_in in
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
                    (* Reverse cumsum of gradient *)
                    let flipped = T.flip g ~axes:[ axis_norm ] in
                    let scanned = T.cumsum ~axis:axis_norm flipped in
                    T.flip scanned ~axes:[ axis_norm ]
                | `Prod ->
                    (* More complex gradient for cumprod *)
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
                    (* cummax gradient: gradient flows to input positions where
                       a new maximum is established.

                       We detect this by comparing res with shifted res: where
                       res > prev_res, a new max was set at that position. *)
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
                    (* Gradient flows where res > prev_res (new max
                       established) *)
                    let active_mask = T.cast dt (T.cmpgt res shifted_res) in
                    T.mul g active_mask
                | `Min ->
                    (* cummin gradient: same logic but detecting new minimums *)
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
                    (* Gradient flows where res < prev_res (new min
                       established) *)
                    let active_mask = T.cast dt (T.cmplt res shifted_res) in
                    T.mul g active_mask
              in
              twg_in.grad <- T.add twg_in.grad grad_contrib;
              fwd)
      (* As Strided (for slicing/indexing) *)
      | E_as_strided { t_in; new_shape; new_strides; offset } ->
          Some
            (fun k ->
              let res =
                op_as_strided t_in
                  (Nx_core.Symbolic_shape.of_ints new_shape)
                  new_strides offset
              in
              let fwd = continue k res in
              let twg_in = get_or_init t_in in
              let twg_res = get_or_init res in
              let g = twg_res.grad in
              let input_shape = T.shape t_in in
              let input_numel = Array.fold_left ( * ) 1 input_shape in
              let output_numel = Array.fold_left ( * ) 1 new_shape in
              let ndim = Array.length new_shape in
              let flat_indices =
                T.init Nx_core.Dtype.Int32 [| output_numel |] (fun out_coords ->
                    let out_flat = out_coords.(0) in
                    let out_idx = Array.make ndim 0 in
                    let temp = ref out_flat in
                    for i = ndim - 1 downto 0 do
                      if new_shape.(i) > 0 then (
                        out_idx.(i) <- !temp mod new_shape.(i);
                        temp := !temp / new_shape.(i))
                    done;
                    let in_flat = ref offset in
                    for i = 0 to ndim - 1 do
                      in_flat := !in_flat + (out_idx.(i) * new_strides.(i))
                    done;
                    Int32.of_int !in_flat)
              in
              let g_flat = T.reshape [| output_numel |] g in
              let zeros_input = T.zeros (T.dtype t_in) [| input_numel |] in
              let grad_contrib =
                op_scatter ~mode:`Add zeros_input flat_indices g_flat 0
              in
              let grad_contrib_reshaped = T.reshape input_shape grad_contrib in
              twg_in.grad <- T.add twg_in.grad grad_contrib_reshaped;
              fwd)
      (* Gather *)
      | E_gather { data; indices; axis } ->
          Some
            (fun k ->
              let res = op_gather data indices axis in
              let fwd = continue k res in
              let twg_data = get_or_init data in
              let _ = get_or_init indices in
              let twg_res = get_or_init res in
              let g = twg_res.grad in
              let zeros_data = T.zeros_like data in
              let scattered_grads =
                op_scatter ~mode:`Add zeros_data indices g axis
              in
              twg_data.grad <- T.add twg_data.grad scattered_grads;
              fwd)
      (* Scatter *)
      | E_scatter { data_template; indices; updates; axis } ->
          Some
            (fun k ->
              let res = op_scatter data_template indices updates axis in
              let fwd = continue k res in
              let twg_dt = get_or_init data_template in
              let twg_upd = get_or_init updates in
              let _ = get_or_init indices in
              let twg_res = get_or_init res in
              let g = twg_res.grad in
              (* Gradient for updates: gather from result gradient *)
              let grad_upd = op_gather g indices axis in
              twg_upd.grad <- T.add twg_upd.grad grad_upd;
              (* Gradient for data_template: masked by scatter *)
              let mask =
                op_scatter
                  (T.ones_like data_template)
                  indices (T.zeros_like updates) axis
              in
              let grad_dt = T.mul g mask in
              twg_dt.grad <- T.add twg_dt.grad grad_dt;
              fwd)
      (* Unfold (for conv) *)
      | E_unfold { t_in; kernel_size; stride; dilation; padding } ->
          Some
            (fun k ->
              let res =
                op_unfold t_in ~kernel_size ~stride ~dilation ~padding
              in
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
                op_fold g ~output_size ~kernel_size ~stride ~dilation ~padding
              in
              twg_in.grad <- T.add twg_in.grad grad_contrib;
              fwd)
      (* Fold (for conv transpose) *)
      | E_fold { t_in; output_size; kernel_size; stride; dilation; padding } ->
          Some
            (fun k ->
              let res =
                op_fold t_in ~output_size ~kernel_size ~stride ~dilation
                  ~padding
              in
              let fwd = continue k res in
              let twg_in = get_or_init t_in in
              let twg_res = get_or_init res in
              let g = twg_res.grad in
              let grad_contrib =
                op_unfold g ~kernel_size ~stride ~dilation ~padding
              in
              twg_in.grad <- T.add twg_in.grad grad_contrib;
              fwd)
      (* Cholesky decomposition *)
      | E_cholesky { t_in; upper } ->
          Some
            (fun k ->
              let l = op_cholesky ~upper t_in in
              let fwd = continue k l in
              let twg_in = get_or_init t_in in
              let twg_l = get_or_init l in
              let dl = twg_l.grad in

              (* VJP for Cholesky: Let L be lower-triangular with A = L L^T.
                 Using Murray (2016), eqs. (9)–(10):

                 P = Φ(L^T L̄) S = L^{-T} P L^{-1} Ā = S + S^T - diag(S)

                 where Φ takes the lower triangular part and halves the
                 diagonal. *)

              (* Normalize to lower-triangular form if op returns upper *)
              let l_lower, dl_lower =
                if upper then (T.transpose l, T.transpose dl) else (l, dl)
              in

              (* C = L^T @ L̄ *)
              let c = T.matmul (T.transpose l_lower) dl_lower in

              (* P = Φ(C) = tril(C) - 0.5 * diag(C) *)
              let p =
                let tril_c = T.tril c in
                let diag_c = T.diagonal c in
                let two = T.add (T.ones_like diag_c) (T.ones_like diag_c) in
                let half_diag = T.div diag_c two in
                T.sub tril_c (T.diag half_diag)
              in

              (* Solve L^T Z = P => Z = L^{-T} P *)
              let z =
                op_triangular_solve ~upper:false ~transpose:true
                  ~unit_diag:false l_lower p
              in

              (* Compute S = Z @ L^{-1} Solve L^T Y = Z^T => Y^T = L^{-T} Z =
                 S *)
              let y =
                op_triangular_solve ~upper:false ~transpose:true
                  ~unit_diag:false l_lower (T.transpose z)
              in
              let s = T.transpose y in

              (* Ā = S + Sᵀ - diag(S) *)
              let s_t = T.transpose s in
              let sum = T.add s s_t in
              let diag_s = T.diagonal s in
              let diag_mat = T.diag diag_s in
              let da_sym = T.sub sum diag_mat in
              (* Cholesky only reads the lower triangle of A, so gradient should
                 be lower triangular to match finite differences. *)
              let da = T.tril da_sym in

              twg_in.grad <- T.add twg_in.grad da;
              fwd)
          (* Triangular solve *)
      | E_triangular_solve { a; b; upper; transpose; unit_diag } ->
          Some
            (fun k ->
              let res = op_triangular_solve ~upper ~transpose ~unit_diag a b in
              let fwd = continue k res in
              let twg_a = get_or_init a in
              let twg_b = get_or_init b in
              let twg_res = get_or_init res in
              let g = twg_res.grad in

              (* op(A) X = B, where op(A) = A (transpose=false) or A^T
                 (transpose=true) *)

              (* dB = op(A)^{-T} dX *)
              let grad_b =
                if transpose then
                  (* op(A) = A^T => dB = A^{-1} dX *)
                  op_triangular_solve ~upper ~transpose:false ~unit_diag a g
                else
                  (* op(A) = A => dB = A^{-T} dX *)
                  op_triangular_solve ~upper ~transpose:true ~unit_diag a g
              in
              twg_b.grad <- T.add twg_b.grad grad_b;

              (* Grad w.r.t A. For op(A) X = B, left-side:

                 transpose=false: Ā = -grad_b @ Xᵀ transpose=true: Ā = -X @
                 grad_bᵀ

                 We also zero out entries outside the specified triangle. *)

              (* Handle vector RHS by promoting to 2D *)
              let res_2d, grad_b_2d =
                let g_ndim = Array.length (T.shape g) in
                if g_ndim = 1 then
                  (T.expand_dims [ -1 ] res, T.expand_dims [ -1 ] grad_b)
                else (res, grad_b)
              in

              let grad_a_full =
                if transpose then
                  (* op(A) = A^T: Ā = -X @ grad_bᵀ *)
                  T.neg (T.matmul res_2d (T.transpose grad_b_2d))
                else
                  (* op(A) = A: Ā = -grad_b @ Xᵀ *)
                  T.neg (T.matmul grad_b_2d (T.transpose res_2d))
              in

              (* Keep only the triangular part that actually exists in A *)
              let grad_a =
                if upper then T.triu grad_a_full else T.tril grad_a_full
              in

              twg_a.grad <- T.add twg_a.grad grad_a;
              fwd)
          (* QR decomposition *)
      | E_qr { t_in; reduced } ->
          Some
            (fun k ->
              (* Forward pass: compute Q, R *)
              let q, r = op_qr ~reduced t_in in
              let fwd = continue k (q, r) in

              (* Ensure input is on the tape *)
              let twg_in = get_or_init t_in in
              let twg_q = get_or_init q in
              let twg_r = get_or_init r in

              (* Incoming cotangents *)
              let gq = twg_q.grad in
              let gr_full = twg_r.grad in

              (* 1. Project R-grad to upper triangular (R is
                 upper-triangular) *)
              let gr =
                let rt = T.transpose gr_full in
                (* tril(rt) is lower-triangular, transpose again -> upper *)
                T.transpose (T.tril rt)
              in

              (* 2. M = R @ gr^T - gq^T @ Q *)
              let m =
                let term1 = T.matmul r (T.transpose gr) in
                let term2 = T.matmul (T.transpose gq) q in
                T.sub term1 term2
              in

              (* 3. copyltu(M): copy lower (incl. diag) to upper This creates a
                 symmetric matrix from the lower triangular part. copyltu(M) =
                 tril_{-1}(M) + tril_{-1}(M)^T + diag(M) *)
              let lower_strict = T.tril ~k:(-1) m in
              let diag_m = T.contiguous (T.diagonal m) in
              let diag_mat = T.diag diag_m in
              let copyltu =
                T.add (T.add lower_strict (T.transpose lower_strict)) diag_mat
              in

              (* 4. rhs = gQ + Q @ copyltu(M) *)
              let rhs = T.add gq (T.matmul q copyltu) in

              (* 5. barA = rhs @ R^{-T} Implement via triangular_solve: solve R
                 @ barA^T = rhs^T -> barA^T = R^{-1} rhs^T *)
              let da_t =
                op_triangular_solve ~upper:true ~transpose:false
                  ~unit_diag:false r (T.transpose rhs)
              in
              let da = T.transpose da_t in

              (* Accumulate into gradient of input A *)
              twg_in.grad <- T.add twg_in.grad da;

              fwd)
      (* FFT Operations *)
      | E_fft { t; axes } ->
          Some
            (fun k ->
              let res = op_fft t ~axes in
              let fwd = continue k res in
              let twg_in = get_or_init t in
              let twg_res = get_or_init res in
              (* VJP for FFT: The adjoint of raw DFT is raw IDFT (no scaling).
                 op_fft/op_ifft are raw operations without normalization. *)
              let g = twg_res.grad in
              let grad_contrib = op_ifft g ~axes in
              twg_in.grad <- T.add twg_in.grad grad_contrib;
              fwd)
      | E_ifft { t; axes } ->
          Some
            (fun k ->
              let res = op_ifft t ~axes in
              let fwd = continue k res in
              let twg_in = get_or_init t in
              let twg_res = get_or_init res in
              (* VJP for IFFT: The adjoint of raw IDFT is raw DFT (no scaling).
                 op_fft/op_ifft are raw operations without normalization. *)
              let g = twg_res.grad in
              let grad_contrib = op_fft g ~axes in
              twg_in.grad <- T.add twg_in.grad grad_contrib;
              fwd)
      | _ -> None
  in
  {
    retc =
      (fun final_result ->
        (* Seed the output gradient before stack unwinds *)
        let twg_final = get_or_init final_result in
        twg_final.grad <- seed_output final_result;
        final_result);
    exnc = raise;
    effc;
  }

(* ───── API ───── *)

let jvp (type a b c d) (f : (a, b) t -> (c, d) t) (primals : (a, b) t)
    (tangents : (a, b) t) : (c, d) t * (c, d) t =
  let dual_map = Physical_tbl.create 16 in
  Physical_tbl.add dual_map primals
    (Any_dual { primal = primals; tangent = tangents });
  let handler = make_jvp_handler dual_map in
  let result = Effect.Deep.match_with f primals handler in
  match Physical_tbl.find dual_map result with
  | Some (Any_dual d) ->
      let d = unwrap_dual (dtype result) (Any_dual d) in
      (d.primal, d.tangent)
  | None -> (result, T.zeros_like result)

let jvps (type a b c d) (f : (a, b) t list -> (c, d) t)
    (primals : (a, b) t list) (tangents : (a, b) t list) : (c, d) t * (c, d) t =
  let dual_map = Physical_tbl.create 16 in
  List.iter2
    (fun p t ->
      Physical_tbl.add dual_map p (Any_dual { primal = p; tangent = t }))
    primals tangents;
  let handler = make_jvp_handler dual_map in
  let result = Effect.Deep.match_with f primals handler in
  match Physical_tbl.find dual_map result with
  | Some (Any_dual d) ->
      let d = unwrap_dual (dtype result) (Any_dual d) in
      (d.primal, d.tangent)
  | None -> (result, T.zeros_like result)

let jvp_aux (type a b c d e) (f : (a, b) t -> (c, d) t * e) (primals : (a, b) t)
    (tangents : (a, b) t) : (c, d) t * (c, d) t * e =
  let dual_map = Physical_tbl.create 16 in
  Physical_tbl.add dual_map primals
    (Any_dual { primal = primals; tangent = tangents });
  let handler = make_jvp_handler dual_map in
  let result, aux = Effect.Deep.match_with f primals handler in
  match Physical_tbl.find dual_map result with
  | Some (Any_dual d) ->
      let d = unwrap_dual (dtype result) (Any_dual d) in
      (d.primal, d.tangent, aux)
  | None -> (result, T.zeros_like result, aux)

let vjp (type a b c d) (f : (a, b) t -> (c, d) t) (x : (a, b) t)
    (cotangent : (c, d) t) : (c, d) t * (a, b) t =
  let tape = Physical_tbl.create 32 in
  let handler = make_vjp_handler tape (fun _ -> cotangent) in
  let y = Effect.Deep.match_with f x handler in
  let grad_x =
    match Physical_tbl.find tape x with
    | Some (Any_twg twg) -> (unwrap_twg (dtype x) (Any_twg twg)).grad
    | None -> T.zeros_like x
  in
  (y, grad_x)

let vjps (type a b c d) (f : (a, b) t list -> (c, d) t) (xs : (a, b) t list)
    (cotangent : (c, d) t) : (c, d) t * (a, b) t list =
  let tape = Physical_tbl.create 32 in
  let handler = make_vjp_handler tape (fun _ -> cotangent) in
  let y = Effect.Deep.match_with f xs handler in
  let grads =
    List.map
      (fun x ->
        match Physical_tbl.find tape x with
        | Some (Any_twg twg) -> (unwrap_twg (dtype x) (Any_twg twg)).grad
        | None -> T.zeros_like x)
      xs
  in
  (y, grads)

let grad (type a b c d) (f : (a, b) t -> (c, d) t) (x : (a, b) t) : (a, b) t =
  let tape = Physical_tbl.create 32 in
  let handler = make_vjp_handler tape T.ones_like in
  let _ = Effect.Deep.match_with f x handler in
  match Physical_tbl.find tape x with
  | Some (Any_twg twg) -> (unwrap_twg (dtype x) (Any_twg twg)).grad
  | None -> T.zeros_like x

let grads (type a b c d) (f : (a, b) t list -> (c, d) t) (xs : (a, b) t list) :
    (a, b) t list =
  let tape = Physical_tbl.create 32 in
  let handler = make_vjp_handler tape T.ones_like in
  let _ = Effect.Deep.match_with f xs handler in
  List.map
    (fun x ->
      match Physical_tbl.find tape x with
      | Some (Any_twg twg) -> (unwrap_twg (dtype x) (Any_twg twg)).grad
      | None -> T.zeros_like x)
    xs

let value_and_grad (type a b c d) (f : (a, b) t -> (c, d) t) (x : (a, b) t) :
    (c, d) t * (a, b) t =
  let tape = Physical_tbl.create 32 in
  let handler = make_vjp_handler tape T.ones_like in
  let y = Effect.Deep.match_with f x handler in
  let grad_x =
    match Physical_tbl.find tape x with
    | Some (Any_twg twg) -> (unwrap_twg (dtype x) (Any_twg twg)).grad
    | None -> T.zeros_like x
  in
  (y, grad_x)

let value_and_grads (type a b c d) (f : (a, b) t list -> (c, d) t)
    (xs : (a, b) t list) : (c, d) t * (a, b) t list =
  let tape = Physical_tbl.create 32 in
  let handler = make_vjp_handler tape T.ones_like in
  let y = Effect.Deep.match_with f xs handler in
  let grads =
    List.map
      (fun x ->
        match Physical_tbl.find tape x with
        | Some (Any_twg twg) -> (unwrap_twg (dtype x) (Any_twg twg)).grad
        | None -> T.zeros_like x)
      xs
  in
  (y, grads)

let detach t = without_autodiff (fun () -> T.copy t)
let no_grad f = without_autodiff f
