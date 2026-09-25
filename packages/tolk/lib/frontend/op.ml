(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Capture before [open Tolk_uop] shadows it with the uop movement module. *)
module Movement_ops = Movement
open Tolk_uop
module Movement = Movement_ops
module D = Dtype
module T = Tensor

let prod = List.fold_left ( * ) 1
let take n l = List.filteri (fun idx _ -> idx < n) l
let drop n l = List.filteri (fun idx _ -> idx >= n) l
let sub_range lo hi l = List.filteri (fun idx _ -> idx >= lo && idx < hi) l

(* In-place assignment. The write is a STORE effect on the destination graph,
   sequenced with AFTER so that reads of the destination depend on it. When
   the destination is a view of a buffer, the AFTER is embedded at the
   buffer-identity level of the view chain and every live tensor aliasing
   that buffer is repointed, so they all observe the write. *)

let assign t x =
  if D.is_weak (T.dtype t) then T.set_uop t (T.uop (Creation.clone t));
  if T.uop t == T.uop x then t
  else begin
    (* Broadcast the value before resolving weak promotion with the destination. *)
    let x = Movement.symbolic_broadcast_to x (T.symbolic_shape t) in
    let x =
      if D.is_weak (T.dtype x) then
        Dtype_ops.cast x (D.least_upper_dtype [ T.dtype t; T.dtype x ])
      else x
    in
    if not (D.equal (T.dtype t) (T.dtype x)) then
      invalid_arg "Op.assign: dtype mismatch";
    (match T.device t, T.device x with
    | Some (Uop.Multi _), Some _ when Uop.axis (T.uop t) <> Uop.axis (T.uop x) ->
        invalid_arg "Op.assign: sharding axis mismatch"
    | _ -> ());
    let dst = T.uop t in
    let assigned_to = Uop.storage_base dst in
    if not (Uop.has_buffer_identity assigned_to)
       && (Uop.op assigned_to <> Ops.Stage || dst == assigned_to)
    then begin
      (* Overwriting a pending value initializes new storage; its old
         computation is dead. A view into pending storage still needs a write. *)
      let value = T.uop x in
      let value =
        if Uop.op value = Ops.Stage then (Uop.src value).(0) else value
      in
      T.set_uop t (T.uop (Creation.clone (T.of_uop value)))
    end else begin
      let store = Uop.store ~dst ~value:(T.uop x) () in
      let held u = List.exists (fun t -> T.uop t == u) (T.live_tensors ()) in
      let rec identity u =
        let op = Uop.op u in
        if (Ops.Group.is_movement op || op = Ops.Bitcast || op = Ops.Detach)
           && not (Uop.has_buffer_identity u && held u)
        then identity (Uop.src u).(0)
        else u
      in
      let ib = identity dst in
      if ib != dst then begin
        let target =
          if Uop.has_buffer_identity ~after_ok:true ib then ib
          else T.uop (Creation.clone (T.of_uop ib))
        in
        let store =
          if target == ib then store
          else Uop.substitute ~walk:true [ ib, target ] store
        in
        T.apply_map [ ib, Uop.after ~src:target ~deps:[ store ] ]
      end else T.set_uop t (Uop.after ~src:dst ~deps:[ store ])
    end;
    t
  end

(* Composed reductions *)

let reduced_count t ?axis () =
  let kd = T.shape (Reduce.sum ?axis ~keepdim:true t) in
  List.fold_left2 (fun acc si so -> if si <> so then acc * si else acc) 1 (T.shape t) kd

let mean ?axis ?(keepdim = false) t =
  let out_dt = if Dtype_ops.is_floating_point t then T.dtype t else D.float32 in
  let acc = D.sum_acc_dtype (Uop.commit_dtype (T.uop t)) in
  let numerator = Reduce.sum ?axis ~keepdim (Dtype_ops.cast t acc) in
  let denom = reduced_count t ?axis () in
  Dtype_ops.cast (Elementwise.div numerator (T.i denom)) out_dt

(* The squares are accumulated at the wider sum dtype: summing them at a narrow
   float loses the variance of a large input entirely. *)
let var ?axis ?(keepdim = false) ?(correction = 1) t =
  let out_dt = if Dtype_ops.is_floating_point t then T.dtype t else D.float32 in
  let m = mean ?axis ~keepdim:true t in
  let squares = Elementwise.square (Elementwise.sub t m) in
  let n = reduced_count squares ?axis () in
  let acc = D.sum_acc_dtype (T.val_dtype squares) in
  let numerator = Reduce.sum ?axis ~keepdim (Dtype_ops.cast squares acc) in
  let denom = Stdlib.max (n - correction) 0 in
  Dtype_ops.cast (Elementwise.div numerator (T.i denom)) out_dt

let std ?axis ?keepdim ?correction t =
  Elementwise.sqrt (var ?axis ?keepdim ?correction t)

let layernorm ?(axis = [ -1 ]) ?(eps = 1e-5) t =
  let y = Elementwise.sub t (mean ~axis ~keepdim:true t) in
  let m = mean ~axis ~keepdim:true (Elementwise.mul y y) in
  Elementwise.mul y
    (Elementwise.rsqrt
       (Elementwise.add m (Creation.const_like m (T.Sfloat eps))))

(* Concatenation *)

let cat ?(dim = 0) t args =
  let dim = T.resolve_dim t dim in
  let n = T.ndim t in
  let off_dim s = List.filteri (fun ax _ -> ax <> dim) s in
  let rest = off_dim (T.symbolic_shape t) in
  List.iter
    (fun x ->
      let xs = T.symbolic_shape x in
      if List.length xs <> n || not (List.for_all2 Uop.equal rest (off_dim xs))
      then invalid_arg "Op.cat: shapes must match off the concatenated axis")
    args;
  (* Every operand converts once to the dtype of the whole join; a fold over
     the pieces would carry the first ones through the dtype of each step. *)
  let dtype = Uop.promo_dtype (List.map T.uop (t :: args)) in
  let t = Dtype_ops.cast t dtype in
  let args = List.map (fun x -> Dtype_ops.cast x dtype) args in
  let extent x = List.nth (T.symbolic_shape x) dim in
  if List.for_all (fun x -> Uop.equal (extent x) (extent t)) args then
    (* Equal extents concatenate by stacking and merging the new axis into
       the old one, which selects on one range instead of summing pads. *)
    Movement.flatten ~start_dim:dim ~end_dim:(dim + 1)
      (Movement.stack ~dim t args)
  else begin
    (* Padding needs a concrete extent on the concatenated axis; the other
       axes may stay symbolic since they receive no padding. *)
    let size x =
      match Uop.const_int_value (extent x) with
      | Some s -> s
      | None -> invalid_arg "Op.cat: symbolic dimension on the cat axis"
    in
    let sizes = List.map size args in
    let total = List.fold_left ( + ) (size t) sizes in
    let place before sz x =
      Movement.pad x
        (List.init n (fun ax ->
             if ax = dim then (before, total - before - sz) else (0, 0)))
    in
    (* Concatenation is selection: a position takes its piece's element where
       the piece's padded footprint is true. The tinygrad counterpart sums
       zero-padded pieces, which turns -0 into +0, quiets a signalling NaN and,
       on devices whose float adds flush subnormals, zeroes them. *)
    let footprint before sz =
      place before sz
        (Creation.ones ~dtype:D.bool ~buffer:false
           (List.init n (fun ax -> if ax = dim then sz else 1)))
    in
    snd
      (List.fold_left2
         (fun (before, acc) x sz ->
           let placed = place before sz x in
           (before + sz, Elementwise.where (footprint before sz) placed acc))
         (size t, place 0 (size t) t)
         args sizes)
  end

(* Matrix multiplication *)

let dot ?dtype a w =
  let dx = T.ndim a and dw = T.ndim w in
  if dx = 0 || dw = 0 then invalid_arg "Op.dot: both tensors must be at least 1D";
  let sa = T.symbolic_shape a and sw = T.symbolic_shape w in
  let axis_w = -min dw 2 in
  let contract_a = List.nth sa (dx - 1) in
  let contract_w = List.nth sw (dw + axis_w) in
  (* The contracted dimensions must be provably equal; an undecidable
     symbolic comparison is an error. *)
  if Uop.resolve (Uop.O.ne contract_a contract_w) then
    invalid_arg "Op.dot: contracted dimensions differ";
  let ones = min (min (dx - 1) (dw - 1)) 1 in
  let ones_dims = List.init ones (fun _ -> Uop.const_int 1) in
  let a2 =
    Movement.symbolic_reshape a (take (dx - 1) sa @ ones_dims @ [ contract_a ])
  in
  let w2 =
    Movement.symbolic_reshape w
      (take (dw - 2) sw @ ones_dims @ drop (dw + axis_w) sw)
  in
  let w2 = Movement.transpose ~dim0:(-1) ~dim1:axis_w w2 in
  let summed = Reduce.sum ~axis:[ -1 ] ?dtype (Elementwise.mul a2 w2) in
  let out_dt =
    match dtype with
    | Some d -> d
    | None -> D.least_upper_dtype [ T.dtype a2; T.dtype w2 ]
  in
  Dtype_ops.cast summed out_dt

let matmul ?dtype a b = dot ?dtype a b

(* Constant padding *)

let pad_value t px value =
  let px = List.map (function None -> (0, 0) | Some p -> p) px in
  let sh = T.shape t in
  let has_neg = List.exists (fun (before, after) -> before < 0 || after < 0) px in
  let x =
    if has_neg then
      Movement.shrink t
        (List.map2
           (fun (before, after) s -> (-min before 0, min (after + s) s))
           px sh)
    else t
  in
  let pads =
    if has_neg then List.map (fun (before, after) -> (max before 0, max after 0)) px
    else px
  in
  let base = Movement.pad x pads in
  if Uop.equal (T.uop value) (Uop.const (Const.zero (T.dtype value))) then base
  else
    let mask =
      Movement.pad (Creation.const_like ~dtype:D.bool x (T.Sbool true)) pads
    in
    Elementwise.where mask base value

(* A fill holds the padded tensor's dtype, so an integer fill wraps as that
   dtype stores it; a NaN or infinite fill has no integer value and is refused. *)
let fill_of t value =
  let dt = T.dtype t in
  (match value with
   | T.Sfloat x when not (Const.converts dt (Const.Float x)) ->
       invalid_arg
         (Printf.sprintf "Op.pad: fill %g has no value at %s" x (D.to_string dt))
   | _ -> ());
  T.of_uop (Uop.const (T.scalar_const dt value))

let pad_constant t px value = pad_value t px (fill_of t value)

let pad_to ?(value = T.Sint 0) t dims =
  let ret = Movement.pad_to t dims in
  let value = fill_of t value in
  if T.uop ret == T.uop t
     || Uop.equal (T.uop value) (Uop.const (Const.zero (T.dtype value))) then ret
  else
    let mask =
      Movement.pad_to (Creation.const_like ~dtype:D.bool t (T.Sbool true)) dims
    in
    Elementwise.where mask ret value

(* Associative scans *)

let dtype_min_tensor t = T.of_uop (Uop.const (Const.min_value (Uop.commit_dtype (T.uop t))))
let dtype_max_tensor t = T.of_uop (Uop.const (Const.max_value (Uop.commit_dtype (T.uop t))))

let identity_element t = function
  | Ops.Add -> T.i 0
  | Ops.Mul -> T.i 1
  | Ops.Max -> dtype_min_tensor t
  | _ -> invalid_arg "Op.cumalu: op must be Add, Mul, or Max"

(* Each output reduces its own window of [k] inputs, so the work is quadratic
   in the axis length. *)
let cumalu t axis op =
  let k = List.nth (T.shape t) axis in
  let xt = Movement.transpose ~dim0:axis ~dim1:(-1) t in
  let nd = T.ndim xt in
  let px = List.init nd (fun idx -> if idx = nd - 1 then Some (k - 1, 0) else None) in
  let pooled = Movement.pool (pad_value xt px (identity_element t op)) ~k:[ k ] () in
  let reduced =
    match op with
    | Ops.Add -> Reduce.sum ~axis:[ -1 ] pooled
    | Ops.Mul -> Reduce.prod ~axis:[ -1 ] pooled
    | Ops.Max -> Reduce.max ~axis:[ -1 ] pooled
    | _ -> assert false
  in
  Movement.transpose ~dim0:axis ~dim1:(-1) reduced

(* An axis longer than two chunks scans in two stages: each chunk of [split]
   on its own, then the chunk totals, whose exclusive prefix is combined back
   into every chunk. The work drops from quadratic in the axis length to
   quadratic in [split] per chunk. *)
let split_cumalu t axis op =
  let axis = T.resolve_dim t axis in
  if T.ndim t = 0 || List.mem 0 (T.shape t) then
    match op with Ops.Add -> Dtype_ops.cast t (T.dtype (Reduce.sum t)) | _ -> t
  else
    let split = 256 in
    let s = List.nth (T.shape t) axis in
    if s <= split * 2 then cumalu t axis op
    else
      let value = identity_element t op in
      let n = (s + split - 1) / split in
      let total = n * split in
      let xt = Movement.transpose ~dim0:axis ~dim1:(-1) t in
      let nd = T.ndim xt in
      let last_pad p = List.init nd (fun idx -> if idx = nd - 1 then Some p else None) in
      let padded = pad_value xt (last_pad (total - s, 0)) value in
      let chunks = cumalu (Movement.unflatten padded (-1) [ n; split ]) nd op in
      let totals =
        Movement.squeeze ~dim:(-1)
          (Movement.shrink chunks
             (List.mapi
                (fun idx d -> if idx = nd then (split - 1, split) else (0, d))
                (T.shape chunks)))
      in
      let base = pad_value (cumalu totals (nd - 1) op) (last_pad (1, -1)) value in
      let scanned =
        Movement.flatten ~start_dim:(-2)
          (T.alu_binary op chunks (Movement.unsqueeze base (-1)))
      in
      let kept =
        Movement.shrink scanned
          (List.mapi
             (fun idx d -> if idx = nd - 1 then (total - s, total) else (0, d))
             (T.shape scanned))
      in
      Movement.transpose ~dim0:axis ~dim1:(-1) kept

let cumsum ?(axis = 0) t = split_cumalu t axis Ops.Add
let cumprod ?(axis = 0) t = split_cumalu t axis Ops.Mul

(* Ranges *)

(* Whether the endpoints of [[lo, hi]] fit the numeric limits of [dt]. *)
let range_fits dt lo hi =
  match (D.min dt, D.max dt) with
  | `Int dlo, `Int dhi ->
      Z.compare lo dlo >= 0 && Z.compare hi dhi <= 0
  | `Float dlo, `Float dhi ->
      Z.to_float lo >= dlo && Z.to_float hi <= dhi
  | _, _ -> true

let arange ?stop ?(step = 1) ?dtype start =
  if step = 0 then invalid_arg "Op.arange: step must be non-zero";
  let start, stop = match stop with None -> (0, start) | Some s -> (start, s) in
  let first = Z.of_int start and last = Z.of_int stop and stride = Z.of_int step in
  let lo, hi =
    if step > 0 then (first, Z.sub last stride) else (Z.sub last stride, first)
  in
  let dt =
    match dtype with
    | Some d -> d
    (* A range wider than the default integer widens rather than wrapping. *)
    | None -> if range_fits D.default_int lo hi then D.default_int else D.int64
  in
  if not (range_fits dt lo hi) then
    invalid_arg
      (Printf.sprintf "Op.arange: [%d, %d) is not representable in %s" start stop
         (D.to_string dt));
  let output_len = Z.cdiv (Z.sub last first) stride in
  if Z.sign output_len <= 0 then Creation.full ~dtype:dt ~buffer:false [ 0 ] (T.Sint 0)
  else begin
    if not (Z.fits_int output_len) then
      invalid_arg "Op.arange: length exceeds the host integer range";
    let acc_dtype =
      if D.is_float dt then D.least_upper_dtype [ dt; D.float32 ] else dt
    in
    let base = Creation.full ~dtype:acc_dtype ~buffer:false [ Z.to_int output_len ] (T.Sint step) in
    let scan = cumalu base 0 Ops.Add in
    let offset = T.of_uop (Uop.const (Const.integer D.weakint (Z.sub first stride))) in
    Dtype_ops.cast (Elementwise.add scan offset) dt
  end

let linspace ?dtype start stop steps =
  if steps < 0 then invalid_arg "Op.linspace: steps must be non-negative";
  let dt = match dtype with Some d -> d | None -> D.default_float in
  if D.is_bool dt then invalid_arg "Op.linspace: bool dtype is not supported";
  if steps = 1 then Creation.full ~dtype:dt ~buffer:false [ 1 ] (T.Sfloat start)
  else
    Dtype_ops.cast
      (Elementwise.add (T.f start)
         (Elementwise.mul
            (arange ~dtype:D.default_float steps)
            (T.f ((stop -. start) /. float_of_int (steps - 1)))))
      dt

let eye ?m ?dtype n =
  let m_ = match m with None -> n | Some m -> m in
  if n < 0 || m_ < 0 then invalid_arg "Op.eye: dimensions must be non-negative";
  let dt = match dtype with Some d -> d | None -> D.default_float in
  Dtype_ops.cast
    (Elementwise.eq (Movement.unsqueeze (arange n) (-1)) (arange m_))
    dt

(* Triangular masks *)

let tri r c ?(diagonal = 0) () =
  Elementwise.le
    (Elementwise.add (Movement.unsqueeze (arange r) (-1)) (T.i diagonal))
    (arange c)

let last2 t =
  let sh = T.shape t in
  let n = List.length sh in
  (List.nth sh (n - 2), List.nth sh (n - 1))

let triu ?(diagonal = 0) t =
  let r, c = last2 t in
  Elementwise.where (tri r c ~diagonal ()) t (Creation.const_like t (T.Sint 0))

let tril ?(diagonal = 0) t =
  let r, c = last2 t in
  Elementwise.where
    (tri r c ~diagonal:(diagonal + 1) ())
    (Creation.const_like t (T.Sint 0))
    t

(* Cumulative extrema *)

let cummax ?(axis = 0) t =
  if T.ndim t = 0 then
    (split_cumalu t axis Ops.Max, Creation.zeros ~dtype:D.int32 ~buffer:false [])
  else
    let axis = T.resolve_dim t axis in
    let values = split_cumalu t axis Ops.Max in
    let n = List.nth (T.shape t) axis in
    let x = Movement.transpose ~dim0:axis ~dim1:(-1) t in
    let values_t = Movement.transpose ~dim0:axis ~dim1:(-1) values in
    let matches =
      Elementwise.mul
        (Elementwise.eq (Movement.unsqueeze x (-1))
           (Movement.unsqueeze values_t (-2)))
        (triu (Creation.ones ~dtype:D.bool ~buffer:false [ n; n ]))
    in
    let counts =
      Reduce.max ~axis:[ -2 ]
        (Elementwise.mul matches
           (Movement.reshape (arange ~stop:0 ~step:(-1) n) [ n; 1 ]))
    in
    let idx = Dtype_ops.int (Elementwise.add (Elementwise.neg counts) (T.i n)) in
    (values, Movement.transpose ~dim0:(-1) ~dim1:axis idx)

let cummin ?(axis = 0) t =
  let values, indices = cummax ~axis (Elementwise.inverse t) in
  (Elementwise.inverse values, indices)

(* One-hot and gather *)

let one_hot_along_dim ?(dim = -1) index num_classes =
  if not (D.is_int (T.dtype index)) then
    invalid_arg "Op.one_hot_along_dim: integer index required";
  let offset = T.ndim index - T.resolve_dim index dim - 1 in
  let classes =
    Movement.reshape (arange num_classes)
      (num_classes :: List.init offset (fun _ -> 1))
  in
  Elementwise.eq index classes

let one_hot index num_classes =
  Elementwise.where
    (one_hot_along_dim (Movement.unsqueeze index (-1)) num_classes)
    (T.i 1) (T.i 0)

let gather t ~dim index =
  if T.ndim index <> T.ndim t then invalid_arg "Op.gather: ndim mismatch";
  let dim = T.resolve_dim t dim in
  let ish = T.shape index in
  let x =
    Movement.shrink_to t (List.mapi (fun d i -> if d = dim then None else Some i) ish)
  in
  let x = Movement.transpose ~dim0:(-1) ~dim1:dim (Movement.unsqueeze x (-1)) in
  let oh = one_hot_along_dim (Movement.unsqueeze index (-1)) (List.nth (T.shape t) dim) in
  Reduce.sum ~axis:[ -1 ] ~dtype:(T.val_dtype t)
    (Elementwise.where oh x (Creation.const_like x (T.Sint 0)))

(* Scatter

   Each element of [src] is written to [t] at the position named by the matching
   element of [index] along [dim]. Shared with [gather], the write is expressed
   as a one-hot mask over the scattered axis: [pre_scatter] lifts [src] and the
   one-hot [mask] into a trailing reduction axis, then a reduction (or a
   last-write-wins merge) collapses it back onto [t]'s shape. *)

let pre_scatter t ~dim index src =
  let dim = T.resolve_dim t dim in
  if T.ndim index <> T.ndim t || T.ndim src <> T.ndim t then
    invalid_arg "Op.scatter: self, index, and src must have equal rank";
  if not (D.equal (T.dtype t) (T.dtype src)) then
    invalid_arg "Op.scatter: self and src must have the same dtype";
  let ish = T.shape index in
  let src = Movement.shrink_to src (List.map (fun s -> Some s) ish) in
  let n = List.nth (T.shape t) dim in
  let src =
    Movement.transpose ~dim0:(-1) ~dim1:dim
      (Movement.expand (Movement.unsqueeze src (-1)) (T.shape src @ [ n ]))
  in
  let mask =
    Movement.transpose ~dim0:(-1) ~dim1:dim
      (one_hot_along_dim (Movement.unsqueeze index (-1)) n)
  in
  let pad = List.map (fun s -> Some s) (T.shape t) @ [ None ] in
  (Movement.pad_to src pad, Movement.pad_to mask pad)

(* Merge [values] into [t] along [axes], last write winning where [mask] repeats
   an index. *)
let masked_merge t values mask axes =
  let values = ref values and mask = ref mask in
  List.iter
    (fun dim ->
      match (Movement.split ~dim !mask 1, Movement.split ~dim !values 1) with
      | m0 :: mrest, v0 :: vrest ->
          let m = ref m0 and v = ref v0 in
          List.iter2
            (fun my vy ->
              let merged = Elementwise.where my vy !v in
              m := Elementwise.bitwise_or !m my;
              v := merged)
            mrest vrest;
          mask := !m;
          values := !v
      | _ -> assert false)
    (List.rev axes);
  List.iter
    (fun dim ->
      mask := Movement.squeeze ~dim !mask;
      values := Movement.squeeze ~dim !values)
    (List.rev axes);
  Elementwise.where !mask !values t

let scatter_reduce t ~dim index src ~reduce ?(include_self = true) () =
  let src, mask = pre_scatter t ~dim index src in
  let outside a b =
    Elementwise.where (Elementwise.logical_not (Reduce.any ~axis:[ -1 ] mask)) a b
  in
  let self_or fill = if include_self then t else outside t fill in
  match reduce with
  | `Sum ->
      Elementwise.add
        (Reduce.sum ~axis:[ -1 ] (Elementwise.where mask src (T.i 0)))
        (self_or (T.i 0))
  | `Prod ->
      Elementwise.mul
        (Reduce.prod ~axis:[ -1 ] (Elementwise.where mask src (T.i 1)))
        (self_or (T.i 1))
  | `Amax ->
      let m = dtype_min_tensor src in
      Elementwise.maximum
        (Reduce.max ~axis:[ -1 ] (Elementwise.where mask src m))
        (self_or (dtype_min_tensor src))
  | `Amin ->
      let m = dtype_max_tensor src in
      Elementwise.minimum
        (Reduce.min ~axis:[ -1 ] (Elementwise.where mask src m))
        (self_or (dtype_max_tensor src))
  | `Mean ->
      let count =
        Elementwise.add
          (Reduce.sum ~axis:[ -1 ] (Elementwise.where mask (T.i 1) (T.i 0)))
          (if include_self then T.i 1 else outside (T.i 1) (T.i 0))
      in
      let acc =
        Elementwise.add
          (Reduce.sum ~axis:[ -1 ] (Elementwise.where mask src (T.i 0)))
          (self_or (T.i 0))
      in
      Elementwise.div acc count

let scatter t ~dim index src =
  let src, mask = pre_scatter t ~dim index src in
  masked_merge t src mask [ -1 ]

(* Kernels over split operands

   No tinygrad counterpart: the reference builds its custom kernels over one
   device's placeholders. A custom kernel whose operands are split across
   devices runs once per device over each device's slices (the multi rewrite
   passes the call through), so it is built from the extents of the slices,
   which its placeholders carry. A split result is an unwritten buffer of one
   slice per device. An index along an axis split across devices is offset by
   the first position the running device holds, which the device range gives.
   A kernel that contracts or selects along a split axis leaves each device a
   partial result, which a sum across the devices completes. *)

(* The axis [t] is split along across devices, if any. *)
let split_axis t =
  match T.device t with Some (Uop.Multi _) -> Uop.axis (T.uop t) | _ -> None

(* The extents of the slice of [t] each device holds. *)
let slices t = Uop.max_shard_shape (T.uop t)

let device_count = function Uop.Multi ds -> List.length ds | _ -> 1

(* The position of the device running a kernel among [n]. *)
let device_position n =
  Uop.range ~size:(Uop.const_int n) ~axis:(-1) ~kind:Axis_type.Device ()

(* Where a kernel's result lives over split operands: whole on each device,
   split along one of its axes, or as a partial on each device. *)
type result = Whole | Split of int | Partial

(* The unwritten storage of a kernel's result of [shape] and [dtype] at
   [result], and the result once the kernel has written it. Partials are
   float32, summed across the devices and rounded once. *)
let result_storage ~dtype ~device result shape =
  let split ~dtype a shape =
    let n = device_count device in
    let local = List.mapi (fun i d -> if i = a then d / n else d) shape in
    T.of_uop
      (Uop.unshard
         ~src:(T.uop (Creation.empty ~dtype ~device local))
         ~axes:[ a ] ())
  in
  match result with
  | Whole -> (Creation.empty ~dtype ~device shape, Fun.id)
  | Split a -> (split ~dtype a shape, Fun.id)
  | Partial ->
      ( split ~dtype:D.float32 0 (device_count device :: shape),
        fun t -> Dtype_ops.cast (Reduce.sum ~axis:[ 0 ] t) dtype )

(* Indexed scatter

   No tinygrad counterpart. [scatter] and [scatter_reduce] range over the
   destination with a trailing axis over the indices, so a write of [k] rows
   into [n] costs [n * k]. [scatter_indexed] ranges over the indices: a custom
   kernel whose store lands in [t]'s storage at the loaded index, the way the
   reference's embedding gradient writes its buffer. Two updates collide only
   when they agree on every coordinate off [dim], so those coordinates are
   parallel lanes and the index range is serial within a lane: updates land in
   index order, the last [`Set] wins and [`Add] accumulates exactly, on every
   device. [unique] is the caller's promise that no two updates of a lane
   share an index, never inferred: it frees the index range too, and leaves
   the kernel's layout to the optimizer; built as is, it would run one thread
   per workgroup. An index outside the axis gates the store off.

   Over a [t] split across devices each device writes its own slice: the
   lanes of a split axis off [dim] are the device's, with [index] and [src]
   split alike, and along a split [dim] every device reads every update and
   keeps those that land in its rows. *)

let scatter_indexed t ~dim index src ~mode ~unique =
  let src =
    if D.is_weak (T.dtype src) then
      Dtype_ops.cast src (D.least_upper_dtype [ T.dtype t; T.dtype src ])
    else src
  in
  let dim = T.resolve_dim t dim in
  let tsh = T.shape t and ish = T.shape index and ssh = T.shape src in
  let rank = List.length tsh in
  if List.length ish <> rank || List.length ssh <> rank then
    invalid_arg "Op.scatter_indexed: self, index, and src must have equal rank";
  if not (D.equal (T.dtype t) (T.dtype src)) then
    invalid_arg "Op.scatter_indexed: self and src must have the same dtype";
  if not (D.is_int (T.dtype index)) then
    invalid_arg "Op.scatter_indexed: integer index required";
  if not (Uop.has_buffer_identity ~after_ok:true (T.uop t)) then
    invalid_arg "Op.scatter_indexed: self must be storage";
  List.iteri
    (fun d extent ->
      let i = List.nth ish d and s = List.nth ssh d in
      let fits =
        if d = dim then i = s else s = extent && (i = extent || i = 1)
      in
      if not fits then
        invalid_arg
          (Printf.sprintf "Op.scatter_indexed: shape mismatch on axis %d" d))
    tsh;
  let split = split_axis t in
  List.iter
    (fun (name, x) ->
      let aligned =
        match (split, split_axis x) with
        | None, None -> true
        | Some a, Some b -> a = b && a <> dim
        | Some a, None -> a = dim || List.nth (T.shape x) a = 1
        | None, Some _ -> false
      in
      if not aligned then
        invalid_arg
          (Printf.sprintf
             "Op.scatter_indexed: %s must be split like self off dim and \
              whole along dim"
             name))
    [ ("index", index); ("src", src) ];
  if List.nth ish dim = 0 || prod tsh = 0 then t
  else begin
    let expanded = Tolk.Prepare.detect_expanded (T.uop index) in
    let index =
      if List.length expanded <> rank then index
      else
        Movement.shrink_to index
          (List.mapi
             (fun d e ->
               if d <> dim && e && split_axis index <> Some d then Some 1
               else None)
             expanded)
    in
    let tsh = slices t and ish = slices index and ssh = slices src in
    let count = List.nth ish dim and extent = List.nth tsh dim in
    let first_row =
      match (split, T.device t) with
      | Some a, Some device when a = dim ->
          Some
            (Uop.alu_binary ~op:Ops.Mul
               ~lhs:(device_position (device_count device))
               ~rhs:(Uop.const_int extent))
      | _ -> None
    in
    let fxn = function
      | [ out; index; src ] ->
          let open Uop.O in
          let flat u sh = Uop.reshape ~src:u ~shape:(Uop.const_int (prod sh)) in
          let out = flat out tsh in
          let index = flat index ish and src = flat src ssh in
          let range d size kind =
            if size = 1 then Uop.const_int 0
            else Uop.range ~size:(Uop.const_int size) ~axis:d ~kind ()
          in
          let serial = if unique then Axis_type.Weak else Axis_type.Reduce in
          let k = range dim count serial in
          let coords =
            List.mapi
              (fun d s -> if d = dim then k else range d s Axis_type.Weak)
              tsh
          in
          let address sh coords =
            let _, terms =
              List.fold_right2
                (fun s c (stride, terms) ->
                  ( Stdlib.( * ) stride s,
                    if s = 1 then terms
                    else (c * Uop.const_int stride) :: terms ))
                sh coords (1, [])
            in
            match terms with
            | [] -> Uop.const_int 0
            | first :: rest -> List.fold_left ( + ) first rest
          in
          let read ptr sh =
            Uop.load ~src:(Uop.index ~ptr ~idxs:[ address sh coords ] ()) ()
          in
          let row = read index ish in
          let row =
            match first_row with
            | None -> row
            | Some first -> Uop.cast ~src:row ~dtype:D.weakint - first
          in
          let bound n = Uop.const (Const.int (Uop.dtype row) n) in
          let in_bounds =
            Uop.alu_binary ~op:Ops.And
              ~lhs:(not_ (row < bound 0))
              ~rhs:(row < bound extent)
          in
          let target =
            let row = Uop.cast ~src:row ~dtype:D.weakint in
            Uop.valid
              ~src:
                (address tsh
                   (List.mapi (fun d c -> if d = dim then row else c) coords))
              ~cond:in_bounds
          in
          let cell = Uop.index ~ptr:out ~idxs:[ target ] () in
          let update = read src ssh in
          let value =
            match mode with
            | `Set -> update
            | `Add -> Uop.load ~src:cell () + update
          in
          let store = Uop.store ~dst:cell ~value () in
          let is_range u = Uop.op u = Ops.Range in
          let body =
            Uop.end_ ~value:store ~ranges:(List.filter is_range coords)
          in
          let name =
            Printf.sprintf "scatter_%s%s_%d_%d_%s"
              (match mode with `Set -> "set" | `Add -> "add")
              (if unique then "_unique" else "")
              dim count
              (String.concat "_" (List.map string_of_int tsh))
          in
          Uop.sink
            ~kernel_info:
              {
                Uop.name;
                applied_opts = [];
                opts_to_apply = (if unique then None else Some []);
                estimates = None;
                beam = 0;
              }
            [ body ]
      | _ -> assert false
    in
    (* A kernel argument is storage. The scheduler realizes a computed operand,
       but a constant owns no storage to realize, so it is computed into a
       fresh buffer. *)
    let device = List.find_map T.device [ t; index; src ] in
    let stored x =
      if Option.is_some (T.device x) then x else Creation.clone ?device x
    in
    List.hd (T.custom_kernel ~fxn [ t; stored index; stored src ])
  end

(* Quantised matrix product

   No tinygrad counterpart. A product with MXFP4 weights written as a tensor
   composition decodes every weight to a float before it multiplies, and the
   matrix-vector options do not apply to a reduction whose source is a decode.
   This custom kernel reads each 32-value group's code bytes and scale byte
   once per tile of rows, decodes the codes in registers and multiplies each
   group's partial sums by its scale. Its options are pinned per device, so
   neither the heuristic nor a search picks them.

   Codes, and inputs narrower than float32, are read as float32 words, eight
   codes or two inputs in each, since only float loads fold into vectors. A
   code becomes the IEEE half with its bits in place, whose value is 2^-14
   times the code's, half codes included; converted to float32 and scaled back
   it is exact.

   With ids, the positions are a global axis that the options never split: the
   loop over a row's groups is an outer loop, whose bound on a GPU is zero for
   a position whose id selects no matrix, around a constant loop that the group
   option splits. Such a position reads its id, runs no multiply-adds and
   stores the reduction's identity. The CPU runs work groups as a loop and
   miscompiles a loop bound that reads that loop's index, so there the bound is
   constant: an id outside the matrices reads matrix 0, every load stays in
   bounds, and a select zeroes the store. Gating each load on the id instead
   made clang spill the narrow-input unpacking, nearly three times the
   instructions at bfloat16. *)

type quant_options = {
  group : int; (* threads splitting a row's groups, dividing k / 32 *)
  local : int; (* output columns per work group, on a local axis *)
  upcast : int; (* output columns per thread *)
  tile : int; (* rows of x per load of a group *)
}

let largest_divisor n ~at_most =
  let rec go d = if n mod d = 0 then d else go (d - 1) in
  go (max 1 (min n at_most))

(* The options measured on a device for [m] rows of [n] outputs over [k]
   inputs. On the M1 Max, one row of gpt-oss's 5760 by 2880 expert reads its
   packed bytes at 180 GB/s and each further row costs about 20 us, bound by
   the reloads of x from the cache; a thread holds all 32 inputs of a group for
   each of its rows, so columns per thread times rows beyond 4 spill registers
   and run ten times slower. A device without measured options runs the kernel
   unoptimised and serves no rows. *)
let quant_row_tile ren =
  match Tolk.Renderer.device ren with "METAL" | "CPU" -> 8 | _ -> 1

let quant_options ren ~m ~n ~k =
  let groups = k / 32 in
  match Tolk.Renderer.device ren with
  | "METAL" ->
      let local = largest_divisor n ~at_most:4 in
      let tile = largest_divisor m ~at_most:(quant_row_tile ren) in
      {
        group = largest_divisor groups ~at_most:15;
        local;
        upcast = largest_divisor (n / local) ~at_most:(max 1 (4 / tile));
        tile;
      }
  | "CPU" ->
      {
        group = 1;
        local = 1;
        upcast = largest_divisor n ~at_most:4;
        tile = largest_divisor m ~at_most:(quant_row_tile ren);
      }
  | _ -> { group = 1; local = 1; upcast = 1; tile = 1 }

(* The row bound: one value per device, the most rows a matrix may meet in the
   kernel before decoding it, or every expert of a grouped product once, and
   multiplying with the block kernel costs less. It is measured on one
   gpt-oss gate_up expert (5760 by 2880) against decoding it, and on
   gpt-oss-20b's MoE block with routes grouped by expert against decoding
   every expert, at float32 and bfloat16, rows by powers of two; where the two
   products disagree it is the count whose largest loss over both, at both
   dtypes, is smallest. On the M1 Max's GPU both products agree: the kernel
   won up to 8 rows (per expert, grouped) and lost from 16, except one
   bfloat16 matrix, 4% faster in the kernel at 16. On its CPU the grouped
   kernel won through 64 rows per expert (bfloat16 1.9 times faster at 64,
   float32 4%), while one matrix decoded faster from 32 rows at float32 (by
   11% at 32, 33% at 64) and from 128 at bfloat16: 64, whose largest loss is
   float32's 33% at 64 rows of one matrix, where 32 would cost grouped
   bfloat16 1.9 times at 64 rows per expert. Both lie within the read model:
   decoding costs [p + 2q] for [p] packed and [q] decoded bytes, and the kernel
   reads [p] once per tile of 8 rows, so it never pays past the largest [r]
   with [ceil (r / 8) p <= p + 2q]: 64 rows at 16-bit dtypes, where the CPU's
   bound sits, and 128 at float32. *)
let quant_row_bound ren =
  match Tolk.Renderer.device ren with "METAL" -> 8 | "CPU" -> 64 | _ -> 0

(* An option splitting the first axis. With several positions that axis is
   theirs, whose loop bound must stay one value per work group, so
   [quant_matmul] refuses such options. *)
let splits_axis0 (opt : Uop.Opt.t) =
  match opt with
  | Split { axis; _ } | Padto { axis; _ } -> axis = 0
  | Swap { axis; with_axis } -> axis = 0 || with_axis = 0
  | Tc _ -> false

let split_opt axis amount kind =
  Uop.Opt.Split { axis; amount; kind; top = false }

let quant_matmul ?ids x ~codes ~scales =
  let xs = T.shape x and cs = T.shape codes and ss = T.shape scales in
  let ix, m, k =
    match xs with
    | [ ix; m; k ] -> (ix, m, k)
    | _ ->
        invalid_arg "Op.quant_matmul: x must be [instances; rows; inputs]"
  in
  let e, n =
    match cs with
    | [ e; n; half ] when 2 * half = k -> (e, n)
    | _ ->
        invalid_arg "Op.quant_matmul: codes must be [matrices; outputs; k/2]"
  in
  if k mod 32 <> 0 then
    invalid_arg "Op.quant_matmul: inputs must be a multiple of 32";
  if ss <> [ e; n; k / 32 ] then
    invalid_arg "Op.quant_matmul: scales must be [matrices; outputs; k/32]";
  if not (D.equal (T.dtype codes) D.uint8 && D.equal (T.dtype scales) D.uint8)
  then invalid_arg "Op.quant_matmul: codes and scales must be uint8";
  let dtype = T.dtype x in
  let narrow =
    match dtype with
    | D.Float32 -> false
    | D.Bfloat16 | D.Float16 -> true
    | _ ->
        invalid_arg "Op.quant_matmul: x must be float32, bfloat16 or float16"
  in
  let i =
    match ids with
    | None -> e
    | Some ids -> (
        if not (D.is_int (T.dtype ids)) then
          invalid_arg "Op.quant_matmul: integer ids required";
        match T.shape ids with
        | [ i ] -> i
        | _ -> invalid_arg "Op.quant_matmul: ids must be [instances]")
  in
  if (ix = 0 && i > 0) || (ix > 0 && i mod ix <> 0) then
    invalid_arg "Op.quant_matmul: x's instances must divide the product's";
  let device =
    match List.find_map T.device [ x; codes; scales ] with
    | Some device -> device
    | None -> invalid_arg "Op.quant_matmul: no operand is placed on a device"
  in
  let ren =
    match device with
    | Uop.Single name | Uop.Multi (name :: _) ->
        Tolk.Device.renderer (Tolk.Device.get name)
    | Uop.Multi [] | Uop.Index _ ->
        invalid_arg "Op.quant_matmul: no device name"
  in
  (* Over split operands each device multiplies its own instances, rows or
     columns. Whole instances over matrices split along their first axis, or
     over split inputs, leave each device a partial product: ids name matrices
     by their position in the whole, and a device offsets them by its first
     matrix. Without ids, instance [t] is matrix [t], so both split alike.
     Instances of one block of [x] stay on one device. *)
  let parts = split_axis codes in
  if split_axis scales <> parts then
    invalid_arg "Op.quant_matmul: codes and scales split differently";
  let instances =
    match ids with Some ids -> split_axis ids | None -> parts
  in
  let result =
    match (split_axis x, instances, parts) with
    | None, None, None -> Whole
    | Some 0, Some 0, None -> Split 0
    | None, Some 0, None when ix = 1 -> Split 0
    | Some 0, Some 0, Some 0 when Option.is_none ids -> Split 0
    | None, Some 0, Some 0 when Option.is_none ids && ix = 1 -> Split 0
    | Some 1, None, None -> Split 1
    | None, None, Some 1 -> Split 2
    | None, None, Some 0 when Option.is_some ids -> Partial
    | Some 2, None, Some 2 -> Partial
    | _ ->
        invalid_arg
          "Op.quant_matmul: operands split other than by instances, rows, \
           columns, matrices or inputs"
  in
  let out, finish = result_storage ~dtype ~device result [ i; m; n ] in
  let zeros () =
    Creation.clone ~device (Creation.zeros ~dtype ~buffer:false [ i; m; n ])
  in
  let ix, m, k =
    match slices x with [ ix; m; k ] -> (ix, m, k) | _ -> assert false
  in
  let e, n =
    match slices codes with [ e; n; _ ] -> (e, n) | _ -> assert false
  in
  let i = match ids with Some ids -> List.hd (slices ids) | None -> e in
  let first =
    if Option.is_some ids && parts = Some 0 then
      Some
        (Uop.alu_binary ~op:Ops.Mul
           ~lhs:(device_position (device_count device))
           ~rhs:(Uop.const_int e))
    else None
  in
  if i * m * n = 0 then finish out
  else if k = 0 then zeros ()
  else begin
    let gpu = Tolk.Renderer.has_local ren in
    let groups = k / 32 and gated = Option.is_some ids in
    (* Without ids, one row per instance makes instances and columns one
       axis: output, codes and scales are all linear in [t * n + col], and
       the range simplifier would merge the two anyway. *)
    let merged = (not gated) && m = 1 in
    let positions = if merged then 1 else i
    and cols = if merged then i * n else n in
    let o = quant_options ren ~m ~n:cols ~k in
    (* The id bounds the outer loop on a GPU only while that loop has two
       iterations or more. In a loop of at most one, every use of the index
       folds to 0, and the reduce over it, left unparented, is rewritten to
       its body times the loop's size (reduce_unparented, as in tinygrad): the
       body's loads then run for an invalid id too, and Metal returns garbage
       there. Under an upcast or a group such a kernel also fails to compile,
       likely from the same rewrite (not traced). Before 8b26ea10a the range
       rule also folded the loop itself. A single group reads matrix 0 for
       such an id instead, as on the CPU, so it runs its multiply-adds. *)
    let group =
      if not gpu then 1
      else if gated then
        largest_divisor groups ~at_most:(min o.group (groups / 2))
      else o.group
    in
    let outer = groups / group and rep = i / ix in
    let x_span = n * rep in
    let bounded = gpu && gated && outer >= 2 in
    (* Axes: positions, rows and columns are global, in that order; the outer
       loop, the group loop and the four words of a group reduce. *)
    let exists size = if size > 1 then 1 else 0 in
    let col_axis = exists positions + exists m
    and row_axis = exists positions in
    (* Split axes index every live range, including the id-bounded reduce.
       Unrolling removes the word axis without moving the earlier axes. *)
    let globals = exists positions + exists m + exists cols in
    let unrollable = globals + exists outer + exists group in
    let group_axis = globals + exists outer in
    let opts =
      List.concat
        [
          [ split_opt unrollable 4 Unroll ];
          (if group > 1 then
             [ split_opt group_axis group Local ]
           else []);
          (if o.local > 1 && gpu then
             [ split_opt col_axis o.local Local ]
           else []);
          (if o.upcast > 1 then
             [ split_opt col_axis o.upcast Upcast ]
           else []);
          (if o.tile > 1 then
             [ split_opt row_axis o.tile Upcast ]
           else []);
        ]
    in
    if positions > 1 then
      Option.iter
        (fun opt ->
          invalid_arg
            (Printf.sprintf "Op.quant_matmul: option %s splits the positions"
               (Uop.Opt.to_string opt)))
        (List.find_opt splits_axis0 opts);
    let fxn srcs =
      let out, x, codes, scales, ids =
        match srcs with
        | [ out; x; codes; scales ] -> (out, x, codes, scales, None)
        | [ out; x; codes; scales; ids ] -> (out, x, codes, scales, Some ids)
        | _ -> assert false
      in
      let flat u size = Uop.reshape ~src:u ~shape:(Uop.const_int size) in
      (* Words: the same storage, read four bytes at a time. *)
      let words slot size =
        Uop.placeholder ~shape:[ size ] ~dtype:D.float32 ~slot ()
      in
      let out = flat out (i * m * n) in
      let x =
        if narrow then words 1 (ix * m * k / 2) else flat x (ix * m * k)
      in
      let x_row = if narrow then k / 2 else k in
      let codes = words 2 (e * n * k / 8) and code_row = k / 8 in
      let scales = flat scales (e * n * groups) in
      let nibble j = 4 * j in
      let open Uop.O in
      let int = Uop.const_int in
      let u32 v = Uop.const (Const.int D.uint32 v) in
      let f32 v = Uop.const (Const.float D.float32 v) in
      let bits op a b = Uop.alu_binary ~op ~lhs:a ~rhs:b in
      (* The float32 value of the half whose bits are the low 16 of [u]. *)
      let of_half u =
        Uop.cast
          ~src:
            (Uop.bitcast ~src:(Uop.cast ~src:u ~dtype:D.uint16) ~dtype:D.float16)
          ~dtype:D.float32
      in
      let range size axis =
        if size = 1 then int 0
        else Uop.range ~size:(int size) ~axis ~kind:Axis_type.Weak ()
      in
      let reduce size axis =
        if size = 1 then int 0
        else Uop.range ~size:(int size) ~axis ~kind:Axis_type.Reduce ()
      in
      let at ptr idx = Uop.index ~ptr ~idxs:[ idx ] () in
      let pos = range positions 0 and row = range m 1 and col = range cols 2 in
      let selects, matrix =
        match ids with
        | None -> (None, pos)
        | Some ids ->
            let id = at (flat ids i) pos in
            let id =
              match first with
              | None -> id
              | Some first -> Uop.cast ~src:id ~dtype:D.weakint - first
            in
            let bound v = Uop.const (Const.int (Uop.dtype id) v) in
            let selects = bits Ops.And (not_ (id < bound 0)) (id < bound e) in
            let id = if bounded then id else where selects id (bound 0) in
            (Some selects, Uop.cast ~src:id ~dtype:D.weakint)
      in
      let t =
        match selects with
        | Some selects when bounded ->
            Uop.range ~size:(where selects (int outer) (int 0)) ~axis:3
              ~kind:Axis_type.Reduce ()
        | _ -> reduce outer 3
      in
      let a = reduce group 4 and q = reduce 4 5 in
      let g = (t * int group) + a in
      let load ptr idx = Uop.load ~src:(at ptr idx) () in
      let line = if merged then col else (matrix * int n) + col in
      let word =
        Uop.bitcast
          ~src:(load codes ((line * int code_row) + (g * int 4) + q))
          ~dtype:D.uint32
      in
      (* Input [j] of the eight a word's codes multiply, as float32. *)
      let xrow =
        if merged then col // int x_span
        else ((if rep = 1 then pos else pos // int rep) * int m) + row
      in
      let input j =
        let idx = (g * int 32) + (q * int 8) + int j in
        if not narrow then load x ((xrow * int x_row) + idx)
        else
          let w =
            Uop.bitcast
              ~src:(load x ((xrow * int x_row) + (idx // int 2)))
              ~dtype:D.uint32
          in
          let half =
            if j land 1 = 0 then bits Ops.And w (u32 0xffff)
            else bits Ops.Shr w (u32 16)
          in
          match dtype with
          | D.Bfloat16 ->
              Uop.bitcast ~src:(bits Ops.Shl half (u32 16)) ~dtype:D.float32
          | _ -> of_half half
      in
      let code j =
        let c = bits Ops.And (bits Ops.Shr word (u32 (nibble j))) (u32 15) in
        of_half
          (bits Ops.Or
             (bits Ops.Shl (bits Ops.And c (u32 7)) (u32 9))
             (bits Ops.Shl (bits Ops.And c (u32 8)) (u32 12)))
        * f32 16384.0
      in
      let scale =
        let s =
          Uop.cast ~src:(load scales ((line * int groups) + g)) ~dtype:D.int32
        in
        let i32 v = Uop.const (Const.int D.int32 v) in
        Uop.bitcast
          ~src:
            (where (s < i32 1) (i32 0x00400000)
               (where (Uop.alu_binary ~op:Ops.Cmpeq ~lhs:s ~rhs:(i32 255))
                  (i32 0x7fc00000)
                  (bits Ops.Shl s (i32 23))))
          ~dtype:D.float32
      in
      let is_range u = Uop.op u = Ops.Range in
      (* A reduce of its own for each loop keeps the outer loop apart from the
         group loop, which the range simplifier would otherwise merge. *)
      let sum r v =
        if is_range r then
          Uop.reduce ~src:v ~ranges:[ r ] ~op:Ops.Add ~dtype:D.float32
        else v
      in
      let terms = List.init 8 (fun j -> input j * code j) in
      let partial =
        sum q (List.fold_left ( + ) (List.hd terms) (List.tl terms))
      in
      let acc = sum a (sum t (partial * scale)) in
      let acc =
        match selects with
        | Some selects when not bounded -> where selects acc (f32 0.0)
        | _ -> acc
      in
      let store =
        Uop.store
          ~dst:
            (at out
               (if merged then col else (((pos * int m) + row) * int n) + col))
          ~value:(Uop.cast ~src:acc ~dtype:(Uop.dtype out)) ()
      in
      Uop.sink
        ~kernel_info:
          {
            Uop.name = Printf.sprintf "quant_matmul_%d_%d_%d_%d" i m n k;
            applied_opts = [];
            opts_to_apply = Some opts;
            estimates = None;
            beam = 0;
          }
        [
          Uop.end_ ~value:store
            ~ranges:(List.filter is_range [ pos; row; col ]);
        ]
    in
    let stored t =
      if Option.is_some (T.device t) then t else Creation.clone ~device t
    in
    let srcs = [ out; stored x; stored codes; stored scales ] in
    let srcs =
      match ids with None -> srcs | Some ids -> srcs @ [ stored ids ]
    in
    finish (List.hd (T.custom_kernel ~fxn srcs))
  end

(* Block matrix product

   No tinygrad counterpart. A product of blocks of rows, each by the matrix of
   a stack that its id addresses, written as [matmul] over [w] gathered by the
   ids costs a copy of the gathered matrices, and a block whose id selects no
   matrix still multiplies. This custom kernel reads each block's matrix in
   place, and splits the contraction in two loops: an outer loop over tiles of
   [depth], whose bound on a GPU is zero for a block whose id selects no matrix,
   around a constant loop of [depth] that the tensor-core option splits. Such
   a block reads its id, runs no multiply-adds and stores the reduction's
   identity. The bound is one value per work group only while the block axis is
   a global dimension of its own, so the options never split it: an upcast
   block axis would make the bound a vector. The CPU runs work groups as a loop
   and miscompiles a loop bound that reads that loop's index, so there the
   bound is constant, the weight's load is gated and a select zeroes the store.
   So is a contraction of one input anywhere: tolk folds the index of a loop
   read from memory whose size is at most 1 to 0, and the loop drops out of its
   reduce whatever its size at run time, so a bounded loop keeps at least two
   trips (workaround: remove the one-input case and [depth]'s [k > 8] when
   tolk keeps such a loop). *)

(* [block_axis_kept ren kernel ~nb] is whether [kernel]'s block axis, axis 0,
   is still one whole parallel dimension of [nb] once its options apply. *)
let block_axis_kept ren kernel ~nb =
  let k = Tolk.Postrange.create kernel ren in
  Tolk.Postrange.convert_loop_to_global k;
  let opts =
    match Uop.as_kernel_info kernel with
    | Some { opts_to_apply = Some opts; _ } -> opts
    | _ -> []
  in
  List.iter (fun opt -> ignore (Tolk.Postrange.apply_opt k opt)) opts;
  List.exists
    (fun r ->
      match Uop.as_range r with
      | Some { axis = 0; sub = []; kind = Axis_type.Global | Axis_type.Weak; _ }
        ->
          Tolk.Postrange.range_int_size r = nb
      | _ -> false)
    (Tolk.Postrange.rngs k)

(* The tensor cores serve a shape on Metal whose outputs and contraction are
   tiles of 8. The kernel multiplies float32, so they are the float32 ones,
   whatever the operands' dtype. They multiply 8 by 8 tiles: the options
   upcast the rows by up to 8 tiles and the columns by 3, then split the
   columns 4 ways across a work group, measured at gpt-oss's shapes. *)
let block_tensor_cores ren ~n ~k =
  Tolk.Renderer.device ren = "METAL"
  && List.exists
       (fun (tc : Tolk.Tc.t) ->
         D.equal tc.dtype_in D.float32 && D.equal tc.dtype_out D.float32)
       (Tolk.Renderer.tensor_cores ren)
  && n mod 8 = 0 && k mod 8 = 0 && k > 8

let row_upcasts = [ 8; 4; 2; 1 ]

let block_row_tiles ren ~n ~k =
  if block_tensor_cores ren ~n ~k then
    List.map (fun u -> 8 * u) row_upcasts
  else []

(* The contraction's tile depth and the options, per renderer and shape,
   pinned from measurements. The block axis is axis 0 when there are several
   blocks; the rows, then the columns, then the contraction's tiles follow
   it.

   On the CPU the options keep a register tile of at most 64 accumulators,
   float32 at every dtype: rows upcast by up to 8, columns by up to 16 within
   that bound, and the loop over the contraction's tiles unrolled by 4. 64 is
   half of a 128-float vector register file (32 NEON registers, or 16 AVX2
   ones), leaving the other half to the loads of x and w; it under-uses
   AVX-512's 32 registers of 16 floats. The constants were measured on one M1
   Max core at gpt-oss's shapes (5760 outputs, 2880 inputs) against no
   options: 9 to 37 times faster at float32 from blocks of one row to 64, and
   8 to 18 times at bfloat16, whose products are float32 too. A tile of 16 by
   8 (128 accumulators) and one of 4 by 8 (32) were both slower than 8 by 8 at
   float32; at bfloat16 the tile of 4 by 8 was 6% faster at blocks of 8 rows
   and one of 8 columns 22% slower at one row, so one rule serves both. *)
let block_options ren ~nb ~m ~n ~k =
  let depth = if k mod 8 = 0 && k > 8 then 8 else 1 in
  let first = if nb > 1 then 1 else 0 in
  let largest l size = List.find (fun u -> size mod u = 0) l in
  let opt amount o = if amount > 1 then [ o ] else [] in
  let opts =
    if block_tensor_cores ren ~n ~k && m mod 8 = 0 then
      let rows = m / 8 and cols = n / 8 in
      let ur = largest row_upcasts rows in
      let uc = largest [ 3; 2; 1 ] cols in
      let lc = largest [ 4; 2; 1 ] (cols / uc) in
      let col = first + if rows / ur > 1 then 1 else 0 in
      (Uop.Opt.Tc { axis = 0; tc_select = -1; tc_opt = 0; use_tc = 1 }
      :: opt ur (split_opt first ur Upcast))
      @ opt uc (split_opt col uc Upcast)
      @ opt lc (split_opt col lc Local)
    else if Tolk.Renderer.device ren = "CPU" then
      let ur = largest row_upcasts m in
      let uc = largest (List.filter (fun u -> ur * u <= 64) [ 16; 8; 4; 2; 1 ]) n in
      let ut = largest [ 4; 2; 1 ] (k / depth) in
      let left size u = if size / u > 1 then 1 else 0 in
      let split u = if u > 1 then 1 else 0 in
      let col = first + left m ur in
      let tile = col + left n uc + split ur + split uc in
      opt ur (split_opt first ur Upcast)
      @ opt uc (split_opt col uc Upcast)
      @ opt ut (split_opt tile ut Unroll)
    else []
  in
  (depth, opts)

let block_matmul ?(transpose = false) x w ~ids =
  let nb, m, k =
    match T.shape x with
    | [ nb; m; k ] -> (nb, m, k)
    | _ -> invalid_arg "Op.block_matmul: x must be [blocks; rows; inputs]"
  in
  let e, n =
    match (T.shape w, transpose) with
    | [ e; k'; n ], false when k' = k -> (e, n)
    | [ e; n; k' ], true when k' = k -> (e, n)
    | _ -> invalid_arg "Op.block_matmul: w does not match x's inputs"
  in
  if T.shape ids <> [ nb ] then
    invalid_arg "Op.block_matmul: ids must be [blocks]";
  if not (D.is_int (T.dtype ids)) then
    invalid_arg "Op.block_matmul: integer ids required";
  let dtype = T.dtype x in
  if not (D.equal dtype (T.dtype w)) then
    invalid_arg "Op.block_matmul: x and w must have the same dtype";
  if not (D.is_float dtype && D.itemsize dtype <= 4) then
    invalid_arg "Op.block_matmul: x must be a float of at most 32 bits";
  let device =
    match List.find_map T.device [ x; w; ids ] with
    | Some device -> device
    | None -> invalid_arg "Op.block_matmul: no operand is placed on a device"
  in
  let ren =
    match device with
    | Uop.Single name | Uop.Multi (name :: _) ->
        Tolk.Device.renderer (Tolk.Device.get name)
    | Uop.Multi [] | Uop.Index _ ->
        invalid_arg "Op.block_matmul: no device name"
  in
  (* Over split operands each device multiplies its own blocks, rows or
     columns. Whole blocks over matrices split along their first axis, or over
     split inputs, leave each device a partial product: ids name matrices by
     their position in the whole, and a device offsets them by its first
     matrix. *)
  let columns = if transpose then 1 else 2
  and inputs = if transpose then 2 else 1 in
  let result =
    match (split_axis x, split_axis ids, split_axis w) with
    | None, None, None -> Whole
    | Some 0, Some 0, None -> Split 0
    | Some 1, None, None -> Split 1
    | None, None, Some c when c = columns -> Split 2
    | None, None, Some 0 -> Partial
    | Some 2, None, Some c when c = inputs -> Partial
    | _ ->
        invalid_arg
          "Op.block_matmul: operands split other than by blocks, rows, \
           columns, matrices or inputs"
  in
  let out, finish = result_storage ~dtype ~device result [ nb; m; n ] in
  let nb, m, k =
    match slices x with [ nb; m; k ] -> (nb, m, k) | _ -> assert false
  in
  let e, n =
    match slices w with
    | [ e; _; n ] when not transpose -> (e, n)
    | [ e; n; _ ] -> (e, n)
    | _ -> assert false
  in
  let first =
    if split_axis w = Some 0 then
      Some
        (Uop.alu_binary ~op:Ops.Mul
           ~lhs:(device_position (device_count device))
           ~rhs:(Uop.const_int e))
    else None
  in
  if nb * m * n = 0 then finish out
  else begin
    let depth, opts = block_options ren ~nb ~m ~n ~k in
    let bounded = Tolk.Renderer.has_local ren && k / depth > 1 in
    let fxn = function
      | [ out; x; w; ids ] ->
          let flat u size = Uop.reshape ~src:u ~shape:(Uop.const_int size) in
          let out = flat out (nb * m * n) and x = flat x (nb * m * k) in
          let w = flat w (e * n * k) and ids = flat ids nb in
          let tiles = k / depth in
          let open Uop.O in
          let range size axis kind =
            if size = 1 then Uop.const_int 0
            else Uop.range ~size:(Uop.const_int size) ~axis ~kind ()
          in
          let block = range nb 0 Axis_type.Weak in
          let row = range m 1 Axis_type.Weak in
          let col = range n 2 Axis_type.Weak in
          let load ptr idx =
            Uop.load ~src:(Uop.index ~ptr ~idxs:[ idx ] ()) ()
          in
          let id = load ids block in
          let id =
            match first with
            | None -> id
            | Some first -> Uop.cast ~src:id ~dtype:D.weakint - first
          in
          let bound v = Uop.const (Const.int (Uop.dtype id) v) in
          let selects =
            Uop.alu_binary ~op:Ops.And
              ~lhs:(not_ (id < bound 0))
              ~rhs:(id < bound e)
          in
          let tile =
            if bounded then
              Uop.range
                ~size:(where selects (Uop.const_int tiles) (Uop.const_int 0))
                ~axis:3 ~kind:Axis_type.Reduce ()
            else range tiles 3 Axis_type.Reduce
          in
          let inner = range depth 4 Axis_type.Reduce in
          let c = (tile * Uop.const_int depth) + inner in
          let id = Uop.cast ~src:id ~dtype:D.weakint in
          let ( *: ) i size = i * Uop.const_int size in
          let xv = load x ((((block *: m) + row) *: k) + c) in
          let waddr =
            if transpose then (((id *: n) + col) *: k) + c
            else (((id *: k) + c) *: n) + col
          in
          let wv =
            load w
              (if bounded then waddr else Uop.valid ~src:waddr ~cond:selects)
          in
          let is_range u = Uop.op u = Ops.Range in
          let f32 v = Uop.cast ~src:v ~dtype:D.float32 in
          let acc =
            Uop.reduce ~src:(f32 xv * f32 wv)
              ~ranges:(List.filter is_range [ tile; inner ])
              ~op:Ops.Add ~dtype:D.float32
          in
          let acc =
            if bounded then acc
            else where selects acc (Uop.const (Const.float D.float32 0.0))
          in
          let cell =
            Uop.index ~ptr:out ~idxs:[ (((block *: m) + row) *: n) + col ] ()
          in
          let store =
            Uop.store ~dst:cell
              ~value:(Uop.cast ~src:acc ~dtype:(Uop.dtype out))
              ()
          in
          let body =
            Uop.end_ ~value:store
              ~ranges:(List.filter is_range [ block; row; col ])
          in
          let kernel =
            Uop.sink
              ~kernel_info:
                {
                  Uop.name =
                    Printf.sprintf "block_matmul%s_%d_%d_%d_%d_%d"
                      (if transpose then "_t" else "")
                      nb m n k e;
                  applied_opts = [];
                  opts_to_apply = Some opts;
                  estimates = None;
                  beam = 0;
                }
              [ body ]
          in
          if nb > 1 && not (block_axis_kept ren kernel ~nb) then
            invalid_arg
              (Printf.sprintf
                 "Op.block_matmul: the options [%s] split the block axis"
                 (String.concat "; " (List.map Uop.Opt.to_string opts)));
          kernel
      | _ -> assert false
    in
    let stored t =
      if Option.is_some (T.device t) then t else Creation.clone ~device t
    in
    finish
      (List.hd (T.custom_kernel ~fxn [ out; stored x; stored w; stored ids ]))
  end

(* Indexing *)

let rec getitem t indices =
  let indices = Movement.normalize_indices t indices in
  let parsed =
    let rec loop dim = function
      | [] -> []
      | index :: rest ->
          let size =
            match index with
            | Movement.New -> Uop.const_int 1
            | _ -> List.nth (T.symbolic_shape t) dim
          in
          let p =
            match index with
            | Movement.T tensor ->
                if Option.is_none (Uop.const_int_value size) then
                  invalid_arg "Op.getitem: advanced indexing needs a concrete axis size";
                if not (D.is_int (T.dtype tensor)) then
                  invalid_arg "Op.getitem: index tensor must be integer";
                (match T.device tensor, T.device t with
                 | Some a, Some b when a <> b ->
                     invalid_arg "Op.getitem: index and tensor devices differ"
                 | _ -> ());
                let tensor =
                  Elementwise.where
                    (Elementwise.lt tensor (T.i 0))
                    (Elementwise.add tensor (T.of_uop size))
                    tensor
                in
                {
                  Movement.size;
                  boundary = (Uop.const_int 0, size);
                  stride = 1;
                  collapse_dim = false;
                  resolved = Movement.Advanced tensor;
                }
            | _ -> Movement.parse_view_index index size
          in
          let next =
            match p.Movement.resolved with Movement.Newaxis -> dim | _ -> dim + 1
          in
          p :: loop next rest
    in
    loop 0 indices
  in
  let is_newaxis p =
    match p.Movement.resolved with Movement.Newaxis -> true | _ -> false
  in
  let is_adv p =
    match p.Movement.resolved with Movement.Advanced _ -> true | _ -> false
  in
  let adv_tensor p =
    match p.Movement.resolved with Movement.Advanced tn -> tn | _ -> assert false
  in
  let mops = List.filter (fun p -> not (is_newaxis p)) parsed in
  let x = Movement.apply_view_ops t mops in
  let x_dims = List.filter (fun p -> not p.Movement.collapse_dim) parsed in
  let x = Movement.symbolic_reshape x (List.map (fun p -> p.Movement.size) x_dims) in
  let tops =
    List.concat
      (List.mapi (fun d p -> if is_adv p then [ (d, adv_tensor p) ] else []) x_dims)
  in
  match tops with
  | [] -> x
  | _ ->
      let dims = List.map fst tops and tensors = List.map snd tops in
      (* Indexed axes that are not consecutive put the broadcast axes first:
         moved to the front, they are consecutive and take the same linear
         gather. *)
      let x, dims =
        let d0 = List.hd dims in
        if dims = List.init (List.length dims) (fun i -> d0 + i) then (x, dims)
        else
          let rest =
            List.filter
              (fun d -> not (List.mem d dims))
              (List.init (T.ndim x) Fun.id)
          in
          (Movement.permute x (dims @ rest), List.init (List.length dims) Fun.id)
      in
      let big_shape = Uop.broadcast_shape (List.map T.symbolic_shape tensors) in
      let bshape_len = List.length big_shape in
      let d0 = List.hd dims in
      let dlast = List.nth dims (List.length dims - 1) in
      let xshape = T.symbolic_shape x in
      let axis_size d =
        match Uop.const_int_value (List.nth xshape d) with
        | Some n -> n
        | None ->
            invalid_arg "Op.getitem: advanced indexing needs a concrete axis size"
      in
      if List.length dims > 1 then (
        (* Several integer-tensor indices: one linear gather over the
           flattened block. The tinygrad counterpart sums a product of one
           mask per axis, which does not reduce to a load: the sum turns -0
           into +0, and split, it multiplies a masked NaN by zero. *)
        let ishp = List.map axis_size dims in
        let strides = List.mapi (fun i _ -> prod (drop (i + 1) ishp)) ishp in
        let linear_idx =
          match
            List.map2
              (fun tn s ->
                Elementwise.mul
                  (Movement.symbolic_broadcast_to tn big_shape) (T.i s))
              tensors strides
          with
          | h :: tl -> Elementwise.usum h tl
          | [] -> assert false
        in
        let valid =
          match
            List.map2
              (fun tn s ->
                Elementwise.bitwise_and
                  (Elementwise.ge tn (T.i 0))
                  (Elementwise.lt tn (T.i s)))
              tensors ishp
          with
          | h :: tl -> Elementwise.uprod h tl
          | [] -> assert false
        in
        let pre = take d0 xshape and post = drop (dlast + 1) xshape in
        let flat =
          Movement.symbolic_reshape x
            (pre @ [ Uop.const_int (prod ishp) ] @ post)
        in
        let gathered =
          getitem flat
            (List.init (List.length pre) (fun _ -> Movement.All)
            @ [ Movement.T (Elementwise.where valid linear_idx (T.i 0)) ])
        in
        let valid_shape =
          List.init (List.length pre) (fun _ -> Uop.const_int 1)
          @ big_shape
          @ List.init (List.length post) (fun _ -> Uop.const_int 1)
        in
        Elementwise.where (Movement.symbolic_reshape valid valid_shape) gathered
          (T.i 0))
      else
        let xndim = T.ndim x in
        let pre_reduce_shape = take d0 xshape @ big_shape @ drop d0 xshape in
        let tn = List.hd tensors in
        let i =
          Movement.symbolic_broadcast_to
            (Movement.symbolic_reshape tn
               (T.symbolic_shape tn
               @ List.init (xndim - d0) (fun _ -> Uop.const_int 1)))
            pre_reduce_shape
        in
        let mask = one_hot_along_dim ~dim:(d0 - xndim) i (axis_size d0) in
        let reshape_arg =
          take d0 xshape @ List.init bshape_len (fun _ -> Uop.const_int 1)
          @ drop d0 xshape
        in
        Reduce.sum ~axis:[ d0 + bshape_len ] ~dtype:(T.val_dtype x)
          (Elementwise.where mask (Movement.symbolic_reshape x reshape_arg) (T.i 0))

(* Boolean selection

   [masked_select] compacts the elements a boolean mask keeps into a
   fixed-length axis: the running count of kept elements gives each one its
   output slot, and a gather reads the source element for each slot. Slots past
   the number kept are filled with [fill_value]. [nonzero] applies this to the
   grid of coordinates so it returns the position of every non-zero element. *)

let masked_select ?(fill_value = T.Sint 0) t mask ~size =
  if not (D.is_bool (T.dtype mask)) then
    invalid_arg "Op.masked_select: mask must be boolean";
  let x = Movement.flatten t in
  let mask = Movement.flatten (Movement.broadcast_to mask (T.shape t)) in
  let mask_cumsum = cumsum mask in
  let counts =
    scatter_reduce
      (Creation.zeros ~dtype:D.int32 ~buffer:false [ size ])
      ~dim:0 mask_cumsum
      (Creation.ones ~dtype:D.int32 ~buffer:false [ T.numel t ])
      ~reduce:`Sum ()
  in
  let gathered = getitem x [ Movement.T (cumsum counts) ] in
  let cond = Elementwise.lt (arange size) (Reduce.sum mask) in
  Dtype_ops.cast
    (Elementwise.where cond gathered (Creation.const_like gathered fill_value))
    (T.dtype t)

let nonzero ?(fill_value = T.Sint 0) t ~size =
  let ndim = T.ndim t in
  if ndim = 0 then Creation.zeros ~dtype:D.int32 ~buffer:false [ size; 0 ]
  else
    let sh = T.shape t in
    let mask = Movement.flatten (Elementwise.ne t (T.i 0)) in
    let coords =
      List.mapi
        (fun i s ->
          Movement.flatten
            (Movement.expand
               (Movement.reshape (arange s)
                  (List.init i (fun _ -> 1) @ [ s ] @ List.init (ndim - i - 1) (fun _ -> 1)))
               sh))
        sh
    in
    let indices =
      match coords with
      | h :: tl -> Movement.stack ~dim:(-1) h tl
      | [] -> assert false
    in
    let mask = Movement.expand (Movement.unsqueeze mask (-1)) (T.shape mask @ [ ndim ]) in
    Movement.reshape
      (masked_select ~fill_value indices mask ~size:(size * ndim))
      [ -1; ndim ]

(* Argmax / argmin *)

let rec argmax ?axis ?(keepdim = false) t =
  match axis with
  | None -> argmax ~axis:0 ~keepdim (Movement.flatten t)
  | Some axis ->
      let axis = T.resolve_dim t axis in
      let n = List.nth (T.shape t) axis in
      let m = Elementwise.eq t (Reduce.max ~axis:[ axis ] ~keepdim:true t) in
      let ranks =
        Movement.reshape
          (arange ~stop:0 ~step:(-1) n)
          (n :: List.init (T.ndim t - axis - 1) (fun _ -> 1))
      in
      Dtype_ops.int
        (Elementwise.sub (T.i n)
           (Reduce.max ~axis:[ axis ] ~keepdim (Elementwise.mul m ranks)))

let argmin ?axis ?keepdim t = argmax ?axis ?keepdim (Elementwise.inverse t)

(* Sorting

   A bitonic sort: pad the axis up to a power of two, then run the fixed
   network of compare-and-swap stages, materialising between stages so the
   swaps do not re-expand into one huge expression. Indices are recovered by
   matching each sorted value back to its source position, breaking ties by the
   running count of equal elements so stable order is preserved. *)

let bit_length n =
  let rec go n acc = if n = 0 then acc else go (n lsr 1) (acc + 1) in
  go n 0

let sort ?(dim = -1) ?(descending = false) t =
  let dim = T.resolve_dim t dim in
  let orig_len = List.nth (T.shape t) dim in
  if orig_len <= 1 then
    (t, Creation.full ~dtype:D.default_int ~buffer:false (T.shape t) (T.Sint 0))
  else begin
    let ndim = T.ndim t in
    let n_stages = bit_length (orig_len - 1) in
    let pad_val =
      if descending then dtype_min_tensor t else dtype_max_tensor t
    in
    let pads =
      List.init ndim (fun i ->
          if i = dim then Some (0, (1 lsl n_stages) - orig_len) else None)
    in
    let x =
      ref
        (Movement.unflatten (pad_value t pads pad_val) dim
           (List.init n_stages (fun _ -> 2)))
    in
    let split2 d = match Movement.split ~dim:d !x 1 with [ a; b ] -> (a, b) | _ -> assert false in
    for stage = 1 to n_stages do
      let crossover_dim = dim + n_stages - stage - 1 in
      let flip_dims = List.init (stage + (ndim - dim)) (fun i -> -(i + 1)) in
      if stage <> n_stages then (
        let blue, green = split2 crossover_dim in
        x :=
          Elementwise.contiguous
            (cat ~dim:crossover_dim blue [ Movement.flip green flip_dims ]));
      for substage = stage - 1 downto 0 do
        let partner_dim = dim + n_stages - substage - 1 in
        let top, bottom = split2 partner_dim in
        let larger = Elementwise.maximum top bottom in
        let smaller = Elementwise.minimum top bottom in
        x :=
          Elementwise.contiguous
            (if descending then cat ~dim:partner_dim larger [ smaller ]
             else cat ~dim:partner_dim smaller [ larger ])
      done;
      if stage <> n_stages then (
        let blue, green = split2 crossover_dim in
        x := cat ~dim:crossover_dim blue [ Movement.flip green flip_dims ])
    done;
    let sorted =
      Movement.shrink_to
        (Movement.flatten ~start_dim:dim ~end_dim:(dim + n_stages - 1) !x)
        (List.map (fun s -> Some s) (T.shape t))
    in
    let mask =
      Movement.reshape
        (tril (Creation.ones ~dtype:D.bool ~buffer:false [ orig_len; orig_len ]))
        ([ orig_len; orig_len ] @ List.init (ndim - dim - 1) (fun _ -> 1))
    in
    let counts u =
      Reduce.sum ~axis:[ dim + 1 ]
        (Elementwise.bitwise_and mask
           (Elementwise.eq (Movement.unsqueeze u dim) (Movement.unsqueeze u (dim + 1))))
    in
    let cond =
      Elementwise.bitwise_and
        (Elementwise.eq (Movement.unsqueeze t (dim + 1)) (Movement.unsqueeze sorted dim))
        (Elementwise.eq
           (Movement.unsqueeze (counts t) (dim + 1))
           (Movement.unsqueeze (counts sorted) dim))
    in
    let ranks =
      Movement.reshape (arange orig_len)
        (List.init ndim (fun i -> if i = dim then orig_len else 1))
    in
    let idx = Reduce.sum ~axis:[ dim ] (Elementwise.mul cond (Movement.unsqueeze ranks (dim + 1))) in
    (sorted, idx)
  end

let argsort ?(dim = -1) ?(descending = false) t = snd (sort ~dim ~descending t)

let topk ?(dim = -1) ?(largest = true) ?(sorted_ = true) t k =
  if not sorted_ then invalid_arg "Op.topk: unsorted top-k is not supported";
  let dim = T.resolve_dim t dim in
  if k > List.nth (T.shape t) dim then invalid_arg "Op.topk: k exceeds the axis";
  let values, indices = sort ~dim ~descending:largest t in
  let bound = List.mapi (fun i _ -> if i = dim then Some k else None) (T.shape t) in
  (Movement.shrink_to values bound, Movement.shrink_to indices bound)

(* Pooling and convolution *)

(* [resolve_pool_pads] returns a flat per-side pad list of length [2*dims] in
   [left; right; top; bottom; ...] order. A single-element [padding] means the
   same pad on every side; length [dims] means one value per axis (doubled);
   length [2*dims] is taken as-is. *)
let resolve_pool_pads padding dims =
  match padding with
  | [ p ] -> List.init (2 * dims) (fun _ -> p)
  | l when List.length l = 2 * dims -> l
  | l when List.length l = dims ->
      List.rev (List.concat_map (fun p -> [ p; p ]) l)
  | _ -> invalid_arg "Op.resolve_pool_pads: bad padding length"

(* [flat_to_grouped] turns the flat per-side list into per-axis
   [(before, after)] pairs in axis order. *)
let flat_to_grouped flat =
  let a = Array.of_list flat in
  let n = Array.length a in
  List.init (n / 2) (fun j -> (a.(n - 2 - (2 * j)), a.(n - 1 - (2 * j))))

let conv2d ?bias ?(groups = 1) ?(stride = [ 1 ]) ?(dilation = [ 1 ])
    ?(padding = [ 0 ]) ?dtype x weight =
  let sx = T.shape x and sw = T.shape weight in
  let bs = List.nth sx 0 and cin_ = List.nth sx 1 in
  let cout = List.nth sw 0 and cin = List.nth sw 1 in
  let hw = drop 2 sw in
  let ndim = T.ndim x in
  let nsp = List.length hw in
  if (groups * cin <> cin_) || ndim <> T.ndim weight then
    invalid_arg "Op.conv2d: input/weight shape mismatch";
  let padding_ = resolve_pool_pads padding nsp in
  let pad_arg =
    List.init (ndim - nsp) (fun _ -> Some (0, 0))
    @ List.map (fun p -> Some p) (flat_to_grouped padding_)
  in
  let x =
    Movement.pool
      (pad_constant x pad_arg (T.Sfloat 0.))
      ~k:hw ~stride ~dilation ()
  in
  let rcout = cout / groups in
  let oyx = sub_range 2 (T.ndim x - nsp) (T.shape x) in
  let noyx = List.length oyx in
  let x = Movement.reshape x ([ bs; groups; cin; 1 ] @ oyx @ hw) in
  let x = Movement.expand x ([ bs; groups; cin; rcout ] @ oyx @ hw) in
  let x =
    Movement.permute x
      ([ 0; 1; 3 ]
      @ List.init noyx (fun i -> 4 + i)
      @ [ 2 ]
      @ List.init nsp (fun i -> 4 + noyx + i))
  in
  let w =
    Movement.reshape weight
      ([ 1; groups; rcout ] @ List.init noyx (fun _ -> 1) @ [ cin ] @ hw)
  in
  let sum_axes = List.init (1 + noyx) (fun i -> -1 - i) in
  let ret =
    Movement.reshape
      (Reduce.sum ~axis:sum_axes ~keepdim:true ?dtype (Elementwise.mul x w))
      ([ bs; cout ] @ oyx)
  in
  match bias with
  | None -> ret
  | Some bias ->
      Elementwise.add ret
        (Movement.reshape bias ([ 1; -1 ] @ List.init nsp (fun _ -> 1)))


let pool_pad x ~k ~pads ~value =
  let ndim = T.ndim x in
  let nk = List.length k in
  let pad_arg =
    List.init (ndim - nk) (fun _ -> Some (0, 0))
    @ List.map (fun p -> Some p) (flat_to_grouped pads)
  in
  pad_value x pad_arg value

let avg_pool2d ?(kernel_size = [ 2; 2 ]) ?stride ?(dilation = [ 1 ])
    ?(padding = [ 0 ]) x =
  let k = kernel_size in
  let stride = match stride with Some s -> s | None -> k in
  let nk = List.length k in
  let axis = List.init nk (fun i -> -nk + i) in
  let pads = resolve_pool_pads padding nk in
  let pooled =
    Movement.pool (pool_pad x ~k ~pads ~value:(T.f 0.)) ~k ~stride ~dilation ()
  in
  mean ~axis pooled

let max_pool2d ?(kernel_size = [ 2; 2 ]) ?stride ?(dilation = [ 1 ])
    ?(padding = [ 0 ]) x =
  let k = kernel_size in
  let stride = match stride with Some s -> s | None -> k in
  let nk = List.length k in
  let axis = List.init nk (fun i -> -nk + i) in
  let pads = resolve_pool_pads padding nk in
  let pooled =
    Movement.pool
      (pool_pad x ~k ~pads ~value:(dtype_min_tensor x))
      ~k ~stride ~dilation ()
  in
  Reduce.max ~axis pooled

(* Log-space reductions *)

let logsumexp ?axis ?(keepdim = false) t =
  let ax = Option.map (fun a -> [ a ]) axis in
  let mx = Reduce.max ?axis:ax ~keepdim:true t in
  let m = Elementwise.where (Elementwise.isfinite mx) mx (T.i 0) in
  let reduced =
    Elementwise.log
      (Reduce.sum ?axis:ax ~keepdim (Elementwise.exp (Elementwise.sub t m)))
  in
  Elementwise.add reduced (if keepdim then m else Movement.squeeze ?dim:axis m)

(* The subtracted maximum is a constant of the forward pass; tinygrad detaches
   it so it carries no gradient. Autodiff is not modelled here, so the forward
   graph is identical without the detach. *)
let softmax_parts ?dtype axis t =
  let m = Elementwise.sub t (Reduce.max ~axis:[ axis ] ~keepdim:true t) in
  let m = match dtype with Some d -> Dtype_ops.cast m d | None -> m in
  let e = Elementwise.exp m in
  (m, e, Reduce.sum ~axis:[ axis ] ~keepdim:true e)

let softmax ?(axis = -1) ?dtype t =
  let _, e, ss = softmax_parts ?dtype axis t in
  Elementwise.mul e (Elementwise.reciprocal ss)

let log_softmax ?(axis = -1) ?dtype t =
  let m, _, ss = softmax_parts ?dtype axis t in
  Elementwise.sub m (Elementwise.log ss)

let softmin ?(axis = -1) ?dtype t = softmax ~axis ?dtype (Elementwise.neg t)

(* Attention *)

let scaled_dot_product_attention ?attn_mask ?(is_causal = false) q k v =
  let d = List.nth (T.shape q) (T.ndim q - 1) in
  let acc_dt =
    D.least_upper_dtype [ T.val_dtype q; T.val_dtype k; D.float32 ]
  in
  let qk =
    Elementwise.div
      (matmul ~dtype:acc_dt q (Movement.transpose ~dim0:(-2) ~dim1:(-1) k))
      (T.f (Float.sqrt (float_of_int d)))
  in
  let attn_mask =
    if is_causal then begin
      if attn_mask <> None then
        invalid_arg
          "Op.scaled_dot_product_attention: attn_mask cannot be combined \
           with is_causal";
      Some (tril (Creation.const_like ~dtype:D.bool qk (T.Sbool true)))
    end
    else attn_mask
  in
  let qk =
    match attn_mask with
    | None -> qk
    | Some m ->
        let m =
          if D.is_bool (T.dtype m) then
            Elementwise.where m (T.f 0.) (T.f Float.neg_infinity)
          else m
        in
        Elementwise.add qk m
  in
  matmul (softmax ~axis:(-1) (Dtype_ops.cast qk (T.dtype q))) v

let logcumsumexp ?(axis = 0) t =
  if T.ndim t = 0 then t
  else
    let axis = T.resolve_dim t axis in
    let x = Movement.transpose ~dim0:axis ~dim1:(-1) t in
    let last = List.nth (T.shape x) (T.ndim x - 1) in
    let x_unsqueezed = Movement.unsqueeze x (-2) in
    let x_cummax, _ = cummax ~axis:(-1) x in
    let x_cummax = Elementwise.where (Elementwise.isfinite x_cummax) x_cummax (T.i 0) in
    let mask = tril (Creation.ones ~dtype:D.bool ~buffer:false [ last; last ]) in
    let diff = Elementwise.sub x_unsqueezed (Movement.unsqueeze x_cummax (-1)) in
    let filled =
      Elementwise.where mask diff (dtype_min_tensor t)
    in
    let ret =
      Elementwise.add
        (Elementwise.log (Reduce.sum ~axis:[ -1 ] (Elementwise.exp filled)))
        x_cummax
    in
    Movement.transpose ~dim0:(-1) ~dim1:axis ret

(* Padding modes *)

type pad_mode = Constant | Reflect | Replicate | Circular

let pad_group px = List.map (function None -> (0, 0) | Some p -> p) px

let pad_circular t px =
  let px = pad_group px in
  let x =
    Movement.shrink t
      (List.map2 (fun (pb, pa) s -> (-min pb 0, min (pa + s) s)) px (T.shape t))
  in
  let px = List.map (fun (pb, pa) -> (max pb 0, max pa 0)) px in
  let orig = T.shape x in
  List.iter2
    (fun (pb, pa) s ->
      if pb > s || pa > s then
        invalid_arg "Op.pad: circular padding wraps around more than once")
    px orig;
  let x =
    Movement.repeat x
      (List.map (fun (pb, pa) -> 1 + (if pb > 0 then 1 else 0) + (if pa > 0 then 1 else 0)) px)
  in
  let bounds =
    List.map2
      (fun ((pb, pa), osh) xs ->
        ((if pb = 0 then 0 else osh - pb), if pa = 0 then xs else xs - osh + pa))
      (List.combine px orig) (T.shape x)
  in
  Movement.shrink x bounds

let pad_reflect_replicate t px ~reflect =
  let px = pad_group px in
  let pads = List.map (fun (pb, pa) -> (max pb 0, max pa 0)) px in
  let ndim = T.ndim t in
  let x = ref t in
  let shrink_axis d bound =
    Movement.shrink !x (List.mapi (fun i s -> if i = d then bound else (0, s)) (T.shape !x))
  in
  List.iteri
    (fun d (pb, pa) ->
      let s = List.nth (T.shape !x) d in
      let xb, xa =
        if reflect then (
          if pb >= s || pa >= s then
            invalid_arg "Op.pad: reflect padding must be smaller than the axis";
          let piece slc =
            getitem !x (List.init ndim (fun i -> if i = d then slc else Movement.All))
          in
          let xb = if pb > 0 then Some (piece (Movement.R (Some pb, Some 0, Some (-1)))) else None in
          let stop = if s - 2 - pa >= 0 then Some (s - 2 - pa) else None in
          let xa = if pa > 0 then Some (piece (Movement.R (Some (s - 2), stop, Some (-1)))) else None in
          (xb, xa))
        else
          let edge bound p =
            if p > 0 then
              Some
                (Movement.expand (shrink_axis d bound)
                   (List.init ndim (fun i -> if i = d then p else -1)))
            else None
          in
          (edge (0, 1) pb, edge (s - 1, s) pa)
      in
      let pieces = List.filter_map Fun.id [ xb; Some !x; xa ] in
      x := cat ~dim:d (List.hd pieces) (List.tl pieces))
    pads;
  Movement.shrink !x
    (List.map2 (fun (pb, pa) s -> (-min pb 0, min (pa + s) s)) px (T.shape !x))

let pad ?(mode = Constant) ?(value = T.Sfloat 0.0) t padding =
  match mode with
  | Constant -> pad_constant t padding value
  | Circular -> pad_circular t padding
  | Reflect -> pad_reflect_replicate t padding ~reflect:true
  | Replicate -> pad_reflect_replicate t padding ~reflect:false
