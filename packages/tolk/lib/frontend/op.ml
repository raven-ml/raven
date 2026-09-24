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
    (match (T.device t, T.device x) with
    | Some dt, Some dx when dt <> dx ->
        invalid_arg "Op.assign: device mismatch"
    | _ -> ());
    if not (D.equal (T.dtype t) (T.dtype x)) then
      invalid_arg "Op.assign: dtype mismatch";
    let dst = T.uop t in
    let assigned_to = Uop.storage_base dst in
    if not (Uop.has_buffer_identity assigned_to)
       && (Uop.op assigned_to <> Ops.Contiguous || dst == assigned_to)
    then begin
      (* Overwriting a pending value initializes new storage; its old
         computation is dead. A view into pending storage still needs a write. *)
      let value = T.uop x in
      let value =
        if Uop.op value = Ops.Contiguous then (Uop.src value).(0) else value
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
  let tensors = t :: args in
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
  let extent x = List.nth (T.symbolic_shape x) dim in
  if List.for_all (fun x -> Uop.equal (extent x) (extent t)) args then
    (* Equal extents concatenate by stacking and merging the new axis into
       the old one, which selects on one range instead of summing pads. *)
    Movement.flatten ~start_dim:dim ~end_dim:(dim + 1)
      (Movement.stack ~dim t args)
  else begin
    (* Padding needs a concrete extent on the concatenated axis; the other
       axes may stay symbolic since they receive no padding. *)
    let sizes =
      List.map
        (fun x ->
          match Uop.const_int_value (extent x) with
          | Some s -> s
          | None -> invalid_arg "Op.cat: symbolic dimension on the cat axis")
        tensors
    in
    let total = List.fold_left ( + ) 0 sizes in
    let combine =
      if D.is_bool (T.dtype t) then Elementwise.bitwise_or else Elementwise.add
    in
    let _, rev_padded =
      List.fold_left2
        (fun (before, acc) x sz ->
          let padding =
            List.init n (fun ax ->
                if ax = dim then (before, total - before - sz) else (0, 0))
          in
          (before + sz, Movement.pad x padding :: acc))
        (0, []) tensors sizes
    in
    match List.rev rev_padded with
    | [] -> t
    | h :: tl -> List.fold_left combine h tl
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

(* A fill value enters as a weak literal, so the padded tensor is promoted by
   the select below rather than by an explicit cast here. *)
let scalar_tensor = function
  | T.Sint n -> T.i n
  | T.Sfloat x -> T.f x
  | T.Sbool v -> T.b v

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

let pad_constant t px value = pad_value t px (scalar_tensor value)

let pad_to ?(value = T.Sint 0) t dims =
  let ret = Movement.pad_to t dims in
  let value = scalar_tensor value in
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

let fdiv a b =
  if a mod b <> 0 && a < 0 <> (b < 0) then (a / b) - 1 else a / b

let iceildiv a b = -fdiv (-a) b

(* Whether the endpoints of [[lo, hi]] fit the numeric limits of [dt]. *)
let range_fits dt lo hi =
  match (D.min dt, D.max dt) with
  | `Int dlo, `Int dhi ->
      Z.compare (Z.of_int lo) dlo >= 0 && Z.compare (Z.of_int hi) dhi <= 0
  | `Float dlo, `Float dhi ->
      float_of_int lo >= dlo && float_of_int hi <= dhi
  | _, _ -> true

let arange ?stop ?(step = 1) ?dtype start =
  if step = 0 then invalid_arg "Op.arange: step must be non-zero";
  let start, stop = match stop with None -> (0, start) | Some s -> (start, s) in
  let lo, hi = if step > 0 then (start, stop - step) else (stop - step, start) in
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
  let output_len = iceildiv (stop - start) step in
  if output_len <= 0 then Creation.full ~dtype:dt ~buffer:false [ 0 ] (T.Sint 0)
  else
    let acc_dtype =
      if D.is_float dt then D.least_upper_dtype [ dt; D.float32 ] else dt
    in
    let base = Creation.full ~dtype:acc_dtype ~buffer:false [ output_len ] (T.Sint step) in
    let scan = cumalu base 0 Ops.Add in
    Dtype_ops.cast (Elementwise.add scan (T.i (start - step))) dt

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

(* Indexed scatter

   No tinygrad counterpart. [scatter] and [scatter_reduce] range over the
   destination with a trailing axis over the indices, so a write of [k] rows
   into [n] costs [n * k]. [scatter_indexed] ranges over the indices: a custom
   kernel whose store lands in [t]'s storage at the loaded index, the way the
   reference's embedding gradient writes its buffer. Two updates collide only
   when they agree on every coordinate off [dim], so those coordinates are
   parallel lanes and the index range is serial within a lane: updates land in
   index order, the last [`Set] wins and [`Add] accumulates exactly, on every
   device. [unique] frees the index range too. An index outside the axis gates
   the store off. *)

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
  let count = List.nth ish dim and extent = List.nth tsh dim in
  if count = 0 || prod tsh = 0 then t
  else begin
    let expanded = Tolk.Rangeify.detect_expanded (T.uop index) in
    let index =
      if List.length expanded <> rank then index
      else
        Movement.shrink_to index
          (List.mapi (fun d e -> if d <> dim && e then Some 1 else None)
             expanded)
    in
    let ish = T.shape index in
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
                axis_types = [];
                dont_use_locals = false;
                applied_opts = [];
                opts_to_apply = Some [];
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
      let big_shape = Uop.broadcast_shape (List.map T.symbolic_shape tensors) in
      let bshape_len = List.length big_shape in
      let d0 = List.hd dims in
      let dlast = List.nth dims (List.length dims - 1) in
      let consecutive = dims = List.init (List.length dims) (fun i -> d0 + i) in
      let xshape = T.symbolic_shape x in
      let axis_size d =
        match Uop.const_int_value (List.nth xshape d) with
        | Some n -> n
        | None ->
            invalid_arg "Op.getitem: advanced indexing needs a concrete axis size"
      in
      if List.length dims > 1 && consecutive then (
        (* Consecutive integer-tensor indices: one linear gather over the
           flattened block instead of a mask per axis. *)
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
        let mask =
          match
            List.map2
              (fun d tn ->
                let i =
                  Movement.symbolic_broadcast_to
                    (Movement.symbolic_reshape tn
                       (T.symbolic_shape tn
                        @ List.init (xndim - d0) (fun _ -> Uop.const_int 1)))
                    pre_reduce_shape
                in
                one_hot_along_dim ~dim:(d - xndim) i (axis_size d))
              dims tensors
          with
          | h :: tl -> Elementwise.uprod h tl
          | [] -> assert false
        in
        let reshape_arg =
          take d0 xshape @ List.init bshape_len (fun _ -> Uop.const_int 1)
          @ drop d0 xshape
        in
        let sum_axis = List.map (fun d -> d + bshape_len) dims in
        let x =
          let reshaped = Movement.symbolic_reshape x reshape_arg in
          Reduce.sum ~axis:sum_axis ~dtype:(T.val_dtype x)
            (Elementwise.where mask reshaped (T.i 0))
        in
        let permuted =
          d0 <> 0 && List.length dims <> 1
          && dims <> List.init (dlast - d0 + 1) (fun i -> d0 + i)
        in
        if not permuted then x
        else
          let nd = T.ndim x in
          Movement.permute x
            (List.init bshape_len (fun i -> d0 + i)
            @ List.init d0 Fun.id
            @ List.init (nd - (d0 + bshape_len)) (fun i -> d0 + bshape_len + i))

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
