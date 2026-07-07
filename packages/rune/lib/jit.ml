(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Just-in-time compilation as an effect handler over Nx operations.

   Tracing: the handler answers every intercepted operation with a fresh
   uninitialized placeholder tensor of the result's shape and dtype, and records
   the corresponding node of a Tolk tensor graph in a side table keyed by tensor
   identity. Running the function once under the handler therefore turns its
   whole computation into a single graph.

   Compiling: the graph is lowered through Tolk's pipeline (allocations,
   scheduling, kernel codegen) into a compiled linear schedule with a persistent
   buffer binding. Compilation happens once per input signature (the leaf dtypes
   and shapes in traversal order); a call with a new signature retraces.

   Replaying: every call binds the current input leaves to the compiled
   program's buffers and runs the schedule. On the CPU device, contiguous inputs
   and captured constants are wrapped in place — kernels read the tensors' own
   memory, so mutations between calls are observed like in eager execution — and
   outputs are computed straight into the returned tensors' storage. Other
   devices and non-contiguous tensors go through byte copies: inputs are
   re-copied on every call, captured tensors are uploaded once when the trace
   compiles and stay resident on the device. Whole-tensor [assign]s are replayed
   by writing the computed value back into the destination; assigned captures
   are re-uploaded on every call so the written-back state (and any eager
   mutation of it) carries into the next call on every device.

   Reading the value of a traced tensor (for example [Nx.item] on a value that
   depends on the inputs) raises [Jit_error]: a compiled trace cannot branch on
   data. Operations Tolk cannot express (FFT, linear algebra, complex dtypes,
   RNG) raise [Jit_error] as well. *)

open Nx_effect
module F = Tolk_frontend
module U = Tolk_uop.Uop
module TD = Tolk_uop.Dtype
module ND = Nx_core.Dtype
module NV = Nx_core.View

(* Element-wise operations need the broadcasting hook installed by [F.Op]'s
   initialiser; referencing the module forces the link. *)
let _op_linked = Sys.opaque_identity F.Op.broadcasted

exception Jit_error of string

let err fmt = Printf.ksprintf (fun s -> raise (Jit_error s)) fmt

let unsupported op =
  err
    "Rune.jit: %s is not supported inside jit; move it outside the jitted \
     function"
    op

(* Dtype bridges *)

let tolk_dtype : type a b. (a, b) ND.t -> TD.t = function
  | ND.Float16 -> TD.float16
  | ND.Float32 -> TD.float32
  | ND.Float64 -> TD.float64
  | ND.BFloat16 -> TD.bfloat16
  | ND.Float8_e4m3 -> TD.fp8e4m3
  | ND.Float8_e5m2 -> TD.fp8e5m2
  | ND.Int8 -> TD.int8
  | ND.UInt8 -> TD.uint8
  | ND.Int16 -> TD.int16
  | ND.UInt16 -> TD.uint16
  | ND.Int32 -> TD.int32
  | ND.UInt32 -> TD.uint32
  | ND.Int64 -> TD.int64
  | ND.UInt64 -> TD.uint64
  | ND.Bool -> TD.bool
  | ND.Int4 -> unsupported "an int4 tensor"
  | ND.UInt4 -> unsupported "a uint4 tensor"
  | ND.Complex64 -> unsupported "a complex tensor"
  | ND.Complex128 -> unsupported "a complex tensor"

let tolk_val_dtype : type a b. (a, b) ND.t -> TD.Val.t = function
  | ND.Float16 -> TD.Val.float16
  | ND.Float32 -> TD.Val.float32
  | ND.Float64 -> TD.Val.float64
  | ND.BFloat16 -> TD.Val.bfloat16
  | ND.Float8_e4m3 -> TD.Val.fp8e4m3
  | ND.Float8_e5m2 -> TD.Val.fp8e5m2
  | ND.Int8 -> TD.Val.int8
  | ND.UInt8 -> TD.Val.uint8
  | ND.Int16 -> TD.Val.int16
  | ND.UInt16 -> TD.Val.uint16
  | ND.Int32 -> TD.Val.int32
  | ND.UInt32 -> TD.Val.uint32
  | ND.Int64 -> TD.Val.int64
  | ND.UInt64 -> TD.Val.uint64
  | ND.Bool -> TD.Val.bool
  | ND.Int4 -> unsupported "an int4 tensor"
  | ND.UInt4 -> unsupported "a uint4 tensor"
  | ND.Complex64 -> unsupported "a complex tensor"
  | ND.Complex128 -> unsupported "a complex tensor"

let scalar_of : type a b. (a, b) ND.t -> a -> F.Tensor.scalar =
 fun dt v ->
  match dt with
  | ND.Float16 -> F.Tensor.Sfloat v
  | ND.Float32 -> F.Tensor.Sfloat v
  | ND.Float64 -> F.Tensor.Sfloat v
  | ND.BFloat16 -> F.Tensor.Sfloat v
  | ND.Float8_e4m3 -> F.Tensor.Sfloat v
  | ND.Float8_e5m2 -> F.Tensor.Sfloat v
  | ND.Int8 -> F.Tensor.Sint v
  | ND.UInt8 -> F.Tensor.Sint v
  | ND.Int16 -> F.Tensor.Sint v
  | ND.UInt16 -> F.Tensor.Sint v
  | ND.Int4 -> F.Tensor.Sint v
  | ND.UInt4 -> F.Tensor.Sint v
  | ND.Int32 -> F.Tensor.Sint (Int32.to_int v)
  | ND.UInt32 -> F.Tensor.Sint (Int32.to_int v)
  | ND.Int64 -> F.Tensor.Sint (Int64.to_int v)
  | ND.UInt64 -> F.Tensor.Sint (Int64.to_int v)
  | ND.Bool -> F.Tensor.Sbool v
  | ND.Complex64 -> unsupported "a complex tensor"
  | ND.Complex128 -> unsupported "a complex tensor"

(* Identity-keyed tables over tensors, as in [Tensor_map]. *)
module Tbl = Hashtbl.Make (struct
  type t = Obj.t

  let equal = ( == )
  let hash = Hashtbl.hash
end)

type packed = Packed : ('a, 'b) ND.t * ('a, 'b) Nx_effect.t -> packed

(* Devices. One instance per canonical name, created on first use. *)

let canonical name =
  let name = String.uppercase_ascii name in
  if String.contains name ':' then name else name ^ ":0"

let devices : (string, Tolk.Device.t) Hashtbl.t = Hashtbl.create 4

let get_device name =
  let name = canonical name in
  match Hashtbl.find_opt devices name with
  | Some d -> d
  | None ->
      let d = Jit_device.create name in
      Hashtbl.add devices name d;
      d

(* Only the CPU device shares host memory with Nx tensors, so only it can run on
   wrapped buffers instead of copies. *)
let is_cpu name = String.starts_with ~prefix:"CPU" (canonical name)
let to_program dev = Tolk.Codegen.to_program dev (Tolk.Device.renderer dev)

(* Environment knobs, read when a jit closure is created (not at module
   initialization) so tests can toggle them with [Unix.putenv]. *)

let env_int name default =
  match Sys.getenv_opt name with
  | Some s -> ( match int_of_string_opt s with Some v -> v | None -> default)
  | None -> default

(* Forces the copy path on the CPU device, so the device-residency machinery
   (staged transfers, deferred outputs, resident feedback) is exercised without
   a GPU. *)
let force_copy () = env_int "RUNE_JIT_FORCE_COPY" 0 <> 0
let jit_debug = lazy (env_int "RUNE_JIT_DEBUG" 0)

(* Transfer accounting. Cumulative byte counters for host-to-device and
   device-to-host copies made by compiled traces; the zero-copy CPU path moves
   no bytes and counts nothing. *)

type stats = {
  bytes_to_device : int;
  bytes_from_device : int;
  resident_bytes : int;
}

let bytes_to_device = ref 0
let bytes_from_device = ref 0
let resident_bytes = ref 0

let stats () =
  {
    bytes_to_device = !bytes_to_device;
    bytes_from_device = !bytes_from_device;
    resident_bytes = !resident_bytes;
  }

let reset_stats () =
  bytes_to_device := 0;
  bytes_from_device := 0

(* Resident outputs

   On devices that do not share host memory, a compiled call's outputs stay on
   the device: each output leaf becomes a deferred host tensor (a handle)
   owning a device buffer bound freshly for that call. Reading the handle
   forces it — synchronize, copy the buffer out, release it to the allocator —
   and a handle dropped unread is released by a GC finalizer. The registry
   maps live handle ids to their buffers so replay can seed a compiled input
   directly with a resident buffer (no transfer) when a handle is fed back
   into a jit call on the same device. *)

type resident_entry = {
  mutable r_id : int;
  r_device : Tolk.Device.t;
  r_nbytes : int;
  mutable r_buf : Tolk.Device.Buffer.t option; (* [None] once released *)
}

let resident : (int, resident_entry) Hashtbl.t = Hashtbl.create 64

(* Finalizers only record the entry; buffers are released at the next safe
   point (a force or a replay), not mid-GC inside arbitrary device code. *)
let pending_release : resident_entry list ref = ref []

let release_entry e =
  match e.r_buf with
  | None -> ()
  | Some buf ->
      e.r_buf <- None;
      Hashtbl.remove resident e.r_id;
      resident_bytes := !resident_bytes - e.r_nbytes;
      (* Deallocation returns the buffer to the device's LRU pool. A base
         buffer with a still-allocated transient view (a kernel-argument slice
         not yet collected) cannot be deallocated; those are reclaimed by the
         buffer's own GC finalizer instead. *)
      (try Tolk.Device.Buffer.deallocate buf with Invalid_argument _ -> ())

let drain_releases () =
  match !pending_release with
  | [] -> ()
  | entries ->
      pending_release := [];
      List.iter release_entry entries

let resident_budget () =
  env_int "RUNE_JIT_RESIDENT_BUDGET" (4 * 1024 * 1024 * 1024)

(* Fresh device buffer for a call's output. Past the resident budget, collect
   dropped handles first; on allocation failure collect and retry once (the
   LRU allocator has flushed its own cache by then). *)
let create_fresh_buffer dev dtolk n =
  drain_releases ();
  if !resident_bytes > resident_budget () then begin
    Gc.major ();
    drain_releases ()
  end;
  let buf = Tolk.Device.create_buffer ~size:n ~dtype:dtolk dev in
  (try Tolk.Device.Buffer.ensure_allocated buf
   with _ ->
     Gc.major ();
     drain_releases ();
     Tolk.Device.Buffer.ensure_allocated buf);
  buf

(* Buffer slots are process-global so hash-consed buffer nodes from distinct
   traces never collide. *)
let next_slot = ref 0

let fresh_slot () =
  let s = !next_slot in
  incr next_slot;
  s

(* Trace state *)

type input = { i_node : U.t; i_buf : Tolk.Device.Buffer.t }

type state = {
  st_device : Tolk.Device.t;
  st_ctx : Nx_effect.context;
  table : F.Tensor.t Tbl.t; (* nx tensor -> tolk tensor *)
  traced : unit Tbl.t; (* placeholders whose bytes are not meaningful *)
  externals : unit Tbl.t; (* inputs and lifted constants *)
  input_index : int Tbl.t; (* input placeholder -> traversal position *)
  wb_seen : unit Tbl.t;
  mutable consts : (U.t * packed) list; (* reverse order *)
  mutable writebacks : packed list; (* reverse order *)
}

let shape_of x = NV.shape (Nx_effect.view x)
let numel shape = Array.fold_left ( * ) 1 shape

(* Wrap a graph buffer node as a tolk tensor of [shape]. Buffers are 1-D on the
   graph; the reshape restores the logical shape. *)
let buffer_tensor node shape =
  F.Movement.reshape (F.Tensor.of_uop node) (Array.to_list shape)

let make_node st dtolk n =
  U.buffer ~slot:(fresh_slot ()) ~dtype:dtolk ~shape:(F.Tensor.shape_uop [ n ])
    ~device:(U.Single (Tolk.Device.name st.st_device))
    ()

(* A tensor entering the trace without a table entry is a closure capture: it
   becomes a buffer input backed by the tensor's own memory when the device can
   share it, and re-read from the tensor on every call otherwise, so mutations
   between calls behave as they do eagerly. *)
let lift_const (type a b) st (x : (a, b) Nx_effect.t) : F.Tensor.t =
  (* The trace table hashes keys structurally, and a deferred capture would
     mutate — changing its hash — if the traced function read it. Captures are
     forced when the trace compiles anyway (their bytes are wrapped or
     uploaded), so force now and keep the key stable. *)
  ignore (Nx_effect.unwrap x);
  let dt = Nx_effect.dtype x in
  let shape = shape_of x in
  let node = make_node st (tolk_dtype dt) (numel shape) in
  st.consts <- (node, Packed (dt, x)) :: st.consts;
  let tt = buffer_tensor node shape in
  Tbl.replace st.table (Obj.repr x) tt;
  Tbl.replace st.externals (Obj.repr x) ();
  tt

let tolk_of : type a b. state -> (a, b) Nx_effect.t -> F.Tensor.t =
 fun st x ->
  match Tbl.find_opt st.table (Obj.repr x) with
  | Some t -> t
  | None -> lift_const st x

(* Composed operations Tolk has no primitive for. *)

(* Round half away from zero (C [round]); Tolk's round is half to even. *)
let round_away t =
  let open F.Elementwise in
  let half = F.Creation.const_like t (F.Tensor.Sfloat 0.5) in
  trunc (add t (copysign half t))

let atan2_graph y x =
  let open F.Elementwise in
  let zero t = F.Creation.const_like t (F.Tensor.Sfloat 0.) in
  let z = atan (div y x) in
  let pi = F.Creation.const_like z (F.Tensor.Sfloat Float.pi) in
  add z (where (lt x (zero x)) (where (ge y (zero y)) pi (neg pi)) (zero z))

(* Sliding windows. [unfold] is pad -> pool -> permute -> reshape: pure
   movement, so the extracted patches fuse into their consumer. [fold]
   (overlapping-add) has no movement expression; it becomes a scatter-add over
   the flattened padded spatial block driven by a precomputed index constant:
   entry [kf * nwin + w] is the flat padded position written by kernel offset
   [kf] of window [w]. Kernel offsets and windows both enumerate row-major,
   matching the eager implementation; contributions that land in the padding are
   shrunk away, matching its drop-padding semantics. *)

let window_indices ~spatial_padded ~kernel_size ~stride ~dilation =
  let k = Array.length kernel_size in
  let out_sp =
    Array.init k (fun d ->
        (spatial_padded.(d) - ((dilation.(d) * (kernel_size.(d) - 1)) + 1))
        / stride.(d)
        + 1)
  in
  let kernel_prod = numel kernel_size in
  let nwin = numel out_sp in
  let sp_strides = Array.make k 1 in
  for d = k - 2 downto 0 do
    sp_strides.(d) <- sp_strides.(d + 1) * spatial_padded.(d + 1)
  done;
  let idx = Nx_buffer.create Nx_buffer.int32 (kernel_prod * nwin) in
  let k_pos = Array.make k 0 in
  let w_pos = Array.make k 0 in
  let bump pos limits =
    let rec go d =
      if d >= 0 then begin
        pos.(d) <- pos.(d) + 1;
        if pos.(d) = limits.(d) then begin
          pos.(d) <- 0;
          go (d - 1)
        end
      end
    in
    go (k - 1)
  in
  let p = ref 0 in
  for _kf = 0 to kernel_prod - 1 do
    Array.fill w_pos 0 k 0;
    for _w = 0 to nwin - 1 do
      let off = ref 0 in
      for d = 0 to k - 1 do
        off :=
          !off
          + ((w_pos.(d) * stride.(d)) + (k_pos.(d) * dilation.(d)))
            * sp_strides.(d)
      done;
      Nx_buffer.unsafe_set idx !p (Int32.of_int !off);
      incr p;
      bump w_pos out_sp
    done;
    bump k_pos kernel_size
  done;
  (kernel_prod, nwin, idx)

let no_padding = Array.for_all (fun (b, a) -> b = 0 && a = 0)

(* Lift the index constant and broadcast it over the leading dimensions. *)
let window_index_tensor st idx lead n =
  let it = tolk_of st (Nx_effect.from_host st.st_ctx idx) in
  let it = F.Movement.reshape it (List.map (fun _ -> 1) lead @ [ n ]) in
  F.Movement.expand it (lead @ [ n ])

let unfold_graph st t_in ~kernel_size ~stride ~dilation ~padding =
  let shape = shape_of t_in in
  let rank = Array.length shape in
  let kd = Array.length kernel_size in
  let nlead = rank - kd in
  let lead = Array.to_list (Array.sub shape 0 nlead) in
  let t = tolk_of st t_in in
  let t =
    if no_padding padding then t
    else
      F.Op.pad t
        (List.map (fun _ -> None) lead
        @ List.init kd (fun d -> Some padding.(d)))
  in
  let pooled =
    F.Movement.pool t
      ~k:(Array.to_list kernel_size)
      ~stride:(Array.to_list stride) ~dilation:(Array.to_list dilation) ()
  in
  (* (lead.., o.., k..) -> (lead.., k.., o..) -> (lead.., prod k, windows) *)
  let perm =
    List.init nlead Fun.id
    @ List.init kd (fun d -> nlead + kd + d)
    @ List.init kd (fun d -> nlead + d)
  in
  let pooled_shape = Array.of_list (F.Tensor.shape pooled) in
  let nwin = numel (Array.sub pooled_shape nlead kd) in
  F.Movement.reshape
    (F.Movement.permute pooled perm)
    (lead @ [ numel kernel_size; nwin ])

let fold_graph st t_in ~output_size ~kernel_size ~stride ~dilation ~padding =
  let shape = shape_of t_in in
  let rank = Array.length shape in
  let kd = Array.length kernel_size in
  let lead = Array.to_list (Array.sub shape 0 (rank - 2)) in
  let spatial_padded =
    Array.init kd (fun d ->
        let before, after = padding.(d) in
        output_size.(d) + before + after)
  in
  let kernel_prod, nwin, idx =
    window_indices ~spatial_padded ~kernel_size ~stride ~dilation
  in
  let src =
    F.Movement.reshape (tolk_of st t_in) (lead @ [ kernel_prod * nwin ])
  in
  let it = window_index_tensor st idx lead (kernel_prod * nwin) in
  let template =
    F.Creation.zeros
      ~dtype:(tolk_val_dtype (Nx_effect.dtype t_in))
      (lead @ [ numel spatial_padded ])
  in
  let scat =
    F.Op.scatter_reduce template ~dim:(List.length lead) it src ~reduce:`Sum
      ~include_self:true ()
  in
  let r = F.Movement.reshape scat (lead @ Array.to_list spatial_padded) in
  if no_padding padding then r
  else
    F.Movement.shrink r
      (List.map (fun d -> (0, d)) lead
      @ List.init kd (fun d ->
          let before, _ = padding.(d) in
          (before, before + output_size.(d))))

(* Handler *)

let handler st =
  let open Effect.Deep in
  (* Answer an intercepted operation: record the graph node and continue with a
     fresh placeholder carrying the result's shape and dtype. *)
  let ret : type a b r.
      ((a, b) Nx_effect.t, r) continuation -> (a, b) ND.t -> F.Tensor.t -> r =
   fun k dt tt ->
    let shape = Array.of_list (F.Tensor.shape tt) in
    let ph : (a, b) Nx_effect.t = Nx_effect.buffer st.st_ctx dt shape in
    Tbl.replace st.table (Obj.repr ph) tt;
    Tbl.replace st.traced (Obj.repr ph) ();
    continue k ph
  in
  let dt x = Nx_effect.dtype x in
  let go x = tolk_of st x in
  let refuse k op =
    discontinue k
      (Jit_error
         (Printf.sprintf
            "Rune.jit: %s is not supported inside jit; move it outside the \
             jitted function"
            op))
  in
  let effc : type c. c Effect.t -> ((c, _) continuation -> _) option =
   fun eff ->
    match eff with
    (* Metadata reads fall back to the placeholder, whose view is the
       result's. *)
    | E_view _ -> None
    (* Reading data is allowed only for tensors whose bytes are real (constants
       created during the trace); reading a traced value would burn data into
       the compiled program. *)
    | E_to_host x ->
        if Tbl.mem st.traced (Obj.repr x) then
          Some
            (fun k ->
              discontinue k
                (Jit_error
                   "Rune.jit: the value of a traced tensor was read during jit \
                    tracing (item, to_host, or a data-dependent branch); \
                    jitted code cannot branch on tensor values"))
        else None
    (* Creation *)
    | E_buffer { dtype; size_in_elements; _ } ->
        Some
          (fun k ->
            let ph = Nx_effect.buffer st.st_ctx dtype [| size_in_elements |] in
            ignore (lift_const st ph);
            continue k ph)
    | E_const_scalar { value; dtype; _ } ->
        Some
          (fun k ->
            (* [buffer:false] keeps the scalar an immediate constant: it
               folds into consuming kernels instead of being stored into a
               one-element buffer by a kernel of its own. *)
            let tt =
              F.Creation.full ~buffer:false
                ~dtype:(tolk_val_dtype dtype)
                [] (scalar_of dtype value)
            in
            let ph = Nx_effect.const_scalar st.st_ctx value dtype in
            Tbl.replace st.table (Obj.repr ph) tt;
            continue k ph)
    | E_from_host { array; _ } ->
        Some
          (fun k ->
            let ph = Nx_effect.from_host st.st_ctx array in
            ignore (lift_const st ph);
            continue k ph)
    (* Binary arithmetic *)
    | E_add { a; b } ->
        Some (fun k -> ret k (dt a) (F.Elementwise.add (go a) (go b)))
    | E_sub { a; b } ->
        Some (fun k -> ret k (dt a) (F.Elementwise.sub (go a) (go b)))
    | E_mul { a; b } ->
        Some (fun k -> ret k (dt a) (F.Elementwise.mul (go a) (go b)))
    | E_idiv { a; b } ->
        Some (fun k -> ret k (dt a) (F.Elementwise.cdiv (go a) (go b)))
    | E_fdiv { a; b } ->
        Some (fun k -> ret k (dt a) (F.Elementwise.div (go a) (go b)))
    | E_max { a; b } ->
        Some (fun k -> ret k (dt a) (F.Elementwise.maximum (go a) (go b)))
    | E_min { a; b } ->
        Some (fun k -> ret k (dt a) (F.Elementwise.minimum (go a) (go b)))
    | E_mod { a; b } ->
        Some (fun k -> ret k (dt a) (F.Elementwise.fmod (go a) (go b)))
    | E_pow { a; b } ->
        Some (fun k -> ret k (dt a) (F.Elementwise.pow (go a) (go b)))
    | E_xor { a; b } ->
        Some (fun k -> ret k (dt a) (F.Elementwise.bitwise_xor (go a) (go b)))
    | E_or { a; b } ->
        Some (fun k -> ret k (dt a) (F.Elementwise.bitwise_or (go a) (go b)))
    | E_and { a; b } ->
        Some (fun k -> ret k (dt a) (F.Elementwise.bitwise_and (go a) (go b)))
    | E_atan2 { a; b } ->
        Some (fun k -> ret k (dt a) (atan2_graph (go a) (go b)))
    (* Comparisons *)
    | E_cmpeq { a; b } ->
        Some (fun k -> ret k ND.bool (F.Elementwise.eq (go a) (go b)))
    | E_cmpne { a; b } ->
        Some (fun k -> ret k ND.bool (F.Elementwise.ne (go a) (go b)))
    | E_cmplt { a; b } ->
        Some (fun k -> ret k ND.bool (F.Elementwise.lt (go a) (go b)))
    | E_cmple { a; b } ->
        Some (fun k -> ret k ND.bool (F.Elementwise.le (go a) (go b)))
    (* Unary arithmetic *)
    | E_neg { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.neg (go t_in)))
    | E_sin { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.sin (go t_in)))
    | E_sqrt { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.sqrt (go t_in)))
    | E_recip { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.reciprocal (go t_in)))
    | E_log { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.log (go t_in)))
    | E_exp { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.exp (go t_in)))
    | E_cos { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.cos (go t_in)))
    | E_abs { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.abs (go t_in)))
    | E_sign { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.sign (go t_in)))
    | E_tan { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.tan (go t_in)))
    | E_asin { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.asin (go t_in)))
    | E_acos { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.acos (go t_in)))
    | E_atan { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.atan (go t_in)))
    | E_sinh { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.sinh (go t_in)))
    | E_cosh { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.cosh (go t_in)))
    | E_tanh { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.tanh (go t_in)))
    | E_erf { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.erf (go t_in)))
    (* Rounding: identity on integers, as eagerly. *)
    | E_trunc { t_in } ->
        Some
          (fun k ->
            let t = go t_in in
            ret k (dt t_in)
              (if ND.is_float (dt t_in) then F.Elementwise.trunc t else t))
    | E_ceil { t_in } ->
        Some
          (fun k ->
            let t = go t_in in
            ret k (dt t_in)
              (if ND.is_float (dt t_in) then F.Elementwise.ceil t else t))
    | E_floor { t_in } ->
        Some
          (fun k ->
            let t = go t_in in
            ret k (dt t_in)
              (if ND.is_float (dt t_in) then F.Elementwise.floor t else t))
    | E_round { t_in } ->
        Some
          (fun k ->
            let t = go t_in in
            ret k (dt t_in) (if ND.is_float (dt t_in) then round_away t else t))
    (* Ternary *)
    | E_where { condition; if_true; if_false } ->
        Some
          (fun k ->
            ret k (dt if_true)
              (F.Elementwise.where (go condition) (go if_true) (go if_false)))
    (* Reductions. The accumulator dtype is pinned to the input's so results
       match eager execution. *)
    | E_reduce_sum { t_in; axes; keepdims } ->
        Some
          (fun k ->
            let t = go t_in in
            ret k (dt t_in)
              (F.Reduce.sum ~axis:(Array.to_list axes) ~keepdim:keepdims
                 ~dtype:(F.Tensor.val_dtype t) t))
    | E_reduce_prod { t_in; axes; keepdims } ->
        Some
          (fun k ->
            let t = go t_in in
            ret k (dt t_in)
              (F.Reduce.prod ~axis:(Array.to_list axes) ~keepdim:keepdims
                 ~dtype:(F.Tensor.val_dtype t) t))
    | E_reduce_max { t_in; axes; keepdims } ->
        Some
          (fun k ->
            ret k (dt t_in)
              (F.Reduce.max ~axis:(Array.to_list axes) ~keepdim:keepdims
                 (go t_in)))
    | E_reduce_min { t_in; axes; keepdims } ->
        Some
          (fun k ->
            ret k (dt t_in)
              (F.Reduce.min ~axis:(Array.to_list axes) ~keepdim:keepdims
                 (go t_in)))
    | E_argmax { t_in; axis; keepdims } ->
        Some
          (fun k ->
            ret k ND.int32
              (F.Dtype_ops.cast
                 (F.Op.argmax ~axis ~keepdim:keepdims (go t_in))
                 TD.int32))
    | E_argmin { t_in; axis; keepdims } ->
        Some
          (fun k ->
            ret k ND.int32
              (F.Dtype_ops.cast
                 (F.Op.argmin ~axis ~keepdim:keepdims (go t_in))
                 TD.int32))
    | E_sort { t_in; axis; descending } ->
        Some
          (fun k ->
            ret k (dt t_in) (fst (F.Op.sort ~dim:axis ~descending (go t_in))))
    | E_argsort { t_in; axis; descending } ->
        Some
          (fun k ->
            ret k ND.int32
              (F.Dtype_ops.cast
                 (F.Op.argsort ~dim:axis ~descending (go t_in))
                 TD.int32))
    | E_associative_scan { t_in; axis; op } ->
        Some
          (fun k ->
            let t = go t_in in
            let r =
              match op with
              | `Sum -> F.Op.cumsum ~axis t
              | `Prod -> F.Op.cumprod ~axis t
              | `Max -> fst (F.Op.cummax ~axis t)
              | `Min -> fst (F.Op.cummin ~axis t)
            in
            ret k (dt t_in) r)
    (* Movement *)
    | E_permute { t_in; axes } ->
        Some
          (fun k ->
            ret k (dt t_in) (F.Movement.permute (go t_in) (Array.to_list axes)))
    | E_reshape { t_in; new_shape } ->
        Some
          (fun k ->
            ret k (dt t_in)
              (F.Movement.reshape (go t_in) (Array.to_list new_shape)))
    | E_expand { t_in; new_target_shape } ->
        Some
          (fun k ->
            ret k (dt t_in)
              (F.Movement.expand (go t_in) (Array.to_list new_target_shape)))
    | E_pad { t_in; padding_config; fill_value } ->
        Some
          (fun k ->
            let pads =
              Array.to_list (Array.map (fun p -> Some p) padding_config)
            in
            ret k (dt t_in)
              (F.Op.pad ~value:(scalar_of (dt t_in) fill_value) (go t_in) pads))
    | E_shrink { t_in; limits } ->
        Some
          (fun k ->
            ret k (dt t_in) (F.Movement.shrink (go t_in) (Array.to_list limits)))
    | E_flip { t_in; dims_to_flip } ->
        Some
          (fun k ->
            let axes = ref [] in
            Array.iteri (fun i f -> if f then axes := i :: !axes) dims_to_flip;
            ret k (dt t_in) (F.Movement.flip (go t_in) (List.rev !axes)))
    | E_cat { t_list; axis } ->
        Some
          (fun k ->
            match t_list with
            | [] -> err "Rune.jit: cat of an empty list"
            | hd :: tl ->
                ret k (dt hd)
                  (F.Op.cat ~dim:axis (go hd) (List.map (fun t -> go t) tl)))
    (* Cast and copies *)
    | E_cast { t_in; target_dtype } ->
        Some
          (fun k ->
            ret k target_dtype
              (F.Dtype_ops.cast (go t_in) (tolk_dtype target_dtype)))
    | E_contiguous { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.contiguous (go t_in)))
    | E_copy { t_in } ->
        Some (fun k -> ret k (dt t_in) (F.Elementwise.contiguous (go t_in)))
    (* In-place assignment: rebind the destination to the assigned value.
       Externally visible destinations (inputs, closure captures) are also
       written back on every call. Assigning into a view would require aliasing
       the destination's base tensor, which a trace cannot see. *)
    | E_assign { dst; src } ->
        Some
          (fun k ->
            let v = Nx_effect.view dst in
            if not (NV.is_c_contiguous v && NV.offset v = 0) then
              discontinue k
                (Jit_error
                   "Rune.jit: assigning into a view (set_item, set_slice, \
                    blit) is not supported inside jit; use scatter instead")
            else begin
              let sv = tolk_of st src in
              if not (Tbl.mem st.table (Obj.repr dst)) then
                ignore (lift_const st dst);
              Tbl.replace st.table (Obj.repr dst) sv;
              let key = Obj.repr dst in
              if Tbl.mem st.externals key && not (Tbl.mem st.wb_seen key) then begin
                Tbl.replace st.wb_seen key ();
                st.writebacks <-
                  Packed (Nx_effect.dtype dst, dst) :: st.writebacks
              end;
              continue k ()
            end)
    (* Indexed access *)
    | E_gather { data; indices; axis } ->
        Some
          (fun k ->
            ret k (dt data) (F.Op.gather (go data) ~dim:axis (go indices)))
    | E_scatter { data_template; indices; updates; axis; mode; _ } ->
        Some
          (fun k ->
            let r =
              match mode with
              | `Set ->
                  F.Op.scatter (go data_template) ~dim:axis (go indices)
                    (go updates)
              | `Add ->
                  F.Op.scatter_reduce (go data_template) ~dim:axis (go indices)
                    (go updates) ~reduce:`Sum ~include_self:true ()
            in
            ret k (dt data_template) r)
    (* Matrix multiplication *)
    | E_matmul { a; b } ->
        Some (fun k -> ret k (dt a) (F.Op.matmul (go a) (go b)))
    (* Device movement is the identity on the single jit device. *)
    | E_to_device { t_in; _ } -> Some (fun k -> Effect.Deep.continue k t_in)
    (* Unsupported operations fail loudly rather than trace wrongly. *)
    | E_threefry _ ->
        Some (fun k -> refuse k "random number generation (threefry)")
    | E_unfold { t_in; kernel_size; stride; dilation; padding } ->
        Some
          (fun k ->
            ret k (dt t_in)
              (unfold_graph st t_in ~kernel_size ~stride ~dilation ~padding))
    | E_fold { t_in; output_size; kernel_size; stride; dilation; padding } ->
        Some
          (fun k ->
            ret k (dt t_in)
              (fold_graph st t_in ~output_size ~kernel_size ~stride ~dilation
                 ~padding))
    | E_fft _ -> Some (fun k -> refuse k "fft")
    | E_ifft _ -> Some (fun k -> refuse k "ifft")
    | E_rfft _ -> Some (fun k -> refuse k "rfft")
    | E_irfft _ -> Some (fun k -> refuse k "irfft")
    | E_cholesky _ -> Some (fun k -> refuse k "cholesky")
    | E_qr _ -> Some (fun k -> refuse k "qr")
    | E_svd _ -> Some (fun k -> refuse k "svd")
    | E_eig _ -> Some (fun k -> refuse k "eig")
    | E_eigh _ -> Some (fun k -> refuse k "eigh")
    | E_triangular_solve _ -> Some (fun k -> refuse k "triangular_solve")
    | E_psum _ ->
        Some
          (fun k ->
            discontinue k
              (Jit_error "Rune.jit: psum is only meaningful under vmap"))
    | _ -> None
  in
  { retc = Fun.id; exnc = raise; effc }

(* Host transfers *)

let itemsize dt = Nx_buffer.kind_size_in_bytes (ND.to_buffer_kind dt)

(* Wrap host memory as a device buffer without copying. The caller must keep the
   memory's owner reachable while the buffer can still be read or written. *)
let wrap_ptr dev dtolk n ptr =
  let buf =
    Tolk.Device.create_buffer ~size:n ~dtype:dtolk
      ~spec:{ Tolk.Device.Buffer_spec.default with external_ptr = Some ptr }
      dev
  in
  Tolk.Device.Buffer.ensure_allocated buf;
  buf

(* Wrap a tensor's memory, or [None] when its elements are not a contiguous
   span. Also returns the value that keeps the memory reachable. *)
let wrap_tensor : type a b.
    Tolk.Device.t -> (a, b) Nx_effect.t -> (Tolk.Device.Buffer.t * Obj.t) option
    =
 fun dev x ->
  let v = Nx_effect.view x in
  if not (NV.is_c_contiguous v) then None
  else
    let dt = Nx_effect.dtype x in
    let host = Nx_effect.to_host x in
    let ptr =
      Nativeint.add
        (Nx_buffer.unsafe_data_ptr host)
        (Nativeint.of_int (NV.offset v * itemsize dt))
    in
    Some (wrap_ptr dev (tolk_dtype dt) (numel (NV.shape v)) ptr, Obj.repr host)

(* A host buffer wired as a kernel output: the computed tensor is built on it
   directly. *)
type host_out = Host : ('a, 'b) ND.t * ('a, 'b) Nx_buffer.t -> host_out

(* Byte staging for device transfers, reused across calls and keyed by size so
   a compiled program does not repopulate the page tables with fresh
   multi-megabyte [Bytes] on every replay. Compiled functions are not
   thread-safe, and each scratch use completes before the next lookup. *)
type scratch = (int, Bytes.t) Hashtbl.t

let scratch_bytes tbl size =
  match Hashtbl.find_opt tbl size with
  | Some b -> b
  | None ->
      let b = Bytes.create size in
      Hashtbl.add tbl size b;
      b

(* Copy a tensor's logical contents into a device buffer. *)
let copyin_tensor : type a b.
    scratch -> Tolk.Device.Buffer.t -> (a, b) Nx_effect.t -> unit =
 fun sc buf x ->
  let xc = Nx_effect.contiguous x in
  let host = Nx_effect.to_host xc in
  let v = Nx_effect.view xc in
  let n = numel (NV.shape v) in
  let bytes = scratch_bytes sc (n * itemsize (Nx_effect.dtype x)) in
  Nx_buffer.blit_to_bytes ~src_off:(NV.offset v) ~len:n host bytes;
  Tolk.Device.Buffer.ensure_allocated buf;
  bytes_to_device := !bytes_to_device + Bytes.length bytes;
  Tolk.Device.Buffer.copyin buf bytes

(* Build a fresh tensor of [dt]/[shape] from a device buffer's contents. *)
let read_out : type a b.
    scratch ->
    Nx_effect.context ->
    (a, b) ND.t ->
    int array ->
    Tolk.Device.Buffer.t ->
    (a, b) Nx_effect.t =
 fun sc ctx dtv shape buf ->
  let n = numel shape in
  let bytes = scratch_bytes sc (Tolk.Device.Buffer.nbytes buf) in
  bytes_from_device := !bytes_from_device + Bytes.length bytes;
  Tolk.Device.Buffer.copyout buf bytes;
  let host = Nx_buffer.create (ND.to_buffer_kind dtv) n in
  Nx_buffer.blit_from_bytes ~len:n bytes host;
  Nx_effect.reshape (Nx_effect.from_host ctx host) shape

(* Wrap a device buffer as a deferred host tensor owning it. Metadata reads
   answer from the handle; the first data access synchronizes the device,
   copies the buffer out and releases it to the allocator. Copy-out failures
   surface at that first read. *)
let make_handle : type a b.
    device:Tolk.Device.t ->
    ctx:Nx_effect.context ->
    scratch:scratch ->
    (a, b) ND.t ->
    int array ->
    Tolk.Device.Buffer.t ->
    (a, b) Nx_effect.t =
 fun ~device ~ctx ~scratch dtv shape buf ->
  let nbytes = Tolk.Device.Buffer.nbytes buf in
  let entry = { r_id = -1; r_device = device; r_nbytes = nbytes; r_buf = Some buf } in
  let n = numel shape in
  let fill () =
    let buf =
      match entry.r_buf with
      | Some buf -> buf
      | None -> assert false (* released only by this fill or the finalizer *)
    in
    Tolk.Device.synchronize device;
    let bytes = scratch_bytes scratch nbytes in
    Tolk.Device.Buffer.copyout buf bytes;
    bytes_from_device := !bytes_from_device + nbytes;
    let host = Nx_buffer.create (ND.to_buffer_kind dtv) n in
    Nx_buffer.blit_from_bytes ~len:n bytes host;
    release_entry entry;
    host
  in
  let handle = Nx_effect.deferred ctx dtv shape fill in
  (match Nx_effect.deferred_id handle with
  | Some id ->
      entry.r_id <- id;
      Hashtbl.replace resident id entry;
      resident_bytes := !resident_bytes + nbytes
  | None -> assert false);
  (* The finalizer must not capture the handle (it would never die); the entry
     alone decides whether the buffer is still owed a release. *)
  Gc.finalise
    (fun _ ->
      if Option.is_some entry.r_buf then
        pending_release := entry :: !pending_release)
    handle;
  handle

let write_into : type a b.
    scratch ->
    Nx_effect.context ->
    (a, b) Nx_effect.t ->
    Tolk.Device.Buffer.t ->
    unit =
 fun sc ctx x buf ->
  Nx_effect.assign x (read_out sc ctx (Nx_effect.dtype x) (shape_of x) buf)

(* Compiled traces *)

type wb_target = Wb_input of int | Wb_const of packed

(* How a captured constant is bound: [Const_wrapped] buffers alias the tensor's
   memory, so its current value is read directly on every call (the packed
   tensor and host view keep the memory reachable); [Const_copy] buffers hold a
   copy of the tensor's bytes, uploaded once at compile time. [refresh] marks
   captures the traced function assigns to: the writeback keeps their host value
   current between calls, so their bytes are re-uploaded on every call and later
   mutations stay observed. *)
type const_binding =
  | Const_wrapped of packed * Obj.t
  | Const_copy of { buf : Tolk.Device.Buffer.t; src : packed; refresh : bool }

type 'q compiled = {
  cp_device : Tolk.Device.t;
  cp_zero_copy : bool;
  cp_ctx : Nx_effect.context;
  cp_linear : U.t;
  cp_vars : (string * int) list;
  cp_binding : Tolk.Realize.Buffers.t;
  cp_inputs : input array; (* one per leaf visit, in traversal order *)
  cp_consts : const_binding array;
  cp_outputs : (Obj.t * packed * U.t) list;
      (* output leaf -> its placeholder (dtype and shape) and buffer node *)
  cp_reserved : (int, unit) Hashtbl.t;
      (* tags of input and constant buffer nodes: outputs must not reseed
         them *)
  cp_skeleton : 'q; (* trace-time output structure *)
  cp_writebacks : (wb_target * U.t) array;
  cp_scratch : scratch; (* staging bytes reused across replays *)
}

let signature_of (module P : Nx.Ptree.S) (params : P.t) =
  let acc = ref [] in
  P.iter
    (fun leaf ->
      acc := (ND.to_string (Nx_effect.dtype leaf), shape_of leaf) :: !acc)
    params;
  List.rev !acc

let trace_compile ~device:dev ~zero_copy ~const_cache (module P : Nx.Ptree.S)
    (module Q : Nx.Ptree.S) (f : P.t -> Q.t) (params : P.t) : Q.t compiled =
  let st =
    {
      st_device = dev;
      st_ctx = Nx_effect.create_context ();
      table = Tbl.create 64;
      traced = Tbl.create 64;
      externals = Tbl.create 16;
      input_index = Tbl.create 16;
      wb_seen = Tbl.create 4;
      consts = [];
      writebacks = [];
    }
  in
  (* One placeholder per distinct leaf; one input record per leaf visit, in
     traversal order, so replay can pair current leaves positionally. Pairing
     placeholders inside [P.map] goes through the identity association because
     the map callback's evaluation order is instance-defined. *)
  let assoc = ref [] in
  let inputs = ref [] in
  let pos = ref 0 in
  P.iter
    (fun leaf ->
      let key = Obj.repr leaf in
      (match List.assq_opt key !assoc with
      | Some (_, inp) -> inputs := inp :: !inputs
      | None ->
          let ldt = Nx_effect.dtype leaf in
          let shape = shape_of leaf in
          let n = numel shape in
          let node = make_node st (tolk_dtype ldt) n in
          let buf =
            Tolk.Device.create_buffer ~size:n ~dtype:(tolk_dtype ldt) dev
          in
          let ph = Nx_effect.buffer st.st_ctx ldt shape in
          Tbl.replace st.table (Obj.repr ph) (buffer_tensor node shape);
          Tbl.replace st.traced (Obj.repr ph) ();
          Tbl.replace st.externals (Obj.repr ph) ();
          Tbl.replace st.input_index (Obj.repr ph) !pos;
          let inp = { i_node = node; i_buf = buf } in
          assoc := (key, (Packed (ldt, ph), inp)) :: !assoc;
          inputs := inp :: !inputs);
      incr pos)
    params;
  let ph_params =
    P.map
      (fun (type a b) (leaf : (a, b) Nx_effect.t) : (a, b) Nx_effect.t ->
        match List.assq_opt (Obj.repr leaf) !assoc with
        | Some (Packed (pdt, ph), _) -> (
            match ND.equal_witness pdt (Nx_effect.dtype leaf) with
            | Some Type.Equal -> ph
            | None -> assert false)
        | None -> assert false)
      params
  in
  let y =
    Gate.with_transform (fun () ->
        Effect.Deep.match_with f ph_params (handler st))
  in
  (* Collect the output leaves; a leaf the trace never saw is a constant passing
     through unchanged. *)
  let out_assoc = ref [] in
  Q.iter
    (fun (type a b) (leaf : (a, b) Nx_effect.t) ->
      let key = Obj.repr leaf in
      if not (List.exists (fun (k, _, _) -> k == key) !out_assoc) then
        out_assoc :=
          (key, Packed (Nx_effect.dtype leaf, leaf), tolk_of st leaf)
          :: !out_assoc)
    y;
  let outs = List.rev !out_assoc in
  let wbs = List.rev st.writebacks in
  (* A result computed purely from trace-time constants (say, the zero
     gradient of an unused parameter) has no device anywhere in its graph, so
     the scheduler would materialize nothing for it. Anchor such results with
     a bitwise-identity multiply by a device-resident scalar one. *)
  let anchor : type a b. (a, b) ND.t -> F.Tensor.t -> F.Tensor.t =
   fun dt tt ->
    match U.device_of (F.Tensor.uop tt) with
    | Some _ -> tt
    | None ->
        let one = Nx_effect.const_scalar st.st_ctx (ND.one dt) dt in
        F.Elementwise.mul tt (tolk_of st one)
  in
  let cont tt = U.contiguous ~src:(F.Tensor.uop tt) () in
  let out_conts =
    List.map
      (fun (key, (Packed (dt, _) as pk), tt) ->
        let tt = anchor dt tt in
        (key, pk, tt, cont tt))
      outs
  in
  let wb_conts =
    List.map
      (fun (Packed (dt, dst) as pk) ->
        let tt = anchor dt (tolk_of st dst) in
        (pk, tt, cont tt))
      wbs
  in
  let sink =
    U.sink
      (List.map (fun (_, _, _, c) -> c) out_conts
      @ List.map (fun (_, _, c) -> c) wb_conts)
  in
  let call, buffer_map = Tolk.Allocations.transform_to_call sink in
  (* Schedule under a capture hook: the captured linear is unplanned, which
     keeps buffer nodes stable so the seeded input and constant bindings survive
     across replays. *)
  let linear, var_vals =
    let captured = ref None in
    Tolk.Realize.capturing :=
      [ (fun linear var_vals -> captured := Some (linear, var_vals)) ];
    Fun.protect
      ~finally:(fun () -> Tolk.Realize.capturing := [])
      (fun () ->
        ignore
          (Tolk.Schedule.create_linear_with_vars
             ~get_kernel_graph:Tolk.Rangeify.get_kernel_graph call));
    match !captured with
    | Some lv -> lv
    | None -> err "Rune.jit: scheduling captured no computation"
  in
  let linear =
    Tolk.Realize.pm_compile ~device:dev ~to_program:(to_program dev) linear
  in
  let binding = Tolk.Realize.Buffers.create ~device:dev in
  let reserved = Hashtbl.create 16 in
  List.iter
    (fun (_, (_, inp)) ->
      Hashtbl.replace reserved (U.tag inp.i_node) ();
      Tolk.Realize.Buffers.seed binding inp.i_node inp.i_buf)
    !assoc;
  (* Bind each constant: alias its memory when the device shares host memory and
     the tensor is contiguous, copy its bytes once otherwise. Captures the
     function assigns to are refreshed from the host on every call. *)
  let scratch = Hashtbl.create 8 in
  let wb_keys = List.map (fun (Packed (_, dst)) -> Obj.repr dst) wbs in
  let consts =
    List.rev_map
      (fun (node, (Packed (cdt, src) as pk)) ->
        Hashtbl.replace reserved (U.tag node) ();
        match if zero_copy then wrap_tensor dev src else None with
        | Some (buf, keep) ->
            Tolk.Realize.Buffers.seed binding node buf;
            Const_wrapped (pk, keep)
        | None ->
            (* One device copy of a capture serves every signature of the
               closure: the bytes are uploaded when the capture is first
               compiled and later compilations reuse the buffer. Assigned
               (refresh) captures share it too — their re-upload happens on
               every call, not here. *)
            let buf =
              match Tbl.find_opt const_cache (Obj.repr src) with
              | Some buf -> buf
              | None ->
                  let n = numel (shape_of src) in
                  let buf =
                    Tolk.Device.create_buffer ~size:n ~dtype:(tolk_dtype cdt)
                      dev
                  in
                  copyin_tensor scratch buf src;
                  Tbl.replace const_cache (Obj.repr src) buf;
                  buf
            in
            Tolk.Realize.Buffers.seed binding node buf;
            Const_copy
              { buf; src = pk; refresh = List.memq (Obj.repr src) wb_keys })
      st.consts
  in
  (* Resolve each output to the buffer node realization assigned it. An output
     whose node is already a graph buffer (an input or constant returned
     unchanged) reads that buffer directly. *)
  let resolve what tt c =
    match Hashtbl.find_opt buffer_map (U.tag c) with
    | Some node -> node
    | None -> (
        let u = F.Tensor.uop tt in
        match U.op u with
        | Tolk_uop.Ops.Buffer -> u
        | _ -> err "Rune.jit: %s was not scheduled to a buffer" what)
  in
  let cp_outputs =
    List.map
      (fun (key, pk, tt, c) ->
        (key, pk, resolve "an output of the traced function" tt c))
      out_conts
  in
  let cp_writebacks =
    List.map
      (fun ((Packed (_, dst) as pk), tt, c) ->
        let node = resolve "an assigned tensor" tt c in
        let target =
          match Tbl.find_opt st.input_index (Obj.repr dst) with
          | Some i -> Wb_input i
          | None -> Wb_const pk
        in
        (target, node))
      wb_conts
  in
  {
    cp_device = dev;
    cp_zero_copy = zero_copy;
    cp_ctx = st.st_ctx;
    cp_linear = linear;
    cp_vars = var_vals;
    cp_binding = binding;
    cp_inputs = Array.of_list (List.rev !inputs);
    cp_consts = Array.of_list consts;
    cp_outputs;
    cp_reserved = reserved;
    cp_skeleton = y;
    cp_writebacks = Array.of_list cp_writebacks;
    cp_scratch = scratch;
  }

let replay (module P : Nx.Ptree.S) (module Q : Nx.Ptree.S) (c : Q.t compiled)
    (params : P.t) : Q.t =
  drain_releases ();
  let in0 = !bytes_to_device and out0 = !bytes_from_device in
  (* Seed the inputs. A leaf that is an unread output of an earlier call on
     this device seeds its input node with the resident buffer directly — no
     transfer, and the handle stays resident (inputs are read-only). Otherwise
     wrap the current leaf's memory when the device shares host memory and the
     leaf is contiguous, and copy its bytes if not. Seeded leaves and wrapped
     hosts are kept reachable until the run completes, so no finalizer can
     release a buffer the kernels still read. *)
  let resident_buffer leaf =
    match Nx_effect.deferred_id leaf with
    | None -> None
    | Some id -> (
        match Hashtbl.find_opt resident id with
        | Some e when e.r_device == c.cp_device -> e.r_buf
        | _ -> None)
  in
  let keep = ref [] in
  let i = ref 0 in
  P.iter
    (fun leaf ->
      let inp = c.cp_inputs.(!i) in
      (match resident_buffer leaf with
      | Some buf ->
          keep := Obj.repr leaf :: !keep;
          Tolk.Realize.Buffers.seed c.cp_binding inp.i_node buf
      | None -> (
          match
            if c.cp_zero_copy then wrap_tensor c.cp_device leaf else None
          with
          | Some (buf, ka) ->
              keep := ka :: !keep;
              Tolk.Realize.Buffers.seed c.cp_binding inp.i_node buf
          | None ->
              Tolk.Realize.Buffers.seed c.cp_binding inp.i_node inp.i_buf;
              copyin_tensor c.cp_scratch inp.i_buf leaf));
      incr i)
    params;
  Array.iter
    (function
      | Const_wrapped _ | Const_copy { refresh = false; _ } -> ()
      | Const_copy { buf; src = Packed (_, src); refresh = true } ->
          copyin_tensor c.cp_scratch buf src)
    c.cp_consts;
  (* Wire the outputs' storage. On the zero-copy device, fresh host buffers
     become the kernels' output storage, so results are written straight into
     the tensors returned to the caller; nodes backed by an input or constant
     buffer keep their binding and are read back through a copy instead. On
     other devices, every distinct non-reserved output node is bound to a
     fresh device buffer for this call, so handles from earlier calls keep
     their own storage and never alias a later call's outputs. *)
  let out_hosts : (int, host_out) Hashtbl.t = Hashtbl.create 8 in
  let out_bufs : (int, Tolk.Device.Buffer.t) Hashtbl.t = Hashtbl.create 8 in
  if c.cp_zero_copy then
    List.iter
      (fun (_, Packed (odt, ph), node) ->
        let tag = U.tag node in
        if
          (not (Hashtbl.mem c.cp_reserved tag))
          && not (Hashtbl.mem out_hosts tag)
        then begin
          let shape = shape_of ph in
          let n = numel shape in
          let host = Nx_buffer.create (ND.to_buffer_kind odt) n in
          let buf =
            wrap_ptr c.cp_device (tolk_dtype odt) n
              (Nx_buffer.unsafe_data_ptr host)
          in
          Tolk.Realize.Buffers.seed c.cp_binding node buf;
          Hashtbl.add out_hosts tag (Host (odt, host))
        end)
      c.cp_outputs
  else
    List.iter
      (fun (_, Packed (odt, ph), node) ->
        let tag = U.tag node in
        if
          (not (Hashtbl.mem c.cp_reserved tag))
          && not (Hashtbl.mem out_bufs tag)
        then begin
          let n = numel (shape_of ph) in
          let buf = create_fresh_buffer c.cp_device (tolk_dtype odt) n in
          Tolk.Realize.Buffers.seed c.cp_binding node buf;
          Hashtbl.add out_bufs tag buf
        end)
      c.cp_outputs;
  Tolk.Realize.run_linear ~device:c.cp_device
    ~to_program:(to_program c.cp_device) c.cp_binding ~var_vals:c.cp_vars
    ~jit:true c.cp_linear;
  (* An output that is an input or a capture returned unchanged keeps its
     reserved binding; copy it into a fresh buffer on the device, so its
     handle never aliases an input and survives later calls. *)
  if not c.cp_zero_copy then
    List.iter
      (fun (_, Packed (odt, ph), node) ->
        let tag = U.tag node in
        if Hashtbl.mem c.cp_reserved tag && not (Hashtbl.mem out_bufs tag)
        then begin
          let src = Tolk.Realize.Buffers.of_buffer_node c.cp_binding node in
          let n = numel (shape_of ph) in
          let dst = create_fresh_buffer c.cp_device (tolk_dtype odt) n in
          if not (Tolk.Device.Buffer.transfer ~dst ~src) then
            Tolk.Device.Buffer.copy_between ~dst ~src;
          Hashtbl.add out_bufs tag dst
        end)
      c.cp_outputs;
  (* Wrapped buffers alias caller memory: wait for in-flight kernels before
     reading results or letting the caller touch the inputs again. *)
  Tolk.Device.synchronize c.cp_device;
  ignore (Sys.opaque_identity !keep);
  (* Output leaves resolving to the same buffer node share one handle, so
     each device buffer has a single owner. *)
  let handles : (int, packed) Hashtbl.t = Hashtbl.create 8 in
  let y =
    Q.map
      (fun (type a b) (leaf : (a, b) Nx_effect.t) : (a, b) Nx_effect.t ->
        match
          List.find_opt (fun (k, _, _) -> k == Obj.repr leaf) c.cp_outputs
        with
        | Some (_, _, node) -> (
            let tag = U.tag node in
            match Hashtbl.find_opt out_hosts tag with
            | Some (Host (hdt, host)) -> (
                match ND.equal_witness hdt (Nx_effect.dtype leaf) with
                | Some Type.Equal ->
                    Nx_effect.reshape
                      (Nx_effect.from_host c.cp_ctx host)
                      (shape_of leaf)
                | None -> assert false)
            | None ->
                if c.cp_zero_copy then
                  let buf =
                    Tolk.Realize.Buffers.of_buffer_node c.cp_binding node
                  in
                  read_out c.cp_scratch c.cp_ctx (Nx_effect.dtype leaf)
                    (shape_of leaf) buf
                else (
                  match Hashtbl.find_opt handles tag with
                  | Some (Packed (hdt, h)) -> (
                      match ND.equal_witness hdt (Nx_effect.dtype leaf) with
                      | Some Type.Equal -> h
                      | None -> assert false)
                  | None ->
                      let buf =
                        match Hashtbl.find_opt out_bufs tag with
                        | Some buf -> buf
                        | None -> assert false (* every node was bound above *)
                      in
                      let dt = Nx_effect.dtype leaf in
                      let h =
                        make_handle ~device:c.cp_device ~ctx:c.cp_ctx
                          ~scratch:c.cp_scratch dt (shape_of leaf) buf
                      in
                      Hashtbl.add handles tag (Packed (dt, h));
                      h))
        | None -> assert false)
      c.cp_skeleton
  in
  Array.iter
    (fun (target, node) ->
      let buf = Tolk.Realize.Buffers.of_buffer_node c.cp_binding node in
      match target with
      | Wb_const (Packed (_, dst)) -> write_into c.cp_scratch c.cp_ctx dst buf
      | Wb_input idx ->
          let j = ref 0 in
          P.iter
            (fun leaf ->
              if !j = idx then write_into c.cp_scratch c.cp_ctx leaf buf;
              incr j)
            params)
    c.cp_writebacks;
  if Lazy.force jit_debug >= 1 then
    Printf.eprintf
      "rune.jit: replay on %s: %d bytes to device, %d bytes from device, %d \
       bytes resident\n\
       %!"
      (Tolk.Device.name c.cp_device)
      (!bytes_to_device - in0)
      (!bytes_from_device - out0)
      !resident_bytes;
  y

(* Public entry points *)

let jit2 ?(device = "CPU") (module P : Nx.Ptree.S) (module Q : Nx.Ptree.S)
    (f : P.t -> Q.t) : P.t -> Q.t =
  let dev = get_device device in
  let zero_copy = is_cpu device && not (force_copy ()) in
  let cache : (_, Q.t compiled) Hashtbl.t = Hashtbl.create 4 in
  (* Device copies of captured tensors, shared by every signature of this
     closure and keyed by capture identity. *)
  let const_cache : Tolk.Device.Buffer.t Tbl.t = Tbl.create 4 in
  fun params ->
    if Gate.transforming () then f params
    else
      let sg = signature_of (module P) params in
      let c =
        match Hashtbl.find_opt cache sg with
        | Some c -> c
        | None ->
            let c =
              trace_compile ~device:dev ~zero_copy ~const_cache
                (module P)
                (module Q)
                f params
            in
            Hashtbl.add cache sg c;
            c
      in
      replay (module P) (module Q) c params

let jit (type c d) ?device (module P : Nx.Ptree.S)
    (f : P.t -> (c, d) Nx_effect.t) : P.t -> (c, d) Nx_effect.t =
  let module Q = struct
    type t = (c, d) Nx_effect.t

    let map (f : 'a 'b. ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t) t = f t

    let map2
        (f :
          'a 'b.
          ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t)
        a b =
      f a b

    let iter (f : 'a 'b. ('a, 'b) Nx_effect.t -> unit) t = f t
  end in
  jit2 ?device (module P) (module Q) f

let jit' (type a b c d) ?device (f : (a, b) Nx_effect.t -> (c, d) Nx_effect.t) :
    (a, b) Nx_effect.t -> (c, d) Nx_effect.t =
  let module L = struct
    type t = (a, b) Nx_effect.t

    let map (f : 'a 'b. ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t) t = f t

    let map2
        (f :
          'a 'b.
          ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t)
        a b =
      f a b

    let iter (f : 'a 'b. ('a, 'b) Nx_effect.t -> unit) t = f t
  end in
  jit ?device (module L) f
