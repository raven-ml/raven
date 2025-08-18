(* schedule/kernelize.ml Convert Grouper.cluster_t clusters into Scheduled IR
   kernel items. *)

open Ir
module S = Ir.Scheduled

(* small helpers *)

let prod (a : int array) = Array.fold_left ( * ) 1 a

let shape_of (md_tbl : (Var.t, var_metadata) Hashtbl.t) (v : Var.t) : int array
    =
  match Hashtbl.find_opt md_tbl v with Some md -> md.shape | None -> [||]

let dtype_of (md_tbl : (Var.t, var_metadata) Hashtbl.t) (v : Var.t) : Dtype.any
    =
  match Hashtbl.find_opt md_tbl v with
  | Some md -> md.dtype
  | None -> Dtype.Any_Dtype Dtype.Unit

let device_of (md_tbl : (Var.t, var_metadata) Hashtbl.t) (v : Var.t) : string =
  match Hashtbl.find_opt md_tbl v with
  | Some md -> ( match md.device with Some d -> d | None -> "default")
  | None -> "default"

let sizeof_any (Dtype.Any_Dtype dt) = Dtype.sizeof_elt dt

let nbytes_of (dt : Dtype.any) (shape : int array) =
  let elems = if Array.length shape = 0 then 1 else prod shape in
  elems * sizeof_any dt

(* choose a "kernel output shape" (used to build iteration space) *)
let pick_output_shape (spec : Grouper.cluster_t) : int array =
  match spec.outputs with
  | v :: _ -> shape_of spec.vars_metadata v
  | [] ->
      (* fallback: walk nodes until we find a var with known shape *)
      let rec first_out = function
        | [] -> [||]
        | Ir.Any_Node n :: tl ->
            let v =
              match n with
              | Placeholder { out_var; _ }
              | Const_Scalar { out_var; _ }
              | Vconst { out_var; _ }
              | Buffer { out_var; _ }
              | Buffer_View { out_var; _ }
              | Binop { out_var; _ }
              | Unary { out_var; _ }
              | Ternary { out_var; _ }
              | Reduce_Axis { out_var; _ }
              | Expand { out_var; _ }
              | Reshape { out_var; _ }
              | Permute { out_var; _ }
              | Pad { out_var; _ }
              | Shrink { out_var; _ }
              | Flip { out_var; _ }
              | Cat { out_var; _ }
              | Cast { out_var; _ }
              | Bitcast { out_var; _ }
              | Contiguous { out_var; _ }
              | Copy { out_var; _ }
              | Assign { out_var; _ }
              | Threefry { out_var; _ }
              | Gather { out_var; _ }
              | Scatter { out_var; _ }
              | View { out_var; _ }
              | Valid { out_var; _ }
              | Index { out_var; _ }
              | Gep { out_var; _ }
              | Vectorize { out_var; _ }
              | Wmma { out_var; _ }
              | Define_Var { out_var; _ }
              | Bind { out_var; _ }
              | Detach { out_var; _ }
              | Contiguous_Backward { out_var; _ }
              | Multi { out_var; _ }
              | Fuse { out_var; _ }
              | Unroll { out_var; _ }
              | Contract { out_var; _ }
              | Kernel { out_var; _ }
              | Unique { out_var; _ }
              | Device { out_var; _ }
              | Custom { out_var; _ }
              | Noop { out_var; _ } ->
                  out_var
              | Sink _ -> Var.fresh ()
            in
            let sh = shape_of spec.vars_metadata v in
            if Array.length sh > 0 then sh else first_out tl
      in
      first_out spec.nodes

(* build S.buffer_info *)

let make_buf_info ~(is_input : bool) ~(is_output : bool)
    (spec : Grouper.cluster_t) (v : Var.t) : S.buffer_info =
  let shape = shape_of spec.vars_metadata v in
  let dtype = dtype_of spec.vars_metadata v in
  let layout = S.default_layout shape in
  (* lifetime filled later by allocator or left as zero-span; scope Global *)
  let alloc = S.default_alloc ~scope:S.Global ~dtype ~layout ~lifetime:(0, 0) in
  { S.buf_var = v; dtype; layout; alloc; is_input; is_output }

(* build S.iter_space from an output shape *)

let make_iter_space (out_shape : int array) : S.iter_space =
  let nd = Array.length out_shape in
  let axes =
    if nd = 0 then
      [| { S.name = "i0"; size = Some 1; sym = None; role = `Normal } |]
    else
      Array.mapi
        (fun i sz ->
          {
            S.name = Printf.sprintf "i%d" i;
            size = Some sz;
            sym = None;
            role = `Normal;
          })
        out_shape
  in
  (* trivial mapping: last axis → thread, others serial *)
  let mapping =
    if nd = 0 then { S.block = []; thread = [ 0 ]; vec = []; serial = [] }
    else
      let last = nd - 1 in
      let serial = if nd > 1 then List.init (nd - 1) (fun i -> i) else [] in
      { S.block = []; thread = [ last ]; vec = []; serial }
  in
  let tiles = Array.init (Array.length axes) (fun _ -> []) in
  { S.axes; mapping; tiles }

(* device / context *)

let pick_device (spec : Grouper.cluster_t) : string =
  match spec.outputs with
  | v :: _ -> device_of spec.vars_metadata v
  | [] -> (
      match spec.inputs with
      | v :: _ -> device_of spec.vars_metadata v
      | [] -> "default")

let make_context (spec : Grouper.cluster_t) (out_shape : int array) :
    S.schedule_context =
  let device = pick_device spec in
  let flat = if Array.length out_shape = 0 then 1 else prod out_shape in
  {
    S.global_dims = [| flat; 1; 1 |];
    S.local_dims = [| 1; 1; 1 |];
    S.upcasted = 1;
    S.device;
    S.stream = None;
  }

(* public: one cluster -> one kernel item + analysis *)

let of_spec (kernel_id : int) (spec : Grouper.cluster_t) :
    S.schedule_item * S.item_analysis =
  let out_shape = pick_output_shape spec in
  let iter = make_iter_space out_shape in
  let context = make_context spec out_shape in

  let inputs =
    List.map (make_buf_info ~is_input:true ~is_output:false spec) spec.inputs
  in
  let outputs =
    List.map (make_buf_info ~is_input:false ~is_output:true spec) spec.outputs
  in

  let op =
    S.S_Kernel
      {
        kernel_id;
        kernel_name = spec.name;
        ops = spec.nodes;
        inputs;
        outputs;
        iter;
        reduce = None;
        hints = [];
        context;
      }
  in
  let item : S.schedule_item =
    { item_id = kernel_id; operation = op; depends_on = []; dependents = [] }
  in

  (* naive analysis: bytes read/written, est_ns placeholder *)
  let bytes_in =
    List.fold_left
      (fun acc b -> acc + nbytes_of b.S.dtype b.S.layout.shape)
      0 inputs
  in
  let bytes_out =
    List.fold_left
      (fun acc b -> acc + nbytes_of b.S.dtype b.S.layout.shape)
      0 outputs
  in
  let analysis : S.item_analysis =
    {
      item_id = kernel_id;
      flops = 0;
      bytes_read = bytes_in;
      bytes_written = bytes_out;
      regs_per_thread = 0;
      smem_bytes = 0;
      occupancy = 1.0;
      est_ns = 1;
    }
  in
  (item, analysis)

(* public: many clusters -> items + analyses *)

let of_specs (clusters : Grouper.cluster_t list) :
    S.schedule_item list * S.item_analysis list =
  let rec go k acc_items acc_ana = function
    | [] -> (List.rev acc_items, List.rev acc_ana)
    | spec :: tl ->
        let item, ana = of_spec k spec in
        go (k + 1) (item :: acc_items) (ana :: acc_ana) tl
  in
  go 0 [] [] clusters
