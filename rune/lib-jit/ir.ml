(* ml - Complete IR with all tinygrad operations *)

(* ───── Scalars & element types ───── *)

module Dtype = struct
  type _ t =
    | Float32 : float t
    | Int32 : int32 t
    | Bool : bool t
    | Uint8 : int t
    | Unit : unit t

  type any = Any_Dtype : 'a t -> any [@@unboxed]

  let to_string : type a. a t -> string = function
    | Float32 -> "float32"
    | Int32 -> "int32"
    | Bool -> "bool"
    | Uint8 -> "uint8"
    | Unit -> "unit"

  let any_to_string (Any_Dtype d) = to_string d

  let sizeof_elt : type a. a t -> int = function
    | Float32 | Int32 -> 4
    | Bool | Uint8 -> 1
    | Unit -> 0
end

(* ───── SSA variables & symbolic variables ───── *)

module Var = struct
  type t = int

  let counter = ref 0

  let fresh () =
    incr counter;
    !counter

  let compare = Int.compare
  let equal = Int.equal
  let hash = Hashtbl.hash
  let pp fmt v = Format.fprintf fmt "v%d" v
  let to_string = Format.asprintf "%a" pp

  module Set = struct
    include Set.Make (struct
      type nonrec t = t

      let compare = compare
    end)

    let pp fmt s =
      Format.fprintf fmt "{%a}"
        (Format.pp_print_list
           ~pp_sep:(fun f () -> Format.pp_print_string f ", ")
           pp)
        (elements s)
  end
end

(* Symbolic variables for dynamic shapes *)
module SymVar = struct
  type t = { name : string; min_val : int; max_val : int }
end

(* ───── Misc enums & types ───── *)

module Special_index_kind = struct
  type t =
    | Global_task_idx of int (* 0=x,1=y,2=z *)
    | Local_thread_idx of int
    | Workgroup_idx of int
end

type var_metadata = {
  dtype : Dtype.any;
  shape : int array;
  device : string option;
}

type kernel_metadata = {
  name : string;
  local_dims : int;
  upcasted : int;
  dont_use_locals : bool;
}

type custom_attr =
  | Attr_Int of int
  | Attr_Float of float
  | Attr_String of string
  | Attr_Shape of int array

(* Shape tracker for VIEW operations *)
type shape_tracker = { views : view list; shape : int array }

and view = {
  shape : int array;
  strides : int array;
  offset : int;
  mask : (int * int) array option; (* for masked/valid regions *)
}

(* ───── Operation kinds ───── *)

type binop_kind =
  | Add
  | Mul
  | Sub
  | Div
  | Idiv
  | Fdiv
  | Mod
  | Pow
  | Max
  | Min
  | Cmplt
  | Cmpne
  | Xor
  | Or
  | And
  | Shl
  | Shr (* bitwise shifts *)

type unary_op_kind = Neg | Log2 | Exp2 | Sin | Sqrt | Recip
type ternary_op_kind = Where | Mulacc (* multiply-accumulate *)
type reduce_op_kind = Reduce_Sum | Reduce_Max | Reduce_Prod

(* ───── High-level graph IR ───── *)

type _ node_t =
  (* ──── Buffer/Memory Operations ──── *)
  | Buffer : {
      dtype : 'a Dtype.t;
      size_in_elements : int;
      device : string;
      out_var : Var.t;
    }
      -> 'a node_t
  | Buffer_View : {
      (* view into existing buffer *)
      buffer_var : Var.t;
      size : int;
      offset : int;
      dtype : 'a Dtype.t;
      out_var : Var.t;
    }
      -> 'a node_t
  | Placeholder : {
      out_var : Var.t;
      dtype : 'a Dtype.t;
      shape : int array;
    }
      -> 'a node_t
  | Const_Scalar : {
      value : 'a;
      dtype : 'a Dtype.t;
      out_var : Var.t;
    }
      -> 'a node_t
  | Vconst : {
      (* vector constant *)
      values : 'a array;
      dtype : 'a Dtype.t;
      out_var : Var.t;
    }
      -> 'a node_t
  (* ──── Compute Operations ──── *)
  | Binop : {
      op : binop_kind;
      a_var : Var.t;
      b_var : Var.t;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  | Unary : {
      op : unary_op_kind;
      in_var : Var.t;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  | Ternary : {
      op : ternary_op_kind;
      a_var : Var.t;
      b_var : Var.t;
      c_var : Var.t;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  (* ──── Movement/Shape Operations ──── *)
  | View : {
      (* zero-copy shape operations *)
      in_var : Var.t;
      shape_tracker : shape_tracker;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  | Reshape : {
      in_var : Var.t;
      new_shape : int array;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  | Permute : {
      in_var : Var.t;
      axes_permutation : int array;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  | Expand : {
      in_var : Var.t;
      new_target_shape : int array;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  | Pad : {
      in_var : Var.t;
      pad_width : (int * int) array;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  | Shrink : {
      in_var : Var.t;
      limits : (int * int) array;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  | Flip : {
      in_var : Var.t;
      axes : int array;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  (* ──── Reduction Operations ──── *)
  | Reduce_Axis : {
      in_var : Var.t;
      reduce_op_kind : reduce_op_kind;
      axes : int array;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  (* ──── Advanced Operations ──── *)
  | Valid : {
      (* masked valid regions *)
      in_var : Var.t;
      shape_tracker : shape_tracker;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  | Index : {
      (* explicit indexing *)
      in_var : Var.t;
      idx_var : Var.t;
      valid_var : Var.t option;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  | Gep : {
      (* get element pointer for vectors *)
      in_var : Var.t;
      indices : int array;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  | Vectorize : {
      (* create vector from scalars *)
      in_vars : Var.t array;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  | Wmma : {
      (* tensor core operations *)
      a_var : Var.t;
      b_var : Var.t;
      c_var : Var.t;
      m : int;
      n : int;
      k : int;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  (* ──── Type Operations ──── *)
  | Cast : {
      in_var : Var.t;
      target_dtype : Dtype.any;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  | Bitcast : {
      (* reinterpret bits *)
      in_var : Var.t;
      target_dtype : Dtype.any;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  (* ──── Memory Operations ──── *)
  | Contiguous : {
      in_var : Var.t;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  | Copy : {
      in_var : Var.t;
      target_device : string;
      clone : bool; (* if true, force copy even on same device *)
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  | Assign : {
      target_var : Var.t;
      updates : (Var.t * Var.t * (int * int) option) array;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  (* ──── Symbolic/Dynamic Shapes ──── *)
  | Define_Var : {
      (* symbolic variables *)
      sym_var : SymVar.t;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  | Bind : {
      (* bind symbolic var to value *)
      sym_var : Var.t;
      value : int;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  (* ──── AutoGrad Support ──── *)
  | Detach : {
      (* stop gradient *)
      in_var : Var.t;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  | Contiguous_Backward : {
      (* backward pass marker *)
      in_var : Var.t;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  (* ──── Kernel/Graph Management ──── *)
  | Sink : {
      (* dependency synchronization *)
      deps : Var.t array;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  | Kernel : {
      (* kernel wrapper *)
      ast : any_node;
      input_vars : Var.t array;
      output_vars : Var.t array;
      metadata : kernel_metadata;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  | Unique : {
      (* unique identifier generation *)
      id : int;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  (* ──── Device Management ──── *)
  | Device : {
      (* device marker *)
      device_name : string;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  | Multi : {
      (* multi-device tensor *)
      device_vars : Var.t array;
      axis : int option;
      real_mask : bool array;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  (* ──── Optimization Directives ──── *)
  | Fuse : {
      (* fusion marker *)
      in_var : Var.t;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  | Unroll : {
      (* loop unroll directive *)
      loop_var : Var.t;
      unroll_factor : int;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  | Contract : {
      (* tensor contraction *)
      in_vars : Var.t array;
      contraction_axes : (int * int) array;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  (* ──── Miscellaneous Operations ──── *)
  | Cat : {
      in_vars : Var.t array;
      axis : int;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  | Threefry : {
      ctr_var : Var.t;
      key_var : Var.t;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  | Gather : {
      src_var : Var.t;
      indices_var : Var.t;
      axis : int;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  | Scatter : {
      indices_var : Var.t;
      updates_var : Var.t;
      axis : int;
      shape : int array;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  | Custom : {
      (* custom operation *)
      op_name : string;
      in_vars : Var.t array;
      attributes : (string * custom_attr) list;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  | Noop : {
      (* no operation *)
      in_var : Var.t option;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t

and any_node = Any_Node : 'a node_t -> any_node [@@unboxed]

type graph_t = {
  nodes : any_node list;
  vars_metadata : (Var.t, var_metadata) Hashtbl.t;
  input_vars : Var.t list;
  output_vars : Var.t list;
  symbolic_vars : SymVar.t list;
}

let buffer ~dtype ~size ~device ~out_var =
  Buffer { dtype; size_in_elements = size; device; out_var }

let unary ~op ~in_var ~out_var ~dtype = Unary { op; in_var; out_var; dtype }

let binary ~op ~a_var ~b_var ~out_var ~dtype =
  Binop { op; a_var; b_var; out_var; dtype }

let ternary ~op ~a_var ~b_var ~c_var ~out_var ~dtype =
  Ternary { op; a_var; b_var; c_var; out_var; dtype }

let const_scalar ~value ~out_var ~dtype = Const_Scalar { value; out_var; dtype }
let vconst ~values ~out_var ~dtype = Vconst { values; out_var; dtype }

let reshape ~in_var ~new_shape ~out_var ~dtype =
  Reshape { in_var; new_shape; out_var; dtype }

let permute ~in_var ~axes_permutation ~out_var ~dtype =
  Permute { in_var; axes_permutation; out_var; dtype }

let expand ~in_var ~new_target_shape ~out_var ~dtype =
  Expand { in_var; new_target_shape; out_var; dtype }

let pad ~in_var ~pad_width ~out_var ~dtype =
  Pad { in_var; pad_width; out_var; dtype }

let shrink ~in_var ~limits ~out_var ~dtype =
  Shrink { in_var; limits; out_var; dtype }

let reduce_axis ~in_var ~reduce_op_kind ~axes ~out_var ~dtype =
  Reduce_Axis { in_var; reduce_op_kind; axes; out_var; dtype }

let cast ~in_var ~target_dtype ~out_var ~dtype =
  Cast { in_var; target_dtype; out_var; dtype }

let bitcast ~in_var ~target_dtype ~out_var ~dtype =
  Bitcast { in_var; target_dtype; out_var; dtype }

let view ~in_var ~shape_tracker ~out_var ~dtype =
  View { in_var; shape_tracker; out_var; dtype }

let copy ~in_var ~target_device ~clone ~out_var ~dtype =
  Copy { in_var; target_device; clone; out_var; dtype }

let cat ~in_vars ~axis ~out_var ~dtype = Cat { in_vars; axis; out_var; dtype }

let gather ~src_var ~indices_var ~axis ~out_var ~dtype =
  Gather { src_var; indices_var; axis; out_var; dtype }

let scatter ~indices_var ~updates_var ~axis ~shape ~out_var ~dtype =
  Scatter { indices_var; updates_var; axis; shape; out_var; dtype }

let fresh_var () = Var.fresh ()

(* ───── Scheduled IR ───── *)

(* ───── Scheduled IR (single module, structured loops/tiles/mapping) ───── *)

module Scheduled = struct
  (* Utilities *)

  let[@inline] prod (arr : int array) = Array.fold_left ( * ) 1 arr

  let[@inline] ensure3 (a : int array) : int array =
    match Array.length a with
    | 3 -> a
    | 0 -> [| 1; 1; 1 |]
    | 1 -> [| a.(0); 1; 1 |]
    | 2 -> [| a.(0); a.(1); 1 |]
    | _ -> [| a.(0); a.(1); a.(2) |]

  let[@inline] contiguous_strides_elems (shape : int array) : int array =
    let n = Array.length shape in
    if n = 0 then [||]
    else
      let s = Array.make n 0 in
      let stride = ref 1 in
      for i = n - 1 downto 0 do
        s.(i) <- !stride;
        stride := !stride * if shape.(i) = 0 then 1 else shape.(i)
      done;
      s

  (* Core scheduling types *)

  type axis_role = [ `Normal | `Reduction ]

  type axis = {
    name : string;
    size : int option; (* known static extent or None (symbolic) *)
    sym : SymVar.t option; (* the symbolic var that bounds the axis *)
    role : axis_role;
  }

  type mapping = {
    block : int list; (* threadblock / grid dims on GPU; core on CPU *)
    thread : int list; (* thread / lane *)
    vec : int list; (* vector lanes (SIMD) *)
    serial : int list; (* remaining serial loops *)
  }

  type iter_space = {
    axes : axis array;
    (* logical iteration axes *)
    (* mapping selects WHICH axis indices (into [axes]) map to each machine
           level *)
    mapping : mapping;
    (* tiling: for each axis i, a list of tile sizes (outer→inner) *)
    tiles : int list array;
  }

  type memory_scope = Global | Shared | Register

  (* Layout strides are measured in ELEMENTS (not bytes); dtype tells byte
     width *)
  type layout = {
    shape : int array; (* logical shape in elements *)
    strides : int array; (* strides in elements (row-major typical) *)
    alignment : int; (* bytes *)
    vector_width : int; (* elements per vector lane *)
    contiguous_axes : int list; (* for coalescing; usually [last;...] *)
  }

  type allocation = {
    scope : memory_scope;
    size_bytes : int; (* final allocated size (post-tiling/packing) *)
    lifetime : int * int; (* inclusive item-id range for reuse *)
    alias_group : int option; (* optional alias set id for in-place plans *)
  }

  type buffer_info = {
    buf_var : Var.t;
    dtype : Dtype.any;
    layout : layout;
    alloc : allocation;
    is_input : bool;
    is_output : bool;
  }

  type loop_hint =
    | Vectorize of { axis : int; width : int }
    | Unroll of { axis : int; factor : int }
    | Prefetch of { var : Var.t; into : memory_scope; distance : int }
    | Pipeline of { axis : int; stages : int; overlap : bool }

  type reduction_plan = {
    axes : int list; (* indices (into iter_space.axes) tagged as reductions *)
    intra_thread : [ `Tree | `Welford | `Shfl | `None ];
    inter_thread : [ `SharedTree | `Atomic | `GridReduce ];
  }

  type schedule_context = {
    global_dims : int array; (* [|gx;gy;gz|] *)
    local_dims : int array; (* [|lx;ly;lz|] *)
    upcasted : int;
    device : string;
    stream : int option;
  }

  type scheduled_op =
    | S_Kernel of {
        kernel_id : int;
        kernel_name : string;
        ops : any_node list; (* HL ops fused into this kernel *)
        inputs : buffer_info list;
        outputs : buffer_info list;
        iter : iter_space; (* explicit loops/tiling/mapping *)
        reduce : reduction_plan option;
        hints : loop_hint list;
        context : schedule_context;
      }
    | S_Memory_Transfer of {
        transfer_id : int;
        src_var : Var.t;
        dst_var : Var.t;
        src_device : string;
        dst_device : string;
        dims : int array; (* ND copy extents in elements *)
        src_strides : int array option; (* elements; pitched if provided *)
        dst_strides : int array option; (* elements *)
        size_bytes : int; (* optional precomputed flat size *)
        is_async : bool;
        stream : int option;
      }
    | S_Synchronization of {
        sync_id : int;
        sync_type : [ `Barrier | `Fence | `Event of int ];
        scope : [ `Threadgroup | `Device | `System ];
        devices : string list;
        stream : int option;
      }
    | S_Host_Callback of {
        callback_id : int;
        callback_name : string;
        input_vars : Var.t list;
        output_vars : Var.t list;
      }

  type dependency = {
    dep_from : int; (* schedule_item id *)
    dep_to : int; (* schedule_item id *)
    dep_vars : Var.t list; (* values creating the edge *)
    kind : [ `Data | `Control ];
  }

  type schedule_item = {
    item_id : int;
    operation : scheduled_op;
    depends_on : int list; (* item ids *)
    dependents : int list; (* filled by validation/toposort *)
  }

  type fusion_opportunity = {
    kernel_a : int; (* item id *)
    kernel_b : int; (* item id *)
    fusion_type : [ `Elementwise | `Reduction | `Mixed ];
    benefit_score : float;
    memory_saved : int; (* bytes *)
  }

  (* Lightweight analysis product kept outside the core op shape *)
  type item_analysis = {
    item_id : int;
    flops : int;
    bytes_read : int;
    bytes_written : int;
    regs_per_thread : int;
    smem_bytes : int;
    occupancy : float; (* 0–1 estimate *)
    est_ns : int; (* estimated latency in ns *)
  }

  type graph_t = {
    schedule_items : schedule_item array;
    dependencies : dependency list;
    fusion_opportunities : fusion_opportunity list;
    analysis : item_analysis array; (* same order as items; may be empty *)
    critical_path : int list; (* item ids *)
    total_memory_usage : int; (* approximate peak, bytes *)
    estimated_runtime_ns : int; (* critical path sum *)
    vars_metadata : (Var.t, var_metadata) Hashtbl.t;
    symbolic_vars : SymVar.t list;
  }

  (* Validation & helpers *)

  let validate_dims3 (a : int array) (label : string) : unit =
    if Array.length a <> 3 then
      invalid_arg (Printf.sprintf "Scheduled.%s must be length-3" label)

  let validate_iter_space (it : iter_space) : unit =
    let n = Array.length it.axes in
    let in_range i =
      if i < 0 || i >= n then
        invalid_arg
          (Printf.sprintf
             "Scheduled.iter_space: axis index %d out of range 0..%d" i (n - 1))
    in
    List.iter in_range it.mapping.block;
    List.iter in_range it.mapping.thread;
    List.iter in_range it.mapping.vec;
    List.iter in_range it.mapping.serial;
    if Array.length it.tiles <> n then
      invalid_arg "Scheduled.iter_space: tiles length must match axes length"

  let size_bytes_of_layout (dt : Dtype.any) (ly : layout) : int =
    let elt =
      match dt with
      | Dtype.Any_Dtype Dtype.Float32 -> 4
      | Dtype.Any_Dtype Dtype.Int32 -> 4
      | Dtype.Any_Dtype Dtype.Uint8 -> 1
      | Dtype.Any_Dtype Dtype.Bool -> 1
      | Dtype.Any_Dtype Dtype.Unit -> 0
    in
    prod ly.shape * elt

  let default_layout ?(vector_width = 1) ?(alignment = 16) (shape : int array) :
      layout =
    {
      shape;
      strides = contiguous_strides_elems shape;
      alignment;
      vector_width;
      contiguous_axes =
        (let n = Array.length shape in
         let rec aux i acc = if i < 0 then acc else aux (i - 1) (i :: acc) in
         aux (n - 1) []);
    }

  let default_alloc ~scope ~dtype ~layout ~lifetime : allocation =
    let sz = size_bytes_of_layout dtype layout in
    { scope; size_bytes = sz; lifetime; alias_group = None }

  (* Build dependents lists from depends_on *)
  let compute_dependents (items : schedule_item array) : unit =
    let n = Array.length items in
    let deps_rev : int list array = Array.make n [] in
    Array.iter
      (fun (it : schedule_item) ->
        List.iter
          (fun (p : int) ->
            if p >= 0 && p < n then deps_rev.(p) <- it.item_id :: deps_rev.(p))
          it.depends_on)
      items;
    Array.iteri
      (fun i (it : schedule_item) ->
        items.(i) <- { it with dependents = List.rev deps_rev.(i) })
      items

  (* Topological order (Kahn). Returns item ids in topo sequence. *)
  let topological_order (items : schedule_item array) : int list =
    let n = Array.length items in
    let indeg = Array.make n 0 in
    Array.iter
      (fun (it : schedule_item) ->
        (* indegree of a node is number of its dependencies *)
        indeg.(it.item_id) <- List.length it.depends_on)
      items;
    let q = Queue.create () in
    for i = 0 to n - 1 do
      if indeg.(i) = 0 then Queue.add i q
    done;
    let order = ref [] in
    while not (Queue.is_empty q) do
      let u = Queue.pop q in
      order := u :: !order;
      List.iter
        (fun v ->
          indeg.(v) <- indeg.(v) - 1;
          if indeg.(v) = 0 then Queue.add v q)
        items.(u).dependents
    done;
    List.rev !order

  let find_critical_path (g : graph_t) : int list =
    let n = Array.length g.schedule_items in
    if n = 0 then []
    else
      let cost = Array.make n 1 in
      Array.iter
        (fun a -> if a.item_id < n then cost.(a.item_id) <- max 1 a.est_ns)
        g.analysis;
      let dist = Array.make n min_int in
      let prev = Array.make n (-1) in
      let indeg = Array.make n 0 in
      Array.iter
        (fun (it : schedule_item) ->
          indeg.(it.item_id) <- List.length it.depends_on)
        g.schedule_items;
      let q = Queue.create () in
      for i = 0 to n - 1 do
        if indeg.(i) = 0 then (
          dist.(i) <- cost.(i);
          Queue.add i q)
      done;
      while not (Queue.is_empty q) do
        let u = Queue.pop q in
        List.iter
          (fun v ->
            (if dist.(u) <> min_int then
               let cand = dist.(u) + cost.(v) in
               if cand > dist.(v) then (
                 dist.(v) <- cand;
                 prev.(v) <- u));
            indeg.(v) <- indeg.(v) - 1;
            if indeg.(v) = 0 then Queue.add v q)
          g.schedule_items.(u).dependents
      done;
      let end_id = ref 0 in
      for i = 1 to n - 1 do
        if dist.(i) > dist.(!end_id) then end_id := i
      done;
      let rec build acc u = if u = -1 then acc else build (u :: acc) prev.(u) in
      build [] !end_id

  let sum_estimated_runtime_ns (g : graph_t) : int =
    List.fold_left
      (fun acc id ->
        match Array.find_opt (fun a -> a.item_id = id) g.analysis with
        | Some a -> acc + max 1 a.est_ns
        | None -> acc + 1)
      0 g.critical_path

  (* Very rough peak memory estimate: sum of distinct kernel allocations at each
     item *)
  let estimate_peak_memory (g : graph_t) : int =
    let mem_at_item (it : schedule_item) : int =
      match it.operation with
      | S_Kernel { inputs; outputs; _ } ->
          let sum l =
            List.fold_left (fun acc b -> acc + b.alloc.size_bytes) 0 l
          in
          (* NOTE: This double counts shared/register; acceptable as an upper
             bound. *)
          sum inputs + sum outputs
      | S_Memory_Transfer { size_bytes; _ } -> size_bytes
      | _ -> 0
    in
    Array.fold_left (fun acc it -> max acc (mem_at_item it)) 0 g.schedule_items

  (* Constructors *)

  let make_iter_space ~axes ~mapping ~tiles : iter_space =
    let s = { axes; mapping; tiles } in
    validate_iter_space s;
    s

  let make_buffer_info ~(buf_var : Var.t) ~(dtype : Dtype.any)
      ~(shape : int array) ~(scope : memory_scope) ~(is_input : bool)
      ~(is_output : bool) ~(lifetime : int * int) : buffer_info =
    let layout = default_layout shape in
    let alloc = default_alloc ~scope ~dtype ~layout ~lifetime in
    { buf_var; dtype; layout; alloc; is_input; is_output }

  let create_kernel ~kernel_id ~kernel_name ~ops ~inputs ~outputs ~iter ~reduce
      ~hints ~context : scheduled_op =
    validate_dims3 context.global_dims "context.global_dims";
    validate_dims3 context.local_dims "context.local_dims";
    validate_iter_space iter;
    S_Kernel
      {
        kernel_id;
        kernel_name;
        ops;
        inputs;
        outputs;
        iter;
        reduce;
        hints;
        context;
      }

  let create_memory_transfer ~transfer_id ~src_var ~dst_var ~src_device
      ~dst_device ~dims ?src_strides ?dst_strides ~size_bytes ~is_async ~stream
      () : scheduled_op =
    S_Memory_Transfer
      {
        transfer_id;
        src_var;
        dst_var;
        src_device;
        dst_device;
        dims;
        src_strides;
        dst_strides;
        size_bytes;
        is_async;
        stream;
      }

  let create_synchronization ~sync_id ~sync_type ~scope ~devices ~stream :
      scheduled_op =
    S_Synchronization { sync_id; sync_type; scope; devices; stream }

  let create_host_callback ~callback_id ~callback_name ~input_vars ~output_vars
      : scheduled_op =
    S_Host_Callback { callback_id; callback_name; input_vars; output_vars }

  let create_schedule_item ~item_id ~operation ~depends_on : schedule_item =
    { item_id; operation; depends_on; dependents = [] }
end

(* ───── Low-level / lowered IR ───── *)

module Lowered = struct
  type alu_op =
    | Binary of binop_kind
    | Unary of unary_op_kind
    | Ternary of ternary_op_kind

  type instruction =
    (* Memory allocation *)
    | L_Buffer of { dtype : Dtype.any; size : int; out : Var.t }
    | L_Local of { dtype : Dtype.any; size : int; out : Var.t }
    | L_Acc of { dtype : Dtype.any; out : Var.t }
    (* Memory definitions *)
    | L_Define_Global of {
        (* global memory definition *)
        ptr : Var.t;
        dtype : Dtype.any;
        size : int;
      }
    (* Constants and indices *)
    | L_Const of { dtype : Dtype.any; value : string; out : Var.t }
    | L_Vconst of {
        (* vector constant *)
        dst : Var.t;
        values : string array;
        dtype : Dtype.any;
      }
    | L_Special of { dst : Var.t; kind : Special_index_kind.t }
    | L_Define_Var of { sym_var : SymVar.t; out : Var.t }
    (* Control flow *)
    | L_Range of { idx : Var.t; bound : Var.t }
    | L_EndRange
    | L_If of { cond : Var.t }
    | L_EndIf
    | L_Barrier
    (* Block operations *)
    | L_Block of {
        (* block marker *)
        block_id : int;
        start : bool; (* true for BLOCKSTART, false for BLOCKEND *)
      }
    (* Unrolling *)
    | L_Unroll of {
        (* unrolled loop *)
        idx : Var.t;
        iterations : int;
      }
    (* Memory access *)
    | L_Load of {
        dst : Var.t;
        buf : Var.t;
        idx : Var.t;
        dtype : Dtype.any;
        valid : Var.t option; (* masked loads *)
      }
    | L_Store of {
        buf : Var.t;
        idx : Var.t;
        src : Var.t;
        valid : Var.t option; (* masked stores *)
      }
    (* Compute *)
    | L_ALU of {
        dst : Var.t;
        op : alu_op;
        args : Var.t list;
        dtype : Dtype.any;
      }
    (* Vector operations *)
    | L_Gep of {
        (* get element from vector *)
        dst : Var.t;
        src : Var.t;
        indices : int array;
        dtype : Dtype.any;
      }
    | L_Vectorize of {
        (* build vector *)
        dst : Var.t;
        srcs : Var.t array;
        dtype : Dtype.any;
      }
    (* Pointer operations *)
    | L_Ptrcat of {
        (* pointer concatenation *)
        dst : Var.t;
        ptrs : Var.t array;
        dtype : Dtype.any;
      }
    (* Tensor core operations *)
    | L_Wmma of {
        dst : Var.t;
        a : Var.t;
        b : Var.t;
        c : Var.t;
        m : int;
        n : int;
        k : int;
        dtype : Dtype.any;
      }
    (* Data movement *)
    | L_Cast of { dst : Var.t; src : Var.t; dtype : Dtype.any }
    | L_Bitcast of { dst : Var.t; src : Var.t; dtype : Dtype.any }
    | L_Assign of { dst : Var.t; src : Var.t }
    (* Custom operations *)
    | L_Custom of {
        dst : Var.t option;
        op_name : string;
        args : Var.t array;
        attributes : (string * custom_attr) list;
        inline : bool; (* CUSTOMI vs CUSTOM *)
      }
    (* No-op *)
    | L_Noop

  type graph_t = {
    instructions : instruction list;
    vars_metadata : (Var.t, var_metadata) Hashtbl.t;
    kernel_input_vars : Var.t list;
    kernel_output_vars : Var.t list;
    symbolic_vars : SymVar.t list;
  }
end
