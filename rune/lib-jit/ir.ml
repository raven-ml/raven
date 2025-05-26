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
