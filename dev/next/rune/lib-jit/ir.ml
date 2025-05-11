(* lib-jit/ir.ml in the new rune_jit library *)

(* Self-contained Dtype definition for the IR. 'a_elt is the OCaml
   representation of the element (e.g., float, int). 'b_layout_phantom is a
   phantom type for distinguishing kinds/layouts if necessary. *)
module Dtype = struct
  type float_elt = float
  type int_elt = int
  type bool_elt = bool

  type unit_elt =
    unit (* For operations that don't produce a value, like Store *)

  type _x_layout_phantom (* A generic phantom for most types' layouts *)
  type unit_layout_phantom (* A specific phantom for unit type layout *)

  type ('a_elt, 'b_layout_phantom) t =
    | Float32 : (float_elt, _x_layout_phantom) t
    | Int32 : (int_elt, _x_layout_phantom) t
    | Bool : (bool_elt, _x_layout_phantom) t
    | Uint8 :
        (int_elt, _x_layout_phantom) t (* OCaml int represents uint8 values *)
    (* Add other necessary types like Float16, Int64, Uint32, etc. as needed *)
    | Unit : (unit_elt, unit_layout_phantom) t
  (* For GADT parameters of effectful ops like Store *)

  let to_string (type a b) (dt : (a, b) t) : string =
    match dt with
    | Float32 -> "float32"
    | Int32 -> "int32"
    | Bool -> "bool"
    | Uint8 -> "uint8"
    | Unit -> "unit"

  type any = Any_Dtype : ('a, 'b) t -> any [@@unboxed]

  let any_to_string (Any_Dtype dt) = to_string dt
end

module Var = struct
  type t = int

  let counter = ref 0

  let fresh () =
    incr counter;
    !counter

  let compare = Int.compare
  let equal = Int.equal
  let hash = Hashtbl.hash
  let pp fmt v = Format.fprintf fmt "ir_var%d" v
end

module Special_index_kind = struct
  type t =
    | Global_task_idx of int (* dimension index, e.g., 0 for x, 1 for y *)
    | Local_thread_idx of int
    | Workgroup_idx of int
  (* This mirrors Backend_intf.special_kind previously used from Nx ecosystem *)
end

type any_dtype = Dtype.any
(** Existential wrapper for Dtype.t, allowing heterogeneous collections. *)

let any_dtype_to_string = Dtype.any_to_string

type var_metadata = {
  dtype : any_dtype; (* Uses the self-contained Dtype.any *)
  shape : int array; (* The logical shape of the tensor/view *)
}

(** Specifies the kind of reduction operation. *)
type reduce_op_kind_t = Reduce_Sum | Reduce_Max
(* Add others like Reduce_Min, Reduce_Prod as needed *)

(** Defines the nodes in the high-level Intermediate Representation graph. The
    GADT parameters ('a_elt, 'b_layout_phantom) refer to the OCaml element type
    and layout phantom of the output variable produced by this node. *)
type ('a_elt, 'b_layout_phantom) node_t =
  | Buffer : {
      dtype : ('a_elt, 'b_layout_phantom) Dtype.t;
      size_in_elements : int;
      out_var : Var.t;
    }
      -> ('a_elt, 'b_layout_phantom) node_t
  | Placeholder : {
      out_var : Var.t;
      dtype : ('a_elt, 'b_layout_phantom) Dtype.t;
      shape : int array;
    }
      -> ('a_elt, 'b_layout_phantom) node_t
  | Const_Scalar : {
      value : 'a_elt; (* Type matches the 'a_elt from the GADT parameter *)
      dtype : ('a_elt, 'b_layout_phantom) Dtype.t;
      out_var : Var.t;
    }
      -> ('a_elt, 'b_layout_phantom) node_t
  | Add : {
      in_a_var : Var.t;
      in_b_var : Var.t;
      out_var : Var.t;
      dtype : ('a_elt, 'b_layout_phantom) Dtype.t;
    }
      -> ('a_elt, 'b_layout_phantom) node_t
  | Mul : {
      in_a_var : Var.t;
      in_b_var : Var.t;
      out_var : Var.t;
      dtype : ('a_elt, 'b_layout_phantom) Dtype.t;
    }
      -> ('a_elt, 'b_layout_phantom) node_t
  | Reduce_Axis : {
      in_var : Var.t;
      reduce_op_kind : reduce_op_kind_t;
      axes : int array;
      out_var : Var.t;
      dtype : ('a_elt, 'b_layout_phantom) Dtype.t;
    }
      -> ('a_elt, 'b_layout_phantom) node_t
  | Expand : {
      in_var : Var.t;
      new_target_shape : int array;
      out_var : Var.t;
      dtype : ('a_elt, 'b_layout_phantom) Dtype.t;
    }
      -> ('a_elt, 'b_layout_phantom) node_t
  | Reshape : {
      in_var : Var.t;
      new_shape : int array;
      out_var : Var.t;
      dtype : ('a_elt, 'b_layout_phantom) Dtype.t;
    }
      -> ('a_elt, 'b_layout_phantom) node_t
  | Permute : {
      in_var : Var.t;
      axes_permutation : int array;
      out_var : Var.t;
      dtype : ('a_elt, 'b_layout_phantom) Dtype.t;
    }
      -> ('a_elt, 'b_layout_phantom) node_t

(** Existential wrapper for any IR node. *)
type any_node = Any_Node : ('a, 'b) node_t -> any_node [@@unboxed]

type graph_t = {
  nodes : any_node list;
  vars_metadata : (Var.t, var_metadata) Hashtbl.t;
  input_vars : Var.t list;
  output_vars : Var.t list;
}

module Lowered = struct
  (** Specifies the type of scalar Arithmetic Logic Unit operation. *)
  type scalar_alu_op_type =
    | Scalar_Add
    | Scalar_Mul
    | Scalar_Max
    | Scalar_CmpLt
  (* Add others as needed *)

  (** Defines the low-level, imperative UOp-like instructions. GADT parameters
      ('a_elt, 'b_layout_phantom) refer to the output of the instruction, if it
      produces a typed scalar. For effectful ops like Store, it's Dtype.Unit. *)
  type ('a_elt, 'b_layout_phantom) instruction_t =
    | LI_Buffer : {
        dtype : ('a_elt, 'b_layout_phantom) Dtype.t;
        size_in_elements : int;
        out_var : Var.t;
      }
        -> ('a_elt, 'b_layout_phantom) instruction_t
    | LI_Const_Scalar : {
        value : 'a_elt;
        dtype : ('a_elt, 'b_layout_phantom) Dtype.t;
        out_var : Var.t;
      }
        -> ('a_elt, 'b_layout_phantom) instruction_t
    | LI_Range : {
        name_hint : string;
        upper_bound_exclusive : Var.t;
        out_var : Var.t; (* Loop index variable, always an integer *)
      }
        -> (Dtype.int_elt, Dtype._x_layout_phantom) instruction_t
    | LI_Special_Index : {
        name_hint : string;
        kind : Special_index_kind.t;
        out_var : Var.t; (* Special index variable, always an integer *)
      }
        -> (Dtype.int_elt, Dtype._x_layout_phantom) instruction_t
    | LI_Load : {
        buffer_source_var : Var.t;
        indices_vars : Var.t list;
        valid_mask_var : Var.t option;
        out_var : Var.t; (* Variable for the loaded scalar value *)
        dtype : ('a_elt, 'b_layout_phantom) Dtype.t;
      }
        -> ('a_elt, 'b_layout_phantom) instruction_t
    | LI_Store : {
        buffer_target_var : Var.t;
        indices_vars : Var.t list;
        scalar_value_to_store_var : Var.t;
        valid_mask_var : Var.t option;
      }
        (* No output var, this is an effect. GADT params reflect this. *)
        -> (Dtype.unit_elt, Dtype.unit_layout_phantom) instruction_t
    | LI_Scalar_ALU : {
        op_type : scalar_alu_op_type;
        inputs_vars : Var.t list;
        out_var : Var.t; (* Variable for the scalar result *)
        dtype : ('a_elt, 'b_layout_phantom) Dtype.t;
      }
        -> ('a_elt, 'b_layout_phantom) instruction_t

  (** Existential wrapper for any Lowered IR instruction. *)
  type any_instruction =
    | Any_Instruction : ('a, 'b) instruction_t -> any_instruction
  [@@unboxed]

  type graph_t = {
    instructions : any_instruction list;
    vars_metadata : (Var.t, var_metadata) Hashtbl.t;
    kernel_input_vars : Var.t list;
    kernel_output_vars : Var.t list;
  }
end
