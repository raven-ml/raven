(* ir.ml *)

(* ────────── Scalars & element types ────────── *)

module Dtype = struct
  type _ t =
    | Float32 : float t
    | Int32 : int t
    | Bool : bool t
    | Uint8 : int t (* OCaml int carries the byte *)
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

(* ────────── SSA variables ────────── *)

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

(* ────────── Misc enums ────────── *)

module Special_index_kind = struct
  type t =
    | Global_task_idx of int (* 0=x,1=y,2=z *)
    | Local_thread_idx of int
    | Workgroup_idx of int
end

type var_metadata = { dtype : Dtype.any; shape : int array }

(* ────────── High-level graph IR ────────── *)

type binop_kind = Add | Mul | Sub | Div | Max | Min

type _ node_t =
  | Buffer : {
      dtype : 'a Dtype.t;
      size_in_elements : int;
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
  | Binop : {
      op : binop_kind;
      a_var : Var.t;
      b_var : Var.t;
      out_var : Var.t;
      dtype : 'a Dtype.t;
    }
      -> 'a node_t
  | Reduce_Axis : {
      in_var : Var.t;
      reduce_op_kind : reduce_op_kind_t;
      axes : int array;
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

and reduce_op_kind_t = Reduce_Sum | Reduce_Max

type any_node = Any_Node : 'a node_t -> any_node [@@unboxed]

type graph_t = {
  nodes : any_node list;
  vars_metadata : (Var.t, var_metadata) Hashtbl.t;
  input_vars : Var.t list;
  output_vars : Var.t list;
}

(* ────────── Low-level / lowered IR ────────── *)

module Lowered = struct
  type scalar_alu_op = Bin of binop_kind | CmpLt

  type instruction =
    | L_Buffer of { dtype : Dtype.any; size : int; out : Var.t }
    | L_Const of { dtype : Dtype.any; value : string; out : Var.t }
    | L_Range of { idx : Var.t; upper : Var.t } (* for-loop header *)
    | L_EndRange
    | L_SpecialIndex of { dst : Var.t; kind : Special_index_kind.t }
    | L_Load of {
        dst : Var.t;
        buf : Var.t;
        idxs : Var.t list;
        mask : Var.t option;
        dtype : Dtype.any;
      }
    | L_Store of {
        buf : Var.t;
        idxs : Var.t list;
        src : Var.t;
        mask : Var.t option;
      }
    | L_ALU of {
        dst : Var.t;
        op : scalar_alu_op;
        args : Var.t list;
        dtype : Dtype.any;
      }

  type graph_t = {
    instructions : instruction list;
    vars_metadata : (Var.t, var_metadata) Hashtbl.t;
    kernel_input_vars : Var.t list;
    kernel_output_vars : Var.t list;
  }
end
