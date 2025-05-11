(* lowered_ir.ml *)
open Nx_core (* For Dtype *)
open Ir

(** Specifies the type of scalar Arithmetic Logic Unit operation. *)
type scalar_alu_op_type =
  | Scalar_Add
  | Scalar_Mul
  | Scalar_Max (* Example for lowering Reduce_Max *)
  | Scalar_CmpLt
(* Example for comparisons *)
(* Add others: Scalar_Sub, Scalar_Div, Scalar_Sin, Scalar_Log, Scalar_Exp, Scalar_Sqrt etc. *)
(* Note: These operate on SCALAR variables. *)

(** Defines the low-level, imperative UOp-like instructions. These are the
    building blocks for kernels. *)
type _ instruction_t =
  (* Memory Allocation (usually for local/scratchpad, main buffers are from IR
     graph) *)
  | LI_Buffer : {
      (* Similar to IR.Buffer, but in this context, it might also represent
         temporary scratch buffers needed during computation if not handled by
         registers. *)
      dtype : ('a, 'b) Dtype.t;
      size_in_elements : int;
      out_var : Var.t; (* The variable (SID) identifying this buffer. *)
    }
      -> ('a, 'b) Nx_rune.t instruction_t
  (* GADT helps type out_var conceptually *)
  (* Constants *)
  | LI_Const_Scalar : {
      (* A literal scalar constant. *)
      value : 'a;
      dtype : ('a, 'b) Dtype.t;
      out_var : Var.t; (* Variable (SID) for this scalar constant. *)
    }
      -> ('a, 'b) Nx_rune.t instruction_t
  (* Control Flow Primitives for Loops *)
  | LI_Range : {
      (* Represents a loop iterator. *)
      name_hint : string; (* e.g., "idx0_loop", "reduce_axis1_loop" *)
      upper_bound_exclusive : Var.t;
      (* Variable holding the loop's upper bound (scalar int). Can be from
             var_metadata.shape or a const. *)
      (* OR: upper_bound_exclusive : int; if known at lowering time *)
      out_var : Var.t; (* Variable (SID) for the scalar loop index. *)
    }
      -> (int, Dtype.int_elt) Nx_rune.t instruction_t (* Index is an int *)
  (* Hardware-Specific Indices (if applicable at this level) *)
  | LI_Special_Index : {
      (* Represents a special hardware index like thread ID or block ID. *)
      name_hint : string;
      kind : Backend_intf.special_kind; (* e.g., Global_task_idx 0 *)
      out_var : Var.t; (* Variable (SID) for the scalar special index. *)
    }
      -> (int, Dtype.int_elt) Nx_rune.t instruction_t
  (* Memory Access *)
  | LI_Load : {
      (* Loads a SCALAR value from a buffer. *)
      buffer_source_var : Var.t;
          (* Variable of the source buffer (from IR.Buffer or
             IR.Placeholder). *)
      indices_vars : Var.t list;
          (* List of SCALAR variables representing the multi-dimensional
             index. *)
      valid_mask_var : Var.t option;
          (* Optional SCALAR boolean variable for masked loads. *)
      out_var : Var.t; (* Variable (SID) for the loaded SCALAR value. *)
      dtype : ('a, 'b) Dtype.t; (* Dtype of the loaded value. *)
    }
      -> ('a, 'b) Nx_rune.t instruction_t
  | LI_Store : {
      (* Stores a SCALAR value into a buffer. *)
      buffer_target_var : Var.t; (* Variable of the target buffer. *)
      indices_vars : Var.t list;
          (* List of SCALAR variables for the multi-dimensional index. *)
      scalar_value_to_store_var : Var.t;
          (* Variable of the SCALAR value to store. *)
      valid_mask_var : Var.t option;
          (* Optional SCALAR boolean variable for masked stores. *)
    }
      -> unit instruction_t (* Store is an effect, no direct tensor output. *)
  (* Scalar Computation *)
  | LI_Scalar_ALU : {
      (* Performs an ALU operation on SCALAR input variables. *)
      op_type : scalar_alu_op_type;
      inputs_vars : Var.t list; (* List of SCALAR input variables. *)
      out_var : Var.t; (* Variable (SID) for the SCALAR result. *)
      dtype : ('a, 'b) Dtype.t; (* Dtype of the result. *)
    }
      -> ('a, 'b) Nx_rune.t instruction_t

(* Other potential low-level ops: | LI_Barrier: For synchronization in parallel
   execution. | LI_Contiguous: If an explicit data copy/reordering is required
   to make data contiguous. This might be a "macro" op at this level too,
   expanding to loads/stores. *)

(** Existential wrapper for any Lowered IR instruction. *)
type any_instruction = Any_Instruction : 'a instruction_t -> any_instruction

type graph_t = {
  instructions : any_instruction list;
      (* Sequence of instructions for the kernel. *)
  vars_metadata : (Var.t, Ir.var_metadata) Hashtbl.t;
  (* Metadata for all variables, potentially shared/derived from Ir.graph_t. *)
  (* Kernel signature information: *)
  kernel_input_vars : Var.t list;
      (* Variables (buffers) passed as arguments to the kernel. *)
  kernel_output_vars : Var.t list;
      (* Variables (buffers) written by the kernel as results. *)
      (* May also include: local memory requirements, loop structures, etc. *)
}
(** The lowered IR graph, representing a sequence of imperative instructions.
    This is closer to a "linearized" kernel. *)
