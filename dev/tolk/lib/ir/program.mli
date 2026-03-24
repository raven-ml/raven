(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Render-ready SSA IR.

    [Program] is the linear backend stage of [ir_next]. Values are array-backed
    sequences of instructions with stable ids and backward-only references.

    - build programs with {!create}, {!emit}, and {!finish};
    - inspect with {!view}, {!dtype}, {!sort}, {!children};
    - validate with {!validate};
    - rewrite with {!map_children}, {!map_alu}, and {!rebuild}. *)

(** {1:types Types} *)

type t
(** A linear SSA program. *)

type id = int
(** Instruction id. An index into the program array. *)

type builder
(** Mutable program builder. *)

type sort = Value | Pointer | Index | Effect
(** Coarse instruction role. *)

type view =
  | Param of { idx : int; dtype : Dtype.ptr }
      (** Global buffer parameter at index [idx]. *)
  | Param_image of { idx : int; dtype : Dtype.ptr; width : int; height : int }
      (** Image buffer parameter with pixel dimensions. *)
  | Define_local of { size : int; dtype : Dtype.ptr }
      (** Local (workgroup-shared) memory buffer of [size] elements. *)
  | Define_reg of { size : int; dtype : Dtype.ptr }
      (** Register-backed buffer of [size] elements. *)
  | Define_var of { name : string; lo : int; hi : int; dtype : Dtype.t }
      (** Scalar loop or index variable bounded by \[[lo];[hi]\]. *)
  | Const of { value : Const.t; dtype : Dtype.t }
      (** Compile-time constant. *)
  | Index of { ptr : id; idxs : id list; gate : id option; dtype : Dtype.ptr }
      (** Indexes into [ptr] with per-dimension [idxs] and optional [gate]. *)
  | Load of { src : id; alt : id option; dtype : Dtype.t }
      (** Loads from pointer [src]. [alt] is used when gated. *)
  | After of { src : id; deps : id list; dtype : Dtype.t }
      (** Sequences [src] after [deps]. *)
  | Store of { dst : id; value : id }
      (** Stores [value] through pointer [dst]. *)
  | Unary of { op : Op.unary; src : id; dtype : Dtype.t }
      (** Unary arithmetic or transcendental. *)
  | Binary of { op : Op.binary; lhs : id; rhs : id; dtype : Dtype.t }
      (** Binary arithmetic, logic, or comparison. *)
  | Ternary of { op : Op.ternary; a : id; b : id; c : id; dtype : Dtype.t }
      (** Ternary operation ([Where] or [Mulacc]). *)
  | Cast of { src : id; dtype : Dtype.t }
      (** Type cast. *)
  | Bitcast of { src : id; dtype : Dtype.t }
      (** Bit-preserving reinterpretation. *)
  | Vectorize of { srcs : id list; dtype : Dtype.t }
      (** Packs scalar [srcs] into a vector. *)
  | Gep of { src : id; idxs : int list; dtype : Dtype.t }
      (** Extracts elements at [idxs] from a vector. When [idxs] has one
          element, the result is scalar. When [idxs] has multiple elements,
          the result is a vector of the extracted elements. *)
  | Range of { size : id; dtype : Dtype.t; axis : int; sub : int list; kind : Axis_kind.t }
      (** Loop variable over \[[0];[size-1]\] on [axis]. *)
  | End_range of { dep : id; range : id }
      (** Closes the loop opened by [range]. [dep] is the last value produced
          inside the loop body, ensuring the body completes before the loop
          closes. *)
  | If of { cond : id; idx_for_dedup : id }
      (** Conditional branch on [cond]. *)
  | Endif of { if_ : id }
      (** Closes the conditional opened by [if_]. *)
  | Barrier  (** Workgroup barrier. *)
  | Special of { dim : Special_dim.t; size : id; dtype : Dtype.t }
      (** Backend-provided hardware index. *)
  | Wmma of {
      name : string;
      a : id;
      b : id;
      c : id;
      dtype : Dtype.t;
      dims : int * int * int;
      dtype_in : Dtype.scalar;
      dtype_out : Dtype.scalar;
      device : string;
      threads : int;
      upcast_axes : (int * int) list * (int * int) list * (int * int) list;
      reduce_axes : int list;
    }  (** Tensor-core matrix multiply-accumulate primitive. *)
  | Custom of { fmt : string; args : id list }
      (** Backend-specific effect or statement. *)
  | Custom_inline of { fmt : string; args : id list; dtype : Dtype.t }
      (** Backend-specific inline value expression. *)
(** Read-only instruction view. Pattern-match via {!view}. *)

(** {1:building Building} *)

val create : unit -> builder
(** [create ()] is an empty program builder. *)

val emit : builder -> view -> id
(** [emit b v] appends [v] to [b] and returns its id. *)

val finish : builder -> t
(** [finish b] is the program built so far. *)

(** {1:inspection Inspecting} *)

val view : t -> id -> view
(** [view t id] is the instruction at [id]. *)

val length : t -> int
(** [length t] is the number of instructions in [t]. *)

val dtype : t -> id -> Dtype.t option
(** [dtype t id] is the value dtype of [id], if any. Effect instructions
    return [None]. *)

val sort : t -> id -> sort
(** [sort t id] is the coarse role of [id]. *)

val children : t -> id -> id list
(** [children t id] are the direct input ids of instruction [id]. *)

val iteri : (id -> view -> unit) -> t -> unit
(** [iteri f t] calls [f id view] for each instruction in program order. *)

(** {1:predicates Predicates} *)

val is_alu : view -> bool
(** [is_alu v] is [true] iff [v] is {!Unary}, {!Binary}, or {!Ternary}. *)

val dtype_of_view : view -> Dtype.t option
(** [dtype_of_view v] is the result dtype of [v], if any. Effect views
    return [None]. For pointer views, returns the base type. *)

val index_gate : t -> id -> id option
(** [index_gate t id] walks through {!Cast}, {!Bitcast}, and {!After} to
    find the underlying {!Index} gate, if any. *)

(** {1:validation Validation} *)

val validate : t -> unit
(** [validate t] checks program invariants.

    Raises [Failure] on the first violation. *)

(** {1:rewriting Rewriting} *)

val map_children : (id -> id) -> view -> view
(** [map_children f v] rebuilds [v] with [f] applied to every child ref.
    Non-reference fields (dtype, constants, options) are preserved. *)

val map_alu : map_ref:(id -> id) -> dtype:Dtype.t -> view -> view
(** [map_alu ~map_ref ~dtype v] remaps child refs and replaces the dtype
    of an ALU view.

    Raises [Invalid_argument] if [v] is not {!Unary}, {!Binary}, or
    {!Ternary}. *)

val rebuild :
  (emit:(view -> id) -> map_ref:(id -> id) -> view -> id option) -> t -> t
(** [rebuild f t] constructs a new program by iterating forward through [t].

    For each instruction, [f ~emit ~map_ref view] may emit replacement
    instructions via [emit] and return the new id, or return [None] to keep
    the instruction with refs automatically remapped via [map_ref]. *)

(** {1:formatting Formatting} *)

val pp_view : Format.formatter -> view -> unit
(** [pp_view] formats one instruction view. *)

val pp : Format.formatter -> t -> unit
(** [pp] formats a whole program. *)
