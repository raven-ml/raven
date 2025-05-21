(* backend_intf.ml *)

(** Defines the low-level operations that a backend must implement. These
    operations closely mirror the Universal Operations (UOps) used in tinygrad.
    Each backend provides a concrete implementation of this signature, enabling
    either eager execution or JIT compilation via effects.

    {4 IMPORTANT NOTE:}

    A module implementing this signature [S] can serve two primary purposes
    without changing the interface itself.

    - (1) **Eager Execution:** Functions like [op_add], [op_load], etc., can be
      implemented to perform the computation immediately upon being called,
      returning concrete tensor values. This allows for direct, interactive use
      similar to standard NumPy or PyTorch eager mode.
    - (2) **JIT Compilation (via Effects):** Alternatively, functions can be
      implemented to raise specific effects instead of performing computations.
      An effect handler can then capture these effects (e.g.,
      [Effect.Add (t1_id, t2_id)], [Effect.Load (buf_id, indices)]) to build an
      intermediate representation (like a computation graph or UOp list). This
      graph can then be optimized and compiled into efficient kernel code for
      later execution.

    his unified interface allows the frontend library to switch between
    different execution strategies (eager vs. JIT) by simply changing the
    backend module. *)
module type S = sig
  type ('a, 'b) t
  (** The abstract type for a tensor managed by this specific backend.

      - ['a] is the OCaml type of the elements (e.g., [float], [int]).
      - ['b] is a phantom type encoding the specific Dtype kind (e.g.,
        [Dtype.float32_elt]). *)

  type context
  (** The type for the backend's execution context. This may hold
      backend-specific state, such as memory pools, command queues, etc. *)

  (* lenses *)

  val view : ('a, 'b) t -> View.t
  (** [view tensor] Returns the logical view ([View.view]) associated with the
      tensor [t]. *)

  val dtype : ('a, 'b) t -> ('a, 'b) Dtype.t
  (** [dtype tensor] Returns the data type ([Dtype.t]) of the elements within
      the tensor [t]. *)

  (* context *)

  val create_context : unit -> context
  (** [create_context ()] Creates a new backend-specific execution context. *)

  (* ops - These are the equivalent of tinygrad's UOp. Do not add anything here
     that's not a UOp in tinygrad. *)

  val op_buffer :
    context -> ('a, 'b) Dtype.t -> int (* size_in_elements *) -> ('a, 'b) t
  (** [op_buffer ctx dtype size_in_elements] Corresponds to the [BUFFER] UOp.
      Creates an uninitialized buffer managed by the backend [ctx] capable of
      holding [size_in_elements] elements of type [dtype]. Returns a new tensor
      handle representing this buffer. The initial view typically represents a
      flat, contiguous array of size [size_in_elements]. *)

  val op_const_scalar : context -> 'a -> ('a, 'b) Dtype.t -> ('a, 'b) t
  (** [op_const_scalar ctx value dtype] Corresponds to the [CONST] UOp for
      scalars. Creates a tensor representing a single scalar constant [value] of
      type [dtype]. The resulting tensor typically has a shape like [()] or
      [(1,)]. Broadcasting this value requires subsequent [op_expand] calls
      coordinated by the frontend. *)

  val op_add :
    context ->
    ('a, 'b) t (* op1 *) ->
    ('a, 'b) t (* op2 *) ->
    ('a, 'b) t (* result *)
  (** [op_add ctx op1 op2] Corresponds to the [ADD] UOp. Performs element-wise
      addition between [op1] and [op2]. The frontend is responsible for ensuring
      that the views of [op1] and [op2] have been prepared (via [op_expand],
      [op_reshape]) to be broadcast-compatible. The backend implementation
      iterates based on the (broadcasted) output shape, using the views of [op1]
      and [op2] to perform the necessary [op_load]s, scalar addition, and
      [op_store]s (or equivalent fused operations). Returns a *new* tensor
      containing the result, with a shape matching the broadcasted shape. Type
      promotion follows standard rules (e.g., int + float -> float), and the
      output tensor type ['e, 'f] reflects this. *)

  val op_mul : context -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [op_mul ctx op1 op2] Corresponds to the [MUL] UOp. Performs element-wise
      multiplication between [op1] and [op2]. The frontend is responsible for
      ensuring that the views of [op1] and [op2] have been prepared (via
      [op_expand], [op_reshape]) to be broadcast-compatible. The backend
      implementation iterates based on the (broadcasted) output shape, using the
      views of [op1] and [op2] to perform the necessary [op_load]s, scalar
      multiplication, and [op_store]s (or equivalent fused operations). Returns
      a *new* tensor containing the result, with a shape matching the
      broadcasted shape. Assumes [op1] and [op2] have the same dtype. *)

  val op_sum :
    context ->
    axes:int array (* axes to reduce along *) ->
    keepdims:bool ->
    ('a, 'b) t (* input_tensor *) ->
    ('a, 'b) t (* result_tensor *)
  (** [op_sum ctx t ~axis ~keepdims] Corresponds to the [SUM] UOp (reduction).
      Sums elements of [t] along the specified [axis]. If [keepdims] is true,
      the reduced axes are kept with dimension 1. Otherwise, they are removed.
      Returns a *new* tensor with the summed values. The dtype remains the same.
  *)

  val op_expand :
    context -> ('a, 'b) t -> int array (* new_target_shape *) -> ('a, 'b) t
  (** [op_expand ctx t new_target_shape] Corresponds to the [EXPAND] UOp.
      Logically expands dimensions of size 1 in tensor [t] to match
      [new_target_shape]. This operation primarily updates the tensor's
      associated view metadata (specifically, setting strides of expanded
      dimensions to 0). It does not move data. Returns a *new* tensor handle
      that shares the underlying buffer with [t] but has the modified view. *)

  val op_reshape :
    context -> ('a, 'b) t -> int array (* new_shape *) -> ('a, 'b) t
  (** [op_reshape ctx t new_shape] Corresponds to the [RESHAPE] UOp. Changes the
      logical shape of the tensor [t] to [new_shape]. The total number of
      elements must remain the same. This operation primarily updates the
      tensor's associated view metadata. It does not move data if the reshape is
      compatible with the existing layout. Returns a *new* tensor handle that
      shares the underlying buffer with [t] but has the modified view. *)
end
