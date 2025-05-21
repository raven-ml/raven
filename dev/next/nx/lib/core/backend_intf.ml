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

  (* Element-wise Binary Ops *)
  (* These ops assume inputs have been broadcast to the same shape and cast to
     the same compatible dtype by the frontend. The output tensor will also have
     this common dtype. *)

  val op_add :
    context ->
    ('a, 'b) t (* op1 *) ->
    ('a, 'b) t (* op2 *) ->
    ('a, 'b) t (* result *)
  (** [op_add ctx op1 op2] Corresponds to the [ADD] UOp. Performs element-wise
      addition. Frontend ensures broadcast and type compatibility. *)

  val op_mul : context -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [op_mul ctx op1 op2] Corresponds to the [MUL] UOp. Performs element-wise
      multiplication. Frontend ensures broadcast and type compatibility. *)

  val op_idiv : context -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [op_idiv ctx op1 op2] Corresponds to the [IDIV] UOp (integer division,
      truncates). Frontend ensures broadcast and type compatibility (typically
      integer types). *)

  val op_fdiv : context -> (float, 'b) t -> (float, 'b) t -> (float, 'b) t
  (** [op_fdiv ctx op1 op2] Corresponds to the [FDIV] UOp (float division).
      Frontend ensures broadcast and type compatibility (float types). *)

  val op_max : context -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [op_max ctx op1 op2] Corresponds to the [MAX] UOp (element-wise). Frontend
      ensures broadcast and type compatibility. *)

  val op_mod : context -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [op_mod ctx op1 op2] Corresponds to the [MOD] UOp. Frontend ensures
      broadcast and type compatibility. *)

  val op_pow : context -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [op_pow ctx base exponent] Corresponds to the [POW] UOp. Frontend ensures
      broadcast and type compatibility (typically float base). *)

  val op_cmplt : context -> ('a, 'b) t -> ('a, 'b) t -> (int, Dtype.uint8_elt) t
  (** [op_cmplt ctx op1 op2] Corresponds to the [CMPLT] UOp. Returns a
      boolean-like tensor (0 or 1 as uint8). Frontend ensures broadcast and type
      compatibility for inputs. *)

  val op_cmpne : context -> ('a, 'b) t -> ('a, 'b) t -> (int, Dtype.uint8_elt) t
  (** [op_cmpne ctx op1 op2] Corresponds to the [CMPNE] UOp. Returns a
      boolean-like tensor (0 or 1 as uint8). Frontend ensures broadcast and type
      compatibility for inputs. *)

  val op_xor : context -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [op_xor ctx op1 op2] Corresponds to the [XOR] UOp. Frontend ensures
      broadcast and type compatibility (typically integer or boolean-like
      types). *)

  val op_or : context -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [op_or ctx op1 op2] Corresponds to the [OR] UOp. Frontend ensures
      broadcast and type compatibility. *)

  val op_and : context -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [op_and ctx op1 op2] Corresponds to the [AND] UOp. Frontend ensures
      broadcast and type compatibility. *)

  (* Element-wise Unary Ops *)

  val op_neg : context -> ('a, 'b) t -> ('a, 'b) t
  (** [op_neg ctx t] Corresponds to the [NEG] UOp. For bools, it's logical not.
      Output dtype is same as input. *)

  val op_log2 : context -> (float, 'b) t -> (float, 'b) t
  (** [op_log2 ctx t] Corresponds to the [LOG2] UOp. Output is float32. *)

  val op_exp2 : context -> (float, 'b) t -> (float, 'b) t
  (** [op_exp2 ctx t] Corresponds to the [EXP2] UOp. Output is float32. *)

  val op_sin : context -> (float, 'b) t -> (float, 'b) t
  (** [op_sin ctx t] Corresponds to the [SIN] UOp. Output is float32. *)

  val op_sqrt : context -> (float, 'b) t -> (float, 'b) t
  (** [op_sqrt ctx t] Corresponds to the [SQRT] UOp. Output is float32. *)

  val op_recip : context -> (float, 'b) t -> (float, 'b) t
  (** [op_recip ctx t] Corresponds to the [RECIP] UOp. Output is float32. *)

  (* Ternary Op *)

  val op_where :
    context ->
    (int, Dtype.uint8_elt) t (* condition (0 or 1) *) ->
    ('a, 'b) t (* if_true *) ->
    ('a, 'b) t (* if_false *) ->
    ('a, 'b) t (* result *)
  (** [op_where ctx cond if_true if_false] Corresponds to the [WHERE] UOp.
      Frontend ensures broadcast and type compatibility for [if_true] and
      [if_false], and that [cond] is boolean-like. *)

  (* Reduction Ops *)

  val op_reduce_sum :
    context ->
    axes:int array (* axes to reduce along *) ->
    keepdims:bool ->
    ('a, 'b) t (* input_tensor *) ->
    ('a, 'b) t (* result_tensor *)
  (** [op_reduce_sum ctx t ~axis ~keepdims] Corresponds to the [SUM] UOp
      (reduction). Sums elements of [t] along the specified [axis]. If
      [keepdims] is true, the reduced axes are kept with dimension 1. Otherwise,
      they are removed. Returns a *new* tensor with the summed values. The dtype
      remains the same. *)

  val op_reduce_max :
    context -> axes:int array -> keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
  (** [op_reduce_max ctx t ~axes ~keepdims] Corresponds to [REDUCE] UOp with
      [MAX]. Dtype remains the same. *)

  val op_reduce_prod :
    context -> axes:int array -> keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
  (** [op_reduce_prod ctx t ~axes ~keepdims] Corresponds to [REDUCE] UOp with
      [MUL]. Dtype remains the same. *)

  (* Movement Ops - these primarily update view metadata *)

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

  val op_permute : context -> ('a, 'b) t -> int array (* axes *) -> ('a, 'b) t
  (** [op_permute ctx t axes] Corresponds to the [PERMUTE] UOp. Reorders the
      dimensions of [t] according to [axes]. Updates view metadata. Returns a
      *new* tensor handle sharing the buffer. *)

  val op_pad :
    context ->
    ('a, 'b) t ->
    (int * int) array (* padding per axis: (pad_before, pad_after) *) ->
    'a (* fill_value *) ->
    ('a, 'b) t
  (** [op_pad ctx t padding_config fill_value] Corresponds to the [PAD] UOp.
      Pads tensor [t] according to [padding_config] using [fill_value]. Updates
      view metadata (offset, mask). May require new buffer if padding cannot be
      represented by view manipulation alone for JIT, or performs copy for
      eager. For now, assumes view manipulation. Returns a *new* tensor handle.
  *)

  val op_shrink :
    context ->
    ('a, 'b) t ->
    (int * int) array (* shrink_args per axis: (new_start, new_end) *) ->
    ('a, 'b) t
  (** [op_shrink ctx t shrink_args] Corresponds to the [SHRINK] UOp. Extracts a
      sub-tensor (slice) from [t]. Updates view metadata. Returns a *new* tensor
      handle sharing the buffer. *)

  val op_flip :
    context -> ('a, 'b) t -> bool array (* flip_axes_booleans *) -> ('a, 'b) t
  (** [op_flip ctx t flip_axes_booleans] Corresponds to the [FLIP] UOp. Flips
      dimensions of [t] where [flip_axes_booleans] is true. Updates view
      metadata. Returns a *new* tensor handle sharing the buffer. *)

  val op_cat : context -> ('a, 'b) t list -> int (* axis *) -> ('a, 'b) t
  (** [op_cat ctx tensors axis] Corresponds to the [CAT] UOp. Concatenates a
      list of [tensors] along the specified [axis]. All tensors must have the
      same shape except in the dimension [axis], and must have the same dtype.
      Returns a *new* tensor. *)

  (* Other Ops *)

  val op_cast : context -> ('a, 'b) t -> ('c, 'd) Dtype.t -> ('c, 'd) t
  (** [op_cast ctx t target_dtype] Corresponds to the [CAST] UOp. Casts elements
      of [t] to [target_dtype]. This typically involves a data copy into a new
      buffer. Returns a *new* tensor. *)

  val op_contiguous : context -> ('a, 'b) t -> ('a, 'b) t
  (** [op_contiguous ctx t] Corresponds to the [CONTIGUOUS] UOp. Ensures the
      output tensor [t'] has a C-contiguous memory layout. If [t] is already
      contiguous and has a standard view, [t'] might be [t]. Otherwise, [t'] is
      a new tensor with data copied from [t]. *)

  val op_copy : context -> ('a, 'b) t -> ('a, 'b) t
  (** [op_copy ctx t] Corresponds to the [COPY] UOp. Creates a new tensor with
      its own buffer, containing a copy of the data from tensor [t]. This is
      used for cloning or ensuring a tensor has its own data. *)

  val op_threefry :
    context ->
    (int32, Dtype.int32_elt) t (* data_tensor, e.g. counts0 *) ->
    (int32, Dtype.int32_elt) t (* seed_tensor, e.g. key0 *) ->
    (int32, Dtype.int32_elt) t (* result_tensor, random bits *)
  (** [op_threefry ctx data_tensor seed_tensor] Corresponds to the [THREEFRY]
      UOp, a pseudo-random number generator. Assumes inputs are uint32-like
      (represented as int32) and produces uint32-like output. Frontend ensures
      broadcast and type compatibility. *)
end
