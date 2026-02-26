(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Lazy, composable data pipelines for training.

    A {!type:t} is a resettable iterator over elements of type ['a]. Pipelines
    are built by composing constructors, transformers, and consumers.

    {[
      Data.of_array examples |> Data.shuffle key
      |> Data.map_batch 32 collate
      |> Data.iter train_step
    ]} *)

(** {1:types Types} *)

type 'a t
(** The type for lazy data pipelines producing elements of type ['a]. *)

(** {1:constructors Constructors} *)

val of_array : 'a array -> 'a t
(** [of_array a] is a pipeline yielding the elements of [a] in order. *)

val of_tensor : ('a, 'b) Rune.t -> ('a, 'b) Rune.t t
(** [of_tensor t] is a pipeline yielding slices along the first dimension of
    [t]. Each element has shape [t.shape[1:]]. *)

val of_tensors :
  ('a, 'b) Rune.t * ('c, 'd) Rune.t -> (('a, 'b) Rune.t * ('c, 'd) Rune.t) t
(** [of_tensors (x, y)] is a pipeline yielding paired slices along the first
    dimension of [x] and [y].

    Raises [Invalid_argument] if [x] and [y] have different first dimension
    sizes. *)

val of_fn : int -> (int -> 'a) -> 'a t
(** [of_fn n f] is a pipeline yielding [f 0], [f 1], ..., [f (n - 1)].

    Raises [Invalid_argument] if [n < 0]. *)

val repeat : int -> 'a -> 'a t
(** [repeat n v] is a pipeline that yields [v] exactly [n] times.

    Raises [Invalid_argument] if [n < 0]. *)

(** {1:transformers Transformers} *)

val map : ('a -> 'b) -> 'a t -> 'b t
(** [map f t] is a pipeline that applies [f] to each element of [t]. *)

val batch : ?drop_last:bool -> int -> 'a t -> 'a array t
(** [batch ?drop_last n t] is a pipeline yielding arrays of [n] consecutive
    elements from [t].

    [drop_last] defaults to [false]. When [true], the final batch is dropped if
    it has fewer than [n] elements.

    Raises [Invalid_argument] if [n <= 0]. *)

val map_batch : ?drop_last:bool -> int -> ('a array -> 'b) -> 'a t -> 'b t
(** [map_batch ?drop_last n f t] is [map f (batch ?drop_last n t)]. *)

val shuffle : Rune.Rng.key -> 'a t -> 'a t
(** [shuffle key t] is a pipeline that yields the elements of [t] in a random
    order determined by [key]. The permutation is computed once when the
    pipeline is created.

    Raises [Invalid_argument] if [t] has unknown length. *)

(** {1:consumers Consumers} *)

val iter : ('a -> unit) -> 'a t -> unit
(** [iter f t] applies [f] to each element of [t]. *)

val iteri : (int -> 'a -> unit) -> 'a t -> unit
(** [iteri f t] applies [f i x] to each element [x] of [t], where [i] is the
    0-based index. *)

val fold : ('acc -> 'a -> 'acc) -> 'acc -> 'a t -> 'acc
(** [fold f init t] folds [f] over the elements of [t]. *)

val to_array : 'a t -> 'a array
(** [to_array t] collects all elements of [t] into an array. *)

val to_seq : 'a t -> 'a Seq.t
(** [to_seq t] is a standard [Seq.t] view of [t]. Does not reset [t]. *)

(** {1:properties Properties} *)

val reset : 'a t -> unit
(** [reset t] resets [t] so that iteration starts from the beginning. *)

val length : 'a t -> int option
(** [length t] is the number of elements in [t], if known. *)

(** {1:utilities Utilities} *)

val stack_batch : ('a, 'b) Rune.t array -> ('a, 'b) Rune.t
(** [stack_batch tensors] stacks an array of tensors along a new first axis.
    Equivalent to [Rune.stack (Array.to_list tensors)]. *)

val prepare :
  ?shuffle:Rune.Rng.key ->
  batch_size:int ->
  ?drop_last:bool ->
  ('a, 'b) Rune.t * ('c, 'd) Rune.t ->
  (('a, 'b) Rune.t * ('c, 'd) Rune.t) t
(** [prepare ?shuffle ~batch_size (x, y)] is a pipeline that yields batched
    tensor pairs from [x] and [y].

    Each yielded pair has shape [[batch_size; ...]] along the first dimension.

    When [shuffle] is provided, elements are yielded in a random order
    determined by the key. [drop_last] defaults to [true].

    Raises [Invalid_argument] if [x] and [y] have different first dimension
    sizes, or if [batch_size <= 0]. *)
