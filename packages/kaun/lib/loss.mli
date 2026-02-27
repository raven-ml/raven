(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Loss functions.

    Losses are differentiable through Rune's autodiff and return scalar means.
    [Invalid_argument] messages are prefixed with [Loss.<function>:]. *)

(** {1:classification Classification} *)

val cross_entropy : (float, 'a) Nx.t -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [cross_entropy logits one_hot_labels] is softmax cross-entropy.

    [logits] has shape [[...; num_classes]] and must be rank >= 1.
    [one_hot_labels] must have the same shape.

    Uses the log-sum-exp trick for numerical stability.

    Raises [Invalid_argument] if ranks or shapes differ, or if [num_classes] is
    not positive. *)

val cross_entropy_sparse : (float, 'a) Nx.t -> ('c, 'd) Nx.t -> (float, 'a) Nx.t
(** [cross_entropy_sparse logits class_indices] is {!cross_entropy} with integer
    labels.

    [class_indices] has shape [[...]] and must match [logits] without the last
    dimension. The class dimension is [logits]' last axis.

    Raises [Invalid_argument] if labels are non-integer, ranks mismatch,
    non-class dimensions differ, or the class dimension is non-positive. *)

val binary_cross_entropy :
  (float, 'a) Nx.t -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [binary_cross_entropy logits labels] is sigmoid binary cross-entropy.

    [logits] are raw (not sigmoid-normalized). [labels] are typically in
    \[[0];[1]\]. Uses log-sigmoid for numerical stability.

    Raises [Invalid_argument] if [logits] and [labels] shapes differ. *)

(** {1:regression Regression} *)

val mse : ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [mse predictions targets] is [mean ((predictions - targets)^2)].

    Shape compatibility follows Nx broadcasting semantics. *)

val mae : ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [mae predictions targets] is [mean (abs (predictions - targets))].

    Shape compatibility follows Nx broadcasting semantics. *)
