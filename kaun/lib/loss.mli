(** Loss functions for neural network training.

    This module provides standard loss functions used in neural network
    training. All loss functions are numerically stable and differentiable
    through Rune's autodiff system.

    Loss functions measure the difference between predictions and targets,
    returning a scalar loss value that can be minimized during training. Most
    functions return the mean loss across all examples in the batch.

    {1 Classification Losses}

    For multi-class and binary classification tasks. These assume specific input
    formats (logits vs probabilities) and target representations (one-hot vs
    indices).

    {1 Regression Losses}

    For continuous value prediction tasks. These measure distance between
    predicted and target values using different metrics. *)

val softmax_cross_entropy :
  (float, 'a, 'b) Rune.t -> (float, 'a, 'b) Rune.t -> (float, 'a, 'b) Rune.t
(** [softmax_cross_entropy logits labels] computes cross-entropy loss between softmax-normalized logits and one-hot encoded labels.

    The function applies softmax to [logits] internally and computes the cross-entropy loss: -sum(labels * log(softmax(logits))) / batch_size.

    Numerically stable implementation using the log-sum-exp trick to prevent overflow.

    @param logits Raw model outputs of shape [batch_size; num_classes]. Not softmax-normalized.
    @param labels One-hot encoded ground truth labels of shape [batch_size; num_classes]. Each row should sum to 1.0.

    @return Scalar tensor containing the mean cross-entropy loss across the batch.

    {4 Example}

    Multi-class classification with 3 classes:
    {[
      let logits = Rune.create device Rune.float32 [|2; 3|] [|2.0; 1.0; 0.1; 0.5; 1.5; 2.1|] in
      let labels = Rune.create device Rune.float32 [|2; 3|] [|1.0; 0.0; 0.0; 0.0; 0.0; 1.0|] in
      let loss = Loss.softmax_cross_entropy logits labels
    ]}

    Mathematical formula: L = -1/N * sum_i sum_c y_{i,c} * log(softmax(x_{i,c}))
    where N is batch size, y is one-hot labels, x is logits. *)

val softmax_cross_entropy_with_indices :
  (float, 'a, 'b) Rune.t -> ('c, 'd, 'b) Rune.t -> (float, 'a, 'b) Rune.t
(** [softmax_cross_entropy_with_indices logits indices] computes cross-entropy
    loss using class indices instead of one-hot labels.

    Converts [indices] to one-hot encoding internally, then applies
    {!softmax_cross_entropy}. More memory-efficient than manually creating
    one-hot labels.

    @param logits Raw model outputs of shape [batch_size; num_classes].
    @param indices
      Class indices of shape [batch_size]. Values must be in range \[0,
      num_classes).

    @return
      Scalar tensor containing the mean cross-entropy loss across the batch.

    @raise Invalid_argument
      if any index is outside the valid range \[0, num_classes).

    {4 Example}

    Classification where first example belongs to class 0, second to class 2:
    {[
      let logits = Rune.create device Rune.float32 [|2; 3|] [|2.0; 1.0; 0.1; 0.5; 1.5; 2.1|] in
      let indices = Rune.create device Rune.int32 [|2|] [|0; 2|] in
      let loss = Loss.softmax_cross_entropy_with_indices logits indices
    ]} *)

val binary_cross_entropy :
  (float, 'a, 'b) Rune.t -> (float, 'a, 'b) Rune.t -> (float, 'a, 'b) Rune.t
(** [binary_cross_entropy predictions labels] computes binary cross-entropy loss
    between probability predictions and binary labels.

    Assumes [predictions] are already sigmoid-normalized probabilities in range
    [0, 1]. For raw logits, use {!sigmoid_binary_cross_entropy} instead.

    The loss is computed as: -mean(labels * log(predictions) + (1 - labels) *
    log(1 - predictions)).

    @param predictions
      Sigmoid-normalized probabilities of shape [batch_size; ...]. Values should
      be in range [0, 1].
    @param labels
      Binary ground truth labels of shape [batch_size; ...]. Values should be
      0.0 or 1.0.

    @return Scalar tensor containing the mean binary cross-entropy loss.

    {4 Example}

    Binary classification with sigmoid outputs:
    {[
      let predictions = Rune.create device Rune.float32 [|4; 1|] [|0.8; 0.3; 0.7; 0.1|] in
      let labels = Rune.create device Rune.float32 [|4; 1|] [|1.0; 0.0; 1.0; 0.0|] in
      let loss = Loss.binary_cross_entropy predictions labels
    ]}

    Mathematical formula: L = -1/N * sum_i (y_i * log(p_i) + (1 - y_i) * log(1 -
    p_i)) where N is batch size, y are labels, p are predictions. *)

val sigmoid_binary_cross_entropy :
  (float, 'a, 'b) Rune.t -> (float, 'a, 'b) Rune.t -> (float, 'a, 'b) Rune.t
(** [sigmoid_binary_cross_entropy logits labels] computes binary cross-entropy
    loss from raw logits.

    Applies sigmoid normalization to [logits] internally. More numerically
    stable than manually applying sigmoid then {!binary_cross_entropy}, as it
    uses log_sigmoid internally.

    Unlike {!binary_cross_entropy}, this returns loss per example without taking
    the mean, allowing for sample-weighted training.

    @param logits
      Raw model outputs of shape [batch_size; ...]. Can be any real values.
    @param labels
      Binary ground truth labels of shape [batch_size; ...]. Values should be
      0.0 or 1.0.

    @return Tensor of shape [batch_size; ...] containing loss per example.

    {4 Example}

    Binary classification with raw logits:
    {[
      let logits = Rune.create device Rune.float32 [|4; 1|] [|1.5; -0.8; 0.9; -2.1|] in
      let labels = Rune.create device Rune.float32 [|4; 1|] [|1.0; 0.0; 1.0; 0.0|] in
      let loss_per_example = Loss.sigmoid_binary_cross_entropy logits labels in
      let mean_loss = Rune.mean loss_per_example
    ]} *)

val mse : ('a, 'b, 'c) Rune.t -> ('a, 'b, 'c) Rune.t -> ('a, 'b, 'c) Rune.t
(** [mse predictions targets] computes mean squared error between predictions
    and targets.

    Suitable for regression tasks. Penalizes large errors more heavily than
    small errors due to squaring.

    @param predictions Model predictions of any shape.
    @param targets Ground truth targets of the same shape as [predictions].

    @return
      Scalar tensor containing the mean squared error: mean((predictions -
      targets)^2).

    {4 Example}

    Regression with continuous targets:
    {[
      let predictions = Rune.create device Rune.float32 [|3|] [|2.1; 0.8; 1.5|] in
      let targets = Rune.create device Rune.float32 [|3|] [|2.0; 1.0; 1.2|] in
      let loss = Loss.mse predictions targets
    ]}

    Mathematical formula: L = 1/N * sum_i (pred_i - target_i)^2 where N is the
    total number of elements. *)

val mae : ('a, 'b, 'c) Rune.t -> ('a, 'b, 'c) Rune.t -> ('a, 'b, 'c) Rune.t
(** [mae predictions targets] computes mean absolute error between predictions
    and targets.

    Suitable for regression tasks where you want equal penalty for all errors
    regardless of magnitude. Less sensitive to outliers than MSE.

    @param predictions Model predictions of any shape.
    @param targets Ground truth targets of the same shape as [predictions].

    @return
      Scalar tensor containing the mean absolute error: mean(|predictions -
      targets|).

    {4 Example}

    Regression with robust loss function:
    {[
      let predictions = Rune.create device Rune.float32 [|3|] [|2.1; 0.8; 1.5|] in
      let targets = Rune.create device Rune.float32 [|3|] [|2.0; 1.0; 1.2|] in
      let loss = Loss.mae predictions targets
    ]}

    Mathematical formula: L = 1/N * sum_i |pred_i - target_i| where N is the
    total number of elements. *)
