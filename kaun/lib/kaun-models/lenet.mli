(** LeNet-5: Classic convolutional neural network for handwritten digit
    recognition.

    LeCun et al., 1998: "Gradient-Based Learning Applied to Document
    Recognition" One of the first successful CNNs, originally designed for MNIST
    digit classification. *)

open Rune

type config = {
  num_classes : int;  (** Number of output classes (default: 10 for digits) *)
  input_channels : int;
      (** Number of input channels (default: 1 for grayscale) *)
  input_size : int * int;  (** Input image size (default: 32x32) *)
  activation : [ `tanh | `relu | `sigmoid ];
      (** Activation function (original used tanh) *)
  dropout_rate : float option;  (** Optional dropout rate for regularization *)
}
(** Configuration for LeNet-5 model *)

val default_config : config
(** Default configuration (original LeNet-5 for MNIST) *)

val mnist_config : config
(** MNIST-specific configuration (28x28 input, padded to 32x32) *)

val cifar10_config : config
(** CIFAR-10 configuration (32x32 RGB input) *)

type t = Kaun.module_
(** LeNet-5 model instance *)

(** Create a new LeNet-5 model *)
val create : ?config:config -> unit -> t
(** [create ?config ()] creates a new LeNet-5 model.

    Architecture:
    - Conv1: 6 filters of 5x5
    - Pool1: 2x2 average pooling
    - Conv2: 16 filters of 5x5
    - Pool2: 2x2 average pooling
    - FC1: 120 units
    - FC2: 84 units
    - Output: num_classes units

    The original paper used average pooling and tanh activation, but modern
    implementations often use max pooling and ReLU.

    Example:
    {[
      let model = LeNet.create ~config:LeNet.mnist_config () in
      let params = Kaun.init model ~rngs ~dtype:Float32 in
      let output = Kaun.apply model params ~training:false input in
    ]} *)

(** Create model for MNIST *)
val for_mnist : unit -> t
(** [for_mnist ()] creates a LeNet-5 model configured for MNIST digits.
    Equivalent to [create ~config:mnist_config ()]. *)

(** Create model for CIFAR-10 *)
val for_cifar10 : unit -> t
(** [for_cifar10 ()] creates a LeNet-5 model configured for CIFAR-10. Uses 3
    input channels for RGB images. *)

(** Forward pass through the model *)
val forward :
  model:t ->
  params:'a Kaun.params ->
  training:bool ->
  input:(float, 'a) Rune.t ->
  (float, 'a) Rune.t
(** [forward ~model ~params ~training ~input] performs a forward pass.

    @param model The LeNet-5 model
    @param params Model parameters
    @param training Whether in training mode (affects dropout if configured)
    @param input Input tensor of shape [batch_size; channels; height; width]
    @return Output logits of shape [batch_size; num_classes] *)

(** Extract feature representations *)
val extract_features :
  model:t ->
  params:'a Kaun.params ->
  input:(float, 'a) Rune.t ->
  (float, 'a) Rune.t
(** [extract_features ~model ~params ~input] extracts feature representations
    from the second-to-last layer (FC2), useful for transfer learning or
    visualization. Returns features of shape [batch_size; 84]. *)

(** Model statistics *)
val num_parameters : 'a Kaun.params -> int
(** [num_parameters params] returns the total number of parameters in the model.
*)

val parameter_breakdown : 'a Kaun.params -> string
(** [parameter_breakdown params] returns a detailed breakdown of parameters by
    layer. *)

(** {2 Training Helpers} *)

type train_config = {
  learning_rate : float;
  batch_size : int;
  num_epochs : int;
  weight_decay : float option;
  momentum : float option;
}
(** Training configuration *)

val default_train_config : train_config
(** Default training configuration for MNIST *)

(** Compute accuracy *)
val accuracy :
  predictions:(float, 'a) Rune.t -> labels:(int, int32_elt) Rune.t -> float
(** [accuracy ~predictions ~labels] computes classification accuracy.
    Predictions should be logits of shape [batch_size; num_classes], labels
    should be class indices of shape [batch_size]. *)
