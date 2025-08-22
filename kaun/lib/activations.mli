(** Activation functions for neural networks *)

open Rune

(** {1 Standard Activations} *)

val relu : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [relu x] applies Rectified Linear Unit: max(0, x) *)

val relu6 : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [relu6 x] applies ReLU6: min(max(0, x), 6) *)

val sigmoid : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [sigmoid x] applies sigmoid: 1 / (1 + exp(-x)) *)

val tanh : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [tanh x] applies hyperbolic tangent *)

val softmax : ?axes:int array -> (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [softmax ?axes x] applies softmax normalization *)

(** {1 Modern Activations} *)

val gelu : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [gelu x] applies Gaussian Error Linear Unit (approximate version) *)

val silu : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [silu x] applies Sigmoid Linear Unit (SiLU/Swish): x * sigmoid(x) *)

val swish : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [swish x] alias for silu *)

val mish : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [mish x] applies Mish: x * tanh(softplus(x)) *)

(** {1 Parametric Activations} *)

val leaky_relu :
  ?negative_slope:float -> (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [leaky_relu ?negative_slope x] applies Leaky ReLU with negative slope *)

val elu : ?alpha:float -> (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [elu ?alpha x] applies Exponential Linear Unit *)

val selu : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [selu x] applies Scaled Exponential Linear Unit *)

val prelu : (float, 'a, 'dev) t -> (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [prelu alpha x] applies Parametric ReLU with learnable alpha *)

(** {1 Gated Linear Units (GLUs)} *)

val glu : (float, 'a, 'dev) t -> (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [glu x gate] applies Gated Linear Unit: x * sigmoid(gate) *)

val swiglu : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [swiglu x] applies SwiGLU: x * silu(x) *)

val geglu : (float, 'a, 'dev) t -> (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [geglu x gate] applies GeGLU: x * gelu(gate) *)

val reglu : (float, 'a, 'dev) t -> (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [reglu x gate] applies ReGLU: x * relu(gate) *)

(** {1 Other Activations} *)

val softplus : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [softplus x] applies softplus: log(1 + exp(x)) *)

val softsign : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [softsign x] applies softsign: x / (1 + |x|) *)

val hard_sigmoid :
  ?alpha:float -> ?beta:float -> (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [hard_sigmoid ?alpha ?beta x] applies piecewise linear sigmoid
    approximation. Default alpha=1/6, beta=0.5 *)

val hard_tanh : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [hard_tanh x] applies hard tanh: clip(x, -1, 1) *)

val hard_swish : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [hard_swish x] applies hard swish: x * relu6(x + 3) / 6 *)
