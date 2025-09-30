(** Activation functions for neural networks.

    This module provides standard and modern activation functions for neural
    networks. All functions are differentiable through Rune's autodiff system
    and optimized for both forward and backward passes.

    Activation functions introduce non-linearity to neural networks, enabling
    them to learn complex patterns. The choice of activation function affects
    training dynamics, gradient flow, and model performance. *)

open Rune

(** {1 Standard Activations}

    Classical activation functions widely used in early neural network
    architectures. While some have known limitations (vanishing gradients), they
    remain useful in specific contexts and serve as building blocks for modern
    variants. *)

val relu : (float, 'a) t -> (float, 'a) t
(** [relu x] applies Rectified Linear Unit activation.

    Outputs the input directly if positive, otherwise outputs zero. Simple and
    computationally efficient, but suffers from dying ReLU problem where neurons
    can become permanently inactive.

    @param x Input tensor of any shape.

    @return Tensor of same shape with ReLU applied element-wise.

    {4 Example}

    Basic ReLU application:
    {[
      let x = Rune.create device Rune.float32 [|2; 3|] [|-1.0; 0.5; 2.0; -0.3; 0.0; 1.5|] in
      let activated = Activations.relu x in
      (* Result: [|0.0; 0.5; 2.0; 0.0; 0.0; 1.5|] *)
    ]}

    Mathematical formula: f(x) = max(0, x)

    {4 Properties}

    - Range: \[0, +∞)
    - Non-saturating for positive inputs
    - Sparse activation (many zeros)
    - Can suffer from dying ReLU problem
    - Excellent for hidden layers in deep networks
    - Zero gradient for negative inputs can halt learning *)

val relu6 : (float, 'a) t -> (float, 'a) t
(** [relu6 x] applies ReLU6 activation with upper bound.

    Caps the output at 6 to prevent activation explosion in mobile/embedded
    models. Commonly used in MobileNet architectures for quantization-friendly
    behavior.

    @param x Input tensor of any shape.

    @return Tensor of same shape with ReLU6 applied element-wise.

    {4 Example}

    ReLU6 with clamping:
    {[
      let x = Rune.create device Rune.float32 [|4|] [|-2.0; 3.0; 8.0; 1.0|] in
      let activated = Activations.relu6 x in
      (* Result: [|0.0; 3.0; 6.0; 1.0|] *)
    ]}

    Mathematical formula: f(x) = min(max(0, x), 6)

    {4 Properties}

    - Range: [0, 6]
    - Bounded output prevents activation explosion
    - Quantization-friendly for mobile deployment
    - Used in MobileNet, EfficientNet architectures
    - Maintains ReLU's computational efficiency *)

val sigmoid : (float, 'a) t -> (float, 'a) t
(** [sigmoid x] applies sigmoid activation.

    Maps any real number to the range (0, 1). Smooth and differentiable
    everywhere, but suffers from vanishing gradient problem for large input
    magnitudes. Primarily used in binary classification output layers.

    @param x Input tensor of any shape.

    @return Tensor of same shape with sigmoid applied element-wise.

    {4 Example}

    Sigmoid for binary classification output:
    {[
      let logits = Rune.create device Rune.float32 [|3|] [|-2.0; 0.0; 2.0|] in
      let probabilities = Activations.sigmoid logits in
      (* Result: [|0.119; 0.5; 0.881|] *)
    ]}

    Mathematical formula: f(x) = 1 / (1 + exp(-x))

    {4 Properties}

    - Range: (0, 1)
    - Smooth and continuously differentiable
    - Saturates for |x| > 4, causing vanishing gradients
    - Outputs not zero-centered (biases next layer)
    - Ideal for binary classification output layers
    - Avoid in hidden layers of deep networks *)

val tanh : (float, 'a) t -> (float, 'a) t
(** [tanh x] applies hyperbolic tangent activation.

    Maps inputs to the range (-1, 1) with zero-centered outputs. Better than
    sigmoid for hidden layers due to zero-centered property, but still suffers
    from vanishing gradients.

    @param x Input tensor of any shape.

    @return Tensor of same shape with tanh applied element-wise.

    {4 Example}

    Tanh for zero-centered activation:
    {[
      let x = Rune.create device Rune.float32 [|4|] [|-2.0; -1.0; 1.0; 2.0|] in
      let activated = Activations.tanh x in
      (* Result: [|-0.964; -0.762; 0.762; 0.964|] *)
    ]}

    Mathematical formula: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    {4 Properties}

    - Range: (-1, 1)
    - Zero-centered outputs
    - Smooth and continuously differentiable
    - Still suffers from vanishing gradient problem
    - Better than sigmoid for hidden layers
    - Used in LSTM/GRU gating mechanisms *)

val softmax : ?axes:int list -> (float, 'a) t -> (float, 'a) t
(** [softmax ?axes x] applies softmax normalization.

    Converts a vector of real numbers into a probability distribution. The
    output values sum to 1 and are all positive. Essential for multi-class
    classification output layers.

    @param axes
      Axes along which to compute softmax. Defaults to last axis. Each slice
      along these axes will sum to 1.
    @param x Input tensor (typically logits from final layer).

    @return Tensor of same shape with softmax applied along specified axes.

    {4 Example}

    Multi-class classification probabilities:
    {[
      let logits = Rune.create device Rune.float32 [|2; 3|] [|1.0; 2.0; 3.0; 0.5; 1.5; 2.5|] in
      let probs = Activations.softmax logits in
      (* Each row sums to 1.0 *)

      (* Softmax along specific axis *)
      let probs_axis0 = Activations.softmax ~axes:[|0|] logits
    ]}

    Mathematical formula: f(x_i) = exp(x_i) / sum_j exp(x_j)

    {4 Properties}

    - Output range: (0, 1) with sum = 1
    - Differentiable probability distribution
    - Amplifies differences between largest values
    - Used exclusively in multi-class output layers
    - Numerically stable implementations use max subtraction
    - Temperature parameter can control sharpness *)

(** {1 Modern Activations}

    State-of-the-art activation functions designed to address limitations of
    classical functions. These typically provide better gradient flow, faster
    convergence, and improved performance on deep networks. *)

val gelu : (float, 'a) t -> (float, 'a) t
(** [gelu x] applies Gaussian Error Linear Unit activation.

    Smooth approximation to ReLU that weights inputs by their percentile in a
    Gaussian distribution. Provides better gradient flow than ReLU and has
    become the standard in transformer architectures.

    @param x Input tensor of any shape.

    @return Tensor of same shape with GELU applied element-wise.

    {4 Example}

    GELU in transformer layers:
    {[
      let hidden = Rune.create device Rune.float32 [|2; 4|] [|-1.0; -0.5; 0.5; 1.0; 0.0; 1.5; -0.8; 2.0|] in
      let activated = Activations.gelu hidden in
      (* Smooth activation with better gradient properties *)
    ]}

    Mathematical formula: f(x) = x * Φ(x) where Φ is the CDF of standard normal
    distribution. Approximated as: f(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x +
    0.044715 * x³)))

    {4 Properties}

    - Smooth, non-monotonic function
    - Better gradient flow than ReLU
    - Standard in BERT, GPT, and modern transformers
    - Computationally more expensive than ReLU
    - Non-zero gradient everywhere
    - Self-gating behavior based on input magnitude *)

val silu : (float, 'a) t -> (float, 'a) t
(** [silu x] applies Sigmoid Linear Unit (also known as Swish) activation.

    Multiplies input by its sigmoid, providing smooth behavior and better
    gradient properties than ReLU. Self-gating mechanism allows the function to
    selectively emphasize or suppress features.

    @param x Input tensor of any shape.

    @return Tensor of same shape with SiLU applied element-wise.

    {4 Example}

    SiLU for better gradient flow:
    {[
      let x = Rune.create device Rune.float32 [|3|] [|-2.0; 0.0; 2.0|] in
      let activated = Activations.silu x in
      (* Result: [|-0.238; 0.0; 1.762|] *)
    ]}

    Mathematical formula: f(x) = x * sigmoid(x) = x / (1 + exp(-x))

    {4 Properties}

    - Smooth and differentiable everywhere
    - Self-gating mechanism
    - Better than ReLU for deep networks
    - Used in EfficientNet, MobileNet v3
    - Non-monotonic with minimum at x ≈ -0.278
    - Bounded below by -0.278x *)

val swish : (float, 'a) t -> (float, 'a) t
(** [swish x] applies Swish activation (alias for {!silu}).

    Identical to SiLU - different names for the same function. Swish was the
    original name from Google's research, while SiLU is the more standardized
    name in recent literature.

    @param x Input tensor of any shape.

    @return Tensor of same shape with Swish applied element-wise.

    See {!silu} for detailed documentation and examples. *)

val mish : (float, 'a) t -> (float, 'a) t
(** [mish x] applies Mish activation function.

    Smooth, non-monotonic activation that combines benefits of ReLU and Swish.
    Shows improved performance on some computer vision tasks but is
    computationally more expensive.

    @param x Input tensor of any shape.

    @return Tensor of same shape with Mish applied element-wise.

    {4 Example}

    Mish for improved accuracy:
    {[
      let x = Rune.create device Rune.float32 [|4|] [|-3.0; -1.0; 1.0; 3.0|] in
      let activated = Activations.mish x in
      (* Smooth activation with self-regularization *)
    ]}

    Mathematical formula: f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 +
    exp(x)))

    {4 Properties}

    - Smooth and continuously differentiable
    - Self-regularizing behavior
    - Can outperform ReLU/Swish on some tasks
    - Computationally expensive due to tanh and softplus
    - Non-monotonic with better gradient flow
    - Range: approximately (-0.31, +∞) *)

(** {1 Parametric Activations}

    Functions with learnable or configurable parameters that can adapt during
    training or be tuned for specific tasks. These offer more flexibility than
    fixed activations. *)

val leaky_relu : ?negative_slope:float -> (float, 'a) t -> (float, 'a) t
(** [leaky_relu ?negative_slope x] applies Leaky ReLU with configurable negative
    slope.

    Addresses the dying ReLU problem by allowing small negative values to pass
    through. The slope for negative inputs is configurable, enabling tuning for
    specific tasks.

    @param negative_slope
      Slope for negative inputs. Default is 0.01. Must be positive and typically
      small (0.01 to 0.3).
    @param x Input tensor of any shape.

    @return Tensor of same shape with Leaky ReLU applied element-wise.

    {4 Example}

    Preventing dying ReLU:
    {[
      let x = Rune.create device Rune.float32 [|4|] [|-2.0; -0.5; 0.5; 2.0|] in
      let activated = Activations.leaky_relu ~negative_slope:0.1 x in
      (* Result: [|-0.2; -0.05; 0.5; 2.0|] *)

      (* Default slope of 0.01 *)
      let activated_default = Activations.leaky_relu x
    ]}

    Mathematical formula: f(x) = max(αx, x) where α is the negative slope

    {4 Properties}

    - Range: (-∞, +∞)
    - Prevents dying ReLU problem
    - Non-zero gradient for negative inputs
    - Computationally efficient
    - Common slopes: 0.01 (default), 0.1, 0.2
    - Good alternative to ReLU in deep networks *)

val elu : ?alpha:float -> (float, 'a) t -> (float, 'a) t
(** [elu ?alpha x] applies Exponential Linear Unit activation.

    Smooth function that approaches -α for large negative inputs. Provides
    zero-centered activations and better gradient flow than ReLU while avoiding
    the dying ReLU problem.

    @param alpha
      Parameter controlling the saturation value for negative inputs. Default is
      1.0. Must be positive.
    @param x Input tensor of any shape.

    @return Tensor of same shape with ELU applied element-wise.

    {4 Example}

    ELU for better gradient properties:
    {[
      let x = Rune.create device Rune.float32 [|4|] [|-3.0; -1.0; 1.0; 3.0|] in
      let activated = Activations.elu ~alpha:1.0 x in
      (* Negative values approach -1.0, positive unchanged *)

      (* Custom alpha *)
      let activated_custom = Activations.elu ~alpha:0.5 x
    ]}

    Mathematical formula: f(x) = x if x > 0, α(exp(x) - 1) if x ≤ 0

    {4 Properties}

    - Range: (-α, +∞)
    - Zero-centered activations push mean closer to zero
    - Smooth everywhere, no dying neuron problem
    - More computationally expensive than ReLU
    - Better gradient flow than ReLU
    - Reduces bias shift effect *)

val selu : (float, 'a) t -> (float, 'a) t
(** [selu x] applies Scaled Exponential Linear Unit activation.

    Self-normalizing activation that maintains zero mean and unit variance
    through the network when used with appropriate weight initialization. Has
    specific mathematical properties that enable self-normalization.

    @param x Input tensor of any shape.

    @return Tensor of same shape with SELU applied element-wise.

    {4 Example}

    Self-normalizing networks:
    {[
      let x = Rune.create device Rune.float32 [|3; 4|] [|-2.0; -1.0; 0.0; 1.0; 2.0; -0.5; 1.5; -1.5; 0.5; -2.5; 3.0; 0.2|] in
      let activated = Activations.selu x in
      (* Self-normalizing properties maintain statistics *)
    ]}

    Mathematical formula: f(x) = λ * x if x > 0, λ * α(exp(x) - 1) if x ≤ 0
    where λ ≈ 1.0507 and α ≈ 1.6733

    {4 Properties}

    - Self-normalizing: maintains zero mean, unit variance
    - Requires specific weight initialization (lecun_normal)
    - Can replace batch normalization in some cases
    - Fixed parameters (not learnable)
    - Works best with dropout, not other normalization
    - Enables very deep networks without normalization *)

val prelu : (float, 'a) t -> (float, 'a) t -> (float, 'a) t
(** [prelu alpha x] applies Parametric ReLU with learnable slope parameters.

    Generalizes Leaky ReLU by making the negative slope learnable. Different
    channels or elements can learn different slopes, providing more flexibility
    than fixed-slope variants.

    @param alpha
      Learnable slope parameters for negative inputs. Can be scalar (shared) or
      same shape as x (element-wise). Values are typically initialized to small
      positive values.
    @param x Input tensor of any shape.

    @return Tensor of same shape with PReLU applied element-wise.

    {4 Example}

    Learnable negative slopes:
    {[
      let x = Rune.create device Rune.float32 [|2; 3|] [|-1.0; 0.5; -0.2; 1.0; -0.8; 2.0|] in

      (* Shared slope for all elements *)
      let alpha_shared = Rune.create device Rune.float32 [|1|] [|0.1|] in
      let activated_shared = Activations.prelu alpha_shared x in

      (* Different slope per element *)
      let alpha_elementwise = Rune.create device Rune.float32 [|2; 3|] [|0.05; 0.1; 0.15; 0.2; 0.25; 0.3|] in
      let activated_elementwise = Activations.prelu alpha_elementwise x
    ]}

    Mathematical formula: f(x_i) = max(α_i * x_i, x_i)

    {4 Properties}

    - Learnable parameters adapt during training
    - Can use shared or element-wise parameters
    - Generalizes ReLU and Leaky ReLU
    - Helps with dying ReLU problem
    - Initialize α close to zero (e.g., 0.01-0.25)
    - Popular in computer vision tasks *)

(** {1 Gated Linear Units (GLUs)}

    Advanced gating mechanisms that selectively allow information to pass
    through. Popular in transformer architectures and modern language models for
    their ability to control information flow dynamically. *)

val glu : (float, 'a) t -> (float, 'a) t -> (float, 'a) t
(** [glu x gate] applies Gated Linear Unit.

    Multiplies input by sigmoid-activated gate values, allowing selective
    information flow. The gating mechanism enables the model to control which
    features are emphasized or suppressed.

    @param x Input values to be gated.
    @param gate Gate values that control information flow.

    @return Element-wise product of x and sigmoid(gate).

    {4 Example}

    Basic gating mechanism:
    {[
      let x = Rune.create device Rune.float32 [|2; 4|] [|1.0; 2.0; -1.0; 0.5; 0.8; -0.3; 1.5; 2.1|] in
      let gate = Rune.create device Rune.float32 [|2; 4|] [|0.5; -0.2; 2.0; -1.0; 1.2; 0.1; -0.5; 0.8|] in
      let gated = Activations.glu x gate in
      (* Each element of x is modulated by sigmoid(gate) *)
    ]}

    Mathematical formula: f(x, g) = x * sigmoid(g)

    {4 Properties}

    - Selective information flow through gating
    - Sigmoid gate outputs in range (0, 1)
    - Foundation for more complex GLU variants
    - Used in language modeling architectures
    - Enables dynamic feature selection
    - Computationally efficient gating mechanism *)

val swiglu : (float, 'a) t -> (float, 'a) t
(** [swiglu x] applies SwiGLU (Swish Gated Linear Unit).

    Self-gating variant where the input is split in half, with one half gating
    the other using SiLU/Swish activation. Popular in transformer architectures,
    especially large language models.

    @param x
      Input tensor where the last dimension is split in half. Last dimension
      must be even.

    @return Tensor with last dimension halved, containing gated values.

    @raise Invalid_argument if the last dimension is odd.

    {4 Example}

    SwiGLU in transformer feed-forward:
    {[
      let x = Rune.create device Rune.float32 [|2; 6|] [|1.0; 2.0; -1.0; 0.5; 0.8; -0.3; 1.5; 2.1; 0.2; -0.7; 1.1; 0.9|] in
      let gated = Activations.swiglu x in
      (* Shape becomes [|2; 3|] - last dim halved *)
    ]}

    Mathematical formula: SwiGLU(x) = SiLU(x_{1:d/2}) * x_{d/2+1:d}
    where x is split along the last dimension

    {4 Properties}

    - Self-gating: no separate gate input needed
    - Reduces parameter count vs separate linear layers
    - State-of-the-art performance in language models
    - Used in LLaMA, PaLM, and other large models
    - Requires even-dimensional input
    - Combines gating with SiLU's smooth properties *)

val geglu : (float, 'a) t -> (float, 'a) t -> (float, 'a) t
(** [geglu x gate] applies GeGLU (GELU Gated Linear Unit).

    Uses GELU activation for gating instead of sigmoid. Combines GELU's smooth
    properties with GLU's selective information flow, popular in vision
    transformers and modern architectures.

    @param x Input values to be gated.
    @param gate Gate values processed with GELU activation.

    @return Element-wise product of x and gelu(gate).

    {4 Example}

    GELU-based gating:
    {[
      let x = Rune.create device Rune.float32 [|2; 4|] [|1.0; -0.5; 2.0; 0.3; -0.8; 1.2; 0.1; -1.5|] in
      let gate = Rune.create device Rune.float32 [|2; 4|] [|0.2; 1.5; -0.3; 0.8; 1.0; -0.6; 2.0; 0.4|] in
      let gated = Activations.geglu x gate
    ]}

    Mathematical formula: f(x, g) = x * GELU(g)

    {4 Properties}

    - Combines GELU's smooth activation with gating
    - Better gradient flow than sigmoid-based GLU
    - Used in vision transformers and modern CNNs
    - More computationally expensive than regular GLU
    - Provides sophisticated gating mechanism
    - Popular in transformer feed-forward networks *)

val reglu : (float, 'a) t -> (float, 'a) t -> (float, 'a) t
(** [reglu x gate] applies ReGLU (ReLU Gated Linear Unit).

    Uses ReLU activation for gating, providing sparse gating where only positive
    gate values allow information to pass through. Simpler and more
    computationally efficient than other GLU variants.

    @param x Input values to be gated.
    @param gate Gate values processed with ReLU activation.

    @return Element-wise product of x and relu(gate).

    {4 Example}

    ReLU-based sparse gating:
    {[
      let x = Rune.create device Rune.float32 [|2; 3|] [|1.0; 2.0; -0.5; 0.8; -1.2; 1.5|] in
      let gate = Rune.create device Rune.float32 [|2; 3|] [|0.3; -0.5; 1.2; -0.2; 0.8; 0.1|] in
      let gated = Activations.reglu x gate in
      (* Negative gate values completely block information *)
    ]}

    Mathematical formula: f(x, g) = x * ReLU(g)

    {4 Properties}

    - Sparse gating: negative gates completely block info
    - Most computationally efficient GLU variant
    - Sharp cutoff at gate = 0
    - Can lead to dead neurons if gates become negative
    - Simpler than smooth GLU variants
    - Good baseline for gating mechanisms *)

(** {1 Other Activations} *)

val softplus : (float, 'a) t -> (float, 'a) t
(** [softplus x] applies softplus activation.

    Smooth approximation to ReLU that is differentiable everywhere. Always
    positive and approaches ReLU for large positive inputs. Used as a smooth
    alternative to ReLU or in probabilistic models.

    @param x Input tensor of any shape.

    @return Tensor of same shape with softplus applied element-wise.

    {4 Example}

    Smooth approximation to ReLU:
    {[
      let x = Rune.create device Rune.float32 [|4|] [|-3.0; -1.0; 1.0; 3.0|] in
      let activated = Activations.softplus x in
      (* Result: [|0.049; 0.313; 1.313; 3.049|] *)
    ]}

    Mathematical formula: f(x) = ln(1 + exp(x))

    {4 Properties}

    - Range: (0, +∞)
    - Smooth approximation to ReLU
    - Always positive output
    - Differentiable everywhere
    - Can cause numerical overflow for large x
    - Used in Mish and probabilistic models *)

val softsign : (float, 'a) t -> (float, 'a) t
(** [softsign x] applies softsign activation.

    Smooth alternative to tanh that saturates more gradually. Computationally
    simpler than tanh while providing similar range and zero-centered outputs.

    @param x Input tensor of any shape.

    @return Tensor of same shape with softsign applied element-wise.

    {4 Example}

    Gradual saturation alternative to tanh:
    {[
      let x = Rune.create device Rune.float32 [|5|] [|-10.0; -2.0; 0.0; 2.0; 10.0|] in
      let activated = Activations.softsign x in
      (* Result: [|-0.909; -0.667; 0.0; 0.667; 0.909|] *)
    ]}

    Mathematical formula: f(x) = x / (1 + |x|)

    {4 Properties}

    - Range: (-1, 1)
    - Zero-centered like tanh
    - Slower saturation than tanh
    - Computationally simpler than tanh
    - Polynomial rather than exponential function
    - Less commonly used than tanh or modern activations *)

val hard_sigmoid : ?alpha:float -> ?beta:float -> (float, 'a) t -> (float, 'a) t
(** [hard_sigmoid ?alpha ?beta x] applies hard sigmoid activation.

    Piecewise linear approximation to sigmoid that is computationally efficient
    and quantization-friendly. Used in mobile and embedded applications where
    computational resources are limited.

    @param alpha
      Slope of the linear region. Default is 1/6 ≈ 0.167. Controls the steepness
      of the activation.
    @param beta
      Horizontal offset. Default is 0.5. Controls the center point of the
      activation.
    @param x Input tensor of any shape.

    @return Tensor of same shape with hard sigmoid applied element-wise.

    {4 Example}

    Efficient sigmoid approximation:
    {[
      let x = Rune.create device Rune.float32 [|5|] [|-4.0; -2.0; 0.0; 2.0; 4.0|] in
      let activated = Activations.hard_sigmoid x in
      (* Using default alpha=1/6, beta=0.5 *)

      (* Custom parameters *)
      let activated_custom = Activations.hard_sigmoid ~alpha:0.2 ~beta:0.6 x
    ]}

    Mathematical formula: f(x) = max(0, min(1, α*x + β))

    {4 Properties}

    - Range: [0, 1]
    - Piecewise linear approximation
    - Computationally very efficient
    - Quantization-friendly for mobile deployment
    - Used in MobileNet architectures
    - Default parameters chosen to approximate sigmoid *)

val hard_tanh : (float, 'a) t -> (float, 'a) t
(** [hard_tanh x] applies hard tanh activation.

    Piecewise linear approximation to tanh that clips values to [-1, 1]. Simple
    and computationally efficient while maintaining tanh's zero-centered
    property.

    @param x Input tensor of any shape.

    @return Tensor of same shape with hard tanh applied element-wise.

    {4 Example}

    Linear tanh approximation:
    {[
      let x = Rune.create device Rune.float32 [|5|] [|-3.0; -0.5; 0.0; 0.5; 3.0|] in
      let activated = Activations.hard_tanh x in
      (* Result: [|-1.0; -0.5; 0.0; 0.5; 1.0|] *)
    ]}

    Mathematical formula: f(x) = max(-1, min(1, x))

    {4 Properties}

    - Range: [-1, 1]
    - Zero-centered outputs
    - Piecewise linear (identity in [-1,1])
    - Computationally very efficient
    - Sharp transitions at boundaries
    - Simple clipping operation *)

val hard_swish : (float, 'a) t -> (float, 'a) t
(** [hard_swish x] applies hard swish activation.

    Computationally efficient approximation to swish/SiLU using hard sigmoid
    instead of regular sigmoid. Designed for mobile applications requiring fast
    inference with minimal accuracy loss.

    @param x Input tensor of any shape.

    @return Tensor of same shape with hard swish applied element-wise.

    {4 Example}

    Efficient swish approximation:
    {[
      let x = Rune.create device Rune.float32 [|4|] [|-3.0; -1.0; 1.0; 3.0|] in
      let activated = Activations.hard_swish x in
      (* Approximates swish with linear operations *)
    ]}

    Mathematical formula: f(x) = x * ReLU6(x + 3) / 6

    {4 Properties}

    - Efficient approximation to swish
    - Uses only linear operations and ReLU6
    - Quantization-friendly
    - Used in MobileNet v3 architecture
    - Maintains most benefits of swish
    - Significantly faster than true swish *)
