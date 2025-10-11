(** JAX-style splittable random number generation for Tensor.

    This module provides a functional, splittable PRNG system similar to JAX,
    ensuring reproducibility and statistical independence of random streams. *)

type key
(** PRNG key type representing the random state *)

val key : int -> key
(** Create a PRNG key from a seed value.

    The seed is used to initialize the random state. Same seed produces same
    sequence of random numbers. *)

val split : ?n:int -> key -> key array
(** Split a PRNG key into multiple new keys.

    [split key n] returns an array of [n] new independent keys derived from the
    input key. The original key should not be reused after splitting to maintain
    statistical guarantees.

    @param key The key to split
    @param n Number of new keys to generate (default: 2)
    @return Array of new independent keys *)

val fold_in : key -> int -> key
(** Fold data into a key to derive a new key.

    [fold_in key data] combines a key with integer data to produce a new key.
    Useful for deriving keys based on iteration indices or other structured
    data.

    @param key The base key
    @param data Integer data to fold into the key
    @return New derived key *)

val to_int : key -> int
(** Convert key to integer representation for debugging.

    [to_int key] returns the internal integer representation of the key. This is
    mainly useful for debugging and should not be used to recreate keys. *)

(** {2 Random Sampling Functions} *)

val uniform : key -> ('a, 'b) Tensor.dtype -> int array -> ('a, 'b) Tensor.t
(** Generate uniform random values in \[0, 1).

    [uniform key dtype shape] generates a tensor of the given shape with values
    uniformly distributed in the half-open interval \[0, 1).

    @param key PRNG key for random generation
    @param dtype Data type of the output tensor
    @param shape Shape of the output tensor
    @return Tensor with uniform random values *)

val normal : key -> ('a, 'b) Tensor.dtype -> int array -> ('a, 'b) Tensor.t
(** Generate standard normal random values.

    [normal key dtype shape] generates a tensor of the given shape with values
    sampled from a standard normal distribution (mean=0, std=1).

    @param key PRNG key for random generation
    @param dtype Data type of the output tensor
    @param shape Shape of the output tensor
    @return Tensor with normal random values *)

val randint : key -> min:int -> max:int -> int array -> Tensor.int32_t
(** Generate random integers in a range.

    [randint key ~min ~max shape] generates a tensor of integers uniformly
    distributed in the half-open interval \[min, max).

    @param key PRNG key for random generation
    @param min Minimum value (inclusive)
    @param max Maximum value (exclusive)
    @param shape Shape of the output tensor
    @return Tensor with random integer values *)

val bernoulli : key -> p:float -> int array -> Tensor.uint8_t
(** Generate Bernoulli random values.

    [bernoulli key ~p shape] generates a tensor of boolean values where each
    element is true with probability [p].

    @param key PRNG key for random generation
    @param p Probability of true (must be in [0, 1])
    @param shape Shape of the output tensor
    @return Tensor with boolean random values *)

val permutation : key -> int -> Tensor.int32_t
(** Generate random permutation.

    [permutation key n] generates a random permutation of integers from 0 to
    n-1.

    @param key PRNG key for random generation
    @param n Number of elements to permute
    @return 1-D tensor containing a random permutation *)

val shuffle : key -> ('a, 'b) Tensor.t -> ('a, 'b) Tensor.t
(** Randomly shuffle the first dimension of a tensor.

    [shuffle key x] returns a copy of tensor [x] with its first dimension
    randomly shuffled.

    @param key PRNG key for random generation
    @param x Tensor to shuffle
    @return Shuffled tensor *)

val categorical :
  key -> ?axis:int -> ?shape:int array -> ('a, 'b) Tensor.t -> Tensor.int32_t
(** Sample from a categorical distribution.

    [categorical key logits ?axis] samples indices from a categorical
    distribution defined by logits along the specified axis.

    @param key PRNG key for random generation
    @param logits Unnormalized log probabilities
    @param shape Shape of the output tensor (default: scalar)
    @param axis Axis along which to sample (default: -1)
    @return Tensor of sampled indices *)

val truncated_normal :
  key ->
  ('a, 'b) Tensor.dtype ->
  lower:'a ->
  upper:'a ->
  int array ->
  ('a, 'b) Tensor.t
(** Generate random values from a truncated normal distribution.

    [truncated_normal key dtype ~lower ~upper shape] generates values from a
    normal distribution truncated to the range [lower, upper].

    @param key PRNG key for random generation
    @param dtype Data type of the output tensor
    @param lower Lower truncation bound
    @param upper Upper truncation bound
    @param shape Shape of the output tensor
    @return Tensor with truncated normal random values *)
