module F = Nx_core.Make_frontend (Nx_rune)
include F

(* ───── Default Context Wrappers ───── *)

let context = Lazy.from_fun Nx_rune.create_context
let create dtype shape arr = create (Lazy.force context) dtype shape arr
let init dtype shape f = init (Lazy.force context) dtype shape f
let empty dtype shape = empty (Lazy.force context) dtype shape
let full dtype shape value = full (Lazy.force context) dtype shape value
let ones dtype shape = ones (Lazy.force context) dtype shape
let zeros dtype shape = zeros (Lazy.force context) dtype shape
let scalar dtype v = scalar (Lazy.force context) dtype v
let eye ?m ?k dtype n = eye (Lazy.force context) ?m ?k dtype n
let identity dtype n = identity (Lazy.force context) dtype n

let arange dtype start stop step =
  arange (Lazy.force context) dtype start stop step

let arange_f dtype start stop step =
  arange_f (Lazy.force context) dtype start stop step

let linspace dtype ?endpoint start stop num =
  linspace (Lazy.force context) dtype ?endpoint start stop num

let logspace dtype ?endpoint ?base start stop num =
  logspace (Lazy.force context) dtype ?endpoint ?base start stop num

let geomspace dtype ?endpoint start stop num =
  geomspace (Lazy.force context) dtype ?endpoint start stop num

let of_bigarray ba = of_bigarray (Lazy.force context) ba
let of_bigarray_ext ba = of_bigarray_ext (Lazy.force context) ba
let to_bigarray = to_bigarray
let to_bigarray_ext = to_bigarray_ext
let rand dtype ~key shape = rand (Lazy.force context) dtype ~key shape
let randn dtype ~key shape = randn (Lazy.force context) dtype ~key shape

let randint dtype ~key ?high shape low =
  randint (Lazy.force context) dtype ~key ?high shape low

module Rng = struct
  include F.Rng

  let uniform ~key dtype shape =
    F.Rng.uniform (Lazy.force context) ~key dtype shape

  let normal ~key dtype shape =
    F.Rng.normal (Lazy.force context) ~key dtype shape

  let randint dtype ~key ?high shape low =
    F.Rng.randint (Lazy.force context) dtype ~key ?high shape low

  let bernoulli ~key ~p shape =
    F.Rng.bernoulli (Lazy.force context) ~key ~p shape

  let permutation ~key n = F.Rng.permutation (Lazy.force context) ~key n
  let shuffle ~key x = F.Rng.shuffle (Lazy.force context) ~key x

  let categorical ~key ?axis ?shape logits =
    F.Rng.categorical (Lazy.force context) ~key ?axis ?shape logits

  let truncated_normal ~key dtype ~(lower : float) ~(upper : float) shape =
    F.Rng.truncated_normal (Lazy.force context) ~key dtype ~lower ~upper shape
end

(* ───── FFT ───── *)

let fftfreq ?d n = fftfreq (Lazy.force context) ?d n
let rfftfreq ?d n = rfftfreq (Lazy.force context) ?d n
