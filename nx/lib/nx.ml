module F = Nx_core.Make_frontend (Nx_native)
include F

(* ───── Overriding functions with default context ───── *)

let context = Lazy.from_fun Nx_native.create_context
let create dtype shape arr = F.create (Lazy.force context) dtype shape arr
let init dtype shape f = F.init (Lazy.force context) dtype shape f
let empty dtype shape = F.empty (Lazy.force context) dtype shape
let full dtype shape value = F.full (Lazy.force context) dtype shape value
let ones dtype shape = F.ones (Lazy.force context) dtype shape
let zeros dtype shape = F.zeros (Lazy.force context) dtype shape
let scalar dtype v = F.scalar (Lazy.force context) dtype v
let eye ?m ?k dtype n = F.eye (Lazy.force context) ?m ?k dtype n
let identity dtype n = F.identity (Lazy.force context) dtype n

let arange dtype start stop step =
  F.arange (Lazy.force context) dtype start stop step

let arange_f dtype start stop step =
  F.arange_f (Lazy.force context) dtype start stop step

let linspace dtype ?endpoint start stop num =
  F.linspace (Lazy.force context) dtype ?endpoint start stop num

let logspace dtype ?endpoint ?base start stop num =
  F.logspace (Lazy.force context) dtype ?endpoint ?base start stop num

let geomspace dtype ?endpoint start stop num =
  F.geomspace (Lazy.force context) dtype ?endpoint start stop num

let of_bigarray ba = F.of_bigarray (Lazy.force context) ba
let rand dtype ?seed shape = F.rand (Lazy.force context) dtype ?seed shape
let randn dtype ?seed shape = F.randn (Lazy.force context) dtype ?seed shape

let randint dtype ?seed ?high shape low =
  F.randint (Lazy.force context) dtype ?seed ?high shape low

(* ───── FFT ───── *)

let fftfreq ?d n = F.fftfreq (Lazy.force context) ?d n
let rfftfreq ?d n = F.rfftfreq (Lazy.force context) ?d n

(* ───── Aliases to unsafe functions ───── *)

let data t = F.unsafe_data t
let to_bigarray t = F.unsafe_to_bigarray t
let to_array t = F.unsafe_to_array t
let get_item indices t = F.unsafe_get indices t
let set_item indices value t = F.unsafe_set indices value t
let map_item f t = F.unsafe_map f t
let iter_item f t = F.unsafe_iter f t
let fold_item f acc t = F.unsafe_fold f acc t
