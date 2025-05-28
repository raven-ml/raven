module B = Nx_core.Make_frontend (Nx_native)
include B

(* ───── Overriding functions with default context ───── *)

let context = Lazy.from_fun Nx_native.create_context
let create dtype shape arr = B.create (Lazy.force context) dtype shape arr
let init dtype shape f = B.init (Lazy.force context) dtype shape f
let empty dtype shape = B.empty (Lazy.force context) dtype shape
let full dtype shape value = B.full (Lazy.force context) dtype shape value
let ones dtype shape = B.ones (Lazy.force context) dtype shape
let zeros dtype shape = B.zeros (Lazy.force context) dtype shape
let scalar dtype v = B.scalar (Lazy.force context) dtype v
let eye ?m ?k dtype n = B.eye (Lazy.force context) ?m ?k dtype n
let identity dtype n = B.identity (Lazy.force context) dtype n

let arange dtype start stop step =
  B.arange (Lazy.force context) dtype start stop step

let arange_f dtype start stop step =
  B.arange_f (Lazy.force context) dtype start stop step

let linspace dtype ?endpoint start stop num =
  B.linspace (Lazy.force context) dtype ?endpoint start stop num

let logspace dtype ?endpoint ?base start stop num =
  B.logspace (Lazy.force context) dtype ?endpoint ?base start stop num

let geomspace dtype ?endpoint start stop num =
  B.geomspace (Lazy.force context) dtype ?endpoint start stop num

let of_bigarray ba = B.of_bigarray (Lazy.force context) ba
let rand dtype ?seed shape = B.rand (Lazy.force context) dtype ?seed shape
let randn dtype ?seed shape = B.randn (Lazy.force context) dtype ?seed shape

let randint dtype ?seed ?high shape low =
  B.randint (Lazy.force context) dtype ?seed ?high shape low

(* ───── Aliases to unsafe functions ───── *)

let data t = B.unsafe_data t
let to_bigarray t = B.unsafe_to_bigarray t
let to_array t = B.unsafe_to_array t
let get_item indices t = B.unsafe_get indices t
let set_item indices value t = B.unsafe_set indices value t
let map_item f t = B.unsafe_map f t
let iter_item f t = B.unsafe_iter f t
let fold_item f acc t = B.unsafe_fold f acc t
