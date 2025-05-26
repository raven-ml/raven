module B = Nx_core.Make_frontend (Nx_native)
include B

(* ───── Overriding functions with default context ───── *)

let context = Nx_native.create_context ()
let create dtype shape arr = B.create context dtype shape arr
let init dtype shape f = B.init context dtype shape f
let empty dtype shape = B.empty context dtype shape
let full dtype shape value = B.full context dtype shape value
let ones dtype shape = B.ones context dtype shape
let zeros dtype shape = B.zeros context dtype shape
let scalar dtype v = B.scalar context dtype v
let eye ?m ?k dtype n = B.eye context ?m ?k dtype n
let identity dtype n = B.identity context dtype n
let arange dtype start stop step = B.arange context dtype start stop step
let arange_f dtype start stop step = B.arange_f context dtype start stop step

let linspace dtype ?endpoint start stop num =
  B.linspace context dtype ?endpoint start stop num

let logspace dtype ?endpoint ?base start stop num =
  B.logspace context dtype ?endpoint ?base start stop num

let geomspace dtype ?endpoint start stop num =
  B.geomspace context dtype ?endpoint start stop num

let of_bigarray ba = B.of_bigarray context ba
let rand dtype ?seed shape = B.rand context dtype ?seed shape
let randn dtype ?seed shape = B.randn context dtype ?seed shape

let randint dtype ?seed ?high shape low =
  B.randint context dtype ?seed ?high shape low

(* ───── Aliases to unsafe functions ───── *)

let data t = B.unsafe_data t
let to_bigarray t = B.unsafe_to_bigarray t
let to_array t = B.unsafe_to_array t
let get_item indices t = B.unsafe_get indices t
let set_item indices value t = B.unsafe_set indices value t
let map_item f t = B.unsafe_map f t
let iter_item f t = B.unsafe_iter f t
let fold_item f acc t = B.unsafe_fold f acc t
