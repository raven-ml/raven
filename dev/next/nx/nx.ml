module B = Nx_core.Make_backend (Nx_native)

type ('a, 'b) t = ('a, 'b) B.t

let context = B.create_context ()
let broadcast_to x target_shape = B.broadcast_to context x target_shape
let add a b = B.add context a b
