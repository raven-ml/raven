module Nx = struct
  type ('element, 'layout) t = Tensor
end

module Wrong = struct
  type t = int

  let map (f : int -> int) value = f value
  let map2 (f : int -> int -> int) left right = f left right
  let iter (f : int -> unit) value = f value
end

type t = { nested : Wrong.t [@ptree.using Wrong] } [@@deriving ptree]
