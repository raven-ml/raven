module Nx = struct
  type ('element, 'layout) t = Tensor
end

type t = { value : string [@ptree.leaf] } [@@deriving ptree]
