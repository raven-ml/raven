module Nx = struct
  type ('element, 'layout) t = Tensor
end

type t = { metadata : string } [@@deriving ptree]
