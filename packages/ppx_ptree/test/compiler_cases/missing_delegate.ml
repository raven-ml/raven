module Nx = struct
  type ('element, 'layout) t = Tensor
end

module Missing = struct
  type t = int
end

type t = { nested : Missing.t } [@@deriving ptree]
