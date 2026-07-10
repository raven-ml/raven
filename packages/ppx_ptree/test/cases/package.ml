module type S = sig end

type t = { value : (module S) } [@@deriving ptree]
