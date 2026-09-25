type t = { name : string [@walk f] } [@@deriving ptree]
