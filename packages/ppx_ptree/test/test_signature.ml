type 'a t = { w : 'a; b : 'a option } [@@deriving ptree]
type state = { scale : Nx.float32_t } [@@deriving ptree]
