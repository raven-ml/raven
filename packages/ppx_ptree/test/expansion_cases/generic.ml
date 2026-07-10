type 'dtype params = { weight : (float, 'dtype) Nx.t } [@@deriving ptree]
type t = Nx.float32_elt params
