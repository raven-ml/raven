type node = { value : Nx.float32_t; next : node option }
and t = { root : node } [@@deriving ptree]
