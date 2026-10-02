type 'a pair = { x : 'a }
and t = { p : Nx.float32_t pair } [@@deriving ptree]
