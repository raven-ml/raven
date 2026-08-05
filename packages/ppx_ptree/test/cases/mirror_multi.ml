type sub = { sw : Nx.float32_t }
and t = { nested : sub; w : Nx.float32_t } [@@deriving ptree ~mirror]
