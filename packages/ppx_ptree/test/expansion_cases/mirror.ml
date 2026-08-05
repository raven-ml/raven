type t = {
  w : Nx.float32_t;
  b : Nx.float32_t option;
  dim : int; [@ptree.ignore]
}
[@@deriving ptree ~mirror]
