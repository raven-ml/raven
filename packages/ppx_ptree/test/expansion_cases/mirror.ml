type t = {
  w : Nx.float32_t;
  b : Nx.float32_t option;
  layers : Nx.float32_t list;
  dim : int; [@ptree.ignore]
}
[@@deriving ptree ~mirror]
