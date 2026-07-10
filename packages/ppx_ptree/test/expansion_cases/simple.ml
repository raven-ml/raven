type t = {
  weight : Nx.float32_t;
  optional : Nx.float32_t option;
  label : string; [@ptree.ignore]
}
[@@deriving ptree]
