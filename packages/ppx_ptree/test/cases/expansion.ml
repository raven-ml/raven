type helper = { state : Nx.int64_t }

and t = {
  weight : Nx.float32_t;
  optional : Nx.float32_t option;
  helper : helper;
  name : string; [@ptree.ignore]
}
[@@deriving ptree]
