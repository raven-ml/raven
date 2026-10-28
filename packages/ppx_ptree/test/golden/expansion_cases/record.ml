type ('a, 'd) t = {
  w : 'a;
  b : 'a option;
  layers : ('a, 'd) Linear.t list;
  steps : 'd Nx.int32_t;
}
[@@deriving ptree]
