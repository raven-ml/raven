type ('a, 'd) tree = Leaf of 'a | Node of ('a, 'd) tree list

and ('a, 'd) t = {
  root : ('a, 'd) tree;
  frozen : ('d Nx.float32_t, 'd) tree;
  window : int option; [@ptree.int]
}

and 'd state = { steps : 'd Nx.int32_t; index : 'd Cache_index.t }
and shape = { depth : int [@ptree.int] } [@@deriving ptree]
