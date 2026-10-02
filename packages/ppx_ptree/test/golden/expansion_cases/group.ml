type 'a tree = Leaf of 'a | Node of 'a tree list
and 'a t = { root : 'a tree; window : int option [@ptree.int] }
and state = { steps : Nx.int32_t; index : Cache_index.t } [@@deriving ptree]
