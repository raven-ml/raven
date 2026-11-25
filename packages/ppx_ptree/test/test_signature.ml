module Array = struct end
module Bool = struct end

let ignore = ()
let ( <> ) = ()

type ('a, 'd) t = { w : 'a; b : 'a option; steps : 'd Nx.int32_t }
[@@deriving ptree]

type 'd state = { scale : 'd Nx.float32_t } [@@deriving ptree]
type 'd frozen = { scale : 'd Nx.float32_t } [@@deriving ptree]

type config = {
  depth : int; [@ptree.int]
  causal : bool; [@ptree.int]
  heads : (int[@ptree.int]) array;
}
[@@deriving ptree]
