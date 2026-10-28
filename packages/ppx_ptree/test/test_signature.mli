(* An interface derives the values an implementation derives. *)

type ('a, 'd) t = { w : 'a; b : 'a option; steps : 'd Nx.int32_t }
[@@deriving ptree]

type 'd state = { scale : 'd Nx.float32_t } [@@deriving ptree]
type config = { depth : int [@ptree.int] } [@@deriving ptree]
