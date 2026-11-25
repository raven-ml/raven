(* An interface derives the values an implementation derives, for a private type
   too. The implementation derives its walks where these names, which a derived
   walk takes from the standard library, mean something else. *)

module Array : sig end
module Bool : sig end

val ignore : unit
val ( <> ) : unit

type ('a, 'd) t = { w : 'a; b : 'a option; steps : 'd Nx.int32_t }
[@@deriving ptree]

type 'd state = { scale : 'd Nx.float32_t } [@@deriving ptree]
type 'd frozen = private { scale : 'd Nx.float32_t } [@@deriving ptree]

type config = {
  depth : int; [@ptree.int]
  causal : bool; [@ptree.int]
  heads : (int[@ptree.int]) array;
}
[@@deriving ptree]
