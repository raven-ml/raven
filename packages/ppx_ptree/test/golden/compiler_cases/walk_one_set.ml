module Nx = struct
  type 'd float32_t

  module Ptree = struct
    type ('s, 'd) t = Structure

    module Walk = struct
      type ('a, 'b, 'd, 'e) cursor = Cursor

      let field (c : ('a, 'b, 'd, 'e) cursor) (_ : string) walk x = walk c x
      let leaf (_ : ('a, 'b, 'd, 'e) cursor) (_ : 'a) : 'b = assert false
      let structure (_ : ('s, 'd) t) (_ : ('a, 'b, 'd, 'd) cursor) (x : 's) = x
    end
  end
end

(* A description at one set, as Nx.Ptree.v makes. *)
let ints () : (int, 'd) Nx.Ptree.t = Nx.Ptree.Structure

type ('a, 'd) t = {
  w : 'a;
  n : int; [@ptree.walk Nx.Ptree.Walk.structure (ints ())]
}
[@@deriving ptree]
