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

module Linear = struct
  type ('a, _) t = { w : 'a }

  let walk c { w } = { w = Nx.Ptree.Walk.field c "w" Nx.Ptree.Walk.leaf w }
end

type ('a, 'd) t = { w : 'a; frozen : (int, 'd) Linear.t } [@@deriving ptree]
