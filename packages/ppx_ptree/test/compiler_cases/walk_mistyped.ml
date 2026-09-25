module Nx = struct
  module Ptree = struct
    type _ t = Structure

    module Walk = struct
      type ('a, 'b) cursor = Cursor

      let field (c : ('a, 'b) cursor) (_ : string) walk x = walk c x
      let leaf (_ : ('a, 'b) cursor) (_ : 'a) : 'b = assert false
      let structure _ (_ : ('a, 'b) cursor) x = x
    end
  end
end

type 'a t = { w : 'a [@ptree.walk 42] } [@@deriving ptree]
