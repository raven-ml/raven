module Nx = struct
  module Ptree = struct
    module Walk = struct
      type ('a, 'b) cursor = Cursor

      let field (c : ('a, 'b) cursor) (_ : string) walk x = walk c x
      let leaf (_ : ('a, 'b) cursor) (_ : 'a) : 'b = assert false
      let structure _ (_ : ('a, 'b) cursor) x = x
    end
  end
end

module Missing = struct
  type t = int
end

type 'a t = { w : 'a; index : Missing.t } [@@deriving ptree]
