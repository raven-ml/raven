(* The suggested fix of each attribute in an argument walked whole, which
   compiles. *)

module Nx = struct
  type ('v, 's, 'd) t
  type float32_elt
  type 'd float32_t = (float, float32_elt, 'd) t
  type 'd packed = P : ('v, 's, 'd) t -> 'd packed

  module Ptree = struct
    module Walk = struct
      type ('a, 'b, 'd, 'e) cursor = Cursor

      let field (c : ('a, 'b, 'd, 'e) cursor) (_ : string) walk x = walk c x
      let leaf (_ : ('a, 'b, 'd, 'e) cursor) (_ : 'a) : 'b = assert false

      let tensor (_ : ('a, 'b, 'd, 'e) cursor) (_ : ('v, 's, 'd) t) :
          ('v, 's, 'e) t =
        assert false

      let list walk (c : ('a, 'b, 'd, 'e) cursor) l = List.map (walk c) l

      let under (outer : ('p, 'q, 'd, 'e) cursor -> 'u -> 'w)
          (_ : ('a, 'b, 'd, 'e) cursor -> 'p -> 'q)
          (_ : ('a, 'b, 'd, 'e) cursor) (x : 'u) : 'w =
        outer Cursor x
    end
  end

  module Rng = struct
    type 'd key

    let walk (_ : ('a, 'b, 'd, 'e) Ptree.Walk.cursor) (_ : 'd key) : 'e key =
      assert false
  end
end

module Foo = struct
  type ('a, _) t = { w : 'a }

  let walk c { w } = { w = Nx.Ptree.Walk.field c "w" Nx.Ptree.Walk.leaf w }
end

let f = Nx.Ptree.Walk.leaf

type ('a, 'd) t = {
  p : (('a, 'd) Foo.t[@ptree.walk Nx.Ptree.Walk.under Foo.walk f]);
  q : ('a, 'd) Foo.t;
  x : 'd Nx.float32_t;
  y : ('d Nx.float32_t[@ptree.walk Nx.Ptree.Walk.tensor]);
  e : (float, Nx.float32_elt, 'd) Nx.t;
  k : 'd Nx.packed;
  s : (('a, 'd) Foo.t, 'd) Foo.t;
}
[@@deriving ptree]

type 'd keys = ('d Nx.Rng.key[@ptree.walk Nx.Rng.walk]) list
[@@deriving ptree]

type 'd plain_keys = 'd Nx.Rng.key list [@@deriving ptree]
