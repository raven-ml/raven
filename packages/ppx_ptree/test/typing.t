The deriver names a walk by convention and checks no role: OCaml types the
derived walk and reports a mistake at the user's part. The cases compile
against a model of Nx with the types of the walk functions they use.

  $ cat > nx.ml <<'EOF'
  > type ('v, 's, 'd) t
  > type float32_elt
  > type 'd float32_t = (float, float32_elt, 'd) t
  > type 'd packed = P : ('v, 's, 'd) t -> 'd packed
  > type ('v, 's, 'd) tensor = ('v, 's, 'd) t
  > 
  > module Ptree = struct
  >   type ('s, 'd) t = Structure
  > 
  >   module Walk = struct
  >     type ('a, 'b, 'd, 'e) cursor = Cursor
  > 
  >     let field (c : ('a, 'b, 'd, 'e) cursor) (_ : string) walk x = walk c x
  >     let leaf (_ : ('a, 'b, 'd, 'e) cursor) (_ : 'a) : 'b = assert false
  > 
  >     let tensor (_ : ('a, 'b, 'd, 'e) cursor) (_ : ('v, 's, 'd) tensor) :
  >         ('v, 's, 'e) tensor =
  >       assert false
  > 
  >     let case (_ : ('a, 'b, 'd, 'e) cursor) (_ : string) = ()
  >     let list walk (c : ('a, 'b, 'd, 'e) cursor) l = List.map (walk c) l
  > 
  >     let under (outer : ('p, 'q, 'd, 'e) cursor -> 'u -> 'w)
  >         (_ : ('a, 'b, 'd, 'e) cursor -> 'p -> 'q)
  >         (_ : ('a, 'b, 'd, 'e) cursor) (x : 'u) : 'w =
  >       outer Cursor x
  > 
  >     let structure (_ : ('s, 'd) t) (_ : ('a, 'b, 'd, 'd) cursor) (x : 's) = x
  >   end
  > end
  > 
  > module Rng = struct
  >   type 'd key
  > 
  >   let walk (_ : ('a, 'b, 'd, 'e) Ptree.Walk.cursor) (_ : 'd key) : 'e key =
  >     assert false
  > end
  > EOF
  $ ocamlc -c nx.ml
  $ derive () { ocamlc -color never -c -ppx './pp.exe --as-ppx' "$1"; }

A module without the walk fails at the part with Unbound value M.walk.

  $ cat > missing_walk.ml <<'EOF'
  > module Missing = struct
  >   type ('a, 'd) t = { x : 'a }
  > end
  > 
  > type ('a, 'd) t = { sub : ('a, 'd) Missing.t } [@@deriving ptree]
  > EOF
  $ derive missing_walk.ml
  File "missing_walk.ml", line 5, characters 26-44:
  5 | type ('a, 'd) t = { sub : ('a, 'd) Missing.t } [@@deriving ptree]
                                ^^^^^^^^^^^^^^^^^^
  Error: Unbound value Missing.walk
  [2]

So does M.t without argument where M has no walk.

  $ cat > missing_plain_walk.ml <<'EOF'
  > module Missing = struct
  >   type t = int
  > 
  >   let ptree = ()
  > end
  > 
  > type ('a, 'd) t = { w : 'a; index : Missing.t } [@@deriving ptree]
  > EOF
  $ derive missing_plain_walk.ml
  File "missing_plain_walk.ml", line 7, characters 36-45:
  7 | type ('a, 'd) t = { w : 'a; index : Missing.t } [@@deriving ptree]
                                          ^^^^^^^^^
  Error: Unbound value Missing.walk
  [2]

Each walk the deriver names is constrained to walk the part's type, so a
walk of another type fails at the part: a field, a constructor's argument,
or an element of a list.

  $ cat > config.ml <<'EOF'
  > type t = { depth : int }
  > 
  > let walk (_ : _ Nx.Ptree.Walk.cursor) (x : int) = x
  > EOF
  $ ocamlc -c config.ml
  $ cat > walk_other_type.ml <<'EOF'
  > type ('a, 'd) t = { w : 'a; cfg : Config.t } [@@deriving ptree]
  > EOF
  $ derive walk_other_type.ml
  File "walk_other_type.ml", line 1, characters 34-42:
  1 | type ('a, 'd) t = { w : 'a; cfg : Config.t } [@@deriving ptree]
                                        ^^^^^^^^
  Error: The value Config.walk has type
           (ptree_a, ptree_b, ptree_d, ptree_e) Nx.Ptree.Walk.cursor ->
           int -> int
         but an expression was expected of type
           (ptree_a, ptree_b, ptree_d, ptree_e) Nx.Ptree.Walk.cursor ->
           Config.t -> Config.t
         Type int is not compatible with type Config.t
  [2]
  $ cat > argument_other_type.ml <<'EOF'
  > type ('a, 'd) t = W of 'a | C of Config.t [@@deriving ptree]
  > EOF
  $ derive argument_other_type.ml
  File "argument_other_type.ml", line 1, characters 33-41:
  1 | type ('a, 'd) t = W of 'a | C of Config.t [@@deriving ptree]
                                       ^^^^^^^^
  Error: The value Config.walk has type
           (ptree_a, ptree_b, ptree_d, ptree_e) Nx.Ptree.Walk.cursor ->
           int -> int
         but an expression was expected of type
           (ptree_a, ptree_b, ptree_d, ptree_e) Nx.Ptree.Walk.cursor ->
           Config.t -> Config.t
         Type int is not compatible with type Config.t
  [2]
  $ cat > element_other_type.ml <<'EOF'
  > type ('a, 'd) t = { w : 'a; cfgs : Config.t list } [@@deriving ptree]
  > EOF
  $ derive element_other_type.ml
  File "element_other_type.ml", line 1, characters 35-43:
  1 | type ('a, 'd) t = { w : 'a; cfgs : Config.t list } [@@deriving ptree]
                                         ^^^^^^^^
  Error: The value Config.walk has type
           (ptree_a, ptree_b, ptree_d, ptree_e) Nx.Ptree.Walk.cursor ->
           int -> int
         but an expression was expected of type
           (ptree_a, ptree_b, ptree_d, ptree_e) Nx.Ptree.Walk.cursor ->
           Config.t -> Config.t
         Type int is not compatible with type Config.t
  [2]

A structure at a closed payload, as (int, 'd) Kaun.Linear.t, fails with a
type error at the part.

  $ cat > closed_payload.ml <<'EOF'
  > module Linear = struct
  >   type ('a, _) t = { w : 'a }
  > 
  >   let walk c { w } = { w = Nx.Ptree.Walk.field c "w" Nx.Ptree.Walk.leaf w }
  > end
  > 
  > type ('a, 'd) t = { w : 'a; frozen : (int, 'd) Linear.t } [@@deriving ptree]
  > EOF
  $ derive closed_payload.ml
  File "closed_payload.ml", line 7, characters 37-55:
  7 | type ('a, 'd) t = { w : 'a; frozen : (int, 'd) Linear.t } [@@deriving ptree]
                                           ^^^^^^^^^^^^^^^^^^
  Error: The value Linear.walk has type
           (ptree_a, ptree_b, ptree_d, ptree_e) Nx.Ptree.Walk.cursor ->
           (ptree_a, 'a) Linear.t -> (ptree_b, 'b) Linear.t
         but an expression was expected of type
           (ptree_a, ptree_b, ptree_d, ptree_e) Nx.Ptree.Walk.cursor ->
           (int, ptree_d) Linear.t -> (int, ptree_e) Linear.t
         Type ptree_a is not compatible with type int
  [2]

Each [@ptree.walk f] is constrained to walk the part's type, so OCaml
reports a mistyped f at f, and a mistyped body inside it.

  $ cat > walk_mistyped.ml <<'EOF'
  > type ('a, 'd) t = { w : 'a [@ptree.walk 42] } [@@deriving ptree]
  > EOF
  $ derive walk_mistyped.ml
  File "walk_mistyped.ml", line 1, characters 40-42:
  1 | type ('a, 'd) t = { w : 'a [@ptree.walk 42] } [@@deriving ptree]
                                              ^^
  Error: The constant 42 has type int but an expression was expected of type
           (ptree_a, ptree_b, ptree_d, ptree_e) Nx.Ptree.Walk.cursor ->
           ptree_a -> ptree_b
  [2]
  $ cat > walk_wrong_type.ml <<'EOF'
  > type ('a, 'd) t = {
  >   w : 'a;
  >   x : 'd Nx.float32_t; [@ptree.walk Nx.Ptree.Walk.leaf]
  > }
  > [@@deriving ptree]
  > EOF
  $ derive walk_wrong_type.ml
  File "walk_wrong_type.ml", line 3, characters 36-54:
  3 |   x : 'd Nx.float32_t; [@ptree.walk Nx.Ptree.Walk.leaf]
                                          ^^^^^^^^^^^^^^^^^^
  Error: The value Nx.Ptree.Walk.leaf has type
           (ptree_a, ptree_b, ptree_d, ptree_e) Nx.Ptree.Walk.cursor ->
           ptree_a -> ptree_b
         but an expression was expected of type
           (ptree_a, ptree_b, ptree_d, ptree_e) Nx.Ptree.Walk.cursor ->
           ptree_d Nx.float32_t -> ptree_e Nx.float32_t
         Type ptree_a is not compatible with type
           ptree_d Nx.float32_t = (float, Nx.float32_elt, ptree_d) Nx.t
  [2]
  $ cat > walk_body_mistyped.ml <<'EOF'
  > type ('a, 'd) t = {
  >   a : 'a;
  >   b : 'd Nx.float32_t; [@ptree.walk fun _c x -> x + 1]
  > }
  > [@@deriving ptree]
  > EOF
  $ derive walk_body_mistyped.ml
  File "walk_body_mistyped.ml", line 3, characters 48-49:
  3 |   b : 'd Nx.float32_t; [@ptree.walk fun _c x -> x + 1]
                                                      ^
  Error: The value x has type
           ptree_d Nx.float32_t = (float, Nx.float32_elt, ptree_d) Nx.t
         but an expression was expected of type int
  [2]

A walk moves values between sets, so an f at one set, as
Nx.Ptree.Walk.structure s, does not type: OCaml reports at f that ptree_d
is not compatible with ptree_e.

  $ cat > walk_one_set.ml <<'EOF'
  > let ints () : (int, 'd) Nx.Ptree.t = Nx.Ptree.Structure
  > 
  > type ('a, 'd) t = {
  >   w : 'a;
  >   n : int; [@ptree.walk Nx.Ptree.Walk.structure (ints ())]
  > }
  > [@@deriving ptree]
  > EOF
  $ derive walk_one_set.ml
  File "walk_one_set.ml", line 5, characters 24-57:
  5 |   n : int; [@ptree.walk Nx.Ptree.Walk.structure (ints ())]
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  Error: This expression has type
           (ptree_a, ptree_b, ptree_d, ptree_d) Nx.Ptree.Walk.cursor ->
           int -> int
         but an expression was expected of type
           (ptree_a, ptree_b, ptree_d, ptree_e) Nx.Ptree.Walk.cursor ->
           int -> int
         Type ptree_d is not compatible with type ptree_e
  [2]

Each fix that the error of an attribute in an argument walked whole
suggests compiles.

  $ cat > foo.ml <<'EOF'
  > type ('a, _) t = { w : 'a }
  > 
  > let walk c { w } = { w = Nx.Ptree.Walk.field c "w" Nx.Ptree.Walk.leaf w }
  > EOF
  $ ocamlc -c foo.ml
  $ cat > argument_fixes.ml <<'EOF'
  > let f = Nx.Ptree.Walk.leaf
  > 
  > type ('a, 'd) t = {
  >   p : (('a, 'd) Foo.t[@ptree.walk Nx.Ptree.Walk.under Foo.walk f]);
  >   q : ('a, 'd) Foo.t;
  >   x : 'd Nx.float32_t;
  >   y : ('d Nx.float32_t[@ptree.walk Nx.Ptree.Walk.tensor]);
  >   e : (float, Nx.float32_elt, 'd) Nx.t;
  >   k : 'd Nx.packed;
  >   s : (('a, 'd) Foo.t, 'd) Foo.t;
  > }
  > [@@deriving ptree]
  > 
  > type 'd keys = ('d Nx.Rng.key[@ptree.walk Nx.Rng.walk]) list
  > [@@deriving ptree]
  > 
  > type 'd plain_keys = 'd Nx.Rng.key list [@@deriving ptree]
  > EOF
  $ derive argument_fixes.ml
