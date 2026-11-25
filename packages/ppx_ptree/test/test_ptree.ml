(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A derived walk is the hand-written walk of its type: it visits the same
   leaves and reports, at the same paths, in the same order, and rebuilds the
   value it walks. Each test derives a type and compares its walk with one
   written by hand from the .mli's rules. *)

open Windtrap
module P = Nx.Ptree

let device = Result.get_ok (Rig.software "ppx_ptree")

module S = (val Nx.devices [ device ])

type host = Nx.host
type f32 = host Nx.float32_t

let f32 xs : f32 = Nx.create Nx.float32 [| Array.length xs |] xs
let i32 xs : host Nx.int32_t = Nx.create Nx.int32 [| Array.length xs |] xs

let visit_equal a b =
  match (a, b) with
  | P.Leaf p, P.Leaf q -> P.Path.equal p q
  | P.Report (p, r), P.Report (q, r') -> P.Path.equal p q && r = r'
  | _ -> false

let visit = Testable.make ~pp:P.pp_visit ~equal:visit_equal

(* Arrays that are the same values ([==]), in order. *)
let same xs ys =
  List.length xs = List.length ys
  && List.for_all2
       (fun (Nx.P a) (Nx.P b) ->
         match Nx.Dtype.equal_witness (Nx.dtype a) (Nx.dtype b) with
         | Some Equal -> a == b
         | None -> false)
       xs ys

let pp_array ppf (Nx.P a) =
  Format.fprintf ppf "%s [%s]"
    (Nx.Dtype.name (Nx.dtype a))
    (String.concat "; " (Array.to_list (Array.map string_of_int (Nx.shape a))))

let arrays =
  Testable.make
    ~pp:(Format.pp_print_list ~pp_sep:Format.pp_print_space pp_array)
    ~equal:same

let placement = Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal

let pp_visits s ppf x =
  Format.pp_print_list ~pp_sep:Format.pp_print_space P.pp_visit ppf
    (P.visits s x)

(* [agree hand derived x]: [derived] visits what [hand] visits, and the value it
   rebuilds from [x]'s own arrays has, as [hand] sees it, [x]'s visits and
   arrays. *)
let agree hand derived x =
  equal ~msg:"visits" (list visit) (P.visits hand x) (P.visits derived x);
  let xs, _ = P.flatten derived x in
  let y = P.rebuild derived ~like:x xs in
  equal ~msg:"visits of the rebuilt value" (list visit) (P.visits hand x)
    (P.visits hand y);
  equal ~msg:"arrays of the rebuilt value" arrays xs (fst (P.flatten hand y))

(* [positions fold] is the paths at which [fold] passes its function a payload,
   in walk order. *)
let positions fold =
  List.rev (fold (fun p _ acc -> P.Path.to_string p :: acc) [])

(* [placements s x] is the placement of each of [x]'s arrays, in walk order. *)
let placements s x =
  List.rev (P.fold s (fun _ a acc -> Nx.placement a :: acc) x [])

let all_on p ps = equal (list placement) (List.map (fun _ -> p) ps) ps

(* Absent and present at even odds: [Gen.option] draws [None] at 0.15, which 100
   cases all miss with probability 9e-8. *)
let optional g = Gen.(frequency [ (1, constant None); (1, map Option.some g) ])

let floats =
  Gen.(map (fun n -> f32 (Array.init n float_of_int)) (int_range 0 2))

let linear_of position =
  let open Gen in
  let+ w = position and+ b = optional position in
  { Kaun.Linear.w; b }

(* [@@deriving ptree] *)

module Example = struct
  type ('a, 'd) t = {
    l1 : ('a, 'd) Kaun.Linear.t;
    l2 : ('a, 'd) Kaun.Linear.t;
    steps : 'd Nx.int32_t;
    window : int option; [@ptree.int]
  }
  [@@deriving ptree]
end

(* The walk the .mli gives as equivalent to its example's. *)
module Example_hand = struct
  type ('a, 'd) t = ('a, 'd) Example.t

  let walk c (x : _ t) : _ t =
    let open P.Walk in
    let l1 = field c "l1" Kaun.Linear.walk x.l1 in
    let l2 = field c "l2" Kaun.Linear.walk x.l2 in
    let steps = field c "steps" tensor x.steps in
    let window = field c "window" (option int) x.window in
    { l1; l2; steps; window }
end

let examples =
  let open Gen in
  let+ l1 = linear_of floats
  and+ l2 = linear_of floats
  and+ steps = int_range 0 2
  and+ window = optional (int_range 0 8) in
  { Example.l1; l2; steps = i32 (Array.init steps Int32.of_int); window }

let law_example (x : (f32, host) Example.t) =
  cover "an absent bias" (Option.is_none x.l1.b);
  cover "a present bias" (Option.is_some x.l1.b);
  cover "an absent window" (Option.is_none x.window);
  cover "a present window" (Option.is_some x.window);
  agree (P.instantiate (module Example_hand)) (P.instantiate (module Example)) x

(* nx's guide (packages/nx/doc/06-structures.md) says ppx_ptree's tests compare
   its derived [Block] with the hand-written one. *)
module Guide = struct
  module Linear = struct
    type ('a, _) t = { w : 'a; b : 'a option }

    let walk c { w; b } =
      let open P.Walk in
      let w = field c "w" leaf w in
      let b = field c "b" (option leaf) b in
      { w; b }
  end

  module Block_hand = struct
    type ('a, 'd) t = {
      proj : ('a, 'd) Linear.t;
      steps : 'd Nx.int32_t;
      window : int option;
    }

    let walk c { proj; steps; window } =
      let open P.Walk in
      let proj = field c "proj" Linear.walk proj in
      let steps = field c "steps" tensor steps in
      let window = field c "window" (option int) window in
      { proj; steps; window }
  end

  module Block = struct
    type ('a, 'd) t = ('a, 'd) Block_hand.t = {
      proj : ('a, 'd) Linear.t;
      steps : 'd Nx.int32_t;
      window : int option; [@ptree.int]
    }
    [@@deriving ptree]
  end
end

let test_guide () =
  agree
    (P.instantiate (module Guide.Block_hand))
    (P.instantiate (module Guide.Block))
    {
      Guide.Block_hand.proj = { w = f32 [| 1. |]; b = Some (f32 [| 2. |]) };
      steps = Nx.zeros Nx.int32 [||];
      window = Some 4;
    }

(* Declarations *)

module Triple = struct
  type ('a, _) t = 'a * 'a option * 'a list [@@deriving ptree]
end

let triples =
  Gen.triple floats (optional floats)
    (Gen.list ~size:(Gen.int_range 0 2) floats)

let law_triple ((_, b, l) as x) =
  cover "no option" (Option.is_none b);
  cover "an empty list" (List.is_empty l);
  agree
    P.(triple tensor (option tensor) (list tensor))
    (P.instantiate (module Triple))
    x

(* Parts that move between sets by their own walks. *)
module State = struct
  type 'd t = {
    steps : 'd Nx.int32_t;
    key : 'd Nx.Rng.key;
    index : 'd Kaun.Cache_index.t;
    weight : 'd Nx_quant.t;
    wide : (Nx.float32_elt, 'd) Nx_wide.t;
    packed : 'd Nx.packed;
  }
  [@@deriving ptree]
end

let walk_state_hand c (x : _ State.t) : _ State.t =
  let open P.Walk in
  let steps = field c "steps" tensor x.steps in
  let key = field c "key" Nx.Rng.walk x.key in
  let index = field c "index" Kaun.Cache_index.walk x.index in
  let weight = field c "weight" Nx_quant.walk x.weight in
  let wide = field c "wide" Nx_wide.walk x.wide in
  let packed =
    field c "packed" (fun c (Nx.P a) -> Nx.P (tensor c a)) x.packed
  in
  { steps; key; index; weight; wide; packed }

let state () : host State.t =
  {
    steps = i32 [| 3l |];
    key = Nx.Rng.key 1;
    index = Kaun.Cache_index.window 4 (Kaun.Cache_index.rows ~context:8 [| 2 |]);
    weight =
      Nx_quant.mxfp4
        ~scales:(Nx.zeros Nx.uint8 [| 1; 1 |])
        (Nx.zeros Nx.uint8 [| 1; 16 |]);
    wide = Nx_wide.v (f32 [| 1.; 2. |]);
    packed = Nx.P (Nx.create Nx.int64 [| 1 |] [| 5L |]);
  }

let test_state () = agree (P.v walk_state_hand) (P.v State.walk) (state ())

(* Its key opened from Nx. *)
module Held = struct
  open Nx

  type (_, 'd) t = { state : 'd State.t; spare : 'd Rng.key } [@@deriving ptree]
end

module Held_hand = struct
  type ('a, 'd) t = ('a, 'd) Held.t

  let walk c (x : _ t) : _ t =
    let open P.Walk in
    let state = field c "state" walk_state_hand x.state in
    let spare = field c "spare" Nx.Rng.walk x.spare in
    { state; spare }
end

(* A type without payload is described at any payload, here [f32]. *)
let held : ((f32, host) Held.t, host) P.t = P.instantiate (module Held)

let held_value () : (f32, host) Held.t =
  { state = state (); spare = Nx.Rng.key 2 }

let test_held () = agree (P.instantiate (module Held_hand)) held (held_value ())

type kind = Dense | Sparse of int [@ptree.int] [@@deriving ptree]

let walk_kind_hand c =
  let open P.Walk in
  function
  | Dense ->
      case c "Dense";
      Dense
  | Sparse k ->
      case c "Sparse";
      Sparse (int c k)

let kinds =
  Gen.(
    frequency
      [ (1, constant Dense); (1, map (fun k -> Sparse k) (int_range 0 3)) ])

module Config = struct
  type t = {
    depth : int; [@ptree.int]
    causal : bool; [@ptree.int]
    kind : kind;
    heads : (int[@ptree.int]) array;
  }
  [@@deriving ptree]
end

let walk_config_hand c (x : Config.t) : Config.t =
  let open P.Walk in
  let depth = field c "depth" int x.depth in
  let causal =
    field c "causal" (fun c b -> int c (Bool.to_int b) <> 0) x.causal
  in
  let kind = field c "kind" walk_kind_hand x.kind in
  let heads =
    field c "heads"
      (fun c a ->
        ignore (int c (Array.length a));
        Array.mapi (fun i n -> index c i int n) a)
      x.heads
  in
  { depth; causal; kind; heads }

let configs =
  let open Gen in
  let+ depth = int_range 0 4
  and+ causal = bool
  and+ kind = kinds
  and+ heads = array ~size:(int_range 0 2) (int_range 1 4) in
  { Config.depth; causal; kind; heads }

let law_config (x : Config.t) =
  cover "a dense kind" (x.kind = Dense);
  cover "causal" x.causal;
  cover "no head" (Array.length x.heads = 0);
  agree (P.v walk_config_hand) (P.v Config.walk) x

(* Generated values *)

(* The types the .mli gives the generated values, which the build checks:
   [Test_signature]'s interface is derived, its implementation derives what that
   interface declares, and each walk is taken here where only a walk of its
   documented type, at any payload and from any set to any other, is. *)
let _ :
    'e Nx.Placement.t ->
    ((float, 's, host) Nx.t, host) Test_signature.t ->
    ((float, 's, 'e) Nx.t, 'e) Test_signature.t =
  P.place (module Test_signature)

let _ :
    'e Nx.Placement.t -> host Test_signature.state -> 'e Test_signature.state =
  P.place_fixed Test_signature.walk_state

let _ :
    'e Nx.Placement.t -> host Test_signature.frozen -> 'e Test_signature.frozen
    =
  P.place_fixed Test_signature.walk_frozen

let _ : 'e Nx.Placement.t -> Test_signature.config -> Test_signature.config =
  P.place_fixed Test_signature.walk_config

(* How a part is walked *)

(* A declaration group: recursion, a type nested at another argument of itself,
   a member nested at a fixed array, and members of each arity. *)
module Tree = struct
  type ('a, 'd) t =
    | Leaf of 'a
    | Node of ('a, 'd) forest
    | Deep of ('a list, 'd) t

  and ('a, 'd) forest = ('a, 'd) t list

  and ('a, 'd) rooted = {
    root : ('a, 'd) t;
    frozen : ('d Nx.float32_t, 'd) t;
    state : 'd state;
    shape : shape;
  }

  and 'd state = { steps : 'd Nx.int32_t; index : 'd Kaun.Cache_index.t }
  and shape = { depth : int [@ptree.int] } [@@deriving ptree]
end

module Tree_hand = struct
  open Tree

  let rec walk :
      'a 'b 'd 'e. ('a, 'b, 'd, 'e) P.Walk.cursor -> ('a, 'd) t -> ('b, 'e) t =
   fun c ->
    let open P.Walk in
    function
    | Leaf x ->
        case c "Leaf";
        Leaf (leaf c x)
    | Node f ->
        case c "Node";
        Node (walk_forest c f)
    | Deep t ->
        case c "Deep";
        Deep (under walk (list leaf) c t)

  and walk_forest :
      'a 'b 'd 'e.
      ('a, 'b, 'd, 'e) P.Walk.cursor -> ('a, 'd) forest -> ('b, 'e) forest =
   fun c l -> P.Walk.list walk c l

  let walk_rooted c (x : _ rooted) : _ rooted =
    let open P.Walk in
    let root = field c "root" walk x.root in
    let frozen = field c "frozen" (under walk tensor) x.frozen in
    let state =
      field c "state"
        (fun c (s : _ state) : _ state ->
          let steps = field c "steps" tensor s.steps in
          let index = field c "index" Kaun.Cache_index.walk s.index in
          { steps; index })
        x.state
    in
    let shape =
      field c "shape"
        (fun c (s : shape) -> { depth = field c "depth" int s.depth })
        x.shape
    in
    { root; frozen; state; shape }
end

module Rooted = struct
  type ('a, 'd) t = ('a, 'd) Tree.rooted

  let walk = Tree.walk_rooted
end

module Rooted_hand = struct
  type ('a, 'd) t = ('a, 'd) Tree.rooted

  let walk = Tree_hand.walk_rooted
end

let rec tree_of : type a. int -> a Gen.t -> (a, host) Tree.t Gen.t =
 fun depth leaf ->
  let open Gen in
  let leaves = map (fun x -> Tree.Leaf x) leaf in
  if depth = 0 then leaves
  else
    frequency
      [
        (1, leaves);
        ( 1,
          map
            (fun l -> Tree.Node l)
            (list ~size:(int_range 0 2) (tree_of (depth - 1) leaf)) );
        ( 1,
          map
            (fun t -> Tree.Deep t)
            (tree_of (depth - 1) (list ~size:(int_range 0 2) leaf)) );
      ]

let rooted_hand : ((f32, host) Tree.rooted, host) P.t =
  P.instantiate (module Rooted_hand)

let rooteds =
  let open Gen in
  (let+ root = tree_of 3 floats
   and+ frozen = tree_of 2 floats
   and+ steps = int_range 0 2
   and+ depth = int_range 0 4 in
   {
     Tree.root;
     frozen;
     state =
       {
         steps = i32 (Array.init steps Int32.of_int);
         index = Kaun.Cache_index.rows ~context:8 [| 2 |];
       };
     shape = { depth };
   })
  |> with_pp (pp_visits rooted_hand)

let law_rooted (x : (f32, host) Tree.rooted) =
  cover "a leaf" (match x.root with Leaf _ -> true | _ -> false);
  cover "a node" (match x.root with Node _ -> true | _ -> false);
  cover "a deep tree" (match x.root with Deep _ -> true | _ -> false);
  agree rooted_hand (P.instantiate (module Rooted)) x

(* Every kind of part a record holds. *)
module Block = struct
  type ('a, 'd) t = {
    w : 'a;
    b : 'a option;
    layers : ('a, 'd) Kaun.Linear.t list;
    cache : ('a, 'd) Kaun.Attention.Cache.t;
    ends : 'a * 'a;
    stack : 'a array;
    frozen : ('d Nx.float32_t, 'd) Kaun.Linear.t;
    forest : ('a, 'd) Tree.forest;
    steps : 'd Nx.int32_t;
    key : 'd Nx.Rng.key;
    window : int option; [@ptree.int]
    causal : bool; [@ptree.int]
    sizes : (int[@ptree.int]) list;
    kind : kind;
    config : Config.t;
    custom : 'a;
        [@ptree.walk
          fun c x -> Nx.Ptree.Walk.field c "inner" Nx.Ptree.Walk.leaf x]
    name : string; [@ptree.skip]
  }
  [@@deriving ptree]
end

module Block_hand = struct
  type ('a, 'd) t = ('a, 'd) Block.t

  let walk c (x : _ t) : _ t =
    let open P.Walk in
    let w = field c "w" leaf x.w in
    let b = field c "b" (option leaf) x.b in
    let layers = field c "layers" (list Kaun.Linear.walk) x.layers in
    let cache = field c "cache" Kaun.Attention.Cache.walk x.cache in
    let ends =
      field c "ends"
        (fun c (p, q) ->
          let p = index c 0 leaf p in
          let q = index c 1 leaf q in
          (p, q))
        x.ends
    in
    let stack =
      field c "stack"
        (fun c a ->
          ignore (int c (Array.length a));
          Array.mapi (fun i x -> index c i leaf x) a)
        x.stack
    in
    let frozen = field c "frozen" (under Kaun.Linear.walk tensor) x.frozen in
    let forest = field c "forest" Tree_hand.walk_forest x.forest in
    let steps = field c "steps" tensor x.steps in
    let key = field c "key" Nx.Rng.walk x.key in
    let window = field c "window" (option int) x.window in
    let causal =
      field c "causal" (fun c b -> int c (Bool.to_int b) <> 0) x.causal
    in
    let sizes = field c "sizes" (list int) x.sizes in
    let kind = field c "kind" walk_kind_hand x.kind in
    let config = field c "config" walk_config_hand x.config in
    let custom =
      field c "custom" (fun c x -> field c "inner" leaf x) x.custom
    in
    {
      w;
      b;
      layers;
      cache;
      ends;
      stack;
      frozen;
      forest;
      steps;
      key;
      window;
      causal;
      sizes;
      kind;
      config;
      custom;
      name = x.name;
    }
end

let block_hand : ((f32, host) Block.t, host) P.t =
  P.instantiate (module Block_hand)

let block : ((f32, host) Block.t, host) P.t = P.instantiate (module Block)

let blocks =
  let open Gen in
  (let+ w = floats
   and+ b = optional floats
   and+ layers = list ~size:(int_range 0 2) (linear_of floats)
   and+ keys = floats
   and+ values = floats
   and+ e0 = floats
   and+ e1 = floats
   and+ stack = array ~size:(int_range 0 2) floats
   and+ frozen = linear_of floats
   and+ forest = list ~size:(int_range 0 2) (tree_of 1 floats)
   and+ steps = int_range 0 2
   and+ seed = int_range 0 3
   and+ window = optional (int_range 0 8)
   and+ causal = bool
   and+ sizes = list ~size:(int_range 0 2) (int_range 0 4)
   and+ kind = kinds
   and+ config = configs
   and+ custom = floats in
   {
     Block.w;
     b;
     layers;
     cache = { keys; values };
     ends = (e0, e1);
     stack;
     frozen;
     forest;
     steps = i32 (Array.init steps Int32.of_int);
     key = Nx.Rng.key seed;
     window;
     causal;
     sizes;
     kind;
     config;
     custom;
     name = "block";
   })
  |> with_pp (pp_visits block_hand)

let law_block (x : (f32, host) Block.t) =
  cover "an absent bias" (Option.is_none x.b);
  cover "a present bias" (Option.is_some x.b);
  cover "no layer" (x.layers = []);
  cover "layers" (x.layers <> []);
  cover "an empty array" (Array.length x.stack = 0);
  cover "an empty forest" (x.forest = []);
  cover "an absent window" (Option.is_none x.window);
  cover "a sparse kind" (x.kind <> Dense);
  cover "causal" x.causal;
  agree block_hand block x;
  equal ~msg:"payload positions" (list string)
    (positions (fun f -> P.Payload.fold (module Block_hand) f x))
    (positions (fun f -> P.Payload.fold (module Block) f x));
  equal ~msg:"the skipped name" string x.name
    (P.map block (fun _ a -> a) x).name

(* Variants name their case before their parts; an attribute after a
   constructor's single argument applies to it. *)
module Weight = struct
  type ('a, 'd) t =
    | Float of 'a
    | Mxfp4 of { blocks : 'd Nx.uint8_t; scales : 'd Nx.uint8_t }
    | Scaled of 'a * 'd Nx.float32_t
    | Window of int [@ptree.int]
    | Frozen of host Nx.float32_t [@ptree.skip]
    | Custom of 'a
        [@ptree.walk
          fun c x -> Nx.Ptree.Walk.field c "inner" Nx.Ptree.Walk.leaf x]
    | Tied
  [@@deriving ptree]
end

module Weight_hand = struct
  type ('a, 'd) t = ('a, 'd) Weight.t

  let walk c : _ t -> _ t =
    let open P.Walk in
    function
    | Float w ->
        case c "Float";
        Float (leaf c w)
    | Mxfp4 { blocks; scales } ->
        case c "Mxfp4";
        let blocks = field c "blocks" tensor blocks in
        let scales = field c "scales" tensor scales in
        Mxfp4 { blocks; scales }
    | Scaled (w, s) ->
        case c "Scaled";
        let w = index c 0 leaf w in
        let s = index c 1 tensor s in
        Scaled (w, s)
    | Window n ->
        case c "Window";
        Window (int c n)
    | Frozen x ->
        case c "Frozen";
        Frozen x
    | Custom x ->
        case c "Custom";
        Custom (field c "inner" leaf x)
    | Tied ->
        case c "Tied";
        Tied
end

(* Each part its own array, so a walk that swaps two parts of one type is
   seen. *)
let weights () : (string * (f32, host) Weight.t) list =
  let u8 () = Nx.create Nx.uint8 [| 1 |] [| 7 |] in
  [
    ("a constructor with one argument", Float (f32 [| 1. |]));
    ("an inline record", Mxfp4 { blocks = u8 (); scales = u8 () });
    ("a constructor with two arguments", Scaled (f32 [| 1. |], f32 [| 2. |]));
    ("[@ptree.int] after the argument", Window 4);
    ("[@ptree.skip] after the argument", Frozen (f32 [| 3. |]));
    ("[@ptree.walk f] after the argument", Custom (f32 [| 4. |]));
    ("a constant constructor", Tied);
  ]

let test_weight (_, x) =
  agree (P.instantiate (module Weight_hand)) (P.instantiate (module Weight)) x

module Aliases = struct
  type 'd t = {
    f64 : 'd Nx.float64_t;
    f32 : 'd Nx.float32_t;
    f16 : 'd Nx.float16_t;
    bf16 : 'd Nx.bfloat16_t;
    f8e4m3fn : 'd Nx.float8_e4m3fn_t;
    f8e5m2 : 'd Nx.float8_e5m2_t;
    f4e2m1fn : 'd Nx.float4_e2m1fn_t;
    i64 : 'd Nx.int64_t;
    u64 : 'd Nx.uint64_t;
    i32 : 'd Nx.int32_t;
    u32 : 'd Nx.uint32_t;
    i16 : 'd Nx.int16_t;
    u16 : 'd Nx.uint16_t;
    i8 : 'd Nx.int8_t;
    u8 : 'd Nx.uint8_t;
    i4 : 'd Nx.int4_t;
    u4 : 'd Nx.uint4_t;
    c128 : 'd Nx.complex128_t;
    c64 : 'd Nx.complex64_t;
    b : 'd Nx.bool_t;
    bit : 'd Nx.bit_t;
    any : (float, Nx.float64_elt, 'd) Nx.t;
  }
  [@@deriving ptree]
end

module Opened = struct
  open Nx

  type 'd t = 'd Aliases.t = {
    f64 : 'd float64_t;
    f32 : 'd float32_t;
    f16 : 'd float16_t;
    bf16 : 'd bfloat16_t;
    f8e4m3fn : 'd float8_e4m3fn_t;
    f8e5m2 : 'd float8_e5m2_t;
    f4e2m1fn : 'd float4_e2m1fn_t;
    i64 : 'd int64_t;
    u64 : 'd uint64_t;
    i32 : 'd int32_t;
    u32 : 'd uint32_t;
    i16 : 'd int16_t;
    u16 : 'd uint16_t;
    i8 : 'd int8_t;
    u8 : 'd uint8_t;
    i4 : 'd int4_t;
    u4 : 'd uint4_t;
    c128 : 'd complex128_t;
    c64 : 'd complex64_t;
    b : 'd bool_t;
    bit : 'd bit_t;
    any : (float, float64_elt, 'd) Nx.t;
  }
  [@@deriving ptree]
end

(* Nx's aliases (packages/nx/lib/nx.mli), each a fixed array of its own. *)
let aliases () : host Aliases.t =
  let z dt = Nx.zeros dt [| 1 |] in
  {
    f64 = z Nx.float64;
    f32 = z Nx.float32;
    f16 = z Nx.float16;
    bf16 = z Nx.bfloat16;
    f8e4m3fn = z Nx.float8_e4m3fn;
    f8e5m2 = z Nx.float8_e5m2;
    f4e2m1fn = z Nx.float4_e2m1fn;
    i64 = z Nx.int64;
    u64 = z Nx.uint64;
    i32 = z Nx.int32;
    u32 = z Nx.uint32;
    i16 = z Nx.int16;
    u16 = z Nx.uint16;
    i8 = z Nx.int8;
    u8 = z Nx.uint8;
    i4 = z Nx.int4;
    u4 = z Nx.uint4;
    c128 = z Nx.complex128;
    c64 = z Nx.complex64;
    b = z Nx.bool;
    bit = z Nx.bit;
    any = z Nx.float64;
  }

let alias_fields =
  [ "f64"; "f32"; "f16"; "bf16"; "f8e4m3fn"; "f8e5m2"; "f4e2m1fn"; "i64";
    "u64"; "i32"; "u32"; "i16"; "u16"; "i8"; "u8"; "i4"; "u4"; "c128"; "c64";
    "b"; "bit"; "any" ]
[@@ocamlformat "disable"]

let words s x = List.map (Format.asprintf "%a" P.pp_visit) (P.visits s x)

let test_aliases () =
  let leaves = List.map (fun f -> f ^ ": a leaf") alias_fields in
  equal ~msg:"qualified" (list string) leaves
    (words (P.v Aliases.walk) (aliases ()));
  equal ~msg:"opened" (list string) leaves
    (words (P.v Opened.walk) (aliases ()))

(* A training state: a model, an optimiser state nested over it, a key. *)
module Train = struct
  type ('a, 'd) t = {
    params : ('a, 'd) Example.t;
    opt : (('a, 'd) Example.t, 'd) Vega.Adam_state.t;
    key : 'd Nx.Rng.key;
  }
  [@@deriving ptree]
end

module Train_hand = struct
  type ('a, 'd) t = ('a, 'd) Train.t

  let walk c (x : _ t) : _ t =
    let open P.Walk in
    let params = field c "params" Example_hand.walk x.params in
    let opt =
      field c "opt" (under Vega.Adam_state.walk Example_hand.walk) x.opt
    in
    let key = field c "key" Nx.Rng.walk x.key in
    { params; opt; key }
end

let test_train () =
  let k = Nx.Rng.split (Nx.Rng.key 0) in
  let params : (f32, host) Example.t =
    {
      l1 = Kaun.Linear.init ~key:k.(0) ~inputs:2 ~outputs:2;
      l2 = Kaun.Linear.init ~key:k.(1) ~inputs:2 ~outputs:1;
      steps = i32 [| 1l |];
      window = Some 2;
    }
  in
  let opt = Vega.adam_init (P.instantiate (module Example)) params in
  agree
    (P.instantiate (module Train_hand))
    (P.instantiate (module Train))
    { params; opt; key = Nx.Rng.key 3 }

(* Attributes *)

(* A nested structure's payload argument is a part, so it takes an attribute. *)
module Inner = struct
  type ('a, 'd) t = {
    w : 'a;
    frozen :
      ( ('d Nx.float32_t
        [@ptree.walk
          fun c x -> Nx.Ptree.Walk.field c "inner" Nx.Ptree.Walk.tensor x]),
        'd )
      Kaun.Linear.t;
  }
  [@@deriving ptree]
end

module Inner_hand = struct
  type ('a, 'd) t = ('a, 'd) Inner.t

  let walk c (x : _ t) : _ t =
    let open P.Walk in
    let w = field c "w" leaf x.w in
    let frozen =
      field c "frozen"
        (under Kaun.Linear.walk (fun c x -> field c "inner" tensor x))
        x.frozen
    in
    { w; frozen }
end

let test_inner () =
  agree
    (P.instantiate (module Inner_hand))
    (P.instantiate (module Inner))
    { w = f32 [| 1. |]; frozen = { w = f32 [| 2. |]; b = Some (f32 [| 3. |]) } }

(* Each form the .mli refuses without an attribute, skipped, and one walked. The
   Nx types the deriver refuses by name are named here too, so a rename in Nx
   fails this build before the deriver goes stale. *)
module Skipped = struct
  type ('a, 'd) t = {
    w : 'a;
    count : int; [@ptree.skip]
    name : string; [@ptree.skip]
    scale : float; [@ptree.skip]
    dtype : (float, Nx.float32_elt) Nx.dtype; [@ptree.skip]
    format : (float, Nx.float32_elt) Nx.Dtype.t; [@ptree.skip]
    at : host Nx.Placement.t; [@ptree.skip]
    mesh : host Nx.Mesh.t; [@ptree.skip]
    devices : host Nx.Devices.t; [@ptree.skip]
    mask : host Nx.bool_t; [@ptree.skip]
    callback : int -> int; [@ptree.skip]
    getter : < get : int >; [@ptree.skip]
    choice : [ `A | `B ]; [@ptree.skip]
    m : (module Set.OrderedType); [@ptree.skip]
    poly : 'x. 'x list; [@ptree.skip]
    later : int lazy_t; [@ptree.skip]
    outcome : (int, string) result; [@ptree.skip]
    cell : 'd Nx.float32_t ref;
        [@ptree.walk fun c r -> ref (Nx.Ptree.Walk.tensor c !r)]
  }
  [@@deriving ptree]
end

module Skipped_hand = struct
  type ('a, 'd) t = ('a, 'd) Skipped.t

  let walk c (x : _ t) : _ t =
    let open P.Walk in
    let w = field c "w" leaf x.w in
    let cell = field c "cell" (fun c r -> ref (tensor c !r)) x.cell in
    { x with w; cell }
end

let test_skipped () =
  let x : (f32, host) Skipped.t =
    {
      w = f32 [| 1. |];
      count = 1;
      name = "skipped";
      scale = 0.5;
      dtype = Nx.float32;
      format = Nx.float32;
      at = Nx.Host.on;
      mesh = Nx.Mesh.v Nx.Host.v [ ("x", 1) ];
      devices = Nx.Host.v;
      mask = Nx.zeros Nx.bool [| 1 |];
      callback = succ;
      getter =
        object
          method get = 1
        end;
      choice = `A;
      m = (module Int);
      poly = [];
      later = lazy 1;
      outcome = Ok 1;
      cell = ref (f32 [| 2. |]);
    }
  in
  agree (P.instantiate (module Skipped_hand)) (P.instantiate (module Skipped)) x

(* What a description, place and cast make of each declaration. *)

let block_example () : (f32, host) Block.t =
  {
    w = f32 [| 1. |];
    b = Some (f32 [| 2. |]);
    layers = [ { w = f32 [| 3. |]; b = None } ];
    cache = { keys = f32 [| 4. |]; values = f32 [| 5. |] };
    ends = (f32 [| 6. |], f32 [| 7. |]);
    stack = [| f32 [| 8. |] |];
    frozen = { w = f32 [| 9. |]; b = Some (f32 [| 10. |]) };
    forest = [ Leaf (f32 [| 11. |]) ];
    steps = i32 [| 1l |];
    key = Nx.Rng.key 0;
    window = Some 4;
    causal = true;
    sizes = [ 2 ];
    kind = Sparse 1;
    config = { depth = 2; causal = false; kind = Dense; heads = [| 1 |] };
    custom = f32 [| 12. |];
    name = "block";
  }

let test_place () =
  let on_s : ((S.d Nx.float32_t, S.d) Block.t, S.d) P.t =
    P.instantiate (module Block)
  in
  all_on S.on (placements on_s (P.place (module Block) S.on (block_example ())))

let test_cast () =
  let x = block_example () in
  let y = P.cast (module Block) Nx.float16 x in
  equal ~msg:"a position" string "float16" (Nx.Dtype.name (Nx.dtype y.w));
  equal ~msg:"fixed arrays" arrays
    [ Nx.P x.frozen.w; Nx.P x.steps ]
    [ Nx.P y.frozen.w; Nx.P y.steps ]

let test_held_place () =
  let on_s : ((S.d Nx.float32_t, S.d) Held.t, S.d) P.t =
    P.instantiate (module Held)
  in
  let x = held_value () in
  all_on S.on (placements on_s (P.place (module Held) S.on x));
  let y = P.cast (module Held) Nx.float16 x in
  equal ~msg:"a cast keeps every array" arrays
    (fst (P.flatten held x))
    (fst (P.flatten (P.instantiate (module Held)) y))

let test_place_fixed () =
  let y = P.place_fixed State.walk S.on (state ()) in
  all_on S.on (placements (P.v State.walk) y)

let () =
  exit
    (run "ppx_ptree"
       [
         group "[@@deriving ptree]"
           [
             prop "the example derives the walk it shows" examples law_example;
             test "the guide's block derives its hand-written walk" test_guide;
           ];
         group "declarations"
           [
             prop "('a, _) t holds positions of the payload alone" triples
               law_triple;
             test "(_, 'd) t holds fixed arrays and keys alone" test_held;
             test "'d t holds fixed arrays, keys, quantised and wide parts"
               test_state;
             prop "t holds integers and cases" configs law_config;
           ];
         group "generated values"
           [
             test "place moves every array of ('a, 'd) t, fixed ones too"
               test_place;
             test "cast casts the payload of ('a, 'd) t and keeps fixed arrays"
               test_cast;
             test "place and cast take the module of (_, 'd) t" test_held_place;
             test "place_fixed takes the walk of 'd t" test_place_fixed;
           ];
         group "how a part is walked"
           [
             prop "a record walks every kind of part in declaration order"
               blocks law_block;
             cases "a variant reports its case, then walks its arguments"
               ~id:fst (weights ()) test_weight;
             prop "a declaration group walks its types with each other's walks"
               rooteds law_rooted;
             test "every tensor alias, qualified or opened, is a fixed array"
               test_aliases;
             test "a structure nested over another walks with Walk.under"
               test_train;
           ];
         group "attributes"
           [
             test "a nested structure's payload argument takes [@ptree.walk]"
               test_inner;
             test
               "each refused form derives under [@ptree.skip] or [@ptree.walk]"
               test_skipped;
           ];
       ])
