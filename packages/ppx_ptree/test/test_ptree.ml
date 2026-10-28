(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Each derived walk visits what a hand-written walk of the same type visits:
   the same leaves and reports, at the same paths, in the same order, and it
   rebuilds a value from its own arrays. Each declaration shape, ('a, 'd) t, (_,
   'd) t, 'd t and t, moves its fixed arrays and keys between sets. *)

open Windtrap
module P = Nx.Ptree

let device = Result.get_ok (Rig.memory_device "ppx_ptree")

module S = (val Nx.devices [ device ])

type host = Nx.host
type f32 = host Nx.float32_t

let f32 xs : f32 = Nx.create Nx.float32 [| Array.length xs |] xs
let i32 xs : host Nx.int32_t = Nx.create Nx.int32 [| Array.length xs |] xs
let words s x = List.map (Format.asprintf "%a" P.pp_visit) (P.visits s x)

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

let pp_leaf ppf (Nx.P a) =
  Format.fprintf ppf "%s [%s]"
    (Nx.Dtype.name (Nx.dtype a))
    (String.concat "; " (Array.to_list (Array.map string_of_int (Nx.shape a))))

let leaves =
  Testable.make
    ~pp:(Format.pp_print_list ~pp_sep:Format.pp_print_space pp_leaf)
    ~equal:same

let placement () = Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal

let pp_visits s ppf x =
  Format.pp_print_list ~pp_sep:Format.pp_print_space P.pp_visit ppf
    (P.visits s x)

(* [agree ~msg hand derived x]: [derived] visits what [hand] visits, and
   rebuilds [x] from [x]'s own arrays. *)
let agree ~msg hand derived x =
  equal ~msg:(msg ^ ": visits") (list visit) (P.visits hand x)
    (P.visits derived x);
  let arrays, _ = P.flatten derived x in
  let y = P.rebuild derived ~like:x arrays in
  equal ~msg:(msg ^ ": rebuild") leaves arrays (fst (P.flatten derived y))

(* [positions map] is the paths at which [map] passes its function a payload, in
   walk order. *)
let positions map =
  let acc = ref [] in
  ignore
    (map (fun p v ->
         acc := P.Path.to_string p :: !acc;
         v));
  List.rev !acc

(* [placements s x] is the placement of each of [x]'s arrays, in walk order. *)
let placements s x =
  List.rev (P.fold s (fun _ a acc -> Nx.placement a :: acc) x [])

let all_on p ~msg ps =
  equal ~msg (list (placement ())) (List.map (fun _ -> p) ps) ps

let floats =
  Gen.(map (fun n -> f32 (Array.init n float_of_int)) (int_range 0 2))

let linear_of position =
  let open Gen in
  let+ w = position and+ b = option position in
  { Kaun.Linear.w; b }

(* ('a, 'd) t: every kind of part. *)

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

module Block = struct
  type ('a, 'd) t = {
    w : 'a;
    b : 'a option;
    layers : ('a, 'd) Kaun.Linear.t list;
    cache : ('a, 'd) Kaun.Attention.Cache.t;
    ends : 'a * 'a;
    stack : 'a array;
    frozen : ('d Nx.float32_t, 'd) Kaun.Linear.t;
    steps : 'd Nx.int32_t;
    key : 'd Nx.Rng.key;
    window : int option; [@ptree.int]
    causal : bool; [@ptree.int]
    sizes : (int[@ptree.int]) list;
    kind : kind;
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
    let steps = field c "steps" tensor x.steps in
    let key = field c "key" Nx.Rng.walk x.key in
    let window = field c "window" (option int) x.window in
    let causal =
      field c "causal" (fun c b -> int c (Bool.to_int b) <> 0) x.causal
    in
    let sizes = field c "sizes" (list int) x.sizes in
    let kind = field c "kind" walk_kind_hand x.kind in
    {
      w;
      b;
      layers;
      cache;
      ends;
      stack;
      frozen;
      steps;
      key;
      window;
      causal;
      sizes;
      kind;
      name = x.name;
    }
end

let block_hand : ((f32, host) Block.t, host) P.t =
  P.instantiate (module Block_hand)

let block : ((f32, host) Block.t, host) P.t = P.instantiate (module Block)

let kinds =
  Gen.(
    frequency
      [ (1, constant Dense); (1, map (fun k -> Sparse k) (int_range 0 3)) ])

let blocks =
  let open Gen in
  (let+ w = floats
   and+ b = option floats
   and+ layers = list ~size:(int_range 0 2) (linear_of floats)
   and+ keys = floats
   and+ values = floats
   and+ e0 = floats
   and+ e1 = floats
   and+ stack = array ~size:(int_range 0 2) floats
   and+ frozen = linear_of floats
   and+ steps = int_range 0 2
   and+ seed = int_range 0 3
   and+ window = option (int_range 0 8)
   and+ causal = bool
   and+ sizes = list ~size:(int_range 0 2) (int_range 0 4)
   and+ kind = kinds in
   {
     Block.w;
     b;
     layers;
     cache = { keys; values };
     ends = (e0, e1);
     stack;
     frozen;
     steps = i32 (Array.init steps Int32.of_int);
     key = Nx.Rng.key seed;
     window;
     causal;
     sizes;
     kind;
     name = "block";
   })
  |> with_pp (pp_visits block_hand)

let law_block (x : (f32, host) Block.t) =
  cover "an absent bias" (Option.is_none x.b);
  cover "a present bias" (Option.is_some x.b);
  cover "no layer" (x.layers = []);
  cover "layers" (x.layers <> []);
  cover "an empty array" (Array.length x.stack = 0);
  cover "an absent window" (Option.is_none x.window);
  cover "a sparse kind" (x.kind <> Dense);
  cover "causal" x.causal;
  agree ~msg:"block" block_hand block x;
  equal ~msg:"payload positions" (list string)
    (positions (fun f -> P.Payload.map (module Block_hand) f x))
    (positions (fun f -> P.Payload.map (module Block) f x));
  let y = P.map block (fun _ a -> a) x in
  equal ~msg:"skip copies the part" string x.name y.name;
  equal ~msg:"a reported bool rebuilds" bool x.causal y.causal;
  equal ~msg:"reported ints rebuild" (list int) x.sizes y.sizes

let example =
  {
    Block.w = f32 [| 1. |];
    b = Some (f32 [| 2. |]);
    layers = [ { w = f32 [| 3. |]; b = None } ];
    cache = { keys = f32 [| 4. |]; values = f32 [| 5. |] };
    ends = (f32 [| 6. |], f32 [| 7. |]);
    stack = [| f32 [| 8. |] |];
    frozen = { w = f32 [| 9. |]; b = Some (f32 [| 10. |]) };
    steps = i32 [| 1l |];
    key = Nx.Rng.key 0;
    window = Some 4;
    causal = true;
    sizes = [ 2 ];
    kind = Sparse 1;
    name = "block";
  }

let test_block_place () =
  let y = P.place (module Block) S.on example in
  let on_s : ((S.d Nx.float32_t, S.d) Block.t, S.d) P.t =
    P.instantiate (module Block)
  in
  all_on S.on ~msg:"every array, fixed ones and the key too" (placements on_s y);
  equal ~msg:"visits" (list visit) (P.visits block example) (P.visits on_s y);
  equal ~msg:"key words" (array int32)
    (Nx.to_array (Nx.Rng.to_tensor example.key))
    (Nx.to_array (Nx.Rng.to_tensor y.key));
  let back = P.place (module Block) Nx.Host.on y in
  all_on Nx.Host.on ~msg:"back on the host" (placements block back)

let test_block_cast () =
  let y = P.cast (module Block) Nx.float16 example in
  equal ~msg:"a position is cast" string "float16"
    (Nx.Dtype.name (Nx.dtype y.w));
  equal ~msg:"a nested position is cast" string "float16"
    (Nx.Dtype.name (Nx.dtype y.cache.keys));
  equal ~msg:"a frozen layer is kept" leaves [ Nx.P example.frozen.w ]
    [ Nx.P y.frozen.w ];
  equal ~msg:"a fixed array is kept" leaves [ Nx.P example.steps ]
    [ Nx.P y.steps ]

(* Variants name their case before their parts; an attribute after a
   constructor's single argument applies to it. *)

module Weight = struct
  type ('a, 'd) t =
    | Float of 'a
    | Mxfp4 of { blocks : 'd Nx.uint8_t; scales : 'd Nx.uint8_t }
    | Scaled of 'a * 'd Nx.float32_t
    | Window of int [@ptree.int]
    | Frozen of host Nx.float32_t [@ptree.skip]
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
    | Tied ->
        case c "Tied";
        Tied
end

let weights : (string * (f32, host) Weight.t) list =
  let u8 = Nx.create Nx.uint8 [| 1 |] [| 7 |] in
  [
    ("a constructor with one argument", Float (f32 [| 1. |]));
    ("an inline record", Mxfp4 { blocks = u8; scales = u8 });
    ("a constructor with two arguments", Scaled (f32 [| 1. |], f32 [| 2. |]));
    ("[@ptree.int] after the argument", Window 4);
    ("[@ptree.skip] after the argument", Frozen (f32 [| 3. |]));
    ("a constant constructor", Tied);
  ]

let test_weight (msg, x) =
  agree ~msg
    (P.instantiate (module Weight_hand))
    (P.instantiate (module Weight))
    x

(* Recursion, and a type nested at another argument of itself. *)

module Tree = struct
  type ('a, 'd) t =
    | Leaf of 'a
    | Node of ('a, 'd) t list
    | Deep of ('a list, 'd) t
  [@@deriving ptree]
end

module Tree_hand = struct
  type ('a, 'd) t = ('a, 'd) Tree.t

  let rec walk :
      'a 'b 'd 'e. ('a, 'b, 'd, 'e) P.Walk.cursor -> ('a, 'd) t -> ('b, 'e) t =
   fun c ->
    let open P.Walk in
    function
    | Leaf x ->
        case c "Leaf";
        Leaf (leaf c x)
    | Node l ->
        case c "Node";
        Node (list walk c l)
    | Deep t ->
        case c "Deep";
        Deep (under walk (list leaf) c t)
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

let tree_hand : ((f32, host) Tree.t, host) P.t =
  P.instantiate (module Tree_hand)

let tree : ((f32, host) Tree.t, host) P.t = P.instantiate (module Tree)
let trees = tree_of 3 floats |> Gen.with_pp (pp_visits tree_hand)

let law_tree (x : (f32, host) Tree.t) =
  cover "a leaf" (match x with Leaf _ -> true | _ -> false);
  cover "a node" (match x with Node _ -> true | _ -> false);
  cover "a deep tree" (match x with Deep _ -> true | _ -> false);
  agree ~msg:"tree" tree_hand tree x;
  equal ~msg:"payload positions" (list string)
    (positions (fun f -> P.Payload.map (module Tree_hand) f x))
    (positions (fun f -> P.Payload.map (module Tree) f x))

(* 'd t: fixed arrays only, with parts that move between sets by their own
   walks. (_, 'd) t: no payload, its key opened from Nx. *)

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

module Held = struct
  open Nx

  type (_, 'd) t = { state : 'd State.t; spare : 'd Rng.key } [@@deriving ptree]
end

let state : host State.t =
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

let test_state () =
  agree ~msg:"state" (P.v walk_state_hand) (P.v State.walk) state;
  agree ~msg:"held"
    (P.instantiate (module Held_hand))
    held
    { state; spare = Nx.Rng.key 2 }

let test_state_place () =
  let x : (f32, host) Held.t = { state; spare = Nx.Rng.key 2 } in
  let y = P.place (module Held) S.on x in
  let on_s : ((S.d Nx.float32_t, S.d) Held.t, S.d) P.t =
    P.instantiate (module Held)
  in
  all_on S.on ~msg:"every array of the state and both keys" (placements on_s y);
  equal ~msg:"visits" (list visit) (P.visits held x) (P.visits on_s y);
  equal ~msg:"the number" (array float_exact) [| 1.; 2. |]
    (Nx.to_array (Nx_wide.hi y.state.wide));
  let back = P.place (module Held) Nx.Host.on y in
  all_on Nx.Host.on ~msg:"back on the host" (placements held back)

(* Every tensor alias of Nx, qualified and opened. *)

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

let alias_fields =
  [ "f64"; "f32"; "f16"; "bf16"; "f8e4m3fn"; "f8e5m2"; "f4e2m1fn"; "i64";
    "u64"; "i32"; "u32"; "i16"; "u16"; "i8"; "u8"; "i4"; "u4"; "c128"; "c64";
    "b"; "bit"; "any" ]
[@@ocamlformat "disable"]

let aliases : host Aliases.t =
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

module Aliases_s = struct
  type (_, 'd) t = 'd Aliases.t

  let walk = Aliases.walk
end

let test_aliases () =
  let expected = List.map (fun f -> f ^ ": a leaf") alias_fields in
  equal ~msg:"qualified" (list string) expected
    (words (P.v Aliases.walk) aliases);
  equal ~msg:"opened" (list string) expected (words (P.v Opened.walk) aliases);
  let y = P.place (module Aliases_s) S.on aliases in
  all_on S.on ~msg:"each alias is a fixed array"
    (placements (P.v Aliases.walk) y)

(* t: integers and cases, no array. *)

module Config = struct
  type t = {
    depth : int; [@ptree.int]
    kind : kind;
    heads : (int[@ptree.int]) array;
  }
  [@@deriving ptree]
end

let walk_config_hand c (x : Config.t) : Config.t =
  let open P.Walk in
  let depth = field c "depth" int x.depth in
  let kind = field c "kind" walk_kind_hand x.kind in
  let heads =
    field c "heads"
      (fun c a ->
        ignore (int c (Array.length a));
        Array.mapi (fun i n -> index c i int n) a)
      x.heads
  in
  { depth; kind; heads }

let configs =
  let open Gen in
  let+ depth = int_range 0 4
  and+ kind = kinds
  and+ heads = array ~size:(int_range 0 2) (int_range 1 4) in
  { Config.depth; kind; heads }

let law_config (x : Config.t) =
  cover "a dense kind" (x.kind = Dense);
  cover "no head" (Array.length x.heads = 0);
  agree ~msg:"config" (P.v walk_config_hand) (P.v Config.walk) x

(* A training state: a model, an optimiser state nested over it, and a key. *)

module Mlp = struct
  type ('a, 'd) t = { l1 : ('a, 'd) Kaun.Linear.t; l2 : ('a, 'd) Kaun.Linear.t }
  [@@deriving ptree]

  let apply m x = Kaun.Linear.apply m.l2 (Kaun.Linear.apply m.l1 x)
end

module Mlp_hand = struct
  type ('a, 'd) t = ('a, 'd) Mlp.t

  let walk c (m : _ t) : _ t =
    let open P.Walk in
    let l1 = field c "l1" Kaun.Linear.walk m.l1 in
    let l2 = field c "l2" Kaun.Linear.walk m.l2 in
    { l1; l2 }
end

module Train = struct
  type ('a, 'd) t = {
    params : ('a, 'd) Mlp.t;
    opt : (('a, 'd) Mlp.t, 'd) Vega.Adam_state.t;
    key : 'd Nx.Rng.key;
  }
  [@@deriving ptree]
end

module Train_hand = struct
  type ('a, 'd) t = ('a, 'd) Train.t

  let walk c (x : _ t) : _ t =
    let open P.Walk in
    let params = field c "params" Mlp_hand.walk x.params in
    let opt = field c "opt" (under Vega.Adam_state.walk Mlp_hand.walk) x.opt in
    let key = field c "key" Nx.Rng.walk x.key in
    { params; opt; key }
end

let mlp : ((f32, host) Mlp.t, host) P.t = P.instantiate (module Mlp)

let params =
  let k = Nx.Rng.split (Nx.Rng.key 0) in
  {
    Mlp.l1 = Kaun.Linear.init ~key:k.(0) ~inputs:2 ~outputs:2;
    l2 = Kaun.Linear.init ~key:k.(1) ~inputs:2 ~outputs:1;
  }

let train : (f32, host) Train.t =
  { params; opt = Vega.adam_init mlp params; key = Nx.Rng.key 3 }

let test_train () =
  agree ~msg:"train"
    (P.instantiate (module Train_hand))
    (P.instantiate (module Train))
    train;
  let y = P.place (module Train) S.on train in
  let on_s : ((S.d Nx.float32_t, S.d) Train.t, S.d) P.t =
    P.instantiate (module Train)
  in
  all_on S.on ~msg:"the moments, the step and the key" (placements on_s y)

let test_grad () =
  let loss m = Nx.sum (Mlp.apply m (f32 [| 1.; 2. |])) in
  let hand = Rune.grad (P.instantiate (module Mlp_hand)) loss params in
  let derived = Rune.grad mlp loss params in
  equal ~msg:"gradients" (array float_exact) (Nx.to_array hand.l1.w)
    (Nx.to_array derived.l1.w)

(* The guide's example (packages/nx/doc/06-structures.md): the derived [Block]
   visits what the hand-written one visits. *)

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
  let block =
    {
      Guide.Block_hand.proj = { w = f32 [| 1. |]; b = Some (f32 [| 2. |]) };
      steps = Nx.zeros Nx.int32 [||];
      window = Some 4;
    }
  in
  agree ~msg:"guide block"
    (P.instantiate (module Guide.Block_hand))
    (P.instantiate (module Guide.Block))
    block

let () =
  exit
    (run "ppx_ptree"
       [
         group "('a, 'd) t"
           [
             prop "a block's derived walk is the hand-written one" blocks
               law_block;
             test "place moves fixed arrays and keys with the payload"
               test_block_place;
             test "cast casts the payload and keeps fixed arrays"
               test_block_cast;
             cases ~name:fst "variants" weights test_weight;
             prop "a recursive type nested at another argument" trees law_tree;
             test "a state nested over a model with Walk.under" test_train;
             test "the guide's block" test_guide;
           ];
         group "'d t and (_, 'd) t"
           [
             test "fixed arrays, keys, quantised and wide parts" test_state;
             test "place moves every part through the walk" test_state_place;
             test "every tensor alias, qualified and opened" test_aliases;
           ];
         group "t" [ prop "integers and cases" configs law_config ];
         group "transformations" [ test "grad" test_grad ];
       ])
