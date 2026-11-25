(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx.Ptree, in nx.mli's order: paths, walks, descriptions, maps and folds,
   donation, placement and casts, payloads, a value's vector arithmetic, visits,
   signatures and flattening. *)

open Windtrap
open Nx_support
module D = Nx_array.Dtype
module P = Nx.Ptree

let m = Nx_support.memory

module S = (val Nx.devices [ m 0 ])
module S2 = (val Nx.devices [ m 0; m 1 ])

type host = Nx.host
type f32 = host Nx.float32_t

let invalid ~by f = raises_match (Exn.invalid_arg ~sub:(by ^ ": ")) f

let message f =
  match f () with _ -> "no exception" | exception Invalid_argument m -> m

let f32 xs : f32 = Nx.create Nx.float32 [| Array.length xs |] xs
let f64 xs : host Nx.float64_t = Nx.create Nx.float64 [| Array.length xs |] xs
let i32 xs : host Nx.int32_t = Nx.create Nx.int32 [| Array.length xs |] xs
let words s x = List.map (Format.asprintf "%a" P.pp_visit) (P.visits s x)

(* Leaves that are the same values, in order. *)
let same xs ys =
  List.length xs = List.length ys
  && List.for_all2
       (fun (Nx.P a) (Nx.P b) ->
         match D.equal_witness (Nx.dtype a) (Nx.dtype b) with
         | Some Equal -> a == b
         | None -> false)
       xs ys

let pp_leaf ppf (Nx.P a) =
  Format.fprintf ppf "%s %a" (D.name (Nx.dtype a)) pp_ints (Nx.shape a)

(* Arrays that are the same values ([==]), in order. *)
let leaves =
  Testable.make
    ~pp:(Format.pp_print_list ~pp_sep:Format.pp_print_space pp_leaf)
    ~equal:same

let leaf = Testable.make ~pp:pp_leaf ~equal:(fun a b -> same [ a ] [ b ])
let is_on p a = equal (placement ()) p (Nx.placement a)

let visit_equal a b =
  match (a, b) with
  | P.Leaf p, P.Leaf q -> P.Path.equal p q
  | P.Report (p, r), P.Report (q, r') -> P.Path.equal p q && r = r'
  | _ -> false

let visit = Testable.make ~pp:P.pp_visit ~equal:visit_equal
let path_t = Testable.make ~pp:P.Path.pp ~equal:P.Path.equal

let pp_seg ppf = function
  | P.Path.Field s -> Format.fprintf ppf "Field %S" s
  | P.Path.Index i -> Format.fprintf ppf "Index %d" i

let seg_t = Testable.make ~pp:pp_seg ~equal:( = )

(* Structures *)

type 'a linear = { w : 'a; b : 'a option }

module Linear = struct
  type ('a, _) t = 'a linear

  let walk c { w; b } =
    let open P.Walk in
    let w = field c "w" leaf w in
    let b = field c "b" (option leaf) b in
    { w; b }
end

type kind = Dense | Sparse of int

(* Every kind of part: the parameter's positions in a list of records and in a
   pair, an array of a fixed type, an integer and a variant. *)
type ('a, 'd) block = {
  layers : 'a linear list;
  steps : 'd Nx.int32_t;
  window : int;
  kind : kind;
  ends : 'a * 'a;
}

module Block = struct
  type ('a, 'd) t = ('a, 'd) block

  let walk c x =
    let open P.Walk in
    let layers = field c "layers" (list Linear.walk) x.layers in
    let steps = field c "steps" tensor x.steps in
    let window = field c "window" int x.window in
    let kind =
      field c "kind"
        (fun c -> function
          | Dense ->
              case c "dense";
              Dense
          | Sparse k ->
              case c "sparse";
              Sparse (int c k))
        x.kind
    in
    let ends =
      field c "ends"
        (fun c (a, b) ->
          let a = index c 0 leaf a in
          let b = index c 1 leaf b in
          (a, b))
        x.ends
    in
    { layers; steps; window; kind; ends }
end

(* A part of a fixed type walked by its own walk, across sets. *)
type ('a, 'd) keyed = { held : 'a; key : 'd Nx.Rng.key }

module Keyed = struct
  type ('a, 'd) t = ('a, 'd) keyed

  let walk c x =
    let open P.Walk in
    let held = field c "held" leaf x.held in
    let key = field c "key" Nx.Rng.walk x.key in
    { held; key }
end

(* [Block] with a fixed array at each position: a structure of fixed arrays
   only. *)
let walk_frozen c x = P.Walk.under Block.walk P.Walk.tensor c x

let walk_two c (a, b) =
  let open P.Walk in
  let a = index c 0 tensor a in
  let b = index c 1 tensor b in
  (a, b)

let linear : (f32 linear, host) P.t = P.instantiate (module Linear)
let block : ((f32, host) block, host) P.t = P.instantiate (module Block)

let example =
  {
    layers =
      [
        { w = f32 [| 1.; 2. |]; b = Some (f32 [| 3. |]) };
        { w = f32 [||]; b = None };
      ];
    steps = i32 [| 7l |];
    window = 8;
    kind = Sparse 3;
    ends = (f32 [| 4. |], f32 [| 5.; 6. |]);
  }

(* A dense block of [layers] whose ends are [a] and [b]. *)
let plain ?(layers = []) a b =
  { layers; steps = i32 [| 1l |]; window = 0; kind = Dense; ends = (a, b) }

let floats =
  let open Gen in
  let+ n = int_range 0 3 in
  f32 (Array.init n float_of_int)

(* A block whose positions are drawn by [position]. *)
let block_of position =
  let open Gen in
  let linear =
    let+ w = position and+ b = option position in
    { w; b }
  in
  let kind =
    frequency
      [ (1, constant Dense); (1, map (fun k -> Sparse k) (int_range 0 1)) ]
  in
  let+ layers = list ~size:(int_range 0 3) linear
  and+ steps = int_range 0 2
  and+ window = int_range 0 1
  and+ kind = kind
  and+ a = position
  and+ b = position in
  let steps = i32 (Array.init steps Int32.of_int) in
  { layers; steps; window; kind; ends = (a, b) }

let pp_visits s ppf x =
  Format.pp_print_list ~pp_sep:Format.pp_print_space P.pp_visit ppf
    (P.visits s x)

let blocks = block_of floats |> Gen.with_pp (pp_visits block)

(* A layer without its bias is about 0.14 of the drawn blocks, which 100 cases
   miss with probability 3e-7: this block has one. *)
let unbiased =
  plain ~layers:[ { w = f32 [| 1. |]; b = None } ] (f32 [| 2. |]) (f32 [||])

let segs =
  let open Gen in
  list ~size:(int_range 0 4)
    (one_of
       [
         map (fun s -> P.Path.Field s) (of_list [ "a"; "b"; "a.b" ]);
         map (fun i -> P.Path.Index i) (int_range 0 3);
       ])
  |> with_pp (fun ppf l -> P.Path.pp ppf (P.Path.v l))

let seg_string = function
  | P.Path.Field s -> s
  | P.Path.Index i -> string_of_int i

(* Paths *)

let paths =
  group "Path"
    [
      test "the root has no step and prints as the empty string" (fun () ->
          equal ~msg:"steps" (list seg_t) [] (P.Path.segments P.Path.root);
          equal ~msg:"printed" string "" (P.Path.to_string P.Path.root));
      test "paths that print alike differ by their steps" (fun () ->
          let one = P.Path.v [ Field "a.b" ]
          and two = P.Path.v [ Field "a"; Field "b" ] in
          equal ~msg:"printed" string (P.Path.to_string one)
            (P.Path.to_string two);
          not_equal ~msg:"compared" path_t one two);
      prop "a path's steps are the ones it was made of, printed joined by dots"
        segs (fun segs ->
          Law.round_trip (list seg_t) path_t P.Path.v P.Path.segments segs;
          equal ~msg:"add" (list seg_t)
            (segs @ [ P.Path.Index 7 ])
            (P.Path.segments (P.Path.add (Index 7) (P.Path.v segs)));
          let p = P.Path.v segs in
          equal ~msg:"to_string" string
            (String.concat "." (List.map seg_string segs))
            (P.Path.to_string p);
          equal ~msg:"pp" string (P.Path.to_string p)
            (Format.asprintf "%a" P.Path.pp p));
    ]

(* Walks *)

(* A block whose positions hold blocks, and a list of blocks, which may hold no
   position: each walked as one structure by [under]. *)
module Blocks = struct
  type ('a, 'd) t = (('a, 'd) block, 'd) block

  let walk c x = P.Walk.under Block.walk Block.walk c x
end

module Block_list = struct
  type ('a, 'd) t = ('a, 'd) block list

  let walk c x = P.Walk.(under (list leaf) Block.walk) c x
end

type nested = ((f32, host) block, host) block

let nested_ref : (nested, host) P.t =
  P.nest (module Block) (P.instantiate (module Block))

(* The positions' arrays as fixed arrays. *)
let nested_fixed : (nested, host) P.t =
  P.v (fun c x -> P.Walk.(under Block.walk (under Block.walk tensor)) c x)

let nested_module : (nested, host) P.t = P.instantiate (module Blocks)
let list_ref : ((f32, host) block list, host) P.t = P.(list block)

let list_fixed : ((f32, host) block list, host) P.t =
  P.v (fun c x -> P.Walk.(under (list leaf) (under Block.walk tensor)) c x)

let list_module : ((f32, host) block list, host) P.t =
  P.instantiate (module Block_list)

(* Each array as its dtype, shape and elements. *)
let contents s x =
  List.map
    (fun (Nx.P a) -> Format.asprintf "%a %s" pp_leaf (Nx.P a) (Nx.to_string a))
    (fst (P.flatten s x))

(* [s] visits, folds and maps [x] as [reference] does. *)
let agree ~msg reference s x =
  let msg what = msg ^ ": " ^ what in
  equal ~msg:(msg "visits") (list visit) (P.visits reference x) (P.visits s x);
  let folded s =
    P.fold s (fun p a acc -> (P.Path.to_string p, Nx.P a) :: acc) x []
    |> List.rev
  in
  equal ~msg:(msg "fold's paths") (list string)
    (List.map fst (folded reference))
    (List.map fst (folded s));
  equal ~msg:(msg "fold's arrays") leaves
    (List.map snd (folded reference))
    (List.map snd (folded s));
  let mapped s =
    let paths = ref [] in
    let y =
      P.map s
        (fun p a ->
          paths := P.Path.to_string p :: !paths;
          Nx.neg a)
        x
    in
    (List.rev !paths, contents s y)
  in
  equal ~msg:(msg "map")
    (pair (list string) (list string))
    (mapped reference) (mapped s)

let nested_blocks =
  block_of (block_of floats) |> Gen.with_pp (pp_visits nested_ref)

(* An outer position without its bias, and an inner block holding layers one of
   which lacks its bias: each about 0.14 of the drawn values, which 100 cases
   miss with probability 3e-7. *)
let nested_example : nested =
  let inner = unbiased in
  plain ~layers:[ { w = inner; b = None } ] inner (plain (f32 [||]) (f32 [||]))

let block_lists =
  Gen.list ~size:(Gen.int_range 0 3) (block_of floats)
  |> Gen.with_pp (pp_visits list_ref)

(* A state whose model nests blocks in a block and whose frozen layer holds
   fixed arrays at its positions. *)
type ('a, 'd) state = {
  model : (('a, 'd) block, 'd) block;
  frozen : 'd Nx.float32_t linear;
}

module State = struct
  type ('a, 'd) t = ('a, 'd) state

  let walk c x =
    let open P.Walk in
    let model = field c "model" (under Block.walk Block.walk) x.model in
    let frozen = field c "frozen" (under Linear.walk tensor) x.frozen in
    { model; frozen }
end

let state : ((f32, host) state, host) P.t = P.instantiate (module State)

let frozen_state () =
  let inner = plain (f32 [| 1. |]) (f32 [| 2. |]) in
  {
    model = plain inner inner;
    frozen = { w = f32 [| 3. |]; b = Some (f32 [| 4. |]) };
  }

let walks =
  group "Walk"
    [
      prop "a block of blocks walked with under is a nested description"
        ~examples:[ nested_example ] nested_blocks (fun x ->
          cover "an outer block with no layer" (x.layers = []);
          cover "an inner block with no layer" ((fst x.ends).layers = []);
          cover "an inner sparse kind" ((snd x.ends).kind <> Dense);
          cover "an absent outer position"
            (List.exists (fun l -> Option.is_none l.b) x.layers);
          cover "an absent inner bias"
            (List.exists (fun l -> Option.is_none l.b) (fst x.ends).layers);
          agree ~msg:"fixed" nested_ref nested_fixed x;
          agree ~msg:"module" nested_ref nested_module x);
      prop "a list of blocks walked with under is a nested description"
        block_lists (fun x ->
          cover "no block" (x = []);
          cover "blocks" (x <> []);
          agree ~msg:"fixed" list_ref list_fixed x;
          agree ~msg:"module" list_ref list_module x);
      test "a frozen layer's arrays are fixed arrays at its paths" (fun () ->
          let x = frozen_state () in
          let inner_at p =
            List.map
              (fun v -> p ^ "." ^ v)
              [
                "layers: length 0";
                "steps: a leaf";
                "window: int 0";
                "kind: case \"dense\"";
                "ends.0: a leaf";
                "ends.1: a leaf";
              ]
          in
          equal (list string)
            ([
               "model.layers: length 0";
               "model.steps: a leaf";
               "model.window: int 0";
               "model.kind: case \"dense\"";
             ]
            @ inner_at "model.ends.0" @ inner_at "model.ends.1"
            @ [ "frozen.w: a leaf"; "frozen.b: Some"; "frozen.b: a leaf" ])
            (words state x));
      test "structure walks a description's parts as their own walk does"
        (fun () ->
          let x = f32 [| 1. |] in
          let structured = P.v (fun c x -> P.Walk.structure P.tensor c x) in
          equal (list visit) (P.visits P.tensor x) (P.visits structured x));
    ]

(* Descriptions *)

(* A tuple's component: a description and a value of it. *)
type component = C : ('a, host) P.t * 'a -> component

let pp_component ppf (C (s, x)) = Format.fprintf ppf "[%a]" (pp_visits s) x

let components =
  let open Gen in
  one_of
    [
      map (fun x -> C (P.tensor, x)) floats;
      constant (C (P.unit, ()));
      map (fun x -> C (P.(option tensor), x)) (option floats);
      map (fun l -> C (P.(list tensor), l)) (list ~size:(int_range 0 2) floats);
    ]
  |> Gen.with_pp pp_component

let tuples = Gen.quad components components components components
let arrayless (C (s, x)) = List.is_empty (fst (P.flatten s x))

(* [under i k] is [k]'s visits, each path after [Index i]. *)
let under i (C (s, x)) =
  let at p = P.Path.v (P.Path.Index i :: P.Path.segments p) in
  List.map
    (function
      | P.Leaf p -> P.Leaf (at p) | P.Report (p, r) -> P.Report (at p, r))
    (P.visits s x)

let cover_arrayless ks =
  List.iteri
    (fun i k -> cover (Printf.sprintf "no array at %d" i) (arrayless k))
    ks

type described =
  | Described : string * ('a, host) P.t * 'a * string list -> described

let x1 = f32 [| 1. |]

let descriptions =
  group "descriptions"
    [
      cases "each combinator walks at the paths it states"
        ~id:(fun (Described (name, _, _, _)) -> name)
        [
          Described ("tensor", P.tensor, x1, [ "the root: a leaf" ]);
          Described ("unit", P.unit, (), []);
          Described
            ( "pair, list",
              P.(pair tensor (list tensor)),
              (x1, [ x1; x1 ]),
              [ "0: a leaf"; "1: length 2"; "1.0: a leaf"; "1.1: a leaf" ] );
          Described
            ( "triple",
              P.(triple tensor (list tensor) (option tensor)),
              (x1, [ x1; x1 ], None),
              [
                "0: a leaf";
                "1: length 2";
                "1.0: a leaf";
                "1.1: a leaf";
                "2: None";
              ] );
          Described
            ( "quad",
              P.(quad tensor unit (option tensor) tensor),
              (x1, (), Some x1, x1),
              [ "0: a leaf"; "2: Some"; "2: a leaf"; "3: a leaf" ] );
          Described ("empty list", P.(list tensor), [], [ "the root: length 0" ]);
          Described ("option", P.(option tensor), None, [ "the root: None" ]);
        ]
        (fun (Described (_, s, x, expected)) ->
          equal (list string) expected (words s x));
      test "a nested description walks its payloads after their positions"
        (fun () ->
          let lists = P.nest (module Linear) P.(list tensor) in
          equal (list string)
            [ "w: length 1"; "w.0: a leaf"; "b: Some"; "b: length 0" ]
            (words lists { w = [ x1 ]; b = Some [] }));
      test "instantiate is nest of tensor" (fun () ->
          agree ~msg:"block" (P.nest (module Block) P.tensor) block example);
      test "a description made with v names the parts it walks as structures"
        (fun () ->
          let train =
            P.v (fun c (params, opt, step) ->
                let open P.Walk in
                let params = field c "params" (structure linear) params in
                let opt = field c "opt" (structure linear) opt in
                let step = field c "step" tensor step in
                (params, opt, step))
          in
          equal (list string)
            [
              "params.w: a leaf";
              "params.b: None";
              "opt.w: a leaf";
              "opt.b: Some";
              "opt.b: a leaf";
              "step: a leaf";
            ]
            (words train
               ({ w = x1; b = None }, { w = x1; b = Some x1 }, i32 [| 1l |])));
      prop "a tuple's visits are its components' after their positions" tuples
        (fun ((k0, k1, k2, k3) as ks) ->
          cover_arrayless [ k0; k1; k2; k3 ];
          let expected ks = List.concat (List.mapi under ks) in
          match ks with
          | C (a, x), C (b, y), C (c, z), C (d, w) ->
              equal ~msg:"triple" (list visit)
                (expected [ k0; k1; k2 ])
                (P.visits P.(triple a b c) (x, y, z));
              equal ~msg:"quad" (list visit)
                (expected [ k0; k1; k2; k3 ])
                (P.visits P.(quad a b c d) (x, y, z, w)));
      prop "rebuild of a tuple's own arrays is the tuple" tuples
        (fun ((k0, k1, k2, k3) as ks) ->
          cover_arrayless [ k0; k1; k2; k3 ];
          let same s v =
            let xs, k = P.flatten s v in
            let xs', k' = P.flatten s (P.rebuild s ~like:v xs) in
            equal ~msg:"skeleton" (option string) None
              (P.Skeleton.diff ~this:"rebuilt" k' ~that:"in the value" k);
            equal ~msg:"arrays" leaves xs xs'
          in
          match ks with
          | C (a, x), C (b, y), C (c, z), C (d, w) ->
              same P.(triple a b c) (x, y, z);
              same P.(quad a b c d) (x, y, z, w));
    ]

(* Maps and folds *)

let leaf_paths s x =
  List.filter_map
    (function P.Leaf p -> Some (P.Path.to_string p) | P.Report _ -> None)
    (P.visits s x)

(* Values whose walk visits a list's arrays without its length, a pair's array
   at a path its name gives, or a hidden dtype: each breaks a description's
   contract, so that values of one description differ where those of a
   contracted one cannot. *)
let unreported = P.v (fun c l -> List.map (P.Walk.tensor c) l)
let named = P.v (fun c (n, a) -> (n, P.Walk.(field c n tensor a)))

let nested_names =
  P.v (fun c (nest, a) ->
      let open P.Walk in
      ( nest,
        if nest then field c "a" (fun c -> field c "b" tensor) a
        else field c "a.b" tensor a ))

let packed = P.v (fun c (Nx.P a) -> Nx.P (P.Walk.tensor c a))

let packed_pair =
  P.v (fun c (a, Nx.P b) ->
      let open P.Walk in
      let a = index c 0 tensor a in
      let b = index c 1 tensor b in
      (a, Nx.P b))

(* A walk that names its one part [a] and [b] on alternate calls: one value
   walked two ways. *)
let flips = ref 0

let flipped () =
  incr flips;
  if !flips mod 2 = 0 then "a" else "b"

let fickle = P.v (fun c x -> P.Walk.field c (flipped ()) P.Walk.tensor x)

module Fickle = struct
  type ('a, _) t = 'a

  let walk c x = P.Walk.field c (flipped ()) P.Walk.leaf x
end

type differ = Differ : string * ('a, host) P.t * 'a * 'a * string -> differ

let differing =
  let one = { example with layers = [ List.hd example.layers ] } in
  let none =
    { example with layers = [ { w = x1; b = None }; { w = x1; b = None } ] }
  in
  [
    Differ
      ( "length",
        block,
        example,
        one,
        "layers: length 2 in the first value, length 1 in the second" );
    Differ
      ( "presence",
        block,
        example,
        none,
        "layers.0.b: Some in the first value, None in the second" );
    Differ
      ( "int",
        block,
        example,
        { example with window = 4 },
        "window: int 8 in the first value, int 4 in the second" );
    Differ
      ( "case",
        block,
        example,
        { example with kind = Dense },
        "kind: case \"sparse\" in the first value, case \"dense\" in the second"
      );
    Differ
      ( "nothing",
        unreported,
        [ x1; x1 ],
        [ x1 ],
        "the root: a leaf in the first value, nothing in the second" );
    Differ
      ( "paths",
        named,
        ("a", x1),
        ("b", x1),
        "a: a leaf in the first value, a leaf at b in the second" );
    Differ
      ( "alike",
        nested_names,
        (false, x1),
        (true, x1),
        "[a.b]: a leaf in the first value, a leaf at [a; b] in the second" );
    Differ
      ( "dtype",
        packed,
        Nx.P x1,
        Nx.P (i32 [| 1l |]),
        "the root: float32 in the first value, int32 in the second" );
    Differ
      ( "triple",
        P.(triple tensor tensor (option tensor)),
        (x1, x1, Some x1),
        (x1, x1, None),
        "2: Some in the first value, None in the second" );
  ]

let maps =
  group "maps and folds"
    [
      prop "map with the identity rebuilds the value from its own arrays"
        ~examples:[ unbiased ] blocks (fun x ->
          cover "an empty list" (x.layers = []);
          cover "an absent bias"
            (List.exists (fun l -> Option.is_none l.b) x.layers);
          let y = P.map block (fun _ a -> a) x in
          equal (list visit) (P.visits block x) (P.visits block y);
          equal leaves (fst (P.flatten block x)) (fst (P.flatten block y)));
      prop "fold and map visit the arrays of flatten at the leaves' paths"
        blocks (fun x ->
          let folded =
            P.fold block
              (fun p a acc -> (P.Path.to_string p, Nx.P a) :: acc)
              x []
            |> List.rev
          in
          equal ~msg:"fold's paths" (list string) (leaf_paths block x)
            (List.map fst folded);
          equal ~msg:"fold's arrays" leaves
            (fst (P.flatten block x))
            (List.map snd folded);
          let mapped = ref [] in
          ignore
            (P.map block
               (fun p a ->
                 mapped := P.Path.to_string p :: !mapped;
                 a)
               x);
          equal ~msg:"map's paths" (list string) (leaf_paths block x)
            (List.rev !mapped));
      prop "map2 applies its function to the arrays at each path" blocks
        (fun x ->
          let y = P.map block (fun _ a -> Nx.copy a) x in
          let z = P.map2 block (fun _ a b -> Nx.add a b) x y in
          List.iter2
            (fun (Nx.P a) (Nx.P c) ->
              match D.equal_witness (Nx.dtype a) (Nx.dtype c) with
              | None -> fail "map2 changed a dtype"
              | Some Equal ->
                  let dt = Nx.dtype a in
                  equal (array int) (Nx.shape a) (Nx.shape c);
                  equal (elements dt)
                    (Array.to_list (Nx.to_array (Nx.add a a)))
                    (Array.to_list (Nx.to_array c)))
            (fst (P.flatten block x))
            (fst (P.flatten block z)));
      cases "every dtype walks as an array"
        ~id:(fun (D.Any dt) -> D.name dt)
        D.all
        (fun (D.Any dt) ->
          let a : (_, _, host) Nx.t = Nx.full dt [| 2 |] (D.one dt) in
          let s = P.(pair tensor (list tensor)) in
          let x = (a, [ a; a ]) in
          let xs, _ = P.flatten s x in
          equal ~msg:"flatten" leaves [ Nx.P a; Nx.P a; Nx.P a ] xs;
          equal ~msg:"rebuild" leaves xs
            (fst (P.flatten s (P.rebuild s ~like:x xs)));
          equal ~msg:"map" leaves xs
            (fst (P.flatten s (P.map s (fun _ a -> a) x))));
      cases "two values that differ name the first visit where they do"
        ~id:(fun (Differ (name, _, _, _, _)) -> name)
        differing
        (fun (Differ (_, s, a, b, expected)) ->
          equal string
            ("Nx.Ptree.map2: " ^ expected)
            (message (fun () -> P.map2 s (fun _ a _ -> a) a b)));
      test "map2 applies nothing to values whose visits differ" (fun () ->
          let calls = ref 0 in
          invalid ~by:"Nx.Ptree.map2" (fun () ->
              P.map2 block
                (fun _ a _ ->
                  incr calls;
                  a)
                example
                { example with window = 4 });
          equal int 0 !calls);
      test "map2 applies its function to the arrays before one of another dtype"
        (fun () ->
          let calls = ref 0 in
          let m =
            message (fun () ->
                P.map2 packed_pair
                  (fun _ a _ ->
                    incr calls;
                    a)
                  (x1, Nx.P x1)
                  (x1, Nx.P (i32 [| 1l |])))
          in
          equal ~msg:"message" string
            "Nx.Ptree.map2: 1: float32 in the first value, int32 in the second"
            m;
          equal ~msg:"applied" int 1 !calls);
      test "map2 and Payload.map2 refuse a walk of one value two ways"
        (fun () ->
          invalid ~by:"Nx.Ptree.map2" (fun () ->
              P.map2 fickle (fun _ a _ -> a) x1 x1);
          invalid ~by:"Nx.Ptree.Payload.map2" (fun () ->
              P.Payload.map2 (module Fickle) (fun _ a _ -> a) 1 1));
    ]

(* Donation *)

type dead_case =
  | Dead : string * ('a, host) P.t * (unit -> 'a) * string -> dead_case

let donation =
  group "donate"
    [
      test "donate gives up every array of the structure" (fun () ->
          let x = { w = f32 [| 1.; 2. |]; b = Some (f32 [| 3. |]) } in
          let d = P.donate linear x in
          equal ~msg:"its visits" (list string) (words linear x)
            (words linear d);
          equal ~msg:"every array donated" (list bool) [ true; true ]
            (List.rev
               (P.fold linear (fun _ a acc -> Nx.Prim.donated a :: acc) d []));
          raises (Invalid_argument "Nx.copy: float32 [2] was donated")
            (fun () -> Nx.copy x.w);
          raises (Invalid_argument "Nx.copy: float32 [1] was donated")
            (fun () -> Nx.copy (Option.get x.b));
          equal ~msg:"a handle" (array float_exact) [| 2.; 4. |]
            (Nx.to_array (Nx.add d.w d.w)));
      test "a value at two paths takes one handle" (fun () ->
          let w = f32 [| 1.; 2. |] in
          let d = P.donate linear { w; b = Some w } in
          equal ~msg:"one handle" leaf (Nx.P d.w) (Nx.P (Option.get d.b));
          raises (Invalid_argument "Nx.copy: float32 [2] was donated")
            (fun () -> Nx.copy w));
      cases "a dead array raises naming its path"
        ~id:(fun (Dead (name, _, _, _)) -> name)
        (let dead () =
           let x = f32 [| 1.; 2. |] in
           ignore (Nx.copy (Nx.donate x));
           x
         in
         [
           Dead
             ( "a field",
               linear,
               (fun () -> { w = dead (); b = None }),
               "Nx.Ptree.donate: w: float32 [2] was donated" );
           Dead
             ( "the root",
               P.tensor,
               dead,
               "Nx.Ptree.donate: the root: float32 [2] was donated" );
         ])
        (fun (Dead (_, s, make, expected)) ->
          let x = make () in
          raises (Invalid_argument expected) (fun () -> P.donate s x));
      test "a structure of no array comes back as it was" (fun () ->
          equal ~msg:"an empty list" (list leaf) []
            (List.map
               (fun a -> Nx.P a)
               (P.donate P.(list tensor) ([] : f32 list)));
          equal ~msg:"an absent option" (option leaf) None
            (Option.map
               (fun a -> Nx.P a)
               (P.donate P.(option tensor) (None : f32 option))));
    ]

(* Placement and casts *)

let law_round_trip (x, split) =
  cover "split" split;
  cover "whole on each device" (not split);
  cover "an absent frozen bias" (Option.is_none x.frozen.b);
  cover "an empty inner block" ((fst x.model.ends).layers = []);
  let p = if split then S2.split ~axis:0 else S2.on in
  let y = P.place (module State) p x in
  let on_s2 : ((S2.d Nx.float32_t, S2.d) state, S2.d) P.t =
    P.instantiate (module State)
  in
  let placements = P.fold on_s2 (fun _ a acc -> Nx.placement a :: acc) y [] in
  equal ~msg:"placed"
    (list (placement ()))
    (List.map (fun _ -> p) placements)
    placements;
  let back = P.place (module State) Nx.Host.on y in
  equal ~msg:"visits" (list visit) (P.visits state x) (P.visits state back);
  equal ~msg:"contents" (list string) (contents state x) (contents state back);
  let placements =
    P.fold state (fun _ a acc -> Nx.placement a :: acc) back []
  in
  equal ~msg:"on the host"
    (list (placement ()))
    (List.map (fun _ -> Nx.Host.on) placements)
    placements

let states =
  let open Gen in
  (let+ model = block_of (block_of floats)
   and+ w = floats
   and+ b = option floats
   and+ split = bool in
   ({ model; frozen = { w; b } }, split))
  |> with_pp (fun ppf (x, split) ->
      Format.fprintf ppf "%s, %a"
        (if split then "split" else "on")
        (pp_visits state) x)

let cut_words p =
  Format.asprintf
    "Nx.Rng.of_tensor: %a cuts a key's words, which lie whole on one device"
    Nx.Placement.pp p

(* The names of the operations [f] applies, under an Extent interpretation that
   computes them. *)
let applied f =
  let log = ref [] in
  let result =
    match
      Nx.Prim.interpret ~name:"test.log" Extent
        (fun _ ~by op ->
          log := Nx.Prim.name op :: !log;
          Nx.Prim.eval ~by op)
        (fun _ -> f ())
    with
    | _ -> "returned"
    | exception Invalid_argument m -> m
  in
  (result, List.rev !log)

let placing =
  group "place and cast"
    [
      test "place moves every array, fixed ones too, onto another set"
        (fun () ->
          let y = P.place (module Block) S.on example in
          is_on S.on y.steps;
          is_on S.on (List.hd y.layers).w;
          equal ~msg:"values" (array float_exact) [| 1.; 2. |]
            (Nx.to_array (List.hd y.layers).w);
          equal ~msg:"steps" (array int32) [| 7l |] (Nx.to_array y.steps));
      test "a frozen layer's arrays move with its structure" (fun () ->
          let y = P.place (module State) S.on (frozen_state ()) in
          is_on S.on y.frozen.w;
          is_on S.on (Option.get y.frozen.b);
          is_on S.on y.model.steps;
          is_on S.on (fst y.model.ends).steps;
          is_on S.on (fst (fst y.model.ends).ends));
      prop "place takes a state walked with under to a set and back" states
        law_round_trip;
      test "a key walked by Rng.walk moves with its structure" (fun () ->
          let x = { held = f32 [| 1. |]; key = Nx.Rng.key 7 } in
          let y = P.place (module Keyed) S.on x in
          is_on S.on (Nx.Rng.to_tensor y.key);
          is_on S.on y.held;
          equal ~msg:"words" (array int32)
            (Nx.to_array (Nx.Rng.to_tensor x.key))
            (Nx.to_array (Nx.Rng.to_tensor y.key)));
      test
        "place raises as Rng.walk does at a placement that cuts a key's words"
        (fun () ->
          let x = { held = f32 [| 1.; 2. |]; key = Nx.Rng.key 7 } in
          let p = S2.split ~axis:0 in
          equal string (cut_words p)
            (message (fun () -> P.place (module Keyed) p x)));
      test "place checks every array before it moves one" (fun () ->
          let b = f32 [| 3. |] in
          ignore (Nx.copy (Nx.donate b));
          let raised, ops =
            applied (fun () ->
                P.place
                  (module Linear)
                  S.on
                  { w = f32 [| 1.; 2. |]; b = Some b })
          in
          starts_with ~msg:"raised" ~prefix:"Nx.Ptree.place: " raised;
          equal ~msg:"operations" (list string) [] ops);
      prop
        "place_fixed keeps a structure's parts and puts each array at the \
         placement"
        ~examples:[ unbiased ] blocks (fun x ->
          cover "an empty list" (x.layers = []);
          cover "an absent bias"
            (List.exists (fun l -> Option.is_none l.b) x.layers);
          let y = P.place_fixed walk_frozen S2.on x in
          let arrays s v =
            P.fold s (fun p a acc -> (p, Nx.P a) :: acc) v [] |> List.rev
          in
          let xs = arrays (P.v walk_frozen) x in
          let ys = arrays (P.v walk_frozen) y in
          let show (p, Nx.P a) =
            P.Path.to_string p ^ ": " ^ Nx.to_string (Nx.place Nx.Host.on a)
          in
          let where (_, Nx.P a) = Nx.placement a in
          equal ~msg:"visits"
            (list visit)
            (P.visits (P.v walk_frozen) x)
            (P.visits (P.v walk_frozen) y);
          equal ~msg:"values" (list string) (List.map show xs)
            (List.map show ys);
          equal ~msg:"placements"
            (list (placement ()))
            (List.map (fun _ -> S2.on) ys)
            (List.map where ys));
      test "place_fixed raises at a dead array, naming its path" (fun () ->
          let a = f32 [| 1.; 2. |] and b = f32 [| 3. |] in
          ignore (Nx.copy (Nx.donate b));
          raises
            (Invalid_argument "Nx.Ptree.place_fixed: 1: float32 [1] was donated")
            (fun () -> P.place_fixed walk_two S.on (a, b)));
      test "place_fixed raises as the walk does" (fun () ->
          let p = S2.split ~axis:0 in
          equal string (cut_words p)
            (message (fun () -> P.place_fixed Nx.Rng.walk p (Nx.Rng.key 7))));
      test "cast changes the parameter's positions and keeps fixed arrays"
        (fun () ->
          let y = P.cast (module Block) Nx.float64 example in
          equal ~msg:"fixed" leaf (Nx.P example.steps) (Nx.P y.steps);
          equal ~msg:"dtype" string "float64" (D.name (Nx.dtype (fst y.ends)));
          equal ~msg:"values" (array float_exact) [| 5.; 6. |]
            (Nx.to_array (snd y.ends)));
      test "cast keeps a frozen layer's and the nested fixed arrays" (fun () ->
          let x = frozen_state () in
          let z = P.cast (module State) Nx.float64 x in
          equal ~msg:"the frozen layer" leaf (Nx.P x.frozen.w) (Nx.P z.frozen.w);
          equal ~msg:"the model's fixed array" leaf (Nx.P x.model.steps)
            (Nx.P z.model.steps);
          equal ~msg:"an inner fixed array" leaf (Nx.P (fst x.model.ends).steps)
            (Nx.P (fst z.model.ends).steps);
          let w = fst (fst z.model.ends).ends in
          equal ~msg:"dtype" string "float64" (D.name (Nx.dtype w));
          equal ~msg:"values" (array float_exact) [| 1. |] (Nx.to_array w));
    ]

(* Payloads *)

module Layers = struct
  type ('a, _) t = 'a linear list

  let walk c x = P.Walk.list Linear.walk c x
end

let payloads =
  group "Payload"
    [
      test "a payload map passes each position its path, in walk order"
        (fun () ->
          let names =
            P.Payload.map
              (module Linear)
              (fun p () -> P.Path.to_string p)
              { w = (); b = Some () }
          in
          equal ~msg:"map" string "w" names.w;
          equal ~msg:"map" (option string) (Some "b") names.b;
          let both =
            P.Payload.map2
              (module Linear)
              (fun _ n k -> (n, k))
              names { w = 1; b = Some 2 }
          in
          equal ~msg:"map2" (pair string int) ("w", 1) both.w);
      test "payload maps keep the first value's fixed arrays" (fun () ->
          let ints =
            P.Payload.map (module Block) (fun _ a -> Nx.numel a) example
          in
          equal ~msg:"map" leaf (Nx.P example.steps) (Nx.P ints.steps);
          let other = { ints with steps = i32 [| 9l |] } in
          let zipped =
            P.Payload.map2 (module Block) (fun _ a _ -> a) ints other
          in
          equal ~msg:"map2" leaf (Nx.P ints.steps) (Nx.P zipped.steps));
      test "a payload map2 of values that differ raises before applying"
        (fun () ->
          let calls = ref 0 in
          let m =
            message (fun () ->
                P.Payload.map2
                  (module Linear)
                  (fun _ a _ ->
                    incr calls;
                    a)
                  { w = 1; b = Some 2 } { w = 1; b = None })
          in
          equal string
            "Nx.Ptree.Payload.map2: b: Some in the first value, None in the \
             second"
            m;
          equal ~msg:"applied" int 0 !calls);
      prop
        "a payload fold passes the payloads of a payload map, at their paths, \
         in walk order"
        ~examples:[ unbiased ] blocks (fun x ->
          cover "an empty list" (x.layers = []);
          cover "an absent bias"
            (List.exists (fun l -> Option.is_none l.b) x.layers);
          let mapped = ref [] in
          ignore
            (P.Payload.map
               (module Block)
               (fun p a ->
                 mapped := (P.Path.to_string p, Nx.P a) :: !mapped;
                 a)
               x);
          let mapped = List.rev !mapped in
          let folded =
            P.Payload.fold
              (module Block)
              (fun p a acc -> (P.Path.to_string p, Nx.P a) :: acc)
              x []
            |> List.rev
          in
          equal ~msg:"paths" (list string) (List.map fst mapped)
            (List.map fst folded);
          equal ~msg:"payloads" leaves (List.map snd mapped)
            (List.map snd folded));
      prop
        "at array payloads, a payload fold is the array fold without the fixed \
         arrays"
        blocks (fun x ->
          let movable =
            P.fold block
              (fun p a acc -> (P.Path.to_string p, Nx.P a) :: acc)
              x []
            |> List.rev
            |> List.filter (fun (p, _) -> p <> "steps")
          in
          let folded =
            P.Payload.fold
              (module Block)
              (fun p a acc -> (P.Path.to_string p, Nx.P a) :: acc)
              x []
            |> List.rev
          in
          equal ~msg:"paths" (list string) (List.map fst movable)
            (List.map fst folded);
          equal ~msg:"arrays" leaves (List.map snd movable)
            (List.map snd folded));
      prop
        "through under, a payload fold is the array fold without the fixed \
         arrays"
        ~examples:[ nested_example ] nested_blocks (fun x ->
          cover "an inner block with layers" ((fst x.ends).layers <> []);
          cover "an absent inner bias"
            (List.exists (fun l -> Option.is_none l.b) (fst x.ends).layers);
          let fixed p =
            List.nth_opt (List.rev (String.split_on_char '.' p)) 0
            = Some "steps"
          in
          let movable =
            P.fold nested_module
              (fun p a acc -> (P.Path.to_string p, Nx.P a) :: acc)
              x []
            |> List.rev
            |> List.filter (fun (p, _) -> not (fixed p))
          in
          let folded =
            P.Payload.fold
              (module Blocks)
              (fun p a acc -> (P.Path.to_string p, Nx.P a) :: acc)
              x []
            |> List.rev
          in
          equal ~msg:"paths" (list string) (List.map fst movable)
            (List.map fst folded);
          equal ~msg:"arrays" leaves (List.map snd movable)
            (List.map snd folded));
      cases "a payload fold applies its function at each payload, in walk order"
        ~id:(fun (name, _, _) -> name)
        (let paths p n acc = (P.Path.to_string p, n) :: acc in
         [
           ( "w and b",
             (fun () ->
               P.Payload.fold (module Linear) paths { w = 1; b = Some 2 } []),
             [ ("b", 2); ("w", 1) ] );
           ( "w only",
             (fun () ->
               P.Payload.fold (module Linear) paths { w = 1; b = None } []),
             [ ("w", 1) ] );
           ( "no position",
             (fun () -> P.Payload.fold (module Layers) paths [] []),
             [] );
           ( "a key walked by Rng.walk",
             (fun () ->
               P.Payload.fold
                 (module Keyed)
                 paths
                 { held = 3; key = Nx.Rng.key 7 }
                 []),
             [ ("held", 3) ] );
         ])
        (fun (_, fold, expected) ->
          equal (list (pair string int)) expected (fold ()));
      test "a payload fold of a value on another set reads no array" (fun () ->
          let y : (S2.d Nx.float32_t, S2.d) block =
            P.place (module Block) S2.on example
          in
          equal (list string)
            [ "layers.0.w"; "layers.0.b"; "layers.1.w"; "ends.0"; "ends.1" ]
            (P.Payload.fold
               (module Block)
               (fun p _ acc -> P.Path.to_string p :: acc)
               y []
            |> List.rev));
      test "a payload fold of a state passes its model's positions only"
        (fun () ->
          let a = f32 [| 1. |] and b = f32 [| 2. |] in
          let c = f32 [| 3. |] and d = f32 [| 4. |] in
          let x =
            {
              model = plain (plain a b) (plain c d);
              frozen = { w = f32 [| 5. |]; b = Some (f32 [| 6. |]) };
            }
          in
          let folded =
            P.Payload.fold
              (module State)
              (fun p a acc -> (P.Path.to_string p, Nx.P a) :: acc)
              x []
            |> List.rev
          in
          equal ~msg:"paths" (list string)
            [
              "model.ends.0.ends.0";
              "model.ends.0.ends.1";
              "model.ends.1.ends.0";
              "model.ends.1.ends.1";
            ]
            (List.map fst folded);
          equal ~msg:"payloads" leaves
            [ Nx.P a; Nx.P b; Nx.P c; Nx.P d ]
            (List.map snd folded));
    ]

(* Arithmetic *)

(* A list of arrays of any dtype, each walked as a fixed array. *)
let packs () =
  P.v (fun c l -> P.Walk.list (fun c (Nx.P a) -> Nx.P (P.Walk.tensor c a)) c l)

(* The dtype of a drawn array: a float, a complex or an int32 counter, which the
   vector leaves out. *)
type format =
  | Real : (float, 's) D.t -> format
  | Cplx : (Complex.t, 's) D.t * (float, 'p) D.t -> format
  | Count

let pp_kind ppf = function
  | Real dt -> Format.pp_print_string ppf (D.name dt)
  | Cplx (dt, _) -> Format.pp_print_string ppf (D.name dt)
  | Count -> Format.pp_print_string ppf "int32"

(* Values every float format holds, so that each product and sum below is exact
   in float64 and rounds once to the array's format. A NaN is one element
   in 27, so about 0.11 of the vectors hold one in [x], and the law draws
   300 vectors, which miss it with probability below 10^-13. *)
let specials =
  Gen.frequency
    [
      (8, Gen.of_list [ -4.; -1.; -0.; 0.; 0.5; 1.; 3. ]);
      (1, Gen.of_list [ Float.nan; Float.infinity; Float.neg_infinity ]);
    ]

(* An array's kind and, for [x] then [y], its real and imaginary parts. *)
type drawn = {
  kind : format;
  x : float array * float array;
  y : float array * float array;
}

let drawn =
  let open Gen in
  let kinds =
    of_list ~pp:pp_kind
      [
        Real Nx.float16;
        Real Nx.bfloat16;
        Real Nx.float32;
        Real Nx.float64;
        Cplx (Nx.complex64, Nx.float32);
        Cplx (Nx.complex128, Nx.float64);
        Count;
      ]
  in
  let* kind = kinds in
  let* n = int_range 0 3 in
  let parts =
    pair (array ~size:(constant n) specials) (array ~size:(constant n) specials)
  in
  let+ x = parts and+ y = parts in
  { kind; x; y }

let vectors =
  let open Gen in
  (let+ arrays = list ~size:(int_range 0 4) drawn
   and+ k = of_list [ 2.; -0.5; -0.; 1. ] in
   (arrays, k))
  |> with_pp (fun ppf (arrays, k) ->
      Format.fprintf ppf "k %g, %a" k
        (Format.pp_print_list ~pp_sep:Format.pp_print_space (fun ppf d ->
             Format.fprintf ppf "%a[%d]" pp_kind d.kind (Array.length (fst d.x))))
        arrays)

let complexes re im = Array.map2 (fun re im -> { Complex.re; im }) re im

let value d (re, im) : host Nx.packed =
  let n = [| Array.length re |] in
  match d.kind with
  | Real dt -> Nx.P (Nx.create dt n re)
  | Cplx (dt, _) -> Nx.P (Nx.create dt n (complexes re im))
  | Count -> Nx.P (Nx.create Nx.int32 n (Array.map (fun _ -> 7l) re))

(* An array's coordinates in the vector. *)
let coordinates d (re, im) =
  match d.kind with Real _ -> re | Cplx _ -> Array.append re im | Count -> [||]

let round (type s) (dt : (float, s) D.t) v = D.of_float dt v

(* [a] at [d]'s dtype, as the result of [step] at each element of each part: the
   expected result of scale and axpy. *)
let expect d (step : 's. (float, 's) D.t -> float -> float -> float)
    (result : host Nx.packed) =
  let parts : type p. (float, p) D.t -> float array * float array -> _ =
   fun dt (re, im) ->
    ( Array.mapi (fun i v -> step dt v (fst d.y).(i)) re,
      Array.mapi (fun i v -> step dt v (snd d.y).(i)) im )
  in
  match d.kind with
  | Real dt ->
      equal (array float_exact)
        (fst (parts dt d.x))
        (Nx.to_array (Nx.unpack dt result))
  | Cplx (dt, p) ->
      let re, im = parts p d.x in
      let got = Nx.to_array (Nx.unpack dt result) in
      equal ~msg:"re" (array float_exact) re
        (Array.map (fun c -> c.Complex.re) got);
      equal ~msg:"im" (array float_exact) im
        (Array.map (fun c -> c.Complex.im) got)
  | Count -> ()

let law_vector (arrays, k) =
  let all =
    List.concat_map (fun d -> Array.to_list (coordinates d d.x)) arrays
  in
  let nan = List.exists Float.is_nan all
  and inf = List.exists (fun v -> Float.abs v = Float.infinity) all in
  cover "a NaN" nan;
  cover "an infinity" inf;
  cover "a NaN beside an infinity" (nan && inf);
  cover "no coordinate" (all = []);
  cover "a negative zero"
    (List.exists (fun v -> 1. /. v = Float.neg_infinity) all);
  cover "a narrow float"
    (List.exists
       (fun d -> match d.kind with Real dt -> D.bits dt = 16 | _ -> false)
       arrays);
  cover "a complex array"
    (List.exists
       (fun d -> match d.kind with Cplx _ -> true | _ -> false)
       arrays);
  let counter d =
    match d.kind with Count -> true | Real _ | Cplx _ -> false
  in
  cover "a counter" (List.exists counter arrays);
  let s = packs () in
  let x = List.map (fun d -> value d d.x) arrays
  and y = List.map (fun d -> value d d.y) arrays in
  (* Re (conj a . b) summed coordinate by coordinate, from +0. *)
  let dot =
    List.fold_left
      (fun acc d ->
        let a = coordinates d d.x and b = coordinates d d.y in
        acc +. Array.fold_left ( +. ) 0. (Array.map2 ( *. ) a b))
      0. arrays
  in
  (* A structure of no array gives its sums no placement to lie at. *)
  if arrays = [] then begin
    invalid ~by:"Nx.Ptree.dot" (fun () -> P.dot s Nx.float64 x y);
    invalid ~by:"Nx.Ptree.norm" (fun () -> P.norm s Nx.float64 x)
  end
  else begin
    equal ~msg:"dot" float_exact dot (Nx.item [] (P.dot s Nx.float64 x y));
    let norm = Nx.item [] (P.norm s Nx.float64 x) in
    if List.exists Float.is_nan all then
      equal ~msg:"norm" float_exact Float.nan norm
    else if List.exists (fun v -> Float.abs v = Float.infinity) all then
      equal ~msg:"norm" float_exact Float.infinity norm
    else
      (* At most 24 coordinates, each divided by the greatest magnitude and
         squared (two roundings), then summed (23), the root and the product by
         the magnitude (two): with u = eps / 2, a relative error near (2 + 23 /
         2 + 2) u, about 8 eps, under the 16 eps allowed. *)
      equal ~msg:"norm"
        (float_rel ~rel:(16. *. epsilon_float) ~abs:0.)
        (Float.sqrt (List.fold_left (fun acc v -> acc +. (v *. v)) 0. all))
        norm
  end;
  let kv = Nx.scalar Nx.float64 k in
  let scaled = P.scale s kv x in
  List.iter2
    (fun d r -> expect d (fun dt v _ -> round dt (k *. v)) r)
    arrays scaled;
  let z = P.axpy s kv x y in
  List.iter2
    (fun d r -> expect d (fun dt v w -> round dt (round dt (k *. v) +. w)) r)
    arrays z;
  (* A counter is kept: [scale]'s from [x], [axpy]'s from [y]. *)
  List.iteri
    (fun i d ->
      if counter d then begin
        equal ~msg:"scale's counter" leaf (List.nth x i) (List.nth scaled i);
        equal ~msg:"axpy's counter" leaf (List.nth y i) (List.nth z i)
      end)
    arrays

let round32 x = Int32.float_of_bits (Int32.bits_of_float x)

let arithmetic =
  group "arithmetic"
    [
      (* A NaN beside an infinity, whose norm is NaN, not the infinite magnitude
         it scales by, is 3.8% of the cases (miss 9e-6 per run): the example
         holds it. *)
      prop ~count:300
        ~examples:
          [
            ( [
                {
                  kind = Real Nx.float64;
                  x = ([| Float.infinity; Float.nan |], [| 0.; 0. |]);
                  y = ([| 1.; 1. |], [| 0.; 0. |]);
                };
              ],
              2. );
          ]
        "dot, norm, scale and axpy over float and complex arrays, every \
         format, NaN, infinities and -0."
        vectors law_vector;
      test "dot sums in walk order from 0, each array cast to its dtype first"
        (fun () ->
          let s = P.(list tensor) in
          (* (1e16 + 1) rounds to 1e16 in float64: walk order gives 0, adding
             the large terms first would give 1. *)
          let ones = [ f64 [| 1. |]; f64 [| 1. |]; f64 [| 1. |] ] in
          equal ~msg:"walk order" float_exact 0.
            (Nx.item []
               (P.dot s Nx.float64
                  [ f64 [| 1e16 |]; f64 [| 1. |]; f64 [| -1e16 |] ]
                  ones));
          (* 1 + 2^-24 casts to 1 in float32; its float64 square would round to
             1 + 2^-23. *)
          let x = [ f64 [| 1. +. ldexp 1. (-24) |] ] in
          equal ~msg:"cast first" float_exact 1.
            (Nx.item [] (P.dot s Nx.float32 x x)));
      test
        "dot and norm lie whole on each device of the first array's placement"
        (fun () ->
          let s = P.(list tensor) in
          let check p =
            let x = [ Nx.place p (f32 [| 1.; 2. |]) ] in
            equal ~msg:"dot" (placement ()) p
              (Nx.placement (P.dot s Nx.float32 x x));
            equal ~msg:"norm" (placement ()) p
              (Nx.placement (P.norm s Nx.float32 x))
          in
          check S2.on;
          check (Nx.Placement.device S2.v (m 1)));
      test "norm squares no coordinate where the norm is representable"
        (fun () ->
          let h xs : host Nx.float16_t =
            Nx.create Nx.float16 [| Array.length xs |] xs
          in
          let s = P.(list tensor) and s32 = P.(list tensor) in
          equal ~msg:"float16" float_exact 500.
            (Nx.item [] (P.norm s Nx.float16 [ h [| 300. |]; h [| 400. |] ]));
          (* Both quotients are 1 and their sum 2, exact; the root and the
             product by the magnitude each round once, and the reference once:
             two float32s apart at most. *)
          let big = round32 2e38 in
          at_most ~msg:"float32 near its greatest" float_exact ~than:2.
            (ulps Nx.float32
               (Float.sqrt 2. *. big)
               (Nx.item [] (P.norm s32 Nx.float32 [ f32 [| big; big |] ]))));
      cases "complex arrays are the vector's real and imaginary coordinates"
        ~id:(fun (D.Any dt) -> D.name dt)
        [ D.Any Nx.complex64; D.Any Nx.complex128 ]
        (fun (D.Any dt) ->
          match D.kind dt with
          | D.Complex ->
              let z re im = { Complex.re; im } in
              let c xs = Nx.create dt [| Array.length xs |] xs in
              let s = P.(pair tensor tensor) in
              let x = (c [| z 1. 2.; z 3. (-1.) |], f32 [| 2. |])
              and y = (c [| z 2. (-1.); z 1. 4. |], f32 [| 5. |]) in
              (* Re (conj a . b) = 1 * 2 + 2 * -1 + 3 * 1 + -1 * 4, then 2 *
                 5. *)
              equal ~msg:"dot" float_exact 9.
                (Nx.item [] (P.dot s Nx.float64 x y));
              (* Quotients by 3 round, as do their squares, the sum of five, the
                 root and the product by 3: about 4 u of sqrt 19, which is 2.2
                 of its float64s, and the reference rounds once. *)
              at_most ~msg:"norm" float_exact ~than:3.
                (ulps Nx.float64 (Float.sqrt 19.)
                   (Nx.item [] (P.norm s Nx.float64 x)));
              let k = Nx.scalar Nx.float32 2. in
              let pp ppf v = Format.fprintf ppf "%g%+gi" v.Complex.re v.im in
              let complex = Testable.make ~pp ~equal:( = ) in
              equal ~msg:"scale" (array complex)
                [| z 2. 4.; z 6. (-2.) |]
                (Nx.to_array (fst (P.scale s k x)));
              equal ~msg:"axpy" (array complex)
                [| z 4. 3.; z 7. 2. |]
                (Nx.to_array (fst (P.axpy s k x y)))
          | _ -> fail "a complex dtype");
      test "scale and axpy cast the factor to each array's format first"
        (fun () ->
          (* round32 (0.1 * 9) is 0.9 rounded, round32 (round32 0.1 * 9) is
             not. *)
          let s = P.(list tensor) and k = Nx.scalar Nx.float64 0.1 in
          let k' = round32 0.1 in
          let x = [ f32 [| 9.; 13. |] ] and y = [ f32 [| 1.; 1. |] ] in
          equal ~msg:"scale" (array float_exact)
            [| round32 (k' *. 9.); round32 (k' *. 13.) |]
            (Nx.to_array (List.hd (P.scale s k x)));
          equal ~msg:"axpy" (array float_exact)
            [|
              round32 (round32 (k' *. 9.) +. 1.);
              round32 (round32 (k' *. 13.) +. 1.);
            |]
            (Nx.to_array (List.hd (P.axpy s k x y))));
      cases
        "a function of two values raises on values that differ, and on a \
         factor with axes"
        ~id:(fun (name, _, _) -> name)
        (let vector = P.(pair (list tensor) tensor) in
         let x () = ([ f32 [| 1.; 2. |] ], i32 [| 0l |]) in
         let y () = ([ f32 [| 1.; 2.; 3. |] ], i32 [| 0l |]) in
         let k () = f32 [| 2. |] in
         let k0 () = Nx.scalar Nx.float32 1. in
         [
           ( "dot of two shapes",
             (fun () -> P.dot vector Nx.float32 (x ()) (y ()) |> ignore),
             `Is "Nx.Ptree.dot: 0.0: [2] in the first value, [3] in the second"
           );
           ( "dot of two visits",
             (fun () ->
               P.dot vector Nx.float32 (x ()) ([], i32 [| 0l |]) |> ignore),
             `Is
               "Nx.Ptree.dot: 0: length 1 in the first value, length 0 in the \
                second" );
           ( "axpy of two shapes",
             (fun () -> P.axpy vector (k0 ()) (x ()) (y ()) |> ignore),
             `Names "Nx.Ptree.axpy" );
           ( "scale by a factor with axes",
             (fun () -> P.scale vector (k ()) (x ()) |> ignore),
             `Names "Nx.Ptree.scale" );
           ( "axpy by a factor with axes",
             (fun () -> P.axpy vector (k ()) (x ()) (x ()) |> ignore),
             `Names "Nx.Ptree.axpy" );
         ])
        (fun (_, call, expected) ->
          match expected with
          | `Is m -> equal string m (message call)
          | `Names by -> invalid ~by call);
    ]

(* Visits *)

let visits =
  group "visits"
    [
      test "a block's visits are its parts in walk order, each at its path"
        (fun () ->
          equal (list string)
            [
              "layers: length 2";
              "layers.0.w: a leaf";
              "layers.0.b: Some";
              "layers.0.b: a leaf";
              "layers.1.w: a leaf";
              "layers.1.b: None";
              "steps: a leaf";
              "window: int 8";
              "kind: case \"sparse\"";
              "kind: int 3";
              "ends.0: a leaf";
              "ends.1: a leaf";
            ]
            (words block example));
    ]

(* Signatures *)

let signatures =
  group "signatures"
    [
      test "a signature lists its arguments, then its result" (fun () ->
          match P.(tensor @-> linear @-> returns unit) with
          | P.Arg (_, P.Arg (_, P.Returns _)) -> ()
          | _ -> fail "arguments");
    ]

(* Flattening *)

(* Two blocks, often with one skeleton. *)
let block_pairs =
  let open Gen in
  frequency [ (1, map (fun x -> (x, x)) blocks); (2, pair blocks blocks) ]

let rebuild_errors =
  let all = fst (P.flatten block example) in
  let n = List.length all in
  [
    ( "fewer",
      List.filteri (fun i _ -> i < 2) all,
      "Nx.Ptree.rebuild: ",
      [ "layers.1.w" ] );
    ( "more",
      all @ [ Nx.P x1 ],
      "Nx.Ptree.rebuild: ",
      [ string_of_int n; string_of_int (n + 1) ] );
    ( "dtype",
      List.mapi (fun i l -> if i = 3 then Nx.P x1 else l) all,
      "Nx.Ptree.rebuild: steps: ",
      [ "int32"; "float32" ] );
  ]

let flattening =
  group "flattening"
    [
      prop "rebuild of a value's own arrays is the value" blocks (fun x ->
          let xs, k = P.flatten block x in
          let y = P.rebuild block ~like:x xs in
          let xs', k' = P.flatten block y in
          equal (option string) None
            (P.Skeleton.diff ~this:"rebuilt" k' ~that:"in the value" k);
          equal leaves xs xs');
      prop
        "skeletons are equal iff their visits are, diff finds nothing then, \
         and equal ones hash equal"
        block_pairs (fun (x, y) ->
          let kx = snd (P.flatten block x) and ky = snd (P.flatten block y) in
          let eq = P.Skeleton.equal kx ky in
          cover "equal" eq;
          cover "different" (not eq);
          let vs = (P.visits block x, P.visits block y) in
          satisfies ~msg:"equal"
            ~claim:"Skeleton.equal holds iff the visits are equal"
            (pair (list visit) (list visit))
            (fun (a, b) -> List.equal visit_equal a b = eq)
            vs;
          satisfies ~msg:"diff"
            ~claim:"diff is None iff the skeletons are equal" (option string)
            (fun d -> Option.is_none d = eq)
            (P.Skeleton.diff ~this:"here" kx ~that:"there" ky);
          if eq then
            equal ~msg:"hash" int (P.Skeleton.hash kx) (P.Skeleton.hash ky));
      test "a diff places each side with its phrases" (fun () ->
          let k w = snd (P.flatten block { example with window = w }) in
          equal (option string)
            (Some "window: int 8 here, int 4 in the previous key")
            (P.Skeleton.diff ~this:"here" (k 8) ~that:"in the previous key"
               (k 4)));
      cases "rebuild names the array it lacks, the counts or the dtypes"
        ~id:(fun (name, _, _, _) -> name)
        rebuild_errors
        (fun (_, xs, prefix, subs) ->
          let m = message (fun () -> P.rebuild block ~like:example xs) in
          starts_with ~prefix m;
          List.iter (fun sub -> contains ~sub m) subs);
    ]

let () =
  exit
    (run "Nx.Ptree"
       [
         paths;
         walks;
         descriptions;
         maps;
         donation;
         placing;
         payloads;
         arithmetic;
         visits;
         signatures;
         flattening;
       ])
