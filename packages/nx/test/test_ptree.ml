(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx.Ptree: what a walk visits; the laws between flatten, rebuild, map and
   fold, at every dtype; the messages of two values that differ; payload maps,
   casts and placement through a structure's module; structures nested across
   sets; and the vector arithmetic of a value's float arrays. *)

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
let i32 xs : host Nx.int32_t = Nx.create Nx.int32 [| Array.length xs |] xs
let words s x = List.map (Format.asprintf "%a" P.pp_visit) (P.visits s x)
let path segs = P.Path.to_string (P.Path.v segs)

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

let segs =
  let open Gen in
  list ~size:(int_range 0 4)
    (one_of
       [
         map (fun s -> P.Path.Field s) (of_list [ "a"; "b"; "a.b" ]);
         map (fun i -> P.Path.Index i) (int_range 0 3);
       ])
  |> with_pp (fun ppf l -> P.Path.pp ppf (P.Path.v l))

let walks =
  group "walks"
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
      test "descriptions walk at the paths their combinators state" (fun () ->
          let x = f32 [| 1. |] in
          equal ~msg:"tensor" (list string) [ "the root: a leaf" ]
            (words P.tensor x);
          equal ~msg:"unit" (list string) [] (words P.unit ());
          equal ~msg:"pair, list" (list string)
            [ "0: a leaf"; "1: length 2"; "1.0: a leaf"; "1.1: a leaf" ]
            (words P.(pair tensor (list tensor)) (x, [ x; x ]));
          equal ~msg:"empty list" (list string) [ "the root: length 0" ]
            (words P.(list tensor) []);
          equal ~msg:"option" (list string) [ "the root: None" ]
            (words P.(option tensor) None));
      test "a nested description walks its payloads after their positions"
        (fun () ->
          let lists = P.nest (module Linear) P.(list tensor) in
          let x = f32 [| 1. |] in
          equal (list string)
            [ "w: length 1"; "w.0: a leaf"; "b: Some"; "b: length 0" ]
            (words lists { w = [ x ]; b = Some [] }));
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
          let x = f32 [| 1. |] in
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
               ({ w = x; b = None }, { w = x; b = Some x }, i32 [| 1l |])));
      test "paths compare by steps and print joined by dots" (fun () ->
          equal ~msg:"root" string "" (P.Path.to_string P.Path.root);
          equal ~msg:"steps" string "blocks.3.fc"
            (path [ Field "blocks"; Index 3; Field "fc" ]);
          let one = P.Path.v [ Field "a.b" ]
          and two = P.Path.v [ Field "a"; Field "b" ] in
          equal ~msg:"alike" string (P.Path.to_string one)
            (P.Path.to_string two);
          equal ~msg:"unequal" bool false (P.Path.equal one two));
      prop "a path's steps are the ones it was made of" segs (fun segs ->
          let p = P.Path.v segs in
          equal ~msg:"v" bool true (P.Path.segments p = segs);
          let q = P.Path.add (Index 7) p in
          equal ~msg:"add" bool true
            (P.Path.segments q = segs @ [ P.Path.Index 7 ]));
    ]

(* Laws *)

let leaf_paths s x =
  List.filter_map
    (function P.Leaf p -> Some (P.Path.to_string p) | P.Report _ -> None)
    (P.visits s x)

(* Two blocks, often with one skeleton. *)
let block_pairs =
  let open Gen in
  frequency [ (1, map (fun x -> (x, x)) blocks); (2, pair blocks blocks) ]

let laws =
  group "laws"
    [
      prop "map with the identity rebuilds the value from its own arrays" blocks
        (fun x ->
          cover "an empty list" (x.layers = []);
          cover "an absent bias"
            (List.exists (fun l -> Option.is_none l.b) x.layers);
          let y = P.map block (fun _ a -> a) x in
          equal (list visit) (P.visits block x) (P.visits block y);
          equal leaves (fst (P.flatten block x)) (fst (P.flatten block y)));
      prop "rebuild of a value's own arrays is the value" blocks (fun x ->
          let xs, k = P.flatten block x in
          let y = P.rebuild block ~like:x xs in
          let xs', k' = P.flatten block y in
          equal (option string) None
            (P.Skeleton.diff ~this:"rebuilt" k' ~that:"in the value" k);
          equal leaves xs xs');
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
      prop
        "skeletons are equal iff diff finds nothing, and equal ones hash equal"
        block_pairs (fun (x, y) ->
          let kx = snd (P.flatten block x) and ky = snd (P.flatten block y) in
          let eq = P.Skeleton.equal kx ky in
          cover "equal" eq;
          cover "different" (not eq);
          equal ~msg:"visits" bool eq
            (List.equal visit_equal (P.visits block x) (P.visits block y));
          equal ~msg:"diff" bool eq
            (Option.is_none (P.Skeleton.diff ~this:"here" kx ~that:"there" ky));
          if eq then
            equal ~msg:"hash" int (P.Skeleton.hash kx) (P.Skeleton.hash ky));
      prop "map2 applies its function to the arrays at each path" blocks
        (fun x ->
          let y = P.map block (fun _ a -> Nx.copy a) x in
          let z = P.map2 block (fun _ a b -> Nx.add a b) x y in
          List.iter2
            (fun (Nx.P a) (Nx.P c) ->
              match D.equal_witness (Nx.dtype a) (Nx.dtype c) with
              | None -> fail "dtypes"
              | Some Equal ->
                  equal (array int) (Nx.shape a) (Nx.shape c);
                  equal string (Nx.to_string (Nx.add a a)) (Nx.to_string c))
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
    ]

(* Messages *)

(* A value whose walk visits a list's arrays without its length, a pair's array
   at a path its name gives, or a hidden dtype: each breaks a description's
   contract, so that values of one description differ where those of a
   contracted one cannot. *)
let unreported = P.v (fun c l -> List.map (P.Walk.tensor c) l)
let named = P.v (fun c (n, a) -> (n, P.Walk.(field c n tensor a)))

let nested =
  P.v (fun c (nest, a) ->
      let open P.Walk in
      ( nest,
        if nest then field c "a" (fun c -> field c "b" tensor) a
        else field c "a.b" tensor a ))

let packed = P.v (fun c (Nx.P a) -> Nx.P (P.Walk.tensor c a))

let messages =
  let x = f32 [| 1. |] in
  let map2 s a b = message (fun () -> P.map2 s (fun _ a _ -> a) a b) in
  group "messages"
    [
      test "two values that differ name the first visit where they do"
        (fun () ->
          let one = { example with layers = [ List.hd example.layers ] } in
          equal ~msg:"length" string
            "Nx.Ptree.map2: layers: length 2 in the first value, length 1 in \
             the second"
            (map2 block example one);
          let none =
            {
              example with
              layers = [ { w = x; b = None }; { w = x; b = None } ];
            }
          in
          equal ~msg:"presence" string
            "Nx.Ptree.map2: layers.0.b: Some in the first value, None in the \
             second"
            (map2 block example none);
          equal ~msg:"int" string
            "Nx.Ptree.map2: window: int 8 in the first value, int 4 in the \
             second"
            (map2 block example { example with window = 4 });
          equal ~msg:"case" string
            "Nx.Ptree.map2: kind: case \"sparse\" in the first value, case \
             \"dense\" in the second"
            (map2 block example { example with kind = Dense });
          equal ~msg:"nothing" string
            "Nx.Ptree.map2: the root: a leaf in the first value, nothing in \
             the second"
            (map2 unreported [ x; x ] [ x ]);
          equal ~msg:"paths" string
            "Nx.Ptree.map2: a: a leaf in the first value, a leaf at b in the \
             second"
            (map2 named ("a", x) ("b", x));
          equal ~msg:"alike" string
            "Nx.Ptree.map2: [a.b]: a leaf in the first value, a leaf at [a; b] \
             in the second"
            (map2 nested (false, x) (true, x));
          equal ~msg:"dtype" string
            "Nx.Ptree.map2: the root: float32 in the first value, int32 in the \
             second"
            (map2 packed (Nx.P x) (Nx.P (i32 [| 1l |]))));
      test "map2 applies nothing to values that differ" (fun () ->
          let calls = ref 0 in
          invalid ~by:"Nx.Ptree.map2" (fun () ->
              P.map2 block
                (fun _ a _ ->
                  incr calls;
                  a)
                example
                { example with window = 4 });
          equal int 0 !calls);
      test "a diff places each side with its phrases" (fun () ->
          let k w = snd (P.flatten block { example with window = w }) in
          equal (option string)
            (Some "window: int 8 here, int 4 in the previous key")
            (P.Skeleton.diff ~this:"here" (k 8) ~that:"in the previous key"
               (k 4)));
      test "rebuild names the array it lacks, the counts or the dtypes"
        (fun () ->
          let leaves = fst (P.flatten block example) in
          let n = List.length leaves in
          let fewer = List.filteri (fun i _ -> i < 2) leaves in
          let m = message (fun () -> P.rebuild block ~like:example fewer) in
          starts_with ~msg:"fewer" ~prefix:"Nx.Ptree.rebuild: " m;
          contains ~msg:"fewer" ~sub:"layers.1.w" m;
          let m =
            message (fun () ->
                P.rebuild block ~like:example (leaves @ [ Nx.P x ]))
          in
          starts_with ~msg:"more" ~prefix:"Nx.Ptree.rebuild: " m;
          contains ~msg:"more" ~sub:(string_of_int n) m;
          contains ~msg:"more" ~sub:(string_of_int (n + 1)) m;
          let swapped =
            List.mapi (fun i l -> if i = 3 then Nx.P x else l) leaves
          in
          let m = message (fun () -> P.rebuild block ~like:example swapped) in
          starts_with ~msg:"dtype" ~prefix:"Nx.Ptree.rebuild: steps: " m;
          contains ~msg:"dtype" ~sub:"int32" m;
          contains ~msg:"dtype" ~sub:"float32" m);
    ]

(* Payloads, dtypes and sets *)

let payloads =
  group "payloads"
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
      test "cast changes the parameter's positions and keeps fixed arrays"
        (fun () ->
          let y = P.cast (module Block) Nx.float64 example in
          equal ~msg:"fixed" leaf (Nx.P example.steps) (Nx.P y.steps);
          equal ~msg:"dtype" string "float64" (D.name (Nx.dtype (fst y.ends)));
          equal ~msg:"values" (array float_exact) [| 5.; 6. |]
            (Nx.to_array (snd y.ends)));
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
          equal string
            (Format.asprintf
               "Nx.Rng.of_tensor: %a cuts a key's words, which lie whole on \
                one device"
               Nx.Placement.pp p)
            (message (fun () -> P.place (module Keyed) p x));
          let y =
            {
              held = Nx.place p x.held;
              key = Nx.Rng.place (Nx.Placement.uncut p) x.key;
            }
          in
          equal ~msg:"placed apart" (array int32)
            (Nx.to_array (Nx.Rng.to_tensor x.key))
            (Nx.to_array (Nx.Rng.to_tensor y.key)));
      test "place moves every array, fixed ones too, onto another set"
        (fun () ->
          let y = P.place (module Block) S.on example in
          is_on S.on y.steps;
          is_on S.on (List.hd y.layers).w;
          equal ~msg:"values" (array float_exact) [| 1.; 2. |]
            (Nx.to_array (List.hd y.layers).w);
          equal ~msg:"steps" (array int32) [| 7l |] (Nx.to_array y.steps));
    ]

(* Structures nested across sets *)

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

let nesting =
  group "nesting"
    [
      prop "a block of blocks walked with under is a nested description"
        nested_blocks (fun x ->
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
      prop "place takes a state walked with under to a set and back" states
        law_round_trip;
      test "a frozen layer's arrays are fixed arrays at its paths" (fun () ->
          let inner =
            {
              layers = [];
              steps = i32 [| 1l |];
              window = 0;
              kind = Dense;
              ends = (f32 [| 1. |], f32 [| 2. |]);
            }
          in
          let x =
            {
              model =
                {
                  layers = [];
                  steps = i32 [| 1l |];
                  window = 0;
                  kind = Dense;
                  ends = (inner, inner);
                };
              frozen = { w = f32 [| 3. |]; b = Some (f32 [| 4. |]) };
            }
          in
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
            (words state x);
          let y = P.place (module State) S.on x in
          is_on S.on y.frozen.w;
          is_on S.on (Option.get y.frozen.b);
          is_on S.on y.model.steps;
          is_on S.on (fst y.model.ends).steps;
          is_on S.on (fst (fst y.model.ends).ends);
          let z = P.cast (module State) Nx.float64 x in
          equal ~msg:"cast keeps the frozen layer" leaf (Nx.P x.frozen.w)
            (Nx.P z.frozen.w);
          equal ~msg:"cast keeps the model's fixed array" leaf
            (Nx.P x.model.steps) (Nx.P z.model.steps);
          equal ~msg:"cast keeps an inner fixed array" leaf
            (Nx.P (fst x.model.ends).steps) (Nx.P (fst z.model.ends).steps);
          let w = fst (fst z.model.ends).ends in
          equal ~msg:"dtype" string "float64" (D.name (Nx.dtype w));
          equal ~msg:"values" (array float_exact) [| 1. |] (Nx.to_array w));
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
   in float64 and rounds once to the array's format. *)
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
  cover "a NaN" (List.exists Float.is_nan all);
  cover "an infinity" (List.exists (fun v -> Float.abs v = Float.infinity) all);
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
      (* The scaled norm rounds each quotient, square and the product. *)
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

let arithmetic =
  group "arithmetic"
    [
      prop
        "dot, norm, scale and axpy over float and complex arrays, every \
         format, NaN, infinities and -0."
        vectors law_vector;
      test "norm does not overflow where the norm is representable" (fun () ->
          let h xs : host Nx.float16_t =
            Nx.create Nx.float16 [| Array.length xs |] xs
          in
          let s = P.(list tensor) and s32 = P.(list tensor) in
          equal ~msg:"float16" float_exact 500.
            (Nx.item [] (P.norm s Nx.float16 [ h [| 300. |]; h [| 400. |] ]));
          let big = Int32.float_of_bits (Int32.bits_of_float 2e38) in
          let expected =
            Int32.float_of_bits (Int32.bits_of_float (Float.sqrt 2. *. big))
          in
          equal ~msg:"float32 near its greatest"
            (float_rel ~rel:1e-6 ~abs:0.)
            expected
            (Nx.item [] (P.norm s32 Nx.float32 [ f32 [| big; big |] ]));
          equal ~msg:"zeros" float_exact 0.
            (Nx.item [] (P.norm s32 Nx.float32 [ f32 [| 0.; 0. |]; f32 [||] ]));
          invalid ~by:"Nx.Ptree.norm" (fun () -> P.norm s32 Nx.float32 []);
          invalid ~by:"Nx.Ptree.dot" (fun () -> P.dot s32 Nx.float32 [] []);
          equal ~msg:"infinity" float_exact Float.infinity
            (Nx.item []
               (P.norm s32 Nx.float32 [ f32 [| 1.; Float.neg_infinity |] ]));
          equal ~msg:"nan" float_exact Float.nan
            (Nx.item []
               (P.norm s32 Nx.float32 [ f32 [| Float.infinity; Float.nan |] ])));
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
              equal ~msg:"norm" (float 1e-12) (Float.sqrt 19.)
                (Nx.item [] (P.norm s Nx.float64 x));
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
      test "arrays of two shapes, and a factor with axes, raise" (fun () ->
          let vector = P.(pair (list tensor) tensor) in
          let x = ([ f32 [| 1.; 2. |] ], i32 [| 0l |])
          and y = ([ f32 [| 1.; 2.; 3. |] ], i32 [| 0l |]) in
          equal string
            "Nx.Ptree.dot: 0.0: [2] in the first value, [3] in the second"
            (message (fun () -> P.dot vector Nx.float32 x y));
          invalid ~by:"Nx.Ptree.axpy" (fun () ->
              P.axpy vector (Nx.scalar Nx.float32 1.) x y);
          let k = f32 [| 2. |] in
          invalid ~by:"Nx.Ptree.scale" (fun () -> P.scale vector k x);
          invalid ~by:"Nx.Ptree.axpy" (fun () -> P.axpy vector k x x));
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

(* Donation *)

let donation =
  group "donation"
    [
      test "donate gives up every array of the structure" (fun () ->
          let x = { w = f32 [| 1.; 2. |]; b = Some (f32 [| 3. |]) } in
          let d = P.donate linear x in
          equal ~msg:"its visits" (list string) (words linear x)
            (words linear d);
          equal ~msg:"every array donated" (list bool) [ true; true ]
            (List.rev
               (P.fold linear (fun _ a acc -> Nx.Prim.donated a :: acc) d []));
          ignore (Nx.add d.w d.w);
          invalid ~by:"Nx.copy" (fun () -> Nx.copy x.w);
          equal ~msg:"another array lives" (array float_exact) [| 3. |]
            (Nx.to_array (Option.get x.b)));
      test "a dead array raises naming its path" (fun () ->
          let x = { w = f32 [| 1.; 2. |]; b = None } in
          ignore (Nx.copy (Nx.donate x.w));
          raises
            (Invalid_argument
               "Nx.Ptree.donate: w: float32 [2] was donated to Nx.copy")
            (fun () -> P.donate linear x));
      test "a dead array at the root raises naming the root" (fun () ->
          let x = f32 [| 1. |] in
          ignore (Nx.copy (Nx.donate x));
          raises
            (Invalid_argument
               "Nx.Ptree.donate: the root: float32 [1] was donated to Nx.copy")
            (fun () -> P.donate P.tensor x));
      test "a structure of no array is itself" (fun () ->
          equal unit () (P.donate P.unit ()));
    ]

let () =
  exit
    (run "nx ptree"
       [
         walks;
         laws;
         messages;
         payloads;
         nesting;
         arithmetic;
         signatures;
         donation;
       ])
