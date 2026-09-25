(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Each derived walk visits what a hand-written walk of the same type visits:
   the same leaves and reports, at the same paths, in the same order. *)

open Windtrap

let f32 = Nx.float32
let vec xs = Nx.create f32 [| Array.length xs |] xs
let ints xs = Nx.create Nx.int32 [| Array.length xs |] xs

let visits s x =
  List.map (Format.asprintf "%a" Nx.Ptree.pp_visit) (Nx.Ptree.visits s x)

let same ~msg ~hand derived x =
  let expected = visits hand x in
  is_true ~msg:(msg ^ " visits something") (expected <> []);
  equal ~msg (list string) expected (visits derived x)

(* Parameter positions, options, lists and tuples. *)

module Record = struct
  type 'a t = { w : 'a; b : 'a option; layers : 'a list; pair : 'a * 'a }
  [@@deriving ptree]
end

module Record_hand = struct
  type 'a t = 'a Record.t

  let walk c (x : _ t) : _ t =
    let open Nx.Ptree.Walk in
    let w = field c "w" leaf x.w in
    let b = field c "b" (option leaf) x.b in
    let layers = field c "layers" (list leaf) x.layers in
    let pair =
      field c "pair"
        (fun c (p, q) ->
          let p = index c 0 leaf p in
          let q = index c 1 leaf q in
          (p, q))
        x.pair
    in
    { w; b; layers; pair }
end

let test_record () =
  let x =
    Record.
      {
        w = vec [| 1. |];
        b = None;
        layers = [ vec [| 2. |]; vec [| 3. |] ];
        pair = (vec [| 4. |], vec [| 5. |]);
      }
  in
  same ~msg:"record"
    ~hand:(Nx.Ptree.instantiate (module Record_hand))
    (Nx.Ptree.instantiate (module Record))
    x;
  same ~msg:"present option"
    ~hand:(Nx.Ptree.instantiate (module Record_hand))
    (Nx.Ptree.instantiate (module Record))
    { x with b = Some (vec [| 6. |]) }

(* Tensors of a fixed type, and the structure of a type without parameter. *)

module Tensors = struct
  open Nx

  type t = {
    scale : Nx.float32_t;
    shift : (float, Nx.float64_elt) Nx.t;
    key : Nx.Rng.key;
    count : int32_t;
  }
  [@@deriving ptree]
end

module Tensors_hand = struct
  type _ t = Tensors.t

  let walk c (x : _ t) : _ t =
    let open Nx.Ptree.Walk in
    let scale = field c "scale" tensor x.scale in
    let shift = field c "shift" tensor x.shift in
    let key = field c "key" tensor x.key in
    let count = field c "count" tensor x.count in
    { scale; shift; key; count }
end

let tensors =
  Tensors.
    {
      scale = vec [| 1. |];
      shift = Nx.create Nx.float64 [| 1 |] [| 2. |];
      key = Nx.Rng.key 0;
      count = ints [| 3l |];
    }

let test_tensors () =
  same ~msg:"tensors"
    ~hand:(Nx.Ptree.instantiate (module Tensors_hand))
    Tensors.ptree tensors;
  let doubled = Nx.Ptree.map Tensors.ptree (fun _ t -> Nx.add t t) tensors in
  equal ~msg:"ptree maps the tensors" (array float_exact) [| 4. |]
    (Nx.to_array doubled.Tensors.shift)

(* Structures: a module's walk at the parameter, a structure at one type for a
   type without parameter, and a fixed instance of a module's [t]. *)

module Model = struct
  type 'a block = { attn : 'a Kaun.Linear.t; mlp : 'a Kaun.Linear.t list }

  and 'a t = {
    blocks : 'a block list;
    head : 'a Kaun.Linear.t;
    stats : Tensors.t;
    index : Kaun.Cache_index.t;
    frozen : Nx.float32_t Kaun.Linear.t;
  }
  [@@deriving ptree]
end

module Model_hand = struct
  type 'a t = 'a Model.t

  let block c (b : _ Model.block) : _ Model.block =
    let open Nx.Ptree.Walk in
    let attn = field c "attn" Kaun.Linear.walk b.attn in
    let mlp = field c "mlp" (list Kaun.Linear.walk) b.mlp in
    { attn; mlp }

  let walk c (m : _ t) : _ t =
    let open Nx.Ptree.Walk in
    let blocks = field c "blocks" (list block) m.blocks in
    let head = field c "head" Kaun.Linear.walk m.head in
    let stats = field c "stats" (structure Tensors.ptree) m.stats in
    let index = field c "index" (structure Kaun.Cache_index.ptree) m.index in
    let frozen =
      field c "frozen"
        (structure (Nx.Ptree.instantiate (module Kaun.Linear)))
        m.frozen
    in
    { blocks; head; stats; index; frozen }
end

let model =
  Model.
    {
      blocks =
        [
          {
            attn = Kaun.Linear.init ~inputs:2 ~outputs:2;
            mlp = [ Kaun.Linear.init ~inputs:2 ~outputs:2 ];
          };
        ];
      head = Kaun.Linear.init ~inputs:2 ~outputs:1;
      stats = tensors;
      index =
        Kaun.Cache_index.window 4 (Kaun.Cache_index.rows ~context:8 [| 0 |]);
      frozen = Kaun.Linear.init ~inputs:1 ~outputs:1;
    }

let test_model () =
  same ~msg:"model"
    ~hand:(Nx.Ptree.instantiate (module Model_hand))
    (Nx.Ptree.instantiate (module Model))
    model;
  let cast = Nx.Ptree.cast (module Model) Nx.float16 model in
  is_true ~msg:"cast casts the parameter" (Nx.dtype cast.head.w = Nx.float16);
  is_true ~msg:"cast keeps a fixed instance" (cast.frozen.w == model.frozen.w)

(* [@ptree.walk e] walks a part with [e]. *)

let model_ptree = Nx.Ptree.instantiate (module Model)
let adam = Vega.adam_ptree model_ptree

module State = struct
  type t = {
    params : Nx.float32_t Model.t;
        [@ptree.walk Nx.Ptree.Walk.structure model_ptree]
    opt : Nx.float32_t Model.t Vega.adam_state;
        [@ptree.walk Nx.Ptree.Walk.structure adam]
  }
  [@@deriving ptree]
end

module State_hand = struct
  type _ t = State.t

  let walk c (s : _ t) : _ t =
    let open Nx.Ptree.Walk in
    let params = field c "params" (structure model_ptree) s.params in
    let opt = field c "opt" (structure adam) s.opt in
    { params; opt }
end

let test_walk_attribute () =
  let state =
    State.{ params = model; opt = Vega.adam_init model_ptree model }
  in
  same ~msg:"walk attribute"
    ~hand:(Nx.Ptree.instantiate (module State_hand))
    State.ptree state

(* [@ptree.int] reports integers and bools; [@ptree.skip] copies a part. *)

module Data = struct
  type 'a t = {
    w : 'a;
    window : int option; [@ptree.int]
    causal : bool; [@ptree.int]
    sizes : (int[@ptree.int]) list;
    name : string; [@ptree.skip]
  }
  [@@deriving ptree]
end

module Data_hand = struct
  type 'a t = 'a Data.t

  let walk c (x : _ t) : _ t =
    let open Nx.Ptree.Walk in
    let w = field c "w" leaf x.w in
    let window = field c "window" (option int) x.window in
    let causal =
      field c "causal" (fun c b -> int c (Bool.to_int b) <> 0) x.causal
    in
    let sizes = field c "sizes" (list int) x.sizes in
    { w; window; causal; sizes; name = x.name }
end

let test_data () =
  let x =
    Data.
      {
        w = vec [| 1. |];
        window = Some 8;
        causal = true;
        sizes = [ 2; 3 ];
        name = "block";
      }
  in
  let derived = Nx.Ptree.instantiate (module Data) in
  same ~msg:"data" ~hand:(Nx.Ptree.instantiate (module Data_hand)) derived x;
  let y = Nx.Ptree.map derived (fun _ t -> t) x in
  equal ~msg:"skip copies the part" string "block" y.name;
  is_true ~msg:"a reported bool rebuilds" y.causal;
  equal ~msg:"a reported int rebuilds" (option int) (Some 8) y.window

(* Variants name their case before their parts. *)

module Weight = struct
  type 'a t =
    | Float of 'a
    | Mxfp4 of { blocks : Nx.uint8_t; scales : Nx.uint8_t }
    | Scaled of 'a * Nx.float32_t
    | Tied
  [@@deriving ptree]
end

module Weight_hand = struct
  type 'a t = 'a Weight.t

  let walk c : _ t -> _ t =
    let open Nx.Ptree.Walk in
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
    | Tied ->
        case c "Tied";
        Tied
end

let test_variant () =
  let u8 = Nx.create Nx.uint8 [| 1 |] [| 7 |] in
  List.iter
    (fun (msg, x) ->
      same ~msg
        ~hand:(Nx.Ptree.instantiate (module Weight_hand))
        (Nx.Ptree.instantiate (module Weight))
        x)
    [
      ("constructor with one argument", Weight.Float (vec [| 1. |]));
      ("inline record", Weight.Mxfp4 { blocks = u8; scales = u8 });
      ( "constructor with two arguments",
        Weight.Scaled (vec [| 1. |], vec [| 2. |]) );
      ("constant constructor", Weight.Tied);
    ]

(* An attribute after a constructor's single argument applies to it. *)

module Marked = struct
  type 'a t =
    | Plain of 'a
    | Window of int [@ptree.int]
    | Frozen of Nx.float32_t [@ptree.skip]
  [@@deriving ptree]
end

module Marked_hand = struct
  type 'a t = 'a Marked.t

  let walk c : _ t -> _ t =
    let open Nx.Ptree.Walk in
    function
    | Plain x ->
        case c "Plain";
        Plain (leaf c x)
    | Window n ->
        case c "Window";
        Window (int c n)
    | Frozen x ->
        case c "Frozen";
        Frozen x
end

let test_constructor_attributes () =
  List.iter
    (fun (msg, x) ->
      same ~msg
        ~hand:(Nx.Ptree.instantiate (module Marked_hand))
        (Nx.Ptree.instantiate (module Marked))
        x)
    [
      ("an argument", Marked.Plain (vec [| 1. |]));
      ("[@ptree.int] on a constructor", Marked.Window 4);
      ("[@ptree.skip] on a constructor", Marked.Frozen (vec [| 2. |]));
    ]

(* Recursive types, arrays and aliases. *)

module Tree = struct
  type 'a t = Leaf of 'a | Node of 'a t list [@@deriving ptree]
end

module Tree_hand = struct
  type 'a t = 'a Tree.t

  let rec walk c : _ t -> _ t =
    let open Nx.Ptree.Walk in
    function
    | Leaf x ->
        case c "Leaf";
        Leaf (leaf c x)
    | Node l ->
        case c "Node";
        Node (list walk c l)
end

let test_recursive () =
  same ~msg:"recursive"
    ~hand:(Nx.Ptree.instantiate (module Tree_hand))
    (Nx.Ptree.instantiate (module Tree))
    Tree.(Node [ Leaf (vec [| 1. |]); Node [ Leaf (vec [| 2. |]) ] ])

module Stack = struct
  type 'a t = { layers : 'a array } [@@deriving ptree]
end

module Stack_hand = struct
  type 'a t = 'a Stack.t

  let walk c (x : _ t) : _ t =
    let open Nx.Ptree.Walk in
    let layers =
      field c "layers"
        (fun c a ->
          ignore (int c (Array.length a));
          Array.mapi (fun i x -> index c i leaf x) a)
        x.layers
    in
    { layers }
end

let test_array () =
  same ~msg:"array"
    ~hand:(Nx.Ptree.instantiate (module Stack_hand))
    (Nx.Ptree.instantiate (module Stack))
    Stack.{ layers = [| vec [| 1. |]; vec [| 2. |] |] }

module Alias = struct
  type 'a t = 'a Kaun.Linear.t * 'a [@@deriving ptree]
end

module Alias_hand = struct
  type 'a t = 'a Alias.t

  let walk c ((l, x) : _ t) : _ t =
    let open Nx.Ptree.Walk in
    let l = index c 0 Kaun.Linear.walk l in
    let x = index c 1 leaf x in
    (l, x)
end

let test_alias () =
  same ~msg:"alias"
    ~hand:(Nx.Ptree.instantiate (module Alias_hand))
    (Nx.Ptree.instantiate (module Alias))
    (Kaun.Linear.init ~inputs:1 ~outputs:1, vec [| 1. |])

(* A type whose parameter is anonymous is a module of [Nx.Ptree.S]. *)

module Phantom = struct
  type _ t = { scale : Nx.float32_t } [@@deriving ptree]
end

let test_phantom () =
  equal ~msg:"phantom" (list string) [ "scale: a leaf" ]
    (visits
       (Nx.Ptree.instantiate (module Phantom))
       { Phantom.scale = vec [| 1. |] })

(* A derived structure serves transformations. *)

let test_grad () =
  let loss (m : Nx.float32_t Model.t) =
    Nx.sum (Kaun.Linear.apply m.head (vec [| 1.; 2. |]))
  in
  let derived = Rune.grad model_ptree loss model in
  let hand = Rune.grad (Nx.Ptree.instantiate (module Model_hand)) loss model in
  equal ~msg:"gradients" (array float_exact) (Nx.to_array hand.head.w)
    (Nx.to_array derived.head.w)

let () =
  run "ppx_ptree"
    [
      group "walks"
        [
          test "records, options, lists and tuples" test_record;
          test "tensors and a type without parameter" test_tensors;
          test "structures" test_model;
          test "[@ptree.walk]" test_walk_attribute;
          test "[@ptree.int] and [@ptree.skip]" test_data;
          test "variants" test_variant;
          test "attributes on constructors" test_constructor_attributes;
          test "recursive types" test_recursive;
          test "arrays" test_array;
          test "aliases" test_alias;
          test "an anonymous parameter" test_phantom;
        ];
      group "transformations" [ test "grad" test_grad ];
    ]
