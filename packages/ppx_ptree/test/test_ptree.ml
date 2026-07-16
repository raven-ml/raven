(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

let f32 values = Nx.create Nx.float32 [| Array.length values |] values
let f64 values = Nx.create Nx.float64 [| Array.length values |] values

module Tensor_alias = struct
  type t = Nx.float32_t [@@deriving ptree]
end

module Tensor_alias_as_ptree : Nx.Ptree.S with type t = Tensor_alias.t =
  Tensor_alias

module Effect_leaf = struct
  type t = (float, Nx.float32_elt) Nx_effect.t [@@deriving ptree]
end

module Effect_leaf_as_ptree : Nx.Ptree.S with type t = Effect_leaf.t =
  Effect_leaf

module All_aliases = struct
  type t = {
    float16 : Nx.float16_t;
    float32 : Nx.float32_t;
    float64 : Nx.float64_t;
    bfloat16 : Nx.bfloat16_t;
    float8_e4m3 : Nx.float8_e4m3_t;
    float8_e5m2 : Nx.float8_e5m2_t;
    int4 : Nx.int4_t;
    uint4 : Nx.uint4_t;
    int8 : Nx.int8_t;
    uint8 : Nx.uint8_t;
    int16 : Nx.int16_t;
    uint16 : Nx.uint16_t;
    int32 : Nx.int32_t;
    uint32 : Nx.uint32_t;
    int64 : Nx.int64_t;
    uint64 : Nx.uint64_t;
    complex64 : Nx.complex64_t;
    complex128 : Nx.complex128_t;
    bool : Nx.bool_t;
  }
  [@@deriving ptree]
end

module All_aliases_as_ptree : Nx.Ptree.S with type t = All_aliases.t =
  All_aliases

module Open_alias = struct
  open Nx

  type t = { value : float32_t } [@@deriving ptree]
end

module Open_alias_as_ptree : Nx.Ptree.S with type t = Open_alias.t = Open_alias

module Simple = struct
  type t = { weight : Nx.float32_t; bias : Nx.float64_t } [@@deriving ptree]
end

module Simple_as_ptree : Nx.Ptree.S with type t = Simple.t = Simple

module Generic = struct
  type 'dtype params = { weight : (float, 'dtype) Nx.t } [@@deriving ptree]
  type t = Nx.float32_elt params
end

module Generic_as_ptree : Nx.Ptree.S with type t = Generic.t = Generic

module Phantom = struct
  type 'phantom params = { weight : Nx.float32_t } [@@deriving ptree]
  type t = unit params
end

module Phantom_as_ptree : Nx.Ptree.S with type t = Phantom.t = Phantom

module Variance = struct
  type +'tag params = { weight : Nx.float32_t; tag : 'tag [@ptree.ignore] }
  [@@deriving ptree]

  type t = string params
end

module Variance_as_ptree : Nx.Ptree.S with type t = Variance.t = Variance

module Aliased_core = struct
  type t = Nx.float32_t as 'tensor [@@deriving ptree]
end

module Aliased_core_as_ptree : Nx.Ptree.S with type t = Aliased_core.t =
  Aliased_core

module type Private_signature = sig
  type t = private { weight : Nx.float32_t } [@@deriving ptree]
end

module Existing = struct
  type t = Nx.float32_t

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) value = f value
  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) = f
  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) value = f value
end

module Existing_params = struct
  type 'dtype params = { value : (float, 'dtype) Nx.t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) params =
    { value = f params.value }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) left
      right =
    { value = f left.value right.value }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) params = f params.value
end

module Automatic_params = struct
  type t = { nested : Nx.float32_elt Existing_params.params } [@@deriving ptree]
end

module Automatic_params_as_ptree : Nx.Ptree.S with type t = Automatic_params.t =
  Automatic_params

module Derived_helper = struct
  type helper = { value : Nx.float32_t } [@@deriving ptree]
end

module Handwritten_outer = struct
  type t = { inner : Derived_helper.helper }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) (value : t) =
    { inner = Derived_helper.map_helper f value.inner }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t)
      (left : t) (right : t) =
    { inner = Derived_helper.map2_helper f left.inner right.inner }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) (value : t) =
    Derived_helper.iter_helper f value.inner
end

module Handwritten_outer_as_ptree :
  Nx.Ptree.S with type t = Handwritten_outer.t =
  Handwritten_outer

module Composite = struct
  type state = { step : Nx.int64_t }

  and t = {
    pair : Nx.float32_t * Nx.float64_t;
    optional : Nx.float32_t option;
    layers : Existing.t list;
    buffers : Existing.t array;
    state : state;
    name : string; [@ptree.ignore]
  }
  [@@deriving ptree]
end

module Recursive = struct
  type node = { value : Nx.float32_t; next : node option }
  and t = { root : node } [@@deriving ptree]
end

module Annotations = struct
  type weight = Nx.float32_t

  type t = {
    weight : weight; [@ptree.leaf]
    delegated : Simple.t; [@ptree.using Simple]
    ignored : int; [@ptree.ignore]
  }
  [@@deriving ptree]
end

module Nested_annotation = struct
  type weight = Nx.float32_t
  type t = { optional : (weight[@ptree.leaf]) option } [@@deriving ptree]
end

module Ignored_unsupported = struct
  type t = {
    callback : int -> int; [@ptree.ignore]
    reference : int ref; [@ptree.ignore]
    lazy_value : int Lazy.t; [@ptree.ignore]
    table : (string, int) Hashtbl.t; [@ptree.ignore]
  }
  [@@deriving ptree]
end

module Mutual = struct
  type left = { value : Nx.float32_t; right : right option }
  and right = { value : Nx.float32_t; left : left option }
  and t = { root : left } [@@deriving ptree]
end

module Ignored_only = struct
  type t = { name : string [@ptree.ignore] } [@@deriving ptree]
end

module Ignored_alias = struct
  type t = (string[@ptree.ignore]) [@@deriving ptree]
end

module Ignored_exotic = struct
  module type S = sig end

  type t = {
    object_value : < get : int >; [@ptree.ignore]
    package_value : (module S); [@ptree.ignore]
  }
  [@@deriving ptree]
end

module Mutable = struct
  type 'tag params = {
    mutable weight : Nx.float32_t;
    tag : 'tag; [@ptree.ignore]
  }
    constraint 'tag = string
  [@@deriving ptree]
end

module Integration = struct
  type t = {
    weight : Nx.float32_t;
    bias : Nx.float64_t;
    label : string; [@ptree.ignore]
  }
  [@@deriving ptree]
end

module Integration_named = struct
  include Integration

  let names value =
    Stdlib.ignore value;
    [ "weight"; "bias" ]
end

module Jit_structure = struct
  type t = { left : Nx.float32_t list; right : Nx.float32_t list }
  [@@deriving ptree]
end

module Manual_simple = struct
  type t = Simple.t

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) (value : t) =
    Simple.{ weight = f value.weight; bias = f value.bias }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t)
      (left : t) (right : t) =
    Simple.
      { weight = f left.weight right.weight; bias = f left.bias right.bias }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) (value : t) =
    f value.weight;
    f value.bias
end

let test_single_aliases_and_delegation () =
  let tensor = f32 [| 1.; 2. |] in
  let count = ref 0 in
  let mapped =
    Tensor_alias.map
      (fun leaf ->
        incr count;
        leaf)
      tensor
  in
  equal int 1 !count;
  equal (array (float 0.)) [| 1.; 2. |] (Nx.to_array mapped);
  let automatic =
    Automatic_params.{ nested = Existing_params.{ value = tensor } }
  in
  let automatic_count = ref 0 in
  Automatic_params.iter
    (fun leaf ->
      Stdlib.ignore leaf;
      incr automatic_count)
    automatic;
  equal int 1 !automatic_count;
  let outer = Handwritten_outer.{ inner = Derived_helper.{ value = tensor } } in
  let outer_count = ref 0 in
  Handwritten_outer.iter
    (fun leaf ->
      Stdlib.ignore leaf;
      incr outer_count)
    outer;
  equal int 1 !outer_count

let test_simple_order () =
  let value = Simple.{ weight = f32 [| 1.; 2. |]; bias = f64 [| 3. |] } in
  let layouts = ref [] in
  Simple.iter (fun tensor -> layouts := Nx.numel tensor :: !layouts) value;
  equal (list int) [ 2; 1 ] (List.rev !layouts);
  let mapped = Simple.map (fun tensor -> tensor) value in
  equal (array (float 0.)) [| 1.; 2. |] (Nx.to_array mapped.weight);
  let combined =
    Simple.map2
      (fun left right ->
        Stdlib.ignore right;
        left)
      value mapped
  in
  equal (array (float 0.)) [| 3. |] (Nx.to_array combined.bias)

let test_matches_handwritten_traversal () =
  let value = Simple.{ weight = f32 [| 1.; 2. |]; bias = f64 [| 3. |] } in
  let handwritten_order = ref [] in
  Manual_simple.iter
    (fun tensor -> handwritten_order := Nx.numel tensor :: !handwritten_order)
    value;
  let derived_order = ref [] in
  Simple.iter
    (fun tensor -> derived_order := Nx.numel tensor :: !derived_order)
    value;
  equal (list int) (List.rev !handwritten_order) (List.rev !derived_order);
  let derived = Simple.map (fun tensor -> tensor) value in
  let handwritten = Manual_simple.map (fun tensor -> tensor) value in
  equal
    (array (float 0.))
    (Nx.to_array handwritten.weight)
    (Nx.to_array derived.weight);
  equal
    (array (float 0.))
    (Nx.to_array handwritten.bias)
    (Nx.to_array derived.bias)

let test_containers_and_helpers () =
  let tensor length =
    f32 (Array.init length (fun index -> float_of_int index))
  in
  let value =
    Composite.
      {
        pair = (tensor 1, f64 [| 2.; 3. |]);
        optional = Some (tensor 3);
        layers = [ tensor 4; tensor 5 ];
        buffers = [| tensor 6 |];
        state =
          { step = Nx.create Nx.int64 [| 7 |] (Array.init 7 Int64.of_int) };
        name = "left";
      }
  in
  let count = ref 0 in
  let order = ref [] in
  Composite.iter
    (fun tensor ->
      order := Nx.numel tensor :: !order;
      incr count)
    value;
  equal int 7 !count;
  equal ~msg:"containers preserve declaration and element order" (list int)
    [ 1; 2; 3; 4; 5; 6; 7 ] (List.rev !order);
  let map_count = ref 0 in
  let identity =
    Composite.map
      (fun leaf ->
        incr map_count;
        leaf)
      value
  in
  equal int 7 !map_count;
  equal string "left" identity.name;
  is_true ~msg:"array mapping returns a fresh array"
    (identity.buffers != value.buffers);
  let identity_order = ref [] in
  Composite.iter
    (fun leaf -> identity_order := Nx.numel leaf :: !identity_order)
    identity;
  equal ~msg:"map identity preserves every leaf" (list int)
    [ 1; 2; 3; 4; 5; 6; 7 ] (List.rev !identity_order);
  let right = { value with name = "right" } in
  let map2_count = ref 0 in
  let paired_sizes = ref [] in
  let mapped =
    Composite.map2
      (fun left right ->
        incr map2_count;
        paired_sizes := (Nx.numel left, Nx.numel right) :: !paired_sizes;
        left)
      value right
  in
  equal int 7 !map2_count;
  equal ~msg:"map2 pairs corresponding leaves"
    (list (pair int int))
    [ (1, 1); (2, 2); (3, 3); (4, 4); (5, 5); (6, 6); (7, 7) ]
    (List.rev !paired_sizes);
  equal string "left" mapped.name;
  raises_invalid_arg "Test_ptree.Composite.map2: list length mismatch at layers"
    (fun () ->
      Stdlib.ignore
        (Composite.map2
           (fun left right ->
             Stdlib.ignore right;
             left)
           value { right with layers = [] }));
  raises_invalid_arg
    "Test_ptree.Composite.map2: option constructor mismatch at optional"
    (fun () ->
      Stdlib.ignore
        (Composite.map2
           (fun left right ->
             Stdlib.ignore right;
             left)
           value
           { right with optional = None }));
  raises_invalid_arg
    "Test_ptree.Composite.map2: array length mismatch at buffers" (fun () ->
      Stdlib.ignore
        (Composite.map2
           (fun left right ->
             Stdlib.ignore right;
             left)
           value
           { right with buffers = [||] }))

let test_recursive () =
  let value =
    Recursive.
      {
        root =
          {
            value = f32 [| 1. |];
            next = Some { value = f32 [| 2. |]; next = None };
          };
      }
  in
  let count = ref 0 in
  Recursive.iter
    (fun tensor ->
      Stdlib.ignore tensor;
      incr count)
    value;
  equal int 2 !count;
  let mapped = Recursive.map (fun tensor -> tensor) value in
  is_some mapped.root.next;
  let mutual =
    Mutual.
      {
        root =
          {
            value = f32 [| 1. |];
            right =
              Some
                {
                  value = f32 [| 2. |];
                  left = Some { value = f32 [| 3. |]; right = None };
                };
          };
      }
  in
  let mutual_count = ref 0 in
  Mutual.iter
    (fun tensor ->
      Stdlib.ignore tensor;
      incr mutual_count)
    mutual;
  equal int 3 !mutual_count

let test_annotations_and_ignored () =
  let simple = Simple.{ weight = f32 [| 2. |]; bias = f64 [| 3. |] } in
  let value =
    Annotations.{ weight = f32 [| 1. |]; delegated = simple; ignored = 7 }
  in
  let count = ref 0 in
  Annotations.iter
    (fun tensor ->
      Stdlib.ignore tensor;
      incr count)
    value;
  equal int 3 !count;
  let right = { value with ignored = 99 } in
  let mapped =
    Annotations.map2
      (fun left right ->
        Stdlib.ignore right;
        left)
      value right
  in
  equal int 7 mapped.ignored;
  let metadata = Ignored_only.{ name = "left" } in
  equal string "left" (Ignored_only.map (fun tensor -> tensor) metadata).name;
  equal string "left"
    (Ignored_only.map2
       (fun left right ->
         Stdlib.ignore right;
         left)
       metadata
       Ignored_only.{ name = "right" })
      .name;
  equal string "left" (Ignored_alias.map (fun tensor -> tensor) "left");
  let nested = Nested_annotation.{ optional = Some (f32 [| 4. |]) } in
  let nested_count = ref 0 in
  Nested_annotation.iter
    (fun tensor ->
      Stdlib.ignore tensor;
      incr nested_count)
    nested;
  equal int 1 !nested_count

let test_generic_and_mutable () =
  let generic = Generic.{ weight = f32 [| 1. |] } in
  equal
    (array (float 0.))
    [| 1. |]
    (Nx.to_array (Generic.map (fun tensor -> tensor) generic).weight);
  let generic64 = Generic.{ weight = f64 [| 2. |] } in
  equal
    (array (float 0.))
    [| 2. |]
    (Nx.to_array (Generic.map (fun tensor -> tensor) generic64).weight);
  let mutable_value = Mutable.{ weight = f32 [| 2. |]; tag = "tag" } in
  let mapped = Mutable.map (fun tensor -> tensor) mutable_value in
  equal string "tag" mapped.tag;
  is_true (mapped != mutable_value)

let integration_loss (params : Integration.t) =
  Nx.add
    (Nx.cast Nx.float64 (Nx.sum (Nx.mul params.weight params.weight)))
    (Nx.sum (Nx.mul params.bias params.bias))

let test_rune_and_vega_integration () =
  let params =
    Integration.
      {
        weight = f32 [| 1.; -2.; 3. |];
        bias = f64 [| 0.5 |];
        label = "first trace";
      }
  in
  let gradients = Rune.grad (module Integration) integration_loss params in
  equal (array (float 0.)) [| 2.; -4.; 6. |] (Nx.to_array gradients.weight);
  equal (array (float 0.)) [| 1. |] (Nx.to_array gradients.bias);
  is_true (Vega.global_norm (module Integration) gradients > 0.);
  let checkpoint =
    Kaun.Checkpoint.of_params (module Integration_named) params
  in
  equal (list string) [ "bias"; "weight" ] (Kaun.Checkpoint.names checkpoint);
  let jitted =
    Rune.jit2
      (module Integration)
      (module Integration)
      (fun value -> Rune.grad (module Integration) integration_loss value)
  in
  let first = jitted params in
  equal (array (float 0.)) [| 2.; -4.; 6. |] (Nx.to_array first.weight);
  let second_params =
    Integration.
      {
        weight = f32 [| 2.; 3.; 4. |];
        bias = f64 [| 1. |];
        label = "changed after trace";
      }
  in
  let second = jitted second_params in
  equal (array (float 0.)) [| 4.; 6.; 8. |] (Nx.to_array second.weight);
  equal ~msg:"ignored metadata is fixed by the first JIT trace" string
    "first trace" second.label;
  let replay_structure =
    Rune.jit2 (module Jit_structure) (module Jit_structure) Fun.id
  in
  let first_structure =
    Jit_structure.{ left = [ f32 [| 1. |] ]; right = [ f32 [| 2. |] ] }
  in
  let first_structure_result = replay_structure first_structure in
  equal int 1 (List.length first_structure_result.left);
  let changed_structure =
    Jit_structure.{ left = [ f32 [| 3. |]; f32 [| 4. |] ]; right = [] }
  in
  let replayed = replay_structure changed_structure in
  equal ~msg:"container structure is fixed by the first JIT trace" int 1
    (List.length replayed.left);
  equal ~msg:"container structure is fixed by the first JIT trace" int 1
    (List.length replayed.right)

let tests =
  [
    test "supports tensor aliases and both delegation directions"
      test_single_aliases_and_delegation;
    test "preserves mixed-dtype leaf order" test_simple_order;
    test "matches a handwritten traversal" test_matches_handwritten_traversal;
    test "traverses containers and helper types" test_containers_and_helpers;
    test "supports recursive products" test_recursive;
    test "honors leaf, using, and ignore annotations"
      test_annotations_and_ignored;
    test "supports generic dtype and mutable records" test_generic_and_mutable;
    test "works directly with Rune and Vega" test_rune_and_vega_integration;
  ]

let () = run "ppx_ptree" tests
