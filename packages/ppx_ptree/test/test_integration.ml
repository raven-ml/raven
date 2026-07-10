(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

let f32 shape values = Nx.create Nx.float32 shape values

module Pmap_input = struct
  type t = { rows : Nx.float32_t; columns : Nx.float32_t } [@@deriving ptree]
end

module Pmap_output = struct
  type t = Nx.float32_t * Nx.float32_t [@@deriving ptree]
end

module Optimizer_params = struct
  type t = {
    weight : Nx.float32_t;
    bias : Nx.float32_t;
    label : string; [@ptree.ignore]
  }
  [@@deriving ptree]
end

module Block = struct
  type t = { projection : Kaun.Linear.t } [@@deriving ptree]
end

module Model = struct
  type t = {
    stem : Kaun.Linear.t;
    blocks : Block.t list;
    head : Kaun.Linear.t option;
  }
  [@@deriving ptree]
end

module Model_named = struct
  include Model

  let linear_names prefix linear =
    List.map (fun name -> prefix ^ "." ^ name) (Kaun.Linear.names linear)

  let names model =
    let stem = linear_names "stem" model.stem in
    let blocks =
      List.mapi
        (fun index block ->
          linear_names
            (Printf.sprintf "blocks.%d.projection" index)
            block.Block.projection)
        model.blocks
      |> List.concat
    in
    let head =
      match model.head with
      | None -> []
      | Some linear -> linear_names "head" linear
    in
    stem @ blocks @ head
end

let test_pmap_axes_and_tuple_output () =
  let rows = f32 [| 4; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |] in
  let columns = f32 [| 2; 4 |] [| 10.; 20.; 30.; 40.; 50.; 60.; 70.; 80. |] in
  let input = Pmap_input.{ rows; columns } in
  let devices = [ "CPU:1"; "CPU:2" ] in
  let axes = [ Some 0; Some 1 ] in
  let eager = Nx.add rows (Nx.transpose columns) in
  let parallel =
    Rune.pmap ~devices ~in_axes:axes
      (module Pmap_input)
      (fun value -> Nx.add value.rows (Nx.transpose value.columns))
  in
  equal (array (float 0.)) (Nx.to_array eager) (Nx.to_array (parallel input));
  let tuple_parallel =
    Rune.pmap2 ~devices ~in_axes:axes
      (module Pmap_input)
      (module Pmap_output)
      (fun value -> (value.rows, Nx.transpose value.columns))
  in
  let rows_result, columns_result = tuple_parallel input in
  equal (array (float 0.)) (Nx.to_array rows) (Nx.to_array rows_result);
  equal
    (array (float 0.))
    (Nx.to_array (Nx.transpose columns))
    (Nx.to_array columns_result)

let optimizer_loss (params : Optimizer_params.t) =
  Nx.add (Nx.sum (Nx.square params.weight)) (Nx.sum (Nx.square params.bias))

let test_vega_optimizer_step () =
  let params =
    Optimizer_params.
      {
        weight = f32 [| 2 |] [| 1.; -2. |];
        bias = f32 [| 1 |] [| 0.5 |];
        label = "model";
      }
  in
  let gradients = Rune.grad (module Optimizer_params) optimizer_loss params in
  let state = Vega.sgd_init (module Optimizer_params) params in
  let updated, state =
    Vega.sgd_step
      (module Optimizer_params)
      ~lr:0.1 state ~params ~grads:gradients
  in
  equal (array (float 1e-6)) [| 0.8; -1.6 |] (Nx.to_array updated.weight);
  equal (array (float 1e-6)) [| 0.4 |] (Nx.to_array updated.bias);
  equal string "model" updated.label;
  equal (array (float 1e-6)) [| 2.; -4. |] (Nx.to_array state.velocity.weight)

let test_nested_kaun_model_and_checkpoint_order () =
  Nx.Rng.run ~seed:7 @@ fun () ->
  let model =
    Model.
      {
        stem = Kaun.Linear.init ~inputs:2 ~outputs:3;
        blocks =
          [
            Block.{ projection = Kaun.Linear.init ~inputs:3 ~outputs:3 };
            Block.{ projection = Kaun.Linear.init ~inputs:3 ~outputs:3 };
          ];
        head = Some (Kaun.Linear.init ~inputs:3 ~outputs:1);
      }
  in
  let traversal_count = ref 0 in
  Model.iter
    (fun tensor ->
      Stdlib.ignore tensor;
      incr traversal_count)
    model;
  let names = Model_named.names model in
  equal ~msg:"one checkpoint name per generated traversal leaf" int
    (List.length names) !traversal_count;
  equal (list string)
    [
      "stem.w";
      "stem.b";
      "blocks.0.projection.w";
      "blocks.0.projection.b";
      "blocks.1.projection.w";
      "blocks.1.projection.b";
      "head.w";
      "head.b";
    ]
    names;
  let checkpoint = Kaun.Checkpoint.of_params (module Model_named) model in
  List.iter
    (fun name ->
      is_some
        ~msg:("checkpoint contains " ^ name)
        (Kaun.Checkpoint.find name checkpoint))
    names

let tests =
  [
    test "preserves derived leaf order through pmap axes and tuple output"
      test_pmap_axes_and_tuple_output;
    test "runs a Vega optimizer step over a derived module"
      test_vega_optimizer_step;
    test "keeps nested Kaun checkpoint names aligned with generated iter"
      test_nested_kaun_model_and_checkpoint_order;
  ]

let () = run "ppx_ptree integration" tests
