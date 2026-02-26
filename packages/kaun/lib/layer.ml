(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type 'layout vars = {
  params : Ptree.t;
  state : Ptree.t;
  dtype : (float, 'layout) Rune.dtype;
}

type ('input, 'output) t = {
  init : 'layout. dtype:(float, 'layout) Rune.dtype -> 'layout vars;
  apply :
    'layout 'in_elt.
    params:Ptree.t ->
    state:Ptree.t ->
    dtype:(float, 'layout) Rune.dtype ->
    training:bool ->
    ?ctx:Context.t ->
    ('input, 'in_elt) Rune.t ->
    ('output, 'layout) Rune.t * Ptree.t;
}

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let params v = v.params
let state v = v.state
let dtype v = v.dtype
let with_params v params = { v with params }
let with_state v state = { v with state }
let make_vars ~params ~state ~dtype = { params; state; dtype }

module Dtype = Nx_core.Dtype

let require_same_float_dtype (type p in_elt) ~ctx
    (expected : (float, p) Rune.dtype) (x : (float, in_elt) Rune.t) :
    (float, p) Rune.t =
  match Dtype.equal_witness expected (Rune.dtype x) with
  | Some Type.Equal -> (x : (float, p) Rune.t)
  | None ->
      invalid_argf "%s: input dtype %s does not match model dtype %s" ctx
        (Dtype.to_string (Rune.dtype x))
        (Dtype.to_string expected)

let require_int32_indices (type in_elt) ~ctx (x : (int32, in_elt) Rune.t) :
    (int32, Bigarray.int32_elt) Rune.t =
  match Dtype.equal_witness Rune.int32 (Rune.dtype x) with
  | Some Type.Equal -> (x : (int32, Bigarray.int32_elt) Rune.t)
  | None ->
      invalid_argf "%s: expected int32 indices, got %s" ctx
        (Dtype.to_string (Rune.dtype x))

let init t ~dtype = t.init ~dtype

let apply (type a b layout in_elt) (t : (a, b) t) (vars : layout vars) ~training
    ?ctx (x : (a, in_elt) Rune.t) =
  let y, state =
    t.apply ~params:vars.params ~state:vars.state ~dtype:vars.dtype ~training
      ?ctx x
  in
  (y, { vars with state })

let compose left right =
  {
    init =
      (fun ~dtype ->
        let k1 = Rune.Rng.next_key () in
        let k2 = Rune.Rng.next_key () in
        let left_vars = Rune.Rng.with_key k1 (fun () -> left.init ~dtype) in
        let right_vars = Rune.Rng.with_key k2 (fun () -> right.init ~dtype) in
        {
          params =
            Ptree.dict
              [ ("left", left_vars.params); ("right", right_vars.params) ];
          state =
            Ptree.dict
              [ ("left", left_vars.state); ("right", right_vars.state) ];
          dtype;
        });
    apply =
      (fun ~params ~state ~dtype ~training ?ctx x ->
        let param_fields =
          Ptree.Dict.fields_exn ~ctx:"Layer.compose.params" params
        in
        let state_fields =
          Ptree.Dict.fields_exn ~ctx:"Layer.compose.state" state
        in
        let left_params =
          Ptree.Dict.find_exn ~ctx:"Layer.compose.params" "left" param_fields
        in
        let right_params =
          Ptree.Dict.find_exn ~ctx:"Layer.compose.params" "right" param_fields
        in
        let left_state =
          Ptree.Dict.find_exn ~ctx:"Layer.compose.state" "left" state_fields
        in
        let right_state =
          Ptree.Dict.find_exn ~ctx:"Layer.compose.state" "right" state_fields
        in
        let y, left_state' =
          left.apply ~params:left_params ~state:left_state ~dtype ~training ?ctx
            x
        in
        let z, right_state' =
          right.apply ~params:right_params ~state:right_state ~dtype ~training
            ?ctx y
        in
        (z, Ptree.dict [ ("left", left_state'); ("right", right_state') ]));
  }

(* Dense *)

let linear ~in_features ~out_features ?weight_init ?bias_init () =
  let weight_init =
    match weight_init with
    | Some init_value -> init_value
    | None -> Init.glorot_uniform ()
  in
  let bias_init =
    match bias_init with Some init_value -> init_value | None -> Init.zeros
  in
  {
    init =
      (fun ~dtype ->
        let weight = weight_init.f [| in_features; out_features |] dtype in
        let bias = bias_init.f [| out_features |] dtype in
        {
          params =
            Ptree.dict
              [ ("weight", Ptree.tensor weight); ("bias", Ptree.tensor bias) ];
          state = Ptree.empty;
          dtype;
        });
    apply =
      (fun ~params ~state ~dtype ~training ?ctx x ->
        ignore (training, ctx);
        let x = require_same_float_dtype ~ctx:"Layer.linear" dtype x in
        let fields = Ptree.Dict.fields_exn ~ctx:"Layer.linear" params in
        let weight = Ptree.Dict.get_tensor_exn fields ~name:"weight" dtype in
        let bias = Ptree.Dict.get_tensor_exn fields ~name:"bias" dtype in
        (Rune.add (Rune.matmul x weight) bias, state));
  }

(* Convolution *)

let conv1d ~in_channels ~out_channels ?(kernel_size = 3) ?(stride = 1)
    ?(dilation = 1) ?(padding = `Same) () =
  let weight_init = Init.glorot_uniform ~in_axis:1 ~out_axis:0 () in
  {
    init =
      (fun ~dtype ->
        let weight =
          weight_init.f [| out_channels; in_channels; kernel_size |] dtype
        in
        let bias = Rune.zeros dtype [| out_channels |] in
        {
          params =
            Ptree.dict
              [ ("weight", Ptree.tensor weight); ("bias", Ptree.tensor bias) ];
          state = Ptree.empty;
          dtype;
        });
    apply =
      (fun ~params ~state ~dtype ~training ?ctx x ->
        ignore (training, ctx);
        let x = require_same_float_dtype ~ctx:"Layer.conv1d" dtype x in
        let fields = Ptree.Dict.fields_exn ~ctx:"Layer.conv1d" params in
        let weight = Ptree.Dict.get_tensor_exn fields ~name:"weight" dtype in
        let bias = Ptree.Dict.get_tensor_exn fields ~name:"bias" dtype in
        let x =
          match padding with
          | `Same | `Valid -> x
          | `Causal ->
              let pad_left = (kernel_size - 1) * dilation in
              Rune.pad [| (0, 0); (0, 0); (pad_left, 0) |] 0.0 x
        in
        let padding =
          match padding with `Same -> `Same | `Valid | `Causal -> `Valid
        in
        (Fn.conv1d ~stride ~dilation ~padding ~bias x weight, state));
  }

let conv2d ~in_channels ~out_channels ?(kernel_size = (3, 3)) () =
  let kh, kw = kernel_size in
  let weight_init = Init.glorot_uniform ~in_axis:1 ~out_axis:0 () in
  {
    init =
      (fun ~dtype ->
        let weight =
          weight_init.f [| out_channels; in_channels; kh; kw |] dtype
        in
        let bias = Rune.zeros dtype [| out_channels |] in
        {
          params =
            Ptree.dict
              [ ("weight", Ptree.tensor weight); ("bias", Ptree.tensor bias) ];
          state = Ptree.empty;
          dtype;
        });
    apply =
      (fun ~params ~state ~dtype ~training ?ctx x ->
        ignore (training, ctx);
        let x = require_same_float_dtype ~ctx:"Layer.conv2d" dtype x in
        let fields = Ptree.Dict.fields_exn ~ctx:"Layer.conv2d" params in
        let weight = Ptree.Dict.get_tensor_exn fields ~name:"weight" dtype in
        let bias = Ptree.Dict.get_tensor_exn fields ~name:"bias" dtype in
        (Fn.conv2d ~padding:`Same ~bias x weight, state));
  }

(* Normalization *)

let layer_norm ~dim ?(eps = 1e-5) () =
  {
    init =
      (fun ~dtype ->
        let gamma = Rune.ones dtype [| dim |] in
        let beta = Rune.zeros dtype [| dim |] in
        {
          params =
            Ptree.dict
              [ ("gamma", Ptree.tensor gamma); ("beta", Ptree.tensor beta) ];
          state = Ptree.empty;
          dtype;
        });
    apply =
      (fun ~params ~state ~dtype ~training ?ctx x ->
        ignore (training, ctx);
        let x = require_same_float_dtype ~ctx:"Layer.layer_norm" dtype x in
        let fields = Ptree.Dict.fields_exn ~ctx:"Layer.layer_norm" params in
        let gamma = Ptree.Dict.get_tensor_exn fields ~name:"gamma" dtype in
        let beta = Ptree.Dict.get_tensor_exn fields ~name:"beta" dtype in
        (Fn.layer_norm ~gamma ~beta ~epsilon:eps x, state));
  }

let rms_norm ~dim ?(eps = 1e-6) () =
  {
    init =
      (fun ~dtype ->
        let scale = Rune.ones dtype [| dim |] in
        {
          params = Ptree.dict [ ("scale", Ptree.tensor scale) ];
          state = Ptree.empty;
          dtype;
        });
    apply =
      (fun ~params ~state ~dtype ~training ?ctx x ->
        ignore (training, ctx);
        let x = require_same_float_dtype ~ctx:"Layer.rms_norm" dtype x in
        let fields = Ptree.Dict.fields_exn ~ctx:"Layer.rms_norm" params in
        let scale = Ptree.Dict.get_tensor_exn fields ~name:"scale" dtype in
        (Fn.rms_norm ~gamma:scale ~epsilon:eps x, state));
  }

let batch_norm ~num_features () =
  {
    init =
      (fun ~dtype ->
        let scale = Rune.ones dtype [| num_features |] in
        let bias = Rune.zeros dtype [| num_features |] in
        let running_mean = Rune.zeros dtype [| num_features |] in
        let running_var = Rune.ones dtype [| num_features |] in
        {
          params =
            Ptree.dict
              [ ("scale", Ptree.tensor scale); ("bias", Ptree.tensor bias) ];
          state =
            Ptree.dict
              [
                ("running_mean", Ptree.tensor running_mean);
                ("running_var", Ptree.tensor running_var);
              ];
          dtype;
        });
    apply =
      (fun ~params ~state ~dtype ~training ?ctx x ->
        ignore ctx;
        let x = require_same_float_dtype ~ctx:"Layer.batch_norm" dtype x in
        let params_fields =
          Ptree.Dict.fields_exn ~ctx:"Layer.batch_norm.params" params
        in
        let state_fields =
          Ptree.Dict.fields_exn ~ctx:"Layer.batch_norm.state" state
        in
        let scale =
          Ptree.Dict.get_tensor_exn params_fields ~name:"scale" dtype
        in
        let bias = Ptree.Dict.get_tensor_exn params_fields ~name:"bias" dtype in
        let running_mean =
          Ptree.Dict.get_tensor_exn state_fields ~name:"running_mean" dtype
        in
        let running_var =
          Ptree.Dict.get_tensor_exn state_fields ~name:"running_var" dtype
        in
        let rank = Array.length (Rune.shape x) in
        let axes =
          match rank with
          | 2 -> [ 0 ]
          | 3 -> [ 0; 2 ]
          | 4 -> [ 0; 2; 3 ]
          | _ -> [ 0 ]
        in
        if training then
          let momentum = 0.99 in
          let one_minus = 1.0 -. momentum in
          let batch_mean = Rune.mean ~axes x in
          let batch_var = Rune.var ~axes x in
          let y = Fn.batch_norm ~axes ~scale ~bias x in
          let running_mean' =
            Rune.add
              (Rune.mul running_mean (Rune.scalar dtype momentum))
              (Rune.mul batch_mean (Rune.scalar dtype one_minus))
          in
          let running_var' =
            Rune.add
              (Rune.mul running_var (Rune.scalar dtype momentum))
              (Rune.mul batch_var (Rune.scalar dtype one_minus))
          in
          let state' =
            Ptree.dict
              [
                ("running_mean", Ptree.tensor running_mean');
                ("running_var", Ptree.tensor running_var');
              ]
          in
          (y, state')
        else
          let scale_eval, bias_eval =
            match rank with
            | 2 ->
                ( Rune.reshape [| 1; num_features |] scale,
                  Rune.reshape [| 1; num_features |] bias )
            | 3 ->
                ( Rune.reshape [| 1; num_features; 1 |] scale,
                  Rune.reshape [| 1; num_features; 1 |] bias )
            | 4 ->
                ( Rune.reshape [| 1; num_features; 1; 1 |] scale,
                  Rune.reshape [| 1; num_features; 1; 1 |] bias )
            | _ -> (scale, bias)
          in
          let y =
            Rune.standardize ~axes ~mean:running_mean ~variance:running_var x
            |> fun normalized ->
            Rune.add (Rune.mul normalized scale_eval) bias_eval
          in
          (y, state));
  }

(* Embedding *)

let embedding ~vocab_size ~embed_dim ?(scale = true) () =
  let emb_init = Init.normal ~stddev:0.02 () in
  {
    init =
      (fun ~dtype ->
        let embedding = emb_init.f [| vocab_size; embed_dim |] dtype in
        {
          params = Ptree.dict [ ("embedding", Ptree.tensor embedding) ];
          state = Ptree.empty;
          dtype;
        });
    apply =
      (fun ~params ~state ~dtype ~training ?ctx indices ->
        ignore (training, ctx);
        let fields = Ptree.Dict.fields_exn ~ctx:"Layer.embedding" params in
        let embedding =
          Ptree.Dict.get_tensor_exn fields ~name:"embedding" dtype
        in
        let indices = require_int32_indices ~ctx:"Layer.embedding" indices in
        (Fn.embedding ~scale ~embedding indices, state));
  }

(* Regularization *)

let dropout ~rate () =
  if rate < 0.0 || rate >= 1.0 then
    invalid_argf "Layer.dropout: expected 0.0 <= rate < 1.0, got %g" rate;
  {
    init = (fun ~dtype -> { params = Ptree.empty; state = Ptree.empty; dtype });
    apply =
      (fun ~params ~state ~dtype ~training ?ctx x ->
        ignore (params, ctx);
        let x = require_same_float_dtype ~ctx:"Layer.dropout" dtype x in
        if (not training) || rate = 0.0 then (x, state)
        else (Fn.dropout ~rate x, state));
  }

(* Activation layers *)

let relu () =
  {
    init = (fun ~dtype -> { params = Ptree.empty; state = Ptree.empty; dtype });
    apply =
      (fun ~params ~state ~dtype ~training ?ctx x ->
        ignore (params, training, ctx);
        let x = require_same_float_dtype ~ctx:"Layer.relu" dtype x in
        (Rune.relu x, state));
  }

let gelu () =
  {
    init = (fun ~dtype -> { params = Ptree.empty; state = Ptree.empty; dtype });
    apply =
      (fun ~params ~state ~dtype ~training ?ctx x ->
        ignore (params, training, ctx);
        let x = require_same_float_dtype ~ctx:"Layer.gelu" dtype x in
        (Activation.gelu x, state));
  }

let silu () =
  {
    init = (fun ~dtype -> { params = Ptree.empty; state = Ptree.empty; dtype });
    apply =
      (fun ~params ~state ~dtype ~training ?ctx x ->
        ignore (params, training, ctx);
        let x = require_same_float_dtype ~ctx:"Layer.silu" dtype x in
        (Activation.silu x, state));
  }

let tanh () =
  {
    init = (fun ~dtype -> { params = Ptree.empty; state = Ptree.empty; dtype });
    apply =
      (fun ~params ~state ~dtype ~training ?ctx x ->
        ignore (params, training, ctx);
        let x = require_same_float_dtype ~ctx:"Layer.tanh" dtype x in
        (Rune.tanh x, state));
  }

let sigmoid () =
  {
    init = (fun ~dtype -> { params = Ptree.empty; state = Ptree.empty; dtype });
    apply =
      (fun ~params ~state ~dtype ~training ?ctx x ->
        ignore (params, training, ctx);
        let x = require_same_float_dtype ~ctx:"Layer.sigmoid" dtype x in
        (Rune.sigmoid x, state));
  }

(* Pooling *)

let max_pool2d ~kernel_size ?stride () =
  let stride = match stride with Some value -> value | None -> kernel_size in
  {
    init = (fun ~dtype -> { params = Ptree.empty; state = Ptree.empty; dtype });
    apply =
      (fun ~params ~state ~dtype ~training ?ctx x ->
        ignore (params, training, ctx);
        let x = require_same_float_dtype ~ctx:"Layer.max_pool2d" dtype x in
        (Fn.max_pool2d ~kernel_size ~stride x, state));
  }

let avg_pool2d ~kernel_size ?stride () =
  let stride = match stride with Some value -> value | None -> kernel_size in
  {
    init = (fun ~dtype -> { params = Ptree.empty; state = Ptree.empty; dtype });
    apply =
      (fun ~params ~state ~dtype ~training ?ctx x ->
        ignore (params, training, ctx);
        let x = require_same_float_dtype ~ctx:"Layer.avg_pool2d" dtype x in
        (Fn.avg_pool2d ~kernel_size ~stride x, state));
  }

(* Reshape *)

let flatten () =
  {
    init = (fun ~dtype -> { params = Ptree.empty; state = Ptree.empty; dtype });
    apply =
      (fun ~params ~state ~dtype ~training ?ctx x ->
        ignore (params, training, ctx);
        let x = require_same_float_dtype ~ctx:"Layer.flatten" dtype x in
        let shape = Rune.shape x in
        if Array.length shape = 0 then
          invalid_arg "Layer.flatten: expected rank >= 1";
        let batch_size = shape.(0) in
        let flat_size =
          Array.fold_left ( * ) 1 (Array.sub shape 1 (Array.length shape - 1))
        in
        let x = if Rune.is_c_contiguous x then x else Rune.contiguous x in
        (Rune.reshape [| batch_size; flat_size |] x, state));
  }

(* Composition *)

let sequential layers =
  {
    init =
      (fun ~dtype ->
        let n = List.length layers in
        let keys = Array.init n (fun _ -> Rune.Rng.next_key ()) in
        let acc_params = ref [] in
        let acc_state = ref [] in
        List.iteri
          (fun i module_ ->
            let vars =
              Rune.Rng.with_key keys.(i) (fun () -> module_.init ~dtype)
            in
            acc_params := vars.params :: !acc_params;
            acc_state := vars.state :: !acc_state)
          layers;
        {
          params = Ptree.list (List.rev !acc_params);
          state = Ptree.list (List.rev !acc_state);
          dtype;
        });
    apply =
      (fun ~params ~state ~dtype ~training ?ctx input ->
        let params_items =
          Ptree.List.items_exn ~ctx:"Layer.sequential.params" params
        in
        let state_items =
          Ptree.List.items_exn ~ctx:"Layer.sequential.state" state
        in
        match (layers, params_items, state_items) with
        | [], [], [] ->
            let input =
              require_same_float_dtype ~ctx:"Layer.sequential" dtype input
            in
            (input, Ptree.list [])
        | first :: rest_layers, p :: ps, s :: ss ->
            let y, first_state =
              first.apply ~params:p ~state:s ~dtype ~training ?ctx input
            in
            let rec go modules param_values state_values x =
              match (modules, param_values, state_values) with
              | [], [], [] -> (x, [])
              | module_ :: modules_tail, p :: ps_tail, s :: ss_tail ->
                  let y, state' =
                    module_.apply ~params:p ~state:s ~dtype ~training ?ctx x
                  in
                  let y_final, state_tail = go modules_tail ps_tail ss_tail y in
                  (y_final, state' :: state_tail)
              | _ ->
                  invalid_arg
                    "Layer.sequential: params/state/layers length mismatch"
            in
            let y_final, rest_states = go rest_layers ps ss y in
            (y_final, Ptree.list (first_state :: rest_states))
        | _ ->
            invalid_arg "Layer.sequential: params/state/layers length mismatch");
  }
