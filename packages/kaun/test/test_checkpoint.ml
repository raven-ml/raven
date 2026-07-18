(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Checkpoint = Kaun.Checkpoint

let f32 = Nx.float32
let f64 = Nx.float64
let vec32 xs = Nx.create f32 [| Array.length xs |] xs
let vec64 xs = Nx.create f64 [| Array.length xs |] xs
let to_arr t = Nx.to_array (Nx.reshape [| -1 |] (Nx.contiguous t))

(* Checkpoints round-trip bit-exactly, so comparisons are exact. *)
let check_arr ~msg expected actual =
  equal ~msg (array (float 0.0)) expected (to_arr actual)

(* Runs [f] with a fresh checkpoint file path in a temporary directory, removed
   afterwards even on failure. *)
let with_ckpt_file f =
  let dir = Filename.temp_dir "kaun_checkpoint" "" in
  Fun.protect
    ~finally:(fun () ->
      Array.iter
        (fun entry -> Sys.remove (Filename.concat dir entry))
        (Sys.readdir dir);
      Sys.rmdir dir)
    (fun () -> f (Filename.concat dir "ckpt.safetensors"))

(* A typed parameter record with mixed dtypes. *)

type params = { w : Nx.float32_t; b : Nx.float32_t; scale : Nx.float64_t }

module Params = struct
  type t = params

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { w; b; scale } =
    { w = f w; b = f b; scale = f scale }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { w = f p.w q.w; b = f p.b q.b; scale = f p.scale q.scale }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { w; b; scale } =
    f w;
    f b;
    f scale

  let names _ = [ "w"; "b"; "scale" ]
end

let params () =
  {
    w = vec32 [| 1.5; -2.0; 3.25 |];
    b = vec32 [| 0.5 |];
    scale = vec64 [| 2.0 |];
  }

let fresh_params () = Params.map (fun leaf -> Nx.zeros_like leaf) (params ())

(* A float32 linear model, for the training stories. *)

type lin = { lw : Nx.float32_t; lb : Nx.float32_t }

module Lin = struct
  type t = lin

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { lw; lb } =
    { lw = f lw; lb = f lb }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { lw = f p.lw q.lw; lb = f p.lb q.lb }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { lw; lb } =
    f lw;
    f lb

  let names _ = [ "w"; "b" ]
end

(* Round-trip *)

let test_round_trip () =
  with_ckpt_file @@ fun path ->
  Checkpoint.save path (Checkpoint.of_params (module Params) (params ()));
  let ckpt = Checkpoint.load path in
  let p = Checkpoint.to_params (module Params) ~like:(fresh_params ()) ckpt in
  check_arr ~msg:"w" [| 1.5; -2.0; 3.25 |] p.w;
  check_arr ~msg:"b" [| 0.5 |] p.b;
  check_arr ~msg:"scale" [| 2.0 |] p.scale

let test_round_trip_dtypes () =
  with_ckpt_file @@ fun path ->
  Checkpoint.save path (Checkpoint.of_params (module Params) (params ()));
  let ckpt = Checkpoint.load path in
  let dtype_of name =
    match Checkpoint.get name ckpt with
    | Rune.Ptree.P x -> Nx_core.Dtype.to_string (Nx.dtype x)
  in
  equal ~msg:"w" string "float32" (dtype_of "w");
  equal ~msg:"scale" string "float64" (dtype_of "scale");
  (* Strict (no-cast) extraction succeeds, so dtypes survived the file. *)
  let _ = Checkpoint.to_params (module Params) ~like:(fresh_params ()) ckpt in
  ()

let test_int_round_trip () =
  with_ckpt_file @@ fun path ->
  Checkpoint.save path (Checkpoint.of_int "step" 12345);
  equal int 12345 (Checkpoint.to_int "step" (Checkpoint.load path))

(* Resume training *)

let xs = Nx.create f32 [| 4; 2 |] [| 0.0; 1.0; 1.0; 0.0; 1.0; 1.0; 0.5; -0.5 |]
let ys = Nx.create f32 [| 4; 1 |] [| 1.0; -1.0; 0.5; 2.0 |]

let loss (p : lin) =
  let d = Nx.sub (Nx.add (Nx.matmul xs p.lw) p.lb) ys in
  Nx.mean (Nx.mul d d)

let adam_train_step (p, st) =
  let grads = Rune.grad (module Lin) loss p in
  Vega.adam_step (module Lin) ~lr:0.05 st ~params:p ~grads

let rec train_adam_steps n s =
  if n = 0 then s else train_adam_steps (n - 1) (adam_train_step s)

let test_resume_training () =
  with_ckpt_file @@ fun path ->
  let p0 =
    { lw = Nx.create f32 [| 2; 1 |] [| 0.2; -0.1 |]; lb = vec32 [| 0.0 |] }
  in
  let p3, st3 = train_adam_steps 3 (p0, Vega.adam_init (module Lin) p0) in
  Checkpoint.save path
    (Checkpoint.concat
       [
         Checkpoint.of_params (module Lin) ~prefix:"model" p3;
         Checkpoint.of_params (module Lin) ~prefix:"optim.mu" st3.Vega.mu;
         Checkpoint.of_params (module Lin) ~prefix:"optim.nu" st3.Vega.nu;
         Checkpoint.of_int "optim.step" st3.Vega.step;
       ]);
  let expected, _ = train_adam_steps 2 (p3, st3) in
  (* Restore into freshly initialized values and continue training. *)
  let ckpt = Checkpoint.load path in
  let like = Vega.adam_init (module Lin) p0 in
  let p3' = Checkpoint.to_params (module Lin) ~prefix:"model" ~like:p0 ckpt in
  let st3' =
    {
      Vega.mu =
        Checkpoint.to_params
          (module Lin)
          ~prefix:"optim.mu" ~like:like.Vega.mu ckpt;
      nu =
        Checkpoint.to_params
          (module Lin)
          ~prefix:"optim.nu" ~like:like.Vega.nu ckpt;
      step = Checkpoint.to_int "optim.step" ckpt;
    }
  in
  let resumed, _ = train_adam_steps 2 (p3', st3') in
  check_arr ~msg:"w" (to_arr expected.lw) resumed.lw;
  check_arr ~msg:"b" (to_arr expected.lb) resumed.lb;
  (* Control: dropping the optimizer state changes the trajectory, so the
     assertions above genuinely depend on restoring it. *)
  let fresh, _ = train_adam_steps 2 (p3', Vega.adam_init (module Lin) p3') in
  is_false ~msg:"fresh optimizer state diverges"
    (to_arr fresh.lw = to_arr expected.lw)

let sgd_train_step (p, st) =
  let grads = Rune.grad (module Lin) loss p in
  Vega.sgd_step (module Lin) ~lr:0.05 ~momentum:0.9 st ~params:p ~grads

let rec train_sgd_steps n s =
  if n = 0 then s else train_sgd_steps (n - 1) (sgd_train_step s)

let test_resume_sgd_momentum () =
  with_ckpt_file @@ fun path ->
  let p0 =
    { lw = Nx.create f32 [| 2; 1 |] [| 0.2; -0.1 |]; lb = vec32 [| 0.0 |] }
  in
  let p3, st3 = train_sgd_steps 3 (p0, Vega.sgd_init (module Lin) p0) in
  Checkpoint.save path
    (Checkpoint.concat
       [
         Checkpoint.of_params (module Lin) ~prefix:"model" p3;
         Checkpoint.of_params
           (module Lin)
           ~prefix:"optim.velocity" st3.Vega.velocity;
       ]);
  let expected, _ = train_sgd_steps 2 (p3, st3) in
  let ckpt = Checkpoint.load path in
  let p3' = Checkpoint.to_params (module Lin) ~prefix:"model" ~like:p0 ckpt in
  let like = Vega.sgd_init (module Lin) p0 in
  let st3' =
    {
      Vega.velocity =
        Checkpoint.to_params
          (module Lin)
          ~prefix:"optim.velocity" ~like:like.Vega.velocity ckpt;
    }
  in
  let resumed, _ = train_sgd_steps 2 (p3', st3') in
  check_arr ~msg:"w" (to_arr expected.lw) resumed.lw;
  check_arr ~msg:"b" (to_arr expected.lb) resumed.lb;
  let fresh, _ = train_sgd_steps 2 (p3', Vega.sgd_init (module Lin) p3') in
  is_false ~msg:"fresh momentum diverges" (to_arr fresh.lw = to_arr expected.lw)

let test_load_pretrained () =
  with_ckpt_file @@ fun path ->
  (* A file with bare named tensors, as produced by another tool. *)
  Checkpoint.save path
    (Checkpoint.concat
       [
         Checkpoint.of_tensor "w" (Nx.create f32 [| 2; 1 |] [| 0.25; -0.75 |]);
         Checkpoint.of_tensor "b" (vec32 [| 0.125 |]);
       ]);
  let fresh = { lw = Nx.zeros f32 [| 2; 1 |]; lb = Nx.zeros f32 [| 1 |] } in
  let m =
    Checkpoint.to_params (module Lin) ~like:fresh (Checkpoint.load path)
  in
  check_arr ~msg:"w" [| 0.25; -0.75 |] m.lw;
  check_arr ~msg:"b" [| 0.125 |] m.lb

(* Naming *)

let test_prefix_names () =
  let ckpt = Checkpoint.of_params (module Params) ~prefix:"model" (params ()) in
  equal (list string)
    [ "model.b"; "model.scale"; "model.w" ]
    (Checkpoint.names ckpt)

let test_ptree_paths () =
  let module T = Rune.Ptree in
  let tree =
    T.dict
      [
        ( "layers",
          T.list
            [
              T.dict [ ("w", T.tensor (vec32 [| 1.0 |])) ];
              T.dict [ ("w", T.tensor (vec32 [| 2.0 |])) ];
            ] );
        ("head", T.tensor (vec64 [| 3.0 |]));
      ]
  in
  let ckpt = Checkpoint.of_params (module Checkpoint.Ptree) tree in
  equal ~msg:"paths" (list string)
    [ "head"; "layers.0.w"; "layers.1.w" ]
    (Checkpoint.names ckpt);
  with_ckpt_file @@ fun path ->
  Checkpoint.save path ckpt;
  let like = T.map (fun leaf -> Nx.zeros_like leaf) tree in
  let tree' =
    Checkpoint.to_params (module Checkpoint.Ptree) ~like (Checkpoint.load path)
  in
  let leaf name =
    match
      Checkpoint.get name (Checkpoint.of_params (module Checkpoint.Ptree) tree')
    with
    | T.P x -> to_arr (Nx.cast f64 x)
  in
  equal ~msg:"layers.1.w" (array (float 0.0)) [| 2.0 |] (leaf "layers.1.w");
  equal ~msg:"head" (array (float 0.0)) [| 3.0 |] (leaf "head")

let test_find_get () =
  let ckpt = Checkpoint.of_tensor "w" (vec32 [| 1.0 |]) in
  is_some ~msg:"find present" (Checkpoint.find "w" ckpt);
  is_none ~msg:"find absent" (Checkpoint.find "nope" ckpt);
  raises_invalid_arg "Checkpoint.get: no entry named \"nope\"" (fun () ->
      Checkpoint.get "nope" ckpt)

(* Error contracts *)

let test_missing_entry () =
  let ckpt =
    Checkpoint.concat
      [
        Checkpoint.of_tensor "w" (vec32 [| 1.0; 2.0; 3.0 |]);
        Checkpoint.of_tensor "scale" (vec64 [| 1.0 |]);
      ]
  in
  raises_invalid_arg "Checkpoint.to_params: missing entry \"b\"" (fun () ->
      Checkpoint.to_params (module Params) ~like:(fresh_params ()) ckpt)

let test_extra_entries_ignored () =
  let ckpt =
    Checkpoint.concat
      [
        Checkpoint.of_params (module Params) (params ());
        Checkpoint.of_tensor "unrelated" (vec32 [| 9.0 |]);
      ]
  in
  let p =
    no_raise (fun () ->
        Checkpoint.to_params (module Params) ~like:(fresh_params ()) ckpt)
  in
  check_arr ~msg:"w" [| 1.5; -2.0; 3.25 |] p.w

let test_shape_mismatch () =
  let ckpt =
    Checkpoint.concat
      [
        Checkpoint.of_tensor "w" (vec32 [| 1.0; 2.0; 3.0 |]);
        Checkpoint.of_tensor "b" (vec32 [| 1.0; 2.0 |]);
        Checkpoint.of_tensor "scale" (vec64 [| 1.0 |]);
      ]
  in
  raises_invalid_arg
    "Checkpoint.to_params: shape mismatch for \"b\": expected [1], got [2]"
    (fun () ->
      Checkpoint.to_params (module Params) ~like:(fresh_params ()) ckpt)

let test_dtype_mismatch_and_cast () =
  let ckpt =
    Checkpoint.concat
      [
        Checkpoint.of_tensor "w" (vec32 [| 1.0; 2.0; 3.0 |]);
        Checkpoint.of_tensor "b" (vec32 [| 1.0 |]);
        Checkpoint.of_tensor "scale" (vec32 [| 4.0 |]);
      ]
  in
  raises_invalid_arg
    "Checkpoint.to_params: dtype mismatch for \"scale\": expected float64, got \
     float32 (pass ~cast:true to convert)" (fun () ->
      Checkpoint.to_params (module Params) ~like:(fresh_params ()) ckpt);
  let p =
    Checkpoint.to_params (module Params) ~cast:true ~like:(fresh_params ()) ckpt
  in
  check_arr ~msg:"scale cast to float64" [| 4.0 |] p.scale

let test_concat_duplicate () =
  raises_invalid_arg "Checkpoint.concat: duplicate name \"w\"" (fun () ->
      Checkpoint.concat
        [
          Checkpoint.of_tensor "w" (vec32 [| 1.0 |]);
          Checkpoint.of_tensor "w" (vec32 [| 2.0 |]);
        ])

module Misnamed = struct
  include Params

  let names _ = [ "w" ]
end

module Duplicated = struct
  include Params

  let names _ = [ "w"; "w"; "scale" ]
end

let test_invalid_names () =
  raises_invalid_arg "Checkpoint.of_params: 1 name(s) for 3 tensor leaves"
    (fun () -> Checkpoint.of_params (module Misnamed) (params ()));
  raises_invalid_arg "Checkpoint.to_params: 1 name(s) for 3 tensor leaves"
    (fun () ->
      Checkpoint.to_params (module Misnamed) ~like:(params ()) Checkpoint.empty);
  raises_invalid_arg "Checkpoint.of_params: duplicate name \"w\"" (fun () ->
      Checkpoint.of_params (module Duplicated) (params ()))

let test_empty_name () =
  raises_invalid_arg "Checkpoint.of_tensor: empty tensor name" (fun () ->
      Checkpoint.of_tensor "" (vec32 [| 1.0 |]))

let test_to_int_errors () =
  let ckpt =
    Checkpoint.concat
      [
        Checkpoint.of_tensor "w" (vec32 [| 1.0 |]);
        Checkpoint.of_tensor "v" (Nx.create Nx.int32 [| 2 |] [| 1l; 2l |]);
      ]
  in
  raises_invalid_arg "Checkpoint.to_int: no entry named \"step\"" (fun () ->
      Checkpoint.to_int "step" ckpt);
  raises_invalid_arg
    "Checkpoint.to_int: \"w\" is not an int32 entry (dtype float32)" (fun () ->
      Checkpoint.to_int "w" ckpt);
  raises_invalid_arg "Checkpoint.to_int: \"v\" is not a scalar (shape [2])"
    (fun () -> Checkpoint.to_int "v" ckpt)

let () =
  run "kaun checkpoint"
    [
      group "round-trip"
        [
          test "save and load preserve values" test_round_trip;
          test "save and load preserve dtypes" test_round_trip_dtypes;
          test "of_int and to_int round-trip through a file" test_int_round_trip;
        ];
      group "training"
        [
          test "resumed Adam training continues identically"
            test_resume_training;
          test "resumed SGD momentum continues identically"
            test_resume_sgd_momentum;
          test "pretrained weights load by name into a fresh model"
            test_load_pretrained;
        ];
      group "naming"
        [
          test "prefix prepends dotted names" test_prefix_names;
          test "ptree leaves are named by their path" test_ptree_paths;
          test "find and get look entries up by name" test_find_get;
        ];
      group "errors"
        [
          test "missing entry raises with its name" test_missing_entry;
          test "unrelated entries are ignored" test_extra_entries_ignored;
          test "shape mismatch raises" test_shape_mismatch;
          test "dtype mismatch raises unless cast" test_dtype_mismatch_and_cast;
          test "concat rejects duplicate names" test_concat_duplicate;
          test "invalid names lists are rejected" test_invalid_names;
          test "empty tensor names are rejected" test_empty_name;
          test "to_int rejects missing and non-scalar entries"
            test_to_int_errors;
        ];
    ]
