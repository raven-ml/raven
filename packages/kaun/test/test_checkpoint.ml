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
  equal ~msg (array float_exact) expected (to_arr actual)

(* Runs [f] with a fresh checkpoint file path in a temporary directory, removed
   afterwards even on failure. A loaded checkpoint stays mapped until its
   tensors are collected, and Windows may refuse to delete a mapped file. *)
let with_ckpt_file f =
  let dir = Filename.temp_dir "kaun_checkpoint" "" in
  Fun.protect
    ~finally:(fun () ->
      Gc.full_major ();
      try
        Array.iter
          (fun entry -> Sys.remove (Filename.concat dir entry))
          (Sys.readdir dir);
        Sys.rmdir dir
      with Sys_error _ when Sys.win32 -> ())
    (fun () -> f (Filename.concat dir "ckpt.safetensors"))

(* A parameter record with mixed leaf dtypes. *)

module Params = struct
  type t = { w : Nx.float32_t; b : Nx.float32_t; scale : Nx.float64_t }

  module Walked = struct
    type nonrec _ t = t

    let walk c { w; b; scale } =
      let open Nx.Ptree.Walk in
      let w = field c "w" tensor w in
      let b = field c "b" tensor b in
      let scale = field c "scale" tensor scale in
      { w; b; scale }
  end

  let ptree : t Nx.Ptree.t = Nx.Ptree.instantiate (module Walked)
end

let params () =
  {
    Params.w = vec32 [| 1.5; -2.0; 3.25 |];
    b = vec32 [| 0.5 |];
    scale = vec64 [| 2.0 |];
  }

let fresh_params () =
  Nx.Ptree.map Params.ptree (fun _ leaf -> Nx.zeros_like leaf) (params ())

(* A float32 linear model, for the training stories. Its leaf paths keep the
   file names [w] and [b] whatever the record's field names. *)

module Lin = struct
  type 'a t = { lw : 'a; lb : 'a }

  let walk c { lw; lb } =
    let open Nx.Ptree.Walk in
    let lw = field c "w" leaf lw in
    let lb = field c "b" leaf lb in
    { lw; lb }
end

let lin : Nx.float32_t Lin.t Nx.Ptree.t = Nx.Ptree.instantiate (module Lin)

(* Round-trip *)

let test_round_trip () =
  with_ckpt_file @@ fun path ->
  Checkpoint.save path (Checkpoint.of_value Params.ptree (params ()));
  let ckpt = Checkpoint.load path in
  let p = Checkpoint.to_value Params.ptree ~like:(fresh_params ()) ckpt in
  check_arr ~msg:"w" [| 1.5; -2.0; 3.25 |] p.Params.w;
  check_arr ~msg:"b" [| 0.5 |] p.Params.b;
  check_arr ~msg:"scale" [| 2.0 |] p.Params.scale

let test_round_trip_dtypes () =
  with_ckpt_file @@ fun path ->
  Checkpoint.save path (Checkpoint.of_value Params.ptree (params ()));
  let ckpt = Checkpoint.load path in
  let dtype_of name =
    match Checkpoint.get name ckpt with
    | Nx.P x -> Nx_dtype.to_string (Nx.dtype x)
  in
  equal ~msg:"w" string "float32" (dtype_of "w");
  equal ~msg:"scale" string "float64" (dtype_of "scale");
  (* Strict (no-cast) extraction succeeds, so dtypes survived the file. *)
  let _ = Checkpoint.to_value Params.ptree ~like:(fresh_params ()) ckpt in
  ()

let test_int_round_trip () =
  with_ckpt_file @@ fun path ->
  Checkpoint.save path (Checkpoint.of_int "step" 12345);
  equal int 12345 (Checkpoint.to_int "step" (Checkpoint.load path))

(* A key saved with its training state resumes the same stream: the loaded key
   is the saved one, word for word. *)
let test_key_round_trip () =
  with_ckpt_file @@ fun path ->
  let key = Nx.Rng.fold_in (Nx.Rng.key 5) 3 in
  Checkpoint.save path (Checkpoint.of_value ~prefix:"rng" Nx.Rng.ptree key);
  let loaded =
    Checkpoint.to_value ~prefix:"rng" Nx.Rng.ptree ~like:(Nx.Rng.key 0)
      (Checkpoint.load path)
  in
  let words (k : Nx.Rng.t) = Nx.to_array (k :> Nx.int32_t) in
  equal ~msg:"key words" (array int32) (words key) (words loaded)

(* Resume training *)

let xs = Nx.create f32 [| 4; 2 |] [| 0.0; 1.0; 1.0; 0.0; 1.0; 1.0; 0.5; -0.5 |]
let ys = Nx.create f32 [| 4; 1 |] [| 1.0; -1.0; 0.5; 2.0 |]

let loss (p : Nx.float32_t Lin.t) =
  let d = Nx.sub (Nx.add (Nx.matmul xs p.Lin.lw) p.Lin.lb) ys in
  Nx.mean (Nx.mul d d)

let adam_train_step (p, st) =
  let grads = Rune.grad lin loss p in
  Vega.adam_step lin ~lr:(Vega.lr 0.05) st ~params:p ~grads

let rec train_adam_steps n s =
  if n = 0 then s else train_adam_steps (n - 1) (adam_train_step s)

let test_resume_training () =
  with_ckpt_file @@ fun path ->
  let p0 =
    { Lin.lw = Nx.create f32 [| 2; 1 |] [| 0.2; -0.1 |]; lb = vec32 [| 0.0 |] }
  in
  let p3, st3 = train_adam_steps 3 (p0, Vega.adam_init lin p0) in
  let opt = Vega.adam_ptree lin in
  let ckpt =
    Checkpoint.concat
      [
        Checkpoint.of_value ~prefix:"model" lin p3;
        Checkpoint.of_value ~prefix:"optim" opt st3;
      ]
  in
  equal ~msg:"today's names" (list string)
    [
      "model.b";
      "model.w";
      "optim.mu.b";
      "optim.mu.w";
      "optim.nu.b";
      "optim.nu.w";
      "optim.step";
    ]
    (Checkpoint.names ckpt);
  Checkpoint.save path ckpt;
  let expected, _ = train_adam_steps 2 (p3, st3) in
  (* Restore into freshly initialized values and continue training. *)
  let ckpt = Checkpoint.load path in
  let p3' = Checkpoint.to_value ~prefix:"model" lin ~like:p0 ckpt in
  let st3' =
    Checkpoint.to_value ~prefix:"optim" opt ~like:(Vega.adam_init lin p0) ckpt
  in
  let resumed, _ = train_adam_steps 2 (p3', st3') in
  check_arr ~msg:"w" (to_arr expected.Lin.lw) resumed.Lin.lw;
  check_arr ~msg:"b" (to_arr expected.Lin.lb) resumed.Lin.lb;
  (* Control: dropping the optimizer state changes the trajectory, so the
     assertions above genuinely depend on restoring it. *)
  let fresh, _ = train_adam_steps 2 (p3', Vega.adam_init lin p3') in
  is_false ~msg:"fresh optimizer state diverges"
    (to_arr fresh.Lin.lw = to_arr expected.Lin.lw)

let sgd_train_step (p, st) =
  let grads = Rune.grad lin loss p in
  Vega.sgd_step lin ~lr:(Vega.lr 0.05) ~momentum:0.9 st ~params:p ~grads

let rec train_sgd_steps n s =
  if n = 0 then s else train_sgd_steps (n - 1) (sgd_train_step s)

let test_resume_sgd_momentum () =
  with_ckpt_file @@ fun path ->
  let p0 =
    { Lin.lw = Nx.create f32 [| 2; 1 |] [| 0.2; -0.1 |]; lb = vec32 [| 0.0 |] }
  in
  let p3, st3 = train_sgd_steps 3 (p0, Vega.sgd_init lin p0) in
  let opt = Vega.sgd_ptree lin in
  Checkpoint.save path
    (Checkpoint.concat
       [
         Checkpoint.of_value ~prefix:"model" lin p3;
         Checkpoint.of_value ~prefix:"optim" opt st3;
       ]);
  let expected, _ = train_sgd_steps 2 (p3, st3) in
  let ckpt = Checkpoint.load path in
  let p3' = Checkpoint.to_value ~prefix:"model" lin ~like:p0 ckpt in
  let st3' =
    Checkpoint.to_value ~prefix:"optim" opt ~like:(Vega.sgd_init lin p0) ckpt
  in
  let resumed, _ = train_sgd_steps 2 (p3', st3') in
  check_arr ~msg:"w" (to_arr expected.Lin.lw) resumed.Lin.lw;
  check_arr ~msg:"b" (to_arr expected.Lin.lb) resumed.Lin.lb;
  let fresh, _ = train_sgd_steps 2 (p3', Vega.sgd_init lin p3') in
  is_false ~msg:"fresh momentum diverges"
    (to_arr fresh.Lin.lw = to_arr expected.Lin.lw)

let test_load_pretrained () =
  with_ckpt_file @@ fun path ->
  (* A file with bare named tensors, as produced by another tool. *)
  Checkpoint.save path
    (Checkpoint.concat
       [
         Checkpoint.of_tensor "w" (Nx.create f32 [| 2; 1 |] [| 0.25; -0.75 |]);
         Checkpoint.of_tensor "b" (vec32 [| 0.125 |]);
       ]);
  let fresh = { Lin.lw = Nx.zeros f32 [| 2; 1 |]; lb = Nx.zeros f32 [| 1 |] } in
  let m = Checkpoint.to_value lin ~like:fresh (Checkpoint.load path) in
  check_arr ~msg:"w" [| 0.25; -0.75 |] m.Lin.lw;
  check_arr ~msg:"b" [| 0.125 |] m.Lin.lb

(* Naming *)

let test_prefix_names () =
  let ckpt = Checkpoint.of_value ~prefix:"model" Params.ptree (params ()) in
  equal (list string)
    [ "model.b"; "model.scale"; "model.w" ]
    (Checkpoint.names ckpt)

(* A model with a list of layers and an optional head. *)
module Stack = struct
  type 'a t = { layers : 'a Kaun.Linear.t list; head : 'a option }

  let walk c { layers; head } =
    let open Nx.Ptree.Walk in
    let layers = field c "layers" (list Kaun.Linear.walk) layers in
    let head = field c "head" (option leaf) head in
    { layers; head }
end

let test_nested_paths () =
  let stack = Nx.Ptree.instantiate (module Stack) in
  let layer w = { Kaun.Linear.w = vec32 [| w |]; b = None } in
  let x =
    { Stack.layers = [ layer 1.0; layer 2.0 ]; head = Some (vec32 [| 3.0 |]) }
  in
  let ckpt = Checkpoint.of_value stack x in
  equal ~msg:"paths" (list string)
    [ "head"; "layers.0.w"; "layers.1.w" ]
    (Checkpoint.names ckpt);
  with_ckpt_file @@ fun path ->
  Checkpoint.save path ckpt;
  let like = Nx.Ptree.map stack (fun _ leaf -> Nx.zeros_like leaf) x in
  let x' = Checkpoint.to_value stack ~like (Checkpoint.load path) in
  check_arr ~msg:"layers.1.w" [| 2.0 |] (List.nth x'.Stack.layers 1).w;
  check_arr ~msg:"head" [| 3.0 |] (Option.get x'.Stack.head)

let test_root_leaf_prefix () =
  let x = vec32 [| 1.0 |] in
  let ckpt = Checkpoint.of_value ~prefix:"w" Nx.Ptree.tensor x in
  equal ~msg:"prefix names the root" (list string) [ "w" ]
    (Checkpoint.names ckpt);
  raises
    (Invalid_argument "Checkpoint.of_value: a leaf at the root needs ~prefix")
    (fun () -> Checkpoint.of_value Nx.Ptree.tensor x);
  raises
    (Invalid_argument "Checkpoint.to_value: a leaf at the root needs ~prefix")
    (fun () -> Checkpoint.to_value Nx.Ptree.tensor ~like:x ckpt)

(* A fixed tensor has an entry of its own. *)
module Counted = struct
  type 'a t = { w : 'a; count : Nx.int32_t }

  let walk c { w; count } =
    let open Nx.Ptree.Walk in
    let w = field c "w" leaf w in
    let count = field c "count" tensor count in
    { w; count }
end

let test_fixed_tensor_entry () =
  let counted = Nx.Ptree.instantiate (module Counted) in
  let x = { Counted.w = vec32 [| 1.0 |]; count = Nx.scalar Nx.int32 7l } in
  let ckpt = Checkpoint.of_value counted x in
  equal ~msg:"names" (list string) [ "count"; "w" ] (Checkpoint.names ckpt);
  let like = { Counted.w = vec32 [| 0.0 |]; count = Nx.scalar Nx.int32 0l } in
  equal ~msg:"the fixed tensor loads" int32 7l
    (Nx.item [] (Checkpoint.to_value counted ~like ckpt).count)

let test_find_get () =
  let ckpt = Checkpoint.of_tensor "w" (vec32 [| 1.0 |]) in
  is_some ~msg:"find present" (Checkpoint.find "w" ckpt);
  is_none ~msg:"find absent" (Checkpoint.find "nope" ckpt);
  raises (Invalid_argument "Checkpoint.get: no entry named \"nope\"") (fun () ->
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
  raises
    (Invalid_argument
       "Checkpoint.to_value: b: no entry in the checkpoint, a leaf in the \
        template") (fun () ->
      Checkpoint.to_value Params.ptree ~like:(fresh_params ()) ckpt)

let test_extra_entries_ignored () =
  let ckpt =
    Checkpoint.concat
      [
        Checkpoint.of_value Params.ptree (params ());
        Checkpoint.of_tensor "unrelated" (vec32 [| 9.0 |]);
      ]
  in
  let p = Checkpoint.to_value Params.ptree ~like:(fresh_params ()) ckpt in
  check_arr ~msg:"w" [| 1.5; -2.0; 3.25 |] p.Params.w

let test_shape_mismatch () =
  let ckpt =
    Checkpoint.concat
      [
        Checkpoint.of_tensor "w" (vec32 [| 1.0; 2.0; 3.0 |]);
        Checkpoint.of_tensor "b" (vec32 [| 1.0; 2.0 |]);
        Checkpoint.of_tensor "scale" (vec64 [| 1.0 |]);
      ]
  in
  raises
    (Invalid_argument
       "Checkpoint.to_value: b: shape [2] in the checkpoint, [1] in the \
        template") (fun () ->
      Checkpoint.to_value Params.ptree ~like:(fresh_params ()) ckpt)

let test_dtype_mismatch () =
  let ckpt =
    Checkpoint.concat
      [
        Checkpoint.of_tensor "w" (vec32 [| 1.0; 2.0; 3.0 |]);
        Checkpoint.of_tensor "b" (vec32 [| 1.0 |]);
        Checkpoint.of_tensor "scale" (vec32 [| 4.0 |]);
      ]
  in
  raises
    (Invalid_argument
       "Checkpoint.to_value: scale: float32 in the checkpoint, float64 in the \
        template") (fun () ->
      Checkpoint.to_value Params.ptree ~like:(fresh_params ()) ckpt)

(* Extraction by name *)

let accessor_ckpt () =
  Checkpoint.concat
    [
      Checkpoint.of_tensor "w"
        (Nx.create f32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]);
      Checkpoint.of_tensor "half" (Nx.cast Nx.bfloat16 (vec32 [| 0.5; -2.0 |]));
      Checkpoint.of_tensor "blocks" (Nx.create Nx.uint8 [| 3 |] [| 1; 2; 255 |]);
      Checkpoint.of_tensor "tiny" (Nx.cast Nx.float8_e4m3 (vec32 [| 1.0 |]));
    ]

let test_to_tensor () =
  let ckpt = accessor_ckpt () in
  let blocks = Checkpoint.to_tensor ~shape:[| 3 |] Nx.uint8 "blocks" ckpt in
  equal ~msg:"uint8 values" (array int) [| 1; 2; 255 |] (Nx.to_array blocks);
  let w = Checkpoint.to_tensor ~shape:[| 2; 2 |] f32 "w" ckpt in
  is_true ~msg:"the entry is returned as stored"
    (w == Nx.unpack f32 (Checkpoint.get "w" ckpt));
  check_arr ~msg:"float8 is read as stored" [| 1.0 |]
    (Nx.cast f32
       (Checkpoint.to_tensor ~shape:[| 1 |] Nx.float8_e4m3 "tiny" ckpt));
  raises (Invalid_argument "Checkpoint.to_tensor: missing entry \"nope\"")
    (fun () -> Checkpoint.to_tensor ~shape:[| 1 |] f32 "nope" ckpt);
  raises
    (Invalid_argument
       "Checkpoint.to_tensor: shape mismatch for \"w\": expected [4], got [2; \
        2]") (fun () -> Checkpoint.to_tensor ~shape:[| 4 |] f32 "w" ckpt);
  raises
    (Invalid_argument
       "Checkpoint.to_tensor: dtype mismatch for \"half\": expected float32, \
        got bfloat16") (fun () ->
      Checkpoint.to_tensor ~shape:[| 2 |] f32 "half" ckpt)

let test_to_float () =
  let ckpt = accessor_ckpt () in
  check_arr ~msg:"cast from bfloat16" [| 0.5; -2.0 |]
    (Checkpoint.to_float ~shape:[| 2 |] f32 "half" ckpt);
  let half = Checkpoint.to_float ~shape:[| 2 |] Nx.bfloat16 "half" ckpt in
  is_true ~msg:"the entry's own dtype casts nothing"
    (half == Nx.unpack Nx.bfloat16 (Checkpoint.get "half" ckpt));
  raises (Invalid_argument "Checkpoint.to_float: missing entry \"nope\"")
    (fun () -> Checkpoint.to_float ~shape:[| 1 |] f32 "nope" ckpt);
  raises
    (Invalid_argument
       "Checkpoint.to_float: shape mismatch for \"half\": expected [3], got [2]")
    (fun () -> Checkpoint.to_float ~shape:[| 3 |] f32 "half" ckpt);
  raises
    (Invalid_argument
       "Checkpoint.to_float: \"blocks\" is not a floating-point entry (dtype \
        uint8)") (fun () ->
      Checkpoint.to_float ~shape:[| 3 |] f32 "blocks" ckpt);
  raises
    (Invalid_argument
       "Checkpoint.to_float: \"tiny\" is a float8_e4m3 entry, whose scales \
        live in other entries: read it with to_tensor") (fun () ->
      Checkpoint.to_float ~shape:[| 1 |] f32 "tiny" ckpt);
  raises
    (Invalid_argument
       "Checkpoint.to_float: \"w\" cannot be cast to float8_e4m3, which needs \
        scales") (fun () ->
      Checkpoint.to_float ~shape:[| 2; 2 |] Nx.float8_e4m3 "w" ckpt)

let test_concat_duplicate () =
  raises (Invalid_argument "Checkpoint.concat: duplicate name \"w\"") (fun () ->
      Checkpoint.concat
        [
          Checkpoint.of_tensor "w" (vec32 [| 1.0 |]);
          Checkpoint.of_tensor "w" (vec32 [| 2.0 |]);
        ])

(* A structure whose paths collide. *)
module Duplicated = struct
  type 'a t = { a : 'a; b : 'a }

  let walk c { a; b } =
    let open Nx.Ptree.Walk in
    let a = field c "w" leaf a in
    let b = field c "w" leaf b in
    { a; b }
end

let test_duplicate_names () =
  let duplicated = Nx.Ptree.instantiate (module Duplicated) in
  let x = { Duplicated.a = vec32 [| 1.0 |]; b = vec32 [| 2.0 |] } in
  raises (Invalid_argument "Checkpoint.of_value: w: two leaves have this name")
    (fun () -> Checkpoint.of_value duplicated x);
  raises (Invalid_argument "Checkpoint.to_value: w: two leaves have this name")
    (fun () ->
      Checkpoint.to_value duplicated ~like:x
        (Checkpoint.of_tensor "w" (vec32 [| 3.0 |])))

let test_empty_name () =
  raises (Invalid_argument "Checkpoint.of_tensor: empty tensor name") (fun () ->
      Checkpoint.of_tensor "" (vec32 [| 1.0 |]))

let test_to_int_errors () =
  let ckpt =
    Checkpoint.concat
      [
        Checkpoint.of_tensor "w" (vec32 [| 1.0 |]);
        Checkpoint.of_tensor "v" (Nx.create Nx.int32 [| 2 |] [| 1l; 2l |]);
      ]
  in
  raises (Invalid_argument "Checkpoint.to_int: no entry named \"step\"")
    (fun () -> Checkpoint.to_int "step" ckpt);
  raises
    (Invalid_argument
       "Checkpoint.to_int: \"w\" is not an int32 entry (dtype float32)")
    (fun () -> Checkpoint.to_int "w" ckpt);
  raises
    (Invalid_argument "Checkpoint.to_int: \"v\" is not a scalar (shape [2])")
    (fun () -> Checkpoint.to_int "v" ckpt)

let () =
  exit
    (run "kaun checkpoint"
       [
         group "round-trip"
           [
             test "save and load preserve values" test_round_trip;
             test "save and load preserve dtypes" test_round_trip_dtypes;
             test "of_int and to_int round-trip through a file"
               test_int_round_trip;
             test "a key round-trips through a file" test_key_round_trip;
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
             test "leaves are named by their path" test_nested_paths;
             test "a root leaf is named by the prefix alone"
               test_root_leaf_prefix;
             test "a fixed tensor has an entry" test_fixed_tensor_entry;
             test "find and get look entries up by name" test_find_get;
           ];
         group "errors"
           [
             test "missing entry raises with its name" test_missing_entry;
             test "unrelated entries are ignored" test_extra_entries_ignored;
             test "shape mismatch raises" test_shape_mismatch;
             test "dtype mismatch raises" test_dtype_mismatch;
             test "to_tensor is strict and returns the entry as stored"
               test_to_tensor;
             test "to_float casts floating-point entries only" test_to_float;
             test "concat rejects duplicate names" test_concat_duplicate;
             test "duplicate leaf paths are rejected" test_duplicate_names;
             test "empty tensor names are rejected" test_empty_name;
             test "to_int rejects missing and non-scalar entries"
               test_to_int_errors;
           ];
       ])
