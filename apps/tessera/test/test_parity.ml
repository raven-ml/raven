(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The models against TESSERA's own code, on upstream/tiny: a random student and
   teacher of the released forms, their embeddings of a tile of 3 x 4 pixels,
   and the loss and gradients of a distillation step of the student and of a
   pretraining group of the teacher, raven's draws fed to TESSERA's model, as
   gen/tiny.py recorded them. *)

open Windtrap
open Tessera

let dir = "upstream/tiny"
let file f = Filename.concat dir f
let npy dt f = Nx.unpack dt (Nx_io.load_npy (file f))
let f64 x = Nx.to_array (Nx.cast Nx.float64 x)

(* [e], [[n; d]], against the recorded [[h; w; d]]: NaN at the same pixels,
   those without an observation, and within [tol] elsewhere. Both are float32:
   they differ by at most 1e-6 here. *)
let close ~tol e r =
  let r = Nx.reshape (Nx.shape e) r in
  equal (array bool) (Nx.to_array (Nx.isnan r)) (Nx.to_array (Nx.isnan e));
  let d = Nx.where (Nx.isnan r) (Nx.zeros_like r) (Nx.abs (Nx.sub e r)) in
  at_most float_exact ~than:tol (Nx.item [] (Nx.max d))

let embed m =
  let px, _ = Pixels.of_tile (Tile.load dir) in
  Embedding.encode
    ~shape:(Batch.typical ~pixels:8 ~slots:4)
    Nx.Host.on (Model.apply m) px

let student = lazy (Model.load (file "student.pt"))

let embeddings =
  [
    test "a student's embedding is TESSERA's" (fun () ->
        close ~tol:1e-5
          (embed (Lazy.force student))
          (npy Nx.float32 "student.npy"));
    test "a teacher's embedding is TESSERA's" (fun () ->
        close ~tol:1e-5
          (embed (Model.load (file "teacher.pt")))
          (npy Nx.float32 "teacher.npy"));
    test "a student's int8 codes are TESSERA's" (fun () ->
        let e = embed (Lazy.force student) in
        let observed =
          Nx.to_array (Nx.logical_not (Nx.isnan (Nx.slice [ A; I 0 ] e)))
        in
        let rows =
          List.filter (fun i -> observed.(i)) (List.init (Nx.dim 0 e) Fun.id)
        in
        let rows =
          Nx.create Nx.int64
            [| List.length rows |]
            (Array.of_list (List.map Int64.of_int rows))
        in
        let codes, scales = Embedding.quantize (Nx.take ~axis:0 rows e) in
        let ref_codes = npy Nx.int8 "student_int8.npy" in
        let differ =
          Nx.abs (Nx.sub (Nx.cast Nx.int32 codes) (Nx.cast Nx.int32 ref_codes))
        in
        at_most int ~than:1 (Int32.to_int (Nx.item [] (Nx.max differ)));
        equal
          (array (float 1e-6))
          (f64 (npy Nx.float32 "student_scales.npy"))
          (f64 scales));
  ]

(* Each leaf of a structure [s] of [x], by path, in float64. *)
let leaves s x =
  Nx.Ptree.fold s
    (fun p a acc -> (Nx.Ptree.Path.to_string p, f64 a) :: acc)
    x []
  |> List.rev

let norm a = Float.sqrt (Array.fold_left (fun s x -> s +. (x *. x)) 0. a)

(* [loss] and the gradient [g] of [p] against the float64 references [ref_loss]
   and [r]: the loss within [tol] of its value, and each leaf's distance within
   [tol] of the whole gradient's norm, so that a leaf whose exact gradient is 0,
   as an attention's key bias, is judged by its rounding. Float32 lies within
   1.5e-7 of them for the distillation, and 9e-7 for the pretraining, whose
   whitening and batch normalisation divide by a group's deviations. *)
let close_gradients ~tol p ~loss ~ref_loss g r =
  equal (float (tol *. ref_loss)) ref_loss (Nx.item [] loss);
  let mine = leaves p g and theirs = leaves p r in
  let scale = norm (Array.concat (List.map snd theirs)) in
  List.iter2
    (fun (path, a) (_, b) ->
      let d = norm (Array.map2 ( -. ) a b) /. scale in
      if d > tol then failf "%s: distance %.3g of the gradient's norm" path d)
    mine theirs

let distillation =
  test "a distillation step's loss and gradients are TESSERA's" (fun () ->
      let m = Lazy.force student in
      let heads = Distil.load_heads (file "heads.npz") m.prefixes in
      let pixels, b, keys = Tiny.step dir in
      let targets = Nx.take ~axis:0 pixels (npy Nx.float32 "targets.npy") in
      let p = Nx.Ptree.instantiate (module Distil.Params) in
      let loss, g =
        Rune.value_and_grad p
          (fun q -> Distil.loss ~dropout:Model.v2_dropout ~keys m q b ~targets)
          { Distil.net = m.net; heads }
      in
      let r = Model.load (file "grads.pt") in
      let r_heads = Distil.load_heads (file "heads_grads.npz") m.prefixes in
      close_gradients ~tol:1e-6 p ~loss
        ~ref_loss:(Nx.item [] (npy Nx.float64 "loss.npy"))
        g
        { Distil.net = r.net; heads = r_heads })

let pretraining =
  test "a pretraining group's loss and gradients are TESSERA's" (fun () ->
      let m = Model.load (file "teacher.pt") in
      let projector = Barlow.load_projector (file "projector.npz") in
      let v = Tiny.views dir in
      let p = Nx.Ptree.instantiate (module Barlow.Params) in
      let loss, g =
        Rune.value_and_grad p
          (fun q ->
            Barlow.loss Barlow.v1_weights ~dropout:Model.v2_dropout m q v)
          { Barlow.net = m.net; projector }
      in
      let r = Model.load (file "barlow_grads.pt") in
      let r_projector =
        Barlow.load_projector (file "barlow_projector_grads.npz")
      in
      close_gradients ~tol:1e-5 p ~loss
        ~ref_loss:(Nx.item [] (npy Nx.float64 "barlow_loss.npy"))
        g
        { Barlow.net = r.net; projector = r_projector })

let () =
  exit
    (run "tessera.parity"
       [
         group "embeddings" embeddings;
         group "steps" [ distillation; pretraining ];
       ])
