(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The released models against TESSERA's own code, on the tiles of
   $TESSERA_FIXTURES/tiles, as gen/release.py recorded their embeddings in
   $TESSERA_FIXTURES/emb: each model whose weights $TESSERA_WEIGHTS holds, the
   students eagerly on the CPU, the teacher compiled there, all at float32. An
   embedding matches when it is NaN at the same pixels and within 2e-5
   elsewhere, and a student's int8 codes when each is within 1 of the recorded
   code at most at 0.01 % of them, as upstream's own embeddings differ across
   devices and batch sizes. *)

open Tessera

let models =
  [
    ("nano", "student_nano.pt");
    ("small", "student_small.pt");
    ("medium", "student_medium.pt");
    ("large", "student_large.pt");
    ("teacher", "tessera_v2_2B_teacher.pt");
  ]

let failed = ref 0

let report name pass text =
  Printf.printf "%-28s %s %s\n%!" name (if pass then "ok  " else "FAIL") text;
  if not pass then incr failed

let max_abs a b =
  let d = Nx.where (Nx.isnan b) (Nx.zeros_like b) (Nx.abs (Nx.sub a b)) in
  Nx.item [] (Nx.max d)

(* The rows of [r] that are not NaN. *)
let rows_with_values r =
  let o = Nx.to_array (Nx.logical_not (Nx.isnan (Nx.slice [ A; I 0 ] r))) in
  let l = List.filter (fun i -> o.(i)) (List.init (Array.length o) Fun.id) in
  Nx.create Nx.int64
    [| List.length l |]
    (Array.of_list (List.map Int64.of_int l))

let check ~fixtures ~tile name m ~jit =
  let px, _ =
    Pixels.of_tile (Tile.load (Filename.concat fixtures ("tiles/" ^ tile)))
  in
  let f =
    if jit then
      Rune.jit
        Nx.Ptree.(
          instantiate (module Model) @-> v Batch.walk @-> returns tensor)
        Model.apply m
    else Model.apply m
  in
  let pixels = if name = "teacher" then 512 else 4096 in
  let e =
    Embedding.encode ~shape:(Batch.typical ~pixels ~slots:32) Nx.Host.on f px
  in
  let file f =
    Filename.concat fixtures (Printf.sprintf "emb/%s/%s%s.npy" tile name f)
  in
  let npy dt f = Nx.unpack dt (Nx_io.load_npy (file f)) in
  let r = Nx.reshape (Nx.shape e) (npy Nx.float32 "") in
  let nan = Nx.to_array (Nx.isnan r) = Nx.to_array (Nx.isnan e) in
  let d = max_abs e r in
  report
    (name ^ "/" ^ tile)
    (nan && d <= 2e-5)
    (Printf.sprintf "max |d| %.3g, NaN at the same pixels %b" d nan);
  if name <> "teacher" then begin
    let codes, _ =
      Embedding.quantize (Nx.take ~axis:0 (rows_with_values r) e)
    in
    let diff =
      Nx.abs
        (Nx.sub (Nx.cast Nx.int32 codes)
           (Nx.cast Nx.int32 (npy Nx.int8 "_int8")))
    in
    let worst = Int32.to_int (Nx.item [] (Nx.max diff)) in
    let share =
      Nx.item [] (Nx.mean (Nx.cast Nx.float64 (Nx.greater_s diff 0l)))
    in
    report
      (name ^ "/" ^ tile ^ "/int8")
      (worst <= 1 && share <= 1e-4)
      (Printf.sprintf "codes off by %d at most, at %.3g of them" worst share)
  end

let () =
  match
    (Sys.getenv_opt "TESSERA_WEIGHTS", Sys.getenv_opt "TESSERA_FIXTURES")
  with
  | Some weights, Some fixtures ->
      let tiles = Sys.readdir (Filename.concat fixtures "tiles") in
      Array.sort compare tiles;
      List.iter
        (fun (name, file) ->
          let file = Filename.concat weights file in
          if Sys.file_exists file then begin
            let m = Model.load file in
            Array.iter
              (fun tile -> check ~fixtures ~tile name m ~jit:(name = "teacher"))
              tiles;
            Gc.full_major ()
          end)
        models;
      exit (if !failed = 0 then 0 else 1)
  | _ ->
      print_endline
        "parity: TESSERA_WEIGHTS and TESSERA_FIXTURES name no models, nothing \
         to check"
