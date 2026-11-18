(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* TESSERA's two workloads on a real tile, the rows [pytorch.py] beside this
   file times with TESSERA's own code: - [encode M]: the embedding of the tile's
   first 16384 pixels by each released architecture, as main.exe's encode
   computes it: the students in float32 in batches of 4096, the teacher in
   bfloat16 in batches of 512 (on a GPU only: its tile takes hours on a CPU).
   The rate is 16384 pixels over the row's time; - [distil step]: one step of
   v2's distillation of the medium student on the tile's first 4096 pixels,
   products in bfloat16, toward targets of 128 values. The weights are random:
   the time is the architecture's, not the values'.

   TESSERA_TILE names the tile, a directory of TESSERA's files; without one
   there is nothing to measure. TESSERA_DEVICE names the device, [cpu-jit]
   (compiled on the host) by default, or a GPU by its name ([Rig_gpus]). A GPU
   opens in each case's worker, which thumper forks, so this process never
   starts a driver. On a GPU, [--warm] calls each case once, in a process of its
   own, to fill tolk's disk cache: a worker forked on a Mac cannot reach Metal's
   compiler. *)

open Tessera

type 'd target = { at : 'd Nx.Placement.t; jit : bool }
type make = { make : 'd. 'd target -> unit -> unit }

let device = Option.value ~default:"cpu-jit" (Sys.getenv_opt "TESSERA_DEVICE")
let gpu = device <> "cpu" && device <> "cpu-jit"

let call c =
  match device with
  | "cpu" -> c.make { at = Nx.Host.on; jit = false }
  | "cpu-jit" -> c.make { at = Nx.Host.on; jit = true }
  | name ->
      let g = Result.error_to_failure (Rig_gpus.open_ name) in
      let module D = (val Nx.devices [ g ]) in
      c.make { at = D.on; jit = true }

let pixels tile n =
  let px, _ = Pixels.of_tile (Tile.load tile) in
  Pixels.sub px ~pos:0 ~len:(min n (Pixels.length px))

let encode tile arch ~batch ~bf16 =
  let make (type d) (t : d target) =
    let px = pixels tile 16384 in
    let m = Model.init ~key:(Nx.Rng.key 7) arch in
    let run (type s) (m : ((float, s, Nx.host) Nx.t, Nx.host) Model.t) =
      let m = Nx.Ptree.place (module Model) t.at m in
      let model = Nx.Ptree.instantiate (module Model) in
      let f =
        if t.jit then
          Rune.jit
            Nx.Ptree.(model @-> v Batch.walk @-> returns tensor)
            Model.apply m
        else Model.apply m
      in
      let shape = Batch.typical ~pixels:batch ~slots:32 in
      fun () -> ignore (Embedding.encode ~shape t.at f px)
    in
    if bf16 then run (Nx.Ptree.cast (module Model) Nx.bfloat16 m) else run m
  in
  { make }

let distil_step tile =
  let make (type d) (t : d target) =
    let px = pixels tile 4096 in
    let key = Nx.Rng.key 7 in
    let m = Model.init ~key Model.medium in
    let heads = Distil.heads ~key ~prefixes:m.prefixes 128 in
    let targets = Nx.Rng.normal ~key Nx.float32 [| Pixels.length px; 128 |] in
    let shape = Distil.shape ~pixels:4096 ~slots:32 px in
    let g, _ = Distil.groups ~key ~shape ~groups:1 ~first:0 px targets in
    let g = Nx.Ptree.place (module Distil.Group) t.at g in
    let st = Nx.Ptree.place (module Distil.State) t.at (Distil.init m heads) in
    let r = Distil.v2_recipe in
    let step (g, st) =
      Distil.step ~compute:Nx.bfloat16 ~dropout:Model.v2_dropout
        ~lr:(Vega.rate r.lr) ~weight_decay:(Vega.rate r.weight_decay)
        ~clip:r.clip (g, st)
    in
    let s = Nx.Ptree.instantiate (module Distil.State) in
    let step =
      if t.jit then
        Rune.jit
          Nx.Ptree.(
            pair (instantiate (module Distil.Group)) s
            @-> returns (pair tensor s))
          step
      else step
    in
    fun () -> ignore (Nx.item [] (Nx.place Nx.Host.on (fst (step (g, st)))))
  in
  { make }

(* A case's inputs are made and its call compiled in the worker, with one
   uncounted call: compiling may start domains, after which thumper cannot
   fork. *)
let case name c =
  Thumper.bench_with_setup name
    ~setup:(fun () ->
      let f = call c in
      f ();
      f)
    (fun f -> f ())

let () =
  match Sys.getenv_opt "TESSERA_TILE" with
  | None ->
      prerr_endline "bench_tessera: no TESSERA_TILE, nothing to measure";
      exit 0
  | Some tile ->
      let warm = Array.mem "--warm" Sys.argv in
      let case name c =
        if warm then begin
          if gpu then (call c) ();
          Gc.full_major ()
        end;
        case name c
      in
      if warm && not gpu then exit 0;
      let students =
        [
          ("nano", Model.nano);
          ("small", Model.small);
          ("medium", Model.medium);
          ("large", Model.large);
        ]
      in
      let encodes =
        List.map
          (fun (n, a) ->
            case ("encode " ^ n) (encode tile a ~batch:4096 ~bf16:false))
          students
        @
        if gpu then
          [
            case "encode teacher"
              (encode tile Model.teacher ~batch:512 ~bf16:true);
          ]
        else []
      in
      if warm then exit 0;
      exit
        (Thumper.run "tessera"
           ~config:Thumper.Config.(default |> deadline 900.)
           [
             Thumper.group device
               (encodes @ [ case "distil step" (distil_step tile) ]);
           ])
