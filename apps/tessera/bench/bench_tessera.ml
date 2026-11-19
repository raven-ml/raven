(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* TESSERA's rows on the tile TESSERA_TILE names, those [pytorch.py] beside
   this file times with TESSERA's own code (../PERFORMANCE.md). Each runs on a
   sample of the tile, every k-th pixel, which keeps its counts of
   observations:
   - [encode/M]: the student M in float32 over 8192 pixels, 32 on a CPU: the
     sample's batches, packed beforehand, each placed on the device, encoded
     and read back. The last batch, part empty, is left out, as a tile's one
     last batch is among a hundred; the setup prints the pixels a call
     encodes;
   - [encode/teacher]: the teacher in bfloat16 over 2048 pixels, in batches of
     512, likewise (GPU only);
   - [distil-step/medium]: one step of v2's distillation of the medium
     student on 2048 pixels in groups of 256, as [main.exe distil] cuts its
     default step, products in bfloat16 (GPU only).
   TESSERA_SAMPLE=N samples N pixels for [encode/M] instead: 250000, the
   whole tile, checks that the sample scales. The whole command's time,
   [main.exe encode] in a new process, is [whole.sh]'s: a suite cannot hold
   it.

   The weights are random: the time is the architecture's, not the values'.
   TESSERA_DEVICE names the devices as [Rig_gpus.open_all] reads them, [cpu]
   by default, and each case computes there compiled by [Rune.jit], or
   eagerly where TESSERA_EAGER is set. The devices open in each case's
   worker, which thumper forks, so this process never starts a driver, and a
   worker on a GPU prints the device memory raven held, its cache included:
   the most its calls held. [--warm] compiles each case in this process
   first, to fill tolk's disk cache: a worker forked on a Mac cannot reach
   Metal's compiler. *)

open Tessera

type 'd target = { at : 'd Nx.Placement.t; jit : bool }
type make = { make : 'd. 'd target -> unit -> unit }

let device = Option.value ~default:"cpu" (Sys.getenv_opt "TESSERA_DEVICE")
let jit = Sys.getenv_opt "TESSERA_EAGER" = None
let tile = Option.value ~default:"" (Sys.getenv_opt "TESSERA_TILE")

(* Whether a device [device] names is not the host's: a name of [Rig_gpus]
   whose kind is not CPU. Read from the names, so that this process opens
   none. *)
let gpu =
  String.split_on_char ',' device
  |> List.exists (fun n ->
      not (String.starts_with ~prefix:"CPU" (String.uppercase_ascii n)))

(* [c] made at the devices TESSERA_DEVICE names, and a function of the bytes
   they hold, their caches included ([Rig.allocated]): after a trial, the
   most its calls held, since the cache keeps the memory they return. *)
let call c =
  let ds = Result.error_to_failure (Rig_gpus.open_all device) in
  let module D = (val Nx.devices ds) in
  ( c.make { at = D.on; jit },
    fun () -> List.fold_left (fun a d -> a + Rig.allocated d) 0 ds )

(* A random model of [arch] at [t], compiled as [main.exe encode] compiles
   it. *)
let forward (type d) (t : d target) ~bf16 arch : d Batch.t -> d Nx.float32_t =
  let run (type s) (m : ((float, s, Nx.host) Nx.t, Nx.host) Model.t) =
    let m = Nx.Ptree.place (module Model) t.at m in
    if not t.jit then Model.apply m
    else
      let model = Nx.Ptree.instantiate (module Model) in
      Rune.jit
        Nx.Ptree.(model @-> v Batch.walk @-> returns tensor)
        Model.apply m
  in
  let m = Model.init ~key:(Nx.Rng.key 7) arch in
  if bf16 then run (Nx.Ptree.cast (module Model) Nx.bfloat16 m) else run m

(* [n] pixels of the tile, every k-th. *)
let sample n =
  let px, _ = Pixels.of_tile (Tile.load tile) in
  let k = max 1 (Pixels.length px / n) in
  let n = min n (Pixels.length px) in
  let indices = Nx.init Nx.int64 [| n |] (fun i -> Int64.of_int (i.(0) * k)) in
  Pixels.take ~indices px

(* [px]'s full batches of [shape], as {!Embedding.encode} packs them: all
   but the last, unless there is one; and the pixels they hold. *)
let batches shape px =
  let counts = Batch.v2_counts px in
  let rec go acc from =
    let b, next = Batch.pack shape px counts ~from in
    if next >= Pixels.length px then
      match acc with
      | [] -> ([ b ], next)
      | (_, last) :: _ -> (List.rev_map fst acc, last)
    else go ((b, next) :: acc) next
  in
  go [] 0

(* Each case compiles in its setup, on one batch. *)

let encode name ~n ~batch ~bf16 arch =
  let make (type d) (t : d target) =
    let f = forward t ~bf16 arch in
    let shape = Batch.typical ~pixels:batch ~slots:32 in
    let bs, held = batches shape (sample n) in
    Printf.eprintf "%s/%s: %d pixels a call\n%!" device name held;
    let run b = ignore (Nx.place Nx.Host.on (f (Batch.place t.at b))) in
    run (List.hd bs);
    fun () -> List.iter run bs
  in
  { make }

(* [main.exe distil]'s default step: 2048 pixels, in the fewest groups of at
   most 256 that hold them. *)
let distil_step =
  let make (type d) (t : d target) =
    let px = sample 2048 in
    let n = Pixels.length px in
    let groups = (n + 255) / 256 in
    let key = Nx.Rng.key 7 in
    let m = Model.init ~key Model.medium in
    let heads = Distil.heads ~key ~prefixes:m.prefixes 128 in
    let targets = Nx.Rng.normal ~key Nx.float32 [| n; 128 |] in
    let shape = Distil.shape ~pixels:((n + groups - 1) / groups) ~slots:32 px in
    let g, taken = Distil.groups ~key ~shape ~groups ~first:0 px targets in
    Printf.eprintf "%s/distil-step/medium: %d pixels in %d groups\n%!" device
      taken groups;
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
      if not t.jit then step
      else
        Rune.jit
          Nx.Ptree.(
            pair (instantiate (module Distil.Group)) s
            @-> returns (pair tensor s))
          step
    in
    let call () =
      ignore (Nx.item [] (Nx.place Nx.Host.on (fst (step (g, st)))))
    in
    call ();
    call
  in
  { make }

let case (name, c) =
  Thumper.bench_with_setup name
    ~setup:(fun () -> call c)
    ~teardown:(fun (_, held) ->
      if gpu then
        Printf.eprintf "%s/%s: device memory %d MiB\n%!" device name
          (held () / 1048576))
    (fun (f, _) -> f ())

let students =
  [
    ("nano", Model.nano);
    ("small", Model.small);
    ("medium", Model.medium);
    ("large", Model.large);
  ]

let size =
  Option.fold
    ~none:(if gpu then 8192 else 32)
    ~some:int_of_string
    (Sys.getenv_opt "TESSERA_SAMPLE")

let rows =
  List.map
    (fun (n, a) ->
      let name = "encode/" ^ n in
      let batch = if gpu then 4096 else 32 in
      (name, encode name ~n:size ~batch ~bf16:false a))
    students
  @
  if gpu then
    [
      ( "encode/teacher",
        encode "encode/teacher" ~n:2048 ~batch:512 ~bf16:true Model.teacher );
      ("distil-step/medium", distil_step);
    ]
  else []

let () =
  if tile = "" then begin
    prerr_endline "bench_tessera: no TESSERA_TILE, nothing to measure";
    exit 0
  end;
  if Array.mem "--warm" Sys.argv then begin
    if gpu then
      List.iter
        (fun (_, c) ->
          ignore (call c : (unit -> unit) * (unit -> int));
          Gc.full_major ())
        rows;
    exit 0
  end;
  exit
    (Thumper.run "tessera"
       ~config:
         Thumper.Config.(
           default |> samples 3 |> warmup 0. |> deadline 600.
           |> metrics [ Thumper.Metric.wall_time ])
       [ Thumper.group device (List.map case rows) ])
