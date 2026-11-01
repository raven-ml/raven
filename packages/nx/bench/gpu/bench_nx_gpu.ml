(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Placing onto this machine's CUDA GPU, each row what a user calls. An
   executable of its own: a process that links the GPU drivers makes every
   full collection longer, and bench/nx's rows run collections. Without a
   CUDA GPU it measures nothing.

   host-to-gpu-16 places sixteen host elements: a placement that waited for
   its copy would add a round trip to the GPU, about 4 us on an RTX 5000 Ada.
   split-axis1-to-gpu-1024 gathers a 1024 x 1024 float32 value split by
   columns over four memory devices: each device's window is 1024 rows of the
   destination, one copy of boxes, where a copy per row would add a round trip
   per row. *)

module A = Nx_array
module D = Nx_array.Dtype

(* The GPU and a memory device are opened in the measuring worker: a device
   opened before the fork is lost in it. *)
let cuda () =
  let path =
    List.find (fun p -> p.Rig_gpus.kind = "CUDA") (Rig_gpus.paths ())
  in
  match path.gpus () with
  | Ok (g :: _) -> Result.get_ok (g.open_ ())
  | Ok [] -> failwith "no CUDA GPU"
  | Error why -> failwith why

let memory k = Result.get_ok (Rig.memory_device (Printf.sprintf "bench-m%d" k))

let filled s =
  let n = Array.fold_left ( * ) 1 s in
  Nx.Repr.of_array Nx.Host.v
    (A.of_array D.Float32 (Iarray.of_array s) (Array.make n 1.))

(* A value and the placement a row puts it at. *)
type move =
  | Move : 'e Nx.Placement.t * (float, D.float32_elt, 'd) Nx.t -> move

let host_16 () =
  let module G = (val Nx.devices [ cuda () ]) in
  Move (G.on, filled [| 16 |])

let split_columns () =
  let module G = (val Nx.devices [ cuda () ]) in
  let module Four = (val Nx.devices (List.init 4 memory)) in
  Move (G.on, Nx.place (Four.split ~axis:1) (filled [| 1024; 1024 |]))

type value = Value : (float, D.float32_elt, 'd) Nx.t -> value

let place (Move (p, x)) = Value (Nx.place p x)

let rows =
  Thumper.group "place"
    [
      Thumper.bench_with_setup "host-to-gpu-16" ~setup:host_16 place;
      Thumper.bench_with_setup "split-axis1-to-gpu-1024" ~setup:split_columns
        place;
    ]

let () =
  if not (Rig_cuda_support.present ()) then exit 0;
  exit (Thumper.run "nx-gpu" [ rows ])
