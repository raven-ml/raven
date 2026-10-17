(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Loading and launching host code, each row beside the floors that bound it.

   A load of affine.c, about 100 bytes of code, is bounded by mapping,
   installing and unmapping a page and by reading the object; a load of
   linked.c adds a lookup in the process, and a load of many.c, 2,000
   relocations, its read. A launch is bounded by releasing and acquiring the
   runtime, and a launch over many groups by the pool's empty job of the same
   threads and chunks. A compute-bound kernel over 4 Mi floats, in 4 groups per
   worker, is bounded by its run in one group divided by the workers. *)

module B = Rig.Buffer
module Sub = Rig.Submission
module S = Rig_host_support

external map_install_unmap : bytes -> unit = "rig_host_bench_map_install_unmap"
external lookup : unit -> int = "rig_host_bench_lookup"
external release_acquire : unit -> unit = "rig_host_bench_release_acquire"

(* The host suite's code; that which other suites also run lives in
   test/fixtures. *)
let obj f =
  match f with
  | "affine" -> S.fixture ~dir:"../../test/fixtures" f
  | _ -> S.fixture ~dir:"../../test/host/fixtures" f

let load o () =
  match Rig.Image.load Rig.host o with Ok i -> i | Error e -> failwith e

let read o () =
  match Rig_elf.of_string o with Ok o -> o | Error e -> failwith e

let workers = S.workers ()

(* As many bytes as [o]'s image. *)
let image_of o = Bytes.create (Int.max 1 (read o ()).size)

let load_rows =
  let affine = obj "affine" and linked = obj "linked" and many = obj "many" in
  let affine_image = image_of affine and many_image = image_of many in
  Thumper.group "load"
    [
      Thumper.bench "affine" (load affine);
      Thumper.bench "linked" (load linked);
      Thumper.bench "many-relocations" (load many);
      Thumper.bench "floor-map-install-unmap-affine" (fun () ->
          map_install_unmap affine_image);
      Thumper.bench "floor-map-install-unmap-many" (fun () ->
          map_install_unmap many_image);
      Thumper.bench "floor-elf-affine" (read affine);
      Thumper.bench "floor-elf-linked" (read linked);
      Thumper.bench "floor-elf-many" (read many);
      Thumper.bench "floor-lookup" lookup;
    ]

(* A run of a submission of one launch of [image]'s [f] over [groups] groups
   of [threads] threads, with [params] bytes that [set] stores. *)
let launch ?(refs = [||]) ?(params = 0) ?(set = fun _ _ -> ()) image f
    ~groups ~threads =
  let fn = Option.get (Rig.Image.resolve image f) in
  let s = Sub.make Rig.host [| Sub.launch ~refs fn ~params |] in
  let run = Sub.Run.make s and k = Sub.block s 0 in
  Sub.Run.groups run k groups 1 1;
  Sub.Run.threads run k threads 1 1;
  set run k;
  run

let submit run buffers () =
  ignore (Rig.submit run ~buffers : Rig.Point.t)

let launch_rows =
  let empty = load (obj "empty") () in
  let one = launch empty "empty" ~groups:1 ~threads:1
  and all = launch empty "empty" ~groups:workers ~threads:1 in
  Thumper.group "launch"
    [
      Thumper.bench "empty" (submit one [||]);
      Thumper.bench "workers-empty" (submit all [||]);
      Thumper.bench "floor-release-acquire" release_acquire;
      Thumper.bench "floor-pool-job" (fun () ->
          S.empty_job ~threads:workers ~total:workers ~chunks:workers);
    ]

let scale_rows =
  let n = 4 * 1024 * 1024 in
  let image = load (obj "scale") () in
  let x = Bigarray.(Array1.create float32 c_layout n) in
  Bigarray.Array1.fill x 1.;
  let buffers = [| B.of_bigarray x |] in
  let scale ~groups =
    launch image "scale" ~refs:Sub.[| write 0 ~at:0 |] ~params:16
      ~set:(fun run k -> Sub.Run.int64 run k 8 n)
      ~groups
      ~threads:((n + groups - 1) / groups)
  in
  Thumper.group "split"
    [
      Thumper.bench "scale-4Mi" (submit (scale ~groups:(4 * workers)) buffers);
      Thumper.bench "serial-scale-4Mi" (submit (scale ~groups:1) buffers);
    ]

let () =
  exit (Thumper.run "rig_host" [ load_rows; launch_rows; scale_rows ])
