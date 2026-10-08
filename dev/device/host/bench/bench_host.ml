(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Linking and calling host programs, each row beside the floors that bound it.

   A link of affine.c, about 100 bytes of code, is bounded by mapping,
   installing and unmapping a page and by reading the object; a link of linked.c
   adds a lookup in the process, and a link of many.c, 2,000 relocations, its
   read. A call from OCaml is bounded by releasing and acquiring the runtime,
   and a split by the pool's empty job of the same threads and blocks. Calls
   from linked code are loop.c's: 1,000 through device_host_call, beside 1,000
   direct calls; and 100 splits, beside the pool's job. A compute-bound kernel
   over 4 Mi floats, split into 4 blocks per worker, is bounded by its serial
   run divided by the workers. *)

module Host = Device_host
module S = Device_host_support

external map_install_unmap : bytes -> unit
  = "device_host_bench_map_install_unmap"

external lookup : unit -> int = "device_host_bench_lookup"
external release_acquire : unit -> unit = "device_host_bench_release_acquire"

let obj = S.fixture ~dir:"../test/fixtures"

let link ~entry o () =
  match Host.link ~entry o with Ok p -> p | Error e -> failwith e

let read o () =
  match Device_elf.of_string o with Ok o -> o | Error e -> failwith e

let workers = Host.workers ()

(* As many bytes as [o]'s image. *)
let image_of o = Bytes.create (Int.max 1 (read o ()).size)

let link_rows =
  let affine = obj "affine" and linked = obj "linked" and many = obj "many" in
  let affine_image = image_of affine and many_image = image_of many in
  Thumper.group "link"
    [
      Thumper.bench "affine" (link ~entry:"affine" affine);
      Thumper.bench "linked" (link ~entry:"linked" linked);
      Thumper.bench "many-relocations" (link ~entry:"many" many);
      Thumper.bench "floor-map-install-unmap-affine" (fun () ->
          map_install_unmap affine_image);
      Thumper.bench "floor-map-install-unmap-many" (fun () ->
          map_install_unmap many_image);
      Thumper.bench "floor-elf-affine" (read affine);
      Thumper.bench "floor-elf-linked" (read linked);
      Thumper.bench "floor-elf-many" (read many);
      Thumper.bench "floor-lookup" lookup;
    ]

let call_rows =
  let p = link ~entry:"empty" (obj "empty") () in
  let buffers = [| 0; 0 |] and values = [| 0; 0; 0 |] in
  let split blocks = Some { Host.extent = blocks; blocks; lo = 0; hi = 1 } in
  let one = split 1 and all = split workers in
  Thumper.group "call"
    [
      Thumper.bench "empty" (fun () -> Host.call p buffers values);
      Thumper.bench "split-1" (fun () -> Host.call ?split:one p buffers values);
      Thumper.bench "split-empty" (fun () ->
          Host.call ?split:all p buffers values);
      Thumper.bench "floor-release-acquire" release_acquire;
      Thumper.bench "floor-pool-job" (fun () ->
          S.empty_job ~threads:workers ~total:workers ~chunks:workers);
    ]

(* loop.c's values: f, count, how, a split's four values, f's n values. *)
let loop_values ~f ~count ~how ~split =
  Array.concat [ [| f; count; how |]; split; [| 3; 0; 0; 0 |] ]

let entry_rows =
  let loop = link ~entry:"loop" (obj "loop") () in
  let empty = link ~entry:"empty" (obj "empty") () in
  let f = Host.address empty in
  let all = [| workers; workers; 0; 1 |] and none = [| 0; 0; 0; 0 |] in
  (* The loop calls [empty] by its address: the closure keeps it mapped. *)
  let run values () =
    Host.call loop [||] values;
    ignore (Sys.opaque_identity empty)
  in
  Thumper.group "entry"
    [
      Thumper.bench "call-1000"
        (run (loop_values ~f ~count:1000 ~how:0 ~split:none));
      Thumper.bench "floor-direct-call-1000"
        (run (loop_values ~f ~count:1000 ~how:2 ~split:none));
      Thumper.bench "split-100"
        (run (loop_values ~f ~count:100 ~how:1 ~split:all));
      Thumper.bench "floor-pool-job-100" (fun () ->
          for _ = 1 to 100 do
            S.empty_job ~threads:workers ~total:workers ~chunks:workers
          done);
    ]

let scale_rows =
  let n = 4 * 1024 * 1024 in
  let p = link ~entry:"scale" (obj "scale") () in
  let x = Bigarray.(Array1.create float32 c_layout n) in
  Bigarray.Array1.fill x 1.;
  let buffers = [| S.address x |] in
  let split = Some { Host.extent = n; blocks = 4 * workers; lo = 0; hi = 1 } in
  let run ?split values () =
    Host.call ?split p buffers values;
    ignore (Sys.opaque_identity x)
  in
  Thumper.group "split"
    [
      Thumper.bench "scale-4Mi" (run ?split [| 0; 0 |]);
      Thumper.bench "serial-scale-4Mi" (run [| 0; n |]);
    ]

let () =
  exit
    (Thumper.run "device_host" [ link_rows; call_rows; entry_rows; scale_rows ])
