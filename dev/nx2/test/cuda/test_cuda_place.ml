(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Placing values between the host and CUDA's GPU 0 from host memory the GPU
   cannot map. Every test skips where CUDA sees no GPU. *)

open Windtrap
module S = Nx_cuda_support
module A = Nx_array
module Dt = Nx_array.Dtype

let gpu =
  let t =
    lazy
      (if Rig_cuda.count () = 0 then None
       else
         match
           Rig.open_
             (module Rig_cuda)
             ~name:"CUDA:0"
             (fun () -> Rig_cuda.open_ 0)
         with
         | Ok d -> Some d
         | Error e -> failwith e)
  in
  fun () ->
    match Lazy.force t with
    | Some d -> d
    | None -> skip ~reason:"CUDA sees no GPU" ()

let rows = 64
let cols = 64

(* A [rows × cols] float32 host array holding [0, 1, …] in C order, over a
   bigarray's memory from its second element on, which never starts on a page,
   so no GPU maps it; transposed, its index [(i, j)] holds [j·cols + i]. *)
let unpaged ~transposed =
  let n = rows * cols in
  let ba = Bigarray.(Array1.create float32 c_layout (n + 1)) in
  for i = 0 to n - 1 do
    ba.{i + 1} <- float_of_int i
  done;
  let mem = Rig.Buffer.of_bigarray (Bigarray.Array1.sub ba 1 n) in
  let a = A.v Dt.Float32 (A.Layout.contiguous [| rows; cols |]) mem in
  if transposed then Option.get (A.move (A.Move.Permute [| 1; 0 |]) a) else a

(* A host value split on [axis] between the host and the GPU, whose half is
   strided in the source: the GPU's array holds its half's bytes alone, and the
   value placed back on the host has the source's elements. *)
let half_window ~transposed ~axis () =
  let gpu = gpu () in
  let module Two = (val Nx.devices [ Rig.host; gpu ]) in
  let a = unpaged ~transposed in
  let y = Nx.place (Two.split ~axis) (Nx.Repr.of_array Nx.Host.v a) in
  let half = (require_some (Nx.Repr.shards y)).(1) in
  equal string ~msg:"its device" (Rig.name gpu) (Rig.name (A.device half));
  equal int ~msg:"its buffer's bytes"
    (Dt.bytes Dt.Float32 (rows * cols / 2))
    (Rig.Buffer.length (A.buffer half));
  let back = require_some (Nx.Repr.array (Nx.place Nx.Host.on y)) in
  equal (list float_exact) ~msg:"its elements"
    (Array.to_list (A.to_array a))
    (Array.to_list (A.to_array back))

(* A 128 × 128 float32 host array, 64 KiB, whose memory starts on a page. *)
let paged () =
  A.of_array Dt.Float32 [| 128; 128 |] (Array.init (128 * 128) float_of_int)

(* The GPU maps host memory without sharing it, so a paged host value placed on
   it is copied into memory of its own. *)
let own_memory () =
  let module One = (val Nx.devices [ gpu () ]) in
  let a = paged () in
  let v = Nx.place One.on (Nx.Repr.of_array Nx.Host.v a) in
  let y = require_some (Nx.Repr.array v) in
  equal ~msg:"borrowed" bool false (Rig.Buffer.is_borrowed (A.buffer y));
  equal ~msg:"its bytes" int
    (Dt.bytes Dt.Float32 (128 * 128))
    (Rig.Buffer.length (A.buffer y));
  let back = require_some (Nx.Repr.array (Nx.place Nx.Host.on v)) in
  equal (list float_exact) ~msg:"its elements"
    (Array.to_list (A.to_array a))
    (Array.to_list (A.to_array back))

let place =
  group "place"
    [
      test "a column window copies only its own bytes to the GPU"
        (half_window ~transposed:false ~axis:1);
      test "a row window of a transposed value copies only its own bytes"
        (half_window ~transposed:true ~axis:0);
      test "a paged host value placed on the GPU is copied to its memory"
        own_memory;
    ]

let () =
  S.hold_gpu ();
  exit (run "nx.cuda place" [ place ])
