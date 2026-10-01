(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Values placed on runtime devices, over test runtimes whose memory is host
   memory: the laws every runtime keeps, and what makes a runtime a device. *)

open Windtrap
open Nx_test

(* A runtime over host memory. *)
let runtime name =
  Nx_device.Driver.device ~name ~arch:"test" ~budget:max_int
    (Host_visible { memory = Nx_device.Driver.host_memory; mapping = None })

let r1 = runtime "R1"
let r2 = runtime "R2"

let devices =
  group "devices"
    [
      test
        "the host shares the runtimes' memory: a copy on it and on a runtime \
         computes and reads back" (fun () ->
          let p = Nx.Placement.replicated [ r1; Nx_device.host ] in
          let x = Nx.place p (Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |]) in
          let y = Nx.add x x in
          is_true (Nx.Placement.equal (Nx.placement y) p);
          equal (array float_exact) [| 2.; 4.; 6. |] (Nx.to_array y));
    ]

(* A buffer a caller hands to the host engine is on the host and of the value's
   format. *)
let host_buffers =
  let not_host = Exn.invalid_arg ~substring:"not CPU"
  and other_format = Exn.invalid_arg ~substring:"float64 buffer read as float32"
  and on_device = Nx_device.Buffer.create r1 Nx_dtype.Scalar.Float32 4
  and float64 = Nx_array.Elements.create Nx.float64 4 in
  group "host buffers"
    [
      test "Nx.Repr.host refuses a buffer on a device or of another format"
        (fun () ->
          let from_host b =
            let view = Nx_array.View.create [| Nx_device.Buffer.length b |] in
            Nx.Repr.host { dtype = Nx.float32; view; buffer = b }
          in
          raises_match not_host (fun () -> from_host on_device);
          raises_match other_format (fun () -> from_host float64));
    ]

(* Values on the disk: files, which the host reads where they lie, in their
   pages, and other devices read into their memory. *)
let disk =
  let on_disk = Runtimes.on_disk in
  let disk = Nx.Placement.device Nx_device.disk in
  let read () = Nx_device.Stats.bytes_out (Nx_device.stats Nx_device.disk) in
  let reads f =
    let before = read () in
    let y = f () in
    (y, read () - before)
  in
  let x = Nx.arange Nx.int32 0 12 1 |> Nx.reshape [| 3; 4 |] in
  let placement = Devices.placement in
  group "disk"
    [
      test "a value on the disk is placed there and read by nothing yet"
        (fun () ->
          let d, bytes = reads (fun () -> on_disk x) in
          equal placement disk (Nx.placement d);
          equal int 0 bytes);
      test
        "an operation computes on its file's pages on the host, and a constant \
         beside it is the host's" (fun () ->
          let d = on_disk x in
          let y, bytes = reads (fun () -> Nx.add d (Nx.ones_like d)) in
          equal placement Nx.Placement.host (Nx.placement y);
          equal ~msg:"bytes read" int 0 bytes;
          equal (array int32) (Nx.to_array (Nx.add_s x 1l)) (Nx.to_array y);
          equal placement Nx.Placement.host (Nx.placement (Nx.full_like d 0l)));
      test "a movement of it stays on the disk and reads nothing" (fun () ->
          let d = on_disk x in
          let y, bytes =
            reads (fun () -> Nx.transpose (Nx.slice [ Nx.R (1, 3) ] d))
          in
          equal placement disk (Nx.placement y);
          equal int 0 bytes;
          let z, bytes = reads (fun () -> Nx.place Nx.Placement.host y) in
          equal ~msg:"placed on the host, its pages" int 0 bytes;
          is_false ~msg:"a view of them"
            (Nx_array.View.is_c_contiguous (view z));
          equal (array int32)
            (Nx.to_array (Nx.transpose (Nx.slice [ Nx.R (1, 3) ] x)))
            (Nx.to_array z));
      test
        "beside a value on a device, it joins that device as a host value does"
        (fun () ->
          let on_d1 = Nx.place (Nx.Placement.device r1) x in
          let y = Nx.add on_d1 (on_disk x) in
          equal placement (Nx.Placement.device r1) (Nx.placement y);
          equal (array int32) (Nx.to_array (Nx.add x x)) (Nx.to_array y));
      test "placed on a device apart from the host, it is read into it"
        (fun () ->
          let y, bytes =
            reads (fun () -> Nx.place (Nx.Placement.device r1) (on_disk x))
          in
          equal ~msg:"bytes read" int 48 bytes;
          equal (array int32) (Nx.to_array x) (Nx.to_array y));
      test "a placement onto the disk raises" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"DISK") (fun () ->
              Nx.place disk x));
    ]

(* A host value's memory is claimed while an operation or a read uses it: a
   compiled call that holds it exclusive, writing over it, is never read. *)

module Claim = Nx_device.Buffer.Claim

let memory x =
  match Nx.Repr.v x with
  | Host a -> a.buffer
  | Placed _ | Traced _ -> fail "expected a host value"

(* [f ()] with [x]'s memory held exclusive, as a consuming call holds it. *)
let held_exclusive x f =
  let m = memory x in
  Claim.read m;
  is_true ~msg:"exclusive" (Claim.try_exclusive m);
  Fun.protect
    ~finally:(fun () ->
      Claim.finish m;
      Claim.release m)
    f

let in_use = Exn.invalid_arg ~substring:"in use"

let claims =
  group "claims"
    [
      test "an operation and a read of memory held exclusive raise" (fun () ->
          let x = Nx.create Nx.float32 [| 2 |] [| 1.; 2. |] in
          held_exclusive x (fun () ->
              raises_match ~msg:"an operation" in_use (fun () ->
                  ignore (Nx.add x x));
              raises_match ~msg:"to_array" in_use (fun () ->
                  ignore (Nx.to_array x));
              raises_match ~msg:"item" in_use (fun () ->
                  ignore (Nx.item [ 0 ] x));
              raises_match ~msg:"fold_item" in_use (fun () ->
                  ignore (Nx.fold_item ( +. ) 0. x)));
          equal (array (float 1e-6)) [| 1.; 2. |] (Nx.to_array x));
      test "an operation and a read release their claims, also when they raise"
        (fun () ->
          let x = Nx.create Nx.float32 [| 2 |] [| 1.; 2. |] in
          ignore (Nx.add x x);
          ignore (Nx.to_array x);
          raises (Failure "f") (fun () ->
              Nx.iter_item (fun _ -> failwith "f") x);
          held_exclusive x ignore);
    ]

let () =
  exit
    (run "nx runtime devices"
       (devices :: host_buffers :: disk :: claims :: Runtimes.laws [ r1; r2 ]))
