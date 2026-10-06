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
          let p =
            Nx.Placement.replicated [ Nx.Device.make r1; Nx.Device.host ]
          in
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

(* Whether [x] is far from the offsets and strides of small buffers. *)
let big x = x > 1 lsl 30 || x < -(1 lsl 30)

(* The representation's constructors build no value that a kernel would read
   outside its buffers: every element a view reaches lies in them, and each
   starts on a byte aligned to the value's elements. *)
let representation =
  let outside = Exn.invalid_arg ~substring:"reaches outside"
  and other_count = Exn.invalid_arg ~substring:"Nx.of_buffer: shape"
  and p = Nx.Placement.on (Nx.Device.make r1) in
  let shapes =
    Gen.(
      let* rank = int_range 0 3 in
      pair (int_range 0 8) (array ~size:(constant rank) edge_dims))
  in
  let host n view =
    Nx.Repr.host
      {
        dtype = Nx.float32;
        view;
        buffer = Nx_array.Elements.create Nx.float32 n;
      }
  and placed n view =
    let b = Nx_device.Buffer.create r1 Nx_dtype.Scalar.Float32 n in
    Nx.Repr.Placed.v p Nx.float32 view (Nx.Repr.Storage.v p [ b ])
  and shards n view =
    let b = Nx_device.Buffer.create r1 Nx_dtype.Scalar.Float32 n in
    Nx.of_shards p Nx.float32 view [ b ]
  in
  let of_buffer n shape =
    let b = Nx_device.Buffer.create r1 Nx_dtype.Scalar.Float32 n in
    Nx.of_buffer Nx.float32 shape b
  in
  (* Each constructor that takes a view builds a value exactly when the view is
     inside its buffers. *)
  let law (n, view) =
    let build = [ ("host", host); ("placed", placed); ("of_shards", shards) ] in
    if view_inside view n then
      List.iter (fun (_, make) -> ignore (make n view)) build
    else
      List.iter
        (fun (msg, make) -> raises_match ~msg outside (fun () -> make n view))
        build
  in
  let row (offset, strides, shape, n) =
    (n, Nx_array.View.create ~offset ~strides shape)
  in
  group "representation"
    [
      prop
        "Nx.Repr.host, Nx.Repr.Placed.v and Nx.of_shards build a value exactly \
         when its view reaches only elements of its buffers"
        edge_views (fun (n, view) ->
          let ok = view_inside view n
          and shape = Nx_array.View.shape view
          and strides = Nx_array.View.strides view in
          cover "a view inside its buffers" ok;
          cover "a view reaching outside them" (not ok);
          cover "a zero-size view" (Array.exists (( = ) 0) shape);
          cover "a negative stride" (Array.exists (fun s -> s < 0) strides);
          cover "a negative dimension" (Array.exists (fun d -> d < 0) shape);
          cover "an element count past max_int" (count_overflows shape);
          cover "an extreme offset or stride"
            (big (Nx_array.View.offset view) || Array.exists big strides);
          law (n, view));
      cases
        ~name:(fun (n, view) -> Format.asprintf "%a" pp_bounded_view (n, view))
        "views whose bounds wrap reach outside their buffers"
        (List.map row
           [
             (max_int, [| 1 |], [| 2 |], 4);
             (0, [| max_int; max_int |], [| 2; 2 |], 4);
             (0, [| 1; 1 |], [| 1 lsl 32; 1 lsl 32 |], 1);
             (0, [| -1; -1 |], [| -2; -2 |], 7);
           ])
        (fun (n, view) ->
          equal bool false (view_inside view n);
          law (n, view));
      prop
        "Nx.of_buffer builds a value exactly when its shape has the buffer's \
         number of elements"
        shapes (fun (n, shape) ->
          let ok =
            (not (Array.exists (fun d -> d < 0) shape))
            && (not (count_overflows shape))
            && Array.fold_left ( * ) 1 shape = n
          in
          cover "a shape of the buffer's elements" ok;
          cover "a negative dimension" (Array.exists (fun d -> d < 0) shape);
          cover "an element count past max_int" (count_overflows shape);
          if ok then ignore (of_buffer n shape)
          else raises_match other_count (fun () -> of_buffer n shape));
      cases
        ~name:(fun (n, shape) ->
          Format.asprintf "shape %a, %d elements" Nx.pp_shape shape n)
        "Nx.of_buffer refuses a shape whose product wraps to the buffer's count"
        [ (0, [| 1 lsl 32; 1 lsl 32 |]); (4, [| -2; -2 |]); (1, [| -1; -1 |]) ]
        (fun (n, shape) ->
          raises_match other_count (fun () -> of_buffer n shape));
      test "Nx.Repr.Placed.v reads a storage's bytes as its dtype" (fun () ->
          let b = Nx_device.Buffer.create r1 Nx_dtype.Scalar.Float64 4 in
          let s = Nx.Repr.Storage.v p [ b ] in
          let x =
            Nx.Repr.Placed.v p Nx.float32 (Nx_array.View.create [| 8 |]) s
          in
          equal int 8 (Nx.numel x);
          raises_match (Exn.invalid_arg ~substring:"outside the storage")
            (fun () ->
              Nx.Repr.Placed.v p Nx.float32 (Nx_array.View.create [| 9 |]) s));
      test "Nx.Repr.Placed.v refuses a buffer not aligned to its dtype"
        (fun () ->
          let b = Nx_device.Buffer.create r1 Nx_dtype.Scalar.UInt8 9 in
          let s =
            Nx.Repr.Storage.v p
              [ Nx_device.Buffer.view b ~offset:1 Nx_dtype.Scalar.UInt8 8 ]
          in
          raises_match (Exn.invalid_arg ~substring:"not aligned") (fun () ->
              Nx.Repr.Placed.v p Nx.float32 (Nx_array.View.create [| 2 |]) s));
      test
        "Nx.Repr.Placed.v reads an odd int4 storage to its last element, and \
         its bytes to their last nibble as uint4" (fun () ->
          let s =
            Nx.Repr.Storage.v p
              [ Nx_device.Buffer.create r1 Nx_dtype.Scalar.Int4 5 ]
          in
          raises_match (Exn.invalid_arg ~substring:"outside the storage")
            (fun () ->
              Nx.Repr.Placed.v p Nx.int4 (Nx_array.View.create [| 6 |]) s);
          equal int 6
            (Nx.numel
               (Nx.Repr.Placed.v p Nx.uint4 (Nx_array.View.create [| 6 |]) s)));
      test
        "Nx.Repr.Placed.v reads bool only from bool storage, whose bytes are 0 \
         and 1" (fun () ->
          let s =
            Nx.Repr.Storage.v p
              [ Nx_device.Buffer.create r1 Nx_dtype.Scalar.UInt8 4 ]
          in
          raises_match (Exn.invalid_arg ~substring:"read as bool") (fun () ->
              Nx.Repr.Placed.v p Nx.bool (Nx_array.View.create [| 4 |]) s);
          let s =
            Nx.Repr.Storage.v p
              [ Nx_device.Buffer.create r1 Nx_dtype.Scalar.Bool 4 ]
          in
          equal int 4
            (Nx.numel
               (Nx.Repr.Placed.v p Nx.bool (Nx_array.View.create [| 4 |]) s)));
    ]

(* Values on the disk: files, which the host reads where they lie, in their
   pages, and other devices read into their memory. *)
let disk =
  let on_disk = Runtimes.on_disk in
  let disk = Nx.Placement.on (Nx.Device.make Nx_device.disk) in
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
          let on_d1 = Nx.place (Nx.Placement.on (Nx.Device.make r1)) x in
          let y = Nx.add on_d1 (on_disk x) in
          equal placement (Nx.Placement.on (Nx.Device.make r1)) (Nx.placement y);
          equal (array int32) (Nx.to_array (Nx.add x x)) (Nx.to_array y));
      test "placed on a device apart from the host, it is read into it"
        (fun () ->
          let y, bytes =
            reads (fun () ->
                Nx.place (Nx.Placement.on (Nx.Device.make r1)) (on_disk x))
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
       (devices :: host_buffers :: representation :: disk :: claims
       :: Runtimes.laws [ r1; r2 ]))
