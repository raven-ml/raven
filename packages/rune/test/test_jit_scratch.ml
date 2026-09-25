(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

let independent_uploads_keep_their_bytes () =
  let name = "CPU:731" in
  let pending = ref None in
  let synchronize () =
    match !pending with
    | None -> ()
    | Some upload ->
        pending := None;
        upload ()
  in
  let renderer = Tolk.Device.renderer (Tolk.Device.get "CPU") in
  let allocator = Tolk_uop.Storage.Host_allocator.make ~synchronize in
  let dev =
    Tolk.Device.make ~name ~allocator:(Tolk.Device.Allocator.Pack allocator)
      ~renderer_set:
        (Tolk.Device.Renderer_set.make ~device:name
           [ ("CLANG", fun _ -> renderer) ])
      ~synchronize:(fun _ -> synchronize ())
      ()
  in
  let placement = Nx.Placement.device (Rune.device name) in
  (* Each upload reaches the full 64 MiB transfer chunk. The second upload runs
     after the first has filled its staging bytes, before its native copy
     starts. This deterministically exercises overlapping caller lifetimes
     without claiming all of Rune is safe for concurrent replay. *)
  let size = 64 * 1024 * 1024 in
  let first = Nx.full Nx.uint8 [| size |] 17 in
  let second = Nx.full Nx.uint8 [| size |] 93 in
  let nested = ref None in
  pending := Some (fun () -> nested := Some (Nx.place placement second));
  let outer = Nx.place placement first in
  let inner = Option.get !nested in
  let check value tensor =
    let bytes = Nx.to_buffer tensor in
    List.iter
      (fun index ->
        equal
          ~msg:(Printf.sprintf "byte %d" index)
          int value
          (Nx_buffer.get bytes index))
      [ 0; size / 2; size - 1 ]
  in
  check 17 outer;
  check 93 inner;
  Tolk.Device.synchronize dev

let concurrent_device_lookups_keep_one_identity () =
  let renderer = Tolk.Device.renderer (Tolk.Device.get "CPU") in
  let names = Array.init 8 (fun i -> Printf.sprintf "CPU:%d" (740 + i)) in
  Array.iter
    (fun name ->
      let allocator =
        Tolk_uop.Storage.Host_allocator.make ~synchronize:(fun () -> ())
      in
      ignore
        (Tolk.Device.make ~name
           ~allocator:(Tolk.Device.Allocator.Pack allocator)
           ~renderer_set:
             (Tolk.Device.Renderer_set.make ~device:name
                [ ("CLANG", fun _ -> renderer) ])
           ~synchronize:(fun _ -> ())
           ()))
    names;
  let start = Atomic.make false in
  let workers =
    Array.init 4 (fun _ ->
        Domain.spawn (fun () ->
            while not (Atomic.get start) do
              Domain.cpu_relax ()
            done;
            Array.init 20 (fun _ ->
                Array.map
                  (fun name -> Rune.device (String.lowercase_ascii name))
                  names)))
  in
  Atomic.set start true;
  let results = Array.map Domain.join workers in
  Array.iter
    (Array.iter
       (Array.iteri (fun i device ->
            is_true
              ~msg:("canonical identity for " ^ names.(i))
              (device == Rune.device names.(i)))))
    results

let () =
  run "rune transfer scratch"
    [
      test "independent uploads keep their bytes"
        independent_uploads_keep_their_bytes;
      test "concurrent device lookups keep one identity"
        concurrent_device_lookups_keep_one_identity;
    ]
