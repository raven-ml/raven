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

let independent_replays_keep_their_intermediates () =
  let name = "CPU:751" in
  let base = Tolk.Device.get name in
  let renderer = Tolk.Device.renderer base in
  let pending = ref None and calls = ref 0 in
  let runtime object_ =
    let program = Tolk.Device.runtime base object_ in
    let call buffers ~global ~local ~vals ~wait ~timeout =
      let elapsed = program.call buffers ~global ~local ~vals ~wait ~timeout in
      incr calls;
      (match !pending with
      | None -> ()
      | Some replay ->
          pending := None;
          replay ());
      elapsed
    in
    { program with call }
  in
  let allocator =
    Tolk_uop.Storage.Host_allocator.make ~synchronize:(fun () -> ())
  in
  ignore
    (Tolk.Device.make ~name ~allocator:(Tolk.Device.Allocator.Pack allocator)
       ~renderer_set:
         (Tolk.Device.Renderer_set.make ~device:name
            [ ("CLANG", fun _ -> renderer) ])
       ~runtime
       ~synchronize:(fun _ -> ())
       ());
  let program activation x =
    let a = activation (Nx.matmul x (Nx.transpose x)) in
    Nx.sum ~axes:[ 1 ] (Nx.matmul a a)
  in
  let f = program Nx.tanh and g = program Nx.sin in
  let devices = [ Rune.device name ] in
  let outer = Rune.jit' ~devices f and inner = Rune.jit' ~devices g in
  let input k =
    Nx.init Nx.float32 [| 16; 8 |] (fun index ->
        sin (float_of_int ((index.(0) * k) + index.(1))) /. 4.)
  in
  let x = input 3 and y = input 7 in
  ignore (Nx.to_array (outer x));
  ignore (Nx.to_array (inner y));
  calls := 0;
  ignore (Nx.to_array (outer x));
  is_true ~msg:"the outer computation has a live intermediate between kernels"
    (!calls >= 2);
  let nested = ref None in
  pending := Some (fun () -> nested := Some (Nx.to_array (inner y)));
  let actual = Nx.to_array (outer x) in
  equal ~msg:"the interrupted outer replay keeps its own intermediate"
    (array (float 1e-4))
    (Nx.to_array (f x))
    actual;
  equal ~msg:"the nested replay computes its own result"
    (array (float 1e-4))
    (Nx.to_array (g y))
    (Option.get !nested)

let () =
  run "rune transfer scratch"
    [
      test "independent uploads keep their bytes"
        independent_uploads_keep_their_bytes;
      test "concurrent device lookups keep one identity"
        concurrent_device_lookups_keep_one_identity;
      test "independent replays keep their intermediates"
        independent_replays_keep_their_intermediates;
    ]
