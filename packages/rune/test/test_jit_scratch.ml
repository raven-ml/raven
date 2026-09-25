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

let concurrent_transfers_preserve_accounting () =
  let name = "CPU:752" in
  let renderer = Tolk.Device.renderer (Tolk.Device.get "CPU") in
  let allocator = Tolk_uop.Storage.Host_allocator.make ~synchronize:(fun () -> ()) in
  ignore (Tolk.Device.make ~name ~allocator:(Tolk.Device.Allocator.Pack allocator)
      ~renderer_set:(Tolk.Device.Renderer_set.make ~device:name ["CLANG", Fun.const renderer])
      ~synchronize:(fun timeout -> ignore timeout) ());
  let placement = Nx.Placement.device (Rune.device name) in
  for _ = 1 to 3 do Gc.full_major () done;
  let before = Rune.jit_stats () in
  let ready = Atomic.make 0 in
  let workers = Array.init 4 (fun worker -> Domain.spawn (fun () ->
      let source = Nx.full Nx.uint8 [|128|] worker in
      ignore (Atomic.fetch_and_add ready 1);
      while Atomic.get ready <> 4 do Domain.cpu_relax () done;
      Array.init 1000 (fun _ ->
          let placed = Nx.place placement source in
          equal (array int) (Array.make 128 worker) (Nx.to_array placed);
          placed))) in
  let placed = Array.map Domain.join workers in
  let after = Rune.jit_stats () in
  let bytes = 4 * 1000 * 128 in
  equal ~msg:"every upload is counted" int bytes
    (after.bytes_to_device - before.bytes_to_device);
  equal ~msg:"every read is counted" int bytes
    (after.bytes_from_device - before.bytes_from_device);
  equal ~msg:"every retained placement is counted" int bytes
    (after.resident_bytes - before.resident_bytes);
  ignore (Sys.opaque_identity placed)

(* Effect fallbacks retain their operands in the last exception backtrace. *)
let[@inline never] raise_collection_marker () = raise Exit

let reads_keep_their_resident_owner_alive () =
  let name = "CPU:753" in
  let pending = ref None and frees = ref 0 in
  let base =
    Tolk_uop.Storage.Host_allocator.make ~synchronize:(fun () -> ())
  in
  let allocator =
    {
      base with
      host = (fun raw ->
        Option.iter (fun callback -> callback ()) !pending;
        base.host raw);
      free = (fun raw size spec ->
        incr frees;
        base.free raw size spec);
    }
  in
  let renderer = Tolk.Device.renderer (Tolk.Device.get "CPU") in
  ignore
    (Tolk.Device.make ~name ~allocator:(Tolk.Device.Allocator.Pack allocator)
       ~renderer_set:
         (Tolk.Device.Renderer_set.make ~device:name
            [ ("CLANG", fun _ -> renderer) ])
       ~synchronize:(fun _ -> ()) ());
  let placement = Nx.Placement.device (Rune.device name) in
  let[@inline never] read_temporary () =
    let placed = Nx.place placement (Nx.full Nx.uint8 [| 32 |] 91) in
    (* [to_buffer] first checks the storage length, then tail-calls the final
       read. Collect during that final host mapping, after its cell is unpacked. *)
    let mappings = ref 0 in
    pending := Some (fun () ->
        incr mappings;
        if !mappings = 2 then begin
          pending := None;
          (try raise_collection_marker () with Exit -> ());
          Gc.full_major ();
          ignore (Rune.jit_stats ());
          equal ~msg:"a read owns its storage until the copy completes" int 0
            !frees
        end);
    Nx.to_buffer placed
  in
  let actual = read_temporary () in
  equal (array int) (Array.make 32 91)
    (Array.init (Nx_buffer.length actual) (Nx_buffer.get actual));
  is_true ~msg:"the final read reached the collection hook"
    (Option.is_none !pending);
  (try raise_collection_marker () with Exit -> ());
  for _ = 1 to 3 do Gc.full_major () done;
  ignore (Rune.jit_stats ());
  equal ~msg:"the temporary storage is eventually released" int 1 !frees

let failed_release_preserves_all_owners () =
  let name = "CPU:754" in
  let base =
    Tolk_uop.Storage.Host_allocator.make ~synchronize:(fun () -> ())
  in
  let allocations = ref [] and calls = ref [] and failed = ref None in
  let error = Failure "resident release failed" in
  let allocator =
    {
      base with
      alloc = (fun size spec ->
          let raw = base.alloc size spec in
          let weak = Weak.create 1 in
          Weak.set weak 0 (Some raw);
          allocations := (Nativeint.to_int raw, weak) :: !allocations;
          raw);
      free = (fun raw size spec ->
          let key = Nativeint.to_int raw in
          calls := key :: !calls;
          match !failed with
          | None ->
              failed := Some key;
              raise error
          | Some _ -> base.free raw size spec);
    }
  in
  let renderer = Tolk.Device.renderer (Tolk.Device.get "CPU") in
  ignore
    (Tolk.Device.make ~name ~allocator:(Tolk.Device.Allocator.Pack allocator)
       ~renderer_set:
         (Tolk.Device.Renderer_set.make ~device:name
            [ ("CLANG", fun _ -> renderer) ])
       ~synchronize:(fun _ -> ()) ());
  let placement = Nx.Placement.device (Rune.device name) in
  let[@inline never] drop_values () =
    let first = Nx.place placement (Nx.full Nx.uint8 [| 32 |] 17) in
    let second = Nx.place placement (Nx.full Nx.uint8 [| 32 |] 93) in
    ignore (Sys.opaque_identity (first, second))
  in
  drop_values ();
  (try raise_collection_marker () with Exit -> ());
  Gc.full_major ();
  (match Rune.jit_stats () with
  | _ -> fail "the allocator failure must propagate"
  | exception actual ->
      is_true ~msg:"the original exception is preserved" (actual == error));
  equal ~msg:"the failing release stops its batch" int 1 (List.length !calls);
  ignore (Rune.jit_stats ());
  (try raise_collection_marker () with Exit -> ());
  for _ = 1 to 3 do
    Gc.full_major ();
    ignore (Rune.jit_stats ())
  done;
  let failed = Option.get !failed in
  equal ~msg:"an uncertain release is never retried" int 1
    (List.length (List.filter (( = ) failed) !calls));
  equal ~msg:"the untouched store is still released exactly once" int 2
    (List.length !calls);
  is_true ~msg:"the failed raw owner remains retained after collection"
    (Weak.check (List.assoc failed !allocations) 0)

let () =
  run "rune transfer scratch"
    [
      test "reads keep their resident owner alive"
        reads_keep_their_resident_owner_alive;
      test "failed release preserves all owners"
        failed_release_preserves_all_owners;
      test "concurrent transfers preserve accounting" concurrent_transfers_preserve_accounting;
      test "independent uploads keep their bytes"
        independent_uploads_keep_their_bytes;
      test "concurrent device lookups keep one identity"
        concurrent_device_lookups_keep_one_identity;
      test "independent replays keep their intermediates"
        independent_replays_keep_their_intermediates;
    ]
