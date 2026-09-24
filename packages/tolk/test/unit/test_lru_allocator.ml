(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk

(* An LRU allocator over one whose buffers are numbered in allocation order,
   which fails once when [fail] is set and records the buffers it frees. *)
let numbering () =
  let count = ref 0 and fail = ref false and freed = ref [] in
  let raw =
    {
      Device.Allocator.alloc =
        (fun _ _ ->
          if !fail then begin
            fail := false;
            failwith "out of memory"
          end;
          incr count;
          !count);
      free = (fun buf _ _ -> freed := buf :: !freed);
      copyin = (fun _ _ -> ());
      copyout = (fun _ _ -> ());
      addr = Nativeint.of_int;
      offset = None;
      transfer = None;
      supports_transfer = false;
      copy_from_disk = None;
      supports_copy_from_disk = false;
    }
  in
  (Device.Lru_allocator.wrap raw, fail, freed)

let spec = Device.Buffer_spec.default

(* [inside_allocation n f] runs [f] inside the [n]th allocation that
   [during] makes, where a GC finaliser could run, or after [during] if it
   makes fewer. *)
let inside_allocation n f ~during =
  let countdown = ref n in
  let run () =
    if !countdown > 0 then begin
      countdown := 0;
      f ()
    end
  in
  let tracker =
    {
      Gc.Memprof.null_tracker with
      alloc_minor =
        (fun _ ->
          if !countdown = 1 then run () else decr countdown;
          None);
    }
  in
  (match Gc.Memprof.start ~sampling_rate:1.0 ~callstack_size:0 tracker with
  | exception Failure reason -> skip ~reason ()
  | _ -> ());
  Fun.protect ~finally:Gc.Memprof.stop during;
  run ()

let () =
  run "Lru_allocator"
    [
      test "keeps a buffer freed while a cached one is taken" (fun () ->
        (* Taking the size-1 buffer scans past the others, allocating; each
           [n] frees a size-100 buffer inside a different allocation of that
           scan, as a buffer's finaliser can. *)
        for n = 1 to 8 do
          let lru, _, _ = numbering () in
          List.iter
            (fun size -> lru.free (lru.alloc size spec) size spec)
            [ 1; 2; 3; 4; 5; 6; 7; 8 ];
          let freed = lru.alloc 100 spec in
          inside_allocation n
            (fun () -> lru.free freed 100 spec)
            ~during:(fun () -> ignore (lru.alloc 1 spec : int));
          equal int freed (lru.alloc 100 spec)
        done);
      test "keeps a buffer freed while the cache is flushed" (fun () ->
        (* A failed allocation flushes the cache, freeing every entry; each
           [n] frees a size-100 buffer inside a different allocation of the
           failed search or of the flush. *)
        for n = 1 to 24 do
          let lru, fail, freed = numbering () in
          List.iter
            (fun size -> lru.free (lru.alloc size spec) size spec)
            [ 1; 2; 3; 4; 5; 6; 7; 8 ];
          let kept = lru.alloc 100 spec in
          fail := true;
          inside_allocation n
            (fun () -> lru.free kept 100 spec)
            ~during:(fun () -> ignore (lru.alloc 200 spec : int));
          is_true (List.mem kept !freed || lru.alloc 100 spec = kept)
        done);
    ]
