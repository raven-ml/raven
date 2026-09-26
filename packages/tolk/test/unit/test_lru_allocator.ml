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
      addr = Some Nativeint.of_int;
      host = (fun _ -> None);
      kind = Type.Id.make ();
      mapping = None;
      synchronize = (fun () -> ());
      offset = None;
    }
  in
  (Device.Lru_allocator.wrap raw, fail, freed)

let spec = Device.Buffer_spec.default

let failing_numbering () =
  let count = ref 0 and fail_alloc = ref false and freed = ref [] in
  let during_free = ref (fun () -> ()) in
  let failed_owner = Stdlib.Weak.create 1 in
  let cleanup_error = Failure "cached buffer retirement failed" in
  let raw =
    {
      Device.Allocator.alloc =
        (fun _ _ ->
          if !fail_alloc then begin
            fail_alloc := false;
            failwith "out of memory"
          end;
          incr count;
          let buf = ref !count in
          if !count = 2 then Stdlib.Weak.set failed_owner 0 (Some buf);
          buf);
      free =
        (fun buf _ _ ->
          freed := !buf :: !freed;
          if !buf = 3 then !during_free ();
          if !buf = 2 then raise cleanup_error);
      addr = Some (fun buf -> Nativeint.of_int !buf);
      host = (fun _ -> None);
      kind = Type.Id.make ();
      mapping = None;
      synchronize = (fun () -> ());
      offset = None;
    }
  in
  let lru = Device.Lru_allocator.wrap raw in
  List.iter (fun size -> lru.free (lru.alloc size spec) size spec) [ 1; 2; 3 ];
  (lru, fail_alloc, freed, during_free, failed_owner, cleanup_error)

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
  exit (run "Lru_allocator"
    [
      test "failed flush restores untouched entries beside concurrent frees" (fun () ->
          let lru, fail_alloc, freed, during_free, failed_owner, error =
            failing_numbering ()
          in
          during_free := (fun () -> lru.free (lru.alloc 1 spec) 1 spec);
          fail_alloc := true;
          raises_match (fun exn -> exn == error)
            (fun () -> lru.alloc 200 spec);
          during_free := (fun () -> ());
          equal (list int) [ 3; 2 ] (List.rev !freed);
          (* The newly freed entry remains newer than the untouched one. *)
          equal int 4 !(lru.alloc 1 spec);
          equal int 1 !(lru.alloc 1 spec);
          equal int 5 !(lru.alloc 2 spec);
          Gc.full_major ();
          Gc.full_major ();
          is_true (Stdlib.Weak.check failed_owner 0);
          equal (list int) [ 3; 2 ] (List.rev !freed));
      test "failed raw retirement outlives its allocator without retry" (fun () ->
          let wrapper = Stdlib.Weak.create 1 in
          let abandon () =
            let lru, fail_alloc, freed, _, failed_owner, error =
              failing_numbering ()
            in
            Stdlib.Weak.set wrapper 0 (Some lru);
            fail_alloc := true;
            raises_match (fun exn -> exn == error)
              (fun () -> lru.alloc 200 spec);
            (failed_owner, freed)
          in
          let failed_owner, freed = abandon () in
          Gc.full_major ();
          Gc.full_major ();
          is_false ~msg:"the wrapper itself is not the quarantine owner"
            (Stdlib.Weak.check wrapper 0);
          is_true ~msg:"uncertain raw backing stays owned after wrapper collection"
            (Stdlib.Weak.check failed_owner 0);
          Tolk_uop.Storage.with_operation (fun () -> ());
          equal (list int) [ 3; 2 ] (List.rev !freed));
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
    ])
