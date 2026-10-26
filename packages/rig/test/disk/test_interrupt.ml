(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Interrupt walks of the disk. A copy, a barrier or a create runs once for
   each allocation it makes, in a forked child, with every allocation from
   that one on raising until the call returns, as an asynchronous exception
   does. After a copy the disk keeps its bound of descriptors, every file
   reads its bytes, and once the files are collected no descriptor of theirs
   stays open. A create that raised leaves no file at its path. *)

open Windtrap
module B = Rig.Buffer
module Support = Rig_disk_support

let strf = Printf.sprintf
let timeout = 60.

(* The disk keeps at most this many descriptors open while no copy uses
   them. *)
let max_descriptors = 64

(* Files *)

(* This executable's directory, under _build, holds the files, which the
   suite removes as it ends. *)
let exe =
  let e = Sys.executable_name in
  if Filename.is_relative e then Filename.concat (Sys.getcwd ()) e else e

let dir = Filename.concat (Filename.dirname exe) "interrupt-files"

let clear () =
  if Sys.file_exists dir then
    Array.iter (fun f -> Sys.remove (Filename.concat dir f)) (Sys.readdir dir)
  else Sys.mkdir dir 0o755

let path name = Filename.concat dir name

(* One file more than the disk keeps open, each holding its index. *)
let count = max_descriptors + 1
let bytes i = strf "file %d" i

let paths =
  lazy
    (List.init count (fun i ->
         let path = Filename.concat dir (strf "f%d" i) in
         Out_channel.with_open_bin path (fun oc -> output_string oc (bytes i));
         path))

let read b =
  let h = B.create Rig.host (B.length b) in
  B.copy ~src:b ~dst:h;
  B.read h Bigarray.char (fun a ->
      String.init (Bigarray.Array1.dim a) (Bigarray.Array1.get a))

(* Collects the dropped buffers, then drains the disk, which closes the
   descriptors of collected files. *)
let collect () =
  for _ = 1 to 10 do
    Gc.full_major ()
  done;
  ignore (B.create Rig_disk.device 0)

let descriptors () = Option.get (Support.descriptors ())

(* In a child *)

(* [in_child what f] is [f ()], run in a forked child, so that what one
   attempt leaves never changes the next one. It fails with the child's
   failure. *)
let in_child what f =
  let r, w = Unix.pipe () in
  match Unix.fork () with
  | 0 ->
      Unix.close r;
      let answer =
        match f () with v -> Ok v | exception e -> Error (Printexc.to_string e)
      in
      let oc = Unix.out_channel_of_descr w in
      Marshal.to_channel oc (answer : ('a, string) result) [];
      close_out oc;
      Unix._exit 0
  | pid -> (
      Unix.close w;
      let ic = Unix.in_channel_of_descr r in
      let answer = In_channel.input_all ic in
      close_in ic;
      match (answer, snd (Unix.waitpid [] pid)) with
      | "", (WEXITED n | WSIGNALED n | WSTOPPED n) ->
          failf "the child ended (%d) without answering%s" n what
      | answer, _ -> (
          match (Marshal.from_string answer 0 : ('a, string) result) with
          | Ok v -> v
          | Error e -> failf "%s%s" e what))

(* Walks *)

(* Opens and reads every file, which leaves the first one's descriptor closed
   for the others, then reads the first file interrupted from its [n]-th
   allocation, where [landing] lands: the read reopens its descriptor and
   closes the least recently used other one, and leaves no lock held. Is
   whether the allocation was reached. *)
let interrupt ?landing n =
  let at = strf ", interrupted from allocation %d" n in
  in_child at (fun () ->
      let base = descriptors () in
      let files = List.map Rig_disk.of_file (Lazy.force paths) in
      let files = List.map Result.get_ok files in
      List.iter (fun b -> ignore (read b)) files;
      let full = descriptors () in
      equal ~msg:"descriptors kept" int (base + max_descriptors) full;
      let first = List.hd files in
      let h = B.create Rig.host (B.length first) in
      let reached =
        Rig_support.interrupted ?landing n (fun () -> B.copy ~src:first ~dst:h)
      in
      equal ~msg:"locks held after" int 0 (Rig_support.holds ());
      List.iteri
        (fun i b -> equal ~msg:(strf "file %d" i) string (bytes i) (read b))
        files;
      equal ~msg:"descriptors kept after" int full (descriptors ());
      ignore (Sys.opaque_identity files);
      collect ();
      equal ~msg:"descriptors once collected" int base (descriptors ());
      reached)

let test_interrupt landing () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  Rig_support.needs_fork ();
  ignore (Lazy.force paths);
  let rec from n = if interrupt ~landing n then from (n + 1) in
  from 1

(* Writes a file, reads another, then syncs the first interrupted from the
   barrier's [n]-th allocation: the barrier holds the file's sync lock and,
   to make the file the most recently used again, the table's lock inside it.
   No lock stays held, a later barrier returns, and the file holds the bytes
   written. Is whether the allocation was reached. *)
let interrupt_barrier n =
  let at = strf ", barrier interrupted from allocation %d" n in
  in_child at (fun () ->
      let name = path "barrier" in
      if Sys.file_exists name then Sys.remove name;
      let b = Result.get_ok (Rig_disk.create_file name 8) in
      let other =
        Result.get_ok (Rig_disk.of_file (List.hd (Lazy.force paths)))
      in
      B.copy ~src:(B.of_string "barriers") ~dst:b;
      ignore (read other);
      let reached = Rig_support.interrupted n (fun () -> Rig_disk.barrier b) in
      equal ~msg:"locks held after" int 0 (Rig_support.holds ());
      Rig_disk.barrier b;
      equal ~msg:"the file's bytes" string "barriers"
        (In_channel.with_open_bin name In_channel.input_all);
      reached)

let test_interrupt_barrier () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  Rig_support.needs_fork ();
  ignore (Lazy.force paths);
  let rec from n = if interrupt_barrier n then from (n + 1) else n - 1 in
  greater ~msg:"allocations interrupted" int ~than:0 (from 1)

(* Creates *)

(* What a create answers until it returns. *)
let not_returned : (B.t, string) result = Error "not returned"

(* [create_cut n] creates a file, interrupted from its [n]-th allocation. A
   create that raised leaves no file at its path, so a create of the path
   after it answers. Is whether the allocation was reached, and what went
   wrong, if anything. *)
let create_cut n =
  in_child (strf ", interrupted from allocation %d" n) (fun () ->
      let name = path "created" in
      if Sys.file_exists name then Sys.remove name;
      let made = ref not_returned in
      let reached =
        Rig_support.interrupted n (fun () ->
            made := Rig_disk.create_file name 8)
      in
      equal ~msg:"locks held after" int 0 (Rig_support.holds ());
      let wrong =
        match !made with
        | Error why when !made != not_returned -> Some ("answered " ^ why)
        | Ok _ -> None
        | Error _ when Sys.file_exists name -> Some "left a file at its path"
        | Error _ -> (
            match Rig_disk.create_file name 8 with
            | Ok _ -> None
            | Error why -> Some ("a create after it answered " ^ why))
      in
      (reached, wrong))

(* Walks [create_cut] from the first allocation to the last, and fails naming
   every allocation whose interruption went wrong. *)
let test_create () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  Rig_support.needs_fork ();
  let rec from n wrongs =
    match create_cut n with
    | false, _ -> List.rev wrongs
    | true, None -> from (n + 1) wrongs
    | true, Some why -> from (n + 1) ((n, why) :: wrongs)
  in
  equal ~msg:"interruptions that went wrong"
    (list (pair int string))
    [] (from 1 [])

(* Calls from a callback *)

(* What a call answers: ["refused"] for an [Invalid_argument] that names the
   call's caller, rig or the disk. *)
let answer (what, f) =
  match f () with
  | () -> (what, "done")
  | exception Invalid_argument why
    when String.starts_with ~prefix:"Rig." why
         || String.starts_with ~prefix:"Rig_disk." why ->
      (what, "refused")
  | exception e -> (what, Printexc.to_string e)

(* [called n] opens a file beyond the disk's bound of descriptors, which
   holds the disk's lock while it opens the descriptor and closes the least
   recently used other one, with calls of the disk from a Gc.Memprof
   callback at the open's [n]-th allocation: a read of the third file, open
   until the open closes it, a write, a barrier and an open of an open file,
   and a drain that frees a collected file whose descriptor is open. Inside a
   lock of the thread's each raises [Invalid_argument], having changed
   nothing; outside every lock each answers. Is whether the allocation was
   reached, and whether the callback ran inside a lock. *)
let called n =
  let at = strf ", called from allocation %d" n in
  in_child at (fun () ->
      let base = descriptors () in
      let files = List.map Rig_disk.of_file (Lazy.force paths) in
      let files = List.map Result.get_ok files in
      if Sys.file_exists (path "written") then Sys.remove (path "written");
      let written = Result.get_ok (Rig_disk.create_file (path "written") 8) in
      List.iter (fun b -> ignore (read b)) files;
      B.copy ~src:(B.of_string "written!") ~dst:written;
      let dropped = ref (Rig_disk.of_file (path "dropped")) in
      let third = List.nth files 2 in
      let rewritten = ref false and held = ref false and answers = ref [] in
      let calls =
        [
          ( "a read",
            fun () -> equal ~msg:"the third file" string (bytes 2) (read third)
          );
          ( "a write",
            fun () ->
              B.copy ~src:(B.of_string "callback") ~dst:written;
              rewritten := true );
          ("a barrier", fun () -> Rig_disk.barrier written);
          ( "an open",
            fun () -> ignore (Result.get_ok (Rig_disk.of_file (path "dropped")))
          );
          ( "a drain",
            fun () ->
              Gc.full_major ();
              ignore (B.create Rig_disk.device 0) );
        ]
      in
      let call () =
        held := Rig_support.holds () > 0;
        answers := List.map answer calls
      in
      let opened = ref None in
      let run () =
        opened := Some (Result.get_ok (Rig_disk.of_file (path "opened")))
      in
      dropped := Error "collected";
      let reached = Rig_support.interrupted ~landing:(Call call) n run in
      if reached then begin
        let expected (what, _) =
          (what, if !held then "refused" else "done")
        in
        equal ~msg:"the calls' answers"
          (list (pair string string))
          (List.map expected calls) !answers;
        equal ~msg:"the callback's write" bool (not !held) !rewritten
      end;
      at_most ~msg:"descriptors after" int ~than:(base + max_descriptors)
        (descriptors ());
      List.iteri
        (fun i b -> equal ~msg:(strf "file %d" i) string (bytes i) (read b))
        files;
      Option.iter
        (fun b -> equal ~msg:"the opened file" string "opened" (read b))
        !opened;
      Rig_disk.barrier written;
      equal ~msg:"the written file" string
        (if !rewritten then "callback" else "written!")
        (read written);
      ignore (Sys.opaque_identity (files, written, !opened));
      opened := None;
      collect ();
      equal ~msg:"descriptors once collected" int base (descriptors ());
      (reached, !held))

let test_called () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  Rig_support.needs_fork ();
  ignore (Lazy.force paths);
  List.iter
    (fun name ->
      Out_channel.with_open_bin (path name) (fun oc -> output_string oc name))
    [ "dropped"; "opened" ];
  (* Each attempt runs in a child of fork, so the landings are counted here,
     from what the child answers, and not by [Rig_support.seen]. *)
  let rec from n inside outside =
    match called n with
    | true, true -> from (n + 1) (inside + 1) outside
    | true, false -> from (n + 1) inside (outside + 1)
    | false, _ -> (inside, outside)
  in
  let inside, outside = from 1 0 0 in
  greater ~msg:"landings inside a lock" int ~than:0 inside;
  greater ~msg:"landings outside every lock" int ~than:0 outside

let () =
  clear ();
  let code =
    run "rig_disk.interrupt"
      [
        group ~timeout "an exception at any allocation"
          [
            test
              "a read that reopens a descriptor and closes another keeps the \
               bound and leaks no descriptor"
              (test_interrupt Raise);
            test
              "a read interrupted by a signal at any allocation leaks no \
               descriptor"
              (test_interrupt Signal);
            test
              "a barrier under its file's lock and the table's leaves no lock \
               held"
              test_interrupt_barrier;
            test "a create that raises leaves no file at its path" test_create;
          ];
        group ~timeout "a call from a callback at any allocation"
          [
            test
              "a read, a write, a barrier, an open and a free in an open \
               raise Invalid_argument inside a lock and answer outside"
              test_called;
          ];
      ]
  in
  clear ();
  Sys.rmdir dir;
  exit code
