(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A separate process: OCaml forbids fork after a Domain has run. *)

open Windtrap
module C = Rig_cuda
module S = Rig_cuda_support

(* The parent collects one graph before the fork and the child the other.
   The child's graph call destroys the queued graphs before it answers, so a
   child that destroyed an inherited graph would call CUDA, which may crash
   or hang a child of fork; the alarm ends a hang. *)
let inherited_graphs () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  S.with_ @@ fun ({ g; _ } as t) ->
  let m, kernel = S.kernels g in
  let graph ks = (C.capability g).graph ks in
  let make _ = Some (require_ok (graph [| S.kernel (kernel "empty") 0 0 |])) in
  let graphs = Array.init 2 make in
  C.unload g m;
  S.close t;
  graphs.(0) <- None;
  Gc.full_major ();
  Rig_support.needs_fork ();
  match Unix.fork () with
  | 0 ->
      ignore (Unix.alarm 5);
      graphs.(1) <- None;
      Gc.full_major ();
      Unix._exit
        (match graph [||] with
        | Error "making the graph: the device was stopped" -> 0
        | Error _ | Ok _ -> 1)
  | pid ->
      (match Unix.waitpid [] pid with
      | _, WEXITED n -> equal int ~msg:"the child's exit" 0 n
      | _, (WSIGNALED n | WSTOPPED n) ->
          failf "the child stopped on signal %d" n);
      graphs.(1) <- None;
      Gc.full_major ();
      S.with_ (fun { g; _ } ->
          ignore (require_ok ((C.capability g).graph [||])))

(* rig's array of words, which an open registered with CUDA, stays the
   child's: a device the child opens takes a slot of it, and its work's
   value reaches the word. The CUDA device the child inherited is lost. *)
let words_in_child () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  S.with_ @@ fun { d; _ } ->
  Rig_support.needs_fork ();
  match Unix.fork () with
  | 0 ->
      ignore (Unix.alarm 5);
      let m = Result.get_ok (Rig.memory_device "graph_fork:child-memory") in
      let run = Rig.Submission.Run.make () in
      let p = Rig.submit (Rig.Submission.make m [||]) ~run ~buffers:[||] in
      Rig.Point.wait p;
      Unix._exit
        (match Rig.lost d with
        | Some "forked" when Rig.signaled m = Rig.Point.value p -> 0
        | _ -> 1)
  | pid -> (
      match Unix.waitpid [] pid with
      | _, WEXITED n -> equal int ~msg:"the child's exit" 0 n
      | _, (WSIGNALED n | WSTOPPED n) ->
          failf "the child stopped on signal %d" n)

let () =
  exit
    (run "rig_cuda.graph_fork"
       [
         test "a child destroys no graph inherited from its parent"
           inherited_graphs;
         test "a child's devices use rig's array of words" words_in_child;
       ])
