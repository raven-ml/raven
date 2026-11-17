(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A separate process: OCaml forbids fork after a Domain has run. *)

open Windtrap
module C = Rig_cuda
module S = Rig_cuda_support

module B = Rig.Buffer
module H = Rig_gpu_support.Host
module Sub = Rig.Submission
module Run = Rig.Submission.Run

(* A submission of [n] launches of fixtures/launch.ptx's [ids], loaded on
   [d] for it, its run stored and released: launch [j] runs one thread,
   which stores [j + 1] into word [j] of the run's buffer. The submission
   keeps the image. *)
let[@inline never] stored d n =
  let image = require_ok (Rig.Image.load d (S.fixture "launch.ptx")) in
  let fn = Option.get (Rig.Image.resolve image "ids") in
  let refs = [| Sub.write 0 ~at:0 |] in
  let s = Sub.make d (Array.init n (fun _ -> Sub.launch ~refs fn ~params:24)) in
  let run = Sub.take s in
  for j = 0 to n - 1 do
    let b = Sub.block s j in
    Run.groups run b 1 1 1;
    Run.threads run b 1 1 1;
    Run.shared run b 0;
    Run.int64 run b 0 (Int64.of_int (4 * j));
    Run.int64 run b 8 (Int64.of_int (j + 1));
    Run.int32 run b 16 0;
    Run.float32 run b 20 0.
  done;
  Run.release run;
  s

(* The parent plays an image's 33 launches through its chains of 17 and 16
   nodes, then forks. The child, whose device is lost, drops the image and
   the submission, collects them and closes the device: an unload or a stop
   there would call CUDA through the parent's driver files, which may crash
   or hang the child or break the parent's chains; the alarm ends a hang.
   The parent's chains then play the launches again. *)
let inherited_chains () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  S.with_ @@ fun { d; g } ->
  let n = 33 in
  let s = ref (Some (stored d n)) in
  let out = B.create ~memory:Pinned d (4 * n) in
  let play msg =
    H.write (H.at out) (String.make (4 * n) '\000');
    let pieces, plain = C.replays g in
    let run = Sub.take (Option.get !s) in
    Rig.Point.wait (Rig.submit run ~buffers:[| out |]);
    let pieces', plain' = C.replays g in
    equal (pair int int)
      ~msg:(msg ^ ", pieces and plain launches")
      (2, 0)
      (pieces' - pieces, plain' - plain);
    equal (list int) ~msg:(msg ^ ", words")
      (List.init n (fun j -> j + 1))
      (List.init n (fun j -> H.get32 (H.at out + (4 * j))))
  in
  play "before the fork";
  Rig_support.needs_fork ();
  match Unix.fork () with
  | 0 ->
      ignore (Unix.alarm 5);
      s := None;
      Gc.full_major ();
      let m = Result.get_ok (Rig.software "fork:child-collection") in
      ignore (B.create m 0);
      let why = Rig.lost d in
      Rig.close d;
      Unix._exit (match why with Some "forked" -> 0 | _ -> 1)
  | pid ->
      (match Unix.waitpid [] pid with
      | _, WEXITED n -> equal int ~msg:"the child's exit" 0 n
      | _, (WSIGNALED n | WSTOPPED n) ->
          failf "the child stopped on signal %d" n);
      play "after the child's exit"

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
      let m = Result.get_ok (Rig.software "fork:child-memory") in
      let run = Rig.Submission.(take (make m [||])) in
      let p = Rig.submit run ~buffers:[||] in
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
    (run "rig_cuda.fork"
       [
         test "a child calls no CUDA for the chains it inherited"
           inherited_chains;
         test "a child's devices use rig's array of words" words_in_child;
       ])
