(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A child of fork opens GPU 0 and makes a round trip through its memory,
   whatever its parent did with NVIDIA's driver before the fork: nothing,
   read the GPU's usage, which makes the process's client, opened the GPU,
   which makes its unified memory too, or holds it open with memory it
   filled. The parent then makes its own round trip again, or reads its fill
   back and fills the memory again. The driver's state stays for the
   process, so each case runs in a fresh process, forked before the suite
   calls the driver. *)

open Windtrap
module P = Rig_nv_nvidia

let n = 1 lsl 20

(* Whether [b]'s bytes are all [c]. *)
let holds b c =
  let s = Bytes.make n '\000' in
  Rig.Buffer.blit_to_bytes b 0 s 0 n;
  if Bytes.for_all (Char.equal c) s then Ok ()
  else Error "the bytes read back differ from the fill"

let fill b c =
  Rig.Point.wait (Rig.Buffer.submit_fill b c);
  holds b c

(* [f b] on [n] bytes [b] of GPU 0's memory, from an open of its own. *)
let with_memory f =
  match Rig.open_ (P.gpu 0) with
  | Error why -> Error ("the open: " ^ why)
  | Ok d ->
      Fun.protect ~finally:(fun () -> Rig.close d) @@ fun () ->
      let b = Rig.Buffer.create d n in
      Fun.protect ~finally:(fun () -> ignore (Rig.Buffer.free b : bool))
      @@ fun () -> f b

(* A fill of [n] bytes of GPU 0's memory, read back: [Ok ()], or why not. *)
let round_trip () = with_memory (fun b -> fill b 'x')

let guarded f () = try f () with e -> Error ("raised " ^ Printexc.to_string e)

(* [f ()] in a forked child, its result sent back through a pipe, or why the
   child sent none. *)
let in_child (f : unit -> ('a, string) result) : ('a, string) result =
  let r, w = Unix.pipe ~cloexec:true () in
  match Unix.fork () with
  | 0 ->
      Unix.close r;
      let result = guarded f () in
      let oc = Unix.out_channel_of_descr w in
      Marshal.to_channel oc result [];
      close_out oc;
      Unix._exit 0
  | pid -> (
      Unix.close w;
      let ic = Unix.in_channel_of_descr r in
      let result =
        try Some (Marshal.from_channel ic) with End_of_file -> None
      in
      close_in ic;
      match (Unix.waitpid [] pid, result) with
      | (_, WEXITED 0), Some result -> result
      | (_, WEXITED k), _ -> Error (Printf.sprintf "the child exited %d" k)
      | (_, (WSIGNALED k | WSTOPPED k)), _ ->
          Error (Printf.sprintf "the child stopped on signal %d" k))

(* What a case's process saw: its own call before the fork, its child's round
   trip, and its own round trip after the child; or why it could not fork. *)
type seen =
  | Seen of {
      before : (unit, string) result;
      child : (unit, string) result;
      after : (unit, string) result;
    }
  | Unforkable of string

(* [before ()], then the child's round trip, then [after ()]. *)
let around before after =
  let before = guarded before () in
  match Rig_support.needs_fork () with
  | exception e -> Ok (Unforkable (Printexc.to_string e))
  | () ->
      let child = in_child round_trip in
      let after = guarded after () in
      Ok (Seen { before; child; after })

let case before () = around before round_trip

(* The parent holds GPU 0 open across the fork, with memory it filled: after
   the child's round trip, it reads its fill back and fills the memory
   again. *)
let held () =
  with_memory @@ fun b ->
  around
    (fun () -> fill b 'y')
    (fun () -> Result.bind (holds b 'y') (fun () -> fill b 'z'))

let usage () = Result.map ignore (P.usage ~window:0.1 0)

let forked_with case () =
  if Result.value (P.count ()) ~default:0 = 0 then
    skip ~reason:"the machine has no NVIDIA GPU" ();
  Rig_support.needs_fork ();
  match require_ok ~msg:"the case's process" (in_child case) with
  | Unforkable reason -> skip ~reason ()
  | Seen { before; child; after } ->
      let ok = result unit string in
      equal ok ~msg:"the parent's call before the fork" (Ok ()) before;
      equal ok ~msg:"the child's round trip" (Ok ()) child;
      equal ok ~msg:"the parent's round trip after the child" (Ok ()) after

let forked before = forked_with (case before)

let () =
  exit
    (run "rig_nv_nvidia fork"
       [
         group ~timeout:60. "fork"
           [
             test "a child of a parent that never called the driver opens GPU 0"
               (forked (fun () -> Ok ()));
             test
               "a child of a parent that read the GPU's usage opens GPU 0 with \
                its own client"
               (forked usage);
             test
               "a child of a parent that opened GPU 0 opens it with its own \
                client and unified memory, and the parent's stays"
               (forked round_trip);
             test
               "a child of a parent that holds GPU 0 open with memory it \
                filled opens it, and the parent's device and memory stay"
               (forked_with held);
           ];
       ])
