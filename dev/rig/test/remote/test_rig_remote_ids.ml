(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* An agent's answer to ids that name nothing of the job, driven through a
   controller's link of this process. *)

open Windtrap
open Remote_job
module Wire = Rig_remote_proxy.Wire
module Link = Rig_remote_proxy.Link

let state =
  Testable.structural ~pp:(fun ppf -> function
    | Link.Open -> Format.pp_print_string ppf "open"
    | Link.Closed -> Format.pp_print_string ppf "closed"
    | Link.Failed why -> Format.fprintf ppf "failed: %S" why)

(* A controller's link to [a], joined. *)
let joined a f =
  let j = Link.job () in
  let fd = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
  Unix.connect fd (Unix.ADDR_INET (Unix.inet_addr_loopback, a.port));
  require_ok (Wire.dial fd ~key ~self:Wire.Controller ~peer:(Wire.Agent 1));
  let l = Link.make j fd ~name:(machine a) ~peer:(Wire.Agent 1) in
  ignore (require_ok (Link.request l (Wire.Join { agents = [ address a ] })));
  Fun.protect
    ~finally:(fun () ->
      if Link.failure j = None then Link.fail j "the test ends")
    (fun () -> f j l)

(* A request naming an unknown id is refused and the job goes on; a drop naming
   one fails the job, which can refuse no frame without an answer. *)
let unknown_ids () =
  with_agents @@ fun agents ->
  let a = List.hd agents in
  joined a @@ fun j l ->
  (match Link.request l (Wire.Map { id = 5; device = 0; region = 999 }) with
  | Error (`Refused _) -> ()
  | Ok _ -> fail "a map of no memory was taken"
  | Error (`Failed why) -> failf "the job failed: %s" why);
  equal ~msg:"after the refusal" state Link.Open (Link.wait j ~ms:0);
  Link.drop l 999;
  match Link.wait j ~ms:5000 with
  | Link.Failed why -> contains ~sub:"malformed" why
  | s -> failf "after the drop: %a" (Testable.pp state) s

let () =
  Watchdog.start ();
  exit
    (run "rig_remote.ids"
       [
         group ~timeout:60. "rig_remote" [ test "ids of no object" unknown_ids ];
       ])
