(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A program that uses Rig_remote_proxy.Link and never names the proxy's
   driver, which does not link that driver's module. A next from a Gc.Memprof
   callback in a next refuses it as when the driver is linked. Naming
   Rig_remote_proxy.Refused here would link the driver's module, so the
   refusal is told by its exception's name. *)

open Windtrap
module Link = Rig_remote_proxy.Link
module Wire = Rig_remote_proxy.Wire

let strf = Printf.sprintf

let refused e = String.ends_with ~suffix:".Refused" (Printexc.exn_slot_name e)

let dropped = function
  | Ok (Wire.Drop id) -> id
  | Ok _ -> fail "a command other than a drop"
  | Error why -> fail why

let with_pair f =
  let j = Link.job () in
  Fun.protect
    ~finally:(fun () ->
      match Link.wait j ~ms:0 with
      | Link.Open -> Link.fail j "the test ended"
      | _ -> ())
    (fun () ->
      let d, a = Link.socket_pair () in
      let c = Link.make j d ~name:"agent" ~peer:(Wire.Agent 1) in
      let a = Link.make j a ~name:"controller" ~peer:Wire.Controller in
      f c a)

(* [walk attempt] runs [attempt n] from [n = 1] until an attempt no longer
   reaches its [n]-th allocation. *)
let walk attempt =
  let rec from n = if attempt n then from (n + 1) in
  from 1

let next_in_next () =
  let refusals = ref 0 in
  walk (fun n ->
      with_pair @@ fun c a ->
      Link.drop c 3;
      Link.drop c 4;
      let outer = ref None and inner = ref None in
      let call () =
        match Link.next a with
        | r -> inner := Some (dropped r)
        | exception e when refused e -> incr refusals
      in
      let reached =
        Rig_support.interrupted ~landing:(Call call) n (fun () ->
            outer := Some (dropped (Link.next a)))
      in
      let inner =
        match !inner with Some id -> id | None -> dropped (Link.next a)
      in
      equal ~msg:(strf "from allocation %d, the commands taken" n) (list int)
        [ 3; 4 ]
        (List.sort compare [ Option.get !outer; inner ]);
      reached);
  greater ~msg:"nexts refused" int ~than:0 !refusals

let () =
  exit
    (run "rig_remote_proxy link alone"
       [
         group ~timeout:120. "without the driver's module"
           [
             test "a next in a next takes a command, or is refused"
               next_in_next;
           ];
       ])
