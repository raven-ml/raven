(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Operations on a Polled device, which the failure and interrupt walks run
   through failures. *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module S = Rig_support

let why = "the walk's failure"

(* How a walk fails an operation: from its [n]-th fallible call on, every call
   faults; its [n]-th call refuses; from its [n]-th call on, every call refuses,
   as a device out of memory does; or a function it gave rig raises [Exit]. *)
type failure = Fault of int | Refuse of int | Exhaust of int | Raise

(* An operation on a device: [prepare d] makes what it needs, and is the
   operation, told the failure it meets. It checks the values it gets, and
   raises on one rig.mli does not allow. [raises] says whether it gives rig a
   function, which raises under [Raise]. *)
type op = {
  name : string;
  opener : string -> Rig.t * P.t;
  prepare : Rig.t -> failure option -> unit;
  raises : bool;
}

let fresh =
  let k = ref 0 in
  fun what ->
    incr k;
    Printf.sprintf "walk:%s:%d" what !k

let page = 1 lsl 16
let pattern n = String.init n (fun i -> Char.chr (i land 0xff))

let collect () =
  Gc.full_major ();
  Gc.full_major ()

let bytes b =
  B.read b Bigarray.char (fun ba ->
      String.init (Bigarray.Array1.dim ba) (Bigarray.Array1.get ba))

let filled n =
  let b = B.create Rig.host n in
  B.write b Bigarray.char (fun a ->
      String.iteri (Bigarray.Array1.set a) (pattern n));
  b

let op ?(opener = fun name -> P.open_ name) ?(raises = false) name prepare =
  { name; opener; prepare; raises }

(* A refusal may reach a call that answers [None] or [Error]. *)
let refusing = function Some (Refuse _ | Exhaust _) -> true | _ -> false

let create memory =
  op
    (match memory with
    | B.Device -> "create"
    | Pinned -> "create pinned"
    | Mapped -> "create mapped")
    (fun d _ -> equal int 64 (B.length (B.create ~memory d 64)))

let borrow_of ?opener name make =
  op ?opener name (fun d ->
      let b = make () in
      fun failure ->
        match B.borrow d b with
        | Some b' -> equal int (B.length b) (B.length b')
        | None -> equal ~msg:"borrow refused" bool true (refusing failure))

(* The device whose memory [borrow peer] borrows, one for every walk of a
   process while it is not lost, opened with the first device that borrows
   from it: a forked child opens its own. *)
let peers = ref None

let peer () =
  let pid = Unix.getpid () in
  match !peers with
  | Some (pid', d) when pid' = pid && Rig.lost d = None -> d
  | _ ->
      let d = fst (P.open_ (fresh (Printf.sprintf "peer:%d" pid))) in
      peers := Some (pid, d);
      d

let borrow_host = borrow_of "borrow host" (fun () -> B.create Rig.host page)

let borrow_peer =
  let opener name =
    ignore (peer ());
    P.open_ name
  in
  borrow_of ~opener "borrow peer" (fun () -> B.create (peer ()) 64)

let bump =
  Sub.fill ~refs:[| Sub.write 0 ~at:0 |] S.bump ~params:8

let once ?(buffers = [||]) run = Rig.submit run ~buffers

let submit =
  op "submit" (fun d ->
      let arg = B.create d 8 in
      B.write arg Bigarray.char (fun a -> Bigarray.Array1.fill a '\000');
      let s = Sub.make d [| bump |] in
      let run = Sub.Run.make s and buffers = [| arg |] in
      Sub.Run.int64 run (Sub.block s 0) 0 0;
      fun _ ->
        Rig.Point.wait (once ~buffers run);
        equal string "\001\000\000\000\000\000\000\000" (bytes arg))

(* A device whose memory the host does not address, which copied through the
   staging memory both ways once, its mappings of host memory refused: what it
   made for those copies lasts as long as the staging memory, for good. *)
let staging name =
  let d, p = P.open_ ~host_visible:false name in
  let on_d = B.create d page in
  P.fail_at p 1 (`Refuse 1);
  B.copy ~src:(B.create Rig.host page) ~dst:on_d;
  P.fail_at p 1 (`Refuse 1);
  B.copy ~src:on_d ~dst:(B.create Rig.host page);
  (d, p)

(* Copies between the host and [d], whose memory the host does not address: [d]
   copies, through its mapping of the host's memory or through the staging
   memory. *)
let copy direction =
  let name =
    match direction with `Into -> "copy into" | `Out_of -> "copy out of"
  in
  op name ~opener:staging (fun d ->
      let on_d = B.create d page in
      let src, dst =
        match direction with
        | `Into -> (filled page, on_d)
        | `Out_of ->
            B.copy ~src:(filled page) ~dst:on_d;
            (on_d, B.create Rig.host page)
      in
      fun _ ->
        B.copy ~src ~dst;
        let back = B.create Rig.host page in
        B.copy ~src:dst ~dst:back;
        equal string (pattern page) (bytes back))

let load =
  op "load" (fun d failure ->
      match Rig.Image.load d "code:64" with
      | Ok p -> (
          match Rig.Image.resolve p "main" with
          | Some _ -> ()
          | None -> fail "main is absent")
      | Error _ -> equal ~msg:"load refused" bool true (refusing failure))

let wait_transport =
  op "wait behind a transport" ~opener:(P.open_ ~transport:true) (fun d ->
      let run = Sub.Run.make (Sub.make d [||]) in
      fun _ -> Rig.Point.wait (once run))

(* Submits on [d] once a submission that keeps a value, leaving the
   submission unreachable. *)
let[@inline never] submit_kept d =
  Rig.Point.wait (once (Sub.Run.make (Sub.make ~keep:(ref ()) d [||])))

let keep =
  op "keep" (fun d _ ->
      submit_kept d;
      collect ();
      ignore (B.create d 0))

(* A submission with a part under a profile: its hand-over takes a time pair,
   and the profile reads it once the value is reached. *)
let profile =
  op "profile" (fun d ->
      let arg = B.create d 8 in
      let s = Sub.make d [| bump |] in
      let run = Sub.Run.make s and buffers = [| arg |] in
      Sub.Run.int64 run (Sub.block s 0) 0 0;
      fun _ ->
        ignore
          (Rig.Profile.take (fun () -> Rig.Point.wait (once ~buffers run))))

let ops =
  [
    create B.Device;
    create B.Pinned;
    create B.Mapped;
    borrow_host;
    borrow_peer;
    submit;
    load;
    wait_transport;
    keep;
    profile;
  ]

(* A copy through the staging memory needs [d] to map it. *)
let staged = [ copy `Into; copy `Out_of ]

let count call p = List.length (List.filter (( = ) call) (P.log p))

(* The peer device this process opened, if any. *)
let peer_devices () =
  match !peers with
  | Some (pid, peer) when pid = Unix.getpid () -> [ peer ]
  | _ -> []

(* Collects what the caller dropped, then returns memory in the order it comes
   back: the frees of the host's, the peer's and [d]'s caches drain each first,
   which moves [d]'s mappings of their memory into [d]'s cache, which [d]'s
   free returns; the memory they mapped is then unreachable, and the same
   frees again take it back. *)
let settle d =
  let devices = Rig.host :: peer_devices () @ [ d ] in
  collect ();
  List.iter Rig.free_cache devices;
  collect ();
  List.iter Rig.free_cache devices
