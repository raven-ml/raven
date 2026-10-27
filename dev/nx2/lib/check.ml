(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module P = Nx_kernel.Prog

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

(* The host's device, at the brand of a value this module reads there: a
   check's flag, a value read out ({!on_host}). No value at it reaches a
   function of nx's own. *)
let host () = Devices.rebrand (Devices.one Devices.host 0)

(* Whether [p] lies on the host's set. *)
let on_host_set p = Devices.number (Devices.set p) = Devices.number Devices.host

let on_host : type v s d. by:string -> (v, s, d) Value.t -> (v, s) A.t =
 fun ~by x ->
  let host : d Devices.placement = host () in
  match x with
  | Value.Deferred _ -> Iarray.get (Prim.arrays (Constant.at host x)) 0
  | Value.Traced { owner; _ } ->
      invalid_argf "%s: the value is traced by %s" by owner.name
  | Value.Array _ | Value.Shards _ | Value.Donated _ ->
      Iarray.get (Prim.arrays (Place.value ~by host x)) 0

(* Pending checks

   A check off the host reduces its flags where they lie and leaves the result
   there unread, since reading it would make the host wait for the set. Each
   domain lists the checks it issued, by set, and a wait on a set drains that
   set's, oldest first, up to the first that fails. A pending check holds its
   data's memory, so that no donation writes the elements its failure
   reports. *)

type ('d, 'e) issued = {
  holds : (bool, D.bool_elt) A.t;
      (* Whether every flag holds, in memory the host addresses where its
         device is of this machine ({!readable}). *)
  mark : int;
      (* [holds]'s device's last value once [holds] was written. *)
  first : (int64, D.int64_elt, 'd) Value.t option;
      (* The first false flag's position in C order; [None] for one flag,
         whose position is [0]. *)
  shape : int array;  (* The flags'. *)
  data : ('d, 'e) Value.data;
  fail : int array -> 'e -> exn;
}

(* A check and the hold on its data's memory, or the exception of one that
   failed before any wait read it. *)
type pending =
  | Pending : ('d, 'e) issued * Rig.Claim.hold -> pending
  | Raised : exn -> pending

module Sets = Map.Make (Int)

(* Each domain's pending checks, by set number, each set's newest first. The
   domain's systhreads share them, so every update is a compare-and-set of the
   whole. *)
let checks : pending list Sets.t Atomic.t Domain.DLS.key =
  Domain.DLS.new_key (fun () -> Atomic.make Sets.empty)

(* [f] of the calling domain's pending checks, which it replaces, atomically
   against the domain's other systhreads. *)
let rec update f =
  let a = Domain.DLS.get checks in
  let sets = Atomic.get a in
  let sets', r = f sets in
  if Atomic.compare_and_set a sets sets' then r else update f

let of_set n sets = Option.value ~default:[] (Sets.find_opt n sets)

let raised = function Raised _ -> true | Pending _ -> false

(* Adds [p] to set [n]'s pending checks, unless an earlier one failed, whose
   exception the next wait raises in place of any later one's: [false] then. *)
let pend n p =
  update (fun sets ->
      let l = of_set n sets in
      if List.exists raised l then (sets, false)
      else (Sets.add n (p :: l) sets, true))

(* Ends a check's hold. *)
let release = function
  | Pending (_, h) -> Rig.Claim.release h
  | Raised _ -> ()

(* [a] in memory the host addresses where its device is of this machine:
   itself where the host addresses the device's memory, else a copy the
   device makes in its order into memory both address, which the host then
   reads without a copy once the device reports the copy done. *)
let readable a =
  let d = A.device a in
  if Rig.reaches Rig.host d || not (Rig.equal (Rig.host_of d) Rig.host) then a
  else
    let dst = A.create ~memory:Pinned d (A.dtype a) (L.shape (A.layout a)) in
    A.transfer ~src:a ~dst;
    dst

(* Whether [c.holds] is written, so that reading it waits for nothing, asked
   while issuing a check on devices [ds]. On one of [ds], its device's report
   decides, which commits nothing and waits for nothing: a device reports its
   work at most its driver's batch of values late while work is submitted on
   it ({!Rig.section-timeline}). Another device may get no more work, and its
   last values would then stay unreported: asking commits them, since an idle
   device has nothing to batch. The final reduction on [c.holds]'s device
   follows the work on the others, so that device alone decides. *)
let reached ds c =
  let d = A.device c.holds in
  if List.exists (Rig.equal d) ds then Rig.signaled d >= c.mark
  else Rig.Buffer.reached (A.buffer c.holds) Read

(* [a]'s element, read on the host: in place where it addresses [a]'s memory,
   else through a copy. *)
let get_on_host a =
  let d = A.device a in
  if Rig.equal (Rig.host_of d) Rig.host then A.get a [||]
  else A.get (A.to_device Rig.host a) [||]

(* The one element of [a], as a 0-d view. *)
let unit_view a = Option.get (A.move (Reshape [||]) a)

(* Set [n]'s pending checks, oldest first, taken off the list. *)
let drained n =
  update (fun sets -> (Sets.remove n sets, List.rev (of_set n sets)))

(* The first datum's placement, if a datum has bytes. *)
let rec data_placement : type d e.
    (d, e) Value.data -> d Devices.placement option = function
  | [] -> None
  | x :: rest -> (
      match Prim.at x with Some p -> Some p | None -> data_placement rest)

(* [data] with each datum that has bytes a live value of its own over them,
   which no donation of the datum kills. *)
let rec kept : type d e. (d, e) Value.data -> (d, e) Value.data = function
  | [] -> []
  | ((Value.Deferred _ | Value.Traced _) as x) :: rest -> x :: kept rest
  | x :: rest ->
      let x =
        Prim.of_arrays (Prim.placement x) ~shape:(Prim.shape x) (Prim.arrays x)
      in
      x :: kept rest

(* The buffers of [data]'s bytes. *)
let rec data_buffers : type d e. (d, e) Value.data -> Rig.Buffer.t list =
  function
  | [] -> []
  | (Value.Deferred _ | Value.Traced _) :: rest -> data_buffers rest
  | x :: rest ->
      Iarray.fold_right
        (fun a bs -> A.buffer a :: bs)
        (Prim.arrays x) (data_buffers rest)

let unravel i shape =
  let r = Array.length shape in
  let idx = Array.make r 0 and k = ref i in
  for a = r - 1 downto 0 do
    idx.(a) <- !k mod shape.(a);
    k := !k / shape.(a)
  done;
  idx

(* Each of [data]'s elements at [idx], read on the host. *)
let rec data_at : type d e. by:string -> int iarray -> (d, e) Value.data -> e
    =
 fun ~by idx -> function
  | [] -> ()
  | x :: rest ->
      let v = A.get (on_host ~by x) idx in
      (v, data_at ~by idx rest)

(* The exception of a check whose first false lies at [first] in C order. *)
let failed ~by first shape data fail =
  let idx = unravel (Int64.to_int first) shape in
  fail idx (data_at ~by (Iarray.of_array idx) data)

(* [c]'s exception, read on the host, where a flag is false. *)
let outcome : type d e. by:string -> (d, e) issued -> exn option =
 fun ~by c ->
  if get_on_host c.holds then None
  else
    let first =
      match c.first with
      | Some f -> A.get (on_host ~by f) [||]
      | None -> 0L
    in
    Some (failed ~by first c.shape c.data c.fail)

(* Reads [checks], oldest first, up to the first that fails, which raises: the
   later ones go unread. Every check's hold ends then, so that a donation of its
   data right after the wait may reuse the memory. *)
let settle ~by checks =
  let failure = function
    | Pending (c, _) -> outcome ~by c
    | Raised e -> Some e
  in
  Fun.protect ~finally:(fun () -> List.iter release checks) (fun () ->
      Option.iter raise (List.find_map failure checks))

(* Settles set [n]'s earlier checks whose flags are computed, so that a domain
   that never waits on the set holds only the data of the checks still
   running: a passing check goes, a failing one keeps only its exception, and
   the checks after it go, since no wait can raise them. It reads them oldest
   first and stops at the first still running, so it asks rig of one check
   more than it settles. The checks are read outside the update, which then
   takes off those still pending, against a wait on another systhread that
   drains them meanwhile. [ds] are the devices of the check being issued
   ({!reached}). [false] where the set then holds a failure. *)
let settle_reached ~by n ds =
  let earlier = List.rev (of_set n (Atomic.get (Domain.DLS.get checks))) in
  let rec read = function
    | [] | Raised _ :: _ -> []
    | (Pending (c, _) as p) :: rest -> (
        if not (reached ds c) then []
        else
          match outcome ~by c with
          | None -> (p, None) :: read rest
          | Some e -> [ (p, Some e) ])
  in
  match read earlier with
  | [] -> not (List.exists raised earlier)
  | outcomes ->
      (* Set [n]'s checks after the update, oldest first, and those it
         ends. *)
      let rec cut = function
        | [] -> ([], [])
        | (Raised _ as r) :: rest -> ([ r ], rest)
        | (Pending _ as p) :: rest -> (
            match List.assq_opt p outcomes with
            | None ->
                let kept, ended = cut rest in
                (p :: kept, ended)
            | Some None ->
                let kept, ended = cut rest in
                (kept, p :: ended)
            | Some (Some e) -> ([ Raised e ], p :: rest))
      in
      let kept, ended =
        update (fun sets ->
            match cut (List.rev (of_set n sets)) with
            | [], ended -> (Sets.remove n sets, ([], ended))
            | kept, ended -> (Sets.add n (List.rev kept) sets, (kept, ended)))
      in
      List.iter release ended;
      not (List.exists raised kept)

let wait : type v s d. by:string -> (v, s, d) Value.t -> unit =
 fun ~by x ->
  match x with
  | Value.Deferred _ -> ()
  | Value.Traced { owner; _ } ->
      invalid_argf "%s: the value is traced by %s" by owner.name
  | Value.Array _ | Value.Shards _ | Value.Donated _ ->
      Iarray.iter (fun a -> Rig.Buffer.wait (A.buffer a) Read) (Prim.arrays x);
      settle ~by (drained (Devices.number (Devices.set (Prim.placement x))))

(* Issuing *)

(* The reduction [r] of the flags [ok], which have bytes, over all their
   axes: computed where its route reads [ok], by windows or by partials. *)
let reduce ~by ok r =
  let shape = Prim.shape ok in
  let op =
    Value.Reduce
      {
        layout = L.contiguous (Iarray.of_array shape);
        axes = Iarray.init (Array.length shape) Fun.id;
        prog = P.of_node ~ins:[| D.Any D.Bool |] (In 0);
        reductions = Value.[ r ];
        loads = [| Plain ok |];
      }
  in
  let { Prim.route; layouts } = Prim.route ~by op in
  (* [ok] has bytes, so the reduction has a route. *)
  let route = Option.get route in
  let place i x = Exec.placer ~by (Devices.rebrand route.operands.(i)) x in
  let op = Prim.mapi place op in
  match route.compute with
  | Partials -> Partial.run ~by route op
  | Windows -> Exec.windows ~by route layouts op

(* The one element of [x], which has bytes: a 0-d view of the array that holds
   it. A cut placement's other devices hold empty windows. *)
let element x =
  let holds a = L.numel (A.layout a) = 1 in
  unit_view (Option.get (Iarray.find_opt holds (Prim.arrays x)))

let issue : type d e.
    by:string ->
    (bool, D.bool_elt, d) Value.t ->
    (d, e) Value.data ->
    (int array -> e -> exn) ->
    unit =
 fun ~by ok data fail ->
  let shape = Prim.shape ok in
  if Array.mem 0 shape then ()
  else
    (* The check computes where [ok]'s bytes lie, else where a datum's do, and
       on the host where none has bytes. A constant [ok] is computed there
       just before its reduction. *)
    let p =
      match Prim.at ok with
      | Some p -> p
      | None -> Option.value ~default:(host ()) (data_placement data)
    in
    let scalar x = A.get (on_host ~by x) [||] in
    (* One flag is its own reduction: no kernel reduces it. *)
    let one = Array.for_all (( = ) 1) shape in
    if not (on_host_set p) then begin
      (* The set's earlier checks settle first, so that a raise there leaves
         this check unissued and none of its data held. A set that holds a
         failure keeps no later check, which then computes nothing. *)
      let set = Devices.set p in
      let n = Devices.number set in
      let ds = Array.map (Devices.rig set) (Grid.devices (Devices.grid p)) in
      if settle_reached ~by n (Array.to_list ds) then begin
        let ok = Constant.at p ok in
        (* One reduction gives both whether every flag holds and the first
           false, so that the check leaves only scalars to read later. The
           mark follows the copy [readable] submits, which, for one flag,
           no reduction orders. The hold keeps a donation from writing a
           flag read in place. *)
        let holds, first =
          if one then (element ok, None)
          else
            let (holds, first), () = reduce ~by ok (Arg (Min, 0, D.Bool)) in
            (Iarray.get (Prim.arrays holds) 0, Some first)
        in
        let holds = readable holds in
        let mark = Rig.submitted (A.device holds) in
        let data = kept data in
        let held = Rig.Claim.hold (A.buffer holds :: data_buffers data) in
        let c = { holds; mark; first; shape; data; fail } in
        if not (pend n (Pending (c, held))) then Rig.Claim.release held
      end
    end
    else begin
      let ok = Constant.at p ok in
      if one then begin
        if not (get_on_host (element ok)) then
          raise (failed ~by 0L shape data fail)
      end
      else
        (* A passing check runs one [Min]; [Arg], for the first [false],
           costs more and runs only on a failure. *)
        let holds, () = reduce ~by ok (Monoid (Min, 0, D.Bool)) in
        if not (scalar holds) then
          let (_, first), () = reduce ~by ok (Arg (Min, 0, D.Bool)) in
          raise (failed ~by (scalar first) shape data fail)
    end
