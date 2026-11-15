(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module A = Nx_array
module D = Nx_array.Dtype

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

(* The host's device, where a value is read out ({!on_host}). *)
let host = Devices.one Devices.host 0

(* A read of a value [owner] traces, which has no elements while [owner] runs
   and none once it has returned. *)
let traced ~by (owner : Value.interpretation) =
  if Atomic.get owner.live then
    invalid_argf "%s: the value is traced by %s" by owner.name
  else
    invalid_argf "%s: a value of %s, used after %s returned" by owner.name
      owner.name

let on_host : type v s d. by:string -> (v, s, d) Value.t -> (v, s) A.t =
 fun ~by x ->
  match x with
  | Value.Traced { owner; _ } -> traced ~by owner
  | Value.Array _ | Value.Shards _ | Value.Handle _ ->
      Iarray.get (Prim.arrays (Exec.value ~by host x)) 0

let written ~own bs = own && List.for_all (fun b -> Rig.Buffer.reached b Read) bs

(* Pending checks *)

type pending = {
  holds : (bool, D.bool_elt) A.t;
  on : Rig.t;
  mark : int;
  kept : (Rig.Buffer.keep * Rig.Buffer.t list) option;
  found : unit -> unit -> exn;
}

(* A pending check, unread, or a failure read as the builder of its exception,
   which holds nothing of a device's. *)
type entry = Unread of pending | Fails of (unit -> exn)

(* A set's entries, newest first, and its claim: the mutex that whoever reads
   or takes off its entries holds, so that one reader at a time decides each
   entry. An issue adds entries without it. *)
type set = { claim : Mutex.t; entries : entry list }

module Sets = Map.Make (Int)

(* Every set with entries, by set number. The process's domains and systhreads
   share them, so every update is a compare-and-set of the whole. A set with
   no entry has no binding. *)
let table : set Sets.t Atomic.t = Atomic.make Sets.empty

(* [f] of the table, which it replaces, atomically against every other update.
   [f] may run more than once. *)
let rec update f =
  let sets = Atomic.get table in
  let sets', r = f sets in
  if Atomic.compare_and_set table sets sets' then r else update f

let oldest n =
  match Sets.find_opt n (Atomic.get table) with
  | None -> []
  | Some s -> List.rev s.entries

let push n e =
  update (fun sets ->
      let s =
        match Sets.find_opt n sets with
        | Some s -> { s with entries = e :: s.entries }
        | None -> { claim = Mutex.create (); entries = [ e ] }
      in
      (Sets.add n s sets, ()))

(* Replaces the entry [e] of set [n] by [by], or takes it off for [None], and
   ends its keep. Only the holder of [n]'s claim calls it. *)
let replace n e by =
  update (fun sets ->
      match Sets.find_opt n sets with
      | None -> (sets, ())
      | Some s -> (
          let at e' = if e' == e then Option.to_list by else [ e' ] in
          match List.concat_map at s.entries with
          | [] -> (Sets.remove n sets, ())
          | entries -> (Sets.add n { s with entries } sets, ())));
  match e with
  | Unread { kept = Some (k, _); _ } -> Rig.Buffer.release k
  | Unread { kept = None; _ } | Fails _ -> ()

(* [Some (f ())] holding set [n]'s claim, [None] where [n] has no entry or,
   unless [wait], where another reader holds the claim. A set that left the
   table while the caller waited for its claim may come back with another. *)
let rec claimed ~wait n f =
  match Sets.find_opt n (Atomic.get table) with
  | None -> None
  | Some s ->
      let locked =
        if wait then begin
          Mutex.lock s.claim;
          true
        end
        else Mutex.try_lock s.claim
      in
      let current () =
        match Sets.find_opt n (Atomic.get table) with
        | Some s' -> s'.claim == s.claim
        | None -> false
      in
      if not locked then None
      else if current () then
        Some (Fun.protect ~finally:(fun () -> Mutex.unlock s.claim) f)
      else begin
        Mutex.unlock s.claim;
        if wait then claimed ~wait n f else None
      end

(* What reading a check finds. *)
type found = Holds | Fails_as of (unit -> exn)

(* Reads [c], the entry [e] of set [n], holding [n]'s claim. A read that
   raises, such as {!Rig.Lost}, takes [e] off before the exception passes
   on. *)
let read n e c =
  match if A.get c.holds [||] then Holds else Fails_as (c.found ()) with
  | r -> r
  | exception x ->
      let bt = Printexc.get_raw_backtrace () in
      replace n e None;
      Printexc.raise_with_backtrace x bt

(* Whether reading [c] waits for nothing, asked while issuing a check on
   devices [ds]. On one of [ds], its device's report of [c.mark] decides,
   which commits nothing and waits for nothing: a device reports its work at
   most its driver's batch of values late while work is submitted on it
   ({!Rig.section-timeline}). Another device may get no more work, and its
   last values would then stay unreported: asking commits them, since an idle
   device has nothing to batch. The final reduction and its copy on [c.on]
   follow the work on the others. A check read in place waits on the buffers
   it keeps too, which another device's queued copy may write. *)
let reached n e ds c =
  match
    let own =
      if List.exists (Rig.equal c.on) ds then Rig.signaled c.on >= c.mark
      else Rig.Buffer.reached (A.buffer c.holds) Read
    in
    written ~own (match c.kept with Some (_, bs) -> bs | None -> [])
  with
  | r -> r
  | exception x ->
      let bt = Printexc.get_raw_backtrace () in
      replace n e None;
      Printexc.raise_with_backtrace x bt

(* Reads set [n]'s unread checks older than [mine], or all of them, whose
   reading waits for nothing, oldest first, stopping at the first still
   running: a passing one goes, a failing one stays as the builder of its
   exception. It gives up where another reader holds the claim, so that an
   issue never waits. *)
let settle ?mine n ds =
  let rec go = function
    | [] -> ()
    | e :: _ when Option.exists (( == ) e) mine -> ()
    | Fails _ :: rest -> go rest
    | (Unread c as e) :: rest ->
        if reached n e ds c then begin
          (match read n e c with
          | Holds -> replace n e None
          | Fails_as b -> replace n e (Some (Fails b)));
          go rest
        end
  in
  ignore (claimed ~wait:false n (fun () -> go (oldest n)))

let add ~devices n c =
  let mine = Unread c in
  push n mine;
  settle ~mine n devices

let add_read ~devices n failure =
  match failure with
  | None -> settle n devices
  | Some b ->
      let mine = Fails b in
      push n mine;
      settle ~mine n devices

(* Building a failure's exception

   A wait takes the failure off its set and lets the claim go before [fail]
   builds the exception, and [fail] may read the values it receives, which
   waits on a set. Such a wait raises no failure, which would replace the one
   being built: the build runs under a handler that answers [Building],
   whatever systhread or fiber performs it. *)
type _ Effect.t += Building : bool Effect.t

(* Whether the caller runs inside the build of a failure's exception. *)
let building () =
  match Effect.perform Building with
  | b -> b
  | exception Effect.Unhandled Building -> false

(* Re-raised with the backtrace of its raise, which a plain [raise] would
   restart at the handler. *)
let reraise e = Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())

(* The exception [build] makes, under the handler. What [fail] raises keeps
   its backtrace. *)
let built build =
  let effc (type a) (e : a Effect.t) =
    match e with
    | Building ->
        Some
          (fun (k : (a, _) Effect.Deep.continuation) ->
            Effect.Deep.continue k true)
    | _ -> None
  in
  Effect.Deep.match_with build () { retc = Fun.id; exnc = reraise; effc }

(* The oldest failure of set [n], taken off holding [n]'s claim: its entries
   are read oldest first, waiting for each, and the passing ones go. *)
let oldest_failure n =
  let rec go = function
    | [] -> None
    | (Fails b as e) :: _ ->
        replace n e None;
        Some b
    | (Unread c as e) :: rest -> (
        match read n e c with
        | Holds ->
            replace n e None;
            go rest
        | Fails_as b ->
            replace n e None;
            Some b)
  in
  Option.join (claimed ~wait:true n (fun () -> go (oldest n)))

let wait : type v s d. by:string -> (v, s, d) Value.t -> unit =
 fun ~by x ->
  match x with
  | Value.Traced { owner; _ } -> traced ~by owner
  | Value.Array _ | Value.Shards _ | Value.Handle _ -> (
      Iarray.iter (fun a -> Rig.Buffer.wait (A.buffer a) Read) (Prim.arrays x);
      let n = Devices.number (Devices.set (Prim.placement x)) in
      if Sets.mem n (Atomic.get table) && not (building ()) then
        match oldest_failure n with Some b -> raise (built b) | None -> ())
