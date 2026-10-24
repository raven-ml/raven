(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

let died_to ~by i earlier =
  invalid_argf "%s: operand %d was donated to %s" by (i + 1) earlier

(* The key the next donor is offered with. *)
let offers = Atomic.make 0

(* The key of the donor word [w], which a live donor is given here: a donor
   keeps it for life, and every handle of every donation of it holds it. A
   word that died, or that a consumer holds, raises naming that consumer. *)
let rec offer ~by (w : Value.death Atomic.Loc.t) =
  match Atomic.Loc.get w with
  | Value.Offered k -> k
  | Value.Died { by = earlier; _ } -> died_to ~by 0 earlier
  | Value.Live ->
      (* Fails only where another donation offered it first. *)
      let k = Atomic.fetch_and_add offers 1 in
      if Atomic.Loc.compare_and_set w Value.Live (Value.Offered k) then k
      else offer ~by w

let donate (type v s d) ~by (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  let handle at shape arrays donor =
    let key = offer ~by donor in
    Value.Donated { key; at; shape; arrays; donor }
  in
  match x with
  | Value.Array r ->
      handle r.at
        (L.shape (A.layout r.a))
        ([| r.a |] : _ iarray)
        [%atomic.loc r.dead]
  | Value.Shards r -> handle r.at r.shape r.arrays [%atomic.loc r.dead]
  | Value.Donated _ | Value.Deferred _ | Value.Traced _ ->
      Prim.alive ~by 0 x;
      x

(* A handle's arrays as a live value, which the operation reads. *)
let live (type v s d) (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  match x with
  | Value.Donated { at; shape; arrays; _ } -> Prim.of_arrays at ~shape arrays
  | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> x

(* A donated operand and its index among the operation's operands. *)
type handle = Handle : int * ('v, 's, 'd) Value.t -> handle

let is_handle (type v s d) (x : (v, s, d) Value.t) =
  match x with
  | Value.Donated _ -> true
  | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> false

(* The key of the handle [h]'s donor. *)
let key (Handle (_, x)) =
  match x with
  | Value.Donated d -> d.key
  | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ ->
      invalid_arg "Donate.key: not a handle"

(* Whether [b] and [b'] are one buffer, or live and share memory: a dead
   buffer's place may lie in memory made since its memory was freed. *)
let share b b' =
  b == b'
  || Option.is_none (Rig.Buffer.dead b)
     && Option.is_none (Rig.Buffer.dead b')
     && Rig.Buffer.overlaps b b'

let twice ~by = invalid_argf "Nx.donate: %s reads one donation twice" by

(* The donated operands among [xs]. Two handles of one donor or over one memory
   raise naming [Nx.donate]: the same handle twice, two donations of one value,
   or one handle and a value a movement passed on. *)
let handles ~by xs =
  let hs =
    List.filter (fun (Handle (_, x)) -> is_handle x)
      (List.mapi (fun i (Value.Any x) -> Handle (i, x)) xs)
  in
  let rec check = function
    | [] -> ()
    | Handle (_, x) :: rest ->
        List.iter
          (fun (Handle (_, y)) ->
            let same =
              match (x, y) with
              | Value.Donated a, Value.Donated b ->
                  a.key = b.key
                  || Iarray.exists
                       (fun p ->
                         Iarray.exists
                           (fun q -> share (A.buffer p) (A.buffer q))
                           b.arrays)
                       a.arrays
              | _ -> false
            in
            if same then twice ~by)
          rest;
        check rest
  in
  check hs;
  hs

(* The buffers of a handle, one per device. *)
let buffers (Handle (_, x)) = Iarray.to_list (Iarray.map A.buffer (Prim.arrays x))

(* Claims

   A consumer claims each handle's donor, in any order, setting the donor's
   word to its death; a donor that died first, on any domain, raises naming its
   consumer. Two consumers racing for one donor is a misuse: one of them
   raises, possibly both, and a donor a loser claimed before it raised stays
   dead. *)

(* Sets the donor word [w] of operand [i] to [died] unless it died first, on
   any domain, which raises naming that consumer. A word that changed since it
   was read is read again. *)
let rec claim_word ~by i (w : Value.death Atomic.Loc.t) died =
  match Atomic.Loc.get w with
  | Value.Died { by = earlier; _ } -> died_to ~by i earlier
  | Value.Live -> invalid_arg "Donate.claim: a handle's donor was not offered"
  | Value.Offered _ as seen ->
      if not (Atomic.Loc.compare_and_set w seen died) then
        claim_word ~by i w died

let claim ~by hs =
  List.iter
    (fun (Handle (i, x)) ->
      match x with
      | Value.Donated d ->
          claim_word ~by i d.donor (Value.Died { by; key = d.key })
      | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ ->
          ())
    hs

(* Gives back the claims of the handles [hs], whose consumer neither wrote nor
   consumed their memory: each donor is offered again under its key. No other
   consumer writes a word once it died, so each is set in place. *)
let unclaim hs =
  List.iter
    (fun (Handle (_, x)) ->
      match x with
      | Value.Donated d -> Atomic.Loc.set d.donor (Value.Offered d.key)
      | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> ())
    hs

(* Whether the word [w] is a donor's of one of the handles [hs]: offered, or
   dead, under one of their keys. Every handle among [hs] was checked live, so
   a word dead under one of their keys died to [hs]' claim. *)
let ours hs (w : Value.death) =
  match w with
  | Value.Offered k | Value.Died { key = k; _ } ->
      List.exists (fun h -> key h = k) hs
  | Value.Live -> false

let called : type v s d.
    handle list -> (v, s, d) Value.t -> (v, s, d) Value.t =
 fun hs y ->
  match y with
  | Value.Donated _ -> live y
  | Value.Array r when ours hs r.dead -> Value.Array { r with dead = Value.Live }
  | Value.Shards r when ours hs r.dead ->
      Value.Shards { r with dead = Value.Live }
  | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> y

let unhandled hs op = Prim.map (fun y -> called hs y) op

let consume ~by xs =
  let hs =
    if List.exists (fun (Value.Any x) -> is_handle x) xs then handles ~by xs
    else []
  in
  List.iteri (fun i (Value.Any x) -> Prim.alive ~by i x) xs;
  claim ~by hs;
  hs

let consume_list ~by xs =
  let hs = consume ~by xs in
  List.map (fun (Value.Any x) -> Value.Any (called hs x)) xs

(* A handle's arrays, which an interpretation may keep: shared, so that no
   later donation of the donor writes where they read. *)
let lent (type v s d) (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  match x with
  | Value.Donated _ ->
      let y = live x in
      Iarray.iter (fun a -> Rig.Claim.share (A.buffer a)) (Prim.arrays y);
      y
  | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> x

(* [b]'s memory consumed for [by], where rig holds it exclusive: no other value
   reads it. The live buffer over it, if so. *)
let take ~by b =
  Rig.Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
      if Rig.Claim.exclusive c b then
        Some (Rig.Claim.consume c ~why:("donated to " ^ by) b)
      else None)

(* Whether no two of [bufs] share memory. Shards placed by borrowing lie in
   one allocation, where each reads the memory the other holds. *)
let rec apart = function
  | [] -> true
  | b :: rest -> (not (List.exists (Rig.Buffer.overlaps b) rest)) && apart rest

(* Ends the claimed handles [hs] once [by] read them: memory no other value
   reads, every shard of a value or none, and that [by] did not write its
   result into ([reused]), is consumed and then freed, so that it returns to
   its device once the work on it is done. Shards that share memory with one
   another are not consumed: each is read by the other, and rig donates no
   two buffers over one memory together. *)
let consumed ~by ~reused hs =
  List.iter
    (fun (Handle (_, x) as h) ->
      match x with
      | Value.Donated _ ->
          let bufs = buffers h in
          let why = "donated to " ^ by in
          if
            (not (List.exists (fun b -> List.memq b reused) bufs))
            && apart bufs
          then
            Rig.Claim.with_ ~read:[] ~donate:bufs (fun c ->
                if List.for_all (Rig.Claim.exclusive c) bufs then
                  List.map (Rig.Claim.consume c ~why) bufs
                else [])
            |> List.iter (fun b -> ignore (Rig.Buffer.free b : bool))
      | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> ())
    hs

(* Whether [a] may hold a result of [dt] and [layout] in its memory: of that
   dtype and layout, over all of its memory. *)
let holds (D.Any dt) layout a =
  D.equal (A.dtype a) dt
  && L.equal (A.layout a) layout
  && Rig.Buffer.spans (A.buffer a)

(* The handle among [hs] whose memory takes [op]'s first result, if one may:
   on one device, at an operand position {!Prim.own_index} gives, of the result's
   dtype, C-contiguous with its layout over all of its memory, beside
   operands at its placement none of which [op] reads at other indices from
   that memory, and whose memory no other value reads (rig holds it
   exclusive). Its memory is then consumed, naming [by], and its position,
   buffer and the live buffer over its memory are the answer. *)
let reuse : type r.
    by:string ->
    r Value.prim ->
    handle list ->
    (int * Rig.Buffer.t * Rig.Buffer.t) option =
 fun ~by op hs ->
  match Prim.own_index op with
  | None -> None
  | Some (own, layout, dt) -> (
      let fits (Handle (i, x)) =
        match x with
        | Value.Donated { at; arrays; _ }
          when Iarray.length arrays = 1 && List.mem i own ->
            let a = Iarray.get arrays 0 in
            let b = A.buffer a and ok = ref true in
            Prim.iteri
              (fun j z ->
                (match Prim.at z with
                | Some q -> if Devices.rebrand q != at then ok := false
                | None -> ());
                if (not (List.mem j own)) && not (Prim.is_constant z) then
                  Iarray.iter
                    (fun a' -> if share (A.buffer a') b then ok := false)
                    (Prim.arrays z))
              op;
            !ok && holds dt layout a
        | Value.Donated _ | Value.Array _ | Value.Shards _ | Value.Deferred _
        | Value.Traced _ ->
            false
      in
      match List.find_opt fits hs with
      | None -> None
      | Some (Handle (i, _) as h) ->
          let b = List.hd (buffers h) in
          Option.map (fun b' -> (i, b, b')) (take ~by b))

