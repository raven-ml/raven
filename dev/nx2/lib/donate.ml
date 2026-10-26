(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

let died_to ~by x earlier =
  invalid_argf "%s: %a was donated to %s" by Prim.pp_value x earlier

(* The key the next donor is offered with. *)
let offers = Atomic.make 0

(* The key of [x]'s donor word [w], which a live donor is given here: a donor
   keeps it for life, and every handle of every donation of it holds it. A
   word that died raises naming its consumer. *)
let rec offer ~by x (w : Value.death Atomic.Loc.t) =
  match Atomic.Loc.get w with
  | Value.Offered k -> k
  | Value.Died earlier -> died_to ~by x earlier
  | Value.Live ->
      (* Fails only where another donation offered it first. *)
      let k = Atomic.fetch_and_add offers 1 in
      if Atomic.Loc.compare_and_set w Value.Live (Value.Offered k) then k
      else offer ~by x w

let donate (type v s d) ~by (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  let handle at shape arrays donor =
    let key = offer ~by x donor in
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
      Prim.alive ~by x;
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

(* The donated operands among [xs], the first of each donor's: two handles of
   one donor are read as a handle beside its donor is, and the donor dies, and
   its memory is consumed, once. *)
let handles xs =
  let rec first seen = function
    | [] -> []
    | h :: rest ->
        let k = key h in
        if List.mem k seen then first seen rest else h :: first (k :: seen) rest
  in
  first []
    (List.filter (fun (Handle (_, x)) -> is_handle x)
       (List.mapi (fun i (Value.Any x) -> Handle (i, x)) xs))

(* The buffers of a handle, one per device. *)
let buffers (Handle (_, x)) = Iarray.to_list (Iarray.map A.buffer (Prim.arrays x))

(* [y] as the consumer of the handles [hs] reads it, before it claims them: a
   handle as its arrays, a donor offered under one of their keys as a live
   value, and any other value as it is. *)
let called (type v s d) hs (y : (v, s, d) Value.t) : (v, s, d) Value.t =
  let ours (w : Value.death) =
    match w with
    | Value.Offered k -> List.exists (fun h -> key h = k) hs
    | Value.Live | Value.Died _ -> false
  in
  match y with
  | Value.Donated _ -> live y
  | Value.Array r when ours r.dead -> Value.Array { r with dead = Value.Live }
  | Value.Shards r when ours r.dead -> Value.Shards { r with dead = Value.Live }
  | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> y

let unhandled hs op = Prim.map (fun y -> called hs y) op

(* Claims

   A consumer claims each handle's donor, in any order, setting the donor's
   word to its death, for good; a donor that died first, on any domain, raises
   naming its consumer. Two consumers racing for one donor is a misuse: one of
   them raises, possibly both, and a donor a loser claimed before it raised
   stays dead. *)

(* Sets the donor word [w] of the handle [x] to [died] unless it died first,
   on any domain, which raises naming that consumer. A word that changed since
   it was read is read again. *)
let rec claim_word ~by x (w : Value.death Atomic.Loc.t) died =
  match Atomic.Loc.get w with
  | Value.Died earlier -> died_to ~by x earlier
  | Value.Live -> invalid_arg "Donate.claim: a handle's donor was not offered"
  | Value.Offered _ as seen ->
      if not (Atomic.Loc.compare_and_set w seen died) then
        claim_word ~by x w died

let claim ~by hs =
  let died = Value.Died by in
  List.iter
    (fun (Handle (_, x)) ->
      match x with
      | Value.Donated d -> claim_word ~by x d.donor died
      | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ ->
          ())
    hs

(* A handle's arrays, which an interpretation or a function of several
   operations may keep: shared, so that no later donation writes where they
   read. *)
let lent (type v s d) (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  match x with
  | Value.Donated _ ->
      let y = live x in
      Iarray.iter (fun a -> Rig.Claim.share (A.buffer a)) (Prim.arrays y);
      y
  | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> x

let read ~by x =
  Prim.alive ~by x;
  lent x

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
            |> List.iter (fun b -> ignore (Rig.Buffer.free ~why b : bool))
      | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> ())
    hs

(* Whether [a] and [a'] are one buffer placing each index at the same bytes
   as the same dtype. *)
let identical a a' =
  A.buffer a == A.buffer a'
  && D.equal (A.dtype a) (A.dtype a')
  && L.equal (A.layout a) (A.layout a')

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
                (* An operand at the result's own index may read the memory
                   as [a] places it, and no other way: a view of [a] in
                   another layout reads other indices of it. *)
                let apart a' =
                  (not (share (A.buffer a') b))
                  || (List.mem j own && identical a a')
                in
                if not (Prim.is_constant z) then
                  Iarray.iter
                    (fun a' -> if not (apart a') then ok := false)
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

