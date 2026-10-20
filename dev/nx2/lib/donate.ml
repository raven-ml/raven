(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

(* The key the next donor is offered with. *)
let offers = Atomic.make 0

(* Gives the live donor word [w] a key of its own, which every handle of every
   donation of it reads. *)
let offer (w : Value.death Atomic.Loc.t) =
  match Atomic.Loc.get w with
  | Value.Live ->
      (* Fails only where another donation offered it first: a donor dies only
         through a handle, which it has after it is offered. *)
      let k = Atomic.fetch_and_add offers 1 in
      ignore (Atomic.Loc.compare_and_set w Value.Live (Value.Offered k))
  | Value.Offered _ | Value.Died _ -> ()

let donate (type v s d) ~by (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  Prim.alive ~by 0 x;
  let handle at shape arrays donor =
    offer donor;
    Value.Donated { at; shape; arrays; donor; spent = Value.Live }
  in
  match x with
  | Value.Array r ->
      handle r.at
        (L.shape (A.layout r.a))
        ([| r.a |] : _ iarray)
        [%atomic.loc r.dead]
  | Value.Shards r -> handle r.at r.shape r.arrays [%atomic.loc r.dead]
  | Value.Donated _ | Value.Deferred _ | Value.Traced _ -> x

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

(* Whether the donor words [w] and [w'] are one donor's: an offered donor has
   one key. *)
let one_donor (w : Value.death Atomic.Loc.t) w' =
  match (Atomic.Loc.get w, Atomic.Loc.get w') with
  | Value.Offered k, Value.Offered k' -> k = k'
  | (Value.Live | Value.Offered _ | Value.Died _), _ -> false

(* Whether [b] and [b'] are one buffer, or live and share memory: a dead
   buffer's place may lie in memory made since its memory was freed. *)
let share b b' =
  b == b'
  || Option.is_none (Rig.Buffer.dead b)
     && Option.is_none (Rig.Buffer.dead b')
     && Rig.Buffer.overlaps b b'

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
                  one_donor a.donor b.donor
                  || Iarray.exists
                       (fun p ->
                         Iarray.exists
                           (fun q -> share (A.buffer p) (A.buffer q))
                           b.arrays)
                       a.arrays
              | _ -> false
            in
            if same then
              invalid_argf "Nx.donate: %s reads one donation twice" by)
          rest;
        check rest
  in
  check hs;
  hs

(* The buffers of a handle, one per device. *)
let buffers (Handle (_, x)) = Iarray.to_list (Iarray.map A.buffer (Prim.arrays x))

(* Claims

   A consumer claims its handles in their donors' key order, each handle's own
   word before its donor's, setting each word to its death; it gives none back
   when it loses one. Each word has one place in the order, since every handle
   of a donor reads the donor's one key, and a consumer's words rise strictly,
   since [handles] refuses two handles of one donor. So of consumers racing for
   the same handles, at least one succeeds: a consumer that loses at a word
   lost it to one that claimed that word, which, if it lost too, lost at a word
   above it. Following who each loser lost to climbs the order, so it ends at a
   consumer that claimed all its words.

   Claimed in operand order, [add h1 h2] and [add h2 h1] on two domains could
   each take its first handle and lose its second to the other. In key order a
   loser may still name a loser. With h1 < h2 < h3, [add h1 h2] takes h1,
   [add h2 h3] takes h2, [add h1 h3] loses h1 naming the first, which loses h2
   naming the second, which takes h3 and succeeds. *)

let died_to ~by i earlier =
  invalid_argf "%s: operand %d was donated to %s" by (i + 1) earlier

(* Sets the death word [w] of operand [i] to [died] unless it died first, on
   any domain, which raises naming that consumer. No death is undone, so a word
   read live is only retried. *)
let rec claim_word ~by i (w : Value.death Atomic.Loc.t) died =
  match Atomic.Loc.get w with
  | Value.Died earlier -> died_to ~by i earlier
  | (Value.Live | Value.Offered _) as seen ->
      if not (Atomic.Loc.compare_and_set w seen died) then
        claim_word ~by i w died

(* Sets the handle [x]'s own word, operand [i], to [died] unless an operation
   read the handle first, which raises naming it. The word is set in place:
   passed to [claim_word], its location would be allocated. *)
let rec spend_word : type v s d.
    by:string -> int -> Value.death -> (v, s, d) Value.t -> unit =
 fun ~by i died x ->
  match x with
  | Value.Donated d -> (
      if not (Atomic.Loc.compare_and_set [%atomic.loc d.spent] Value.Live died)
      then
        match d.spent with
        | Value.Died earlier -> died_to ~by i earlier
        | Value.Live | Value.Offered _ -> spend_word ~by i died x)
  | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> ()

(* Marks the handle [x], operand [i], read by the movement [by], which passes a
   new handle on: one operation reads a handle. *)
let spend ~by i x = spend_word ~by i (Value.Died by) x

(* Claims the handle [x], operand [i], for its consumer [by], before it reads
   it: its own word, then its donor's, which orders the consumers of all the
   donor's handles. *)
let claim_handle (type v s d) ~by i (x : (v, s, d) Value.t) =
  match x with
  | Value.Donated d ->
      let died = Value.Died by in
      spend_word ~by i died x;
      claim_word ~by i d.donor died
  | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> ()

(* The key of the handle [x]'s donor, operand [i]. A donor that died raises
   naming its consumer. *)
let key (type v s d) ~by i (x : (v, s, d) Value.t) =
  match x with
  | Value.Donated d -> (
      match Atomic.Loc.get d.donor with
      | Value.Offered k -> k
      | Value.Died earlier -> died_to ~by i earlier
      | Value.Live ->
          invalid_arg "Donate.key: a handle's donor was not offered")
  | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ ->
      invalid_arg "Donate.key: not a handle"

(* Claims the handles [hs] for their consumer [by], in key order. Their keys
   differ: [handles] refused two handles of one donor. *)
let claim ~by hs =
  let keyed = List.map (fun (Handle (i, x) as h) -> (key ~by i x, h)) hs in
  List.iter
    (fun (_, Handle (i, x)) -> claim_handle ~by i x)
    (List.sort (fun (k, _) (k', _) -> Int.compare k k') keyed)

(* The keys of the donors of the handles [hs] that are still offered. *)
let offered hs =
  let key (Handle (_, x)) =
    match x with
    | Value.Donated d -> (
        match Atomic.Loc.get d.donor with
        | Value.Offered k -> Some k
        | Value.Live | Value.Died _ -> None)
    | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ ->
        None
  in
  List.filter_map key hs

(* Whether [death] is that of a donor offered under one of [keys]. *)
let among keys = function
  | Value.Offered k -> List.mem k keys
  | Value.Live | Value.Died _ -> false

(* [y] read as it is when its call is made, before the call's claim kills the
   donors of its handles, whose keys are [keys]: a handle as its arrays, and
   one of those donors as a live value over its arrays. *)
let as_called : type v s d. int list -> (v, s, d) Value.t -> (v, s, d) Value.t
    =
 fun keys y ->
  match y with
  | Value.Donated _ -> live y
  | Value.Array r when among keys r.dead ->
      Value.Array { r with dead = Value.Live }
  | Value.Shards r when among keys r.dead ->
      Value.Shards { r with dead = Value.Live }
  | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> y

(* [op] read as it is when it is called, before its claim kills the donors of
   its handles [hs]. *)
let unhandled hs op =
  let keys = offered hs in
  Prim.map (fun y -> as_called keys y) op

let consume ~by op =
  let (Prim.Operands xs) = Prim.operands op in
  let hs = if Prim.exists is_handle op then handles ~by xs else [] in
  Prim.iteri (fun i x -> Prim.alive ~by i x) op;
  let read = unhandled hs op in
  claim ~by hs;
  read

let resolve (type d a) ~by (xs : d Value.any list)
    (k : ('v 's. ('v, 's, d) Value.t -> ('v, 's, d) Value.t) -> a) : a =
  let hs =
    if List.exists (fun (Value.Any x) -> is_handle x) xs then handles ~by xs
    else []
  in
  let alive () = List.iteri (fun i (Value.Any x) -> Prim.alive ~by i x) xs in
  alive ();
  match hs with
  | [] -> k (fun y -> y)
  | hs ->
      (* The hold shares the donors' memory while [k] runs, so that a consumer
         that claims a handle meanwhile computes into fresh memory and frees
         none. A buffer already dead was consumed by a consumer that claimed
         its handle first, which [alive] names. *)
      let held =
        match Rig.Claim.hold (List.concat_map buffers hs) with
        | h -> h
        | exception (Invalid_argument _ as e) ->
            alive ();
            raise e
      in
      let keys = offered hs in
      Fun.protect
        ~finally:(fun () -> Rig.Claim.release held)
        (fun () ->
          let result = k (fun y -> as_called keys y) in
          claim ~by hs;
          result)

let consume_list ~by xs =
  let hs =
    if List.exists (fun (Value.Any x) -> is_handle x) xs then handles ~by xs
    else []
  in
  List.iteri (fun i (Value.Any x) -> Prim.alive ~by i x) xs;
  let keys = offered hs in
  let read = List.map (fun (Value.Any x) -> Value.Any (as_called keys x)) xs in
  claim ~by hs;
  read

(* [b]'s memory consumed for [by], where rig holds it exclusive: no other value
   reads it. The live buffer over it, if so. *)
let take ~by b =
  Rig.Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
      if Rig.Claim.exclusive c b then
        Some (Rig.Claim.consume c ~why:("donated to " ^ by) b)
      else None)

(* Ends the claimed handles [hs] once [by] read them: memory no other value
   reads, every shard of a value or none, and that [by] did not write its
   result into ([reused]), is consumed and then freed, so that it returns to
   its device once the work on it is done. *)
let consumed ~by ~reused hs =
  List.iter
    (fun (Handle (_, x) as h) ->
      match x with
      | Value.Donated _ ->
          let bufs = buffers h in
          let why = "donated to " ^ by in
          if not (List.exists (fun b -> List.memq b reused) bufs) then
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

