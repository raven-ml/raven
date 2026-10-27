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

let donate (type v s d) ~by (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  Prim.alive ~by x;
  match x with
  | Value.Array r ->
      let shape = Iarray.to_array (L.shape (A.layout r.a)) in
      Value.Handle { donor = x; at = r.at; shape; arrays = [| r.a |] }
  | Value.Shards r ->
      Value.Handle { donor = x; at = r.at; shape = r.shape; arrays = r.arrays }
  | Value.Handle _ | Value.Deferred _ | Value.Traced _ -> x

(* A handle's arrays as a live value, which the operation reads. *)
let live (type v s d) (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  match x with
  | Value.Handle { at; shape; arrays; _ } -> Prim.of_arrays at ~shape arrays
  | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> x

let is_handle (type v s d) (x : (v, s, d) Value.t) =
  match x with
  | Value.Handle _ -> true
  | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> false

(* A handle's arrays, which an interpretation or a function of several
   operations may keep: shared, so that no later donation writes where they
   read. *)
let lent (type v s d) (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  match x with
  | Value.Handle _ ->
      let y = live x in
      Iarray.iter (fun a -> Rig.Claim.share (A.buffer a)) (Prim.arrays y);
      y
  | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> x

let read ~by x =
  Prim.alive ~by x;
  lent x

(* Donors *)

(* A handle among a consumer's operands, at its position. *)
type handle = At : int * ('v, 's, 'd) Value.t -> handle

(* Whether [x] and [y] are one value. Donors are mutable blocks, each made
   once, so physical equality tells them; their types are an operand's,
   whose dtype and brand are existential, so it is taken on their
   representations. *)
let same x y = Obj.repr x == Obj.repr y

(* Whether [y] is the donor of the handle [h]. *)
let donates (At (_, x)) y =
  match x with
  | Value.Handle { donor; _ } -> same donor y
  | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> false

(* Whether the handles [h] and [h'] have one donor. *)
let one_donor h (At (_, x')) =
  match x' with
  | Value.Handle { donor; _ } -> donates h donor
  | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> false

(* The handles among [xs], the first of each donor's: two handles of one
   donor are read as a handle beside its donor is, and the donor dies, and its
   memory is consumed, once. *)
let handles xs =
  let rec first seen = function
    | [] -> []
    | h :: rest ->
        if List.exists (one_donor h) seen then first seen rest
        else h :: first (h :: seen) rest
  in
  first []
    (List.filter (fun (At (_, x)) -> is_handle x)
       (List.mapi (fun i (Value.Any x) -> At (i, x)) xs))

(* Claims

   A consumer claims each handle's donor in operand order, setting the
   donor's word from [Live] to its death, for good; a donor that died first,
   on any domain, raises naming its consumer. Two consumers racing for one
   donor is a misuse: one of them raises, possibly both, and a donor a loser
   claimed before it raised stays dead. [taken] holds the buffers whose
   memory {!reuse} gave a result, dead since, and [ended] guards the steps
   after {!consumed}. *)
type 'd claim = {
  by : string;
  operands : 'd Value.any array;
  handles : handle list;
  mutable taken : Rig.Buffer.t list;
  mutable ended : bool;
}

(* The word of a donor, an [Array] or [Shards]. *)
let word (type v s d) (donor : (v, s, d) Value.t) : Value.death Atomic.Loc.t =
  match donor with
  | Value.Array r -> [%atomic.loc r.dead]
  | Value.Shards r -> [%atomic.loc r.dead]
  | Value.Handle _ | Value.Deferred _ | Value.Traced _ ->
      invalid_arg "Donate.claim: a donor is an array or shards"

(* Sets the donor word [w] of the handle [x] from [Live] to [died], or raises
   naming the consumer it died to first, on any domain. The compare-and-set
   fails only where another consumer set the word since it was read, which
   the next read finds. *)
let rec kill ~by x w died =
  match Atomic.Loc.get w with
  | Value.Died earlier -> died_to ~by x earlier
  | Value.Live ->
      if not (Atomic.Loc.compare_and_set w Value.Live died) then
        kill ~by x w died

let claim ~by xs =
  let handles = handles xs in
  let died = Value.Died by in
  List.iter
    (fun (At (_, x)) ->
      match x with
      | Value.Handle { donor; _ } -> kill ~by x (word donor) died
      | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ ->
          ())
    handles;
  { by; operands = Array.of_list xs; handles; taken = []; ended = false }

(* [y] as the consumer of the handles [hs] reads it: a handle as its arrays, a
   donor of one of [hs] as a live value, and any other value as it is. *)
let as_read (type v s d) hs (y : (v, s, d) Value.t) : (v, s, d) Value.t =
  let ours () = List.exists (fun h -> donates h y) hs in
  match y with
  | Value.Handle _ -> live y
  | Value.Array r when ours () -> Value.Array { r with dead = Value.Live }
  | Value.Shards r when ours () -> Value.Shards { r with dead = Value.Live }
  | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> y

let called c op = Prim.map (fun y -> as_read c.handles y) op

let operand c i =
  if c.ended then invalid_argf "%s: the consumer ended its handles" c.by;
  if i < 0 || i >= Array.length c.operands then
    invalid_argf "%s: no operand %d of %d" c.by i (Array.length c.operands);
  c.operands.(i)

(* Reuse *)

(* Whether [b] and [b'] are one buffer, or live and share memory: a dead
   buffer's place may lie in memory made since its memory was freed. *)
let share b b' =
  b == b'
  || Option.is_none (Rig.Buffer.dead b)
     && Option.is_none (Rig.Buffer.dead b')
     && Rig.Buffer.overlaps b b'

(* Whether [a] and [a'] are one buffer placing each index at the same bytes
   as the same dtype. *)
let identical a a' =
  A.buffer a == A.buffer a'
  && D.equal (A.dtype a) (A.dtype a')
  && L.equal (A.layout a) (A.layout a')

(* Whether [a] may hold a result of [dt] and [layout] in its memory: of that
   dtype and layout, over all of its memory. *)
let holds dt layout a =
  D.equal (A.dtype a) dt
  && L.equal (A.layout a) layout
  && Rig.Buffer.spans (A.buffer a)

(* [b]'s memory consumed for [by], where rig holds it exclusive: no other value
   reads it. The live buffer over it, if so. *)
let take ~by b =
  Rig.Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
      if Rig.Claim.exclusive c b then
        Some (Rig.Claim.consume c ~why:("donated to " ^ by) b)
      else None)

let has_bytes (type v s d) (x : (v, s, d) Value.t) =
  match x with
  | Value.Array _ | Value.Shards _ | Value.Handle _ -> true
  | Value.Deferred _ | Value.Traced _ -> false

let claimed c i = List.exists (fun (At (j, _)) -> j = i) c.handles

let reuse (type w r d) (c : d claim) ~own (dt : (w, r) D.t) layout :
    (Rig.Buffer.t * (w, r) A.t) option =
  let n = Array.length c.operands in
  (* The buffer of operand [i], a claimed handle on one device whose array
     holds the result, if every operand lies apart from its memory but those
     at positions of [own], which may read it as that array places it and no
     other way: a view of the array in another layout reads other indices of
     it. *)
  let fits i =
    let (Value.Any x) = operand c i in
    match x with
    | Value.Handle { arrays; _ } when Iarray.length arrays = 1 && claimed c i
      ->
        let a = Iarray.get arrays 0 in
        let b = A.buffer a in
        let apart j a' =
          (not (share (A.buffer a') b)) || (List.mem j own && identical a a')
        in
        let rec alone j =
          j = n
          ||
          let (Value.Any y) = c.operands.(j) in
          ((not (has_bytes y)) || Iarray.for_all (apart j) (Prim.arrays y))
          && alone (j + 1)
        in
        if holds dt layout a && (not (List.memq b c.taken)) && alone 0 then
          Some b
        else None
    | Value.Handle _ | Value.Array _ | Value.Shards _ | Value.Deferred _
    | Value.Traced _ ->
        None
  in
  List.find_map
    (fun i ->
      match fits i with
      | None -> None
      | Some b -> (
          match take ~by:c.by b with
          | Some b' ->
              c.taken <- b :: c.taken;
              Some (b, A.v dt layout b')
          | None -> None))
    own

(* Ending *)

(* The buffers of a handle, one per device. *)
let buffers (At (_, x)) = Iarray.to_list (Iarray.map A.buffer (Prim.arrays x))

(* Whether no two of [bufs] share memory. Shards placed by borrowing lie in
   one allocation, where each reads the memory the other holds. *)
let rec apart = function
  | [] -> true
  | b :: rest -> (not (List.exists (Rig.Buffer.overlaps b) rest)) && apart rest

(* Memory no other value reads, every shard of a value or none, and that no
   result took, is consumed and then freed, so that it returns to its device
   once the work on it is done. Shards that share memory with one another are
   not consumed: each is read by the other, and rig donates no two buffers
   over one memory together. *)
let consumed c =
  c.ended <- true;
  let why = "donated to " ^ c.by in
  List.iter
    (fun h ->
      let bufs = buffers h in
      if (not (List.exists (fun b -> List.memq b c.taken) bufs)) && apart bufs
      then
        Rig.Claim.with_ ~read:[] ~donate:bufs (fun cl ->
            if List.for_all (Rig.Claim.exclusive cl) bufs then
              List.map (Rig.Claim.consume cl ~why) bufs
            else [])
        |> List.iter (fun b -> ignore (Rig.Buffer.free ~why b : bool)))
    c.handles

(* Consumers *)

let with_consumer ~by xs f =
  List.iter (fun (Value.Any x) -> Prim.alive ~by x) xs;
  let c = claim ~by xs in
  (* The claim ends on a raise too: its donors are dead, so nothing else
     would return their memory. *)
  Fun.protect ~finally:(fun () -> consumed c) (fun () -> f c)

let arrays c i =
  let (Value.Any x) = operand c i in
  match x with
  | Value.Deferred _ | Value.Traced _ ->
      invalid_argf "%s: operand %d, %a, has no bytes" c.by i Prim.pp_value x
  | Value.Array _ | Value.Shards _ | Value.Handle _ -> Prim.any (Prim.arrays x)
