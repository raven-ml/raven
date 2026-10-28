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
  | Value.Array _ | Value.Shards _ -> Value.Handle { donor = x; over = x }
  | Value.Handle _ | Value.Traced _ -> x

(* A handle's arrays as a live value, which the operation reads: a value of
   its own, since [over] may be the donor, whose word a claim sets. *)
let live (type v s d) (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  match x with
  | Value.Handle { over; _ } ->
      Prim.of_arrays (Prim.placement over) ~shape:(Prim.shape over)
        (Prim.arrays over)
  | Value.Array _ | Value.Shards _ | Value.Traced _ -> x

let is_handle (type v s d) (x : (v, s, d) Value.t) =
  match x with
  | Value.Handle _ -> true
  | Value.Array _ | Value.Shards _ | Value.Traced _ -> false

(* A handle's arrays, which an interpretation or a function of several
   operations may keep: shared, so that no later donation writes where they
   read. *)
let lent (type v s d) (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  match x with
  | Value.Handle _ ->
      let y = live x in
      Iarray.iter (fun a -> Rig.Claim.share (A.buffer a)) (Prim.arrays y);
      y
  | Value.Array _ | Value.Shards _ | Value.Traced _ -> x

let read ~by x =
  Prim.alive ~by x;
  lent x

(* Donors *)

(* Whether [x] and [y] are one value. Donors are mutable blocks, each made
   once, so physical equality tells them; their types are an operand's,
   whose dtype and brand are existential, so it is taken on their
   representations. *)
let same x y = Obj.repr x == Obj.repr y

(* Whether [y] is the donor of the handle [h]. *)
let donates (Value.Any h) y =
  match h with
  | Value.Handle { donor; _ } -> same donor y
  | Value.Array _ | Value.Shards _ | Value.Traced _ -> false

(* Claims

   A consumer claims each handle's donor in operand order, setting the
   donor's word from [Live] to its death, for good; a donor that died first,
   on any domain, raises naming its consumer. Two consumers racing for one
   donor is a misuse: one of them raises, possibly both, and a donor a loser
   claimed before it raised stays dead. [handles] holds every handle among
   [operands], in their order, and [ended] guards the steps after
   {!consumed}. *)
type 'd claim = {
  by : string;
  operands : 'd Value.any array;
  handles : 'd Value.any list;
  mutable ended : bool;
}

(* The word of a donor, an [Array] or [Shards]. *)
let word (type v s d) (donor : (v, s, d) Value.t) : Value.death Atomic.Loc.t =
  match donor with
  | Value.Array r -> [%atomic.loc r.dead]
  | Value.Shards r -> [%atomic.loc r.dead]
  | Value.Handle _ | Value.Traced _ ->
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
  let handles = List.filter (fun (Value.Any x) -> is_handle x) xs in
  let died = Value.Died by in
  (* Each handle kills its donor unless a handle before it has that donor:
     two handles of one donor are read as a handle beside its donor is, and
     the donor dies once, at the first. *)
  let rec kills seen = function
    | [] -> ()
    | (Value.Any x as h) :: rest ->
        (match x with
        | Value.Handle { donor; _ } ->
            if not (List.exists (fun s -> donates s donor) seen) then
              kill ~by x (word donor) died
        | Value.Array _ | Value.Shards _ | Value.Traced _ ->
            ());
        kills (h :: seen) rest
  in
  kills [] handles;
  { by; operands = Array.of_list xs; handles; ended = false }

(* [y] as the consumer of the handles [hs] reads it: a handle as its arrays, a
   donor of one of [hs] as a live value, and any other value as it is. *)
let as_read (type v s d) hs (y : (v, s, d) Value.t) : (v, s, d) Value.t =
  let ours () = List.exists (fun h -> donates h y) hs in
  match y with
  | Value.Handle _ -> live y
  | Value.Array r when ours () -> Value.Array { r with dead = Value.Live }
  | Value.Shards r when ours () -> Value.Shards { r with dead = Value.Live }
  | Value.Array _ | Value.Shards _ | Value.Traced _ -> y

let called c op = Prim.map (fun y -> as_read c.handles y) op

let operand c i =
  if c.ended then invalid_argf "%s: the consumer ended its handles" c.by;
  if i < 0 || i >= Array.length c.operands then
    invalid_argf "%s: no operand %d of %d" c.by i (Array.length c.operands);
  c.operands.(i)

(* Reuse *)

(* Whether no consumption killed [b]: a buffer over memory a {!reuse} took,
   or that the ending of a claim consumed, is dead. *)
let unconsumed b = Option.is_none (Rig.Buffer.dead b)

(* Whether [b] and [b'] are one buffer, or live and share memory: a dead
   buffer's place may lie in memory made since its memory was freed. *)
let share b b' =
  b == b' || (unconsumed b && unconsumed b' && Rig.Buffer.overlaps b b')

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
let take_memory ~by b =
  Rig.Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
      if Rig.Claim.exclusive c b then
        Some (Rig.Claim.consume c ~why:("donated to " ^ by) b)
      else None)

let has_bytes (type v s d) (x : (v, s, d) Value.t) =
  match x with
  | Value.Array _ | Value.Shards _ | Value.Handle _ -> true
  | Value.Traced _ -> false

let reuse (type w r d) (c : d claim) ~own (dt : (w, r) D.t) layout :
    (Rig.Buffer.t * (w, r) A.t) option =
  let n = Array.length c.operands in
  (* The buffer of operand [i], a handle on one device whose array holds the
     result over memory no earlier reuse took, if every operand lies apart
     from its memory but those at positions of [own], which may read it as
     that array places it and no other way: a view of the array in another
     layout reads other indices of it. *)
  let fits i =
    let (Value.Any x) = operand c i in
    match x with
    | Value.Handle { over = Value.Array { a; _ }; _ } ->
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
        if holds dt layout a && unconsumed b && alone 0 then Some b else None
    | Value.Handle _ | Value.Array _ | Value.Shards _ | Value.Traced _ ->
        None
  in
  List.find_map
    (fun i ->
      match fits i with
      | None -> None
      | Some b -> (
          match take_memory ~by:c.by b with
          | Some b' -> Some (b, A.v dt layout b')
          | None -> None))
    own

(* Ending *)

(* The buffers of a handle, one per device, that no consumption killed. *)
let buffers (Value.Any x) =
  List.filter unconsumed (Iarray.to_list (Iarray.map A.buffer (Prim.arrays x)))

(* Whether no two of [bufs] share memory. Shards placed by borrowing lie in
   one allocation, where each reads the memory the other holds. *)
let rec apart = function
  | [] -> true
  | b :: rest -> (not (List.exists (Rig.Buffer.overlaps b) rest)) && apart rest

(* The memory of each handle that no other value reads, every live shard of
   it or none, is consumed and then freed, so that it returns to its device
   once the work on it is done. Memory a result took is dead, and so is
   memory an earlier handle's ending consumed, which the second of two
   handles over one memory finds: each memory is consumed once. Shards that
   share memory with one another are not consumed: each is read by the
   other, and rig donates no two buffers over one memory together. *)
let consumed c =
  c.ended <- true;
  let why = "donated to " ^ c.by in
  List.iter
    (fun h ->
      let bufs = buffers h in
      if apart bufs then
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

(* Operand [i], which must have bytes: a traced value has none. *)
let bytes_at c i =
  let (Value.Any x as y) = operand c i in
  if not (has_bytes x) then
    invalid_argf "%s: operand %d, %a, has no bytes" c.by i Prim.pp_value x;
  y

let arrays c i =
  let (Value.Any x) = bytes_at c i in
  Prim.any (Prim.arrays x)

(* A result written over operand [i] reads it, at any index, through the
   array [take] gives, so no other position may share its memory. *)
let take c i dt l =
  let (Value.Any _) = bytes_at c i in
  Option.map snd (reuse c ~own:[ i ] dt l)
