(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Make (K : sig
  type t

  val equal : t -> t -> bool
  val hash : t -> int
end) =
struct
  type key = K.t

  (* A bucket's cells. A cell's [next] changes only to unlink its successor,
     in one store. *)
  type 'a cell =
    | Nil
    | Cell of { key : key; mutable v : 'a; mutable next : 'a cell }

  type 'a t = { mutable buckets : 'a cell array; mutable size : int }

  (* The buckets number a power of two, which a mask indexes. *)
  let rec power n p = if p >= n then p else power n (2 * p)
  let create n = { buckets = Array.make (power n 8) Nil; size = 0 }
  let length t = t.size
  let index buckets k = K.hash k land (Array.length buckets - 1)

  let rec cell k = function
    | Nil -> Nil
    | Cell c as found -> if K.equal c.key k then found else cell k c.next

  let find t k =
    match cell k t.buckets.(index t.buckets k) with
    | Cell c -> c.v
    | Nil -> raise Not_found

  let find_opt t k =
    match cell k t.buckets.(index t.buckets k) with
    | Cell c -> Some c.v
    | Nil -> None

  let mem t k = cell k t.buckets.(index t.buckets k) != Nil

  let rec copy_into buckets = function
    | Nil -> ()
    | Cell c ->
        let i = index buckets c.key in
        buckets.(i) <- Cell { key = c.key; v = c.v; next = buckets.(i) };
        copy_into buckets c.next

  (* Builds the doubled buckets from fresh cells, then publishes them. *)
  let grow t =
    let buckets = Array.make (2 * Array.length t.buckets) Nil in
    Array.iter (copy_into buckets) t.buckets;
    t.buckets <- buckets

  (* Links [cell] at the head of the bucket [i], counted. *)
  let[@poll error] link t i cell =
    t.size <- t.size + 1;
    t.buckets.(i) <- cell

  (* A new binding grows the table first, then links its cell. *)
  let replace t k v =
    match cell k t.buckets.(index t.buckets k) with
    | Cell c -> c.v <- v
    | Nil ->
        if t.size >= 2 * Array.length t.buckets then grow t;
        let i = index t.buckets k in
        link t i (Cell { key = k; v; next = t.buckets.(i) })

  (* Unlinks the cell after [prev] in the bucket [i], or its first if [prev]
     is [Nil], leaving [next] in its place, counted. *)
  let[@poll error] unlinked t i prev next =
    (match prev with Nil -> t.buckets.(i) <- next | Cell p -> p.next <- next);
    t.size <- t.size - 1

  (* Unlinks [k]'s cell from the bucket whose cell [prev] precedes [cell]. *)
  let rec unlink t i k prev = function
    | Nil -> ()
    | Cell c when K.equal c.key k -> unlinked t i prev c.next
    | Cell c as cell -> unlink t i k cell c.next

  let remove t k =
    let i = index t.buckets k in
    unlink t i k Nil t.buckets.(i)

  let rec fold_cells f acc = function
    | Nil -> acc
    | Cell c -> fold_cells f (f c.key c.v acc) c.next

  let fold f t acc =
    Array.fold_left (fun acc cell -> fold_cells f acc cell) acc t.buckets

  let[@poll error] emptied t buckets =
    t.buckets <- buckets;
    t.size <- 0

  let reset t = emptied t (Array.make 8 Nil)

  let trim t = if t.size = 0 && Array.length t.buckets > 8 then reset t
end
