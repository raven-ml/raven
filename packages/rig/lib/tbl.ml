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

  let rec find k = function
    | Nil -> Nil
    | Cell c as cell -> if K.equal c.key k then cell else find k c.next

  let find_opt t k =
    match find k t.buckets.(index t.buckets k) with
    | Cell c -> Some c.v
    | Nil -> None

  let mem t k = find k t.buckets.(index t.buckets k) != Nil

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

  (* A new binding grows the table first, then links its cell: the link is
     the last store, so the call has bound [k] only once it returns. *)
  let replace t k v =
    match find k t.buckets.(index t.buckets k) with
    | Cell c -> c.v <- v
    | Nil ->
        if t.size >= 2 * Array.length t.buckets then grow t;
        let i = index t.buckets k in
        let cell = Cell { key = k; v; next = t.buckets.(i) } in
        t.size <- t.size + 1;
        t.buckets.(i) <- cell

  (* Unlinks [k]'s cell from the bucket whose cell [prev] precedes [cell]. *)
  let rec unlink t i k prev = function
    | Nil -> ()
    | Cell c when K.equal c.key k ->
        (match prev with
        | Nil -> t.buckets.(i) <- c.next
        | Cell p -> p.next <- c.next);
        t.size <- t.size - 1
    | Cell c as cell -> unlink t i k cell c.next

  let remove t k =
    let i = index t.buckets k in
    unlink t i k Nil t.buckets.(i)

  let rec fold_cells f acc = function
    | Nil -> acc
    | Cell c -> fold_cells f (f c.key c.v acc) c.next

  let fold f t acc =
    Array.fold_left (fun acc cell -> fold_cells f acc cell) acc t.buckets

  let reset t =
    t.buckets <- Array.make 8 Nil;
    t.size <- 0

  let trim t = if t.size = 0 && Array.length t.buckets > 8 then reset t
end
