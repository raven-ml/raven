(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A multiply by an odd constant moves every bit up; the shift brings the high
   ones back down to the low bits a table indexes by. *)
let mix k =
  let h = k * 0x2545F4914F6CDD1D in
  h lxor (h lsr 29)

module Address = struct
  type key = int

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
  let index buckets k = mix k land (Array.length buckets - 1)

  let rec cell k = function
    | Nil -> Nil
    | Cell c as found -> if Int.equal c.key k then found else cell k c.next

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

  (* A new binding grows the table first, then links its cell: the link is
     the last store, so the call has bound [k] only once it returns. *)
  let replace t k v =
    match cell k t.buckets.(index t.buckets k) with
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
    | Cell c when Int.equal c.key k ->
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
end

module Range = Hashtbl.Make (struct
  type t = int * int

  let equal (a, n) (b, m) = Int.equal a b && Int.equal n m
  let hash (a, n) = mix (a lxor (n lsl 1))
end)

module Window = Hashtbl.Make (struct
  type t = Window.t

  (* Windows of two machines may lie at one address: all of a window's fields
     tell it apart. *)
  let equal (a : t) b = a = b
  let hash w = mix (Window.address w)
end)
