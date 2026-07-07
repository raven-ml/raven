(**************************************************************************)
(*                                                                        *)
(*  Copyright (C) Jean-Christophe Filliatre                               *)
(*                                                                        *)
(*  This software is free software; you can redistribute it and/or        *)
(*  modify it under the terms of the GNU Library General Public           *)
(*  License version 2.1, with the special exception on linking            *)
(*  described in file LICENSE.                                            *)
(*                                                                        *)
(*  This software is distributed in the hope that it will be useful,      *)
(*  but WITHOUT ANY WARRANTY; without even the implied warranty of        *)
(*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                  *)
(*                                                                        *)
(**************************************************************************)

(* Vendored from https://github.com/backtracking/hashcons
   Modifications:
   - Removed the non-functorial generic interface.
   - Removed [Hmap] and [Hset]. *)

type +'a hash_consed = { hkey : int; tag : int; node : 'a }

let tag_counter = ref 0
let gentag () = incr tag_counter; !tag_counter
let gentag_peek () = !tag_counter

module type HashedType = sig
  type t
  val equal : t -> t -> bool
  val hash : t -> int
end

module type S = sig
  type key
  type t
  val create : int -> t
  val clear : t -> unit
  val hashcons : t -> key -> key hash_consed
  val iter : (key hash_consed -> unit) -> t -> unit
  val stats : t -> int * int * int * int * int * int
end

module Make (H : HashedType) : S with type key = H.t = struct
  type key = H.t
  type data = H.t hash_consed

  type t = {
    mutable table : data Weak.t array;
    mutable totsize : int;
    mutable limit : int;
  }

  let emptybucket = Weak.create 0

  let create sz =
    let sz = if sz < 7 then 7 else sz in
    let sz = if sz > Sys.max_array_length then Sys.max_array_length else sz in
    { table = Array.make sz emptybucket; totsize = 0; limit = 3 }

  let clear t =
    for i = 0 to Array.length t.table - 1 do
      t.table.(i) <- emptybucket
    done;
    t.totsize <- 0;
    t.limit <- 3

  let iter f t =
    let rec iter_bucket i b =
      if i >= Weak.length b then ()
      else
        match Weak.get b i with
        | Some v -> f v; iter_bucket (i + 1) b
        | None -> iter_bucket (i + 1) b
    in
    Array.iter (iter_bucket 0) t.table

  let count t =
    let rec count_bucket i b accu =
      if i >= Weak.length b then accu
      else count_bucket (i + 1) b (accu + if Weak.check b i then 1 else 0)
    in
    Array.fold_right (count_bucket 0) t.table 0

  let next_sz n = min (3 * n / 2 + 3) (Sys.max_array_length - 1)

  let rec resize t =
    let oldlen = Array.length t.table in
    let newlen = next_sz oldlen in
    if newlen > oldlen then begin
      let newt = create newlen in
      newt.limit <- t.limit + 100;
      iter (fun d -> add newt d) t;
      t.table <- newt.table
    end

  and add t d =
    let index = d.hkey mod Array.length t.table in
    let bucket = t.table.(index) in
    let sz = Weak.length bucket in
    let rec loop i =
      if i >= sz then begin
        let newsz = min (3 * sz / 2 + 3) (Sys.max_array_length - 1) in
        if newsz <= sz then
          failwith "Hashcons.Make: hash bucket cannot grow more";
        let newbucket = Weak.create newsz in
        Weak.blit bucket 0 newbucket 0 sz;
        Weak.set newbucket i (Some d);
        t.table.(index) <- newbucket;
        t.totsize <- t.totsize + (newsz - sz);
        if t.totsize > t.limit * Array.length t.table then resize t
      end else begin
        if Weak.check bucket i then loop (i + 1)
        else Weak.set bucket i (Some d)
      end
    in
    loop 0

  let hashcons t d =
    let hkey = H.hash d land max_int in
    let index = hkey mod Array.length t.table in
    let bucket = t.table.(index) in
    let sz = Weak.length bucket in
    let rec loop i =
      if i >= sz then begin
        let hnode = { hkey; tag = gentag (); node = d } in
        add t hnode;
        hnode
      end else
        match Weak.get bucket i with
        | Some v when H.equal v.node d -> begin
            match Weak.get bucket i with
            | Some v -> v
            | None -> loop (i + 1)
          end
        | _ -> loop (i + 1)
    in
    loop 0

  let stats t =
    let len = Array.length t.table in
    let lens = Array.map Weak.length t.table in
    Array.sort compare lens;
    let totlen = Array.fold_left ( + ) 0 lens in
    (len, count t, totlen, lens.(0), lens.(len / 2), lens.(len - 1))
end
