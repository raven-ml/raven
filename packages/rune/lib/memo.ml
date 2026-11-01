(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module P = Nx_kernel.Prog

type key = { prog : P.t; mask : int; codes : int array }

(* A plan and its key, as [last] holds it: the table keeps each one made, so
   that a plan found there becomes [last] without an allocation. *)
type 'v entry = { key : key; v : 'v }

(* A domain's plans. [last] is read without the lock: a systhread switch never
   splits a word's write. The table is read and written under [lock], as
   Stdlib's Hashtbl relinks its buckets while it grows. *)
type 'v state = {
  mutable last : 'v entry option;
  table : (key, 'v entry option) Hashtbl.t;
  lock : Mutex.t;
}

type 'v t = 'v state Domain.DLS.key

let entries = 4096
let fits n = n < Sys.int_size - 1

let operands k =
  Array.init (Iarray.length k.prog.ins) (fun j -> (k.mask lsr j) land 1 = 1)

let create () =
  Domain.DLS.new_key (fun () ->
      { last = None; table = Hashtbl.create 64; lock = Mutex.create () })

external reraise : exn -> 'a = "%reraise"

(* The table read or written under the lock. Written as [Mutex.protect] is,
   nothing allocated between the lock and the handler, so that an
   asynchronous exception, as a signal's, cannot leave the lock held. *)
let[@inline never] lookup st key =
  Mutex.lock st.lock;
  match Hashtbl.find st.table key with
  | e ->
      Mutex.unlock st.lock;
      e
  | exception Not_found ->
      Mutex.unlock st.lock;
      None
  | exception e ->
      Mutex.unlock st.lock;
      reraise e

let[@inline never] add st key v =
  Mutex.lock st.lock;
  match
    if Hashtbl.length st.table >= entries then Hashtbl.reset st.table;
    Hashtbl.replace st.table key v
  with
  | () -> Mutex.unlock st.lock
  | exception e ->
      Mutex.unlock st.lock;
      reraise e

(* [make] runs outside the lock: a plan's derivation may raise, and the
   systhreads of the domain go on reading the table meanwhile. Two threads
   that miss one key both make its plan, and the table keeps the last. *)
let find t prog mask ~same ~codes ~make x =
  let st = Domain.DLS.get t in
  match st.last with
  | Some { key = k; v } when k.prog == prog && k.mask = mask && same k.codes x
    ->
      v
  | _ -> (
      let key = { prog; mask; codes = codes x } in
      match lookup st key with
      | Some { key = k; v } as e ->
          (* The plan's own key holds the program that made it: an equal
             program built since is kept as the last, so that its next call
             finds it without hashing. *)
          st.last <- (if k.prog == prog then e else Some { key; v });
          v
      | None ->
          let v = make key x in
          let e = Some { key; v } in
          add st key e;
          st.last <- e;
          v)

let any _ _ = true
let none _ = [||]
let apply k make = make k

let plain t prog mask ~make =
  find t prog mask ~same:any ~codes:none ~make:apply make
