(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module P = Nx_kernel.Prog

type key = { prog : P.t; mask : int; codes : int array }

(* A domain's plans. [last] is read without the lock: a systhread switch never
   splits a word's write. The table is read and written under [lock], as
   Stdlib's Hashtbl relinks its buckets while it grows. *)
type 'v state = {
  mutable last : (key * 'v) option;
  table : (key, 'v) Hashtbl.t;
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
  match Hashtbl.find_opt st.table key with
  | v ->
      Mutex.unlock st.lock;
      v
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
  | Some (k, v) when k.prog == prog && k.mask = mask && same k.codes x -> v
  | _ ->
      let key = { prog; mask; codes = codes x } in
      let v =
        match lookup st key with
        | Some v -> v
        | None ->
            let v = make key x in
            add st key v;
            v
      in
      st.last <- Some (key, v);
      v

let any _ _ = true
let none _ = [||]
let apply k make = make k

let plain t prog mask ~make =
  find t prog mask ~same:any ~codes:none ~make:apply make
