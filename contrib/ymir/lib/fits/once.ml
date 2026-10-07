(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A value computed on first use, once: the first domain to ask computes it
   under a lock, and the others asking meanwhile wait for it. A computation
   that raises leaves the value to the next ask. *)

type 'a t = { lock : Mutex.t; value : 'a option Atomic.t; compute : unit -> 'a }

let make compute = { lock = Mutex.create (); value = Atomic.make None; compute }

let of_value v =
  {
    lock = Mutex.create ();
    value = Atomic.make (Some v);
    compute = (fun () -> v);
  }

let is_computed t = Option.is_some (Atomic.get t.value)

let get t =
  match Atomic.get t.value with
  | Some v -> v
  | None ->
      Mutex.protect t.lock (fun () ->
          match Atomic.get t.value with
          | Some v -> v
          | None ->
              let v = t.compute () in
              Atomic.set t.value (Some v);
              v)
