(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external reraise : exn -> 'a = "%reraise"

(* [f ()] with [m] locked, or [held ()] where the calling thread holds [m]:
   locking raises there, where waiting would wait for itself. Written as
   [Mutex.protect] is, nothing allocated between the lock and the handler, so
   that an asynchronous exception cannot leave [m] locked. *)
let[@inline never] latched m ~held f =
  match Mutex.lock m with
  | exception Sys_error _ -> held ()
  | () -> (
      match f () with
      | x ->
          Mutex.unlock m;
          x
      | exception e ->
          Mutex.unlock m;
          reraise e)

module Make (K : Hashtbl.HashedType) = struct
  module H = Hashtbl.Make (K)

  type 'v entry = {
    latch : Mutex.t;  (* Held by the thread making the value. *)
    value : 'v option Atomic.t;
  }

  type 'v t = { lock : Mutex.t; table : 'v entry H.t }

  let create () = { lock = Mutex.create (); table = H.create 16 }

  let find t k ~miss make =
    let e =
      Mutex.protect t.lock @@ fun () ->
      match H.find_opt t.table k with
      | Some e -> e
      | None ->
          miss ();
          let e = { latch = Mutex.create (); value = Atomic.make None } in
          H.add t.table k e;
          e
    in
    match Atomic.get e.value with
    | Some v -> v
    | None ->
        (* The thread holding the latch is making the value: a call from inside
           the making, or from a fiber resumed while it is suspended, makes its
           own. Another thread, of any domain, waits. *)
        latched e.latch ~held:make (fun () ->
            match Atomic.get e.value with
            | Some v -> v
            | None ->
                let v = make () in
                Atomic.set e.value (Some v);
                v)
end
