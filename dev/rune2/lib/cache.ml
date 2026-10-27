(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Make (K : Hashtbl.HashedType) = struct
  module H = Hashtbl.Make (K)

  type 'v entry = {
    latch : Mutex.t;
    value : 'v option Atomic.t;
    maker : Domain.id option Atomic.t;  (** The domain making it, if any. *)
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
          let e =
            {
              latch = Mutex.create ();
              value = Atomic.make None;
              maker = Atomic.make None;
            }
          in
          H.add t.table k e;
          e
    in
    match Atomic.get e.value with
    | Some v -> v
    | None when Atomic.get e.maker = Some (Domain.self ()) ->
        (* The making runs on this domain: a call from inside it, or from a
           fiber resumed while it is suspended, would wait for itself. *)
        make ()
    | None -> (
        Mutex.protect e.latch @@ fun () ->
        match Atomic.get e.value with
        | Some v -> v
        | None ->
            Atomic.set e.maker (Some (Domain.self ()));
            Fun.protect
              ~finally:(fun () -> Atomic.set e.maker None)
              (fun () ->
                let v = make () in
                Atomic.set e.value (Some v);
                v))
end
