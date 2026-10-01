(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let assoc cell latch k make =
  match List.assq_opt k (Atomic.get cell) with
  | Some v -> v
  | None -> (
      Mutex.protect latch @@ fun () ->
      match List.assq_opt k (Atomic.get cell) with
      | Some v -> v
      | None ->
          let v = make () in
          Atomic.set cell ((k, v) :: Atomic.get cell);
          v)

module Make (K : Hashtbl.HashedType) = struct
  module H = Hashtbl.Make (K)

  type 'v entry = { latch : Mutex.t; value : 'v option Atomic.t }
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
    | None -> (
        Mutex.protect e.latch @@ fun () ->
        match Atomic.get e.value with
        | Some v -> v
        | None ->
            let v = make () in
            Atomic.set e.value (Some v);
            v)
end
