(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Names of maps, and the gather that addresses one.

   [lanes a x] performs [E_lanes]. The map named [a] answers it with every
   lane's [x] stacked on a new leading axis; every other map passes it on,
   swapping the lanes it batches behind the gathered axis. Unhandled, no map
   named [a] lies around the call and there is one lane. *)

type t = int

let counter = Atomic.make 0
let make () = Atomic.fetch_and_add counter 1

type _ Effect.t +=
  | E_lanes : { axis : t; t_in : ('a, 'b) Nx.t } -> ('a, 'b) Nx.t Effect.t

let lanes axis t_in =
  try Effect.perform (E_lanes { axis; t_in })
  with Effect.Unhandled _ -> Nx.unsqueeze ~axes:[ 0 ] t_in
