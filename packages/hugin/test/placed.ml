(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A device whose values live in memory of their own, as on a GPU: nx reads them
   through its memory, which counts the reads. *)

type Nx_effect.storage += Mem of Nx_device.Buffer.t

let reads = ref 0

let rec memory =
  {
    Nx_effect.read =
      (fun r ->
        incr reads;
        match r.r_cell.state with
        | Live (Mem mem) -> Nx_array.Elements.gather mem r.r_view
        | _ -> assert false);
    place = (fun p x -> place p x);
  }

and place : type a b.
    Nx_effect.placement -> (a, b) Nx_effect.t -> (a, b) Nx_effect.t =
 fun p x ->
  let x = Nx.place Nx.Placement.host x in
  let mem = Nx_array.Elements.gather (Nx_effect.read x) (Nx_effect.view x) in
  Nx_effect.placed p (Nx.dtype x)
    (Nx_array.View.create (Nx.shape x))
    (Nx_effect.cell ~placement:p
       ~length:(Nx_device.Buffer.length mem)
       (Mem mem))

let device = Nx_effect.Device.make "TEST" memory
let place x = Nx.place (Nx.Placement.device device) x
