(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A device whose values live in memory of their own, as on a GPU: nx reads them
   through its engine, which counts the reads, and [Nx.data] of one raises. *)

type Nx_effect.storage +=
  | Mem : ('a, 'b) Nx_dtype.t * ('a, 'b) Nx_buffer.t -> Nx_effect.storage

let reads = ref 0

(* The elements view [v] reaches in [mem], in C order. *)
let gather (type a b) (mem : (a, b) Nx_buffer.t) v : (a, b) Nx_buffer.t =
  let shape = Nx_core.View.shape v and strides = Nx_core.View.strides v in
  let dst = Nx_buffer.create (Nx_buffer.dtype mem) (Nx_core.View.numel v) in
  for i = 0 to Nx_buffer.length dst - 1 do
    let idx = Nx_core.Shape.unravel_index i shape in
    let off = ref (Nx_core.View.offset v) in
    Array.iteri (fun d k -> off := !off + (k * strides.(d))) idx;
    Nx_buffer.set dst i (Nx_buffer.get mem !off)
  done;
  dst

let rec engine =
  {
    Nx_effect.read =
      (fun (type a b) (r : (a, b) Nx_effect.resident) : (a, b) Nx_buffer.t ->
        incr reads;
        match r.r_cell.state with
        | Live (Mem (dt, mem)) -> (
            match Nx_dtype.equal_witness dt r.r_dtype with
            | Some Type.Equal -> gather mem r.r_view
            | None -> assert false)
        | _ -> assert false);
    place = (fun p x -> place p x);
  }

and place : type a b.
    Nx_effect.placement -> (a, b) Nx_effect.t -> (a, b) Nx_effect.t =
 fun p x ->
  let src = Nx.to_buffer (Nx.place Nx.Placement.host x) in
  let mem = Nx_buffer.create (Nx_buffer.dtype src) (Nx_buffer.length src) in
  Nx_buffer.blit ~src ~dst:mem;
  Nx_effect.placed p (Nx.dtype x)
    (Nx_core.View.create (Nx.shape x))
    (Nx_effect.cell ~placement:p ~length:(Nx_buffer.length mem)
       (Mem (Nx.dtype x, mem)))

let device = Nx_effect.Device.make "TEST" engine
let place x = Nx.place (Nx.Placement.device device) x
