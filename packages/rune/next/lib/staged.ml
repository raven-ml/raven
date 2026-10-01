(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let install s f =
  let call : type r. r Construct.t -> (unit -> r) option =
   fun c ->
    match[@warning "@4@8"] c with
    | Detach x -> Some (fun () -> x)
    | Scan _ | Remat _ | Barrier _ | Custom _ | Lanes _ | Lane_index _
    | Lane_count _ | Add _ ->
        None
  in
  let op = { Nx.Op.run = (fun o -> Lower.op s o); claims = (fun _ -> true) } in
  Construct.install { op = Some op; call } f
