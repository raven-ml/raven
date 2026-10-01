(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk_next

(* Whether [x] is a value of the trace: traced by its lowering, or one it
   captures. *)
let ours x =
  match Nx.Repr.v x with
  | Traced t -> (
      match Nx.Repr.Traced.node t with Lower.Uop _ -> true | _ -> false)
  | Host _ | Placed _ -> true

(* The node of [x] if the trace computes it: neither storage it reads, nor a
   constant, nor a value of another transformation. *)
let computed s x =
  if not (ours x) then None
  else
    let u = Lower.value s x in
    match Ops.op (Ops.unsharded_base u) with
    | Op.Buffer | Op.Const -> None
    | _ -> Some u

(* [kept s x] is [x] with its own storage when the trace computes it. *)
let kept s (Nx.P x) =
  match computed s x with
  | Some u ->
      Nx.P (Lower.traced (Nx.placement x) (Nx.dtype x) (Ops.contiguous u))
  | None -> Nx.P x

(* [after s values deps] is each of [values] the trace computes read through a
   copy stored once [deps] exist. *)
let after s values deps =
  let deps =
    List.filter_map
      (fun (Nx.P d) ->
        if ours d then Some (Ops.contiguous (Lower.value s d)) else None)
      deps
  in
  List.map
    (fun (Nx.P x) ->
      match computed s x with
      | None -> Nx.P x
      | Some u ->
          let at = Nx.placement x and dt = Nx.dtype x in
          let copy =
            Lower.output s ~slot:(Ops.unique_num ()) at dt (Nx.shape x)
          in
          Nx.P
            (Lower.traced at dt
               (Ops.after copy
                  [ Ops.store copy (Ops.after (Ops.contiguous u) deps) ])))
    values

let rec install : 'a. Lower.scope -> (unit -> 'a) -> 'a =
 fun s f ->
  let call : type r. r Construct.t -> (unit -> r) option =
   fun c ->
    match[@warning "@4@8"] c with
    | Detach x -> Some (fun () -> x)
    | Remat { recomputed = true; p; f; args; _ } ->
        Some
          (fun () ->
            let leaves, _ = Nx.Ptree.flatten p args in
            let args =
              Nx.Ptree.rebuild p ~like:args (List.map (kept s) leaves)
            in
            install s (fun () -> f args))
    | Barrier { values; after = deps } -> Some (fun () -> after s values deps)
    | Remat { recomputed = false; _ }
    | Scan _ | Custom _ | Lanes _ | Lane_index _ | Lane_count _ | Add _ ->
        None
  in
  let op = { Nx.Op.run = (fun o -> Lower.op s o); claims = (fun _ -> true) } in
  Construct.install { op = Some op; call } f
