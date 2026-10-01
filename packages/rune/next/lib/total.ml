(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let shape_mismatch v zero =
  invalid_arg
    (Format.asprintf "Rune.Total.add: shape %a does not match the total's %a"
       Nx.pp_shape (Nx.shape v) Nx.pp_shape (Nx.shape zero))

(* [threaded t ~zero r] is the scan [r] with one more carry leaf, the sum of
   each run of the step's additions to [t] from the carried sum, performed
   outward; it is the result and the final sum. *)
let rec threaded : type a b.
    (a, b) Construct.total ->
    zero:(a, b) Nx.t ->
    Scan.request ->
    Scan.result * (a, b) Nx.t =
 fun t ~zero r ->
  let n = List.length r.req_carry in
  let split l = (List.filteri (fun i _ -> i < n) l, List.nth l n) in
  let req_step c x =
    let c, s = split c in
    let (c', y), s' =
      collect t ~zero:(Nx.unpack (Nx.dtype zero) s) (fun () -> r.req_step c x)
    in
    (c' @ [ Nx.P s' ], y)
  in
  let result =
    Construct.perform
      (Scan
         {
           r with
           req_carry = r.req_carry @ [ Nx.P (Nx.zeros_like zero) ];
           req_step;
         })
  in
  let carry, s = split result.r_carry in
  ({ result with r_carry = carry }, Nx.unpack (Nx.dtype zero) s)

and collect : type a b r.
    (a, b) Construct.total -> zero:(a, b) Nx.t -> (unit -> r) -> r * (a, b) Nx.t
    =
 fun t ~zero f ->
  let total = ref zero in
  let receive v =
    if Nx.shape v <> Nx.shape zero then shape_mismatch v zero;
    total := Nx.add !total v
  in
  let answer : type c. c Construct.t -> (unit -> c) option =
   fun c ->
    match[@warning "@4@8"] c with
    | Add (t', v) -> (
        match Type.Id.provably_equal t t' with
        | Some Equal -> Some (fun () -> receive v)
        | None -> None)
    | Scan r ->
        Some
          (fun () ->
            let result, s = threaded t ~zero r in
            receive s;
            result)
    | Remat _ | Barrier _ | Custom _ | Lanes _ | Lane_index _ | Lane_count _
    | Detach _ ->
        None
  in
  let marker = { Nx.Op.run = Nx.Op.eval; claims = (fun _ -> false) } in
  let r = Construct.install { op = Some marker; call = answer } f in
  (r, !total)

let rec discarding : type r. (unit -> r) -> r =
 fun f ->
  let answer : type c. c Construct.t -> (unit -> c) option =
   fun c ->
    match[@warning "@4@8"] c with
    | Add _ -> Some ignore
    | Scan r ->
        let req_step c x = discarding (fun () -> r.req_step c x) in
        Some (fun () -> Construct.perform (Scan { r with req_step }))
    | Remat _ | Barrier _ | Custom _ | Lanes _ | Lane_index _ | Lane_count _
    | Detach _ ->
        None
  in
  Construct.install { op = None; call = answer } f
