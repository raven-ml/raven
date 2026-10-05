(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let shape_mismatch v zero =
  invalid_arg
    (Format.asprintf "Rune.Total.add: shape %a does not match the total's %a"
       Nx.pp_shape (Nx.shape v) Nx.pp_shape (Nx.shape zero))

(* [threaded t ~zero r] is the loop [r] with one more carry leaf, the sum of
   each run of the step's additions to [t] from the carried sum, performed
   outward; it is the result and the final sum. *)
let rec threaded : type a b.
    (a, b) Construct.total ->
    zero:(a, b) Nx.t ->
    Trips.request ->
    Trips.result * (a, b) Nx.t =
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
  let req_trips : Trips.trips =
    match r.req_trips with
    | Rows _ as rows -> rows
    | Until stop ->
        Until { stop with until = (fun c -> stop.until (fst (split c))) }
  in
  let result =
    Construct.perform
      (Loop
         {
           req_carry = r.req_carry @ [ Nx.P (Nx.zeros_like zero) ];
           req_trips;
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
    | Loop r ->
        Some
          (fun () ->
            let result, s = threaded t ~zero r in
            receive s;
            result)
    | Compiled { p; q; f; args; compiler } ->
        let f (args, zero) = collect t ~zero (fun () -> f args) in
        let p = Nx.Ptree.pair p Nx.Ptree.tensor
        and q = Nx.Ptree.pair q Nx.Ptree.tensor in
        let compiler = compiler.derive (Totals t) in
        Some
          (fun () ->
            let args = (args, Nx.zeros_like zero) in
            let y, s =
              Construct.perform (Compiled { p; q; f; args; compiler })
            in
            receive s;
            y)
    | Root ({ solve; _ } as r) ->
        (* Only derivatives run the other functions, under [discarding]. *)
        let sums = ref [] in
        let solve () =
          let x, s = collect t ~zero:(Nx.zeros_like zero) solve in
          sums := s :: !sums;
          x
        in
        Some
          (fun () ->
            let x = Construct.perform (Root { r with solve }) in
            List.iter receive !sums;
            x)
    | Remat { p; q; f; args; recomputed } ->
        let f args = collect t ~zero:(Nx.zeros_like zero) (fun () -> f args) in
        let q = Nx.Ptree.pair q Nx.Ptree.tensor in
        Some
          (fun () ->
            let y, s =
              Construct.perform (Remat { p; q; f; args; recomputed })
            in
            receive s;
            y)
    | Barrier _ | Custom _ | At_map _ | Lanes _ | Lane_index _ | Lane_count _
    | Detach _ ->
        None
  in
  let r = Construct.install { op = None; call = answer } f in
  (r, !total)

let rec discarding : type r. (unit -> r) -> r =
 fun f ->
  let answer : type c. c Construct.t -> (unit -> c) option =
   fun c ->
    match[@warning "@4@8"] c with
    | Add _ -> Some ignore
    | Loop r ->
        let req_step c x = discarding (fun () -> r.req_step c x) in
        let req_trips : Trips.trips =
          match r.req_trips with
          | Rows _ as rows -> rows
          | Until stop ->
              Until
                {
                  stop with
                  until = (fun c -> discarding (fun () -> stop.until c));
                }
        in
        Some (fun () -> Construct.perform (Loop { r with req_step; req_trips }))
    | Remat ({ f; _ } as r) ->
        let f args = discarding (fun () -> f args) in
        Some (fun () -> Construct.perform (Remat { r with f }))
    | Root r ->
        let residual x = discarding (fun () -> r.residual x)
        and solve () = discarding r.solve
        and linear_solve op b = discarding (fun () -> r.linear_solve op b) in
        Some
          (fun () ->
            Construct.perform (Root { r with residual; solve; linear_solve }))
    | Compiled ({ f; compiler; _ } as c) ->
        let f args = discarding (fun () -> f args) in
        let compiler = compiler.derive Discarding in
        Some (fun () -> Construct.perform (Compiled { c with f; compiler }))
    | Barrier _ | Custom _ | At_map _ | Lanes _ | Lane_index _ | Lane_count _
    | Detach _ ->
        None
  in
  Construct.install { op = None; call = answer } f
