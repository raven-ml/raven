(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Write-only totals.

   A total is a sum that code anywhere inside a function adds to, and that the
   caller reads when the function returns: [collect t ~zero f] is [f ()] with
   [zero] plus everything [f] added to [t]. Nothing reads a total before its
   [collect] returns, so an addition never changes a value the function
   computes, and an addition with no open scope is unobservable.

   The scope that collects a total owns it and threads it through scans and
   remats itself, so a staged loop stays one loop, a replay recomputes the total
   and a restarted trace discards its additions. It discharges its total by one
   rule: code a handler runs away from its call site runs under a nested scope
   started at [Nx.zeros_like zero], its sum leaves as a value, and the scope
   adds it. An addition that crosses a transformation is transformed by that
   transformation's handler: jvp passes it on (it has no tangent), vmap replaces
   it with the sum over its lanes, and reverse drops it on a rerun tape. *)

type ('a, 'b) t = { id : int }

let counter = Atomic.make 0
let make () = { id = Atomic.fetch_and_add counter 1 }
let same a b = a.id = b.id

type _ Effect.t += E_total_add : ('a, 'b) t * ('a, 'b) Nx.t -> unit Effect.t

(* [perform t v] adds [v] to the innermost open scope of [t], if any: with no
   scope the addition is dropped. Handlers re-perform it in their own context
   after transforming [v]. *)
let perform t v =
  match Effect.perform (E_total_add (t, v)) with
  | () -> ()
  | exception Effect.Unhandled _ -> ()

let add = perform

(* [with_no_additions f] runs [f] with every addition dropped. Reverse runs the
   functions it runs in its own context — a custom call's [fwd], a custom_jvp's
   [f] — under it whenever it re-runs them over a rerun tape. *)
let with_no_additions : type r. (unit -> r) -> r =
 fun f ->
  let open Effect.Deep in
  let effc : type c. c Effect.t -> ((c, _) continuation -> _) option =
   fun eff ->
    match eff with E_total_add _ -> Some (fun k -> continue k ()) | _ -> None
  in
  match_with f () { retc = Fun.id; exnc = raise; effc }

let err_shape what a b =
  invalid_arg
    (Printf.sprintf
       "Rune.Total: %s shape [%s] does not match the scope's zero [%s]" what
       (Structure.shape_string (Nx.shape a))
       (Structure.shape_string (Nx.shape b)))

(* The scope of a total. It accumulates every addition made to its total while
   the body runs, and passes everything else on. *)
let rec scope : type a b r.
    (a, b) t ->
    zero:(a, b) Nx.t ->
    (a, b) Nx.t ref ->
    (r, r) Effect.Deep.handler =
 fun t ~zero acc ->
  let open Effect.Deep in
  let effc : type c. c Effect.t -> ((c, _) continuation -> _) option =
   fun eff ->
    match eff with
    | E_total_add (t', v) -> (
        match
          (same t t', Nx_dtype.equal_witness (Nx.dtype v) (Nx.dtype zero))
        with
        | true, Some Type.Equal ->
            if Nx.shape v <> Nx.shape zero then
              err_shape "the addition's" v zero;
            Some
              (fun k ->
                acc := Nx.add !acc v;
                continue k ())
        | true, None ->
            Some
              (fun _ ->
                invalid_arg
                  (Printf.sprintf
                     "Rune.Total: the addition's dtype %s does not match the \
                      scope's zero %s"
                     (Nx_dtype.to_string (Nx.dtype v))
                     (Nx_dtype.to_string (Nx.dtype zero))))
        | false, _ -> None)
    | Scan.E_scan_probe -> Some (fun k -> continue k (Scan.probe ()))
    | Scan.E_scan req -> Some (fun k -> stage_scan k t ~zero ~acc req)
    | Remat.E_remat (Remat.Call { params_s; result_s; params; f; residuals }) ->
        Some
          (fun k ->
            stage_remat k t ~zero ~acc ~params_s ~result_s ~params ~f ~residuals)
    | _ -> None
  in
  { retc = Fun.id; exnc = raise; effc }

(* A staged scan. When a stager lies beyond, the scope passes the scan on with
   one more carry leaf, [Nx.zeros_like zero], and a step that runs the received
   step under a nested scope started at that leaf and returns [c' @ [total']];
   the scope adds the final leaf. Otherwise it declines with [Not_staged], and
   the performer folds where it performed the scan, inside the scope, so each
   step's additions reach the scope as they are made. *)
and stage_scan : type a b r.
    (Scan.scan_res, r) Effect.Deep.continuation ->
    (a, b) t ->
    zero:(a, b) Nx.t ->
    acc:(a, b) Nx.t ref ->
    Scan.scan_req ->
    r =
 fun k t ~zero ~acc req ->
  let open Effect.Deep in
  if not (Scan.probe ()) then discontinue k Scan.Not_staged
  else begin
    let n = List.length req.req_carry in
    let carry_zero = Nx.zeros_like zero in
    let step c x =
      let c, total = Scan.split n c in
      let total =
        match total with
        | [ Nx.P leaf ] -> Nx.unpack (Nx.dtype zero) (Nx.P leaf)
        | _ -> assert false
      in
      let acc_nested = ref total in
      let nested = scope t ~zero:total acc_nested in
      let c', y = match_with (fun () -> req.req_step.run c x) () nested in
      (c' @ [ Nx.P !acc_nested ], y)
    in
    let req =
      {
        req with
        req_carry = req.req_carry @ [ Nx.P carry_zero ];
        req_step = { run = step };
      }
    in
    match Effect.perform (Scan.E_scan req) with
    | res ->
        let c, total = Scan.split n res.r_carry in
        let total =
          match total with
          | [ Nx.P leaf ] -> Nx.unpack (Nx.dtype zero) (Nx.P leaf)
          | _ -> assert false
        in
        acc := Nx.add !acc total;
        continue k { res with r_carry = c }
    | exception Scan.Not_staged -> discontinue k Scan.Not_staged
    | exception e -> discontinue k e
  end

(* A remat. The scope passes on the remat of [f'], which runs [f] under a nested
   scope started at [Nx.zeros_like zero] and returns its sum as an extra result;
   the scope adds that sum. A recompute of [f'] discards the extra result. *)
and stage_remat : type a b r q p.
    (q, r) Effect.Deep.continuation ->
    (a, b) t ->
    zero:(a, b) Nx.t ->
    acc:(a, b) Nx.t ref ->
    params_s:p Nx.Ptree.t ->
    result_s:q Nx.Ptree.t ->
    params:p ->
    f:(p -> q) ->
    residuals:bool ->
    r =
 fun k t ~zero ~acc ~params_s ~result_s ~params ~f ~residuals ->
  let open Effect.Deep in
  let f' params =
    let nested_zero = Nx.zeros_like zero in
    let acc_nested = ref nested_zero in
    let nested = scope t ~zero:nested_zero acc_nested in
    let y = match_with (fun () -> f params) () nested in
    (y, !acc_nested)
  in
  let y, total =
    Remat.run
      (Remat.Call
         {
           params_s;
           result_s = Nx.Ptree.pair result_s Nx.Ptree.tensor;
           params;
           f = f';
           residuals;
         })
  in
  acc := Nx.add !acc total;
  continue k y

(* [collect t ~zero f] runs [f] with [t]'s scope open and returns its result
   with the sum of the additions made to [t]. The scope is a transformation:
   [jit] inside it runs its function eagerly, as it does inside [grad] and
   [vmap]. *)
let collect : type a b.
    (a, b) t -> zero:(a, b) Nx.t -> (unit -> 'r) -> 'r * (a, b) Nx.t =
 fun t ~zero f ->
  let acc = ref (Nx.zeros_like zero) in
  let handler = scope t ~zero acc in
  let r =
    Gate.with_transform (fun () ->
        Effect.Deep.match_with (fun () -> f ()) () handler)
  in
  (r, !acc)
