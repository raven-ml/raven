(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Totals: write-only sums that code anywhere inside a function adds to, read by
   the scope that collects them once the function returns.

   [add t v] performs [E_add]; unhandled, no scope of [t] is open and the
   addition is dropped. A transformation passes an addition on in the context of
   its own: vmap sums its lanes, reverse drops one it makes again while
   rerunning code, the others pass it as it is.

   The scope discharges its total itself. Code a handler runs away from its call
   site, a staged scan's step or a remat's function, runs under a nested scope
   started at zero whose sum leaves as a value: an extra carry leaf of the scan,
   an extra result of the remat. The scope adds it. A trace that another claimer
   restarts discards its additions with its carry, and a replay computes them
   again. *)

type ('a, 'b) t = ('a, 'b) Nx.t Type.Id.t

let make () = Type.Id.make ()

type _ Effect.t += E_add : ('a, 'b) t * ('a, 'b) Nx.t -> unit Effect.t

let add t v = try Effect.perform (E_add (t, v)) with Effect.Unhandled _ -> ()

(* [dropping f] is [f ()] with every addition it makes dropped. *)
let dropping f =
  let rule : type c. c Effect.t -> (unit -> c) option = function
    | E_add _ -> Some (fun () -> ())
    | _ -> None
  in
  let effc : type c. c Effect.t -> ((c, _) Effect.Deep.continuation -> _) option
      =
   fun eff -> Option.map Answer.deliver (rule eff)
  in
  Effect.Deep.match_with f () { retc = Fun.id; exnc = raise; effc }

let rec collect : type a b r.
    (a, b) t -> zero:(a, b) Nx.t -> (unit -> r) -> r * (a, b) Nx.t =
 fun t ~zero f ->
  let open Effect.Deep in
  let dtype = Nx.dtype zero and shape = Nx.shape zero in
  let total = ref zero in
  let receive v =
    if Nx.shape v <> shape then
      invalid_arg
        (Printf.sprintf
           "Rune.Total.add: shape [%s] does not match the total's [%s]"
           (Structure.shape_string (Nx.shape v))
           (Structure.shape_string shape));
    total := Nx.add !total v
  in
  (* The step passes on with one more carry leaf, the sum of the received step's
     additions, started at zero every time the scan runs. *)
  let stage (req : Scan.scan_req) =
    let nc = List.length req.req_carry in
    let run c x =
      let c, s = Scan.split nc c in
      let s = Nx.unpack dtype (List.hd s) in
      let (c', y), s' = collect t ~zero:s (fun () -> req.req_step.run c x) in
      (c' @ [ Nx.P s' ], y)
    in
    let res =
      Effect.perform
        (Scan.E_scan
           {
             req with
             req_carry = req.req_carry @ [ Nx.P (Nx.zeros_like zero) ];
             req_step = { run };
           })
    in
    let c, s = Scan.split nc res.r_carry in
    receive (Nx.unpack dtype (List.hd s));
    { res with r_carry = c }
  in
  let rule : type c. c Effect.t -> (unit -> c) option = function
    | E_add (t', v) -> (
        match Type.Id.provably_equal t t' with
        | Some Type.Equal -> Some (fun () -> receive v)
        | None -> None)
    (* A scan no stager lies beyond is declined: its performer folds it where it
       performed it, inside this scope, past no handler. *)
    | Scan.E_scan_probe -> Some Scan.probe
    | Scan.E_scan req ->
        Some
          (fun () ->
            Scan.pass_on
              ~fold:(fun () -> raise Scan.Not_staged)
              (fun () -> stage req))
    | Remat.E_remat (Remat.Call { params_s; result_s; params; f; residuals }) ->
        Some
          (fun () ->
            let f params =
              collect t ~zero:(Nx.zeros_like zero) (fun () -> f params)
            in
            let y, s =
              Remat.run
                (Remat.Call
                   {
                     params_s;
                     result_s = Nx.Ptree.pair result_s Nx.Ptree.tensor;
                     params;
                     f;
                     residuals;
                   })
            in
            receive s;
            y)
    | _ -> None
  in
  let effc : type c. c Effect.t -> ((c, _) continuation -> _) option =
   fun eff -> Option.map Answer.deliver (rule eff)
  in
  (* The scope interprets its operations as they are, so that a compiled
     function called inside it runs as code whose additions reach it. *)
  match_with
    (fun () -> Nx_effect.intercept { run = Nx_effect.eval } f)
    ()
    { retc = (fun r -> (r, !total)); exnc = raise; effc }
