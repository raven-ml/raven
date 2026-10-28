(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Program

(* Outputs [i] to [i + n - 1]. *)
let sub = Array.sub

(* Calls *)

(* [f]'s rule traced over stand-ins of [args]: its primal, returning the results
   and then what [inner], traced inside it over stand-ins of [over] the results,
   captures, its residuals. *)
let traced ~name args f =
  let ny = ref 0 and inner = ref None in
  let primal, captures =
    Trace.trace Body ~name (Array.map form args) (fun vs ->
        let ys, over, g = f vs in
        ny := Array.length ys;
        let p, residuals = Trace.trace Body ~name (Array.map form over) g in
        inner := Some p;
        Array.append ys residuals)
  in
  (primal, Option.get !inner, !ny, captures)

(* The innermost interpretation around the call says how the rule runs. Where
   none runs it and no interpretation traces an argument, the call is the
   rule's results and its derivative never runs. A call a tracer stages is
   traced whatever its arguments. *)
let call ~name mode args rule trace_call =
  let untraced (V x) = Option.is_none (Nx.Prim.owner x) in
  if staging () then trace_call ()
  else
    (* A draw without a key in the rule or its derivative raises where either
       runs, as it does where they are traced. *)
    let rule args =
      let ys, d = Nx.Rng.refusing name (fun () -> rule args) in
      (ys, fun ts -> Nx.Rng.refusing name (fun () -> d ts))
    in
    match Program.rule { name; mode; args; rule } with
    | Some run -> run ()
    | None when Array.for_all untraced args -> fst (rule args)
    | None -> trace_call ()

let jvp_call ~name args rule =
  call ~name Forward args rule @@ fun () ->
  let primal, tangent, ny, captures =
    traced ~name args (fun vs ->
        let ys, map = rule vs in
        (ys, vs, map))
  in
  let derivative = Tangent tangent in
  let c =
    Custom
      {
        name;
        primal;
        derivative;
        ny;
        args;
        captures;
        residuals = false;
        applied = false;
        seeds = [||];
      }
  in
  sub (run c) 0 ny

let vjp_call ~name args rule =
  call ~name Reverse args rule @@ fun () ->
  let primal, pullback, ny, captures =
    traced ~name args (fun vs ->
        let ys, pb = rule vs in
        (ys, ys, pb))
  in
  let derivative = Pullback pullback in
  let c =
    Custom
      {
        name;
        primal;
        derivative;
        ny;
        args;
        captures;
        residuals = false;
        applied = false;
        seeds = [||];
      }
  in
  sub (run c) 0 ny

(* [f] and its transpose are traced apart: what each captures is its own. *)
let transpose_call ~name args f =
  let ny = ref 0 and inner = ref None in
  let apply, captures =
    Trace.trace Body ~name (Array.map form args) (fun vs ->
        let ys, b = f vs in
        ny := Array.length ys;
        inner := Some (Array.map form ys, b);
        ys)
  in
  let forms, b = Option.get !inner in
  let transpose, residuals = Trace.trace Body ~name forms b in
  (* A linear function's additions would be made by its transpose's transposes
     too, outside the maps it ran in. *)
  if Array.length apply.adds > 0 || Array.length transpose.adds > 0 then
    invalid_arg
      (name
     ^ ": the function or its transpose adds to a total; a linear call makes \
        no addition");
  let c =
    Linear
      {
        name;
        apply = Some apply;
        transpose = Traced transpose;
        args;
        captures;
        residuals;
        forms;
      }
  in
  sub (run c) 0 !ny

(* The residuals among [p]'s outputs, after its [ny] results. *)
let residual_count (p : Program.t) ~ny =
  Array.length p.outputs - ny - Array.length p.adds
