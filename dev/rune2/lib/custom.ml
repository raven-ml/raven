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

let jvp_call ~name args rule =
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
      { name; apply = Some apply; transpose; args; captures; residuals; forms }
  in
  sub (run c) 0 !ny

(* The residuals among [p]'s outputs, after its [ny] results. *)
let residual_count (p : Program.t) ~ny =
  Array.length p.outputs - ny - Array.length p.adds
