(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Operation logging as a generic interpreter. Purely observational: each
   operation is printed, then evaluated in the enclosing interpretation, so
   other transformations compose as usual. It treats every operation alike, so
   it names none. *)

open Nx_effect
module T = Nx

let shape_string s =
  "[" ^ String.concat "," (Array.to_list (Array.map string_of_int s)) ^ "]"

let rec handler : type r. Format.formatter -> (r, r) Effect.Deep.handler =
 fun ppf ->
  let open Effect.Deep in
  (* Logs [name] with the output shape, then continues with the output. *)
  let obs (type a b) name (out : (a, b) t) =
    Format.fprintf ppf "%s -> %s@." name (shape_string (T.shape out));
    out
  in
  let rule : type c. c Effect.t -> (unit -> c) option =
   fun eff ->
    match eff with
    | E_op op ->
        Some
          (fun () ->
            Format.fprintf ppf "%a@." Op.pp op;
            eval op)
    | Nx_quant.Effect.E_quant { w; op } ->
        let name =
          match op with
          | Apply { transpose = false; _ } -> "quant_apply"
          | Apply { transpose = true; _ } -> "quant_apply_transposed"
          | Dequant _ -> "quant_dequant"
        in
        Some (fun () -> obs name (Nx_quant.Effect.perform w op))
    (* A remat passes on with its function logged. *)
    | Remat.E_remat (Remat.Call c) ->
        Some
          (fun () ->
            let f params = match_with c.f params (handler ppf) in
            Remat.run (Remat.Call { c with f }))
    | _ -> None
  in
  let effc : type c. c Effect.t -> ((c, _) continuation -> _) option =
   fun eff -> Option.map Gate.deliver (rule eff)
  in
  { retc = Fun.id; exnc = raise; effc }

let with_debug ?(ppf = Format.err_formatter) f =
  Gate.with_transform (fun () -> Effect.Deep.match_with f () (handler ppf))
