(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Operation logging as a generic interpreter. Purely observational: each
   operation is printed, then evaluated in the enclosing interpretation, so
   other transformations compose as usual. It treats every operation alike, so
   it names none. *)

open Nx.Op
module T = Nx

let shape_string s =
  "[" ^ String.concat "," (Array.to_list (Array.map string_of_int s)) ^ "]"

let rec install : type a. Format.formatter -> (unit -> a) -> a =
 fun ppf f ->
  let open Effect.Deep in
  (* Logs [name] with the output shape, then continues with the output. *)
  let obs (type a b) name (out : (a, b) T.t) =
    Format.fprintf ppf "%s -> %s@." name (shape_string (T.shape out));
    out
  in
  let rule : type c. c Effect.t -> (unit -> c) option =
   fun eff ->
    match eff with
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
            let f params = install ppf (fun () -> c.f params) in
            Remat.run (Remat.Call { c with f }))
    | _ -> None
  in
  let effc : type c. c Effect.t -> ((c, a) continuation -> a) option =
   fun eff -> Option.map Answer.deliver (rule eff)
  in
  let run op =
    Format.fprintf ppf "%a@." Nx.Op.pp op;
    eval op
  in
  match_with
    (fun () -> intercept { run } f)
    ()
    { retc = Fun.id; exnc = raise; effc }

let with_debug ?(ppf = Format.err_formatter) f = install ppf f
