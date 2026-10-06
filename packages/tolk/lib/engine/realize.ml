(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Shape
open Call

(* Calls *)

let get_call_arg_uops call =
  List.filter (fun s -> not (is_bound_var s)) (src_without_body call)

let get_call_var_uops call prg =
  let bound =
    List.filter_map
      (fun s ->
        match arg s with
        | Param { bound = Some v; _ } when is_bound_var s ->
            Some (expr s, const (v :> Dtype.const))
        | _ -> None)
      (src_without_body call)
  in
  match arg prg with
  | Program p ->
      let named v = List.assoc_opt (expr v) bound in
      let value =
        match arg (nth prg 0) with
        | Kernel { split = Some s; _ } -> (
            (* A split program a queue launches runs one block, the whole
               loop. *)
            let iterations =
              match s.iterations with
              | Int n -> const (`Int (Bigint.of_int n))
              | Sym n ->
                  let values =
                    List.filter_map
                      (fun v ->
                        if is_variable v then
                          Option.map (fun c -> (v, c)) (named v)
                        else None)
                      (toposort ~calls:Enter n)
                  in
                  substitute ~calls:Skip ~pass:Fixed_point n values
            in
            fun v ->
              match arg v with
              | Param { slot; _ } when slot = s.lo ->
                  Some (const (`Int Bigint.zero))
              | Param { slot; _ } when slot = s.hi -> Some iterations
              | _ -> named v)
        | _ -> named
      in
      List.map (fun v -> Option.value (value v) ~default:v) p.vars
  | _ -> invalid_arg "get_call_var_uops takes a program"

let aux call = match arg call with Call { aux; _ } -> aux | _ -> None

let get_call_outs_ins call =
  let ast = body call in
  match (aux call, op ast, arg ast) with
  | Some _, _, _ -> ([], [])
  | None, Op.Program, Program p -> (p.outs, p.ins)
  | None, Op.Store, _ -> ([ 0 ], [ 1 ])
  | _ -> ([], [])

let get_call_written_bufs call =
  match aux call with
  | Some info -> info.written_bufs
  | None ->
      let args = get_call_arg_uops call
      and outs, ins = get_call_outs_ins call in
      let storage k =
        let b = storage_base (List.nth args k) in
        if op b = Op.Mselect then storage_base (nth b 0) else b
      in
      List.filter (fun k -> not (List.mem k ins)) outs
      |> List.map storage
      |> List.filter (fun b -> op b = Op.Buffer)
      |> Helpers.dedup (module Ops)

let devices u =
  match device u with
  | Some (Single d) -> [ d ]
  | Some (Multi ds) -> ds
  | None -> []

let bytes_of u = Sint.(numel u * Int (Dtype.itemsize (dtype u)))

let get_call_name ?(var_vals = []) call bufs =
  let size u = Helpers.size_to_str (sym_infer (bytes_of u) var_vals) in
  let dev_str u =
    String.concat ", "
      (List.map (fun d -> String.sub d 0 (min 7 (String.length d))) (devices u))
  in
  let ast = body call in
  match (op ast, arg (nth ast 0)) with
  | Op.Program, Kernel k -> k.name
  | Op.Store, _ ->
      Helpers.colored Helpers.Yellow
        (Printf.sprintf "copy %10s, %7s <- %-7s"
           (size (List.hd (get_call_arg_uops call)))
           (dev_str (List.nth bufs 0))
           (dev_str (List.nth bufs 1)))
  | _ -> invalid_arg "get_call_name names programs and copies"

(* Stat *)

let estimate_uop call =
  let call = without_after call in
  let ast = body call in
  match (aux call, op ast) with
  | Some info, _ -> info.estimates
  | None, Op.Program -> (
      match arg (nth ast 0) with
      | Kernel { estimates = Some e; _ } -> e
      | _ -> Renderer.Estimates.zero)
  | None, Op.Store ->
      let nbytes = bytes_of (nth call 1) in
      { Renderer.Estimates.zero with lds = nbytes; mem = nbytes }
  | None, _ -> Renderer.Estimates.zero

(* Parallel lowering and compilation *)

let get_call_to_compile targets c =
  let ast = body c in
  match (op ast, arg ast) with
  | Op.Sink, Kernel _ ->
      let t = targets (List.hd (devices c)) in
      Some (ast, Result.fold ~ok:Fun.id ~error:invalid_arg (Device.renderer t))
  | _ -> None

let lower_and_compile ?search ~targets linear =
  let calls =
    List.filter_map
      (fun c ->
        if op c <> Op.Call then None
        else Option.map (fun a -> (c, a)) (get_call_to_compile targets c))
      (toposort ~calls:Enter linear)
  in
  let same (a0, r0) (a1, r1) = a0 == a1 && r0 == r1 in
  let todo =
    List.fold_left
      (fun todo (_, a) -> if List.exists (same a) todo then todo else a :: todo)
      [] calls
    |> List.rev
  in
  (* A beam search times its candidates, which concurrent compilations would
     disturb. *)
  let beam (_, (ast, _)) =
    match arg ast with Kernel k -> k.beam > 0 | _ -> false
  in
  let map =
    if List.compare_length_with todo 1 <= 0 || List.exists beam calls then
      List.map
    else Worker.map
  in
  let compiled =
    List.combine todo
      (map (fun (ast, ren) -> Codegen.to_program ?beam:search ast ren) todo)
  in
  let program a = snd (List.find (fun (b, _) -> same a b) compiled) in
  substitute ~calls:Skip ~pass:Fixed_point linear
    (List.map
       (fun (c, a) -> (c, replace c ~src:(program a :: List.tl (src c))))
       calls)
