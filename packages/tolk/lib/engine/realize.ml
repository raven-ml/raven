(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Shape

(* Calls *)

let get_call_arg_uops call =
  List.filter (fun s -> addrspace s <> Some Dtype.Alu) (src_without_body call)

(* What [call] gives its program's scalar parameter [v]: a nameless parameter,
   its argument in its slot; a variable, the variable [call] binds to its name,
   whose value is a default the run's binding of its name overrides, or else
   the variable a run binds. *)
let call_value call v =
  let args = src_without_body call in
  match arg v with
  | Param { name = None; slot; _ } -> List.nth args slot
  | Param p -> (
      let binds s = is_bound_var s && expr s = expr v in
      match List.find_opt binds args with
      | Some b -> b
      | None -> replace v ~arg:(Param { p with slot = -1 }))
  | _ -> v

let call_values call u =
  let scalar v = op v = Op.Param && addrspace v = Some Dtype.Alu in
  let subs =
    List.map
      (fun v -> (v, call_value call v))
      (List.filter scalar (toposort ~calls:Skip u))
  in
  substitute ~calls:Skip ~pass:Once u subs

let get_call_kernel_args call prg =
  let args = src_without_body call in
  let value =
    match arg (nth prg 0) with
    | Kernel { split = Some s; _ } -> (
        (* A split program a queue launches runs one block, the whole loop. *)
        let iterations =
          match s.iterations with
          | Int n -> const (`Int (Bigint.of_int n))
          | Sym n -> call_values call n
        in
        fun v ->
          match arg v with
          | Param { slot; _ } when slot = s.lo -> const (`Int Bigint.zero)
          | Param { slot; _ } when slot = s.hi -> iterations
          | _ -> call_value call v)
    | _ -> call_value call
  in
  List.map
    (fun p ->
      match arg p with
      | Param { addrspace = Dtype.Alu; _ } -> value p
      | Param { slot; _ } -> List.nth args slot
      | _ -> invalid_arg "get_call_kernel_args: a parameter is no Param")
    (kernel_params prg)

let get_call_outs_ins call =
  let ast = body call in
  match (op ast, arg ast) with
  | Op.Program, Program p -> (p.outs, p.ins)
  | Op.Store, _ -> ([ 0 ], [ 1 ])
  | _ -> ([], [])

let devices u = Option.to_list (device u)

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
  match op ast with
  | Op.Program -> (
      match arg (nth ast 0) with
      | Kernel { estimates = Some e; _ } -> e
      | _ -> Renderer.Estimates.zero)
  | Op.Store ->
      let nbytes = bytes_of (nth call 1) in
      { Renderer.Estimates.zero with lds = nbytes; mem = nbytes }
  | _ -> Renderer.Estimates.zero

(* Parallel lowering and compilation *)

let get_call_to_compile targets c =
  let ast = body c in
  match (op ast, arg ast) with
  | Op.Sink, Kernel _ ->
      (* A kernel's parameters name no device: it runs on its call's first
         storage argument's, as the engine launches it. *)
      let d =
        match Option.bind (List.nth_opt (get_call_arg_uops c) 0) device with
        | Some d -> d
        | None -> invalid_arg "a kernel's first storage needs a device"
      in
      let t = targets d in
      Some (ast, Result.fold ~ok:Fun.id ~error:invalid_arg (Device.renderer t))
  | _ -> None

type kernel = {
  name : string;
  key : string;
  ast : Ops.t;
  renderer : Renderer.t;
  opts : Opt.t list;
}

(* The kernels [lower_and_compile] finds on a domain while a [kernels] runs
   there, latest first. *)
let found : kernel list ref option Domain.DLS.key =
  Domain.DLS.new_key (fun () -> None)

let kernels f =
  let outer = Domain.DLS.get found and inner = ref [] in
  Domain.DLS.set found (Some inner);
  let v = Fun.protect ~finally:(fun () -> Domain.DLS.set found outer) f in
  (v, List.rev !inner)

(* [prg]'s linearized sink names its kernel as optimisation named it. *)
let note prg key ast renderer =
  match (Domain.DLS.get found, arg (nth prg 0)) with
  | Some l, Kernel k ->
      l := { name = k.name; key; ast; renderer; opts = k.applied_opts } :: !l
  | Some _, _ | None, _ -> ()

(* [linear] with each launch of more members than its renderer's program
   holds ({!Renderer.t.launch_members}) run as several, in order: members
   run in any order. Alike members make alike launches, one program. *)
let split_launches targets linear =
  let rec chunks n = function
    | [] -> []
    | ms ->
        let first = List.filteri (fun i _ -> i < n) ms in
        first :: chunks n (List.filteri (fun i _ -> i >= n) ms)
  in
  let launched = function [ m ] -> m | ms -> Shape.launch ms in
  let rec entries l = List.concat_map entry (src l)
  and entry e =
    match op e with
    | Op.Call when Shape.is_launch (body e) -> (
        match get_call_to_compile targets e with
        | Some (ast, ren) when List.length (src ast) > ren.launch_members ->
            List.map launched
              (chunks ren.launch_members (Shape.launch_members e))
        | _ -> [ e ])
    | (Op.End | Op.Backedge) when op (nth e 0) = Op.Linear ->
        let body = nth e 0 in
        [ replace e ~src:(replace body ~src:(entries body) :: List.tl (src e)) ]
    | _ -> [ e ]
  in
  replace linear ~src:(entries linear)

let lower_and_compile ?search ~targets linear =
  let linear = split_launches targets linear in
  let calls =
    List.filter_map
      (fun c ->
        if op c <> Op.Call then None
        else Option.map (fun a -> (c, a)) (get_call_to_compile targets c))
      (toposort ~calls:Skip linear)
  in
  (* Kernels of one program key take one program, whatever devices their
     calls' arguments lie on. A program made or kept on disk is read on the
     calling domain: a read took 0.4 to 0.5 ms on fudge, less than a domain's
     start and the collections it then joins, so a schedule whose programs are
     all made or kept spawns no domain. *)
  Codegen.to_programs ?beam:search (List.map snd calls);
  let compile (ast, ren) = Codegen.to_program ?beam:search ast ren in
  (* Each kernel now finds its program kept. *)
  substitute ~calls:Skip ~pass:Fixed_point linear
    (List.map
       (fun (c, ((ast, ren) as a)) ->
         let prg = compile a in
         note prg (Codegen.program_key ast ren) ast ren;
         (c, replace c ~src:(prg :: List.tl (src c))))
       calls)

(* A kernel that asks for no beam search asks for one of the setting's width. *)
let pm_beam width =
  Pattern_matcher.v (fun () ->
      [
        Pattern_matcher.rule
          (Upat.op ~name:"call" ~allow_any_len:true
             ~src:[ Upat.op ~name:"sink" Op.Sink ]
             Op.Call)
          (fun m ->
            match arg (m "sink") with
            | Kernel k when k.beam = 0 ->
                let sink =
                  replace (m "sink") ~arg:(Kernel { k with beam = width })
                in
                Some
                  (replace (m "call") ~src:(sink :: List.tl (src (m "call"))))
            | _ -> None);
      ])

let compile_linear ?search ~targets linear =
  let width = Setting.value Setting.beam in
  let linear =
    if width >= 1 then
      graph_rewrite ~calls:Skip ~pass:Once ~ctx:() linear
        (After_sources (pm_beam width))
    else linear
  in
  lower_and_compile ?search ~targets linear
