(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* The width [jit_lower] searches at: [beam], or else JITBEAM's, or else
   BEAM's. *)
let width ?beam () =
  match beam with
  | Some beam -> beam
  | None -> (
      match Setting.value Setting.jitbeam with
      | Some beam -> beam
      | None -> Setting.value Setting.beam)

(* The settings a lowering compiles under: BEAM at its width, and JITBEAM,
   read by then, unset. *)
let bindings ?beam () =
  [ Setting.B (Setting.beam, width ?beam ()); B (Setting.jitbeam, None) ]

let shaping ?beam () = Setting.shaping_under (bindings ?beam ())

type storage = Linked | Per_run

let jit_lower ?beam ?search ~storage ~targets ~held_bufs ~inputs ~input_vars
    linear =
  let param i u =
    let align, phase = Shape.storage_phase u in
    ( u,
      Shape.param
        ~shape:[ Int (Shape.max_numel u) ]
        ?device:(Ops.device u) ~phase ~align i (Ops.dtype u) )
  in
  let linear =
    Ops.substitute ~calls:Skip ~pass:Once linear (List.mapi param inputs)
  in
  (* The input variables change every run, which binds them: they are
     unbound. Any other binding was fixed at capture and stays. *)
  let unbound =
    List.filter_map
      (fun v ->
        if Shape.is_bound_var v && List.mem_assoc (Ops.expr v) input_vars then
          Some (v, Shape.unbound v)
        else None)
      (Ops.toposort ~calls:Skip linear)
  in
  let linear = Ops.substitute ~calls:Skip ~pass:Once linear unbound in
  let per_run = storage = Per_run in
  let linear = Memory.memory_plan_rewrite ~held_bufs ~per_run linear in
  (* The buffers left unheld hold only what one run computes: they become
     storage of the program, allocated where it is linked. *)
  let scratch =
    List.filter_map
      (fun u ->
        if Ops.op u = Op.Buffer && not (List.memq u held_bufs) then
          let n = Shape.max_numel u in
          Some (u, Shape.alloc ?device:(Ops.device u) [ Int n ] (Ops.dtype u))
        else None)
      (Ops.toposort ~calls:Skip linear)
  in
  let linear = Ops.substitute ~calls:Skip ~pass:Once linear scratch in
  Setting.context (bindings ?beam ()) (fun () ->
      Realize.compile_linear ?search ~targets linear)
