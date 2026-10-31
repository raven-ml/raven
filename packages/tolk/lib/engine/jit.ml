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

let jit_lower ?beam ?search ~targets ~held_bufs ~inputs linear =
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
  let linear = Memory.memory_plan_rewrite ~held_bufs linear in
  Setting.context (bindings ?beam ()) (fun () ->
      Realize.compile_linear ?search ~targets linear)
