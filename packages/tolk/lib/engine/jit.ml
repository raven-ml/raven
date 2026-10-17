(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

let jit_lower ?beam ?search ~targets ~held_bufs ~inputs linear =
  let param i u =
    let align, phase = Call.storage_phase u in
    ( u,
      Call.param
        ~shape:[ Int (Shape.max_numel u) ]
        ?device:(Ops.device u) ~phase ~align i (Ops.dtype u) )
  in
  let linear =
    Ops.substitute ~calls:Skip ~pass:Once linear (List.mapi param inputs)
  in
  let linear = Memory.memory_plan_rewrite ~held_bufs linear in
  let beam =
    match beam with
    | Some beam -> beam
    | None -> (
        match Setting.value Setting.jitbeam with
        | Some beam -> beam
        | None -> Setting.value Setting.beam)
  in
  Setting.context
    [ B (Setting.beam, beam) ]
    (fun () -> Realize.compile_linear ?search ~targets linear)
