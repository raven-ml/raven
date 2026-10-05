(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

let jit_lower ?beam ?search ~profile ~devices ~held_bufs ~inputs linear =
  let param i u =
    ( u,
      Ops.param
        ~shape:[ Int (Ops.max_numel u) ]
        ?device:(Ops.device u) i (Ops.dtype u) )
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
    (fun () -> Hcq2.compile_linear ?search ~profile ~devices linear)
