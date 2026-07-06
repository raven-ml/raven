(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tracing gate shared by the differentiation handlers: [no_grad] and [detach]
   turn interception off for a scope, in both reverse and forward mode. *)

let enabled = ref true

let without_tracing f =
  let prev = !enabled in
  enabled := false;
  Fun.protect f ~finally:(fun () -> enabled := prev)
