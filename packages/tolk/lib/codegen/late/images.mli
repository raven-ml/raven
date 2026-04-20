(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Late lowering for OpenCL image operations.

    See also {!Devectorizer} and {!Linearizer}. *)

val rewrite : Renderer.t -> Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t
(** [rewrite renderer root] lowers {!Tolk_ir.Kernel.view.Param_image}-based
    memory operations into explicit OpenCL image builtins for [renderer].

    Raises [Failure] if [root] uses images and [renderer] does not
    support them. *)
