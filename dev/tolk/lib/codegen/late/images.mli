(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Late lowering for OpenCL image operations.

    See also {!Devectorizer} and {!Linearizer}. *)

val rewrite : Renderer.t -> Tolk_ir.Program.t -> Tolk_ir.Program.t
(** [rewrite renderer program] lowers {!Tolk_ir.Program.Param_image}-based
    memory operations into explicit OpenCL image builtins for [renderer].

    Raises [Failure] if [program] uses images and [renderer] does not
    support them. *)
