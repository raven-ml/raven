(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Ppxlib

(** Deriver registration for [@@deriving ptree]. *)

(** {1:internal Internal} *)

val ptree_structure :
  ctxt:Expansion_context.Deriver.t ->
  Asttypes.rec_flag * Parsetree.type_declaration list ->
  bool ->
  Parsetree.structure
(** [ptree_structure ~ctxt decls mirror] is the structure generator behind
    [@@deriving ptree ~mirror:mirror]. Internal: shared with [ppx_jit], which
    calls it on synthesized declarations instead of emitting [@@deriving ptree]
    attributes; not part of the public API. *)
