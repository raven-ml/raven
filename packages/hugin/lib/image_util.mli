(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Image encoding utilities.

    {b Internal module.} Base64 encoding for data URIs. *)

(** {1:base64 Base64} *)

val base64_encode : string -> string
(** [base64_encode s] is the base64 encoding of [s]. *)
