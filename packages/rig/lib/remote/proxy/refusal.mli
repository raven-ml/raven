(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The exception of a refused call, which {!Link} raises and
    {!Rig_remote_proxy} exports as its driver's [Refused]. It lives here so
    a program that uses {!Link} alone, and so does not link the driver's
    module, raises it as one that does. *)

exception Refused of string
