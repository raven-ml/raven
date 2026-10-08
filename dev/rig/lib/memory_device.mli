(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Memory devices: the driver {!Rig.memory_device} opens.

    Its memory is the host's, and its work runs in the submitting thread: the
    hand-over runs a submission's copies and fills in order, then stores the
    value into the word. A {!Rig.Submission.Words} part, or a fill that declares
    ring units or segment bytes, never fits. A device's state is its word, alone
    in a page, which other devices may map and which is never freed. *)

val open_ : string -> (Def.device, string) result
(** {!Rig.memory_device}. *)
