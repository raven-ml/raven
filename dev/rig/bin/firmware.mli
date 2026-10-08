(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** [rig firmware]: a directory filled with a driver's pinned firmware images.

    Each image is downloaded by running [curl], and checked against its pin with
    {!Rig_pci.Firmware.digest} before it is written. *)

val fetch : string -> string -> 'a
(** [fetch list dir] fills [dir] with the images of [list], a driver's
    [firmware.tsv] as {!Images} holds it: a line per image, its path, digest and
    URL separated by tabs, after comment lines starting with [#]. It exits with
    the status the page of [rig firmware] gives. *)
