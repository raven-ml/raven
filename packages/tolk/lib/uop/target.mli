(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Device, renderer and interface selection. *)

type t = {
  device : string;
  renderer : string;
  arch : string;
  interface : string;
  indices : string;
}
(** A target specification. Empty fields leave the selection to the consumer.
    [device] and [renderer] use uppercase names; [arch] and [interface] retain
    their spelling. [indices] selects visible devices, for example ["0,2"]
    or ["1-3"]. A device instance such as ["NV:1"] is a separate name. *)

val of_string : string -> t
(** [of_string s] parses a device with optional renderer and architecture,
    prefixed by an optional interface and indices separated by [+]. Colons
    separate fields on either side. Device and renderer names are uppercased.
    For example,
    ["PCI:0,2+nv:cuda:sm_89"] selects [NV], its [CUDA] renderer and the
    [PCI] interface. Empty fields and an empty string are accepted.

    Raises [Invalid_argument] on extra [+] or target [:] separators. Device,
    renderer, architecture and index availability are checked by consumers. *)

val to_string : t -> string
(** [to_string t] formats [t], omitting trailing empty fields. *)
