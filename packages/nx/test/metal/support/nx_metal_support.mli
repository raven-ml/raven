(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the Metal suite shares: the GPU and the harness's kernels, each run as
    one launch of rig's. *)

(** {1:gpu The GPU} *)

val device : unit -> Rig.t
(** [device ()] is the device of {!Nx_kernels_support.metal}[ ()]: the machine's
    Metal GPU 0, on which nx.metal computes. It skips the test where there is
    none. *)

(** {1:launches Launches} *)

(** The type for a harness kernel's parameters, laid out in order as 8-byte
    words. *)
type word =
  | A of Rig.Buffer.t  (** The GPU address of a buffer's first byte. *)
  | W of int  (** A 64-bit word. *)

val launch :
  string -> groups:int -> threads:int -> shared:int -> word list -> unit
(** [launch k ~groups ~threads ~shared ws] runs the harness kernel [k] over
    [groups] groups of [threads] threads along x, each with [shared] bytes of
    threadgroup memory, with parameters [ws], every buffer it addresses written,
    on {!device}[ ()], and returns once it is done.

    Raises [Invalid_argument] if the harness has no kernel [k]. *)

(** {1:parts Parts} *)

val parted :
  ?budget:float -> ?launches:int -> ?seconds:float -> (unit -> 'a) -> 'a
(** [parted ~budget ~launches ~seconds f] is [f ()] with nx.metal's calls cut
    into parts of at most [launches] launches (defaults to
    {!Nx_metal.Plan.part_launches}) and [seconds] seconds together (defaults to
    {!Nx_metal.Plan.part_budget}) ({!Nx_metal.Plan.bound_parts}), each launch
    within [budget] seconds where given ({!Nx_metal.Plan.bound_budget}), and the
    library's bounds restored once no [parted] runs. Domains that run [parted]
    at once share the bounds of the first, so each must give the same. *)
