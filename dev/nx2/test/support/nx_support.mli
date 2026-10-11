(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Devices and kernel libraries for nx's suites. *)

val memory : int -> Rig.t
(** [memory k] is the memory device [k], [0 <= k < 4], named ["m<k>"]: its
    memory is the host's and its work runs in this process
    ({!Rig.memory_device}). *)

(** nx.cpu, counting each call of its entries, named ["nx.test"]. *)
module Counting : sig
  include Nx_kernel.S

  val calls : unit -> int
  (** [calls ()] is the number of entry calls since the last {!reset}. *)

  val reset : unit -> unit
end

module Declining : Nx_kernel.S
(** nx.cpu, declining [Add] and [Fill], named ["nx.test"]. *)
