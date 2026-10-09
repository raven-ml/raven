(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Values: nx's central types. This module has no implementation.

    A value is concrete: arrays on the devices of its placement. The brand ['d]
    is phantom: every function that makes a value checks that its arrays lie on
    its placement's devices, and nothing at run time reads ['d]. *)

type ('v, 's) dtype = ('v, 's) Nx_array.Dtype.t

type ('v, 's, 'd) t =
  | Array of { at : 'd Devices.placement; a : ('v, 's) Nx_array.t }
      (** On [at]'s one device. *)
  | Shards of { at : 'd Devices.placement; arrays : ('v, 's) Nx_array.t array }
      (** On [at]'s devices, two or more: per device, in
          [Grid.devices (Devices.grid at)]'s order, its window of the value. *)

and ('v, 's, 'd) form = {
  dtype : ('v, 's) dtype;
  layout : Nx_array.Layout.t;
  placement : 'd Devices.placement;
}
(** A value without its bytes. A sharded value's layout is the C-contiguous
    layout of the whole. *)
