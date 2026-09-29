(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Shape = Shape
module View = View
module Elements = Elements
module Backend_intf = Backend_intf

(* The field order is nx.cpu's C ABI (nx_c.h). *)
type ('a, 'b) t = {
  dtype : ('a, 'b) Nx_dtype.t;
  view : View.t;
  buffer : Nx_device.Buffer.t;
}
