(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Spk = struct
  type segment = Spk.segment = {
    name : string;
    target : int;
    center : int;
    frame : int;
    kind : int;
    start : (Ymir.Time.tdb, Nx.host) Ymir.Time.t;
    stop : (Ymir.Time.tdb, Nx.host) Ymir.Time.t;
  }

  type 'd t = 'd Spk.t

  let read = Spk.read
  let of_string = Spk.of_string
  let segments = Spk.segments
  let state = Spk.state
  let position = Spk.position
end
