(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type summary = [ `Min | `Max | `Mean | `Last | `None ]
type goal = [ `Minimize | `Maximize ]
type sample = { step : int; timestamp : float; value : float }

type def = {
  summary : summary;
  step_metric : string option;
  goal : goal option;
}

type t = { key : string; append : step:int -> timestamp:float -> float -> unit }

let make ~key ~append = { key; append }
let key t = t.key

let log t ~step ?timestamp value =
  let timestamp = Option.value timestamp ~default:(Unix.gettimeofday ()) in
  t.append ~step ~timestamp value
