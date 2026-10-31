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

(* GPU metrics' keys *)

let gpu_prefix = "sys/gpu"
let gpu_key i name =
  if i < 0 then invalid_arg (Printf.sprintf "Metric.gpu_key: GPU %d < 0" i);
  if name = "" then invalid_arg "Metric.gpu_key: empty name";
  Printf.sprintf "%s%d_%s" gpu_prefix i name

(* The digits after the prefix, as [gpu_key] prints them, without a leading
   zero, up to the first '_', then a name. *)
let gpu_metric key =
  let p = String.length gpu_prefix and n = String.length key in
  if not (String.starts_with ~prefix:gpu_prefix key) then None
  else
    match String.index_from_opt key p '_' with
    | None -> None
    | Some u ->
        let digits = String.sub key p (u - p) in
        let all_digits =
          digits <> ""
          && String.for_all (function '0' .. '9' -> true | _ -> false) digits
          && (digits = "0" || digits.[0] <> '0')
        in
        if (not all_digits) || u + 1 = n then None
        else
          Option.map
            (fun i -> (i, String.sub key (u + 1) (n - u - 1)))
            (int_of_string_opt digits)
