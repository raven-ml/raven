(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The usual gap is the median of the last [kept - 1] gaps, and three of them
   are never less than [floor] seconds. These constants have no reference: they
   bound how long a run reads live after its last sample. *)
let kept = 32
let floor = 10.

type t = { samples : int; arrivals : float list (* newest first *) }

let start ~samples = { samples; arrivals = [] }

let observe w ~now ~samples =
  if samples > w.samples then
    let arrivals = List.filteri (fun i _ -> i < kept) (now :: w.arrivals) in
    { samples; arrivals }
  else { w with samples }

let last_arrival w = match w.arrivals with t :: _ -> Some t | [] -> None

let usual_gap w =
  let rec gaps = function
    | a :: (b :: _ as rest) -> (a -. b) :: gaps rest
    | _ -> []
  in
  match List.sort Float.compare (gaps w.arrivals) with
  | [] -> None
  | g ->
      let n = List.length g in
      let a = Array.of_list g in
      Some
        (if n mod 2 = 1 then a.(n / 2) else (a.((n / 2) - 1) +. a.(n / 2)) /. 2.)

let stalled w ~now =
  match (usual_gap w, last_arrival w) with
  | Some g, Some t -> now -. t > Float.max (3. *. g) floor
  | _ -> false

type state =
  | Starting
  | Running
  | Live
  | Stalled
  | Writing
  | Silent
  | Crashed
  | Finished
  | Failed
  | Killed

let state (status : Munin.Run.status) w ~now =
  match (status, w) with
  | `Running, None -> Running
  | `Running, Some w ->
      if w.samples = 0 then Starting
      else if w.arrivals = [] then Running
      else if stalled w ~now then Stalled
      else Live
  | `Silent, Some w when w.arrivals <> [] && not (stalled w ~now) -> Writing
  | `Silent, _ -> Silent
  | `Crashed, _ -> Crashed
  | `Finished, _ -> Finished
  | `Failed _, _ -> Failed
  | `Killed, _ -> Killed

let word = function
  | Starting -> "starting"
  | Running -> "running"
  | Live -> "live"
  | Stalled -> "stalled"
  | Writing -> "writing"
  | Silent -> "silent"
  | Crashed -> "crashed"
  | Finished -> "finished"
  | Failed -> "failed"
  | Killed -> "killed"

let live = function
  | Starting | Running | Live | Writing -> true
  | Stalled | Silent | Crashed | Finished | Failed | Killed -> false

let broken = function Failed | Crashed -> true | _ -> false
