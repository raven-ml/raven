(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The usual gap is the median of the last [kept - 1] gaps, and three of them
   are never less than [floor] seconds. These constants have no reference: they
   bound how long a run reads live after its last sample. *)
let kept = 32
let floor = 10.

type t = {
  since : float;
  samples : int;
  arrivals : float list (* newest first *);
}

let start ~now ~samples = { since = now; samples; arrivals = [] }

let observe w ~now ~samples =
  if samples > w.samples then
    let arrivals = List.filteri (fun i _ -> i < kept) (now :: w.arrivals) in
    { w with samples; arrivals }
  else { w with samples }

let since w = w.since
let samples w = w.samples
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

let word (status : Munin.Run.status) w ~now =
  match (status, w) with
  | `Running, None -> "running"
  | `Running, Some w ->
      if w.samples = 0 then "starting"
      else if w.arrivals = [] then "running"
      else if stalled w ~now then "stalled"
      else "live"
  | `Silent, Some w when w.arrivals <> [] && not (stalled w ~now) -> "writing"
  | `Silent, _ -> "silent"
  | `Crashed, _ -> "crashed"
  | `Finished, _ -> "finished"
  | `Failed _, _ -> "failed"
  | `Killed, _ -> "killed"
