(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Munin

let commit ?(dirty = "+") (p : Provenance.t) =
  Option.map
    (fun (g : Provenance.git) ->
      String.sub g.commit 0 (min 7 (String.length g.commit))
      ^ if g.dirty then dirty else "")
    p.git

let arrow = function
  | Some `Minimize -> "↓"
  | Some `Maximize -> "↑"
  | None -> ""

let latest_step stats =
  List.fold_left
    (fun acc (_, (k : History.key_stats)) ->
      if k.kind = `Metric then
        Some (max k.stats.last.step (Option.value acc ~default:0))
      else acc)
    None stats

let progress ?(bar = true) ~wide ~steps latest =
  let n = if wide then Text.count else Text.short in
  match (latest, steps) with
  | None, _ -> ""
  | Some s, Some t when s < t ->
      let p = Float.of_int s /. Float.of_int t in
      if wide && bar then
        Printf.sprintf "%s of %s  %s %.0f%%" (n s) (n t) (Text.bar 16 p)
          (100. *. p)
      else if wide then Printf.sprintf "%s of %s" (n s) (n t)
      else Printf.sprintf "%s/%s %.0f%%" (n s) (n t) (100. *. p)
  | Some s, _ -> n s

let wall r =
  Option.value (Run.ended_at r) ~default:(Run.updated_at r) -. Run.started_at r

let pauses h =
  let started = Run.started_at (History.run h) in
  match History.resumes h with
  | [] -> []
  | resumes ->
      let times =
        List.concat_map
          (fun k ->
            match History.series h k with
            | s -> List.init (Metric.length s) (Metric.time s)
            | exception Invalid_argument _ -> [])
          (History.metrics h)
        |> List.sort Float.compare |> Array.of_list
      in
      (* The latest sample time before [r], by bisection. *)
      let before r =
        let lo = ref 0 and hi = ref (Array.length times) in
        while !lo < !hi do
          let mid = (!lo + !hi) / 2 in
          if times.(mid) < r then lo := mid + 1 else hi := mid
        done;
        if !lo = 0 then started else Float.max started times.(!lo - 1)
      in
      fst
        (List.fold_left
           (fun (acc, total) (r : History.resume) ->
             let total = total +. Float.max 0. (r.time -. before r.time) in
             ((r.time, total) :: acc, total))
           ([], 0.) resumes)

let active_at pauses started t =
  let off =
    match List.find_opt (fun (r, _) -> r <= t) pauses with
    | Some (_, p) -> p
    | None -> 0.
  in
  t -. started -. off
