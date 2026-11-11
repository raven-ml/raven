(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Munin

type run = {
  run : Run.t;
  state : Watch.state;
  stats : (string * History.key_stats) list;
  metrics : string list;
  latest : int option;
  non_finite : (string * Metric.sample * float list) option;
  last_arrival : float option;
  active : float;
  rate : float option;
  x_end : float;
}

let name r = Option.value (Run.name r) ~default:(Run.id r)
let stats f key = List.assoc_opt key f.stats
let goal f key = Option.bind (stats f key) (fun (s : History.key_stats) -> s.goal)

let best f key =
  match stats f key with
  | Some { goal = Some g; stats; _ } -> Metric.best g stats
  | _ -> None

let commit (p : Provenance.t) =
  Option.map
    (fun (g : Provenance.git) -> String.sub g.commit 0 (min 7 (String.length g.commit)))
    p.git

let value : Value.t -> string = function
  | `Float x -> Numbers.given x
  | `Int n -> string_of_int n
  | `String s -> s
  | `Bool b -> string_of_bool b

let printed : Value.t -> string = function
  | `Float x -> Numbers.value x
  | `Int n -> Numbers.value (Float.of_int n)
  | v -> value v

let starts_with p s = String.starts_with ~prefix:p s
let ends_with p s = String.ends_with ~suffix:p s

let bound x =
  if not (x > 0.) then 1.
  else if x = Float.infinity then Float.max_float
  else
    let k = Numbers.decade x in
    let rounds =
      List.map (fun (m, k) -> Numbers.decimal m k) [ (1, k); (2, k); (5, k); (1, k + 1) ]
    in
    match List.find_opt (fun r -> r >= x) rounds with
    | Some r when Float.is_finite r -> r
    | _ -> Float.max_float

(* Statistics of the keys a history read, as a hint states them. *)
let key_stats h =
  let of_key kind k =
    Option.map
      (fun stats -> (k, { History.kind; goal = History.goal h k; stats }))
      (History.stats h k)
  in
  List.sort
    (fun (a, _) (b, _) -> String.compare a b)
    (List.filter_map (of_key `Metric) (History.metrics h)
    @ List.filter_map (of_key `Reading) (History.readings h))

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

let active_at pauses ~started t =
  let off =
    match List.find_opt (fun (r, _) -> r <= t) pauses with
    | Some (_, p) -> p
    | None -> 0.
  in
  t -. started -. off

(* The window over which the rate is taken: the last 10 minutes, or the last
   tenth of the active time when longer, once it holds a minute. *)
let rate_window = 600.
let rate_least = 60.

let rate h ~active =
  let longest =
    List.fold_left
      (fun acc k ->
        match History.series h k with
        | s when Metric.length s > (match acc with Some (_, n) -> n | None -> 1)
          ->
            Some (s, Metric.length s)
        | _ -> acc
        | exception Invalid_argument _ -> acc)
      None (History.metrics h)
  in
  match longest with
  | None -> None
  | Some (s, n) ->
      let span = Float.max rate_window (0.1 *. active) in
      let t1 = Metric.time s (n - 1) in
      let k = ref (n - 1) in
      while !k > 0 && t1 -. Metric.time s (!k - 1) <= span do
        decr k
      done;
      let dt = t1 -. Metric.time s !k in
      let ds = Metric.step s (n - 1) - Metric.step s !k in
      if dt < rate_least || ds <= 0 then None else Some (Float.of_int ds /. dt)

(* Losses with a goal first, then the other metrics with a goal, then the
   rest, each by key. *)
let order stats =
  let metrics =
    List.filter_map
      (fun (k, (s : History.key_stats)) ->
        if s.kind = `Metric then Some (k, s.goal <> None) else None)
      stats
  in
  let losses, rest = List.partition (fun (k, g) -> g && ends_with "loss" k) metrics in
  let goals, rest = List.partition snd rest in
  List.map fst (losses @ goals @ rest)

let latest_step stats =
  List.fold_left
    (fun acc (_, (k : History.key_stats)) ->
      if k.kind = `Metric then Some (max k.stats.last.step (Option.value acc ~default:0))
      else acc)
    None stats

let first_non_finite stats =
  let firsts =
    List.filter_map
      (fun (k, (s : History.key_stats)) ->
        match (s.kind, s.stats.non_finite) with
        | `Metric, Some (_, f) -> Some (k, f)
        | _ -> None)
      stats
  in
  match firsts with
  | [] -> None
  | first :: rest ->
      let k, f =
        List.fold_left
          (fun (k, (f : Metric.sample)) (k', (f' : Metric.sample)) ->
            if f'.step < f.step then (k', f') else (k, f))
          first rest
      in
      Some (k, f, List.map (fun (_, (f : Metric.sample)) -> f.value) firsts)

let of_run ~now ~history ~hint ~watch ~pauses r =
  let r = match history with Some h -> History.run h | None -> r in
  let stats =
    match (history, hint) with
    | Some h, _ -> key_stats h
    | None, Some s -> s
    | None, None -> []
  in
  let until =
    match (Run.ended_at r, Run.status r) with
    | Some e, _ -> e
    | None, (`Running | `Silent) -> now
    | None, _ -> Run.updated_at r
  in
  let active = active_at pauses ~started:(Run.started_at r) until in
  let latest = latest_step stats in
  let x_end =
    match (Run.steps r, latest) with
    | Some n, Some l when n > 0 && l <= n -> Float.of_int n
    | Some n, None when n > 0 -> Float.of_int n
    | _, Some l -> bound (Float.of_int l)
    | _, None -> 1.
  in
  {
    run = r;
    state = Watch.state (Run.status r) watch ~now;
    stats;
    metrics = order stats;
    latest;
    non_finite = first_non_finite stats;
    last_arrival = Option.bind watch Watch.last_arrival;
    active;
    rate = Option.bind history (fun h -> rate h ~active);
    x_end;
  }

let glyph f =
  match f.state with
  | Failed | Crashed -> "✗"
  | Killed -> "■"
  | _ when f.non_finite <> None -> "×"
  | Live | Writing -> "●"
  | Running | Starting -> "○"
  | Stalled | Silent -> "◌"
  | Finished -> "✓"

(* Experiments *)

type experiment = {
  name : string;
  runs : string list;
  key : string option;
  varying : string list;
}

let key_metric runs =
  (* One entry per key, a goal winning over none. *)
  let keys = Hashtbl.create 16 in
  List.iter
    (fun f ->
      List.iter
        (fun (k, (s : History.key_stats)) ->
          if s.kind = `Metric then
            match Hashtbl.find_opt keys k with
            | Some (Some _) -> ()
            | _ -> Hashtbl.replace keys k s.goal)
        f.stats)
    runs;
  let keys =
    List.sort (fun (a, _) (b, _) -> String.compare a b)
      (Hashtbl.fold (fun k g acc -> (k, g) :: acc) keys [])
  in
  let goals = List.filter (fun (_, g) -> g <> None) keys in
  let outside = List.filter (fun (k, _) -> not (starts_with "train/" k)) goals in
  let losses = List.filter (fun (k, _) -> ends_with "loss" k) outside in
  match (losses, outside, goals, keys) with
  | (k, _) :: _, _, _, _
  | [], (k, _) :: _, _, _
  | [], [], (k, _) :: _, _
  | [], [], [], (k, _) :: _ ->
      Some k
  | [], [], [], [] -> None

let varying runs =
  let params = List.map (fun f -> Run.params f.run) runs in
  let keys = List.sort_uniq String.compare (List.concat_map (List.map fst) params) in
  match params with
  | [] -> []
  | first :: rest ->
      List.filter
        (fun k ->
          let x = List.assoc_opt k first in
          List.exists (fun p -> List.assoc_opt k p <> x) rest)
        keys

let experiments runs =
  let order = ref [] and members = Hashtbl.create 16 in
  List.iter
    (fun f ->
      let e = Run.experiment f.run in
      match Hashtbl.find_opt members e with
      | Some l -> Hashtbl.replace members e (f :: l)
      | None ->
          order := e :: !order;
          Hashtbl.replace members e [ f ])
    runs;
  List.rev_map
    (fun e ->
      let runs = List.rev (Hashtbl.find members e) in
      {
        name = e;
        runs = List.map (fun f -> Run.id f.run) runs;
        key = key_metric runs;
        varying = varying runs;
      })
    !order
