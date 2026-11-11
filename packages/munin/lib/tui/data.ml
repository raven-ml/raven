(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Munin
module Smap = Map.Make (String)
module Sset = Set.Make (String)

type plot = {
  run : string;
  key : string;
  x : float * float;
  dots : int;
  smooth : bool;
}

module Pmap = Map.Make (struct
  type t = plot

  let compare = compare
end)

type at = Latest of string | At of int
type readout = { key : string; runs : string list; at : at; tau : float }
type compare = { focus : string; other : string; keys : string list }

type wants = {
  focus : string option;
  every : bool;
  plots : (plot * (float -> int)) list;
  readouts : readout list;
  compare : compare option;
}

let nothing = { focus = None; every = false; plots = []; readouts = []; compare = None }

let same a b =
  a.focus = b.focus && a.every = b.every
  && List.equal ( = ) (List.map fst a.plots) (List.map fst b.plots)
  && a.readouts = b.readouts && a.compare = b.compare

(* The keys a history keeps the samples of. *)
type keys = All | Keys of Sset.t

(* A history held, and its generation: a history read from its log's start,
   or followed into a restart, starts a new one, so that a reduction made from
   another generation is never extended by its series. *)
type held = { history : History.t; keys : keys; gen : int }

(* A reduction, with the generation and the resumes of the history it read,
   since a resume can orphan samples it reduced, and
   whether that history had read its run's end: only such a reduction is
   whole. *)
type made = { reduction : Reduce.t; from : int; resumes : int; whole : bool }

type t = {
  time : float;
  listed : bool;
  listing : Run.t list;
  facts : Facts.run list;
  by_id : Facts.run Smap.t;
  experiments : Facts.experiment Smap.t;
  held : held Smap.t;
  watches : Watch.t Smap.t;
  hints : (string * History.key_stats) list Smap.t;
  pauses : (float * float) list Smap.t;
  reductions : made Pmap.t;
  values : (readout * (string * float option * float option) list) list;
  differences : (compare * (string * (int * float * float)) list) option;
  errors : (string * string) list;
  gen : int;
}

let empty =
  {
    time = 0.;
    listed = false;
    listing = [];
    facts = [];
    by_id = Smap.empty;
    experiments = Smap.empty;
    held = Smap.empty;
    watches = Smap.empty;
    hints = Smap.empty;
    pauses = Smap.empty;
    reductions = Pmap.empty;
    values = [];
    differences = None;
    errors = [];
    gen = 0;
  }

let time d = d.time
let listed d = d.listed
let runs d = d.facts
let run d id = Smap.find_opt id d.by_id
let experiment d name = Smap.find_opt name d.experiments
let history d id = Option.map (fun h -> h.history) (Smap.find_opt id d.held)
let errors d = d.errors

(* A plot's dot count and smoothing follow its run, key and x in the order
   of plots, so the reductions of a series are one range. *)
let reductions d ~run ~key ~x =
  let first = { run; key; x; dots = min_int; smooth = false } in
  let series (p : plot) = p.run = run && p.key = key && p.x = x in
  Pmap.to_seq_from first d.reductions
  |> Seq.take_while (fun (p, _) -> series p)
  |> Seq.map (fun (_, m) -> m.reduction)
  |> List.of_seq

let values d r = Option.value (List.assoc_opt r d.values) ~default:[]

let difference d c key =
  match d.differences with
  | Some (c', l) when c' = c -> List.assoc_opt key l
  | _ -> None

(* Keys *)

let covers have want =
  match (have, want) with
  | All, _ -> true
  | Keys _, All -> false
  | Keys h, Keys w -> Sset.subset w h

let union a b =
  match (a, b) with All, _ | _, All -> All | Keys a, Keys b -> Keys (Sset.union a b)

let add_need id keys m =
  Smap.update id (function Some k -> Some (union k keys) | None -> Some keys) m

let ended r = Run.ended_at r <> None

(* A key's samples in the history the run kept: a reading logged before a
   resume that went back below its step is left out, as a metric's orphans
   are, so a chart over the steps draws one attempt at each step. *)
let series h key =
  match History.surviving h key with s -> Some s | exception Invalid_argument _ -> None

let step_of s i = Float.of_int (Metric.step s i)

(* The poll *)

(* The samples a history read, all keys together: a poll that finds more has
   timed an arrival. *)
let samples h =
  List.fold_left
    (fun n k -> match History.stats h k with Some s -> n + s.count | None -> n)
    0 (History.metrics h)

(* A reduction made from [h] that the poll can extend: of [h]'s generation and
   resumes, over a series no shorter than the one it read. *)
let reusable m (h : held) s =
  m.from = h.gen
  && m.resumes = List.length (History.resumes h.history)
  && Reduce.length m.reduction <= Metric.length s

let poll store ~now ~list wants d =
  let errors = ref [] in
  let guard id f =
    try Some (f ())
    with Sys_error e | Failure e | Invalid_argument e ->
      errors := (id, e) :: !errors;
      None
  in
  let listing =
    if not list then d.listing
    else
      let prev = if d.listed then Some d.listing else None in
      match guard "" (fun () -> Store.runs ?prev store) with
      | Some l -> l
      | None -> d.listing
  in
  let listed = d.listed || (list && !errors = []) in
  let by_listing = List.fold_left (fun m r -> Smap.add (Run.id r) r m) Smap.empty listing in
  let in_listing id = Smap.mem id by_listing in
  let find id =
    match Smap.find_opt id by_listing with
    | Some r -> Some r
    | None -> Option.join (guard id (fun () -> Store.run store id))
  in
  (* What each run's history must keep. A history held whose run had not
     ended when last read is followed once more by the listing that shows
     the end, so that what was made from it reaches its last samples. *)
  let lacks (p, _) =
    match Pmap.find_opt p d.reductions with None -> true | Some m -> not m.whole
  in
  let need = ref Smap.empty in
  List.iter
    (fun r -> if not (ended r) then need := add_need (Run.id r) (Keys Sset.empty) !need)
    listing;
  if list then
    List.iter
      (fun r ->
        match Smap.find_opt (Run.id r) d.held with
        | Some h when ended r && not (ended (History.run h.history)) ->
            need := add_need (Run.id r) h.keys !need
        | _ -> ())
      listing;
  List.iter
    (fun ((p : plot), dot) ->
      if lacks (p, dot) then need := add_need p.run (Keys (Sset.singleton p.key)) !need)
    wants.plots;
  (match wants.focus with
  | Some id ->
      let live =
        match Smap.find_opt id by_listing with Some r -> not (ended r) | None -> true
      in
      if live || wants.every then need := add_need id All !need
  | None -> ());
  List.iter
    (fun (r : readout) ->
      List.iter (fun id -> need := add_need id (Keys (Sset.singleton r.key)) !need) r.runs)
    wants.readouts;
  Option.iter
    (fun (c : compare) ->
      let ks = Keys (Sset.of_list c.keys) in
      need := add_need c.focus ks (add_need c.other ks !need))
    wants.compare;
  (* Read what is needed: a history held with the keys is followed when the
     poll lists, and kept as it is otherwise; any other is read from its
     log's start with the keys both ask for. *)
  let gen = ref d.gen in
  let fresh () =
    incr gen;
    !gen
  in
  let read id want =
    let fresh_read keys =
      Option.bind (find id) (fun r ->
          let series = match keys with All -> None | Keys k -> Some (Sset.elements k) in
          guard id (fun () ->
              { history = History.read ?series r; keys; gen = fresh () }))
    in
    match Smap.find_opt id d.held with
    | Some h when covers h.keys want ->
        if list && not (ended (History.run h.history)) then
          match guard id (fun () -> History.follow h.history) with
          | Some h' ->
              let gen = if History.restarted h' then fresh () else h.gen in
              Some { h with history = h'; gen }
          | None -> Some h
        else Some h
    | Some h -> (
        match fresh_read (union h.keys want) with Some h' -> Some h' | None -> Some h)
    | None -> fresh_read want
  in
  let held =
    Smap.fold
      (fun id want acc -> match read id want with Some h -> Smap.add id h acc | None -> acc)
      !need Smap.empty
  in
  (* Arrivals, timed at this poll for the histories it followed or read. *)
  let watches =
    Smap.fold
      (fun id (h : held) acc ->
        let n = samples h.history in
        let w =
          match Smap.find_opt id d.watches with
          | Some w -> Watch.observe w ~now ~samples:n
          | None -> Watch.start ~samples:n
        in
        Smap.add id w acc)
      held
      (Smap.filter (fun id _ -> in_listing id) d.watches)
  in
  (* The statistics of each ended run not held, from its hint, once. *)
  let hints =
    let kept = Smap.filter (fun id _ -> in_listing id) d.hints in
    if not list then kept
    else
      List.fold_left
        (fun acc r ->
          let id = Run.id r in
          if (not (ended r)) || Smap.mem id acc || Smap.mem id held then acc
          else
            match guard id (fun () -> History.stats_of r) with
            | Some s -> Smap.add id s acc
            | None -> acc)
        kept listing
  in
  (* Pauses change only with resumes, so a run's are read again only when a
     history held of it has another number of them, or none was known. *)
  let pauses =
    Smap.fold
      (fun id (h : held) acc ->
        let known = Smap.find_opt id d.pauses in
        let resumes = List.length (History.resumes h.history) in
        match (h.keys, known) with
        | All, Some p
          when List.length p = resumes && not (History.restarted h.history) ->
            Smap.add id p acc
        | All, _ -> Smap.add id (Facts.pauses h.history) acc
        | Keys _, Some p -> Smap.add id p acc
        | Keys _, None -> acc)
      held
      (Smap.filter (fun id _ -> in_listing id) d.pauses)
  in
  let facts =
    List.map
      (fun r ->
        let id = Run.id r in
        Facts.of_run ~now
          ~history:(Option.map (fun (h : held) -> h.history) (Smap.find_opt id held))
          ~hint:(Smap.find_opt id hints) ~watch:(Smap.find_opt id watches)
          ~pauses:(Option.value (Smap.find_opt id pauses) ~default:[])
          r)
      listing
  in
  let by_id =
    List.fold_left (fun m (f : Facts.run) -> Smap.add (Run.id f.run) f m) Smap.empty facts
  in
  let experiments =
    List.fold_left
      (fun m (e : Facts.experiment) -> Smap.add e.name e m)
      Smap.empty (Facts.experiments facts)
  in
  (* Reductions: each wanted one made from its run's history, extended by
     what the history gained, or kept as it was when nothing holds it. *)
  let make_reduction ((p : plot), dot) =
    let prev = Pmap.find_opt p d.reductions in
    let from_history (h : held) =
      match series h.history p.key with
      | None -> prev
      | Some s ->
          let x = step_of s in
          let resumes = List.length (History.resumes h.history) in
          let reduction =
            match prev with
            | Some m when reusable m h s -> Reduce.extend m.reduction ~x s
            | _ ->
                let lo, hi = p.x in
                let tau = if p.smooth then Some (Plot.tau p.x) else None in
                Reduce.make ?tau { lo; hi; dots = p.dots; dot } ~x s
          in
          Some { reduction; from = h.gen; resumes; whole = ended (History.run h.history) }
    in
    match Smap.find_opt p.run held with
    | Some h when covers h.keys (Keys (Sset.singleton p.key)) -> (
        match guard p.run (fun () -> from_history h) with
        | Some m -> m
        | None -> prev)
    | _ -> prev
  in
  let wanted =
    List.fold_left
      (fun m ((p : plot), dot) ->
        match make_reduction (p, dot) with Some r -> Pmap.add p r m | None -> m)
      Pmap.empty wants.plots
  in
  (* A whole reduction stays while its run is listed, at the dot counts the
     screen draws. *)
  let drawn = List.sort_uniq Int.compare (List.map (fun ((p : plot), _) -> p.dots) wants.plots) in
  let reductions =
    Pmap.fold
      (fun (p : plot) m acc ->
        if Pmap.mem p acc then acc
        else if m.whole && in_listing p.run && List.mem p.dots drawn then
          Pmap.add p m acc
        else acc)
      d.reductions wanted
  in
  let values =
    List.map
      (fun (r : readout) ->
        let step =
          match r.at with
          | At s -> Some s
          | Latest id -> Option.bind (Smap.find_opt id by_id) (fun (f : Facts.run) -> f.latest)
        in
        let value step id =
          Option.bind (Smap.find_opt id held) (fun (h : held) ->
              Option.map
                (fun s ->
                  let x = step_of s in
                  let v = Float.of_int step in
                  (id, Reduce.at ~x s v, Reduce.at ~tau:r.tau ~x s v))
                (series h.history r.key))
        in
        (r, match step with Some s -> List.filter_map (value s) r.runs | None -> []))
      wants.readouts
  in
  let differences =
    Option.map
      (fun (c : compare) ->
        let pair key =
          match (Smap.find_opt c.focus held, Smap.find_opt c.other held) with
          | Some a, Some b -> (
              match (series a.history key, series b.history key) with
              | Some sa, Some sb -> Option.map (fun d -> (key, d)) (Reduce.common sa sb)
              | _ -> None)
          | _ -> None
        in
        (c, List.filter_map pair c.keys))
      wants.compare
  in
  {
    time = now;
    listed;
    listing;
    facts;
    by_id;
    experiments;
    held;
    watches;
    hints;
    pauses;
    reductions;
    values;
    differences;
    errors = List.rev !errors;
    gen = !gen;
  }
