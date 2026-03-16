type live_status = [ `Live | `Stopped | `Done of Run.status ]

type t = {
  run : Run.t;
  mutable ic : in_channel option;
  mutable pos : int;
  mutable last_event_time : float;
  latest : (string, Run.metric) Hashtbl.t;
  histories : (string, (int * float) list) Hashtbl.t;
  defs : (string, Run.metric_def) Hashtbl.t;
  mutable finished : Run.status option;
}

let stopped_timeout = 5.0
let events_path dir = Filename.concat dir "events.jsonl"

let start run =
  {
    run;
    ic = None;
    pos = 0;
    last_event_time = Unix.gettimeofday ();
    latest = Hashtbl.create 16;
    histories = Hashtbl.create 16;
    defs = Hashtbl.create 8;
    finished = None;
  }

let close t =
  Option.iter close_in t.ic;
  t.ic <- None

let read_new_events t =
  let path = events_path (Run.dir t.run) in
  if not (Sys.file_exists path) then []
  else
    let stat = Unix.stat path in
    let size = stat.Unix.st_size in
    if size <= t.pos then []
    else
      let ic =
        match t.ic with
        | Some ic ->
            (* Check for truncation/rotation *)
            if size < t.pos then (
              close_in ic;
              let new_ic = open_in path in
              t.ic <- Some new_ic;
              t.pos <- 0;
              new_ic)
            else ic
        | None ->
            let ic = open_in path in
            t.ic <- Some ic;
            ic
      in
      seek_in ic t.pos;
      let events = ref [] in
      (try
         while true do
           let line = input_line ic in
           match Event_log.decode_line line with
           | Some event -> events := event :: !events
           | None -> ()
         done
       with End_of_file -> ());
      t.pos <- pos_in ic;
      List.rev !events

let process_event t = function
  | Event_log.Metric { step; timestamp; key; value } ->
      let metric = Run.{ step; timestamp; value } in
      Hashtbl.replace t.latest key metric;
      let history =
        match Hashtbl.find_opt t.histories key with Some h -> h | None -> []
      in
      Hashtbl.replace t.histories key ((step, value) :: history);
      t.last_event_time <- timestamp
  | Define_metric { key; summary = s; step_metric; goal } ->
      Hashtbl.replace t.defs key { Run.summary = s; step_metric; goal }
  | Finished { status; ended_at = _ } ->
      t.finished <- Some (Run.status_of_string status)
  | Resumed _ ->
      t.finished <- None;
      t.last_event_time <- Unix.gettimeofday ()
  | Media _ | Summary _ | Notes _ | Tags _ | Artifact_output _
  | Artifact_input _ ->
      ()

let poll t =
  let events = read_new_events t in
  List.iter (process_event t) events

let live_status t =
  match t.finished with
  | Some status -> `Done status
  | None ->
      let elapsed = Unix.gettimeofday () -. t.last_event_time in
      if elapsed > stopped_timeout then `Stopped else `Live

let metrics t =
  Hashtbl.to_seq t.latest |> List.of_seq
  |> List.sort (fun (a, _) (b, _) -> String.compare a b)

let history t key =
  match Hashtbl.find_opt t.histories key with
  | Some h -> List.rev h
  | None -> []

let metric_defs t =
  Hashtbl.to_seq t.defs |> List.of_seq
  |> List.sort (fun (a, _) (b, _) -> String.compare a b)

let contains_sub ~sub s =
  let ls = String.length sub and lk = String.length s in
  ls <= lk
  &&
  let rec loop i = i <= lk - ls && (String.sub s i ls = sub || loop (i + 1)) in
  loop 0

let is_loss_like key =
  let key = String.lowercase_ascii key in
  contains_sub ~sub:"loss" key || contains_sub ~sub:"error" key

let best t key =
  match Hashtbl.find_opt t.histories key with
  | None | Some [] -> None
  | Some history ->
      let minimize =
        match Hashtbl.find_opt t.defs key with
        | Some { goal = Some `Minimize; _ } -> true
        | Some { goal = Some `Maximize; _ } -> false
        | _ -> is_loss_like key
      in
      let compare =
        if minimize then fun a b -> Float.compare a b
        else fun a b -> Float.compare b a
      in
      let best_step, best_value =
        List.fold_left
          (fun (bs, bv) (s, v) -> if compare v bv < 0 then (s, v) else (bs, bv))
          (List.hd history) (List.tl history)
      in
      let timestamp =
        match Hashtbl.find_opt t.latest key with
        | Some m -> m.timestamp
        | None -> 0.0
      in
      Some Run.{ step = best_step; timestamp; value = best_value }
