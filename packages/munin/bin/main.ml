(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* munin, the command-line reader of a store. Each command prints header-free
   tab-separated lines for the next program, and an error on stderr with exit
   status 1. *)

open Munin
open Cmdliner
open Munin_float_text

(* Fields *)

let fail fmt = Printf.ksprintf failwith fmt
let line fields = print_endline (String.concat "\t" fields)

(* A value prints as Value.pp prints it, so that a string a run recorded, which
   may hold a tab or a line feed, never splits a line. Texts, ids and paths hold
   no control character in a store munin wrote, and print as they are, so that a
   path stays usable on Windows. *)
let value v = Format.asprintf "%a" Value.pp v
let float x = value (`Float x)
let escaped s = value (`String s)
let or_dash f = function Some x -> f x | None -> "-"
let joined = function [] -> "-" | l -> String.concat "," l

let status_name : Run.status -> string = function
  | `Running -> "running"
  | `Silent -> "silent"
  | `Crashed -> "crashed"
  | `Finished -> "finished"
  | `Failed _ -> "failed"
  | `Killed -> "killed"

let statuses =
  [ "running"; "silent"; "crashed"; "finished"; "failed"; "killed" ]

(* Times *)

(* The civil date of a day counted from 1970-01-01, by Howard Hinnant's
   days_from_civil inverse, exact over the proleptic Gregorian calendar. *)
let civil days =
  let z = days + 719468 in
  let era = (if z >= 0 then z else z - 146096) / 146097 in
  let doe = z - (era * 146097) in
  let yoe = (doe - (doe / 1460) + (doe / 36524) - (doe / 146096)) / 365 in
  let doy = doe - ((365 * yoe) + (yoe / 4) - (yoe / 100)) in
  let mp = ((5 * doy) + 2) / 153 in
  let d = doy - (((153 * mp) + 2) / 5) + 1 in
  let m = if mp < 10 then mp + 3 else mp - 9 in
  ((yoe + (era * 400) + if m <= 2 then 1 else 0), m, d)

(* RFC 3339 writes years 0001 to 9999: 0001-01-01 and 10000-01-01 in seconds
   since the epoch. *)
let first_second = -62135596800.
let end_second = 253402300800.

(* [t] in RFC 3339, UTC, to the microsecond, rounded once from its exact value
   as Float_text rounds. A time outside the years RFC 3339 writes prints as a
   float. *)
let time t =
  let s = Float.floor t in
  let frac = Float_text.f 6 (t -. s) in
  let s, micros =
    if frac.[0] = '1' then (s +. 1., "000000") else (s, String.sub frac 2 6)
  in
  if not (s >= first_second && s < end_second) then float t
  else
    let s = Float.to_int s in
    let days = if s >= 0 then s / 86400 else (s - 86399) / 86400 in
    let secs = s - (days * 86400) in
    let y, m, d = civil days in
    Printf.sprintf "%04d-%02d-%02dT%02d:%02d:%02d.%sZ" y m d (secs / 3600)
      (secs / 60 mod 60)
      (secs mod 60) micros

(* Stores *)

(* An explicit root must exist, so that a mistyped one fails. The default root
   of a user who has recorded nothing yet is an empty store. *)
let open_store = function
  | None -> Store.open_ ()
  | Some dir ->
      if not (Sys.file_exists dir && Sys.is_directory dir) then
        fail "no store at %s" dir;
      Store.open_ ~root:dir ()

let find_run store id =
  match Store.run store id with Some r -> r | None -> fail "no run %s" id

let find_version store name n =
  match Store.artifact store name (`Version n) with
  | Some a -> a
  | None -> fail "no version %d of %s" n name

let version_number s =
  match int_of_string_opt s with
  | Some n when n >= 1 -> n
  | _ -> fail "%s is not a version number" s

(* [best_value goal st] is the value a key's row shows: its best under its goal,
   or its last when it has none. A goal with no finite sample has no best. *)
let best_value goal (st : Metric.stats) =
  match goal with
  | None -> Some st.last.value
  | Some g -> Option.map (fun (s : Metric.sample) -> s.value) (Metric.best g st)

let non_finite (st : Metric.stats) =
  match st.non_finite with Some (n, _) -> n | None -> 0

(* Commands *)

let root store =
  print_endline (Store.root store);
  0

let runs ~experiment ~status ~tags ~parent store =
  let matches want v = match want with None -> true | Some w -> w = v in
  let keep r =
    matches experiment (Run.experiment r)
    && matches status (status_name (Run.status r))
    && List.for_all (fun t -> List.mem t (Run.tags r)) tags
    && (parent = None || parent = Run.parent r)
  in
  let print r =
    line
      [
        Run.id r;
        status_name (Run.status r);
        time (Run.started_at r);
        Run.experiment r;
        or_dash Fun.id (Run.name r);
        joined (Run.tags r);
      ]
  in
  List.iter (fun r -> if keep r then print r) (Store.runs store);
  0

let show_provenance (p : Provenance.t) =
  let field k v = line [ "provenance." ^ k; v ] in
  field "command" (String.concat " " (List.map escaped p.command));
  field "cwd" (escaped p.cwd);
  Option.iter (fun h -> field "hostname" (escaped h)) p.hostname;
  field "pid" (string_of_int p.pid);
  field "ocaml" (escaped p.ocaml);
  field "raven" (escaped p.raven);
  field "os" (escaped p.os);
  field "arch" (escaped p.arch);
  Option.iter (fun c -> field "cpu" (escaped c)) p.cpu;
  Option.iter
    (fun (g : Provenance.git) ->
      field "git.commit" g.commit;
      field "git.dirty" (string_of_bool g.dirty))
    p.git;
  List.iter (fun (k, n) -> field ("seed." ^ k) (string_of_int n)) p.seeds;
  List.iter (fun (k, v) -> field ("env." ^ k) (escaped v)) p.env

let show_metric h k =
  let st = Option.get (History.stats h k) in
  let field f v = line [ Printf.sprintf "metric.%s.%s" k f; v ] in
  field "count" (string_of_int st.count);
  field "last" (float st.last.value);
  Option.iter
    (fun (s : Metric.sample) -> field "best" (float s.value))
    (History.best h k);
  field "non_finite" (string_of_int (non_finite st))

(* A line per thing the run has: an absent name, end or note prints no line, so
   that [grep ^ended] tells whether the run ended. *)
let show id store =
  let h = History.read ~series:[] (find_run store id) in
  let r = History.run h in
  let field k v = line [ k; v ] in
  let some k f o = Option.iter (fun x -> field k (f x)) o in
  field "id" (Run.id r);
  field "experiment" (Run.experiment r);
  some "name" Fun.id (Run.name r);
  some "parent" Fun.id (Run.parent r);
  field "status" (status_name (Run.status r));
  (match Run.status r with `Failed e -> field "error" (escaped e) | _ -> ());
  field "started" (time (Run.started_at r));
  some "ended" time (Run.ended_at r);
  field "updated" (time (Run.updated_at r));
  some "steps" string_of_int (Run.steps r);
  List.iter (fun (k, v) -> field ("param." ^ k) (value v)) (Run.params r);
  show_provenance (Run.provenance r);
  some "diff" Fun.id (Run.diff r);
  List.iter (field "tag") (Run.tags r);
  some "notes" escaped (Run.notes r);
  List.iter (fun (k, v) -> field ("summary." ^ k) (value v)) (History.summary h);
  List.iter (show_metric h) (History.metrics h);
  0

let kind_name = function `Metric -> "metric" | `Reading -> "reading"

let metric_stats (k, (ks : History.key_stats)) =
  let st = ks.stats in
  let best =
    Option.bind ks.goal (fun g ->
        Option.map (fun (s : Metric.sample) -> s.value) (Metric.best g st))
  in
  line
    [
      k;
      kind_name ks.kind;
      string_of_int st.count;
      float st.last.value;
      or_dash float best;
      string_of_int (non_finite st);
    ]

let series run k =
  let h = History.read ~series:[ k ] run in
  if not (List.mem k (History.metrics h) || List.mem k (History.readings h))
  then fail "run %s has no samples of %s" (Run.id run) k;
  let s = History.series h k in
  for i = 0 to Metric.length s - 1 do
    line
      [
        string_of_int (Metric.step s i);
        time (Metric.time s i);
        float (Metric.value s i);
      ]
  done

let metrics id key store =
  let run = find_run store id in
  (match key with
  | None -> List.iter metric_stats (History.stats_of run)
  | Some k -> series run k);
  0

(* Compare *)

let union keys_of runs =
  List.concat_map keys_of runs |> List.sort_uniq String.compare

(* A param row prints when the runs do not all give it the same value. Two
   values are the same when they print the same, so that NaN equals NaN and
   [`Int 1] differs from [`Float 1.]. *)
let param_rows runs =
  let cell r k = or_dash value (List.assoc_opt k (Run.params r)) in
  union (fun r -> List.map fst (Run.params r)) runs
  |> List.iter (fun k ->
      let cells = List.map (fun r -> cell r k) runs in
      if List.exists (fun c -> c <> List.hd cells) cells then
        line (("param." ^ k) :: cells))

(* The index of the best of [values] under [goal]: finite values alone compete,
   and the earliest of equal values wins, as History.best picks a sample. Strict
   comparisons keep the earliest, and [-0.] equals [0.]. *)
let best_index goal values =
  let better a b = match goal with `Minimize -> a < b | `Maximize -> a > b in
  let pick (best, i) v =
    let best =
      match (v, best) with
      | Some v, _ when not (Float.is_finite v) -> best
      | Some v, None -> Some (i, v)
      | Some v, Some (_, b) when better v b -> Some (i, v)
      | _ -> best
    in
    (best, i + 1)
  in
  Option.map fst (fst (List.fold_left pick (None, 0) values))

(* The goal a row is starred under: the one every run that has the key gives.
   Runs that disagree, or a key without a goal, star nothing. *)
let row_goal entries =
  match List.filter_map Fun.id entries with
  | [] -> None
  | (g, _) :: rest ->
      if List.for_all (fun (g', _) -> g' = g) rest then g else None

let metric_row stats k =
  let entries =
    List.map
      (fun st ->
        Option.map
          (fun (ks : History.key_stats) -> (ks.goal, ks.stats))
          (List.assoc_opt k st))
      stats
  in
  let values =
    List.map (Option.map (fun (g, st) -> best_value g st)) entries
    |> List.map Option.join
  in
  let star = Option.bind (row_goal entries) (fun g -> best_index g values) in
  let cell i v =
    let text = or_dash float v in
    if star = Some i then text ^ "*" else text
  in
  line (("metric." ^ k) :: List.mapi cell values)

let compare ids store =
  let runs = List.map (find_run store) ids in
  param_rows runs;
  let metrics_of r =
    List.filter
      (fun (_, (ks : History.key_stats)) -> ks.kind = `Metric)
      (History.stats_of r)
  in
  let stats = List.map metrics_of runs in
  List.iter (metric_row stats) (union (List.map fst) stats);
  0

(* Artifacts *)

let artifacts name store =
  let aliases = Hashtbl.create 8 in
  let aliases_of a =
    let name = Artifact.name a in
    match Hashtbl.find_opt aliases name with
    | Some l -> l
    | None ->
        let l = Store.aliases store name in
        Hashtbl.add aliases name l;
        l
  in
  let print a =
    let names =
      List.filter_map
        (fun (alias, n) -> if n = Artifact.version a then Some alias else None)
        (aliases_of a)
    in
    line
      [
        Artifact.name a;
        string_of_int (Artifact.version a);
        Artifact.digest a;
        string_of_int (Artifact.size_bytes a);
        Artifact.run a;
        or_dash string_of_int (Artifact.step a);
        joined names;
      ]
  in
  List.iter print (Store.artifacts ?name store);
  0

let alias ~remove args store =
  match (remove, args) with
  | false, [ name; version; alias ] ->
      let a = find_version store name (version_number version) in
      Store.set_alias store a alias;
      `Ok 0
  | true, [ name; alias ] ->
      Store.remove_alias store ~name alias;
      `Ok 0
  | false, _ -> `Error (true, "alias takes NAME VERSION ALIAS")
  | true, _ -> `Error (true, "alias --remove takes NAME ALIAS")

let tag ~remove id tags store =
  ignore (find_run store id);
  if remove then Store.remove_tags store id tags
  else Store.add_tags store id tags;
  0

let note ~clear id text store =
  let r = find_run store id in
  match (clear, text) with
  | true, Some _ -> `Error (true, "note --clear takes no TEXT")
  | true, None ->
      Store.set_notes store id None;
      `Ok 0
  | false, Some t ->
      Store.set_notes store id (Some t);
      `Ok 0
  | false, None ->
      Option.iter
        (fun n ->
          print_string n;
          if not (String.ends_with ~suffix:"\n" n) then print_newline ())
        (Run.notes r);
      `Ok 0

(* Each run is removed or reports why not, as rm does, and the status is 1 if
   any was not. *)
let rm ids store =
  let remove id =
    match Store.delete_run store (Run.id (find_run store id)) with
    | () -> true
    | exception (Failure m | Sys_error m) ->
        prerr_endline ("munin: " ^ m);
        false
  in
  if List.for_all Fun.id (List.map remove ids) then 0 else 1

(* A resume at the largest step orphans nothing; ending it as killed records
   that the run stopped without an end of its own. A silent run's resume trusts
   that its writer is gone (Store_format, section resume). *)
let end_run id store =
  let r = find_run store id in
  match Run.status r with
  | `Running ->
      prerr_endline ("munin: run " ^ id ^ " is being written");
      1
  | `Finished | `Failed _ | `Killed ->
      prerr_endline ("munin: run " ^ id ^ " has ended");
      1
  | `Crashed | `Silent ->
      let s = Session.resume r ~step:(1 lsl 53) in
      Session.finish ~status:`Killed s;
      0

let rm_artifact name version store =
  Store.delete_artifact store (find_version store name version);
  0

let gc store =
  Printf.printf "removed %d\n" (Store.gc store);
  0

(* [dashboard screen store] shows the dashboard of [store] until the user
   quits. *)
let dashboard screen store =
  match Munin_tui.run store screen with
  | Ok () -> 0
  | Error why -> fail "%s" why

let check store =
  match Store.check store with
  | [] -> 0
  | problems ->
      List.iter (fun (path, why) -> line [ path; why ]) problems;
      1

(* Command line *)

(* [guard error f] is [f ()], or [error] once it printed why [f] failed. *)
let guard error f =
  try f ()
  with Failure m | Invalid_argument m | Sys_error m ->
    prerr_endline ("munin: " ^ m);
    error

let root_arg =
  let doc =
    "The store's root. Defaults to $(b,RAVEN_TRACKING_DIR), then \
     $(b,XDG_DATA_HOME)/raven/munin, then ~/.local/share/raven/munin."
  in
  Arg.(value & opt (some string) None & info [ "root" ] ~docv:"DIR" ~doc)

let exits =
  [
    Cmd.Exit.info 0 ~doc:"on success.";
    Cmd.Exit.info 1 ~doc:"on an error, or a problem $(b,check) found.";
  ]

(* [cmd name doc term] runs [term]'s command on the store at [--root]. *)
let cmd name doc term =
  let run root f = guard 1 (fun () -> f (open_store root)) in
  Cmd.v (Cmd.info name ~doc ~exits) Term.(const run $ root_arg $ term)

(* As [cmd], for a command that checks how many arguments it was given. *)
let cmd_ret name doc term =
  let run root f = guard (`Ok 1) (fun () -> f (open_store root)) in
  Cmd.v (Cmd.info name ~doc ~exits) Term.(ret (const run $ root_arg $ term))

let run_pos n = Arg.(required & pos n (some string) None & info [] ~docv:"RUN")
let runs_pos = Arg.(non_empty & pos_all string [] & info [] ~docv:"RUN")
let name_pos = Arg.(required & pos 0 (some string) None & info [] ~docv:"NAME")

let opt_string name docv doc =
  Arg.(value & opt (some string) None & info [ name ] ~docv ~doc)

let remove_flag doc = Arg.(value & flag & info [ "remove" ] ~doc)
let root_cmd = cmd "root" "Print the store's root." (Term.const root)

let runs_cmd =
  let status =
    let doc = "Only runs of status $(docv): " ^ String.concat ", " statuses in
    let status_conv = Arg.enum (List.map (fun s -> (s, s)) statuses) in
    Arg.(value & opt (some status_conv) None & info [ "status" ] ~docv:"S" ~doc)
  in
  let tags =
    let doc = "Only runs tagged $(docv). Repeat for runs with every tag." in
    Arg.(value & opt_all string [] & info [ "tag" ] ~docv:"T" ~doc)
  in
  let f experiment status tags parent =
    runs ~experiment ~status ~tags ~parent
  in
  cmd "runs"
    "List runs, newest first: id, status, start, experiment, name, tags."
    Term.(
      const f
      $ opt_string "experiment" "E" "Only runs of experiment $(docv)."
      $ status $ tags
      $ opt_string "parent" "ID" "Only runs whose parent is $(docv).")

let show_cmd =
  cmd "show" "Print a run as key-value lines." Term.(const show $ run_pos 0)

let metrics_cmd =
  let key = Arg.(value & pos 1 (some string) None & info [] ~docv:"KEY") in
  cmd "metrics"
    "List a run's metrics and readings: key, kind, count, last, best, \
     non-finite count. With $(i,KEY), its samples: step, time, value."
    Term.(const metrics $ run_pos 0 $ key)

let compare_cmd =
  cmd "compare"
    "Compare runs: the params that differ, and each metric's best, or last \
     without a goal, one column per run. A star marks a row's best."
    Term.(const compare $ runs_pos)

let artifacts_cmd =
  let name = Arg.(value & pos 0 (some string) None & info [] ~docv:"NAME") in
  cmd "artifacts"
    "List artifact versions: name, version, digest, size, run, step, aliases."
    Term.(const artifacts $ name)

let alias_cmd =
  let args = Arg.(value & pos_all string [] & info [] ~docv:"ARG") in
  let f remove args = alias ~remove args in
  cmd_ret "alias"
    "Make $(i,ALIAS) name version $(i,VERSION) of $(i,NAME): $(b,alias) \
     $(i,NAME) $(i,VERSION) $(i,ALIAS), or remove it: $(b,alias --remove) \
     $(i,NAME) $(i,ALIAS)."
    Term.(const f $ remove_flag "Remove the alias." $ args)

let tag_cmd =
  let tags = Arg.(non_empty & pos_right 0 string [] & info [] ~docv:"TAG") in
  let f remove id tags = tag ~remove id tags in
  cmd "tag" "Add tags to a run, or remove them."
    Term.(const f $ remove_flag "Remove the tags." $ run_pos 0 $ tags)

let note_cmd =
  let text = Arg.(value & pos 1 (some string) None & info [] ~docv:"TEXT") in
  let clear = Arg.(value & flag & info [ "clear" ] ~doc:"Remove the notes.") in
  let f clear id text = note ~clear id text in
  cmd_ret "note"
    "Set a run's notes to $(i,TEXT), remove them, or print them without \
     $(i,TEXT)."
    Term.(const f $ clear $ run_pos 0 $ text)

let rm_cmd =
  cmd "rm" "Delete runs and every file they hold." Term.(const rm $ runs_pos)

let end_cmd =
  cmd "end"
    "End a crashed or silent run as killed. A silent run's writer cannot be \
     probed: end it only when that writer is known to be gone."
    Term.(const end_run $ run_pos 0)

let rm_artifact_cmd =
  let version =
    Arg.(required & pos 1 (some int) None & info [] ~docv:"VERSION")
  in
  cmd "rm-artifact" "Delete an artifact version."
    Term.(const rm_artifact $ name_pos $ version)

let gc_cmd =
  cmd "gc" "Remove blobs no version names and what crashed writers left."
    (Term.const gc)

let watch_cmd =
  let id = Arg.(value & pos 0 (some string) None & info [] ~docv:"RUN") in
  cmd "watch"
    "Monitor a run in the terminal: $(i,RUN), else the newest run without an \
     end that has not crashed, else the newest run, else the first run to \
     start. Esc goes to the store's runs."
    Term.(const (fun id -> dashboard (Munin_tui.Run id)) $ id)

let check_cmd =
  cmd "check" "List the paths that break the store's format, with why."
    (Term.const check)

(* Bare munin shows the store's runs on a terminal, and its help elsewhere. *)
let default =
  let run root =
    if Unix.isatty Unix.stdin && Unix.isatty Unix.stdout then
      `Ok (guard 1 (fun () -> dashboard Munin_tui.Runs (open_store root)))
    else `Help (`Plain, None)
  in
  Term.(ret (const run $ root_arg))

let main =
  Cmd.group ~default
    (Cmd.info "munin"
       ~doc:"Read and change a munin store, or show it in the terminal" ~exits)
    [
      root_cmd;
      runs_cmd;
      show_cmd;
      metrics_cmd;
      compare_cmd;
      artifacts_cmd;
      alias_cmd;
      tag_cmd;
      note_cmd;
      end_cmd;
      rm_cmd;
      rm_artifact_cmd;
      gc_cmd;
      check_cmd;
      watch_cmd;
    ]

(* Every error, a command line's included, exits with status 1. *)
let () =
  exit
    (match Cmd.eval_value main with
    | Ok (`Ok code) -> code
    | Ok (`Help | `Version) -> 0
    | Error _ -> 1)
