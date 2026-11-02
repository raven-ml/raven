(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Smap = Map.Make (String)

type kind = [ `Metric | `Reading ]

(* A session appends to its log through [fd], opened for appending and holding
   the run's exclusive lock. [good] is the log's length after the last whole
   record: a write that fails is cut back to it, so a torn line never glues to
   the next record. [good] follows a write that returns; an asynchronous
   exception, such as Ctrl-C's Sys.Break, between a write and its update leaves
   it short, and nothing here promises otherwise. Every field below [mutex]
   changes under it. *)
type t = {
  dir : string;
  id : string;
  fd : Unix.file_descr;
  mutex : Mutex.t;
  latest : int Atomic.t;
  stopping : bool Atomic.t;
  mutable good : int;
  mutable finished : bool;
  mutable sync_failed : bool;
  mutable kinds : kind Smap.t;
  mutable accs : Acc.t Smap.t;
  mutable defs : (Metric.goal option * string option) Smap.t;
      (* each declared metric's goal and step metric *)
  store : Store.t;
  mutable samplers : Thread.t list;
  mutable failure : (exn * Printexc.raw_backtrace) option;
  buf : Buffer.t; (* each record is encoded here *)
}

type metric = { key : string; owner : t }
type reading = unit -> (string * float) list

let max_step = 1 lsl 53

(* Checks *)

let invalid fmt = Printf.ksprintf invalid_arg fmt

let check_step fn step =
  if step < 0 || step > max_step then
    invalid "Session.%s: step %d is outside [0;2^53]" fn step

let check_values fn values =
  (match File.repeat fst values with
  | Some k -> invalid "Session.%s: key %S repeats" fn k
  | None -> ());
  List.iter
    (fun (k, v) ->
      if not (File.is_text k) then
        invalid "Session.%s: key %S is not a text" fn k;
      match v with
      | `String s when not (Json.utf8 s) ->
          invalid "Session.%s: the value of %S is not UTF-8" fn k
      | _ -> ())
    values

(* Writing *)

let ensure_open fn s =
  if s.finished then invalid "Session.%s: the session is finished" fn;
  if s.sync_failed then
    failwith
      (Printf.sprintf
         "Session.%s: a sync of run %s failed, so the system may have dropped \
          records written before it"
         fn s.id)

(* [append s r] writes [r] under [s.mutex]. *)
(* [append ?time s r] writes [r] under [s.mutex]. *)
let append ?(time = Unix.gettimeofday ()) s r =
  let line = Record.encode ~buf:s.buf ~time r in
  match File.write_all s.fd line with
  | () -> s.good <- s.good + String.length line
  | exception Sys_error e ->
      (try Unix.ftruncate s.fd s.good with Unix.Unix_error _ -> ());
      raise (Sys_error (Listing.log s.dir ^ ": " ^ e))

let sync_locked s =
  match File.sync s.fd with
  | () -> ()
  | exception e ->
      s.sync_failed <- true;
      raise e

let add_samples s kind ~step ~time values =
  List.iter
    (fun (k, v) ->
      s.kinds <- Smap.add k kind s.kinds;
      let acc = Option.value ~default:Acc.empty (Smap.find_opt k s.accs) in
      s.accs <- Smap.add k (Acc.add acc { Metric.step; time; value = v }) s.accs)
    values

(* Starting *)

let id_now () =
  let t = Unix.gettimeofday () in
  let tm = Unix.gmtime t in
  Printf.sprintf "%04d%02d%02dT%02d%02d%02dZ-%08x" (tm.tm_year + 1900)
    (tm.tm_mon + 1) tm.tm_mday tm.tm_hour tm.tm_min tm.tm_sec
    (Random.State.bits (Random.State.make_self_init ()) land 0xFFFFFFFF)

let rec claim_dir runs =
  let id = id_now () in
  let dir = Filename.concat runs id in
  match Unix.mkdir dir 0o777 with
  | () -> (id, dir)
  | exception Unix.Unix_error (Unix.EEXIST, _, _) -> claim_dir runs
  | exception Unix.Unix_error (e, _, _) ->
      raise (Sys_error (dir ^ ": " ^ Unix.error_message e))

(* Read-write: on NFS a lock is a byte-range lock, and an exclusive one needs a
   descriptor open for writing, a shared one (the resumer's probe) one open for
   reading (flock(2), NFS details). *)
let log_flags =
  [ Unix.O_RDWR; Unix.O_APPEND; Unix.O_CLOEXEC; Unix.O_SHARE_DELETE ]
let identities () = (File.machine_id (), File.boot_id ())

(* Who writes the run now, beside the log, replaced before the start or resume
   record (Store_format, section status). *)
let write_writer dir (machine, boot) =
  let b = Buffer.create 128 in
  let opt k = function
    | None -> []
    | Some v -> [ (k, fun b -> Json.add_string b v) ]
  in
  Json.add_object b (opt "machine" machine @ opt "boot" boot);
  Buffer.add_char b '\n';
  File.replace (Filename.concat dir "writer") (Buffer.contents b)

let make ~store ~dir ~id ~fd ~good =
  {
    store;
    dir;
    id;
    fd;
    mutex = Mutex.create ();
    latest = Atomic.make 0;
    stopping = Atomic.make false;
    good;
    finished = false;
    sync_failed = false;
    kinds = Smap.empty;
    accs = Smap.empty;
    defs = Smap.empty;
    samplers = [];
    failure = None;
    buf = Buffer.create 256;
  }

let start ?store ?name ?parent ?(tags = []) ?(params = []) ?steps ?provenance
    ~experiment () =
  if not (File.is_text experiment) then
    invalid "Session.start: experiment %S is not a text" experiment;
  Option.iter
    (fun n ->
      if not (File.is_text n) then
        invalid "Session.start: name %S is not a text" n)
    name;
  List.iter
    (fun t ->
      if not (File.is_tag t) then
        invalid "Session.start: tag %S is not a tag: a text without \",\"" t)
    tags;
  check_values "start" params;
  Option.iter (fun n -> check_step "start" n) steps;
  let store = match store with Some s -> s | None -> Store.open_ () in
  Option.iter
    (fun p ->
      match Store.run store p with
      | Some _ -> ()
      | None ->
          failwith
            (Printf.sprintf "Session.start: parent %s is no run of %s" p
               (Store.root store))
      | exception Failure _ -> ())
    parent;
  let provenance =
    match provenance with Some p -> p | None -> Provenance.detect ()
  in
  let runs = Filename.concat (Store.root store) "runs" in
  File.mkdirs runs;
  let id, dir = claim_dir runs in
  let log = Listing.log dir in
  let fd =
    try Unix.openfile log (Unix.O_CREAT :: Unix.O_EXCL :: log_flags) 0o666
    with Unix.Unix_error (e, _, _) ->
      raise (Sys_error (log ^ ": " ^ Unix.error_message e))
  in
  match
    (* The lock comes before the start, so no listing sees a started run without
       its writer. *)
    if not (File.try_lock fd `Exclusive) then
      raise (Sys_error (log ^ ": the file system refused a lock"));
    write_writer dir (identities ());
    List.iter (fun tag -> File.replace (Listing.tag_file dir tag) tag) tags;
    (match provenance.git with
    | Some { dirty = true; _ } -> (
        match
          File.output
            [| "git"; "-C"; provenance.cwd; "diff"; "--binary"; "HEAD" |]
        with
        | Some diff -> File.replace (Filename.concat dir "git.diff") diff
        | None ->
            failwith ("Session.start: git diff failed in " ^ provenance.cwd))
    | _ -> ());
    let s = make ~store ~dir ~id ~fd ~good:0 in
    append s
      (Record.Start
         {
           format = Store_format.version;
           experiment;
           name;
           parent;
           steps;
           params;
           provenance;
         });
    File.sync fd;
    File.sync_dir dir;
    File.sync_dir runs;
    s
  with
  | s -> s
  | exception e ->
      Unix.close fd;
      raise e

let resume ?provenance (r : Run.t) ~step =
  check_step "resume" step;
  let log = Listing.log r.dir in
  let fd =
    try Unix.openfile log log_flags 0
    with Unix.Unix_error (e, _, _) ->
      raise (Sys_error (log ^ ": " ^ Unix.error_message e))
  in
  match
    if not (File.take_writer fd) then
      failwith (Printf.sprintf "Session.resume: run %s is being written" r.id);
    (* Under the lock, read the log once: its end, its keys' kinds and
       statistics, with this resume's orphans left out. *)
    let h = Fold.for_resume r ~step in
    if Fold.ended h then
      failwith (Printf.sprintf "Session.resume: run %s has ended" r.id);
    let good = Fold.offset h in
    (try Unix.ftruncate fd good
     with Unix.Unix_error (e, _, _) ->
       raise (Sys_error (log ^ ": " ^ Unix.error_message e)));
    let provenance =
      match provenance with Some p -> p | None -> Provenance.detect ()
    in
    let store =
      Store.open_ ~root:(Filename.dirname (Filename.dirname r.dir)) ()
    in
    let s = make ~store ~dir:r.dir ~id:r.id ~fd ~good in
    List.iter
      (fun (k, kind) -> s.kinds <- Smap.add k kind s.kinds)
      (Fold.kinds h);
    List.iter
      (fun (k, (_, acc)) -> s.accs <- Smap.add k acc s.accs)
      (Fold.accs h);
    List.iter (fun (k, d) -> s.defs <- Smap.add k d s.defs) (Fold.defs h);
    Atomic.set s.latest step;
    write_writer r.dir (identities ());
    append s (Record.Resume { step; provenance });
    File.sync fd;
    s
  with
  | s -> s
  | exception e ->
      Unix.close fd;
      raise e

let id s = s.id

let sync s =
  Mutex.protect s.mutex @@ fun () ->
  ensure_open "sync" s;
  sync_locked s

(* Metrics *)

let metric s ?goal ?step_metric key =
  if not (File.is_key key) then invalid "Session.metric: %S is not a key" key;
  Option.iter
    (fun m ->
      if m.owner != s then
        invalid "Session.metric: the step metric of %S is another session's" key)
    step_metric;
  Mutex.protect s.mutex @@ fun () ->
  ensure_open "metric" s;
  if Smap.find_opt key s.kinds = Some `Reading then
    invalid "Session.metric: a reading of run %s gave %S" s.id key;
  let def = (goal, Option.map (fun m -> m.key) step_metric) in
  (match Smap.find_opt key s.defs with
  | Some d when d = def -> ()
  | Some _ ->
      invalid "Session.metric: %S is declared otherwise in run %s" key s.id
  | None ->
      append s (Record.Metric { key; goal; step_metric = snd def });
      s.kinds <- Smap.add key `Metric s.kinds;
      s.defs <- Smap.add key def s.defs);
  { key; owner = s }

let store s = s.store
let key m = m.key

let log_metrics s ~step ?time pairs =
  check_step "log_metrics" step;
  (match time with
  | Some t when not (Float.is_finite t) ->
      invalid "Session.log_metrics: time %g is not finite" t
  | _ -> ());
  List.iter
    (fun (m, _) ->
      if m.owner != s then
        invalid "Session.log_metrics: %S is another session's metric" m.key)
    pairs;
  (match File.repeat (fun (m, _) -> m.key) pairs with
  | Some k -> invalid "Session.log_metrics: %S appears twice" k
  | None -> ());
  if pairs <> [] then (
    Mutex.protect s.mutex @@ fun () ->
    ensure_open "log_metrics" s;
    let time = match time with Some t -> t | None -> Unix.gettimeofday () in
    let values = List.map (fun (m, v) -> (m.key, v)) pairs in
    append ~time s (Record.Metrics { step; values });
    add_samples s `Metric ~step ~time values;
    Atomic.set s.latest step)

let log m ~step ?time v = log_metrics m.owner ~step ?time [ (m, v) ]

let set_summary s values =
  check_values "set_summary" values;
  Mutex.protect s.mutex @@ fun () ->
  ensure_open "set_summary" s;
  append s (Record.Summary values)

(* Sampling *)

(* [log_reading s pairs] logs one reading's pairs. Raises Invalid_argument for a
   pair the format refuses. *)
let log_reading s pairs =
  if pairs <> [] then begin
    List.iter
      (fun (k, _) ->
        if not (File.is_key k) then
          invalid "Session.sample: a reading gave %S, which is not a key" k)
      pairs;
    (match File.repeat fst pairs with
    | Some k -> invalid "Session.sample: a reading gave %S twice" k
    | None -> ());
    let time = Unix.gettimeofday () in
    Mutex.protect s.mutex @@ fun () ->
    if not s.finished then begin
      List.iter
        (fun (k, _) ->
          if Smap.find_opt k s.kinds = Some `Metric then
            invalid "Session.sample: a reading gave %S, a metric of run %s" k
              s.id)
        pairs;
      let step = Atomic.get s.latest in
      append ~time s (Record.Reading { step; values = pairs });
      add_samples s `Reading ~step ~time pairs
    end
  end

let call_reading s r =
  match r () with pairs -> log_reading s pairs | exception Sys_error _ -> ()

(* The sampler waits in slices of [slice] seconds, so that finish stops it
   within one slice and a reading in progress. *)
let slice = 0.1

let sample s ~every readings =
  if not (Float.is_finite every && every > 0.) then
    invalid "Session.sample: every %g is not positive and finite" every;
  Mutex.protect s.mutex (fun () -> ensure_open "sample" s);
  List.iter (call_reading s) readings;
  let run () =
    let rec wait until =
      if Atomic.get s.stopping then false
      else
        let left = until -. File.monotonic () in
        if left <= 0. then true
        else (
          Unix.sleepf (Float.min slice left);
          wait until)
    in
    (* On a clock that never steps, so a wall clock set back or forward neither
       stalls the sampler nor runs it back to back. A call that overran its
       period skips the periods it missed. *)
    let rec loop next =
      if wait next then begin
        List.iter (call_reading s) readings;
        let now = File.monotonic () in
        let rec after t = if t > now then t else after (t +. every) in
        loop (after (next +. every))
      end
    in
    try loop (File.monotonic () +. every)
    with e ->
      let bt = Printexc.get_raw_backtrace () in
      Mutex.protect s.mutex (fun () ->
          if s.failure = None then s.failure <- Some (e, bt))
  in
  let th = Thread.create run () in
  Mutex.protect s.mutex (fun () -> s.samplers <- th :: s.samplers)

(* Files *)

let log_file s ~step ~key path =
  if not (File.is_key key) then invalid "Session.log_file: %S is not a key" key;
  check_step "log_file" step;
  Mutex.protect s.mutex (fun () -> ensure_open "log_file" s);
  let rel_dir = "files/" ^ key in
  let dir = Filename.concat s.dir rel_dir in
  File.mkdirs dir;
  let tmp = File.temp_name dir in
  let ext = Filename.extension path in
  let rec claim k =
    let name =
      if k = 0 then Printf.sprintf "%d%s" step ext
      else Printf.sprintf "%d-%d%s" step k ext
    in
    match Unix.link tmp (Filename.concat dir name) with
    | () -> name
    | exception Unix.Unix_error (Unix.EEXIST, _, _) -> claim (k + 1)
    | exception Unix.Unix_error (e, _, _) ->
        raise (Sys_error (dir ^ ": " ^ Unix.error_message e))
  in
  let name =
    Fun.protect
      ~finally:(fun () -> try Unix.unlink tmp with Unix.Unix_error _ -> ())
      (fun () ->
        ignore (File.copy ~dst:tmp path);
        claim 0)
  in
  Mutex.protect s.mutex @@ fun () ->
  ensure_open "log_file" s;
  append s (Record.File { key; step; path = rel_dir ^ "/" ^ name })

(* Artifacts *)

let log_artifact s ?step ?(metadata = []) ~name path =
  if not (File.is_segment name) then
    invalid "Session.log_artifact: %S is not a name" name;
  Option.iter (check_step "log_artifact") step;
  check_values "log_artifact" metadata;
  (* The log first, so that a version is never durable before the samples up to
     its step (Store_format, section durability). *)
  Mutex.protect s.mutex (fun () ->
      ensure_open "log_artifact" s;
      sync_locked s);
  Version.publish ~root:(Store.root s.store) ~run:s.id ~step ~metadata ~name
    path

let use_artifact s a =
  Mutex.protect s.mutex @@ fun () ->
  ensure_open "use_artifact" s;
  append s (Record.Use { name = Artifact.name a; version = Artifact.version a })

(* Ending *)

let finish ?(status = `Finished) s =
  let already = Mutex.protect s.mutex (fun () -> s.finished) in
  if not already then begin
    Atomic.set s.stopping true;
    let samplers = Mutex.protect s.mutex (fun () -> s.samplers) in
    List.iter Thread.join samplers;
    let hint_error =
      Mutex.protect s.mutex @@ fun () ->
      Fun.protect ~finally:(fun () ->
          s.finished <- true;
          (* Closing releases the lock. *)
          try Unix.close s.fd with Unix.Unix_error _ -> ())
      @@ fun () ->
      if s.sync_failed then
        failwith (Printf.sprintf "Session.finish: a sync of run %s failed" s.id);
      append s (Record.End status);
      sync_locked s;
      let keys =
        Smap.fold
          (fun k acc l ->
            match Acc.stats acc with
            | None -> l
            | Some stats ->
                let kind =
                  Option.value ~default:`Metric (Smap.find_opt k s.kinds)
                in
                let goal = Option.bind (Smap.find_opt k s.defs) fst in
                (k, { Hint.kind; goal; stats }) :: l)
          s.accs []
        |> List.rev
      in
      match
        File.replace
          (Filename.concat s.dir "stats.json")
          (* The log's own length, which an interrupted count cannot miss. *)
          (Hint.encode ~length:(File.fstat s.fd).Unix.st_size keys)
      with
      | () -> None
      | exception (Sys_error _ as e) -> Some e
    in
    (match s.failure with
    | Some (e, bt) -> Printexc.raise_with_backtrace e bt
    | None -> ());
    Option.iter raise hint_error
  end

let within s f =
  match f s with
  | v ->
      finish s;
      v
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      let status =
        match e with
        | Sys.Break -> `Killed
        | e -> `Failed (Printexc.to_string e)
      in
      (try finish ~status s with _ -> ());
      Printexc.raise_with_backtrace e bt
