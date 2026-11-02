(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = { root : string }

let absolute p =
  if Filename.is_relative p then Filename.concat (Sys.getcwd ()) p else p

let nonempty v =
  match Sys.getenv_opt v with Some "" | None -> None | Some s -> Some s

(* XDG Base Directory: "If $XDG_DATA_HOME is either not set or empty, a default
   equal to $HOME/.local/share should be used", and a relative path in it is
   invalid and ignored. *)
let default_root () =
  let ( / ) = Filename.concat in
  match nonempty "RAVEN_TRACKING_DIR" with
  | Some d -> d
  | None -> (
      match nonempty "XDG_DATA_HOME" with
      | Some d when not (Filename.is_relative d) -> d / "raven" / "munin"
      | _ -> (
          match nonempty "HOME" with
          | Some h -> h / ".local" / "share" / "raven" / "munin"
          | None -> (
              match (Sys.win32, nonempty "LOCALAPPDATA") with
              | true, Some d -> d / "raven" / "munin"
              | _ ->
                  failwith
                    "Store.open_: no root: set RAVEN_TRACKING_DIR or HOME")))

let open_ ?root () =
  let root = match root with Some r -> r | None -> default_root () in
  { root = absolute root }

let root s = s.root
let runs_dir s = Filename.concat s.root "runs"
let run_dir s id = Filename.concat (runs_dir s) id

(* Runs *)

let compare_runs (a : Listing.t) (b : Listing.t) =
  match Float.compare b.started_at a.started_at with
  | 0 -> String.compare b.id a.id
  | c -> c

let runs ?prev s =
  let prev =
    match prev with
    | None -> Hashtbl.create 0
    | Some l ->
        let h = Hashtbl.create (List.length l) in
        List.iter (fun (r : Listing.t) -> Hashtbl.replace h r.id r) l;
        h
  in
  File.entries (runs_dir s)
  |> List.filter_map (fun id ->
      let prev = Hashtbl.find_opt prev id in
      match Listing.read ?prev ~id (run_dir s id) with
      | Ok r -> Some r
      | Error _ -> None)
  |> List.sort compare_runs

let run s id =
  if not (File.is_segment id) then None
  else
    match Listing.read ~id (run_dir s id) with
    | Ok r -> Some r
    | Error `Not_a_run -> None
    | Error (`Unreadable why) -> failwith (Printf.sprintf "run %s: %s" id why)

let existing s id =
  match run s id with
  | Some r -> r
  | None -> failwith (Printf.sprintf "no run %s in %s" id s.root)

let check_tag t =
  if not (File.is_tag t) then
    invalid_arg
      (Printf.sprintf "Store: %S is not a tag: a text without \",\"" t)

(* Tags and notes take no lock, so a deletion may remove the run's directory
   meanwhile. They create no directory above [tags/], so that they never make
   the run's directory again: they raise instead. *)
let add_tags s id tags =
  List.iter check_tag tags;
  let r = existing s id in
  List.iter
    (fun tag ->
      let path = Listing.tag_file r.dir tag in
      if not (Sys.file_exists path) then begin
        File.mkdir (Filename.dirname path);
        File.replace ~parents:false path tag
      end)
    tags

let remove_tags s id tags =
  let r = existing s id in
  List.iter
    (fun tag ->
      if File.is_tag tag then File.remove_tree (Listing.tag_file r.dir tag))
    tags

let set_notes s id notes =
  (match notes with
  | Some n when not (Json.utf8 n) ->
      invalid_arg "Store.set_notes: notes are not UTF-8"
  | _ -> ());
  let r = existing s id in
  let path = Filename.concat r.dir "notes" in
  match notes with
  | None -> File.remove_tree path
  | Some n -> File.replace ~parents:false path n

(* [remove_in_place dir] removes the directory [dir] of a deleted run, its mark
   last, so that a removal cut short leaves the directory marked. *)
let remove_in_place dir =
  let mark = Listing.mark dir in
  Array.iter
    (fun n ->
      let p = Filename.concat dir n in
      if p <> mark then File.remove_tree p)
    (try Sys.readdir dir with Sys_error _ -> [||]);
  File.remove_tree dir

(* [remove_deleted s dir] moves [dir] to a [.]-named entry of runs/ and removes
   it there, so that no writer that still holds its path, such as {!add_tags},
   adds to it. Windows refuses the move while another program holds a file in
   [dir] open, and [dir] is then removed in place. *)
let remove_deleted s dir =
  let gone = File.temp_name (runs_dir s) in
  match Unix.rename dir gone with
  | () -> File.remove_tree gone
  | exception Unix.Unix_error _ -> remove_in_place dir

let delete_run s id =
  let r = existing s id in
  if r.status = `Silent then
    failwith
      (Printf.sprintf
         "run %s is silent: its writer cannot be probed. Resume it, finish it \
          as killed, then delete it"
         id);
  let log = Listing.log r.dir in
  (* Read-write, for an exclusive lock on NFS, as Session's log. A user who may
     write the run's directory and not its log, as in a group-shared store,
     opens it read-only: a local flock ignores the open mode, and NFS's
     emulation then refuses the lock, which raises. *)
  let fd =
    let open_ flags =
      Unix.openfile log (Unix.O_CLOEXEC :: Unix.O_SHARE_DELETE :: flags) 0
    in
    try open_ [ Unix.O_RDWR ] with
    | Unix.Unix_error (Unix.EACCES, _, _) -> (
        try open_ [ Unix.O_RDONLY ]
        with Unix.Unix_error (e, _, _) ->
          raise (Sys_error (log ^ ": " ^ Unix.error_message e)))
    | Unix.Unix_error (e, _, _) ->
        raise (Sys_error (log ^ ": " ^ Unix.error_message e))
  in
  (* Renaming the log to its mark under the lock ends the run at once for
     readers and writers, so that none meets half a run. The directory goes
     once the lock is let go: Windows renames no directory while a file in it
     is open, and the lock is such a file. *)
  (Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
   if not (File.take_writer fd) then
     failwith (Printf.sprintf "run %s is being written" id);
   try Unix.rename log (Listing.mark r.dir)
   with Unix.Unix_error (e, _, _) ->
     raise (Sys_error (log ^ ": " ^ Unix.error_message e)));
  (* The run is deleted. Whatever its removal leaves, gc removes. *)
  try remove_deleted s r.dir with Sys_error _ -> ()

(* Artifacts *)

let artifacts ?name s =
  let names =
    match name with Some n -> [ n ] | None -> Version.names ~root:s.root
  in
  List.concat_map
    (fun n ->
      if not (File.is_segment n) then []
      else
        List.filter_map
          (fun v ->
            match Version.load ~root:s.root n v with
            | `Version a -> Some a
            | _ -> None)
          (Version.numbers ~root:s.root n))
    names

let alias_dir s name =
  Filename.concat
    (Filename.concat (Filename.concat s.root "artifacts") name)
    "aliases"

let alias_target s name alias =
  match File.read_opt (Filename.concat (alias_dir s name) alias) with
  | Some text -> int_of_string_opt (String.trim text)
  | None | (exception Sys_error _) -> None

let artifact s name sel =
  if not (File.is_segment name) then None
  else
    let version n =
      match Version.load ~root:s.root name n with
      | `Version a -> Some a
      | _ -> None
    in
    match sel with
    | `Version n -> version n
    | `Alias a ->
        if File.is_segment a then Option.bind (alias_target s name a) version
        else None
    | `Latest ->
        List.find_map version (List.rev (Version.numbers ~root:s.root name))

let aliases s name =
  if not (File.is_segment name) then []
  else
    File.entries (alias_dir s name)
    |> List.filter_map (fun a ->
        Option.map (fun n -> (a, n)) (alias_target s name a))

(* An alias and a tombstone are written through a [.]-named temporary under
   [artifacts/], which gc removes under its exclusive lock: their writers hold
   the shared lock, as a publication does, so gc never removes one in progress
   (Store_format, section blobs). *)
let set_alias s a alias =
  if not (File.is_segment alias) then
    invalid_arg (Printf.sprintf "Store.set_alias: %S is not a name" alias);
  Version.with_lock s.root `Shared @@ fun () ->
  File.replace
    (Filename.concat (alias_dir s (Artifact.name a)) alias)
    (string_of_int (Artifact.version a) ^ "\n")

let remove_alias s ~name alias =
  if File.is_segment name && File.is_segment alias then
    File.remove_tree (Filename.concat (alias_dir s name) alias)

let delete_artifact s a =
  let name = Artifact.name a in
  List.iter
    (fun (alias, n) ->
      if n = Artifact.version a then remove_alias s ~name alias)
    (aliases s name);
  Version.with_lock s.root `Shared @@ fun () -> Version.tombstone ~root:s.root a

(* Maintenance *)

let dot_entries dir =
  match Sys.readdir dir with
  | exception Sys_error _ -> []
  | names ->
      Array.to_list names
      |> List.filter (fun n -> n <> "" && n.[0] = '.')
      |> List.map (Filename.concat dir)

(* A collection of a store that does not exist removes nothing, and must not
   create it. *)
let with_store_lock s kind f =
  if Sys.file_exists s.root then Version.with_lock s.root kind f else f ()

let with_publication_lock s f = Version.with_lock s.root `Shared f

let gc s =
  with_store_lock s `Exclusive @@ fun () ->
  let named = Hashtbl.create 64 in
  List.iter
    (fun a -> Hashtbl.replace named (Filename.basename (Artifact.path a)) ())
    (artifacts s);
  let blobs = Version.blob_dir s.root in
  let unnamed =
    File.entries blobs
    |> List.filter (fun h -> not (Hashtbl.mem named h))
    |> List.map (Filename.concat blobs)
  in
  let leftovers =
    dot_entries blobs
    @ dot_entries (Filename.concat s.root "artifacts")
    @ List.concat_map
        (fun n ->
          let d = Filename.concat (Filename.concat s.root "artifacts") n in
          dot_entries d @ dot_entries (Filename.concat d "aliases"))
        (Version.names ~root:s.root)
  in
  let runs = runs_dir s in
  let deleted =
    File.entries runs
    |> List.map (Filename.concat runs)
    |> List.filter Listing.deleted
  in
  let removed = unnamed @ leftovers @ dot_entries runs in
  List.iter File.remove_tree removed;
  List.iter remove_in_place deleted;
  List.length removed + List.length deleted

(* Check *)

let check s =
  let problems = ref [] in
  let add path what = problems := (path, what) :: !problems in
  let runs = runs_dir s in
  List.iter
    (fun id ->
      let dir = Filename.concat runs id in
      let deleted = Listing.deleted dir in
      if not deleted then
        List.iter (fun p -> add p "left by a writer") (dot_entries dir);
      match Listing.read ~id dir with
      | Error `Not_a_run when deleted -> add dir "left by a deletion"
      | Error `Not_a_run ->
          add dir "not a run: its log holds no whole first line"
      | Error (`Unreadable why) -> add dir why
      | Ok r -> (
          (* Tags *)
          let tags = Filename.concat dir "tags" in
          List.iter
            (fun name ->
              if Listing.read_tag dir name = None then
                add (Filename.concat tags name) "damaged tag file")
            (File.entries tags);
          (* The log *)
          let h = Fold.read ~keep:[] r in
          if Fold.damaged h > 0 then
            add (Listing.log dir)
              (Printf.sprintf "damaged lines: %d" (Fold.damaged h));
          let size =
            match File.stamp (Listing.log dir) with
            | Some (_, n, _) -> n
            | None -> 0
          in
          (match r.status with
          | (`Crashed | `Finished | `Failed _ | `Killed)
            when Fold.offset h < size ->
              add (Listing.log dir)
                (Printf.sprintf "%d bytes at offset %d are not whole lines"
                   (size - Fold.offset h)
                   (Fold.offset h))
          | _ -> ());
          (* Files no record names *)
          let named = Fold.named_files h in
          let rec walk rel =
            List.iter
              (fun n ->
                let rel = rel ^ "/" ^ n in
                let p =
                  List.fold_left Filename.concat dir
                    (String.split_on_char '/' rel)
                in
                if Sys.is_directory p then walk rel
                else if not (List.mem rel named) then
                  add p "a file no record names")
              (File.entries
                 (List.fold_left Filename.concat dir
                    (String.split_on_char '/' rel)))
          in
          walk "files";
          (* The statistics hint *)
          let hint = Filename.concat dir "stats.json" in
          match File.read_opt hint with
          | None -> ()
          | Some text -> (
              match Hint.decode text with
              | None ->
                  add hint
                    "a statistics hint that does not parse or is of another \
                     format"
              | Some (length, keys) ->
                  if length = size && not (Hint.equal keys (Fold.hint_keys h))
                  then add hint "a statistics hint that disagrees with its log")
          | exception Sys_error _ -> add hint "unreadable"))
    (File.entries runs);
  List.iter (fun p -> add p "left by a writer") (dot_entries runs);
  (* Artifacts *)
  List.iter
    (fun name ->
      let d = Filename.concat (Filename.concat s.root "artifacts") name in
      List.iter (fun p -> add p "left by a writer") (dot_entries d);
      let versions = Version.numbers ~root:s.root name in
      List.iter
        (fun n ->
          match Version.load ~root:s.root name n with
          | `Missing ->
              add
                (Filename.concat d (string_of_int n ^ ".json"))
                "unreadable version file"
          | `Deleted -> ()
          | `Version a ->
              if not (Sys.file_exists (Artifact.path a)) then
                add (Artifact.path a) "a version's blob is missing"
              else
                let hex, _ = Version.tree (Artifact.path a) in
                if "blake2b-256:" ^ hex <> Artifact.digest a then
                  add (Artifact.path a) "content does not match its digest")
        versions;
      List.iter
        (fun alias ->
          let p = Filename.concat (Filename.concat d "aliases") alias in
          match alias_target s name alias with
          | Some n
            when match Version.load ~root:s.root name n with
                 | `Version _ -> true
                 | _ -> false ->
              ()
          | _ -> add p "an alias that names no version")
        (File.entries (Filename.concat d "aliases")))
    (Version.names ~root:s.root);
  List.iter
    (fun p -> add p "left by a writer")
    (dot_entries (Version.blob_dir s.root)
    @ dot_entries (Filename.concat s.root "artifacts"));
  List.sort compare !problems
