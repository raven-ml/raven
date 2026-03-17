type t = { root : string }

let root t = t.root

let open_ ?root () =
  let root = Option.value root ~default:(Env.root ()) in
  Fs.ensure_dir (Filename.concat root "experiments");
  Fs.ensure_dir (Filename.concat root "artifacts");
  Fs.ensure_dir (Filename.concat (Filename.concat root "blobs") "sha256");
  { root }

let list_experiments t = Fs.list_dirs (Filename.concat t.root "experiments")

(* Index-based listing: filter on header fields, return lazy Run.t *)
let list_runs_indexed index ~root ?experiment ?status ?tag ?parent ?group () =
  Hashtbl.to_seq index |> List.of_seq
  |> List.filter_map (fun (id, (entry : Index.entry)) ->
      if
        Option.fold ~none:true ~some:(String.equal entry.experiment) experiment
        && Option.fold ~none:true ~some:(( = ) entry.status) status
        && Option.fold ~none:true
             ~some:(fun t -> List.exists (String.equal t) entry.tags)
             tag
        && Option.fold ~none:true
             ~some:(fun p -> entry.parent_id = Some p)
             parent
        && Option.fold ~none:true ~some:(fun g -> entry.group = Some g) group
      then Some (Run.load_from_index ~root id entry)
      else None)
  |> List.sort (fun a b -> String.compare (Run.id b) (Run.id a))

(* Filesystem-based listing: fallback when no index *)
let list_runs_scan ~root ?experiment ?status ?tag ?parent ?group () =
  let runs =
    match experiment with
    | Some experiment ->
        Run.list ~root ~experiment ?status ?tag ?parent ?group ()
    | None ->
        Fs.list_dirs (Filename.concat root "experiments")
        |> List.concat_map (fun experiment ->
            Run.list ~root ~experiment ?status ?tag ?parent ?group ())
  in
  List.sort (fun a b -> String.compare (Run.id b) (Run.id a)) runs

let list_runs t ?experiment ?status ?tag ?parent ?group () =
  match Index.read t.root with
  | Some index ->
      list_runs_indexed index ~root:t.root ?experiment ?status ?tag ?parent
        ?group ()
  | None ->
      list_runs_scan ~root:t.root ?experiment ?status ?tag ?parent ?group ()

let find_run t id =
  match Index.read t.root with
  | Some index -> (
      match Hashtbl.find_opt index id with
      | Some entry -> Some (Run.load_from_index ~root:t.root id entry)
      | None -> None)
  | None ->
      List.find_map
        (fun experiment -> Run.load ~root:t.root ~experiment ~id)
        (list_experiments t)

let latest_run t ?experiment ?status ?tag ?group () =
  match list_runs t ?experiment ?status ?tag ?group () with
  | run :: _ -> Some run
  | [] -> None

let find_artifact t ~name ~version = Artifact.load ~root:t.root ~name ~version

let list_artifacts t ?name ?kind ?alias ?producer_run ?consumer_run () =
  Artifact.list ~root:t.root ?name ?kind ?alias ?producer_run ?consumer_run ()

let delete_run t run =
  Fs.remove_tree (Run.dir run);
  let exp_dir =
    Filename.concat
      (Filename.concat t.root "experiments")
      (Run.experiment_name run)
  in
  if Fs.list_dirs (Filename.concat exp_dir "runs") = [] then
    Fs.remove_tree exp_dir;
  Index.remove t.root ~id:(Run.id run)

let gc t =
  let blobs_dir = Filename.concat (Filename.concat t.root "blobs") "sha256" in
  let referenced = Hashtbl.create 64 in
  List.iter
    (fun artifact -> Hashtbl.replace referenced (Artifact.digest artifact) ())
    (list_artifacts t ());
  let removed = ref 0 in
  Fs.list_dirs blobs_dir
  |> List.iter (fun digest ->
      if not (Hashtbl.mem referenced digest) then (
        Fs.remove_tree (Filename.concat blobs_dir digest);
        incr removed));
  !removed
