type kind = [ `dataset | `model | `checkpoint | `file | `dir | `other ]
type payload = [ `file | `dir ]

type t = {
  root : string;
  name : string;
  kind : kind;
  payload : payload;
  version : string;
  digest : string;
  materialized_rel_path : string;
  size_bytes : int;
  metadata : (string * Jsont.json) list;
  aliases : string list;
  producer_run_id : string option;
  consumer_run_ids : string list;
  created_at : float;
}

let schema_version = 2
let name t = t.name
let kind t = t.kind
let payload t = t.payload
let version t = t.version
let digest t = t.digest
let size_bytes t = t.size_bytes
let metadata t = List.map (fun (k, v) -> (k, Value.of_json v)) t.metadata
let aliases t = t.aliases
let producer_run_id t = t.producer_run_id
let consumer_run_ids t = t.consumer_run_ids
let created_at t = t.created_at
let path t = Filename.concat t.root t.materialized_rel_path
let has_alias t alias = List.exists (String.equal alias) t.aliases

let kind_of_string = function
  | "dataset" -> `dataset
  | "model" -> `model
  | "checkpoint" -> `checkpoint
  | "file" -> `file
  | "dir" -> `dir
  | _ -> `other

let kind_to_string : kind -> string = function
  | `dataset -> "dataset"
  | `model -> "model"
  | `checkpoint -> "checkpoint"
  | `file -> "file"
  | `dir -> "dir"
  | `other -> "other"

let payload_of_string = function "dir" -> `dir | _ -> `file

let payload_to_string : payload -> string = function
  | `file -> "file"
  | `dir -> "dir"

let versions_dir root name =
  Filename.concat
    (Filename.concat (Filename.concat root "artifacts") name)
    "versions"

let manifest_path root name version =
  Filename.concat
    (Filename.concat (versions_dir root name) version)
    "manifest.json"

let load_manifest root path =
  try
    let json = Fs.read_file path |> Json_utils.json_of_string in
    let schema_ok =
      match
        Json_utils.json_mem "schema_version" json |> Json_utils.json_number
      with
      | Some v -> int_of_float v = schema_version
      | None -> false
    in
    if not schema_ok then None
    else
      match
        ( Json_utils.json_mem "name" json |> Json_utils.json_string,
          Json_utils.json_mem "version" json |> Json_utils.json_string,
          Json_utils.json_mem "kind" json |> Json_utils.json_string,
          Json_utils.json_mem "payload" json |> Json_utils.json_string,
          Json_utils.json_mem "digest" json |> Json_utils.json_string,
          Json_utils.json_mem "path" json |> Json_utils.json_string,
          Json_utils.json_mem "size_bytes" json |> Json_utils.json_number )
      with
      | ( Some name,
          Some version,
          Some kind,
          Some payload,
          Some digest,
          Some materialized_rel_path,
          Some size_bytes ) ->
          Some
            {
              root;
              name;
              kind = kind_of_string kind;
              payload = payload_of_string payload;
              version;
              digest;
              materialized_rel_path;
              size_bytes = int_of_float size_bytes;
              metadata =
                Json_utils.json_mem "metadata" json |> Json_utils.json_assoc;
              aliases =
                Json_utils.json_mem "aliases" json
                |> Json_utils.json_string_list;
              producer_run_id =
                Json_utils.json_mem "producer_run_id" json
                |> Json_utils.json_string;
              consumer_run_ids =
                Json_utils.json_mem "consumer_run_ids" json
                |> Json_utils.json_string_list;
              created_at =
                Option.value
                  (Json_utils.json_mem "created_at" json
                  |> Json_utils.json_number)
                  ~default:0.0;
            }
      | _ -> None
  with _ -> None

let write_manifest root name version artifact =
  let json =
    Json_utils.json_obj
      ([
         ("schema_version", Jsont.Json.int schema_version);
         ("name", Jsont.Json.string artifact.name);
         ("version", Jsont.Json.string artifact.version);
         ("kind", Jsont.Json.string (kind_to_string artifact.kind));
         ("payload", Jsont.Json.string (payload_to_string artifact.payload));
         ("digest", Jsont.Json.string artifact.digest);
         ("path", Jsont.Json.string artifact.materialized_rel_path);
         ("size_bytes", Jsont.Json.int artifact.size_bytes);
         ("metadata", Json_utils.json_obj artifact.metadata);
         ( "aliases",
           Jsont.Json.list (List.map Jsont.Json.string artifact.aliases) );
         ( "consumer_run_ids",
           Jsont.Json.list
             (List.map Jsont.Json.string artifact.consumer_run_ids) );
         ("created_at", Jsont.Json.number artifact.created_at);
       ]
      @
      match artifact.producer_run_id with
      | None -> []
      | Some run_id -> [ ("producer_run_id", Jsont.Json.string run_id) ])
  in
  Fs.write_file
    (manifest_path root name version)
    (Json_utils.json_to_string ~pretty:true json ^ "\n")

let version_number version =
  if String.length version >= 2 && version.[0] = 'v' then
    int_of_string_opt (String.sub version 1 (String.length version - 1))
  else None

let compare_version a b =
  match (version_number a.version, version_number b.version) with
  | Some a_num, Some b_num -> Int.compare a_num b_num
  | _ ->
      let by_name = String.compare a.name b.name in
      if by_name <> 0 then by_name else String.compare a.version b.version

let resolve_alias root name alias =
  Fs.list_dirs (versions_dir root name)
  |> List.filter_map (fun version ->
      load_manifest root (manifest_path root name version))
  |> List.filter (fun artifact -> has_alias artifact alias)
  |> List.sort compare_version |> List.rev
  |> function
  | artifact :: _ -> Some artifact
  | [] -> None

let load ~root ~name ~version =
  let path = manifest_path root name version in
  if Sys.file_exists path then load_manifest root path
  else resolve_alias root name version

let list ~root ?name ?kind ?alias ?producer_run ?consumer_run () =
  let names =
    match name with
    | Some name -> [ name ]
    | None -> Fs.list_dirs (Filename.concat root "artifacts")
  in
  List.concat_map
    (fun name ->
      Fs.list_dirs (versions_dir root name)
      |> List.filter_map (fun version ->
          load_manifest root (manifest_path root name version)))
    names
  |> List.filter (fun artifact ->
      Option.fold ~none:true ~some:(fun k -> artifact.kind = k) kind
      && Option.fold ~none:true ~some:(has_alias artifact) alias
      && Option.fold ~none:true
           ~some:(fun run_id -> artifact.producer_run_id = Some run_id)
           producer_run
      && Option.fold ~none:true
           ~some:(fun run_id ->
             List.exists (String.equal run_id) artifact.consumer_run_ids)
           consumer_run)
  |> List.sort (fun a b ->
      let by_name = String.compare a.name b.name in
      if by_name <> 0 then by_name else compare_version a b)

let create ~root ~name ~kind ~payload ~digest ~path:size_path ~metadata ~aliases
    ~producer_run_id =
  let version =
    let max_version =
      Fs.list_dirs (versions_dir root name)
      |> List.filter_map version_number
      |> List.fold_left max 0
    in
    Printf.sprintf "v%d" (max_version + 1)
  in
  let artifact =
    {
      root;
      name;
      kind;
      payload;
      version;
      digest;
      materialized_rel_path = size_path;
      size_bytes = Fs.path_size (Filename.concat root size_path);
      metadata;
      aliases;
      producer_run_id;
      consumer_run_ids = [];
      created_at = Unix.gettimeofday ();
    }
  in
  write_manifest root name version artifact;
  artifact

let add_consumer ~root ~name ~version run_id =
  match load ~root ~name ~version with
  | None -> ()
  | Some artifact ->
      if List.exists (String.equal run_id) artifact.consumer_run_ids then ()
      else
        let artifact =
          {
            artifact with
            consumer_run_ids = artifact.consumer_run_ids @ [ run_id ];
          }
        in
        write_manifest root name version artifact
