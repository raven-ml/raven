type artifact_entry = { kind : Artifact.kind; label : string; slug : string }

type t = {
  version : int;
  step : int option;
  created_at : float;
  tags : string list;
  metadata : (string * string) list;
  artifacts : artifact_entry list;
}

let current_version = 1

let create ?step ?(tags : string list = [])
    ?(metadata : (string * string) list = []) ~artifacts () =
  {
    version = current_version;
    step;
    created_at = Util.now_unix ();
    tags;
    metadata;
    artifacts;
  }

let artifact_entry_to_yojson { kind; label; slug } =
  `Assoc
    [
      ("kind", `String (Artifact.kind_to_string kind));
      ("label", `String label);
      ("slug", `String slug);
    ]

let artifact_entry_of_yojson = function
  | `Assoc fields -> (
      match
        ( List.assoc_opt "kind" fields,
          List.assoc_opt "label" fields,
          List.assoc_opt "slug" fields )
      with
      | Some (`String kind_str), Some (`String label), Some (`String slug) ->
          let kind =
            match Artifact.kind_of_string kind_str with
            | Some kind -> kind
            | None -> Artifact.Unknown kind_str
          in
          { kind; label; slug }
      | _ -> failwith "Manifest.artifact_entry_of_yojson: missing fields")
  | _ -> failwith "Manifest.artifact_entry_of_yojson: expected object"

let to_yojson { version; step; created_at; tags; metadata; artifacts } =
  `Assoc
    [
      ("version", `Int version);
      ("created_at", `Float created_at);
      ("tags", `List (List.map (fun tag -> `String tag) tags));
      ("metadata", `Assoc (List.map (fun (k, v) -> (k, `String v)) metadata));
      ("artifacts", `List (List.map artifact_entry_to_yojson artifacts));
      ("step", match step with Some s -> `Int s | None -> `Null);
    ]

let of_yojson json =
  try
    match json with
    | `Assoc fields ->
        let version =
          match List.assoc_opt "version" fields with
          | Some (`Int v) -> v
          | _ -> failwith "Manifest.of_yojson: missing version"
        in
        let created_at =
          match List.assoc_opt "created_at" fields with
          | Some (`Float ts) -> ts
          | Some (`Int ts) -> float_of_int ts
          | _ -> failwith "Manifest.of_yojson: missing created_at"
        in
        let tags =
          match List.assoc_opt "tags" fields with
          | Some (`List tags) ->
              List.map
                (function
                  | `String tag -> tag
                  | _ -> failwith "Manifest.of_yojson: invalid tag")
                tags
          | _ -> []
        in
        let metadata =
          match List.assoc_opt "metadata" fields with
          | Some (`Assoc kvs) ->
              List.map
                (fun (k, v) ->
                  match v with
                  | `String value -> (k, value)
                  | _ ->
                      failwith
                        "Manifest.of_yojson: metadata values must be strings")
                kvs
          | _ -> []
        in
        let artifacts =
          match List.assoc_opt "artifacts" fields with
          | Some (`List artifacts) ->
              List.map artifact_entry_of_yojson artifacts
          | Some _ -> failwith "Manifest.of_yojson: artifacts must be a list"
          | None -> []
        in
        let step =
          match List.assoc_opt "step" fields with
          | Some (`Int s) -> Some s
          | Some `Null | None -> None
          | _ -> failwith "Manifest.of_yojson: invalid step"
        in
        Ok { version; step; created_at; tags; metadata; artifacts }
    | _ -> Error "Manifest.of_yojson: expected object"
  with Failure msg -> Error msg
