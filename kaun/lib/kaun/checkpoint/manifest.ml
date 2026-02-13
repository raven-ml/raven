(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

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

let json_obj pairs =
  Jsont.Json.object' (List.map (fun (k, v) -> (Jsont.Json.name k, v)) pairs)

let json_to_string j =
  match Jsont_bytesrw.encode_string ~format:Jsont.Minify Jsont.json j with
  | Ok s -> s
  | Error e -> failwith e

let json_of_file path =
  let ic = open_in path in
  let s =
    Fun.protect
      ~finally:(fun () -> close_in ic)
      (fun () -> really_input_string ic (in_channel_length ic))
  in
  match Jsont_bytesrw.decode_string Jsont.json s with
  | Ok v -> v
  | Error e -> failwith e

let json_to_file path j =
  let s = json_to_string j in
  let oc = open_out path in
  Fun.protect ~finally:(fun () -> close_out oc) (fun () -> output_string oc s)

let artifact_entry_to_json { kind; label; slug } =
  json_obj
    [
      ("kind", Jsont.Json.string (Artifact.kind_to_string kind));
      ("label", Jsont.Json.string label);
      ("slug", Jsont.Json.string slug);
    ]

let artifact_entry_of_json json =
  match json with
  | Jsont.Object (mems, _) -> (
      match
        ( Jsont.Json.find_mem "kind" mems,
          Jsont.Json.find_mem "label" mems,
          Jsont.Json.find_mem "slug" mems )
      with
      | ( Some (_, Jsont.String (kind_str, _)),
          Some (_, Jsont.String (label, _)),
          Some (_, Jsont.String (slug, _)) ) ->
          let kind =
            match Artifact.kind_of_string kind_str with
            | Some kind -> kind
            | None -> Artifact.Unknown kind_str
          in
          { kind; label; slug }
      | _ -> failwith "Manifest.artifact_entry_of_json: missing fields")
  | _ -> failwith "Manifest.artifact_entry_of_json: expected object"

let to_json { version; step; created_at; tags; metadata; artifacts } =
  json_obj
    [
      ("version", Jsont.Json.int version);
      ("created_at", Jsont.Json.number created_at);
      ("tags", Jsont.Json.list (List.map Jsont.Json.string tags));
      ( "metadata",
        json_obj (List.map (fun (k, v) -> (k, Jsont.Json.string v)) metadata) );
      ("artifacts", Jsont.Json.list (List.map artifact_entry_to_json artifacts));
      ( "step",
        match step with
        | Some s -> Jsont.Json.int s
        | None -> Jsont.Json.null () );
    ]

let of_json json =
  try
    match json with
    | Jsont.Object (mems, _) ->
        let version =
          match Jsont.Json.find_mem "version" mems with
          | Some (_, Jsont.Number (f, _)) -> int_of_float f
          | _ -> failwith "Manifest.of_json: missing version"
        in
        let created_at =
          match Jsont.Json.find_mem "created_at" mems with
          | Some (_, Jsont.Number (f, _)) -> f
          | _ -> failwith "Manifest.of_json: missing created_at"
        in
        let tags =
          match Jsont.Json.find_mem "tags" mems with
          | Some (_, Jsont.Array (tags, _)) ->
              List.map
                (function
                  | Jsont.String (tag, _) -> tag
                  | _ -> failwith "Manifest.of_json: invalid tag")
                tags
          | _ -> []
        in
        let metadata =
          match Jsont.Json.find_mem "metadata" mems with
          | Some (_, Jsont.Object (kvs, _)) ->
              List.map
                (fun ((k, _), v) ->
                  match v with
                  | Jsont.String (value, _) -> (k, value)
                  | _ ->
                      failwith
                        "Manifest.of_json: metadata values must be strings")
                kvs
          | _ -> []
        in
        let artifacts =
          match Jsont.Json.find_mem "artifacts" mems with
          | Some (_, Jsont.Array (artifacts, _)) ->
              List.map artifact_entry_of_json artifacts
          | Some _ -> failwith "Manifest.of_json: artifacts must be a list"
          | None -> []
        in
        let step =
          match Jsont.Json.find_mem "step" mems with
          | Some (_, Jsont.Number (f, _)) -> Some (int_of_float f)
          | Some (_, Jsont.Null _) | None -> None
          | _ -> failwith "Manifest.of_json: invalid step"
        in
        Ok { version; step; created_at; tags; metadata; artifacts }
    | _ -> Error "Manifest.of_json: expected object"
  with Failure msg -> Error msg
