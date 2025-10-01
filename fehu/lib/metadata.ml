open Yojson.Safe.Util

type t = {
  render_modes : string list;
  render_fps : int option;
  authors : string list;
  description : string option;
  version : string option;
  supported_vector_modes : string list;
  tags : string list;
  extra : Yojson.Safe.t option;
}

let default =
  {
    render_modes = [];
    render_fps = None;
    authors = [];
    description = None;
    version = None;
    supported_vector_modes = [];
    tags = [];
    extra = None;
  }

let add_render_mode mode metadata =
  if List.exists (String.equal mode) metadata.render_modes then metadata
  else { metadata with render_modes = metadata.render_modes @ [ mode ] }

let supports_render_mode mode metadata =
  List.exists (String.equal mode) metadata.render_modes

let with_render_fps render_fps metadata = { metadata with render_fps }
let with_description description metadata = { metadata with description }
let with_version version metadata = { metadata with version }

let add_author author metadata =
  if List.exists (String.equal author) metadata.authors then metadata
  else { metadata with authors = metadata.authors @ [ author ] }

let add_tag tag metadata =
  if List.exists (String.equal tag) metadata.tags then metadata
  else { metadata with tags = metadata.tags @ [ tag ] }

let set_tags tags metadata = { metadata with tags }

let to_yojson metadata =
  let assoc =
    [
      ( "render_modes",
        `List (List.map (fun mode -> `String mode) metadata.render_modes) );
      ( "supported_vector_modes",
        `List
          (List.map (fun mode -> `String mode) metadata.supported_vector_modes)
      );
      ( "authors",
        `List (List.map (fun author -> `String author) metadata.authors) );
      ("tags", `List (List.map (fun tag -> `String tag) metadata.tags));
    ]
  in
  let assoc =
    match metadata.render_fps with
    | None -> assoc
    | Some fps -> ("render_fps", `Int fps) :: assoc
  in
  let assoc =
    match metadata.description with
    | None -> assoc
    | Some description -> ("description", `String description) :: assoc
  in
  let assoc =
    match metadata.version with
    | None -> assoc
    | Some version -> ("version", `String version) :: assoc
  in
  let assoc =
    match metadata.extra with
    | None -> assoc
    | Some extra -> ("extra", extra) :: assoc
  in
  `Assoc assoc

let string_list_member key json =
  match member key json with
  | `Null -> []
  | `List values -> List.map to_string values
  | _ ->
      raise
        (Failure (Printf.sprintf "metadata.%s must be a list of strings" key))

let optional_string key json =
  match member key json with
  | `Null -> None
  | `String s -> Some s
  | `Assoc _ | `List _ | `Bool _ | `Float _ | `Int _ | `Intlit _ | `Tuple _
  | `Variant _ ->
      raise
        (Failure (Printf.sprintf "metadata.%s must be a string or null" key))

let optional_int key json =
  match member key json with
  | `Null -> None
  | `Int i -> Some i
  | `Intlit lit ->
      raise
        (Failure
           (Printf.sprintf
              "metadata.%s must be an int or null (string literal %S received)"
              key lit))
  | `Float _ | `Bool _ | `String _ | `Assoc _ | `List _ | `Tuple _ | `Variant _
    ->
      raise (Failure (Printf.sprintf "metadata.%s must be an int or null" key))

let extra_member json =
  match member "extra" json with `Null -> None | value -> Some value

let of_yojson json =
  try
    let render_modes = string_list_member "render_modes" json in
    let supported_vector_modes =
      string_list_member "supported_vector_modes" json
    in
    let authors = string_list_member "authors" json in
    let tags = string_list_member "tags" json in
    let render_fps = optional_int "render_fps" json in
    let description = optional_string "description" json in
    let version = optional_string "version" json in
    let extra = extra_member json in
    Ok
      {
        render_modes;
        render_fps;
        authors;
        description;
        version;
        supported_vector_modes;
        tags;
        extra;
      }
  with Failure msg -> Error msg
