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
  let list_field key values =
    (key, `List (List.map (fun v -> `String v) values))
  in
  let base_fields =
    [
      list_field "authors" metadata.authors;
      list_field "render_modes" metadata.render_modes;
      list_field "supported_vector_modes" metadata.supported_vector_modes;
      list_field "tags" metadata.tags;
    ]
  in
  let add_opt key to_json opt acc =
    match opt with None -> acc | Some value -> (key, to_json value) :: acc
  in
  let fields =
    base_fields
    |> add_opt "description" (fun s -> `String s) metadata.description
    |> add_opt "extra" (fun json -> json) metadata.extra
    |> add_opt "render_fps" (fun fps -> `Int fps) metadata.render_fps
    |> add_opt "version" (fun v -> `String v) metadata.version
  in
  let sorted = List.sort (fun (a, _) (b, _) -> String.compare a b) fields in
  `Assoc sorted

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
  | _ ->
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
  | _ ->
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

let with_render_modes modes metadata = { metadata with render_modes = modes }

let with_supported_vector_modes modes metadata =
  { metadata with supported_vector_modes = modes }

let add_supported_vector_mode mode metadata =
  if List.exists (String.equal mode) metadata.supported_vector_modes then
    metadata
  else
    {
      metadata with
      supported_vector_modes = metadata.supported_vector_modes @ [ mode ];
    }

let with_authors authors metadata = { metadata with authors }
let with_extra extra metadata = { metadata with extra }
