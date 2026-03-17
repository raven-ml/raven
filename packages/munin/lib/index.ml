type status = [ `running | `finished | `failed | `killed ]

type entry = {
  experiment : string;
  name : string option;
  group : string option;
  parent_id : string option;
  status : status;
  tags : string list;
  started_at : float;
}

let index_path root = Filename.concat root "index.json"

let status_to_string = function
  | `running -> "running"
  | `finished -> "finished"
  | `failed -> "failed"
  | `killed -> "killed"

let entry_to_json entry =
  Json_utils.json_obj
    ([
       ("experiment", Jsont.Json.string entry.experiment);
       ("status", Jsont.Json.string (status_to_string entry.status));
       ("tags", Jsont.Json.list (List.map Jsont.Json.string entry.tags));
       ("started_at", Jsont.Json.number entry.started_at);
     ]
    @ (match entry.name with
      | Some n -> [ ("name", Jsont.Json.string n) ]
      | None -> [])
    @ (match entry.group with
      | Some g -> [ ("group", Jsont.Json.string g) ]
      | None -> [])
    @
    match entry.parent_id with
    | Some p -> [ ("parent_id", Jsont.Json.string p) ]
    | None -> [])

let entry_of_json json =
  match Json_utils.json_mem "experiment" json |> Json_utils.json_string with
  | None -> None
  | Some experiment ->
      let status : status =
        match Json_utils.json_mem "status" json |> Json_utils.json_string with
        | Some "finished" -> `finished
        | Some "failed" -> `failed
        | Some "killed" -> `killed
        | _ -> `running
      in
      let tags =
        Json_utils.json_mem "tags" json |> Json_utils.json_string_list
      in
      let started_at =
        Option.value
          (Json_utils.json_mem "started_at" json |> Json_utils.json_number)
          ~default:0.0
      in
      Some
        {
          experiment;
          name = Json_utils.json_mem "name" json |> Json_utils.json_string;
          group = Json_utils.json_mem "group" json |> Json_utils.json_string;
          parent_id =
            Json_utils.json_mem "parent_id" json |> Json_utils.json_string;
          status;
          tags;
          started_at;
        }

let read root =
  let path = index_path root in
  if not (Sys.file_exists path) then None
  else
    try
      let json = Fs.read_file path |> Json_utils.json_of_string in
      let tbl = Hashtbl.create 64 in
      List.iter
        (fun (id, value) ->
          match entry_of_json value with
          | Some entry -> Hashtbl.replace tbl id entry
          | None -> ())
        (Json_utils.json_assoc json);
      Some tbl
    with _ -> None

let write root tbl =
  let entries =
    Hashtbl.to_seq tbl |> List.of_seq
    |> List.sort (fun (a, _) (b, _) -> String.compare b a)
    |> List.map (fun (id, entry) -> (id, entry_to_json entry))
  in
  let json = Json_utils.json_obj entries in
  Fs.write_file (index_path root)
    (Json_utils.json_to_string ~pretty:true json ^ "\n")

let modify root f =
  let tbl =
    match read root with Some tbl -> tbl | None -> Hashtbl.create 16
  in
  f tbl;
  write root tbl

let add root ~id entry = modify root (fun tbl -> Hashtbl.replace tbl id entry)

let update_status root ~id status =
  modify root (fun tbl ->
      match Hashtbl.find_opt tbl id with
      | Some entry -> Hashtbl.replace tbl id { entry with status }
      | None -> ())

let remove root ~id = modify root (fun tbl -> Hashtbl.remove tbl id)
