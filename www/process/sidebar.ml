let read_sidebar_json path =
  if Sys.file_exists path then
    let ic = open_in path in
    let content =
      Fun.protect ~finally:(fun () -> close_in ic) (fun () ->
          In_channel.input_all ic)
    in
    match Jsont_bytesrw.decode_string Jsont.json content with
    | Ok v -> v
    | Error e -> failwith e
  else failwith (Printf.sprintf "sidebar.json not found at %s" path)

let find_field name mems =
  match Jsont.Json.find_mem name mems with
  | Some (_, v) -> v
  | None -> raise Not_found

type entry = {
  name : string;
  url : string option;
  kind : string option;
  children : entry list;
}

let rec process_entry json =
  match json with
  | Jsont.Object (mems, _) -> (
      let node =
        try find_field "node" mems
        with Not_found -> failwith "No node field in entry"
      in
      let children =
        try
          match find_field "children" mems with
          | Jsont.Array (l, _) -> l
          | _ -> []
        with Not_found -> []
      in
      match node with
      | Jsont.Object (node_fields, _) ->
          let url =
            try
              match find_field "url" node_fields with
              | Jsont.String (s, _) -> Some s
              | Jsont.Null _ -> None
              | _ -> None
            with Not_found -> None
          in
          let content =
            try
              match find_field "content" node_fields with
              | Jsont.String (s, _) -> s
              | _ -> ""
            with Not_found -> ""
          in
          let kind =
            try
              match find_field "kind" node_fields with
              | Jsont.String (s, _) -> Some s
              | Jsont.Null _ -> None
              | _ -> None
            with Not_found -> None
          in
          let children_entries = List.filter_map process_entry children in
          Some { name = content; url; kind; children = children_entries }
      | _ -> None)
  | _ -> None

let rec collect_modules entry =
  match entry.kind with
  | Some "module" | Some "module-type" -> [ entry ]
  | _ ->
      List.concat_map collect_modules entry.children

let process_sidebar json =
  match json with
  | Jsont.Array (entries, _) ->
      let all_entries = List.filter_map process_entry entries in
      List.concat_map collect_modules all_entries
  | _ -> failwith "Expected array at top level of sidebar.json"

let rec generate_html ~base_path entries =
  let open Printf in
  List.map
    (fun entry ->
      match entry.url with
      | Some url ->
          (* Convert URL from nx/... to /docs/nx/api/... *)
          let link =
            if String.length url > 3 && String.sub url 0 3 = "nx/" then
              let rest = String.sub url 3 (String.length url - 3) in
              sprintf "/docs/nx/api/%s" rest
            else sprintf "%s%s" base_path url
          in
          let children_html =
            if entry.children = [] then ""
            else
              let child_modules =
                List.concat_map collect_modules entry.children
              in
              if child_modules = [] then ""
              else
                sprintf "\n<ul>\n%s</ul>"
                  (generate_html ~base_path child_modules)
          in
          sprintf "<li><a href=\"%s\">%s</a>%s</li>" link entry.name
            children_html
      | None ->
          (* Group without URL - shouldn't happen for modules *)
          "")
    entries
  |> List.filter (fun s -> s <> "")
  |> String.concat "\n"

let generate_api_nav_html ~library:_ ~sidebar_json_path ~output_path =
  let json = read_sidebar_json sidebar_json_path in
  let entries = process_sidebar json in
  let html = generate_html ~base_path:"/" entries in
  let full_html =
    if html = "" then
      "<ul class=\"nav-links\">\n<li>No API documentation available</li>\n</ul>"
    else Printf.sprintf "<ul class=\"nav-links\">\n%s\n</ul>" html
  in
  let oc = open_out output_path in
  output_string oc full_html;
  close_out oc

let () =
  if Array.length Sys.argv < 4 then
    failwith "Usage: sidebar <library> <sidebar.json> <output.html>"
  else
    let library = Sys.argv.(1) in
    let sidebar_json_path = Sys.argv.(2) in
    let output_path = Sys.argv.(3) in
    generate_api_nav_html ~library ~sidebar_json_path ~output_path
