let read_sidebar_json path =
  if Sys.file_exists path then Yojson.Safe.from_file path
  else failwith (Printf.sprintf "sidebar.json not found at %s" path)

type entry = {
  name : string;
  url : string option;
  kind : string option;
  children : entry list;
}

let rec process_entry json =
  match json with
  | `Assoc fields -> (
      let node =
        try List.assoc "node" fields
        with Not_found -> failwith "No node field in entry"
      in
      let children =
        try match List.assoc "children" fields with `List l -> l | _ -> []
        with Not_found -> []
      in
      match node with
      | `Assoc node_fields ->
          let url =
            try
              match List.assoc "url" node_fields with
              | `String s -> Some s
              | `Null -> None
              | _ -> None
            with Not_found -> None
          in
          let content =
            try
              match List.assoc "content" node_fields with
              | `String s -> s
              | _ -> ""
            with Not_found -> ""
          in
          let kind =
            try
              match List.assoc "kind" node_fields with
              | `String s -> Some s
              | `Null -> None
              | _ -> None
            with Not_found -> None
          in
          let children_entries = List.filter_map process_entry children in
          (* Create entry with processed children *)
          Some { name = content; url; kind; children = children_entries }
      | _ -> None)
  | _ -> None

let rec collect_modules entry =
  match entry.kind with
  | Some "module" | Some "module-type" -> [ entry ]
  | _ ->
      (* For library groups or other containers, collect modules from
         children *)
      List.concat_map collect_modules entry.children

let process_sidebar json =
  match json with
  | `List entries ->
      let all_entries = List.filter_map process_entry entries in
      (* Collect all modules from the tree *)
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
