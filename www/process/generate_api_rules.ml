let read_sidebar_json path =
  if Sys.file_exists path then
    Yojson.Safe.from_file path
  else
    failwith (Printf.sprintf "sidebar.json not found at %s" path)

type module_info = {
  name: string; [@warning "-69"]
  url: string;
  path: string list; [@warning "-69"]
}

let rec extract_modules ?(path=[]) json =
  match json with
  | `Assoc fields ->
    let node = 
      try List.assoc "node" fields 
      with Not_found -> failwith "No node field in entry"
    in
    let children = 
      try 
        match List.assoc "children" fields with
        | `List l -> l
        | _ -> []
      with Not_found -> []
    in
    (match node with
    | `Assoc node_fields ->
      let url = 
        try 
          match List.assoc "url" node_fields with
          | `String s -> Some s
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
          | _ -> None
        with Not_found -> None
      in
      (match kind, url with
      | Some ("module" | "module-type"), Some url ->
        let module_info = { name = content; url; path } in
        module_info :: List.concat_map (extract_modules ~path:(path @ [content])) children
      | _ ->
        List.concat_map (extract_modules ~path) children)
    | _ -> [])
  | _ -> []

let process_sidebar json =
  match json with
  | `List entries -> 
    List.concat_map (extract_modules ~path:[]) entries
  | _ -> failwith "Expected array at top level of sidebar.json"

let generate_dune_rule library module_info =
  let open Printf in
  (* Skip URLs with anchors *)
  if String.contains module_info.url '#' then
    None
  else
    (* Extract the path from the URL *)
    let url_parts = String.split_on_char '/' module_info.url in
    let odoc_path = 
      match List.filter (fun s -> s <> "") url_parts with
      | "nx" :: rest -> String.concat "/" rest
      | _ -> failwith (sprintf "Unexpected URL format: %s" module_info.url)
    in
    let target_path = 
      String.sub odoc_path 0 (String.length odoc_path - 10) (* Remove "index.html" *)
    in
    let depth = List.length (String.split_on_char '/' target_path) in
    let back_path = String.concat "/" (List.init (depth + 4) (fun _ -> "..")) in
    
    Some (sprintf "(rule
  (mode promote)
  (deps
   (:index %s/process/index.exe)
   (:source %s/odoc/%s/%s))
  (targets %sindex.html)
  (action
   (run %%{index} %%{source} %s %sindex.html)))"
      back_path back_path library odoc_path target_path library target_path)

let () =
  if Array.length Sys.argv < 4 then
    failwith "Usage: generate_api_rules <library> <sidebar.json> <output.inc>"
  else
    let library = Sys.argv.(1) in
    let sidebar_json_path = Sys.argv.(2) in
    let output_path = Sys.argv.(3) in
    
    let json = read_sidebar_json sidebar_json_path in
    let modules = process_sidebar json in
    
    (* Skip the main Nx module as it's already handled *)
    let other_modules = List.filter (fun m -> m.url <> "nx/nx/Nx/index.html") modules in
    
    let rules = List.filter_map (generate_dune_rule library) other_modules in
    let output = String.concat "\n\n" rules in
    
    let oc = open_out output_path in
    output_string oc output;
    close_out oc