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

type module_info = {
  name : string; [@warning "-69"]
  url : string;
  path : string list; [@warning "-69"]
}

let rec extract_modules ?(path = []) json =
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
      | Jsont.Object (node_fields, _) -> (
          let url =
            try
              match find_field "url" node_fields with
              | Jsont.String (s, _) -> Some s
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
              | _ -> None
            with Not_found -> None
          in
          match (kind, url) with
          | Some ("module" | "module-type"), Some url ->
              let module_info = { name = content; url; path } in
              module_info
              :: List.concat_map
                   (extract_modules ~path:(path @ [ content ]))
                   children
          | _ -> List.concat_map (extract_modules ~path) children)
      | _ -> [])
  | _ -> []

let process_sidebar json =
  match json with
  | Jsont.Array (entries, _) ->
      List.concat_map (extract_modules ~path:[]) entries
  | _ -> failwith "Expected array at top level of sidebar.json"

let generate_dune_rule library module_info =
  let open Printf in
  (* Skip URLs with anchors *)
  if String.contains module_info.url '#' then None
  else
    (* Extract the path from the URL *)
    let url_parts = String.split_on_char '/' module_info.url in
    let odoc_path =
      match List.filter (fun s -> s <> "") url_parts with
      | "nx" :: rest -> String.concat "/" rest
      | _ -> failwith (sprintf "Unexpected URL format: %s" module_info.url)
    in
    let target_path =
      String.sub odoc_path 0 (String.length odoc_path - 10)
      (* Remove "index.html" *)
    in
    let depth = List.length (String.split_on_char '/' target_path) in
    let back_path = String.concat "/" (List.init (depth + 4) (fun _ -> "..")) in

    Some
      (sprintf
         "(rule\n\
         \  (mode promote)\n\
         \  (deps\n\
         \   (:index %s/process/index.exe)\n\
         \   (:source %s/odoc/%s/%s))\n\
         \  (targets %sindex.html)\n\
         \  (action\n\
         \   (run %%{index} %%{source} %s %sindex.html)))"
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
    let other_modules =
      List.filter (fun m -> m.url <> "nx/nx/Nx/index.html") modules
    in

    let rules = List.filter_map (generate_dune_rule library) other_modules in
    let output = String.concat "\n\n" rules in

    let oc = open_out output_path in
    output_string oc output;
    close_out oc
