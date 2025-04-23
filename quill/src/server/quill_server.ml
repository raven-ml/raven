let is_safe_path base_dir requested_path =
  let resolved_base = Unix.realpath base_dir in
  let resolved_requested = Unix.realpath requested_path in
  String.starts_with ~prefix:resolved_base resolved_requested

module Handler = struct
  let not_found _req = Dream.respond ~status:`Not_Found "Not Found"

  let editor_loader _ path req =
    let asset_opt = Asset_editor.read path in
    match asset_opt with
    | None -> not_found req
    | Some asset ->
        Dream.respond
          ~headers:[ ("Cache-Control", "no-store, max-age=0") ]
          asset

  let asset_loader _ path req =
    let asset_opt = Asset.read path in
    match asset_opt with
    | None -> not_found req
    | Some asset ->
        Dream.respond
          ~headers:[ ("Cache-Control", "no-store, max-age=0") ]
          asset

  let serve_document_content base_dir filename_md req =
    let full_path = Filename.concat base_dir filename_md in
    if Sys.file_exists full_path && is_safe_path base_dir full_path then
      if String.equal (Filename.extension filename_md) ".md" then
        Dream.from_filesystem base_dir filename_md req
      else not_found req
    else not_found req

  let serve_document_editor _req = Dream.html (Template_document.render ())

  let serve_directory_index base_dir _req =
    try
      let entries = Sys.readdir base_dir in
      let md_files =
        Array.to_list entries
        |> List.filter (fun name ->
               let entry_path = Filename.concat base_dir name in
               (not (Sys.is_directory entry_path))
               && String.equal (Filename.extension name) ".md"
               && is_safe_path base_dir entry_path)
        |> List.sort String.compare
      in
      Dream.html (Template_index.render ~files:md_files)
    with Sys_error _msg ->
      Dream.respond ~status:`Internal_Server_Error "Error reading directory"

  let handle_root file_or_dir_path req =
    if Sys.is_regular_file file_or_dir_path then
      if String.equal (Filename.extension file_or_dir_path) ".md" then
        serve_document_editor req
      else
        Dream.respond ~status:`Bad_Request "Root path must be a Markdown file"
    else if Sys.is_directory file_or_dir_path then
      serve_directory_index file_or_dir_path req
    else not_found req

  let handle_named_document base_dir req =
    let filename_md = Dream.param req "filename_md" in
    let full_path = Filename.concat base_dir filename_md in
    if
      Sys.file_exists full_path
      && is_safe_path base_dir full_path
      && String.equal (Filename.extension filename_md) ".md"
    then serve_document_editor req
    else not_found req

  let execute_code req =
    let open Lwt.Syntax in
    let* body = Dream.body req in
    match Yojson.Safe.from_string body with
    | `Assoc [ ("code", `String code) ] ->
        let id = "default" in
        (* Use a unique ID per session/document if needed *)
        Quill_top.initialize_toplevel id;
        let result = Quill_top.eval ~id code in
        let json =
          `Assoc
            [
              ("output", `String result.output);
              ( "error",
                match result.error with Some e -> `String e | None -> `Null );
              ( "status",
                `String
                  (match result.status with
                  | `Success -> "success"
                  | `Error -> "error") );
            ]
        in
        Dream.json (Yojson.Safe.to_string json)
    | _ -> Dream.respond ~status:`Bad_Request "Invalid JSON"
end

let create_router file_or_dir_path =
  let base_dir =
    if Sys.is_directory file_or_dir_path then file_or_dir_path
    else Filename.dirname file_or_dir_path
  in
  let is_single_file_mode = Sys.is_regular_file file_or_dir_path in

  Dream.router
    ([
       Dream.post "/api/execute" Handler.execute_code;
       Dream.get "/asset/**" (Dream.static ~loader:Handler.asset_loader "");
       Dream.get "/editor/**" (Dream.static ~loader:Handler.editor_loader "");
       Dream.get "/" (Handler.handle_root file_or_dir_path);
     ]
    @
    if is_single_file_mode then
      [
        Dream.get "/api/doc" (fun req ->
            Handler.serve_document_content base_dir
              (Filename.basename file_or_dir_path)
              req);
      ]
    else
      [
        Dream.get "/api/doc/:filename_md" (fun req ->
            let filename_md = Dream.param req "filename_md" in
            Handler.serve_document_content base_dir filename_md req);
        Dream.get "/:filename_md" (Handler.handle_named_document base_dir);
      ])

let start path =
  if not (Sys.file_exists path) then (
    Printf.eprintf "Error: Path '%s' does not exist.\n" path;
    exit 1);

  if
    Sys.is_regular_file path
    && not (String.equal (Filename.extension path) ".md")
  then (
    Printf.eprintf "Error: Input file '%s' must be a Markdown (.md) file.\n"
      path;
    exit 1);

  Dream.run ~interface:"localhost" ~port:8080
  @@ Dream.logger @@ create_router path
