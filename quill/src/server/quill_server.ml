let is_safe_path base_dir requested_path =
  let resolved_base = Unix.realpath base_dir in
  let resolved_requested = Unix.realpath requested_path in
  String.starts_with ~prefix:resolved_base resolved_requested

module Handler = struct
  let not_found _req = Dream.respond ~status:`Not_Found "Not Found"

  (* Release Mode Static Serving *)

  (* Loader for crunched editor assets (Release Mode) *)
  let editor_loader_crunched _ path req =
    let asset_opt = Asset_editor.read path in
    match asset_opt with
    | None -> not_found req
    | Some asset ->
        Dream.respond
          ~headers:
            [
              (* ("Cache-Control", "public, max-age=3600") *)
              ("Cache-Control", "no-store, max-age=0");
            ]
          asset

  (* Loader for crunched general assets (Release Mode) *)
  let asset_loader_crunched _ path req =
    let asset_opt = Asset.read path in
    match asset_opt with
    | None -> not_found req
    | Some asset ->
        Dream.respond
          ~headers:
            [
              (* ("Cache-Control", "public, max-age=3600") *)
              ("Cache-Control", "no-store, max-age=0");
            ]
          asset

  (* Dev Mode Static Serving *)

  let project_root =
    let binary_path = Filename.dirname Sys.executable_name in
    Dream.log "Project root: %s" binary_path;
    binary_path

  let editor_source_dir = "_build/default/quill/src/editor/"
  let asset_source_dir = "_build/default/quill/src/server/asset"
  let serve_editor_static = Dream.static editor_source_dir
  let serve_asset_static = Dream.static asset_source_dir

  (* Handlers *)

  let serve_document_content base_dir filename_md req =
    let full_path = Filename.concat base_dir filename_md in
    if
      Sys.file_exists full_path && String.starts_with ~prefix:base_dir full_path
    then
      if String.equal (Filename.extension filename_md) ".md" then
        Dream.from_filesystem base_dir filename_md req
      else not_found req
    else not_found req

  let serve_document_editor _req = Dream.html (Template_document.render ())

  let serve_directory_index base_dir _req =
    try
      let resolved_base =
        try Unix.realpath base_dir with Unix.Unix_error _ -> base_dir
      in
      let entries = Sys.readdir resolved_base in
      let md_files =
        Array.to_list entries
        |> List.filter (fun name ->
               let entry_path = Filename.concat resolved_base name in
               try
                 (not (Sys.is_directory entry_path))
                 && String.equal (Filename.extension name) ".md"
               with Sys_error _ -> false)
        |> List.sort String.compare
      in
      Dream.html (Template_index.render ~files:md_files)
    with Sys_error msg ->
      Dream.log "Error reading directory %s: %s" base_dir msg;
      Dream.respond ~status:`Internal_Server_Error "Error reading directory"

  let handle_root file_or_dir_path req =
    let path =
      try Unix.realpath file_or_dir_path
      with Unix.Unix_error _ -> file_or_dir_path
    in
    if Sys.file_exists path then
      if Sys.is_directory path then serve_directory_index path req
      else if String.equal (Filename.extension path) ".md" then
        serve_document_editor req
      else
        Dream.respond ~status:`Bad_Request
          "Root path must be a Markdown file or a directory"
    else not_found req

  let handle_named_document base_dir req =
    let filename_md = Dream.param req "filename_md" in
    let full_path = Filename.concat base_dir filename_md in
    if
      Sys.file_exists full_path
      && String.starts_with ~prefix:base_dir full_path
      && String.equal (Filename.extension filename_md) ".md"
    then serve_document_editor req
    else not_found req

  (* Helper to get a unique ID for the toplevel session. For now, just use the
     filename. Later, could incorporate session ID. *)
  let get_toplevel_id _req = "default"

  let execute_code req =
    let open Lwt.Syntax in
    let* body = Dream.body req in
    let toplevel_id = get_toplevel_id req in
    Dream.log "Executing code for toplevel ID: %s" toplevel_id;
    try
      let json = Yojson.Safe.from_string body in
      let request = Quill_api.code_execution_request_of_yojson json in
      match request with
      | Error err ->
          Dream.log "Failed to parse JSON: %s" err;
          Dream.respond ~status:`Bad_Request "Invalid JSON format"
      | Ok request ->
          let code = request.Quill_api.code in
          let result = Top.eval ~id:toplevel_id code in
          let response =
            Quill_api.
              {
                output = String.trim result.output;
                error = Option.map String.trim result.error;
                status = result.status;
              }
          in
          let response_json =
            Quill_api.code_execution_result_to_yojson response
          in
          Dream.json (Yojson.Safe.to_string response_json)
    with Yojson.Json_error msg ->
      Dream.log "Failed to parse JSON: %s" msg;
      Dream.respond ~status:`Bad_Request "Invalid JSON format"
end

let create_router file_or_dir_path =
  let base_dir =
    let path =
      try Unix.realpath file_or_dir_path
      with Unix.Unix_error _ -> file_or_dir_path
    in
    if Sys.file_exists path then
      if Sys.is_directory path then path else Filename.dirname path
    else
      raise
        (Invalid_argument
           (Printf.sprintf "Path '%s' does not exist." file_or_dir_path))
  in
  let is_single_file_mode =
    Sys.file_exists file_or_dir_path && Sys.is_regular_file file_or_dir_path
  in

  let asset_routes =
    if Config.is_release_mode then
      [
        Dream.get "/asset/**"
          (Dream.static ~loader:Handler.asset_loader_crunched "");
        Dream.get "/editor/**"
          (Dream.static ~loader:Handler.editor_loader_crunched "");
      ]
    else
      [
        (* In dev mode, fonts come from CDN (via template), editor files served
           directly *)
        Dream.get "/editor/**" Handler.serve_editor_static;
        Dream.get "/asset/**" Handler.serve_asset_static;
      ]
  in

  Dream.router
    (asset_routes
    @ [
        Dream.post "/api/execute" Handler.execute_code;
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

  let is_valid_start_path =
    if Sys.is_directory path then true
    else if Sys.is_regular_file path then
      String.equal (Filename.extension path) ".md"
    else false
  in

  if not is_valid_start_path then (
    Printf.eprintf
      "Error: Start path '%s' must be a directory or a Markdown (.md) file.\n"
      path;
    exit 1);

  Dream.run ~interface:"localhost" ~port:8080
  @@ Dream.logger @@ Dream.memory_sessions @@ create_router path
