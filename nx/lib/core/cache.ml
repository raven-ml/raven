(* cache.ml *)

let get_cache_base_dir ?(getenv = Sys.getenv_opt) () =
  match getenv "RAVEN_CACHE_ROOT" with
  | Some dir when dir <> "" -> dir
  | _ ->
      let xdg = Xdg.create ~env:getenv () in
      Filename.concat (Xdg.cache_dir xdg) "raven"

let get_cache_dir ?(getenv = Sys.getenv_opt) ~scope name =
  let base = get_cache_base_dir ~getenv () in
  let path = List.fold_left Filename.concat base [ scope; name ] in
  let sep = Filename.dir_sep.[0] in
  if path <> "" && path.[String.length path - 1] = sep then path
  else path ^ Filename.dir_sep

