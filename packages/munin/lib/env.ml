let root () =
  match Sys.getenv_opt "RAVEN_TRACKING_DIR" with
  | Some dir -> dir
  | None ->
      let data_home =
        match Sys.getenv_opt "XDG_DATA_HOME" with
        | Some dir -> dir
        | None ->
            Filename.concat
              (Filename.concat (Sys.getenv "HOME") ".local")
              "share"
      in
      Filename.concat (Filename.concat data_home "raven") "munin"
