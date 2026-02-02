(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let base_dir () =
  match Sys.getenv_opt "RAVEN_RUNS_DIR" with
  | Some dir -> dir
  | None ->
      let cache =
        match Sys.getenv_opt "XDG_CACHE_HOME" with
        | Some d -> d
        | None -> Filename.concat (Sys.getenv "HOME") ".cache"
      in
      Filename.concat (Filename.concat cache "raven") "runs"
