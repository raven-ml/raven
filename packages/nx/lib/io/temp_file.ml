(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let remove_if_exists path = try Sys.remove path with Sys_error _ -> ()

let sibling path =
  Filename.temp_file ~temp_dir:(Filename.dirname path)
    (Filename.basename path ^ ".")
    ".tmp"

let mode = 0o640

let replace temp path =
  match
    Unix.chmod temp mode;
    Unix.rename temp path
  with
  | () -> ()
  | exception exn ->
      remove_if_exists temp;
      raise exn
