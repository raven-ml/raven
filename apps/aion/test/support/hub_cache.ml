(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let rec mkdir_p d =
  if not (Sys.file_exists d) then begin
    mkdir_p (Filename.dirname d);
    Sys.mkdir d 0o755
  end

let plant ?(repo = Aion.Hub.repo) ?(revision = Aion.Hub.revision) dir file a =
  let path = Kaun_hf.cache_path ~cache_dir:dir ~revision ~file repo in
  mkdir_p (Filename.dirname path);
  Nx_io.save_safetensors path a
