(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module C = Configurator.V1

let () =
  C.main ~name:"sdl2-config" (fun c ->
      let pkg_config =
        match C.Pkg_config.get c with
        | None -> C.die "pkg-config not found"
        | Some pc -> pc
      in
      let sdl2 =
        match C.Pkg_config.query pkg_config ~package:"sdl2" with
        | None -> C.die "SDL2 not found via pkg-config"
        | Some info -> info
      in
      let cflags = "-fPIC" :: sdl2.C.Pkg_config.cflags in
      let clibs = sdl2.C.Pkg_config.libs in
      C.Flags.write_sexp "cflags.sexp" cflags;
      C.Flags.write_sexp "clibs.sexp" clibs)
