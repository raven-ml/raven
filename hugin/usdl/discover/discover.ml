module C = Configurator.V1

let () =
  C.main ~name:"sdl2-config" (fun c ->
      let pkg_config =
        match C.Pkg_config.get c with
        | None -> C.die "pkg-config tool not found. Please install it."
        | Some pc -> pc
      in
      let sdl2_info =
        match C.Pkg_config.query pkg_config ~package:"sdl2" with
        | None ->
            C.die
              "SDL2 library configuration not found using pkg-config.\n\
               Please install SDL2 development libraries (e.g., libsdl2-dev on \
               Debian/Ubuntu, sdl2-devel on Fedora, sdl2 via brew on macOS)."
        | Some info -> info
      in
      let cflags = "-fPIC" :: sdl2_info.C.Pkg_config.cflags in
      let clibs = sdl2_info.C.Pkg_config.libs in
      C.Flags.write_sexp "cflags.sexp" cflags;
      C.Flags.write_sexp "clibs.sexp" clibs)
