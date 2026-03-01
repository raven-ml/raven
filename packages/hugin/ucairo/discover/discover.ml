(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module C = Configurator.V1
module P = C.Pkg_config

let default_cairo c =
  let sys = C.ocaml_config_var_exn c "system" in
  if sys = "msvc" || sys = "win64" then
    {
      P.cflags = [ "-I"; "C:\\gtk\\include\\cairo" ];
      libs = [ "/LC:\\gtk\\lib"; "cairo.lib" ];
    }
  else { P.cflags = [ "-I/usr/include/cairo" ]; libs = [ "-lcairo" ] }

let () =
  C.main ~name:"cairo-config" (fun c ->
      let p =
        match P.get c with
        | Some p -> (
            match P.query p ~package:"cairo" with
            | Some p -> p
            | None -> default_cairo c)
        | None -> default_cairo c
      in
      let cflags =
        match Sys.getenv "CAIRO_CFLAGS" with
        | exception Not_found -> "-fPIC" :: p.P.cflags
        | alt -> C.Flags.extract_blank_separated_words alt
      in
      let libs =
        match Sys.getenv "CAIRO_LIBS" with
        | exception Not_found -> p.P.libs
        | alt -> C.Flags.extract_blank_separated_words alt
      in
      C.Flags.write_sexp "cflags.sexp" cflags;
      C.Flags.write_sexp "clibs.sexp" libs)
