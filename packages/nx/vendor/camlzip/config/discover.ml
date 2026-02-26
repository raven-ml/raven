module C = Configurator.V1

let () =
C.main ~name:"zip" (fun c ->

let stale_gzip : C.Pkg_config.package_conf = {
 libs = [ "-lz" ];
 cflags = []
} in

let conf =
  match C.Pkg_config.get c with
  | None -> C.die "'pkg-config' missing"
  | Some pc ->
    match (C.Pkg_config.query pc ~package:"zlib") with
      | None -> stale_gzip
      | Some deps -> deps
  in

  (* Add -fPIC on Linux and BSD systems for position-independent code.
     This is required when building shared libraries on x86-64 Linux to avoid
     relocation errors like "relocation R_X86_64_32 against `.data' can not be 
     used when making a shared object" *)
  let cflags = 
    match C.ocaml_config_var c "system" with
    | Some "linux" | Some "freebsd" | Some "netbsd" | Some "openbsd" 
    | Some "dragonfly" | Some "gnu" -> "-fPIC" :: conf.cflags
    | _ -> conf.cflags
  in

  C.Flags.write_sexp "c_flags.sexp"         cflags;
  C.Flags.write_sexp "c_library_flags.sexp" conf.libs)