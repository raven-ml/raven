module C = Configurator.V1

let ( /^ ) = Filename.concat
let file_exists = Sys.file_exists

let xla_flags () =
  let config ~lib_dir =
    let cflags = [ "-isystem"; lib_dir /^ "include" ] in
    let libs =
      if file_exists (lib_dir /^ "libxla_extension.so") then
        [
          "-Wl,-rpath," ^ lib_dir;
          lib_dir /^ "libxla_extension.so" (* Use full path on macOS *);
          "-ldl" (* for dlopen *);
        ]
      else
        [
          "-Wl,-rpath," ^ lib_dir;
          "-L" ^ lib_dir;
          "-lxla_extension";
          "-ldl" (* for dlopen *);
        ]
    in
    { C.Pkg_config.cflags; libs }
  in

  (* First check environment variable *)
  match Sys.getenv_opt "XLA_EXTENSION_DIR" with
  | Some lib_dir -> config ~lib_dir
  | None -> (
      (* Check vendor directory relative to where we are *)
      let vendor_dir = "../../vendor" in
      if file_exists (vendor_dir /^ "libxla_extension.so") then
        let abs_vendor_dir =
          let cwd = Sys.getcwd () in
          if Filename.is_relative vendor_dir then Filename.concat cwd vendor_dir
          else vendor_dir
        in
        config ~lib_dir:abs_vendor_dir
      else
        (* Check OPAM installation *)
        match Sys.getenv_opt "OPAM_SWITCH_PREFIX" with
        | Some prefix ->
            let lib_dir = prefix /^ "lib" /^ "xla" in
            if file_exists lib_dir then config ~lib_dir
            else { C.Pkg_config.cflags = []; libs = [] }
        | None -> { C.Pkg_config.cflags = []; libs = [] })

let () =
  C.main ~name:"xla" (fun c ->
      let default : C.Pkg_config.package_conf = { libs = []; cflags = [] } in

      let conf =
        match C.Pkg_config.get c with
        | None -> default
        | Some pc -> (
            match C.Pkg_config.query pc ~package:"xla" with
            | None -> default
            | Some deps -> deps)
      in

      let xla_flags = xla_flags () in

      C.Flags.write_sexp "c_flags.sexp" (xla_flags.cflags @ conf.cflags);
      C.Flags.write_sexp "c_library_flags.sexp" (xla_flags.libs @ conf.libs))
