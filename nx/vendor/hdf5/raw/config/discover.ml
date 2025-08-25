module C = Configurator.V1

let test_hdf5_compilation c conf =
  (* Test if we can compile and link a simple HDF5 program *)
  let test_code =
    {|
#include <hdf5.h>
int main() {
  unsigned maj, min, rel;
  H5get_libversion(&maj, &min, &rel);
  return 0;
}
|}
  in
  C.c_test c test_code ~c_flags:conf.C.Pkg_config.cflags
    ~link_flags:conf.C.Pkg_config.libs

let hdf5_from_paths () =
  (* Try common locations for HDF5 *)
  let lib_paths =
    [
      "/opt/homebrew/lib";
      (* macOS ARM Homebrew *)
      "/usr/local/lib";
      (* macOS Intel Homebrew *)
      "/usr/lib/x86_64-linux-gnu/hdf5/serial";
      (* Ubuntu/Debian *)
      "/usr/lib64";
      (* RedHat/Fedora *)
      "/usr/lib";
      (* Generic Unix *)
    ]
  in
  let inc_paths =
    [
      "/opt/homebrew/include";
      (* macOS ARM Homebrew *)
      "/usr/local/include";
      (* macOS Intel Homebrew *)
      "/usr/include/hdf5/serial";
      (* Ubuntu/Debian *)
      "/usr/include";
      (* Generic Unix *)
    ]
  in

  let libs =
    let base_libs = [ "-lhdf5"; "-lhdf5_hl" ] in
    match List.find_opt Sys.file_exists lib_paths with
    | Some path -> base_libs @ [ Printf.sprintf "-L%s" path ]
    | None -> base_libs
  in

  let cflags =
    match List.find_opt Sys.file_exists inc_paths with
    | Some path -> [ Printf.sprintf "-I%s" path ]
    | None -> []
  in

  C.Pkg_config.{ cflags; libs }

let () =
  C.main ~name:"hdf5-raw" (fun c ->
      let conf_opt =
        (* First try pkg-config *)
        match C.Pkg_config.get c with
        | None -> None
        | Some pc ->
            (* Try HDF5 HL first (includes both libraries), then fallback to
               base packages *)
            let packages =
              [ "hdf5_hl"; "hdf5"; "hdf5-serial"; "hdf5-openmpi"; "hdf5-mpich" ]
            in
            let rec try_packages = function
              | [] -> None
              | pkg :: rest -> (
                  match C.Pkg_config.query pc ~package:pkg with
                  | None -> try_packages rest
                  | Some conf -> Some conf)
            in
            try_packages packages
      in

      (* If pkg-config failed, try hardcoded paths *)
      let conf_opt =
        match conf_opt with
        | Some conf -> Some conf
        | None ->
            let conf = hdf5_from_paths () in
            if test_hdf5_compilation c conf then Some conf else None
      in

      match conf_opt with
      | None ->
          (* HDF5 not found - write empty flags to make the library optional *)
          Printf.printf "HDF5 not found - library will not be available\n";
          C.Flags.write_sexp "c_flags.sexp" [];
          C.Flags.write_sexp "c_library_flags.sexp" []
      | Some conf ->
          (* Ensure we have both hdf5 and hdf5_hl libraries *)
          let libs =
            if List.mem "-lhdf5_hl" conf.libs then conf.libs
            else conf.libs @ [ "-lhdf5_hl" ]
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
          let conf = { conf with libs; cflags } in
          Printf.printf "HDF5 found and configured\n";
          C.Flags.write_sexp "c_flags.sexp" conf.cflags;
          C.Flags.write_sexp "c_library_flags.sexp" conf.libs)
