(* discover.ml *)
open Configurator.V1

let () =
  main ~name:"objc_discover" (fun c ->
      (* --- Determine Operating System --- *)
      let os_type =
        match ocaml_config_var c "system" with
        | Some "macosx" -> `MacOS
        | Some "linux" -> `Linux
        | Some s -> `Unknown_os s (* Capture the unknown system string *)
        | None -> `Unknown_os "undefined"
      in

      (* --- Determine C Compiler Type (more robustly) --- *)
      (* Configurator.V1.C_compiler.cc_type uses the configured C compiler
       to check for specific compiler families. *)
      let ccomp_type = C_compiler.cc_type c in

      (* --- Generate Compiler Flags --- *)
      let common_cflags = [ "-Wall" ] in
      (* Common flags for all compilers *)

      let os_specific_cflags =
        match os_type with
        | `Linux ->
            [
              (* GNUstep include paths for Linux *)
              "-I/usr/GNUstep/System/Library/Headers/";
              "-I/usr/GNUstep/Local/Library/Headers/";
              "-I/usr/local/GNUstep/System/Library/Headers/";
              "-I/usr/local/GNUstep/Local/Library/Headers/";
            ]
        | `MacOS | `Unknown_os _ -> [] (* Default for macOS or unknown *)
      in

      let compiler_specific_cflags =
        match ccomp_type with
        | `GCC -> [ "-Wincompatible-pointer-types" ]
        | `Clang -> [ "-Wno-error=incompatible-function-pointer-types" ]
        | `MSVC -> [] (* Add MSVC specific flags if needed *)
        | `Other _ -> [] (* For other unknown compilers *)
      in

      let final_cflags =
        List.concat
          [ common_cflags; os_specific_cflags; compiler_specific_cflags ]
      in
      Flags.write_sexp "compiler_flags.sexp" final_cflags;

      (* --- Generate Linker Flags --- *)
      let final_ldflags =
        match os_type with
        | `MacOS ->
            [ "-lobjc"; "-framework"; "Foundation" ]
            (* Note: -framework Foundation is two args *)
        | `Linux ->
            [
              (* GNUstep library paths and libraries for Linux *)
              "-L/usr/GNUstep/System/Library/Libraries";
              "-L/usr/GNUstep/Local/Library/Libraries";
              "-L/usr/local/GNUstep/System/Library/Libraries";
              "-L/usr/local/GNUstep/Local/Library/Libraries";
              "-L/usr/local/lib";
              "-L/usr/lib";
              "-L/usr/lib/x86_64-linux-gnu";
              "-L/usr/lib/aarch64-linux-gnu";
              "-lobjc";
              "-lgnustep-base";
            ]
        | `Unknown_os _ ->
            (* It's good to warn or error if flags are essential and OS is
               unknown *)
            prerr_endline
              "Warning: Unknown OS detected, linker flags may be incomplete.";
            []
      in
      Flags.write_sexp "linker_flags.sexp" final_ldflags;

      (* --- (Optional) Generate OCaml config modules if still needed --- *)
      (* If you still want OCaml modules like os_config.ml, you can generate them here.
       This is useful if your OCaml code needs to know the OS/compiler at runtime,
       beyond just setting build flags. *)
      let os_config_content =
        let os_to_string = function
          | `MacOS -> "MacOS"
          | `Linux -> "Linux"
          | `Unknown_os s -> Printf.sprintf "(Unknown_os %S)" s
        in
        Printf.sprintf
          "type t = MacOS | Linux | Unknown_os of string\nlet current = %s\n"
          (os_to_string os_type)
      in
      Flags.write_lines "os_config.ml" [ os_config_content ];

      let ccomp_config_content =
        let ccomp_to_string = function
          | `GCC -> "GCC"
          | `Clang -> "Clang"
          | `MSVC -> "MSVC"
          | `Other s -> Printf.sprintf "(Other %S)" s
        in
        Printf.sprintf
          "type t = GCC | Clang | MSVC | Other of string\nlet current = %s\n"
          (ccomp_to_string ccomp_type)
      in
      Flags.write_lines "ccomp_config.ml" [ ccomp_config_content ])
