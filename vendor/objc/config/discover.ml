open Configurator.V1

let get_ccomp_type c =
  match ocaml_config_var c "ccomp_type" with
  | Some "gcc" -> `GCC
  | Some "clang" -> `Clang
  | Some "msvc" -> `MSVC
  | Some other -> `Other_ccomp other
  | None -> `Other_ccomp "unknown"

let get_target_platform c =
  let system_name = ocaml_config_var_exn c "system" in
  match system_name with
  | "macosx" ->
      let target_triplet = ocaml_config_var_exn c "target" in
      let parts = String.split_on_char '-' target_triplet in
      let has_exact_part target_part =
        List.exists (fun p -> p = target_part) parts
      in
      if has_exact_part "macabi" then `Catalyst
      else if has_exact_part "ios" then `IOS
      else `MacOS
  | "linux" -> `Linux
  | s -> `Unknown_platform s

let get_target_arch c =
  match ocaml_config_var c "architecture" with
  | Some "x86_64" | Some "amd64" -> `Amd64
  | Some "aarch64" | Some "arm64" -> `Arm64
  | Some arch -> `Other_arch arch
  | None -> `Other_arch "unknown"

let () =
  main ~name:"objc_discover" (fun c ->
      let platform = get_target_platform c in
      let ccomp_type = get_ccomp_type c in
      let arch = get_target_arch c in

      let ccopts =
        let common_flags = [ "-Wall"; "-g" ] in
        let platform_defines =
          match platform with
          | `Linux -> [ "-DGNUSTEP=1" ]
          | `MacOS | `IOS | `Catalyst -> [ "-DOBJC_APPLE=1" ]
          | `Unknown_platform _ -> []
        in
        let ccompiler_specific_warnings =
          match ccomp_type with
          | `GCC -> [ "-Wincompatible-pointer-types" ]
          | `Clang ->
              [ "-Wno-incompatible-function-pointer-types"; "-fno-common" ]
          | `MSVC | `Other_ccomp _ -> []
        in
        let objc_compile_flags =
          match platform with
          | `MacOS | `IOS | `Catalyst -> [ "-fobjc-arc" ]
          | `Linux -> [ "-fobjc-exceptions" ]
          | `Unknown_platform _ -> []
        in
        let gnustep_includes =
          match platform with
          | `Linux ->
              [
                "-I/usr/GNUstep/System/Library/Headers/";
                "-I/usr/GNUstep/Local/Library/Headers/";
                "-I/usr/local/GNUstep/System/Library/Headers/";
                "-I/usr/local/GNUstep/Local/Library/Headers/";
              ]
          | _ -> []
        in
        List.concat
          [
            common_flags;
            platform_defines;
            ccompiler_specific_warnings;
            objc_compile_flags;
            gnustep_includes;
          ]
      in
      Flags.write_sexp "ccopts.sexp" ccopts;

      let cclibs =
        match platform with
        | `MacOS | `IOS | `Catalyst -> [ "-lobjc"; "-framework"; "Foundation" ]
        | `Linux ->
            [
              "-L/usr/GNUstep/System/Library/Libraries";
              "-L/usr/GNUstep/Local/Library/Libraries";
              "-L/usr/local/GNUstep/System/Library/Libraries";
              "-L/usr/local/GNUstep/Local/Library/Libraries";
              "-L/usr/local/lib";
              "-L/usr/lib";
              "-L/usr/lib/x86_64-linux-gnu";
              "-L/usr/lib/aarch64-linux-gnu";
              "-lgnustep-base";
              "-lobjc";
            ]
        | `Unknown_platform _ -> []
      in
      Flags.write_sexp "cclibs.sexp" cclibs;

      let platform_module_name = "platform" in
      let platform_type_string =
        "type t = MacOS | IOS | Catalyst | Linux | Unknown_platform of string"
      in
      let platform_current_value_string =
        match platform with
        | `MacOS -> "MacOS"
        | `IOS -> "IOS"
        | `Catalyst -> "Catalyst"
        | `Linux -> "Linux"
        | `Unknown_platform s -> Printf.sprintf "(Unknown_platform %S)" s
      in
      Flags.write_lines
        (platform_module_name ^ ".ml")
        [
          platform_type_string;
          Printf.sprintf "let current = %s" platform_current_value_string;
          "let is_apple = match current with MacOS | IOS | Catalyst -> true | \
           _ -> false";
          "let is_gnustep = match current with Linux -> true | _ -> false";
        ];
      Flags.write_lines
        (platform_module_name ^ ".mli")
        [
          platform_type_string;
          "val current : t";
          "val is_apple : bool";
          "val is_gnustep : bool";
        ];

      let ccomp_module_name = "ccomp" in
      let ccomp_type_string =
        "type t = GCC | Clang | MSVC | Other_ccomp of string"
      in
      let ccomp_current_value_string =
        match ccomp_type with
        | `GCC -> "GCC"
        | `Clang -> "Clang"
        | `MSVC -> "MSVC"
        | `Other_ccomp s -> Printf.sprintf "(Other_ccomp %S)" s
      in
      Flags.write_lines
        (ccomp_module_name ^ ".ml")
        [
          ccomp_type_string;
          Printf.sprintf "let current = %s" ccomp_current_value_string;
        ];
      Flags.write_lines
        (ccomp_module_name ^ ".mli")
        [ ccomp_type_string; "val current : t" ];

      let arch_module_name = "arch" in
      let arch_type_string = "type t = Amd64 | Arm64 | Other_arch of string" in
      let arch_current_value_string =
        match arch with
        | `Amd64 -> "Amd64"
        | `Arm64 -> "Arm64"
        | `Other_arch s -> Printf.sprintf "(Other_arch %S)" s
      in
      Flags.write_lines (arch_module_name ^ ".ml")
        [
          arch_type_string;
          Printf.sprintf "let current = %s" arch_current_value_string;
        ];
      Flags.write_lines
        (arch_module_name ^ ".mli")
        [ arch_type_string; "val current : t" ])
