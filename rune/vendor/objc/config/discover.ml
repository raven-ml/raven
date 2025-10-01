open Configurator.V1
open Printf

module Platform = struct
  type t = MacOS | IOS | Catalyst | GNUStep | Unknown of string

  let of_context c =
    match Configurator.V1.ocaml_config_var_exn c "system" with
    | "macosx" ->
        let has part =
          String.split_on_char '-'
            (Configurator.V1.ocaml_config_var_exn c "target")
          |> List.mem part
        in
        if has "macabi" then Catalyst else if has "ios" then IOS else MacOS
    | "linux" -> GNUStep
    | s -> Unknown s

  let defines = function
    | GNUStep -> [ "-DGNUSTEP=1" ]
    | MacOS | IOS | Catalyst -> [ "-DOBJC_APPLE=1" ]
    | Unknown _ -> []

  let objc_flags = function
    | MacOS | IOS | Catalyst -> [ "-fobjc-arc" ]
    | GNUStep -> [ "-fobjc-exceptions" ]
    | Unknown _ -> []

  let include_paths = function
    | GNUStep ->
        [
          "/usr/GNUstep/System/Library/Headers/";
          "/usr/GNUstep/Local/Library/Headers/";
          "/usr/local/GNUstep/System/Library/Headers/";
          "/usr/local/GNUstep/Local/Library/Headers/";
        ]
        |> List.map (fun d -> "-I" ^ d)
    | _ -> []

  let link_libs = function
    | MacOS | IOS | Catalyst -> [ "-framework"; "Foundation" ]
    | GNUStep ->
        let ldirs =
          [
            "/usr/GNUstep/System/Library/Libraries";
            "/usr/GNUstep/Local/Library/Libraries";
            "/usr/local/GNUstep/System/Library/Libraries";
            "/usr/local/GNUstep/Local/Library/Libraries";
            "/usr/local/lib";
            "/usr/lib";
            "/usr/lib/x86_64-linux-gnu";
            "/usr/lib/aarch64-linux-gnu";
          ]
          |> List.map (fun d -> "-L" ^ d)
        in
        ldirs @ [ "-lgnustep-base"; "-lobjc" ]
    | Unknown _ -> []
end

module Ccomp = struct
  type t = GCC | Clang | MSVC | Other of string

  let of_context c =
    match Configurator.V1.ocaml_config_var c "ccomp_type" with
    | Some "gcc" -> GCC
    | Some "clang" -> Clang
    | Some "msvc" -> MSVC
    | Some other -> Other other
    | None -> Other "unknown"

  let warn_flags = function
    | GCC -> [ "-Wincompatible-pointer-types" ]
    | Clang -> [ "-Wno-incompatible-function-pointer-types"; "-Wno-incompatible-pointer-types"; "-fno-common" ]
    | MSVC | Other _ -> []
end

module Arch = struct
  type t = Amd64 | Arm64 | Other of string

  let of_context c =
    match Configurator.V1.ocaml_config_var c "architecture" with
    | Some ("x86_64" | "amd64") -> Amd64
    | Some ("aarch64" | "arm64") -> Arm64
    | Some other -> Other other
    | None -> Other "unknown"
end

let write_ocaml_module ~name ~type_def ~current_value =
  Flags.write_lines (name ^ ".ml")
    [ type_def; sprintf "let current = %s" current_value ];
  Flags.write_lines (name ^ ".mli") [ type_def; "val current : t" ]

let () =
  main ~name:"objc_discover" (fun c ->
      let platform = Platform.of_context c in
      let ccomp = Ccomp.of_context c in
      let arch = Arch.of_context c in

      let ccopts =
        [ "-Wall"; "-g" ] @ Platform.defines platform @ Ccomp.warn_flags ccomp
        @ Platform.objc_flags platform
        @ Platform.include_paths platform
      in
      Flags.write_sexp "ccopts.sexp" ccopts;

      let cclibs = Platform.link_libs platform in
      Flags.write_sexp "cclibs.sexp" cclibs;

      write_ocaml_module ~name:"platform"
        ~type_def:
          "type t = MacOS | IOS | Catalyst | GNUStep | Unknown of string"
        ~current_value:
          (match platform with
          | Platform.MacOS -> "MacOS"
          | IOS -> "IOS"
          | Catalyst -> "Catalyst"
          | GNUStep -> "GNUStep"
          | Unknown s -> sprintf "(Unknown %S)" s);

      write_ocaml_module ~name:"ccomp"
        ~type_def:"type t = GCC | Clang | MSVC | Other of string"
        ~current_value:
          (match ccomp with
          | Ccomp.GCC -> "GCC"
          | Clang -> "Clang"
          | MSVC -> "MSVC"
          | Other s -> sprintf "(Other %S)" s);

      write_ocaml_module ~name:"arch"
        ~type_def:"type t = Amd64 | Arm64 | Other of string"
        ~current_value:
          (match arch with
          | Arch.Amd64 -> "Amd64"
          | Arm64 -> "Arm64"
          | Other s -> sprintf "(Other %S)" s))
