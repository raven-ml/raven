module C = Configurator.V1

let () =
  C.main ~name:"metrics-config" (fun c ->
      let system = C.ocaml_config_var_exn c "system" in
      let c_flags, c_library_flags =
        if String.equal system "macosx" then
          (* macOS: link against IOKit and CoreFoundation frameworks *)
          ([], [ "-framework"; "IOKit"; "-framework"; "CoreFoundation" ])
        else
          (* Linux: no extra flags needed *)
          ([], [])
      in
      C.Flags.write_sexp "c_flags.sexp" c_flags;
      C.Flags.write_sexp "c_library_flags.sexp" c_library_flags)
