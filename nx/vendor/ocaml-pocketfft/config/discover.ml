module C = Configurator.V1

let () =
  C.main ~name:"pocketfft" (fun c ->
      let architecture = C.ocaml_config_var_exn c "architecture" in
      let word_size = C.ocaml_config_var_exn c "word_size" in

      let arch_flags =
        match architecture with
        | "amd64" | "x86_64" -> [ "-march=native"; "-mtune=native" ]
        | "arm64" | "aarch64" -> [ "-mcpu=native" ]
        | "power" | "ppc" | "ppc64" | "ppc64le" -> [ "-mcpu=native" ]
        | "riscv32" -> [ "-march=rv32gc" ]
        | "riscv64" -> [ "-march=rv64gc" ]
        | "riscv" ->
            if word_size = "64" then [ "-march=rv64gc" ]
            else [ "-march=rv32gc" ]
        | "s390x" -> [ "-march=native" ]
        | _ -> []
      in

      let cxx_flags =
        [ "-O3" ]
        @ arch_flags
        @ [
            "-flto";
            "-ffast-math";
            "-DNDEBUG";
            "-funroll-loops";
            "-fomit-frame-pointer";
            "-finline-functions";
            "-fno-rtti";
            "-std=c++17";
            "-I.";
            "-DPOCKETFFT_NO_MULTITHREADING=0";
            "-DPOCKETFFT_CACHE_SIZE=32768";
          ]
      in

      C.Flags.write_sexp "cxx_flags.sexp" cxx_flags)
