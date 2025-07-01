(* Discover program for platform-specific compiler flags *)
module C = Configurator.V1

let () =
  C.main ~name:"nx_c" (fun c ->
      let system = C.ocaml_config_var_exn c "system" in
      let architecture = C.ocaml_config_var_exn c "architecture" in
      let word_size = C.ocaml_config_var_exn c "word_size" in

      (* Base optimization flags based on architecture *)
      let base_flags =
        match architecture with
        | "amd64" | "x86_64" -> [ "-O3"; "-march=native"; "-fPIC" ]
        | "arm64" | "aarch64" -> [ "-O3"; "-mcpu=native"; "-fPIC" ]
        | "power" | "ppc" | "ppc64" | "ppc64le" ->
            [ "-O3"; "-mcpu=native"; "-fPIC" ]
        | "riscv32" -> [ "-O3"; "-march=rv32gc"; "-fPIC" ]
        | "riscv64" -> [ "-O3"; "-march=rv64gc"; "-fPIC" ]
        | "riscv" ->
            (* For generic riscv, check word size to determine 32 vs 64 bit *)
            if word_size = "64" then [ "-O3"; "-march=rv64gc"; "-fPIC" ]
            else [ "-O3"; "-march=rv32gc"; "-fPIC" ]
        | "s390x" -> [ "-O3"; "-march=native"; "-fPIC" ]
        | _ -> [ "-O3"; "-fPIC" ]
      in

      (* Platform-specific adjustments *)
      let opt_flags =
        match system with
        | "macosx" ->
            (* macOS doesn't need -fPIC as it's implied *)
            List.filter (( <> ) "-fPIC") base_flags
        | _ -> base_flags
      in

      (* Discover OpenMP support *)
      let test_openmp c_flags =
        C.c_test c ~c_flags
          "#include <omp.h>\nint main() { return omp_get_num_threads(); }"
      in

      let c_flags, lib_flags =
        match system with
        | "macosx" ->
            (* Try macOS with Homebrew libomp *)
            if test_openmp [ "-Xpreprocessor"; "-fopenmp" ] then
              ( opt_flags @ [ "-Xpreprocessor"; "-fopenmp" ],
                if C.c_test c ~link_flags:[ "-lomp" ] "int main() { return 0; }"
                then [ "-lomp" ]
                else [] )
            else (opt_flags, [])
        | "linux" | "linux_elf" ->
            (* Linux with standard OpenMP *)
            if test_openmp [ "-fopenmp" ] then
              (opt_flags @ [ "-fopenmp" ], [ "-fopenmp" ])
            else (opt_flags, [])
        | "mingw" | "mingw64" | "cygwin" ->
            (* Windows with OpenMP *)
            if test_openmp [ "-fopenmp" ] then
              (opt_flags @ [ "-fopenmp" ], [ "-fopenmp" ])
            else (opt_flags, [])
        | _ ->
            (* Other systems: try standard OpenMP *)
            if test_openmp [ "-fopenmp" ] then
              (opt_flags @ [ "-fopenmp" ], [ "-fopenmp" ])
            else (opt_flags, [])
      in

      (* Write discovered flags *)
      C.Flags.write_sexp "c_flags.sexp" c_flags;
      C.Flags.write_sexp "c_library_flags.sexp" lib_flags)
