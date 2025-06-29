(* Discover program to find BLAS library *)
module C = Configurator.V1

let () =
  C.main ~name:"nx_cblas" (fun c ->
      (* Discover BLAS library *)
      let blas_libs_base =
        match C.ocaml_config_var c "system" with
        | Some "macosx" ->
            (* On macOS, use Accelerate framework *)
            [ "-framework"; "Accelerate" ]
        | _ ->
            (* On other systems, try to find BLAS *)
            let libs =
              if C.c_test c ~link_flags:[ "-lblas" ] "int main() { return 0; }"
              then [ "-lblas" ]
              else if
                C.c_test c ~link_flags:[ "-lopenblas" ]
                  "int main() { return 0; }"
              then [ "-lopenblas" ]
              else if
                C.c_test c ~link_flags:[ "-lmkl_rt" ] "int main() { return 0; }"
              then [ "-lmkl_rt" ]
              else (
                prerr_endline "Warning: No BLAS library found. Using -lblas.";
                [ "-lblas" ])
            in
            libs
      in

      (* Discover OpenMP support and platform-specific flags *)
      let c_flags, has_openmp =
        let base_flags =
          let opt_flags = 
            match C.ocaml_config_var c "architecture" with
            | Some "amd64" | Some "x86_64" | Some "i386" | Some "i686" ->
                [ "-O3"; "-march=native"; "-fPIC" ]
            | Some "power" | Some "ppc" | Some "ppc64" | Some "ppc64le" ->
                [ "-O3"; "-mcpu=native"; "-fPIC" ]
            | Some "arm64" | Some "aarch64" ->
                [ "-O3"; "-mcpu=native"; "-fPIC" ]
            | Some "riscv" | Some "riscv64" ->
                [ "-O3"; "-march=native"; "-fPIC" ]
            | Some "s390x" ->
                [ "-O3"; "-march=native"; "-fPIC" ]
            | _ ->
                (* For unknown architectures, use safe optimization flags *)
                [ "-O3"; "-fPIC" ]
          in
          match C.ocaml_config_var c "system" with
          | Some "macosx" ->
              (* Use new BLAS interface on macOS to avoid deprecation warnings *)
              (* macOS doesn't need -fPIC as it's implied *)
              let flags_without_fpic = List.filter ((<>) "-fPIC") opt_flags in
              "-DACCELERATE_NEW_LAPACK" :: flags_without_fpic
          | _ -> opt_flags
        in
        let test_openmp flags =
          C.c_test c ~c_flags:flags
            "#include <omp.h>\nint main() { return omp_get_num_threads(); }"
        in
        if test_openmp [ "-fopenmp" ] then ("-fopenmp" :: base_flags, true)
        else if test_openmp [ "-Xpreprocessor"; "-fopenmp" ] then
          (* macOS with Homebrew libomp *)
          ("-Xpreprocessor" :: "-fopenmp" :: base_flags, true)
        else
          (* No OpenMP support, just optimization flags *)
          (base_flags, false)
      in

      (* Add OpenMP library to link flags if OpenMP is available *)
      let blas_libs =
        if has_openmp then
          match C.ocaml_config_var c "system" with
          | Some "macosx" ->
              (* On macOS with Homebrew, we need to link libomp explicitly *)
              if C.c_test c ~link_flags:[ "-lomp" ] "int main() { return 0; }"
              then "-lomp" :: blas_libs_base
              else blas_libs_base
          | _ ->
              (* On Linux, link with gomp (GNU OpenMP) *)
              "-lgomp" :: blas_libs_base
        else blas_libs_base
      in

      C.Flags.write_sexp "c_library_flags.sexp" blas_libs;
      C.Flags.write_sexp "c_flags.sexp" c_flags)
