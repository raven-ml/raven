(* Discover program to find BLAS library *)
module C = Configurator.V1

let () =
  C.main ~name:"nx_cblas" (fun c ->
      (* Discover BLAS library *)
      let blas_libs =
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
      let base_flags =
        match C.ocaml_config_var c "system" with
        | Some "macosx" ->
            (* Use new BLAS interface on macOS to avoid deprecation
               warnings *)
            [ "-DACCELERATE_NEW_LAPACK"; "-O3"; "-march=native" ]
        | _ -> [ "-O3"; "-march=native" ]
      in
      let test_openmp c_flags link_flags =
        C.c_test c ~c_flags:(c_flags @ base_flags) ~link_flags
          "#include <omp.h>\nint main() { return omp_get_num_threads(); }"
      in
      let omp_c_flags, omp_libs =
        if test_openmp ["-fopenmp"] ["-lgomp"] then
          (["-fopenmp"], ["-lgomp"])
        else if test_openmp ["-fopenmp"] [] then
          (["-fopenmp"], [])
        else if test_openmp ["-Xpreprocessor"; "-fopenmp"] ["-lomp"] then
          (* macOS with Homebrew libomp *)
          (["-Xpreprocessor"; "-fopenmp"], ["-lomp"])
        else
          (* No OpenMP support, just optimization flags *)
          ([], [])
      in

      let final_c_flags = omp_c_flags @ base_flags in
      let final_libs = blas_libs @ omp_libs in

      C.Flags.write_sexp "c_library_flags.sexp" final_libs;
      C.Flags.write_sexp "c_flags.sexp" final_c_flags)
