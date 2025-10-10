module C = Configurator.V1

let test_blas =
  {|
#include <cblas.h>

int main() {
  float a[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float b[6] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  float c[9] = {0.0f};
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 2, 1.0f, a, 2,
              b, 3, 0.0f, c, 3);
  return (int)c[0];
}
|}

let test_lapacke =
  {|
#include <lapacke.h>

int main() {
  double a[4] = {4.0, 1.0, 1.0, 3.0};
  lapack_int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', 2, a, 2);
  return (int)info;
}
|}

let openblas_default : C.Pkg_config.package_conf =
  let search =
    [ "/usr/local/opt/openblas/lib"; "/opt/OpenBLAS/lib/"; "/usr/lib" ]
    |> List.filter Sys.file_exists
  in
  let libs = List.map (fun path -> "-L" ^ path) search @ [ "-lopenblas" ] in
  let cflags =
    if Sys.file_exists "/usr/include/openblas" then
      [ "-I/usr/include/openblas" ]
    else []
  in
  { C.Pkg_config.cflags; libs }

let default_ldlibs = [ "-lm" ]

let string_contains ~needle haystack =
  let h_len = String.length haystack in
  let n_len = String.length needle in
  let rec aux i =
    if i + n_len > h_len then false
    else if String.sub haystack i n_len = needle then true
    else aux (i + 1)
  in
  if n_len = 0 then true else aux 0

let list_find_map f lst =
  let rec aux = function
    | [] -> None
    | x :: xs -> ( match f x with None -> aux xs | some -> some)
  in
  aux lst

let libomp_paths c =
  let env = Sys.getenv_opt "LIBOMP_PREFIX" in
  let brew_prefix =
    if C.Process.run_ok c "brew" [ "--prefix"; "libomp" ] then
      Some
        (C.Process.run_capture_exn c "brew" [ "--prefix"; "libomp" ]
        |> String.trim)
    else None
  in
  let candidates =
    List.filter_map
      (fun x -> x)
      [
        env;
        brew_prefix;
        Some "/opt/homebrew/opt/libomp";
        Some "/usr/local/opt/libomp";
      ]
  in
  list_find_map
    (fun prefix ->
      let include_dir = Filename.concat prefix "include" in
      let lib_dir = Filename.concat prefix "lib" in
      let header = Filename.concat include_dir "omp.h" in
      if Sys.file_exists header then
        Some ([ "-I" ^ include_dir ], [ "-L" ^ lib_dir ])
      else None)
    candidates

let compiler_is_clang c =
  let compiler =
    Sys.getenv_opt "CC"
    |> Option.value
         ~default:
           (match C.ocaml_config_var c "c_compiler" with
           | Some cc when String.trim cc <> "" -> cc
           | _ -> "cc")
  in
  if C.Process.run_ok c compiler [ "--version" ] then
    let version =
      C.Process.run_capture_exn c compiler [ "--version" ]
      |> String.lowercase_ascii
    in
    string_contains ~needle:"clang" version
  else false

let detect_openmp c system base_flags =
  let test c_flags link_flags =
    C.c_test c ~c_flags ~link_flags
      "#include <omp.h>\nint main(){return omp_get_num_threads();}"
  in
  match system with
  | "macosx" ->
      if compiler_is_clang c then
        let include_flags, lib_dir_flags =
          match libomp_paths c with Some paths -> paths | None -> ([], [])
        in
        let openmp_flags = include_flags @ [ "-Xpreprocessor"; "-fopenmp" ] in
        let openmp_libs = lib_dir_flags @ [ "-lomp" ] in
        if test openmp_flags openmp_libs then
          (base_flags @ openmp_flags, openmp_libs)
        else (base_flags, [])
      else if test [ "-fopenmp" ] [ "-fopenmp" ] then
        (base_flags @ [ "-fopenmp" ], [ "-fopenmp" ])
      else (base_flags, [])
  | "linux" | "linux_elf" | "mingw" | "mingw64" | "cygwin" ->
      if test [ "-fopenmp" ] [ "-fopenmp" ] then
        (base_flags @ [ "-fopenmp" ], [ "-fopenmp" ])
      else (base_flags, [])
  | _ ->
      if test [ "-fopenmp" ] [ "-fopenmp" ] then
        (base_flags @ [ "-fopenmp" ], [ "-fopenmp" ])
      else (base_flags, [])

let pkg_query c package =
  match C.Pkg_config.get c with
  | None -> None
  | Some pc -> C.Pkg_config.query pc ~package

let ensure_lapacke c c_flags libs pkg_query_fn =
  if C.c_test c test_lapacke ~c_flags ~link_flags:libs then (libs, c_flags)
  else
    let lapacke_conf =
      match pkg_query_fn "llapacke" with
      | Some conf -> conf
      | None -> (
          match pkg_query_fn "lapacke" with
          | Some conf -> conf
          | None -> { C.Pkg_config.cflags = []; libs = [ "-llapacke" ] })
    in
    let libs = lapacke_conf.libs @ libs in
    let c_flags = lapacke_conf.cflags @ c_flags in
    if C.c_test c test_lapacke ~c_flags ~link_flags:libs then (libs, c_flags)
    else (
      Printf.printf
        {|
Unable to link against LAPACKE even after adding (%s) to the link flags.
Verify that a LAPACKE implementation is installed and visible to the
build system (consider installing lapacke or setting PKG_CONFIG_PATH).
|}
        (String.concat " " lapacke_conf.libs);
      failwith "Unable to link against lapacke.")

let () =
  C.main ~name:"nx_c" (fun c ->
      let system = C.ocaml_config_var_exn c "system" in
      let architecture = C.ocaml_config_var_exn c "architecture" in
      let word_size = C.ocaml_config_var_exn c "word_size" in

      let base_flags =
        let opt_flags =
          match architecture with
          | "amd64" | "x86_64" -> [ "-O3"; "-march=native"; "-fPIC" ]
          | "arm64" | "aarch64" -> [ "-O3"; "-mcpu=native"; "-fPIC" ]
          | "power" | "ppc" | "ppc64" | "ppc64le" ->
              [ "-O3"; "-mcpu=native"; "-fPIC" ]
          | "riscv32" -> [ "-O3"; "-march=rv32gc"; "-fPIC" ]
          | "riscv64" -> [ "-O3"; "-march=rv64gc"; "-fPIC" ]
          | "riscv" ->
              if word_size = "64" then [ "-O3"; "-march=rv64gc"; "-fPIC" ]
              else [ "-O3"; "-march=rv32gc"; "-fPIC" ]
          | "s390x" -> [ "-O3"; "-march=native"; "-fPIC" ]
          | _ -> [ "-O3"; "-fPIC" ]
        in
        (* Suppress vectorization failure warnings from clang *)
        if compiler_is_clang c then opt_flags @ [ "-Wno-pass-failed" ]
        else opt_flags
      in

      let opt_flags =
        match system with
        | "macosx" -> List.filter (fun flag -> flag <> "-fPIC") base_flags
        | _ -> base_flags
      in

      let opt_flags, openmp_libs = detect_openmp c system opt_flags in

      let openblas_conf =
        match pkg_query c "openblas" with
        | Some conf -> conf
        | None -> openblas_default
      in

      let openblas_cflags =
        List.filter (fun flag -> flag <> "-fopenmp") openblas_conf.cflags
      in
      let openblas_libs =
        List.filter (fun flag -> flag <> "-fopenmp") openblas_conf.libs
      in
      let c_flags = opt_flags @ openblas_cflags in
      let libs = openblas_libs @ openmp_libs @ default_ldlibs in

      if not (C.c_test c test_blas ~c_flags ~link_flags:libs) then (
        Printf.printf
          {|
Unable to link against OpenBLAS: the current values for cflags and libs
are respectively (%s) and (%s).
Check that OpenBLAS is installed and, if necessary, extend PKG_CONFIG_PATH
with the directory containing openblas.pc.
|}
          (String.concat " " openblas_cflags)
          (String.concat " " openblas_libs);
        failwith "Unable to link against openblas.");

      let libs, c_flags = ensure_lapacke c c_flags libs (pkg_query c) in

      C.Flags.write_sexp "c_flags.sexp" c_flags;
      C.Flags.write_sexp "c_library_flags.sexp" libs)
