(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

external comgr_load : string -> (unit, string) result = "caml_tolk_comgr_load"

external comgr_compile :
  string ->
  string ->
  bool ->
  string array ->
  string array ->
  (string, string) result = "caml_tolk_comgr_compile"

(* comgr's library. *)
let rocm_path =
  Helpers.Context_var.string ~reach:Process "ROCM_PATH" "/opt/rocm"

let library () =
  let rocm = Helpers.Context_var.value rocm_path in
  C.findlib "comgr" [ rocm ^ "/lib/libamd_comgr.so"; "amd_comgr" ]

(* The library is loaded once, by whichever domain first compiles. *)
let comgr =
  let lock = Mutex.create () and loaded = ref None in
  fun () ->
    Mutex.protect lock @@ fun () ->
    match !loaded with
    | Some l -> l
    | None ->
        let l =
          match library () with
          | None -> Error "comgr not available: try setting COMGR_PATH?"
          | Some path ->
              Result.map_error
                (fun e -> "comgr not available: " ^ e)
                (comgr_load path)
        in
        loaded := Some l;
        l

(* comgr takes options as a list, split at spaces. *)
let options s = Array.of_list (String.split_on_char ' ' s)

(* The options comgr compiles HIP with, and links it with. *)
let compile_options arch =
  String.concat " "
    [
      "-O3";
      "-ffp-contract=off";
      "-mcumode";
      "--hip-version=6.0.32830";
      "-DHIP_VERSION_MAJOR=6";
      "-DHIP_VERSION_MINOR=0";
      "-DHIP_VERSION_PATCH=32830";
      "-D__HIPCC_RTC__";
      "-std=c++14";
      "-nogpuinc";
      "-Wno-gnu-line-marker";
      "-Wno-missing-prototypes";
      "--offload-arch=" ^ arch;
      "-I/opt/rocm/include";
      "-Xclang -disable-llvm-passes";
      "-Xclang -aux-triple";
      "-Xclang x86_64-unknown-linux-gnu";
    ]

let link_options = "-O3 -mllvm -amdgpu-internalize-symbols"

let compile_hip src ~arch ~asm =
  comgr_compile src
    ("amdgcn-amd-amdhsa--" ^ arch)
    asm
    (options (compile_options arch))
    (options link_options)

(* HIP *)

let hip arch =
  let compile src =
    let asm = String.trim (List.hd (String.split_on_char '\n' src)) = ".text" in
    match Result.bind (comgr ()) (fun () -> compile_hip src ~arch ~asm) with
    | Ok lib -> lib
    | Error e -> raise (Renderer.Compiler.Compile_error e)
  in
  let table () =
    let identity =
      String.concat "\n"
        [ C.identity (library ()); compile_options arch; link_options ]
    in
    Printf.sprintf "compile_hip_%s_%s" arch
      (Digest.to_hex (Digest.string identity))
  in
  Renderer.Compiler.v ~cachekey:table ~disassemble:Helpers.amdgpu_disassemble
    compile
