(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* comgr's library. *)

let library () =
  let rocm = Setting.value Setting.rocm_path in
  C.findlib "comgr" [ rocm ^ "/lib/libamd_comgr.so"; "amd_comgr" ]

(* comgr serialises the compiles of a process on a mutex of its own, so
   compiles run in processes of their own: workers, a C program embedded in the
   library ([compiler_amd_worker.c]), each serving compiles in turn. *)
let worker = Compiler_worker.v ~name:"comgr" Comgr_worker.program

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

(* comgr runs Clang's driver, which in a release build has the compiler proper
   skip LLVM's verifier. From comgr 2.8 (ROCm 6.2, LLVM 18), the driver takes
   [verified], which turns it back on in both steps, so that IR an optimization
   makes invalid stops the compile; earlier drivers reject the option. *)
let verified_from = ("2", "8")
let verified = "-fverify-intermediate-code"

(* The worker's request: the ISA, the assemble flag, the counts and options of
   the compile and of code generation, the version from which both are given
   [verified], its count and itself, NUL-terminated, then the source. *)
let request src ~arch ~asm =
  let options s =
    let o = String.split_on_char ' ' s in
    string_of_int (List.length o) :: o
  in
  let fields =
    [ "amdgcn-amd-amdhsa--" ^ arch; (if asm then "1" else "0") ]
    @ options (compile_options arch)
    @ options link_options
    @ [ fst verified_from; snd verified_from ]
    @ options verified
  in
  String.concat "" (List.map (fun f -> f ^ "\000") fields) ^ src

let compile_hip comgr src ~arch ~asm =
  Compiler_worker.call worker comgr (request src ~arch ~asm)

(* HIP *)

let hip arch =
  let comgr =
    match library () with
    | Some comgr -> comgr
    | None -> failwith "comgr not available: try setting COMGR_PATH?"
  in
  let compile src =
    let asm = String.trim (List.hd (String.split_on_char '\n' src)) = ".text" in
    match compile_hip comgr src ~arch ~asm with
    | Ok lib -> lib
    | Error e -> raise (Renderer.Compiler.Compile_error e)
  in
  let table () =
    let identity =
      String.concat "\n"
        [
          C.identity (Some comgr);
          compile_options arch;
          link_options;
          fst verified_from ^ "." ^ snd verified_from ^ " " ^ verified;
        ]
    in
    Printf.sprintf "compile_hip_%s_%s" arch
      (Digest.to_hex (Digest.string identity))
  in
  Renderer.Compiler.v ~cachekey:table ~disassemble:Helpers.amdgpu_disassemble
    compile
