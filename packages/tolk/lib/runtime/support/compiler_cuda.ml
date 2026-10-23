(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

let cuda_targets =
  let machine = List.hd (String.split_on_char '-' C.multiarch) in
  List.concat_map
    (fun pre ->
      List.map
        (fun tgt -> Printf.sprintf "/%s/cuda/targets/%s/lib" pre tgt)
        [ machine ^ "-linux"; "sbsa-linux" ])
    [ "opt"; "usr/local" ]

(* NVRTC's library. *)
let library () = C.findlib ~extra_paths:cuda_targets "nvrtc" [ "nvrtc" ]

(* NVRTC's assembler crashes on some valid sources, so compiles run in
   processes of their own: workers, a C program embedded in the library
   ([compiler_cuda_worker.c]), each serving compiles in turn. *)
let worker = Compiler_worker.v ~name:"nvrtc" Nvrtc_worker.program
let worker_digest = Digest.to_hex (Digest.string Nvrtc_worker.program)

(* The worker's request: whether to make PTX, the count then the options, the
   version from which NVRTC is given the next options, the count then those,
   NUL-terminated, then the source. From version 12.4, NVRTC compiles faster
   without textures, surfaces and the device runtime API. *)
let request src ~ptx options =
  let list o = string_of_int (List.length o) :: o in
  let fields =
    ((if ptx then "1" else "0") :: list options)
    @ ("12" :: "4" :: list [ "--minimal" ])
  in
  String.concat "" (List.map (fun f -> f ^ "\000") fields) ^ src

let cuda_disassemble ~ptx arch lib =
  try
    let fn =
      Filename.concat
        (Filename.get_temp_dir_name ())
        ("tinycuda_" ^ Digest.to_hex (Digest.string lib))
    in
    let rec rstrip_nul s =
      if String.ends_with ~suffix:"\000" s then
        rstrip_nul (String.sub s 0 (String.length s - 1))
      else s
    in
    Out_channel.with_open_bin fn (fun oc ->
        output_string oc (if ptx then rstrip_nul lib else lib));
    if ptx then
      ignore
        (Helpers.system (Printf.sprintf "ptxas -arch=%s -o %s %s" arch fn fn));
    print_endline (Helpers.system ("nvdisasm " ^ fn))
  with Failure e | Sys_error e ->
    print_endline
      ("Failed to generate SASS " ^ e
     ^ " Make sure your PATH contains ptxas/nvdisasm binary of compatible \
        version.")

(* NVRTC *)

let nvrtc ?(ptx = true) ?(cache_key = "cuda") arch =
  let includes =
    match Setting.value Setting.cuda_path with
    | "" ->
        [ "-I/usr/local/cuda/include"; "-I/usr/include"; "-I/opt/cuda/include" ]
    | cuda_path -> [ "-I" ^ cuda_path ^ "/include" ]
  in
  let options = ("--gpu-architecture=" ^ arch) :: "--fmad=false" :: includes in
  let compile src =
    match library () with
    | None ->
        raise
          (Renderer.Compiler.Compile_error
             "failed to load library nvrtc: try setting NVRTC_PATH?")
    | Some nvrtc -> (
        match Compiler_worker.call worker nvrtc (request src ~ptx options) with
        | Ok lib -> lib
        | Error e -> raise (Renderer.Compiler.Compile_error e))
  in
  (* NVRTC's library, whose version picks the options it is given beyond
     [options], the worker that runs it, and whether it makes PTX, name a binary
     with its source. *)
  let table () =
    let identity =
      String.concat "\n"
        (C.identity (library ()) :: worker_digest :: string_of_bool ptx
       :: options)
    in
    Printf.sprintf "compile_%s_%s_%s" cache_key arch
      (Digest.to_hex (Digest.string identity))
  in
  Renderer.Compiler.v ~cachekey:table
    ~disassemble:(cuda_disassemble ~ptx arch)
    compile
