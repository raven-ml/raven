(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

external nvrtc_load : string -> (int * int, string) result
  = "caml_tolk_nvrtc_load"

external nvrtc_compile :
  string -> string array -> bool -> (string, string) result
  = "caml_tolk_nvrtc_compile"

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

(* The library is loaded once, by whichever domain first compiles. *)
let nvrtc_version =
  let lock = Mutex.create () and version = ref None in
  fun () ->
    Mutex.protect lock @@ fun () ->
    match !version with
    | Some v -> v
    | None ->
        let v =
          match library () with
          | None ->
              Error "failed to load library nvrtc: try setting NVRTC_PATH?"
          | Some path ->
              Result.map_error
                (fun e -> "failed to load library nvrtc: " ^ e)
                (nvrtc_load path)
        in
        version := Some v;
        v

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

(* The toolkit whose headers kernels include. *)
let cuda_path = Setting.string ~reach:Output "CUDA_PATH" ""

let nvrtc ?(ptx = true) ?(cache_key = "cuda") arch =
  let includes =
    match Setting.value cuda_path with
    | "" ->
        [ "-I/usr/local/cuda/include"; "-I/usr/include"; "-I/opt/cuda/include" ]
    | cuda_path -> [ "-I" ^ cuda_path ^ "/include" ]
  in
  let options = ("--gpu-architecture=" ^ arch) :: "--fmad=false" :: includes in
  let compile src =
    match nvrtc_version () with
    | Error e -> raise (Renderer.Compiler.Compile_error e)
    | Ok version -> (
        let options =
          options @ if version >= (12, 4) then [ "--minimal" ] else []
        in
        match nvrtc_compile src (Array.of_list options) ptx with
        | Ok lib -> lib
        | Error e -> raise (Renderer.Compiler.Compile_error e))
  in
  (* NVRTC's library, whose version picks the options it is given beyond
     [options], and whether it makes PTX, name a binary with its source. *)
  let table () =
    let identity =
      String.concat "\n"
        (C.identity (library ()) :: string_of_bool ptx :: options)
    in
    Printf.sprintf "compile_%s_%s_%s" cache_key arch
      (Digest.to_hex (Digest.string identity))
  in
  Renderer.Compiler.v ~cachekey:table
    ~disassemble:(cuda_disassemble ~ptx arch)
    compile
