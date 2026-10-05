(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

external worker_memfd : string -> int = "caml_tolk_comgr_worker_memfd"

(* comgr's library. *)
let rocm_path =
  Helpers.Context_var.string ~reach:Process "ROCM_PATH" "/opt/rocm"

let library () =
  let rocm = Helpers.Context_var.value rocm_path in
  C.findlib "comgr" [ rocm ^ "/lib/libamd_comgr.so"; "amd_comgr" ]

(* The worker *)

(* comgr serialises the compiles of a process on a mutex of its own, so each
   compile runs in a process of its own: the worker, a C program embedded in the
   library ([compiler_amd_worker.c]), written out once per process where it can
   be run. On Linux that is a memory file, which runs where the home directory
   is mounted noexec; elsewhere a file in [Helpers.cache_dir] named by the
   program's digest, written aside and renamed into place. *)

let rec mkdir_p dir =
  if not (Sys.file_exists dir) then begin
    mkdir_p (Filename.dirname dir);
    try Sys.mkdir dir 0o755 with Sys_error _ when Sys.file_exists dir -> ()
  end

let program_file program =
  let digest = Digest.to_hex (Digest.string program) in
  let path =
    Filename.concat Helpers.cache_dir
      ("comgr-worker-" ^ digest ^ if Sys.win32 then ".exe" else "")
  in
  let whole () =
    Sys.file_exists path && Digest.to_hex (Digest.file path) = digest
  in
  try
    if not (whole ()) then begin
      mkdir_p Helpers.cache_dir;
      let tmp, oc =
        Filename.open_temp_file ~mode:[ Open_binary ]
          ~temp_dir:Helpers.cache_dir "comgr-worker-" ".tmp"
      in
      Fun.protect
        ~finally:(fun () -> close_out_noerr oc)
        (fun () -> output_string oc program);
      Unix.chmod tmp 0o755;
      try Sys.rename tmp path with Sys_error _ when whole () -> Sys.remove tmp
    end;
    if not (whole ()) then failwith (path ^ ": differs from the program");
    path
  with Sys_error e | Unix.Unix_error (_, _, e) ->
    failwith (Printf.sprintf "comgr worker %s: %s" path e)

(* Written once, by whichever domain first compiles. A failure is raised to that
   compile and tried again by the next. *)
let worker =
  let lock = Mutex.create () and written = ref None in
  fun () ->
    Mutex.protect lock @@ fun () ->
    match !written with
    | Some path -> path
    | None ->
        let path =
          if Host_config.system = "linux" then
            "/proc/self/fd/" ^ string_of_int (worker_memfd Comgr_worker.program)
          else program_file Comgr_worker.program
        in
        written := Some path;
        path

(* [reply s] is the reply [s] holds if it is complete. *)
let reply s =
  match
    Scanf.sscanf_opt s "%s %u\n%n" (fun kind n start -> (kind, n, start))
  with
  | Some (kind, n, start) when start + n = String.length s -> (
      let data = String.sub s start n in
      match kind with
      | "ok" -> Some (Ok data)
      | "error" -> Some (Error data)
      | _ -> None)
  | _ -> None

let signal_name s =
  let names =
    Sys.
      [
        (sigsegv, "SIGSEGV");
        (sigbus, "SIGBUS");
        (sigabrt, "SIGABRT");
        (sigill, "SIGILL");
        (sigfpe, "SIGFPE");
        (sigkill, "SIGKILL");
        (sigterm, "SIGTERM");
        (sigint, "SIGINT");
      ]
  in
  match List.assoc_opt s names with
  | Some name -> name
  | None -> "signal " ^ string_of_int s

(* [run program args request] runs [program] with [args], then the files of
   [request] and of its reply, its output and errors going to a third file:
   files, so that neither process ever waits for the other to read. The reply
   decides, whatever the exit status says, which is lost when SIGCHLD is
   ignored. A program that ends without a whole reply is [Error] with why and
   its diagnostics. Raises [Failure] if the program cannot be started: that is
   the machine, not the request. *)
let run program args request =
  let temp_file suffix contents =
    let path = Filename.temp_file "tolk-comgr" suffix in
    Out_channel.with_open_bin path (fun oc -> output_string oc contents);
    path
  in
  let request_file = temp_file ".request" request
  and reply_file = temp_file ".reply" ""
  and diagnostics_file = temp_file ".log" "" in
  Fun.protect ~finally:(fun () ->
      List.iter Sys.remove [ request_file; reply_file; diagnostics_file ])
  @@ fun () ->
  let pid =
    let null = Unix.openfile Filename.null [ O_RDONLY; O_CLOEXEC ] 0 in
    let diagnostics =
      Unix.openfile diagnostics_file [ O_WRONLY; O_CLOEXEC ] 0
    in
    Fun.protect ~finally:(fun () -> List.iter Unix.close [ null; diagnostics ])
    @@ fun () ->
    let argv =
      Array.of_list ((program :: args) @ [ request_file; reply_file ])
    in
    try Unix.create_process program argv null diagnostics diagnostics
    with Unix.Unix_error (e, _, _) ->
      failwith
        (Printf.sprintf "comgr worker %s: %s" program (Unix.error_message e))
  in
  let rec wait () =
    match Unix.waitpid [] pid with
    | _, status -> Some status
    | exception Unix.Unix_error (EINTR, _, _) -> wait ()
    | exception Unix.Unix_error (ECHILD, _, _) -> None
  in
  let status = wait () in
  let read path = In_channel.with_open_bin path In_channel.input_all in
  let diagnostics = read diagnostics_file in
  match reply (read reply_file) with
  | Some r ->
      prerr_string diagnostics;
      r
  | None ->
      let why =
        match status with
        | Some (WSIGNALED s) -> "was killed by " ^ signal_name s
        | Some (WEXITED n) -> "exited with " ^ string_of_int n
        | Some (WSTOPPED s) -> "was stopped by " ^ signal_name s
        | None -> "ended"
      in
      Error
        (Printf.sprintf "comgr worker %s without a reply\n%s" why diagnostics)

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

(* The worker's request: the ISA, the assemble flag, the counts and options of
   the compile and of code generation, NUL-terminated, then the source. *)
let request src ~arch ~asm =
  let options s =
    let o = String.split_on_char ' ' s in
    string_of_int (List.length o) :: o
  in
  let fields =
    [ "amdgcn-amd-amdhsa--" ^ arch; (if asm then "1" else "0") ]
    @ options (compile_options arch)
    @ options link_options
  in
  String.concat "" (List.map (fun f -> f ^ "\000") fields) ^ src

let compile_hip comgr src ~arch ~asm =
  run (worker ())
    [ comgr; string_of_int (Unix.getpid ()) ]
    (request src ~arch ~asm)

(* HIP *)

let hip arch =
  let compile src =
    let asm = String.trim (List.hd (String.split_on_char '\n' src)) = ".text" in
    let result =
      match library () with
      | None -> Error "comgr not available: try setting COMGR_PATH?"
      | Some comgr -> compile_hip comgr src ~arch ~asm
    in
    match result with
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
