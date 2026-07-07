(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk

let uname flag =
  try
    let ic = Unix.open_process_in ("uname " ^ flag) in
    let value = input_line ic in
    let _ = Unix.close_process_in ic in
    String.trim value
  with _ -> ""

let cc =
  let var = Helpers.Context_var.string ~key:"CC" ~default:"clang" in
  fun () -> Helpers.Context_var.get var

let is_windows = String.equal Sys.os_type "Win32"

(* Host Target *)

type arch = X86_64 | Arm64 | Riscv64

let windows_machine () =
  match Sys.getenv_opt "PROCESSOR_ARCHITECTURE" with
  | Some arch when String.trim arch <> "" -> arch
  | _ -> "amd64"

let host_machine () =
  let machine = if is_windows then windows_machine () else uname "-m" in
  String.lowercase_ascii machine

let arch_of_machine machine =
  match String.lowercase_ascii machine with
  | "x86_64" | "amd64" -> X86_64
  | "arm64" | "aarch64" -> Arm64
  | "riscv64" -> Riscv64
  | arch ->
      raise
        (Compiler.Compile_error
           (Printf.sprintf "unsupported arch: %S" arch))

let target = function
  | X86_64 -> "x86_64"
  | Arm64 -> "arm64"
  | Riscv64 -> "riscv64"

let arch_args = function
  | X86_64 -> [ "-march=native" ]
  | Arm64 -> [ "-ffixed-x18"; "-mcpu=native" ]
  | Riscv64 -> [ "-march=rv64g" ]

(* Subprocess Helpers *)

let close_noerr fd = try Unix.close fd with Unix.Unix_error _ -> ()

let rec waitpid pid =
  try Unix.waitpid [] pid
  with Unix.Unix_error (Unix.EINTR, _, _) -> waitpid pid

let write_all_fd fd bytes =
  let len = Bytes.length bytes in
  let rec loop off =
    if off < len then begin
      match Unix.write fd bytes off (len - off) with
      | 0 -> raise (Unix.Unix_error (Unix.EPIPE, "write", ""))
      | wrote -> loop (off + wrote)
      | exception Unix.Unix_error (Unix.EINTR, _, _) -> loop off
    end
  in
  loop 0

let read_pipes stdout_fd stderr_fd =
  let buf = Bytes.create 4096 in
  let stdout_buf = Buffer.create 4096 in
  let stderr_buf = Buffer.create 4096 in
  Fun.protect
    ~finally:(fun () ->
      close_noerr stdout_fd;
      close_noerr stderr_fd)
    (fun () ->
      let rec select fds =
        try Unix.select fds [] [] (-1.)
        with Unix.Unix_error (Unix.EINTR, _, _) -> select fds
      in
      let rec read fd =
        try Unix.read fd buf 0 (Bytes.length buf)
        with Unix.Unix_error (Unix.EINTR, _, _) -> read fd
      in
      let rec loop fds =
        match fds with
        | [] -> ()
        | _ ->
            let ready, _, _ = select fds in
            let fds' =
              List.fold_left
                (fun acc fd ->
                  if not (List.mem fd ready) then fd :: acc
                  else
                    match read fd with
                    | 0 -> acc
                    | n ->
                        let target =
                          if fd = stdout_fd then stdout_buf else stderr_buf
                        in
                        Buffer.add_string target (Bytes.sub_string buf 0 n);
                        fd :: acc)
                [] fds
            in
            loop (List.rev fds')
      in
      loop [ stdout_fd; stderr_fd ];
      (Buffer.contents stdout_buf, Buffer.contents stderr_buf))

(* Compilation *)

(* Mirrors tinygrad's ClangCompiler for CPU: compile C source to a relocatable
   ELF object for the normalized host architecture. *)
let compile_clang src =
  let compiler = cc () in
  let arch = arch_of_machine (host_machine ()) in
  let base_args =
    [
      "-O2";
      "-fPIC";
      "-ffreestanding";
      "-fno-math-errno";
      "-nostdlib";
      "-fno-ident";
      Printf.sprintf "--target=%s-none-unknown-elf" (target arch);
    ]
  in
  let stdin_r, stdin_w = Unix.pipe () in
  let stdout_r, stdout_w = Unix.pipe () in
  let stderr_r, stderr_w = Unix.pipe () in
  Unix.set_close_on_exec stdin_w;
  Unix.set_close_on_exec stdout_r;
  Unix.set_close_on_exec stderr_r;
  let argv =
    Array.of_list
      ((compiler :: "-c" :: "-x" :: "c" :: base_args)
      @ arch_args arch @ [ "-"; "-o"; "-" ])
  in
  let pid =
    try Unix.create_process compiler argv stdin_r stdout_w stderr_w
    with Unix.Unix_error (err, fn, arg) ->
      List.iter close_noerr
        [ stdin_r; stdin_w; stdout_r; stdout_w; stderr_r; stderr_w ];
      let msg =
        Printf.sprintf "%s: %s%s" fn (Unix.error_message err)
          (if String.equal arg "" then "" else " (" ^ arg ^ ")")
      in
      raise (Compiler.Compile_error msg)
  in
  close_noerr stdin_r;
  close_noerr stdout_w;
  close_noerr stderr_w;
  let obj, err, status =
    try
      write_all_fd stdin_w (Bytes.of_string src);
      close_noerr stdin_w;
      let obj, err = read_pipes stdout_r stderr_r in
      let _, status = waitpid pid in
      (obj, err, status)
    with exn ->
      close_noerr stdin_w;
      close_noerr stdout_r;
      close_noerr stderr_r;
      ignore (waitpid pid);
      raise exn
  in
  match status with
  | Unix.WEXITED 0 -> Bytes.of_string obj
  | _ ->
      let label = "clang -x c" in
      let msg =
        if String.equal err "" then label ^ " failed (no stderr output)"
        else Printf.sprintf "%s failed:\n%s" label err
      in
      raise (Compiler.Compile_error msg)
