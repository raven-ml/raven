(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk

(* Host Detection *)

let uname flag =
  try
    let ic = Unix.open_process_in ("uname " ^ flag) in
    let value = input_line ic in
    let _ = Unix.close_process_in ic in
    String.trim value
  with _ -> ""

let cc =
  let var = Device.Context.string ~name:"CC" ~default:"clang" in
  fun () -> Device.Context.get var

let host_arch = uname "-m"
let is_windows = String.equal Sys.os_type "Win32"

(* Subprocess Helpers *)

let write_all_fd fd bytes =
  let len = Bytes.length bytes in
  let rec loop off =
    if off < len then
      let wrote = Unix.write fd bytes off (len - off) in
      loop (off + wrote)
  in
  loop 0

let read_pipes stdout_fd stderr_fd =
  let buf = Bytes.create 4096 in
  let stdout_buf = Buffer.create 4096 in
  let stderr_buf = Buffer.create 4096 in
  Fun.protect
    ~finally:(fun () ->
      (try Unix.close stdout_fd with Unix.Unix_error _ -> ());
      try Unix.close stderr_fd with Unix.Unix_error _ -> ())
    (fun () ->
      let rec loop fds =
        match fds with
        | [] -> ()
        | _ ->
            let ready, _, _ = Unix.select fds [] [] (-1.) in
            let fds' =
              List.fold_left
                (fun acc fd ->
                  if not (List.mem fd ready) then fd :: acc
                  else
                    match Unix.read fd buf 0 (Bytes.length buf) with
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

(* Key flags: -fno-math-errno ensures __builtin_sqrt becomes a single
   instruction, not a function call. -ffixed-x18 avoids ARM's platform-reserved
   register (macOS context switch / Windows TEB). *)
let compile ~lang src =
  let arch = if is_windows then "AMD64" else host_arch in
  let target = if is_windows then "x86_64" else arch in
  let arch_flag =
    match arch with
    | "x86_64" | "AMD64" -> "-march=native"
    | "riscv64" -> "-march=rv64g"
    | _ -> "-mcpu=native"
  in
  let extra_args =
    if String.equal target "arm64" then [ "-ffixed-x18" ] else []
  in
  let base_args =
    [
      Printf.sprintf "--target=%s-none-unknown-elf" target;
      arch_flag;
      "-O2";
      "-fPIC";
      "-ffreestanding";
      "-fno-math-errno";
      "-nostdlib";
      "-fno-ident";
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
      ((cc () :: "-c" :: "-x" :: lang :: base_args)
      @ extra_args @ [ "-"; "-o"; "-" ])
  in
  let pid = Unix.create_process (cc ()) argv stdin_r stdout_w stderr_w in
  Unix.close stdin_r;
  Unix.close stdout_w;
  Unix.close stderr_w;
  write_all_fd stdin_w (Bytes.of_string src);
  Unix.close stdin_w;
  let obj, err = read_pipes stdout_r stderr_r in
  let _, status = Unix.waitpid [] pid in
  match status with
  | Unix.WEXITED 0 -> Bytes.of_string obj
  | _ ->
      let label = Printf.sprintf "clang -x %s" lang in
      let msg =
        if String.equal err "" then label ^ " failed (no stderr output)"
        else Printf.sprintf "%s failed:\n%s" label err
      in
      raise (Device.Compiler.Compile_error msg)

let compile_clang src = compile ~lang:"c" src

(* Compiles LLVM IR to object code by invoking clang with -x ir.
   This avoids a library dependency on LLVM at the cost of per-compilation
   subprocess overhead. *)
let compile_llvmir src = compile ~lang:"ir" src
