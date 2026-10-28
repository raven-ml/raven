(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A value over a file's pages, copied onto a GPU whose memory the host does not
   address after a truncation cut its pages off: the copy raises Sys_error
   naming the file. A child copies, since a host read of such a page may end
   the process. It skips without such a GPU. *)

open Windtrap

let exe = Sys.executable_name

(* Saves a value of 64 KiB to [path], loads it over the file's pages, truncates
   the file to 4 bytes, places the value onto the first GPU and prints how the
   placement ended, or why it skipped. *)
let truncated_child path =
  match Rig_gpus.first () with
  | Error why -> print_string ("skip: no GPU: " ^ why)
  | Ok d when Rig.reaches Rig.host d ->
      print_string "skip: the host addresses the GPU's memory"
  | Ok d -> (
      let module G = (val Nx.devices [ d ]) in
      let n = 1 lsl 14 in
      Nx_io.save_npy path (Nx.create Nx.float32 [| n |] (Array.make n 1.));
      let x = Nx.unpack Nx.float32 (Nx_io.load_npy path) in
      let fd = Unix.openfile path [ Unix.O_WRONLY ] 0 in
      Unix.ftruncate fd 4;
      Unix.close fd;
      match Nx.place G.on x with
      | _ -> print_string "placed"
      | exception Sys_error m when String.starts_with ~prefix:path m ->
          print_string "Sys_error naming the file"
      | exception e -> print_string (Printexc.to_string e))

let test_truncated () =
  let path = Filename.temp_file "nx_io_gpu" ".npy" in
  Fun.protect ~finally:(fun () -> Sys.remove path) @@ fun () ->
  let ic = Unix.open_process_args_in exe [| exe; "truncated"; path |] in
  let out = String.trim (In_channel.input_all ic) in
  match Unix.close_process_in ic with
  | WEXITED 0 when String.starts_with ~prefix:"skip: " out ->
      skip ~reason:(String.sub out 6 (String.length out - 6)) ()
  | WEXITED 0 -> equal string "Sys_error naming the file" out
  | WEXITED n -> failf "the child exited %d: %s" n out
  | WSIGNALED n | WSTOPPED n -> failf "the child ended by signal %d: %s" n out

let () =
  match Array.to_list Sys.argv with
  | [ _; "truncated"; path ] -> truncated_child path
  | _ ->
      exit
        (run "nx.io on a GPU"
           [
             group "truncation"
               [
                 test
                   "a copy of a value whose file a truncation cut short \
                    raises Sys_error naming the file"
                   test_truncated;
               ];
           ])
