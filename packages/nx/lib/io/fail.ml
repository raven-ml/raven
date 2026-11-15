(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let failf fmt = Printf.ksprintf failwith fmt

let pp_shape s =
  "[" ^ String.concat "; " (Array.to_list (Array.map string_of_int s)) ^ "]"

(* [file path f] is [f ()], whose failures to read or write [path] or to parse
   its contents raise [Failure] starting with [path]. Code under it raises
   [Failure] without the path, and rig.disk's [Sys_error] starts with it. *)
let file path f =
  try f () with
  | Failure m -> failwith (path ^ ": " ^ m)
  | Sys_error m when String.starts_with ~prefix:path m -> failwith m
  | Sys_error m -> failwith (path ^ ": " ^ m)
  | Unix.Unix_error (e, _, _) -> failwith (path ^ ": " ^ Unix.error_message e)
