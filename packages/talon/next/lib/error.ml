(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = { file : string option; bytes : (int * int) option; msg : string }

let v ?file ?bytes msg =
  (match bytes with
  | Some (first, last) when first < 0 || last < first ->
      invalid_arg
        (Printf.sprintf "Error.v: invalid byte range (%d, %d)" first last)
  | _ -> ());
  { file; bytes; msg }

let pp_bytes ppf = function
  | first, last when first = last -> Format.fprintf ppf "byte %d" first
  | first, last -> Format.fprintf ppf "bytes %d-%d" first last

let pp ppf e =
  Option.iter (Format.fprintf ppf "%s: ") e.file;
  Option.iter (Format.fprintf ppf "%a: " pp_bytes) e.bytes;
  Format.pp_print_string ppf e.msg

let get_ok = function
  | Ok v -> v
  | Error e -> failwith (Format.asprintf "%a" pp e)
