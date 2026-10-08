(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let colons s = String.fold_left (fun n c -> if c = ':' then n + 1 else n) 0 s

(* ["[h]rest"] as [Some (h, rest)]. *)
let bracketed s =
  if s = "" || s.[0] <> '[' then None
  else
    match String.index_opt s ']' with
    | None -> None
    | Some i ->
        Some
          ( String.sub s 1 (i - 1),
            String.sub s (i + 1) (String.length s - i - 1) )

let machine s =
  let no_port () = Error (strf "'%s': a machine takes no port" s) in
  match bracketed s with
  | Some (h, "") when h <> "" && colons h > 0 -> Ok h
  | Some (h, rest) when h <> "" && rest <> "" && rest.[0] = ':' -> no_port ()
  | Some _ -> Error (strf "'%s' is no machine" s)
  | None when s.[0] = '[' -> Error (strf "'%s' is no machine" s)
  | None -> (
      match colons s with
      | 0 -> Ok s
      | 1 -> no_port ()
      | _ ->
          Error (strf "'%s': write an IPv6 address in brackets, as [ADDRESS]" s)
      )

let port s p =
  let digits = p <> "" && String.length p <= 5 in
  let digits = digits && String.for_all (fun c -> c >= '0' && c <= '9') p in
  match if digits then int_of_string_opt p else None with
  | Some n when n <= 65535 -> Ok n
  | _ -> Error (strf "the port of '%s' is no number from 0 to 65535" s)

let host_port s =
  let no () = Error (strf "'%s' is no HOST:PORT" s) in
  match bracketed s with
  | Some (h, rest) when h <> "" && rest <> "" && rest.[0] = ':' ->
      Result.map
        (fun p -> (h, p))
        (port s (String.sub rest 1 (String.length rest - 1)))
  | Some _ -> no ()
  | None when s <> "" && s.[0] = '[' -> no ()
  | None -> (
      match String.index_opt s ':' with
      | None -> no ()
      | Some _ when colons s > 1 ->
          Error
            (strf "'%s': write an IPv6 address in brackets, as [ADDRESS]:PORT" s)
      | Some 0 -> no ()
      | Some i ->
          let h = String.sub s 0 i in
          Result.map
            (fun p -> (h, p))
            (port s (String.sub s (i + 1) (String.length s - i - 1))))

let with_port host port =
  if String.contains host ':' then strf "[%s]:%d" host port
  else strf "%s:%d" host port
