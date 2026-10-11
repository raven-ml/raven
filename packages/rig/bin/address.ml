(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

(* [s] as its host, its brackets taken off, and its port's text if it has one,
   or [None] if [s] is malformed. A bare IPv6 address splits at its first colon,
   so its port's text holds a colon. *)
let split s =
  let n = String.length s in
  let sub i j = String.sub s i (j - i) in
  match String.index_opt s ']' with
  | Some i when s.[0] = '[' && i > 1 ->
      if i = n - 1 then Some (sub 1 i, None)
      else if s.[i + 1] = ':' then Some (sub 1 i, Some (sub (i + 2) n))
      else None
  | Some _ -> None
  | None when s = "" || s.[0] = '[' -> None
  | None -> (
      match String.index_opt s ':' with
      | None -> Some (s, None)
      | Some i -> Some (sub 0 i, Some (sub (i + 1) n)))

let ipv6 s form =
  Error (strf "'%s': write an IPv6 address in brackets, as %s" s form)

let machine s =
  match split s with
  | Some (h, None) -> Ok h
  | Some (_, Some p) when String.contains p ':' -> ipv6 s "[ADDRESS]"
  | Some (h, Some _) when h <> "" ->
      Error (strf "'%s': a machine takes no port" s)
  | _ -> Error (strf "'%s' is no machine" s)

let port s p =
  let digits = p <> "" && String.length p <= 5 in
  let digits = digits && String.for_all (fun c -> c >= '0' && c <= '9') p in
  match if digits then int_of_string_opt p else None with
  | Some n when n <= 65535 -> Ok n
  | _ -> Error (strf "the port of '%s' is no number from 0 to 65535" s)

let host_port s =
  match split s with
  | Some (_, Some p) when String.contains p ':' -> ipv6 s "[ADDRESS]:PORT"
  | Some (h, Some p) when h <> "" -> Result.map (fun p -> (h, p)) (port s p)
  | _ -> Error (strf "'%s' is no HOST:PORT" s)

let with_port host port =
  if String.contains host ':' then strf "[%s]:%d" host port
  else strf "%s:%d" host port
