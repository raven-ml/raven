(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

type t =
  | Agent of string
  | Started
  | Waiting
  | Listening of string
  | Closed
  | Failed of string
  | Died of string

let flat s = String.map (function '\n' | '\r' -> ' ' | c -> c) s

let to_string = function
  | Agent v -> strf "rig-agent %s\n" (flat v)
  | Started -> "started\n"
  | Waiting -> "waiting\n"
  | Listening a -> strf "listening %s\n" (flat a)
  | Closed -> "closed\n"
  | Failed why -> strf "failed %s\n" (flat why)
  | Died cause -> strf "died %s\n" (flat cause)

let of_string s =
  let after p =
    String.sub s (String.length p) (String.length s - String.length p)
  in
  let starts p = String.starts_with ~prefix:p s in
  match s with
  | "started" -> Some Started
  | "waiting" -> Some Waiting
  | "closed" -> Some Closed
  | _ when starts "rig-agent " -> Some (Agent (after "rig-agent "))
  | _ when starts "listening " -> Some (Listening (after "listening "))
  | _ when starts "failed " -> Some (Failed (after "failed "))
  | _ when starts "died " -> Some (Died (after "died "))
  | _ -> None

(* Readers *)

type reader = { fd : Unix.file_descr; pending : Buffer.t; mutable ended : bool }

let chunk = Bytes.create 4096

let reader fd =
  Unix.set_nonblock fd;
  { fd; pending = Buffer.create 256; ended = false }

let fd r = r.fd
let ended r = r.ended

(* Splits the complete lines off the pending bytes. *)
let lines r =
  let s = Buffer.contents r.pending in
  match String.rindex_opt s '\n' with
  | None -> []
  | Some i ->
      Buffer.clear r.pending;
      Buffer.add_string r.pending
        (String.sub s (i + 1) (String.length s - i - 1));
      String.split_on_char '\n' (String.sub s 0 i)

let rec fill r =
  match Unix.read r.fd chunk 0 (Bytes.length chunk) with
  | 0 ->
      r.ended <- true;
      Unix.close r.fd
  | n ->
      Buffer.add_subbytes r.pending chunk 0 n;
      fill r
  | exception
      Unix.Unix_error ((Unix.EAGAIN | Unix.EWOULDBLOCK | Unix.EINTR), _, _) ->
      ()

let read r =
  if r.ended then []
  else begin
    fill r;
    let ls = lines r in
    if r.ended then Buffer.clear r.pending;
    ls
  end
