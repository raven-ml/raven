(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let ( let* ) = Result.bind

(* Errors *)

let err_key fn n =
  strf "Rig_remote_proxy.Wire.%s: the key has %d bytes, not 16 to 4096" fn n

(* Constants *)

let magic = "rig-job\n"
let version = 1
let min_key = 16
let max_key = 4096
let nonce_bytes = 32
let proof_bytes = 32
let timeout_s = 10.0

(* The longest refusal read: a refusal is a sentence. *)
let max_why = 4096

(* The proofs' labels: each end's differs, so neither answers for the other. *)
let dialing_label = "rig-job dialing"
let accepting_label = "rig-job accepting"

type process = Controller | Agent of int

let process_code = function Controller -> 0 | Agent i -> i
let process_of_code n = if n = 0 then Controller else Agent n

(* Proofs: HMAC (RFC 2104) over BLAKE2b-256, whose block is 128 bytes. *)

let block = 128
let hash = Digest.BLAKE256.string

let hmac key msg =
  let key = if String.length key > block then hash key else key in
  let pad c =
    String.init block @@ fun i ->
    let k = if i < String.length key then Char.code key.[i] else 0 in
    Char.chr (k lxor c)
  in
  hash (pad 0x5c ^ hash (pad 0x36 ^ msg))

let u32 n =
  let b = Bytes.create 4 in
  Bytes.set_int32_le b 0 (Int32.of_int n);
  Bytes.unsafe_to_string b

let proof key label ~dialing ~accepting ~dialing_nonce ~accepting_nonce =
  hmac key
    (String.concat ""
       [
         label;
         u32 (process_code dialing);
         u32 (process_code accepting);
         accepting_nonce;
         dialing_nonce;
       ])

(* [a = b] in a time independent of where they differ. *)
let same a b =
  String.length a = String.length b
  &&
  let d = ref 0 in
  String.iteri (fun i c -> d := !d lor (Char.code c lxor Char.code b.[i])) a;
  !d = 0

(* Sockets *)

external send : Unix.file_descr -> string -> unit = "caml_rig_remote_wire_send"
external nonce : unit -> string = "caml_rig_remote_wire_nonce"

(* The stream ended or broke, or a field is malformed: the handshake's [Error],
   with its cause. *)
exception Stop of string

let recv fd n =
  let b = Bytes.create n in
  let rec go off =
    if off < n then
      match Unix.read fd b off (n - off) with
      | 0 -> raise (Stop "the connection closed during the handshake")
      | k -> go (off + k)
  in
  go 0;
  Bytes.unsafe_to_string b

let recv_u32 fd =
  Int32.to_int (String.get_int32_le (recv fd 4) 0) land 0xFFFF_FFFF

let recv_u8 fd = Char.code (recv fd 1).[0]
let string_bytes s = u32 (String.length s) ^ s

let recv_string fd =
  let n = recv_u32 fd in
  if n > max_why then raise (Stop "a malformed handshake");
  recv fd n

(* Runs [f] with [fd]'s reads and writes bounded by the handshake's timeout, and
   turns the stream's failures into [Error]. *)
let guarded fd f =
  let set t =
    Unix.setsockopt_float fd Unix.SO_RCVTIMEO t;
    Unix.setsockopt_float fd Unix.SO_SNDTIMEO t
  in
  match
    set timeout_s;
    let r = f () in
    set 0.0;
    r
  with
  | r -> r
  | exception Stop why -> Error why
  | exception Unix.Unix_error ((Unix.EAGAIN | Unix.EWOULDBLOCK), _, _) ->
      Error (strf "no answer within %.0f s" timeout_s)
  | exception Unix.Unix_error (e, _, _) -> Error (Unix.error_message e)

let check_key fn key =
  let n = String.length key in
  if n < min_key || n > max_key then invalid_arg (err_key fn n)

(* Dialing *)

let greeting fd =
  if recv fd (String.length magic) <> magic then
    raise (Stop "the peer is no process of a job");
  let v = recv_u32 fd in
  if v <> version then
    raise
      (Stop
         (strf "the peer speaks protocol version %d, this process %d" v version));
  match recv_u8 fd with
  | 0 -> Ok (recv fd nonce_bytes)
  | 1 -> Error (recv_string fd)
  | _ -> raise (Stop "a malformed handshake")

let dial fd ~key ~self ~peer =
  check_key "dial" key;
  (match peer with
  | Controller ->
      invalid_arg "Rig_remote_proxy.Wire.dial: the peer is the controller"
  | Agent _ when peer = self ->
      invalid_arg "Rig_remote_proxy.Wire.dial: the peer is this process"
  | Agent i when i < 1 ->
      invalid_argf "Rig_remote_proxy.Wire.dial: agent %d is no agent" i
  | Agent _ -> ());
  guarded fd @@ fun () ->
  let* accepting_nonce = greeting fd in
  let dialing_nonce = nonce () in
  let p label =
    proof key label ~dialing:self ~accepting:peer ~dialing_nonce
      ~accepting_nonce
  in
  send fd
    (String.concat ""
       [
         dialing_nonce;
         u32 (process_code self);
         u32 (process_code peer);
         p dialing_label;
       ]);
  match recv_u8 fd with
  | 1 -> Error (recv_string fd)
  | 0 ->
      if same (recv fd proof_bytes) (p accepting_label) then Ok ()
      else Error "the peer does not know the job's key"
  | _ -> raise (Stop "a malformed handshake")

(* Accepting *)

let refusal why = "\001" ^ string_bytes why

let accept fd ~key ~admit =
  check_key "accept" key;
  guarded fd @@ fun () ->
  let accepting_nonce = nonce () in
  send fd (String.concat "" [ magic; u32 version; "\000"; accepting_nonce ]);
  let dialing_nonce = recv fd nonce_bytes in
  let dialing = process_of_code (recv_u32 fd) in
  let accepting = process_of_code (recv_u32 fd) in
  let given = recv fd proof_bytes in
  let p label =
    proof key label ~dialing ~accepting ~dialing_nonce ~accepting_nonce
  in
  let refuse why =
    send fd (refusal why);
    Error why
  in
  if accepting = Controller || accepting = dialing then
    raise (Stop "a malformed handshake");
  if not (same given (p dialing_label)) then
    refuse "the dialing end does not know the job's key"
  else
    match admit dialing with
    | Error why -> refuse why
    | Ok () ->
        send fd ("\000" ^ p accepting_label);
        Ok (dialing, accepting)

let refuse fd why =
  (try send fd (String.concat "" [ magic; u32 version; refusal why ])
   with Unix.Unix_error _ -> ());
  try Unix.close fd with Unix.Unix_error _ -> ()

(* Requests, hand-overs and commands: the types of the interface, encoded in
   [link.ml]. *)

type account = {
  id : int;
  name : string;
  arch : string;
  budget : int;
  reaches : int list;
}

type _ request =
  | Join : { agents : (string * int) list } -> account request
  | Open : string -> account list request
  | Alloc : {
      id : int;
      device : int;
      memory : [ `Device | `Pinned | `Mapped ];
      bytes : int;
    }
      -> bool request
  | Map : { id : int; device : int; region : int } -> bool request
  | Load : { id : int; binary : string } -> unit request
  | Entry : { image : int; name : string } -> int option request
  | Rail : {
      id : int;
      peer : process;
      send : Rig_remote_abi.transfer array;
      receive : Rig_remote_abi.transfer array;
    }
      -> unit request

type side = Region of { id : int; offset : int } | Local
type part = Words of string | Copy of { src : side; dst : side; bytes : int }

type handover = {
  device : int;
  value : int;
  waits : (int * int) array;
  parts : part array;
}

type command =
  | Request : 'a request -> command
  | Handover : handover * Rig_remote_abi.area array -> command
  | Drop : int -> command
  | Close : command
