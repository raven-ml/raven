(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let ( let* ) = Result.bind

(* Errors *)

let err_key fn n =
  strf "Rig_remote_proxy.Wire.%s: a key of %d bytes is outside 16 to 4096" fn n

let err_malformed = "a malformed handshake"

(* Constants *)

let magic = "rig-job\n"
let version = 1
let min_key = 16
let max_key = 4096
let nonce_bytes = 32
let proof_bytes = 32
let max_agent = 0xFFFF_FFFF

(* The time an end waits for each answer. *)
let answer_s = 10.0

(* The most bytes of a refusal's reason: a reason is a sentence. *)
let max_why = 4096

(* The proofs' labels: each end's differs, so neither answers for the other. *)
let dialing_label = "rig-job dialing"
let accepting_label = "rig-job accepting"

type process = Controller | Agent of int

let process_code = function Controller -> 0 | Agent i -> i
let process_of_code n = if n = 0 then Controller else Agent n

(* Proofs: HMAC (RFC 2104) over BLAKE2b-256, whose block is 128 bytes, keyed by
   the hash of the job's key: HMAC pads a shorter key with zeros, so a key and
   the key with a trailing NUL would prove each other. *)

let block = 128
let hash = Digest.BLAKE256.string

let hmac key msg =
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
  hmac (hash key)
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

(* The stream ended or broke, an answer came late, or a field is malformed: the
   handshake's [Error], with its cause. *)
exception Stop of string

(* An answer is due by [until], on the clock of [Unix.gettimeofday]. *)
let due () = Unix.gettimeofday () +. answer_s

let recv until fd n =
  let b = Bytes.create n in
  let rec go off =
    if off < n then begin
      let left = until -. Unix.gettimeofday () in
      if left <= 0. then raise (Stop (strf "no answer within %.0f s" answer_s));
      match Unix.select [ fd ] [] [] left with
      | exception Unix.Unix_error (Unix.EINTR, _, _) -> go off
      | [], _, _ -> go off
      | _ -> (
          match Unix.read fd b off (n - off) with
          | 0 -> raise (Stop "the connection closed during the handshake")
          | k -> go (off + k))
    end
  in
  go 0;
  Bytes.unsafe_to_string b

let recv_u32 until fd =
  Int32.to_int (String.get_int32_le (recv until fd 4) 0) land 0xFFFF_FFFF

let recv_u8 until fd = Char.code (recv until fd 1).[0]

let recv_string until fd =
  let n = recv_u32 until fd in
  if n > max_why then raise (Stop err_malformed);
  recv until fd n

(* A refusal, its reason cut to [max_why] bytes. *)
let refusal why =
  let why =
    if String.length why > max_why then String.sub why 0 max_why else why
  in
  "\001" ^ u32 (String.length why) ^ why

(* Runs [f], its sends bounded by the answer's time, and turns the stream's
   failures into [Error]. *)
let guarded fd f =
  let set t = Unix.setsockopt_float fd Unix.SO_SNDTIMEO t in
  match
    set answer_s;
    let r = f () in
    set 0.0;
    r
  with
  | r -> r
  | exception Stop why -> Error why
  | exception Unix.Unix_error ((Unix.EAGAIN | Unix.EWOULDBLOCK), _, _) ->
      Error (strf "no answer within %.0f s" answer_s)
  | exception Unix.Unix_error (e, _, _) -> Error (Unix.error_message e)

let check_key fn key =
  let n = String.length key in
  if n < min_key || n > max_key then invalid_arg (err_key fn n)

let check_agent = function
  | Controller -> ()
  | Agent i when i >= 1 && i <= max_agent -> ()
  | Agent i ->
      invalid_argf "Rig_remote_proxy.Wire.dial: agent %d is outside 1 to %d" i
        max_agent

(* Dialing *)

let greeting until fd =
  if recv until fd (String.length magic) <> magic then
    raise (Stop "the peer is no process of a job");
  let v = recv_u32 until fd in
  if v <> version then
    raise
      (Stop
         (strf "the peer speaks protocol version %d, this process %d" v version));
  match recv_u8 until fd with
  | 0 -> Ok (recv until fd nonce_bytes)
  | 1 -> Error (recv_string until fd)
  | _ -> raise (Stop err_malformed)

let dial fd ~key ~self ~peer =
  check_key "dial" key;
  check_agent self;
  check_agent peer;
  if peer = Controller then
    invalid_arg "Rig_remote_proxy.Wire.dial: the peer is the controller";
  if peer = self then
    invalid_arg "Rig_remote_proxy.Wire.dial: the peer is this process";
  guarded fd @@ fun () ->
  let* accepting_nonce = greeting (due ()) fd in
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
  let until = due () in
  match recv_u8 until fd with
  | 1 -> Error (recv_string until fd)
  | 0 ->
      if same (recv until fd proof_bytes) (p accepting_label) then Ok ()
      else Error "the peer does not know the job's key"
  | _ -> raise (Stop err_malformed)

(* Accepting *)

let accept fd ~key ~admit =
  check_key "accept" key;
  guarded fd @@ fun () ->
  let accepting_nonce = nonce () in
  send fd (String.concat "" [ magic; u32 version; "\000"; accepting_nonce ]);
  let until = due () in
  let dialing_nonce = recv until fd nonce_bytes in
  let dialing = process_of_code (recv_u32 until fd) in
  let accepting = process_of_code (recv_u32 until fd) in
  let given = recv until fd proof_bytes in
  let p label =
    proof key label ~dialing ~accepting ~dialing_nonce ~accepting_nonce
  in
  let refuse why =
    send fd (refusal why);
    Error why
  in
  if accepting = Controller || accepting = dialing then
    raise (Stop err_malformed);
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
