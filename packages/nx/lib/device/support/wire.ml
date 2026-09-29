(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The remote protocol's framing, shared by the client and the server.

   After the handshake, a request is a header of [header] bytes, a command and
   four 64-bit arguments, followed by the command's payload; a response is a
   status and two 64-bit results, followed by the command's payload. Every
   integer is little-endian. *)

let magic = "NXREMOTE"
let version = 1

type cmd =
  | Ping
  | Probe
  | Take
  | Release
  | Cfg_read
  | Cfg_write
  | Resize_bar
  | Reset
  | Bar
  | Map_bar
  | Reserve
  | Sysmem_alloc
  | Sysmem_free
  | Alloc
  | Free
  | Pin
  | Unpin
  | Read
  | Write
  | Copy
  | Load
  | Unload
  | Call

let cmds =
  [|
    Ping;
    Probe;
    Take;
    Release;
    Cfg_read;
    Cfg_write;
    Resize_bar;
    Reset;
    Bar;
    Map_bar;
    Reserve;
    Sysmem_alloc;
    Sysmem_free;
    Alloc;
    Free;
    Pin;
    Unpin;
    Read;
    Write;
    Copy;
    Load;
    Unload;
    Call;
  |]

let code c =
  let rec find i = if cmds.(i) = c then i else find (i + 1) in
  find 0

let cmd_of_code i =
  if i >= 0 && i < Array.length cmds then Some cmds.(i) else None

(* The commands that have no response: their failure closes the connection. *)
let posted = function Write | Copy | Call -> true | _ -> false

(* Statuses. *)
let ok = 0
let error = 1
let fatal = 2
let header = 36
let response = 17

let encode_header cmd a0 a1 a2 a3 =
  let b = Bytes.create header in
  Bytes.set_int32_le b 0 (Int32.of_int (code cmd));
  Bytes.set_int64_le b 4 (Int64.of_int a0);
  Bytes.set_int64_le b 12 (Int64.of_int a1);
  Bytes.set_int64_le b 20 (Int64.of_int a2);
  Bytes.set_int64_le b 28 (Int64.of_int a3);
  Bytes.unsafe_to_string b

let encode_response status r0 r1 =
  let b = Bytes.create response in
  Bytes.set_uint8 b 0 status;
  Bytes.set_int64_le b 1 (Int64.of_int r0);
  Bytes.set_int64_le b 9 (Int64.of_int r1);
  Bytes.unsafe_to_string b

let int64 s off = Int64.to_int (String.get_int64_le s off)

(* Sockets *)

exception Closed

let rec write_all fd s off n =
  if n > 0 then begin
    let k = Unix.write_substring fd s off n in
    write_all fd s (off + k) (n - k)
  end

let send fd s = write_all fd s 0 (String.length s)

let recv fd n =
  let b = Bytes.create n in
  let rec go off =
    if off < n then
      match Unix.read fd b off (n - off) with
      | 0 -> raise Closed
      | k -> go (off + k)
  in
  go 0;
  Bytes.unsafe_to_string b

(* The [n] bytes at [a] in the process, sent or received without a copy. *)
let send_from fd a n =
  let ba = Mmio.bigarray (Mmio.v a n) in
  let rec go off =
    if off < n then go (off + Unix.write_bigarray fd ba off (n - off))
  in
  go 0

let recv_into fd a n =
  let ba = Mmio.bigarray (Mmio.v a n) in
  let rec go off =
    if off < n then
      match Unix.read_bigarray fd ba off (n - off) with
      | 0 -> raise Closed
      | k -> go (off + k)
  in
  go 0

(* A string of a length and bytes. *)
let send_string fd s =
  let b = Bytes.create 4 in
  Bytes.set_int32_le b 0 (Int32.of_int (String.length s));
  send fd (Bytes.unsafe_to_string b ^ s)

let recv_string fd =
  let n = Int32.to_int (String.get_int32_le (recv fd 4) 0) in
  if n < 0 || n > 1 lsl 20 then raise Closed;
  recv fd n

let words l =
  let b = Bytes.create (8 * List.length l) in
  List.iteri (fun i w -> Bytes.set_int64_le b (8 * i) (Int64.of_int w)) l;
  Bytes.unsafe_to_string b

let of_words s = List.init (String.length s / 8) (fun i -> int64 s (8 * i))

(* Authentication: HMAC-SHA-256 over the nonces of both parties. *)

external sha256 : string -> string = "caml_nx_sha256"
external random : int -> string = "caml_nx_random"

let nonce () = random 32
let min_key = 16

let hmac key msg =
  let key = if String.length key > 64 then sha256 key else key in
  let pad c =
    String.init 64 (fun i ->
        let k = if i < String.length key then Char.code key.[i] else 0 in
        Char.chr (k lxor c))
  in
  sha256 (pad 0x5c ^ sha256 (pad 0x36 ^ msg))

(* Compares in time independent of where the strings differ. *)
let same a b =
  String.length a = String.length b
  &&
  let d = ref 0 in
  String.iteri (fun i c -> d := !d lor (Char.code c lxor Char.code b.[i])) a;
  !d = 0

let client_proof key ~server ~client = hmac key ("client" ^ server ^ client)
let server_proof key ~server ~client = hmac key ("server" ^ client ^ server)
