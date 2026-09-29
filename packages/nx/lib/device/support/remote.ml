(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = {
  name : string;
  fd : Unix.file_descr;
  timeout_ms : int;
  lock : Mutex.t; (* one command at a time *)
  mutable failed : string option;
  arch : string;
  page : int;
}

let name r = r.name
let arch r = r.arch
let page r = r.page
let failed r = Mutex.protect r.lock (fun () -> r.failed)

(* Fails [r] for good with [why]; [r] is taken. *)
let lost r why =
  if r.failed = None then begin
    r.failed <- Some (Printf.sprintf "%s: %s" r.name why);
    try Unix.close r.fd with Unix.Unix_error _ -> ()
  end;
  failwith (Option.get r.failed)

let transport_error r = function
  | Unix.EAGAIN | Unix.EWOULDBLOCK | Unix.ETIMEDOUT ->
      Printf.sprintf "no answer within %d ms" r.timeout_ms
  | e -> "connection lost: " ^ Unix.error_message e

(* Runs [f] with [r] taken. A broken stream fails [r]; a refused command raises
   [Failure] and leaves it usable. *)
let with_connection r f =
  Mutex.protect r.lock (fun () ->
      Option.iter failwith r.failed;
      match f () with
      | v -> v
      | exception Wire.Closed -> lost r "the server closed the connection"
      | exception Unix.Unix_error (e, _, _) -> lost r (transport_error r e))

let send_request r cmd a0 a1 a2 a3 payload =
  Wire.send r.fd (Wire.encode_header cmd a0 a1 a2 a3);
  payload r.fd

(* The answer to the last request: [reply fd r0 r1] reads its payload. *)
let answer r reply =
  let s = Wire.recv r.fd Wire.response in
  let status = Char.code s.[0]
  and r0 = Wire.int64 s 1
  and r1 = Wire.int64 s 9 in
  if status = Wire.ok then reply r.fd r0 r1
  else
    let why = Wire.recv r.fd r0 in
    if status = Wire.error then failwith (Printf.sprintf "%s: %s" r.name why)
    else lost r why

let rpc r ?(a0 = 0) ?(a1 = 0) ?(a2 = 0) ?(a3 = 0) ?(payload = ignore) cmd reply
    =
  with_connection r (fun () ->
      send_request r cmd a0 a1 a2 a3 payload;
      answer r reply)

let post r ?(a0 = 0) ?(a1 = 0) ?(a2 = 0) ?(a3 = 0) ?(payload = ignore) cmd =
  with_connection r (fun () -> send_request r cmd a0 a1 a2 a3 payload)

let unit_reply _ _ _ = ()
let r0_reply _ r0 _ = r0

(* Connecting *)

(* Connects to the first address of [host] that answers, without blocking past
   the timeout on any. *)
let connect_socket host port timeout_ms =
  let hints = [ Unix.AI_SOCKTYPE Unix.SOCK_STREAM ] in
  let attempt (ai : Unix.addr_info) =
    let fd = Unix.socket ~cloexec:true ai.ai_family ai.ai_socktype 0 in
    match
      Unix.set_nonblock fd;
      (try Unix.connect fd ai.ai_addr
       with Unix.Unix_error ((Unix.EINPROGRESS | Unix.EWOULDBLOCK), _, _) -> (
         match Unix.select [] [ fd ] [] (float_of_int timeout_ms /. 1000.) with
         | _, [], _ -> raise (Unix.Unix_error (Unix.ETIMEDOUT, "connect", ""))
         | _ -> (
             match Unix.getsockopt_error fd with
             | None -> ()
             | Some e -> raise (Unix.Unix_error (e, "connect", "")))));
      Unix.clear_nonblock fd;
      let t = float_of_int timeout_ms /. 1000. in
      Unix.setsockopt_float fd Unix.SO_RCVTIMEO t;
      Unix.setsockopt_float fd Unix.SO_SNDTIMEO t;
      Unix.setsockopt fd Unix.TCP_NODELAY true
    with
    | () -> fd
    | exception e ->
        Unix.close fd;
        raise e
  in
  let rec first = function
    | [] -> failwith "no such host"
    | [ ai ] -> attempt ai
    | ai :: rest -> ( try attempt ai with Unix.Unix_error _ -> first rest)
  in
  first (Unix.getaddrinfo host (string_of_int port) hints)

(* The server speaks first: busy, or its nonce. The client proves the key over
   both nonces, then the server proves it back and describes its machine. *)
let handshake fd key =
  let hello = Wire.recv fd (String.length Wire.magic + 5) in
  if String.sub hello 0 8 <> Wire.magic then failwith "not an nx-remote server";
  let version = Int32.to_int (String.get_int32_le hello 8) in
  if Char.code hello.[12] <> 0 then failwith (Wire.recv_string fd);
  if version <> Wire.version then
    failwith
      (Printf.sprintf "the server speaks version %d of the protocol, not %d"
         version Wire.version);
  let server = Wire.recv fd 32 and client = Wire.nonce () in
  Wire.send fd (client ^ Wire.client_proof key ~server ~client);
  if Char.code (Wire.recv fd 1).[0] <> 0 then failwith (Wire.recv_string fd);
  let proof = Wire.recv fd 32 in
  if not (Wire.same proof (Wire.server_proof key ~server ~client)) then
    failwith "the server does not know the key";
  let page = Int32.to_int (String.get_int32_le (Wire.recv fd 4) 0) in
  let arch = Wire.recv_string fd in
  (page, arch)

(* Writing to a connection the other end closed raises [EPIPE] instead of
   killing the process. *)
let quiet_sigpipe () =
  if Sys.unix then Sys.set_signal Sys.sigpipe Sys.Signal_ignore

let connect ?(timeout_ms = 30_000) ~key host port =
  quiet_sigpipe ();
  if String.length key < Wire.min_key then
    invalid_arg
      (Printf.sprintf "Remote.connect: a key of %d bytes, fewer than %d"
         (String.length key) Wire.min_key);
  let name = Printf.sprintf "%s:%d" host port in
  let fail why = failwith (Printf.sprintf "%s: %s" name why) in
  match connect_socket host port timeout_ms with
  | exception Unix.Unix_error (e, _, _) -> fail (Unix.error_message e)
  | exception Failure why -> fail why
  | fd -> (
      match handshake fd key with
      | page, arch ->
          {
            name;
            fd;
            timeout_ms;
            lock = Mutex.create ();
            failed = None;
            arch;
            page;
          }
      | exception e ->
          Unix.close fd;
          fail
            (match e with
            | Failure why -> why
            | Wire.Closed -> "the server closed the connection"
            | Unix.Unix_error ((Unix.EAGAIN | Unix.EWOULDBLOCK), _, _) ->
                Printf.sprintf "no answer within %d ms" timeout_ms
            | Unix.Unix_error (e, _, _) -> Unix.error_message e
            | e -> raise e))

let close r =
  Mutex.protect r.lock (fun () ->
      if r.failed = None then begin
        r.failed <- Some (r.name ^ ": the connection is closed");
        Unix.close r.fd
      end)

let ping r = rpc r Wire.Ping unit_reply

(* PCI functions *)

let scan r ~vendor ?class_ ids =
  let pairs =
    List.concat_map (fun (m, l) -> List.map (fun d -> (m, d)) l) ids
  in
  let words = List.concat_map (fun (m, d) -> [ m; d ]) pairs in
  rpc r Wire.Probe ~a0:vendor
    ~a1:(Option.value ~default:(-1) class_)
    ~a2:(List.length pairs)
    ~payload:(fun fd -> Wire.send fd (Wire.words words))
    (fun fd n _ ->
      String.split_on_char '\n' (Wire.recv fd n)
      |> List.filter (fun s -> s <> ""))

let take r ~lock bus =
  rpc r Wire.Take
    ~payload:(fun fd ->
      Wire.send_string fd lock;
      Wire.send_string fd bus)
    r0_reply

let release r f = rpc r Wire.Release ~a0:f unit_reply
let read_config r f off n = rpc r Wire.Cfg_read ~a0:f ~a1:off ~a2:n r0_reply

let write_config r f off n v =
  rpc r Wire.Cfg_write ~a0:f ~a1:off ~a2:n ~a3:v unit_reply

let bar r f i = rpc r Wire.Bar ~a0:f ~a1:i (fun _ a n -> (a, n))
let resize_bar r f i = rpc r Wire.Resize_bar ~a0:f ~a1:i unit_reply
let reset r f = rpc r Wire.Reset ~a0:f unit_reply

(* Memory *)

let read_string r a n =
  rpc r Wire.Read ~a1:(Nativeint.to_int a) ~a2:n (fun fd _ _ -> Wire.recv fd n)

let write_string r a s =
  post r Wire.Write ~a1:(Nativeint.to_int a) ~a2:(String.length s)
    ~payload:(fun fd -> Wire.send fd s)

let access r = { Mmio.read = read_string r; write = write_string r }

let map_bar r f i =
  let a, n = rpc r Wire.Map_bar ~a0:f ~a1:i (fun _ a n -> (a, n)) in
  Mmio.remote (access r) (Nativeint.of_int a) n

let reserve r ~base n = rpc r Wire.Reserve ~a1:base ~a2:n unit_reply
let pages_reply fd r0 r1 = (r0, Wire.of_words (Wire.recv fd (8 * r1)))

let alloc_sysmem r ?(contiguous = false) ?va n =
  let a, pages =
    rpc r Wire.Sysmem_alloc
      ~a1:(Option.value ~default:0 va)
      ~a2:n ~a3:(Bool.to_int contiguous) pages_reply
  in
  let n = (n + r.page - 1) / r.page * r.page in
  (Mmio.remote (access r) (Nativeint.of_int a) n, pages)

let free_sysmem r m =
  rpc r Wire.Sysmem_free
    ~a1:(Nativeint.to_int (Mmio.address m))
    ~a2:(Mmio.length m) unit_reply

let alloc r n =
  match rpc r Wire.Alloc ~a2:n r0_reply with
  | 0 -> None
  | a -> Some (Nativeint.of_int a)

let free r a = rpc r Wire.Free ~a1:(Nativeint.to_int a) unit_reply
let pin r a n = snd (rpc r Wire.Pin ~a1:(Nativeint.to_int a) ~a2:n pages_reply)
let unpin r a n = rpc r Wire.Unpin ~a1:(Nativeint.to_int a) ~a2:n unit_reply

let read r ~src ~dst n =
  rpc r Wire.Read ~a1:(Nativeint.to_int src) ~a2:n (fun fd _ _ ->
      Wire.recv_into fd dst n)

let write r ~dst ~src n =
  post r Wire.Write ~a1:(Nativeint.to_int dst) ~a2:n ~payload:(fun fd ->
      Wire.send_from fd src n)

let copy r ~dst ~src n =
  post r Wire.Copy ~a1:(Nativeint.to_int dst) ~a2:(Nativeint.to_int src) ~a3:n

(* Programs *)

let load r ~binary ~name =
  rpc r Wire.Load ~a1:(String.length binary) ~a2:(String.length name)
    ~payload:(fun fd -> Wire.send fd (binary ^ name))
    r0_reply

let unload r p = rpc r Wire.Unload ~a0:p unit_reply

let call r p buffers values =
  let words =
    Array.to_list buffers
    |> List.concat_map (fun (a, n) -> [ Nativeint.to_int a; n ])
  in
  post r Wire.Call ~a0:p ~a1:(Array.length buffers) ~a2:(Array.length values)
    ~payload:(fun fd ->
      Wire.send fd (Wire.words (words @ Array.to_list values)))
