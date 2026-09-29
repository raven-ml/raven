(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Serves this machine's PCI functions, memory and host programs to the process
   of another machine that proves the key. *)

module Remote_server = Nx_device_support.Remote_server

let usage =
  "nx-remote [--listen ADDR:PORT] --key-file FILE\n\n\
   Serves this machine's PCI functions, memory and host programs to one \
   process of another machine at a time, which drives its GPUs and network \
   adapters with nx. The client proves that it holds the key of FILE: 16 to \
   4096 bytes, of a file of this user that no other user may read, made with \
   head -c 32 /dev/urandom > FILE && chmod 600 FILE. The client is root here: \
   listen only where every host that can connect may drive this machine, such \
   as the loopback behind a tunnel that fails when it cannot forward (ssh -o \
   ExitOnForwardFailure=yes -L ...), or the machines' own network.\n"

let fail fmt =
  Printf.ksprintf
    (fun s ->
      prerr_endline ("nx-remote: " ^ s);
      exit 1)
    fmt

let address s =
  match String.rindex_opt s ':' with
  | None -> fail "%s is not ADDR:PORT" s
  | Some i -> (
      let host = String.sub s 0 i
      and port = String.sub s (i + 1) (String.length s - i - 1) in
      let host =
        if String.length host > 1 && host.[0] = '[' then
          String.sub host 1 (String.length host - 2)
        else host
      in
      match
        ( int_of_string_opt port,
          Unix.getaddrinfo host port [ Unix.AI_SOCKTYPE Unix.SOCK_STREAM ] )
      with
      | Some _, a :: _ -> a.Unix.ai_addr
      | None, _ -> fail "%s is not a port" port
      | _, [] -> fail "%s is not an address" host)

let () =
  let listen = ref "127.0.0.1:6667" and key_file = ref "" in
  Arg.parse
    [
      ( "--listen",
        Arg.Set_string listen,
        "ADDR:PORT where to listen (127.0.0.1:6667)" );
      ("--key-file", Arg.Set_string key_file, "FILE the key clients prove");
    ]
    (fun a -> fail "unexpected argument %s" a)
    usage;
  if !key_file = "" then fail "no --key-file";
  let key =
    try Nx_device_support.Remote.read_key !key_file
    with Failure why -> fail "%s" why
  in
  let addr = address !listen in
  let s =
    try Nx_remote_device.listen ~key addr
    with Unix.Unix_error (e, _, _) ->
      fail "%s: %s" !listen (Unix.error_message e)
  in
  (* Stopping disconnects the client and stops the DMA of its functions before
     the process exits and releases their memory. *)
  let quit = Atomic.make false in
  let stop = Sys.Signal_handle (fun _ -> Atomic.set quit true) in
  Sys.set_signal Sys.sigint stop;
  if Sys.unix then Sys.set_signal Sys.sigterm stop;
  (match Remote_server.address s with
  | Unix.ADDR_INET (a, p) ->
      Printf.printf "listening on %s:%d\n%!" (Unix.string_of_inet_addr a) p
  | Unix.ADDR_UNIX p -> Printf.printf "listening on %s\n%!" p);
  while not (Atomic.get quit) do
    Unix.sleepf 0.2
  done;
  Remote_server.stop s
