(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external host_alloc : int -> nativeint = "caml_nx_host_alloc"
external host_free : nativeint -> int -> unit = "caml_nx_host_free"

type programs = {
  load : binary:string -> name:string -> int;
  call : int -> Mmio.t array -> int array -> unit;
  unload : int -> unit;
}

let arch = match Host_arch.architecture with "amd64" -> "x86_64" | a -> a

(* Sessions *)

(* Memory a client allocated or mapped: its host memory, system memory, or a
   function's BAR, which is accessed a register at a time. *)
type kind = Host | Sysmem | Bar
type range = { kind : kind; mmio : Mmio.t }

type session = {
  fd : Unix.file_descr;
  stopping : bool Atomic.t; (* the server's *)
  programs : programs option;
  functions : (int, Pci.t) Hashtbl.t;
  mutable next_function : int;
  bars : (int * int, Mmio.t) Hashtbl.t;
  mutable ranges : range list;
  pins : (nativeint * int, int) Hashtbl.t; (* counted *)
  loaded : (int, unit) Hashtbl.t;
  mutable reserved : (int * int) list; (* address ranges, which it releases *)
}

let fail fmt = Printf.ksprintf failwith fmt

(* A request whose payload cannot be read: the stream is out of step. *)
exception Malformed of string

(* The payload of [n] bytes a request announces. *)
let payload fd n =
  if n < 0 || n > 1 lsl 30 then
    raise (Malformed (Printf.sprintf "a payload of %d bytes" n));
  Wire.recv fd n

let first r = Nativeint.to_int (Mmio.address r.mmio)

(* The range that holds the [n] bytes at [a], compared without overflow for
   whatever [a] and [n] a request names. *)
let find s a n =
  match
    List.find_opt
      (fun r ->
        let off = a - first r in
        a >= first r
        && n >= 0
        && off <= Mmio.length r.mmio
        && n <= Mmio.length r.mmio - off)
      s.ranges
  with
  | Some r -> (r, a - first r)
  | None -> fail "0x%x (%d bytes) is no memory of this connection" a n

let memory s a n =
  match find s a n with
  | ({ kind = Host | Sysmem; _ } as r), off -> Mmio.sub r.mmio off n
  | { kind = Bar; _ }, _ -> fail "0x%x is a BAR, not memory" a

let func s id =
  match Hashtbl.find_opt s.functions id with
  | Some p -> p
  | None -> fail "no function %d on this connection" id

let add_range s kind mmio = s.ranges <- { kind; mmio } :: s.ranges

(* A lock name is a word, and a bus a PCI address such as ["0000:03:00.0"]: both
   name files. *)
let word w =
  w <> ""
  && String.for_all
       (function 'a' .. 'z' | '0' .. '9' | '_' -> true | _ -> false)
       w

let pci_address b =
  let hex = function '0' .. '9' | 'a' .. 'f' -> true | _ -> false in
  String.length b = 12
  && List.for_all Fun.id
       (List.init 12 (fun i ->
            match i with
            | 4 | 7 -> b.[i] = ':'
            | 10 -> b.[i] = '.'
            | 11 -> b.[i] >= '0' && b.[i] <= '7'
            | _ -> hex b.[i]))

let remove_range s kind a =
  match List.partition (fun r -> r.kind = kind && first r = a) s.ranges with
  | r :: _, rest ->
      s.ranges <- rest;
      r.mmio
  | [], _ -> fail "0x%x is no allocation of this connection" a

(* Registers are read and written at their width. *)
let read_bar m off n =
  match n with
  | 1 -> String.make 1 (Char.chr (Mmio.get8 m off))
  | 4 when off land 3 = 0 ->
      let b = Bytes.create 4 in
      Bytes.set_int32_le b 0 (Int32.of_int (Mmio.get32 m off));
      Bytes.unsafe_to_string b
  | 8 when off land 7 = 0 ->
      let b = Bytes.create 8 in
      Bytes.set_int64_le b 0 (Mmio.get64 m off);
      Bytes.unsafe_to_string b
  | n -> Mmio.read m off n

let write_bar m off s =
  Mmio.barrier ();
  match String.length s with
  | 1 -> Mmio.set8 m off (Char.code s.[0])
  | 4 when off land 3 = 0 ->
      Mmio.set32 m off (Int32.to_int (String.get_int32_le s 0) land 0xffff_ffff)
  | 8 when off land 7 = 0 -> Mmio.set64 m off (String.get_int64_le s 0)
  | _ -> Mmio.write m off s

let log fmt = Printf.ksprintf (fun m -> prerr_endline ("nx-remote: " ^ m)) fmt

(* Stops the function's DMA. *)
let master_off p =
  let command = 0x04 and master = 0x04 in
  Pci.write_config p command 2 (Pci.read_config p command 2 land lnot master)

(* An access of configuration space: 1, 2 or 4 bytes, aligned, in its 4096. *)
let config off n =
  if
    (not (List.mem n [ 1; 2; 4 ]))
    || off < 0
    || off > 4096 - n
    || off land (n - 1) <> 0
  then fail "a configuration access of %d bytes at 0x%x" n off

(* A BAR's bytes go over the connection in pieces of at most this many. *)
let piece = 1 lsl 20

let ok ?(r0 = 0) ?(r1 = 0) fd =
  Wire.send fd (Wire.encode_response Wire.ok r0 r1)

(* Runs one command. A refused command raises [Failure] once its payload is
   read, so the stream stays in step. *)
let run s cmd a0 a1 a2 a3 =
  let fd = s.fd in
  match (cmd : Wire.cmd) with
  | Ping -> ok fd
  | Probe ->
      let words = Array.of_list (Wire.of_words (payload fd (16 * a2))) in
      let ids =
        List.init
          (Array.length words / 2)
          (fun i -> (words.(2 * i), [ words.((2 * i) + 1) ]))
      in
      let class_ = if a1 < 0 then None else Some a1 in
      let buses = Pci.scan ~vendor:a0 ?class_ ids in
      let reply = String.concat "\n" buses in
      ok ~r0:(String.length reply) fd;
      Wire.send fd reply
  | Take ->
      let lock = Wire.recv_string fd in
      let bus = Wire.recv_string fd in
      if not (word lock) then fail "%S is no lock name" lock;
      if not (pci_address bus) then fail "%S is no PCI address" bus;
      if Hashtbl.fold (fun _ p acc -> acc || Pci.bus p = bus) s.functions false
      then fail "%s is already taken by this connection" bus;
      let p = Pci.take ~lock bus in
      (* The system memory the server gives is reached physically. *)
      if Pci.addressing p = Pci.Iommu then begin
        Pci.release p;
        fail "%s is behind an IOMMU, which the server does not map memory for"
          bus
      end;
      let id = s.next_function in
      s.next_function <- id + 1;
      Hashtbl.replace s.functions id p;
      ok ~r0:id fd
  | Release ->
      let p = func s a0 in
      (* A function that cannot be stopped stays held: the cleanup tries again,
         and keeps the memory it may reach. *)
      master_off p;
      Hashtbl.filter_map_inplace
        (fun (f, _) m ->
          if f = a0 then begin
            s.ranges <- List.filter (fun r -> r.mmio != m) s.ranges;
            Pci.unmap_bar m;
            None
          end
          else Some m)
        s.bars;
      Hashtbl.remove s.functions a0;
      Pci.release p;
      ok fd
  | Cfg_read ->
      config a1 a2;
      ok ~r0:(Pci.read_config (func s a0) a1 a2) fd
  | Cfg_write ->
      config a1 a2;
      Pci.write_config (func s a0) a1 a2 a3;
      ok fd
  | Resize_bar ->
      Pci.resize_bar (func s a0) a1;
      ok fd
  | Reset ->
      Pci.reset (func s a0);
      ok fd
  | Bar ->
      let a, n = Pci.bar (func s a0) a1 in
      ok ~r0:a ~r1:n fd
  | Map_bar ->
      let m =
        match Hashtbl.find_opt s.bars (a0, a1) with
        | Some m -> m
        | None ->
            let m = Pci.map_bar (func s a0) a1 in
            Hashtbl.replace s.bars (a0, a1) m;
            add_range s Bar m;
            m
      in
      ok ~r0:(Nativeint.to_int (Mmio.address m)) ~r1:(Mmio.length m) fd
  | Reserve ->
      if not (List.mem (a1, a2) s.reserved) then begin
        Sysmem.reserve ~base:a1 a2;
        s.reserved <- (a1, a2) :: s.reserved
      end;
      ok fd
  | Sysmem_alloc ->
      let va = if a1 = 0 then None else Some a1 in
      let contiguous = a3 <> 0 in
      (* Memory at an address replaces whatever is mapped there: only a free
         part of this connection's reservations may be named. *)
      Option.iter
        (fun va ->
          let n = Sysmem.extent ~contiguous a2 in
          if
            n <= 0
            || not
                 (List.exists
                    (fun (b, m) -> va >= b && va - b <= m && n <= m - (va - b))
                    s.reserved)
          then
            fail "0x%x (%d bytes) is outside this connection's reservations" va
              a2;
          if
            List.exists
              (fun r -> va < first r + Mmio.length r.mmio && first r < va + n)
              s.ranges
          then fail "0x%x (%d bytes) overlaps memory of this connection" va a2)
        va;
      let m, pages = Sysmem.alloc ~contiguous ?va a2 in
      add_range s Sysmem m;
      ok ~r0:(Nativeint.to_int (Mmio.address m)) ~r1:(List.length pages) fd;
      Wire.send fd (Wire.words pages)
  | Sysmem_free ->
      Sysmem.free (remove_range s Sysmem a1);
      ok fd
  | Alloc ->
      if a2 <= 0 then fail "an allocation of %d bytes" a2;
      let a = host_alloc a2 in
      if a <> 0n then add_range s Host (Mmio.v a a2);
      ok ~r0:(Nativeint.to_int a) fd
  | Free ->
      let r, _ = find s a1 0 in
      if
        Hashtbl.fold
          (fun (a, _) _ acc ->
            acc
            || Nativeint.to_int a >= a1
               && Nativeint.to_int a < a1 + Mmio.length r.mmio)
          s.pins false
      then fail "0x%x is pinned" a1;
      let m = remove_range s Host a1 in
      host_free (Mmio.address m) (Mmio.length m);
      ok fd
  | Pin ->
      (match find s a1 a2 with
      | { kind = Host; _ }, _ -> ()
      | _ -> fail "0x%x is not host memory" a1);
      let a = Nativeint.of_int a1 in
      let pages = Sysmem.pin a a2 in
      Hashtbl.replace s.pins (a, a2)
        (1 + Option.value ~default:0 (Hashtbl.find_opt s.pins (a, a2)));
      ok ~r1:(List.length pages) fd;
      Wire.send fd (Wire.words pages)
  | Unpin ->
      let a = Nativeint.of_int a1 in
      (match Hashtbl.find_opt s.pins (a, a2) with
      | None -> fail "0x%x (%d bytes) is not pinned" a1 a2
      | Some 1 -> Hashtbl.remove s.pins (a, a2)
      | Some k -> Hashtbl.replace s.pins (a, a2) (k - 1));
      Sysmem.unpin a a2;
      ok fd
  | Read -> (
      match find s a1 a2 with
      | { kind = Bar; mmio }, off when a2 <= 8 ->
          let data = read_bar mmio off a2 in
          ok fd;
          Wire.send fd data
      | { kind = Bar; mmio }, off ->
          ok fd;
          let rec go at =
            if at < a2 then begin
              let k = Int.min piece (a2 - at) in
              Wire.send fd (Mmio.read mmio (off + at) k);
              go (at + k)
            end
          in
          go 0
      | { mmio; _ }, off ->
          ok fd;
          Wire.send_from fd (Mmio.address (Mmio.sub mmio off a2)) a2)
  | Write -> (
      match find s a1 a2 with
      | { kind = Bar; mmio }, off when a2 <= 8 ->
          write_bar mmio off (payload fd a2)
      | { kind = Bar; mmio }, off ->
          Mmio.barrier ();
          let rec go at =
            if at < a2 then begin
              let k = Int.min piece (a2 - at) in
              Mmio.write mmio (off + at) (Wire.recv fd k);
              go (at + k)
            end
          in
          go 0
      | { mmio; _ }, off ->
          Wire.recv_into fd (Mmio.address (Mmio.sub mmio off a2)) a2)
  | Copy ->
      let dst = memory s a1 a3 and src = memory s a2 a3 in
      Bigarray.Array1.blit (Mmio.bigarray src) (Mmio.bigarray dst)
  | Load -> (
      let binary = payload fd a1 in
      let name = payload fd a2 in
      match s.programs with
      | None -> fail "this server loads no programs"
      | Some p ->
          let id = p.load ~binary ~name in
          Hashtbl.replace s.loaded id ();
          ok ~r0:id fd)
  | Unload -> (
      match s.programs with
      | Some p when Hashtbl.mem s.loaded a0 ->
          Hashtbl.remove s.loaded a0;
          p.unload a0;
          ok fd
      | _ -> fail "no program %d on this connection" a0)
  | Call -> (
      if a1 < 0 || a2 < 0 then raise (Malformed "a negative count");
      let words =
        Array.of_list (Wire.of_words (payload fd (8 * ((2 * a1) + a2))))
      in
      let buffers =
        Array.init a1 (fun i ->
            let a = words.(2 * i) and n = words.((2 * i) + 1) in
            let r, off = find s a n in
            Mmio.sub r.mmio off n)
      in
      let values = Array.sub words (2 * a1) a2 in
      match s.programs with
      | Some p when Hashtbl.mem s.loaded a0 -> p.call a0 buffers values
      | _ -> fail "no program %d on this connection" a0)

let error_of = function
  | Failure why | Invalid_argument why | Sys_error why -> Some why
  | Unix.Unix_error (e, fn, _) -> Some (fn ^ ": " ^ Unix.error_message e)
  | Out_of_memory -> Some "out of memory"
  | _ -> None

let answer_error fd status why =
  Wire.send fd (Wire.encode_response status (String.length why) 0);
  Wire.send fd why

(* The header of the client's next command, or [Wire.Closed] once the server
   stops. The wait looks at the server each second: on Windows, the shutdown of
   the socket that ends the session leaves a read blocked on it. *)
let rec header s =
  if Atomic.get s.stopping then raise Wire.Closed;
  match Unix.select [ s.fd ] [] [] 1. with
  | [], _, _ | (exception Unix.Unix_error (Unix.EINTR, _, _)) -> header s
  | _ -> Wire.recv s.fd Wire.header

(* Serves commands until the client leaves or the server stops. A posted command
   that fails ends the session with a fatal answer the client reads at its next
   command, and so does a failure that is no refusal, such as a program's
   exception. *)
let loop s =
  let rec next () =
    match header s with
    | exception (Wire.Closed | Unix.Unix_error _) -> ()
    | h -> (
        let arg i = Wire.int64 h (4 + (8 * i)) in
        let code = Int32.to_int (String.get_int32_le h 0) in
        match Wire.cmd_of_code code with
        | None ->
            answer_error s.fd Wire.fatal (Printf.sprintf "no command %d" code)
        | Some cmd -> (
            match run s cmd (arg 0) (arg 1) (arg 2) (arg 3) with
            | () -> next ()
            | exception (Wire.Closed | Unix.Unix_error _) -> ()
            | exception Malformed why -> answer_error s.fd Wire.fatal why
            | exception e -> (
                match error_of e with
                | Some why when not (Wire.posted cmd) ->
                    answer_error s.fd Wire.error why;
                    next ()
                | Some why -> answer_error s.fd Wire.fatal why
                | None -> answer_error s.fd Wire.fatal (Printexc.to_string e))))
  in
  next ()

(* Frees what the client held once its functions' DMA is off: memory a function
   may still write is leaked on purpose, never handed back to the system. Each
   step is attempted whatever the others do. *)
let cleanup s =
  let attempt f = try f () with e when error_of e <> None -> () in
  let stopped =
    Hashtbl.fold
      (fun _ p stopped ->
        (try
           master_off p;
           true
         with e when error_of e <> None -> false)
        && stopped)
      s.functions true
  in
  Hashtbl.iter (fun _ m -> attempt (fun () -> Pci.unmap_bar m)) s.bars;
  Hashtbl.iter (fun _ p -> attempt (fun () -> Pci.release p)) s.functions;
  if stopped then begin
    Hashtbl.iter
      (fun (a, n) k ->
        for _ = 1 to k do
          attempt (fun () -> Sysmem.unpin a n)
        done)
      s.pins;
    List.iter
      (fun r ->
        match r.kind with
        | Host ->
            attempt (fun () ->
                host_free (Mmio.address r.mmio) (Mmio.length r.mmio))
        | Sysmem -> attempt (fun () -> Sysmem.free r.mmio)
        | Bar -> ())
      s.ranges;
    List.iter
      (fun (base, n) -> attempt (fun () -> Sysmem.unreserve ~base n))
      s.reserved
  end
  else
    log
      "the DMA of a client's functions could not be stopped: its memory stays \
       allocated";
  Option.iter
    (fun p ->
      Hashtbl.iter (fun id () -> attempt (fun () -> p.unload id)) s.loaded)
    s.programs

(* Connections *)

(* A handshake has this long to complete, and this many run at once: a peer that
   does not know the key holds no more than a slot of these, for no longer. *)
let handshake_s = 10.
let handshakes = 64

(* A client that answers no keepalive probe for this long is gone: its session
   ends, and cleanup stops its functions' DMA. *)
let keepalive_idle_s = 30
let keepalive_interval_s = 10
let keepalive_count = 3

external keepalive : Unix.file_descr -> int -> int -> int -> unit
  = "caml_nx_keepalive"

let close fd = try Unix.close fd with Unix.Unix_error _ -> ()

let hello status =
  let b = Bytes.create 5 in
  Bytes.set_int32_le b 0 (Int32.of_int Wire.version);
  Bytes.set_uint8 b 4 status;
  Wire.magic ^ Bytes.unsafe_to_string b

let refuse fd why =
  Wire.send fd "\001";
  Wire.send_string fd why

(* Sends the server's proof and a description of this machine. *)
let welcome fd key ~server ~client =
  let b = Bytes.create 4 in
  Bytes.set_int32_le b 0 (Int32.of_int Sysmem.page);
  Wire.send fd
    ("\000" ^ Wire.server_proof key ~server ~client ^ Bytes.unsafe_to_string b);
  Wire.send_string fd arch

(* Serves an authenticated client until it leaves, then cleans up after it. It
   never raises: whatever ends the session, the cleanup runs. *)
let serve ~programs ~stopping fd =
  let s =
    {
      fd;
      stopping;
      programs;
      functions = Hashtbl.create 4;
      next_function = 0;
      bars = Hashtbl.create 8;
      ranges = [];
      pins = Hashtbl.create 16;
      loaded = Hashtbl.create 16;
      reserved = [];
    }
  in
  (try
     Unix.setsockopt_float fd Unix.SO_RCVTIMEO 0.;
     Unix.setsockopt_float fd Unix.SO_SNDTIMEO 0.;
     keepalive fd keepalive_idle_s keepalive_interval_s keepalive_count;
     loop s
   with e -> log "a session ended: %s" (Printexc.to_string e));
  try cleanup s with e -> log "cleanup failed: %s" (Printexc.to_string e)

(* A connection whose handshake is under way: the nonce the server sent, and the
   client's nonce and proof as they arrive. *)
type pending = {
  fd : Unix.file_descr;
  nonce : string;
  reply : Bytes.t;
  mutable got : int;
  deadline : float;
}

type t = {
  address : Unix.sockaddr;
  stopping : bool Atomic.t;
  acceptor : unit Domain.t;
}

(* The acceptor runs every handshake itself, without blocking on any, and gives
   a client the session only once it proved the key and no other client holds
   it. Nothing one connection does ends the acceptor. *)
let accept_loop ~key ~programs socket stopping =
  let pending = ref [] in
  (* The session and whether it ended, and its socket. *)
  let session = ref None in
  let busy () =
    match !session with
    | Some (_, ended, _) -> not (Atomic.get ended)
    | None -> false
  in
  let join () =
    Option.iter (fun (d, _, _) -> Domain.join d) !session;
    session := None
  in
  let drop p =
    pending := List.filter (fun p' -> p'.fd != p.fd) !pending;
    close p.fd
  in
  let admit () =
    match Unix.accept ~cloexec:true socket with
    | exception
        Unix.Unix_error
          ((Unix.EMFILE | Unix.ENFILE | Unix.ENOBUFS | Unix.ENOMEM), _, _) ->
        Unix.sleepf 0.1
    | exception Unix.Unix_error _ -> ()
    | fd, _ when Atomic.get stopping -> close fd
    | fd, _ when List.length !pending >= handshakes ->
        (try
           Wire.send fd (hello 1);
           Wire.send_string fd "the server has too many connections to answer"
         with Unix.Unix_error _ -> ());
        close fd
    | fd, _ -> (
        let nonce = Wire.nonce () in
        match
          Unix.set_nonblock fd;
          Unix.setsockopt fd Unix.TCP_NODELAY true;
          Wire.send fd (hello 0 ^ nonce)
        with
        | () ->
            pending :=
              {
                fd;
                nonce;
                reply = Bytes.create 64;
                got = 0;
                deadline = Unix.gettimeofday () +. handshake_s;
              }
              :: !pending
        | exception _ -> close fd)
  in
  (* The client's nonce and proof, then the answer: refused, busy, or the
     session. *)
  let advance p =
    match Unix.read p.fd p.reply p.got (64 - p.got) with
    | exception
        Unix.Unix_error ((Unix.EAGAIN | Unix.EWOULDBLOCK | Unix.EINTR), _, _) ->
        ()
    | exception _ -> drop p
    | 0 -> drop p
    | k when p.got + k < 64 -> p.got <- p.got + k
    | _ -> (
        pending := List.filter (fun p' -> p'.fd != p.fd) !pending;
        let reply = Bytes.to_string p.reply in
        let client = String.sub reply 0 32 and proof = String.sub reply 32 32 in
        match
          Unix.clear_nonblock p.fd;
          Unix.setsockopt_float p.fd Unix.SO_SNDTIMEO handshake_s;
          if
            not
              (Wire.same proof (Wire.client_proof key ~server:p.nonce ~client))
          then (
            refuse p.fd "the client does not know the key";
            false)
          else if busy () then (
            refuse p.fd "the server is busy with another client";
            false)
          else (
            welcome p.fd key ~server:p.nonce ~client;
            true)
        with
        | exception _ -> close p.fd
        | false -> close p.fd
        | true ->
            join ();
            let ended = Atomic.make false and fd = p.fd in
            let d =
              Domain.spawn (fun () ->
                  serve ~programs ~stopping fd;
                  close fd;
                  Atomic.set ended true)
            in
            session := Some (d, ended, fd))
  in
  let rec go () =
    if not (Atomic.get stopping) then begin
      let now = Unix.gettimeofday () in
      List.iter (fun p -> if p.deadline <= now then drop p) !pending;
      let wait =
        List.fold_left (fun w p -> Float.min w (p.deadline -. now)) 1. !pending
      in
      (match
         Unix.select (socket :: List.map (fun p -> p.fd) !pending) [] [] wait
       with
      | exception Unix.Unix_error _ -> ()
      | ready, _, _ ->
          List.iter (fun p -> if List.memq p.fd ready then advance p) !pending;
          if List.memq socket ready then admit ());
      go ()
    end
  in
  Fun.protect
    ~finally:(fun () ->
      close socket;
      List.iter (fun p -> close p.fd) !pending;
      Option.iter
        (fun (_, _, fd) ->
          try Unix.shutdown fd Unix.SHUTDOWN_ALL with Unix.Unix_error _ -> ())
        !session;
      join ())
    go

let listen ~key ?programs addr =
  if String.length key < Wire.min_key then
    invalid_arg
      (Printf.sprintf "Remote_server.listen: a key of %d bytes, fewer than %d"
         (String.length key) Wire.min_key);
  if Sys.unix then Sys.set_signal Sys.sigpipe Sys.Signal_ignore;
  let socket =
    Unix.socket ~cloexec:true (Unix.domain_of_sockaddr addr) Unix.SOCK_STREAM 0
  in
  match
    Unix.setsockopt socket Unix.SO_REUSEADDR true;
    Unix.bind socket addr;
    Unix.listen socket 64;
    Unix.getsockname socket
  with
  | exception e ->
      close socket;
      raise e
  | address ->
      let stopping = Atomic.make false in
      let acceptor =
        Domain.spawn (fun () -> accept_loop ~key ~programs socket stopping)
      in
      { address; stopping; acceptor }

let address s = s.address
let wait s = Domain.join s.acceptor

(* The acceptor waits at most a second in [select]: a connection of our own
   wakes it at once. *)
let stop s =
  if not (Atomic.exchange s.stopping true) then begin
    let fd =
      Unix.socket ~cloexec:true
        (Unix.domain_of_sockaddr s.address)
        Unix.SOCK_STREAM 0
    in
    (try Unix.connect fd s.address with Unix.Unix_error _ -> ());
    close fd
  end;
  wait s
