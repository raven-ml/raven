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
  programs : programs option;
  functions : (int, Pci.t) Hashtbl.t;
  mutable next_function : int;
  bars : (int * int, Mmio.t) Hashtbl.t;
  mutable ranges : range list;
  pins : (nativeint * int, int) Hashtbl.t; (* counted *)
  loaded : (int, unit) Hashtbl.t;
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
        a >= first r && n >= 0 && off <= Mmio.length r.mmio
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

(* A lock name is a word, and a bus a PCI address such as ["0000:03:00.0"]:
   both name files. *)
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

let ok ?(r0 = 0) ?(r1 = 0) fd =
  Wire.send fd (Wire.encode_response Wire.ok r0 r1)

(* Runs one command. A refused command raises [Failure] once its payload is
   read, so the stream stays in step. *)
let run s cmd a0 a1 a2 a3 =
  let fd = s.fd in
  match (cmd : Wire.cmd) with
  | Ping -> ok fd
  | Probe ->
      let words = Wire.of_words (payload fd (16 * a2)) in
      let rec pairs = function
        | m :: d :: l -> (m, [ d ]) :: pairs l
        | _ -> []
      in
      let class_ = if a1 < 0 then None else Some a1 in
      let buses = Pci.scan ~vendor:a0 ?class_ (pairs words) in
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
      let id = s.next_function in
      s.next_function <- id + 1;
      Hashtbl.replace s.functions id p;
      ok ~r0:id fd
  | Release ->
      let p = func s a0 in
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
  | Cfg_read -> ok ~r0:(Pci.read_config (func s a0) a1 a2) fd
  | Cfg_write ->
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
      Sysmem.reserve ~base:a1 a2;
      ok fd
  | Sysmem_alloc ->
      let va = if a1 = 0 then None else Some a1 in
      let m, pages = Sysmem.alloc ~contiguous:(a3 <> 0) ?va a2 in
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
      | { kind = Bar; mmio }, off ->
          let data = read_bar mmio off a2 in
          ok fd;
          Wire.send fd data
      | { mmio; _ }, off ->
          ok fd;
          Wire.send_from fd (Mmio.address (Mmio.sub mmio off a2)) a2)
  | Write -> (
      match find s a1 a2 with
      | { kind = Bar; mmio }, off -> write_bar mmio off (payload fd a2)
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

(* Serves commands until the client leaves. A posted command that fails ends the
   session with a fatal answer the client reads at its next command. *)
let loop s =
  let rec next () =
    match Wire.recv s.fd Wire.header with
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
                | None -> raise e
                | Some why when Wire.posted cmd ->
                    answer_error s.fd Wire.fatal why
                | Some why ->
                    answer_error s.fd Wire.error why;
                    next ())))
  in
  next ()

(* Stops the functions' DMA first, then frees what the client held. Each step is
   attempted whatever the others do. *)
let cleanup s =
  let attempt f = try f () with e when error_of e <> None -> () in
  let command = 0x04 and master = 0x04 in
  Hashtbl.iter
    (fun _ p ->
      attempt (fun () ->
          Pci.write_config p command 2
            (Pci.read_config p command 2 land lnot master)))
    s.functions;
  Hashtbl.iter (fun _ m -> attempt (fun () -> Pci.unmap_bar m)) s.bars;
  Hashtbl.iter (fun _ p -> attempt (fun () -> Pci.release p)) s.functions;
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
  Option.iter
    (fun p ->
      Hashtbl.iter (fun id () -> attempt (fun () -> p.unload id)) s.loaded)
    s.programs

(* The handshake: the server's nonce, the client's proof, then the server's
   proof and a description of this machine. *)
let hello status =
  let b = Bytes.create 5 in
  Bytes.set_int32_le b 0 (Int32.of_int Wire.version);
  Bytes.set_uint8 b 4 status;
  Wire.magic ^ Bytes.unsafe_to_string b

let authenticate fd key =
  let server = Wire.nonce () in
  Wire.send fd (hello 0 ^ server);
  let reply = Wire.recv fd 64 in
  let client = String.sub reply 0 32 and proof = String.sub reply 32 32 in
  if Wire.same proof (Wire.client_proof key ~server ~client) then begin
    let b = Bytes.create 4 in
    Bytes.set_int32_le b 0 (Int32.of_int Sysmem.page);
    Wire.send fd
      ("\000" ^ Wire.server_proof key ~server ~client ^ Bytes.unsafe_to_string b);
    Wire.send_string fd arch;
    true
  end
  else begin
    Wire.send fd "\001";
    Wire.send_string fd "the client does not know the key";
    false
  end

let serve ~key ~programs fd =
  let s =
    {
      fd;
      programs;
      functions = Hashtbl.create 4;
      next_function = 0;
      bars = Hashtbl.create 8;
      ranges = [];
      pins = Hashtbl.create 16;
      loaded = Hashtbl.create 16;
    }
  in
  Fun.protect
    ~finally:(fun () -> cleanup s)
    (fun () ->
      match authenticate fd key with
      | true -> loop s
      | false -> ()
      | exception (Wire.Closed | Unix.Unix_error _) -> ())

(* Listening *)

type t = {
  address : Unix.sockaddr;
  stopping : bool Atomic.t;
  acceptor : unit Domain.t;
}

let busy fd =
  Wire.send fd (hello 1);
  Wire.send_string fd "the server is busy with another client"

let close fd = try Unix.close fd with Unix.Unix_error _ -> ()

let accept_loop ~key ~programs socket stopping client =
  let session = ref None in
  let join () = Option.iter Domain.join !session in
  let rec go () =
    match Unix.accept ~cloexec:true socket with
    | exception Unix.Unix_error ((Unix.EINTR | Unix.ECONNABORTED), _, _) ->
        go ()
    | fd, _ when Atomic.get stopping -> close fd
    | fd, _ ->
        if Option.is_some (Atomic.get client) then begin
          (try busy fd with Unix.Unix_error _ -> ());
          close fd
        end
        else begin
          join ();
          Unix.setsockopt fd Unix.TCP_NODELAY true;
          Atomic.set client (Some fd);
          session :=
            Some
              (Domain.spawn (fun () ->
                   Fun.protect
                     ~finally:(fun () ->
                       Atomic.set client None;
                       close fd)
                     (fun () -> serve ~key ~programs fd)))
        end;
        go ()
  in
  Fun.protect
    ~finally:(fun () ->
      close socket;
      Option.iter
        (fun fd ->
          try Unix.shutdown fd Unix.SHUTDOWN_ALL with Unix.Unix_error _ -> ())
        (Atomic.get client);
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
    Unix.listen socket 4;
    Unix.getsockname socket
  with
  | exception e ->
      close socket;
      raise e
  | address ->
      let stopping = Atomic.make false in
      (* The client being served. *)
      let client = Atomic.make None in
      let acceptor =
        Domain.spawn (fun () ->
            accept_loop ~key ~programs socket stopping client)
      in
      { address; stopping; acceptor }

let address s = s.address
let wait s = Domain.join s.acceptor

(* The acceptor blocks in [accept]: a connection of our own wakes it. *)
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
