(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Other machines' devices.

    Another machine runs [nx-remote], a server of its PCI functions, memory and
    host programs, and this process connects to it: {!connect} is that machine's
    host as a device, named ["CPU@HOST:PORT"]. Its buffers are memory of that
    machine, which {!Nx_device.Buffer.copy} moves over the network, and it runs
    host programs there ({!Nx_device.Program}). The libraries that open GPUs and
    network adapters, such as [nx.amd.device], [nx.nv.device] and
    [nx.rdma.device], open those of that machine given its host.

    {b Security.} The server gives this process the machine. Both ends prove
    they hold the same key, and nothing protects the traffic after that: use it
    on the machines' own network, or through a tunnel.

    {b Failures.} The connection fails for good when the machine does not answer
    within the host's {!Nx_device.timeout} at connection, when the stream
    breaks, or when an operation sent without waiting fails there. Each device
    of the machine is then lost ({!Nx_device.Lost}) with that error at its next
    operation that reaches the machine; this machine's devices go on. The server
    stops the DMA of the functions this process took and frees its memory once
    the connection is gone. {!connect} then connects anew, to a fresh host. *)

val default_port : int
(** [default_port] is [6667], the port [nx-remote] listens on by default. *)

val connect :
  ?port:int ->
  ?timeout_ms:int ->
  key:string ->
  string ->
  (Nx_device.t, string) result
(** [connect ~key host] is the host of the machine whose server listens at
    [host] and [port] (defaults to {!default_port}), proving [key] to it. The
    same host and port give the same device while its connection holds. Once the
    connection failed, [connect] connects anew: the host is a fresh device,
    unequal to the lost one, whose devices and memory stay lost. [timeout_ms]
    (defaults to {!Nx_device.Driver.default_timeout}) bounds the connection and
    every answer of the server, and is the device's {!Nx_device.timeout}, which
    {!Nx_device.set_timeout} changes later.

    [Error why] if the server cannot be reached, is busy with another client,
    speaks another version of the protocol, or either end does not know the key.

    Raises [Invalid_argument] if [key] is shorter than 16 bytes or if
    [timeout_ms <= 0]. *)

val listen : key:string -> Unix.sockaddr -> Nx_device_support.Remote_server.t
(** [listen ~key addr] serves this machine at [addr], as [nx-remote] does:
    {!Nx_device_support.Remote_server.listen} running the host programs of
    clients with {!Nx_device.Driver.host_programs}.

    Raises as {!Nx_device_support.Remote_server.listen} does. *)

val remote : Nx_device.t -> Nx_device_support.Remote.t option
(** [remote d] is the connection to the machine whose host is [d], if [d] is the
    host of another machine, and [None] if [d] is {!Nx_device.host}.

    Raises [Invalid_argument] if [d] is neither. *)
