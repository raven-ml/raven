(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The protocol between the processes of a job.

    A job's processes are its {e controller} and an {e agent} on each other
    machine ({!process}). Each pair of them has one TCP connection, which one
    end {e dials} and the other {e accepts}: the controller dials every agent,
    and each agent dials the agents after it. Every integer is little-endian; a
    process is a u32, [0] for the controller and [i] for agent [i], from [1] to
    [2]{^ 32}[ - 1]; a string is its length (u32) and its bytes.

    {1:handshake Handshake}

    The accepting end speaks first. Then each end proves that it holds the job's
    key over both ends' nonces, the dialing end first:

    {v
     accepting: "rig-job\n", version 1 (u32), 0, its nonce (32 bytes)
                or "rig-job\n", version 1 (u32), 1 and why
     dialing:   its nonce (32 bytes), its process, the accepting end's
                process, its proof (32 bytes)
     accepting: 0 and its proof (32 bytes)
                or 1 and why
    v}

    The first [why] refuses a connection before any proof, such as one beyond
    those the accepting end can serve; the second refuses a dialing end that
    does not prove the key, or that the accepting end does not admit, such as a
    second controller. A reason has at most 4096 bytes: a longer one is cut
    there, and one announced longer is malformed.

    A proof is HMAC-BLAKE2b-256, keyed by the BLAKE2b-256 hash of the job's key,
    over its end's label (["rig-job dialing"] or ["rig-job accepting"]), the
    dialing end's process, the accepting end's process, the accepting end's
    nonce and the dialing end's nonce. The two ends' labels differ, so neither
    proof answers for the other. The key itself never crosses the connection.

    {1:frames Frames}

    After the handshake, each direction is a sequence of {e frames}: the
    payload's length (u64), the frame's kind (u8) and the payload. A frame whose
    kind is unknown or not expected from its sender, whose payload is malformed,
    or that names an id of no object of the job or of another kind than it
    needs, fails the job; so does one longer than the receiving process can
    hold. Frames carry no integrity check: the handshake admits the processes,
    and the job trusts the network between them.

    {v
     kind  name      sent by     payload
     1     request   controller  a request ({!request})
     2     answer    agent       0 and the answer, or 1 and why
     3     handover  controller  a hand-over ({!handover}), then the bytes of
                                 its copies from the controller's memory
     4     drop      controller  id (u64)
     5     word      agent       device (u64), value (u64)
     6     bytes     agent       device (u64), value (u64), bytes of a copy
     7     rail      any         rail (u64), count (u64), the transfer's bytes
     8     beat      any         nothing
     9     abort     any         why
     10    close     any         nothing
    v}

    The agent answers requests in the order they came, and applies the
    controller's frames in that order. The bytes of a hand-over's copies into
    the controller's memory come back as [bytes] frames, one per copy, in its
    parts' order, before the [word] frame of its value.

    A process sends a [beat] after a second in which it sent nothing on a
    connection, and fails the job when no byte came on one for 10 seconds. An
    [abort] fails the job with its reason, which has at most 4096 bytes. A
    connection ends in order once each end sent a [close] and received the
    other's; one that ends otherwise fails the job.

    {1:references References}

    - H. Krawczyk, M. Bellare and R. Canetti.
      {{:https://www.rfc-editor.org/rfc/rfc2104}RFC 2104},
      {e HMAC: Keyed-Hashing for Message Authentication}: the proof, over a hash
      of 128-byte blocks.
    - M-J. Saarinen and J-P. Aumasson.
      {{:https://www.rfc-editor.org/rfc/rfc7693}RFC 7693},
      {e The BLAKE2 Cryptographic Hash and Message Authentication Code}:
      BLAKE2b, its block and digest sizes. *)

(** {1:handshakes Handshakes} *)

val min_key : int
(** [min_key] is the fewest bytes of a job's key: 16. *)

val max_key : int
(** [max_key] is the most bytes of a job's key: 4096. *)

(** The type for the processes of a job. *)
type process =
  | Controller  (** The controller, on its own machine. *)
  | Agent of int  (** Agent [i], from [1]. *)

val dial :
  Unix.file_descr ->
  key:string ->
  self:process ->
  peer:process ->
  (unit, string) result
(** [dial fd ~key ~self ~peer] runs the dialing end's handshake on the connected
    socket [fd], as [self], to [peer]. It waits at most 10 seconds for each
    answer.

    [Error why] if the accepting end refused, before the proofs or after them,
    [why] its reason; speaks another version, [why] naming both; is no process
    of a job; proves another key; sends a malformed handshake; or fails to
    answer in time, or the stream fails.

    Raises [Invalid_argument] if [key] has fewer than {!min_key} or more than
    {!max_key} bytes, [self] or [peer] is an agent outside [1] to
    [2]{^ 32}[ - 1], or [peer] is [Controller] or [self]. *)

val accept :
  Unix.file_descr ->
  key:string ->
  admit:(process -> (unit, string) result) ->
  (process * process, string) result
(** [accept fd ~key ~admit] runs the accepting end's handshake on the connected
    socket [fd], and is the dialing end's process and this end's, as the dialing
    end names it. Once the dialing end proved [key], [admit p] decides, [p] the
    dialing end's process, and its [Error why] refuses it with [why], cut to
    4096 bytes. It waits at most 10 seconds for each answer.

    [Error why] if the dialing end proves another key, is not admitted, sends a
    malformed handshake, or fails to answer in time, or the stream fails.

    Raises [Invalid_argument] if [key] has fewer than {!min_key} or more than
    {!max_key} bytes. *)

val refuse : Unix.file_descr -> string -> unit
(** [refuse fd why] sends the refusal [why], cut to 4096 bytes, in place of a
    greeting, then closes [fd]. It raises nothing: a failed send is the peer's
    loss. *)

(** {1:requests Requests} *)

type account = {
  id : int;  (** Its id on its machine: [0] for the host. *)
  name : string;  (** Its name on its machine, such as ["CUDA:3"]. *)
  arch : string;  (** Its [Rig.arch] there. *)
  budget : int;  (** Its [Rig.budget] there when it opened. *)
  reaches : int list;
      (** The ids of the machine's other devices whose own memory it reaches
          there ([Rig.reaches]). *)
}
(** The type for an agent's account of one of its devices. *)

(** The type for requests, each answered by a value of its parameter. The ids of
    memory, images and rails are the controller's, from one count: each names
    one object of the job on the agent from its request until the controller
    drops it ([Drop]). *)
type _ request =
  | Join : { agents : (string * int) list } -> account request
      (** Makes the agent its job's agent, the job's agents being [agents],
          every agent's host and port in the order of their processes from [1].
          Answered with its host's account once it is connected to every other
          agent. *)
  | Open : string -> account list request
      (** Opens the agent's devices of a kind, such as ["CUDA"]: answered with
          their accounts, in its index order. *)
  | Alloc : {
      id : int;
      device : int;
      memory : [ `Device | `Pinned | `Mapped ];
      bytes : int;
    }
      -> bool request
      (** Allocates [bytes] of [device]'s [memory] as [id]: [false] if the
          device has not the room. *)
  | Map : { id : int; device : int; region : int } -> bool request
      (** Maps the memory [region], of another device of the machine, on
          [device] as [id]: [false] if [device] cannot map it. *)
  | Load : { id : int; binary : string } -> unit request
      (** Loads [binary] on the machine's host as [id]. *)
  | Entry : { image : int; name : string } -> int option request
      (** The function [name] of the image [image]: [None] if it has none. *)
  | Rail : {
      id : int;
      peer : process;
      send : Rig_remote_abi.transfer array;
      receive : Rig_remote_abi.transfer array;
    }
      -> unit request
      (** Makes the agent's end of the rail [id] to [peer]: its transfers [send]
          from the agent's machine and [receive] to it. *)

(** {1:handovers Hand-overs} *)

(** The type for the memory on one side of a copy. *)
type side =
  | Region of { id : int; offset : int }
      (** Memory of the agent's machine: the object [id], from its byte
          [offset]. *)
  | Local  (** Memory of the controller's process. *)

(** The type for the parts of a hand-over, which the agent runs in order. *)
type part =
  | Words of string  (** A run's 32-bit words, little-endian. *)
  | Copy of { src : side; dst : side; bytes : int }
      (** A copy of [bytes] bytes, [src] and [dst] not both [Local]. *)

type handover = {
  device : int;  (** The device whose work it is. *)
  value : int;  (** The value of its device's word it is the work of. *)
  waits : (int * int) array;
      (** The values of devices of the machine its work follows: device and
          value pairs. *)
  parts : part array;  (** Its work. *)
}
(** The type for hand-overs: the work of one value of a device's word.

    {v
     device (u64), value (u64), waits (u32) and each device (u64) and
     value (u64), parts (u32) and each part:
       0, words (u32) and the words
       1, bytes (u64), src side, dst side
     side: 0, id (u64), offset (u64)   or   1
    v} *)

(** {1:commands Commands} *)

(** The type for the frames an agent reads from its controller. *)
type command =
  | Request : 'a request -> command
      (** A request, which the agent answers in the order requests came. *)
  | Handover : handover * Rig_remote_abi.area array -> command
      (** A hand-over and the bytes of its copies from [Local], one area per
          such copy, in its parts' order. *)
  | Drop : int -> command
      (** Releases the object [id] once the work handed over before needs it no
          more. *)
  | Close : command
      (** Ends the job in order, after the work handed over before. *)
