(** A queue encoder for tests: the commands of tinygrad's [NULL] device.

    Each command is four little-endian [uint64] words: an opcode, then up to
    three operands, zero-padded. Storage and command sequences are written by
    address, other words by value. A program's arguments are a ["kernargs"]
    {!Tolk.Op.Linear} of their addresses and values. The queue is submitted by
    storing its first byte into a placeholder tagged ["doorbell"]. *)

open Tolk

val exec : int
(** [exec] is [0]: [exec kernargs nargs event]. *)

val copy : int
(** [copy] is [1]: [copy dst src event]. *)

val wait : int
(** [wait] is [2]: [wait signal value 0]. *)

val store : int
(** [store] is [3]: [store signal value]. *)

val timestamp : int
(** [timestamp] is [4]: [timestamp address], the second word of a slot. *)

type events
(** The type for the numbering of the events commands report: each kernel by its
    device, function name and program key, each copy by its source. *)

val events : unit -> events
(** [events ()] has numbered no event. *)

val commands : events -> Hcq2.Queue.t -> Hcq2.commands
(** [commands events q] encodes the queue [q] with the [NULL] commands,
    numbering their events in [events]. *)

val program : events -> int -> Ops.t
(** [program events e] is the compiled program of the kernel event [e].

    Raises [Not_found] if [e] is no kernel's event. *)
