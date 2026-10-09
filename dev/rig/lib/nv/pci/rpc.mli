(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The GSP's calls and messages, as bytes.

    A message between the process and the GSP is a record of one or more 4 KiB
    elements: the element's header (checksum, sequence number, element count),
    the RPC's header (signature, function, result, length), then its body. A
    body longer than 16 elements continues in records of the function
    [CONTINUATION_RECORD]. A record's checksum makes the XOR of its 64-bit words
    folded to 32 bits zero ([message_queue_cpu.c]). The bodies are release
    570.144's ([g_rpc-structures.h], [rmgspseq.h]). Pure functions. *)

(** {1:records Records} *)

val element_size : int
(** [element_size] is the size of an element, 4 KiB. *)

val header_size : int
(** [header_size] is the size of an element's and its RPC's headers, where a
    record's body starts. *)

val records : seq:int -> int -> string -> string list
(** [records ~seq fn body] is the records [body] is sent as, whole elements
    each, with their checksums: [fn] with the first bytes that fit 16 elements,
    then [CONTINUATION_RECORD]s with the rest, numbered from [seq]. *)

val elements : string -> int
(** [elements s] is the number of elements the record that starts [s] says it
    takes. [s] holds at least {!header_size} bytes. *)

type message = { fn : int; result : int; body : string }
(** The type for messages the GSP sends: its function or event, its result ([0]
    for success) and its body. *)

val message : string -> (message * int, string) result
(** [message s] is the message that starts the bytes [s] and the number of
    elements it takes, or [Error] if its header is not a GSP's or it is longer
    than [s]. *)

val fault : message -> string option
(** [fault m] is the report of a fault of the GPU's work that [m] carries,
    naming the channel and the error: a channel the GSP stopped ([RC_TRIGGERED],
    its exception type named as [nverror.h] names it), or a fault the MMU queued
    ([MMU_FAULT_QUEUED]). It is [None] for any other message. An error log
    ([OS_ERROR_LOG]) is [None]: the GSP stops a channel itself and says so with
    [RC_TRIGGERED] ([_kgspRpcRCTriggered]), and logs errors that stop nothing,
    such as a retired page. *)

(** {1:calls Calls} *)

val rm_alloc :
  client:int -> parent:int -> obj:int -> cls:int -> string -> string
(** [rm_alloc ~client ~parent ~obj ~cls p] is the body of the RPC [GSP_RM_ALLOC]
    that makes the object [obj] of class [cls] under [parent] of [client] with
    the parameters [p]. *)

val rm_control : client:int -> obj:int -> cmd:int -> string -> string
(** [rm_control ~client ~obj ~cmd p] is the body of the RPC [GSP_RM_CONTROL]
    that runs the command [cmd] on [obj] of [client] with the parameters [p]. *)

val rm_answer : [ `Alloc | `Control ] -> string -> (int * string, string) result
(** [rm_answer k body] is the RM's status and the parameters it wrote back in
    the GSP's answer [body] to an RPC of kind [k], or [Error] if [body] is
    shorter than its header says. *)

val page_directory :
  client:int -> device:int -> vaspace:int -> root:int -> entries:int -> string
(** [page_directory ~client ~device ~vaspace ~root ~entries] is the body of
    [SET_PAGE_DIRECTORY], which points the virtual address space [vaspace] to
    the root table at the physical address [root] of the GPU's memory, of
    [entries] entries. *)

val unloading : string
(** [unloading] is the body of [UNLOADING_GUEST_DRIVER], unloading to level 6:
    the GSP stops every channel and stays idle for the next boot. *)

val registry : (string * int) list -> string
(** [registry keys] is the RM's registry the GSP reads at boot
    ([PACKED_REGISTRY_TABLE]): each key with its 32-bit value. *)

(** {1:sequences CPU sequences} *)

(** The type for the steps of a register sequence the GSP asks of the CPU, in
    the sequencer's own terms. Registers are offsets in BAR 0. *)
type step =
  | Write of int * int  (** [Write (r, x)] writes [x] to [r]. *)
  | Modify of int * int * int
      (** [Modify (r, mask, x)] writes [(r land lnot mask) lor x] to [r]: the
          bits of [x] outside [mask] are set too. *)
  | Poll of int * int * int
      (** [Poll (r, mask, x)] waits until [r land mask] is [x]. *)
  | Delay_us of int  (** [Delay_us n] waits [n] microseconds. *)
  | Store of int * int
      (** [Store (r, i)] saves [r] in slot [i] of the GSP's save area, which the
          GSP does not read back from the CPU. *)
  | Core_reset  (** Resets the GSP's falcon. *)
  | Core_start  (** Starts it. *)
  | Core_wait_for_halt  (** Waits for it to halt. *)
  | Core_resume  (** Resumes the GSP from its suspended state. *)

val sequence : string -> (step list, string) result
(** [sequence body] is the register sequence a [GSP_RUN_CPU_SEQUENCER] event's
    [body] asks of the CPU. [Error] names an opcode it does not know or a
    sequence that ends inside a command. *)
