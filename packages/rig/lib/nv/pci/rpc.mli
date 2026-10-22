(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The GSP's calls and messages, as bytes: pure functions over release
    570.144's structures. A message is a record of one or more 4 KiB
    elements, the first holding the element's header and the RPC's header
    before the body. *)

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
    naming the channel and the error: a channel the GSP stopped
    ([RC_TRIGGERED]), or a fault the MMU queued ([MMU_FAULT_QUEUED]). It is
    [None] for any other message, an error log ([OS_ERROR_LOG]) included,
    since the GSP logs there errors that stop nothing, such as a retired
    page. *)

(** {1:calls Calls} *)

val rm_alloc :
  client:int -> parent:int -> obj:int -> cls:int -> string -> string
(** [rm_alloc ~client ~parent ~obj ~cls p] is the body of the RPC [GSP_RM_ALLOC]
    that makes the object [obj] of class [cls] under [parent] of [client] with
    the parameters [p]. *)

val rm_control : client:int -> obj:int -> cmd:int -> string -> string
(** [rm_control ~client ~obj ~cmd p] is the body of the RPC [GSP_RM_CONTROL]
    that runs the command [cmd] on [obj] of [client] with the parameters [p]. *)

val free : client:int -> obj:int -> string
(** [free ~client ~obj] is the body of the RPC [FREE] that frees the object
    [obj] of [client] and every object under it. The RM's status is the
    reply's result. *)

val rm_answer : [ `Alloc | `Control ] -> string -> (int * string, string) result
(** [rm_answer k body] is the RM's status and the parameters it wrote back in
    the GSP's reply [body] to an RPC of kind [k], or [Error] if [body] is
    shorter than its header says. *)

val page_directory :
  client:int -> device:int -> vaspace:int -> root:int -> entries:int -> string
(** [page_directory ~client ~device ~vaspace ~root ~entries] is the body of
    [SET_PAGE_DIRECTORY], which points the virtual address space [vaspace] to
    the root table at the physical address [root] of the GPU's memory, of
    [entries] entries. *)

val reserved : int * int
(** [reserved] is the GSP's own addresses in an address space whose page
    tables the process keeps, as their first address and their size: the 512
    MiB from 4 GiB. The GSP maps its own buffers of that space there, so the
    process must map nothing there. *)

val reserved_pdes : Page_entry.version -> int list -> string
(** [reserved_pdes v tables] is the parameters of the command
    [COPY_SERVER_RESERVED_PDES], which hands the GSP the tables that translate
    {!reserved}: the physical addresses [tables] in the GPU's memory, root
    first, down to the dual level, whose entries map 2 MiB. The GSP takes
    them over as its own, the root as the space's page directory, and writes
    only the dual level's entries and the tables below them. The process
    writes the tables above, so the addresses it maps in the space translate
    through them.

    Raises [Invalid_argument] if [tables] is not one table per level from the
    root to the dual level. *)

val unloading : string
(** [unloading] is the body of [UNLOADING_GUEST_DRIVER], unloading to level 6:
    the GSP stops every channel and stays idle for the next boot. *)

val registry : (string * int) list -> string
(** [registry keys] is the RM's registry the GSP reads at boot
    ([PACKED_REGISTRY_TABLE]): each key with its 32-bit value. *)

(** {1:sequences CPU sequences} *)

val sequence : libos:int -> string -> (Falcon.op list, string) result
(** [sequence ~libos body] is the register accesses a
    [GSP_RUN_CPU_SEQUENCER] event's [body] asks of the CPU, as the falcons
    run them, in order:
    - a write, a modify and a poll of a register of BAR 0. A modify sets the
      value's bits outside its mask too, and a poll waits as {!Falcon.Poll}
      does;
    - a delay;
    - nothing for a store into the GSP's save area, which the GSP does not
      read back from the CPU;
    - for the GSP's core, its falcon reset ({!Falcon.reset}) and allowed to
      read physical memory without a context, started ({!Falcon.start}) or
      waited for to halt ({!Falcon.wait_halt}); a resumption resets it as a
      RISC-V core, gives it its libos arguments at the bus address [libos]
      in its mailboxes, starts SEC2 and waits for SEC2 to hand it over.

    [Error] names an opcode it does not know or a sequence that ends inside a
    command. *)
