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

val report : message -> (string * string) option
(** [report m] is what [m] says of an error of the GSP, as the error's
    subject and its report: an error the GSP logged ([OS_ERROR_LOG]), its
    Xid and its text; a record of the GSP's error journal
    ([GSP_POST_NOCAT_RECORD]), its source and its error code. A text or a
    source ends at its first zero byte or its field's end. A message shorter
    than its structure is a report that says so. It is [None] for any other
    message. *)

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

type promotion = {
  buffer : int;
  phys : (int * int) option;
  virt : int option;
}
(** The type for a buffer of a graphics context handed to the GSP: its promote
    id ([NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_*]), its physical address and
    size in the GPU's memory if the GSP initializes it there, and its address
    in the channel's virtual address space if the GSP maps it there. *)

val promote_ctx : client:int -> channel:int -> promotion list -> string
(** [promote_ctx ~client ~channel l] is the parameters of the command
    [GPU_PROMOTE_CTX] that hands the GSP the buffers [l] of the graphics
    context of the channel [channel] of [client]. A buffer the GSP initializes
    is in the GPU's memory, uncached by the GPU, and unmapped unless it has a
    virtual address.

    Raises [Invalid_argument] if [l] has more than 16 buffers or one with
    neither address. *)

type context_buffer = { pa : int; size : int; va : int }
(** The type for a buffer of a graphics context's own: its physical address
    and size in the GPU's memory, and its address in the channel's virtual
    address space. *)

val compute_context :
  main:context_buffer ->
  pm:context_buffer ->
  patch:context_buffer ->
  globals:(int * int) list ->
  promotion list * promotion list
(** [compute_context ~main ~pm ~patch ~globals] is what the channel of a
    compute object in an address space another owner keeps hands the GSP, as
    NVIDIA's RM and its UVM do, in two calls: first, before the object is
    made, the context's main, PM and patch buffers to initialize; then, once
    it is made, the virtual address of each of the context's buffers, those
    three and the GPU's global buffers [globals], by promote id. *)

val unloading : string
(** [unloading] is the body of [UNLOADING_GUEST_DRIVER], unloading to level 6:
    the GSP stops every channel and stays idle for the next boot. *)

val registry : (string * int) list -> string
(** [registry keys] is the RM's registry the GSP reads at boot
    ([PACKED_REGISTRY_TABLE]): each key with its 32-bit value. *)

(** {1:channels Channels}

    A GPU tells channels apart by a hardware id, which NVIDIA's RM chooses
    when it runs beside a GSP and gives the GSP in the channel's allocation. *)

val channel_ids : int
(** [channel_ids] is the number of hardware ids a channel's flags hold, 4096,
    from [0]. *)

val channel_flags : int -> int -> int
(** [channel_flags flags id] is the flags [flags] of a channel's allocation
    ([NVOS04_FLAGS]) with the channel's hardware id fixed to [id]: its USERD's
    page of 8, [id / 8], fixed, and its index within the page, [id mod 8], not
    fixed by itself. The other flags are [flags]'s.

    Raises [Invalid_argument] if [id] is outside \[[0];[channel_ids - 1]\]. *)

val free_channel_id : (int -> bool) -> int option
(** [free_channel_id held] is the hardware id NVIDIA's RM gives a channel
    beside a GSP: the lowest from [1] that [held] does not hold, below
    {!channel_ids}, or [None] if [held] holds them all. Id [0] is kept for the
    GSP's own channel. *)

val channel_id : int -> int option
(** [channel_id flags] is the hardware id the flags [flags] fix as
    {!channel_flags} does, or [None] if they fix no USERD page. *)

val work_submit_token : runlist_doorbell:bool -> runlist:int -> int -> int
(** [work_submit_token ~runlist_doorbell ~runlist id] is the work submit token
    of the channel of hardware id [id] on the runlist [runlist], the word the
    host writes to the GPU's doorbell to have it fetch the channel's work:
    [runlist] in bits 22:16 and [id] in bits 11:0, and with
    [runlist_doorbell], as on Blackwell, bit 30, which rings the runlist's own
    doorbell. NVIDIA's RM makes the token itself, without the GSP.

    Raises [Invalid_argument] if [runlist] is outside \[[0];[127]\] or [id]
    outside \[[0];[4095]\]. *)

val perf_boost : flags:int -> duration:int -> string
(** [perf_boost ~flags ~duration] is the parameters of
    [INTERNAL_PERF_BOOST_SET_2X], the command NVIDIA's RM sends the GSP for a
    [PERF_BOOST] of [flags] and [duration]: [duration]'s low 32 bits, and
    [flags] cut to their low byte, which the command's boolean field holds.
    Every flag the boost defines lies in that byte. *)

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
