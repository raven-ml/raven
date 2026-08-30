(** Typed access to the generated AMD hardware tables.

    The [Amd_*_defs] modules hold raw generated data: per-family register maps,
    PM4 and SDMA packet constants and field encoders, structure layouts, and
    address-space segment bases. This module resolves that data into usable
    values: registers with absolute addresses, and the packet-constant set for
    a given hardware generation.

    Packet flavors are exposed as first-class modules ([pm4], [sdma]) rather
    than records: each flavor's constants already live in a generated module
    with the shared surface, so selection is a one-line coercion and call
    sites read [P.packet3_dispatch_direct] after [let module P = (val ...)].
    Symbols that exist for only one flavor (for example the gfx9 cache-flush
    controls) are not part of the shared signature; access them directly from
    [Amd_pm4_defs.Soc15] or [Amd_pm4_defs.Nv] inside code already branched on
    the generation. *)

module Reg : sig
  type t = {
    name : string;
    offset : int;  (** Dword offset within the block's address-space segment. *)
    segment : int;  (** Index into the block's segment-base table. *)
    fields : (string * (int * int)) array;
        (** Field name to [(lo, hi)] bit range, inclusive. *)
    addr : int;  (** Absolute dword address: segment base plus [offset]. *)
  }

  val encode : t -> (string * int) list -> int
  (** [encode t values] ors each value shifted to its field's low bit. Values
      are not masked to the field width. Raises [Invalid_argument] on an
      unknown field name. *)

  val decode : t -> int -> (string * int) list
  (** [decode t v] extracts every field of [t] from the register value [v]. *)

  val fields_mask : t -> string list -> int
  (** [fields_mask t names] is the bitmask covering the named fields. Raises
      [Invalid_argument] on an unknown field name. *)
end

module Ip : sig
  type t
  (** A hardware IP block: a versioned register family resolved against the
      die's segment bases. *)

  val create : name:string -> version:int * int * int -> bases:int array -> t
  (** [create ~name ~version ~bases] resolves the register family for the IP
      block [name] (["gc"], ["nbio"], or ["nbif"]): the greatest available
      family with the same major version that is [<= version]. [bases] are the
      block's address-space segment bases for die instance 0; each register's
      [Reg.addr] is [bases.(segment) + offset]. Raises [Invalid_argument] when
      no family matches. *)

  val name : t -> string

  val version : t -> int * int * int
  (** The version requested at [create] time (from IP discovery), not the
      resolved family's version. *)

  val reg : t -> string -> Reg.t
  (** [reg t nm] looks up the register [nm], falling back to the
      ["mm"]-prefixed spelling when the ["reg"]-prefixed one is absent.
      Raises [Invalid_argument] on an unknown register. *)
end

(** The PM4 packet surface shared by both compute-packet flavors. *)
module type Pm4 = sig
  val packet3 : int -> int -> int
  (** [packet3 op count] builds a type-3 packet header; [count] is the number
      of payload dwords minus one. *)

  val packet3_nop : int
  val packet3_set_sh_reg : int
  val packet3_set_sh_reg_start : int
  val packet3_set_sh_reg_end : int
  val packet3_set_uconfig_reg : int
  val packet3_set_uconfig_reg_start : int
  val packet3_acquire_mem : int
  val packet3_release_mem : int
  val packet3_wait_reg_mem : int
  val packet3_dispatch_direct : int
  val packet3_event_write : int
  val packet3_indirect_buffer : int
  val indirect_buffer_valid : int
  val packet3_pred_exec : int
  val cache_flush_and_inv_ts_event : int
  val event_index__mec_release_mem__end_of_pipe : int
  val data_sel__mec_release_mem__send_32_bit_low : int
  val data_sel__mec_release_mem__send_64_bit_data : int
  val data_sel__mec_release_mem__send_gpu_clock_counter : int
  val int_sel__mec_release_mem__none : int
  val int_sel__mec_release_mem__send_interrupt_after_write_confirm : int
  val wait_reg_mem_function : int -> int
  val wait_reg_mem_mem_space : int -> int
  val wait_reg_mem_operation : int -> int
  val wait_reg_mem_engine : int -> int
  val event_type : int -> int
  val event_index : int -> int
end

val pm4 : gfx9:bool -> (module Pm4)
(** [pm4 ~gfx9] is the compute-packet constant set for the generation:
    [Amd_pm4_defs.Soc15] when [gfx9], [Amd_pm4_defs.Nv] otherwise. *)

(** The SDMA packet surface shared by both packet-format versions. *)
module type Sdma = sig
  val sdma_op_copy : int
  val sdma_op_write : int
  val sdma_op_indirect : int
  val sdma_op_fence : int
  val sdma_op_trap : int
  val sdma_op_poll_regmem : int
  val sdma_op_timestamp : int
  val sdma_subop_copy_linear : int
  val sdma_subop_timestamp_get_global : int
  val sdma_pkt_copy_linear_header_sub_op : int -> int
  val sdma_pkt_copy_linear_count_count : int -> int
  val sdma_pkt_indirect_header_vmid : int -> int
  val sdma_pkt_poll_regmem_header_func : int -> int
  val sdma_pkt_poll_regmem_header_mem_poll : int -> int
  val sdma_pkt_poll_regmem_dw5_interval : int -> int
  val sdma_pkt_poll_regmem_dw5_retry_count : int -> int
  val sdma_pkt_timestamp_get_header_sub_op : int -> int
  val sdma_pkt_trap_int_context_int_context : int -> int
end

val sdma : version:int * int * int -> (module Sdma)
(** [sdma ~version] is the SDMA packet-format constant set for the engine's
    discovered version, capped at 6.0.0: major 4 selects
    [Amd_sdma_defs.V4_0_0], majors 6 and above select [Amd_sdma_defs.V6_0_0].
    Raises [Invalid_argument] for other majors. The 6.0.0-only fence-header
    mtype encoder lives in [Amd_sdma_defs.V6_0_0] outside this signature. *)

val tmpring_size : target_major:int -> waves:int -> wavesize:int -> int
(** [tmpring_size ~target_major ~waves ~wavesize] encodes the scratch-ring
    size register for target generation 9, 11, or 12; the generations differ
    in the wavesize field's width. Raises [Invalid_argument] for other
    generations or when a value does not fit its field. *)
