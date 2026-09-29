(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The scratch memory of a GPU's kernels: its size, the waves it serves, and the
   buffer descriptor an AQL queue hands them. *)

module D = Amd_defs

type gpu = {
  gc : Am_reg.version;
  compute_units : int; (* of one XCC *)
  slots : int; (* scratch wave slots of a compute unit *)
  shader_engines : int; (* of one XCC *)
  xccs : int;
}

let round_up n a = (n + a - 1) / a * a

let major g =
  let m, _, _ = g.gc in
  m

(* Waves of 64 lanes, each lane taking [n] bytes rounded to the alignment of its
   generation. *)
let per_thread g n =
  let align = if major g <> 9 then 256 else 1024 in
  (round_up (Int.max n 128) (align / 64), align)

let bytes g n =
  let t, _ = per_thread g n in
  t * 64 * g.slots * g.compute_units * g.xccs

let tmpring_size g n =
  let t, align = per_thread g n in
  let per_xcc = t * 64 * g.slots * g.compute_units in
  let max_waves = g.compute_units * g.slots * g.xccs in
  let wave = ((64 * t) + align - 1) / align in
  let waves =
    per_xcc / (wave * align) / if major g <> 9 then g.shader_engines else 1
  in
  let r =
    match
      List.assoc_opt "regCOMPUTE_TMPRING_SIZE"
        (Am_reg.registers "gc" g.gc ~bases:[])
    with
    | Some r -> r
    | None -> failwith "no COMPUTE_TMPRING_SIZE register"
  in
  Am_reg.encode r [ ("waves", Int.min waves max_waves); ("wavesize", wave) ]

(* The four words of the buffer descriptor of [n] bytes of scratch at [base],
   split among the XCCs. *)
let descriptor g ~base n =
  let bits (off, width) v = (v land ((1 lsl width) - 1)) lsl off in
  let module W = (val D.sq_buf_rsrc (major g) : D.SQ_BUF_RSRC) in
  let word1 =
    bits W.base_address_hi (base lsr 32) lor bits W.swizzle_enable 1
  in
  let word3 =
    bits W.dst_sel_x D.sq_sel_x
    lor bits W.dst_sel_y D.sq_sel_y
    lor bits W.dst_sel_z D.sq_sel_z
    lor bits W.dst_sel_w D.sq_sel_w
    lor bits W.add_tid_enable 1 lor bits W.type_ D.sq_rsrc_buf
    lor
    if major g = 9 then
      bits (Option.get W.num_format) D.buf_num_format_uint
      lor bits (Option.get W.data_format) D.buf_data_format_32
      lor bits (Option.get W.element_size) 1
      lor bits (Option.get W.index_stride) 3
    else
      bits (Option.get W.format) D.buf_format_32_uint
      lor bits (Option.get W.oob_select) 2
  in
  List.map (fun w -> w land 0xffff_ffff) [ base; word1; n / g.xccs; word3 ]
