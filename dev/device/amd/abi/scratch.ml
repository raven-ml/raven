(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A wave's lanes, which scratch is sized for, and the least a lane takes. *)
let lanes = 64
let min_per_lane = 128

let major (g : Gpu.t) =
  let m, _, _ = g.gc in
  m

(* A wave's scratch is a multiple of 1024 bytes on GFX9, of 256 after. *)
let granule g = if major g = 9 then 1024 else 256

let per_lane g n =
  let unit = granule g / lanes in
  (Int.max n min_per_lane + unit - 1) / unit * unit

let size (g : Gpu.t) n =
  per_lane g n * lanes * g.scratch_slots * g.compute_units * g.xccs

let tmpring (g : Gpu.t) n =
  let r =
    match Register.find g "regCOMPUTE_TMPRING_SIZE" with
    | Some r -> r
    | None ->
        let a, b, c = g.gc in
        invalid_arg
          (Printf.sprintf
             "Scratch.tmpring: GC %d.%d.%d has no COMPUTE_TMPRING_SIZE" a b c)
  in
  let granule = granule g in
  let per_die = per_lane g n * lanes * g.scratch_slots * g.compute_units in
  let wave = ((lanes * per_lane g n) + granule - 1) / granule in
  let engines = if major g = 9 then 1 else g.shader_engines in
  let waves = per_die / (wave * granule) / engines in
  Register.encode r
    [
      ("waves", Int.min waves (g.compute_units * g.scratch_slots * g.xccs));
      ("wavesize", wave);
    ]

(* The descriptor's fields as HSA's runtime sets them for scratch: elements of 4
   bytes, an index stride of 64, and bounds checked against the records alone,
   past GFX9. *)
let element_size_4 = 1
let index_stride_64 = 3
let oob_select_raw = 2
let mask32 = 0xffff_ffff

let descriptor (g : Gpu.t) ~base n =
  let no_layout () =
    let a, b, c = g.gc in
    invalid_arg
      (Printf.sprintf
         "Scratch.descriptor: GC %d.%d.%d has no buffer descriptor layout" a b c)
  in
  let l =
    match List.assoc_opt (major g) Defs.sq_buf_rsrc with
    | Some l -> l
    | None -> no_layout ()
  in
  let bits (off, width) v = (v land ((1 lsl width) - 1)) lsl off in
  let format =
    match l with
    | { format = Some format; oob_select = Some oob; _ } ->
        bits format Defs.buf_format_32_uint lor bits oob oob_select_raw
    | {
     num_format = Some num;
     data_format = Some data;
     element_size = Some element;
     index_stride = Some stride;
     _;
    } ->
        bits num Defs.buf_num_format_uint
        lor bits data Defs.buf_data_format_32
        lor bits element element_size_4
        lor bits stride index_stride_64
    | _ -> no_layout ()
  in
  let word1 =
    bits l.base_address_hi (base lsr 32) lor bits l.swizzle_enable 1
  in
  let word3 =
    bits l.dst_sel_x Defs.sq_sel_x
    lor bits l.dst_sel_y Defs.sq_sel_y
    lor bits l.dst_sel_z Defs.sq_sel_z
    lor bits l.dst_sel_w Defs.sq_sel_w
    lor bits l.add_tid_enable 1
    lor bits l.type_ Defs.sq_rsrc_buf
    lor format
  in
  let b = Bytes.create 16 in
  let set i w = Bytes.set_int32_le b (4 * i) (Int32.of_int (w land mask32)) in
  set 0 base;
  set 1 word1;
  set 2 (n / g.xccs);
  set 3 word3;
  Bytes.unsafe_to_string b
