(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Defs

let strf = Printf.sprintf
let ( let* ) = Result.bind

type t = {
  sos : (int * string) list;
  smu : (int list * string) option;
  pieces : (int list * string) list;
  starts : (string * int) list;
  mec : int;
}

let pinned = D.pinned
let origin = D.origin

(* A defect of an image, named by the caller. *)
exception Bad of string

let bad fmt = Printf.ksprintf (fun s -> raise (Bad s)) fmt

(* Names *)

(* MP1 13.0.12 has no image: its firmware boots with the PSP's. *)
let smu_without_image = (13, 0, 12)
let ver (a, b, c) = strf "%d_%d_%d" a b c
let dotted (a, b, c) = strf "%d.%d.%d" a b c

(* A CP engine's image: version 1 is code and a jump table, which only the MEC's
   has; version 2 is RS64 code and its stack, with a start address. *)
type engine = {
  name : string;
  code : int;
  jump : int option;
  rs64 : int;
  stack : int;
}

let pfp =
  D.
    {
      name = "PFP";
      code = gfx_fw_type_cp_pfp;
      jump = None;
      rs64 = gfx_fw_type_rs64_pfp;
      stack = gfx_fw_type_rs64_pfp_p0_stack;
    }

let me =
  D.
    {
      name = "ME";
      code = gfx_fw_type_cp_me;
      jump = None;
      rs64 = gfx_fw_type_rs64_me;
      stack = gfx_fw_type_rs64_me_p0_stack;
    }

let mec =
  D.
    {
      name = "MEC";
      code = gfx_fw_type_cp_mec;
      jump = Some gfx_fw_type_cp_mec_me1;
      rs64 = gfx_fw_type_rs64_mec;
      stack = gfx_fw_type_rs64_mec_p0_stack;
    }

type kind = Sos | Smu of Discovery.version | Sdma | Cp of engine | Imu | Rlc

(* The images the blocks of [d] name, in load order, each with its kind, its
   block's name and version, and its path. *)
let wanted d =
  let block b =
    match Discovery.version d b with
    | Some v -> Ok v
    | None -> Error (strf "the GPU has no %s block" (Discovery.name b))
  in
  let* mp0 = block D.mp0_hwid in
  let* mp1 = block D.mp1_hwid in
  let* sdma = block D.sdma0_hwid in
  let* gc = block D.gc_hwid in
  let image kind b v file = (kind, Discovery.name b, v, "amdgpu/" ^ file) in
  let gc_image kind e =
    image kind D.gc_hwid gc (strf "gc_%s_%s.bin" (ver gc) e)
  in
  Ok
    (List.concat
       [
         [ image Sos D.mp0_hwid mp0 (strf "psp_%s_sos.bin" (ver mp0)) ];
         (if mp1 = smu_without_image then []
          else [ image (Smu gc) D.mp1_hwid mp1 (strf "smu_%s.bin" (ver mp1)) ]);
         [ image Sdma D.sdma0_hwid sdma (strf "sdma_%s.bin" (ver sdma)) ];
         (if gc >= (12, 0, 0) then
            [ gc_image (Cp pfp) "pfp"; gc_image (Cp me) "me" ]
          else []);
         [ gc_image (Cp mec) "mec" ];
         (if gc >= (11, 0, 0) then [ gc_image Imu "imu" ] else []);
         [ gc_image Rlc "rlc" ];
       ])

let digest path = List.assoc_opt path pinned

let names d =
  let* images = wanted d in
  let rec check = function
    | [] -> Ok (List.map (fun (_, _, _, p) -> p) images)
    | (_, block, v, path) :: rest -> (
        match digest path with
        | Some _ -> check rest
        | None ->
            Error
              (strf "%s %s is a version this library does not boot: no image %s"
                 block (dotted v) path))
  in
  check images

(* Headers *)

(* [get img (off, n)] is the little-endian field of [n] bytes at [off]. *)
let get img (off, n) =
  if off < 0 || off + n > String.length img then
    bad "a header field at byte %d lies outside its %d bytes" off
      (String.length img);
  match n with
  | 2 -> String.get_uint16_le img off
  | 4 -> Int32.to_int (String.get_int32_le img off) land 0xffff_ffff
  | n -> invalid_arg (strf "Images: a field of %d bytes" n)

(* [cut img off n] is the [n] bytes of [img] at [off]. *)
let cut img off n =
  if off < 0 || n < 0 || off + n > String.length img then
    bad "a piece of %d bytes at byte %d lies outside its %d bytes" n off
      (String.length img);
  String.sub img off n

let major img = get img D.Common_firmware_header.header_version_major
let minor img = get img D.Common_firmware_header.header_version_minor
let payload img = get img D.Common_firmware_header.ucode_array_offset_bytes
let payload_bytes img = get img D.Common_firmware_header.ucode_size_bytes

let unread img =
  bad "its header is of version %d.%d, which this library does not read"
    (major img) (minor img)

(* The whole payload, as [types]. *)
let whole img types = (types, cut img (payload img) (payload_bytes img))

(* The PSP's components. Version 2.1 lists its components up to the first
   auxiliary one. *)
let sos img =
  let count, bins =
    match (major img, minor img) with
    | 2, 0 ->
        let open D.Psp_firmware_header_v2_0 in
        (get img psp_fw_bin_count, fst psp_fw_bin)
    | 2, 1 ->
        let open D.Psp_firmware_header_v2_1 in
        (get img psp_aux_fw_bin_index, fst psp_fw_bin)
    | _ -> unread img
  in
  List.init count (fun i ->
      let at = bins + (i * D.Psp_fw_bin_desc.sizeof) in
      let field f = get img (at + fst f, snd f) in
      let open D.Psp_fw_bin_desc in
      ( field fw_type,
        cut img (payload img + field offset_bytes) (field size_bytes) ))

(* The tables a power manager of GC 9 boots with: its soft tables of the P2S
   table ID. *)
let p2s_table_id = 0x50325358

let smu_tables img =
  let open D.Smc_firmware_header_v2_1 in
  if (major img, minor img) <> (2, 1) then unread img;
  let first = get img pptable_entry_offset in
  List.init (get img pptable_count) (fun i ->
      let at = first + (i * D.Smc_soft_pptable_entry.sizeof) in
      let field f = get img (at + fst f, snd f) in
      let open D.Smc_soft_pptable_entry in
      if field id <> p2s_table_id then None
      else
        Some
          ( [ D.gfx_fw_type_p2s_table ],
            cut img (field ppt_offset_bytes) (field ppt_size_bytes) ))
  |> List.filter_map Fun.id

let sdma img =
  match major img with
  | 1 ->
      [
        whole img
          D.
            [
              gfx_fw_type_sdma0;
              gfx_fw_type_sdma1;
              gfx_fw_type_sdma2;
              gfx_fw_type_sdma3;
            ];
      ]
  | 2 ->
      let open D.Sdma_firmware_header_v2_0 in
      [
        ( [ D.gfx_fw_type_sdma_ucode_th1 ],
          cut img (get img ctl_ucode_offset) (get img ctl_ucode_size_bytes) );
        ( [ D.gfx_fw_type_sdma_ucode_th0 ],
          cut img (payload img) (get img ctx_ucode_size_bytes) );
      ]
  | 3 ->
      [
        ( [ D.gfx_fw_type_sdma_ucode_th0 ],
          cut img (payload img)
            (get img D.Sdma_firmware_header_v3_0.ucode_size_bytes) );
      ]
  | _ -> unread img

let cp img e =
  match major img with
  | 1 -> (
      let open D.Gfx_firmware_header_v1_0 in
      let jt = 4 * get img jt_size in
      match e.jump with
      | None -> unread img
      | Some jump ->
          ( [
              ([ e.code ], cut img (payload img) (payload_bytes img - jt));
              ([ jump ], cut img (payload img + (4 * get img jt_offset)) jt);
            ],
            None ))
  | 2 ->
      let open D.Gfx_firmware_header_v2_0 in
      ( [
          ([ e.rs64 ], cut img (payload img) (get img ucode_size_bytes));
          ( [ e.stack ],
            cut img (get img data_offset_bytes) (get img data_size_bytes) );
        ],
        Some
          (get img ucode_start_addr_lo lor (get img ucode_start_addr_hi lsl 32))
      )
  | _ -> unread img

let imu img =
  let open D.Imu_firmware_header_v1_0 in
  if major img <> 1 then unread img;
  let iram = get img imu_iram_ucode_size_bytes in
  [
    ([ D.gfx_fw_type_imu_i ], cut img (payload img) iram);
    ( [ D.gfx_fw_type_imu_d ],
      cut img (payload img + iram) (get img imu_dram_ucode_size_bytes) );
  ]

(* The RLC's: each minor version adds pieces to the one before. *)
let rlc img =
  if major img <> 2 then unread img;
  let piece ty (off, n) = ([ ty ], cut img (get img off) (get img n)) in
  let v = minor img in
  List.concat
    [
      (if v = 1 then
         let open D.Rlc_firmware_header_v2_1 in
         [
           piece D.gfx_fw_type_rlc_restore_list_srm_cntl
             ( save_restore_list_cntl_offset_bytes,
               save_restore_list_cntl_size_bytes );
           piece D.gfx_fw_type_rlc_restore_list_gpm_mem
             ( save_restore_list_gpm_offset_bytes,
               save_restore_list_gpm_size_bytes );
           piece D.gfx_fw_type_rlc_restore_list_srm_mem
             ( save_restore_list_srm_offset_bytes,
               save_restore_list_srm_size_bytes );
         ]
       else []);
      (if v >= 2 then
         let open D.Rlc_firmware_header_v2_2 in
         [
           piece D.gfx_fw_type_rlc_iram
             (rlc_iram_ucode_offset_bytes, rlc_iram_ucode_size_bytes);
           piece D.gfx_fw_type_rlc_dram_boot
             (rlc_dram_ucode_offset_bytes, rlc_dram_ucode_size_bytes);
         ]
       else []);
      (if v = 3 then
         let open D.Rlc_firmware_header_v2_3 in
         [
           piece D.gfx_fw_type_rlc_p
             (rlcp_ucode_offset_bytes, rlcp_ucode_size_bytes);
           piece D.gfx_fw_type_rlc_v
             (rlcv_ucode_offset_bytes, rlcv_ucode_size_bytes);
         ]
       else []);
      [ whole img [ D.gfx_fw_type_rlc_g ] ];
    ]

(* Loading *)

let empty = { sos = []; smu = None; pieces = []; starts = []; mec = 0 }

(* [add fw kind img] is [fw] with the pieces of [img], an image of [kind];
   pieces are kept in reverse until [load] ends. *)
let add fw kind img =
  match kind with
  | Sos -> { fw with sos = sos img }
  | Smu gc when gc >= (11, 0, 0) ->
      { fw with smu = Some (whole img [ D.gfx_fw_type_smu ]) }
  | Smu _ -> { fw with pieces = List.rev_append (smu_tables img) fw.pieces }
  | Sdma -> { fw with pieces = List.rev_append (sdma img) fw.pieces }
  | Cp e ->
      let ps, start = cp img e in
      let starts =
        match start with
        | Some a -> (e.name, a) :: fw.starts
        | None -> fw.starts
      in
      let version = get img D.Common_firmware_header.ucode_version in
      let mec = if e.name = "MEC" then version else fw.mec in
      { fw with pieces = List.rev_append ps fw.pieces; starts; mec }
  | Imu -> { fw with pieces = List.rev_append (imu img) fw.pieces }
  | Rlc -> { fw with pieces = List.rev_append (rlc img) fw.pieces }

let load find d =
  let* images = wanted d in
  let rec go fw = function
    | [] ->
        Ok { fw with pieces = List.rev fw.pieces; starts = List.rev fw.starts }
    | (kind, block, v, path) :: rest -> (
        match digest path with
        | None ->
            Error
              (strf "%s %s is a version this library does not boot: no image %s"
                 block (dotted v) path)
        | Some digest -> (
            let* img = find path ~digest in
            match add fw kind img with
            | fw -> go fw rest
            | exception Bad why -> Error (strf "%s: %s" path why)))
  in
  go empty images
