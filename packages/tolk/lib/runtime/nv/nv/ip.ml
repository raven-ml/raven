(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

module G = Nv_gsp_defs
module Mmio = Tolk_hcq.Hcq.Mmio

let debug = Tolk.Helpers.getenv "DEBUG" 0
let round_up n align = (n + align - 1) / align * align
let ceildiv n d = (n + d - 1) / d

external monotonic_ms : unit -> int = "caml_tolk_hcq_monotonic_ms" [@@noalloc]

exception Timeout_error of string

(* helpers.py:538 wait_cond, over an injected clock so tests can script
   the passage of time. *)
let wait_cond now_ms ?(timeout_ms = 10000) ~value ~msg cb =
  let start = now_ms () in
  let rec go last =
    if now_ms () - start < timeout_ms then begin
      let v = cb () in
      if v = value then () else go v
    end
    else
      raise
        (Timeout_error
           (Printf.sprintf
              "%s. Timed out after %d ms, condition not met: %d != %d" msg
              timeout_ms last value))
  in
  go 0

(* Little-endian field access over a plain byte image, keyed on the
   (byte offset, byte size) layout pairs from the generated tables. A
   [base] shifts the whole layout, so a structure embedded at a nonzero
   offset or an array element reuses the same field pairs. *)

let read_field ?(base = 0) b (off, size) =
  let p = base + off in
  match size with
  | 1 -> Char.code (Bytes.get b p)
  | 2 -> Bytes.get_uint16_le b p
  | 4 -> Int32.to_int (Bytes.get_int32_le b p) land 0xffffffff
  | 8 -> Int64.to_int (Bytes.get_int64_le b p)
  | _ -> invalid_arg "Ip: unsupported field size"

let set_field ?(base = 0) b (off, size) v =
  let p = base + off in
  match size with
  | 1 -> Bytes.set_uint8 b p (v land 0xff)
  | 2 -> Bytes.set_uint16_le b p (v land 0xffff)
  | 4 -> Bytes.set_int32_le b p (Int32.of_int (v land 0xffffffff))
  | 8 -> Bytes.set_int64_le b p (Int64.of_int v)
  | _ -> invalid_arg "Ip: unsupported field size"

let u32 b off = read_field b (off, 4)

(* The four-byte words of a byte image as unsigned integers, dropping a
   trailing partial word. *)
let u32_array b =
  Array.init (Bytes.length b / 4) (fun i -> u32 b (i * 4))

(* Firmware images *)

module Firmware = struct
  let hex digest =
    let n = Bytes.length digest in
    let out = Bytes.create (n * 2) in
    for i = 0 to n - 1 do
      let v = Bytes.get_uint8 digest i in
      Bytes.set out (2 * i) "0123456789abcdef".[v lsr 4];
      Bytes.set out ((2 * i) + 1) "0123456789abcdef".[v land 0xf]
    done;
    Bytes.unsafe_to_string out

  (* A compressed image is decoded through the system [zstd] into a
     temporary file, then read back. *)
  let zstd_dc path =
    let tmp = Filename.temp_file "tolk_nv_fw" ".bin" in
    Fun.protect
      ~finally:(fun () -> try Sys.remove tmp with Sys_error _ -> ())
      (fun () ->
        let cmd =
          Printf.sprintf "zstd -q -d -c %s > %s" (Filename.quote path)
            (Filename.quote tmp)
        in
        let status = Sys.command cmd in
        if status <> 0 then
          failwith
            (Printf.sprintf "zstd -d -c %s failed with exit code %d" path status);
        Bytes.of_string (In_channel.with_open_bin tmp In_channel.input_all))

  let url ~chip_dir name =
    Printf.sprintf "%s/nvidia/%s/gsp/%s" G.upstream chip_dir name

  let digest ~chip_dir name =
    match List.assoc_opt name G.firmware with
    | None ->
        failwith (Printf.sprintf "firmware %s has no pinned digest table" name)
    | Some by_chip -> (
        match List.assoc_opt chip_dir by_chip with
        | Some sha -> sha
        | None ->
            failwith
              (Printf.sprintf "firmware %s has no pinned digest for chip dir %s"
                 name chip_dir))

  let fetch ?dir ~chip_dir name =
    let dir =
      match dir with
      | Some d -> d
      | None -> Tolk.Helpers.getenv_str "NV_FW_PATH" "/lib/firmware/nvidia"
    in
    let sha = digest ~chip_dir name in
    let plain = Filename.concat (Filename.concat dir chip_dir) (Filename.concat "gsp" name) in
    let zst = plain ^ ".zst" in
    let blob =
      if Sys.file_exists plain then
        Bytes.of_string (In_channel.with_open_bin plain In_channel.input_all)
      else if Sys.file_exists zst then zstd_dc zst
      else
        failwith
          (Printf.sprintf
             "firmware %s not found: searched %s and %s; fetch it from %s \
              (expected sha256 %s)"
             name plain zst (url ~chip_dir name) sha)
    in
    let actual = hex (Tolk.Helpers.sha256 blob) in
    if not (String.equal actual sha) then
      failwith
        (Printf.sprintf
           "firmware %s sha mismatch: expected %s but got %s for %s (pinned \
            source %s)"
           name sha actual plain (url ~chip_dir name));
    blob
end

(* Rpc_queue: ip.py NVRpcQueue *)

module Rpc_queue = struct
  type t = {
    view : Mmio.t;
    queue : Mmio.t;
    msg_size : int;
    msg_count : int;
    rx_hdr_off : int;
    mutable rx : Mmio.t option;
    mutable seq : int;
    notify : unit -> unit;
    on_run_cpu_seq : bytes -> unit;
    on_error : unit -> unit;
    devfmt : string;
    now_ms : unit -> int;
  }

  (* Unsigned 32-bit access into a mapped view. *)
  let u32r m off = Int32.to_int (Mmio.read32 m off) land 0xffffffff
  let u32w m off v = Mmio.write32 m off (Int32.of_int (v land 0xffffffff))

  (* ip.py:20 NVRpcQueue.__init__ *)
  let make ?completion_q_view ?(now_ms = monotonic_ms) ?(devfmt = "?") ~notify
      ~on_run_cpu_seq ~on_error view =
    wait_cond now_ms ~value:0x1000 ~msg:"RPC queue not initialized" (fun () ->
        u32r view (fst G.Msgq_tx_header.entryoff));
    let hdr = Mmio.read_bytes view ~off:0 ~len:G.Msgq_tx_header.sizeof in
    let msg_size = read_field hdr G.Msgq_tx_header.msgsize in
    let msg_count = read_field hdr G.Msgq_tx_header.msgcount in
    let entry_off = read_field hdr G.Msgq_tx_header.entryoff in
    let rx =
      match completion_q_view with
      | None -> None
      | Some comp ->
          Some (Mmio.view comp ~off:(u32r comp (fst G.Msgq_tx_header.rxhdroff)) ())
    in
    {
      view;
      queue = Mmio.view view ~off:entry_off ~size:(msg_size * msg_count) ();
      msg_size;
      msg_count;
      rx_hdr_off = read_field hdr G.Msgq_tx_header.rxhdroff;
      rx;
      seq = 0;
      notify;
      on_run_cpu_seq;
      on_error;
      devfmt;
      now_ms;
    }

  let rx_hdr_off t = t.rx_hdr_off
  let set_rx_view t rx = t.rx <- Some rx

  (* ip.py:32 NVRpcQueue._checksum: xor of the frame's little-endian
     64-bit words (zero-padded to a whole word), folded to 32 bits. *)
  let checksum data =
    let padded =
      let r = round_up (Bytes.length data) 8 in
      if r = Bytes.length data then data
      else begin
        let p = Bytes.make r '\000' in
        Bytes.blit data 0 p 0 (Bytes.length data);
        p
      end
    in
    let c = ref 0L in
    for i = 0 to (Bytes.length padded / 8) - 1 do
      c := Int64.logxor !c (Bytes.get_int64_le padded (i * 8))
    done;
    Int64.to_int
      (Int64.logxor (Int64.shift_right_logical !c 32) (Int64.logand !c 0xffffffffL))

  (* ip.py:38 NVRpcQueue._send_rpc_record *)
  let send_rpc_record t func msg =
    let header = Bytes.make G.Rpc_message_header.sizeof '\000' in
    set_field header G.Rpc_message_header.signature G.nv_vgpu_msg_signature_valid;
    set_field header G.Rpc_message_header.rpc_result G.nv_vgpu_msg_result_rpc_pending;
    set_field header G.Rpc_message_header.rpc_result_private
      G.nv_vgpu_msg_result_rpc_pending;
    set_field header G.Rpc_message_header.header_version (3 lsl 24);
    set_field header G.Rpc_message_header.func func;
    set_field header G.Rpc_message_header.length (Bytes.length msg + 0x20);
    let msg = Bytes.cat header msg in
    let elem_count =
      ceildiv (Bytes.length msg + G.Gsp_msg_queue_element.sizeof) t.msg_size
    in
    let phdr = Bytes.make G.Gsp_msg_queue_element.sizeof '\000' in
    set_field phdr G.Gsp_msg_queue_element.elemcount elem_count;
    set_field phdr G.Gsp_msg_queue_element.seqnum t.seq;
    set_field phdr G.Gsp_msg_queue_element.checksum (checksum (Bytes.cat phdr msg));
    let frame = Bytes.make (elem_count * t.msg_size) '\000' in
    Bytes.blit phdr 0 frame 0 G.Gsp_msg_queue_element.sizeof;
    Bytes.blit msg 0 frame G.Gsp_msg_queue_element.sizeof (Bytes.length msg);
    let wp = u32r t.view (fst G.Msgq_tx_header.writeptr) in
    let off = wp * t.msg_size in
    let first = min (Bytes.length frame) (Mmio.size t.queue - off) in
    Mmio.blit_bytes t.queue ~off (Bytes.sub frame 0 first);
    if first < Bytes.length frame then
      Mmio.blit_bytes t.queue ~off:0
        (Bytes.sub frame first (Bytes.length frame - first));
    u32w t.view (fst G.Msgq_tx_header.writeptr) ((wp + elem_count) mod t.msg_count);
    Mmio.fence ();
    t.seq <- t.seq + 1;
    t.notify ()

  (* ip.py:57 NVRpcQueue.send_rpc: a message larger than one 16-element
     frame continues in follow-on continuation records. *)
  let send_rpc t func msg =
    let max_payload =
      (t.msg_size * 16) - G.Gsp_msg_queue_element.sizeof
      - G.Rpc_message_header.sizeof
    in
    let len = Bytes.length msg in
    send_rpc_record t func (Bytes.sub msg 0 (min len max_payload));
    let off = ref max_payload in
    while !off < len do
      send_rpc_record t G.nv_vgpu_msg_function_continuation_record
        (Bytes.sub msg !off (min max_payload (len - !off)));
      off := !off + max_payload
    done

  (* ip.py:62 NVRpcQueue.read_resp *)
  let read_resp t =
    let rx =
      match t.rx with
      | Some rx -> rx
      | None -> invalid_arg "Ip.Rpc_queue: queue has no read pointer"
    in
    let qsize = Mmio.size t.queue in
    let rec step () =
      let rp = u32r rx 0 in
      if rp = u32r t.view (fst G.Msgq_tx_header.writeptr) then Seq.Nil
      else begin
        let off = rp * t.msg_size in
        let hdr =
          Mmio.read_bytes t.queue
            ~off:(off + G.Gsp_msg_queue_element.sizeof)
            ~len:G.Rpc_message_header.sizeof
        in
        let func = read_field hdr G.Rpc_message_header.func in
        let length = read_field hdr G.Rpc_message_header.length in
        let result = read_field hdr G.Rpc_message_header.rpc_result in
        let start =
          off + G.Gsp_msg_queue_element.sizeof + G.Rpc_message_header.sizeof
        in
        let msg =
          Mmio.read_bytes t.queue ~off:start
            ~len:(max 0 (min length (qsize - start)))
        in
        (* Handling special functions *)
        if func = G.nv_vgpu_msg_event_gsp_run_cpu_sequencer then
          t.on_run_cpu_seq msg
        else if func = G.nv_vgpu_msg_event_os_error_log then begin
          let text =
            if Bytes.length msg > 12 then
              Bytes.sub_string msg 12 (Bytes.length msg - 12)
            else ""
          in
          let stop = ref (String.length text) in
          while !stop > 0 && text.[!stop - 1] = '\000' do
            decr stop
          done;
          Printf.printf "nv %s: GSP LOG: %s\n%!" t.devfmt (String.sub text 0 !stop)
        end;
        if
          func = G.nv_vgpu_msg_event_os_error_log
          || func = G.nv_vgpu_msg_event_mmu_fault_queued
        then t.on_error ();
        (* Update the read pointer *)
        u32w rx 0 ((rp + (round_up length t.msg_size / t.msg_size)) mod t.msg_count);
        Mmio.fence ();
        if debug >= 3 then begin
          let nm =
            match List.assoc_opt func G.rpc_fns with
            | Some nm -> nm
            | None -> (
                match List.assoc_opt func G.rpc_events with
                | Some nm -> nm
                | None -> Printf.sprintf "ev:%x" func)
          in
          Printf.printf "nv %s: in RPC: %s, res:0x%x\n%!" t.devfmt nm result
        end;
        if result <> 0 then
          failwith
            (Printf.sprintf "RPC call %d failed with result %d" func result);
        Seq.Cons ((func, msg), step)
      end
    in
    fun () ->
      Mmio.fence ();
      step ()

  (* ip.py:87 NVRpcQueue.wait_resp *)
  let wait_resp t ?(timeout_ms = 10000) cmd =
    let start = t.now_ms () in
    let rec attempt () =
      if t.now_ms () - start < timeout_ms then
        let found =
          Seq.find_map
            (fun (func, msg) -> if func = cmd then Some msg else None)
            (read_resp t)
        in
        match found with Some msg -> msg | None -> attempt ()
      else
        failwith
          (Printf.sprintf "Timeout waiting for RPC response for command %d" cmd)
    in
    attempt ()
end

(* Falcon microcontroller boot images and execution *)

module R = Nvdev.Nv_reg

let lo32 v = v land 0xffffffff
let hi32 v = v lsr 32

(* A device register resolved by name and rebased onto a block. *)
let based nvdev name base = R.with_base (Nvdev.reg nvdev name) base

(* The device clock shaped for [wait_cond]. *)
let dev_clock nvdev () = Nvdev.now_ms nvdev

(* time.sleep over the device clock, so a scripted clock makes the boot
   delays consume no wall time. *)
let sleep_ms nvdev ms =
  let start = Nvdev.now_ms nvdev in
  while Nvdev.now_ms nvdev - start < ms do
    ()
  done

module Flcn = struct
  type desc = {
    imem_load_size : int;
    imem_phys_base : int;
    imem_virt_base : int;
    dmem_phys_base : int;
    dmem_load_size : int;
    pkc_data_offset : int;
    engine_id_mask : int;
    ucode_id : int;
    stored_size : int;
    interface_offset : int;
  }

  type ucode = { desc : desc; frts_offset : int; frts_image : bytes }

  (* The VBIOS image lives behind a fixed 1 MiB register window; read it
     word by word into a flat byte image. [read32] answers with the
     unsigned 32-bit value at a byte address in the register aperture. *)
  let read_vbios ~read32 =
    let base = 0x00300000 and len = 0x00100000 in
    let b = Bytes.create len in
    for i = 0 to (len / 4) - 1 do
      let w = read32 (base + (i * 4)) in
      Bytes.set_int32_le b (i * 4) (Int32.of_int (w land 0xffffffff))
    done;
    b

  (* Walk the PCI expansion-ROM images to the extended VBIOS block; its
     base is where the falcon ucode tables are addressed from. Each image
     header carries a pointer to its PCI data structure, whose block count
     gives the image length and whose code type distinguishes the base
     image from the extended one. *)
  let expansion_rom_off rom =
    let rec walk off block_size =
      let pci_blck = read_field rom (off + G.offsetof_pci_exp_rom_pci_data_struct_ptr, 2) in
      let imglen =
        read_field rom (off + pci_blck + G.offsetof_pci_data_struct_image_len, 2)
        * G.pci_rom_image_block_size
      in
      let code_type =
        read_field rom (off + pci_blck + G.offsetof_pci_data_struct_code_type, 1)
      in
      if code_type = G.nv_bcrt_hash_info_base_code_type_vbios_ext then
        off - block_size
      else
        let block_size =
          if code_type = G.nv_bcrt_hash_info_base_code_type_vbios_base then imglen
          else block_size
        in
        walk (off + imglen) block_size
    in
    walk 0 0

  (* Locate the production FWSEC ucode descriptor by walking the BIT
     tokens to the falcon data token, then its ucode table. Returns the
     descriptor's byte offset in the ROM and its size. *)
  let fwsec_desc rom expansion =
    let bit_addr = 0x1b0 in
    let signature = read_field rom ~base:bit_addr G.Bit_header_v1_00.signature in
    if signature <> 0x00544942 then
      failwith
        (Printf.sprintf "Invalid BIT header signature 0x%x" signature);
    let token_entries = read_field rom ~base:bit_addr G.Bit_header_v1_00.tokenentries in
    let header_size = read_field rom ~base:bit_addr G.Bit_header_v1_00.headersize in
    let token_size = read_field rom ~base:bit_addr G.Bit_header_v1_00.tokensize in
    let found = ref None in
    for i = 0 to token_entries - 1 do
      let tbase = bit_addr + header_size + (i * token_size) in
      let token_id = read_field rom ~base:tbase G.Bit_token_v1_00.tokenid in
      let data_version = read_field rom ~base:tbase G.Bit_token_v1_00.dataversion in
      let data_size = read_field rom ~base:tbase G.Bit_token_v1_00.datasize in
      if
        token_id = G.bit_token_falcon_data && data_version = 2
        && data_size >= G.bit_data_falcon_data_v2_size_4
      then begin
        let data_ptr = read_field rom ~base:tbase G.Bit_token_v1_00.dataptr in
        let table_ptr =
          expansion
          + read_field rom
              ~base:(data_ptr land 0xffff)
              G.Bit_data_falcon_data_v2.falconucodetableptr
        in
        let entry_count = read_field rom ~base:table_ptr G.Falcon_ucode_table_hdr_v1.entrycount in
        let hdr_size = read_field rom ~base:table_ptr G.Falcon_ucode_table_hdr_v1.headersize in
        let entry_size = read_field rom ~base:table_ptr G.Falcon_ucode_table_hdr_v1.entrysize in
        for j = 0 to entry_count - 1 do
          let ebase = table_ptr + hdr_size + (j * entry_size) in
          let app_id = read_field rom ~base:ebase G.Falcon_ucode_table_entry_v1.applicationid in
          if app_id = G.falcon_ucode_entry_appid_fwsec_prod then begin
            let desc_off =
              expansion + read_field rom ~base:ebase G.Falcon_ucode_table_entry_v1.descptr
            in
            let vdesc = read_field rom ~base:desc_off G.Falcon_ucode_desc_header.vdesc in
            found := Some (desc_off, vdesc lsr 16)
          end
        done
      end
    done;
    match !found with
    | Some d -> d
    | None -> failwith "FWSEC production ucode descriptor not found in VBIOS"

  let read_desc rom off =
    {
      imem_load_size = read_field rom ~base:off G.Falcon_ucode_desc_v3.imemloadsize;
      imem_phys_base = read_field rom ~base:off G.Falcon_ucode_desc_v3.imemphysbase;
      imem_virt_base = read_field rom ~base:off G.Falcon_ucode_desc_v3.imemvirtbase;
      dmem_phys_base = read_field rom ~base:off G.Falcon_ucode_desc_v3.dmemphysbase;
      dmem_load_size = read_field rom ~base:off G.Falcon_ucode_desc_v3.dmemloadsize;
      pkc_data_offset = read_field rom ~base:off G.Falcon_ucode_desc_v3.pkcdataoffset;
      engine_id_mask = read_field rom ~base:off G.Falcon_ucode_desc_v3.engineidmask;
      ucode_id = read_field rom ~base:off G.Falcon_ucode_desc_v3.ucodeid;
      stored_size = read_field rom ~base:off G.Falcon_ucode_desc_v3.storedsize;
      interface_offset = read_field rom ~base:off G.Falcon_ucode_desc_v3.interfaceoffset;
    }

  (* The FWSEC command that reserves the frame-buffer resident tables. Its
     region descriptor points at the top 1 MiB of the 2 MiB reservation
     below the VGA workspace. *)
  let frts_cmd ~frts_offset =
    let b = Bytes.make G.Fwseclic_frts_cmd.sizeof '\000' in
    set_field b G.Fwseclic_read_vbios_desc.version 0x1;
    set_field b G.Fwseclic_read_vbios_desc.size G.Fwseclic_read_vbios_desc.sizeof;
    set_field b G.Fwseclic_read_vbios_desc.flags 2;
    let rbase = fst G.Fwseclic_frts_cmd.frtsregiondesc in
    set_field b ~base:rbase G.Fwseclic_frts_region_desc.version 0x1;
    set_field b ~base:rbase G.Fwseclic_frts_region_desc.size G.Fwseclic_frts_region_desc.sizeof;
    set_field b ~base:rbase G.Fwseclic_frts_region_desc.frtsregionoffset4k (frts_offset lsr 12);
    set_field b ~base:rbase G.Fwseclic_frts_region_desc.frtsregionsize 0x100;
    set_field b ~base:rbase G.Fwseclic_frts_region_desc.frtsregionmediatype 2;
    b

  (* Patch the FWSEC image for boot: select the FRTS command in the DMEM
     mapper, splice the command payload into its input buffer, and splice
     the trailing 0x180-byte signature over the descriptor's PKC data
     region. Offsets in the image are relative to the loaded IMEM. *)
  let patch_fwsec ~desc ~image ~signature ~cmd_id ~cmd =
    let patched = Bytes.copy image in
    let app_hdr_off = desc.imem_load_size + desc.interface_offset in
    let entry_count = read_field image ~base:app_hdr_off G.Falcon_application_interface_header_v1.entrycount in
    let dmem_offset = ref 0 in
    for i = 0 to entry_count - 1 do
      let ebase =
        app_hdr_off + G.Falcon_application_interface_header_v1.sizeof
        + (i * G.Falcon_application_interface_entry_v1.sizeof)
      in
      if read_field image ~base:ebase G.Falcon_application_interface_entry_v1.id
         = G.falcon_application_interface_entry_id_dmemmapper
      then
        dmem_offset := read_field image ~base:ebase G.Falcon_application_interface_entry_v1.dmemoffset
    done;
    let dmem_mapper_offset = desc.imem_load_size + !dmem_offset in
    set_field patched ~base:dmem_mapper_offset G.Falcon_application_interface_dmem_mapper_v3.init_cmd cmd_id;
    let cmd_in_buffer_offset =
      read_field image ~base:dmem_mapper_offset G.Falcon_application_interface_dmem_mapper_v3.cmd_in_buffer_offset
    in
    let cmd_off = desc.imem_load_size + cmd_in_buffer_offset in
    Bytes.blit cmd 0 patched cmd_off (Bytes.length cmd);
    let sig_off = desc.imem_load_size + desc.pkc_data_offset in
    Bytes.blit signature (Bytes.length signature - 0x180) patched sig_off 0x180;
    patched

  let prep_ucode ~rom ~vram_size =
    let expansion = expansion_rom_off rom in
    let desc_off, desc_size = fwsec_desc rom expansion in
    let desc = read_desc rom desc_off in
    let sig_total_size = desc_size - G.falcon_ucode_desc_v3_size_44 in
    let signature = Bytes.sub rom (desc_off + G.falcon_ucode_desc_v3_size_44) sig_total_size in
    let image = Bytes.sub rom (desc_off + desc_size) (round_up desc.stored_size 256) in
    (* The FRTS reservation is the top 1 MiB of the 2 MiB below the VGA
       workspace at the end of VRAM. *)
    let frts_offset = vram_size - 0x100000 - 0x100000 in
    let frts_image =
      patch_fwsec ~desc ~image ~signature ~cmd_id:0x15 ~cmd:(frts_cmd ~frts_offset)
    in
    { desc; frts_offset; frts_image }

  type booter = {
    image : bytes;
    data_off : int;
    data_sz : int;
    code_off : int;
    code_sz : int;
  }

  (* Parse a heavy-secured bootloader image: the nvfw container header
     points at a heavy-secure header, whose load header and app entry give
     the code and data spans, and whose patch points name where the
     production signature is spliced into the image. *)
  let prep_booter ~blob =
    let header_offset = read_field blob G.Nvfw_bin_hdr.header_offset in
    let data_offset = read_field blob G.Nvfw_bin_hdr.data_offset in
    let data_size = read_field blob G.Nvfw_bin_hdr.data_size in
    let hs_hdr_off = read_field blob ~base:header_offset G.Nvfw_hs_header_v2.header_offset in
    let patch_loc = u32 blob (read_field blob ~base:header_offset G.Nvfw_hs_header_v2.patch_loc) in
    let patch_sig = u32 blob (read_field blob ~base:header_offset G.Nvfw_hs_header_v2.patch_sig) in
    let num_sig = u32 blob (read_field blob ~base:header_offset G.Nvfw_hs_header_v2.num_sig) in
    let sig_prod_offset = read_field blob ~base:header_offset G.Nvfw_hs_header_v2.sig_prod_offset in
    let sig_prod_size = read_field blob ~base:header_offset G.Nvfw_hs_header_v2.sig_prod_size in
    let os_data_offset = read_field blob ~base:hs_hdr_off G.Nvfw_hs_load_header_v2.os_data_offset in
    let os_data_size = read_field blob ~base:hs_hdr_off G.Nvfw_hs_load_header_v2.os_data_size in
    let app_base = hs_hdr_off + G.Nvfw_hs_load_header_v2.sizeof in
    let app_offset = read_field blob ~base:app_base G.Nvfw_hs_load_header_v2_app.offset in
    let app_size = read_field blob ~base:app_base G.Nvfw_hs_load_header_v2_app.size in
    let sig_len = sig_prod_size / num_sig in
    let sig_off = sig_prod_offset + patch_sig in
    let image = Bytes.sub blob data_offset data_size in
    Bytes.blit blob sig_off image patch_loc sig_len;
    {
      image;
      data_off = os_data_offset;
      data_sz = os_data_size;
      code_off = app_offset;
      code_sz = app_size;
    }

  (* The prepared boot state produced by {!init_sw}: the parsed images
     and the device-local addresses they were loaded at. *)
  type prepared = {
    ucode : ucode;
    frts_image_paddr : int;
    booter_prep : booter;
    booter_image_paddr : int;
  }

  (* The falcon execution layer drives two microcontrollers at their
     fixed register-block bases: the GSP falcon and the SEC2 booter. *)
  type t = {
    nvdev : Nvdev.t;
    falcon : int;
    sec2 : int;
    mutable prepared : prepared option;
  }

  let create nvdev =
    { nvdev; falcon = 0x00110000; sec2 = 0x00840000; prepared = None }

  let falcon t = t.falcon
  let sec2 t = t.sec2

  let require_prepared t =
    match t.prepared with
    | Some p -> p
    | None -> invalid_arg "Flcn: init_sw must run before init_hw"

  (* ip.py:94 NV_FLCN.wait_for_reset: the boot-progress scratch reports
     the device is out of reset and secure boot has completed. *)
  let wait_for_reset t =
    wait_cond (dev_clock t.nvdev) ~value:1 ~msg:"waiting for reset" (fun () ->
        let plm =
          R.read_bitfields
            (Nvdev.reg t.nvdev
               "NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK")
        in
        let scratch =
          R.read
            (R.with_idx
               (Nvdev.reg t.nvdev "NV_PGC6_AON_SECURE_SCRATCH_GROUP_05")
               0)
        in
        if
          List.assoc "read_protection_level0" plm = 1
          && scratch land 0xff = 0xff
        then 1
        else 0)

  (* ip.py:271 reset: pulse the engine reset, wait for memory scrubbing,
     then bring the RISC-V core out of reset — either into bootloader
     fetch ([riscv]) or by selecting the falcon core and stamping the
     chip id. *)
  let reset t ?(riscv = false) base =
    let engine_reg =
      if base = t.falcon then Nvdev.reg t.nvdev "NV_PGSP_FALCON_ENGINE"
      else Nvdev.reg t.nvdev "NV_PSEC_FALCON_ENGINE"
    in
    R.write engine_reg [ ("reset", 1) ];
    sleep_ms t.nvdev 100;
    R.write engine_reg [ ("reset", 0) ];
    wait_cond (dev_clock t.nvdev) ~value:0 ~msg:"Scrubbing not completed"
      (fun () ->
        List.assoc "mem_scrubbing"
          (R.read_bitfields (based t.nvdev "NV_PFALCON_FALCON_HWCFG2" base)));
    if riscv then
      R.write
        (based t.nvdev "NV_PRISCV_RISCV_BCR_CTRL" base)
        [ ("core_select", 1); ("valid", 0); ("brfetch", 1) ]
    else if
      List.assoc "riscv"
        (R.read_bitfields (based t.nvdev "NV_PFALCON_FALCON_HWCFG2" base))
      = 1
    then begin
      R.write
        (based t.nvdev "NV_PRISCV_RISCV_BCR_CTRL" base)
        [ ("core_select", 0) ];
      wait_cond (dev_clock t.nvdev) ~value:1 ~msg:"RISCV core not booted"
        (fun () ->
          List.assoc "valid"
            (R.read_bitfields
               (based t.nvdev "NV_PRISCV_RISCV_BCR_CTRL" base)));
      R.write
        (based t.nvdev "NV_PFALCON_FALCON_RM" base)
        ~value:(Nvdev.chip_id t.nvdev) []
    end

  (* ip.py:267 disable_ctx_req: allow context-free physical DMA and clear
     the DMA control so transfers need no bound context. *)
  let disable_ctx_req t base =
    R.update
      (based t.nvdev "NV_PFALCON_FBIF_CTL" base)
      [ ("allow_phys_no_ctx", 1) ];
    R.write (based t.nvdev "NV_PFALCON_FALCON_DMACTL" base) ~value:0 []

  (* ip.py:229 start_cpu: start the core through its alias when the alias
     path is enabled, otherwise directly. *)
  let start_cpu t base =
    let cpuctl = based t.nvdev "NV_PFALCON_FALCON_CPUCTL" base in
    if List.assoc "alias_en" (R.read_bitfields cpuctl) = 1 then
      Nvdev.wreg t.nvdev
        (base + Nvdev.const t.nvdev "NV_PFALCON_FALCON_CPUCTL_ALIAS")
        0x2
    else R.write cpuctl [ ("startcpu", 1) ]

  (* ip.py:234 wait_cpu_halted *)
  let wait_cpu_halted t base =
    wait_cond (dev_clock t.nvdev) ~value:1 ~msg:"not halted" (fun () ->
        List.assoc "halted"
          (R.read_bitfields (based t.nvdev "NV_PFALCON_FALCON_CPUCTL" base)))

  (* ip.py:212 execute_dma: point the DMA base at the source, then walk it
     across the region in 256-byte transfers, waiting on the queue between
     each and for the engine to idle at the end. *)
  let execute_dma t base ~cmd ~dest ~mem_off ~src ~size =
    let poll_full () =
      wait_cond (dev_clock t.nvdev) ~value:0 ~msg:"DMA does not progress"
        (fun () ->
          List.assoc "full"
            (R.read_bitfields
               (based t.nvdev "NV_PFALCON_FALCON_DMATRFCMD" base)))
    in
    poll_full ();
    R.write
      (based t.nvdev "NV_PFALCON_FALCON_DMATRFBASE" base)
      ~value:(lo32 (src lsr 8)) [];
    R.write
      (based t.nvdev "NV_PFALCON_FALCON_DMATRFBASE1" base)
      ~value:(hi32 (src lsr 8) land 0x1ff) [];
    let xfered = ref 0 in
    while !xfered < size do
      poll_full ();
      R.write
        (based t.nvdev "NV_PFALCON_FALCON_DMATRFMOFFS" base)
        ~value:(dest + !xfered) [];
      R.write
        (based t.nvdev "NV_PFALCON_FALCON_DMATRFFBOFFS" base)
        ~value:(mem_off + !xfered) [];
      R.write (based t.nvdev "NV_PFALCON_FALCON_DMATRFCMD" base) ~value:cmd [];
      xfered := !xfered + 256
    done;
    wait_cond (dev_clock t.nvdev) ~value:1 ~msg:"DMA does not complete"
      (fun () ->
        List.assoc "idle"
          (R.read_bitfields
             (based t.nvdev "NV_PFALCON_FALCON_DMATRFCMD" base)))

  (* ip.py:236 execute_hs: load a heavy-secured image into the falcon's
     instruction and data memory over physical DMA, program the boot ROM
     with the signature parameters, and run the core to completion.
     Returns the mailbox pair when a [mailbox] seeds it, else [None]. *)
  let execute_hs t base ~img_paddr ~code_off ~data_off ~imem_pa ~imem_va
      ~imem_sz ~dmem_pa ~dmem_va ~dmem_sz ~pkc_off ~engid ~ucodeid ?mailbox () =
    disable_ctx_req t base;
    let ctx_dma = 0 in
    R.update
      (R.with_idx (based t.nvdev "NV_PFALCON_FBIF_TRANSCFG" base) ctx_dma)
      [
        ("target", 0);
        ( "mem_type",
          Nvdev.const t.nvdev "NV_PFALCON_FBIF_TRANSCFG_MEM_TYPE_PHYSICAL" );
      ];
    let dmatrfcmd = based t.nvdev "NV_PFALCON_FALCON_DMATRFCMD" base in
    let size_256b =
      Nvdev.const t.nvdev "NV_PFALCON_FALCON_DMATRFCMD_SIZE_256B"
    in
    let cmd =
      R.encode dmatrfcmd
        [
          ("write", 0); ("size", size_256b); ("ctxdma", ctx_dma); ("imem", 1);
          ("sec", 1);
        ]
    in
    execute_dma t base ~cmd ~dest:imem_pa ~mem_off:imem_va
      ~src:(img_paddr + code_off - imem_va) ~size:imem_sz;
    let cmd =
      R.encode dmatrfcmd
        [
          ("write", 0); ("size", size_256b); ("ctxdma", ctx_dma); ("imem", 0);
          ("sec", 0);
        ]
    in
    execute_dma t base ~cmd ~dest:dmem_pa ~mem_off:dmem_va
      ~src:(img_paddr + data_off - dmem_va) ~size:dmem_sz;
    R.write
      (R.with_idx (based t.nvdev "NV_PFALCON2_FALCON_BROM_PARAADDR" base) 0)
      ~value:pkc_off [];
    R.write
      (based t.nvdev "NV_PFALCON2_FALCON_BROM_ENGIDMASK" base)
      ~value:engid [];
    R.write
      (based t.nvdev "NV_PFALCON2_FALCON_BROM_CURR_UCODE_ID" base)
      [ ("val", ucodeid) ];
    R.write
      (based t.nvdev "NV_PFALCON2_FALCON_MOD_SEL" base)
      [ ("algo", Nvdev.const t.nvdev "NV_PFALCON2_FALCON_MOD_SEL_ALGO_RSA3K") ];
    R.write (based t.nvdev "NV_PFALCON_FALCON_BOOTVEC" base) ~value:imem_va [];
    (match mailbox with
    | Some mb ->
        R.write
          (based t.nvdev "NV_PFALCON_FALCON_MAILBOX0" base)
          ~value:(lo32 mb) [];
        R.write
          (based t.nvdev "NV_PFALCON_FALCON_MAILBOX1" base)
          ~value:(hi32 mb) []
    | None -> ());
    start_cpu t base;
    wait_cpu_halted t base;
    match mailbox with
    | Some _ ->
        Some
          ( R.read (based t.nvdev "NV_PFALCON_FALCON_MAILBOX0" base),
            R.read (based t.nvdev "NV_PFALCON_FALCON_MAILBOX1" base) )
    | None -> None

  (* ip.py:98 init_sw: resolve the falcon register families, read the
     VBIOS, and load the FWSEC ucode and the booter image into device
     memory, remembering where they landed. *)
  let init_sw t =
    let nvdev = t.nvdev in
    List.iter
      (fun (family, arch) -> Nvdev.include_regs nvdev ~family ~arch)
      [
        ("dev_gsp", "ga102"); ("dev_falcon_v4", "ga102");
        ("dev_riscv_pri", "ga102"); ("dev_fbif_v4", "ga102");
        ("dev_falcon_second_pri", "ga102"); ("dev_sec_pri", "ga102");
        ("dev_bus", "tu102");
      ];
    let load name image =
      match Nvdev.alloc_boot_mem nvdev ~data:image ~sysmem:false
              (Bytes.length image)
      with
      | _, Some paddr, _ -> paddr
      | _, None, _ ->
          failwith
            (Printf.sprintf "Flcn: %s image needs a device-local address" name)
    in
    let rom = read_vbios ~read32:(Nvdev.rreg nvdev) in
    let ucode = prep_ucode ~rom ~vram_size:(Nvdev.vram_size nvdev) in
    let frts_image_paddr = load "FRTS" ucode.frts_image in
    let booter_prep =
      prep_booter
        ~blob:
          (Firmware.fetch ~chip_dir:(Nvdev.fw_name nvdev)
             "booter_load-570.144.bin")
    in
    let booter_image_paddr = load "booter" booter_prep.image in
    t.prepared <-
      Some { ucode; frts_image_paddr; booter_prep; booter_image_paddr }

  (* ip.py:186 init_hw: run the FWSEC ucode to reserve the resident
     tables, restart the GSP falcon as RISC-V with the boot arguments in
     its mailbox, then run the booter to unlock the write-protected
     region, and confirm the GSP core is alive. *)
  let init_hw t ~libos_args_sysmem ~wpr_meta_sysmem =
    let p = require_prepared t in
    let nvdev = t.nvdev in
    let desc = p.ucode.desc in
    reset t t.falcon;
    ignore
      (execute_hs t t.falcon ~img_paddr:p.frts_image_paddr ~code_off:0x0
         ~data_off:desc.imem_load_size ~imem_pa:desc.imem_phys_base
         ~imem_va:desc.imem_virt_base ~imem_sz:desc.imem_load_size
         ~dmem_pa:desc.dmem_phys_base ~dmem_va:0x0 ~dmem_sz:desc.dmem_load_size
         ~pkc_off:desc.pkc_data_offset ~engid:desc.engine_id_mask
         ~ucodeid:desc.ucode_id ()
        : (int * int) option);
    if R.read (Nvdev.reg nvdev "NV_PFB_PRI_MMU_WPR2_ADDR_HI") = 0 then
      failwith "WPR2 is not initialized";
    reset t ~riscv:true t.falcon;
    R.write
      (Nvdev.reg nvdev "NV_PGSP_FALCON_MAILBOX0")
      ~value:(lo32 libos_args_sysmem) [];
    R.write
      (Nvdev.reg nvdev "NV_PGSP_FALCON_MAILBOX1")
      ~value:(hi32 libos_args_sysmem) [];
    reset t t.sec2;
    let mbx =
      execute_hs t t.sec2 ~img_paddr:p.booter_image_paddr
        ~code_off:p.booter_prep.code_off ~data_off:p.booter_prep.data_off
        ~imem_pa:0x0 ~imem_va:p.booter_prep.code_off
        ~imem_sz:p.booter_prep.code_sz ~dmem_pa:0x0 ~dmem_va:0x0
        ~dmem_sz:p.booter_prep.data_sz ~pkc_off:0x10 ~engid:1 ~ucodeid:3
        ~mailbox:wpr_meta_sysmem ()
    in
    (match mbx with
    | Some (m0, m1) when m0 <> 0 ->
        failwith
          (Printf.sprintf "Booter failed to execute, mailbox is %08x, %08x" m0
             m1)
    | _ -> ());
    R.write
      (R.with_base (Nvdev.reg nvdev "NV_PFALCON_FALCON_OS") t.falcon)
      ~value:0x0 [];
    if
      List.assoc "active_stat"
        (R.read_bitfields
           (R.with_base (Nvdev.reg nvdev "NV_PRISCV_RISCV_CPUCTL") t.falcon))
      <> 1
    then failwith "GSP Core is not active"
end

(* Chain-of-trust falcon boot image (Blackwell) *)

module Flcn_cot = struct
  type fmc = {
    image : bytes;
    hash : int array;
    signature : int array;
    public_key : int array;
  }

  (* The chain-of-trust firmware is an ELF whose named sections carry the
     bootable image and the three verification blobs. The hash, signature
     and public key are consumed as 32-bit words; the public key is padded
     so its final word is whole. *)
  let init_fmc_image ~blob =
    let elf = Tolk.Elf.load blob in
    let section name =
      match Tolk.Elf.find_section elf name with
      | Some (s : Tolk.Elf.section) -> s.content
      | None -> failwith (Printf.sprintf "FMC image missing %S section" name)
    in
    {
      image = section "image";
      hash = u32_array (section "hash");
      signature = u32_array (section "signature");
      public_key = u32_array (Bytes.cat (section "publickey") (Bytes.make 3 '\000'));
    }

  (* The prepared boot state produced by {!init_sw}: the chain-of-trust
     image and the addresses of the boot-argument region and the booter
     image in the memory the security processor reaches. *)
  type prepared = {
    fmc : fmc;
    fmc_boot_args_view : Mmio.t;
    fmc_boot_args_sysmem : int;
    fmc_booter_bar1 : int;
  }

  type t = { nvdev : Nvdev.t; falcon : int; mutable prepared : prepared option }

  let create nvdev = { nvdev; falcon = 0x00110000; prepared = None }

  let require_prepared t =
    match t.prepared with
    | Some p -> p
    | None -> invalid_arg "Flcn_cot: init_sw must run before init_hw"

  (* ip.py:286 NV_FLCN_COT.wait_for_reset: the thermal scratch reports the
     device is out of reset. *)
  let wait_for_reset t =
    Nvdev.include_regs t.nvdev ~family:"dev_therm" ~arch:"gb202";
    wait_cond (dev_clock t.nvdev) ~value:0xff ~msg:"waiting for reset"
      (fun () -> R.read (Nvdev.reg t.nvdev "NV_THERM_I2CS_SCRATCH"))

  (* ip.py:328 kfsp_send_msg: push a message to the security processor
     through its external memory window and wait for its reply, then drain
     the reply queue. *)
  let kfsp_send_msg t ~nvmd buf =
    let nvdev = t.nvdev in
    let headers = Bytes.create 8 in
    (* single-packet framing: start and end of message to seid 0, then
       the NVDM channel and message type. *)
    set_field headers (0, 4) ((1 lsl 31) lor (1 lsl 30));
    set_field headers (4, 4) (0x7e lor (0x10de lsl 8) lor (nvmd lsl 24));
    let body = Bytes.cat headers buf in
    let msg =
      Bytes.cat body (Bytes.make (4 - (Bytes.length body mod 4)) '\000')
    in
    if Bytes.length msg >= 0x400 then
      failwith
        (Printf.sprintf "FSP message too long: %d bytes, max 1024 bytes"
           (Bytes.length msg));
    let ememc = R.with_idx (Nvdev.reg nvdev "NV_PFSP_EMEMC") 0 in
    R.write ememc [ ("offs", 0); ("blk", 0); ("aincw", 1); ("aincr", 0) ];
    let ememd = R.with_idx (Nvdev.reg nvdev "NV_PFSP_EMEMD") 0 in
    let i = ref 0 in
    while !i < Bytes.length msg do
      R.write ememd ~value:(u32 msg !i) [];
      i := !i + 4
    done;
    R.write
      (R.with_idx (Nvdev.reg nvdev "NV_PFSP_QUEUE_TAIL") 0)
      ~value:(Bytes.length msg - 4) [];
    R.write (R.with_idx (Nvdev.reg nvdev "NV_PFSP_QUEUE_HEAD") 0) ~value:0 [];
    let msgq_head () =
      R.read (R.with_idx (Nvdev.reg nvdev "NV_PFSP_MSGQ_HEAD") 0)
    in
    let msgq_tail () =
      R.read (R.with_idx (Nvdev.reg nvdev "NV_PFSP_MSGQ_TAIL") 0)
    in
    wait_cond (dev_clock nvdev) ~value:1 ~msg:"FSP didn't respond to message"
      (fun () -> if msgq_head () <> msgq_tail () then 1 else 0);
    R.write ememc [ ("offs", 0); ("blk", 0); ("aincw", 0); ("aincr", 1) ];
    R.write
      (R.with_idx (Nvdev.reg nvdev "NV_PFSP_MSGQ_TAIL") 0)
      ~value:(msgq_head ()) []

  (* ip.py:290 init_sw: resolve the chain-of-trust register families,
     reserve the GSP boot-argument region, and load the chain-of-trust
     firmware image. *)
  let init_sw t =
    let nvdev = t.nvdev in
    List.iter
      (fun (family, arch) -> Nvdev.include_regs nvdev ~family ~arch)
      [
        ("dev_gsp", "ga102"); ("dev_falcon_v4", "gh100"); ("dev_vm", "gh100");
        ("dev_fsp_pri", "gh100"); ("dev_bus", "tu102");
      ];
    let fmc_boot_args_view, _, fmc_boot_addrs =
      Nvdev.alloc_boot_mem nvdev
        ~data:(Bytes.make G.Gsp_fmc_boot_params.sizeof '\000')
        G.Gsp_fmc_boot_params.sizeof
    in
    let fmc_boot_args_sysmem = List.hd fmc_boot_addrs in
    let fmc =
      init_fmc_image
        ~blob:
          (Firmware.fetch ~chip_dir:(Nvdev.fw_name nvdev) "fmc-570.144.bin")
    in
    let _, _, booter_addrs =
      Nvdev.alloc_boot_mem nvdev ~data:fmc.image (Bytes.length fmc.image)
    in
    let fmc_booter_bar1 = List.hd booter_addrs in
    t.prepared <-
      Some { fmc; fmc_boot_args_view; fmc_boot_args_sysmem; fmc_booter_bar1 }

  (* ip.py:311 init_hw: fill the boot-argument region with the GSP-RM and
     RM parameter blocks, build the chain-of-trust payload naming the
     boot arguments and the booter image, hand it to the security
     processor, and wait for the RISC-V boot lockdown to clear. *)
  let init_hw t ~libos_args_sysmem ~wpr_meta_sysmem =
    let p = require_prepared t in
    let nvdev = t.nvdev in
    let boot_args = Bytes.make G.Gsp_acr_boot_gsp_rm_params.sizeof '\000' in
    set_field boot_args G.Gsp_acr_boot_gsp_rm_params.gsprmdescoffset
      wpr_meta_sysmem;
    set_field boot_args G.Gsp_acr_boot_gsp_rm_params.gsprmdescsize
      G.Gsp_fw_wpr_meta.sizeof;
    set_field boot_args G.Gsp_acr_boot_gsp_rm_params.target
      G.gsp_dma_target_coherent_system;
    set_field boot_args G.Gsp_acr_boot_gsp_rm_params.bisgsprmboot 1;
    let rm_args = Bytes.make G.Gsp_rm_params.sizeof '\000' in
    set_field rm_args G.Gsp_rm_params.bootargsoffset libos_args_sysmem;
    set_field rm_args G.Gsp_rm_params.target G.gsp_dma_target_coherent_system;
    let fmc_params = Bytes.make G.Gsp_fmc_boot_params.sizeof '\000' in
    Bytes.blit boot_args 0 fmc_params
      (fst G.Gsp_fmc_boot_params.bootgsprmparams)
      G.Gsp_acr_boot_gsp_rm_params.sizeof;
    Bytes.blit rm_args 0 fmc_params
      (fst G.Gsp_fmc_boot_params.gsprmparams)
      G.Gsp_rm_params.sizeof;
    Mmio.blit_bytes p.fmc_boot_args_view ~off:0 fmc_params;
    let cot = Bytes.make G.Nvdm_payload_cot.sizeof '\000' in
    set_field cot G.Nvdm_payload_cot.version 0x2;
    set_field cot G.Nvdm_payload_cot.size G.Nvdm_payload_cot.sizeof;
    set_field cot G.Nvdm_payload_cot.frtsvidmemoffset 0x1c00000;
    set_field cot G.Nvdm_payload_cot.frtsvidmemsize 0x100000;
    set_field cot G.Nvdm_payload_cot.gspbootargssysmemoffset
      p.fmc_boot_args_sysmem;
    set_field cot G.Nvdm_payload_cot.gspfmcsysmemoffset p.fmc_booter_bar1;
    let put_words off elem_size xs =
      Array.iteri
        (fun i x -> set_field cot (off + (i * elem_size), elem_size) x)
        xs
    in
    put_words G.Nvdm_payload_cot.hash384_offset
      G.Nvdm_payload_cot.hash384_elem_size p.fmc.hash;
    put_words G.Nvdm_payload_cot.signature_offset
      G.Nvdm_payload_cot.signature_elem_size p.fmc.signature;
    put_words G.Nvdm_payload_cot.publickey_offset
      G.Nvdm_payload_cot.publickey_elem_size p.fmc.public_key;
    kfsp_send_msg t ~nvmd:G.nvdm_type_cot cot;
    wait_cond (dev_clock nvdev) ~value:0
      ~msg:"RISCV boot lockdown not cleared" (fun () ->
        List.assoc "riscv_br_priv_lockdown"
          (R.read_bitfields (based nvdev "NV_PFALCON_FALCON_HWCFG2" t.falcon)))
end

(* GSP firmware image and its page hierarchy *)

module Gsp = struct
  type radix3 = { npages : int array; offsets : int array; image_off : int }

  (* A three-level page hierarchy covering the GSP image: the deepest
     level maps the image's 4 KiB pages, and each level above holds one
     8-byte pointer per page of the level below, 512 (a 4 KiB page's
     worth) per directory page. [offsets] are the byte positions of each
     level within the hierarchy region, and [image_off] is where the image
     itself begins, right after the last directory level. *)
  let radix3 ~image_len =
    let npages = Array.make 4 0 in
    npages.(3) <- round_up image_len 0x1000 / 0x1000;
    for i = 3 downto 1 do
      npages.(i - 1) <-
        ((npages.(i) - 1) lsr (G.libos_memory_region_radix_page_log2 - 3)) + 1
    done;
    let offsets = Array.make 4 0 in
    for i = 1 to 3 do
      offsets.(i) <- offsets.(i - 1) + (npages.(i - 1) * 0x1000)
    done;
    { npages; offsets; image_off = offsets.(3) }

  type split = { image : bytes; signature : bytes }

  (* Split the GSP ELF into the firmware image and the per-chip signature
     section. The signature section is named for the chip family, e.g.
     [.fwsignature_ga10x]. *)
  let split_gsp_image ~blob ~chip_name =
    let elf = Tolk.Elf.load blob in
    let section name =
      match Tolk.Elf.find_section elf name with
      | Some (s : Tolk.Elf.section) -> s.content
      | None -> failwith (Printf.sprintf "GSP image missing %S section" name)
    in
    let family = String.lowercase_ascii (String.sub chip_name 0 4) in
    {
      image = section ".fwimage";
      signature = section (Printf.sprintf ".fwsignature_%sx" family);
    }

  type bootloader = {
    image : bytes;
    monitor_code_offset : int;
    monitor_data_offset : int;
    manifest_offset : int;
  }

  (* Parse the RISC-V bootloader container: the nvfw header points at the
     ucode descriptor giving the monitor code, data and manifest offsets,
     and the data span carries the bootloader image itself. *)
  let init_boot_binary_image ~blob =
    let header_offset = read_field blob G.Nvfw_bin_hdr.header_offset in
    let data_offset = read_field blob G.Nvfw_bin_hdr.data_offset in
    let data_size = read_field blob G.Nvfw_bin_hdr.data_size in
    {
      image = Bytes.sub blob data_offset data_size;
      monitor_code_offset = read_field blob ~base:header_offset G.Rm_riscv_ucode_desc.monitorcodeoffset;
      monitor_data_offset = read_field blob ~base:header_offset G.Rm_riscv_ucode_desc.monitordataoffset;
      manifest_offset = read_field blob ~base:header_offset G.Rm_riscv_ucode_desc.manifestoffset;
    }

  (* ip.py:618 NV_GSP.run_cpu_seq *)
  let run_cpu_seq ~rreg ~wreg ~now_ms ~sleep_us ~core_reset ~core_start
      ~core_wait_halted ~core_resume buf =
    let hdr_sz = G.Rpc_run_cpu_sequencer.sizeof in
    let cmd_index = read_field buf G.Rpc_run_cpu_sequencer.cmdindex in
    let words = min cmd_index ((Bytes.length buf - hdr_sz) / 4) in
    let save = Array.make G.Rpc_run_cpu_sequencer.regsavearea_count 0 in
    let pos = ref 0 in
    let next () =
      if !pos >= words then failwith "run_cpu_seq: truncated command stream";
      let v = u32 buf (hdr_sz + (!pos * 4)) in
      incr pos;
      v
    in
    while !pos < words do
      match next () with
      | 0x0 ->
          (* reg write *)
          let addr = next () in
          let v = next () in
          wreg addr v
      | 0x1 ->
          (* reg modify *)
          let addr = next () in
          let v = next () in
          let mask = next () in
          wreg addr ((rreg addr land lnot mask) lor (v land mask))
      | 0x2 ->
          (* reg poll; the trailing two words of the command are unused *)
          let addr = next () in
          let mask = next () in
          let v = next () in
          ignore (next ());
          ignore (next ());
          wait_cond now_ms ~value:v
            ~msg:
              (Printf.sprintf "Register %#x not equal to %#x after polling" addr
                 v)
            (fun () -> rreg addr land mask)
      | 0x3 -> sleep_us (next ()) (* delay us *)
      | 0x4 ->
          (* save reg *)
          let addr = next () in
          let index = next () in
          save.(index) <- rreg addr
      | 0x5 -> core_reset ()
      | 0x6 -> core_start ()
      | 0x7 -> core_wait_halted ()
      | 0x8 -> core_resume ()
      | op -> failwith (Printf.sprintf "Unknown op code %d in run_cpu_seq" op)
    done
end
