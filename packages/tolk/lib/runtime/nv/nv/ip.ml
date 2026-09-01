(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

module G = Nv_gsp_defs
module D = Nv_defs
module Mmio = Tolk_hcq.Hcq.Mmio

let debug = Tolk.Helpers.getenv "DEBUG" 0
let round_up n align = (n + align - 1) / align * align
let round_down n align = n / align * align
let ceildiv n d = (n + d - 1) / d

(* The number of bits needed to represent [n], so [bit_length (1 lsl k) = k + 1]. *)
let bit_length n =
  let rec go n acc = if n = 0 then acc else go (n lsr 1) (acc + 1) in
  go n 0

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
  let nvdev t = t.nvdev

  let require_prepared t =
    match t.prepared with
    | Some p -> p
    | None -> invalid_arg "Flcn: init_sw must run before init_hw"

  let frts_offset t = (require_prepared t).ucode.frts_offset

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

  (* The bytes of a driver parameter blob, for embedding into an RPC
     payload; the driver reads and writes these structures in place. *)
  let blob_bytes (b : Nv_tables.blob) =
    Bytes.init (Bigarray.Array1.dim b) (fun i -> Bigarray.Array1.get b i)

  (* A parameter blob holding [b], for RPCs that take one. *)
  let blob_of_bytes b =
    let blob = Nv_tables.create_blob (Bytes.length b) in
    for i = 0 to Bytes.length b - 1 do
      Bigarray.Array1.set blob i (Bytes.get b i)
    done;
    blob

  (* Fill an embedded NV_MEMORY_DESC_PARAMS region of a channel-allocation
     blob: the physical base, the byte size, the aperture ([2] is device
     memory), and the cache attributes. *)
  let set_mem_desc params ~region ~mem_base ~size ~aspace ~cache =
    let off = fst region in
    Nv_tables.set_field ~base:off params G.Nv_memory_desc_params.base mem_base;
    Nv_tables.set_field ~base:off params G.Nv_memory_desc_params.size size;
    Nv_tables.set_field ~base:off params G.Nv_memory_desc_params.addressspace
      aspace;
    Nv_tables.set_field ~base:off params G.Nv_memory_desc_params.cacheattrib
      cache

  (* ip.py:536 the channel-allocation memory descriptors: the RAM-FC and
     instance memory come from one contiguous page, the method buffer from
     boot memory, and — for a user channel with an error notifier — the
     error-notifier and user-D descriptors. [userd] is the user-D physical
     base when it applies, absent otherwise. *)
  let set_channel_gpfifo_descs ~params ~ramfc_paddr ~method_paddr ~userd =
    let module C = G.Nv_channelgpfifo_allocation_parameters in
    set_mem_desc params ~region:C.ramfcmem ~mem_base:ramfc_paddr ~size:0x200
      ~aspace:2 ~cache:0;
    set_mem_desc params ~region:C.instancemem ~mem_base:ramfc_paddr ~size:0x1000
      ~aspace:2 ~cache:0;
    set_mem_desc params ~region:C.mthdbufmem ~mem_base:method_paddr ~size:0x5000
      ~aspace:2 ~cache:0;
    match userd with
    | Some base ->
        set_mem_desc params ~region:C.errornotifiermem ~mem_base:0 ~size:0xecc
          ~aspace:0 ~cache:0;
        set_mem_desc params ~region:C.userdmem ~mem_base:base ~size:0x400
          ~aspace:2 ~cache:0
    | None -> ()

  (* ip.py:476 the reserved-PDE copy parameters: the server-reserved page
     directory entries covering [virt_addr_lo, virt_addr_hi], one level entry
     per page-table level [(physaddress, size, pageshift, aperture)]. *)
  let reserved_pdes_params ~page_size ~virt_addr_lo ~virt_addr_hi ~num_levels
      ~levels =
    let module P = G.Nv90f1_ctrl_vaspace_copy_server_reserved_pdes_params in
    let module L =
      G.Nv90f1_ctrl_vaspace_copy_server_reserved_pdes_params_level
    in
    let b = Bytes.make P.sizeof '\000' in
    set_field b P.pagesize page_size;
    set_field b P.virtaddrlo virt_addr_lo;
    set_field b P.virtaddrhi virt_addr_hi;
    set_field b P.numlevelstocopy num_levels;
    List.iteri
      (fun i (physaddress, size, pageshift, aperture) ->
        let base = P.levels_offset + (i * P.levels_elem_size) in
        set_field b ~base L.physaddress physaddress;
        set_field b ~base L.size size;
        set_field b ~base L.aperture aperture;
        set_field b ~base L.pageshift pageshift)
      levels;
    b

  type promote_entry = {
    buffer_id : int;
    gpu_virt_addr : int;
    gpu_phys_addr : int;
    size : int;
    phys_attr : int;
    b_initialize : bool;
    b_nonmapped : bool;
  }

  (* ip.py:457 the context-promotion parameters: one entry per graphics
     context buffer, binding it to the channel by physical and/or virtual
     address. *)
  let promote_ctx_params ~client ~obj ~entries =
    let module P = G.Nv2080_ctrl_gpu_promote_ctx_params in
    let module E = G.Nv2080_ctrl_gpu_promote_ctx_buffer_entry in
    let b = Bytes.make P.sizeof '\000' in
    set_field b P.entrycount (List.length entries);
    set_field b P.enginetype 0x1;
    set_field b P.hchanclient client;
    set_field b P.hobject obj;
    List.iteri
      (fun i e ->
        let base = P.promoteentry_offset + (i * P.promoteentry_elem_size) in
        set_field b ~base E.bufferid e.buffer_id;
        set_field b ~base E.gpuvirtaddr e.gpu_virt_addr;
        set_field b ~base E.binitialize (if e.b_initialize then 1 else 0);
        set_field b ~base E.gpuphysaddr e.gpu_phys_addr;
        set_field b ~base E.size e.size;
        set_field b ~base E.physattr e.phys_attr;
        set_field b ~base E.bnonmapped (if e.b_nonmapped then 1 else 0))
      entries;
    b

  (* ip.py:605 rpc_set_registry_table: the packed table of registry keys the
     GSP reads at boot. Each entry names its key by an offset into the
     trailing string blob. *)
  let registry_table () =
    let entries = [ ("RMForcePcieConfigSave", 0x1); ("RMSecBusResetEnable", 0x1) ] in
    let hdr_size = G.Packed_registry_table.sizeof in
    let entries_size = G.Packed_registry_entry.sizeof * List.length entries in
    let entries_buf = Buffer.create 64 and data_buf = Buffer.create 64 in
    List.iter
      (fun (k, v) ->
        let e = Bytes.make G.Packed_registry_entry.sizeof '\000' in
        set_field e G.Packed_registry_entry.nameoffset
          (hdr_size + entries_size + Buffer.length data_buf);
        set_field e G.Packed_registry_entry.typ G.registry_table_entry_type_dword;
        set_field e G.Packed_registry_entry.data v;
        set_field e G.Packed_registry_entry.length 4;
        Buffer.add_bytes entries_buf e;
        Buffer.add_string data_buf k;
        Buffer.add_char data_buf '\000')
      entries;
    let header = Bytes.make hdr_size '\000' in
    set_field header G.Packed_registry_table.size
      (hdr_size + Buffer.length entries_buf + Buffer.length data_buf);
    set_field header G.Packed_registry_table.numentries (List.length entries);
    Bytes.concat Bytes.empty
      [ header; Buffer.to_bytes entries_buf; Buffer.to_bytes data_buf ]

  (* ip.py:412 the radix3 region image: the GSP image copied in at its
     offset, and each directory level filled with the physical addresses of
     the pages of the level below. [page_addrs] are the region's own page
     physical addresses. *)
  let radix3_fill ~image ~(layout : radix3) ~page_addrs =
    let region = Bytes.make (layout.image_off + Bytes.length image) '\000' in
    Bytes.blit image 0 region layout.image_off (Bytes.length image);
    for i = 0 to 2 do
      let cur =
        Array.fold_left ( + ) 0 (Array.sub layout.npages 0 (i + 1))
      in
      for j = 0 to layout.npages.(i + 1) - 1 do
        set_field region (layout.offsets.(i) + (j * 8), 8) page_addrs.(cur + j)
      done
    done;
    region

  (* ip.py:433 init_wpr_meta: the write-protect-region metadata the booter
     reads to place the GSP image, heap and reserved tables in VRAM. The
     chain-of-trust boot uses a fixed reservation layout; the falcon boot
     packs the regions down from the top of VRAM and its reserved-tables
     offset must match the FWSEC ucode's. *)
  let wpr_meta ~vram_size ~fmc_boot ~radix3_addr ~radix3_size ~booter_addr
      ~booter_size ~signature_addr ~code_off ~data_off ~manifest_off
      ~frts_offset =
    let m = Bytes.make G.Gsp_fw_wpr_meta.sizeof '\000' in
    Bytes.set_int64_le m (fst G.Gsp_fw_wpr_meta.magic) G.gsp_fw_wpr_meta_magic;
    set_field m G.Gsp_fw_wpr_meta.revision G.gsp_fw_wpr_meta_revision;
    set_field m G.Gsp_fw_wpr_meta.sysmemaddrofradix3elf radix3_addr;
    set_field m G.Gsp_fw_wpr_meta.sizeofradix3elf radix3_size;
    set_field m G.Gsp_fw_wpr_meta.sysmemaddrofbootloader booter_addr;
    set_field m G.Gsp_fw_wpr_meta.sizeofbootloader booter_size;
    set_field m G.Gsp_fw_wpr_meta.bootloadercodeoffset code_off;
    set_field m G.Gsp_fw_wpr_meta.bootloaderdataoffset data_off;
    set_field m G.Gsp_fw_wpr_meta.bootloadermanifestoffset manifest_off;
    set_field m G.Gsp_fw_wpr_meta.sysmemaddrofsignature signature_addr;
    set_field m G.Gsp_fw_wpr_meta.sizeofsignature 0x1000;
    if fmc_boot then begin
      set_field m G.Gsp_fw_wpr_meta.vgaworkspacesize 0x20000;
      set_field m G.Gsp_fw_wpr_meta.pmureservedsize 0x1820000;
      set_field m G.Gsp_fw_wpr_meta.nonwprheapsize 0x220000;
      set_field m G.Gsp_fw_wpr_meta.gspfwheapsize 0x8700000;
      set_field m G.Gsp_fw_wpr_meta.frtssize 0x100000
    end
    else begin
      let vga_sz = 0x100000 in
      let vga_off = vram_size - vga_sz in
      let frts_sz = 0x100000 in
      let frts_off = vga_off - frts_sz in
      if frts_off <> frts_offset then
        failwith
          (Printf.sprintf "FRTS mismatch: %d != %d" frts_offset frts_off);
      let boot_off = frts_off - booter_size in
      let gsp_off = round_down (boot_off - radix3_size) 0x10000 in
      let gsp_heap_sz = 0x8100000 in
      let gsp_heap_off = round_down (gsp_off - gsp_heap_sz) 0x100000 in
      let wpr_st = round_down (gsp_heap_off - 0x1000) 0x100000 in
      let non_wpr_sz = 0x100000 in
      let non_wpr_off = round_down (wpr_st - non_wpr_sz) 0x100000 in
      set_field m G.Gsp_fw_wpr_meta.vgaworkspacesize vga_sz;
      set_field m G.Gsp_fw_wpr_meta.vgaworkspaceoffset vga_off;
      set_field m G.Gsp_fw_wpr_meta.gspfwwprend vga_off;
      set_field m G.Gsp_fw_wpr_meta.frtssize frts_sz;
      set_field m G.Gsp_fw_wpr_meta.frtsoffset frts_off;
      set_field m G.Gsp_fw_wpr_meta.bootbinoffset boot_off;
      set_field m G.Gsp_fw_wpr_meta.gspfwoffset gsp_off;
      set_field m G.Gsp_fw_wpr_meta.gspfwheapsize gsp_heap_sz;
      set_field m G.Gsp_fw_wpr_meta.fbsize vram_size;
      set_field m G.Gsp_fw_wpr_meta.gspfwheapoffset gsp_heap_off;
      set_field m G.Gsp_fw_wpr_meta.gspfwwprstart wpr_st;
      set_field m G.Gsp_fw_wpr_meta.nonwprheapsize non_wpr_sz;
      set_field m G.Gsp_fw_wpr_meta.nonwprheapoffset non_wpr_off;
      set_field m G.Gsp_fw_wpr_meta.gspfwrsvdstart non_wpr_off
    end;
    m

  (* ip.py:591 the BDF address the GSP system info carries, packed from the
     bus address string; a non-PCI (usb/remote) transport reports zero. *)
  let bdf_as_int s =
    let starts p = String.length s >= String.length p && String.sub s 0 (String.length p) = p in
    if starts "usb" || starts "remote" then 0
    else
      let hex a n = int_of_string ("0x" ^ String.sub s a n) in
      (hex 5 2 lsl 8) lor (hex 8 2 lsl 3)
      lor int_of_string ("0x" ^ String.make 1 s.[String.length s - 1])

  (* ip.py:594 GspSystemInfo: the host and PCI topology the GSP firmware
     needs, from the device's BARs and configuration space. *)
  let gsp_system_info ~gpu_phys ~gpu_phys_fb ~gpu_phys_inst ~fmc_boot ~bdf
      ~pci_device_id ~pci_subdevice_id ~pci_revision_id =
    let b = Bytes.make G.Gsp_system_info.sizeof '\000' in
    set_field b G.Gsp_system_info.gpuphysaddr gpu_phys;
    set_field b G.Gsp_system_info.gpuphysfbaddr gpu_phys_fb;
    set_field b G.Gsp_system_info.gpuphysinstaddr gpu_phys_inst;
    set_field b G.Gsp_system_info.pciconfigmirrorbase
      (if fmc_boot then 0x92000 else 0x88000);
    set_field b G.Gsp_system_info.pciconfigmirrorsize 0x1000;
    set_field b G.Gsp_system_info.nvdomainbusdevicefunc bdf;
    set_field b G.Gsp_system_info.bispassthru 1;
    set_field b G.Gsp_system_info.pcideviceid pci_device_id;
    set_field b G.Gsp_system_info.pcisubdeviceid pci_subdevice_id;
    set_field b G.Gsp_system_info.pcirevisionid pci_revision_id;
    set_field b G.Gsp_system_info.maxuserva 0x7ffffffff000;
    b

  (* ip.py:547 the GSP_RM_ALLOC payload: the allocation envelope naming the
     client, parent, new object handle and class, followed by the class's
     parameter structure. *)
  let rm_alloc_request ~client ~hparent ~hobject ~hclass ~params =
    let a = Bytes.make G.Rpc_gsp_rm_alloc.sizeof '\000' in
    set_field a G.Rpc_gsp_rm_alloc.hclient client;
    set_field a G.Rpc_gsp_rm_alloc.hparent hparent;
    set_field a G.Rpc_gsp_rm_alloc.hobject hobject;
    set_field a G.Rpc_gsp_rm_alloc.hclass hclass;
    set_field a G.Rpc_gsp_rm_alloc.flags 0;
    set_field a G.Rpc_gsp_rm_alloc.paramssize (Bytes.length params);
    Bytes.cat a params

  (* ip.py:572 the GSP_RM_CONTROL payload: the control envelope naming the
     client, object and command, followed by the command's parameter
     structure. *)
  let rm_control_request ~client ~hobject ~cmd ~params =
    let a = Bytes.make G.Rpc_gsp_rm_control.sizeof '\000' in
    set_field a G.Rpc_gsp_rm_control.hclient client;
    set_field a G.Rpc_gsp_rm_control.hobject hobject;
    set_field a G.Rpc_gsp_rm_control.cmd cmd;
    set_field a G.Rpc_gsp_rm_control.flags 0;
    set_field a G.Rpc_gsp_rm_control.paramssize (Bytes.length params);
    Bytes.cat a params

  (* ip.py:576 the driver's in-place update of a control's parameter
     structure: the response carries the envelope echo then the updated
     parameters, copied back into the caller's blob. GB20x parts need the
     work-submit enable bit patched into the returned token. *)
  let rm_control_apply ~chip_name ~cmd ~response ~(params : Nv_tables.blob) =
    let start = G.Rpc_gsp_rm_control.sizeof in
    for i = 0 to Bigarray.Array1.dim params - 1 do
      Bigarray.Array1.set params i (Bytes.get response (start + i))
    done;
    if
      String.length chip_name >= 3
      && String.sub chip_name 0 3 = "GB2"
      && cmd = D.nvc36f_ctrl_cmd_gpfifo_get_work_submit_token
    then begin
      let module W = D.Nvc36f_ctrl_cmd_gpfifo_get_work_submit_token_params in
      let tok = Nv_tables.get_field params W.worksubmittoken in
      Nv_tables.set_field params W.worksubmittoken (tok lor (1 lsl 30))
    end

  (* GSP boot state and execution *)

  (* The chip boots the GSP through the falcon bootloader or, on the
     newest parts, the chain-of-trust security processor. *)
  type boot = Falcon of Flcn.t | Cot of Flcn_cot.t

  type core_actions = {
    core_reset : unit -> unit;
    core_start : unit -> unit;
    core_wait_halted : unit -> unit;
    core_resume : unit -> unit;
  }

  (* ip.py:634 the boot actions the CPU sequencer drives during GSP boot,
     over the falcon primitives. Resume restarts the GSP falcon as RISC-V
     with the boot arguments in its mailbox, runs the SEC2 booter, and waits
     for the secure-boot handoff. *)
  let falcon_core_actions f ~libos_args_sysmem =
    let nvdev = Flcn.nvdev f in
    let falcon = Flcn.falcon f and sec2 = Flcn.sec2 f in
    {
      core_reset =
        (fun () ->
          Flcn.reset f falcon;
          Flcn.disable_ctx_req f falcon);
      core_start = (fun () -> Flcn.start_cpu f falcon);
      core_wait_halted = (fun () -> Flcn.wait_cpu_halted f falcon);
      core_resume =
        (fun () ->
          Flcn.reset f ~riscv:true falcon;
          R.write
            (Nvdev.reg nvdev "NV_PGSP_FALCON_MAILBOX0")
            ~value:(lo32 libos_args_sysmem) [];
          R.write
            (Nvdev.reg nvdev "NV_PGSP_FALCON_MAILBOX1")
            ~value:(hi32 libos_args_sysmem) [];
          Flcn.start_cpu f sec2;
          wait_cond (dev_clock nvdev) ~value:1 ~msg:"SEC2 didn't hand off"
            (fun () ->
              List.assoc "boot_stage_3_handoff"
                (R.read_bitfields
                   (Nvdev.reg nvdev "NV_PGC6_BSI_SECURE_SCRATCH_14")));
          let mailbox =
            R.read (based nvdev "NV_PFALCON_FALCON_MAILBOX0" sec2)
          in
          if mailbox <> 0 then
            failwith
              (Printf.sprintf "Falcon SEC2 failed to execute, mailbox is %08x"
                 mailbox));
    }

  (* A graphics context buffer: its byte size and whether it is promoted by
     physical address, virtual address, or both; [local] buffers are per
     channel and excluded from the golden-image promotion. *)
  type grbuf = { size : int; virt : bool; phys : bool; local : bool }

  type t = {
    nvdev : Nvdev.t;
    boot : boot;
    mutable handle_gen : int;
    mutable cmd_q : Rpc_queue.t option;
    mutable stat_q : Rpc_queue.t option;
    mutable cmd_q_view : Mmio.t option;
    mutable stat_q_view : Mmio.t option;
    mutable rm_args_sysmem : int;
    mutable libos_args_sysmem : int;
    mutable wpr_meta_sysmem : int;
    mutable gpfifo_class : int;
    mutable compute_class : int;
    mutable dma_class : int;
    mutable priv_root : int;
    mutable device : int;
    mutable subdevice : int;
    mutable grctx_bufs : (int * grbuf) list;
  }

  let create nvdev ~boot =
    {
      nvdev;
      boot;
      handle_gen = 0xcf000000;
      cmd_q = None;
      stat_q = None;
      cmd_q_view = None;
      stat_q_view = None;
      rm_args_sysmem = 0;
      libos_args_sysmem = 0;
      wpr_meta_sysmem = 0;
      gpfifo_class = 0;
      compute_class = 0;
      dma_class = 0;
      priv_root = 0;
      device = 0;
      subdevice = 0;
      grctx_bufs = [];
    }

  let falcon_boot f = Falcon f
  let cot_boot c = Cot c
  let libos_args_sysmem t = t.libos_args_sysmem
  let wpr_meta_sysmem t = t.wpr_meta_sysmem
  let gpfifo_class t = t.gpfifo_class
  let compute_class t = t.compute_class
  let dma_class t = t.dma_class
  let next_handle t = let h = t.handle_gen in t.handle_gen <- h + 1; h
  let cmd_q t = Option.get t.cmd_q
  let stat_q t = Option.get t.stat_q

  (* ip.py:55 the doorbell that signals the GSP a record was published. *)
  let notify t () =
    R.write (R.with_idx (Nvdev.reg t.nvdev "NV_PGSP_QUEUE_HEAD") 0) ~value:0 []

  (* The CPU-sequencer callback: the GSP sends register command streams
     during boot, interpreted over the device and the falcon boot actions.
     The chain-of-trust boot processor never drives the core actions, so
     they raise there. *)
  let on_run_cpu_seq t buf =
    let nvdev = t.nvdev in
    let ca =
      match t.boot with
      | Falcon f -> falcon_core_actions f ~libos_args_sysmem:t.libos_args_sysmem
      | Cot _ ->
          let unexpected () =
            failwith "Gsp: cpu-sequencer core action on the chain-of-trust boot"
          in
          {
            core_reset = unexpected;
            core_start = unexpected;
            core_wait_halted = unexpected;
            core_resume = unexpected;
          }
    in
    run_cpu_seq ~rreg:(Nvdev.rreg nvdev) ~wreg:(Nvdev.wreg nvdev)
      ~now_ms:(fun () -> Nvdev.now_ms nvdev)
      ~sleep_us:(fun us -> sleep_ms nvdev (us / 1000))
      ~core_reset:ca.core_reset ~core_start:ca.core_start
      ~core_wait_halted:ca.core_wait_halted ~core_resume:ca.core_resume buf

  (* ip.py:363 init_rm_args: allocate the shared command and status queues in
     system memory, publish their page table and the RM arguments, write the
     command queue's transmit header, and bring the command queue up. *)
  let init_rm_args t =
    let nvdev = t.nvdev in
    let queue_size = 0x40000 in
    let queue_pte_cnt = queue_size * 2 / 0x1000 in
    let pte_cnt = queue_pte_cnt + (round_up (queue_pte_cnt * 8) 0x1000 / 0x1000) in
    let pt_size = round_up (pte_cnt * 8) 0x1000 in
    let queues_view, _, queues_sysmem =
      Nvdev.alloc_boot_mem nvdev ~sysmem:true (pt_size + (queue_size * 2))
    in
    List.iteri
      (fun i sysmem -> Mmio.write64 queues_view (i * 8) (Int64.of_int sysmem))
      queues_sysmem;
    let queue_args = Bytes.make G.Message_queue_init_arguments.sizeof '\000' in
    set_field queue_args G.Message_queue_init_arguments.sharedmemphysaddr
      (List.hd queues_sysmem);
    set_field queue_args G.Message_queue_init_arguments.pagetableentrycount
      pte_cnt;
    set_field queue_args G.Message_queue_init_arguments.cmdqueueoffset pt_size;
    set_field queue_args G.Message_queue_init_arguments.statqueueoffset
      (pt_size + queue_size);
    let cached = Bytes.make G.Gsp_arguments_cached.sizeof '\000' in
    set_field cached G.Gsp_arguments_cached.bdmemstack 1;
    Bytes.blit queue_args 0 cached
      (fst G.Gsp_arguments_cached.messagequeueinitarguments)
      G.Message_queue_init_arguments.sizeof;
    let _, _, rm_args_addrs =
      Nvdev.alloc_boot_mem nvdev ~data:cached G.Gsp_arguments_cached.sizeof
    in
    t.rm_args_sysmem <- List.hd rm_args_addrs;
    let cmd_q_view = Mmio.view queues_view ~off:pt_size () in
    let stat_q_view = Mmio.view queues_view ~off:(pt_size + queue_size) () in
    t.cmd_q_view <- Some cmd_q_view;
    t.stat_q_view <- Some stat_q_view;
    let hdr = Bytes.make G.Msgq_tx_header.sizeof '\000' in
    set_field hdr G.Msgq_tx_header.version 0;
    set_field hdr G.Msgq_tx_header.size queue_size;
    set_field hdr G.Msgq_tx_header.entryoff 0x1000;
    set_field hdr G.Msgq_tx_header.msgsize 0x1000;
    set_field hdr G.Msgq_tx_header.msgcount ((queue_size - 0x1000) / 0x1000);
    set_field hdr G.Msgq_tx_header.writeptr 0;
    set_field hdr G.Msgq_tx_header.flags 1;
    set_field hdr G.Msgq_tx_header.rxhdroff G.Msgq_tx_header.sizeof;
    Mmio.blit_bytes cmd_q_view ~off:0 hdr;
    t.cmd_q <-
      Some
        (Rpc_queue.make ~devfmt:(Nvdev.devfmt nvdev) ~notify:(notify t)
           ~on_run_cpu_seq:(on_run_cpu_seq t)
           ~on_error:(fun () -> Nvdev.set_err_state nvdev true)
           cmd_q_view)

  (* ip.py:388 init_libos_args: reserve the GSP log buffers and describe them,
     and the RM arguments, as libos memory regions keyed by an 8-byte name. *)
  let be_int s = String.fold_left (fun acc c -> (acc * 256) + Char.code c) 0 s

  let init_libos_args t =
    let nvdev = t.nvdev in
    let _, _, logbuf_addrs = Nvdev.alloc_boot_mem nvdev (2 lsl 20) in
    let libos_args_view, _, libos_addrs = Nvdev.alloc_boot_mem nvdev 0x1000 in
    t.libos_args_sysmem <- List.hd libos_addrs;
    let logbuf0 = List.hd logbuf_addrs in
    let region ~size ~id8 ~pa =
      let b = Bytes.make G.Libos_memory_region_init_argument.sizeof '\000' in
      set_field b G.Libos_memory_region_init_argument.kind
        G.libos_memory_region_contiguous;
      set_field b G.Libos_memory_region_init_argument.loc
        G.libos_memory_region_loc_sysmem;
      set_field b G.Libos_memory_region_init_argument.size size;
      set_field b G.Libos_memory_region_init_argument.id8 id8;
      set_field b G.Libos_memory_region_init_argument.pa pa;
      b
    in
    let logs =
      List.mapi
        (fun i name ->
          region ~size:0x10000
            ~id8:(be_int ("LOG" ^ name))
            ~pa:(logbuf0 + (0x10000 * i)))
        [ "INIT"; "INTR"; "RM"; "MNOC"; "KRNL" ]
    in
    let rmargs =
      region ~size:0x1000 ~id8:(be_int "RMARGS") ~pa:t.rm_args_sysmem
    in
    Mmio.blit_bytes libos_args_view ~off:0
      (Bytes.concat Bytes.empty (logs @ [ rmargs ]))

  (* ip.py:433 init_wpr_meta: load the GSP firmware image behind its page
     hierarchy and the RISC-V bootloader, then build the write-protect-region
     metadata that places them in VRAM. *)
  let init_wpr_meta t =
    let nvdev = t.nvdev in
    let split =
      split_gsp_image
        ~blob:(Firmware.fetch ~chip_dir:"ga102" "gsp-570.144.bin")
        ~chip_name:(Nvdev.chip_name nvdev)
    in
    let layout = radix3 ~image_len:(Bytes.length split.image) in
    let region_size = layout.image_off + Bytes.length split.image in
    let radix_view, _, radix_addrs = Nvdev.alloc_boot_mem nvdev region_size in
    Mmio.blit_bytes radix_view ~off:0
      (radix3_fill ~image:split.image ~layout
         ~page_addrs:(Array.of_list radix_addrs));
    let _, _, sig_addrs =
      Nvdev.alloc_boot_mem nvdev ~data:split.signature
        (Bytes.length split.signature)
    in
    let bl =
      init_boot_binary_image
        ~blob:(Firmware.fetch ~chip_dir:(Nvdev.fw_name nvdev) "bootloader-570.144.bin")
    in
    let _, _, bl_addrs =
      Nvdev.alloc_boot_mem nvdev ~data:bl.image (Bytes.length bl.image)
    in
    let frts_offset =
      match t.boot with Falcon f -> Flcn.frts_offset f | Cot _ -> 0
    in
    let meta =
      wpr_meta ~vram_size:(Nvdev.vram_size nvdev)
        ~fmc_boot:(Nvdev.fmc_boot nvdev) ~radix3_addr:(List.hd radix_addrs)
        ~radix3_size:(Bytes.length split.image) ~booter_addr:(List.hd bl_addrs)
        ~booter_size:(Bytes.length bl.image) ~signature_addr:(List.hd sig_addrs)
        ~code_off:bl.monitor_code_offset ~data_off:bl.monitor_data_offset
        ~manifest_off:bl.manifest_offset ~frts_offset
    in
    let _, _, wpr_addrs =
      Nvdev.alloc_boot_mem nvdev ~data:meta G.Gsp_fw_wpr_meta.sizeof
    in
    t.wpr_meta_sysmem <- List.hd wpr_addrs

  (* ### RPCs *)

  (* ip.py:590 rpc_set_gsp_system_info: prefill the command queue with the
     host and PCI topology the GSP reads at boot. *)
  let rpc_set_gsp_system_info t =
    let nvdev = t.nvdev in
    let pci =
      match Nvdev.pci_dev nvdev with
      | Some p -> p
      | None -> failwith "Gsp: rpc_set_gsp_system_info needs a PCI device"
    in
    let bar b = fst (Tolk_hcq.System.Pci_device.bar_info pci b) in
    let cfg ~offset ~size =
      Tolk_hcq.System.Pci_device.read_config pci ~offset ~size
    in
    let data =
      gsp_system_info ~gpu_phys:(bar 0) ~gpu_phys_fb:(bar 1)
        ~gpu_phys_inst:(bar 3) ~fmc_boot:(Nvdev.fmc_boot nvdev)
        ~bdf:(bdf_as_int (Nvdev.devfmt nvdev))
        ~pci_device_id:(cfg ~offset:0x0 ~size:4)
        ~pci_subdevice_id:(cfg ~offset:0x2c ~size:4)
        ~pci_revision_id:(cfg ~offset:0x8 ~size:1)
    in
    Rpc_queue.send_rpc (cmd_q t) G.nv_vgpu_msg_function_gsp_set_system_info data

  (* ip.py:605 rpc_set_registry_table: prefill the command queue with the
     registry keys the GSP applies at boot. *)
  let rpc_set_registry_table t =
    Rpc_queue.send_rpc (cmd_q t) G.nv_vgpu_msg_function_set_registry
      (registry_table ())

  (* ip.py:583 rpc_set_page_directory: hand the GSP the root page-table
     address for a user virtual-address space. *)
  let rpc_set_page_directory t ~device ~hvaspace ~pdir_paddr ~client
      ?(pasid = 0xffffffff) () =
    let params =
      Bytes.make G.Nv0080_ctrl_dma_set_page_directory_params.sizeof '\000'
    in
    set_field params G.Nv0080_ctrl_dma_set_page_directory_params.physaddress
      pdir_paddr;
    set_field params G.Nv0080_ctrl_dma_set_page_directory_params.numentries
      (Tolk.Memory.pte_cnt (Nvdev.mm t.nvdev) 0);
    set_field params G.Nv0080_ctrl_dma_set_page_directory_params.flags 0x8;
    set_field params G.Nv0080_ctrl_dma_set_page_directory_params.hvaspace hvaspace;
    set_field params G.Nv0080_ctrl_dma_set_page_directory_params.pasid pasid;
    set_field params G.Nv0080_ctrl_dma_set_page_directory_params.subdeviceid 1;
    set_field params G.Nv0080_ctrl_dma_set_page_directory_params.chid 0;
    let args = Bytes.make G.Rpc_set_page_directory.sizeof '\000' in
    set_field args G.Rpc_set_page_directory.hclient client;
    set_field args G.Rpc_set_page_directory.hdevice device;
    set_field args G.Rpc_set_page_directory.pasid pasid;
    Bytes.blit params 0 args
      (fst G.Rpc_set_page_directory.params)
      G.Nv0080_ctrl_dma_set_page_directory_params.sizeof;
    Rpc_queue.send_rpc (cmd_q t) G.nv_vgpu_msg_function_set_page_directory args;
    ignore
      (Rpc_queue.wait_resp (stat_q t) G.nv_vgpu_msg_function_set_page_directory)

  (* ip.py:561 rpc_rm_control: invoke a control command on an object and copy
     the driver's in-place parameter update back into the caller's blob. *)
  let rpc_rm_control t ~hobject ~cmd ?params ~client () =
    let params_bytes =
      match params with Some p -> blob_bytes p | None -> Bytes.empty
    in
    Rpc_queue.send_rpc (cmd_q t) G.nv_vgpu_msg_function_gsp_rm_control
      (rm_control_request ~client ~hobject ~cmd ~params:params_bytes);
    let res =
      Rpc_queue.wait_resp (stat_q t) G.nv_vgpu_msg_function_gsp_rm_control
    in
    match params with
    | Some p ->
        rm_control_apply ~chip_name:(Nvdev.chip_name t.nvdev) ~cmd ~response:res
          ~params:p
    | None -> ()

  (* ip.py:456 promote_ctx: promote a set of graphics context buffers to a
     channel. Each buffer is allocated (unless one is supplied in [bufs]) and
     bound by physical and/or virtual address; the allocations are returned so
     a second pass can bind the same memory the other way. *)
  let promote_ctx t ~client ~subdevice ~obj ~ctxbufs ?bufs ?virt ?phys () =
    let mm = Nvdev.mm t.nvdev in
    let res = ref [] in
    let entries =
      List.map
        (fun (buf, desc) ->
          let use_v = match virt with Some v -> v | None -> desc.virt in
          let use_p = match phys with Some p -> p | None -> desc.phys in
          let x =
            match bufs with
            | Some bs when List.mem_assoc buf bs -> List.assoc buf bs
            | _ -> Tolk.Memory.valloc mm desc.size ~contiguous:true ()
          in
          res := (buf, x) :: !res;
          {
            buffer_id = buf;
            gpu_virt_addr = (if use_v then x.Tolk.Memory.va_addr else 0);
            gpu_phys_addr =
              (if use_p then fst (List.hd x.Tolk.Memory.paddrs) else 0);
            size = (if use_p then desc.size else 0);
            phys_attr = (if use_p then 0x4 else 0);
            b_initialize = use_p;
            b_nonmapped = use_p && not use_v;
          })
        ctxbufs
    in
    rpc_rm_control t ~hobject:subdevice ~cmd:G.nv2080_ctrl_cmd_gpu_promote_ctx
      ~params:(blob_of_bytes (promote_ctx_params ~client ~obj ~entries))
      ~client ();
    List.rev !res

  (* ip.py:534 rpc_rm_alloc: allocate a driver object under a parent and is
     its handle. A channel allocation gets its memory descriptors filled; a
     user address space gets its page directory set; and a user compute object
     promotes the graphics context to its channel. *)
  let rpc_rm_alloc t ~hparent ~hclass ?params ~client () =
    (if hclass = t.gpfifo_class then
       match params with
       | None ->
           failwith "Gsp: a channel allocation needs its parameter structure"
       | Some p ->
           let module C = G.Nv_channelgpfifo_allocation_parameters in
           let ramfc = Tolk.Memory.valloc (Nvdev.mm t.nvdev) 0x1000 ~contiguous:true () in
           let ramfc_paddr = fst (List.hd ramfc.Tolk.Memory.paddrs) in
           let method_paddr =
             match Nvdev.alloc_boot_mem t.nvdev ~sysmem:false 0x5000 with
             | _, Some paddr, _ -> paddr
             | _, None, _ ->
                 failwith "Gsp: the channel method buffer needs device memory"
           in
           let userd =
             if client <> t.priv_root && Nv_tables.get_field p C.hobjecterror <> 0
             then
               Some
                 (Nv_tables.get_field p
                    (C.huserdmemory_offset, C.huserdmemory_elem_size)
                 + Nv_tables.get_field p
                     (C.userdoffset_offset, C.userdoffset_elem_size))
             else None
           in
           set_channel_gpfifo_descs ~params:p ~ramfc_paddr ~method_paddr ~userd);
    let obj = next_handle t in
    let params_bytes =
      match params with Some p -> blob_bytes p | None -> Bytes.empty
    in
    Rpc_queue.send_rpc (cmd_q t) G.nv_vgpu_msg_function_gsp_rm_alloc
      (rm_alloc_request ~client ~hparent ~hobject:obj ~hclass ~params:params_bytes);
    ignore (Rpc_queue.wait_resp (stat_q t) G.nv_vgpu_msg_function_gsp_rm_alloc);
    if hclass = D.fermi_vaspace_a && client <> t.priv_root then
      rpc_set_page_directory t ~device:hparent ~hvaspace:obj
        ~pdir_paddr:
          (Nvdev.Nv_page_table.paddr
             (Tolk.Memory.root_page_table (Nvdev.mm t.nvdev)))
        ~client ();
    if hclass = D.nv01_device_0 && client <> t.priv_root then t.device <- obj;
    if hclass = D.nv20_subdevice_0 then t.subdevice <- obj;
    if hclass = t.compute_class && client <> t.priv_root then begin
      let bufs012 =
        List.filter (fun (b, _) -> List.mem b [ 0; 1; 2 ]) t.grctx_bufs
      in
      let phys_gr_ctx =
        promote_ctx t ~client ~subdevice:t.subdevice ~obj:hparent
          ~ctxbufs:bufs012 ~virt:false ()
      in
      ignore
        (promote_ctx t ~client ~subdevice:t.subdevice ~obj:hparent
           ~ctxbufs:bufs012 ~bufs:phys_gr_ctx ~phys:false ())
    end;
    if hclass = D.nv01_root then client else obj

  (* ip.py:600 rpc_unloading_guest_driver: tell the GSP the driver is
     unloading, for a fast device shutdown. *)
  let rpc_unloading_guest_driver t =
    let data = Bytes.make G.Rpc_unloading_guest_driver.sizeof '\000' in
    set_field data G.Rpc_unloading_guest_driver.binpmtransition 0;
    set_field data G.Rpc_unloading_guest_driver.bgc6entering 0;
    set_field data G.Rpc_unloading_guest_driver.newlevel (1 lsl 6);
    Rpc_queue.send_rpc (cmd_q t)
      G.nv_vgpu_msg_function_unloading_guest_driver data;
    ignore
      (Rpc_queue.wait_resp (stat_q t)
         G.nv_vgpu_msg_function_unloading_guest_driver)

  (* ip.py:347 init_sw: build the shared queues and boot arguments, prefill
     the system info and registry, and select the engine classes for the
     chip. *)
  let init_sw t =
    init_rm_args t;
    init_libos_args t;
    init_wpr_meta t;
    rpc_set_gsp_system_info t;
    rpc_set_registry_table t;
    let chip = Nvdev.chip_name t.nvdev in
    let prefix = if String.length chip >= 2 then String.sub chip 0 2 else chip in
    let g, c, d =
      match prefix with
      | "AD" -> (D.ampere_channel_gpfifo_a, D.ada_compute_a, D.ampere_dma_copy_b)
      | "GB" ->
          ( D.blackwell_channel_gpfifo_a,
            D.blackwell_compute_b,
            D.blackwell_dma_copy_b )
      | _ ->
          (D.ampere_channel_gpfifo_a, D.ampere_compute_b, D.ampere_dma_copy_b)
    in
    t.gpfifo_class <- g;
    t.compute_class <- c;
    t.dma_class <- d

  (* ip.py:467 init_golden_image: allocate a privileged client, device,
     subdevice and address space, copy the server-reserved page directory
     entries into it, allocate a channel, size the graphics context buffers
     from the static engine info, and promote them — recording the buffer
     descriptors so a user compute object can promote the same set. *)
  let init_golden_image t =
    let nvdev = t.nvdev in
    let mm = Nvdev.mm nvdev in
    ignore
      (rpc_rm_alloc t ~hparent:0x0 ~hclass:0x0
         ~params:(Nv_tables.create_blob D.Nv0000_alloc_parameters.sizeof)
         ~client:t.priv_root ());
    let dev_params = Nv_tables.create_blob D.Nv0080_alloc_parameters.sizeof in
    Nv_tables.set_field dev_params D.Nv0080_alloc_parameters.hclientshare
      t.priv_root;
    let dev =
      rpc_rm_alloc t ~hparent:t.priv_root ~hclass:D.nv01_device_0
        ~params:dev_params ~client:t.priv_root ()
    in
    let subdev =
      rpc_rm_alloc t ~hparent:dev ~hclass:D.nv20_subdevice_0
        ~params:(Nv_tables.create_blob D.Nv2080_alloc_parameters.sizeof)
        ~client:t.priv_root ()
    in
    let vp = (Nv_tables.defs_for_driver ~major:570).nv_vaspace_allocation_parameters in
    let vaspace =
      rpc_rm_alloc t ~hparent:dev ~hclass:D.fermi_vaspace_a
        ~params:(Nv_tables.create_blob vp.sizeof) ~client:t.priv_root ()
    in
    (* reserve 512 MiB for the server-reserved page directory entries *)
    let res_sz = 512 lsl 20 in
    let res_va = Tolk.Memory.alloc_vaddr mm res_sz () in
    let levels =
      List.mapi
        (fun i pt ->
          ( Nvdev.Nv_page_table.paddr pt,
            (if i = 0 then Tolk.Memory.pte_cnt mm 0 * 8 else 0x1000),
            bit_length (Tolk.Memory.pte_covers mm i) - 1,
            1 ))
        (Tolk.Memory.page_tables mm ~vaddr:res_va ~size:res_sz)
    in
    rpc_rm_control t ~hobject:vaspace
      ~cmd:G.nv90f1_ctrl_cmd_vaspace_copy_server_reserved_pdes
      ~params:
        (blob_of_bytes
           (reserved_pdes_params ~page_size:res_sz ~virt_addr_lo:res_va
              ~virt_addr_hi:(res_va + res_sz - 1) ~num_levels:3 ~levels))
      ~client:t.priv_root ();
    (* the golden-image channel *)
    let module C = G.Nv_channelgpfifo_allocation_parameters in
    let gpfifo_area = Tolk.Memory.valloc mm (4 lsl 10) ~contiguous:true () in
    let gpfifo_paddr = fst (List.hd gpfifo_area.Tolk.Memory.paddrs) in
    let gg = Nv_tables.create_blob C.sizeof in
    Nv_tables.set_field gg C.gpfifooffset gpfifo_area.Tolk.Memory.va_addr;
    Nv_tables.set_field gg C.gpfifoentries 32;
    Nv_tables.set_field gg C.enginetype 0x1;
    Nv_tables.set_field gg C.cid 3;
    Nv_tables.set_field gg C.hvaspace vaspace;
    Nv_tables.set_field gg (C.userdoffset_offset, C.userdoffset_elem_size)
      (0x20 * 8);
    set_mem_desc gg ~region:C.userdmem ~mem_base:(gpfifo_paddr + (0x20 * 8))
      ~size:0x20 ~aspace:2 ~cache:0;
    Nv_tables.set_field gg C.internalflags 0x1a;
    Nv_tables.set_field gg C.flags 0x200320;
    let ch_gpfifo =
      rpc_rm_alloc t ~hparent:dev ~hclass:t.gpfifo_class ~params:gg
        ~client:t.priv_root ()
    in
    (* size the graphics context buffers from the static engine info *)
    let module IP =
      G.Nv2080_ctrl_internal_static_kgr_get_context_buffers_info_params
    in
    let module CBI = G.Nv2080_ctrl_internal_static_gr_context_buffers_info in
    let module EBI = G.Nv2080_ctrl_internal_engine_context_buffer_info in
    let cbi = Nv_tables.create_blob IP.sizeof in
    rpc_rm_control t ~hobject:subdev
      ~cmd:G.nv2080_ctrl_cmd_internal_static_kgr_get_context_buffers_info
      ~params:cbi ~client:t.priv_root ();
    let engine_base idx =
      IP.enginecontextbuffersinfo_offset + CBI.engine_offset
      + (idx * CBI.engine_elem_size)
    in
    let ctx_info ?(add = 0) ?align idx =
      round_up
        (Nv_tables.get_field ~base:(engine_base idx) cbi EBI.size + add)
        (match align with
        | Some a -> a
        | None -> Nv_tables.get_field ~base:(engine_base idx) cbi EBI.alignment)
    in
    let gr_size =
      ctx_info ~add:0x40000
        G.nv0080_ctrl_fifo_get_engine_context_properties_engine_id_graphics
    in
    let patch_size =
      ctx_info
        G.nv0080_ctrl_fifo_get_engine_context_properties_engine_id_graphics_patch
    in
    (* indices 3–10 map to engine rows 17–24; index 5 aligns to 2 MiB *)
    let cfg x =
      ctx_info ?align:(if x = 5 then Some (2 lsl 20) else None) (x + 14)
    in
    let grctx =
      [
        (0, { size = gr_size; phys = true; virt = true; local = false });
        (1, { size = patch_size; phys = true; virt = true; local = true });
        (2, { size = patch_size; phys = true; virt = true; local = false });
        (3, { size = cfg 3; phys = false; virt = true; local = false });
        (4, { size = cfg 4; phys = false; virt = true; local = false });
        (5, { size = cfg 5; phys = false; virt = true; local = false });
        (6, { size = cfg 6; phys = false; virt = true; local = false });
        (9, { size = cfg 9; phys = true; virt = true; local = false });
        (10, { size = cfg 10; phys = true; virt = false; local = false });
        (11, { size = cfg 10; phys = true; virt = true; local = false });
      ]
    in
    t.grctx_bufs <- grctx;
    ignore
      (promote_ctx t ~client:t.priv_root ~subdevice:subdev ~obj:ch_gpfifo
         ~ctxbufs:(List.filter (fun (_, v) -> not v.local) grctx)
         ());
    ignore
      (rpc_rm_alloc t ~hparent:ch_gpfifo ~hclass:t.compute_class
         ~client:t.priv_root ());
    ignore
      (rpc_rm_alloc t ~hparent:ch_gpfifo ~hclass:t.dma_class ~client:t.priv_root
         ())

  (* ip.py:506 init_hw: bring the status queue up, wait for the GSP to report
     initialization done, program the BAR1 block, and initialize the golden
     image. *)
  let init_hw t =
    let nvdev = t.nvdev in
    let cmd_q_view = Option.get t.cmd_q_view
    and stat_q_view = Option.get t.stat_q_view in
    let stat_q =
      Rpc_queue.make ~completion_q_view:cmd_q_view ~devfmt:(Nvdev.devfmt nvdev)
        ~notify:(notify t) ~on_run_cpu_seq:(on_run_cpu_seq t)
        ~on_error:(fun () -> Nvdev.set_err_state nvdev true)
        stat_q_view
    in
    t.stat_q <- Some stat_q;
    Rpc_queue.set_rx_view (cmd_q t)
      (Mmio.view stat_q_view ~off:(Rpc_queue.rx_hdr_off stat_q) ());
    ignore (Rpc_queue.wait_resp stat_q G.nv_vgpu_msg_event_gsp_init_done);
    R.write
      (Nvdev.reg nvdev "NV_PBUS_BAR1_BLOCK")
      [ ("mode", 0); ("target", 0); ("ptr", 0) ];
    if Nvdev.fmc_boot nvdev then
      R.write
        (Nvdev.reg nvdev "NV_VIRTUAL_FUNCTION_PRIV_FUNC_BAR1_BLOCK_LOW_ADDR")
        [ ("mode", 0); ("target", 0); ("ptr", 0) ];
    t.priv_root <- 0xc1e00004;
    init_golden_image t

  (* ip.py:518 fini_hw: unload the guest driver for a fast shutdown. *)
  let fini_hw t = rpc_unloading_guest_driver t

  (* Consume any pending status-queue messages, dispatching their events;
     a fault event latches the device error state. *)
  let drain_responses t = Seq.iter (fun _ -> ()) (Rpc_queue.read_resp (stat_q t))
end
