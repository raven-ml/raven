(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

module Hcq = Tolk_hcq.Hcq
module Nv_tables = Nv_tables
module Defs = Nv_defs
module Q = Hcq.Q

let lo32 v = Int64.to_int (Int64.logand v 0xFFFFFFFFL)
let hi32 v = Int64.to_int (Int64.shift_right_logical v 32)
let va64 = Int64.of_nativeint
let round_up n align = (n + align - 1) / align * align

(* Method-argument values sit at the bit range their register describes. *)
let bits (_hi, lo) v = v lsl lo

(* Raw stores into a mapped descriptor. The narrow stores reject values
   that do not fit rather than truncating them. *)
let write_u32 view off v =
  if v < 0 || v > 0xffffffff then
    invalid_arg (Printf.sprintf "value 0x%x does not fit in 32 bits" v);
  Hcq.Mmio.write32 view off (Int32.of_int v)

let write_u16 view off v =
  if v < 0 || v > 0xffff then
    invalid_arg (Printf.sprintf "value 0x%x does not fit in 16 bits" v);
  let b = Bytes.create 2 in
  Bytes.set_uint16_le b 0 v;
  Hcq.Mmio.blit_bytes view ~off b

let write_u8 view off v =
  if v < 0 || v > 0xff then
    invalid_arg (Printf.sprintf "value 0x%x does not fit in 8 bits" v);
  Hcq.Mmio.blit_bytes view ~off (Bytes.make 1 (Char.chr v))

(* Launch descriptors *)

module Qmd = struct
  type t = {
    ver : int;
    size : int;
    view : Hcq.Mmio.t;
    fields : (string, int * int) Hashtbl.t;
  }

  let table entries =
    let h = Hashtbl.create 512 in
    List.iter (fun (name, range) -> Hashtbl.replace h name range) entries;
    h

  let fields_v3 = lazy (table Nv_defs.nvc6c0_qmdv03_00_fields)
  let fields_v5 = lazy (table Nv_defs.nvcec0_qmdv05_00_fields)

  let layout ~compute_class =
    if compute_class >= Nv_defs.blackwell_compute_a then (5, 0x180, fields_v5)
    else (3, 0x100, fields_v3)

  let sizeof ~compute_class =
    let _, size, _ = layout ~compute_class in
    size

  let create ~view ~compute_class =
    let ver, size, fields = layout ~compute_class in
    if Hcq.Mmio.size view < size then
      invalid_arg
        (Printf.sprintf "Qmd.create: view holds %d bytes, the descriptor %d"
           (Hcq.Mmio.size view) size);
    { ver; size; view; fields = Lazy.force fields }

  let version t = t.ver

  let range t name =
    match Hashtbl.find_opt t.fields (String.uppercase_ascii name) with
    | Some r -> r
    | None -> invalid_arg ("Qmd: unknown field " ^ name)

  let field_offset t name = snd (range t name) / 8

  (* A field is at most 32 bits wide at an arbitrary bit offset, so it spans
     at most five bytes of the descriptor. *)
  let read t name =
    let hi, lo = range t name in
    let first = lo / 8 in
    let b = Hcq.Mmio.read_bytes t.view ~off:first ~len:((hi / 8) - first + 1) in
    let num = ref 0 in
    for i = Bytes.length b - 1 downto 0 do
      num := (!num lsl 8) lor Char.code (Bytes.unsafe_get b i)
    done;
    (!num lsr (lo mod 8)) land ((1 lsl (hi - lo + 1)) - 1)

  let write_field t name v =
    let hi, lo = range t name in
    let width = hi - lo + 1 in
    if v < 0 || v lsr width > 0 then
      invalid_arg (Printf.sprintf "Qmd: 0x%x does not fit in field %s" v name);
    let first = lo / 8 in
    let len = (hi / 8) - first + 1 in
    let b = Hcq.Mmio.read_bytes t.view ~off:first ~len in
    let num = ref 0 in
    for i = len - 1 downto 0 do
      num := (!num lsl 8) lor Char.code (Bytes.unsafe_get b i)
    done;
    let mask = ((1 lsl width) - 1) lsl (lo mod 8) in
    let num = !num land lnot mask lor (v lsl (lo mod 8)) in
    for i = 0 to len - 1 do
      Bytes.unsafe_set b i (Char.unsafe_chr ((num lsr (8 * i)) land 0xff))
    done;
    Hcq.Mmio.blit_bytes t.view ~off:first b

  let write t fields = List.iter (fun (name, v) -> write_field t name v) fields

  let set_constant_buf_addr t i addr =
    let a = va64 addr in
    if t.ver < 4 then
      write t
        [
          (Printf.sprintf "constant_buffer_addr_upper_%d" i, hi32 a);
          (Printf.sprintf "constant_buffer_addr_lower_%d" i, lo32 a);
        ]
    else
      let a = Int64.shift_right_logical a 6 in
      write t
        [
          (Printf.sprintf "constant_buffer_addr_upper_shifted6_%d" i, hi32 a);
          (Printf.sprintf "constant_buffer_addr_lower_shifted6_%d" i, lo32 a);
        ]

  let to_bytes t = Hcq.Mmio.read_bytes t.view ~off:0 ~len:t.size
end

(* Devices *)

type 'meta device = {
  compute_class : int;
  dma_class : int;
  gpfifo_class : int;
  sass_version : int;
  mutable slm_per_thread : int;
  shared_mem_window : nativeint;
  local_mem_window : nativeint;
  cmdq_page : 'meta Hcq.Buffer.t;
  cmdq_allocator : Tolk.Bump.t;
  cmdq : Hcq.Mmio.t;
  gpu_mmio : Hcq.Mmio.t;
}

let device ~compute_class ~dma_class ~gpfifo_class ~sass_version
    ?(slm_per_thread = 0) ~shared_mem_window ~local_mem_window ~cmdq_page
    ~gpu_mmio () =
  {
    compute_class;
    dma_class;
    gpfifo_class;
    sass_version;
    slm_per_thread;
    shared_mem_window;
    local_mem_window;
    cmdq_page;
    cmdq_allocator =
      Tolk.Bump.create
        ~size:(Hcq.Buffer.size cmdq_page)
        ~base:(Nativeint.to_int (Hcq.Buffer.va cmdq_page))
        ~wrap:true ();
    cmdq = Hcq.Buffer.cpu_view cmdq_page;
    gpu_mmio;
  }

(* Programs *)

type 'meta program = { dev : 'meta device; qmd : Qmd.t; cbuf0_size : int }

(* Queue descriptors *)

module Queue_desc = struct
  type t = {
    ring : Hcq.Mmio.t;
    gpput : Hcq.Mmio.t;
    token : int;
    mutable put_value : int;
  }
end

(* Method streams. A type-2 header names an engine subchannel and a method
   id, carries its argument count, and auto-increments the method for each
   argument dword that follows. *)
let nvm q subchannel mthd args =
  Q.push q
    ((2 lsl 28)
    lor (Array.length args lsl 16)
    lor (subchannel lsl 13) lor (mthd lsr 2));
  for i = 0 to Array.length args - 1 do
    Q.push q (Array.unsafe_get args i)
  done

(* The host-class semaphore methods work on any engine's channel. *)
let push_sem_wait q ~addr ~value =
  let a = va64 addr and v = Int64.of_int value in
  nvm q 0 Defs.nvc56f_sem_addr_lo
    [|
      lo32 a;
      hi32 a;
      lo32 v;
      hi32 v;
      bits Defs.nvc56f_sem_execute_operation
        Defs.nvc56f_sem_execute_operation_acq_circ_geq
      lor bits Defs.nvc56f_sem_execute_payload_size
            Defs.nvc56f_sem_execute_payload_size_64bit;
    |]

(* Submission: stage the stream in the device's command buffer, point the
   next ring entry at it, publish the new put position, and only then ring
   the work-submission doorbell, so the device never fetches a stale
   entry. *)
let submit_to_gpfifo (dev : 'meta device) q (qd : Queue_desc.t) =
  let n = Q.length q in
  let cmdq_addr = Tolk.Bump.alloc dev.cmdq_allocator (n * 4) ~align:16 () in
  let base = cmdq_addr - Nativeint.to_int (Hcq.Buffer.va dev.cmdq_page) in
  for i = 0 to n - 1 do
    Hcq.Mmio.write32 dev.cmdq (base + (i * 4)) (Int32.of_int (Q.get q i))
  done;
  let entries = Hcq.Mmio.size qd.ring / 8 in
  Hcq.Mmio.write64 qd.ring
    (qd.put_value mod entries * 8)
    (Int64.of_int (((cmdq_addr / 4) lsl 2) lor (n lsl 42) lor (1 lsl 41)));
  Hcq.Mmio.write32 qd.gpput 0 (Int32.of_int ((qd.put_value + 1) mod entries));
  Hcq.Mmio.fence ();
  Hcq.Mmio.write32 dev.gpu_mmio 0x90 (Int32.of_int qd.token);
  qd.put_value <- qd.put_value + 1

(* Compute queue *)

module Compute_queue = struct
  type 'meta t = {
    dev : 'meta device;
    q : Q.t;
    mutable active_qmd : Qmd.t option;
  }

  let create dev = { dev; q = Q.create (); active_qmd = None }
  let q t = t.q

  let setup t ?compute_class ?local_mem_window ?shared_mem_window ?local_mem
      ?local_mem_tpc_bytes () =
    Option.iter
      (fun c -> nvm t.q 1 Defs.nvc6c0_set_object [| c |])
      compute_class;
    Option.iter
      (fun w ->
        let w = va64 w in
        nvm t.q 1 Defs.nvc6c0_set_shader_local_memory_window_a
          [| hi32 w; lo32 w |])
      local_mem_window;
    Option.iter
      (fun w ->
        let w = va64 w in
        nvm t.q 1 Defs.nvc6c0_set_shader_shared_memory_window_a
          [| hi32 w; lo32 w |])
      shared_mem_window;
    Option.iter
      (fun a ->
        let a = va64 a in
        nvm t.q 1 Defs.nvc6c0_set_shader_local_memory_a [| hi32 a; lo32 a |])
      local_mem;
    Option.iter
      (fun b ->
        let b = Int64.of_int b in
        nvm t.q 1 Defs.nvc6c0_set_shader_local_memory_non_throttled_a
          [| hi32 b; lo32 b; 0xff |])
      local_mem_tpc_bytes

  let wait t ?(value = 0) sg =
    push_sem_wait t.q ~addr:(Hcq.Signal.value_addr sg) ~value;
    t.active_qmd <- None

  let memory_barrier t =
    nvm t.q 1 Defs.nvc6c0_invalidate_shader_caches_no_wfi
      [|
        bits Defs.nvc6c0_invalidate_shader_caches_no_wfi_instruction
          Defs.nvc6c0_invalidate_shader_caches_no_wfi_instruction_true
        lor bits Defs.nvc6c0_invalidate_shader_caches_no_wfi_global_data
              Defs.nvc6c0_invalidate_shader_caches_no_wfi_global_data_true
        lor bits Defs.nvc6c0_invalidate_shader_caches_no_wfi_constant
              Defs.nvc6c0_invalidate_shader_caches_no_wfi_constant_true;
      |];
    t.active_qmd <- None

  let exec t (prg : 'meta program) ~kernargs ~global_size:(gx, gy, gz)
      ~local_size:(lx, ly, lz) =
    let compute_class = t.dev.compute_class in
    let qmd_buf =
      Hcq.Buffer.offset kernargs
        ~off:(round_up prg.cbuf0_size 256)
        ~size:(Qmd.sizeof ~compute_class) ()
    in
    let view = Hcq.Buffer.cpu_view qmd_buf in
    Hcq.Mmio.blit_bytes view ~off:0 (Qmd.to_bytes prg.qmd);
    let va = va64 (Hcq.Buffer.va qmd_buf) in
    (* the launch methods and the dependent pointer carry the descriptor
       address shifted right by 8, in 32 bits *)
    if Int64.shift_right_logical va 40 <> 0L then
      invalid_arg
        (Printf.sprintf
           "Compute_queue.exec: descriptor address 0x%Lx does not fit in 40 \
            bits" va);
    let qmd = Qmd.create ~view ~compute_class in
    (* the geometry stores are whole-word writes at the fields' offsets;
       the extra bits they cover are reserved, except after the third
       block dimension, hence its single-byte store *)
    let grid_off =
      Qmd.field_offset qmd
        (if Qmd.version qmd < 4 then "cta_raster_width" else "grid_width")
    in
    write_u32 view grid_off gx;
    write_u32 view (grid_off + 4) gy;
    write_u32 view (grid_off + 8) gz;
    let dim_off = Qmd.field_offset qmd "cta_thread_dimension0" in
    write_u16 view dim_off lx;
    write_u16 view (dim_off + 2) ly;
    write_u8 view (Qmd.field_offset qmd "cta_thread_dimension2") lz;
    Qmd.set_constant_buf_addr qmd 0 (Hcq.Buffer.va kernargs);
    let ptr = Int64.to_int (Int64.shift_right_logical va 8) in
    (match t.active_qmd with
    | None ->
        nvm t.q 1 Defs.nvc6c0_send_pcas_a [| ptr |];
        nvm t.q 1 Defs.nvc6c0_send_signaling_pcas2_b [| 9 |]
    | Some prev ->
        Qmd.write prev
          [
            ("dependent_qmd0_pointer", ptr);
            ("dependent_qmd0_action", 1);
            ("dependent_qmd0_prefetch", 1);
            ("dependent_qmd0_enable", 1);
          ]);
    t.active_qmd <- Some qmd

  let signal t ?(value = 0) sg =
    let patched =
      match t.active_qmd with
      | None -> false
      | Some qmd ->
          let v3 = Qmd.version qmd < 4 in
          let view = qmd.Qmd.view in
          let rec claim i =
            if i > 1 then false
            else if Qmd.read qmd (Printf.sprintf "release%d_enable" i) <> 0
            then claim (i + 1)
            else begin
              Qmd.write qmd [ (Printf.sprintf "release%d_enable" i, 1) ];
              let addr_off =
                Qmd.field_offset qmd
                  (if v3 then Printf.sprintf "release%d_address_lower" i
                   else Printf.sprintf "release_semaphore%d_addr_lower" i)
              in
              let a = va64 (Hcq.Signal.value_addr sg) in
              Hcq.Mmio.write32 view addr_off (Int32.of_int (lo32 a));
              (* the top address bits share their word with other release
                 fields, the enable bit included: touch only the low
                 nibble *)
              let upper =
                Int32.to_int (Hcq.Mmio.read32 view (addr_off + 4))
                land 0xffffffff
              in
              Hcq.Mmio.write32 view (addr_off + 4)
                (Int32.of_int (upper land lnot 0xf lor hi32 a));
              let val_off =
                Qmd.field_offset qmd
                  (if v3 then Printf.sprintf "release%d_payload_lower" i
                   else Printf.sprintf "release_semaphore%d_payload_lower" i)
              in
              let v = Int64.of_int value in
              Hcq.Mmio.write32 view val_off (Int32.of_int (lo32 v));
              Hcq.Mmio.write32 view (val_off + 4) (Int32.of_int (hi32 v));
              true
            end
          in
          claim 0
    in
    if not patched then begin
      let a = va64 (Hcq.Signal.value_addr sg) and v = Int64.of_int value in
      nvm t.q 0 Defs.nvc56f_sem_addr_lo
        [|
          lo32 a;
          hi32 a;
          lo32 v;
          hi32 v;
          bits Defs.nvc56f_sem_execute_operation
            Defs.nvc56f_sem_execute_operation_release
          lor bits Defs.nvc56f_sem_execute_release_wfi
                Defs.nvc56f_sem_execute_release_wfi_en
          lor bits Defs.nvc56f_sem_execute_payload_size
                Defs.nvc56f_sem_execute_payload_size_64bit
          lor bits Defs.nvc56f_sem_execute_release_timestamp
                Defs.nvc56f_sem_execute_release_timestamp_en;
        |];
      nvm t.q 0 Defs.nvc56f_non_stall_interrupt [| 0 |];
      t.active_qmd <- None
    end

  let timestamp t sg = signal t ~value:0 sg

  let write t ?(b64 = false) buf value =
    let a = va64 (Hcq.Buffer.va buf) in
    nvm t.q 0 Defs.nvc56f_sem_addr_lo
      [|
        lo32 a;
        hi32 a;
        lo32 value;
        hi32 value;
        bits Defs.nvc56f_sem_execute_operation
          Defs.nvc56f_sem_execute_operation_release
        lor bits Defs.nvc56f_sem_execute_release_wfi
              Defs.nvc56f_sem_execute_release_wfi_en
        lor bits Defs.nvc56f_sem_execute_payload_size
              (if b64 then Defs.nvc56f_sem_execute_payload_size_64bit
               else Defs.nvc56f_sem_execute_payload_size_32bit);
      |];
    t.active_qmd <- None

  let poll_bit t buf ~value ~mask =
    let a = va64 (Hcq.Buffer.va buf) in
    let payload =
      Int64.of_int (if value = 0 then lnot mask land 0xffffffff else value)
    in
    nvm t.q 0 Defs.nvc56f_sem_addr_lo
      [|
        lo32 a;
        hi32 a;
        lo32 payload;
        hi32 payload;
        bits Defs.nvc56f_sem_execute_operation
          (if value = 0 then Defs.nvc56f_sem_execute_operation_acq_nor
           else Defs.nvc56f_sem_execute_operation_acq_and)
        lor bits Defs.nvc56f_sem_execute_payload_size
              Defs.nvc56f_sem_execute_payload_size_32bit;
      |];
    t.active_qmd <- None

  let submit t qd = submit_to_gpfifo t.dev t.q qd
end

(* Copy queue *)

module Copy_queue = struct
  type 'meta t = { dev : 'meta device; q : Q.t }

  let create dev = { dev; q = Q.create () }
  let q t = t.q

  let setup t ?copy_class () =
    Option.iter (fun c -> nvm t.q 4 Defs.nvc6c0_set_object [| c |]) copy_class

  let copy t ~dest ~src size =
    (* one transfer moves at most 2 GiB; larger copies split into chunks *)
    let step = 1 lsl 31 in
    let off = ref 0 in
    while !off < size do
      let s = Int64.add (va64 (Hcq.Buffer.va src)) (Int64.of_int !off) in
      let d = Int64.add (va64 (Hcq.Buffer.va dest)) (Int64.of_int !off) in
      nvm t.q 4 Defs.nvc6b5_offset_in_upper [| hi32 s; lo32 s; hi32 d; lo32 d |];
      nvm t.q 4 Defs.nvc6b5_line_length_in [| min (size - !off) step |];
      nvm t.q 4 Defs.nvc6b5_launch_dma
        [|
          bits Defs.nvc6b5_launch_dma_data_transfer_type
            Defs.nvc6b5_launch_dma_data_transfer_type_non_pipelined
          lor bits Defs.nvc6b5_launch_dma_src_memory_layout
                Defs.nvc6b5_launch_dma_src_memory_layout_pitch
          lor bits Defs.nvc6b5_launch_dma_dst_memory_layout
                Defs.nvc6b5_launch_dma_dst_memory_layout_pitch;
        |];
      off := !off + step
    done

  let signal t ?(value = 0) sg =
    let a = va64 (Hcq.Signal.value_addr sg) in
    nvm t.q 4 Defs.nvc6b5_set_semaphore_a [| hi32 a; lo32 a; value |];
    nvm t.q 4 Defs.nvc6b5_launch_dma
      [|
        bits Defs.nvc6b5_launch_dma_flush_enable
          Defs.nvc6b5_launch_dma_flush_enable_true
        lor bits Defs.nvc6b5_launch_dma_semaphore_type
              Defs.nvc6b5_launch_dma_semaphore_type_release_four_word_semaphore;
      |]

  let timestamp t sg = signal t ~value:0 sg

  let wait t ?(value = 0) sg =
    push_sem_wait t.q ~addr:(Hcq.Signal.value_addr sg) ~value

  let submit t qd = submit_to_gpfifo t.dev t.q qd
end
