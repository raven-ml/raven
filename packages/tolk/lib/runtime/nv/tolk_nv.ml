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
  mutable shader_local_mem : 'meta Hcq.Buffer.t option;
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
    shader_local_mem = None;
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

(* Driver interface seam *)

module Nv_iface = struct
  exception Out_of_memory of string

  type mem = { h_memory : int; owner_id : int }
  type nvdev = ..

  type usermode = {
    handle : int;
    mmio : Hcq.Mmio.t;
    compute_class : int;
    dma_class : int;
    gpfifo_class : int;
  }

  type t = {
    root : int;
    gpu_instance : int;
    count : int;
    set_device : nvdevice:int -> subdevice:int -> virtmem:int -> unit;
    rm_alloc : parent:int -> cls:int -> ?params:Nv_tables.blob -> unit -> int;
    rm_control : obj:int -> cmd:int -> ?params:Nv_tables.blob -> unit -> unit;
    alloc :
      ?host:bool ->
      ?uncached:bool ->
      ?cpu_access:bool ->
      ?contiguous:bool ->
      ?map_flags:int ->
      ?cpu_addr:nativeint ->
      int ->
      mem Hcq.Buffer.t;
    free : mem Hcq.Buffer.t -> unit;
    map : mem Hcq.Buffer.t -> mem Hcq.Buffer.t;
    setup_usermode : unit -> usermode;
    setup_vm : vaspace:int -> unit;
    setup_gpfifo_vm : gpfifo:int -> unit;
    sleep : int -> unit;
    device_fini : unit -> unit;
    nvdev : nvdev option;
  }

  let is_nvd t = t.nvdev <> None
end

(* Kernel-driver interface *)

module Nvk_iface = struct
  module File_io = Hcq.File_io

  type gpu = { gpu_id : int; minor_number : int }

  type state = {
    fd_ctl : int;
    fd_uvm : int;
    fd_uvm_2 : int;
    root : int;
    defs : Nv_defs_versions.t;
    gpus_info : gpu array;
  }

  type t = {
    device_id : int;
    fd_dev : int;
    gpu_minor : int;
    gpu_instance : int;
    mutable nvdevice : int;
    mutable subdevice : int;
    mutable virtmem : int;
    mutable gpu_uuid : bytes;
  }

  (* Driver-wide state shared by every device in the process: the control
     and memory-manager file descriptors, the root client, the installed
     driver's parameter-structure generation, and the visible cards. *)
  let state : state option ref = ref None

  (* Host objects take handles from this private enumerator, so a handle
     at or below its mark is one of ours rather than the driver's. *)
  let host_object_enumerator = ref 0x1000

  (* The 48-bit device virtual address space splits at 0x2000000000: the
     64 GiB below are reserved for CPU-visible mappings, everything above
     holds device-only ranges. Addresses are process-global and never
     reused. *)
  let low_uvm_vaddr_allocator =
    Tolk.Bump.create ~size:0x1000000000 ~base:0x1000000000 ~wrap:false ()

  let uvm_vaddr_allocator =
    Tolk.Bump.create ~size:((1 lsl 48) - 1) ~base:0x2000000000 ~wrap:false ()

  let alloc_gpu_vaddr ?(alignment = 4 lsl 10) ?(force_low = false) size =
    Nativeint.of_int
      (Tolk.Bump.alloc
         (if force_low then low_uvm_vaddr_allocator else uvm_vaddr_allocator)
         size ~align:alignment ())

  let error_str defs status =
    Printf.sprintf "%d: %s" status
      (match
         List.assoc_opt status defs.Nv_defs_versions.nv_status_codes
       with
      | Some name -> name
      | None -> "Unknown error")

  (* The address of a nested parameter blob travels through another blob
     as a raw integer, invisible to the garbage collector: pin the nested
     blob until after the driver call. *)
  let keep_alive b = ignore (Sys.opaque_identity b)

  let blit_bytes b ~off src =
    for i = 0 to Bytes.length src - 1 do
      Bigarray.Array1.set b (off + i) (Bytes.get src i)
    done

  let read_bytes b ~off ~len =
    Bytes.init len (fun i -> Bigarray.Array1.get b (off + i))

  (* Escape calls travel under request codes that embed the parameter
     size; a nonzero return is a transport failure, the driver status
     arrives inside the blob. *)
  let escape fd ~nr b =
    let request = Nv_tables.escape_code ~nr ~size:(Bigarray.Array1.dim b) in
    let r = Nv_tables.ioctl ~fd ~request b in
    if r <> 0 then failwith (Printf.sprintf "ioctl returned %d" r)

  (* Memory-manager commands are their own request numbers and report
     status in the parameter structure's status field. *)
  let uvm' ~defs ~fd ~cmd ~rmstatus b =
    let r = Nv_tables.ioctl ~fd ~request:cmd b in
    if r <> 0 then failwith (Printf.sprintf "ioctl returned %d" r);
    let status = Nv_tables.get_field b rmstatus in
    if status <> 0 then failwith ("uvm returned " ^ error_str defs status)

  let driver_version_major b =
    let off, len =
      Defs.Nv0000_ctrl_system_get_build_version_v2_params.driverversionbuffer
    in
    let stop = ref 0 in
    while !stop < len && Bigarray.Array1.get b (off + !stop) <> '\000' do
      incr stop
    done;
    let s = String.init !stop (fun i -> Bigarray.Array1.get b (off + i)) in
    let major =
      match String.index_opt s '.' with
      | Some i -> String.sub s 0 i
      | None -> s
    in
    match int_of_string_opt major with
    | Some v -> v
    | None -> failwith (Printf.sprintf "cannot parse driver version %S" s)

  (* Parameter-structure constructors *)

  let nvos21_params ~root ~parent ~cls ?params () =
    let module P = Defs.Nvos21_parameters in
    let b = Nv_tables.create_blob P.sizeof in
    Nv_tables.set_field b P.hroot root;
    Nv_tables.set_field b P.hobjectparent parent;
    Nv_tables.set_field b P.hclass cls;
    Option.iter
      (fun p ->
        Nv_tables.set_field b P.pallocparms
          (Nativeint.to_int (Nv_tables.blob_addr p)))
      params;
    b

  let memory_allocation_params ~root ~size ~page_size ~uncached ~contiguous
      ~read_only =
    let attr =
      ((if contiguous then Defs.nvos32_attr_physicality_contiguous
        else Defs.nvos32_attr_physicality_allow_noncontiguous)
      lsl 27)
      lor ((if page_size > 0x1000 then Defs.nvos32_attr_page_size_huge else 0)
          lsl 23)
      lor ((if uncached then Defs.nvos32_attr_location_pci else 0) lsl 25)
    in
    let attr2 =
      ((if uncached then Defs.nvos32_attr2_gpu_cacheable_no
        else Defs.nvos32_attr2_gpu_cacheable_yes)
      lsl 2)
      lor ((if page_size > 0x1000 then Defs.nvos32_attr2_page_size_huge_2mb
            else 0)
          lsl 20)
      lor Defs.nvos32_attr2_zbc_prefer_no_zbc
      lor (if read_only then Defs.nvos32_attr2_protection_user_read_only lsl 22
           else 0)
    in
    let flags =
      Defs.nvos32_alloc_flags_map_not_required
      lor Defs.nvos32_alloc_flags_memory_handle_provided
      lor Defs.nvos32_alloc_flags_alignment_force
      lor Defs.nvos32_alloc_flags_ignore_bank_placement
      lor (if not uncached then Defs.nvos32_alloc_flags_persistent_vidmem
           else 0)
    in
    let cls = if uncached then Defs.nv1_memory_system else Defs.nv1_memory_user in
    let module P = Defs.Nv_memory_allocation_params in
    let p = Nv_tables.create_blob P.sizeof in
    Nv_tables.set_field p P.owner root;
    Nv_tables.set_field p P.typ
      (if uncached then Defs.nvos32_type_notifier else Defs.nvos32_type_image);
    Nv_tables.set_field p P.flags flags;
    Nv_tables.set_field p P.attr attr;
    Nv_tables.set_field p P.attr2 attr2;
    Nv_tables.set_field p P.format 6;
    Nv_tables.set_field p P.size size;
    Nv_tables.set_field p P.alignment page_size;
    Nv_tables.set_field p P.limit (size - 1);
    (cls, p)

  let map_external_params ~rm_ctrl_fd ~root ~va ~size ~mem_handle ~gpu_uuid =
    if Bytes.length gpu_uuid <> 16 then
      invalid_arg "map_external_params: gpu uuid must be 16 bytes";
    let module P = Defs.Uvm_map_external_allocation_params in
    let module A = Defs.Uvm_gpu_mapping_attributes in
    let b = Nv_tables.create_blob P.sizeof in
    Nv_tables.set_field b P.base (Nativeint.to_int va);
    Nv_tables.set_field b P.length size;
    Nv_tables.set_field b P.rmctrlfd rm_ctrl_fd;
    Nv_tables.set_field b P.hclient root;
    Nv_tables.set_field b P.hmemory mem_handle;
    Nv_tables.set_field b P.gpuattributescount 1;
    blit_bytes b ~off:(P.pergpuattributes_offset + fst A.gpuuuid) gpu_uuid;
    Nv_tables.set_field ~base:P.pergpuattributes_offset b A.gpumappingtype 1;
    b

  (* Object allocation and control *)

  let rm_alloc' ~fd_ctl ~defs ~root ~parent ~cls ?params () =
    let module P = Defs.Nvos21_parameters in
    let b = nvos21_params ~root ~parent ~cls ?params () in
    escape fd_ctl ~nr:Defs.nv_esc_rm_alloc b;
    keep_alive params;
    let status = Nv_tables.get_field b P.status in
    if status = Defs.nv_err_no_memory then
      raise
        (Nv_iface.Out_of_memory ("rm_alloc returned " ^ error_str defs status));
    if status <> 0 then failwith ("rm_alloc returned " ^ error_str defs status);
    Nv_tables.get_field b P.hobjectnew

  let rm_control' ~fd_ctl ~defs ~root ~obj ~cmd ?params () =
    let module P = Defs.Nvos54_parameters in
    let b = Nv_tables.create_blob P.sizeof in
    Nv_tables.set_field b P.hclient root;
    Nv_tables.set_field b P.hobject obj;
    Nv_tables.set_field b P.cmd cmd;
    Option.iter
      (fun p ->
        Nv_tables.set_field b P.paramssize (Bigarray.Array1.dim p);
        Nv_tables.set_field b P.params
          (Nativeint.to_int (Nv_tables.blob_addr p)))
      params;
    escape fd_ctl ~nr:Defs.nv_esc_rm_control b;
    keep_alive params;
    let status = Nv_tables.get_field b P.status in
    if status <> 0 then failwith ("rm_control returned " ^ error_str defs status)

  (* Root bootstrap *)

  let init_root () =
    match !state with
    | Some st -> st
    | None ->
        let fd_ctl = File_io.openfile "/dev/nvidiactl" ~flags:File_io.o_rdwr in
        let fd_uvm = File_io.openfile "/dev/nvidia-uvm" ~flags:File_io.o_rdwr in
        let fd_uvm_2 =
          File_io.openfile "/dev/nvidia-uvm" ~flags:File_io.o_rdwr
        in
        (* the root client exists before the driver generation is known;
           the bootstrap decodes with the oldest layouts *)
        let boot = Nv_defs_versions.v570 in
        let root =
          rm_alloc' ~fd_ctl ~defs:boot ~root:0 ~parent:0
            ~cls:Defs.nv01_root_client ()
        in
        let module V = Defs.Nv0000_ctrl_system_get_build_version_v2_params in
        let vb = Nv_tables.create_blob V.sizeof in
        rm_control' ~fd_ctl ~defs:boot ~root ~obj:root
          ~cmd:Defs.nv0000_ctrl_cmd_system_get_build_version_v2 ~params:vb ();
        let defs = Nv_tables.defs_for_driver ~major:(driver_version_major vb) in
        let module I = Defs.Uvm_initialize_params in
        let ib = Nv_tables.create_blob I.sizeof in
        uvm' ~defs ~fd:fd_uvm ~cmd:Defs.uvm_initialize ~rmstatus:I.rmstatus ib;
        (* the memory-manager handshake may be unsupported; that failure
           is expected and harmless *)
        (try
           let module M = Defs.Uvm_mm_initialize_params in
           let mb = Nv_tables.create_blob M.sizeof in
           Nv_tables.set_field mb M.uvmfd fd_uvm;
           uvm' ~defs ~fd:fd_uvm_2 ~cmd:Defs.uvm_mm_initialize
             ~rmstatus:M.rmstatus mb
         with Failure _ -> ());
        let module C = Defs.Nv_ioctl_card_info in
        let cards = 64 in
        let cb = Nv_tables.create_blob (cards * C.sizeof) in
        escape fd_ctl ~nr:Defs.nv_esc_card_info cb;
        let gpus = ref [] in
        for i = cards - 1 downto 0 do
          let base = i * C.sizeof in
          if Nv_tables.get_field ~base cb C.valid <> 0 then
            gpus :=
              {
                gpu_id = Nv_tables.get_field ~base cb C.gpu_id;
                minor_number = Nv_tables.get_field ~base cb C.minor_number;
              }
              :: !gpus
        done;
        let st =
          {
            fd_ctl;
            fd_uvm;
            fd_uvm_2;
            root;
            defs;
            gpus_info = Array.of_list !gpus;
          }
        in
        state := Some st;
        st

  let rm_alloc st ~parent ~cls ?params () =
    rm_alloc' ~fd_ctl:st.fd_ctl ~defs:st.defs ~root:st.root ~parent ~cls
      ?params ()

  let rm_control st ~obj ~cmd ?params () =
    rm_control' ~fd_ctl:st.fd_ctl ~defs:st.defs ~root:st.root ~obj ~cmd
      ?params ()

  let uvm st ?fd ~cmd ~rmstatus b =
    uvm' ~defs:st.defs
      ~fd:(Option.value fd ~default:st.fd_uvm)
      ~cmd ~rmstatus b

  (* Devices *)

  let new_gpu_fd st ~minor =
    let fd =
      File_io.openfile (Printf.sprintf "/dev/nvidia%d" minor)
        ~flags:File_io.o_rdwr
    in
    let module P = Defs.Nv_ioctl_register_fd in
    let b = Nv_tables.create_blob P.sizeof in
    Nv_tables.set_field b P.ctl_fd st.fd_ctl;
    escape fd ~nr:Defs.nv_esc_register_fd b;
    fd

  let create st ~device_id =
    if device_id >= Array.length st.gpus_info then
      failwith
        (Printf.sprintf
           "No device found for %d. Requesting more devices than the system \
            has?"
           device_id);
    let gpu = st.gpus_info.(device_id) in
    let fd_dev = new_gpu_fd st ~minor:gpu.minor_number in
    let module P = Defs.Nv0000_ctrl_gpu_get_id_info_v2_params in
    let b = Nv_tables.create_blob P.sizeof in
    Nv_tables.set_field b P.gpuid gpu.gpu_id;
    rm_control st ~obj:st.root ~cmd:Defs.nv0000_ctrl_cmd_gpu_get_id_info_v2
      ~params:b ();
    {
      device_id;
      fd_dev;
      gpu_minor = gpu.minor_number;
      gpu_instance = Nv_tables.get_field b P.deviceinstance;
      nvdevice = 0;
      subdevice = 0;
      virtmem = 0;
      gpu_uuid = Bytes.make 16 '\000';
    }

  (* Memory *)

  let gpu_map_to_cpu st t ~memory_handle ~size ?target ?(flags = 0)
      ?(system = false) () =
    let fd =
      if system then File_io.openfile "/dev/nvidiactl" ~flags:File_io.o_rdwr
      else new_gpu_fd st ~minor:t.gpu_minor
    in
    let module W = Defs.Nv_ioctl_nvos33_parameters_with_fd in
    let module P = Defs.Nvos33_parameters in
    let b = Nv_tables.create_blob W.sizeof in
    Nv_tables.set_field b P.hclient st.root;
    Nv_tables.set_field b P.hdevice t.nvdevice;
    Nv_tables.set_field b P.hmemory memory_handle;
    Nv_tables.set_field b P.length size;
    Nv_tables.set_field b P.flags flags;
    Nv_tables.set_field b W.fd fd;
    escape st.fd_ctl ~nr:Defs.nv_esc_rm_map_memory b;
    let status = Nv_tables.get_field b P.status in
    if status <> 0 then
      failwith ("_gpu_map_to_cpu returned " ^ error_str st.defs status);
    File_io.mmap
      ~addr:(Option.value target ~default:0n)
      ~size
      ~prot:(File_io.prot_read lor File_io.prot_write)
      ~flags:
        (File_io.map_shared
        lor (if target = None then 0 else File_io.map_fixed))
      ~fd ~offset:0L

  let gpu_uvm_map st t ~va ~size ~mem_handle ?(create_range = true)
      ?(has_cpu_mapping = false) ?owner_id () =
    if create_range then begin
      let module C = Defs.Uvm_create_external_range_params in
      let cb = Nv_tables.create_blob C.sizeof in
      Nv_tables.set_field cb C.base (Nativeint.to_int va);
      Nv_tables.set_field cb C.length size;
      uvm st ~cmd:Defs.uvm_create_external_range ~rmstatus:C.rmstatus cb;
      let open Nv_defs_versions in
      let p46 = st.defs.nvos46_parameters in
      let b = Nv_tables.create_blob p46.sizeof in
      Nv_tables.set_field b p46.hclient st.root;
      Nv_tables.set_field b p46.hdevice t.nvdevice;
      Nv_tables.set_field b p46.hdma t.virtmem;
      Nv_tables.set_field b p46.hmemory mem_handle;
      Nv_tables.set_field b p46.length size;
      Nv_tables.set_field b p46.flags
        ((Defs.nvos46_flags_page_size_4kb lsl 8)
        lor (Defs.nvos46_flags_cache_snoop_enable lsl 4)
        lor (Defs.nvos46_flags_dma_offset_fixed_true lsl 15));
      Nv_tables.set_field b p46.dmaoffset (Nativeint.to_int va);
      escape st.fd_ctl ~nr:Defs.nv_esc_rm_map_memory_dma b;
      let status = Nv_tables.get_field b p46.status in
      if status <> 0 then
        failwith ("nv_sys_alloc 1 returned " ^ error_str st.defs status);
      assert (Nv_tables.get_field b p46.dmaoffset = Nativeint.to_int va)
    end;
    let module M = Defs.Uvm_map_external_allocation_params in
    let mb =
      map_external_params ~rm_ctrl_fd:st.fd_ctl ~root:st.root ~va ~size
        ~mem_handle ~gpu_uuid:t.gpu_uuid
    in
    uvm st ~cmd:Defs.uvm_map_external_allocation ~rmstatus:M.rmstatus mb;
    Hcq.Buffer.make ~va ~size
      ?view:
        (if has_cpu_mapping then Some (Hcq.Mmio.make ~addr:va ~size) else None)
      ~meta:
        {
          Nv_iface.h_memory = mem_handle;
          owner_id = Option.value owner_id ~default:t.device_id;
        }
      ()

  let alloc st t ?(host = false) ?(uncached = false) ?(cpu_access = false)
      ?(contiguous = false) ?(map_flags = 0) ?cpu_addr ?(read_only = false)
      size =
    (* uncached memory lives in system pages; huge pages only serve large
       device-memory allocations *)
    let page_size =
      if uncached || host then 0x1000
      else if size >= 8 lsl 20 then 2 lsl 20
      else 4 lsl 10
    in
    let size = round_up size page_size in
    let alloced = cpu_addr = None in
    let va =
      match cpu_addr with
      | Some a -> a
      | None -> alloc_gpu_vaddr ~alignment:page_size ~force_low:cpu_access size
    in
    if host then begin
      let va =
        if alloced then
          File_io.mmap ~addr:va ~size
            ~prot:(File_io.prot_read lor File_io.prot_write)
            ~flags:
              (File_io.map_fixed lor File_io.map_shared
             lor File_io.map_anonymous)
            ~fd:(-1) ~offset:0L
        else va
      in
      let flags =
        (Defs.nvos02_flags_physicality_noncontiguous lsl 4)
        lor (Defs.nvos02_flags_coherency_cached lsl 12)
        lor (Defs.nvos02_flags_mapping_no_map lsl 30)
      in
      incr host_object_enumerator;
      let module W = Defs.Nv_ioctl_nvos02_parameters_with_fd in
      let module P = Defs.Nvos02_parameters in
      let b = Nv_tables.create_blob W.sizeof in
      Nv_tables.set_field b P.hroot st.root;
      Nv_tables.set_field b P.hobjectparent t.nvdevice;
      Nv_tables.set_field b P.flags flags;
      Nv_tables.set_field b P.hobjectnew !host_object_enumerator;
      Nv_tables.set_field b P.hclass Defs.nv01_memory_system_os_descriptor;
      Nv_tables.set_field b P.pmemory (Nativeint.to_int va);
      Nv_tables.set_field b P.limit (size - 1);
      Nv_tables.set_field b W.fd (-1);
      escape t.fd_dev ~nr:Defs.nv_esc_rm_alloc_memory b;
      let status = Nv_tables.get_field b P.status in
      if status <> 0 then
        failwith ("host alloc returned " ^ error_str st.defs status);
      let mem_handle = Nv_tables.get_field b P.hobjectnew in
      gpu_uvm_map st t ~va ~size ~mem_handle ~has_cpu_mapping:true ()
    end
    else begin
      let cls, params =
        memory_allocation_params ~root:st.root ~size ~page_size ~uncached
          ~contiguous ~read_only
      in
      let mem_handle = rm_alloc st ~parent:t.nvdevice ~cls ~params () in
      let va =
        if cpu_access then
          gpu_map_to_cpu st t ~memory_handle:mem_handle ~size ~target:va
            ~flags:map_flags ~system:uncached ()
        else va
      in
      gpu_uvm_map st t ~va ~size ~mem_handle ~has_cpu_mapping:cpu_access ()
    end

  let free st t buf =
    let buf = Hcq.Buffer.base buf in
    let meta = Hcq.Buffer.meta buf in
    if meta.Nv_iface.owner_id = t.device_id then begin
      (* a handle above the enumerator came from the driver: release its
         physical memory; host objects only unregister through the
         address-range free below *)
      if meta.Nv_iface.h_memory > !host_object_enumerator then begin
        let module P = Defs.Nvos00_parameters in
        let b = Nv_tables.create_blob P.sizeof in
        Nv_tables.set_field b P.hroot st.root;
        Nv_tables.set_field b P.hobjectparent t.nvdevice;
        Nv_tables.set_field b P.hobjectold meta.Nv_iface.h_memory;
        escape st.fd_ctl ~nr:Defs.nv_esc_rm_free b;
        let status = Nv_tables.get_field b P.status in
        if status <> 0 then
          failwith ("_gpu_free returned " ^ error_str st.defs status)
      end;
      let open Nv_defs_versions in
      let fp = st.defs.uvm_free_params in
      let b = Nv_tables.create_blob fp.sizeof in
      Nv_tables.set_field b fp.base
        (Nativeint.to_int (Hcq.Buffer.va buf));
      Option.iter
        (fun f -> Nv_tables.set_field b f (Hcq.Buffer.size buf))
        fp.length;
      uvm st ~cmd:Defs.uvm_free ~rmstatus:fp.rmstatus b;
      match Hcq.Buffer.view buf with
      | Some _ ->
          File_io.munmap (Hcq.Buffer.va buf) ~size:(Hcq.Buffer.size buf)
      | None -> ()
    end

  (* An import maps an already-created range: no new range or physical
     memory, and the original allocator keeps ownership. *)
  let map st t buf =
    let meta = Hcq.Buffer.meta buf in
    gpu_uvm_map st t ~va:(Hcq.Buffer.va buf) ~size:(Hcq.Buffer.size buf)
      ~mem_handle:meta.Nv_iface.h_memory ~create_range:false
      ~owner_id:meta.Nv_iface.owner_id ()

  (* Channel set-up *)

  let setup_usermode st t =
    let module P = Defs.Nv0080_ctrl_gpu_get_classlist_params in
    let nb = Nv_tables.create_blob P.sizeof in
    rm_control st ~obj:t.nvdevice ~cmd:Defs.nv0080_ctrl_cmd_gpu_get_classlist
      ~params:nb ();
    let n = Nv_tables.get_field nb P.numclasses in
    let listing = Nv_tables.create_blob (n * 4) in
    let cb = Nv_tables.create_blob P.sizeof in
    Nv_tables.set_field cb P.numclasses n;
    Nv_tables.set_field cb P.classlist
      (Nativeint.to_int (Nv_tables.blob_addr listing));
    rm_control st ~obj:t.nvdevice ~cmd:Defs.nv0080_ctrl_cmd_gpu_get_classlist
      ~params:cb ();
    keep_alive listing;
    let n = Nv_tables.get_field cb P.numclasses in
    let classes = List.init n (fun i -> Nv_tables.get_field listing (i * 4, 4)) in
    let pick name candidates =
      match List.find_opt (fun c -> List.mem c classes) candidates with
      | Some c -> c
      | None -> failwith ("setup_usermode: no supported " ^ name ^ " class")
    in
    let usermode_class =
      pick "usermode" [ Defs.hopper_usermode_a; Defs.turing_usermode_a ]
    in
    let gpfifo_class =
      pick "gpfifo"
        [ Defs.blackwell_channel_gpfifo_a; Defs.ampere_channel_gpfifo_a ]
    in
    let compute_class =
      pick "compute"
        [ Defs.blackwell_compute_b; Defs.ada_compute_a; Defs.ampere_compute_b ]
    in
    let dma_class =
      pick "dma" [ Defs.blackwell_dma_copy_b; Defs.ampere_dma_copy_b ]
    in
    let handle = rm_alloc st ~parent:t.subdevice ~cls:usermode_class () in
    let mmio_size = 0x10000 in
    let addr = gpu_map_to_cpu st t ~memory_handle:handle ~size:mmio_size () in
    {
      Nv_iface.handle;
      mmio = Hcq.Mmio.make ~addr ~size:mmio_size;
      compute_class;
      dma_class;
      gpfifo_class;
    }

  let setup_vm st t ~vaspace =
    let module P = Defs.Nv2080_ctrl_gpu_get_gid_info_params in
    let b = Nv_tables.create_blob P.sizeof in
    Nv_tables.set_field b P.flags
      Defs.nv2080_gpu_cmd_gpu_get_gid_flags_format_binary;
    Nv_tables.set_field b P.length 16;
    rm_control st ~obj:t.subdevice ~cmd:Defs.nv2080_ctrl_cmd_gpu_get_gid_info
      ~params:b ();
    t.gpu_uuid <- read_bytes b ~off:(fst P.data) ~len:16;
    let module R = Defs.Uvm_register_gpu_params in
    let rb = Nv_tables.create_blob R.sizeof in
    blit_bytes rb ~off:(fst R.gpu_uuid) t.gpu_uuid;
    Nv_tables.set_field rb R.rmctrlfd (-1);
    uvm st ~cmd:Defs.uvm_register_gpu ~rmstatus:R.rmstatus rb;
    let module V = Defs.Uvm_register_gpu_vaspace_params in
    let vb = Nv_tables.create_blob V.sizeof in
    blit_bytes vb ~off:(fst V.gpuuuid) t.gpu_uuid;
    Nv_tables.set_field vb V.rmctrlfd st.fd_ctl;
    Nv_tables.set_field vb V.hclient st.root;
    Nv_tables.set_field vb V.hvaspace vaspace;
    uvm st ~cmd:Defs.uvm_register_gpu_vaspace ~rmstatus:V.rmstatus vb

  let setup_gpfifo_vm st t ~gpfifo =
    let module P = Defs.Uvm_register_channel_params in
    let b = Nv_tables.create_blob P.sizeof in
    blit_bytes b ~off:(fst P.gpuuuid) t.gpu_uuid;
    Nv_tables.set_field b P.rmctrlfd st.fd_ctl;
    Nv_tables.set_field b P.hclient st.root;
    Nv_tables.set_field b P.hchannel gpfifo;
    Nv_tables.set_field b P.base
      (Nativeint.to_int (alloc_gpu_vaddr ~force_low:true 0x4000000));
    Nv_tables.set_field b P.length 0x4000000;
    uvm st ~cmd:Defs.uvm_register_channel ~rmstatus:P.rmstatus b

  let iface ~device_id : Nv_iface.t =
    let st = init_root () in
    let t = create st ~device_id in
    {
      Nv_iface.root = st.root;
      gpu_instance = t.gpu_instance;
      count = Array.length st.gpus_info;
      set_device =
        (fun ~nvdevice ~subdevice ~virtmem ->
          t.nvdevice <- nvdevice;
          t.subdevice <- subdevice;
          t.virtmem <- virtmem);
      rm_alloc =
        (fun ~parent ~cls ?params () -> rm_alloc st ~parent ~cls ?params ());
      rm_control =
        (fun ~obj ~cmd ?params () -> rm_control st ~obj ~cmd ?params ());
      alloc =
        (fun ?host ?uncached ?cpu_access ?contiguous ?map_flags ?cpu_addr size ->
          alloc st t ?host ?uncached ?cpu_access ?contiguous ?map_flags
            ?cpu_addr size);
      free = (fun buf -> free st t buf);
      map = (fun buf -> map st t buf);
      setup_usermode = (fun () -> setup_usermode st t);
      setup_vm = (fun ~vaspace -> setup_vm st t ~vaspace);
      setup_gpfifo_vm = (fun ~gpfifo -> setup_gpfifo_vm st t ~gpfifo);
      (* no driver wait channel exists: signal waits spin *)
      sleep = (fun (_ : int) -> ());
      device_fini = (fun () -> ());
      nvdev = None;
    }
end

(* Loaded programs *)

module Program = struct
  type 'meta t = {
    params : 'meta program;
    name : string;
    lib_gpu : 'meta Hcq.Buffer.t;
    regs_usage : int;
    shmem_usage : int;
    lcmem_usage : int;
    constbufs : (int * (nativeint * int)) list;
    cbuf_0 : int array;
    max_threads : int;
    kernargs_alloc_size : int;
  }

  let r_cuda_64 = 2

  (* ".nv.constant<N>" or ".nv.constant<N>.<kernel>" names carry the
     constant-bank index right after the prefix. *)
  let constant_index name =
    let prefix = ".nv.constant" in
    let plen = String.length prefix in
    if String.length name <= plen || not (String.starts_with ~prefix name) then
      None
    else begin
      let i = ref plen and v = ref 0 and digits = ref 0 in
      while !i < String.length name && name.[!i] >= '0' && name.[!i] <= '9' do
        v := (!v * 10) + Char.code name.[!i] - Char.code '0';
        incr digits;
        incr i
      done;
      if !digits = 0 then None else Some !v
    end

  let u16le b off = Bytes.get_uint16_le b off
  let u32le b off = Int32.to_int (Bytes.get_int32_le b off) land 0xffffffff

  (* ".nv.info" entries: a value-format byte, an attribute byte, and a
     16-bit size that is the value itself except for format 4, where it
     counts the payload bytes that follow. [f] receives the attribute
     and the payload offset of each format-4 entry. *)
  let iter_info (s : Tolk.Elf.section) f =
    let off = ref 0 in
    while !off < s.Tolk.Elf.size do
      let typ = Char.code (Bytes.get s.content !off) in
      let param = Char.code (Bytes.get s.content (!off + 1)) in
      let sz = u16le s.content (!off + 2) in
      (match typ with
      | 1 | 2 | 3 -> ()
      | 4 -> f param (!off + 4)
      | _ ->
          failwith (Printf.sprintf "unknown EIATTR format %d in %s" typ s.name));
      off := !off + (if typ = 4 then sz else 0) + 4
    done

  let load (dev : 'meta device) ~alloc ~ensure_local_memory ~name lib =
    let elf = Tolk.Elf.load ~force_section_align:128 lib in
    let image = Tolk.Elf.image elf in
    let sections = Tolk.Elf.sections elf in
    (* at least 4 KiB of guard space after the image mitigates prefetch
       memory faults *)
    let lib_gpu = alloc (round_up (Bytes.length image) 0x1000 + 0x1000) in
    let va = Nativeint.to_int (Hcq.Buffer.va lib_gpu) in
    let regs_usage = ref 0
    and shmem_usage = ref 0x400
    and lcmem_usage = ref 0x240
    and cbuf0_size = ref 0 in
    let prog_addr = ref va and prog_sz = ref (Bytes.length image) in
    let constbufs = ref [ (0, (0n, 0x160)) ] in
    let set_constbuf i entry =
      if List.mem_assoc i !constbufs then
        constbufs :=
          List.map
            (fun (j, e) -> if j = i then (j, entry) else (j, e))
            !constbufs
      else constbufs := !constbufs @ [ (i, entry) ]
    in
    Array.iter
      (fun (s : Tolk.Elf.section) ->
        if s.name = ".nv.shared." ^ name then
          shmem_usage := round_up (0x400 + s.size) 128;
        if s.name = ".text." ^ name then begin
          prog_addr := va + s.addr;
          prog_sz := s.size
        end
        else
          match constant_index s.name with
          | Some i -> set_constbuf i (Nativeint.of_int (va + s.addr), s.size)
          | None ->
              if String.starts_with ~prefix:".nv.info" s.name then
                iter_info s (fun param data ->
                    (* attribute 0xa is the kernel's constant-bank
                       descriptor: the bank size follows a 32-bit bank
                       ordinal; 0x12 is the minimum stack size, to
                       which the engine adds a 0x240-byte reserve;
                       0x2f is the register count *)
                    if s.name = ".nv.info." ^ name && param = 0xa then
                      cbuf0_size := u16le s.content (data + 4)
                    else if s.name = ".nv.info" && param = 0x12 then
                      lcmem_usage := u32le s.content (data + 4) + 0x240
                    else if s.name = ".nv.info" && param = 0x2f then
                      regs_usage := u32le s.content (data + 4)))
      sections;
    List.iter
      (fun (r : Tolk.Elf.reloc) ->
        if r.symbol.shndx = 0 then
          failwith
            ("Attempting to relocate against an undefined symbol "
           ^ r.symbol.name);
        let target = va + sections.(r.symbol.shndx).addr + r.symbol.value in
        if r.r_type = r_cuda_64 then
          Bytes.set_int64_le image r.offset (Int64.of_int target)
        else if r.r_type = 0x38 then
          Bytes.set_int32_le image (r.offset + 4)
            (Int32.of_int (target land 0xffffffff))
        else if r.r_type = 0x39 then
          Bytes.set_int32_le image (r.offset + 4) (Int32.of_int (target lsr 32))
        else failwith (Printf.sprintf "unknown NV reloc %d" r.r_type))
      (Tolk.Elf.relocs elf);
    (* driver parameters occupy constant-buffer-0 entries up to index
       223 on Blackwell, up to 11 before *)
    let min_cbuf0_entries =
      if dev.compute_class >= Defs.blackwell_compute_a then 224 else 12
    in
    let cbuf_0 = Array.make (max (!cbuf0_size / 4) min_cbuf0_entries) 0 in
    ensure_local_memory !lcmem_usage;
    Hcq.Mmio.blit_bytes (Hcq.Buffer.cpu_view lib_gpu) ~off:0 image;
    let compute_class = dev.compute_class in
    let qmd_size = Qmd.sizeof ~compute_class in
    (* the template's backing bytes belong to the program; [free]
       releases them *)
    let qmd_addr =
      Hcq.File_io.mmap ~addr:0n ~size:qmd_size
        ~prot:(Hcq.File_io.prot_read lor Hcq.File_io.prot_write)
        ~flags:(Hcq.File_io.map_private lor Hcq.File_io.map_anonymous)
        ~fd:(-1) ~offset:0L
    in
    let qmd =
      Qmd.create
        ~view:(Hcq.Mmio.make ~addr:qmd_addr ~size:qmd_size)
        ~compute_class
    in
    let sw = va64 dev.shared_mem_window and lw = va64 dev.local_mem_window in
    let version_fields =
      if compute_class >= Defs.blackwell_compute_a then begin
        cbuf_0.(188) <- lo32 sw;
        cbuf_0.(189) <- hi32 sw;
        cbuf_0.(190) <- lo32 lw;
        cbuf_0.(191) <- hi32 lw;
        cbuf_0.(223) <- 0xfffdc0;
        let pa4 = !prog_addr lsr 4 in
        [
          ("qmd_major_version", 5);
          ("qmd_type", Defs.nvcec0_qmdv05_00_qmd_type_grid_cta);
          ("program_address_upper_shifted4", pa4 lsr 32);
          ("program_address_lower_shifted4", pa4 land 0xffffffff);
          ("register_count", !regs_usage);
          ("shared_memory_size_shifted7", !shmem_usage lsr 7);
          ("shader_local_memory_high_size_shifted4", dev.slm_per_thread lsr 4);
        ]
      end
      else begin
        cbuf_0.(6) <- lo32 sw;
        cbuf_0.(7) <- hi32 sw;
        cbuf_0.(8) <- lo32 lw;
        cbuf_0.(9) <- hi32 lw;
        cbuf_0.(10) <- 0xfffdc0;
        [
          ("qmd_major_version", 3);
          ("sm_global_caching_enable", 1);
          ("program_address_upper", !prog_addr lsr 32);
          ("program_address_lower", !prog_addr land 0xffffffff);
          ("shared_memory_size", !shmem_usage);
          ("register_count_v", !regs_usage);
          ("shader_local_memory_high_size", dev.slm_per_thread);
        ]
      end
    in
    let smem_cfg =
      match
        List.find_opt (fun c -> c * 1024 >= !shmem_usage) [ 32; 64; 100 ]
      with
      | Some c -> (c * 1024 / 4096) + 1
      | None ->
          failwith
            (Printf.sprintf
               "shared memory size 0x%x exceeds the largest configuration"
               !shmem_usage)
    in
    Qmd.write qmd
      (version_fields
      @ [
          ("qmd_group_id", 0x3f);
          ("invalidate_texture_header_cache", 1);
          ("invalidate_texture_sampler_cache", 1);
          ("invalidate_texture_data_cache", 1);
          ("invalidate_shader_data_cache", 1);
          ("api_visible_call_limit", 1);
          ("sampler_index", 1);
          ("barrier_count", 1);
          ( "cwd_membar_type",
            Defs.nvc6c0_qmdv03_00_cwd_membar_type_l1_sysmembar );
          ("constant_buffer_invalidate_0", 1);
          ("min_sm_config_shared_mem_size", smem_cfg);
          ("target_sm_config_shared_mem_size", smem_cfg);
          ("max_sm_config_shared_mem_size", 0x1a);
          ("program_prefetch_size", min (!prog_sz lsr 8) 0x1ff);
          ("sass_version", dev.sass_version);
          ("program_prefetch_addr_upper_shifted", !prog_addr lsr 40);
          ("program_prefetch_addr_lower_shifted", !prog_addr lsr 8);
        ]);
    List.iter
      (fun (i, (addr, sz)) ->
        Qmd.set_constant_buf_addr qmd i addr;
        Qmd.write qmd
          [
            (Printf.sprintf "constant_buffer_size_shifted4_%d" i, sz);
            (Printf.sprintf "constant_buffer_valid_%d" i, 1);
          ])
      !constbufs;
    (* register allocation granularity is 256 per warp, warp allocation
       granularity is 4, register file size 65536 *)
    let max_threads =
      65536 / round_up (max 1 !regs_usage * 32) 256 / 4 * 4 * 32
    in
    let cbuf0_bytes = snd (List.assoc 0 !constbufs) in
    {
      params = { dev; qmd; cbuf0_size = cbuf0_bytes };
      name;
      lib_gpu;
      regs_usage = !regs_usage;
      shmem_usage = !shmem_usage;
      lcmem_usage = !lcmem_usage;
      constbufs = !constbufs;
      cbuf_0;
      max_threads;
      kernargs_alloc_size = round_up cbuf0_bytes 256 + 0x800;
    }

  let free ~free:release t =
    release t.lib_gpu;
    Hcq.File_io.munmap
      (Hcq.Mmio.addr t.params.qmd.Qmd.view)
      ~size:t.params.qmd.Qmd.size

  let call t ~kernargs ~queue ~timeline ~timeline_value ?wait ?timeout_ms ~bufs
      ~vals ~global_size ~local_size () =
    let gx, gy, gz = global_size and lx, ly, lz = local_size in
    let threads = lx * ly * lz in
    if
      threads > 1024 || t.max_threads < threads
      || t.lcmem_usage > t.params.dev.slm_per_thread
    then
      failwith
        (Printf.sprintf
           "Too many resources requested for launch, %d threads, max %d"
           threads t.max_threads);
    if
      gx > 2147483647 || gy > 65535 || gz > 65535 || lx > 1024 || ly > 1024
      || lz > 64
    then
      failwith
        (Printf.sprintf "Invalid global/local dims (%d, %d, %d), (%d, %d, %d)"
           gx gy gz lx ly lz);
    let slot = Hcq.Kernargs.alloc kernargs t.kernargs_alloc_size in
    Hcq.Kernargs.write_args ~prefix:t.cbuf_0 slot ~bufs ~vals;
    let cq = Compute_queue.create t.params.dev in
    Compute_queue.wait cq ~value:(timeline_value - 1) timeline;
    Compute_queue.memory_barrier cq;
    (match wait with
    | Some (st, _) -> Compute_queue.timestamp cq st
    | None -> ());
    Compute_queue.exec cq t.params ~kernargs:slot ~global_size ~local_size;
    (match wait with
    | Some (_, en) -> Compute_queue.timestamp cq en
    | None -> ());
    Compute_queue.signal cq ~value:timeline_value timeline;
    Compute_queue.submit cq queue;
    match wait with
    | None -> None
    | Some (st, en) ->
        Hcq.Signal.wait timeline ?timeout_ms timeline_value;
        Some ((Hcq.Signal.timestamp en -. Hcq.Signal.timestamp st) /. 1e6)
end

(* Local-memory sizing *)

let ensure_has_local_memory (dev : 'meta device) ~alloc ~free ~num_gpcs
    ~num_tpc_per_gpc ~num_sm_per_tpc ~max_warps_per_sm ~tl ~queue required =
  if dev.slm_per_thread < required then begin
    let old_slm_per_thread = dev.slm_per_thread in
    dev.slm_per_thread <- round_up required 32;
    let bytes_per_tpc =
      round_up
        (round_up (dev.slm_per_thread * 32) 0x200
        * max_warps_per_sm * num_sm_per_tpc)
        0x8000
    in
    let old = dev.shader_local_mem in
    Option.iter free old;
    let shader_local_mem =
      match
        alloc (round_up (bytes_per_tpc * num_tpc_per_gpc * num_gpcs) 0x20000)
      with
      | buf -> buf
      | exception Nv_iface.Out_of_memory _ when Option.is_some old ->
          (* out of memory: reallocate the old size so the device stays
             usable, and restore the sizing state *)
          let buf = alloc (Hcq.Buffer.size (Option.get old)) in
          dev.slm_per_thread <- old_slm_per_thread;
          buf
    in
    dev.shader_local_mem <- Some shader_local_mem;
    let cq = Compute_queue.create dev in
    Compute_queue.wait cq
      ~value:(tl.Hcq.Timeline.timeline_value - 1)
      tl.Hcq.Timeline.timeline;
    Compute_queue.setup cq
      ~local_mem:(Hcq.Buffer.va shader_local_mem)
      ~local_mem_tpc_bytes:bytes_per_tpc ();
    Compute_queue.signal cq
      ~value:(Hcq.Timeline.next_timeline tl)
      tl.Hcq.Timeline.timeline;
    Compute_queue.submit cq queue
  end
