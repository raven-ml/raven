(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

module Hcq = Tolk_hcq.Hcq
module Nv_tables = Nv_tables
module Nvdev = Nvdev
module Ip = Ip
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
  type storage = Host of bytes | Mapped of Hcq.Mmio.t
  type t = {
    ver : int;
    size : int;
    storage : storage;
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
    { ver; size; storage = Mapped view; fields = Lazy.force fields }

  let empty ~compute_class =
    let ver, size, fields = layout ~compute_class in
    {ver; size; storage = Host (Bytes.make size '\000'); fields = Lazy.force fields}

  let read_bytes t ~off ~len = match t.storage with
    | Host bytes -> Bytes.sub bytes off len
    | Mapped view -> Hcq.Mmio.read_bytes view ~off ~len
  let blit_bytes t ~off bytes = match t.storage with
    | Host target -> Bytes.blit bytes 0 target off (Bytes.length bytes)
    | Mapped view -> Hcq.Mmio.blit_bytes view ~off bytes
  let mapped_view t = match t.storage with
    | Mapped view -> view
    | Host _ -> invalid_arg "Qmd: descriptor has no mapped view"

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
    let b = read_bytes t ~off:first ~len:((hi / 8) - first + 1) in
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
    let b = read_bytes t ~off:first ~len in
    let num = ref 0 in
    for i = len - 1 downto 0 do
      num := (!num lsl 8) lor Char.code (Bytes.unsafe_get b i)
    done;
    let mask = ((1 lsl width) - 1) lsl (lo mod 8) in
    let num = !num land lnot mask lor (v lsl (lo mod 8)) in
    for i = 0 to len - 1 do
      Bytes.unsafe_set b i (Char.unsafe_chr ((num lsr (8 * i)) land 0xff))
    done;
    blit_bytes t ~off:first b

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

  let to_bytes t = read_bytes t ~off:0 ~len:t.size
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
  let put = Int32.to_int (Hcq.Mmio.read32 qd.gpput 0) in
  Hcq.Mmio.write64 qd.ring
    (put mod entries * 8)
    (Int64.of_int (((cmdq_addr / 4) lsl 2) lor (n lsl 42) lor (1 lsl 41)));
  Hcq.Mmio.fence ();
  Hcq.Mmio.write32 qd.gpput 0 (Int32.of_int ((put + 1) mod entries));
  Hcq.Mmio.fence ();
  Hcq.Mmio.write32 dev.gpu_mmio 0x90 (Int32.of_int qd.token)

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

  let release t ~value ~timestamp sg =
    let patched =
      match t.active_qmd with
      | None -> false
      | Some qmd ->
          let v3 = Qmd.version qmd < 4 in
          let view = Qmd.mapped_view qmd in
          let rec claim i =
            if i > 1 then false
            else if Qmd.read qmd (Printf.sprintf "release%d_enable" i) <> 0
            then claim (i + 1)
            else begin
              Qmd.write qmd
                ([Printf.sprintf "release%d_enable" i, 1;
                  (if v3 then Printf.sprintf "release%d_structure_size" i
                   else Printf.sprintf "release_structure_size_%d" i),
                  (if timestamp then 0 else 2)]
                @ if v3 then [Printf.sprintf "release%d_payload64b" i, 1] else []);
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
                (if timestamp then Defs.nvc56f_sem_execute_release_timestamp_en else 0);
        |];
      if not timestamp then nvm t.q 0 Defs.nvc56f_non_stall_interrupt [| 0 |];
      t.active_qmd <- None
    end

  let signal t ?(value = 0) sg = release t ~value ~timestamp:false sg
  let timestamp t sg = release t ~value:0 ~timestamp:true sg

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

  let release t ~value ~timestamp sg =
    let value = if Hcq.Signal.is_timeline sg then value land 0xffffffff else value in
    let a = va64 (Hcq.Signal.value_addr sg) in
    nvm t.q 4 Defs.nvc6b5_set_semaphore_a [| hi32 a; lo32 a; value |];
    nvm t.q 4 Defs.nvc6b5_launch_dma
      [|
        bits Defs.nvc6b5_launch_dma_flush_enable
          Defs.nvc6b5_launch_dma_flush_enable_true
        lor bits Defs.nvc6b5_launch_dma_semaphore_type
              (if timestamp then Defs.nvc6b5_launch_dma_semaphore_type_release_four_word_semaphore
               else Defs.nvc6b5_launch_dma_semaphore_type_release_one_word_semaphore);
      |]

  let signal t ?(value = 0) sg = release t ~value ~timestamp:false sg
  let timestamp t sg = release t ~value:0 ~timestamp:true sg

  let wait t ?(value = 0) sg =
    push_sem_wait t.q ~addr:(Hcq.Signal.value_addr sg) ~value

  let submit t qd = submit_to_gpfifo t.dev t.q qd
end

(* Driver interface seam *)

module Nv_iface = struct
  exception Out_of_memory of string

  type nvdev = ..

  type usermode = {
    handle : int;
    mmio : Hcq.Mmio.t;
    compute_class : int;
    dma_class : int;
    gpfifo_class : int;
  }

  type 'mem t = {
    root : int;
    gpu_instance : int;
    count : int;
    defs : Nv_defs_versions.t;
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
      'mem Hcq.Buffer.t;
    free : 'mem Hcq.Buffer.t -> unit;
    kind : 'mem Hcq.Buffer.t Type.Id.t;
    hmemory : 'mem Hcq.Buffer.t -> int;
    map : Tolk.Device.Buffer.t -> 'mem Hcq.Buffer.t;
    unmap : 'mem Hcq.Buffer.t -> unit;
    setup_usermode : unit -> usermode;
    setup_vm : vaspace:int -> unit;
    setup_gpfifo_vm : gpfifo:int -> unit;
    sleep : int -> unit;
    device_fini : unit -> unit;
    nvdev : nvdev option;
  }

  type packed = Pack : 'mem t -> packed

  let is_nvd t = t.nvdev <> None
end

(* Kernel-driver interface *)

module Nvk_iface = struct
  type ownership = Owned | Registered | Imported
  type mem = { h_memory : int; ownership : ownership }
  let kind : mem Hcq.Buffer.t Type.Id.t = Type.Id.make ()
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
  let is_initialized () = !state <> None

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
            gpus_info =
              Array.of_list (Tolk_hcq.System.filter_visible_devices "NV" !gpus);
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
    let registered = ref false in
    Fun.protect ~finally:(fun () -> if not !registered then File_io.close fd)
      (fun () ->
        let module P = Defs.Nv_ioctl_register_fd in
        let b = Nv_tables.create_blob P.sizeof in
        Nv_tables.set_field b P.ctl_fd st.fd_ctl;
        escape fd ~nr:Defs.nv_esc_register_fd b;
        registered := true;
        fd)

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
    Fun.protect ~finally:(fun () -> File_io.close fd) (fun () ->
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
        ~fd ~offset:0L)

  let free_memory st t handle =
    let module P = Defs.Nvos00_parameters in
    let b = Nv_tables.create_blob P.sizeof in
    Nv_tables.set_field b P.hroot st.root;
    Nv_tables.set_field b P.hobjectparent t.nvdevice;
    Nv_tables.set_field b P.hobjectold handle;
    escape st.fd_ctl ~nr:Defs.nv_esc_rm_free b;
    let status = Nv_tables.get_field b P.status in
    if status <> 0 then failwith ("_gpu_free returned " ^ error_str st.defs status)

  let free_range st ~va ~size =
    let open Nv_defs_versions in
    let fp = st.defs.uvm_free_params in
    let b = Nv_tables.create_blob fp.sizeof in
    Nv_tables.set_field b fp.base (Nativeint.to_int va);
    Option.iter (fun f -> Nv_tables.set_field b f size) fp.length;
    uvm st ~cmd:Defs.uvm_free ~rmstatus:fp.rmstatus b

  let gpu_uvm_map st t ~va ~size ~mem_handle ?(create_range = true)
      ?(has_cpu_mapping = false) ?(ownership = Owned) () =
    let created = ref false and complete = ref false in
    Fun.protect
      ~finally:(fun () -> if !created && not !complete then free_range st ~va ~size)
      (fun () ->
        if create_range then begin
          let module C = Defs.Uvm_create_external_range_params in
          let cb = Nv_tables.create_blob C.sizeof in
          Nv_tables.set_field cb C.base (Nativeint.to_int va);
          Nv_tables.set_field cb C.length size;
          uvm st ~cmd:Defs.uvm_create_external_range ~rmstatus:C.rmstatus cb;
          created := true;
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
        let buffer = Hcq.Buffer.make ~va ~size
          ?view:
            (if has_cpu_mapping then Some (Hcq.Mmio.make ~addr:va ~size) else None)
          ~meta:
            {
              h_memory = mem_handle; ownership;
            }
          () in
        complete := true;
        buffer)

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
    let memory = ref None and mapping = ref None and complete = ref false in
    Fun.protect
      ~finally:(fun () -> if not !complete then
        Fun.protect
          ~finally:(fun () -> Option.iter (fun addr -> File_io.munmap addr ~size) !mapping)
          (fun () -> Option.iter (free_memory st t) !memory))
      (fun () ->
        let buffer =
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
            if alloced then mapping := Some va;
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
            memory := Some mem_handle;
            gpu_uvm_map st t ~va ~size ~mem_handle ~has_cpu_mapping:true
              ~ownership:(if alloced then Owned else Registered) ()
          end
          else begin
            let cls, params =
              memory_allocation_params ~root:st.root ~size ~page_size ~uncached
                ~contiguous ~read_only
            in
            let mem_handle = rm_alloc st ~parent:t.nvdevice ~cls ~params () in
            memory := Some mem_handle;
            let va =
              if cpu_access then
                gpu_map_to_cpu st t ~memory_handle:mem_handle ~size ~target:va
                  ~flags:map_flags ~system:uncached ()
              else va
            in
            if cpu_access then mapping := Some va;
            gpu_uvm_map st t ~va ~size ~mem_handle ~has_cpu_mapping:cpu_access ()
          end in
        complete := true;
        buffer)

  let free st t buf =
    let buf = Hcq.Buffer.base buf in
    let meta = Hcq.Buffer.meta buf in
    if meta.ownership <> Imported then begin
      (* a handle above the enumerator came from the driver: release its
         physical memory; host objects only unregister through the
         address-range free below *)
      if meta.h_memory > !host_object_enumerator then free_memory st t meta.h_memory;
      free_range st ~va:(Hcq.Buffer.va buf) ~size:(Hcq.Buffer.size buf);
      match Hcq.Buffer.view buf with
      | Some view when meta.ownership = Owned ->
          File_io.munmap (Hcq.Mmio.addr view) ~size:(Hcq.Mmio.size view)
      | Some _ | None -> ()
    end

  (* An import maps an already-created range: no new range or physical
     memory, and the original allocator keeps ownership. *)
  let map st t buf =
    let meta = Hcq.Buffer.meta buf in
    gpu_uvm_map st t ~va:(Hcq.Buffer.va buf) ~size:(Hcq.Buffer.size buf)
      ~mem_handle:meta.h_memory ~create_range:false
      ~ownership:Imported ()

  let map_storage st t source =
    let module B = Tolk.Device.Buffer in
    let Tolk.Device.Allocator.Pack allocator = B.allocator source in
    let existing = match Type.Id.provably_equal kind allocator.kind with
      | Some Type.Equal -> B.get kind source
      | None -> B.find_mapping kind source in
    match existing with
    | Some raw -> map st t raw
    | None ->
        let address = match B.host_addr source with
          | Some address -> address
          | None -> invalid_arg "NVK map requires NVK or host-accessible storage" in
        if Nativeint.logand address 0xfffn <> 0n then
          invalid_arg "NVK host mapping requires page alignment";
        alloc st t ~host:true ~cpu_addr:address (B.nbytes source)

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
      | None ->
          failwith
            (Printf.sprintf
               "setup_usermode: no supported %s class: wanted one of [%s], GPU \
                advertises [%s]"
               name
               (String.concat "; "
                  (List.map (Printf.sprintf "0x%x") candidates))
               (String.concat "; " (List.map (Printf.sprintf "0x%x") classes)))
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

  let iface ~device_id : mem Nv_iface.t =
    let st = init_root () in
    let t = create st ~device_id in
    {
      Nv_iface.root = st.root;
      gpu_instance = t.gpu_instance;
      count = Array.length st.gpus_info;
      defs = st.defs;
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
      kind;
      hmemory = (fun b -> (Hcq.Buffer.meta b).h_memory);
      map = (fun source -> map_storage st t source);
      unmap = (fun buf -> free st t buf);
      setup_usermode = (fun () -> setup_usermode st t);
      setup_vm = (fun ~vaspace -> setup_vm st t ~vaspace);
      setup_gpfifo_vm = (fun ~gpfifo -> setup_gpfifo_vm st t ~gpfifo);
      (* no driver wait channel exists: signal waits spin *)
      sleep = (fun (_ : int) -> ());
      device_fini = (fun () -> ());
      nvdev = None;
    }
end

(* Driver-less PCI interface: ops_nv.py:556-581 PCIIface *)

module Pci_iface = struct
  module Base = Tolk_hcq.System.Pci_iface_base
  module System = Tolk_hcq.System

  (* The booted device the interface drives: the passive device state and
     the GSP boot-and-RPC layer over it. *)
  type nv_boot = { nvdev : Nvdev.t; gsp : Ip.Gsp.t }

  type t = {
    base : (nv_boot, Nvdev.Nv_page_table.t) Base.t;
    root : int;
  }

  type Nv_iface.nvdev += Nv_pci of nv_boot

  let vendor = 0x10de

  (* ops_nv.py:560 the supported consumer device ids, masked by 0xff00. *)
  let pci_ids =
    [
      ( 0xff00,
        [
          0x2200; 0x2400; 0x2500; 0x2600; 0x2700; 0x2800; 0x2b00; 0x2c00;
          0x2d00; 0x2f00;
        ] );
    ]

  let impl t = Base.dev_impl t.base

  (* nvdev.py:74 the device bring-up: construct the falcon and GSP layers
     over the reset-settled device, then run their software and hardware
     init in order — the falcon's software init reserves the tables the
     GSP's software init checks, and the GSP's software init produces the
     boot arguments the falcon's hardware init consumes. *)
  let boot pci_dev =
    let nvdev = Nvdev.create pci_dev in
    (* the client is reinitialized every run, so leave the booting state
       only for the memory manager the device already set up *)
    Nvdev.set_is_booting nvdev false;
    if Nvdev.fmc_boot nvdev then begin
      let flcn = Ip.Flcn_cot.create nvdev in
      let gsp = Ip.Gsp.create nvdev ~boot:(Ip.Gsp.cot_boot flcn) in
      Ip.Flcn_cot.init_sw flcn;
      Ip.Gsp.init_sw gsp;
      Ip.Flcn_cot.init_hw flcn
        ~libos_args_sysmem:(Ip.Gsp.libos_args_sysmem gsp)
        ~wpr_meta_sysmem:(Ip.Gsp.wpr_meta_sysmem gsp);
      Ip.Gsp.init_hw gsp;
      { nvdev; gsp }
    end
    else begin
      let flcn = Ip.Flcn.create nvdev in
      let gsp = Ip.Gsp.create nvdev ~boot:(Ip.Gsp.falcon_boot flcn) in
      Ip.Flcn.init_sw flcn;
      Ip.Gsp.init_sw gsp;
      Ip.Flcn.init_hw flcn
        ~libos_args_sysmem:(Ip.Gsp.libos_args_sysmem gsp)
        ~wpr_meta_sysmem:(Ip.Gsp.wpr_meta_sysmem gsp);
      Ip.Gsp.init_hw gsp;
      { nvdev; gsp }
    end

  (* nvdev.py:88 fini: the device finalizes its IPs in reverse
     initialization order (the boot layers hold no hardware state, so
     their finalization is a no-op). *)
  let fini (b : nv_boot) =
    Ip.Gsp.fini_hw b.gsp;
    match Ip.Gsp.boot b.gsp with
    | Ip.Gsp.Falcon f -> Ip.Flcn.fini_hw f
    | Ip.Gsp.Cot c -> Ip.Flcn_cot.fini_hw c

  (* ops_nv.py:557 PCIIface.__init__: opening the PCI interface after the
     kernel driver has run would overwrite its memory manager's mappings. *)
  let create ~device_id =
    if Nvk_iface.is_initialized () then
      failwith
        "Cannot use the PCI interface after the kernel driver has been \
         initialized (would corrupt UVM memory)";
    let base =
      Base.create ~name:"NV" ~devpref:"NV" ~dev_id:device_id ~vendor
        ~devices:pci_ids ~base_class:0x03 ~vram_bar:1
        ~va_start:(Nativeint.of_int Nvdev.va_base)
        ~va_size:Nvdev.va_size ~dev_impl:boot
        ~mm:(fun impl -> Nvdev.mm impl.nvdev)
        ()
    in
    let t = { base; root = 0xc1000000 } in
    (* ops_nv.py:564: register the client with the driver *)
    let (_ : int) =
      Ip.Gsp.rpc_rm_alloc (impl t).gsp ~hparent:0 ~hclass:Defs.nv01_root
        ~params:(Nv_tables.create_blob Defs.Nv0000_alloc_parameters.sizeof)
        ~client:t.root ()
    in
    t

  (* ops_nv.py:570 setup_usermode: the work-submission doorbell lives in a
     fixed window of BAR0; the engine classes come from the GSP. *)
  let setup_usermode t =
    let g = (impl t).gsp in
    {
      Nv_iface.handle = 0xce000000;
      mmio =
        System.Pci_device.map_bar (Base.pci_dev t.base) ~off:0xbb0000
          ~size:0x10000 0;
      compute_class = Ip.Gsp.compute_class g;
      dma_class = Ip.Gsp.dma_class g;
      gpfifo_class = Ip.Gsp.gpfifo_class g;
    }

  (* ops_nv.py:579 sleep: drain the status queue for GSP events and surface
     a latched device fault. *)
  let sleep t =
    Ip.Gsp.drain_responses (impl t).gsp;
    if Nvdev.is_err_state (impl t).nvdev then failwith "Device fault detected"

  let iface t : Base.mem Nv_iface.t =
    let g = (impl t).gsp in
    {
      Nv_iface.root = t.root;
      gpu_instance = 0;
      count = Base.count t.base;
      defs = Nv_tables.defs_for_driver ~major:570;
      (* the GSP tracks the device and subdevice handles itself *)
      set_device = (fun ~nvdevice:_ ~subdevice:_ ~virtmem:_ -> ());
      rm_alloc =
        (fun ~parent ~cls ?params () ->
          Ip.Gsp.rpc_rm_alloc g ~hparent:parent ~hclass:cls ?params
            ~client:t.root ());
      rm_control =
        (fun ~obj ~cmd ?params () ->
          Ip.Gsp.rpc_rm_control g ~hobject:obj ~cmd ?params ~client:t.root ());
      alloc =
        (fun ?host ?uncached ?cpu_access ?contiguous ?map_flags:_ ?cpu_addr:_
             size ->
          Base.alloc t.base ?host ?uncached ?cpu_access ?contiguous size);
      free = Base.free t.base;
      kind = Base.kind;
      hmemory = Base.hmemory;
      map = Base.map t.base;
      unmap = Base.unmap t.base;
      setup_usermode = (fun () -> setup_usermode t);
      (* the driver-less path sets the vaspace page directory in rm_alloc *)
      setup_vm = (fun ~vaspace:_ -> ());
      setup_gpfifo_vm = (fun ~gpfifo:_ -> ());
      (* ops_nv.py:29 long waits back off to draining GSP events, which
         also surface a latched device fault *)
      sleep = (fun spent_ms -> if spent_ms > 200 then sleep t);
      device_fini = (fun () -> fini (impl t));
      nvdev = Some (Nv_pci (impl t));
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

  type data = {
    image : bytes;
    relocations : (int * int * int * int) list;
    prog_offset : int;
    prog_size : int;
    regs_usage : int;
    shmem_usage : int;
    lcmem_usage : int;
    constbufs : (int * (int * int)) list;
    cbuf0_size : int;
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

  let image ~name lib =
    let elf = Tolk.Elf.load ~force_section_align:128 lib in
    let image = Tolk.Elf.image elf in
    let sections = Tolk.Elf.sections elf in
    let regs_usage = ref 0
    and shmem_usage = ref 0x400
    and lcmem_usage = ref 0x240
    and cbuf0_size = ref 0 in
    let prog_addr = ref 0 and prog_sz = ref (Bytes.length image) in
    let constbufs = ref [ (0, (0, 0x160)) ] in
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
          prog_addr := s.addr;
          prog_sz := s.size
        end
        else
          match constant_index s.name with
          | Some i -> set_constbuf i (s.addr, s.size)
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
    let relocations = List.map (fun (r : Tolk.Elf.reloc) ->
        if r.symbol.shndx = 0 then
          failwith ("Attempting to relocate against an undefined symbol " ^ r.symbol.name);
        let target = sections.(r.symbol.shndx).addr + r.symbol.value in
        let offset, width, shift =
          if r.r_type = r_cuda_64 then r.offset, 8, 0
          else if r.r_type = 0x38 then r.offset + 4, 4, 0
          else if r.r_type = 0x39 then r.offset + 4, 4, 32
          else failwith (Printf.sprintf "unknown NV reloc %d" r.r_type) in
        if offset < 0 || offset > Bytes.length image - width then
          invalid_arg "NV relocation extends beyond the image";
        offset, target, width, shift) (Tolk.Elf.relocs elf) in
    let padded = Bytes.make (round_up (Bytes.length image) 0x1000 + 0x1000) '\000' in
    Bytes.blit image 0 padded 0 (Bytes.length image);
    {image = padded; relocations; prog_offset = !prog_addr; prog_size = !prog_sz;
     regs_usage = !regs_usage; shmem_usage = !shmem_usage;
     lcmem_usage = !lcmem_usage; constbufs = !constbufs; cbuf0_size = !cbuf0_size}

  let template (dev : 'meta device) (data : data) =
    (* driver parameters occupy constant-buffer-0 entries up to index
       223 on Blackwell, up to 11 before *)
    let min_cbuf0_entries =
      if dev.compute_class >= Defs.blackwell_compute_a then 224 else 12
    in
    let cbuf_0 = Array.make (max (data.cbuf0_size / 4) min_cbuf0_entries) 0 in
    let compute_class = dev.compute_class in
    let qmd = Qmd.empty ~compute_class in
    let sw = va64 dev.shared_mem_window and lw = va64 dev.local_mem_window in
    let version_fields =
      if compute_class >= Defs.blackwell_compute_a then begin
        cbuf_0.(188) <- lo32 sw;
        cbuf_0.(189) <- hi32 sw;
        cbuf_0.(190) <- lo32 lw;
        cbuf_0.(191) <- hi32 lw;
        cbuf_0.(223) <- 0xfffdc0;
        [
          ("qmd_major_version", 5);
          ("qmd_type", Defs.nvcec0_qmdv05_00_qmd_type_grid_cta);
          ("register_count", data.regs_usage);
          ("shared_memory_size_shifted7", data.shmem_usage lsr 7);
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
          ("shared_memory_size", data.shmem_usage);
          ("register_count_v", data.regs_usage);
          ("shader_local_memory_high_size", dev.slm_per_thread);
        ]
      end
    in
    let smem_cfg =
      match
        List.find_opt (fun c -> c * 1024 >= data.shmem_usage) [ 32; 64; 100 ]
      with
      | Some c -> (c * 1024 / 4096) + 1
      | None ->
          failwith
            (Printf.sprintf
               "shared memory size 0x%x exceeds the largest configuration"
               data.shmem_usage)
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
          ("program_prefetch_size", min (data.prog_size lsr 8) 0x1ff);
          ("sass_version", dev.sass_version);
        ]);
    List.iter
      (fun (i, (offset, sz)) ->
        ignore offset;
        Qmd.write qmd
          [
            (Printf.sprintf "constant_buffer_size_shifted4_%d" i, sz);
            (Printf.sprintf "constant_buffer_valid_%d" i, 1);
          ])
      data.constbufs;
    qmd, cbuf_0

  let load (dev : 'meta device) ~alloc ~ensure_local_memory ~name lib =
    let data = image ~name lib in
    ensure_local_memory data.lcmem_usage;
    let qmd, cbuf_0 = template dev data in
    (* A guard page after the image mitigates instruction prefetch faults. *)
    let lib_gpu = alloc (Bytes.length data.image) in
    let va = Nativeint.to_int (Hcq.Buffer.va lib_gpu) in
    let prog_addr = va + data.prog_offset in
    let constbufs = List.map (fun (i, (offset, size)) ->
        i, (Nativeint.of_int (va + offset), size)) data.constbufs in
    let image = Bytes.copy data.image in
    List.iter (fun (offset, target, width, shift) ->
        let address = Int64.shift_right_logical (Int64.of_int (va + target)) shift in
        if width = 8 then Bytes.set_int64_le image offset address
        else Bytes.set_int32_le image offset (Int64.to_int32 address)) data.relocations;
    List.iter (fun (i, (address, _)) -> Qmd.set_constant_buf_addr qmd i address) constbufs;
    let address = if Qmd.version qmd < 4 then prog_addr else prog_addr lsr 4 in
    let suffix = if Qmd.version qmd < 4 then "" else "_shifted4" in
    Qmd.write qmd ["program_address_upper" ^ suffix, address lsr 32;
      "program_address_lower" ^ suffix, address land 0xffffffff;
      "program_prefetch_addr_upper_shifted", prog_addr lsr 40;
      "program_prefetch_addr_lower_shifted", prog_addr lsr 8];
    Hcq.Mmio.blit_bytes (Hcq.Buffer.cpu_view lib_gpu) ~off:0 image;
    (* register allocation granularity is 256 per warp, warp allocation
       granularity is 4, register file size 65536 *)
    let max_threads =
      65536 / round_up (max 1 data.regs_usage * 32) 256 / 4 * 4 * 32
    in
    let cbuf0_bytes = snd (List.assoc 0 constbufs) in
    {
      params = { dev; qmd; cbuf0_size = cbuf0_bytes };
      name;
      lib_gpu;
      regs_usage = data.regs_usage;
      shmem_usage = data.shmem_usage;
      lcmem_usage = data.lcmem_usage;
      constbufs = constbufs;
      cbuf_0;
      max_threads;
      kernargs_alloc_size = round_up cbuf0_bytes 256 + 0x800;
    }

  let free ~free:release t = release t.lib_gpu

  let call t ~layout ~kernargs ~queue ~timeline ~timeline_value ?wait ?timeout_ms ~bufs
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
    Hcq.Kernargs.write_args ~prefix:t.cbuf_0 layout slot ~bufs ~vals;
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

module Encoded_queue = struct
  open Tolk_uop
  module U = Uop
  module D = Dtype

  let u32 n = U.const (Const.int D.uint32 n)
  let u64 n = U.const (Const.int D.uint64 n)
  let cast dtype src = U.cast ~src ~dtype
  let op op lhs rhs = U.alu_binary ~op ~lhs ~rhs
  let add = op Ops.Add
  let shr = op Ops.Shr
  let bor = op Ops.Or
  let band = op Ops.And
  let addr name src = U.getaddr ~device:name ~src ()
  let index ptr = U.index ~ptr ~idxs:[U.const_int 0] ()
  let placeholder ?allocation ?(volatile = false) name tag dtype size =
    U.placeholder ~shape:[size] ~dtype ~slot:0 ~device:(U.Single name)
      ?allocation ~volatile () |> U.with_tag tag
  let context name = U.placeholder ~shape:[2] ~dtype:D.uint64 ~slot:0
      ~device:(U.Single name) ~volatile:true ~allocation:("hcq_submission", "") ()
  let words value =
    if D.itemsize (U.dtype value) = 8 then
      [cast D.uint32 value; cast D.uint32 (shr value (u64 32))]
    else [cast D.uint32 value]
  let hilo value = [cast D.uint32 (shr value (u64 32)); cast D.uint32 value]

  (* Merge symbolic fields by dword. Descriptor fields share words, and
     several span a boundary, so independently storing fields would clobber
     their neighbours. Link patches stay naturally aligned. *)
  type descriptor = { qmd : Qmd.t; words : (int, U.t) Hashtbl.t;
    buffer : U.t; blob : bytes; mutable releases : int }

  let field t name value =
    let hi, lo = Qmd.range t.qmd name in
    for word = lo / 32 to hi / 32 do
      let start = max lo (word * 32) and finish = min hi (word * 32 + 31) in
      let width = finish - start + 1 in
      let mask = ((1 lsl width) - 1) lsl (start mod 32) in
      let old = match Hashtbl.find_opt t.words word with
        | Some old -> old
        | None -> u32 (Int32.to_int (Bytes.get_int32_le t.blob (word * 4)) land 0xffffffff) in
      let value = cast D.uint32 (shr (cast D.uint64 value) (u64 (start - lo))) in
      let value = op Ops.Shl value (u32 (start mod 32)) in
      Hashtbl.replace t.words word (bor (band old (u32 (0xffffffff lxor mask)))
        (band value (u32 mask)))
    done

  let lower = Hcq.Submission.lower

  let encode (dev : 'meta device) ~name ~compute_entries ~copy_entries
      ~compute_token ~copy_token u =
    match U.op u, U.arg u, U.children u with
    | Ops.Custom_function, U.Arg.String ("submit_nv_compute" | "submit_nv_copy" as kind),
        [linear; dependency] ->
        let compute = kind = "submit_nv_compute" in
        let commands = ref [] and descriptors = ref [] and previous = ref None in
        let q xs = commands := List.rev_append xs !commands in
        let nvm subchannel method_ xs =
          let xs = List.concat_map words xs in
          q (u32 ((2 lsl 28) lor (List.length xs lsl 16) lor (subchannel lsl 13)
            lor (method_ lsr 2)) :: xs) in
        let sem signal value flags = nvm 0 Defs.nvc56f_sem_addr_lo
            [addr name signal; cast D.uint64 value;
             u32 (flags lor bits Defs.nvc56f_sem_execute_payload_size
                Defs.nvc56f_sem_execute_payload_size_64bit)] in
        let release ~timestamp signal value =
          match !previous with
          | Some d when d.releases < 2 ->
              let i = d.releases in
              d.releases <- i + 1;
              let v3 = Qmd.version d.qmd < 4 in
              let set fmt v = field d (Printf.sprintf fmt i) v in
              set "release%d_enable" (u32 1);
              set (if v3 then "release%d_structure_size" else "release_structure_size_%d")
                (u32 (if timestamp then 0 else 2));
              if v3 then set "release%d_payload64b" (u32 1);
              let address = addr name signal in
              set (if v3 then "release%d_address_lower" else "release_semaphore%d_addr_lower") address;
              set (if v3 then "release%d_address_upper" else "release_semaphore%d_addr_upper") (shr address (u64 32));
              set (if v3 then "release%d_payload_lower" else "release_semaphore%d_payload_lower") value;
              set (if v3 then "release%d_payload_upper" else "release_semaphore%d_payload_upper") (shr (cast D.uint64 value) (u64 32))
          | _ ->
              previous := None;
              sem signal value (bits Defs.nvc56f_sem_execute_operation Defs.nvc56f_sem_execute_operation_release
                lor bits Defs.nvc56f_sem_execute_release_wfi Defs.nvc56f_sem_execute_release_wfi_en
                lor bits Defs.nvc56f_sem_execute_release_timestamp
                  (if timestamp then Defs.nvc56f_sem_execute_release_timestamp_en else 0));
              if not timestamp then nvm 0 Defs.nvc56f_non_stall_interrupt [u32 0] in
        let dims xs = List.init 3 (fun i -> if i >= List.length xs then u32 1 else
            match List.nth xs i with
            | U.Launch_int n -> u32 n | U.Launch_float f -> u32 (int_of_float f)
            | U.Launch_sym v -> cast D.uint32 v) in
        List.iter (fun node -> match U.as_call node, U.arg node with
          | Some {body; args}, _ when U.op body = Ops.Program && compute ->
              let info = Option.get (U.as_program_info body) in
              let object_ = U.to_elf body in
              let data = Program.image ~name:object_.name object_.lib in
              let template_dev = {dev with slm_per_thread = max dev.slm_per_thread (round_up data.lcmem_usage 32)} in
              let qmd, prefix = Program.template template_dev data in
              let image = U.placeholder ~shape:[Bytes.length data.image] ~dtype:D.uint8
                  ~slot:(U.fresh_buffer_slot ()) ~device:(U.Single name)
                  ~allocation:("nv_image", string_of_int data.lcmem_usage) () |> U.with_tag "program" in
              let relocations = List.concat_map (fun (offset, target, width, shift) ->
                  let address = shr (add (addr name image) (u64 target)) (u64 shift) in
                  if width = 8 then [offset, cast D.uint32 address;
                    offset + 4, cast D.uint32 (shr address (u64 32))]
                  else [offset, cast D.uint32 address]) data.relocations in
              let image = Tolk.Hcq2.patch ~blob:(Bytes.to_string data.image) image relocations in
              let layout = Tiny_elf.layout object_.signature in
              let qmd_size = round_up (Qmd.sizeof ~compute_class:dev.compute_class) 256 in
              let at = qmd_size + Array.length prefix * 4 in
              let size = round_up (max (qmd_size + snd (List.assoc 0 data.constbufs))
                  (at + List.fold_left (fun n (f : Tiny_elf.field) -> max n (f.offset + f.size)) 0 layout)) 256 in
              let buffer = U.placeholder ~shape:[size] ~dtype:D.uint8 ~slot:(U.fresh_buffer_slot ())
                  ~device:(U.Single name) () |> U.with_tag "qmd" in
              let blob = Bytes.make size '\000' in
              let template = Qmd.to_bytes qmd in
              Bytes.blit template 0 blob 0 (Bytes.length template);
              Array.iteri (fun i word -> Bytes.set_int32_le blob (qmd_size + i * 4) (Int32.of_int word)) prefix;
              let d = {qmd; words = Hashtbl.create 32; buffer; blob; releases = 0} in
              let address = addr name buffer in
              let check_dims limits values = List.iteri (fun i dim ->
                  let value = match dim with U.Launch_int n -> Some n
                    | U.Launch_float f -> Some (int_of_float f) | U.Launch_sym _ -> None in
                  Option.iter (fun n -> if n < 1 || n > List.nth limits i then
                    invalid_arg "NV queue: invalid launch dimensions") value) values in
              check_dims [2147483647; 65535; 65535] info.global_size;
              check_dims [1024; 1024; 64] info.local_size;
              let local = dims info.local_size and global = dims info.global_size in
              let max_threads = 65536 / round_up (max 1 data.regs_usage * 32) 256 / 4 * 4 * 32 in
              let threads = List.fold_left (fun n dim -> n * Bound.to_int (U.vmax dim)) 1 local in
              if threads > min 1024 max_threads then
                invalid_arg "NV queue: too many threads for the kernel's register allocation";
              let grid = if Qmd.version qmd < 4 then
                  ["cta_raster_width"; "cta_raster_height"; "cta_raster_depth"]
                else ["grid_width"; "grid_height"; "grid_depth"] in
              List.iter2 (field d) grid global;
              List.iteri (fun i v -> field d ("cta_thread_dimension" ^ string_of_int i) v) local;
              let program_address = add (addr name image) (u64 data.prog_offset) in
              let suffix, shift = if Qmd.version qmd < 4 then "", 0 else "_shifted4", 4 in
              field d ("program_address_lower" ^ suffix) (shr program_address (u64 shift));
              field d ("program_address_upper" ^ suffix) (shr program_address (u64 (32 + shift)));
              field d "program_prefetch_addr_lower_shifted" (shr program_address (u64 8));
              field d "program_prefetch_addr_upper_shifted" (shr program_address (u64 40));
              List.iter (fun (i, (offset, _)) ->
                  let address = if i = 0 then add address (u64 qmd_size)
                    else add (addr name image) (u64 offset) in
                  let suffix, shift = if Qmd.version qmd < 4 then "", 0 else "_shifted6", 6 in
                  field d (Printf.sprintf "constant_buffer_addr_lower%s_%d" suffix i) (shr address (u64 shift));
                  field d (Printf.sprintf "constant_buffer_addr_upper%s_%d" suffix i) (shr address (u64 (32 + shift)))) data.constbufs;
              let buffers = List.filter (fun a -> not (U.is_bound_var a)) args in
              let bound = List.filter_map (fun a -> match U.as_bind a with
                  | Some {var; value} -> Option.map (fun n -> n, value) (U.program_var_name var)
                  | None -> None) args in
              let vars = List.map (fun v -> match U.program_var_name v with
                  | Some n -> Option.value (List.assoc_opt n bound) ~default:v | None -> v) info.vars in
              let actuals = List.map (fun i -> addr name (List.nth buffers i)) info.globals @ vars in
              let rows = List.map (fun (f : Tiny_elf.field) ->
                  let dtype = if f.argument.addrspace = D.Alu then f.argument.dtype else D.uint64 in
                  at + f.offset, cast dtype (List.nth actuals f.argument.slot)) layout in
              (match !previous with
               | None -> nvm 1 Defs.nvc6c0_send_pcas_a [cast D.uint32 (shr address (u64 8))];
                   nvm 1 Defs.nvc6c0_send_signaling_pcas2_b [u32 9]
               | Some prev -> field prev "dependent_qmd0_pointer" (shr address (u64 8));
                   List.iter (fun key -> field prev key (u32 1))
                     ["dependent_qmd0_action"; "dependent_qmd0_prefetch"; "dependent_qmd0_enable"]);
              previous := Some d;
              descriptors := (d, rows) :: !descriptors
          | Some {body; args = [dst; src]}, _ when U.op body = Ops.Store && not compute ->
              let bytes = U.max_numel dst * D.itemsize (U.dtype dst) in
              let offset = ref 0 in
              while !offset < bytes do
                let size = min (1 lsl 31) (bytes - !offset) in
                nvm 4 Defs.nvc6b5_offset_in_upper
                  (hilo (add (addr name src) (u64 !offset)) @ hilo (add (addr name dst) (u64 !offset)));
                nvm 4 Defs.nvc6b5_line_length_in [u32 size];
                nvm 4 Defs.nvc6b5_launch_dma [u32 (bits Defs.nvc6b5_launch_dma_data_transfer_type
                  Defs.nvc6b5_launch_dma_data_transfer_type_non_pipelined lor
                  bits Defs.nvc6b5_launch_dma_src_memory_layout Defs.nvc6b5_launch_dma_src_memory_layout_pitch lor
                  bits Defs.nvc6b5_launch_dma_dst_memory_layout Defs.nvc6b5_launch_dma_dst_memory_layout_pitch)];
                offset := !offset + size
              done
          | _, U.Arg.Typed ("barrier", _) ->
              previous := None;
              if compute then begin
                let cq = Compute_queue.create dev in
                Compute_queue.memory_barrier cq;
                q (Array.to_list (Q.dwords (Compute_queue.q cq)) |> List.map u32)
              end
          | _, U.Arg.Typed ("wait", _) ->
              previous := None;
              let args = U.src node in
              sem args.(0) args.(1) (bits Defs.nvc56f_sem_execute_operation Defs.nvc56f_sem_execute_operation_acq_circ_geq)
          | _, U.Arg.Typed (("store" | "timestamp" as kind), _) ->
              let timestamp = kind = "timestamp" in
              let args = U.src node in
              let value = if timestamp then u64 0 else args.(1) in
              if compute then release ~timestamp args.(0) value else begin
                nvm 4 Defs.nvc6b5_set_semaphore_a (hilo (addr name args.(0)) @ [cast D.uint32 value]);
                nvm 4 Defs.nvc6b5_launch_dma [u32 (bits Defs.nvc6b5_launch_dma_flush_enable 1 lor
                    bits Defs.nvc6b5_launch_dma_semaphore_type (if timestamp then 2 else 1))]
              end
          | _ -> invalid_arg "NV queue: unsupported instruction") (U.children linear);
        let patches = List.rev_map (fun (d, rows) ->
            let fields = Hashtbl.fold (fun word value rows -> (word * 4, value) :: rows) d.words []
                |> List.sort (fun (a, _) (b, _) -> Int.compare a b) in
            Tolk.Hcq2.patch ~blob:(Bytes.to_string d.blob) ~after:[dependency] d.buffer (fields @ rows)) !descriptors in
        let commands = List.rev !commands in
        let size = List.length commands * 4 in
        if size / 4 > 0x1fffff then invalid_arg "NV command stream exceeds its FIFO entry";
        let suffix = if compute then "compute" else "copy" in
        let buffer = U.placeholder ~shape:[size] ~dtype:D.uint8 ~slot:(U.fresh_buffer_slot ())
            ~device:(U.Single name) () |> U.with_tag ("cmdbuf_" ^ suffix) in
        let buffer = Tolk.Hcq2.patch ~blob:(String.make size '\000') ~after:(dependency :: patches)
            buffer (List.mapi (fun i value -> i * 4, value) commands) in
        let entries = if compute then compute_entries else copy_entries in
        let ring = placeholder ~volatile:true name ("ring_" ^ suffix) D.uint64 entries in
        let put = placeholder ~volatile:true name ("gpput_" ^ suffix) D.uint32 1 in
        let doorbell = placeholder ~volatile:true name "doorbell" D.uint32 1 in
        let entry = bor (addr name buffer) (u64 (((size / 4) lsl 42) lor (1 lsl 41))) in
        Some (Tolk.Hcq2.ccall ~host:name ~after:[buffer] ~name:"tolk_hcq_gpfifo" ~dtype:D.void
          [index (context name); index ring; index put; index doorbell; entry;
           u32 (if compute then compute_token else copy_token); u32 entries])
    | _ -> None
end

let ensure_has_local_memory (dev : 'meta device) ~alloc ~free ~num_gpcs
    ~num_tpc_per_gpc ~num_sm_per_tpc ~max_warps_per_sm ~tl ~queue required =
  if dev.slm_per_thread < required then begin
    Hcq.Timeline.prepare tl;
    let slm_per_thread = round_up required 32 in
    let bytes_per_tpc =
      round_up (round_up (slm_per_thread * 32) 0x200
        * max_warps_per_sm * num_sm_per_tpc) 0x8000 in
    let old = dev.shader_local_mem in
    let shader_local_mem =
      alloc (round_up (bytes_per_tpc * num_tpc_per_gpc * num_gpcs) 0x20000) in
    dev.slm_per_thread <- slm_per_thread;
    dev.shader_local_mem <- Some shader_local_mem;
    let cq = Compute_queue.create dev in
    Compute_queue.wait cq
      ~value:(Hcq.Timeline.submitted tl)
      tl.Hcq.Timeline.timeline;
    Compute_queue.setup cq
      ~local_mem:(Hcq.Buffer.va shader_local_mem)
      ~local_mem_tpc_bytes:bytes_per_tpc ();
    Compute_queue.signal cq
      ~value:(Hcq.Timeline.next_timeline tl)
      tl.Hcq.Timeline.timeline;
    Compute_queue.submit cq queue;
    Option.iter free old
  end

(* Device runtime *)

module Timeline = Hcq.Timeline

(* The chip's shader ISA identity, derived from the reported SM version:
   the generation and revision digits name the architecture, and the same
   two components packed into a byte are the ISA revision launch
   descriptors carry. The 0xa04 report names sm_120. *)
let arch_of_sm_version sm_version =
  if sm_version = 0xa04 then "sm_120"
  else
    let v = sm_version land 0xff in
    Printf.sprintf "sm_%d%d"
      ((sm_version lsr 8) land 0xff)
      (if v > 0xf then v lsr 4 else v)

let sass_of_sm_version sm_version =
  ((sm_version land 0xf00) lsr 4) lor (sm_version land 0xf)

let query_gpu_info (iface : 'mem Nv_iface.t) ~subdevice indices =
  match iface.Nv_iface.nvdev with
  | Some _ ->
      (* an interface programming the hardware directly answers from the
         static engine information of the first engine *)
      let module P = Defs.Nv2080_ctrl_internal_static_gr_get_info_params in
      let module I = Defs.Nv2080_ctrl_internal_static_gr_info in
      let b = Nv_tables.create_blob P.sizeof in
      iface.Nv_iface.rm_control ~obj:subdevice
        ~cmd:Defs.nv2080_ctrl_cmd_internal_static_kgr_get_info ~params:b ();
      List.map
        (fun idx ->
          let base =
            P.engineinfo_offset + I.infolist_offset
            + (idx * I.infolist_elem_size)
          in
          Nv_tables.get_field ~base b Defs.Nv2080_ctrl_internal_gr_info.data)
        indices
  | None ->
      let module I = Defs.Nv2080_ctrl_gr_info in
      let n = List.length indices in
      let infos = Nv_tables.create_blob (n * I.sizeof) in
      List.iteri
        (fun i idx ->
          Nv_tables.set_field ~base:(i * I.sizeof) infos I.index idx)
        indices;
      let module P = Defs.Nv2080_ctrl_gr_get_info_params in
      let b = Nv_tables.create_blob P.sizeof in
      Nv_tables.set_field b P.grinfolistsize n;
      Nv_tables.set_field b P.grinfolist
        (Nativeint.to_int (Nv_tables.blob_addr infos));
      iface.Nv_iface.rm_control ~obj:subdevice
        ~cmd:Defs.nv2080_ctrl_cmd_gr_get_info ~params:b ();
      List.mapi
        (fun i _ -> Nv_tables.get_field ~base:(i * I.sizeof) infos I.data)
        indices

(* One mapped hardware channel: an error notifier, the channel object over
   its slice of the shared ring area, the engine object bound to it, the
   work-submission token, and the memory-manager registration. The compute
   channel also creates the debugger objects fault reports are read
   through; their handles are returned alongside the descriptor. *)
let new_gpfifo (iface : 'mem Nv_iface.t) ~(usermode : Nv_iface.usermode) ~nvdevice
    ~gpfifo_area ~ctxshare ~channel_group ~offset ~entries ~compute =
  let notifier = iface.Nv_iface.alloc ~uncached:true (48 lsl 20) in
  let open Nv_defs_versions in
  let p = iface.Nv_iface.defs.nv_channelgpfifo_allocation_parameters in
  let b = Nv_tables.create_blob p.sizeof in
  Nv_tables.set_field b p.gpfifooffset
    (Nativeint.to_int (Hcq.Buffer.va gpfifo_area) + offset);
  Nv_tables.set_field b p.gpfifoentries entries;
  Nv_tables.set_field b p.hcontextshare ctxshare;
  Nv_tables.set_field b p.hobjecterror
    (iface.Nv_iface.hmemory notifier);
  Nv_tables.set_field b p.hobjectbuffer
    (iface.Nv_iface.hmemory gpfifo_area);
  Nv_tables.set_field b p.huserdmemory
    (iface.Nv_iface.hmemory gpfifo_area);
  Nv_tables.set_field b p.userdoffset ((entries * 8) + offset);
  Nv_tables.set_field b p.enginetype 0;
  let gpfifo =
    iface.Nv_iface.rm_alloc ~parent:channel_group
      ~cls:usermode.Nv_iface.gpfifo_class ~params:b ()
  in
  let debug =
    if compute then begin
      let debug_compute_obj =
        iface.Nv_iface.rm_alloc ~parent:gpfifo
          ~cls:usermode.Nv_iface.compute_class ()
      in
      let module D = Defs.Nv83de_alloc_parameters in
      let db = Nv_tables.create_blob D.sizeof in
      Nv_tables.set_field db D.happclient iface.Nv_iface.root;
      Nv_tables.set_field db D.hclass3dobject debug_compute_obj;
      Some
        ( iface.Nv_iface.rm_alloc ~parent:nvdevice ~cls:Defs.gt200_debugger
            ~params:db (),
          gpfifo )
    end
    else begin
      let (_ : int) =
        iface.Nv_iface.rm_alloc ~parent:gpfifo
          ~cls:usermode.Nv_iface.dma_class ()
      in
      None
    end
  in
  let module W = Defs.Nvc36f_ctrl_cmd_gpfifo_get_work_submit_token_params in
  let wb = Nv_tables.create_blob W.sizeof in
  Nv_tables.set_field wb W.worksubmittoken (-1);
  iface.Nv_iface.rm_control ~obj:gpfifo
    ~cmd:Defs.nvc36f_ctrl_cmd_gpfifo_get_work_submit_token ~params:wb ();
  iface.Nv_iface.setup_gpfifo_vm ~gpfifo;
  let area = Hcq.Buffer.cpu_view gpfifo_area in
  ( {
      Queue_desc.ring = Hcq.Mmio.view area ~off:offset ~size:(entries * 8) ();
      gpput =
        Hcq.Mmio.view area
          ~off:(offset + (entries * 8) + Defs.ampere_a_control_gpfifo_gpput)
          ~size:4 ();
      token = Nv_tables.get_field wb W.worksubmittoken;
    },
    debug )

(* Fault reports: the per-SM error states read through the debugger, and
   when they record an MMU fault, its address, type and access decoded by
   name. *)
let on_device_hang (iface : 'mem Nv_iface.t) ~debugger ~debug_channel () =
  let report = ref [] in
  let add line = report := line :: !report in
  let module P = Defs.Nv83de_ctrl_debug_read_all_sm_error_states_params in
  let b = Nv_tables.create_blob P.sizeof in
  Nv_tables.set_field b P.htargetchannel debug_channel;
  Nv_tables.set_field b P.numsmstoread 100;
  iface.Nv_iface.rm_control ~obj:debugger
    ~cmd:Defs.nv83de_ctrl_cmd_debug_read_all_sm_error_states ~params:b ();
  if Nv_tables.get_field b P.mmufault_valid <> 0 then begin
    let module M = Defs.Nv83de_ctrl_debug_read_mmu_fault_info_params in
    let module E = Defs.Nv83de_ctrl_debug_read_mmu_fault_info_entry in
    let mb = Nv_tables.create_blob M.sizeof in
    iface.Nv_iface.rm_control ~obj:debugger
      ~cmd:Defs.nv83de_ctrl_cmd_debug_read_mmu_fault_info ~params:mb ();
    let name table id =
      match List.assoc_opt id table with
      | Some n -> n
      | None -> string_of_int id
    in
    for i = 0 to Nv_tables.get_field mb M.count - 1 do
      let base = M.mmufaultinfolist_offset + (i * E.sizeof) in
      add
        (Printf.sprintf "MMU fault: 0x%X | %s | %s"
           (Nv_tables.get_field ~base mb E.faultaddress)
           (name Defs.nv_pfault_fault_type
              (Nv_tables.get_field ~base mb E.faulttype))
           (name Defs.nv_pfault_access_type
              (Nv_tables.get_field ~base mb E.accesstype)))
    done
  end
  else begin
    let module R = Defs.Nv83de_sm_error_state_registers in
    for i = 0 to P.smerrorstatearray_count - 1 do
      let base = P.smerrorstatearray_offset + (i * R.sizeof) in
      let esr = Nv_tables.get_field ~base b R.hwwglobalesr in
      let warp_esr = Nv_tables.get_field ~base b R.hwwwarpesr in
      if esr <> 0 || warp_esr <> 0 then
        add
          (Printf.sprintf "SM %d fault: esr=%d warp_esr=0x%x warp_pc=0x%x" i
             esr warp_esr
             (Nv_tables.get_field ~base b R.hwwwarpesrpc64))
    done
  end;
  failwith (String.concat "\n" (List.rev !report))

module State = struct
  type 'mem t = {
    name : string;
    iface : 'mem Nv_iface.t;
    hw : 'mem device;
    subdevice : int;
    compute_queue : Queue_desc.t;
    dma_queue : Queue_desc.t;
    kernargs : 'mem Hcq.Kernargs.t;
    pool : 'mem Hcq.Signal.Pool.t;
    tl : ('mem, 'mem device) Timeline.t;
    submission : Hcq.Submission.t;
    num_gpcs : int;
    num_tpc_per_gpc : int;
    num_sm_per_tpc : int;
    max_warps_per_sm : int;
    (* The device's LRU-wrapped allocator; set right after creation and used
       for local-memory sizing. *)
    mutable allocator :
      'mem Hcq.Buffer.t Tolk.Device.Allocator.t option;
  }

  let check_submission t =
    Timeline.guarded_wait t.tl (fun () -> Hcq.Submission.check t.submission)

  let prepare t =
    check_submission t;
    Timeline.prepare t.tl;
    Hcq.Submission.prepare ~timeout_ms:(Tolk.Helpers.getenv "HCQ_TIMEOUT_MS" 30000)
      t.submission

  let synchronize t =
    check_submission t;
    Timeline.synchronize t.tl

  let invalidate_caches t =
    if Nv_iface.is_nvd t.iface then
      t.iface.Nv_iface.rm_control ~obj:t.subdevice
        ~cmd:Defs.nv2080_ctrl_cmd_internal_bus_flush_with_sysmembar ()
    else begin
      let module P = Defs.Nv2080_ctrl_fb_flush_gpu_cache_params in
      let b = Nv_tables.create_blob P.sizeof in
      Nv_tables.set_field b P.flags
        ((Defs.nv2080_ctrl_fb_flush_gpu_cache_flags_write_back_yes lsl 2)
        lor (Defs.nv2080_ctrl_fb_flush_gpu_cache_flags_invalidate_yes lsl 3)
        lor (Defs.nv2080_ctrl_fb_flush_gpu_cache_flags_flush_mode_full_cache
            lsl 4));
      t.iface.Nv_iface.rm_control ~obj:t.subdevice
        ~cmd:Defs.nv2080_ctrl_cmd_fb_flush_gpu_cache ~params:b ()
    end
end

module Allocator = struct
  (* One DMA stream ordered against the device timeline: wait for the last
     submitted work, append the packets of [build], advance the timeline. *)
  let submit_copy state build =
    let tl = state.State.tl in
    State.prepare state;
    let cp = Copy_queue.create state.State.hw in
    Copy_queue.wait cp
      ~value:(Timeline.submitted tl)
      tl.Timeline.timeline;
    build cp;
    Copy_queue.signal cp ~value:(Timeline.next_timeline tl) tl.Timeline.timeline;
    Copy_queue.submit cp state.State.dma_queue

  let submit_chunk state ~dest ~src len =
    submit_copy state (fun cp -> Copy_queue.copy cp ~dest ~src len)

  let copyin state buf bytes =
    Timeline.copyin state.State.tl ~submit_chunk:(submit_chunk state) buf bytes

  let copyout state bytes buf =
    State.synchronize state;
    Timeline.copyout state.State.tl ~submit_chunk:(submit_chunk state) bytes
      buf

  let transfer state ~dest ~src ~dest_device ~src_device nbytes =
    if Tolk.Device.canonicalize dest_device <> Tolk.Device.canonicalize src_device then false
    else begin
      submit_copy state (fun cp -> Copy_queue.copy cp ~dest ~src nbytes);
      true
    end

  let raw state =
    let alloc size (spec : Tolk.Device.Buffer_spec.t) =
      match spec.external_ptr with
      | Some _ -> invalid_arg "NV buffers cannot adopt an external pointer"
      | None ->
          state.State.iface.Nv_iface.alloc ~host:spec.host
            ~cpu_access:spec.cpu_access size
    in
    (* A queued kernel may still use the memory. *)
    let free buf _size (_ : Tolk.Device.Buffer_spec.t) =
      State.synchronize state;
      state.State.iface.Nv_iface.free buf
    in
    let offset buf size byte_offset =
      Hcq.Buffer.offset buf ~off:byte_offset ~size ()
    in
    {
      Tolk.Device.Allocator.kind = state.State.iface.Nv_iface.kind;
      host = (fun buf -> Option.map Hcq.Mmio.addr (Hcq.Buffer.view buf));
      mapping = Some {
        map = state.State.iface.Nv_iface.map;
        unmap = (fun b -> State.synchronize state;
          state.State.iface.Nv_iface.unmap b);
      };
      synchronize = (fun () -> State.synchronize state);
      alloc;
      free;
      copyin = copyin state;
      copyout = copyout state;
      addr = Some Hcq.Buffer.va;
      offset = Some offset;
      transfer = Some (transfer state);
      supports_transfer = true;
      copy_from_disk = None;
      supports_copy_from_disk = false;
    }

  let create state =
    let allocator = Tolk.Device.Lru_allocator.wrap (raw state) in
    state.State.allocator <- Some allocator;
    Tolk.Device.Allocator.Pack allocator
end

module Runtime = struct
  (* Local-memory backing goes through the LRU allocator so resizes reuse
     freed device memory. *)
  let ensure_local_memory state size =
    State.check_submission state;
    let allocator = Option.get state.State.allocator in
    ensure_has_local_memory state.State.hw
      ~alloc:(fun size ->
        allocator.Tolk.Device.Allocator.alloc size
          Tolk.Device.Buffer_spec.default)
      ~free:(fun buf ->
        allocator.Tolk.Device.Allocator.free buf (Hcq.Buffer.size buf)
          Tolk.Device.Buffer_spec.default)
      ~num_gpcs:state.State.num_gpcs
      ~num_tpc_per_gpc:state.State.num_tpc_per_gpc
      ~num_sm_per_tpc:state.State.num_sm_per_tpc
      ~max_warps_per_sm:state.State.max_warps_per_sm ~tl:state.State.tl
      ~queue:state.State.compute_queue size

  let default_local = [| 1; 1; 1 |]

  let runtime state (obj : Tolk_uop.Tiny_elf.t) =
    let name = obj.name and lib = obj.lib in
    let layout = Tolk_uop.Tiny_elf.layout obj.signature in
    let prg =
      Program.load state.State.hw
        ~alloc:(fun size ->
          state.State.iface.Nv_iface.alloc ~cpu_access:true size)
        ~ensure_local_memory:(ensure_local_memory state) ~name lib
    in
    let call bufs ~global ~local ~vals ~wait ~timeout:_ =
      let bufs = Array.map (fun buf ->
          match Tolk.Device.Buffer.get ~device:state.State.name state.State.iface.Nv_iface.kind buf with
          | Some raw -> Hcq.Buffer.va raw | None -> 0n) bufs in
      let local = Option.value local ~default:default_local in
      let tl = state.State.tl in
      State.check_submission state;
      let timeline_value = Timeline.next_timeline tl in
      let launch ?timing () =
        Program.call prg ~layout ~kernargs:state.State.kernargs
          ~queue:state.State.compute_queue ~timeline:tl.Timeline.timeline
          ~timeline_value ?wait:timing ~bufs ~vals
          ~global_size:(global.(0), global.(1), global.(2))
          ~local_size:(local.(0), local.(1), local.(2))
          ()
      in
      if not wait then launch ()
      else begin
        (match tl.Timeline.error_state with Some e -> raise e | None -> ());
        let st_slot = Hcq.Signal.Pool.get state.State.pool in
        let en_slot = Hcq.Signal.Pool.get state.State.pool in
        Fun.protect
          ~finally:(fun () ->
            Hcq.Signal.Pool.put state.State.pool en_slot;
            Hcq.Signal.Pool.put state.State.pool st_slot)
          (fun () ->
            let st = Hcq.Signal.make st_slot in
            let en = Hcq.Signal.make en_slot in
            Timeline.guarded_wait tl (fun () -> launch ~timing:(st, en) ()))
      end
    in
    let free () = Program.free ~free:state.State.iface.Nv_iface.free prg in
    { Tolk.Device.call; free; handle = 0n }
end

module Queue = struct
  open Tolk
  open Tolk_uop
  module U = Uop
  module B = Device.Buffer

  let bufferize state u =
    let name = state.State.name in
    let size = U.max_numel u and dtype = U.dtype u in
    let borrow_view view =
      let allocator = Storage.Host_allocator.make ~synchronize:(fun () -> State.synchronize state) in
      let spec = {Device.Buffer_spec.default with external_ptr = Some (Hcq.Mmio.addr view); nolru = true} in
      B.create ~device:"CPU" ~size ~dtype ~spec (Device.Allocator.Pack allocator) in
    let borrowed raw =
      let allocator = { (Allocator.raw state) with
        alloc = (fun _ _ -> raw); free = (fun _ _ _ -> State.synchronize state) } in
      B.create ~device:name ~size ~dtype
        ~spec:{Device.Buffer_spec.default with nolru = true} (Device.Allocator.Pack allocator) in
    let allocate ?(host = false) ?(cpu_access = true) () =
      let spec = {Device.Buffer_spec.default with host; cpu_access; nolru = true} in
      B.create ~device:name ~size ~dtype ~spec (Device.Allocator.Pack (Allocator.raw state)) in
    match U.as_param u with
    | Some {param = {allocation = Some ("hcq_submission", _); _}; _} ->
        Some (Hcq.Submission.buffer state.State.submission)
    | Some {param = {allocation = Some ("cfunc", data); _}; _} ->
        let libs, symbol = (Marshal.from_string data 0 : string list * string) in
        if libs <> [] then invalid_arg "NV host helpers do not load libraries";
        let allocator = Storage.Host_allocator.make ~synchronize:(fun () -> ()) in
        let b = B.create ~device:"CPU" ~size:1 ~dtype:Dtype.uint64 (Device.Allocator.Pack allocator) in
        let bytes = Bytes.create 8 in
        Bytes.set_int64_le bytes 0 (Int64.of_nativeint (Hcq.Submission.symbol symbol));
        B.ensure_allocated b; B.copyin b bytes; Some b
    | Some {param = {allocation = Some ("nv_image", requested); _}; _} ->
        Runtime.ensure_local_memory state (int_of_string requested);
        Some (allocate ())
    | Some _ ->
        (match U.node_tag u with
         | Some "timeline" -> Some (borrowed (Hcq.Signal.buf state.State.tl.Timeline.timeline))
         | Some "slots" -> Some (allocate ~host:true ())
         | Some ("qmd" | "cmdbuf_compute" | "cmdbuf_copy") -> Some (allocate ())
         | Some "doorbell" -> Some (borrow_view (Hcq.Mmio.view state.State.hw.gpu_mmio ~off:0x90 ~size:4 ()))
         | Some tag ->
             let descriptor, suffix = if Filename.check_suffix tag "_compute" then
                 Some state.State.compute_queue, "_compute"
               else if Filename.check_suffix tag "_copy" then Some state.State.dma_queue, "_copy"
               else None, "" in
             Option.bind descriptor (fun q ->
                 let field = String.sub tag 0 (String.length tag - String.length suffix) in
                 Option.map borrow_view (match field with
                   | "ring" -> Some q.Queue_desc.ring
                   | "gpput" -> Some q.Queue_desc.gpput
                   | _ -> None))
         | None -> None)
    | None -> None

  let create state =
    let host = try Device.get "CPU" with Failure _ -> Tolk_cpu.create "CPU" in
    let copy call = match U.as_call call with
      | Some {args; _} -> List.for_all (fun arg ->
          U.device_of arg = Some (U.Single state.State.name)) args
      | None -> false in
    Device.{timestamp_divider = 1000.; prepare = (fun () -> State.prepare state); host = Device.name host; copy;
      encode = Encoded_queue.encode state.State.hw ~name:state.State.name
        ~compute_entries:(Hcq.Mmio.size state.State.compute_queue.Queue_desc.ring / 8)
        ~copy_entries:(Hcq.Mmio.size state.State.dma_queue.Queue_desc.ring / 8)
        ~compute_token:state.State.compute_queue.Queue_desc.token
        ~copy_token:state.State.dma_queue.Queue_desc.token;
      lower = Encoded_queue.lower state.State.name;
      compile = Codegen.to_program ~optimize:false host (Device.renderer host)}
end

(* The shared device open path over the selected interface: everything from
   object allocation through channel set-up and renderer wiring is
   interface-independent. *)
let open_device ~name (iface : 'mem Nv_iface.t) =
  let module D = Defs.Nv0080_alloc_parameters in
  let db = Nv_tables.create_blob D.sizeof in
  Nv_tables.set_field db D.deviceid iface.Nv_iface.gpu_instance;
  Nv_tables.set_field db D.hclientshare iface.Nv_iface.root;
  Nv_tables.set_field db D.vamode
    Defs.nv_device_allocation_vamode_optional_multiple_vaspaces;
  let nvdevice =
    iface.Nv_iface.rm_alloc ~parent:iface.Nv_iface.root ~cls:Defs.nv01_device_0
      ~params:db ()
  in
  let subdevice =
    iface.Nv_iface.rm_alloc ~parent:nvdevice ~cls:Defs.nv20_subdevice_0
      ~params:(Nv_tables.create_blob Defs.Nv2080_alloc_parameters.sizeof)
      ()
  in
  let module V = Defs.Nv_memory_virtual_allocation_params in
  let vb = Nv_tables.create_blob V.sizeof in
  Nv_tables.set_field vb V.limit 0x1ffffffffffff;
  let virtmem =
    iface.Nv_iface.rm_alloc ~parent:nvdevice ~cls:Defs.nv01_memory_virtual
      ~params:vb ()
  in
  iface.Nv_iface.set_device ~nvdevice ~subdevice ~virtmem;
  let usermode = iface.Nv_iface.setup_usermode () in
  let module B = Defs.Nv2080_ctrl_perf_boost_params in
  let bb = Nv_tables.create_blob B.sizeof in
  Nv_tables.set_field bb B.duration 0xffffffff;
  Nv_tables.set_field bb B.flags
    ((Defs.nv2080_ctrl_perf_boost_flags_cuda_yes lsl 4)
    lor (Defs.nv2080_ctrl_perf_boost_flags_cuda_priority_high lsl 6)
    lor Defs.nv2080_ctrl_perf_boost_flags_cmd_boost_to_max);
  iface.Nv_iface.rm_control ~obj:subdevice ~cmd:Defs.nv2080_ctrl_cmd_perf_boost
    ~params:bb ();
  let open Nv_defs_versions in
  let vp = iface.Nv_iface.defs.nv_vaspace_allocation_parameters in
  let vsb = Nv_tables.create_blob vp.sizeof in
  Nv_tables.set_field vsb vp.vabase 0x1000;
  Nv_tables.set_field vsb vp.vasize 0x1fffffb000000;
  Nv_tables.set_field vsb vp.flags
    (Defs.nv_vaspace_allocation_flags_enable_page_faulting
    lor Defs.nv_vaspace_allocation_flags_is_externally_owned);
  let vaspace =
    iface.Nv_iface.rm_alloc ~parent:nvdevice ~cls:Defs.fermi_vaspace_a
      ~params:vsb ()
  in
  iface.Nv_iface.setup_vm ~vaspace;
  let module C = Defs.Nv_channel_group_allocation_parameters in
  let cb = Nv_tables.create_blob C.sizeof in
  Nv_tables.set_field cb C.enginetype Defs.nv2080_engine_type_graphics;
  let channel_group =
    iface.Nv_iface.rm_alloc ~parent:nvdevice ~cls:Defs.kepler_channel_group_a
      ~params:cb ()
  in
  let gpfifo_area =
    iface.Nv_iface.alloc ~contiguous:true ~cpu_access:true
      ~map_flags:(Defs.nvos33_flags_caching_type_writecombined lsl 23)
      0x300000
  in
  let module X = Defs.Nv_ctxshare_allocation_parameters in
  let xb = Nv_tables.create_blob X.sizeof in
  Nv_tables.set_field xb X.hvaspace vaspace;
  Nv_tables.set_field xb X.flags
    Defs.nv_ctxshare_allocation_flags_subcontext_async;
  let ctxshare =
    iface.Nv_iface.rm_alloc ~parent:channel_group
      ~cls:Defs.fermi_context_share_a ~params:xb ()
  in
  let compute_queue, debug =
    new_gpfifo iface ~usermode ~nvdevice ~gpfifo_area ~ctxshare ~channel_group
      ~offset:0 ~entries:0x10000 ~compute:true
  in
  let dma_queue, _ =
    new_gpfifo iface ~usermode ~nvdevice ~gpfifo_area ~ctxshare ~channel_group
      ~offset:0x100000 ~entries:0x10000 ~compute:false
  in
  let debugger, debug_channel = Option.get debug in
  let sp = iface.Nv_iface.defs.nva06c_ctrl_gpfifo_schedule_params in
  let sb = Nv_tables.create_blob sp.sizeof in
  Nv_tables.set_field sb sp.benable 1;
  iface.Nv_iface.rm_control ~obj:channel_group
    ~cmd:Defs.nva06c_ctrl_cmd_gpfifo_schedule ~params:sb ();
  let cmdq_page = iface.Nv_iface.alloc ~cpu_access:true 0x200000 in
  let num_gpcs, num_tpc_per_gpc, num_sm_per_tpc, max_warps_per_sm, sm_version =
    match
      query_gpu_info iface ~subdevice
        [
          Defs.nv2080_ctrl_gr_info_index_litter_num_gpcs;
          Defs.nv2080_ctrl_gr_info_index_litter_num_tpc_per_gpc;
          Defs.nv2080_ctrl_gr_info_index_litter_num_sm_per_tpc;
          Defs.nv2080_ctrl_gr_info_index_max_warps_per_sm;
          Defs.nv2080_ctrl_gr_info_index_sm_version;
        ]
    with
    | [ a; b; c; d; e ] -> (a, b, c, d, e)
    | _ -> assert false
  in
  let arch = arch_of_sm_version sm_version in
  let hw =
    device ~compute_class:usermode.Nv_iface.compute_class
      ~dma_class:usermode.Nv_iface.dma_class
      ~gpfifo_class:usermode.Nv_iface.gpfifo_class
      ~sass_version:(sass_of_sm_version sm_version)
      ~shared_mem_window:0x729400000000n ~local_mem_window:0x729300000000n
      ~cmdq_page ~gpu_mmio:usermode.Nv_iface.mmio ()
  in
  let pool =
    Hcq.Signal.Pool.create ~alloc_page:(fun () ->
        iface.Nv_iface.alloc ~host:true ~uncached:true ~cpu_access:true 0x1000)
  in
  let timeline_signal () =
    Hcq.Signal.make ~is_timeline:true ~sleep:iface.Nv_iface.sleep ~owner:hw
      (Hcq.Signal.Pool.get pool)
  in
  let bounce_count = 32 and bounce_size = 2 lsl 20 in
  let state =
    {
      State.name = name;
      submission = Hcq.Submission.create ();
      iface;
      hw;
      subdevice;
      compute_queue;
      dma_queue;
      kernargs =
        Hcq.Kernargs.create
          (iface.Nv_iface.alloc ~cpu_access:true (16 lsl 20));
      pool;
      tl =
        {
          Timeline.timeline = timeline_signal ();

          error_state = None;
          bounce =
            Array.init bounce_count (fun _ ->
                iface.Nv_iface.alloc ~host:true bounce_size);
          bounce_timeline = Array.make bounce_count 0;
          bounce_next = 0;
          on_hang = on_device_hang iface ~debugger ~debug_channel;
        };
      num_gpcs;
      num_tpc_per_gpc;
      num_sm_per_tpc;
      max_warps_per_sm;
      allocator = None;
    }
  in
  (* the initial queue set-up: bind the engine classes and the memory
     windows to the fresh channels, ordered on the timeline *)
  let tl = state.State.tl in
  let cq = Compute_queue.create hw in
  Compute_queue.setup cq ~compute_class:usermode.Nv_iface.compute_class
    ~local_mem_window:hw.local_mem_window
    ~shared_mem_window:hw.shared_mem_window ();
  Compute_queue.signal cq ~value:(Timeline.next_timeline tl)
    tl.Timeline.timeline;
  Compute_queue.submit cq compute_queue;
  let cp = Copy_queue.create hw in
  Copy_queue.wait cp ~value:(Timeline.submitted tl)
    tl.Timeline.timeline;
  Copy_queue.setup cp ~copy_class:usermode.Nv_iface.dma_class ();
  Copy_queue.signal cp ~value:(Timeline.next_timeline tl) tl.Timeline.timeline;
  Copy_queue.submit cp dma_queue;
  Timeline.synchronize tl;
  at_exit (fun () ->
      (* finalize even when the device faulted, so shutdown still reaches
         the driver *)
      (try State.synchronize state
       with e ->
         Printf.eprintf "%s synchronization failed before finalizing: %s\n%!"
           name (Printexc.to_string e));
      iface.Nv_iface.device_fini ());
  let allocator = Allocator.create state in
  let renderer_set = Tolk.Device.Renderer_set.make ~device:name ~arch
      [ "CUDA", (fun target ->
          let arch = match Tolk.Gpu_target.parse_cuda_arch target.Tolk_uop.Target.arch with
            | Some arch -> arch
            | None -> invalid_arg ("unsupported NV architecture: " ^ target.arch) in
          Tolk.Renderer.with_compiler
            (Tolk_nvrtc.Compiler_nvrtc.create ~ptx:false ~cache_key:"nv" target.arch)
            (Tolk.Cstyle.cuda ~device:"NV" arch)) ] in
  Tolk.Device.make ~name ~allocator ~renderer_set
    ~runtime:(Runtime.runtime state)
    ~synchronize:(fun () -> State.synchronize state)
    ~invalidate_caches:(fun () -> State.invalidate_caches state)
    ~queue:(Queue.create state) ~bufferize:(Queue.bufferize state) ()

let create name =
  let device_id =
    match String.index_opt name ':' with
    | Some i -> (
        let suffix = String.sub name (i + 1) (String.length name - i - 1) in
        match int_of_string_opt suffix with
        | Some id -> id
        | None -> invalid_arg (Printf.sprintf "invalid NV device %S" name))
    | None -> 0
  in
  let nvk () = Nv_iface.Pack (Nvk_iface.iface ~device_id) in
  let pci () = Nv_iface.Pack (Pci_iface.iface (Pci_iface.create ~device_id)) in
  let Nv_iface.Pack iface = Tolk.Helpers.select_interface ~device:name
      [ "NVK", nvk; "PCI", pci ] in
  open_device ~name iface
