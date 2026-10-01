(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
module G = Nv_gpu

let u32 = Hcq2.Queue.dword
let u64 n = int ~dtype:Dtype.Uint64 n
let hilo a = [ cast (shr a (int 32)) Dtype.Uint32; cast a Dtype.Uint32 ]
let bits (_, lo) v = v lsl lo
let binary s = v Op.Binary ~arg:(Bytes s)

let nvm ?(typ = 2) subc mthd vals =
  let n =
    List.fold_left (fun n v -> n + (Dtype.itemsize (dtype v) / 4)) 0 vals
  in
  u32 ((typ lsl 28) lor (n lsl 16) lor (subc lsl 13) lor (mthd lsr 2)) :: vals

(* A region of [blob], with [patches] over it at their byte offsets. *)
let region name blob patches =
  let patches = List.sort (fun (a, _) (b, _) -> Int.compare a b) patches in
  let bytes pos stop =
    if stop > pos then [ binary (String.sub blob pos (stop - pos)) ] else []
  in
  let rec words pos = function
    | [] -> bytes pos (String.length blob)
    | (off, w) :: rest ->
        bytes pos off @ (w :: words (off + Dtype.itemsize (dtype w)) rest)
  in
  v Op.Linear ~src:(words 0 patches) ~arg:(Region { name; align = 256 })

(* Launch descriptors *)

type channel = { entries : int; token : int }

type props = {
  compute_class : int;
  sass_version : int;
  shared_window : int;
  local_window : int;
  compute : channel;
  copy : channel;
}

module Qmd = struct
  type t = {
    ver : int;
    fields : (string * (int * int)) list;
    mv : Bytes.t;
    mutable patches : (int * Ops.t) list; (* by byte offset, the latest kept *)
  }

  let make props =
    let ver, sz, fields =
      if props.compute_class >= G.blackwell_compute_a then (5, 0x60, G.qmd_v5)
      else (3, 0x40, G.qmd_v3)
    in
    { ver; fields; mv = Bytes.make (sz * 4) '\000'; patches = [] }

  let copy q = { q with mv = Bytes.copy q.mv }

  let range q k =
    match List.assoc_opt k q.fields with
    | Some r -> r
    | None -> invalid_arg ("no field " ^ k ^ " in a launch descriptor")

  let number q lo hi =
    let n = ref 0 in
    for i = hi / 8 downto lo / 8 do
      n := (!n lsl 8) lor Char.code (Bytes.get q.mv i)
    done;
    !n

  let read q k =
    let hi, lo = range q k in
    (number q lo hi lsr (lo mod 8)) land ((1 lsl (hi - lo + 1)) - 1)

  let write q k v =
    let hi, lo = range q k in
    if v lsr (hi - lo + 1) <> 0 then
      invalid_arg (Printf.sprintf "%s=0x%x does not fit" k v);
    let mask = ((1 lsl (hi - lo + 1)) - 1) lsl (lo mod 8) in
    let n = number q lo hi land lnot mask lor (v lsl (lo mod 8)) in
    for i = lo / 8 to hi / 8 do
      Bytes.set q.mv i (Char.chr ((n lsr (8 * (i - (lo / 8)))) land 0xff))
    done

  (* A word written when the batch is linked or run, of the widest unsigned type
     the field holds. *)
  let patch q k u =
    let hi, lo = range q k in
    if lo mod 8 <> 0 then invalid_arg (k ^ " is not byte aligned");
    let t =
      List.find
        (fun t -> Dtype.itemsize t * 8 <= hi - lo + 1)
        Dtype.[ Uint64; Uint32; Uint16; Uint8 ]
    in
    q.patches <- (lo / 8, ccast u t) :: List.remove_assoc (lo / 8) q.patches

  let set_addr q name ?(sfx = "") addr =
    patch q (name ^ "_lower" ^ sfx) addr;
    patch q (name ^ "_upper" ^ sfx) (shr addr (int 32))

  let set_constant_buf_addr q i addr =
    let v4 = q.ver >= 4 in
    set_addr q "constant_buffer_addr"
      ~sfx:(Printf.sprintf (if v4 then "_shifted6_%d" else "_%d") i)
      (shr addr (int (if v4 then 6 else 0)))

  let set_program_addr q addr =
    let v4 = q.ver >= 4 in
    set_addr q "program_address"
      ~sfx:(if v4 then "_shifted4" else "")
      (shr addr (int (if v4 then 4 else 0)));
    set_addr q "program_prefetch_addr" ~sfx:"_shifted" (shr addr (int 8))

  (* One of the two releases once the launch completes, if one is free. *)
  let set_release q addr payload ~timestamp =
    match
      List.find_opt
        (fun i -> read q (Printf.sprintf "release%d_enable" i) = 0)
        [ 0; 1 ]
    with
    | None -> false
    | Some i ->
        let v4 = q.ver >= 4 in
        let name s = Printf.sprintf s i in
        set_addr q
          (if v4 then name "release_semaphore%d_addr"
           else name "release%d_address")
          addr;
        set_addr q
          (if v4 then name "release_semaphore%d_payload"
           else name "release%d_payload")
          payload;
        write q (name "release%d_enable") 1;
        write q
          (if v4 then name "release_structure_size_%d"
           else name "release%d_structure_size")
          (if timestamp then 0 else 2);
        if not v4 then write q (name "release%d_payload64b") 1;
        true

  let grid q =
    if q.ver >= 4 then [ "grid_width"; "grid_height"; "grid_depth" ]
    else [ "cta_raster_width"; "cta_raster_height"; "cta_raster_depth" ]

  let blob q = Bytes.to_string q.mv
end

(* Programs *)

type program = {
  constbufs : (int * (int * int)) list; (* index, (offset in the image, size) *)
  prog_off : int;
  cbuf_0 : int array; (* the driver's parameters in constant buffer 0 *)
  vars : Dtype.t list;
  kernargs_size : int;
  local_bytes : int;
  qmd : Qmd.t; (* the template of the program's launches *)
  max_threads : int;
}

(* The attributes of an .nv.info section: (type, parameter, data or size). *)
let elf_info (sh : Nx_device_elf.section) =
  let rec go off =
    if off >= sh.size then []
    else
      let typ = Char.code sh.contents.[off]
      and param = Char.code sh.contents.[off + 1]
      and sz = String.get_uint16_le sh.contents (off + 2) in
      let data = if typ = 4 then String.sub sh.contents (off + 4) sz else "" in
      (typ, param, data) :: go (off + (if typ = 4 then sz else 0) + 4)
  in
  go 0

let u32_at s off = Int32.to_int (String.get_int32_le s off) land 0xffff_ffff

let program_data props (obj : Device.Tiny_elf.t) =
  let { Nx_device_elf.image; sections; _ } =
    try Nx_device_elf.load ~align:128 obj.lib
    with Failure why -> invalid_arg ("an NV program is no cubin: " ^ why)
  in
  let name = obj.name in
  let constbufs = ref [ (0, (0, 0x160)) ] and prog_off = ref 0 in
  let prog_sz = ref (String.length image) in
  let regs = ref 0 and shmem = ref 0x400 and lcmem = ref 0x240 in
  let cbuf0_size = ref 0 in
  let set_bank i b =
    constbufs :=
      if List.mem_assoc i !constbufs then
        List.map (fun (j, c) -> if j = i then (j, b) else (j, c)) !constbufs
      else !constbufs @ [ (i, b) ]
  in
  List.iter
    (fun (sh : Nx_device_elf.section) ->
      if sh.name = ".nv.shared." ^ name then
        shmem := Helpers.round_up (0x400 + sh.size) 128;
      if sh.name = ".text." ^ name then (
        prog_off := sh.offset;
        prog_sz := sh.size)
      else if String.starts_with ~prefix:".nv.constant" sh.name then (
        let rest = String.sub sh.name 12 (String.length sh.name - 12) in
        let digits =
          String.to_seq rest
          |> Seq.take_while (fun c -> c >= '0' && c <= '9')
          |> String.of_seq
        in
        if digits <> "" then set_bank (int_of_string digits) (sh.offset, sh.size))
      else if String.starts_with ~prefix:".nv.info" sh.name then
        List.iter
          (fun (_, param, data) ->
            if sh.name = ".nv.info." ^ name && param = 0xa then
              cbuf0_size := String.get_uint16_le data 4 (* EIATTR_PARAM_CBANK *)
            else if sh.name = ".nv.info" && param = 0x12 then
              lcmem := u32_at data 4 + 0x240 (* EIATTR_MIN_STACK_SIZE *)
            else if sh.name = ".nv.info" && param = 0x2f then
              regs := u32_at data 4 (* EIATTR_REGCOUNT *))
          (elf_info sh))
    sections;
  let blackwell = props.compute_class >= G.blackwell_compute_a in
  (* Minimum cbuf_0 size for driver params: Blackwell needs index 223 (224
     entries), older GPUs need index 11 (12 entries) *)
  let cbuf_0 =
    Array.make (max (!cbuf0_size / 4) (if blackwell then 224 else 12)) 0
  in
  let sig_ = obj.signature in
  let nbufs =
    List.length
      (List.filter (fun (p : Device.Tiny_elf.param) -> p.name = None) sig_)
  in
  let vars =
    List.filteri (fun i _ -> i >= nbufs) sig_
    |> List.map (fun (p : Device.Tiny_elf.param) -> p.dtype)
  in
  let kernargs_size =
    Helpers.round_up
      (max
         (snd (List.assoc 0 !constbufs))
         ((Array.length cbuf_0 * 4) + (List.length sig_ * 8)))
      256
  in
  let words a =
    let lo, hi = Helpers.data64_le a in
    [ lo; hi ]
  in
  let fill at l = List.iteri (fun i w -> cbuf_0.(at + i) <- w) l in
  let qmd = Qmd.make props in
  if blackwell then (
    fill 188 (words props.shared_window @ words props.local_window);
    cbuf_0.(223) <- 0xfffdc0;
    List.iter
      (fun (k, v) -> Qmd.write qmd k v)
      [
        ("qmd_major_version", 5);
        ("qmd_type", G.nvcec0_qmdv05_00_qmd_type_grid_cta);
        ("register_count", !regs);
        ("shared_memory_size_shifted7", !shmem lsr 7);
      ])
  else (
    fill 6
      (words props.shared_window @ words props.local_window @ words 0xfffdc0);
    List.iter
      (fun (k, v) -> Qmd.write qmd k v)
      [
        ("qmd_major_version", 3);
        ("sm_global_caching_enable", 1);
        ("shared_memory_size", !shmem);
        ("register_count_v", !regs);
      ]);
  let smem_cfg =
    match List.find_opt (fun c -> c * 1024 >= !shmem) [ 32; 64; 100 ] with
    | Some c -> (c * 1024 / 4096) + 1
    | None ->
        invalid_arg
          (Printf.sprintf "%s needs %d bytes of shared memory" name !shmem)
  in
  List.iter
    (fun (k, v) -> Qmd.write qmd k v)
    [
      ("qmd_group_id", 0x3f);
      ("invalidate_texture_header_cache", 1);
      ("invalidate_texture_sampler_cache", 1);
      ("invalidate_texture_data_cache", 1);
      ("invalidate_shader_data_cache", 1);
      ("api_visible_call_limit", 1);
      ("sampler_index", 1);
      ("barrier_count", 1);
      ("cwd_membar_type", G.nvc6c0_qmdv03_00_cwd_membar_type_l1_sysmembar);
      ("constant_buffer_invalidate_0", 1);
      ("min_sm_config_shared_mem_size", smem_cfg);
      ("target_sm_config_shared_mem_size", smem_cfg);
      ("max_sm_config_shared_mem_size", 0x1a);
      ("program_prefetch_size", min (!prog_sz lsr 8) 0x1ff);
      ("sass_version", props.sass_version);
    ];
  List.iter
    (fun (i, (_, sz)) ->
      Qmd.write qmd (Printf.sprintf "constant_buffer_size_shifted4_%d" i) sz;
      Qmd.write qmd (Printf.sprintf "constant_buffer_valid_%d" i) 1)
    !constbufs;
  (* Registers allocation granularity per warp is 256, warp allocation
     granularity is 4. Register file size is 65536. *)
  let max_threads =
    65536 / Helpers.round_up (max 1 !regs * 32) 256 / 4 * 4 * 32
  in
  ( Helpers.round_up (String.length image) 0x1000 + 0x1000,
    {
      constbufs = !constbufs;
      prog_off = !prog_off;
      cbuf_0;
      vars;
      kernargs_size;
      local_bytes = !lcmem;
      qmd;
      max_threads;
    } )

(* The local memory word of kernels that need [bytes] per thread. *)
let local_word devs bytes =
  load
    (index
       (placeholder ~device:(Multi devs)
          ~tag:(Tag.Tuple [ String "nv_local"; Int bytes ])
          [ 1 ] Dtype.Uint32)
       [ int 0 ])
    []

(* Each program's launch template is built once for its devices. Its cubin is
   the engine's to load, relocated by the device that runs it (DIVERGENCES D38):
   the placeholder names the cubin and its kernel, and is as long as the image
   the device lays out, with room for the GPU's prefetch after it. *)
let programs = Hashtbl.create 16
let programs_lock = Mutex.create ()

let build_program props devs prg =
  let obj = Device.Tiny_elf.of_program prg in
  Mutex.protect programs_lock @@ fun () ->
  match Hashtbl.find_opt programs (obj.lib, devs, props) with
  | Some p -> p
  | None ->
      let image_size, data = program_data props obj in
      let local = local_word devs data.local_bytes in
      if data.qmd.ver >= 4 then
        Qmd.patch data.qmd "shader_local_memory_high_size_shifted4"
          (shr local (int 4))
      else Qmd.patch data.qmd "shader_local_memory_high_size" local;
      let p =
        ( data,
          placeholder ~slot:0 ~device:(Multi devs)
            ~tag:
              (Tag.Tuple [ String "program"; Bytes obj.lib; String obj.name ])
            [ image_size ] Dtype.Uint8 )
      in
      Hashtbl.replace programs (obj.lib, devs, props) p;
      p

(* Queues *)

let queue props q : Hcq2.commands =
  let devs = Hcq2.Queue.devices q and name = Hcq2.Queue.name q in
  let dev = Multi devs and on = List.hd devs in
  let emit words = ignore (Hcq2.Queue.q q words) in
  let nvm subc mthd vals = emit (nvm subc mthd vals) in
  let addr u = getaddr ~device:on u in
  let copy_queue = String.starts_with ~prefix:"COPY" name in
  (* NVQueue *)
  let sem a value flags =
    nvm 0 G.nvc56f_sem_addr_lo
      [
        a;
        ccast value Dtype.Uint64;
        u32
          (bits G.nvc56f_sem_execute_payload_size
             G.nvc56f_sem_execute_payload_size_64bit
          lor flags);
      ]
  in
  let wait signal value =
    sem (addr signal) value
      (bits G.nvc56f_sem_execute_operation
         G.nvc56f_sem_execute_operation_acq_circ_geq)
  in
  let release signal value ~timestamp =
    sem (addr signal) value
      (bits G.nvc56f_sem_execute_operation
         G.nvc56f_sem_execute_operation_release
      lor bits G.nvc56f_sem_execute_release_wfi
            G.nvc56f_sem_execute_release_wfi_en
      lor bits G.nvc56f_sem_execute_release_timestamp
            (if timestamp then G.nvc56f_sem_execute_release_timestamp_en
             else G.nvc56f_sem_execute_release_timestamp_dis));
    if not timestamp then nvm 0 G.nvc56f_non_stall_interrupt [ u32 0 ]
  in
  let submit_cmdbuf cmdbuf =
    let fifo = if copy_queue then props.copy else props.compute in
    let ib, off = Hcq2.unwrap_view cmdbuf in
    let word nm dt sz =
      placeholder ~device:dev ~volatile:true
        ~tag:(Tag.String (Hcq2.to_name [ nm; name ]))
        [ sz ] dt
    in
    let ring = word "ring" Dtype.Uint64 fifo.entries
    and gpput = word "gpput" Dtype.Uint32 1
    and doorbell = word "doorbell" Dtype.Uint32 1
    and put = word "put_value" Dtype.Uint64 1 in
    let dwords = max_numel cmdbuf * Dtype.itemsize (dtype cmdbuf) / 4 in
    (* An entry's length field holds 21 bits of words (DIVERGENCES D43). *)
    if dwords >= 1 lsl 21 then
      raise
        (Hcq2.Over_capacity
           (Printf.sprintf
              "an NV command buffer of %d words exceeds a GPFIFO entry's %d"
              dwords
              ((1 lsl 21) - 1)));
    let gpentry =
      Hcq2.patch
        (word "gpentry" Dtype.Uint64 1)
        [
          (int 0, add (addr ib) (u64 (off lor (dwords lsl 42) lor (1 lsl 41))));
        ]
    in
    let p = load (index put [ int 0 ]) [] in
    let written =
      barrier
        (store
           (index (after ring [ cmdbuf ])
              [ cast (mod_ p (int fifo.entries)) Dtype.Int32 ])
           (load (index gpentry [ int 0 ]) []))
        [ store (index put [ int 0 ]) (add p (int 1)) ]
    in
    let queued =
      barrier
        (store
           (index (after gpput [ written ]) [ int 0 ])
           (cast (mod_ (add p (int 1)) (int fifo.entries)) Dtype.Uint32))
        []
    in
    store (index (after doorbell [ queued ]) [ int 0 ]) (u32 fifo.token)
  in
  (* NVComputeQueue *)
  let chain = ref [] in
  (* The launches of a chain end with it: each descriptor, built back to front,
     points to the next, and the channel schedules the first. *)
  let end_chain () =
    let rec build = function
      | [] -> None
      | qmd :: rest ->
          let region_of_next = build rest in
          Option.iter
            (fun r ->
              Qmd.patch qmd "dependent_qmd0_pointer" (shr (addr r) (int 8)))
            region_of_next;
          Some (region "qmd" (Qmd.blob qmd) qmd.patches)
    in
    Option.iter
      (fun head ->
        nvm 1 G.nvc6c0_send_pcas_a
          [ cast (shr (addr head) (int 8)) Dtype.Uint32 ];
        nvm 1 G.nvc6c0_send_signaling_pcas2_b
          [ u32 G.nvc6c0_send_signaling_pcas2_b_pcas_action_prefetch_schedule ])
      (build !chain);
    chain := []
  in
  let exec call prg =
    if copy_queue then invalid_arg "an NV copy queue runs no program";
    let data, lib = build_program props devs prg in
    let info =
      match arg prg with
      | Program p -> p
      | _ -> invalid_arg "an NV command runs a compiled program"
    in
    let known = List.map (function Int n -> n | Sym _ -> 1) in
    let threads = Helpers.prod (known info.local_size) in
    if threads > 1024 || data.max_threads < threads then
      invalid_arg
        (Printf.sprintf
           "Too many resources requested for launch, prod(local_size)=%d, \
            data.max_threads=%d"
           threads data.max_threads);
    let exceeds sizes limits =
      List.exists2
        (fun g m -> match g with Int g -> g > m | Sym _ -> false)
        sizes limits
    in
    if
      exceeds info.global_size [ 2147483647; 65535; 65535 ]
      || exceeds info.local_size [ 1024; 1024; 64 ]
    then invalid_arg "Invalid global/local dims";
    let qmd = Qmd.copy data.qmd in
    let dims =
      List.combine
        (Qmd.grid qmd @ List.init 3 (Printf.sprintf "cta_thread_dimension%d"))
        (info.global_size @ info.local_size)
    in
    List.iter
      (fun (k, s) ->
        match s with Int n -> Qmd.write qmd k n | Sym u -> Qmd.patch qmd k u)
      dims;
    Qmd.set_program_addr qmd (add (addr lib) (int data.prog_off));
    (* constant buffer 0: the driver params, then the arguments *)
    let bufs = Realize.get_call_arg_uops call in
    let vals = Realize.get_call_var_uops call prg in
    let at = Array.length data.cbuf_0 * 4 in
    let args =
      Hcq2.layout_args ~offset:at
        (List.map (fun g -> addr (List.nth bufs g)) info.globals
        @ List.map2 ccast vals data.vars)
    in
    let driver = Bytes.make data.kernargs_size '\000' in
    Array.iteri
      (fun i w -> Bytes.set_int32_le driver (4 * i) (Int32.of_int w))
      data.cbuf_0;
    let cbuf = region "cbuf" (Bytes.to_string driver) args in
    List.iter
      (fun (j, (off, _)) ->
        Qmd.set_constant_buf_addr qmd j
          (if j = 0 then addr cbuf else add (addr lib) (int off)))
      data.constbufs;
    match List.rev !chain with
    | prev :: _ ->
        List.iter
          (fun k -> Qmd.write prev k 1)
          [
            "dependent_qmd0_action";
            "dependent_qmd0_prefetch";
            "dependent_qmd0_enable";
          ];
        chain := !chain @ [ qmd ]
    | [] -> chain := [ qmd ]
  in
  let compute_release signal value ~timestamp =
    match List.rev !chain with
    | prev :: _ when Qmd.set_release prev (addr signal) value ~timestamp -> ()
    | _ ->
        end_chain ();
        release signal value ~timestamp
  in
  (* NVCopyQueue *)
  let semaphore a value typ =
    nvm 4 G.nvc6b5_set_semaphore_a (hilo a @ [ ccast value Dtype.Uint32 ]);
    nvm 4 G.nvc6b5_launch_dma
      [
        u32
          (bits G.nvc6b5_launch_dma_flush_enable
             G.nvc6b5_launch_dma_flush_enable_true
          lor bits G.nvc6b5_launch_dma_semaphore_type typ);
      ]
  in
  let copy dst src n =
    if not copy_queue then invalid_arg "an NV compute queue copies nothing";
    let dst = addr dst and src = addr src in
    let step = 1 lsl 31 in
    let rec go off =
      if off < n then (
        nvm 4 G.nvc6b5_offset_in_upper
          (hilo (add src (u64 off)) @ hilo (add dst (u64 off)));
        nvm 4 G.nvc6b5_line_length_in [ u32 (min (n - off) step) ];
        nvm 4 G.nvc6b5_launch_dma
          [
            u32
              (bits G.nvc6b5_launch_dma_data_transfer_type
                 G.nvc6b5_launch_dma_data_transfer_type_non_pipelined
              lor bits G.nvc6b5_launch_dma_src_memory_layout
                    G.nvc6b5_launch_dma_src_memory_layout_pitch
              lor bits G.nvc6b5_launch_dma_dst_memory_layout
                    G.nvc6b5_launch_dma_dst_memory_layout_pitch);
          ];
        go (off + step))
    in
    go 0
  in
  (* The copy engine writes a 64-bit value in 32-bit words: its low word, then
     its high word, where the value lives when the low word wrapped to 0 and
     into a word of its own otherwise (DIVERGENCES D40). *)
  let copy_signal signal value =
    let a = addr signal in
    let one = G.nvc6b5_launch_dma_semaphore_type_release_one_word_semaphore in
    let sink =
      placeholder ~device:dev ~volatile:true ~tag:(Tag.String "nv_sink") [ 1 ]
        Dtype.Uint32
    in
    semaphore a value one;
    semaphore
      (where (eq (cast value Dtype.Uint32) (u32 0)) (add a (u64 4)) (addr sink))
      (shr value (int 32))
      one
  in
  let timestamp slot =
    if copy_queue then
      semaphore (addr slot) (u32 0)
        G.nvc6b5_launch_dma_semaphore_type_release_four_word_semaphore
    else compute_release slot (u64 0) ~timestamp:true
  in
  {
    exec;
    copy;
    wait =
      (fun signal value ->
        end_chain ();
        wait signal value);
    signal =
      (fun signal value ->
        if copy_queue then copy_signal signal value
        else compute_release signal value ~timestamp:false);
    timestamp;
    memory_barrier =
      (fun () ->
        if not copy_queue then (
          end_chain ();
          nvm 1 G.nvc6c0_invalidate_shader_caches_no_wfi
            [
              u32
                (bits G.nvc6c0_invalidate_shader_caches_no_wfi_instruction
                   G.nvc6c0_invalidate_shader_caches_no_wfi_instruction_true
                lor bits G.nvc6c0_invalidate_shader_caches_no_wfi_global_data
                      G.nvc6c0_invalidate_shader_caches_no_wfi_global_data_true
                lor bits G.nvc6c0_invalidate_shader_caches_no_wfi_constant
                      G.nvc6c0_invalidate_shader_caches_no_wfi_constant_true);
            ]));
    (* A chain ends at a loop's edges and at the end of each trip: a trip's
       launches chain onto its own descriptors. *)
    loop =
      (fun r body ->
        end_chain ();
        Hcq2.Queue.loop q r (fun () ->
            body ();
            end_chain ()));
    submit =
      (fun () ->
        end_chain ();
        submit_cmdbuf (Hcq2.bufferize_cmdbuf q "cmdbuf"));
  }

let queues ~host ~reaches props =
  { Hcq2.commands = queue props; copy_queue = true; host; reaches }

(* Linking *)

type storage =
  | Program of { binary : string; name : string }
  | Ring of string
  | Gp_put of string
  | Put of string
  | Doorbell of string
  | Local of int

let storage u =
  match tag u with
  | Some (Tag.Tuple [ String "program"; Bytes binary; String name ]) ->
      Some (Program { binary; name })
  | Some (Tag.Tuple [ String "nv_local"; Int bytes ]) -> Some (Local bytes)
  | Some (Tag.String t) -> (
      (* [name_queue_index], as submit tags them. *)
      match List.rev (String.split_on_char '_' t) with
      | i :: queue :: name -> (
          let queue = String.uppercase_ascii queue ^ ":" ^ i in
          match String.concat "_" (List.rev name) with
          | "ring" -> Some (Ring queue)
          | "gpput" -> Some (Gp_put queue)
          | "put_value" -> Some (Put queue)
          | "doorbell" -> Some (Doorbell queue)
          | _ -> None)
      | _ -> None)
  | _ -> None
