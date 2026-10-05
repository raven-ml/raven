(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

module D = Defs

(* Terms and words *)

type 'v term = Value of 'v | Add of 'v term * int | Shift of 'v term * int

let rec eval = function
  | Value v -> v
  | Add (t, n) -> eval t + n
  | Shift (t, n) -> eval t lsr n

type 'v word = Dword of int | W32 of 'v term | W64 of 'v term

let mask32 = 0xffff_ffff

let dwords ws =
  List.concat_map
    (function
      | Dword n -> [ n land mask32 ]
      | W32 t -> [ eval t land mask32 ]
      | W64 t ->
          let n = eval t in
          [ n land mask32; (n lsr 32) land mask32 ])
    ws

(* [v] in the field [(lo, _)] of a word. *)
let bits (lo, _) v = v lsl lo

(* Methods *)

type subchannel = Host | Compute | Copy

let subchannel = function Host -> 0 | Compute -> 1 | Copy -> 4

(* Incrementing methods of [s] from [mthd], one per word. *)
let methods s mthd ws =
  let n =
    List.fold_left
      (fun n w -> n + match w with Dword _ | W32 _ -> 1 | W64 _ -> 2)
      0 ws
  in
  Dword ((2 lsl 28) lor (n lsl 16) lor (subchannel s lsl 13) lor (mthd lsr 2))
  :: ws

(* A copy's line is at most 2 GiB. *)
let line = 1 lsl 31

module Methods = struct
  (* An address as the copy engine takes it: its high word, then its low. *)
  let hi_lo a = [ W32 (Shift (a, 32)); W32 a ]

  let semaphore a v flags =
    methods Host D.nvc56f_sem_addr_lo
      [
        W64 (Value a);
        W64 (Value v);
        Dword
          (bits D.nvc56f_sem_execute_payload_size
             D.nvc56f_sem_execute_payload_size_64bit
          lor flags);
      ]

  let acquire a v =
    semaphore a v
      (bits D.nvc56f_sem_execute_operation
         D.nvc56f_sem_execute_operation_acq_circ_geq)

  let released stamp =
    bits D.nvc56f_sem_execute_operation D.nvc56f_sem_execute_operation_release
    lor bits D.nvc56f_sem_execute_release_wfi
          D.nvc56f_sem_execute_release_wfi_en
    lor bits D.nvc56f_sem_execute_release_timestamp stamp

  let release a v =
    semaphore a v (released D.nvc56f_sem_execute_release_timestamp_dis)
    @ methods Host D.nvc56f_non_stall_interrupt [ Dword 0 ]

  let release_stamp a v =
    semaphore a v (released D.nvc56f_sem_execute_release_timestamp_en)

  let set_object s cls = methods s D.nvc6c0_set_object [ Dword cls ]

  let local_memory_window a =
    methods Compute D.nvc6c0_set_shader_local_memory_window_a (hi_lo (Value a))

  let shared_memory_window a =
    methods Compute D.nvc6c0_set_shader_shared_memory_window_a (hi_lo (Value a))

  (* The third word is the most streaming multiprocessors the memory serves: all
     of them. *)
  let local_memory a ~per_tpc =
    methods Compute D.nvc6c0_set_shader_local_memory_a (hi_lo (Value a))
    @ methods Compute D.nvc6c0_set_shader_local_memory_non_throttled_a
        (hi_lo (Value per_tpc) @ [ Dword 0xff ])

  let invalidate_caches =
    methods Compute D.nvc6c0_invalidate_shader_caches_no_wfi
      [
        Dword
          (bits D.nvc6c0_invalidate_shader_caches_no_wfi_instruction
             D.nvc6c0_invalidate_shader_caches_no_wfi_instruction_true
          lor bits D.nvc6c0_invalidate_shader_caches_no_wfi_global_data
                D.nvc6c0_invalidate_shader_caches_no_wfi_global_data_true
          lor bits D.nvc6c0_invalidate_shader_caches_no_wfi_constant
                D.nvc6c0_invalidate_shader_caches_no_wfi_constant_true);
      ]

  let schedule a =
    methods Compute D.nvc6c0_send_pcas_a [ W32 (Shift (Value a, 8)) ]
    @ methods Compute D.nvc6c0_send_signaling_pcas2_b
        [ Dword D.nvc6c0_send_signaling_pcas2_b_pcas_action_prefetch_schedule ]

  let copy ~dst ~src n =
    let launch =
      bits D.nvc6b5_launch_dma_data_transfer_type
        D.nvc6b5_launch_dma_data_transfer_type_non_pipelined
      lor bits D.nvc6b5_launch_dma_src_memory_layout
            D.nvc6b5_launch_dma_src_memory_layout_pitch
      lor bits D.nvc6b5_launch_dma_dst_memory_layout
            D.nvc6b5_launch_dma_dst_memory_layout_pitch
    in
    let rec go off acc =
      if off >= n then List.concat (List.rev acc)
      else
        let words =
          methods Copy D.nvc6b5_offset_in_upper
            (hi_lo (Add (Value src, off)) @ hi_lo (Add (Value dst, off)))
          @ methods Copy D.nvc6b5_line_length_in
              [ Dword (Stdlib.Int.min line (n - off)) ]
          @ methods Copy D.nvc6b5_launch_dma [ Dword launch ]
        in
        go (off + line) (words :: acc)
    in
    go 0 []

  let copy_semaphore a v kind =
    methods Copy D.nvc6b5_set_semaphore_a (hi_lo (Value a) @ [ v ])
    @ methods Copy D.nvc6b5_launch_dma
        [
          Dword
            (bits D.nvc6b5_launch_dma_flush_enable
               D.nvc6b5_launch_dma_flush_enable_true
            lor bits D.nvc6b5_launch_dma_semaphore_type kind);
        ]

  let copy_release a v =
    copy_semaphore a (W32 (Value v))
      D.nvc6b5_launch_dma_semaphore_type_release_one_word_semaphore

  let copy_stamp a =
    copy_semaphore a (Dword 0)
      D.nvc6b5_launch_dma_semaphore_type_release_four_word_semaphore
end

(* Channel rings *)

module Gpfifo = struct
  let max_words = (1 lsl snd D.nvc56f_gp_entry1_length) - 1

  (* An address below 2^40, 4-byte aligned, is the entry's low 40 bits: its word
     0 holds bits 2 to 31 at bit 2, its word 1 bits 32 to 39 at bit 0. *)
  let entry a ~offset ~words =
    if words < 0 || words > max_words then
      invalid_arg
        (Printf.sprintf "Gpfifo.entry: %d words, at most %d" words max_words);
    let flags =
      bits D.nvc56f_gp_entry1_level D.nvc56f_gp_entry1_level_subroutine
      lor bits D.nvc56f_gp_entry1_length words
    in
    Add (Value a, offset lor (flags lsl 32))
end

(* Launches *)

let round_up n a = (n + a - 1) / a * a

(* The driver reserves the first 1 KiB of a block's shared memory, and 576 bytes
   of each thread's local memory. *)
let reserved_shared = 0x400
let reserved_local = 0x240

(* The shared memory a streaming multiprocessor may be configured with, in
   KiB. *)
let shared_configs = [ 32; 64; 100 ]

(* Bank 0 when the cubin has none: the driver's parameters alone. *)
let default_bank0 = { Nx_nv_cubin.index = 0; offset = 0; bytes = 0x160 }

(* The driver's stack limit, and where its parameters sit in constant bank 0, in
   words: the shared and local memory windows, the stack limit, and the words
   through the last it reads. *)
let stack_limit = 0xfffdc0

type params = { words : int; shared : int; local : int; stack : int }

let ada_params = { words = 12; shared = 6; local = 8; stack = 10 }
let blackwell_params = { words = 224; shared = 188; local = 190; stack = 223 }

module Program = struct
  type t = {
    kernel : Nx_nv_cubin.kernel;
    blackwell : bool;
    sass_version : int;
    shared_window : int;
    local_window : int;
    shared_bytes : int;
    shared_config : int;
  }

  let make ~compute_class ~sass_version ~shared_window ~local_window
      (kernel : Nx_nv_cubin.kernel) =
    let shared_bytes = round_up (reserved_shared + kernel.shared_bytes) 128 in
    match List.find_opt (fun c -> c * 1024 >= shared_bytes) shared_configs with
    | None ->
        Error
          (Printf.sprintf
             "the kernel needs %d bytes of shared memory, more than %d KiB"
             shared_bytes
             (List.fold_left Stdlib.Int.max 0 shared_configs))
    | Some c ->
        Ok
          {
            kernel;
            blackwell = compute_class >= D.blackwell_compute_a;
            sass_version;
            shared_window;
            local_window;
            shared_bytes;
            shared_config = (c * 1024 / 4096) + 1;
          }

  let code p = p.kernel.code

  let banks p =
    List.fold_left
      (fun banks (b : Nx_nv_cubin.bank) ->
        if List.exists (fun (x : Nx_nv_cubin.bank) -> x.index = b.index) banks
        then
          List.map
            (fun (x : Nx_nv_cubin.bank) -> if x.index = b.index then b else x)
            banks
        else banks @ [ b ])
      [ default_bank0 ] p.kernel.banks

  let driver_parameters p =
    let l = if p.blackwell then blackwell_params else ada_params in
    let b =
      Bytes.make
        (4 * Stdlib.Int.max (p.kernel.params_offset / 4) l.words)
        '\000'
    in
    Bytes.set_int64_le b (4 * l.shared) (Int64.of_int p.shared_window);
    Bytes.set_int64_le b (4 * l.local) (Int64.of_int p.local_window);
    Bytes.set_int32_le b (4 * l.stack) (Int32.of_int stack_limit);
    Bytes.to_string b

  let local_bytes p = p.kernel.stack_bytes + reserved_local

  (* Registers are allocated per warp in units of 256, warps in units of 4, from
     a register file of 65536. *)
  let max_threads p =
    65536 / round_up (Stdlib.Int.max 1 p.kernel.registers * 32) 256 / 4 * 4 * 32
end

type 'v hole = { at : int; bytes : int; value : 'v term }
type 'v structure = { bytes : string; holes : 'v hole list }

let fill (s : int structure) =
  let b = Bytes.of_string s.bytes in
  List.iter
    (fun (h : int hole) ->
      for i = 0 to h.bytes - 1 do
        Bytes.set b (h.at + i) (Char.chr ((eval h.value lsr (8 * i)) land 0xff))
      done)
    s.holes;
  Bytes.to_string b

type axis = X | Y | Z
type dim = Grid of axis | Block of axis

module Qmd = struct
  type 'v t = {
    ver : int;
    fields : (string * (int * int)) list;
    mv : Bytes.t;
    mutable holes : 'v hole list; (* the latest at each offset *)
  }

  let range q k =
    match List.assoc_opt k q.fields with
    | Some r -> r
    | None -> invalid_arg ("no field " ^ k ^ " in a launch descriptor")

  (* The bytes holding the bits [lo] to [hi], as one integer. *)
  let number q lo hi =
    let n = ref 0 in
    for i = hi / 8 downto lo / 8 do
      n := (!n lsl 8) lor Char.code (Bytes.get q.mv i)
    done;
    !n

  let read q k =
    let lo, w = range q k in
    (number q lo (lo + w - 1) lsr (lo mod 8)) land ((1 lsl w) - 1)

  let write q k v =
    let lo, w = range q k in
    if v lsr w <> 0 then invalid_arg (Printf.sprintf "%s=0x%x does not fit" k v);
    let hi = lo + w - 1 in
    let mask = ((1 lsl w) - 1) lsl (lo mod 8) in
    let n = number q lo hi land lnot mask lor (v lsl (lo mod 8)) in
    for i = lo / 8 to hi / 8 do
      Bytes.set q.mv i (Char.chr ((n lsr (8 * (i - (lo / 8)))) land 0xff))
    done

  (* A hole of the widest unsigned word the field holds. *)
  let patch q k v =
    let lo, w = range q k in
    if lo mod 8 <> 0 then invalid_arg (k ^ " is not byte aligned");
    let bytes = List.find (fun n -> n * 8 <= w) [ 8; 4; 2; 1 ] in
    let at = lo / 8 in
    q.holes <-
      { at; bytes; value = v } :: List.filter (fun h -> h.at <> at) q.holes

  let v4 q = q.ver >= 4

  let set_addr q name ?(sfx = "") addr =
    patch q (name ^ "_lower" ^ sfx) addr;
    patch q (name ^ "_upper" ^ sfx) (Shift (addr, 32))

  let make (p : Program.t) =
    (* The version, its size in words, and its fields. *)
    let ver, words, fields =
      if p.blackwell then (5, 0x60, D.qmd_v5) else (3, 0x40, D.qmd_v3)
    in
    let q = { ver; fields; mv = Bytes.make (words * 4) '\000'; holes = [] } in
    let k = p.kernel in
    let own =
      if p.blackwell then
        [
          ("qmd_major_version", 5);
          ("qmd_type", D.nvcec0_qmdv05_00_qmd_type_grid_cta);
          ("register_count", k.registers);
          ("shared_memory_size_shifted7", p.shared_bytes lsr 7);
        ]
      else
        [
          ("qmd_major_version", 3);
          ("sm_global_caching_enable", 1);
          ("shared_memory_size", p.shared_bytes);
          ("register_count_v", k.registers);
        ]
    in
    List.iter
      (fun (f, v) -> write q f v)
      (own
      @ [
          ("qmd_group_id", 0x3f);
          ("invalidate_texture_header_cache", 1);
          ("invalidate_texture_sampler_cache", 1);
          ("invalidate_texture_data_cache", 1);
          ("invalidate_shader_data_cache", 1);
          ("api_visible_call_limit", 1);
          ("sampler_index", 1);
          ("barrier_count", 1);
          ("cwd_membar_type", D.nvc6c0_qmdv03_00_cwd_membar_type_l1_sysmembar);
          ("constant_buffer_invalidate_0", 1);
          ("min_sm_config_shared_mem_size", p.shared_config);
          ("target_sm_config_shared_mem_size", p.shared_config);
          ("max_sm_config_shared_mem_size", 0x1a);
          ("program_prefetch_size", Stdlib.Int.min (k.code_bytes lsr 8) 0x1ff);
          ("sass_version", p.sass_version);
        ]);
    List.iter
      (fun (b : Nx_nv_cubin.bank) ->
        write q
          (Printf.sprintf "constant_buffer_size_shifted4_%d" b.index)
          ((b.bytes + 15) lsr 4);
        write q (Printf.sprintf "constant_buffer_valid_%d" b.index) 1)
      (Program.banks p);
    q

  let copy q = { q with mv = Bytes.copy q.mv }
  let axis = function X -> 0 | Y -> 1 | Z -> 2

  let dim q = function
    | Grid a when v4 q ->
        [| "grid_width"; "grid_height"; "grid_depth" |].(axis a)
    | Grid a ->
        [| "cta_raster_width"; "cta_raster_height"; "cta_raster_depth" |].(axis
                                                                             a)
    | Block a -> Printf.sprintf "cta_thread_dimension%d" (axis a)

  let set_dim q d n = write q (dim q d) n
  let patch_dim q d v = patch q (dim q d) (Value v)

  let set_program q addr =
    let addr = Value addr in
    if v4 q then set_addr q "program_address" ~sfx:"_shifted4" (Shift (addr, 4))
    else set_addr q "program_address" (Shift (addr, 0));
    set_addr q "program_prefetch_addr" ~sfx:"_shifted" (Shift (addr, 8))

  let set_bank q i addr =
    let addr = Value addr in
    if v4 q then
      set_addr q "constant_buffer_addr"
        ~sfx:(Printf.sprintf "_shifted6_%d" i)
        (Shift (addr, 6))
    else
      set_addr q "constant_buffer_addr" ~sfx:(Printf.sprintf "_%d" i)
        (Shift (addr, 0))

  let set_local_memory q bytes =
    if v4 q then
      patch q "shader_local_memory_high_size_shifted4" (Shift (Value bytes, 4))
    else patch q "shader_local_memory_high_size" (Value bytes)

  (* One of the two releases, if one is free: a 64-bit payload, or with [stamp]
     the payload and the timer. *)
  let add_release q addr v ~stamp =
    let enable i =
      if v4 q then Printf.sprintf "release_enable_%d" i
      else Printf.sprintf "release%d_enable" i
    in
    match List.find_opt (fun i -> read q (enable i) = 0) [ 0; 1 ] with
    | None -> false
    | Some i ->
        let name s = Printf.sprintf s i in
        if v4 q then (
          set_addr q (name "release_semaphore%d_addr") (Value addr);
          set_addr q (name "release_semaphore%d_payload") (Value v))
        else (
          set_addr q (name "release%d_address") (Value addr);
          set_addr q (name "release%d_payload") (Value v));
        write q (enable i) 1;
        write q
          (if v4 q then name "release_structure_size_%d"
           else name "release%d_structure_size")
          (if stamp then 0 else 2);
        if not (v4 q) then write q (name "release%d_payload64b") 1;
        true

  let release q addr v = add_release q addr v ~stamp:false
  let release_stamp q addr v = add_release q addr v ~stamp:true

  let chain q addr =
    List.iter
      (fun f -> write q f 1)
      [
        "dependent_qmd0_action";
        "dependent_qmd0_prefetch";
        "dependent_qmd0_enable";
      ];
    patch q "dependent_qmd0_pointer" (Shift (Value addr, 8))

  let structure q =
    {
      bytes = Bytes.to_string q.mv;
      holes = List.sort (fun a b -> Stdlib.Int.compare a.at b.at) q.holes;
    }
end
