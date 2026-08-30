(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

let round_up n align = (n + align - 1) / align * align

let bit_length n =
  let rec go acc n = if n = 0 then acc else go (acc + 1) (n lsr 1) in
  go 0 n

type addr_space = Phys | Sys | Peer

type virt_mapping = {
  va_addr : int;
  size : int;
  paddrs : (int * int) list;
  aspace : addr_space;
  uncached : bool;
  snooped : bool;
}

type 'pt pt_ops = {
  make : paddr:int -> lv:int -> 'pt;
  set_entry :
    'pt ->
    idx:int ->
    paddr:int ->
    ?table:bool ->
    ?uncached:bool ->
    ?aspace:addr_space ->
    ?snooped:bool ->
    ?frag:int ->
    valid:bool ->
    unit ->
    unit;
  entry : 'pt -> int -> int64;
  valid : 'pt -> int -> bool;
  address : 'pt -> int -> int;
  is_page : 'pt -> int -> bool;
  supports_huge_page : 'pt -> paddr:int -> bool;
  paddr : 'pt -> int;
  lv : 'pt -> int;
}

type 'pt t = {
  pt_ops : 'pt pt_ops;
  vram_size : int;
  va_base : int;
  va_bits : int;
  level_cnt : int;
  pte_covers : int array;
  pte_cnt : int array;
  palloc_ranges : (int * int) array;
  reserve_ptable : bool;
  boot_allocator : Tlsf.t;
  ptable_allocator : Tlsf.t;
  pa_allocator : Tlsf.t;
  va_allocator : Tlsf.t;
  zero_vram : paddr:int -> size:int -> unit;
  on_range_mapped : unit -> unit;
  is_booting : unit -> bool;
  dbg_name : string;
  root_page_table : 'pt;
  mutable identity_vas : (bool * int) list;
}

let vram_size t = t.vram_size
let va_base t = t.va_base
let va_bits t = t.va_bits
let level_cnt t = t.level_cnt
let pte_covers t lv = t.pte_covers.(lv)
let pte_cnt t lv = t.pte_cnt.(lv)
let root_page_table t = t.root_page_table

let create ~pt_ops ~vram_size ~boot_size ~va_bits ~va_shifts ~va_base
    ~palloc_ranges ~va_allocator ~is_booting ~zero_vram ?(first_lv = 0)
    ?(reserve_ptable = false) ?(smi_dev = false) ?(dbg_name = "mm")
    ?(on_range_mapped = fun () -> ()) () =
  let shifts = Array.of_list va_shifts in
  let n = Array.length shifts in
  let msb i = if i = n then va_bits + 1 else shifts.(i) in
  let pte_covers = Array.init n (fun i -> 1 lsl shifts.(n - 1 - i)) in
  let pte_cnt = Array.init n (fun i -> 1 lsl (msb (n - i) - msb (n - 1 - i))) in
  let boot_allocator = Tlsf.create ~size:boot_size () in
  let ptable_size =
    if reserve_ptable then round_up (vram_size / 512) (1 lsl 20) else 0
  in
  let ptable_allocator = Tlsf.create ~size:ptable_size ~base:boot_size () in
  let off_sz = boot_size + ptable_size in
  let pa_allocator = Tlsf.create ~size:(vram_size - off_sz) ~base:off_sz () in
  if not (is_booting ()) then
    invalid_arg "During booting, only boot memory can be allocated";
  let root_paddr = Tlsf.alloc boot_allocator 0x1000 ~align:0x1000 () in
  if not smi_dev then zero_vram ~paddr:root_paddr ~size:0x1000;
  {
    pt_ops;
    vram_size;
    va_base;
    va_bits;
    level_cnt = n;
    pte_covers;
    pte_cnt;
    palloc_ranges = Array.of_list palloc_ranges;
    reserve_ptable;
    boot_allocator;
    ptable_allocator;
    pa_allocator;
    va_allocator;
    zero_vram;
    on_range_mapped;
    is_booting;
    dbg_name;
    root_page_table = pt_ops.make ~paddr:root_paddr ~lv:first_lv;
    identity_vas = [];
  }

let palloc t size ?(align = 0x1000) ?(zero = true) ?(boot = false)
    ?(ptable = false) () =
  if t.is_booting () <> boot then
    invalid_arg "During booting, only boot memory can be allocated";
  let allocator =
    if boot then t.boot_allocator
    else if t.reserve_ptable && ptable then t.ptable_allocator
    else t.pa_allocator
  in
  let paddr = Tlsf.alloc allocator (round_up size 0x1000) ~align () in
  if zero then t.zero_vram ~paddr ~size;
  paddr

let pfree t paddr ?(ptable = false) () =
  Tlsf.free
    (if t.reserve_ptable && ptable then t.ptable_allocator else t.pa_allocator)
    paddr

(* Page-table traversal *)

type 'pt ctx = {
  mm : 'pt t;
  mutable vaddr : int;
  create_pts : bool;
  free_pts : bool;
  inspect : bool;
  boot : bool;
  mutable pt_stack : ('pt * int * int) list;
  (* head is the deepest level; each element is (pt, pte_idx, pte_covers) *)
}

let pte_cnt_at mm lv = mm.pte_cnt.(lv)
let pte_size_of mm pt = mm.pte_covers.(mm.pt_ops.lv pt)
let pte_idx_of mm pt va = va / pte_size_of mm pt mod pte_cnt_at mm (mm.pt_ops.lv pt)

let ctx_make mm pt vaddr ~create_pts ~free_pts ~inspect ~boot =
  let vaddr = vaddr - mm.va_base in
  {
    mm;
    vaddr;
    create_pts;
    free_pts;
    inspect;
    boot;
    pt_stack = [ (pt, pte_idx_of mm pt vaddr, pte_size_of mm pt) ];
  }

let top ctx = match ctx.pt_stack with x :: _ -> x | [] -> assert false

let level_down ctx =
  let mm = ctx.mm in
  let ops = mm.pt_ops in
  let pt, pte_idx, _ = top ctx in
  if not (ops.valid pt pte_idx) then begin
    if not ctx.create_pts then
      invalid_arg "Not allowed to create new page table";
    let paddr = palloc mm 0x1000 ~zero:true ~boot:ctx.boot ~ptable:true () in
    ops.set_entry pt ~idx:pte_idx ~paddr ~table:true ~valid:true ()
  end;
  if ops.is_page pt pte_idx then
    invalid_arg
      (Printf.sprintf "Must be table pt=0x%x lv=%d pte_idx=%d entry=0x%Lx"
         (ops.paddr pt) (ops.lv pt) pte_idx (ops.entry pt pte_idx));
  let child = ops.make ~paddr:(ops.address pt pte_idx) ~lv:(ops.lv pt + 1) in
  let entry = (child, pte_idx_of mm child ctx.vaddr, pte_size_of mm child) in
  ctx.pt_stack <- entry :: ctx.pt_stack;
  entry

let try_free_pt ctx =
  let mm = ctx.mm in
  let ops = mm.pt_ops in
  match ctx.pt_stack with
  | (pt, _, _) :: (parent_pt, parent_idx, _) :: _
    when ctx.free_pts
         && ops.paddr pt <> ops.paddr mm.root_page_table
         &&
         let cnt = pte_cnt_at mm (ops.lv pt) in
         let rec all_invalid i =
           i >= cnt || ((not (ops.valid pt i)) && all_invalid (i + 1))
         in
         all_invalid 0 ->
      pfree mm (ops.paddr pt) ~ptable:true ();
      ops.set_entry parent_pt ~idx:parent_idx ~paddr:0x0 ~valid:false ();
      true
  | _ -> false

let rec level_up ctx =
  let mm = ctx.mm in
  let ops = mm.pt_ops in
  let top_exhausted () =
    let pt, pte_idx, _ = top ctx in
    pte_idx = pte_cnt_at mm (ops.lv pt)
  in
  if try_free_pt ctx || top_exhausted () then begin
    (match ctx.pt_stack with
    | (pt, pt_cnt, _) :: rest ->
        ctx.pt_stack <- rest;
        if pt_cnt = pte_cnt_at mm (ops.lv pt) then begin
          match ctx.pt_stack with
          | (p, i, c) :: rest -> ctx.pt_stack <- (p, i + 1, c) :: rest
          | [] -> assert false
        end
    | [] -> assert false);
    level_up ctx
  end

let ctx_next ctx ~size ?paddr ?(off = 0) f =
  let mm = ctx.mm in
  let ops = mm.pt_ops in
  let size = ref size and off = ref off in
  while !size > 0 do
    if ctx.create_pts then begin
      let paddr =
        match paddr with
        | Some p -> p
        | None ->
            invalid_arg "paddr must be provided when allocating new page tables"
      in
      let rec descend (pt, _, pte_covers) =
        if
          pte_covers > !size
          || (not (ops.supports_huge_page pt ~paddr:(paddr + !off)))
          || ctx.vaddr land (pte_covers - 1) <> 0
        then descend (level_down ctx)
      in
      descend (top ctx)
    end
    else begin
      let rec descend (pt, pte_idx, _) =
        if
          (not (ops.is_page pt pte_idx))
          && (ctx.free_pts || ops.valid pt pte_idx)
        then descend (level_down ctx)
      in
      descend (top ctx)
    end;
    let pt, pte_idx, pte_covers = top ctx in
    let entries =
      max
        (min (!size / pte_covers) (pte_cnt_at mm (ops.lv pt) - pte_idx))
        (if ctx.inspect then 1 else 0)
    in
    if entries <= 0 then
      invalid_arg
        (Printf.sprintf "Invalid entries size=0x%x, pte_covers=0x%x" !size
           pte_covers);
    f ~off:!off ~pt ~pte_idx ~n_ptes:entries ~pte_covers;
    size := !size - (entries * pte_covers);
    off := !off + (entries * pte_covers);
    ctx.vaddr <- ctx.vaddr + (entries * pte_covers);
    (match ctx.pt_stack with
    | (p, _, c) :: rest -> ctx.pt_stack <- (p, pte_idx + entries, c) :: rest
    | [] -> assert false);
    level_up ctx
  done

(* Memory manager operations *)

let frag_size ?(must_cover = true) va sz =
  (* [1 lsl 61] stands in for infinity: it is the largest power of two
     a native int can hold without going negative, and addresses stay
     below [2^48]. *)
  let va_pwr2_div = if va > 0 then va land -va else 1 lsl 61 in
  let sz_pwr2_div = sz land -sz in
  let sz_pwr2_max = 1 lsl (bit_length sz - 1) in
  bit_length
    (if must_cover then min va_pwr2_div sz_pwr2_div
     else min va_pwr2_div sz_pwr2_max)
  - 1 - 12

let page_tables t ~vaddr ~size =
  let exception Stop in
  let ctx =
    ctx_make t t.root_page_table vaddr ~create_pts:true ~free_pts:false
      ~inspect:false ~boot:false
  in
  let result = ref [] in
  (try
     ctx_next ctx ~size ~paddr:0
       (fun ~off:_ ~pt:_ ~pte_idx:_ ~n_ptes:_ ~pte_covers:_ ->
         result := List.rev_map (fun (pt, _, _) -> pt) ctx.pt_stack;
         raise Stop)
   with Stop -> ());
  !result

let map_range t ~vaddr ~size paddrs aspace ?(uncached = false)
    ?(snooped = false) ?(boot = false) () =
  if Helpers.getenv "MM_DEBUG" 0 <> 0 then
    Printf.printf "%s: mapping vaddr=0x%x (size=0x%x)\n%!" t.dbg_name vaddr size;
  let paddrs_size = List.fold_left (fun acc (_, s) -> acc + s) 0 paddrs in
  if size <> paddrs_size then
    invalid_arg (Printf.sprintf "Size mismatch size=%d paddrs=%d" size paddrs_size);
  let ctx =
    ctx_make t t.root_page_table vaddr ~create_pts:false ~free_pts:false
      ~inspect:true ~boot
  in
  ctx_next ctx ~size (fun ~off:_ ~pt ~pte_idx ~n_ptes ~pte_covers:_ ->
      for pte_off = 0 to n_ptes - 1 do
        if t.pt_ops.valid pt (pte_idx + pte_off) then
          invalid_arg
            (Printf.sprintf "PTE already mapped: 0x%Lx"
               (t.pt_ops.entry pt (pte_idx + pte_off)))
      done);
  let ctx =
    ctx_make t t.root_page_table vaddr ~create_pts:true ~free_pts:false
      ~inspect:false ~boot
  in
  List.iter
    (fun (paddr, psize) ->
      ctx_next ctx ~size:psize ~paddr
        (fun ~off ~pt ~pte_idx ~n_ptes ~pte_covers ->
          for pte_off = 0 to n_ptes - 1 do
            t.pt_ops.set_entry pt ~idx:(pte_idx + pte_off)
              ~paddr:(paddr + off + (pte_off * pte_covers))
              ~uncached ~aspace ~snooped
              ~frag:(frag_size (ctx.vaddr + off) (n_ptes * pte_covers))
              ~valid:true ()
          done))
    paddrs;
  t.on_range_mapped ();
  { va_addr = vaddr; size; paddrs; aspace; uncached; snooped }

let unmap_range t ~vaddr ~size =
  if Helpers.getenv "MM_DEBUG" 0 <> 0 then
    Printf.printf "%s: unmapping vaddr=0x%x (size=0x%x)\n%!" t.dbg_name vaddr
      size;
  let ctx =
    ctx_make t t.root_page_table vaddr ~create_pts:false ~free_pts:true
      ~inspect:false ~boot:false
  in
  ctx_next ctx ~size (fun ~off:_ ~pt ~pte_idx ~n_ptes ~pte_covers:_ ->
      for pte_id = pte_idx to pte_idx + n_ptes - 1 do
        if not (t.pt_ops.valid pt pte_id) then
          invalid_arg
            (Printf.sprintf "PTE not mapped: 0x%Lx" (t.pt_ops.entry pt pte_id));
        t.pt_ops.set_entry pt ~idx:pte_id ~paddr:0x0 ~valid:false ()
      done)

let alloc_vaddr t size ?(align = 0x1000) () =
  if size <= 0 then invalid_arg "size must be positive";
  Tlsf.alloc t.va_allocator size
    ~align:(max (1 lsl (bit_length size - 1)) align)
    ()

let identity_va t ~uncached =
  match List.assoc_opt uncached t.identity_vas with
  | Some va -> va
  | None ->
      let va = alloc_vaddr t t.vram_size ~align:t.vram_size () in
      let (_ : virt_mapping) =
        map_range t ~vaddr:va ~size:t.vram_size
          [ (0, t.vram_size) ]
          Phys ~uncached ()
      in
      t.identity_vas <- (uncached, va) :: t.identity_vas;
      va

let valloc t size ?(align = 0x1000) ?(uncached = false) ?(contiguous = false) ()
    =
  let size = round_up size 0x1000 in
  if Helpers.getenv "GMMU" 1 = 0 then begin
    let paddr = palloc t size ~align ~zero:false () in
    {
      va_addr = identity_va t ~uncached + paddr;
      size;
      paddrs = [ (paddr, size) ];
      aspace = Phys;
      uncached;
      snooped = false;
    }
  end
  else begin
    (* Allocate physical memory and map it to the virtual address. *)
    let va = alloc_vaddr t size ~align () in
    let paddrs =
      if contiguous then [ (palloc t size ~zero:true (), size) ]
      else begin
        (* Allocate the longest possible segments to reduce TLB pressure,
           moving to smaller ranges as the larger ones run out. *)
        let n_ranges = Array.length t.palloc_ranges in
        let nxt_range = ref 0 and rem_size = ref size in
        let paddrs = ref [] in
        while !rem_size > 0 do
          while fst t.palloc_ranges.(!nxt_range) > !rem_size do
            incr nxt_range
          done;
          let try_sz, try_align = t.palloc_ranges.(!nxt_range) in
          match palloc t try_sz ~align:try_align ~zero:false () with
          | paddr ->
              paddrs := (paddr, try_sz) :: !paddrs;
              rem_size := !rem_size - try_sz
          | exception Tlsf.Out_of_memory _ ->
              incr nxt_range;
              if !nxt_range = n_ranges then begin
                List.iter (fun (paddr, _) -> pfree t paddr ()) !paddrs;
                raise
                  (Tlsf.Out_of_memory
                     (Printf.sprintf
                        "Failed to allocate memory (OOM). Request size=0x%x"
                        size))
              end
        done;
        List.rev !paddrs
      end
    in
    map_range t ~vaddr:va ~size paddrs Phys ~uncached ()
  end

let vfree t vm =
  if Helpers.getenv "GMMU" 1 = 0 then pfree t (fst (List.hd vm.paddrs)) ()
  else begin
    unmap_range t ~vaddr:vm.va_addr ~size:vm.size;
    Tlsf.free t.va_allocator vm.va_addr;
    List.iter (fun (paddr, _) -> pfree t paddr ()) vm.paddrs
  end
