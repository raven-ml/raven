open Metal

type buffer_entry = { buffer : Buffer.t; size : int; mutable in_use : bool }

type t = {
  device : Device.t;
  (* Free buffers grouped by size for O(1) lookup *)
  free_buffers : (int, buffer_entry list) Hashtbl.t;
  (* In-use buffers indexed by buffer for O(1) release *)
  in_use_buffers : (Buffer.t, buffer_entry) Hashtbl.t;
  mutable total_allocated : int;
  mutable total_in_use : int;
}

let create device =
  {
    device;
    free_buffers = Hashtbl.create 32;
    in_use_buffers = Hashtbl.create 128;
    total_allocated = 0;
    total_in_use = 0;
  }

let round_up_size size =
  (* Round up to next power of 2 or multiple of 256 bytes for better reuse *)
  let min_size = 256 in
  let size = max size min_size in
  if size <= 4096 then
    (* For small sizes, round to next power of 2 *)
    let rec next_pow2 n = if n >= size then n else next_pow2 (n * 2) in
    next_pow2 min_size
  else
    (* For larger sizes, round up to 4KB boundaries *)
    (size + 4095) / 4096 * 4096

let allocate pool size =
  let size = round_up_size size in

  (* Try to find a free buffer of the exact size - O(1) operation *)
  match Hashtbl.find_opt pool.free_buffers size with
  | Some (entry :: rest) ->
      (* Found a free buffer of the right size *)
      entry.in_use <- true;
      pool.total_in_use <- pool.total_in_use + entry.size;
      (* Update free list *)
      if rest = [] then Hashtbl.remove pool.free_buffers size
      else Hashtbl.replace pool.free_buffers size rest;
      (* Add to in-use table *)
      Hashtbl.add pool.in_use_buffers entry.buffer entry;
      entry.buffer
  | Some [] | None ->
      (* No suitable buffer found, allocate a new one *)
      let options =
        ResourceOptions.(storage_mode_shared + hazard_tracking_mode_tracked)
      in
      let buffer = Buffer.on_device pool.device ~length:size options in
      let entry = { buffer; size; in_use = true } in
      (* Add to in-use table *)
      Hashtbl.add pool.in_use_buffers buffer entry;
      pool.total_allocated <- pool.total_allocated + size;
      pool.total_in_use <- pool.total_in_use + size;
      buffer

let release pool buffer =
  (* O(1) lookup in the in-use table *)
  match Hashtbl.find_opt pool.in_use_buffers buffer with
  | Some entry ->
      entry.in_use <- false;
      pool.total_in_use <- pool.total_in_use - entry.size;
      (* Remove from in-use table *)
      Hashtbl.remove pool.in_use_buffers buffer;
      (* Add to free list for this size *)
      let current_list =
        Option.value ~default:[] (Hashtbl.find_opt pool.free_buffers entry.size)
      in
      Hashtbl.replace pool.free_buffers entry.size (entry :: current_list)
  | None ->
      (* Buffer not found - this shouldn't happen in normal operation *)
      ()

let cleanup pool =
  (* Remove all free buffers to reclaim memory *)
  let freed_size = ref 0 in
  Hashtbl.iter
    (fun size entries ->
      List.iter (fun entry -> freed_size := !freed_size + entry.size) entries)
    pool.free_buffers;
  Hashtbl.clear pool.free_buffers;
  pool.total_allocated <- pool.total_allocated - !freed_size

let stats pool =
  let num_in_use = Hashtbl.length pool.in_use_buffers in
  let num_free = ref 0 in
  Hashtbl.iter
    (fun _ entries -> num_free := !num_free + List.length entries)
    pool.free_buffers;
  let num_buffers = num_in_use + !num_free in
  Printf.sprintf
    "BufferPool: %d buffers, %d in use, %d bytes allocated, %d bytes in use"
    num_buffers num_in_use pool.total_allocated pool.total_in_use
