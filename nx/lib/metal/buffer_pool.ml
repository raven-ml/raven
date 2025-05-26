open Metal

type buffer_entry = { buffer : Buffer.t; size : int; mutable in_use : bool }

type t = {
  device : Device.t;
  mutable buffers : buffer_entry list;
  mutable total_allocated : int;
  mutable total_in_use : int;
}

let create device =
  { device; buffers = []; total_allocated = 0; total_in_use = 0 }

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

  (* First, try to find a free buffer of sufficient size *)
  let rec find_free_buffer = function
    | [] -> None
    | entry :: rest ->
        if (not entry.in_use) && entry.size >= size then (
          entry.in_use <- true;
          pool.total_in_use <- pool.total_in_use + entry.size;
          Some entry.buffer)
        else find_free_buffer rest
  in

  match find_free_buffer pool.buffers with
  | Some buffer -> buffer
  | None ->
      (* No suitable buffer found, allocate a new one *)
      let options =
        ResourceOptions.(storage_mode_shared + hazard_tracking_mode_tracked)
      in
      let buffer = Buffer.on_device pool.device ~length:size options in
      let entry = { buffer; size; in_use = true } in
      pool.buffers <- entry :: pool.buffers;
      pool.total_allocated <- pool.total_allocated + size;
      pool.total_in_use <- pool.total_in_use + size;
      buffer

let release pool buffer =
  let rec mark_free = function
    | [] -> ()
    | entry :: rest ->
        if entry.buffer = buffer then (
          entry.in_use <- false;
          pool.total_in_use <- pool.total_in_use - entry.size)
        else mark_free rest
  in
  mark_free pool.buffers

let cleanup pool =
  (* Remove buffers that haven't been used recently *)
  (* For now, just remove all unused buffers *)
  let in_use, free = List.partition (fun e -> e.in_use) pool.buffers in
  let freed_size = List.fold_left (fun acc e -> acc + e.size) 0 free in
  pool.buffers <- in_use;
  pool.total_allocated <- pool.total_allocated - freed_size

let stats pool =
  let num_buffers = List.length pool.buffers in
  let num_in_use = List.length (List.filter (fun e -> e.in_use) pool.buffers) in
  Printf.sprintf
    "BufferPool: %d buffers, %d in use, %d bytes allocated, %d bytes in use"
    num_buffers num_in_use pool.total_allocated pool.total_in_use
