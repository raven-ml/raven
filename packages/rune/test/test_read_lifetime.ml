(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A read borrows its buffer's memory while it copies the elements out: the
   buffer stays while it does, even when the value read is a temporary and a
   collection runs in between, forced here at every bigarray allocation. The
   value is uploaded from a mapped file, so its buffer bypasses the allocator's
   cache and a buffer freed under the copy returns to the system: the read
   faults. A plain executable, since whether the temporary is dead at the copy
   depends on how its caller is compiled. *)

let cpu1 = Rune.device "CPU:1"

(* An int32 tensor of [n] elements over a mapping of a fresh file. *)
let mapped n =
  let path = Filename.temp_file "rune_read_" ".bin" in
  let oc = open_out_bin path in
  for i = 0 to (4 * n) - 1 do
    output_char oc (Char.chr (i land 0xff))
  done;
  close_out oc;
  let fd = Unix.openfile path [ Unix.O_RDONLY ] 0 in
  let st = Unix.fstat fd in
  let m =
    Nx_buffer.of_bigarray1
      (Bigarray.array1_of_genarray
         (Unix.map_file fd Bigarray.int8_unsigned Bigarray.c_layout false
            [| -1 |]))
  in
  Unix.close fd;
  Nx_buffer.register_file
    { path; size = 4 * n; mtime = st.st_mtime; inode = st.st_ino }
    m;
  (Nx.of_buffer (Nx_buffer.reinterpret Nx_dtype.Int32 m) ~shape:[| n |], path)

(* [f ()] with a full collection at every bigarray allocation. *)
let collecting f =
  let full (info : Gc.Memprof.allocation) =
    if info.source = Gc.Memprof.Custom then Gc.full_major ();
    None
  in
  ignore
    (Gc.Memprof.start ~sampling_rate:1.0
       { Gc.Memprof.null_tracker with alloc_minor = full; alloc_major = full });
  Fun.protect ~finally:Gc.Memprof.stop f

let () =
  let x, path = mapped (1 lsl 16) in
  for _ = 1 to 3 do
    let differing =
      collecting (fun () ->
          Nx.not_equal (Nx.place (Nx.Placement.device cpu1) x) x)
    in
    if Nx.item [] (Nx.sum (Nx.cast Nx.int32 differing)) <> 0l then
      failwith "a read of a placed temporary returned other elements"
  done;
  Sys.remove path
