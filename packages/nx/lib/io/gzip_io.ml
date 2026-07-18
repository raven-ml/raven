(*--------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  --------------------------------------------------------------------------*)

open Bigarray

type bytes = (int, int8_unsigned_elt, c_layout) Array1.t

let error fmt = Printf.ksprintf failwith fmt

let byte data off =
  if off < 0 || off >= Array1.dim data then error "truncated gzip stream";
  Array1.unsafe_get data off

let u16 data off = byte data off lor (byte data (off + 1) lsl 8)

let u32 data off =
  let value =
    Int64.logor
      (Int64.of_int (byte data off))
      (Int64.logor
         (Int64.shift_left (Int64.of_int (byte data (off + 1))) 8)
         (Int64.logor
            (Int64.shift_left (Int64.of_int (byte data (off + 2))) 16)
            (Int64.shift_left (Int64.of_int (byte data (off + 3))) 24)))
  in
  if value > Int64.of_int max_int then
    error "gzip member exceeds the OCaml integer range";
  Int64.to_int value

let i32_bits data off =
  let open Int32 in
  logor
    (of_int (byte data off))
    (logor
       (shift_left (of_int (byte data (off + 1))) 8)
       (logor
          (shift_left (of_int (byte data (off + 2))) 16)
          (shift_left (of_int (byte data (off + 3))) 24)))

let map_file fd size =
  if size = 0 then Array1.create int8_unsigned c_layout 0
  else
    Unix.map_file fd int8_unsigned c_layout false [| size |]
    |> Bigarray.array1_of_genarray

let zero_terminated data start limit field =
  let rec scan pos =
    if pos >= limit then error "unterminated gzip %s" field;
    if Array1.unsafe_get data pos = 0 then pos + 1 else scan (pos + 1)
  in
  scan start

let member_header data start =
  let limit = Array1.dim data in
  if start > limit - 10 then error "truncated gzip member header";
  if byte data start <> 0x1f || byte data (start + 1) <> 0x8b then
    error "invalid gzip magic at byte %d" start;
  if byte data (start + 2) <> 8 then error "unsupported gzip compression method";
  let flags = byte data (start + 3) in
  if flags land 0xe0 <> 0 then error "invalid reserved gzip flags";
  let pos = ref (start + 10) in
  if flags land 0x04 <> 0 then (
    if !pos > limit - 2 then error "truncated gzip extra field";
    let length = u16 data !pos in
    pos := !pos + 2;
    if length > limit - !pos then error "truncated gzip extra field";
    pos := !pos + length);
  if flags land 0x08 <> 0 then pos := zero_terminated data !pos limit "filename";
  if flags land 0x10 <> 0 then pos := zero_terminated data !pos limit "comment";
  if flags land 0x02 <> 0 then (
    if !pos > limit - 2 then error "truncated gzip header checksum";
    let expected = u16 data !pos in
    let actual =
      Nx_io_codec.crc32 data ~off:start ~len:(!pos - start) |> Int32.to_int
      |> fun crc -> crc land 0xffff
    in
    if actual <> expected then error "gzip header checksum mismatch";
    pos := !pos + 2);
  !pos

let output_member fd data start =
  let deflate = member_header data start in
  let available = Array1.dim data - deflate in
  let consumed, output_size, actual_crc =
    Nx_io_codec.inflate_raw_member_to_fd fd data ~src_off:deflate
      ~src_len:available
  in
  let footer = deflate + consumed in
  if footer > Array1.dim data - 8 then error "truncated gzip member footer";
  let expected_crc = i32_bits data footer in
  let expected_size = u32 data (footer + 4) in
  let actual_size =
    Int64.logand (Int64.of_int output_size) 0xffffffffL |> Int64.to_int
  in
  if actual_crc <> expected_crc then error "gzip payload checksum mismatch";
  if actual_size <> expected_size then error "gzip payload size mismatch";
  footer + 8

let gunzip ~src ~dst =
  let input_fd = Unix.openfile src [ Unix.O_RDONLY ] 0 in
  let data =
    Fun.protect
      ~finally:(fun () -> Unix.close input_fd)
      (fun () -> map_file input_fd (Unix.fstat input_fd).st_size)
  in
  if Array1.dim data = 0 then error "empty gzip stream";
  let temp =
    Filename.temp_file ~temp_dir:(Filename.dirname dst)
      (Filename.basename dst ^ ".")
      ".tmp"
  in
  match
    let output_fd = Unix.openfile temp [ Unix.O_WRONLY; Unix.O_TRUNC ] 0 in
    Fun.protect
      ~finally:(fun () -> Unix.close output_fd)
      (fun () ->
        let rec members off =
          let next = output_member output_fd data off in
          if next < Array1.dim data then members next
          else if next > Array1.dim data then error "truncated gzip stream"
        in
        members 0)
  with
  | () -> (
      match
        Unix.chmod temp 0o640;
        Unix.rename temp dst
      with
      | () -> ()
      | exception exn ->
          (try Sys.remove temp with Sys_error _ -> ());
          raise exn)
  | exception exn ->
      (try Sys.remove temp with Sys_error _ -> ());
      raise exn
