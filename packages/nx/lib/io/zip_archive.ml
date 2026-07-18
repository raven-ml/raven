(*--------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  The ZIP subset used by NPZ: stored and DEFLATE entries, data descriptors,
  UTF-8 names, and ZIP64 sizes and offsets. Multi-disk and encrypted archives
  are intentionally rejected.
  --------------------------------------------------------------------------*)

open Bigarray

type bytes = (int, int8_unsigned_elt, c_layout) Array1.t
type method_ = Store | Deflate

type entry = {
  name : string;
  method_ : method_;
  flags : int;
  crc32 : int32;
  compressed_size : int;
  uncompressed_size : int;
  local_offset : int;
}

type in_file = { fd : Unix.file_descr; data : bytes; entries : entry list }
type written_entry = { entry : entry; zip64_sizes : bool }

type out_file = {
  fd : Unix.file_descr;
  mutable offset : int;
  mutable entries_rev : written_entry list;
  names : (string, unit) Hashtbl.t;
  mutable closed : bool;
}

let error fmt = Printf.ksprintf failwith fmt

let valid_utf8 text =
  let length = String.length text in
  let continuation index =
    index < length
    &&
    let byte = Char.code text.[index] in
    byte land 0xc0 = 0x80
  in
  let rec loop index =
    if index = length then true
    else
      let first = Char.code text.[index] in
      if first < 0x80 then loop (index + 1)
      else if first >= 0xc2 && first <= 0xdf then
        continuation (index + 1) && loop (index + 2)
      else if first >= 0xe0 && first <= 0xef then
        index + 2 < length
        && continuation (index + 1)
        && continuation (index + 2)
        &&
        let second = Char.code text.[index + 1] in
        (first <> 0xe0 || second >= 0xa0)
        && (first <> 0xed || second < 0xa0)
        && loop (index + 3)
      else if first >= 0xf0 && first <= 0xf4 then
        index + 3 < length
        && continuation (index + 1)
        && continuation (index + 2)
        && continuation (index + 3)
        &&
        let second = Char.code text.[index + 1] in
        (first <> 0xf0 || second >= 0x90)
        && (first <> 0xf4 || second < 0x90)
        && loop (index + 4)
      else false
  in
  loop 0

let safe_name name =
  let length = String.length name in
  length > 0
  && name.[0] <> '/'
  && (not (String.contains name '\\'))
  && (not (String.contains name '\x00'))
  && (not (length >= 2 && name.[1] = ':'))
  && List.for_all
       (fun component ->
         component <> "" && component <> "." && component <> "..")
       (String.split_on_char '/' name)

let validate_name ~utf8 name =
  if not (safe_name name) then
    error "unsafe or ambiguous ZIP entry name %S" name;
  if utf8 then (
    if not (valid_utf8 name) then error "invalid UTF-8 ZIP entry name")
  else
    String.iter
      (fun byte ->
        if Char.code byte >= 0x80 then
          error "non-ASCII legacy ZIP entry name %S" name)
      name

let checked_add context a b =
  if a < 0 || b < 0 || a > max_int - b then error "ZIP %s is too large" context;
  a + b

let byte data off =
  if off < 0 || off >= Array1.dim data then error "truncated ZIP structure";
  Array1.unsafe_get data off

let u16 data off = byte data off lor (byte data (off + 1) lsl 8)

let u32_i64 data off =
  Int64.logor
    (Int64.of_int (byte data off))
    (Int64.logor
       (Int64.shift_left (Int64.of_int (byte data (off + 1))) 8)
       (Int64.logor
          (Int64.shift_left (Int64.of_int (byte data (off + 2))) 16)
          (Int64.shift_left (Int64.of_int (byte data (off + 3))) 24)))

let u64 data off =
  let value = ref 0L in
  for i = 7 downto 0 do
    value :=
      Int64.logor
        (Int64.shift_left !value 8)
        (Int64.of_int (byte data (off + i)))
  done;
  if !value < 0L || !value > Int64.of_int max_int then
    error "ZIP64 value exceeds the OCaml integer range";
  Int64.to_int !value

let u32 data off =
  let value = u32_i64 data off in
  if value > Int64.of_int max_int then
    error "ZIP value exceeds the OCaml integer range";
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

let signature data off expected =
  if u32_i64 data off <> expected then
    error "invalid ZIP signature at byte %d" off

let substring data off len =
  if off < 0 || len < 0 || off > Array1.dim data || len > Array1.dim data - off
  then error "truncated ZIP string";
  String.init len (fun i -> Char.chr (Array1.unsafe_get data (off + i)))

let map_file fd size =
  if size = 0 then Array1.create int8_unsigned c_layout 0
  else
    Unix.map_file fd int8_unsigned c_layout false [| size |]
    |> Bigarray.array1_of_genarray

let find_eocd data =
  let size = Array1.dim data in
  if size < 22 then error "file is too short to be a ZIP archive";
  let first = max 0 (size - 22 - 0xffff) in
  let rec scan off =
    if off < first then error "ZIP end-of-central-directory record not found"
    else if u32_i64 data off = 0x06054b50L then
      let comment_len = u16 data (off + 20) in
      if off + 22 + comment_len = size then off else scan (off - 1)
    else scan (off - 1)
  in
  scan (size - 22)

let zip64_directory data eocd =
  if eocd < 20 then error "missing ZIP64 locator";
  let locator = eocd - 20 in
  signature data locator 0x07064b50L;
  if u32 data (locator + 4) <> 0 || u32 data (locator + 16) <> 1 then
    error "multi-disk ZIP archives are unsupported";
  let record = u64 data (locator + 8) in
  signature data record 0x06064b50L;
  let record_size = u64 data (record + 4) in
  if
    record_size < 44
    || record > Array1.dim data
    || record_size > Array1.dim data - record - 12
  then error "truncated ZIP64 end record";
  if u32 data (record + 16) <> 0 || u32 data (record + 20) <> 0 then
    error "multi-disk ZIP archives are unsupported";
  let entries_on_disk = u64 data (record + 24) in
  let entry_count = u64 data (record + 32) in
  if entries_on_disk <> entry_count then
    error "split ZIP archives are unsupported";
  (entry_count, u64 data (record + 40), u64 data (record + 48))

let classic_directory data eocd =
  let disk = u16 data (eocd + 4) in
  let central_disk = u16 data (eocd + 6) in
  let entries_on_disk = u16 data (eocd + 8) in
  let entry_count = u16 data (eocd + 10) in
  if disk <> 0 || central_disk <> 0 || entries_on_disk <> entry_count then
    error "multi-disk ZIP archives are unsupported";
  (entry_count, u32 data (eocd + 12), u32 data (eocd + 16))

let parse_zip64_extra data off len ~compressed ~uncompressed ~local_offset =
  let limit = checked_add "extra field" off len in
  if limit > Array1.dim data then error "truncated ZIP extra field";
  let rec fields pos =
    if pos = limit then None
    else if pos > limit - 4 then error "truncated ZIP extra field"
    else
      let tag = u16 data pos in
      let size = u16 data (pos + 2) in
      let payload = pos + 4 in
      if payload > limit || size > limit - payload then
        error "truncated ZIP extra field";
      if tag = 0x0001 then Some (payload, size) else fields (payload + size)
  in
  match fields off with
  | None -> error "missing ZIP64 extended information"
  | Some (pos, size) ->
      let cursor = ref pos in
      let remaining () = pos + size - !cursor in
      let take () =
        if remaining () < 8 then error "truncated ZIP64 extended information";
        let value = u64 data !cursor in
        cursor := !cursor + 8;
        value
      in
      let uncompressed =
        if uncompressed = 0xffffffffL then take ()
        else Int64.to_int uncompressed
      in
      let compressed =
        if compressed = 0xffffffffL then take () else Int64.to_int compressed
      in
      let local_offset =
        if local_offset = 0xffffffffL then take ()
        else Int64.to_int local_offset
      in
      (compressed, uncompressed, local_offset)

let parse_entries data ~count ~central_size ~central_offset =
  let size = Array1.dim data in
  if central_offset > size || central_size > size - central_offset then
    error "ZIP central directory is out of bounds";
  if count > central_size / 46 then
    error "ZIP entry count exceeds the central-directory size";
  let central_end = central_offset + central_size in
  let rec loop index off entries =
    if index = count then (
      if off <> central_end then error "ZIP central-directory size mismatch";
      List.rev entries)
    else (
      if off > central_end - 46 then error "truncated ZIP central directory";
      signature data off 0x02014b50L;
      let flags = u16 data (off + 8) in
      if flags land 0x0001 <> 0 then
        error "encrypted ZIP entries are unsupported";
      let method_ =
        match u16 data (off + 10) with
        | 0 -> Store
        | 8 -> Deflate
        | method_ -> error "unsupported ZIP compression method %d" method_
      in
      let allowed_flags =
        match method_ with Store -> 0x0808 | Deflate -> 0x080e
      in
      if flags land lnot allowed_flags <> 0 then
        error "unsupported ZIP flags 0x%04x" flags;
      let crc32 = i32_bits data (off + 16) in
      let compressed32 = u32_i64 data (off + 20) in
      let uncompressed32 = u32_i64 data (off + 24) in
      let name_len = u16 data (off + 28) in
      let extra_len = u16 data (off + 30) in
      let comment_len = u16 data (off + 32) in
      if u16 data (off + 34) <> 0 then
        error "multi-disk ZIP archives are unsupported";
      let local32 = u32_i64 data (off + 42) in
      let variable =
        checked_add "central entry" name_len
          (checked_add "central entry" extra_len comment_len)
      in
      let next = checked_add "central entry" (off + 46) variable in
      if next > central_end then error "truncated ZIP central entry";
      let name = substring data (off + 46) name_len in
      validate_name ~utf8:(flags land 0x0800 <> 0) name;
      let extra_off = off + 46 + name_len in
      let compressed_size, uncompressed_size, local_offset =
        if
          compressed32 = 0xffffffffL
          || uncompressed32 = 0xffffffffL
          || local32 = 0xffffffffL
        then
          parse_zip64_extra data extra_off extra_len ~compressed:compressed32
            ~uncompressed:uncompressed32 ~local_offset:local32
        else
          ( Int64.to_int compressed32,
            Int64.to_int uncompressed32,
            Int64.to_int local32 )
      in
      if method_ = Store && compressed_size <> uncompressed_size then
        error "stored ZIP entry %S has inconsistent sizes" name;
      let entry =
        {
          name;
          method_;
          flags;
          crc32;
          compressed_size;
          uncompressed_size;
          local_offset;
        }
      in
      loop (index + 1) next (entry :: entries))
  in
  loop 0 central_offset []

let validate_local_entry data ~central_offset entry =
  let off = entry.local_offset in
  if off < 0 || off > central_offset - 30 then
    error "ZIP local header for %S is out of bounds" entry.name;
  signature data off 0x04034b50L;
  let flags = u16 data (off + 6) in
  let method_code = u16 data (off + 8) in
  let expected_method = match entry.method_ with Store -> 0 | Deflate -> 8 in
  if flags <> entry.flags || method_code <> expected_method then
    error "ZIP local header disagrees with central entry %S" entry.name;
  let name_len = u16 data (off + 26) in
  let extra_len = u16 data (off + 28) in
  let payload = checked_add "local entry" (off + 30) (name_len + extra_len) in
  if
    payload > central_offset || entry.compressed_size > central_offset - payload
  then error "ZIP entry %S overlaps the central directory" entry.name;
  let local_name = substring data (off + 30) name_len in
  if not (String.equal local_name entry.name) then
    error "ZIP local filename disagrees with central entry %S" entry.name;
  let compressed32 = u32_i64 data (off + 18) in
  let uncompressed32 = u32_i64 data (off + 22) in
  let data_end = payload + entry.compressed_size in
  if flags land 0x0008 = 0 then (
    let local_compressed, local_uncompressed, _ =
      if compressed32 = 0xffffffffL || uncompressed32 = 0xffffffffL then
        parse_zip64_extra data
          (off + 30 + name_len)
          extra_len ~compressed:compressed32 ~uncompressed:uncompressed32
          ~local_offset:0L
      else (Int64.to_int compressed32, Int64.to_int uncompressed32, 0)
    in
    if
      i32_bits data (off + 14) <> entry.crc32
      || local_compressed <> entry.compressed_size
      || local_uncompressed <> entry.uncompressed_size
    then error "ZIP local metadata disagrees with central entry %S" entry.name;
    (off, data_end))
  else
    let zip64 =
      compressed32 = 0xffffffffL
      || uncompressed32 = 0xffffffffL
      || entry.compressed_size > 0xffffffff
      || entry.uncompressed_size > 0xffffffff
    in
    let size_bytes = if zip64 then 16 else 8 in
    let descriptor_at descriptor =
      if
        descriptor > central_offset - 4
        || size_bytes > central_offset - descriptor - 4
      then None
      else
        let crc = i32_bits data descriptor in
        let compressed, uncompressed =
          if zip64 then (u64 data (descriptor + 4), u64 data (descriptor + 12))
          else (u32 data (descriptor + 4), u32 data (descriptor + 8))
        in
        if
          crc = entry.crc32
          && compressed = entry.compressed_size
          && uncompressed = entry.uncompressed_size
        then Some (descriptor + 4 + size_bytes)
        else None
    in
    let finish =
      if data_end <= central_offset - 4 && u32_i64 data data_end = 0x08074b50L
      then
        match descriptor_at (data_end + 4) with
        | Some finish -> Some finish
        | None -> descriptor_at data_end
      else descriptor_at data_end
    in
    match finish with
    | Some finish -> (off, finish)
    | None -> error "invalid ZIP data descriptor for %S" entry.name

let validate_entries data ~central_offset entries =
  let names = Hashtbl.create (List.length entries) in
  List.iter
    (fun entry ->
      if Hashtbl.mem names entry.name then
        error "duplicate ZIP entry %S" entry.name;
      Hashtbl.add names entry.name ())
    entries;
  let spans =
    List.map (validate_local_entry data ~central_offset) entries
    |> List.sort (fun (left, _) (right, _) -> Int.compare left right)
  in
  let rec disjoint previous_end = function
    | [] -> ()
    | (start, finish) :: rest ->
        if start < previous_end then error "overlapping ZIP local records";
        disjoint finish rest
  in
  disjoint 0 spans

let open_in path =
  let fd = Unix.openfile path [ Unix.O_RDONLY ] 0 in
  match
    let size = (Unix.fstat fd).st_size in
    let data = map_file fd size in
    let eocd = find_eocd data in
    let classic_count = u16 data (eocd + 10) in
    let classic_size = u32_i64 data (eocd + 12) in
    let classic_offset = u32_i64 data (eocd + 16) in
    let count, central_size, central_offset =
      if
        classic_count = 0xffff || classic_size = 0xffffffffL
        || classic_offset = 0xffffffffL
      then zip64_directory data eocd
      else classic_directory data eocd
    in
    if central_offset > eocd || central_size > eocd - central_offset then
      error "ZIP central directory overlaps its end record";
    let entries = parse_entries data ~count ~central_size ~central_offset in
    validate_entries data ~central_offset entries;
    { fd; data; entries }
  with
  | value -> value
  | exception exn ->
      Unix.close fd;
      raise exn

let close_in (archive : in_file) = Unix.close archive.fd
let entries (archive : in_file) = archive.entries

let find_entry (archive : in_file) name =
  match
    List.find_opt (fun entry -> String.equal entry.name name) archive.entries
  with
  | Some entry -> entry
  | None -> raise Not_found

let data_span (archive : in_file) entry =
  let data = archive.data in
  let off = entry.local_offset in
  if off > Array1.dim data - 30 then
    error "truncated ZIP local header for %S" entry.name;
  signature data off 0x04034b50L;
  let flags = u16 data (off + 6) in
  let method_code = u16 data (off + 8) in
  let expected_method = match entry.method_ with Store -> 0 | Deflate -> 8 in
  if flags <> entry.flags || method_code <> expected_method then
    error "ZIP local header disagrees with central entry %S" entry.name;
  let name_len = u16 data (off + 26) in
  let extra_len = u16 data (off + 28) in
  let payload = checked_add "local entry" (off + 30) (name_len + extra_len) in
  if
    payload > Array1.dim data
    || entry.compressed_size > Array1.dim data - payload
  then error "ZIP entry %S is out of bounds" entry.name;
  let local_name = substring data (off + 30) name_len in
  if not (String.equal local_name entry.name) then
    error "ZIP local filename disagrees with central entry %S" entry.name;
  (payload, entry.compressed_size)

let npy_suffix = ".npy"

let logical_name filename =
  let suffix_len = String.length npy_suffix in
  let length = String.length filename in
  if
    length >= suffix_len
    && String.sub filename (length - suffix_len) suffix_len = npy_suffix
  then String.sub filename 0 (length - suffix_len)
  else filename

let npy_entries (archive : in_file) =
  List.filter_map
    (fun entry ->
      let logical = logical_name entry.name in
      if String.length logical = String.length entry.name then None
      else Some logical)
    archive.entries

let read_npy (archive : in_file) name =
  let entry = find_entry archive (name ^ npy_suffix) in
  let src_off, src_len = data_span archive entry in
  match entry.method_ with
  | Store ->
      let header = Npy.parse_header archive.data ~off:src_off ~len:src_len in
      if
        header.data_offset + header.data_size <> src_len
        || entry.uncompressed_size <> src_len
      then error "NPY size mismatch in ZIP entry %S" entry.name;
      let crc = Nx_io_codec.crc32 archive.data ~off:src_off ~len:src_len in
      if crc <> entry.crc32 then
        error "CRC-32 mismatch in ZIP entry %S" entry.name;
      Npy.materialize header
        (Npy.Stored { src = archive.data; off = src_off + header.data_offset })
      |> fst
  | Deflate ->
      let prefix =
        Nx_io_codec.inflate_raw_prefix archive.data ~off:src_off ~len:src_len
          ~max_output:(Npy.max_header_size + 12)
      in
      let header = Npy.parse_header prefix ~off:0 ~len:(Array1.dim prefix) in
      if header.data_offset + header.data_size <> entry.uncompressed_size then
        error "NPY size mismatch in ZIP entry %S" entry.name;
      let packed, crc =
        Npy.materialize header
          (Npy.Deflated
             { src = archive.data; src_off; src_len; skip = header.data_offset })
      in
      if crc <> Some entry.crc32 then
        error "CRC-32 mismatch in ZIP entry %S" entry.name;
      packed

let le16 value =
  String.init 2 (fun i -> Char.chr ((value lsr (8 * i)) land 0xff))

let le32_i64 value =
  String.init 4 (fun i ->
      Char.chr
        (Int64.to_int
           (Int64.logand (Int64.shift_right_logical value (8 * i)) 0xffL)))

let le32 value = le32_i64 (Int64.of_int value)

let le32_bits value =
  String.init 4 (fun i ->
      Char.chr
        (Int32.to_int
           (Int32.logand (Int32.shift_right_logical value (8 * i)) 0xffl)))

let le64 value =
  let value = Int64.of_int value in
  String.init 8 (fun i ->
      Char.chr
        (Int64.to_int
           (Int64.logand (Int64.shift_right_logical value (8 * i)) 0xffL)))

let write_string fd text =
  let rec loop off =
    if off < String.length text then (
      let written =
        Unix.write_substring fd text off (String.length text - off)
      in
      if written = 0 then raise (Unix.Unix_error (Unix.EIO, "write", ""));
      loop (off + written))
  in
  loop 0

let add_offset archive amount =
  archive.offset <- checked_add "output" archive.offset amount

let open_out ?(exclusive = false) path =
  let flags =
    if exclusive then [ Unix.O_CREAT; Unix.O_EXCL; Unix.O_WRONLY ]
    else [ Unix.O_CREAT; Unix.O_TRUNC; Unix.O_WRONLY ]
  in
  let fd = Unix.openfile path flags 0o640 in
  {
    fd;
    offset = 0;
    entries_rev = [];
    names = Hashtbl.create 16;
    closed = false;
  }

let worst_deflate_size size =
  checked_add "DEFLATE output" size (((size / 65535) + 1) * 5)

let sample_size = 64 * 1024

let choose_method (Npy.E encoded) =
  let available = max 0 (sample_size - String.length encoded.header) in
  let data_size = min encoded.data_size available in
  let compressed =
    Nx_io_codec.deflate_raw ~prefix:encoded.header encoded.data ~off:0
      ~len:data_size
  in
  if Array1.dim compressed < String.length encoded.header + data_size then
    Deflate
  else Store

let add_npy archive name packed =
  if archive.closed then invalid_arg "Zip_archive.add_npy: archive is closed";
  if name = "" then invalid_arg "Zip_archive.add_npy: empty entry name";
  if (not (safe_name name)) || not (valid_utf8 name) then
    invalid_arg "Zip_archive.add_npy: unsafe or invalid UTF-8 entry name";
  let filename = name ^ npy_suffix in
  if Hashtbl.mem archive.names filename then error "duplicate NPZ entry %S" name;
  let (Npy.E encoded as encoded_npy) = Npy.encode packed in
  let method_ = choose_method encoded_npy in
  Hashtbl.add archive.names filename ();
  let uncompressed_size =
    checked_add "NPY entry" (String.length encoded.header) encoded.data_size
  in
  let maximum_size =
    match method_ with
    | Store -> uncompressed_size
    | Deflate -> worst_deflate_size uncompressed_size
  in
  let zip64_sizes = maximum_size > 0xffffffff in
  let local_offset = archive.offset in
  let flags = 0x0808 in
  let version = if zip64_sizes then 45 else 20 in
  let method_code = match method_ with Store -> 0 | Deflate -> 8 in
  let extra =
    if zip64_sizes then le16 0x0001 ^ le16 16 ^ le64 0 ^ le64 0 else ""
  in
  let size_marker = if zip64_sizes then le32_i64 0xffffffffL else le32 0 in
  let local =
    le32_i64 0x04034b50L ^ le16 version ^ le16 flags ^ le16 method_code ^ le16 0
    ^ le16 0 ^ le32 0 ^ size_marker ^ size_marker
    ^ le16 (String.length filename)
    ^ le16 (String.length extra)
    ^ filename ^ extra
  in
  write_string archive.fd local;
  add_offset archive (String.length local);
  let stats =
    match method_ with
    | Store ->
        Nx_io_codec.store_to_fd archive.fd ~prefix:encoded.header encoded.data
          ~off:0 ~len:encoded.data_size
    | Deflate ->
        Nx_io_codec.deflate_raw_to_fd archive.fd ~prefix:encoded.header
          encoded.data ~off:0 ~len:encoded.data_size
  in
  add_offset archive stats.output_size;
  let descriptor =
    le32_i64 0x08074b50L ^ le32_bits stats.crc32
    ^
    if zip64_sizes then le64 stats.output_size ^ le64 stats.input_size
    else le32 stats.output_size ^ le32 stats.input_size
  in
  write_string archive.fd descriptor;
  add_offset archive (String.length descriptor);
  let entry =
    {
      name = filename;
      method_;
      flags;
      crc32 = stats.crc32;
      compressed_size = stats.output_size;
      uncompressed_size = stats.input_size;
      local_offset;
    }
  in
  archive.entries_rev <- { entry; zip64_sizes } :: archive.entries_rev

let central_record written =
  let entry = written.entry in
  let zip64_offset = entry.local_offset > 0xffffffff in
  let zip64 = written.zip64_sizes || zip64_offset in
  let version = if zip64 then 45 else 20 in
  let extra_payload =
    (if written.zip64_sizes then
       le64 entry.uncompressed_size ^ le64 entry.compressed_size
     else "")
    ^ if zip64_offset then le64 entry.local_offset else ""
  in
  let extra =
    if extra_payload = "" then ""
    else le16 0x0001 ^ le16 (String.length extra_payload) ^ extra_payload
  in
  let compressed =
    if written.zip64_sizes then le32_i64 0xffffffffL
    else le32 entry.compressed_size
  in
  let uncompressed =
    if written.zip64_sizes then le32_i64 0xffffffffL
    else le32 entry.uncompressed_size
  in
  let offset =
    if zip64_offset then le32_i64 0xffffffffL else le32 entry.local_offset
  in
  let method_code = match entry.method_ with Store -> 0 | Deflate -> 8 in
  le32_i64 0x02014b50L ^ le16 0x031e ^ le16 version ^ le16 entry.flags
  ^ le16 method_code ^ le16 0 ^ le16 0 ^ le32_bits entry.crc32 ^ compressed
  ^ uncompressed
  ^ le16 (String.length entry.name)
  ^ le16 (String.length extra)
  ^ le16 0 ^ le16 0 ^ le16 0 ^ le32 0 ^ offset ^ entry.name ^ extra

let close_out archive =
  if not archive.closed then (
    let central_offset = archive.offset in
    let records = List.rev archive.entries_rev |> List.map central_record in
    List.iter
      (fun record ->
        write_string archive.fd record;
        add_offset archive (String.length record))
      records;
    let central_size = archive.offset - central_offset in
    let count = List.length archive.entries_rev in
    let zip64_directory =
      count > 0xffff || central_size > 0xffffffff || central_offset > 0xffffffff
    in
    if zip64_directory then (
      let zip64_offset = archive.offset in
      let record =
        le32_i64 0x06064b50L ^ le64 44 ^ le16 0x031e ^ le16 45 ^ le32 0 ^ le32 0
        ^ le64 count ^ le64 count ^ le64 central_size ^ le64 central_offset
      in
      write_string archive.fd record;
      add_offset archive (String.length record);
      let locator =
        le32_i64 0x07064b50L ^ le32 0 ^ le64 zip64_offset ^ le32 1
      in
      write_string archive.fd locator;
      add_offset archive (String.length locator));
    let count16 = if count > 0xffff then 0xffff else count in
    let size32 =
      if central_size > 0xffffffff then le32_i64 0xffffffffL
      else le32 central_size
    in
    let offset32 =
      if central_offset > 0xffffffff then le32_i64 0xffffffffL
      else le32 central_offset
    in
    let eocd =
      le32_i64 0x06054b50L ^ le16 0 ^ le16 0 ^ le16 count16 ^ le16 count16
      ^ size32 ^ offset32 ^ le16 0
    in
    write_string archive.fd eocd;
    archive.closed <- true;
    Unix.close archive.fd)

let abort_out archive =
  if not archive.closed then (
    archive.closed <- true;
    Unix.close archive.fd)
