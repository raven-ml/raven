(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  The zip subset npz uses: stored and DEFLATE entries, data descriptors, UTF-8
  names, and ZIP64 sizes and offsets. Archives over several disks are refused,
  and so are npy entries that are encrypted or of another method. The other
  entries are not read. Of them, only the records' bounds and agreement and the
  names' distinctness are checked, since a name is only an archive's key.
  ---------------------------------------------------------------------------*)

open Bigarray
module Crc32 = Compress_deflate.Crc32
module Encoder = Compress_deflate.Encoder

type bytes = (int, int8_unsigned_elt, c_layout) Array1.t

(* The compression methods of npy entries (APPNOTE, 4.4.5). *)
let stored = 0
let deflated = 8

type entry = {
  name : string;
  method_ : int;
  flags : int;
  crc32 : Crc32.t;
  compressed_size : int;
  uncompressed_size : int;
  local_offset : int;
}

(* [file] is the archive's bytes on the host. *)
type in_file = { file : Rig.Buffer.t; entries : entry list }

type out_file = {
  fd : Unix.file_descr;
  mutable offset : int;
  mutable entries_rev : entry list;
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

let signature data off expected =
  if u32_i64 data off <> expected then
    error "invalid ZIP signature at byte %d" off

let substring data off len =
  if off < 0 || len < 0 || off > Array1.dim data || len > Array1.dim data - off
  then error "truncated ZIP string";
  String.init len (fun i -> Char.chr (Array1.unsafe_get data (off + i)))

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
  let locator = eocd - 20 in
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
      let method_ = u16 data (off + 10) in
      let crc32 = u32 data (off + 16) in
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
  if flags <> entry.flags || method_code <> entry.method_ then
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
      u32 data (off + 14) <> entry.crc32
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
        let crc = u32 data descriptor in
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

let open_in file =
  let file = Storage.pages file in
  Storage.read file @@ fun data ->
  let eocd = find_eocd data in
  (* A ZIP64 end record's locator lies just before the end record. Without one,
     the classic fields hold their values, at their marker values too: Python's
     zipfile writes an archive of 65535 entries so. *)
  let count, central_size, central_offset =
    if eocd >= 20 && u32_i64 data (eocd - 20) = 0x07064b50L then
      zip64_directory data eocd
    else classic_directory data eocd
  in
  if central_offset > eocd || central_size > eocd - central_offset then
    error "ZIP central directory overlaps its end record";
  let entries = parse_entries data ~count ~central_size ~central_offset in
  validate_entries data ~central_offset entries;
  { file; entries }

let data_span data entry =
  let off = entry.local_offset in
  if off > Array1.dim data - 30 then
    error "truncated ZIP local header for %S" entry.name;
  signature data off 0x04034b50L;
  let name_len = u16 data (off + 26) in
  let extra_len = u16 data (off + 28) in
  let payload = checked_add "local entry" (off + 30) (name_len + extra_len) in
  if
    payload > Array1.dim data
    || entry.compressed_size > Array1.dim data - payload
  then error "ZIP entry %S is out of bounds" entry.name;
  (payload, entry.compressed_size)

let npy_suffix = ".npy"

(* The bytes of an entry: where they lie in the archive, or inflated. *)
type data = Stored of int * int | Inflated of bytes

(* [check_npy entry] fails unless the npy entry [entry] is stored or deflated,
   with no flag but a data descriptor, UTF-8 and DEFLATE's options. Only an npy
   entry is checked so: the archive's other entries are not read. *)
let check_npy entry =
  if entry.flags land 0x0001 <> 0 then
    error "encrypted ZIP entries are unsupported";
  let allowed_flags =
    if entry.method_ = stored then 0x0808
    else if entry.method_ = deflated then 0x080e
    else error "unsupported ZIP compression method %d" entry.method_
  in
  if entry.flags land lnot allowed_flags <> 0 then
    error "unsupported ZIP flags 0x%04x" entry.flags;
  if entry.method_ = stored && entry.compressed_size <> entry.uncompressed_size
  then error "stored ZIP entry %S has inconsistent sizes" entry.name

(* The array of the npy entry [entry]. *)
let read_npy (archive : in_file) entry =
  check_npy entry;
  let data =
    Storage.read archive.file @@ fun data ->
    let off, len = data_span data entry in
    let src = Array1.sub data off len in
    if entry.method_ = stored then (
      if Crc32.bigbytes src <> entry.crc32 then
        error "CRC-32 mismatch in ZIP entry %S" entry.name;
      Stored (off, len))
    else (
      (* Deflate expands at most 1032 times: 258 bytes from two bits. *)
      if entry.uncompressed_size / 1032 > len then
        error "ZIP entry %S declares more data than its DEFLATE data holds"
          entry.name;
      let inflated =
        Array1.create int8_unsigned c_layout entry.uncompressed_size
      in
      (match Compress_deflate.Deflate.decompress_into src inflated with
      | Ok () -> ()
      | Error e -> error "invalid DEFLATE data in ZIP entry %S: %s" entry.name e);
      if Crc32.bigbytes inflated <> entry.crc32 then
        error "CRC-32 mismatch in ZIP entry %S" entry.name;
      Inflated inflated)
  in
  let entry b value =
    let read pos n = Storage.string b ~pos n in
    value (Npy.header read ~length:(Rig.Buffer.length b)) b
  in
  match data with
  | Stored (first, length) ->
      entry (Rig.Buffer.view archive.file ~first ~length) Npy.value
  | Inflated a ->
      (* The inflated bytes are the entry's alone: its value owns a copy, in
         memory rig may hand to a donation, which a bigarray's never is. *)
      entry (Rig.Buffer.of_bigarray a) Npy.copied

let read_npys (archive : in_file) =
  let n = String.length npy_suffix in
  List.filter_map
    (fun e ->
      if String.ends_with ~suffix:npy_suffix e.name then
        let name = String.sub e.name 0 (String.length e.name - n) in
        Some (name, read_npy archive e)
      else None)
    archive.entries

let le16 value =
  String.init 2 (fun i -> Char.chr ((value lsr (8 * i)) land 0xff))

let le32_i64 value =
  String.init 4 (fun i ->
      Char.chr
        (Int64.to_int
           (Int64.logand (Int64.shift_right_logical value (8 * i)) 0xffL)))

let le32 value = le32_i64 (Int64.of_int value)

let le64 value =
  let value = Int64.of_int value in
  String.init 8 (fun i ->
      Char.chr
        (Int64.to_int
           (Int64.logand (Int64.shift_right_logical value (8 * i)) 0xffL)))

let add_offset archive amount =
  archive.offset <- checked_add "output" archive.offset amount

let write_string archive s =
  Save.string archive.fd s;
  add_offset archive (String.length s)

let open_out fd = { fd; offset = 0; entries_rev = [] }
let sample_size = 64 * 1024

(* The first [n] bytes of [header] followed by [data]. *)
let prefix header (data : bytes) n =
  let b = Bytes.create n in
  let from_header = Int.min n (String.length header) in
  Bytes.blit_string header 0 b 0 from_header;
  for i = from_header to n - 1 do
    Bytes.unsafe_set b i
      (Char.unsafe_chr (Array1.unsafe_get data (i - String.length header)))
  done;
  Bytes.unsafe_to_string b

(* How an entry's bytes are written: as they are, as the DEFLATE data already
   made of them, or deflated while they are written. *)
type writing = Stored | Deflated of string | Deflating

(* An entry is deflated when deflate shrinks its first 64 KiB. Where those are
   the whole entry, their DEFLATE data is the entry's, which spares an encoder,
   whose state takes about 1 MiB. *)
let writing header (data : bytes) =
  let size = String.length header + Array1.dim data in
  let sample = prefix header data (Int.min sample_size size) in
  let compressed = Compress_deflate.Deflate.compress sample in
  if String.length compressed >= String.length sample then Stored
  else if String.length sample = size then Deflated compressed
  else Deflating

(* Writes [header] and [data] to the archive as [w] says. Returns their CRC-32
   and their size as written. *)
let write_entry archive w header (data : bytes) =
  let size = Array1.dim data in
  let crc = Crc32.bigbytes ~crc:(Crc32.string header) data in
  match w with
  | Stored ->
      Save.string archive.fd header;
      Save.bytes archive.fd data;
      (crc, String.length header + size)
  | Deflated deflated ->
      Save.string archive.fd deflated;
      (crc, String.length deflated)
  | Deflating ->
      let e = Compress_deflate.Deflate.encoder () in
      let chunk = Bytes.create sample_size in
      (* Gives [e] the data from [first], at most a chunk, and the end of the
         input once the data is given. *)
      let give first =
        let n = Int.min sample_size (size - first) in
        for i = 0 to n - 1 do
          Bytes.unsafe_set chunk i
            (Char.unsafe_chr (Array1.unsafe_get data (first + i)))
        done;
        Encoder.src e chunk 0 n;
        first + n
      in
      let rec loop first written =
        match Encoder.encode e with
        | `Await -> loop (give first) written
        | `Data (b, off, len) ->
            ignore (Unix.write archive.fd b off len);
            loop first (written + len)
        | `End -> (crc, written)
      in
      let header = Bytes.of_string header in
      Encoder.src e header 0 (Bytes.length header);
      loop 0 0

let valid_name name = safe_name name && valid_utf8 name

(* Every entry has ZIP64 sizes and offset and the archive a ZIP64 end record, as
   numpy's np.savez writes them, so that no reader takes a 32-bit field at its
   marker value for the value itself. *)
let zip64_version = 45
let marker16 = 0xffff
let marker32 = 0xffffffff

let add_npy archive name header (data : bytes) =
  let filename = name ^ npy_suffix in
  let w = writing header data in
  let method_ =
    match w with Stored -> stored | Deflated _ | Deflating -> deflated
  in
  let uncompressed_size =
    checked_add "NPY entry" (String.length header) (Array1.dim data)
  in
  let local_offset = archive.offset in
  let flags = 0x0808 in
  (* The sizes follow the data, in its descriptor: the local header's ZIP64
     field holds zeros in their place. *)
  write_string archive
    (le32_i64 0x04034b50L ^ le16 zip64_version ^ le16 flags ^ le16 method_
   ^ le16 0 ^ le16 0 ^ le32 0 ^ le32 marker32 ^ le32 marker32
    ^ le16 (String.length filename)
    ^ le16 20 ^ filename ^ le16 0x0001 ^ le16 16 ^ le64 0 ^ le64 0);
  let crc32, compressed_size = write_entry archive w header data in
  add_offset archive compressed_size;
  write_string archive
    (le32_i64 0x08074b50L ^ le32 crc32 ^ le64 compressed_size
   ^ le64 uncompressed_size);
  let entry =
    {
      name = filename;
      method_;
      flags;
      crc32;
      compressed_size;
      uncompressed_size;
      local_offset;
    }
  in
  archive.entries_rev <- entry :: archive.entries_rev

let central_record entry =
  let extra =
    le64 entry.uncompressed_size
    ^ le64 entry.compressed_size ^ le64 entry.local_offset
  in
  le32_i64 0x02014b50L ^ le16 0x031e ^ le16 zip64_version ^ le16 entry.flags
  ^ le16 entry.method_ ^ le16 0 ^ le16 0 ^ le32 entry.crc32 ^ le32 marker32
  ^ le32 marker32
  ^ le16 (String.length entry.name)
  ^ le16 (4 + String.length extra)
  ^ le16 0 ^ le16 0 ^ le16 0 ^ le32 0 ^ le32 marker32 ^ entry.name ^ le16 0x0001
  ^ le16 (String.length extra)
  ^ extra

let close_out archive =
  let central_offset = archive.offset in
  List.iter
    (fun entry -> write_string archive (central_record entry))
    (List.rev archive.entries_rev);
  let central_size = archive.offset - central_offset in
  let count = List.length archive.entries_rev in
  let zip64_end = archive.offset in
  write_string archive
    (le32_i64 0x06064b50L ^ le64 44 ^ le16 0x031e ^ le16 zip64_version ^ le32 0
   ^ le32 0 ^ le64 count ^ le64 count ^ le64 central_size ^ le64 central_offset
    );
  write_string archive (le32_i64 0x07064b50L ^ le32 0 ^ le64 zip64_end ^ le32 1);
  (* The classic end record holds each value that fits its field, and the marker
     otherwise, as Python's zipfile writes it. *)
  let count = Int.min count marker16 in
  write_string archive
    (le32_i64 0x06054b50L ^ le16 0 ^ le16 0 ^ le16 count ^ le16 count
    ^ le32 (Int.min central_size marker32)
    ^ le32 (Int.min central_offset marker32)
    ^ le16 0)
