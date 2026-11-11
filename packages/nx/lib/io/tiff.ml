(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  TIFF 6.0 (Adobe, 1992), with BigTIFF's 64-bit offsets (libtiff's "BigTIFF
  Design"), Deflate (Adobe Photoshop TIFF Technical Notes, 2002) and the
  floating point predictor (Adobe Photoshop TIFF Technical Note 3, 2005).

  A load reads the tags of every IFD, from a file or a partial file, and
  names the ranges a partial file lacks. [window] plans one IFD's chunks,
  checks each against the file before it allocates, names the ranges of
  the chunks it covers that a partial file lacks, then decodes each with
  lz, undoes its predictor and byte order or unpacks its samples in
  nx_io_tiff.c, and copies its rows in the window into place. [samples] is
  the window of the whole raster, or a value over the file's pages.

  A save writes a placeholder of its header, IFD and tag values, its chunks
  after them, then the directory again with the chunks' offsets and counts:
  the directory's length depends on its entries' types and counts alone.
  ---------------------------------------------------------------------------*)

open Bigarray
module A = Nx_array
module B = Rig.Buffer
module D = Nx_array.Dtype
module L = Nx_array.Layout

type value =
  | Byte of int array
  | Ascii of string
  | Short of int array
  | Long of int array
  | Rational of (int * int) array
  | Sbyte of int array
  | Undefined of string
  | Sshort of int array
  | Slong of int array
  | Srational of (int * int) array
  | Float of float array
  | Double of float array
  | Long8 of int64 array
  | Slong8 of int64 array

type compression = Uncompressed | Deflate

(* Where an IFD's bytes are: a file, or a partial file. *)
type source = File of B.t | Part of Partial.t

let length = function File b -> B.length b | Part p -> Partial.length p

(* An IFD: its tags in file order, its file's byte order, and the file or
   partial file, named [name], from which [window] and [samples] read. *)
type t = {
  name : string;
  source : source;
  big : bool;
  tags : (int * value) list;
}

let tags t = t.tags

(* Tags (TIFF 6.0, sections 8, 14, 15 and 19) *)

let image_width = 256
let image_length = 257
let bits_per_sample = 258
let compression_tag = 259
let photometric = 262
let fill_order = 266
let strip_offsets = 273
let samples_per_pixel = 277
let rows_per_strip = 278
let strip_byte_counts = 279
let planar_configuration = 284
let predictor_tag = 317
let tile_width = 322
let tile_length = 323
let tile_offsets = 324
let tile_byte_counts = 325
let extra_samples = 338
let sample_format = 339

let names =
  [
    (image_width, "ImageWidth");
    (image_length, "ImageLength");
    (bits_per_sample, "BitsPerSample");
    (compression_tag, "Compression");
    (photometric, "Photometric");
    (fill_order, "FillOrder");
    (strip_offsets, "StripOffsets");
    (samples_per_pixel, "SamplesPerPixel");
    (rows_per_strip, "RowsPerStrip");
    (strip_byte_counts, "StripByteCounts");
    (planar_configuration, "PlanarConfiguration");
    (predictor_tag, "Predictor");
    (tile_width, "TileWidth");
    (tile_length, "TileLength");
    (tile_offsets, "TileOffsets");
    (tile_byte_counts, "TileByteCounts");
    (extra_samples, "ExtraSamples");
    (sample_format, "SampleFormat");
  ]

(* A tag as messages name it: ["Compression (259)"], or ["tag 33550"]. *)
let field tag =
  match List.assoc_opt tag names with
  | Some n -> Printf.sprintf "%s (%d)" n tag
  | None -> Printf.sprintf "tag %d" tag

(* Field types (TIFF 6.0, section 2; BigTIFF Design) *)

let type_name = function
  | Byte _ -> "BYTE"
  | Ascii _ -> "ASCII"
  | Short _ -> "SHORT"
  | Long _ -> "LONG"
  | Rational _ -> "RATIONAL"
  | Sbyte _ -> "SBYTE"
  | Undefined _ -> "UNDEFINED"
  | Sshort _ -> "SSHORT"
  | Slong _ -> "SLONG"
  | Srational _ -> "SRATIONAL"
  | Float _ -> "FLOAT"
  | Double _ -> "DOUBLE"
  | Long8 _ -> "LONG8"
  | Slong8 _ -> "SLONG8"

(* The bytes of one value of the field type [code], or 0 for a type this
   reader skips, as TIFF 6.0 tells readers to: "Readers should skip over
   fields containing an unexpected field type." *)
let type_size = function
  | 1 | 2 | 6 | 7 -> 1
  | 3 | 8 -> 2
  | 4 | 9 | 11 -> 4
  | 5 | 10 | 12 | 16 | 17 -> 8
  | _ -> 0

(* Reading *)

(* A file's byte order, and whether it is BigTIFF, whose entry counts,
   values' counts and offsets take 8 bytes where classic TIFF's take 2, 4
   and 4. *)
type form = { big : bool; bigtiff : bool }

let u16 f s i =
  if f.big then String.get_uint16_be s i else String.get_uint16_le s i

let i16 f s i =
  if f.big then String.get_int16_be s i else String.get_int16_le s i

let i32 f s i =
  Int32.to_int
    (if f.big then String.get_int32_be s i else String.get_int32_le s i)

let u32 f s i = i32 f s i land 0xffff_ffff

let i64 f s i =
  if f.big then String.get_int64_be s i else String.get_int64_le s i

(* A LONG8 that addresses a byte: an offset or a count. *)
let address what v =
  if Int64.compare v 0L < 0 || Int64.compare v (Int64.of_int max_int) > 0 then
    Fail.failf "%s %Lu is past any file" what v;
  Int64.to_int v

let word f s i =
  if f.bigtiff then address "an offset" (i64 f s i) else u32 f s i

let word_size f = if f.bigtiff then 8 else 4
let entry_size f = if f.bigtiff then 20 else 12

(* The [n] values of field type [code] in [s]. *)
let decode f code n s =
  let ints get = Array.init n get in
  match code with
  | 1 -> Byte (ints (fun k -> Char.code s.[k]))
  | 2 ->
      let len = String.length s in
      if len > 0 && s.[len - 1] = '\000' then Ascii (String.sub s 0 (len - 1))
      else Ascii s
  | 3 -> Short (ints (fun k -> u16 f s (2 * k)))
  | 4 -> Long (ints (fun k -> u32 f s (4 * k)))
  | 5 -> Rational (ints (fun k -> (u32 f s (8 * k), u32 f s ((8 * k) + 4))))
  | 6 -> Sbyte (ints (fun k -> String.get_int8 s k))
  | 7 -> Undefined s
  | 8 -> Sshort (ints (fun k -> i16 f s (2 * k)))
  | 9 -> Slong (ints (fun k -> i32 f s (4 * k)))
  | 10 -> Srational (ints (fun k -> (i32 f s (8 * k), i32 f s ((8 * k) + 4))))
  | 11 ->
      Float
        (ints (fun k -> Int32.float_of_bits (Int32.of_int (u32 f s (4 * k)))))
  | 12 -> Double (ints (fun k -> Int64.float_of_bits (i64 f s (8 * k))))
  | 16 -> Long8 (ints (fun k -> i64 f s (8 * k)))
  | _ -> Slong8 (ints (fun k -> i64 f s (8 * k)))

(* [take length taken n] counts [n] more bytes of IFDs and values in
   [taken], and refuses the file once they pass its [length]: IFDs and
   values may not share bytes past it, so what a load reads is linear in
   the file's length. *)
let take length taken n =
  taken := !taken + n;
  if !taken > length then
    Fail.failf "the IFDs and their values take more than the file's %d bytes"
      length

let ( let* ) = Option.bind
let ( let+ ) x f = Option.map f x

(* The [n] bytes of [source] from byte [pos], or [None] where a partial
   file lacks some of them, whose ranges [need] then gains. Raises
   [Failure] if they do not lie in the file. *)
let get source need ~pos n =
  match source with
  | File b -> Some (Storage.string b ~pos n)
  | Part p -> (
      Storage.within ~length:(Partial.length p) ~pos n;
      match Partial.lacking p ~pos n with
      | [] -> Some (Partial.string p ~pos n)
      | rs ->
          need := rs @ !need;
          None)

(* The IFD at [at]: its tags and the offset of the next IFD, 0 after the
   last, or [None] where a partial file lacks its entries. A value is in
   its entry's last field when it fits there, and at the offset that field
   holds otherwise; a value a partial file lacks is left out, and [need]
   gains its range. [taken] counts the bytes read. *)
let ifd source f taken need at =
  let length = length source in
  let wide = word_size f and entry = entry_size f in
  let* n, first =
    if f.bigtiff then
      let+ s = get source need ~pos:at 8 in
      (address "an entry count" (i64 f s 0), at + 8)
    else
      let+ s = get source need ~pos:at 2 in
      (u16 f s 0, at + 2)
  in
  if n > (length - first) / entry then
    Fail.failf "the IFD at byte %d states %d entries, past the file's end" at
      n;
  take length taken (first - at + (n * entry) + wide);
  let+ es = get source need ~pos:first ((n * entry) + wide) in
  let next = word f es (n * entry) in
  let seen = Hashtbl.create n in
  let rec entries k acc =
    if k = n then List.rev acc
    else
      let e = k * entry in
      let tag = u16 f es e and code = u16 f es (e + 2) in
      let count =
        if f.bigtiff then address "a count" (i64 f es (e + 4))
        else u32 f es (e + 4)
      in
      let size = type_size code in
      if size = 0 then entries (k + 1) acc
      else begin
        if Hashtbl.mem seen tag then
          Fail.failf "the IFD at byte %d holds tag %d twice" at tag;
        Hashtbl.add seen tag ();
        if count > length / size then
          Fail.failf "%s states %d values, past the file's end" (field tag)
            count;
        let bytes = count * size and value = e + 4 + wide in
        let s =
          if bytes <= wide then Some (String.sub es value bytes)
          else begin
            take length taken bytes;
            get source need ~pos:(word f es value) bytes
          end
        in
        match s with
        | Some s -> entries (k + 1) ((tag, decode f code count s) :: acc)
        | None -> entries (k + 1) acc
      end
  in
  (entries 0 [], next)

(* The header (TIFF 6.0, section 2; BigTIFF Design): the byte order, 42 or
   43, BigTIFF's offset size and reserved zeros ("If they're not, a reader
   should give up"), then the first IFD's offset. *)
let header source need =
  let* s = get source need ~pos:0 8 in
  let big =
    match String.sub s 0 2 with
    | "II" -> false
    | "MM" -> true
    | _ -> failwith "not a TIFF file: no byte order II or MM"
  in
  let f = { big; bigtiff = false } in
  match u16 f s 2 with
  | 42 -> Some (f, u32 f s 4)
  | 43 ->
      if u16 f s 4 <> 8 || u16 f s 6 <> 0 then
        failwith "a BigTIFF header whose offsets are not of 8 bytes";
      let f = { big; bigtiff = true } in
      let+ s = get source need ~pos:8 8 in
      (f, word f s 0)
  | v -> Fail.failf "TIFF version %d, neither 42 nor BigTIFF's 43" v

(* The IFDs of [source], or the ranges a partial file lacks of the bytes
   the chain's reading reads next: the header, an IFD's entries, and the
   values of every IFD it reaches. It follows the chain past an IFD whose
   values the partial file lacks, which only adds ranges: every check
   reads only bytes held, and no value's bytes can fail one. *)
let chain name source =
  let need = ref [] in
  let ifds =
    match header source need with
    | None -> []
    | Some (f, first) ->
        if first = 0 then failwith "the file has no IFD";
        let seen = Hashtbl.create 8 and taken = ref 0 in
        let rec walk at acc =
          if at = 0 then List.rev acc
          else begin
            if Hashtbl.mem seen at then
              Fail.failf "the IFDs' chain comes back to byte %d" at;
            Hashtbl.add seen at ();
            match ifd source f taken need at with
            | None -> List.rev acc
            | Some (tags, next) ->
                walk next ({ name; source; big = f.big; tags } :: acc)
          end
        in
        walk first []
  in
  match !need with [] -> Ok ifds | rs -> Error (Partial.ranges rs)

let read path file =
  match chain path (File file) with
  | Ok ds -> ds
  | Error _ -> assert false (* A file holds every byte. *)

let decode p =
  let name = Partial.name p in
  Fail.file name (fun () -> chain name (Part p))

(* Plans *)

let refuse tag v why = Fail.failf "%s %d: %s" (field tag) v why

(* The values of the unsigned integer field [tag], which TIFF 6.0 tells
   readers to accept as BYTE, SHORT or LONG, and BigTIFF as LONG8. *)
let ints t tag =
  match List.assoc_opt tag t.tags with
  | None -> None
  | Some (Byte a | Short a | Long a) -> Some a
  | Some (Long8 a) -> Some (Array.map (address (field tag)) a)
  | Some v -> Fail.failf "%s of type %s" (field tag) (type_name v)

let required t tag =
  match ints t tag with
  | Some a -> a
  | None -> Fail.failf "%s is missing" (field tag)

(* A field of one value, or [default] without it. *)
let one ?default t tag =
  match (ints t tag, default) with
  | Some [| v |], _ -> v
  | None, Some d -> d
  | None, None -> Fail.failf "%s is missing" (field tag)
  | Some a, _ ->
      Fail.failf "%s holds %d values, not 1" (field tag) (Array.length a)

(* A field of a value per sample, all equal, or [default] without it. One
   value stands for every sample, as some writers state it. *)
let per_sample t tag ~spp ~default =
  match ints t tag with
  | None -> default
  | Some [| v |] -> v
  | Some a when Array.length a = spp ->
      if Array.exists (( <> ) a.(0)) a then
        Fail.failf "%s differs between samples" (field tag);
      a.(0)
  | Some a ->
      Fail.failf "%s holds %d values for %d samples" (field tag)
        (Array.length a) spp

(* The dtype of samples of [bits] bits of SampleFormat [format]: for
   unsigned samples, the narrowest that holds them. *)
let dtype ~bits ~format =
  match (format, bits) with
  (* TIFF 6.0 stores a 24-bit sample as three bytes, most significant
     first, but libtiff swaps it as an integer in the file's byte order,
     and GDAL writes either: in each byte order a major writer departs from
     the specification. *)
  | 1, 24 ->
      refuse bits_per_sample 24
        "unsigned, whose bytes TIFF 6.0 orders most significant first and \
         libtiff in the file's byte order"
  | 1, b when b >= 1 && b <= 8 -> D.Any D.Uint8
  | 1, b when b >= 9 && b <= 16 -> D.Any D.Uint16
  | 1, b when b >= 17 && b <= 32 -> D.Any D.Uint32
  | 2, 16 -> D.Any D.Int16
  | 2, 32 -> D.Any D.Int32
  | 3, 32 -> D.Any D.Float32
  | 3, 64 -> D.Any D.Float64
  | (1 | 2 | 3), _ ->
      refuse bits_per_sample bits
        (Printf.sprintf
           "with SampleFormat %d; read are 1 to 32 bits unsigned but 24, 16 \
            and 32 signed, 32 and 64 float"
           format)
  | _ -> refuse sample_format format "read are 1, 2 and 3"

(* Samples of 8, 16, 32 or 64 bits lie in the file as SHORTs, LONGs and
   DOUBLEs do, in its byte order; samples of any other width are packed,
   most significant bit first, whatever its byte order: "Otherwise BYTE"
   (TIFF 6.0, section 8, Compression). *)
let packed bits = bits <> 8 && bits <> 16 && bits <> 32 && bits <> 64

(* The most bytes one byte of a stream decodes to: deflate's 258-byte
   match, from a length code and a distance code of a bit each (RFC 1951),
   and LZW's longest string, 3839 bytes after 3838 entries, from a 12-bit
   code. *)
let deflate_ratio = 1032
let lzw_ratio = 2560

(* How an IFD's samples lie in its chunks: [rows] by [cols] pixels each,
   [across] to a row of chunks, chunk [k] at [offsets.(k)] of
   [counts.(k)] bytes. A sample takes [bits] bits in a chunk and [bytes]
   bytes in the raster. *)
type plan = {
  h : int;
  w : int;
  spp : int;
  bits : int;
  bytes : int;
  dt : D.any;
  codec : int;
  predictor : int;
  rows : int;
  cols : int;
  across : int;
  offsets : int array;
  counts : int array;
  tiled : bool;
}

let ceil_div a b = (a + b - 1) / b

let check_values t ~spp ~bits ~format ~codec =
  let predictor = one ~default:1 t predictor_tag in
  (match predictor with
  | 1 -> ()
  | (2 | 3) when codec = 1 ->
      refuse predictor_tag predictor "with Compression 1"
  | 2 when packed bits ->
      refuse predictor_tag 2
        (Printf.sprintf
           "on samples of %d bits; read is 1 for samples not of 8, 16 or 32 \
            bits"
           bits)
  | 3 when format <> 3 -> refuse predictor_tag predictor "on integer samples"
  | 2 | 3 -> ()
  | p -> refuse predictor_tag p "read are 1, 2 and 3");
  let planar = one ~default:1 t planar_configuration in
  if planar <> 1 && not (planar = 2 && spp = 1) then
    refuse planar_configuration planar "read is 1, and 2 for one sample";
  let fill = one ~default:1 t fill_order in
  if fill <> 1 then refuse fill_order fill "read is 1";
  (match ints t photometric with
  | Some [| 6 |] ->
      refuse photometric 6 "(YCbCr), whose samples may be subsampled"
  | _ -> ());
  predictor

let plan t =
  let w = one t image_width and h = one t image_length in
  if w = 0 then refuse image_width 0 "no column";
  if h = 0 then refuse image_length 0 "no row";
  let spp = one ~default:1 t samples_per_pixel in
  if spp = 0 then refuse samples_per_pixel 0 "no sample";
  let bits = per_sample t bits_per_sample ~spp ~default:1 in
  let format = per_sample t sample_format ~spp ~default:1 in
  let (D.Any d as dt) = dtype ~bits ~format in
  let codec = one ~default:1 t compression_tag in
  (* 32946 is Deflate's obsolete code: "We recommend that TIFF
     implementations recognize and read the obsolete code" (Adobe's TIFF
     technical notes). *)
  if not (List.mem codec [ 1; 5; 8; 32946 ]) then
    refuse compression_tag codec
      "read are 1 (none), 5 (LZW), and 8 and 32946 (Deflate)";
  let predictor = check_values t ~spp ~bits ~format ~codec in
  Storage.check_shape [| h; w; spp |];
  let tiled = List.mem_assoc tile_offsets t.tags in
  if tiled && List.mem_assoc strip_offsets t.tags then
    failwith "both StripOffsets (273) and TileOffsets (324)";
  if not (tiled || List.mem_assoc strip_offsets t.tags) then
    failwith "neither StripOffsets (273) nor TileOffsets (324)";
  let offsets_tag, counts_tag =
    if tiled then (tile_offsets, tile_byte_counts)
    else (strip_offsets, strip_byte_counts)
  in
  let rows, cols =
    if tiled then begin
      let tw = one t tile_width and tl = one t tile_length in
      if tw = 0 then refuse tile_width 0 "no column";
      if tl = 0 then refuse tile_length 0 "no row";
      (tl, tw)
    end
    else
      let rps = one ~default:0xffff_ffff t rows_per_strip in
      if rps = 0 then refuse rows_per_strip 0 "no row";
      (Int.min rps h, w)
  in
  Storage.check_shape [| rows; cols; spp |];
  let offsets = required t offsets_tag and counts = required t counts_tag in
  let across = ceil_div w cols in
  let n = across * ceil_div h rows in
  if Array.length offsets <> n then
    Fail.failf "%s holds %d offsets for %d chunks" (field offsets_tag)
      (Array.length offsets) n;
  if Array.length counts <> n then
    Fail.failf "%s holds %d counts for %d chunks" (field counts_tag)
      (Array.length counts) n;
  let bytes = D.bits d / 8 in
  {
    h; w; spp; bits; bytes; dt; codec; predictor; rows; cols; across;
    offsets; counts; tiled;
  }

let unit p = if p.tiled then "tile" else "strip"
let counts_tag p = if p.tiled then tile_byte_counts else strip_byte_counts
(* The bytes of a chunk's row: its samples' bits, padded to a byte, "Each
   row is padded to the next BYTE/SHORT/LONG boundary" (TIFF 6.0, section
   8, Compression). *)
let chunk_row p = ceil_div (p.cols * p.spp * p.bits) 8

(* Chunk [k]'s first row and column, and its rows and columns that lie in
   the raster. *)
let place p k =
  let r = k / p.across * p.rows and c = k mod p.across * p.cols in
  (r, c, Int.min p.rows (p.h - r), Int.min p.cols (p.w - c))

(* The bytes of chunk [k]'s rows in the raster. *)
let needed p k =
  let _, _, rows, _ = place p k in
  rows * chunk_row p

(* The bytes chunk [k] decodes to: its rows in the raster, and every row of
   a compressed tile, since "all the tiles in an image are the same size
   when uncompressed" (TIFF 6.0, section 15). *)
let decoded p k =
  if p.tiled && p.codec <> 1 then p.rows * chunk_row p else needed p k

(* Every chunk lies in the file and holds its rows' bytes, or a stream
   that could decode to them, and the chunks' bytes total at most the
   file's [length], so that a raster takes at most the codec's ratio times
   the file's length. *)
let check_chunks p length =
  Array.iteri
    (fun k off ->
      let count = p.counts.(k) in
      if count = 0 then
        Fail.failf "%s: %s %d has no byte, as GDAL's sparse files leave it"
          (field (counts_tag p)) (unit p) k;
      if off > length - count then
        Fail.failf "%s %d at byte %d ends past the file's %d bytes" (unit p) k
          off length;
      let need = decoded p k in
      let ratio =
        match p.codec with 1 -> 1 | 5 -> lzw_ratio | _ -> deflate_ratio
      in
      if need / ratio > count || (p.codec = 1 && count < need) then
        Fail.failf "%s: %s %d holds %d bytes, too few for its %d"
          (field (counts_tag p)) (unit p) k count need)
    p.offsets;
  (* Each count is at most the file's length, so the sum stays below twice
     it. *)
  ignore
    (Array.fold_left
       (fun total count ->
         let total = total + count in
         if total > length then
           Fail.failf "%s: the %ss take more than the file's %d bytes"
             (field (counts_tag p)) (unit p) length;
         total)
       0 p.counts
      : int);

external unpredict :
  Storage.bytes ->
  int ->
  Storage.bytes ->
  int ->
  int ->
  int ->
  int ->
  int ->
  bool ->
  unit = "caml_nx_io_tiff_unpredict_byte" "caml_nx_io_tiff_unpredict"
[@@noalloc]

external predict :
  Storage.bytes -> Storage.bytes -> int -> int -> int -> int -> int -> unit
  = "caml_nx_io_tiff_predict_byte" "caml_nx_io_tiff_predict"
[@@noalloc]

(* [copy src ~at ~stride dst ~at ~stride rows span] copies [rows] rows of
   [span] bytes, each at its buffer's offset plus its row times the
   buffer's stride, with no allocation. Every row must lie in its
   buffer. *)
external copy :
  Storage.bytes ->
  int ->
  int ->
  Storage.bytes ->
  int ->
  int ->
  int ->
  int ->
  unit = "caml_nx_io_tiff_copy_byte" "caml_nx_io_tiff_copy"
[@@noalloc]

(* [unpack src ~at ~stride ~first dst ~at ~stride rows n bits bytes]
   reads [rows] rows of [n] samples of [bits] bits, packed most significant
   bit first, from sample [first] of each row of [src], into samples of
   [bytes] bytes in the host's byte order in [dst], with no allocation.
   Rows lie as [copy]'s do, and every row must lie in its buffer. *)
external unpack :
  Storage.bytes ->
  int ->
  int ->
  int ->
  Storage.bytes ->
  int ->
  int ->
  int ->
  int ->
  int ->
  int ->
  unit = "caml_nx_io_tiff_unpack_byte" "caml_nx_io_tiff_unpack"
[@@noalloc]

let create n = Array1.create int8_unsigned c_layout n

(* Chunk [k]'s stream [src] decoded into [buf]. *)
let inflate p k src buf =
  let decode = if p.codec = 5 then Lz.Lzw.decode else Lz.Zlib.decode in
  match decode src buf with
  | Ok { produced; _ } ->
      if produced < needed p k then
        Fail.failf "Compression (259) %d: %s %d decodes to %d bytes, not %d"
          p.codec (unit p) k produced (needed p k)
  | Error e ->
      Fail.failf "Compression (259) %d: %s %d at byte %d: %s" p.codec (unit p)
        k p.offsets.(k)
        (Format.asprintf "%a" Lz.pp_error e)

(* The samples lie in the file as one array in C order: uncompressed, in
   chunks as wide as the raster that follow each other. *)
let contiguous p =
  let stride = p.rows * chunk_row p in
  p.codec = 1 && p.cols = p.w && not (packed p.bits)
  && Array.for_all Fun.id
       (Array.mapi (fun k off -> off = p.offsets.(0) + (k * stride)) p.offsets)

(* [with_chunks source f] is [f bytes], [bytes at n] the bytes [at] to
   [at + n - 1] of [source], which a partial file must hold: the file's
   pages, which the host borrows for the extent of [f], or the partial
   file's own. *)
let with_chunks source f =
  match source with
  | File b ->
      Storage.read (Storage.pages b) (fun file ->
          f (fun at n -> Array1.sub file at n))
  | Part p -> f (fun at n -> Partial.bytes p ~pos:at n)

(* Along a side of [side] pixels a chunk, the [n] pixels from pixel [at]
   lie in [over side at n] chunks from chunk [first side at]. A window
   covers [across] chunks a row from chunk row [r0] and chunk column [c0],
   and [chunk p ~r0 ~c0 ~across i] is the [i]th, in the file's order. *)
let first side at = at / side

let over side at n =
  if n = 0 then 0 else ((at + n - 1) / side) - first side at + 1

let chunk p ~r0 ~c0 ~across i =
  ((r0 + (i / across)) * p.across) + c0 + (i mod across)

(* The window's samples, from the chunks it covers, all held. Each chunk is
   decoded whole; its rows in the window are swapped or unpredicted, row by
   row as the predictors work, and copied or unpacked into place. *)
let decode_window (t : t) p ~row ~col ~rows ~cols =
  let (D.Any dt) = p.dt in
  let packed = packed p.bits in
  let swapped = t.big <> Sys.big_endian && p.bytes > 1 && not packed in
  let px = p.spp * p.bytes and crow = chunk_row p in
  let orow = cols * px in
  let out = B.create Rig.host (rows * orow) in
  let r0 = first p.rows row and c0 = first p.cols col in
  let down = over p.rows row rows and across = over p.cols col cols in
  (* Chunk 0 has the most rows, so it decodes to the most bytes, which
     [check_chunks] bounds by the file's. *)
  let largest = if down * across = 0 then 0 else decoded p 0 in
  let buf = create largest and scratch = create crow in
  B.write out int8_unsigned (fun dst ->
      with_chunks t.source (fun bytes ->
          for i = 0 to (down * across) - 1 do
            let k = chunk p ~r0 ~c0 ~across i in
            let r, c, krows, kcols = place p k in
            let src = bytes p.offsets.(k) p.counts.(k) in
            (* The chunk's rows [y0 - r] to [y1 - r - 1] and columns
               [x0 - c] to [x1 - c - 1] lie in the window. *)
            let y0 = Int.max r row and x0 = Int.max c col in
            let y1 = Int.min (r + krows) (row + rows)
            and x1 = Int.min (c + kcols) (col + cols) in
            let first = (y0 - r) * crow and n = y1 - y0 in
            let span = (x1 - x0) * p.spp in
            (* An uncompressed chunk of packed samples or in the host's
               byte order is read in place; any other is decoded or copied
               into [buf] first. *)
            let data =
              if p.codec <> 1 then (
                inflate p k src buf;
                buf)
              else if swapped then (
                Array1.blit
                  (Array1.sub src first (n * crow))
                  (Array1.sub buf first (n * crow));
                buf)
              else src
            in
            let at = ((y0 - row) * orow) + ((x0 - col) * px) in
            if packed then
              unpack data first crow ((x0 - c) * p.spp) dst at orow n span
                p.bits p.bytes
            else begin
              if data == buf then
                unpredict buf first scratch n (p.cols * p.spp) p.spp p.bytes
                  p.predictor t.big;
              copy data (first + ((x0 - c) * px)) crow dst at orow n
                (span * p.bytes)
            end
          done));
  let layout = L.contiguous (Iarray.of_array [| rows; cols; p.spp |]) in
  Nx.P (Nx.Repr.unsafe_of_array Nx.Host.v (A.v dt layout out))

(* The ranges a partial file lacks of the chunks a window covers, none for
   a file, after the checks of every chunk. *)
let lacking (t : t) p ~row ~col ~rows ~cols =
  check_chunks p (length t.source);
  match t.source with
  | File _ -> []
  | Part q ->
      let r0 = first p.rows row and c0 = first p.cols col in
      let across = over p.cols col cols in
      let lacks i =
        let k = chunk p ~r0 ~c0 ~across i in
        Partial.lacking q ~pos:p.offsets.(k) p.counts.(k)
      in
      Partial.ranges
        (List.concat (List.init (over p.rows row rows * across) lacks))

let window (t : t) ~row ~col ~rows ~cols =
  Fail.file t.name @@ fun () ->
  let p = plan t in
  if row < 0 || rows < 0 || row > p.h - rows then
    Fail.invalid_argf
      "Nx_io.Tiff.window: %d rows from row %d lie outside the %d rows" rows
      row p.h;
  if col < 0 || cols < 0 || col > p.w - cols then
    Fail.invalid_argf
      "Nx_io.Tiff.window: %d columns from column %d lie outside the %d \
       columns"
      cols col p.w;
  match lacking t p ~row ~col ~rows ~cols with
  | [] -> Ok (decode_window t p ~row ~col ~rows ~cols)
  | rs -> Error rs

let samples (t : t) =
  Fail.file t.name @@ fun () ->
  let p = plan t in
  match t.source with
  | File file when contiguous p ->
      check_chunks p (B.length file);
      let (D.Any dt) = p.dt in
      let total = p.h * p.w * p.spp * p.bytes and first = p.offsets.(0) in
      let layout = L.contiguous (Iarray.of_array [| p.h; p.w; p.spp |]) in
      let x = Storage.value (B.view file ~first ~length:total) dt layout in
      let swapped = t.big <> Sys.big_endian && p.bytes > 1 in
      Nx.P (if swapped then Storage.byteswap x else x)
  | File _ | Part _ -> (
      match lacking t p ~row:0 ~col:0 ~rows:p.h ~cols:p.w with
      | [] -> decode_window t p ~row:0 ~col:0 ~rows:p.h ~cols:p.w
      | (at, n) :: _ ->
          Fail.failf "the partial file lacks bytes %d to %d" at (at + n - 1))

(* Saving *)

let by = "Nx_io.save_tiff"

(* The SampleFormat and BitsPerSample of [dt]'s samples. *)
let format_of (type v s) (dt : (v, s) D.t) =
  match dt with
  | D.Uint8 -> Some (1, 8)
  | D.Uint16 -> Some (1, 16)
  | D.Uint32 -> Some (1, 32)
  | D.Int16 -> Some (2, 16)
  | D.Int32 -> Some (2, 32)
  | D.Float32 -> Some (3, 32)
  | D.Float64 -> Some (3, 64)
  | _ -> None

let type_code = function
  | Byte _ -> 1
  | Ascii _ -> 2
  | Short _ -> 3
  | Long _ -> 4
  | Rational _ -> 5
  | Sbyte _ -> 6
  | Undefined _ -> 7
  | Sshort _ -> 8
  | Slong _ -> 9
  | Srational _ -> 10
  | Float _ -> 11
  | Double _ -> 12
  | Long8 _ -> 16
  | Slong8 _ -> 17

(* A field's Count: its values, an ASCII string's bytes and its NUL. *)
let count = function
  | Byte a | Short a | Long a | Sbyte a | Sshort a | Slong a -> Array.length a
  | Rational a | Srational a -> Array.length a
  | Float a | Double a -> Array.length a
  | Long8 a | Slong8 a -> Array.length a
  | Ascii s -> String.length s + 1
  | Undefined s -> String.length s

(* A value's bytes in a little-endian file. *)
let bytes_of v =
  let b = Buffer.create 16 in
  let i32 x = Buffer.add_int32_le b (Int32.of_int x) in
  let pair (n, d) =
    i32 n;
    i32 d
  in
  (match v with
  | Byte a -> Array.iter (Buffer.add_uint8 b) a
  | Ascii s ->
      Buffer.add_string b s;
      Buffer.add_char b '\000'
  | Short a -> Array.iter (Buffer.add_uint16_le b) a
  | Long a | Slong a -> Array.iter i32 a
  | Rational a | Srational a -> Array.iter pair a
  | Sbyte a -> Array.iter (Buffer.add_int8 b) a
  | Undefined s -> Buffer.add_string b s
  | Sshort a -> Array.iter (Buffer.add_int16_le b) a
  | Float a ->
      Array.iter (fun x -> Buffer.add_int32_le b (Int32.bits_of_float x)) a
  | Double a ->
      Array.iter (fun x -> Buffer.add_int64_le b (Int64.bits_of_float x)) a
  | Long8 a | Slong8 a -> Array.iter (Buffer.add_int64_le b) a);
  Buffer.contents b

(* [v], the value of [tag], holds a value or more, each in its type's
   range. *)
let check_value tag v =
  let bad m = Fail.invalid_argf "%s: tag %d: %s" by tag m in
  let range lo hi a =
    Array.iter
      (fun x ->
        if x < lo || x > hi then
          bad (Printf.sprintf "%d is outside %s" x (type_name v)))
      a
  in
  let pairs lo hi a =
    range lo hi (Array.map fst a);
    range lo hi (Array.map snd a)
  in
  if count v = 0 then bad "no value";
  match v with
  | Byte a -> range 0 0xff a
  | Short a -> range 0 0xffff a
  | Long a -> range 0 0xffff_ffff a
  | Sbyte a -> range (-0x80) 0x7f a
  | Sshort a -> range (-0x8000) 0x7fff a
  | Slong a -> range (-0x8000_0000) 0x7fff_ffff a
  | Rational a -> pairs 0 0xffff_ffff a
  | Srational a -> pairs (-0x8000_0000) 0x7fff_ffff a
  | Float a ->
      Array.iter
        (fun x ->
          let back = Int32.float_of_bits (Int32.bits_of_float x) in
          if back <> x && not (Float.is_nan x) then
            bad (Printf.sprintf "%h is not a FLOAT" x))
        a
  | Ascii _ | Undefined _ | Double _ | Long8 _ | Slong8 _ -> ()

(* The tags a save states itself. *)
let own =
  [
    image_width; image_length; bits_per_sample; compression_tag;
    strip_offsets; samples_per_pixel; rows_per_strip; strip_byte_counts;
    planar_configuration; predictor_tag; tile_width; tile_length;
    tile_offsets; tile_byte_counts; sample_format;
  ]

let check_tags tags =
  let seen = Hashtbl.create 16 in
  List.iter
    (fun (tag, v) ->
      if tag < 0 || tag > 0xffff then
        Fail.invalid_argf "%s: tag %d is not a SHORT" by tag;
      if List.mem tag own then
        Fail.invalid_argf "%s: %s is the save's own" by (field tag);
      if Hashtbl.mem seen tag then
        Fail.invalid_argf "%s: tag %d is given twice" by tag;
      Hashtbl.add seen tag ();
      check_value tag v)
    tags

let align8 n = (n + 7) land lnot 7

(* The header, the IFD of [entries], sorted by tag, and the values that do
   not fit their entries, each at a multiple of 8 bytes, as BigTIFF asks
   ("All values must begin at an 8-byte-aligned address") and as classic
   TIFF's word boundary allows. *)
let directory ~bigtiff entries =
  let f = { big = false; bigtiff } in
  let wide = word_size f and n = List.length entries in
  let header = if bigtiff then 16 else 8 in
  let base =
    align8 (header + (if bigtiff then 8 else 2) + (n * entry_size f) + wide)
  in
  let b = Buffer.create base and values = Buffer.create 1024 in
  let word n =
    if bigtiff then Buffer.add_int64_le b (Int64.of_int n)
    else Buffer.add_int32_le b (Int32.of_int n)
  in
  let zeros n = String.make n '\000' in
  Buffer.add_string b "II";
  if bigtiff then List.iter (Buffer.add_uint16_le b) [ 43; 8; 0 ]
  else Buffer.add_uint16_le b 42;
  word header;
  if bigtiff then word n else Buffer.add_uint16_le b n;
  List.iter
    (fun (tag, v) ->
      Buffer.add_uint16_le b tag;
      Buffer.add_uint16_le b (type_code v);
      word (count v);
      let s = bytes_of v in
      let len = String.length s in
      if len <= wide then Buffer.add_string b (s ^ zeros (wide - len))
      else begin
        word (base + Buffer.length values);
        Buffer.add_string values (s ^ zeros (align8 len - len))
      end)
    entries;
  word 0;
  Buffer.add_string b (zeros (base - Buffer.length b));
  Buffer.add_buffer b values;
  Buffer.contents b

(* TIFF 6.0's strips: "Set RowsPerStrip such that the size of each strip
   is about 8K bytes", or one row when a row is larger. *)
let strip_bytes = 8192

(* The largest offset and count of a classic file, a LONG. *)
let classic_max = 0xffff_ffff

(* The raster's sides and samples, checked. *)
let raster_shape x =
  let h, w, c =
    match Nx.shape x with
    | [| h; w; c |] -> (h, w, c)
    | s ->
        Fail.invalid_argf "%s: %d axes, a TIFF raster has 3" by
          (Array.length s)
  in
  if h = 0 || w = 0 || c = 0 then
    Fail.invalid_argf "%s: the raster has no sample" by;
  if h > classic_max || w > classic_max then
    Fail.invalid_argf "%s: %d by %d pixels, a LONG holds at most %d a side" by
      h w classic_max;
  if c > 0xffff then
    Fail.invalid_argf "%s: %d samples a pixel, a SHORT holds at most 65535" by
      c;
  (h, w, c)

let save ?tile ?(compression = Uncompressed) ?(tags = []) path x =
  let dt = Nx.dtype x in
  let format, bits =
    match format_of dt with
    | Some f -> f
    | None -> Fail.invalid_argf "%s: TIFF has no %s" by (D.name dt)
  in
  let h, w, c = raster_shape x in
  (match tile with
  | Some (r, k) when r <= 0 || k <= 0 || r mod 16 <> 0 || k mod 16 <> 0 ->
      Fail.invalid_argf "%s: tiles of %d by %d, whose sides are not multiples \
                         of 16"
        by r k
  | _ -> ());
  check_tags tags;
  let bytes = bits / 8 in
  let row = w * c * bytes in
  let tiled = Option.is_some tile in
  let rows, cols =
    match tile with
    | Some t -> t
    | None -> (Int.max 1 (Int.min h (strip_bytes / row)), w)
  in
  let across = ceil_div w cols in
  let n = across * ceil_div h rows in
  let crow = cols * c * bytes in
  (* A tile is whole, padded; the last strip ends with the raster. *)
  let chunk_bytes k =
    if tiled then rows * crow else Int.min rows (h - (k / across * rows)) * crow
  in
  let codec, predictor =
    match compression with
    | Uncompressed -> (1, 1)
    | Deflate -> (8, if format = 3 then 3 else 2)
  in
  let given tag = List.mem_assoc tag tags in
  let defaults =
    if given photometric then []
    else if c > 1 && not (given extra_samples) then
      [
        (photometric, Short [| 1 |]);
        (extra_samples, Short (Array.make (c - 1) 0));
      ]
    else [ (photometric, Short [| 1 |]) ]
  in
  let entries ~bigtiff offsets counts =
    let long a = if bigtiff then Long8 (Array.map Int64.of_int a) else Long a in
    let chunks =
      if tiled then
        [
          (tile_width, Long [| cols |]);
          (tile_length, Long [| rows |]);
          (tile_offsets, long offsets);
          (tile_byte_counts, long counts);
        ]
      else
        [
          (strip_offsets, long offsets);
          (rows_per_strip, Long [| rows |]);
          (strip_byte_counts, long counts);
        ]
    in
    let predicted =
      if predictor = 1 then [] else [ (predictor_tag, Short [| predictor |]) ]
    in
    List.stable_sort
      (fun (a, _) (b, _) -> Int.compare a b)
      ([
         (image_width, Long [| w |]);
         (image_length, Long [| h |]);
         (bits_per_sample, Short (Array.make c bits));
         (compression_tag, Short [| codec |]);
         (samples_per_pixel, Short [| c |]);
         (planar_configuration, Short [| 1 |]);
         (sample_format, Short (Array.make c format));
       ]
      @ chunks @ defaults @ predicted @ tags)
  in
  let zeros = Array.make n 0 in
  (* Classic when no tag is of BigTIFF's types and the largest file the save
     could write fits classic offsets, each chunk at most its bound. *)
  let largest =
    let dir = directory ~bigtiff:false (entries ~bigtiff:false zeros zeros) in
    let sum = ref (String.length dir) in
    for k = 0 to n - 1 do
      let b = chunk_bytes k in
      sum := !sum + if compression = Deflate then Lz.Zlib.bound b else b
    done;
    !sum
  in
  let wide = function _, (Long8 _ | Slong8 _) -> true | _ -> false in
  let bigtiff = largest > classic_max || List.exists wide tags in
  Storage.with_bytes ~by x @@ fun data ->
  Save.write path @@ fun fd ->
  let first = directory ~bigtiff (entries ~bigtiff zeros zeros) in
  Save.string fd first;
  let offsets = Array.make n 0 and counts = Array.make n 0 in
  let buf = create (rows * crow) and scratch = create crow in
  let out =
    create (if compression = Deflate then Lz.Zlib.bound (rows * crow) else 0)
  in
  let pos = ref (String.length first) in
  for k = 0 to n - 1 do
    let r = k / across * rows and c0 = k mod across * cols in
    let rk = Int.min rows (h - r) and ck = Int.min cols (w - c0) in
    let len = chunk_bytes k in
    (* An uncompressed strip is the raster's own bytes; any other chunk is
       gathered into [buf], padded with zeros, and predicted. *)
    let chunk =
      if (not tiled) && compression = Uncompressed then
        Array1.sub data (r * row) len
      else begin
        if rk < rows || ck < cols then Array1.fill buf 0;
        copy data ((r * row) + (c0 * c * bytes)) row buf 0 crow rk
          (ck * c * bytes);
        let chunk = Array1.sub buf 0 len in
        predict chunk scratch (len / crow) (cols * c) c bytes predictor;
        chunk
      end
    in
    let chunk =
      match compression with
      | Uncompressed -> chunk
      | Deflate -> (
          (* [bound]'s destination always fits (lz.mli). *)
          match Lz.Zlib.encode chunk out with
          | Some m -> Array1.sub out 0 m
          | None -> assert false)
    in
    Save.bytes fd chunk;
    offsets.(k) <- !pos;
    counts.(k) <- Array1.dim chunk;
    pos := !pos + Array1.dim chunk
  done;
  ignore (Unix.lseek fd 0 Unix.SEEK_SET : int);
  Save.string fd (directory ~bigtiff (entries ~bigtiff offsets counts))
