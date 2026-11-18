(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  WAV: RIFF's form WAVE (Multimedia Programming Interface and Data
  Specifications 1.0, IBM and Microsoft, 1991), its fmt chunk plain or as
  WAVEFORMATEXTENSIBLE (Microsoft's mmreg.h), and RF64's 64-bit sizes (EBU Tech
  3306).

  A load walks the chunks to [data], reading [fmt ] on the way and nothing after
  [data]. Samples whose container is 1, 2, 4 or 8 bytes and whose valid bits
  fill it lie in the file as one array; any other is widened and checked in
  nx_io_wav.c.
  ---------------------------------------------------------------------------*)

open Bigarray
module A = Nx_array
module B = Rig.Buffer
module D = Nx_array.Dtype
module L = Nx_array.Layout

type coding = Pcm | Ieee_float | Alaw | Mulaw

(* A WAV file: its [data] chunk's rows of [channels] containers of [container]
   bytes, from byte [data], of [coding]; [bits] valid bits a sample, and [dt]
   the dtype they load at. *)
type t = {
  file : B.t;
  rate : int;
  channels : int;
  bits : int;
  channel_mask : int;
  coding : coding;
  container : int;
  data : int;
  length : int;
  dt : D.any;
}

let rate t = t.rate
let length t = t.length
let channels t = t.channels
let bits t = t.bits
let channel_mask t = t.channel_mask

(* Formats (mmreg.h) *)

let format_pcm = 0x0001
let format_ieee_float = 0x0003
let format_alaw = 0x0006
let format_mulaw = 0x0007
let format_extensible = 0xFFFE

(* A KSDATAFORMAT_SUBTYPE GUID is its format code, little-endian, then these 14
   bytes. *)
let subtype_tail = "\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"

(* The speakers of a file of [c] channels that states none: front centre for
   one, front left and right for two, and none named for more, which the
   specification leaves undefined. *)
let default_mask c = match c with 1 -> 0x4 | 2 -> 0x3 | _ -> 0

(* G.711's values in bits: 13 for A-law, 14 for mu-law. *)
let alaw_bits = 13
let mulaw_bits = 14

(* The dtype a sample of [container] bytes of [coding] loads at: the narrowest
   that holds the container, unsigned for 8-bit PCM, which the specification
   stores offset by 128. *)
let dtype coding container =
  match (coding, container) with
  | Pcm, 1 -> D.Any D.Uint8
  | Pcm, 2 | (Alaw | Mulaw), _ -> D.Any D.Int16
  | Pcm, (3 | 4) -> D.Any D.Int32
  | Pcm, _ -> D.Any D.Int64
  | Ieee_float, 4 -> D.Any D.Float32
  | Ieee_float, _ -> D.Any D.Float64

(* Reading *)

let u16 s i = String.get_uint16_le s i
let u32 s i = Int32.to_int (String.get_int32_le s i) land 0xffff_ffff

let u64 s i =
  let v = String.get_int64_le s i in
  if Int64.compare v 0L < 0 || Int64.compare v (Int64.of_int max_int) > 0 then
    Fail.failf "a 64-bit size of %Lu bytes is past any file" v;
  Int64.to_int v

(* The 32-bit size of a chunk whose writer could not seek back to state it. No
   RIFF file states it: a form's size counts 4 bytes of its own, the smallest
   fmt chunk's 8 + 14 and a data chunk's 8 besides the data, so a data size past
   [2^32 - 35] makes the form's pass [2^32 - 1]. In RF64 it says that the ds64
   chunk states the size. *)
let unstated = 0xffff_ffff

(* The coding of a fmt chunk [s]: the format, its channels, rate, container
   bytes, valid bits and channel mask. *)
let fmt s =
  let len = String.length s in
  if len < 16 then Fail.failf "the fmt chunk has %d bytes, fewer than 16" len;
  let tag = u16 s 0 and channels = u16 s 2 and rate = u32 s 4 in
  let align = u16 s 12 and stored = u16 s 14 in
  if channels = 0 then failwith "the fmt chunk states no channel";
  if align mod channels <> 0 || align = 0 || align / channels > 8 then
    Fail.failf "the fmt chunk states blocks of %d bytes for %d channels" align
      channels;
  let container = align / channels in
  let code, valid, mask =
    if tag <> format_extensible then (tag, stored, default_mask channels)
    else begin
      if len < 40 || u16 s 16 < 22 then
        failwith "the fmt chunk is WAVEFORMATEXTENSIBLE in fewer than 40 bytes";
      if stored <> 8 * container then
        Fail.failf "the fmt chunk states %d bits in containers of %d bytes"
          stored container;
      if String.sub s 26 14 <> subtype_tail then
        failwith "the fmt chunk's SubFormat is not a format of mmreg.h";
      (u16 s 24, u16 s 18, u32 s 20)
    end
  in
  let refuse () =
    Fail.failf "the fmt chunk states format %d, %d bits in %d bytes" code valid
      container
  in
  let coding, bits =
    if code = format_pcm then begin
      if valid = 0 then failwith "the fmt chunk states 0 valid bits";
      (* A plain PCM container is the bits rounded up to bytes. *)
      let fits =
        if tag = format_extensible then valid <= 8 * container
        else (valid + 7) / 8 = container
      in
      if not fits then refuse ();
      (Pcm, valid)
    end
    else if code = format_ieee_float then begin
      if (container <> 4 && container <> 8) || valid <> 8 * container then
        refuse ();
      (Ieee_float, valid)
    end
    else if code = format_alaw || code = format_mulaw then begin
      if container <> 1 || valid <> 8 then refuse ();
      if code = format_alaw then (Alaw, alaw_bits) else (Mulaw, mulaw_bits)
    end
    else refuse ()
  in
  (coding, channels, rate, container, bits, mask)

(* RF64's ds64 chunk [s]: the data's size and the sizes of other chunks by their
   identifiers (EBU Tech 3306, 3.4). *)
let ds64 s =
  let len = String.length s in
  if len < 28 then Fail.failf "the ds64 chunk has %d bytes, fewer than 28" len;
  let entries = u32 s 24 in
  if entries > (len - 28) / 12 then
    failwith "the ds64 chunk's table ends past the chunk";
  let table =
    List.init entries (fun i ->
        let at = 28 + (12 * i) in
        (String.sub s at 4, u64 s (at + 4)))
  in
  (u64 s 8, table)

let read file =
  let length = B.length file in
  let head = Storage.string file ~pos:0 12 in
  let form = String.sub head 0 4 in
  if (form <> "RIFF" && form <> "RF64") || String.sub head 8 4 <> "WAVE" then
    failwith "the file is no RIFF or RF64 form WAVE";
  let chunk pos =
    let h = Storage.string file ~pos 8 in
    (String.sub h 0 4, u32 h 4)
  in
  let sizes =
    if form = "RIFF" then None
    else
      match chunk 12 with
      | "ds64", n -> Some (ds64 (Storage.string file ~pos:20 n))
      | _ -> failwith "the RF64 form's first chunk is no ds64 chunk"
  in
  (* A chunk's size: in RF64, an unstated size is the ds64 chunk's. *)
  let size id n =
    match sizes with
    | Some (data, table) when n = unstated -> (
        if id = "data" then data
        else
          match List.assoc_opt id table with
          | Some n -> n
          | None -> Fail.failf "the ds64 chunk states no size for %S" id)
    | _ -> n
  in
  let rec walk pos fmt =
    if pos > length - 8 then failwith "the file has no data chunk";
    let id, n = chunk pos in
    let n = size id n in
    let body = pos + 8 in
    match (id, fmt) with
    | "data", None -> failwith "the data chunk comes before the fmt chunk"
    | "data", Some f ->
        let n = if n = unstated && sizes = None then length - body else n in
        if n > length - body then
          Fail.failf
            "the data chunk at byte %d states %d bytes, and the file holds %d \
             after it"
            pos n (length - body);
        (f, body, n)
    | _ ->
        if n > length - body then
          Fail.failf "the %S chunk at byte %d ends past the file" id pos;
        let fmt =
          match (id, fmt) with
          | "fmt ", None -> Some (Storage.string file ~pos:body n)
          | "fmt ", Some _ -> failwith "the file has two fmt chunks"
          | _ -> fmt
        in
        (* A chunk of an odd size is followed by a zero byte. *)
        walk (body + n + (n land 1)) fmt
  in
  let f, data, n = walk 12 None in
  let coding, channels, rate, container, bits, channel_mask = fmt f in
  let align = channels * container in
  if n mod align <> 0 then
    Fail.failf "the data chunk's %d bytes hold part of a row of %d bytes" n
      align;
  {
    file;
    rate;
    channels;
    bits;
    channel_mask;
    coding;
    container;
    data;
    length = n / align;
    dt = dtype coding container;
  }

(* Samples *)

external widen :
  Storage.bytes -> int -> Storage.bytes -> int -> int -> int -> int -> int
  = "caml_nx_io_wav_widen_byte" "caml_nx_io_wav_widen"

external narrow :
  Storage.bytes -> int -> Storage.bytes -> int -> int -> int -> int -> int
  = "caml_nx_io_wav_narrow_byte" "caml_nx_io_wav_narrow"

external g711 : Storage.bytes -> int -> Storage.bytes -> int -> bool -> unit
  = "caml_nx_io_wav_g711"

(* The rows lie in the file as one array: containers of a power of two bytes up
   to 8 whose valid bits fill them, or floats. *)
let direct t =
  match t.coding with
  | Ieee_float -> true
  | Alaw | Mulaw -> false
  | Pcm -> t.bits = 8 * t.container && t.container land (t.container - 1) = 0

let layout t rows = L.contiguous (Iarray.of_array [| rows; t.channels |])

(* Rows [row] to [row + rows - 1] in memory of their own, from their bytes
   alone; no row reads nothing. *)
let window t ~row ~rows =
  let (D.Any dt) = t.dt in
  let align = t.channels * t.container in
  let first = t.data + (row * align) in
  if rows = 0 then Nx.P (Nx.zeros dt [| 0; t.channels |])
  else if direct t then
    let bytes = B.view t.file ~first ~length:(rows * align) in
    let x = Storage.copied bytes dt (layout t rows) in
    Nx.P (if Sys.big_endian then Storage.byteswap x else x)
  else
    let w = D.bytes dt 1 and n = rows * t.channels in
    let out = B.create Rig.host (n * w) in
    let bad =
      B.write out int8_unsigned @@ fun dst ->
      Storage.read (Storage.pages t.file ~first ~length:(rows * align))
      @@ fun src ->
      match t.coding with
      | Alaw | Mulaw ->
          g711 src 0 dst n (t.coding = Alaw);
          -1
      | Pcm | Ieee_float -> widen src 0 dst n t.container w t.bits
    in
    if bad >= 0 then
      Fail.failf "row %d: a sample has a set bit below its %d valid bits"
        (row + (bad / t.channels))
        t.bits;
    Nx.P (Nx.Repr.unsafe_of_array Nx.Host.v (A.v dt (layout t rows) out))

(* A value over the file's pages where the rows lie in it as one array, in the
   host's byte order at a multiple of a sample's bytes. *)
let samples t =
  if not (direct t) then window t ~row:0 ~rows:t.length
  else
    let (D.Any dt) = t.dt in
    let bytes = t.length * t.channels * t.container in
    let x =
      Storage.little
        (B.view t.file ~first:t.data ~length:bytes)
        dt (layout t t.length)
    in
    Nx.P x

(* Saving *)

let by = "Nx_io.save_wav"

(* The coding of a dtype, and the fewest container bytes that load back at
   it. *)
let coding_of (type v s) (dt : (v, s) D.t) =
  match dt with
  | D.Uint8 -> (Pcm, 1)
  | D.Int16 -> (Pcm, 2)
  | D.Int32 -> (Pcm, 3)
  | D.Int64 -> (Pcm, 5)
  | D.Float32 -> (Ieee_float, 4)
  | D.Float64 -> (Ieee_float, 8)
  | D.Int8 ->
      Fail.invalid_argf
        "%s: WAV stores 8-bit samples unsigned, offset by 128, as uint8" by
  | dt -> Fail.invalid_argf "%s: WAV stores no %s" by (D.name dt)

let le n v = String.init n (fun i -> Char.chr ((v lsr (8 * i)) land 0xff))
let chunk id body = id ^ le 4 (String.length body) ^ body

(* [junk at k] is a JUNK chunk that puts the samples of a data chunk whose
   header would start at byte [at] at a multiple of their [k] bytes, so that a
   load maps them, or none where they lie there already or would not map. A
   chunk takes 8 bytes or more, and offsets are even. *)
let junk at k =
  let gap = (k - ((at + 8) mod k)) mod k in
  if (k <> 4 && k <> 8) || gap = 0 then ""
  else chunk "JUNK" (String.make ((((gap - 8) mod k) + k) mod k) '\000')

(* The rows a save converts at a time, about 1 MiB of a row's elements. *)
let rows_at_once align = Int.max 1 ((1 lsl 20) / align)

let save ?bits ?channel_mask path ~rate x =
  let dt = Nx.dtype x in
  let n, c =
    match Nx.shape x with
    | [| n; c |] -> (n, c)
    | s ->
        Fail.invalid_argf "%s: %d axes, a WAV file's samples have 2" by
          (Array.length s)
  in
  let coding, least = coding_of dt in
  if c = 0 then Fail.invalid_argf "%s: the samples have no channel" by;
  let w = D.bytes dt 1 in
  let bits =
    match (bits, coding) with
    | None, _ -> 8 * w
    | Some _, Ieee_float ->
        Fail.invalid_argf "%s: bits is given for %s samples" by (D.name dt)
    | Some b, _ ->
        if b < 1 || b > 8 * w then
          Fail.invalid_argf "%s: %d bits in %s elements" by b (D.name dt);
        b
  in
  let k = Int.max least ((bits + 7) / 8) in
  let align = k * c in
  if align > 0xffff then
    Fail.invalid_argf "%s: a row of %d bytes, past WAV's 65535" by align;
  if rate < 1 || rate > 0xffff_ffff || rate * align > 0xffff_ffff then
    Fail.invalid_argf "%s: a rate of %d rows a second of %d bytes" by rate align;
  let default = default_mask c in
  let mask = Option.value channel_mask ~default in
  if mask < 0 || mask > 0xffff_ffff then
    Fail.invalid_argf "%s: the channel mask %d is not 32 bits" by mask;
  let code = match coding with Pcm -> format_pcm | _ -> format_ieee_float in
  let extensible =
    c > 2 || (coding = Pcm && bits > 16) || bits <> 8 * k || mask <> default
  in
  let common = le 2 c ^ le 4 rate ^ le 4 (rate * align) ^ le 2 align in
  let fmt =
    chunk "fmt "
      (if extensible then
         le 2 format_extensible ^ common
         ^ le 2 (8 * k)
         ^ le 2 22 ^ le 2 bits ^ le 4 mask ^ le 2 code ^ subtype_tail
       else if coding = Pcm then le 2 code ^ common ^ le 2 (8 * k)
       else le 2 code ^ common ^ le 2 (8 * k) ^ le 2 0)
  in
  let data = n * align in
  let pad = data land 1 in
  let fact rf64 =
    if coding = Ieee_float then
      chunk "fact" (le 4 (if rf64 then unstated else n))
    else ""
  in
  (* The chunks between WAVE and data, after the ds64 chunk [ds64]. *)
  let body rf64 ds64 =
    let fact = fact rf64 in
    let at = 12 + String.length ds64 + String.length fmt + String.length fact in
    ds64 ^ fmt ^ fact ^ junk at k
  in
  let plain = body false "" in
  let riff = 4 + String.length plain + 8 + data + pad in
  (* A size field that would reach [unstated] makes the file RF64, whose ds64
     chunk states the sizes (EBU Tech 3306, 3.4). *)
  let header =
    if riff < unstated then
      "RIFF" ^ le 4 riff ^ "WAVE" ^ plain ^ "data" ^ le 4 data
    else
      let ds64_bytes = 36 in
      let size =
        4
        + String.length (body true (String.make ds64_bytes ' '))
        + 8 + data + pad
      in
      let ds64 = chunk "ds64" (le 8 size ^ le 8 data ^ le 8 n ^ le 4 0) in
      "RF64" ^ le 4 unstated ^ "WAVE" ^ body true ds64 ^ "data" ^ le 4 unstated
  in
  Storage.with_bytes ~by x @@ fun src ->
  Save.write path @@ fun fd ->
  Save.string fd header;
  let step = rows_at_once align in
  let buf = Array1.create int8_unsigned c_layout (step * align) in
  let rec rows r =
    if r < n then begin
      let m = Int.min step (n - r) in
      let bad = narrow src (r * c) buf (m * c) w k bits in
      if bad >= 0 then
        Fail.invalid_argf "%s: row %d: a sample has a set bit below its %d bits"
          by
          (r + (bad / c))
          bits;
      Save.bytes fd (Array1.sub buf 0 (m * align));
      rows (r + m)
    end
  in
  rows 0;
  if pad = 1 then Save.string fd "\000"
