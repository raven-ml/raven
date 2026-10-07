(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* An HDU is a header and the bytes of its data unit, in a buffer: the disk
   buffer of a file, the host buffer of bytes held elsewhere, or the encoding of
   a constructed HDU. *)

open Err
module B = Nx_device.Buffer

let strf = Printf.sprintf
let block = Header.block_size

(* The data unit's bytes: [size] of them from byte [offset] of [buffer],
   padding excluded. *)
type store = { buffer : B.t; offset : int; size : int }

type t = {
  name : string;  (** the file's path, [of_bytes]'s name, or [""] *)
  index : int;  (** position in the file; [0] for a constructed HDU *)
  digest : string Lazy.t;  (** of the file's headers *)
  header : Header.t Lazy.t;
  data : store Lazy.t;
  header_bytes : (int * int) option;  (** a read HDU's header, in [buffer] *)
  slabs : slabs option;
      (** a constructed HDU whose data [write] streams without encoding it whole
      *)
}

(* [slabs.write emit] calls [emit] with host buffers of the data unit's bytes,
   in order. *)
and slabs = { write : (B.t -> unit) -> unit }

let header h = Lazy.force h.header
let name h = h.name
let digest h = Lazy.force h.digest
let place h = Header.place (header h)

(* Host bytes *)

let host_bytes n = B.create Nx_device.host Nx_dtype.Scalar.UInt8 n
let bigbytes b = B.bigarray Bigarray.int8_unsigned b

(* [copy_in src ~offset dst ~at n] copies [n] bytes from byte [offset] of
   [src] to byte [at] of the host buffer [dst]. *)
let copy_in src ~offset dst ~at n =
  if n > 0 then
    B.copy
      ~src:(B.view src ~offset Nx_dtype.Scalar.UInt8 n)
      ~dst:(B.view dst ~offset:at Nx_dtype.Scalar.UInt8 n)

let read_string src ~offset n =
  let b = host_bytes n in
  copy_in src ~offset b ~at:0 n;
  let a = bigbytes b in
  String.init n (fun i -> Char.unsafe_chr (Bigarray.Array1.get a i))

(* Places *)

let hdu_place name index header =
  let ext =
    match Header.find_struct Value.string "EXTNAME" header with
    | Ok (Some n) -> strf " (%s)" n
    | _ -> ""
  in
  Err.sub (Err.file name) (strf "HDU %d%s" index ext)

(* Sizes *)

type kind =
  | Primary of { groups : bool }
  | Extension of string  (** XTENSION, trailing spaces dropped *)

let get_struct h v k =
  match Header.find_struct v k h with
  | Ok (Some x) -> x
  | Ok None -> fail_at (Header.place h) "%s is absent" k
  | Error e -> fail "%s" e

let find_struct h v k = ok_or_fail (Header.find_struct v k h)

let card_fail h k what =
  match Header.cards h k with
  | i :: _ -> fail_at (Header.card_place h i k) "%s" what
  | [] -> fail_at (Header.place h) "%s" what

let bitpix h =
  let b = get_struct h Value.int "BITPIX" in
  match b with
  | 8 | 16 | 32 | 64 | -32 | -64 -> b
  | _ -> card_fail h "BITPIX" (strf "%d is not 8, 16, 32, 64, -32 or -64" b)

(* NAXIS1 ... NAXISn, in file order. *)
let naxes h =
  let n = get_struct h Value.int "NAXIS" in
  if n < 0 || n > 999 then card_fail h "NAXIS" (strf "%d is outside 0-999" n);
  Array.init n (fun i ->
      let k = strf "NAXIS%d" (i + 1) in
      let v = get_struct h Value.int k in
      if v < 0 then card_fail h k (strf "%d is negative" v);
      v)

let kind_of first h =
  if first then begin
    if not (get_struct h Value.bool "SIMPLE") then
      card_fail h "SIMPLE" "the file does not conform to FITS (SIMPLE = F)";
    let groups = find_struct h Value.bool "GROUPS" = Some true in
    Primary { groups }
  end
  else Extension (get_struct h Value.string "XTENSION")

let product what h l =
  Array.fold_left
    (fun acc n ->
      match Err.mul acc n with
      | Some p -> p
      | None -> fail_at (Header.place h) "the %s's size overflows" what)
    1 l

let checked h = function
  | Some n -> n
  | None -> fail_at (Header.place h) "the data unit's size overflows"

(* The data unit's size in bytes (FITS 4.0 Eq. 1, 2 and 4). *)
let data_size kind h =
  let bytes = abs (bitpix h) / 8 in
  let axes = naxes h in
  let pcount () =
    let p = get_struct h Value.int "PCOUNT" in
    if p < 0 then card_fail h "PCOUNT" (strf "%d is negative" p);
    p
  in
  let gcount () =
    let g = get_struct h Value.int "GCOUNT" in
    if g < 0 then card_fail h "GCOUNT" (strf "%d is negative" g);
    g
  in
  let elements = if axes = [||] then 0 else product "data unit" h axes in
  match kind with
  | Primary { groups = true } when Array.length axes > 0 && axes.(0) = 0 ->
      let group =
        product "group" h (Array.sub axes 1 (Array.length axes - 1))
      in
      let g = gcount () and p = pcount () in
      checked h
        (Option.bind (Err.add p group) (fun n ->
             Option.bind (Err.mul g n) (Err.mul bytes)))
  | Primary _ -> checked h (Err.mul bytes elements)
  | Extension _ ->
      let g = gcount () and p = pcount () in
      checked h
        (Option.bind (Err.add p elements) (fun n ->
             Option.bind (Err.mul g n) (Err.mul bytes)))

let padded n = (n + block - 1) / block * block

(* Walking a file *)

let gzip_magic = "\x1f\x8b"

(* The header starting at byte [pos], and the offset after its last block.
   Blocks are copied while they lie inside the file. *)
let read_header src len ~name ~index pos =
  let rec go acc pos =
    if pos + block > len then
      fail_at
        (Err.sub
           (Err.sub (Err.file name) (strf "HDU %d" index))
           (strf "bytes %d-%d" pos (pos + block - 1)))
        "the header ends past the end of the file, at byte %d" len
    else
      let s = read_string src ~offset:pos block in
      let rec records acc k =
        if k = block / Header.record_size then `More acc
        else
          let r = String.sub s (k * Header.record_size) Header.record_size in
          if Header.is_end r then `End acc else records (r :: acc) (k + 1)
      in
      match records acc 0 with
      | `More acc -> go acc (pos + block)
      | `End acc -> (Array.of_list (List.rev acc), pos + block)
  in
  go [] pos

let walk ~name src =
  let len = B.length src in
  if len >= 2 && read_string src ~offset:0 2 = gzip_magic then
    fail
      "%s: a gzip stream; decompress it with Nx_io.gunzip and read the result"
      name;
  if len < 6 || read_string src ~offset:0 6 <> "SIMPLE" then
    fail "%s: not a FITS file: it does not start with SIMPLE" name;
  let rec go acc index pos =
    let records, data_start = read_header src len ~name ~index pos in
    let h0 = Header.of_records records in
    let h = Header.of_records ~place:(hdu_place name index h0) records in
    let kind = kind_of (index = 0) h in
    let size = data_size kind h in
    if data_start + size > len then
      fail_at
        (Err.sub (Header.place h)
           (strf "bytes %d-%d" data_start (data_start + size - 1)))
        "the data unit ends past the end of the file, at byte %d" len;
    let hdu =
      (h, { buffer = src; offset = data_start; size }, (pos, data_start - pos))
    in
    let next = data_start + padded size in
    let acc = hdu :: acc in
    if next + 8 <= len && read_string src ~offset:next 8 = "XTENSION" then
      go acc (index + 1) next
    else List.rev acc
  in
  let parts = go [] 0 0 in
  let digest =
    lazy
      (let ctx = List.map (fun (h, _, _) -> Header.to_string h) parts in
       "blake2b-256:"
       ^ Digest.BLAKE256.to_hex (Digest.BLAKE256.string (String.concat "" ctx)))
  in
  List.mapi
    (fun index (header, store, hb) ->
      {
        name;
        index;
        digest;
        header = Lazy.from_val header;
        data = Lazy.from_val store;
        header_bytes = Some hb;
        slabs = None;
      })
    parts

let read path =
  match B.of_file path with
  | Error e -> Error e
  | Ok src -> catch (fun () -> walk ~name:path src)

(* The C-order bytes of [t] in a buffer the host reads: its own storage on the
   host or the disk, a host copy otherwise. *)
let buffer_of (t : (int, Nx.uint8_elt) Nx.t) =
  let b = Nx.to_buffer t in
  let d = B.device b in
  if Nx_device.equal d Nx_device.host || Nx_device.equal d Nx_device.disk then b
  else begin
    let h = host_bytes (B.nbytes b) in
    B.copy ~src:b ~dst:h;
    h
  end

let of_bytes ~name t = catch (fun () -> walk ~name (buffer_of t))

(* Data *)

let store h = Lazy.force h.data

let data h =
  let s = store h in
  Nx.of_buffer Nx.uint8 [| s.size |]
    (B.view s.buffer ~offset:s.offset Nx_dtype.Scalar.UInt8 s.size)

(* Lookup *)

let extname h =
  match Header.find_struct Value.string "EXTNAME" (header h) with
  | Ok (Some n) -> Some n
  | _ -> None

let extver h =
  match Header.find_struct Value.int "EXTVER" (header h) with
  | Ok (Some v) -> v
  | _ -> 1

let get ?ver name hdus =
  let named = List.filter (fun h -> extname h = Some name) hdus in
  let names () =
    List.mapi
      (fun i h ->
        match extname h with
        | Some n -> strf "%d %s (EXTVER %d)" i n (extver h)
        | None -> strf "%d unnamed" i)
      hdus
    |> String.concat ", "
  in
  let file =
    match hdus with h :: _ when h.name <> "" -> h.name ^ ": " | _ -> ""
  in
  match (ver, named) with
  | None, [ h ] -> Ok h
  | None, [] ->
      Error (strf "%sno HDU is named %s; the HDUs are %s" file name (names ()))
  | None, _ ->
      Error
        (strf "%s%d HDUs are named %s; give ~ver; the HDUs are %s" file
           (List.length named) name (names ()))
  | Some v, _ -> (
      match List.filter (fun h -> extver h = v) named with
      | [ h ] -> Ok h
      | [] ->
          Error
            (strf "%sno HDU is named %s with EXTVER %d; the HDUs are %s" file
               name v (names ()))
      | l ->
          Error
            (strf "%s%d HDUs are named %s with EXTVER %d; the HDUs are %s" file
               (List.length l) name v (names ())))

(* Constructed HDUs *)

let header_digest header =
  lazy
    ("blake2b-256:"
    ^ Digest.BLAKE256.to_hex
        (Digest.BLAKE256.string (Header.to_string (Lazy.force header))))

(* [constructed_lazy encoded] is the HDU whose header and data [encoded]
   computes when first asked, as a tiled image's whose PCOUNT is its
   compressed size. *)
let constructed_lazy ?slabs encoded =
  let header =
    lazy
      (let h, _ = Lazy.force encoded in
       Header.with_place (hdu_place "" 0 h) h)
  in
  {
    name = "";
    index = 0;
    digest = header_digest header;
    header;
    data = lazy (snd (Lazy.force encoded));
    header_bytes = None;
    slabs;
  }

let constructed ?slabs header data =
  constructed_lazy ?slabs (lazy (header, Lazy.force data))

let host_store b = { buffer = b; offset = 0; size = B.nbytes b }
