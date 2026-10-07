(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tile-compressed images (FITS 4.0 §10): a BINTABLE with ZIMAGE = T whose
   rows are tiles, each coded in the heap. *)

open Err
module A = Bigarray.Array1

let strf = Printf.sprintf

type ints = Quantize.ints
type floats = Quantize.floats

type codec =
  | Rice of { block : int; bytepix : int }
  | Gzip_1
  | Gzip_2
  | Nocompress

let codec_name = function
  | Rice _ -> "RICE_1"
  | Gzip_1 -> "GZIP_1"
  | Gzip_2 -> "GZIP_2"
  | Nocompress -> "NOCOMPRESS"

(* A per-tile parameter: a column of the table, a keyword, or absent. *)
type 'a param = Column of Bintable.column | Keyword of 'a | Absent

type t = {
  table : Bintable.t;
  zbitpix : int;
  axes : int array;  (** ZNAXISn, in file order *)
  tile : int array;  (** ZTILEn, in file order *)
  counts : int array;  (** tiles along each axis *)
  codec : codec;
  quantized : bool;
  dither : Quantize.dither;
  dither_name : string;
  zdither0 : int;
  zscale : float param;
  zzero : float param;
  zblank : int param;
  compressed : Bintable.column;
  gzip : Bintable.column option;
  uncompressed : Bintable.column option;
  null_mask : Bintable.column option;
}

let tile_place t n = Err.sub t.table.place (strf "tile %d" (n + 1))

(* Description *)

let describe h (store : Hdu.store) =
  let place = Header.place h in
  let table = Bintable.layout h store in
  let get v k = Hdu.get_struct h v k in
  let find v k = Hdu.find_struct h v k in
  let zbitpix = get Value.int "ZBITPIX" in
  (match zbitpix with
  | 8 | 16 | 32 | 64 | -32 | -64 -> ()
  | b -> Hdu.card_fail h "ZBITPIX" (strf "%d is not a BITPIX" b));
  let rank = get Value.int "ZNAXIS" in
  if rank < 1 || rank > 999 then
    Hdu.card_fail h "ZNAXIS" (strf "%d is outside 1-999" rank);
  let axes =
    Array.init rank (fun i ->
        let k = strf "ZNAXIS%d" (i + 1) in
        let v = get Value.int k in
        if v < 0 then Hdu.card_fail h k (strf "%d is negative" v);
        v)
  in
  let tile =
    Array.init rank (fun i ->
        let k = strf "ZTILE%d" (i + 1) in
        let v =
          match find Value.int k with
          | Some v -> v
          | None -> if i = 0 then axes.(0) else 1
        in
        if v < 1 && axes.(i) > 0 then
          Hdu.card_fail h k (strf "%d is not positive" v);
        Int.max 1 v)
  in
  let counts = Array.mapi (fun i n -> (n + tile.(i) - 1) / tile.(i)) axes in
  let ntiles =
    Array.fold_left
      (fun acc c ->
        match Err.mul acc c with
        | Some p -> p
        | None -> fail_at place "the tile count overflows")
      1 counts
  in
  if ntiles <> table.rows then
    fail_at place "NAXIS2 is %d, the tiles number %d" table.rows ntiles;
  let pixels =
    Array.fold_left
      (fun acc n ->
        match Err.mul acc n with
        | Some p -> p
        | None -> fail_at place "the image's size overflows")
      1 axes
  in
  (* A gzip tile stores at least one byte per pixel and deflate expands at
     most 1032-fold. *)
  if pixels > 0 && pixels / 1032 > table.heap_size then
    fail_at place "%d pixels in a heap of %d bytes, past 1032 pixels per byte"
      pixels table.heap_size;
  let names = ref [] in
  for i = 1 to 99 do
    match find Value.string (strf "ZNAME%d" i) with
    | Some n -> names := (n, i) :: !names
    | None -> ()
  done;
  let zval name default =
    match List.assoc_opt name !names with
    | None -> default
    | Some i -> (
        match find Value.int (strf "ZVAL%d" i) with
        | Some v -> v
        | None -> Hdu.card_fail h (strf "ZVAL%d" i) "is absent")
  in
  let cmp = get Value.string "ZCMPTYPE" in
  let codec =
    match cmp with
    | "RICE_1" | "RICE_ONE" ->
        let block = zval "BLOCKSIZE" 32 in
        let bytepix =
          if
            List.mem_assoc "NOISEBIT" !names
            && not (List.mem_assoc "BYTEPIX" !names)
          then 4
          else zval "BYTEPIX" 4
        in
        if block <> 16 && block <> 32 then
          Hdu.card_fail h "ZCMPTYPE" (strf "BLOCKSIZE %d is not 16 or 32" block);
        (match bytepix with
        | 1 | 2 | 4 -> ()
        | 8 ->
            Hdu.card_fail h "ZCMPTYPE" "Rice with BYTEPIX 8 has no definition"
        | b -> Hdu.card_fail h "ZCMPTYPE" (strf "BYTEPIX %d is not 1, 2 or 4" b));
        Rice { block; bytepix }
    | "GZIP_1" -> Gzip_1
    | "GZIP_2" -> Gzip_2
    | "NOCOMPRESS" -> Nocompress
    | ("HCOMPRESS_1" | "PLIO_1") as c ->
        Hdu.card_fail h "ZCMPTYPE"
          (strf "%s is not read; decompress the file with funpack" c)
    | c -> Hdu.card_fail h "ZCMPTYPE" (strf "%S is no tile compression" c)
  in
  let column name = Bintable.find table name in
  let compressed =
    match column "COMPRESSED_DATA" with
    | Some c when c.form.heap <> Row -> c
    | _ -> fail_at place "no COMPRESSED_DATA heap column"
  in
  let heap_column name =
    match column name with
    | Some c when c.form.heap <> Row -> Some c
    | Some c ->
        fail_at (Bintable.column_place table c) "%s is not a heap column" name
    | None -> None
  in
  let param v name =
    match column name with
    | Some c ->
        (match (c.form.heap, c.form.elt, c.form.repeat) with
        | Row, (D | E | J | K | I | B), 1 -> ()
        | _ -> fail_at (Bintable.column_place table c) "%s holds no scalar" name);
        Column c
    | None -> ( match find v name with Some x -> Keyword x | None -> Absent)
  in
  let zscale = param Value.float "ZSCALE"
  and zzero = param Value.float "ZZERO" in
  let check_finite k = function
    | Keyword x when not (Float.is_finite x) ->
        Hdu.card_fail h k "is not finite"
    | _ -> ()
  in
  check_finite "ZSCALE" zscale;
  check_finite "ZZERO" zzero;
  let quant = find Value.string "ZQUANTIZ" in
  let quantized = zbitpix < 0 && zscale <> Absent && quant <> Some "NONE" in
  let dither, dither_name =
    match quant with
    | None | Some "NO_DITHER" | Some "NONE" ->
        (Quantize.No_dither, Option.value ~default:"NO_DITHER" quant)
    | Some "SUBTRACTIVE_DITHER_1" -> (Subtractive_1, "SUBTRACTIVE_DITHER_1")
    | Some "SUBTRACTIVE_DITHER_2" -> (Subtractive_2, "SUBTRACTIVE_DITHER_2")
    | Some q -> Hdu.card_fail h "ZQUANTIZ" (strf "%S is no quantization" q)
  in
  let zdither0 =
    if (not quantized) || dither = No_dither then 0
    else
      match find Value.int "ZDITHER0" with
      | Some d when d >= 1 && d <= 10000 -> d
      | Some d -> Hdu.card_fail h "ZDITHER0" (strf "%d is outside 1-10000" d)
      | None -> fail_at place "ZDITHER0 is absent from a dithered image"
  in
  let zblank =
    match column "ZBLANK" with
    | Some c -> Column c
    | None -> (
        match find Value.int "ZBLANK" with
        | Some b -> Keyword b
        | None -> Absent)
  in
  {
    table;
    zbitpix;
    axes;
    tile;
    counts;
    codec;
    quantized;
    dither;
    dither_name;
    zdither0;
    zscale;
    zzero;
    zblank;
    compressed;
    gzip = heap_column "GZIP_COMPRESSED_DATA";
    uncompressed = heap_column "UNCOMPRESSED_DATA";
    null_mask = heap_column "NULL_PIXEL_MASK";
  }

(* Decoding one tile *)

(* A decoded tile: integers that fit [int], int64s, or floats. *)
type values =
  | Ints of ints
  | Longs of (int64, Bigarray.int64_elt, Bigarray.c_layout) A.t
  | Reals of floats

let ints n = A.create Bigarray.int Bigarray.c_layout n
let floats n = A.create Bigarray.float64 Bigarray.c_layout n

(* [gunzip t n src] is the data of the gzip stream [src], at most [8 n]
   bytes. *)
let gunzip place n (src : Checksum.bigbytes) =
  let cap = 8 * n in
  let out = Bytes.create cap in
  let len = ref 0 in
  let d = Compress_deflate.Gzip.decoder () in
  let input = Bytes.create (A.dim src) in
  for i = 0 to A.dim src - 1 do
    Bytes.unsafe_set input i (Char.unsafe_chr (A.unsafe_get src i))
  done;
  Compress_deflate.Decoder.src d input 0 (Bytes.length input);
  let fed = ref false in
  let rec loop () =
    match Compress_deflate.Decoder.decode d with
    | `Await ->
        if !fed then fail_at place "the gzip stream ends early"
        else begin
          fed := true;
          Compress_deflate.Decoder.src d input 0 0;
          loop ()
        end
    | `Data (b, first, l) ->
        if !len + l > cap then
          fail_at place "the gzip stream holds more than 8 bytes per pixel";
        Bytes.blit b first out !len l;
        len := !len + l;
        loop ()
    | `End -> ()
    | `Error e -> fail_at place "%s" e
  in
  loop ();
  (out, !len)

(* [of_bytes t place ~quantized_ints bytes len n] reads [n] big-endian values
   whose width the length decides (FITS 4.0 §10.4.2). *)
let of_bytes t place ~as_ints get len n =
  if n = 0 then Ints (ints 0)
  else
    let w = len / n in
    if w * n <> len || not (List.mem w [ 1; 2; 4; 8 ]) then
      fail_at place "%d bytes decode to %d pixels, not 1, 2, 4 or 8 bytes each"
        len n;
    let be i =
      let v = ref 0 in
      for j = 0 to w - 1 do
        v := (!v lsl 8) lor get ((i * w) + j)
      done;
      !v
    in
    if t.zbitpix > 0 || as_ints then begin
      if t.zbitpix > 0 && 8 * w > t.zbitpix then
        fail_at place "%d-byte values in a BITPIX %d image" w t.zbitpix;
      if t.zbitpix < 0 && w <> 4 then
        fail_at place "%d-byte values in a quantized image" w;
      if w = 8 then begin
        let a = A.create Bigarray.int64 Bigarray.c_layout n in
        for i = 0 to n - 1 do
          let v = ref 0L in
          for j = 0 to 7 do
            v :=
              Int64.logor (Int64.shift_left !v 8)
                (Int64.of_int (get ((i * 8) + j)))
          done;
          A.unsafe_set a i !v
        done;
        Longs a
      end
      else
        let a = ints n in
        for i = 0 to n - 1 do
          A.unsafe_set a i (Rice.signed w (be i))
        done;
        Ints a
    end
    else
      let a = floats n in
      (match w with
      | 4 ->
          for i = 0 to n - 1 do
            A.unsafe_set a i (Int32.float_of_bits (Int32.of_int (be i)))
          done
      | 8 ->
          for i = 0 to n - 1 do
            let v = ref 0L in
            for j = 0 to 7 do
              v :=
                Int64.logor (Int64.shift_left !v 8)
                  (Int64.of_int (get ((i * 8) + j)))
            done;
            A.unsafe_set a i (Int64.float_of_bits !v)
          done
      | _ -> fail_at place "%d-byte values in a float image" w);
      Reals a

let param_value (type a) rows (p : a param) (of_float : float -> a)
    (of_int : int -> a) : a option =
  match p with
  | Absent -> None
  | Keyword x -> Some x
  | Column c -> (
      match c.form.elt with
      | E | D -> Some (of_float (Bintable.float_at rows c.start c.form.elt))
      | J -> Some (of_int (Bintable.be_int32 rows c.start))
      | I -> Some (of_int (Rice.signed 2 (Bintable.be_int rows c.start 2)))
      | B -> Some (of_int (Bintable.be_int rows c.start 1))
      | K -> Some (of_int (Int64.to_int (Bintable.be_int64 rows c.start)))
      | _ -> None)

(* [decode_tile t n npix budget] decodes tile [n] of [npix] pixels. [budget]
   counts the heap bytes a read may still decode. *)
let decode_tile t n npix budget =
  let place = tile_place t n in
  let rows = Bintable.read_rows t.table n (n + 1) in
  let fetch c =
    let count, offset = Bintable.descriptor t.table c rows 0 in
    let bytes = count * Bintable.size c.Bintable.form.elt in
    if bytes > !budget then
      fail_at place "the tiles decode more bytes than the heap holds";
    budget := !budget - bytes;
    ( count,
      if count = 0 then None else Some (Bintable.heap t.table offset bytes) )
  in
  (match t.null_mask with
  | Some c when fst (Bintable.descriptor t.table c rows 0) > 0 ->
      fail_at
        (Bintable.column_place t.table c)
        "a tile has a null pixel mask, which is not read"
  | _ -> ());
  let scale = param_value rows t.zscale Fun.id Float.of_int in
  let zero = param_value rows t.zzero Fun.id Float.of_int in
  let blank = param_value rows t.zblank int_of_float Fun.id in
  let dequantize i =
    let scale = Option.value ~default:1. scale
    and zero = Option.value ~default:0. zero in
    if not (Float.is_finite scale && Float.is_finite zero) then
      fail_at place "ZSCALE or ZZERO is not finite";
    let out = floats npix in
    Quantize.dequantize ~dither:t.dither ~row:(n + t.zdither0) ~scale ~zero
      ~blank i npix out;
    Reals out
  in
  let raw c = match fetch c with _, None -> None | _, Some b -> Some b in
  match raw t.compressed with
  | Some b -> (
      let get i = A.unsafe_get b i in
      match t.codec with
      | Rice { block; bytepix } ->
          if t.zbitpix < 0 && not t.quantized then
            fail_at place
              "Rice codes integers; the float image is not quantized";
          let out = ints npix in
          (try Rice.decode ~width:bytepix ~block b 0 (A.dim b) out npix
           with Fail e -> fail_at place "%s" e);
          if t.quantized then dequantize out else Ints out
      | Gzip_1 | Gzip_2 | Nocompress -> (
          let data, len =
            if t.codec = Nocompress then
              (Bytes.init (A.dim b) (fun i -> Char.unsafe_chr (get i)), A.dim b)
            else gunzip place npix b
          in
          let w = if npix = 0 then 1 else len / npix in
          let get =
            if t.codec = Gzip_2 && w > 1 && w * npix = len then
              (* byte j of pixel k is at j * npix + k *)
              fun i ->
              Char.code (Bytes.unsafe_get data ((i mod w * npix) + (i / w)))
            else fun i -> Char.code (Bytes.unsafe_get data i)
          in
          match of_bytes t place ~as_ints:t.quantized get len npix with
          | Ints i when t.quantized -> dequantize i
          | v -> v))
  | None -> (
      let lossless c f =
        match raw c with Some b -> Some (f b) | None -> None
      in
      let from_gzip =
        Option.bind t.gzip (fun c ->
            lossless c (fun b ->
                let data, len = gunzip place npix b in
                of_bytes t place ~as_ints:false
                  (fun i -> Char.code (Bytes.unsafe_get data i))
                  len npix))
      in
      match from_gzip with
      | Some v -> v
      | None -> (
          let from_raw =
            Option.bind t.uncompressed (fun c ->
                lossless c (fun b ->
                    let w = Bintable.size c.form.elt in
                    if A.dim b <> w * npix then
                      fail_at place
                        "UNCOMPRESSED_DATA holds %d bytes for %d pixels"
                        (A.dim b) npix;
                    of_bytes t place ~as_ints:false
                      (fun i -> A.unsafe_get b i)
                      (A.dim b) npix))
          in
          match from_raw with
          | Some v -> v
          | None ->
              if npix = 0 then Ints (ints 0)
              else fail_at place "the tile holds no data"))

(* Reading a window *)

(* The output: a host buffer of the window in ZBITPIX's format, filled in
   place. *)
type out =
  | U8 of (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) A.t
  | I16 of (int, Bigarray.int16_signed_elt, Bigarray.c_layout) A.t
  | I32 of (int32, Bigarray.int32_elt, Bigarray.c_layout) A.t
  | I64 of (int64, Bigarray.int64_elt, Bigarray.c_layout) A.t
  | F32 of (float, Bigarray.float32_elt, Bigarray.c_layout) A.t
  | F64 of (float, Bigarray.float64_elt, Bigarray.c_layout) A.t

let set place out k (v : values) j =
  let range lo hi x =
    if x < lo || x > hi then fail_at place "%d is outside BITPIX's type" x
    else x
  in
  match (out, v) with
  | U8 o, Ints a -> A.unsafe_set o k (range 0 255 (A.unsafe_get a j))
  | I16 o, Ints a -> A.unsafe_set o k (range (-32768) 32767 (A.unsafe_get a j))
  | I32 o, Ints a ->
      A.unsafe_set o k
        (Int32.of_int (range (-2147483648) 2147483647 (A.unsafe_get a j)))
  | I64 o, Ints a -> A.unsafe_set o k (Int64.of_int (A.unsafe_get a j))
  | I64 o, Longs a -> A.unsafe_set o k (A.unsafe_get a j)
  | F32 o, Reals a -> A.unsafe_set o k (A.unsafe_get a j)
  | F64 o, Reals a -> A.unsafe_set o k (A.unsafe_get a j)
  | (U8 _ | I16 _ | I32 _ | I64 _), Reals _ ->
      fail_at place "floats in an integer image"
  | (F32 _ | F64 _), (Ints _ | Longs _) ->
      fail_at place "integers in a float image"
  | (U8 _ | I16 _ | I32 _), Longs _ ->
      fail_at place "8-byte integers in a narrower image"

(* [read t bounds] is the window [bounds], given in tensor axis order, in
   ZBITPIX's format: the tiles it meets decoded, each once. *)
let read t bounds : Nx.packed =
  let rank = Array.length t.axes in
  (* file order *)
  let fb = Array.init rank (fun i -> bounds.(rank - 1 - i)) in
  let ext = Array.map (fun (a, b) -> b - a) fb in
  let n = Array.fold_left ( * ) 1 ext in
  let w = abs t.zbitpix / 8 in
  let host = Hdu.host_bytes (n * w) in
  let out =
    match t.zbitpix with
    | 8 -> U8 (Nx_device.Buffer.bigarray Bigarray.int8_unsigned host)
    | 16 -> I16 (Nx_device.Buffer.bigarray Bigarray.int16_signed host)
    | 32 -> I32 (Nx_device.Buffer.bigarray Bigarray.int32 host)
    | 64 -> I64 (Nx_device.Buffer.bigarray Bigarray.int64 host)
    | -32 -> F32 (Nx_device.Buffer.bigarray Bigarray.float32 host)
    | _ -> F64 (Nx_device.Buffer.bigarray Bigarray.float64 host)
  in
  if n > 0 then begin
    let budget = ref t.table.heap_size in
    let lo = Array.mapi (fun i (a, _) -> a / t.tile.(i)) fb in
    let hi = Array.mapi (fun i (_, b) -> (b - 1) / t.tile.(i)) fb in
    let ti = Array.copy lo in
    (* output strides in file order: axis 0 is the fastest *)
    let ostride = Array.make rank 1 in
    for i = 1 to rank - 1 do
      ostride.(i) <- ostride.(i - 1) * ext.(i - 1)
    done;
    let tstride = Array.make rank 1 in
    let finished = ref false in
    while not !finished do
      let number = ref 0 and mult = ref 1 in
      for i = 0 to rank - 1 do
        number := !number + (ti.(i) * !mult);
        mult := !mult * t.counts.(i)
      done;
      let start = Array.mapi (fun i k -> k * t.tile.(i)) ti in
      let len =
        Array.mapi (fun i s -> Int.min t.tile.(i) (t.axes.(i) - s)) start
      in
      for i = 1 to rank - 1 do
        tstride.(i) <- tstride.(i - 1) * len.(i - 1)
      done;
      let npix = Array.fold_left ( * ) 1 len in
      let place = tile_place t !number in
      let v = decode_tile t !number npix budget in
      (* the intersection of the tile and the window *)
      let a = Array.mapi (fun i s -> Int.max s (fst fb.(i))) start in
      let b =
        Array.mapi (fun i s -> Int.min (s + len.(i)) (snd fb.(i))) start
      in
      let idx = Array.copy a in
      let run = b.(0) - a.(0) in
      let rec loop axis =
        if axis = 0 then begin
          let tk = ref 0 and ok = ref 0 in
          for i = 1 to rank - 1 do
            tk := !tk + ((idx.(i) - start.(i)) * tstride.(i));
            ok := !ok + ((idx.(i) - fst fb.(i)) * ostride.(i))
          done;
          let tk = !tk + (a.(0) - start.(0))
          and ok = !ok + (a.(0) - fst fb.(0)) in
          for r = 0 to run - 1 do
            set place out (ok + r) v (tk + r)
          done
        end
        else
          for x = a.(axis) to b.(axis) - 1 do
            idx.(axis) <- x;
            loop (axis - 1)
          done
      in
      loop (rank - 1);
      (* next tile, axis 0 fastest *)
      let rec bump i =
        if i = rank then finished := true
        else if ti.(i) < hi.(i) then ti.(i) <- ti.(i) + 1
        else begin
          ti.(i) <- lo.(i);
          bump (i + 1)
        end
      in
      bump 0
    done
  end;
  let shape = Array.map (fun (a, b) -> b - a) bounds in
  let view s = Nx_device.Buffer.view host ~offset:0 s n in
  match t.zbitpix with
  | 8 -> Nx.P (Nx.of_buffer Nx.uint8 shape host)
  | 16 -> Nx.P (Nx.of_buffer Nx.int16 shape (view Int16))
  | 32 -> Nx.P (Nx.of_buffer Nx.int32 shape (view Int32))
  | 64 -> Nx.P (Nx.of_buffer Nx.int64 shape (view Int64))
  | -32 -> Nx.P (Nx.of_buffer Nx.float32 shape (view Float32))
  | _ -> Nx.P (Nx.of_buffer Nx.float64 shape (view Float64))
