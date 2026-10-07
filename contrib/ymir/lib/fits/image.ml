(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Err
module B = Nx_device.Buffer
module S = Nx_dtype.Scalar

let strf = Printf.sprintf

(* Elements *)

(* The dtypes that hold every value of an element exactly, in the order an
   error lists them. *)
let holders : S.t -> S.t list = function
  | UInt8 ->
      [
        UInt8;
        Int16;
        UInt16;
        Int32;
        UInt32;
        Int64;
        UInt64;
        Float16;
        BFloat16;
        Float32;
        Float64;
        Complex64;
        Complex128;
      ]
  | Int8 ->
      [
        Int8;
        Int16;
        Int32;
        Int64;
        Float16;
        BFloat16;
        Float32;
        Float64;
        Complex64;
        Complex128;
      ]
  | Int16 -> [ Int16; Int32; Int64; Float32; Float64; Complex64; Complex128 ]
  | UInt16 ->
      [
        UInt16;
        Int32;
        UInt32;
        Int64;
        UInt64;
        Float32;
        Float64;
        Complex64;
        Complex128;
      ]
  | Int32 -> [ Int32; Int64; Float64; Complex128 ]
  | UInt32 -> [ UInt32; Int64; UInt64; Float64; Complex128 ]
  | Int64 -> [ Int64 ]
  | UInt64 -> [ UInt64 ]
  | Float32 -> [ Float32; Float64; Complex64; Complex128 ]
  | Float64 -> [ Float64; Complex128 ]
  | Complex64 -> [ Complex64; Complex128 ]
  | s -> [ s ]

let is_complex : S.t -> bool = function
  | Complex64 | Complex128 -> true
  | _ -> false

let or_list = function
  | [] -> ""
  | [ x ] -> x
  | l ->
      let r = List.rev l in
      String.concat ", " (List.rev (List.tl r)) ^ " or " ^ List.hd r

(* [check_holds place element dtype] fails unless [dtype] holds [element]. *)
let check_holds place element (dtype : S.t) =
  let hs = holders element in
  if not (List.exists (S.equal dtype) hs) then
    fail_at place "%s does not read exactly as %s; read it as %s"
      (S.to_string element) (S.to_string dtype)
      (or_list
         (List.filter_map
            (fun s -> if is_complex s then None else Some (S.to_string s))
            hs))

(* Byte order *)

let stored_dtype bitpix : Nx.packed =
  match bitpix with
  | 8 -> Nx.P (Nx.zeros Nx.uint8 [| 0 |])
  | 16 -> Nx.P (Nx.zeros Nx.int16 [| 0 |])
  | 32 -> Nx.P (Nx.zeros Nx.int32 [| 0 |])
  | 64 -> Nx.P (Nx.zeros Nx.int64 [| 0 |])
  | -32 -> Nx.P (Nx.zeros Nx.float32 [| 0 |])
  | _ -> Nx.P (Nx.zeros Nx.float64 [| 0 |])

(* [of_big_endian dtype shape bytes] reads the C-order elements of [dtype]
   held big-endian in the host buffer [bytes]: one pass that reverses each
   element's bytes on a little-endian machine. *)
let of_big_endian (type a b) (dtype : (a, b) Nx.dtype) shape bytes : (a, b) Nx.t
    =
  let w = Nx_dtype.itemsize dtype in
  let n = B.nbytes bytes / w in
  if w = 1 then
    Nx.reshape shape (Nx.bitcast dtype (Nx.of_buffer Nx.uint8 [| n |] bytes))
  else
    let u = Nx.of_buffer Nx.uint8 [| n; w |] bytes in
    let u = if Sys.big_endian then u else Nx.flip ~axes:[ 1 ] u in
    Nx.reshape shape (Nx.bitcast dtype u)

(* [to_big_endian t] is a host buffer of [t]'s elements in C order, each
   big-endian. *)
let to_big_endian (type a b) (t : (a, b) Nx.t) : B.t =
  let t = Nx.contiguous (Nx.place Nx.Placement.host t) in
  let w = Nx_dtype.itemsize (Nx.dtype t) in
  let src = Nx.to_buffer (Nx.bitcast Nx.uint8 t) in
  if w = 1 || Sys.big_endian then src
  else begin
    (* each element's bytes reversed, in one pass that allocates nothing *)
    let n = B.nbytes src in
    let dst = B.create Nx_device.host Nx_dtype.Scalar.UInt8 n in
    let a = B.bigarray Bigarray.int8_unsigned src
    and b = B.bigarray Bigarray.int8_unsigned dst in
    let i = ref 0 in
    while !i < n do
      for j = 0 to w - 1 do
        Bigarray.Array1.unsafe_set b (!i + j)
          (Bigarray.Array1.unsafe_get a (!i + w - 1 - j))
      done;
      i := !i + w
    done;
    dst
  end

(* Descriptions *)

type storage = Plain | Tiled of Tiles.t

type t = {
  place : Err.place;
  shape : int array;  (** C order *)
  bitpix : int;
  element : S.t;
  scaled : bool;
  bscale : float;
  bzero : float;
  blank : int64 option;
  storage : storage;
}

let shape t = Array.copy t.shape
let element t = t.element
let scaled t = t.scaled

let pp_shape ppf s =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))

let pp ppf t =
  Format.fprintf ppf "%s %a" (S.to_string t.element) pp_shape t.shape;
  if t.scaled then
    Format.fprintf ppf ", scaled (BSCALE %s, BZERO %s)"
      (Value.print_float t.bscale)
      (Value.print_float t.bzero);
  Option.iter (fun b -> Format.fprintf ppf ", BLANK %Ld" b) t.blank;
  match t.storage with
  | Plain -> ()
  | Tiled tl ->
      Format.fprintf ppf ", %s tiles %a"
        (Tiles.codec_name tl.codec)
        pp_shape
        (Array.of_list (List.rev (Array.to_list tl.tile)));
      if tl.quantized then Format.fprintf ppf ", quantized (%s)" tl.dither_name

let max_rank = 32

(* The offsets of FITS 4.0 Table 11 that make an integer format unsigned, or
   signed for bytes. *)
let offset_element bitpix : (string * S.t) option =
  match bitpix with
  | 8 -> Some ("-128", Int8)
  | 16 -> Some ("32768", UInt16)
  | 32 -> Some ("2147483648", UInt32)
  | 64 -> Some ("9223372036854775808", UInt64)
  | _ -> None

let own_element bitpix : S.t =
  match bitpix with
  | 8 -> UInt8
  | 16 -> Int16
  | 32 -> Int32
  | 64 -> Int64
  | -32 -> Float32
  | _ -> Float64

let number h k ~default =
  match Header.find_struct Value.text k h with
  | Error e -> fail "%s" e
  | Ok None -> (default, float_of_string default)
  | Ok (Some t) -> (
      match Value.read Value.Float t None with
      | Ok x -> (t, x)
      | Error e ->
          fail_at (Header.card_place h (List.hd (Header.cards h k)) k) "%s" e)

let stored_range bitpix =
  match bitpix with
  | 8 -> (0L, 255L)
  | 16 -> (-32768L, 32767L)
  | 32 -> (Int64.of_int32 Int32.min_int, Int64.of_int32 Int32.max_int)
  | _ -> (Int64.min_int, Int64.max_int)

let blank_of h key bitpix =
  if bitpix < 0 then None
  else
    match Header.find_struct Value.text key h with
    | Error e -> fail "%s" e
    | Ok None -> None
    | Ok (Some t) ->
        let at () =
          Header.card_place h (List.hd (Header.cards h "BLANK")) "BLANK"
        in
        let digits =
          String.length t > 0
          && String.for_all
               (fun c -> c >= '0' && c <= '9')
               (String.sub t 1 (String.length t - 1))
        in
        let v =
          match t.[0] with
          | ('+' | '-' | '0' .. '9') when digits ->
              Int64.of_string_opt
                (if t.[0] = '+' then String.sub t 1 (String.length t - 1) else t)
          | _ -> None
        in
        let v =
          match v with
          | Some v -> v
          | None -> fail_at (at ()) "%s is not an integer within int64" t
        in
        let lo, hi = stored_range bitpix in
        if Int64.compare v lo < 0 || Int64.compare v hi > 0 then
          fail_at (at ()) "%Ld is outside BITPIX %d's stored range %Ld to %Ld" v
            bitpix lo hi;
        Some v

(* [describe place h ~bitpix ~axes ~blank storage] is the description of an
   image of [bitpix] and [axes] (in file order), scaled by [h]'s BSCALE and
   BZERO. *)
let describe place h ~bitpix ~axes ~blank storage =
  if Array.length axes = 0 then
    fail_at place "NAXIS = 0: the HDU holds no image";
  if Array.length axes > max_rank then
    fail_at place "%d axes, past the 32 nx holds; Fits.data reads the bytes"
      (Array.length axes);
  let shape = Array.of_list (List.rev (Array.to_list axes)) in
  let bscale_t, bscale = number h "BSCALE" ~default:"1" in
  let bzero_t, bzero = number h "BZERO" ~default:"0" in
  let one = Decimal.equal bscale_t "1" and zero = Decimal.equal bzero_t "0" in
  let element, scaled =
    if one && zero then (own_element bitpix, false)
    else
      match offset_element bitpix with
      | Some (off, e) when one && Decimal.equal bzero_t off -> (e, false)
      | _ -> (own_element bitpix, true)
  in
  { place; shape; bitpix; element; scaled; bscale; bzero; blank; storage }

let of_hdu hdu =
  catch (fun () ->
      let h = Hdu.header hdu in
      let place = Hdu.place hdu in
      let plain () =
        let bitpix = Hdu.bitpix h in
        describe place h ~bitpix ~axes:(Hdu.naxes h)
          ~blank:(blank_of h "BLANK" bitpix)
          Plain
      in
      match
        Hdu.kind_of (Header.find_struct Value.string "XTENSION" h = Ok None) h
      with
      | Primary { groups = true }
        when match Hdu.naxes h with [||] -> false | a -> a.(0) = 0 ->
          fail_at place "random groups are no image; Fits.data reads the bytes"
      | Primary _ | Extension "IMAGE" -> plain ()
      | Extension "BINTABLE"
        when Hdu.find_struct h Value.bool "ZIMAGE" = Some true ->
          let t = Tiles.describe h (Hdu.store hdu) in
          let blank =
            if t.zbitpix < 0 then None
            else
              match blank_of h "BLANK" t.zbitpix with
              | Some b -> Some b
              | None -> blank_of h "ZBLANK" t.zbitpix
          in
          describe place h ~bitpix:t.zbitpix ~axes:t.axes ~blank (Tiled t)
      | Extension x -> fail_at place "a %s extension is no image" x)

(* Windows *)

let window_bounds t window =
  match window with
  | None -> Array.map (fun n -> (0, n)) t.shape
  | Some w ->
      Array.iter
        (fun (a, b) ->
          if a < 0 || a > b then
            invalid_arg
              (strf
                 "Fits.Image: the window range (%d, %d) is not 0 <= start <= \
                  stop"
                 a b))
        w;
      if Array.length w <> Array.length t.shape then
        fail_at t.place "a window of rank %d on an image of shape %s"
          (Array.length w)
          (Format.asprintf "%a" pp_shape t.shape);
      Array.iteri
        (fun i (_, b) ->
          if b > t.shape.(i) then
            fail_at t.place "the window %s passes the shape %s on axis %d"
              (String.concat "; "
                 (Array.to_list
                    (Array.map (fun (a, b) -> strf "(%d, %d)" a b) w)))
              (Format.asprintf "%a" pp_shape t.shape)
              i)
        w;
      Array.copy w

(* Plain storage: the window's runs of bytes copied into host memory, then
   read big-endian. *)
let read_plain t store bounds =
  let w = abs t.bitpix / 8 in
  let rank = Array.length t.shape in
  let out = Array.map (fun (a, b) -> b - a) bounds in
  let n = Array.fold_left ( * ) 1 out in
  let host = Hdu.host_bytes (n * w) in
  if n > 0 then begin
    (* Axes from [k] on are copied as one run: the last axis and every axis
       after the first the window does not cover whole. *)
    let k = ref (rank - 1) in
    while !k > 0 && fst bounds.(!k) = 0 && snd bounds.(!k) = t.shape.(!k) do
      decr k
    done;
    let k = !k in
    let strides = Array.make rank 1 in
    for i = rank - 2 downto 0 do
      strides.(i) <- strides.(i + 1) * t.shape.(i + 1)
    done;
    let run = out.(k) * strides.(k) * w in
    let idx = Array.map fst bounds in
    let at = ref 0 in
    let rec loop axis =
      if axis = k then begin
        let e = ref 0 in
        for i = 0 to rank - 1 do
          e := !e + (idx.(i) * strides.(i))
        done;
        Hdu.copy_in store.Hdu.buffer
          ~offset:(store.Hdu.offset + (!e * w))
          host ~at:!at run;
        at := !at + run
      end
      else
        for v = fst bounds.(axis) to snd bounds.(axis) - 1 do
          idx.(axis) <- v;
          loop (axis + 1)
        done
    in
    loop 0
  end;
  (host, out)

(* The stored numbers of the window, in BITPIX's own format. *)
let stored t hdu bounds : Nx.packed =
  match t.storage with
  | Plain ->
      let host, out = read_plain t (Hdu.store hdu) bounds in
      let (Nx.P z) = stored_dtype t.bitpix in
      Nx.P (of_big_endian (Nx.dtype z) out host)
  | Tiled tiles -> Tiles.read tiles bounds

(* The stored numbers as the element: the offsets of Table 11 added, modulo
   the width, which flips the sign bit. *)
let with_offset element (Nx.P s) : Nx.packed =
  match ((element : S.t), Nx.dtype s) with
  | Int8, UInt8 -> Nx.P (Nx.add_s (Nx.bitcast Nx.int8 s) (-128))
  | UInt16, Int16 -> Nx.P (Nx.add_s (Nx.bitcast Nx.uint16 s) 32768)
  | UInt32, Int32 -> Nx.P (Nx.add_s (Nx.bitcast Nx.uint32 s) Int32.min_int)
  | UInt64, Int64 -> Nx.P (Nx.add_s (Nx.bitcast Nx.uint64 s) Int64.min_int)
  | _ -> Nx.P s

let as_element t s = with_offset t.element s

let raw (type a b) ?window (dtype : (a, b) Nx.dtype) hdu :
    ((a, b) Nx.t, string) result =
  catch (fun () ->
      let t = ok_or_fail (of_hdu hdu) in
      let bounds = window_bounds t window in
      check_holds t.place t.element (S.of_dtype dtype);
      let (Nx.P e) = as_element t (stored t hdu bounds) in
      Nx.cast dtype e)

(* [undefined t s] is where the stored numbers [s] are undefined: BLANK on an
   integer format, NaN on a float one. *)
let undefined t (Nx.P s) : Nx.bool_t option =
  let blank (type a b) (s : (a, b) Nx.t) v : Nx.bool_t =
    match Nx.dtype s with
    | UInt8 -> Nx.equal_s s (Int64.to_int v)
    | Int16 -> Nx.equal_s s (Int64.to_int v)
    | Int32 -> Nx.equal_s s (Int64.to_int32 v)
    | Int64 -> Nx.equal_s s v
    | _ -> Nx.zeros Nx.bool (Nx.shape s)
  in
  if t.bitpix < 0 then Some (Nx.isnan s) else Option.map (blank s) t.blank

let values (type b) ?window (dtype : (float, b) Nx.dtype) hdu :
    ((float, b) Nx.t, string) result =
  catch (fun () ->
      let t = ok_or_fail (of_hdu hdu) in
      let bounds = window_bounds t window in
      let s = stored t hdu bounds in
      let v =
        if not t.scaled then begin
          check_holds t.place t.element (S.of_dtype dtype);
          let (Nx.P e) = as_element t s in
          Nx.cast dtype e
        end
        else
          (* BZERO + BSCALE × s in float64, rounded once, then once more to the
             dtype. *)
          let (Nx.P s) = s in
          let x = Nx.cast Nx.float64 s in
          let p =
            Nx.fma
              (Nx.scalar Nx.float64 t.bscale)
              x
              (Nx.scalar Nx.float64 t.bzero)
          in
          Nx.cast dtype p
      in
      match (t.bitpix > 0, undefined t s) with
      | true, Some u -> Nx.where u (Nx.full dtype (Nx.shape v) Float.nan) v
      | _ -> v)

let validity ?window hdu =
  catch (fun () ->
      let t = ok_or_fail (of_hdu hdu) in
      let bounds = window_bounds t window in
      if t.bitpix > 0 && t.blank = None then None
      else
        match undefined t (stored t hdu bounds) with
        | None -> None
        | Some u ->
            if not (Nx.item [] (Nx.any u)) then None
            else Some (Nx.cast Nx.bit (Nx.logical_not u)))

(* Writing *)

(* BITPIX and the BZERO that store a dtype, or the cast that would. *)
let format_of (type a b) (dtype : (a, b) Nx.dtype) : int * string option =
  let no cast =
    invalid_arg
      (strf "Fits.Image: FITS has no image of %s; %s" (Nx_dtype.to_string dtype)
         cast)
  in
  match dtype with
  | UInt8 -> (8, None)
  | Int8 -> (8, Some "-128")
  | Int16 -> (16, None)
  | UInt16 -> (16, Some "32768")
  | Int32 -> (32, None)
  | UInt32 -> (32, Some "2147483648")
  | Int64 -> (64, None)
  | UInt64 -> (64, Some "9223372036854775808")
  | Float32 -> (-32, None)
  | Float64 -> (-64, None)
  | Bool | Bit -> no "cast it to uint8"
  | Int4 -> no "cast it to int8"
  | UInt4 -> no "cast it to uint8"
  | Float16 | BFloat16 | Float8_e4m3 | Float8_e5m2 -> no "cast it to float32"
  | Complex64 | Complex128 ->
      no "store its real and imaginary parts as two images"

(* The stored numbers of an element: the offset subtracted, modulo the width. *)
let to_stored (type a b) (t : (a, b) Nx.t) : Nx.packed =
  match Nx.dtype t with
  | Int8 -> Nx.P (Nx.bitcast Nx.uint8 (Nx.add_s t (-128)))
  | UInt16 -> Nx.P (Nx.bitcast Nx.int16 (Nx.add_s t 32768))
  | UInt32 -> Nx.P (Nx.bitcast Nx.int32 (Nx.add_s t Int32.min_int))
  | UInt64 -> Nx.P (Nx.bitcast Nx.int64 (Nx.add_s t Int64.min_int))
  | _ -> Nx.P t

let encode t =
  let (Nx.P s) = to_stored t in
  to_big_endian s

(* Rows of at most this many bytes make one slab of a streamed write. *)
let slab_bytes = 1 lsl 22

(* [each_slab t f] calls [f] with the big-endian elements of [t]'s slabs
   along its leading axis, in order. *)
let each_slab (type a b) (t : (a, b) Nx.t) f =
  let shape = Nx.shape t in
  let lead = shape.(0) in
  let row = Nx.nbytes t / Int.max 1 lead in
  let step = Int.max 1 (slab_bytes / Int.max 1 row) in
  let rec go a =
    if a < lead then begin
      let b = Int.min lead (a + step) in
      f (encode (Nx.slice [ Nx.R (a, b) ] t));
      go b
    end
  in
  go 0

let plain_stream header t =
  {
    Hdu.provisional = header;
    write =
      (fun sink ->
        each_slab t sink.Hdu.append;
        header);
  }

(* The tile shape in file order, from [tiles] in tensor axis order, each
   at most its axis. *)
let tile_shape fn shape tiles =
  if Array.length tiles <> Array.length shape then
    invalid_arg
      (strf "Fits.Image.%s: %d tile axes for an image of %d" fn
         (Array.length tiles) (Array.length shape));
  Array.iter
    (fun n ->
      if n < 1 then invalid_arg (strf "Fits.Image.%s: a tile axis of %d" fn n))
    tiles;
  Array.of_list
    (List.rev
       (Array.to_list
          (Array.map2 (fun t n -> Int.max 1 (Int.min t n)) tiles shape)))

(* A tiled image HDU: a BINTABLE of the tiles coded by [plan], tile row by
   tile row from slabs of [t] along its leading axis. *)
let tiled header t ~bitpix ~others ~tile plan =
  let axes = Array.of_list (List.rev (Array.to_list (Nx.shape t))) in
  let slab a b = Hdu.bigbytes (encode (Nx.slice [ Nx.R (a, b) ] t)) in
  (* a quantized image is read once before it is coded, for its seed *)
  let zdither0 =
    Once.make (fun () ->
        match plan with
        | Tiles.Quantized _ ->
            let s = Checksum.summer () in
            each_slab t (fun b ->
                Checksum.feed s (Hdu.bigbytes b) 0 (B.nbytes b));
            Tiles.zdither0 (Checksum.total s)
        | Lossless _ -> 0)
  in
  let header_of (tb : Tiles.table) =
    let columns =
      List.concat
        (List.mapi
           (fun i (name, form) ->
             Structure.
               [
                 string (strf "TTYPE%d" (i + 1)) name;
                 string (strf "TFORM%d" (i + 1)) form;
               ])
           tb.columns)
    in
    let prefix =
      Structure.
        [
          string "XTENSION" "BINTABLE";
          int "BITPIX" 8;
          int "NAXIS" 2;
          int "NAXIS1" tb.row_bytes;
          int "NAXIS2" tb.ntiles;
          int "PCOUNT" tb.pcount;
          int "GCOUNT" 1;
          int "TFIELDS" (List.length tb.columns);
        ]
    in
    let owned k =
      Structure.table_owned k || Structure.tile_owned k
      || Structure.image_owned k
      || (bitpix < 0 && k = "BLANK")
    in
    Structure.apply ~owned ~prefix ~others:(columns @ tb.keys @ others) header
  in
  let code heap =
    Tiles.code ~bitpix ~axes ~tile plan ~zdither0:(Once.get zdither0) ~slab
      ~heap
  in
  let encoded =
    Once.make (fun () ->
        let heap = Buffer.create 4096 in
        let rows, tb = code (Buffer.add_string heap) in
        let data = rows ^ Buffer.contents heap in
        let b = Hdu.host_bytes (String.length data) in
        let a = Hdu.bigbytes b in
        String.iteri
          (fun i c -> Bigarray.Array1.unsafe_set a i (Char.code c))
          data;
        (header_of tb, Hdu.host_store b))
  in
  let write (sink : Hdu.sink) =
    (* the rows go first as zeros, then the heap tile row by tile row, then
       the rows themselves *)
    let tb0 =
      Tiles.table ~bitpix ~axes ~tile plan ~zdither0:0 ~maxes:(0, 0) ~heap:0
    in
    let zeros = Hdu.host_bytes (tb0.ntiles * tb0.row_bytes) in
    Bigarray.Array1.fill (Hdu.bigbytes zeros) 0;
    sink.append zeros;
    let rows, tb =
      code (fun s ->
          let b = Hdu.host_bytes (String.length s) in
          let a = Hdu.bigbytes b in
          String.iteri
            (fun i c -> Bigarray.Array1.unsafe_set a i (Char.code c))
            s;
          sink.append b)
    in
    sink.patch 0 rows;
    header_of tb
  in
  let provisional =
    Once.make (fun () ->
        header_of
          (Tiles.table ~bitpix ~axes ~tile plan ~zdither0:(Once.get zdither0)
             ~maxes:(0, 0) ~heap:0))
  in
  Hdu.constructed_lazy
    ~stream:
      (Once.make (fun () -> { Hdu.provisional = Once.get provisional; write }))
    encoded

let check_shape fn t =
  if Array.length (Nx.shape t) = 0 then
    invalid_arg
      (strf "Fits.Image.%s: a scalar is no image; reshape it to [|1|]" fn)

let hdu ?tiles header (t : ('a, 'b) Nx.t) =
  let bitpix, bzero = format_of (Nx.dtype t) in
  check_shape "hdu" t;
  let shape = Nx.shape t in
  let others =
    match bzero with
    | None -> []
    | Some z -> [ Structure.decimal "BZERO" z; Structure.int "BSCALE" 1 ]
  in
  match tiles with
  | Some tiles ->
      let tile = tile_shape "hdu" shape tiles in
      (* Rice for integers up to 32 bits, GZIP_2 otherwise, lossless. *)
      let codec =
        match bitpix with
        | 8 | 16 | 32 -> Tiles.Rice { block = 32; bytepix = bitpix / 8 }
        | _ -> Tiles.Gzip_2
      in
      tiled header t ~bitpix ~others ~tile (Lossless codec)
  | None ->
      let axes = Array.of_list (List.rev (Array.to_list shape)) in
      let owned k = Structure.image_owned k || (bitpix < 0 && k = "BLANK") in
      let header =
        Structure.apply ~owned
          ~prefix:(Structure.image_prefix ~primary:false ~bitpix ~axes)
          ~others header
      in
      let data = Once.make (fun () -> Hdu.host_store (encode t)) in
      Hdu.constructed
        ~stream:(Once.of_value (plain_stream header t))
        header data

let quantized (type b) ?tiles q header (t : (float, b) Nx.t) =
  if not (Float.is_finite q && q > 0.) then
    invalid_arg (strf "Fits.Image.quantized: %g is not a positive step" q);
  check_shape "quantized" t;
  let bitpix =
    match Nx.dtype t with
    | Float32 -> -32
    | Float64 -> -64
    | d ->
        invalid_arg
          (strf
             "Fits.Image.quantized: FITS quantizes float32 and float64; cast \
              the %s image to float32"
             (Nx_dtype.to_string d))
  in
  let shape = Nx.shape t in
  let tiles =
    match tiles with
    | Some t -> t
    | None ->
        Array.mapi
          (fun i n -> if i = Array.length shape - 1 then n else 1)
          shape
  in
  tiled header t ~bitpix ~others:[]
    ~tile:(tile_shape "quantized" shape tiles)
    (Quantized { q })
