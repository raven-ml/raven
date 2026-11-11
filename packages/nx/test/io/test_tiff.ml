(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* TIFF. Files built here byte by byte from TIFF 6.0, BigTIFF and Adobe's
   notes load as the raster they encode, for every value of a tag nx.io
   reads, packed samples of every width among them; every other value is
   refused, naming its tag; a raster saved loads as itself, bit for bit,
   with its tags; files GDAL wrote load as the arrays it was given, and a
   tile of a Sentinel-2 file as GDAL decodes it; a file cut short or
   corrupted raises [Failure] naming it, and never anything else. *)

open Windtrap
open Io_support
module T = Nx_io.Tiff

(* Byte order *)

(* A file's form: its byte order, MM (big-endian) or II, and BigTIFF's
   8-byte offsets or classic TIFF's 4. *)
type form = { mm : bool; bigtiff : bool }

let pp_form ppf f =
  Format.fprintf ppf "%s %s"
    (if f.mm then "MM" else "II")
    (if f.bigtiff then "BigTIFF" else "classic")

(* [n] in [w] bytes, in [f]'s byte order. *)
let num f w n =
  String.init w (fun i ->
      let k = if f.mm then w - 1 - i else i in
      let b = Int64.logand (Int64.shift_right_logical n (8 * k)) 0xffL in
      Char.chr (Int64.to_int b))

let u16 f n = num f 2 (Int64.of_int n)
let u32 f n = num f 4 (Int64.of_int n)
let word f n = if f.bigtiff then num f 8 (Int64.of_int n) else u32 f n

(* Fields *)

(* A field: its tag, type, count and values as the file holds them. *)
type field = { tag : int; typ : int; count : int; data : string }

let field tag typ count data = { tag; typ; count; data }
let values put vs = String.concat "" (List.map put vs)
let short f tag vs = field tag 3 (List.length vs) (values (u16 f) vs)
let long f tag vs = field tag 4 (List.length vs) (values (u32 f) vs)

let long8 f tag vs =
  field tag 16 (List.length vs) (values (fun v -> num f 8 (Int64.of_int v)) vs)

(* Offsets and counts: LONG in classic TIFF, LONG8 in BigTIFF. *)
let offsets f tag vs = if f.bigtiff then long8 f tag vs else long f tag vs

(* [replace fd fs] is [fs] with its field of [fd]'s tag replaced by [fd],
   [drop tag fs] without it. *)
let replace fd fs = fd :: List.filter (fun x -> x.tag <> fd.tag) fs
let drop tag fs = List.filter (fun x -> x.tag <> tag) fs
let by_tag fs = List.sort (fun a b -> Int.compare a.tag b.tag) fs

(* Files *)

(* The file of [f] whose IFDs each hold [blobs], written first after the
   header, and [fields at], given each blob's offset. The IFDs follow the
   data they describe, which TIFF 6.0 allows ("an Image File Directory may
   follow the image data it describes"), each with its values after it. *)
let build f ifds =
  let b = Buffer.create 1024 in
  let wide = if f.bigtiff then 8 else 4 in
  let entry = if f.bigtiff then 20 else 12 in
  Buffer.add_string b (if f.mm then "MM" else "II");
  if f.bigtiff then Buffer.add_string b (u16 f 43 ^ u16 f 8 ^ u16 f 0)
  else Buffer.add_string b (u16 f 42);
  let link = ref (Buffer.length b) in
  Buffer.add_string b (String.make wide '\000');
  let place s =
    let at = Buffer.length b in
    Buffer.add_string b s;
    at
  in
  let ifds =
    List.map (fun (blobs, fields) -> fields (List.map place blobs)) ifds
  in
  let patch at n =
    let s = Buffer.to_bytes b in
    Bytes.blit_string (word f n) 0 s at wide;
    Buffer.clear b;
    Buffer.add_bytes b s
  in
  List.iter
    (fun fields ->
      if Buffer.length b mod 2 = 1 then Buffer.add_char b '\000';
      patch !link (Buffer.length b);
      let n = List.length fields in
      let count = if f.bigtiff then 8 else 2 in
      let after = Buffer.length b + count + (n * entry) + wide in
      let tail = Buffer.create 256 in
      Buffer.add_string b (if f.bigtiff then word f n else u16 f n);
      List.iter
        (fun fd ->
          Buffer.add_string b (u16 f fd.tag ^ u16 f fd.typ);
          Buffer.add_string b
            (if f.bigtiff then num f 8 (Int64.of_int fd.count)
             else u32 f fd.count);
          let len = String.length fd.data in
          if len <= wide then
            Buffer.add_string b (fd.data ^ String.make (wide - len) '\000')
          else begin
            Buffer.add_string b (word f (after + Buffer.length tail));
            Buffer.add_string tail fd.data;
            if Buffer.length tail mod 2 = 1 then Buffer.add_char tail '\000'
          end)
        fields;
      link := Buffer.length b;
      Buffer.add_string b (String.make wide '\000');
      Buffer.add_buffer b tail)
    ifds;
  Buffer.contents b

(* Rasters *)

(* A raster's samples as their bits, row-major: [h] rows of [w] pixels of
   [c] samples of [bits] bits each, of SampleFormat [format]. *)
type raster = {
  h : int;
  w : int;
  c : int;
  bits : int;
  format : int;
  samples : int64 array;
}

(* The samples of [x], of shape [[|h; w; c|]], as their bits. *)
let raster (Nx.P x) =
  let dt = Nx.dtype x in
  let h, w, c =
    match Nx.shape x with [| h; w; c |] -> (h, w, c) | _ -> assert false
  in
  let k = D.bytes dt 1 in
  let b = bytes x in
  let sample i =
    let v = ref 0L in
    for j = k - 1 downto 0 do
      let j = if Sys.big_endian then k - 1 - j else j in
      v := Int64.logor (Int64.shift_left !v 8) (Int64.of_int b.((i * k) + j))
    done;
    !v
  in
  let format = match D.kind dt with D.Float -> 3 | D.Signed -> 2 | _ -> 1 in
  { h; w; c; bits = 8 * k; format; samples = Array.init (h * w * c) sample }

(* How a built file cuts and stores its raster: strips of rows or tiles of
   rows by columns, Compression 1, 5 or 8, Predictor 1, 2 or 3, and
   whether a last strip holds rows past the raster's. *)
type chunks = Strips of int | Tiles of int * int

type layout = {
  form : form;
  chunks : chunks;
  codec : int;
  predictor : int;
  padded : bool;
}

(* LZW codes of single bytes (TIFF 6.0, section 13), most significant bit
   first: ClearCode, each byte, with a ClearCode every 250 codes so that
   the table never holds string 510 and every code is 9 bits wide, then
   EndOfInformation. *)
let lzw_literals s =
  let b = Buffer.create (String.length s * 2) in
  let acc = ref 0 and bits = ref 0 in
  let put c =
    acc := (!acc lsl 9) lor c;
    bits := !bits + 9;
    while !bits >= 8 do
      bits := !bits - 8;
      Buffer.add_char b (Char.chr ((!acc lsr !bits) land 0xff))
    done;
    acc := !acc land ((1 lsl !bits) - 1)
  in
  put 256;
  String.iteri
    (fun i ch ->
      if i > 0 && i mod 250 = 0 then put 256;
      put (Char.code ch))
    s;
  put 257;
  if !bits > 0 then
    Buffer.add_char b (Char.chr ((!acc lsl (8 - !bits)) land 0xff));
  Buffer.contents b

let mask bits v =
  if bits = 64 then v
  else Int64.logand v (Int64.pred (Int64.shift_left 1L bits))

let byte_of v k =
  Int64.to_int (Int64.logand (Int64.shift_right_logical v (8 * k)) 0xffL)

(* Whether samples of [bits] bits are packed, as TIFF 6.0 stores any width
   but 8, 16, 32 and 64 bits: "Otherwise BYTE". *)
let is_packed bits = not (List.mem bits [ 8; 16; 32; 64 ])

(* A row of samples of [bits] bits, packed most significant bit first and
   padded to a byte (TIFF 6.0, section 8, Compression and FillOrder). *)
let pack bits row =
  let b = Bytes.make (((Array.length row * bits) + 7) / 8) '\000' in
  Array.iteri
    (fun i v ->
      for k = 0 to bits - 1 do
        let bit = Int64.shift_right_logical v (bits - 1 - k) in
        if Int64.logand bit 1L = 1L then begin
          let at = (i * bits) + k in
          let old = Char.code (Bytes.get b (at / 8)) in
          Bytes.set b (at / 8) (Char.chr (old lor (0x80 lsr (at mod 8))))
        end
      done)
    row;
  Bytes.to_string b

(* One chunk's rows of samples, [n] to a row, as [l] stores them: after
   its predictor, in its byte order, compressed. *)
let encode l r rows n =
  let bytes = r.bits / 8 in
  let rows = Array.map Array.copy rows in
  if l.predictor = 2 then
    Array.iter
      (fun row ->
        for i = n - 1 downto r.c do
          row.(i) <- mask r.bits (Int64.sub row.(i) row.(i - r.c))
        done)
      rows;
  (* Technical Note 3: byte [k] of every sample, the most significant
     first, then each byte less the byte a pixel before. *)
  let planes row =
    let p =
      Bytes.init (n * bytes) (fun i ->
          Char.chr (byte_of row.(i mod n) (bytes - 1 - (i / n))))
    in
    for i = (n * bytes) - 1 downto r.c do
      let d = Char.code (Bytes.get p i) - Char.code (Bytes.get p (i - r.c)) in
      Bytes.set p i (Char.chr (d land 0xff))
    done;
    Bytes.to_string p
  in
  let row_bytes row =
    if l.predictor = 3 then planes row
    else if is_packed r.bits then pack r.bits row
    else String.concat "" (Array.to_list (Array.map (num l.form bytes) row))
  in
  let raw = String.concat "" (Array.to_list (Array.map row_bytes rows)) in
  match l.codec with
  | 5 -> lzw_literals raw
  | 8 | 32946 -> Lz.Zlib.encode_string raw
  | _ -> raw

let sides l r =
  match l.chunks with Strips k -> (Int.min k r.h, r.w) | Tiles (a, b) -> (a, b)

(* The chunks of [r] as [l] cuts them, each its rows of samples, zero past
   the raster's edges. *)
let cut l r =
  let rows, cols = sides l r in
  let across = (r.w + cols - 1) / cols and down = (r.h + rows - 1) / rows in
  let chunk k =
    let r0 = k / across * rows and c0 = k mod across * cols in
    let n =
      match l.chunks with
      | Strips _ when not l.padded -> Int.min rows (r.h - r0)
      | _ -> rows
    in
    Array.init n (fun i ->
        Array.init (cols * r.c) (fun j ->
            let y = r0 + i and x = c0 + (j / r.c) in
            if y < r.h && x < r.w then
              r.samples.((((y * r.w) + x) * r.c) + (j mod r.c))
            else 0L))
  in
  List.init (across * down) chunk

(* The tags that state [r] cut by [l], given its chunks' offsets [at] and
   byte counts. *)
let fields l r ~at ~counts =
  let f = l.form in
  let per_sample v = List.init r.c (fun _ -> v) in
  let rows, cols = sides l r in
  let chunks =
    match l.chunks with
    | Strips k -> [ offsets f 273 at; long f 278 [ k ]; offsets f 279 counts ]
    | Tiles _ ->
        [
          long f 322 [ cols ]; long f 323 [ rows ]; offsets f 324 at;
          offsets f 325 counts;
        ]
  in
  [
    long f 256 [ r.w ]; long f 257 [ r.h ]; short f 258 (per_sample r.bits);
    short f 259 [ l.codec ]; short f 262 [ 1 ]; short f 277 [ r.c ];
    short f 284 [ 1 ]; short f 317 [ l.predictor ];
    short f 339 (per_sample r.format);
  ]
  @ chunks

(* An IFD of [blobs] described by [l]'s fields of [r], with [edit]. *)
let ifd ?(edit = Fun.id) l r blobs =
  let counts = List.map String.length blobs in
  (blobs, fun at -> by_tag (edit (fields l r ~at ~counts)))

(* The file of [r] laid out by [l], with [edit] applied to its fields. *)
let tiff ?edit l r =
  let encoded rows = encode l r rows (Array.length rows.(0)) in
  let blobs = List.map encoded (cut l r) in
  build l.form [ ifd ?edit l r blobs ]

let load_one path =
  match Nx_io.load_tiff path with
  | [ d ] -> T.samples d
  | ds -> failf "%d IFDs" (List.length ds)

let tags_of path = T.tags (List.hd (Nx_io.load_tiff path))

(* Tag values compare structurally, and print as their type and values. *)
let value =
  let pp ppf v =
    let all f a = String.concat " " (Array.to_list (Array.map f a)) in
    let ints = all string_of_int in
    let s =
      match v with
      | T.Byte a -> "BYTE " ^ ints a
      | T.Short a -> "SHORT " ^ ints a
      | T.Long a -> "LONG " ^ ints a
      | T.Sbyte a -> "SBYTE " ^ ints a
      | T.Sshort a -> "SSHORT " ^ ints a
      | T.Slong a -> "SLONG " ^ ints a
      | T.Ascii s -> Printf.sprintf "ASCII %S" s
      | T.Undefined s -> Printf.sprintf "UNDEFINED %S" s
      | T.Rational a | T.Srational a ->
          all (fun (n, d) -> Printf.sprintf "%d/%d" n d) a
      | T.Float a | T.Double a -> all (Printf.sprintf "%h") a
      | T.Long8 a | T.Slong8 a -> all Int64.to_string a
    in
    Format.pp_print_string ppf s
  in
  Testable.make ~pp ~equal:( = )

(* Drawn files *)

let tiff_dtypes =
  D.
    [
      Any Uint8; Any Uint16; Any Int16; Any Uint32; Any Int32; Any Float32;
      Any Float64;
    ]

(* Rasters of a dtype TIFF stores: 1 to 40 rows and columns, one in five a
   single row or column, of 1 to 4 samples, any values. *)
let rasters =
  let open Gen in
  let* (D.Any dt) =
    of_list ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt) tiff_dtypes
  in
  let side = frequency [ (1, constant 1); (4, int_range 1 40) ] in
  let* h = side and+ w = side and+ c = int_range 1 4 in
  let+ vs = array ~size:(constant (h * w * c)) (element dt) in
  Nx.P (Nx.create dt [| h; w; c |] vs)

let layouts (Nx.P x) =
  let open Gen in
  let float = D.is D.Float (Nx.dtype x) in
  let* form = map (fun (mm, bigtiff) -> { mm; bigtiff }) (pair bool bool) in
  let side = int_range 1 3 in
  let tile = map (fun (a, b) -> Tiles (16 * a, 16 * b)) (pair side side) in
  let* chunks = one_of [ map (fun k -> Strips k) (int_range 1 50); tile ] in
  let* codec = frequency [ (3, of_list [ 1; 5; 8 ]); (1, constant 32946) ] in
  let* predictor =
    if codec = 1 then constant 1
    else of_list (if float then [ 1; 2; 3 ] else [ 1; 2 ])
  in
  let+ padded = bool in
  { form; chunks; codec; predictor; padded }

let pp_layout ppf l =
  Format.fprintf ppf "%a %s, Compression %d, Predictor %d%s" pp_form l.form
    (match l.chunks with
    | Strips k -> Printf.sprintf "strips of %d rows" k
    | Tiles (a, b) -> Printf.sprintf "tiles of %d by %d" a b)
    l.codec l.predictor
    (if l.padded then ", last strip padded" else "")

let built =
  Gen.with_pp
    (fun ppf (x, l) -> Format.fprintf ppf "%a, %a" pp_packed x pp_layout l)
    Gen.(
      let* x = rasters in
      let+ l = layouts x in
      (x, l))

let reading =
  prop "a file built from the specifications loads as the raster it encodes"
    built (fun (x, l) ->
      let r = raster x in
      let tiled, edge, strips =
        match l.chunks with
        | Tiles (a, b) -> (true, r.h mod a <> 0 || r.w mod b <> 0, false)
        | Strips k -> (false, false, k < r.h)
      in
      cover "MM" l.form.mm;
      cover "BigTIFF" l.form.bigtiff;
      cover "tiles" tiled;
      cover "a tile past the raster's edge" edge;
      cover "several strips" strips;
      cover "a last strip padded" (strips && l.padded);
      cover "LZW" (l.codec = 5);
      cover "Deflate" (l.codec = 8);
      cover "Deflate's obsolete code" (l.codec = 32946);
      cover "horizontal differencing" (l.predictor = 2);
      cover "the floating point predictor" (l.predictor = 3);
      cover "several samples a pixel" (r.c > 1);
      equal packed x (load_one (file (tiff l r))))

(* Packed samples *)

(* The widths every run reaches. *)
let widths = [ 1; 3; 12; 15; 17; 31 ]

(* [r]'s samples at the narrowest unsigned dtype that holds them. *)
let unsigned r =
  let s = [| r.h; r.w; r.c |] and to_int = Array.map Int64.to_int in
  if r.bits <= 8 then Nx.P (Nx.create Nx.uint8 s (to_int r.samples))
  else if r.bits <= 16 then Nx.P (Nx.create Nx.uint16 s (to_int r.samples))
  else Nx.P (Nx.create Nx.uint32 s (Array.map Int64.to_int32 r.samples))

(* Rasters of unsigned samples of 1 to 32 bits, the widths above often, any
   values, zero and the largest among them, and layouts of them: strips or
   tiles, every compression, Predictor 1, either byte order. *)
let packed_files =
  let open Gen in
  let any = List.filter (( <> ) 24) (List.init 32 succ) in
  let* bits = frequency [ (3, of_list widths); (1, of_list any) ] in
  let side = frequency [ (1, constant 1); (4, int_range 1 40) ] in
  let* h = side and+ w = side and+ c = int_range 1 4 in
  let top = Int64.pred (Int64.shift_left 1L bits) in
  let sample =
    frequency
      [
        (1, constant 0L); (1, constant top); (4, map (Int64.logand top) int64);
      ]
  in
  let* samples = array ~size:(constant (h * w * c)) sample in
  let* mm = bool in
  let* bigtiff = bool in
  let side = int_range 1 3 in
  let tile = map (fun (a, b) -> Tiles (16 * a, 16 * b)) (pair side side) in
  let* chunks = one_of [ map (fun k -> Strips k) (int_range 1 50); tile ] in
  let* codec = of_list [ 1; 5; 8; 32946 ] in
  let+ padded = bool in
  let l = { form = { mm; bigtiff }; chunks; codec; predictor = 1; padded } in
  ({ h; w; c; bits; format = 1; samples }, l)

let packs =
  prop "packed samples of 1 to 32 bits but 24 load as the values they pack"
    (Gen.with_pp
       (fun ppf (r, l) ->
         Format.fprintf ppf "%d by %d by %d of %d bits, %a" r.h r.w r.c r.bits
           pp_layout l)
       packed_files)
    (fun (r, l) ->
      let cols = snd (sides l r) in
      List.iter
        (fun b -> cover (Printf.sprintf "%d bits" b) (r.bits = b))
        widths;
      cover "MM" l.form.mm;
      cover "tiles" (match l.chunks with Tiles _ -> true | Strips _ -> false);
      cover "a row that ends inside a byte" (cols * r.c * r.bits mod 8 <> 0);
      cover "uncompressed" (l.codec = 1);
      cover "LZW" (l.codec = 5);
      cover "Deflate" (l.codec = 8);
      equal packed (unsigned r) (load_one (file (tiff l r))))

(* A small raster, classic little-endian strips of it, and its file with a
   field edit. *)
let small =
  Nx.P
    (Nx.init Nx.uint16 [| 4; 5; 3 |] (fun i ->
         (i.(0) * 100) + (i.(1) * 10) + i.(2)))

let f = { mm = false; bigtiff = false }

let plain =
  { form = f; chunks = Strips 2; codec = 1; predictor = 1; padded = false }

let edited ?(l = plain) ?(x = small) edit = file (tiff ~edit l (raster x))
let grey = Nx.P (Nx.init Nx.uint8 [| 3; 2; 1 |] (fun i -> i.(0) + i.(1)))

let variants =
  group "values TIFF 6.0 lets a file state"
    [
      cases "an unsigned integer field of BYTE, SHORT or LONG" ~id:Fun.id
        [ "BYTE"; "SHORT"; "LONG" ]
        (fun ty ->
          let width =
            match ty with
            | "BYTE" -> field 256 1 1 "\005"
            | "SHORT" -> short f 256 [ 5 ]
            | _ -> long f 256 [ 5 ]
          in
          equal packed small (load_one (edited (replace width))));
      test "one BitsPerSample and SampleFormat for every sample" (fun () ->
          let one fs =
            replace (short f 258 [ 16 ]) (replace (short f 339 [ 1 ]) fs)
          in
          equal packed small (load_one (edited one)));
      test "no RowsPerStrip, one strip of the whole raster" (fun () ->
          let l = { plain with chunks = Strips 4 } in
          equal packed small (load_one (edited ~l (drop 278))));
      test "no SampleFormat, Predictor, PlanarConfiguration or \
            SamplesPerPixel, at their defaults" (fun () ->
          let none fs = drop 339 (drop 317 (drop 284 (drop 277 fs))) in
          equal packed grey (load_one (edited ~x:grey none)));
      test "PlanarConfiguration 2 with one sample a pixel" (fun () ->
          let planar = replace (short f 284 [ 2 ]) in
          equal packed grey (load_one (edited ~x:grey planar)));
      test "an IFD before its data, and a chain of two" (fun () ->
          (* A save writes its IFD first; [build] after the data. *)
          let a = saved ".tif" (fun p (Nx.P x) -> Nx_io.save_tiff p x) small in
          equal packed small (load_one a);
          let r = raster small in
          let blobs = List.map (fun x -> encode plain r x 15) (cut plain r) in
          let path = file (build f [ ifd plain r blobs; ifd plain r blobs ]) in
          equal (list packed) [ small; small ]
            (List.map T.samples (Nx_io.load_tiff path)));
      test "fields of types TIFF does not define are left out" (fun () ->
          let odd fs =
            field 330 13 1 (u32 f 8) :: field 40000 99 2 "ab" :: fs
          in
          let tags = tags_of (edited odd) in
          equal (list int) []
            (List.filter (fun t -> t = 330 || t = 40000) (List.map fst tags)));
      test "an ASCII field without its NUL keeps its bytes" (fun () ->
          let ascii fs = field 305 2 3 "abc" :: field 306 2 4 "abc\000" :: fs in
          let tags = tags_of (edited ascii) in
          equal (list (option value))
            [ Some (T.Ascii "abc"); Some (T.Ascii "abc") ]
            (List.map (fun t -> List.assoc_opt t tags) [ 305; 306 ]));
    ]

(* Refusals *)

let refused ?l ?x ~naming edit =
  let path = edited ?l ?x edit in
  fails ~path ~naming (fun () -> load_one path)

(* A file of [small]'s strips laid out by [l], whose blobs are [blobs]. *)
let with_blobs l blobs = file (build f [ ifd l (raster small) blobs ])

let refusals =
  group "values nx.io does not read, refused naming their tag"
    [
      cases "Compression" ~id:string_of_int [ 2; 7; 32773; 50000 ]
        (fun v ->
          refused
            ~naming:(Printf.sprintf "Compression (259) %d" v)
            (replace (short f 259 [ v ])));
      cases "Predictor" ~id:(fun (name, _, _) -> name)
        [
          ("4 with Deflate", { plain with codec = 8 }, 4);
          ("2 with Compression 1", plain, 2);
          ("3 on integers with Deflate", { plain with codec = 8 }, 3);
          ("3 on integers with LZW", { plain with codec = 5 }, 3);
        ]
        (fun (_, l, v) ->
          refused ~l
            ~naming:(Printf.sprintf "Predictor (317) %d" v)
            (replace (short f 317 [ v ])));
      cases "Predictor 2 on packed samples" ~id:string_of_int [ 1; 12; 15 ]
        (fun bits ->
          let l = { plain with codec = 8 } in
          refused ~l
            ~naming:
              (Printf.sprintf "Predictor (317) 2: on samples of %d bits" bits)
            (fun fs ->
              replace (short f 317 [ 2 ])
                (replace (short f 258 [ bits; bits; bits ]) fs)));
      cases "24-bit unsigned samples, in either byte order"
        ~id:(Format.asprintf "%a" pp_form)
        [ f; { f with mm = true } ]
        (fun form ->
          let l = { plain with form } in
          refused ~l ~naming:"BitsPerSample (258) 24: unsigned"
            (replace (short form 258 [ 24; 24; 24 ])));
      cases "BitsPerSample and SampleFormat"
        ~id:(fun (b, s) -> Printf.sprintf "%d bits, SampleFormat %d" b s)
        [ (0, 1); (33, 1); (64, 1); (8, 2); (64, 2); (16, 3); (16, 4) ]
        (fun (bits, format) ->
          let naming =
            if format = 4 then "SampleFormat (339) 4"
            else Printf.sprintf "BitsPerSample (258) %d" bits
          in
          refused ~naming (fun fs ->
              replace (short f 258 [ bits ])
                (replace (short f 339 [ format ]) fs)));
      test "BitsPerSample that differs between samples" (fun () ->
          refused ~naming:"BitsPerSample (258) differs"
            (replace (short f 258 [ 16; 16; 8 ])));
      test "PlanarConfiguration 2 with three samples a pixel" (fun () ->
          refused ~naming:"PlanarConfiguration (284) 2"
            (replace (short f 284 [ 2 ])));
      test "FillOrder 2" (fun () ->
          refused ~naming:"FillOrder (266) 2" (fun fs ->
              short f 266 [ 2 ] :: fs));
      test "Photometric 6 (YCbCr)" (fun () ->
          refused ~naming:"Photometric (262) 6" (replace (short f 262 [ 6 ])));
      test "ImageWidth of type ASCII" (fun () ->
          refused ~naming:"ImageWidth (256) of type ASCII"
            (replace (field 256 2 2 "5\000")));
      test "ImageWidth 0" (fun () ->
          refused ~naming:"ImageWidth (256) 0" (replace (long f 256 [ 0 ])));
      test "no ImageLength" (fun () ->
          refused ~naming:"ImageLength (257) is missing" (drop 257));
      test "strips and tiles both" (fun () ->
          refused ~naming:"both StripOffsets (273) and TileOffsets (324)"
            (fun fs -> long f 324 [ 8 ] :: fs));
      test "neither strips nor tiles" (fun () ->
          refused ~naming:"neither StripOffsets (273) nor TileOffsets (324)"
            (drop 273));
      test "no StripByteCounts" (fun () ->
          refused ~naming:"StripByteCounts (279) is missing" (drop 279));
      test "an offset too few for the strips" (fun () ->
          refused ~naming:"StripOffsets (273) holds 1 offsets for 2 chunks"
            (replace (long f 273 [ 8 ])));
      test "a tile of no byte, as GDAL's sparse files leave it" (fun () ->
          let l = { plain with chunks = Tiles (16, 16) } in
          refused ~l ~naming:"TileByteCounts (325): tile 0 has no byte"
            (replace (long f 325 [ 0 ])));
      test "a strip past the file's end" (fun () ->
          refused ~naming:"strip 1 at byte"
            (replace (long f 279 [ 60; 1000 ])));
      test "an uncompressed strip shorter than its rows" (fun () ->
          refused ~naming:"StripByteCounts (279): strip 0 holds 59 bytes"
            (replace (long f 279 [ 59; 60 ])));
      cases "strips that share bytes past the file's length" ~id:fst
        [
          (* 40 one-row strips of 500 bytes, all one row's. *)
          ("uncompressed", (1, String.make 500 'a', 500));
          (* 64 one-row strips of 1024 bytes, all one zlib stream's. *)
          ("Deflate", (8, Lz.Zlib.encode_string (String.make 1024 'a'), 1024));
        ]
        (fun (_, (codec, blob, w)) ->
          let h = if codec = 1 then 40 else 64 in
          let x = Nx.P (Nx.zeros Nx.uint8 [| h; w; 1 |]) in
          let l = { plain with chunks = Strips 1; codec } in
          let at = List.init h (fun _ -> 8) in
          let counts = List.init h (fun _ -> String.length blob) in
          let fields _ = by_tag (fields l (raster x) ~at ~counts) in
          let path = file (build f [ ([ blob ], fields) ]) in
          fails ~path
            ~naming:"StripByteCounts (279): the strips take more than"
            (fun () -> load_one path));
      test "an LZW strip with a code past the table" (fun () ->
          let l = { plain with codec = 5 } in
          let r = raster small in
          let blobs = List.map (fun rows -> encode l r rows 15) (cut l r) in
          (* The strip's second code, its bits 9 to 17, set to 300 where
             the table holds no entry. *)
          let s = Bytes.of_string (List.hd blobs) in
          let set i v = Bytes.set s i (Char.chr v) in
          set 1 ((Char.code (Bytes.get s 1) land 0x80) lor (300 lsr 2));
          set 2
            ((Char.code (Bytes.get s 2) land 0x3f) lor ((300 land 3) lsl 6));
          let path = with_blobs l (Bytes.to_string s :: List.tl blobs) in
          fails ~path
            ~naming:
              "Compression (259) 5: strip 0 at byte 8: malformed at byte 1: \
               invalid code"
            (fun () -> load_one path));
      test "a Deflate strip that decodes to fewer bytes than its rows"
        (fun () ->
          let z n = Lz.Zlib.encode_string (String.make n 'a') in
          let path = with_blobs { plain with codec = 8 } [ z 59; z 60 ] in
          fails ~path ~naming:"strip 0 decodes to 59 bytes, not 60" (fun () ->
              load_one path));
    ]

(* Headers and directories *)

(* A classic little-endian file of one IFD at byte 8 of the [entries]
   given as bytes, chained to [next]. *)
let raw_ifd entries next =
  "II*\000\008\000\000\000"
  ^ u16 f (List.length entries)
  ^ String.concat "" entries ^ u32 f next

let headers =
  group "headers and directories"
    [
      cases "a file that is not TIFF" ~id:String.escaped
        [
          ""; "GIF89a"; "II\042"; "II\044\000\008\000\000\000";
          "II+\000\004\000\000\000"; "II*\000\000\000\000\000";
        ]
        (fun s ->
          let path = file s in
          fails ~path (fun () -> Nx_io.load_tiff path));
      test "an IFD that chains back to itself" (fun () ->
          let width = u16 f 256 ^ u16 f 3 ^ u32 f 1 ^ u32 f 5 in
          let path = file (raw_ifd [ width ] 8) in
          fails ~path ~naming:"the IFDs' chain comes back to byte 8" (fun () ->
              Nx_io.load_tiff path));
      test "an IFD that holds a tag twice" (fun () ->
          let path = edited (fun fs -> short f 256 [ 5 ] :: fs) in
          fails ~path ~naming:"holds tag 256 twice" (fun () ->
              Nx_io.load_tiff path));
      test "a value past the file's end" (fun () ->
          let width = u16 f 256 ^ u16 f 3 ^ u32 f 4 ^ u32 f 100 in
          let path = file (raw_ifd [ width ] 0) in
          fails ~path ~naming:"the file ends at byte 26" (fun () ->
              Nx_io.load_tiff path));
      test "values that share bytes past the file's length" (fun () ->
          (* 50 UNDEFINED fields of 1000 bytes, each at byte 0 of a file
             of 1100: 50,000 bytes of values. *)
          let shared t = u16 f t ^ u16 f 7 ^ u32 f 1000 ^ u32 f 0 in
          let ifd = raw_ifd (List.init 50 (fun i -> shared (40000 + i))) 0 in
          let path = file (ifd ^ String.make (1100 - String.length ifd) 'x') in
          fails ~path ~naming:"the IFDs and their values take more than"
            (fun () -> Nx_io.load_tiff path));
      test "an IFD's tags load from a file whose chunks are cut off"
        (fun () ->
          (* A save's IFD comes first: its first 300 bytes hold every tag,
             and its samples fail. *)
          let big = Nx.P (Nx.zeros Nx.uint8 [| 64; 64; 1 |]) in
          let path = saved ".tif" (fun p (Nx.P x) -> Nx_io.save_tiff p x) big in
          let path = file (String.sub (read path) 0 300) in
          let d = List.hd (Nx_io.load_tiff path) in
          equal (option value) (Some (T.Long [| 64 |]))
            (List.assoc_opt 256 (T.tags d));
          fails ~path ~naming:"ends past the file's" (fun () -> T.samples d));
    ]

(* Every IFD's samples of a file, which a corrupted file loads or
   refuses. *)
let load_all path =
  List.iter (fun d -> ignore (T.samples d)) (Nx_io.load_tiff path)

(* Three files of a few hundred bytes, so that a file per prefix and per
   byte stays small: LZW strips, Deflate tiles with the floating point
   predictor in BigTIFF, and big-endian uncompressed strips. *)
let damaged =
  let ints =
    raster
      (Nx.P
         (Nx.init Nx.uint16 [| 5; 7; 2 |] (fun i -> (i.(0) * i.(1)) + i.(2))))
  in
  let fls =
    raster
      (Nx.P
         (Nx.init Nx.float32 [| 5; 7; 1 |] (fun i ->
              Float.of_int (i.(0) - i.(1)) /. 3.)))
  in
  let tiles =
    {
      plain with
      form = { mm = false; bigtiff = true };
      chunks = Tiles (16, 16);
      codec = 8;
      predictor = 3;
    }
  in
  [
    ("LZW strips", tiff { plain with codec = 5; predictor = 2 } ints);
    ("Deflate tiles, BigTIFF", tiff tiles fls);
    ("big-endian strips", tiff { plain with form = { f with mm = true } } ints);
  ]

let corrupted =
  cases "every proper prefix fails, and any byte flipped loads or fails"
    ~id:fst damaged (fun (_, whole) ->
      prefixes_fail load_all whole;
      flips_load_or_fail load_all whole)

(* Round trips *)

let views = [ Plain; Transposed; Flipped; Broadcast ]

(* [x] laid out by [view]: transposed or flipped over its first two axes,
   or broadcast from its first row. *)
let viewed view (Nx.P x) =
  match view with
  | Plain -> Nx.P x
  | Transposed -> Nx.P (Nx.transpose ~axes:[ 1; 0; 2 ] x)
  | Flipped -> Nx.P (Nx.flip ~axes:[ 0 ] x)
  | Broadcast ->
      Nx.P (Nx.broadcast_to (Nx.shape x) (Nx.slice [ Nx.R (0, 1) ] x))

let pp_saved ppf (view, x, tile, compression) =
  Format.fprintf ppf "%a %a, %s, %s" pp_view view pp_packed x
    (match tile with
    | None -> "strips"
    | Some (a, b) -> Printf.sprintf "tiles of %d by %d" a b)
    (match compression with
    | T.Uncompressed -> "uncompressed"
    | T.Deflate -> "Deflate")

let saves =
  Gen.with_pp pp_saved
    Gen.(
      let* x = rasters in
      let* view = of_list views in
      let side = int_range 1 2 in
      let tiles = map (fun (a, b) -> Some (16 * a, 16 * b)) (pair side side) in
      let* tile = frequency [ (1, constant None); (2, tiles) ] in
      let+ compression = of_list [ T.Uncompressed; T.Deflate ] in
      (view, x, tile, compression))

let saved_tags =
  [
    (269, T.Ascii "name");
    (282, T.Rational [| (300, 1) |]);
    (33550, T.Double [| 10.; 10.; 0. |]);
    ( 33922,
      T.Double [| 0.; 0.; 0.; 301822.86256412434; 5798366.72759726; 0. |] );
    (34735, T.Short [| 1; 1; 0; 1; 3072; 0; 1; 32631 |]);
    (42113, T.Ascii "0");
    (50000, T.Byte [| 0; 255 |]);
    (50001, T.Sbyte [| -128; 127 |]);
    (50002, T.Undefined "\000\001");
    (50003, T.Sshort [| -32768; 32767 |]);
    (50004, T.Slong [| -0x8000_0000; 0x7fff_ffff |]);
    (50005, T.Srational [| (-1, 3) |]);
    (50006, T.Float [| 0.5; Float.infinity |]);
    (50007, T.Long [| 0; 0xffff_ffff |]);
  ]

let round_trips =
  let (Nx.P s) = small in
  group "round trips"
    [
      prop "a raster saved loads as itself, bit for bit" saves
        (fun (view, x, tile, compression) ->
          let (Nx.P y) = viewed view x in
          cover "Deflate" (compression = T.Deflate);
          cover "tiles" (Option.is_some tile);
          cover "a view" (view <> Plain);
          let path = temp_file ~suffix:".tif" () in
          Nx_io.save_tiff ?tile ~compression path y;
          equal packed (Nx.P (Nx.copy y)) (load_one path));
      test "tags of every type round trip, GeoTIFF's among them" (fun () ->
          let path = temp_file ~suffix:".tif" () in
          Nx_io.save_tiff ~tags:saved_tags path s;
          let read = tags_of path in
          equal (list (option value))
            (List.map (fun (_, v) -> Some v) saved_tags)
            (List.map (fun (t, _) -> List.assoc_opt t read) saved_tags);
          equal packed small (load_one path));
      test "a LONG8 tag makes the file BigTIFF" (fun () ->
          let path = temp_file ~suffix:".tif" () in
          Nx_io.save_tiff ~tags:[ (50000, T.Long8 [| -1L |]) ] path s;
          equal string "II+\000" (String.sub (read path) 0 4);
          equal (option value) (Some (T.Long8 [| -1L |]))
            (List.assoc_opt 50000 (tags_of path));
          equal packed small (load_one path));
      test "the save states Photometric 1 and ExtraSamples unless told"
        (fun () ->
          let path = temp_file ~suffix:".tif" () in
          let read t = List.assoc_opt t (tags_of path) in
          Nx_io.save_tiff path s;
          equal (list (option value))
            [ Some (T.Short [| 1 |]); Some (T.Short [| 0; 0 |]) ]
            [ read 262; read 338 ];
          Nx_io.save_tiff ~tags:[ (262, T.Short [| 2 |]) ] path s;
          equal (list (option value))
            [ Some (T.Short [| 2 |]); None ]
            [ read 262; read 338 ]);
      test "uncompressed strips and tiles as wide as the raster load over the \
            file's pages" (fun () ->
          let path = temp_file ~suffix:".tif" () in
          let x =
            Nx.init Nx.float32 [| 64; 64; 2 |] (fun i ->
                Float.of_int (i.(0) - i.(1)))
          in
          Nx_io.save_tiff path x;
          equal memory Over_file (memory_of (load_one path));
          Nx_io.save_tiff ~tile:(16, 64) path x;
          equal memory Over_file (memory_of (load_one path));
          Nx_io.save_tiff ~tile:(16, 32) path x;
          equal memory Own_memory (memory_of (load_one path));
          Nx_io.save_tiff ~compression:Deflate path x;
          equal memory Own_memory (memory_of (load_one path));
          let mm = { plain with form = { f with mm = true } } in
          let path = file (tiff mm (raster small)) in
          equal memory Own_memory (memory_of (load_one path)));
    ]

let save_refusals =
  let (Nx.P x) = small in
  let invalid save =
    invalid ~by:"Nx_io.save_tiff" (fun () -> save (temp_file ~suffix:".tif" ()))
  in
  group "save refusals"
    [
      cases "a dtype TIFF does not store" ~id:(fun (D.Any dt) -> D.name dt)
        D.
          [
            Any Int8; Any Int64; Any Uint64; Any Float16; Any Bfloat16;
            Any Bool; Any Complex64;
          ]
        (fun (D.Any dt) ->
          invalid (fun p -> Nx_io.save_tiff p (Nx.zeros dt [| 2; 2; 1 |])));
      cases "a shape" ~id:(Format.asprintf "%a" pp_shape)
        [ [| 2; 2 |]; [| 0; 2; 1 |]; [| 2; 2; 0 |]; [| 1; 1; 65536 |] ]
        (fun s -> invalid (fun p -> Nx_io.save_tiff p (Nx.zeros Nx.uint8 s)));
      cases "a tile that is not a positive multiple of 16"
        ~id:(fun (a, b) -> Printf.sprintf "%d by %d" a b)
        [ (15, 16); (16, 0); (-16, 16); (24, 16) ]
        (fun tile -> invalid (fun p -> Nx_io.save_tiff ~tile p x));
      cases "tags it cannot write" ~id:fst
        [
          ("the save's own", [ (259, T.Short [| 8 |]) ]);
          ("twice", [ (300, T.Short [| 1 |]); (300, T.Short [| 2 |]) ]);
          ("past a SHORT's tag numbers", [ (65536, T.Short [| 1 |]) ]);
          ("no value", [ (300, T.Short [||]) ]);
          ("a SHORT of 70000", [ (300, T.Short [| 70000 |]) ]);
          ("a FLOAT that is not a binary32", [ (300, T.Float [| 0.1 |]) ]);
        ]
        (fun (_, tags) -> invalid (fun p -> Nx_io.save_tiff ~tags p x));
    ]

(* Files GDAL wrote *)

let gdal =
  cases "files GDAL wrote load as the arrays it was given" ~id:Fun.id
    [
      "tiff_lzw_u16_tiles"; "tiff_deflate_f32_bigtiff_mm";
      "tiff_lzw_f64_strips"; "tiff_deflate_i16_mm_tiles"; "tiff_none_u32";
      "tiff_lzw_i32"; "tiff_lzw_u8_rgb"; "tiff_none_u1"; "tiff_none_u3_mm";
      "tiff_lzw_u12_rgb"; "tiff_deflate_u15_tiles"; "tiff_deflate_u17_bigtiff";
      "tiff_deflate_u31_mm_tiles";
    ]
    (fun name ->
      equal packed
        (Nx_io.load_npy (fixture (name ^ ".npy")))
        (load_one (fixture (name ^ ".tif"))))

(* Partial files *)

module P = Nx_io.Partial

(* The partial file "part" of [whole] that holds the bytes [held] marks,
   added in pieces of at most 61 bytes, so that a chunk often spans
   several. *)
let partial whole held =
  let n = String.length whole in
  let rec from i p =
    if i = n then p
    else if not held.(i) then from (i + 1) p
    else begin
      let j = ref i in
      while !j < n && held.(!j) && !j - i < 61 do
        incr j
      done;
      from !j (P.add p ~at:i (String.sub whole i (!j - i)))
    end
  in
  from 0 (P.v ~name:"part" ~length:n)

(* What a call did: returned, lacked ranges, or raised, its message past
   its file's name. A call on a whole file returns [Ok]. *)
type 'a outcome =
  | Done of 'a
  | Lacks of (int * int) list
  | Raised of string
  | Refused of string

let past name m =
  let prefix = name ^ ": " in
  let k = String.length prefix in
  if String.starts_with ~prefix m then String.sub m k (String.length m - k)
  else m

let outcome name f =
  match f () with
  | Ok x -> Done x
  | Error rs -> Lacks rs
  | exception Failure m -> Raised (past name m)
  | exception Invalid_argument m -> Refused m

let outcomes t =
  let pp ppf = function
    | Done x -> Format.fprintf ppf "Done %a" (Testable.pp t) x
    | Lacks rs ->
        Format.fprintf ppf "Lacks [%s]"
          (String.concat "; "
             (List.map (fun (a, n) -> Printf.sprintf "(%d, %d)" a n) rs))
    | Raised m -> Format.fprintf ppf "Raised %S" m
    | Refused m -> Format.fprintf ppf "Refused %S" m
  in
  let equal a b =
    match (a, b) with
    | Done x, Done y -> Testable.equal t x y
    | Done _, _ | _, Done _ -> false
    | _ -> a = b
  in
  Testable.make ~pp ~equal

let ifds = outcomes (list (list (pair int value)))
let windowed = outcomes packed
let decoded_tags = List.map T.tags

let tags = function
  | Done ds -> Done (decoded_tags ds)
  | Lacks rs -> Lacks rs
  | Raised m -> Raised m
  | Refused m -> Refused m

(* A partial file's IFDs, and a window of an IFD, as outcomes. *)
let decode_of p = outcome "part" (fun () -> Nx_io.decode_tiff p)

let window_of name d (row, rows, col, cols) =
  outcome name (fun () -> T.window d ~row ~col ~rows ~cols)

(* The same on the whole file [whole]. *)
let whole_of whole k win =
  let path = file whole in
  let ds = outcome path (fun () -> Ok (Nx_io.load_tiff path)) in
  let w =
    match ds with
    | Done ds when k < List.length ds ->
        Some (window_of path (List.nth ds k) win)
    | _ -> None
  in
  (tags ds, w)

(* [rs] is a list of ranges of a file of [n] bytes in normal form: not
   empty, each of a byte or more, sorted, apart, and lacking from
   [held]. *)
let check_ranges ~held n rs =
  if rs = [] then failf "no range";
  ignore
    (List.fold_left
       (fun prev (at, len) ->
         if len <= 0 || at < prev || at + len > n then
           failf "range (%d, %d) after byte %d, in %d bytes" at len prev n;
         for i = at to at + len - 1 do
           if held.(i) then failf "range (%d, %d) holds byte %d" at len i
         done;
         at + len + 1)
       0 rs
      : int)

let hold held rs = List.iter (fun (at, n) -> Array.fill held at n true) rs

(* Drawn partial files: a file of one or two IFDs of a raster built from
   the specifications, packed samples among them, half with a byte flipped,
   anywhere or in the first IFD; the bytes held, none or all but a few
   holes, and a few ranges more; an IFD, and a window of its raster: whole,
   of no pixel or any. *)
type drawn = {
  whole : string;
  r : raster;
  l : layout;
  count : int;
  held : bool array;
  more : (int * int) list;
  k : int;
  win : int * int * int * int;
}

let pp_drawn ppf d =
  let holes = Array.fold_left (fun n h -> if h then n else n + 1) 0 d.held in
  let row, rows, col, cols = d.win in
  Format.fprintf ppf
    "%d by %d by %d of %d bits, %a, %d IFDs, %d of %d bytes lacking, IFD %d, \
     %d rows from %d, %d columns from %d"
    d.r.h d.r.w d.r.c d.r.bits pp_layout d.l d.count holes
    (String.length d.whole) d.k rows row cols col

let span n =
  let open Gen in
  frequency
    [
      (1, constant (0, n));
      (1, map (fun a -> (a, 0)) (int_range 0 n));
      ( 4,
        let* a = int_range 0 (n - 1) in
        let+ k = int_range 1 (n - a) in
        (a, k) );
    ]

let built_raster =
  let open Gen in
  let* x = rasters in
  let+ l = layouts x in
  (raster x, l)

let drawn =
  let open Gen in
  let* r, l = frequency [ (3, built_raster); (1, packed_files) ] in
  let* count = frequency [ (3, constant 1); (1, constant 2) ] in
  let encoded rows = encode l r rows (Array.length rows.(0)) in
  let blobs = List.map encoded (cut l r) in
  let built = build l.form (List.init count (fun _ -> ifd l r blobs)) in
  let n = String.length built in
  (* The first IFD's offset, from the header, and its first 200 bytes. *)
  let first =
    match (l.form.bigtiff, l.form.mm) with
    | false, false -> Int32.to_int (String.get_int32_le built 4)
    | false, true -> Int32.to_int (String.get_int32_be built 4)
    | true, false -> Int64.to_int (String.get_int64_le built 8)
    | true, true -> Int64.to_int (String.get_int64_be built 8)
  in
  let* flip =
    frequency
      [
        (2, constant None);
        (1, map Option.some (int_range 0 (n - 1)));
        (1, map Option.some (int_range first (Int.min (n - 1) (first + 200))));
      ]
  in
  let whole = match flip with None -> built | Some i -> flip_byte built i in
  let range =
    let* at = int_range 0 (n - 1) in
    let+ len = int_range 1 (Int.max 1 (n / 8)) in
    (at, Int.min len (n - at))
  in
  let* held =
    frequency
      [
        (1, constant (Array.make n false));
        ( 4,
          let+ holes = list ~size:(int_range 0 4) range in
          let held = Array.make n true in
          List.iter (fun (at, len) -> Array.fill held at len false) holes;
          held );
      ]
  in
  let* more = list ~size:(int_range 0 3) range in
  let* k = int_range 0 (count - 1) in
  let+ row, rows = span r.h and+ col, cols = span r.w in
  { whole; r; l; count; held; more; k; win = (row, rows, col, cols) }

let drawn = Gen.with_pp pp_drawn drawn

(* [fill whole p rs] is [p] holding [whole]'s bytes of the ranges [rs]. *)
let fill whole p rs =
  List.fold_left (fun p (at, n) -> P.add p ~at (String.sub whole at n)) p rs

let raised = function Raised _ -> true | _ -> false

let covers_layout d =
  cover "two IFDs" (d.count = 2);
  cover "packed samples" (is_packed d.r.bits);
  cover "tiles" (match d.l.chunks with Tiles _ -> true | Strips _ -> false);
  cover "LZW" (d.l.codec = 5);
  cover "Deflate" (d.l.codec = 8);
  cover "horizontal differencing" (d.l.predictor = 2);
  cover "the floating point predictor" (d.l.predictor = 3);
  cover "MM" d.l.form.mm;
  cover "BigTIFF" d.l.form.bigtiff

let agreement =
  prop ~count:400
    "a partial file decodes as its whole file, once it holds the ranges it \
     names"
    drawn (fun d ->
      let n = String.length d.whole and held = Array.copy d.held in
      let decoded, window = whole_of d.whole d.k d.win in
      (* Each call's ranges are added until it returns or raises: a round
         adds a byte or more, so a file takes at most a round a byte. *)
      let rounds = ref 0 in
      let rec loop p call =
        incr rounds;
        if !rounds > n + 2 then failf "%d rounds for %d bytes" !rounds n;
        match call p with
        | Lacks rs ->
            check_ranges ~held n rs;
            hold held rs;
            loop (fill d.whole p rs) call
        | o -> (o, p)
      in
      let o, p = loop (partial d.whole d.held) decode_of in
      covers_layout d;
      cover "nothing held at first" (Array.for_all not d.held);
      cover "raised" (raised o);
      equal ifds decoded (tags o);
      let first = !rounds in
      let window_at p =
        match decode_of p with
        | Done ds -> window_of "part" (List.nth ds d.k) d.win
        | o -> failf "decoded once, then %a" (Testable.pp ifds) (tags o)
      in
      match (o, window) with
      | Done ds, Some window when d.k < List.length ds ->
          let w, _ = loop p window_at in
          let _, rows, _, cols = d.win in
          cover "a window that lacked a chunk" (!rounds > first + 1);
          cover "a window of no pixel" (rows = 0 || cols = 0);
          equal windowed window w
      | _ -> ())

let unread =
  prop ~count:300 "no result depends on a byte the partial file lacks" drawn
    (fun d ->
      let n = String.length d.whole in
      let flip i c =
        if d.held.(i) then c else Char.chr (Char.code c lxor 0xff)
      in
      let other = String.mapi flip d.whole in
      let decoded, window = whole_of d.whole d.k d.win in
      let decoded', window' = whole_of other d.k d.win in
      let lacking = Array.exists not d.held in
      let o = decode_of (partial d.whole d.held) in
      let w =
        match o with
        | Lacks rs ->
            check_ranges ~held:d.held n rs;
            None
        | o -> (
            equal ifds decoded (tags o);
            equal ifds decoded' (tags o);
            match o with
            | Done ds when d.k < List.length ds -> (
                match window_of "part" (List.nth ds d.k) d.win with
                | Lacks rs ->
                    check_ranges ~held:d.held n rs;
                    None
                | w ->
                    equal (option windowed) window (Some w);
                    equal (option windowed) window' (Some w);
                    Some w)
            | _ -> None)
      in
      cover "IFDs decoded beside a hole"
        (lacking && match o with Done _ -> true | _ -> false);
      cover "a window beside a hole" (lacking && Option.is_some w);
      cover "a raise beside a hole"
        (lacking && (raised o || Option.fold ~none:false ~some:raised w)))

let monotone =
  prop ~count:300 "adding bytes never turns a result into Error" drawn
    (fun d ->
      let p = partial d.whole d.held in
      let q = fill d.whole p d.more in
      match decode_of p with
      | Done ds when d.k < List.length ds -> (
          let ds' =
            match decode_of q with
            | Done ds' -> ds'
            | o -> failf "decoded, then %a" (Testable.pp ifds) (tags o)
          in
          equal ifds (Done (decoded_tags ds)) (Done (decoded_tags ds'));
          let old = List.nth ds d.k and fresh = List.nth ds' d.k in
          match window_of "part" old d.win with
          | Done x ->
              cover "a window, then more bytes" (d.more <> []);
              equal windowed (Done x) (window_of "part" fresh d.win)
          | Lacks rs ->
              (* The IFD decoded before the ranges were added never reads
                 them; one decoded after does. *)
              let q = fill d.whole q rs in
              let fresh = List.nth (Result.get_ok (Nx_io.decode_tiff q)) d.k in
              cover "an IFD decoded before its ranges were added" true;
              equal windowed (Lacks rs) (window_of "part" old d.win);
              if window_of "part" fresh d.win = Lacks rs then
                failf "the ranges added, the same ranges again"
          | Raised _ | Refused _ -> ())
      | _ -> ())

(* [small]'s file: its two strips of 60 bytes at bytes 8 and 68, then its
   IFD. *)
let strips = tiff plain (raster small)
let strips_held lo hi =
  Array.init (String.length strips) (fun i -> i < lo || i >= hi)
let lacks rs = Lacks rs

let partial_cases =
  group "partial files"
    [
      test "ranges merge where they touch and name a chunk's tail" (fun () ->
          let win p rows =
            match Nx_io.decode_tiff p with
            | Ok [ d ] -> window_of "part" d (0, rows, 0, 5)
            | _ -> failf "not one IFD"
          in
          let p = partial strips (strips_held 8 128) in
          equal windowed (lacks [ (8, 120) ]) (win p 4);
          equal windowed (lacks [ (8, 60) ]) (win p 2);
          let p = partial strips (strips_held 18 128) in
          equal windowed (lacks [ (18, 110) ]) (win p 4));
      test "a header, then an IFD's entry count, then its entries" (fun () ->
          let p = P.v ~name:"part" ~length:(String.length strips) in
          equal ifds (lacks [ (0, 8) ]) (tags (decode_of p));
          let p = fill strips p [ (0, 8) ] in
          equal ifds (lacks [ (128, 2) ]) (tags (decode_of p));
          let p = fill strips p [ (128, 2) ] in
          let n = String.get_uint16_le strips 128 in
          equal ifds (lacks [ (130, (12 * n) + 4) ]) (tags (decode_of p)));
      test "a BigTIFF header's offset after its first 8 bytes" (fun () ->
          let l = { plain with form = { f with bigtiff = true } } in
          let whole = tiff l (raster small) in
          let p = partial whole (Array.init (String.length whole) (( > ) 8)) in
          equal ifds (lacks [ (8, 8) ]) (tags (decode_of p)));
      test "a file shorter than a header" (fun () ->
          equal ifds
            (Raised "the file ends at byte 5, before byte 8")
            (tags (decode_of (P.v ~name:"part" ~length:5))));
      test "a window of no row reads no chunk" (fun () ->
          let p = partial strips (strips_held 8 128) in
          let d = List.hd (Result.get_ok (Nx_io.decode_tiff p)) in
          let (Nx.P s) = small in
          let none = Nx.P (Nx.slice [ Nx.R (0, 0) ] s) in
          equal windowed (Done none) (window_of "part" d (4, 0, 0, 5)));
      test "samples raise naming the first range lacking" (fun () ->
          let p = partial strips (strips_held 30 40) in
          let d = List.hd (Result.get_ok (Nx_io.decode_tiff p)) in
          raises (Failure "part: the partial file lacks bytes 30 to 39")
            (fun () -> T.samples d));
      test "a window outside the raster" (fun () ->
          let d = List.hd (Nx_io.load_tiff (file strips)) in
          List.iter
            (fun (row, rows, col, cols) ->
              invalid ~by:"Nx_io.Tiff.window" (fun () ->
                  T.window d ~row ~col ~rows ~cols))
            [ (0, 5, 0, 5); (-1, 1, 0, 5); (0, 4, 1, 5); (0, 4, 0, -1) ]);
      test "added bytes that agree, and bytes that differ" (fun () ->
          let p = P.v ~name:"part" ~length:10 in
          let p = P.add (P.add p ~at:0 "abcd") ~at:2 "cdef" in
          let q = P.add p ~at:6 "gh" in
          raises (Failure "part: byte 3 differs from the byte already held")
            (fun () -> P.add q ~at:3 "xy");
          let invalid f = invalid ~by:"Nx_io.Partial.add" f in
          invalid (fun () -> P.add p ~at:(-1) "a");
          invalid (fun () -> P.add p ~at:8 "abc");
          raises_match (Exn.invalid_arg ~sub:"Nx_io.Partial.v: ") (fun () ->
              P.v ~name:"part" ~length:(-1)));
      test "an add leaves the partial file it was given as it was" (fun () ->
          let p = partial strips (strips_held 8 128) in
          let before = tags (decode_of p) in
          ignore (fill strips p [ (8, 120) ] : P.t);
          equal ifds before (tags (decode_of p)));
    ]

(* A Sentinel-2 file *)

(* Bytes of a cloud-optimised GeoTIFF of the Microsoft Planetary Computer,
   Sentinel-2 L2A tile T31UCT's B04 band of 2025-07-01 (baseline N05.11),
   [s2_length] bytes: its first 6556, which hold its five IFDs and their
   values, and tile 348 of its first IFD, at [s2_at], a Deflate stream of
   512 by 512 samples of 15 bits, partly nodata. cut/NOTICE gives their
   source. *)
let s2_length = 246_755_406
let s2_at = 207_111_118
let s2_head = Filename.concat "cut" "s2_b04_head.bin"
let s2_tile = Filename.concat "cut" "s2_b04_tile348.bin"

(* The tile as GDAL 3.12.2 decodes it from the whole file: the MD5 of its
   samples as little-endian [uint16], in C order, and their largest value,
   sum and count of nonzero. *)
let s2_md5 = "9e8cb3052738828e8d6d1470be4772aa"
let s2_stats = [ 2478; 6_822_003; 3292 ]

let s2_digest p =
  let x = Nx.unpack Nx.uint16 p in
  let vs = Nx.to_array x in
  let le = Bytes.create (2 * Array.length vs) in
  Array.iteri (fun i v -> Bytes.set_uint16_le le (2 * i) v) vs;
  ( Nx.shape x,
    Digest.to_hex (Digest.bytes le),
    [
      Array.fold_left Int.max 0 vs;
      Array.fold_left ( + ) 0 vs;
      Array.fold_left (fun n v -> if v = 0 then n else n + 1) 0 vs;
    ] )

let s2_expected = ([| 512; 512; 1 |], s2_md5, s2_stats)
let digest = triple (array int) string (list int)

let sentinel =
  test "a tile of a Sentinel-2 file loads as GDAL decodes it" (fun () ->
      (* The tile under an IFD of the file's first IFD's sample tags, its
         raster cut to the tile. *)
      let first = T.tags (List.hd (Nx_io.load_tiff s2_head)) in
      equal (list (option value))
        [
          Some (T.Short [| 15 |]); Some (T.Short [| 8 |]);
          Some (T.Short [| 1 |]); Some (T.Short [| 1 |]);
          Some (T.Short [| 512 |]); Some (T.Short [| 512 |]);
        ]
        (List.map
           (fun t -> List.assoc_opt t first)
           [ 258; 259; 317; 339; 322; 323 ]);
      let tile = read s2_tile in
      let fields at =
        by_tag
          [
            short f 256 [ 512 ]; short f 257 [ 512 ]; short f 258 [ 15 ];
            short f 259 [ 8 ]; short f 262 [ 1 ]; short f 277 [ 1 ];
            short f 284 [ 1 ]; short f 317 [ 1 ]; short f 322 [ 512 ];
            short f 323 [ 512 ]; long f 324 at;
            long f 325 [ String.length tile ]; short f 339 [ 1 ];
          ]
      in
      let path = file (build f [ ([ tile ], fields) ]) in
      equal digest s2_expected (s2_digest (load_one path)))

let sentinel_ranges =
  test "a tile of a Sentinel-2 file read by its ranges" (fun () ->
      let p = P.add (P.v ~name:"s2" ~length:s2_length) ~at:0 (read s2_head) in
      let ds = Result.get_ok (Nx_io.decode_tiff p) in
      equal (list (list (pair int value)))
        (decoded_tags (Nx_io.load_tiff s2_head))
        (decoded_tags ds);
      (* Tile 348 is row 15 and column 18 of 22 tiles a row. *)
      let tile d = window_of "s2" d (15 * 512, 512, 18 * 512, 512) in
      equal windowed (lacks [ (s2_at, 6141) ]) (tile (List.hd ds));
      let p = P.add p ~at:s2_at (read s2_tile) in
      match tile (List.hd (Result.get_ok (Nx_io.decode_tiff p))) with
      | Done x -> equal digest s2_expected (s2_digest x)
      | o -> failf "%a" (Testable.pp windowed) o)

let () =
  exit
    (run "nx.io TIFF"
       [
         reading; packs; variants; refusals; headers; corrupted; round_trips;
         save_refusals; gdal; sentinel; agreement; unread; monotone;
         partial_cases; sentinel_ranges;
       ])
