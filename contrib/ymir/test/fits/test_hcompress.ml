(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* H-compressed tiles read as cfitsio decodes them, through astropy
   (gen/fixtures.py): lossless integers of every width, lossy tiles with and
   without SMOOTH, a quantized float, a cube, edge tiles of one row and one
   column, and lossy tiles past their type, clamped. A window reads as the
   slice of the whole, and as a dense decoder (hcompress_dense.ml) reads
   changed streams. A tile of one pixel, built by hand, holds its first
   coefficient. Broken streams are an Error naming their tile, and a window
   of a constant tile of 2^31 pixels, coded in 26 bytes, reads in little
   memory and time. *)

open Windtrap
open Ymir_fits
module H = Fits.Header
module V = Fits.Value
module I = Fits.Image

let hdus = lazy (require_ok (Fits.read "golden/hcompress.fits"))
let hdu name = require_ok (Fits.get name (Lazy.force hdus))

let names =
  [
    "H_I16";
    "H_U16";
    "H_U8";
    "H_I32";
    "H_S4";
    "H_S8_SMOOTH";
    "H_S32_SMOOTH";
    "H_I32_S16";
    "H_Q";
    "H_CUBE";
    "H_ROW";
    "H_COL";
    "H_CLIP";
    "H_CLIP_U8";
    "H_CLIP_I32";
  ]

let expected name =
  In_channel.with_open_text "golden/hcompress.values" In_channel.input_lines
  |> List.find_map (fun l ->
      match String.split_on_char ' ' l with
      | n :: values when n = name -> Some (Array.of_list values)
      | _ -> None)
  |> Option.get

let texts (type a b) (t : (a, b, Nx.host) Nx.t) : string array =
  let a = Nx.to_array (Nx.reshape [| Nx.numel t |] t) in
  match Nx.dtype t with
  | Uint8 -> Array.map string_of_int a
  | Int16 -> Array.map string_of_int a
  | Uint16 -> Array.map string_of_int a
  | Int32 -> Array.map Int32.to_string a
  | _ -> failwith "texts: an integer dtype"

let ints (type a b) (dtype : (a, b) Nx.dtype) name =
  equal (array string) (expected name)
    (texts (require_ok (I.raw dtype (hdu name))))

let decoded =
  cases "tiles decode as cfitsio decodes them" ~id:fst
    [
      ("H_I16", fun n -> ints Nx.int16 n);
      ("H_U16", fun n -> ints Nx.uint16 n);
      ("H_U8", fun n -> ints Nx.uint8 n);
      ("H_I32", fun n -> ints Nx.int32 n);
      ("H_S4", fun n -> ints Nx.int16 n);
      ("H_S8_SMOOTH", fun n -> ints Nx.int16 n);
      ("H_S32_SMOOTH", fun n -> ints Nx.int16 n);
      ("H_I32_S16", fun n -> ints Nx.int32 n);
      ("H_CUBE", fun n -> ints Nx.int16 n);
      ("H_ROW", fun n -> ints Nx.int16 n);
      ("H_COL", fun n -> ints Nx.int16 n);
      ("H_CLIP", fun n -> ints Nx.int16 n);
      ("H_CLIP_U8", fun n -> ints Nx.uint8 n);
      ("H_CLIP_I32", fun n -> ints Nx.int32 n);
      ( "H_Q",
        fun n ->
          let t = require_ok (I.values Nx.float32 (hdu n)) in
          equal (array float_exact)
            (Array.map float_of_string (expected n))
            (Nx.to_array (Nx.reshape [| Nx.numel t |] t)) );
    ]
    (fun (name, f) -> f name)

let described () =
  let pp name = Format.asprintf "%a" Fits.pp (hdu name) in
  expect (String.concat "\n" (List.map pp [ "H_I16"; "H_Q"; "H_CUBE" ]))
  @@ __POS_OF__
       {|
    BINTABLE H_I16 1: int16 [10; 13], HCOMPRESS_1 tiles [5; 7]
    BINTABLE H_Q 1: float32 [10; 13], HCOMPRESS_1 tiles [5; 13], quantized (SUBTRACTIVE_DITHER_1)
    BINTABLE H_CUBE 1: int16 [2; 5; 7], HCOMPRESS_1 tiles [1; 5; 7]
    |}

(* Windows *)

let shape name = I.shape (require_ok (I.of_hdu (hdu name)))

let window_gen shape =
  Gen.(
    let axis n =
      map
        (fun (a, b) -> (Int.min a b, Int.max a b))
        (pair (int_range 0 n) (int_range 0 n))
    in
    let some =
      Array.fold_right
        (fun n acc -> map (fun (r, rs) -> r :: rs) (pair (axis n) acc))
        shape (constant [])
    in
    let pixel =
      map
        (fun i -> Array.map (fun n -> (i mod n, (i mod n) + 1)) shape)
        (int_range 0 1000)
    in
    frequency
      [
        (1, constant (Array.map (fun n -> (0, n)) shape));
        (2, pixel);
        (6, map Array.of_list some);
      ])

let slice w t =
  Nx.slice (Array.to_list (Array.map (fun (a, b) -> Nx.R (a, b)) w)) t

let windows =
  let names = Array.of_list names in
  prop "a window reads as the slice of the whole" ~count:600
    Gen.(
      bind
        (int_range 0 (Array.length names - 1))
        (fun k ->
          map (fun w -> (names.(k), w)) (window_gen (shape names.(k)))))
    (fun (name, w) ->
      Array.iter (fun n -> cover n (n = name)) names;
      cover "one pixel" (Array.for_all (fun (a, b) -> b = a + 1) w);
      cover "empty" (Array.exists (fun (a, b) -> a = b) w);
      let whole = require_ok (I.values Nx.float64 (hdu name)) in
      let part = require_ok (I.values ~window:w Nx.float64 (hdu name)) in
      equal (array int) (Nx.shape (slice w whole)) (Nx.shape part);
      equal (array float_exact)
        (Nx.to_array (slice w whole))
        (Nx.to_array part))

(* Tiles by hand. One tile's table is one descriptor, [count; offset], and
   the heap that follows is the tile's stream. *)

let be32 d at =
  (d.(at) lsl 24) lor (d.(at + 1) lsl 16) lor (d.(at + 2) lsl 8) lor d.(at + 3)

let set32 d at v =
  for i = 0 to 3 do
    d.(at + i) <- (v lsr (8 * (3 - i))) land 0xFF
  done

let heap = 8

(* [stream ~nx ~ny ~scale ~h0] is a stream of [nx × ny] pixels whose
   coefficients other than the first, [h0], are 0: no bit planes. *)
let stream ~nx ~ny ~scale ~h0 =
  let b = Buffer.create 26 in
  Buffer.add_string b "\xDD\x99";
  Buffer.add_int32_be b (Int32.of_int nx);
  Buffer.add_int32_be b (Int32.of_int ny);
  Buffer.add_int32_be b (Int32.of_int scale);
  Buffer.add_int64_be b (Int64.of_int h0);
  Buffer.add_string b "\x00\x00\x00\x00";
  Buffer.contents b

(* [one_tile shape s] is H_S8_SMOOTH's image reshaped to one tile of
   [shape], in tensor axis order, coded by [s]. *)
let one_tile shape s =
  let n = String.length s in
  let rank = Array.length shape in
  let h = ref (Fits.header (hdu "H_S8_SMOOTH")) in
  h :=
    !h |> H.set V.int "PCOUNT" n
    |> H.set V.string "TFORM1" (Printf.sprintf "1PB(%d)" n);
  for i = 1 to rank do
    h :=
      !h
      |> H.set V.int (Printf.sprintf "ZNAXIS%d" i) shape.(rank - i)
      |> H.set V.int (Printf.sprintf "ZTILE%d" i) shape.(rank - i)
  done;
  let d = Array.make (heap + n) 0 in
  set32 d 0 n;
  String.iteri (fun i c -> d.(heap + i) <- Char.code c) s;
  Fits.v !h (Nx.create Nx.uint8 [| heap + n |] d)

let one_pixel =
  (* With no level to invert, the pixel is the first coefficient times the
     scale. *)
  cases "a tile of one pixel holds its first coefficient"
    ~id:(fun (s, _) -> Printf.sprintf "scale %d" s)
    [ (0, 12345); (1, -3); (5, -7) ]
    (fun (scale, h0) ->
      let x = one_tile [| 1; 1 |] (stream ~nx:1 ~ny:1 ~scale ~h0) in
      equal
        (result (array string) string)
        (Ok [| string_of_int (h0 * Int.max 1 scale) |])
        (Result.map texts (I.raw Nx.int16 x)))

(* A dense decoder reads changed streams as the library does. *)

let bytes_of h = Nx.to_array (Fits.data h)

let one_tiles =
  [| "H_S8_SMOOTH"; "H_I32_S16"; "H_CLIP"; "H_CLIP_U8"; "H_CLIP_I32" |]

let range = function
  | "H_CLIP_U8" -> (0, 255)
  | "H_I32_S16" | "H_CLIP_I32" -> (-2147483648, 2147483647)
  | _ -> (-32768, 32767)

(* [dense_window name src w] is the window [w] of the tile [src] codes as
   the dense decoder reads it, its lossy pixels clamped, or [Error ()] if
   the decoder fails or a pixel is past the type. An empty window decodes
   nothing. *)
let dense_window name src w =
  let rows, cols =
    match shape name with [| r; c |] -> (r, c) | _ -> assert false
  in
  let smooth = H.get V.int "ZVAL2" (Fits.header (hdu name)) = Ok 1 in
  if Array.exists (fun (a, b) -> a = b) w then Ok [||]
  else
    match Hcompress_dense.decode ~smooth src (rows * cols) with
    | exception Failure _ -> Error ()
    | nx, ny, _, _ when nx <> rows || ny <> cols -> Error ()
    | _, _, v, scale ->
        let lo, hi = range name in
        let v =
          if scale > 1 then Array.map (fun x -> Int.max lo (Int.min hi x)) v
          else v
        in
        let (r0, r1), (c0, c1) = (w.(0), w.(1)) in
        let part =
          Array.concat
            (List.init (r1 - r0) (fun r ->
                 Array.sub v (((r0 + r) * cols) + c0) (c1 - c0)))
        in
        if Array.exists (fun x -> x < lo || x > hi) part then Error ()
        else Ok part

let dense =
  prop "a changed stream reads as the dense decoder reads it" ~count:300
    Gen.(
      bind
        (int_range 0 (Array.length one_tiles - 1))
        (fun k ->
          let name = one_tiles.(k) in
          map
            (fun (change, w) -> (name, change, w))
            (pair
               (triple (int_range 0 1_000_000) (int_range 0 255)
                  (int_range 0 3))
               (window_gen (shape name)))))
    (fun (name, (at, byte, change), w) ->
      let h = hdu name in
      let d = bytes_of h in
      let len = be32 d 0 in
      let at = at mod len in
      (* one draw in four leaves the stream whole *)
      if change > 0 then d.(heap + at) <- byte;
      cover "whole stream" (change = 0);
      cover "header" (change > 0 && at < 25);
      cover "bits" (change > 0 && at >= 25);
      let x =
        Fits.v (Fits.header h) (Nx.create Nx.uint8 [| Array.length d |] d)
      in
      let src = String.init len (fun i -> Char.chr d.(heap + i)) in
      let got =
        match I.raw ~window:w Nx.int32 x with
        | Ok t ->
            Ok
              (Array.map Int32.to_int
                 (Nx.to_array (Nx.reshape [| Nx.numel t |] t)))
        | Error _ -> Error ()
      in
      let want = dense_window name src w in
      cover "pixels read" (match want with Ok a -> a <> [||] | _ -> false);
      cover "an Error" (want = Error ());
      equal (result (array int) unit) want got)

(* Broken streams *)

let broken f =
  let h = hdu "H_S8_SMOOTH" in
  let d = bytes_of h in
  f d;
  let x = Fits.v (Fits.header h) (Nx.create Nx.uint8 [| Array.length d |] d) in
  match I.raw Nx.int16 x with
  | Ok _ -> fail "a broken stream read"
  | Error e -> contains ~sub:"tile 1" e

let truncated () =
  (* Every length short of the stream's ends it early or before its signs. *)
  let len = be32 (bytes_of (hdu "H_S8_SMOOTH")) 0 in
  for n = 0 to len - 1 do
    broken (fun d -> set32 d 0 n)
  done

let bad_magic () = broken (fun d -> d.(heap) <- 0xDE)

let wrong_size () =
  (* The stream states 24 × 30 pixels for a tile of 24 × 29. *)
  broken (fun d -> set32 d (heap + 6) 30)

let bad_plane_code () =
  (* The first quadrant's first plane starts at byte 25, its code the high
     nybble: 7 is neither 0 nor 15. *)
  broken (fun d -> d.(heap + 25) <- (d.(heap + 25) land 0x0F) lor 0x70)

let too_many_planes () = broken (fun d -> d.(heap + 22) <- 63)

let wide () =
  (* cfitsio's decoder packs its pixels into 32 bits. *)
  let h = hdu "H_I32" in
  let x = Fits.v (H.set V.int "ZBITPIX" 64 (Fits.header h)) (Fits.data h) in
  match I.raw Nx.int64 x with
  | Ok _ -> fail "a 64-bit HCOMPRESS image read"
  | Error e -> contains ~sub:"HCOMPRESS codes up to 32-bit integers" e

(* A huge tile *)

(* [limited flag] runs this executable with [flag] under 2 GiB of address
   space and 20 s of processor time, and is what it printed and its exit
   status. Only Linux, whose /proc/self/limits shows them, enforces the
   limits: macOS ignores [ulimit -v] and Windows has no ulimit. *)
let limited flag =
  if not (Sys.os_type = "Unix" && Sys.file_exists "/proc/self/limits") then
    skip ~reason:"ulimit -v is enforced on Linux only" ();
  let out = temp_file ~suffix:".txt" () in
  let cmd =
    Printf.sprintf "ulimit -v 2097152 && ulimit -t 20 && exec %s %s > %s 2>&1"
      (Filename.quote Sys.executable_name)
      flag (Filename.quote out)
  in
  let status = Sys.command cmd in
  (In_channel.with_open_text out In_channel.input_all, status)

(* A constant tile of 46341 × 46340 pixels, past 2^31, in 26 bytes: 16
   levels, so a pixel is the first coefficient over 2^17. *)
let huge_value = 1234
let huge_flag = "--huge-window"

let huge_child () =
  let x =
    one_tile [| 46341; 46340 |]
      (stream ~nx:46341 ~ny:46340 ~scale:0 ~h0:(huge_value lsl 17))
  in
  match I.raw ~window:[| (46000, 46001); (123, 124) |] Nx.int16 x with
  | Ok t -> print_string (String.concat " " (Array.to_list (texts t)))
  | Error e -> print_string e

let huge_window () =
  (* Decoding the whole tile would take 17 GB. *)
  equal (pair string int) (string_of_int huge_value, 0) (limited huge_flag)

(* A dithered tile *)

(* A constant quantized tile of (2^31 - 1)^2 pixels, near 2^62, in one row
   of H_Q's table, ZSCALE, ZZERO and ZDITHER0 as below: 31 levels, so its
   stored integer is the first coefficient over 2^32. *)
let side = (1 lsl 31) - 1
let stored = 7
let zscale = 0.5
let zzero = 100.25

let dithered () =
  let s = stream ~nx:side ~ny:side ~scale:0 ~h0:(stored lsl 32) in
  let n = String.length s in
  let row = 32 in
  let h =
    Fits.header (hdu "H_Q")
    |> H.set V.int "NAXIS2" 1 |> H.set V.int "PCOUNT" n
    |> H.set V.string "TFORM1" (Printf.sprintf "1PB(%d)" n)
    |> H.set V.int "ZNAXIS1" side |> H.set V.int "ZNAXIS2" side
    |> H.set V.int "ZTILE1" side |> H.set V.int "ZTILE2" side
  in
  let b = Buffer.create (row + n) in
  Buffer.add_int32_be b (Int32.of_int n);
  Buffer.add_int32_be b 0l;
  Buffer.add_int64_be b 0L;
  Buffer.add_int64_be b (Int64.bits_of_float zscale);
  Buffer.add_int64_be b (Int64.bits_of_float zzero);
  Buffer.add_string b s;
  let d = Buffer.contents b in
  Fits.v h
    (Nx.init Nx.uint8 [| String.length d |] (fun i -> Char.code d.[i.(0)]))

(* [deviate row k] is the dither deviate of pixel [k] of the tile whose row
   is [row], from FITS 4.0 Appendix I's generator: seed [(row - 1) mod
   10000] gives the sequence's first index, [500 r] in float32, and each
   seed's deviates run to the last before the next seed's. Whole cycles of
   the 10000 seeds are left out, then the seeds walked one by one. *)
let deviate row k =
  let f32 x = Int32.float_of_bits (Int32.bits_of_float x) in
  let seed = ref 1. in
  let r =
    Array.init 10000 (fun _ ->
        let t = 16807. *. !seed in
        let m = 2147483647. in
        seed := t -. (m *. Float.of_int (int_of_float (t /. m)));
        f32 (!seed /. 2147483647.))
  in
  let first i = int_of_float (f32 (r.(i) *. 500.)) in
  let cycle = ref 0 in
  for i = 0 to 9999 do
    cycle := !cycle + 10000 - first i
  done;
  let seed = ref ((row - 1) mod 10000) and k = ref k in
  let next () = seed := (!seed + 1) mod 10000 in
  if !k >= 10000 - first !seed then begin
    k := !k - (10000 - first !seed);
    next ();
    k := !k mod !cycle
  end;
  while !k >= 10000 - first !seed do
    k := !k - (10000 - first !seed);
    next ()
  done;
  r.(first !seed + !k)

let dithered_flag = "--dithered-window"

let dithered_child () =
  let w = [| (side - 1, side); (side - 1, side) |] in
  match I.values ~window:w Nx.float32 (dithered ()) with
  | Ok t -> print_string (Printf.sprintf "%h" (Nx.item [ 0; 0 ] t))
  | Error e -> print_string e

let dithered_window () =
  (* Its last pixel, past 4.6e18 others: the sequence skips to it. *)
  let k = (side * side) - 1 in
  let x = ((Float.of_int stored -. deviate 3 k) +. 0.5) *. zscale +. zzero in
  let want = Int32.float_of_bits (Int32.bits_of_float x) in
  equal (pair string int) (Printf.sprintf "%h" want, 0) (limited dithered_flag)

let () =
  if Array.exists (( = ) huge_flag) Sys.argv then huge_child ()
  else if Array.exists (( = ) dithered_flag) Sys.argv then dithered_child ()
  else
    exit
    @@ run "Fits HCOMPRESS tiles"
         [
           decoded;
           test "descriptions" described;
           windows;
           one_pixel;
           dense;
           test "a window of a constant tile past 2^31 pixels" huge_window;
           test "the last pixel of a dithered tile of 2^62 pixels"
             dithered_window;
           group "broken streams"
             [
               test "a truncated stream is an Error" truncated;
               test "a stream without the magic is an Error" bad_magic;
               test "a stream of another size is an Error" wrong_size;
               test "a bit plane code past 0 and 15 is an Error" bad_plane_code;
               test "more than 62 bit planes is an Error" too_many_planes;
               test "a 64-bit image is an Error" wide;
             ];
         ]
