(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* PLIO_1 tiles: astropy's files of masks (gen/fixtures.py) read as their
   source pixels; line lists of every instruction and both headers read as
   the masks they code; broken lists are an Error. *)

open Windtrap
open Ymir_fits
module H = Fits.Header
module V = Fits.Value
module I = Fits.Image

let hdus = lazy (require_ok (Fits.read "golden/plio.fits"))
let hdu name = require_ok (Fits.get name (Lazy.force hdus))

let golden =
  lazy
    (In_channel.with_open_text "golden/plio.values" In_channel.input_lines
    |> List.map (fun l ->
        match String.split_on_char ' ' l with
        | name :: values ->
            (name, Array.of_list (List.map int_of_string values))
        | [] -> assert false))

let expected name = List.assoc name (Lazy.force golden)

let ints (type a b) (dtype : (a, b) Nx.dtype) (t : (a, b, Nx.host) Nx.t) :
    int array =
  let a = Nx.to_array (Nx.reshape [| Nx.numel t |] t) in
  match dtype with
  | Uint8 -> a
  | Int16 -> a
  | Uint16 -> a
  | Int32 -> Array.map Int32.to_int a
  | _ -> failwith "ints: uint8, int16, uint16 or int32"

let raw ?window (type a b) (dtype : (a, b) Nx.dtype) h =
  Result.map (ints dtype) (I.raw ?window dtype h)

let check (type a b) name (dtype : (a, b) Nx.dtype) =
  equal (result (array int) string) (Ok (expected name)) (raw dtype (hdu name))

let decoded =
  cases "tiles read as astropy's source pixels" ~id:fst
    [
      ("PLIO_I32", fun n -> check n Nx.int32);
      ("PLIO_I16", fun n -> check n Nx.int16);
      ("PLIO_U8", fun n -> check n Nx.uint8);
      ("PLIO_HOLES", fun n -> check n Nx.int32);
      ("PLIO_ZERO", fun n -> check n Nx.int16);
      ("PLIO_ROWS", fun n -> check n Nx.int32);
    ]
    (fun (name, f) -> f name)

let described () =
  let pp name = Format.asprintf "%a" Fits.pp (hdu name) in
  expect
    (String.concat "\n" (List.map pp [ "PLIO_I32"; "PLIO_U8"; "PLIO_ROWS" ]))
  @@ __POS_OF__
       {|
    BINTABLE PLIO_I32 1: int32 [10; 13], PLIO_1 tiles [3; 4]
    BINTABLE PLIO_U8 1: uint8 [10; 13], PLIO_1 tiles [3; 4]
    BINTABLE PLIO_ROWS 1: int32 [3; 9000], PLIO_1 tiles [1; 9000]
    |}

let names =
  [|
    "PLIO_I32"; "PLIO_I16"; "PLIO_U8"; "PLIO_HOLES"; "PLIO_ZERO"; "PLIO_ROWS";
  |]

let shape name = I.shape (require_ok (I.of_hdu (hdu name)))

let window_gen shape =
  Gen.(
    let axis n =
      map
        (fun (a, b) -> (Int.min a b, Int.max a b))
        (pair (int_range 0 n) (int_range 0 n))
    in
    let pixel =
      map
        (fun i -> Array.map (fun n -> (i mod n, (i mod n) + 1)) shape)
        (int_range 0 100_000)
    in
    let some =
      match shape with
      | [| r; c |] -> map (fun (x, y) -> [| x; y |]) (pair (axis r) (axis c))
      | _ -> assert false
    in
    frequency
      [
        (1, constant (Array.map (fun n -> (0, n)) shape));
        (2, pixel);
        (6, some);
      ])

let windows =
  prop "a window reads as the slice of the whole" ~count:300
    Gen.(
      bind
        (int_range 0 (Array.length names - 1))
        (fun k ->
          map (fun w -> (names.(k), w)) (window_gen (shape names.(k)))))
    (fun (name, w) ->
      Array.iter (fun n -> cover n (n = name)) names;
      cover "one pixel" (Array.for_all (fun (a, b) -> b = a + 1) w);
      cover "empty" (Array.exists (fun (a, b) -> a = b) w);
      let h = hdu name in
      let whole = require_ok (I.values Nx.float64 h) in
      let part = require_ok (I.values ~window:w Nx.float64 h) in
      let slice =
        Nx.slice (Array.to_list (Array.map (fun (a, b) -> Nx.R (a, b)) w)) whole
      in
      equal (array int) (Nx.shape slice) (Nx.shape part);
      equal (array float_exact) (Nx.to_array slice) (Nx.to_array part))

(* Line lists *)

(* IRAF's masks reach 27 bits (PL_MAXDEPTH). *)
let max_value = (1 lsl 27) - 1

(* Instruction words, PLIO.hlp's opcodes over 12 bits of data. *)
let zn d = d
let sh d = (1 lsl 12) lor d
let ih d = (2 lsl 12) lor d
let dh d = (3 lsl 12) lor d
let hn d = (4 lsl 12) lor d
let pn d = (5 lsl 12) lor d
let is d = (6 lsl 12) lor d
let ds d = (7 lsl 12) lor d

(* A list with the 7-word header of IRAF's current version, or the 3-word
   one of its first. *)
let with_header ?(old = false) body =
  let n = List.length body in
  if old then [ 0; 0; n + 3 ] @ body
  else [ 0; 7; -100; (n + 7) land 0x7FFF; (n + 7) lsr 15; 0; 0 ] @ body

(* [image ?keys ~shape ~tile rows words] is an int32 image of [shape] in
   PLIO_1 tiles of [tile], both in tensor axis order, whose heap is [words]
   and whose rows' descriptors are [rows], pairs of a count of words and a
   byte offset. [keys] changes its header. *)
let image ?(keys = Fun.id) ~shape ~tile rows words =
  let rank = Array.length shape in
  let ones = Array.make rank 1 in
  let x = I.hdu ~tiles:ones H.empty (Nx.zeros Nx.int32 ones) in
  let k = List.length words in
  let longest = List.fold_left (fun acc (c, _) -> Int.max acc c) 0 rows in
  let h =
    Fits.header x
    |> H.set V.string "ZCMPTYPE" "PLIO_1"
    |> H.remove "ZNAME1" |> H.remove "ZVAL1" |> H.remove "ZNAME2"
    |> H.remove "ZVAL2"
    |> H.set V.string "TFORM1" (Printf.sprintf "1PI(%d)" longest)
    |> H.set V.int "NAXIS2" (List.length rows)
    |> H.set V.int "PCOUNT" (2 * k)
  in
  let h = ref h in
  for i = 1 to rank do
    h :=
      !h
      |> H.set V.int (Printf.sprintf "ZNAXIS%d" i) shape.(rank - i)
      |> H.set V.int (Printf.sprintf "ZTILE%d" i) tile.(rank - i)
  done;
  let b = Buffer.create ((8 * List.length rows) + (2 * k)) in
  List.iter
    (fun (c, o) ->
      Buffer.add_int32_be b (Int32.of_int c);
      Buffer.add_int32_be b (Int32.of_int o))
    rows;
  List.iter (fun w -> Buffer.add_uint16_be b (w land 0xFFFF)) words;
  let s = Buffer.contents b in
  Fits.v (keys !h)
    (Nx.init Nx.uint8 [| String.length s |] (fun i -> Char.code s.[i.(0)]))

(* [one_tile ?keys n words] is an image of [n] int32 pixels in one tile
   whose list is [words]. *)
let one_tile ?keys n words =
  image ?keys ~shape:[| n |] ~tile:[| n |] [ (List.length words, 0) ] words

(* A test encoder: each run of the mask coded by one of the instruction
   sequences that regenerate it, as [pick k] chooses among [k], with
   instructions of count 0 between runs. *)
let encode pick runs =
  let hi = ref 1 and out = ref [] in
  let emit w = out := w :: !out in
  let chunks op len =
    let left = ref len in
    while !left > 0 do
      let c = Int.min !left (1 + pick 4095) in
      emit (op c);
      left := !left - c
    done
  in
  let nothing = [| zn 0; hn 0; pn 0; ih 0; dh 0 |] in
  List.iter
    (fun (v, len) ->
      if pick 3 = 0 then emit nothing.(pick (Array.length nothing));
      if v = 0 then chunks zn len
      else begin
        let dv = v - !hi in
        hi := v;
        let small = dv <> 0 && abs dv <= 4095 in
        if small && len = 1 && pick 2 = 0 then
          emit (if dv > 0 then is dv else ds (-dv))
        else begin
          (if dv = 0 then ()
           else if small && pick 2 = 0 then
             emit (if dv > 0 then ih dv else dh (-dv))
           else begin
             emit (sh (v land 0xFFF));
             emit (v lsr 12)
           end);
          if len > 1 then chunks hn len
          else
            match pick 4 with
            | 0 -> emit (pn 1)
            | 1 -> emit (is 0)
            | 2 -> emit (ds 0)
            | _ -> emit (hn 1)
        end
      end)
    runs;
  List.rev !out

let level =
  Gen.(
    frequency
      [
        (3, constant 0);
        (2, int_range 1 10);
        (2, int_range 4090 4100);
        (2, int_range ((1 lsl 24) - 4096) ((1 lsl 24) + 4096));
        (1, constant max_value);
        (1, int_range 0 max_value);
      ])

let length =
  Gen.(
    frequency
      [ (6, int_range 1 6); (2, int_range 7 300); (1, int_range 4090 9000) ])

let round_trip =
  prop "a line list reads as the mask it codes"
    Gen.(
      quad
        (list ~size:(int_range 1 12) (pair level length))
        (list ~size:(int_range 1 30) (int_range 0 1_000_000))
        bool bool)
    (fun (runs, picks, old, short) ->
      let picks = Array.of_list picks and next = ref 0 in
      let pick k =
        let p = picks.(!next mod Array.length picks) in
        incr next;
        p mod k
      in
      (* A short list leaves out its trailing zeros, as IRAF's pl_r2l
         does. *)
      let rec coded = function
        | [] -> []
        | runs -> (
            match List.rev runs with
            | (0, _) :: rest -> coded (List.rev rest)
            | _ -> runs)
      in
      let body = encode pick (if short then coded runs else runs) in
      let old = old && List.length body + 3 <= 0x7FFF in
      let n = List.fold_left (fun acc (_, l) -> acc + l) 0 runs in
      let mask =
        Array.concat (List.map (fun (v, l) -> Array.make l v) runs)
      in
      cover "old header" old;
      cover "a run past 4095" (List.exists (fun (_, l) -> l > 4095) runs);
      cover "past 2^24" (List.exists (fun (v, _) -> v > 1 lsl 24) runs);
      cover "27 bits" (List.exists (fun (v, _) -> v = max_value) runs);
      cover "a step of one pixel"
        (List.exists (fun (v, l) -> v > 0 && l = 1) runs);
      cover "trailing zeros left out"
        (short && match List.rev runs with (0, _) :: _ -> true | _ -> false);
      (* Instructions past the last pixel, a word no instruction has
         included, are ignored. *)
      let body = if short then body else body @ [ hn 5; 0x8000 ] in
      equal (result (array int) string) (Ok mask)
        (raw Nx.int32 (one_tile n (with_header ~old body))))

let instructions =
  cases "lists read as PLIO.hlp's instructions say" ~id:(fun (id, _, _) -> id)
    [
      ( "ZN, HN and PN of 0 output nothing",
        [ zn 0; hn 0; pn 0; hn 2 ],
        [| 1; 1 |] );
      ("IS 0 and DS 0 output the high value", [ is 0; ih 2; ds 0 ], [| 1; 3 |]);
      ("PN outputs zeros, then the high value", [ pn 3 ], [| 0; 0; 1 |]);
      ( "a PN past the end outputs its zeros",
        [ sh 1; 0x1000; pn 3 ],
        [| 0; 0 |] );
      ("SH sets the high value", [ sh 5; 2; hn 1 ], [| (2 * 4096) + 5 |]);
      ("2^24", [ sh 0; 0x1000; hn 1 ], [| 1 lsl 24 |]);
      ("past 2^24", [ zn 1; sh 1; 0x1000; hn 1 ], [| 0; (1 lsl 24) + 1 |]);
      ("27 bits", [ sh 0xFFF; 0x7FFF; hn 1 ], [| max_value |]);
      ("below 0", [ hn 1; dh 2; hn 1 ], [| 1; -1 |]);
      ("a negative SH", [ sh 0; 0xFFFF; hn 1 ], [| -4096 |]);
      ("zeros past the end", [ hn 2 ], [| 1; 1; 0; 0 |]);
      ( "IRAF's line of one range to xe - 1",
        [ hn 9 ],
        Array.init 10 (fun i -> if i < 9 then 1 else 0) );
      ("an empty list", [], [| 0; 0; 0; 0 |]);
    ]
    (fun (_, body, want) ->
      equal (result (array int) string) (Ok want)
        (raw Nx.int32 (one_tile (Array.length want) (with_header body))))

(* Shared lists *)

let zeros () =
  (* 32760 zeros in 15 words, 30 bytes, more pixels per byte than gzip
     can hold. *)
  let n = 8 * 4095 in
  equal (result (array int) string) (Ok (Array.make n 0))
    (raw Nx.int32 (one_tile n (with_header (List.init 8 (fun _ -> zn 4095)))))

let shared () =
  (* Rows 1, 4 and 5 point at the empty line's list, 2 and 3 at another:
     each list is decoded once, within the heap's bytes. *)
  let empty = with_header [ zn 4 ] and ones = with_header [ hn 4 ] in
  let e = (List.length empty, 0)
  and o = (List.length ones, 2 * List.length empty) in
  let x =
    image ~shape:[| 5; 4 |] ~tile:[| 1; 4 |] [ e; o; o; e; e ] (empty @ ones)
  in
  equal (result (array int) string)
    (Ok (Array.init 20 (fun i -> if i >= 4 && i < 12 then 1 else 0)))
    (raw Nx.int32 x);
  (* Tiles of 3 and 1 pixels share one list, read once at each size: the
     heap holds the bytes of two. *)
  let l = with_header [ hn 2 ] in
  let d = (List.length l, 0) in
  equal (result (array int) string)
    (Ok [| 1; 1; 0; 1; 1; 1; 0; 1 |])
    (raw Nx.int32
       (image ~shape:[| 2; 4 |] ~tile:[| 1; 3 |] [ d; d; d; d ] (l @ l)))

let in_turn () =
  (* Three lists in turn, each decoded again, pass the heap's bytes. *)
  let a = with_header [ hn 1 ] and b = with_header [ zn 1 ] in
  let c = with_header [ ih 1; hn 1 ] in
  let at = [ (8, 0); (8, 16); (9, 32) ] in
  let x = image ~shape:[| 6; 1 |] ~tile:[| 1; 1 |] (at @ at) (a @ b @ c) in
  match raw Nx.int32 x with
  | Ok v -> failf "%d pixels read" (Array.length v)
  | Error e -> contains ~sub:"more bytes than the heap holds" e

let iraf_mask () =
  (* An empty 4096 x 4096 mask as IRAF writes it: every line points at one
     list of 9 words, an 18-byte heap. *)
  let empty = with_header [ zn 4094; zn 2 ] in
  let rows = List.init 4096 (fun _ -> (List.length empty, 0)) in
  let x = image ~shape:[| 4096; 4096 |] ~tile:[| 1; 4096 |] rows empty in
  equal int 18 (List.length empty * 2);
  let t = require_ok (I.raw Nx.int32 x) in
  equal (array int) [| 4096; 4096 |] (Nx.shape t);
  equal int32 0l (Nx.item [] (Nx.max t));
  equal int32 0l (Nx.item [] (Nx.min t))

(* A tile of 2^31 pixels whose 8-word list codes 9 ones and leaves out the
   rest, its zeros. *)
let huge_flag = "--huge-window"
let huge_size = 1 lsl 31

let huge_child () =
  let l = with_header [ hn 9 ] in
  let x =
    image ~shape:[| 1; huge_size |] ~tile:[| 1; huge_size |]
      [ (List.length l, 0) ]
      l
  in
  match raw Nx.int32 ~window:[| (0, 1); (5, 10) |] x with
  | Ok v ->
      print_string
        (String.concat " " (Array.to_list (Array.map string_of_int v)))
  | Error e -> print_string e

let huge_window () =
  (* Decoding the whole tile would take 17 GB; the child may map 2 GiB and
     run 20 s. Only Linux, whose /proc/self/limits shows them, enforces
     those limits: macOS ignores [ulimit -v] and Windows has no ulimit. *)
  if not (Sys.os_type = "Unix" && Sys.file_exists "/proc/self/limits") then
    skip ~reason:"ulimit -v is enforced on Linux only" ();
  let out = temp_file ~suffix:".txt" () in
  let cmd =
    Printf.sprintf "ulimit -v 2097152 && ulimit -t 20 && exec %s %s > %s 2>&1"
      (Filename.quote Sys.executable_name)
      huge_flag (Filename.quote out)
  in
  let status = Sys.command cmd in
  equal string "1 1 1 1 0" (In_channel.with_open_text out In_channel.input_all);
  equal int 0 status

let unsigned () =
  (* A 16-bit image of BZERO 32768 and BSCALE 1 stores its unsigned values
     in the list; under another BSCALE, the list holds the stored ones. *)
  let keys bscale h =
    h |> H.set V.int "ZBITPIX" 16 |> H.set V.int "BZERO" 32768
    |> H.set V.int "BSCALE" bscale
  in
  let words = with_header [ zn 1; sh 0xFFF; 15; hn 1; dh 4095; is 1 ] in
  equal (result (array int) string)
    (Ok [| 0; 65535; 61441 |])
    (raw Nx.uint16 (one_tile ~keys:(keys 1) 3 words));
  match raw Nx.int32 (one_tile ~keys:(keys 2) 3 words) with
  | Ok v -> failf "65535 read as an int16 of %d" v.(1)
  | Error e -> contains ~sub:"65535 is outside BITPIX's type" e

(* Broken lists *)

let broken =
  let error ?keys ?(floats = false) n body sub =
    let h = one_tile ?keys n body in
    let read =
      if floats then Result.map Nx.numel (I.raw Nx.float32 h)
      else Result.map Array.length (raw Nx.int32 h)
    in
    match read with
    | Ok k -> failf "a broken list read as %d values" k
    | Error e ->
        contains ~sub:"tile 1" e;
        contains ~sub e
  in
  cases "a broken list is an Error" ~id:(fun (id, _) -> id)
    [
      ("short header", fun () -> error 4 [ 0; 0 ] "short of a header");
      ( "header under 7 words",
        fun () ->
          error 4 [ 0; 3; -100; 8; 0; 0; 0; hn 4 ] "header is 3 words" );
      ( "length past the stored words",
        fun () ->
          let w = with_header [ hn 4 ] in
          error 4
            (List.mapi (fun i x -> if i = 3 then 12 else x) w)
            "the line list is 12 words, 8 are stored" );
      ( "old length past the stored words",
        fun () -> error 4 [ 0; 0; 5; hn 4 ] "5 words, 4 are stored" );
      ( "SH without its second word",
        fun () ->
          error 4 (with_header [ hn 1; sh 3 ]) "lacks its second word" );
      ( "sign bit",
        fun () ->
          error 4
            (with_header [ hn 1; 0x8003 ])
            "word 9 of the line list has its sign bit set" );
      ( "value past ZBITPIX",
        fun () ->
          error
            ~keys:(H.set V.int "ZBITPIX" 16)
            1
            (with_header [ sh 0; 8; hn 1 ])
            "32768 is outside BITPIX's type" );
      ( "unquantized floats",
        fun () ->
          error ~floats:true
            ~keys:(H.set V.int "ZBITPIX" (-32))
            1 (with_header [ hn 1 ]) "PLIO codes integers" );
    ]
    (fun (_, f) -> f ())

let () =
  if Array.exists (( = ) huge_flag) Sys.argv then huge_child ()
  else
    exit
    @@ run "Fits PLIO tiles"
         [
           decoded;
           test "descriptions" described;
           windows;
           round_trip;
           instructions;
           test "zeros past gzip's bound" zeros;
           test "rows share lists" shared;
           test "three lists in turn pass the heap" in_turn;
           test "an empty 4096 x 4096 IRAF mask" iraf_mask;
           test "a window of a tile of 2^31 pixels" huge_window;
           test "unsigned 16-bit lists" unsigned;
           broken;
         ]
