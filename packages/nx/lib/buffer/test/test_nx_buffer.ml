(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_dtype
open Nx_buffer
open Windtrap

let invalid_argument = function Invalid_argument _ -> true | _ -> false

(* Test creation of different buffer types *)
let test_create_bfloat16 () =
  let buf = create bfloat16 10 in
  equal ~msg:"bfloat16 buffer size" int 10 (length buf);
  set buf 0 1.0;
  set buf 5 2.5;
  equal ~msg:"bfloat16 get" (float 0.1) 1.0 (get buf 0);
  equal ~msg:"bfloat16 get" (float 0.1) 2.5 (get buf 5)

let test_create_bool () =
  let buf = create Nx_dtype.bool 8 in
  equal ~msg:"bool buffer size" int 8 (length buf);
  set buf 0 true;
  set buf 1 false;
  set buf 7 true;
  equal ~msg:"bool get" bool true (get buf 0);
  equal ~msg:"bool get" bool false (get buf 1);
  equal ~msg:"bool get" bool true (get buf 7)

let test_create_int4 () =
  let buf = create int4 16 in
  equal ~msg:"int4 buffer size" int 16 (length buf);
  set buf 0 (-8);
  set buf 1 7;
  set buf 2 0;
  equal ~msg:"int4 get" int (-8) (get buf 0);
  equal ~msg:"int4 get" int 7 (get buf 1);
  equal ~msg:"int4 get" int 0 (get buf 2)

let test_create_uint4 () =
  let buf = create uint4 16 in
  equal ~msg:"uint4 buffer size" int 16 (length buf);
  set buf 0 0;
  set buf 1 15;
  set buf 2 8;
  equal ~msg:"uint4 get" int 0 (get buf 0);
  equal ~msg:"uint4 get" int 15 (get buf 1);
  equal ~msg:"uint4 get" int 8 (get buf 2)

let test_create_float8_e4m3 () =
  let buf = create float8_e4m3 10 in
  equal ~msg:"float8_e4m3 buffer size" int 10 (length buf);
  set buf 0 0.0;
  set buf 1 1.0;
  set buf 2 (-1.5);
  equal ~msg:"float8_e4m3 get" (float 0.1) 0.0 (get buf 0);
  equal ~msg:"float8_e4m3 get" (float 0.1) 1.0 (get buf 1);
  equal ~msg:"float8_e4m3 get" (float 0.1) (-1.5) (get buf 2)

let test_create_float8_e5m2 () =
  let buf = create float8_e5m2 10 in
  equal ~msg:"float8_e5m2 buffer size" int 10 (length buf);
  set buf 0 0.0;
  set buf 1 2.0;
  set buf 2 (-0.5);
  equal ~msg:"float8_e5m2 get" (float 0.1) 0.0 (get buf 0);
  equal ~msg:"float8_e5m2 get" (float 0.1) 2.0 (get buf 1);
  equal ~msg:"float8_e5m2 get" (float 0.1) (-0.5) (get buf 2)

(* Test genarray creation *)
let test_genarray_creation () =
  let dims = [| 2; 3; 4 |] in
  let ga_bf16 = genarray_create bfloat16 Bigarray.c_layout dims in
  let ga_bool = genarray_create Nx_dtype.bool Bigarray.c_layout dims in
  let ga_fp8 = genarray_create float8_e4m3 Bigarray.c_layout dims in
  equal ~msg:"Genarray bfloat16 dims" int 3
    (Array.length (Bigarray.Genarray.dims ga_bf16));
  equal ~msg:"Genarray bool dims" int 3
    (Array.length (Bigarray.Genarray.dims ga_bool));
  equal ~msg:"Genarray float8 dims" int 3
    (Array.length (Bigarray.Genarray.dims ga_fp8));
  equal ~msg:"Genarray dim 0" int 2 (Bigarray.Genarray.nth_dim ga_bf16 0);
  equal ~msg:"Genarray dim 1" int 3 (Bigarray.Genarray.nth_dim ga_bf16 1);
  equal ~msg:"Genarray dim 2" int 4 (Bigarray.Genarray.nth_dim ga_bf16 2)

(* The C stub maps storage flag bits back to GADT constructor indices by
   declaration order; a mismatch shows up as [dtype] returning the wrong
   constructor. Check the round-trip for every constructor. *)
let test_dtype_roundtrip () =
  let check : type a b. (a, b) Nx_dtype.t -> unit =
   fun k ->
    equal
      ~msg:(Nx_dtype.to_string k ^ " dtype round-trip")
      bool true
      (dtype (create k 2) = k)
  in
  check Nx_dtype.float16;
  check Nx_dtype.float32;
  check Nx_dtype.float64;
  check Nx_dtype.bfloat16;
  check Nx_dtype.float8_e4m3;
  check Nx_dtype.float8_e5m2;
  check Nx_dtype.int4;
  check Nx_dtype.uint4;
  check Nx_dtype.int8;
  check Nx_dtype.uint8;
  check Nx_dtype.int16;
  check Nx_dtype.uint16;
  check Nx_dtype.int32;
  check Nx_dtype.uint32;
  check Nx_dtype.int64;
  check Nx_dtype.uint64;
  check Nx_dtype.complex64;
  check Nx_dtype.complex128;
  check Nx_dtype.bool

(* Test itemsize *)
let test_itemsize () =
  equal ~msg:"bfloat16 size" int 2 (Nx_dtype.itemsize bfloat16);
  equal ~msg:"bool size" int 1 (Nx_dtype.itemsize Nx_dtype.bool);
  equal ~msg:"int4 size" int 1 (Nx_dtype.itemsize int4);
  equal ~msg:"uint4 size" int 1 (Nx_dtype.itemsize uint4);
  equal ~msg:"float8_e4m3 size" int 1 (Nx_dtype.itemsize float8_e4m3);
  equal ~msg:"float8_e5m2 size" int 1 (Nx_dtype.itemsize float8_e5m2);
  equal ~msg:"uint32 size" int 4 (Nx_dtype.itemsize uint32);
  equal ~msg:"uint64 size" int 8 (Nx_dtype.itemsize uint64);
  equal ~msg:"float32 size" int 4 (Nx_dtype.itemsize float32);
  equal ~msg:"float64 size" int 8 (Nx_dtype.itemsize float64);
  equal ~msg:"int32 size" int 4 (Nx_dtype.itemsize Nx_dtype.int32)

(* Test blit *)
let test_blit () =
  let src = create float32 4 in
  let dst = create float32 4 in
  set src 0 1.0;
  set src 1 2.0;
  set src 2 3.0;
  set src 3 4.0;
  blit ~src ~dst;
  equal ~msg:"blit[0]" (float 1e-6) 1.0 (get dst 0);
  equal ~msg:"blit[3]" (float 1e-6) 4.0 (get dst 3)

(* Test fill *)
let test_fill () =
  let buf = create float32 4 in
  fill buf 7.0;
  equal ~msg:"fill[0]" (float 1e-6) 7.0 (get buf 0);
  equal ~msg:"fill[3]" (float 1e-6) 7.0 (get buf 3)

(* Conversion semantics: store a float through the packed representation and
   read it back. These vectors pin round-to-nearest-even, saturation, NaN and
   subnormal handling; the JavaScript stubs mirror the same algorithms. *)
let store_get k v =
  let buf = create k 1 in
  set buf 0 v;
  get buf 0

let store_code k v =
  let buf = create k 1 in
  set buf 0 v;
  get (reinterpret uint8 buf) 0

let test_bfloat16_semantics () =
  let rt = store_get bfloat16 in
  equal ~msg:"bf16 1.0" float_exact 1.0 (rt 1.0);
  (* 1 + 2^-8 is halfway between 1 and 1 + 2^-7: ties to even, down. *)
  equal ~msg:"bf16 tie to even down" float_exact 1.0 (rt 1.00390625);
  (* 1 + 2^-7 + 2^-8 is halfway with an odd mantissa: ties to even, up. *)
  equal ~msg:"bf16 tie to even up" float_exact 1.015625 (rt 1.01171875);
  equal ~msg:"bf16 inf" float_exact Float.infinity (rt Float.infinity);
  (* Max finite float32 is beyond the max finite bfloat16: rounds to inf. *)
  equal ~msg:"bf16 overflow" float_exact Float.infinity (rt 3.4028234e38);
  equal ~msg:"bf16 nan" bool true (Float.is_nan (rt Float.nan))

let test_float8_e4m3_semantics () =
  let rt = store_get float8_e4m3 in
  equal ~msg:"e4m3 1.0" float_exact 1.0 (rt 1.0);
  (* Exponent 15 is a normal binade in e4m3fn: 256..448 are representable. *)
  equal ~msg:"e4m3 256" float_exact 256.0 (rt 256.0);
  equal ~msg:"e4m3 max finite" float_exact 448.0 (rt 448.0);
  equal ~msg:"e4m3 300 rounds to 288" float_exact 288.0 (rt 300.0);
  (* No infinities: overflow and infinities convert to NaN, matching the
     ml_dtypes and PyTorch e4m3fn casts. 464 is the round-to-nearest boundary
     past 448 and ties to the even finite value. *)
  equal ~msg:"e4m3 460 rounds back to 448" float_exact 448.0 (rt 460.0);
  equal ~msg:"e4m3 tie at 464 stays finite" float_exact 448.0 (rt 464.0);
  equal ~msg:"e4m3 overflow is nan" bool true (Float.is_nan (rt 465.0));
  equal ~msg:"e4m3 512 is nan" bool true (Float.is_nan (rt 512.0));
  equal ~msg:"e4m3 inf is nan" bool true (Float.is_nan (rt Float.infinity));
  equal ~msg:"e4m3 -inf is nan" bool true (Float.is_nan (rt Float.neg_infinity));
  (* Subnormals: min subnormal is 2^-9. *)
  equal ~msg:"e4m3 min subnormal" float_exact 0x1p-9 (rt 0x1p-9);
  equal ~msg:"e4m3 subnormal rounds up" float_exact 0x1p-9 (rt 0x1.8p-10);
  (* Half the min subnormal ties to even: zero. *)
  equal ~msg:"e4m3 underflow" float_exact 0.0 (rt 0x1p-10);
  equal ~msg:"e4m3 nan" bool true (Float.is_nan (rt Float.nan));
  equal ~msg:"e4m3 -nan keeps its sign" int 0xFF
    (store_code float8_e4m3 (-.Float.nan))

let test_float8_e5m2_semantics () =
  let rt = store_get float8_e5m2 in
  equal ~msg:"e5m2 1.0" float_exact 1.0 (rt 1.0);
  equal ~msg:"e5m2 max finite" float_exact 57344.0 (rt 57344.0);
  equal ~msg:"e5m2 below tie stays finite" float_exact 57344.0 (rt 61439.0);
  (* 61440 is halfway between 57344 and 65536: ties to even, to inf. *)
  equal ~msg:"e5m2 tie overflows to inf" float_exact Float.infinity (rt 61440.0);
  equal ~msg:"e5m2 inf" float_exact Float.infinity (rt Float.infinity);
  (* Subnormals: min subnormal is 2^-16. *)
  equal ~msg:"e5m2 min subnormal" float_exact 0x1p-16 (rt 0x1p-16);
  (* 1.5 * 2^-16 is halfway between 2^-16 and 2^-15: ties to even, up. *)
  equal ~msg:"e5m2 subnormal tie to even" float_exact 0x1p-15 (rt 0x1.8p-16);
  (* Half the min subnormal ties to even: zero. *)
  equal ~msg:"e5m2 underflow" float_exact 0.0 (rt 0x1p-17);
  equal ~msg:"e5m2 nan" bool true (Float.is_nan (rt Float.nan));
  equal ~msg:"e5m2 -nan keeps its sign" int 0xFF
    (store_code float8_e5m2 (-.Float.nan))

(* A float64 rounds once. Each value lies within 2^-40 of a tie at the stored
   precision, close enough that rounding to float32 first would land on the tie
   and then round it to even, the wrong way. *)
let test_float64_rounds_once () =
  let above = Float.succ and below = Float.pred in
  let check name k cases =
    List.iter
      (fun (x, want) ->
        equal
          ~msg:(Printf.sprintf "%s %h" name x)
          float_exact want (store_get k x))
      cases
  in
  check "bf16" bfloat16
    [
      (above 1.00390625, 1.0078125);
      (below 1.01171875, 1.0078125);
      (1e39, Float.infinity);
      (-1e-50, -0.0);
    ];
  check "e4m3" float8_e4m3
    [
      (above 336.0, 352.0);
      (below 368.0, 352.0);
      (below 0x1.8p-9, 0x1p-9);
      (-1e-50, -0.0);
    ];
  check "e5m2" float8_e5m2
    [
      (above 288.0, 320.0);
      (below 352.0, 320.0);
      (below 0x1.8p-16, 0x1p-16);
      (1e39, Float.infinity);
    ];
  equal ~msg:"e4m3 1e39 is nan" bool true
    (Float.is_nan (store_get float8_e4m3 1e39))

let test_int4_clamping () =
  equal ~msg:"int4 clamps high" int 7 (store_get int4 9);
  equal ~msg:"int4 clamps low" int (-8) (store_get int4 (-9));
  equal ~msg:"uint4 clamps high" int 15 (store_get uint4 99);
  equal ~msg:"uint4 clamps low" int 0 (store_get uint4 (-1))

let test_uint64_roundtrip () =
  equal ~msg:"uint64 all ones" int64 (-1L) (store_get uint64 (-1L));
  equal ~msg:"uint64 max int64 + 1" int64 Int64.min_int
    (store_get uint64 Int64.min_int)

(* Int4 packs two elements per byte: the bytes blits move whole bytes, so
   element offsets must be even and map to byte offset [off / 2]. *)
let test_int4_bytes_blit_roundtrip () =
  let src = create int4 8 in
  for i = 0 to 7 do
    set src i (i - 4)
  done;
  let bytes = Bytes.create 4 in
  blit_to_bytes src bytes;
  let dst = create int4 8 in
  blit_from_bytes bytes dst;
  for i = 0 to 7 do
    equal ~msg:(Printf.sprintf "int4 roundtrip[%d]" i) int (i - 4) (get dst i)
  done

let test_int4_bytes_blit_offsets () =
  let src = create int4 8 in
  for i = 0 to 7 do
    set src i (i mod 8)
  done;
  (* Elements 4..7 of [src], packed into two bytes. *)
  let bytes = Bytes.create 2 in
  blit_to_bytes ~src_off:4 ~len:4 src bytes;
  (* Into elements 2..5 of [dst]. *)
  let dst = create int4 8 in
  blit_from_bytes ~dst_off:2 ~len:4 bytes dst;
  equal ~msg:"int4 offset dst[1]" int 0 (get dst 1);
  equal ~msg:"int4 offset dst[2]" int 4 (get dst 2);
  equal ~msg:"int4 offset dst[5]" int 7 (get dst 5);
  equal ~msg:"int4 offset dst[6]" int 0 (get dst 6)

let test_int4_bytes_blit_odd_raises () =
  let buf = create int4 8 in
  let bytes = Bytes.create 4 in
  raises_match ~msg:"odd src_off to bytes" invalid_argument (fun () ->
      blit_to_bytes ~src_off:1 ~len:2 buf bytes);
  raises_match ~msg:"odd dst_off from bytes" invalid_argument (fun () ->
      blit_from_bytes ~dst_off:1 ~len:2 bytes buf);
  raises_match ~msg:"odd len not reaching the end" invalid_argument (fun () ->
      blit_from_bytes ~len:3 bytes buf);
  (* An odd length is fine when the copy reaches the end of the buffer: the
     trailing nibble is padding. *)
  let tail = create int4 5 in
  blit_from_bytes ~dst_off:2 ~len:3 bytes tail

(* Test bigarray conversions *)
let test_bigarray_roundtrip () =
  let buf = create float32 3 in
  set buf 0 1.0;
  set buf 1 2.0;
  set buf 2 3.0;
  let ba1 = to_bigarray1 buf in
  equal ~msg:"to_bigarray1 dim" int 3 (Bigarray.Array1.dim ba1);
  let buf2 = of_bigarray1 ba1 in
  equal ~msg:"roundtrip[0]" (float 1e-6) 1.0 (get buf2 0);
  equal ~msg:"roundtrip[2]" (float 1e-6) 3.0 (get buf2 2)

let test_genarray_roundtrip () =
  let buf = create float32 6 in
  for i = 0 to 5 do
    set buf i (float_of_int i)
  done;
  let ga = to_genarray buf [| 2; 3 |] in
  equal ~msg:"genarray dims" (array int) [| 2; 3 |] (Bigarray.Genarray.dims ga);
  let buf2 = of_genarray ga in
  equal ~msg:"genarray roundtrip length" int 6 (length buf2);
  equal ~msg:"genarray roundtrip[0]" (float 1e-6) 0.0 (get buf2 0);
  equal ~msg:"genarray roundtrip[5]" (float 1e-6) 5.0 (get buf2 5)

(* Extended dtypes have no faithful Bigarray.kind: viewing one as a bigarray
   would let stdlib operations misread it. *)
let test_to_bigarray1_extended_raises () =
  let buf = create bfloat16 4 in
  raises_match ~msg:"to_bigarray1 on bfloat16" invalid_argument (fun () ->
      to_bigarray1 buf)

(* Extended-dtype genarrays stay usable through the genarray bridge. *)
let test_genarray_extended_roundtrip () =
  let buf = create bfloat16 6 in
  for i = 0 to 5 do
    set buf i (float_of_int i)
  done;
  let ga = to_genarray buf [| 2; 3 |] in
  equal ~msg:"extended genarray dtype" bool true (genarray_dtype ga = BFloat16);
  let buf2 = of_genarray ga in
  equal ~msg:"extended genarray roundtrip[5]" (float 1e-6) 5.0 (get buf2 5)

let test_of_bigarray1_unsupported_raises () =
  let ba_int = Bigarray.Array1.create Bigarray.int Bigarray.c_layout 4 in
  raises_match ~msg:"of_bigarray1 on int" invalid_argument (fun () ->
      of_bigarray1 ba_int);
  let ba_char = Bigarray.Array1.create Bigarray.char Bigarray.c_layout 4 in
  raises_match ~msg:"of_bigarray1 on char" invalid_argument (fun () ->
      of_bigarray1 ba_char);
  let ba_nat = Bigarray.Array1.create Bigarray.nativeint Bigarray.c_layout 4 in
  raises_match ~msg:"of_bigarray1 on nativeint" invalid_argument (fun () ->
      of_bigarray1 ba_nat)

(* Reinterpretation *)

type any_dtype = K : ('a, 'b) Nx_dtype.t -> any_dtype

let byte_dtypes =
  [
    K Float16;
    K Float32;
    K Float64;
    K BFloat16;
    K Float8_e4m3;
    K Float8_e5m2;
    K Int8;
    K UInt8;
    K Int16;
    K UInt16;
    K Int32;
    K UInt32;
    K Int64;
    K UInt64;
    K Complex64;
    K Complex128;
    K Bool;
  ]

let pattern = Bytes.init 32 (fun i -> Char.chr (((i * 37) + 129) land 0xff))

let pattern_buffer () =
  let buf = create uint8 (Bytes.length pattern) in
  blit_from_bytes pattern buf;
  buf

let test_reinterpret_bit_exact () =
  List.iter
    (fun (K k) ->
      let name = Nx_dtype.to_string k in
      let n = Bytes.length pattern / Nx_dtype.itemsize k in
      let view = reinterpret k (pattern_buffer ()) in
      equal ~msg:(name ^ " dtype") string name (Nx_dtype.to_string (dtype view));
      equal ~msg:(name ^ " length") int n (length view);
      let stored = Bytes.create (Bytes.length pattern) in
      blit_to_bytes view stored;
      equal ~msg:(name ^ " bytes") bytes pattern stored;
      let owned = create k n in
      blit_from_bytes pattern owned;
      for i = 0 to n - 1 do
        is_true
          ~msg:(Printf.sprintf "%s element %d" name i)
          (compare (get view i) (get owned i) = 0)
      done;
      let back = Bytes.create (Bytes.length pattern) in
      blit_to_bytes (reinterpret UInt8 view) back;
      equal ~msg:(name ^ " back to bytes") bytes pattern back)
    byte_dtypes

let test_reinterpret_aliases () =
  let source = pattern_buffer () in
  let view = reinterpret UInt32 source in
  set view 1 0x04030201l;
  equal ~msg:"a write through the view reaches the source" (list int)
    (if Sys.big_endian then [ 4; 3; 2; 1 ] else [ 1; 2; 3; 4 ])
    (List.init 4 (fun i -> get source (4 + i)))

let sub_bytes buf off len =
  of_bigarray1 (Bigarray.Array1.sub (to_bigarray1 buf) off len)

let test_reinterpret_raises () =
  let source = pattern_buffer () in
  raises_match ~msg:"size not a multiple" invalid_argument (fun () ->
      reinterpret Int16 (sub_bytes source 0 3));
  raises_match ~msg:"misaligned address" invalid_argument (fun () ->
      reinterpret BFloat16 (sub_bytes source 1 8));
  raises_match ~msg:"to int4" invalid_argument (fun () ->
      reinterpret Int4 source);
  raises_match ~msg:"to uint4" invalid_argument (fun () ->
      reinterpret UInt4 source);
  raises_match ~msg:"from int4" invalid_argument (fun () ->
      reinterpret UInt8 (create int4 8));
  raises_match ~msg:"from uint4" invalid_argument (fun () ->
      reinterpret UInt8 (create uint4 8))

(* Whether the file [path] is mapped in this process, where the system can
   tell. *)
let is_mapped path =
  match open_in "/proc/self/maps" with
  | exception Sys_error _ -> None
  | ic ->
      Fun.protect
        ~finally:(fun () -> close_in ic)
        (fun () ->
          let rec scan () =
            match input_line ic with
            | exception End_of_file -> false
            | line -> String.ends_with ~suffix:path line || scan ()
          in
          Some (scan ()))

let map_bytes path =
  let fd = Unix.openfile path [ Unix.O_RDONLY ] 0 in
  Fun.protect
    ~finally:(fun () -> Unix.close fd)
    (fun () ->
      of_genarray
        (Unix.map_file fd Bigarray.int8_unsigned Bigarray.c_layout false
           [| -1 |]))

(* The functions below are not inlined, so that once they return no slot of the
   caller's frame holds the arrays they built. *)

let[@inline never] mapped_view path n =
  reinterpret BFloat16 (sub_bytes (map_bytes path) 2 (2 * n))

let[@inline never] read_mapped_view path n =
  let view = mapped_view path n in
  Gc.full_major ();
  Option.iter (is_true ~msg:"mapped while viewed") (is_mapped path);
  let stored = Bytes.create (2 * n) in
  blit_to_bytes view stored;
  for i = 0 to (2 * n) - 1 do
    if Bytes.get_uint8 stored i <> (i + 2) land 0xff then
      failf "byte %d of the view is %d" i (Bytes.get_uint8 stored i)
  done

(* A reinterpreted sub-array alone keeps a mapped file mapped, and the file is
   unmapped once it is unreachable. *)
let test_reinterpret_mapped_lifetime () =
  let n = 1 lsl 16 in
  let path = Filename.temp_file "nx_buffer_mapped_" ".bin" in
  Fun.protect
    ~finally:(fun () -> try Sys.remove path with Sys_error _ -> ())
    (fun () ->
      let oc = open_out_bin path in
      for i = 0 to (2 * n) + 1 do
        output_char oc (Char.chr (i land 0xff))
      done;
      close_out oc;
      read_mapped_view path n;
      Gc.full_major ();
      Option.iter
        (fun mapped -> is_true ~msg:"unmapped once unreachable" (not mapped))
        (is_mapped path))

(* Mapped files *)

let with_mapped_file n f =
  let path = Filename.temp_file "nx_buffer_file_" ".bin" in
  Fun.protect
    ~finally:(fun () -> try Sys.remove path with Sys_error _ -> ())
    (fun () ->
      let oc = open_out_bin path in
      for i = 0 to n - 1 do
        output_char oc (Char.chr (i land 0xff))
      done;
      close_out oc;
      let stat = Unix.stat path in
      let file =
        { path; size = n; mtime = stat.st_mtime; inode = stat.st_ino }
      in
      f file)

(* The mapping itself, as [Unix.map_file] returns it: not a view of it. *)
let map_root path =
  let fd = Unix.openfile path [ Unix.O_RDONLY ] 0 in
  Fun.protect
    ~finally:(fun () -> Unix.close fd)
    (fun () ->
      of_bigarray1
        (Bigarray.array1_of_genarray
           (Unix.map_file fd Bigarray.int8_unsigned Bigarray.c_layout false
              [| -1 |])))

let offset_in file buf =
  match file_range buf with
  | Some (f, offset) when f = file -> Some offset
  | Some _ -> fail "file_range answered another file"
  | None -> None

let[@inline never] check_mapped_ranges file =
  let mapped = map_root file.path in
  register_file file mapped;
  equal ~msg:"the mapping" (option int) (Some 0) (offset_in file mapped);
  let sub = sub_bytes mapped 24 64 in
  equal ~msg:"a sub-array" (option int) (Some 24) (offset_in file sub);
  let halves = reinterpret BFloat16 (sub_bytes sub 8 32) in
  equal ~msg:"a reinterpreted sub-array" (option int) (Some 32)
    (offset_in file halves);
  equal ~msg:"an ordinary buffer" (option int) None
    (offset_in file (create UInt8 64));
  raises_match ~msg:"a mapping that already has views" invalid_argument
    (fun () -> register_file file mapped);
  halves

let test_file_range () =
  with_mapped_file 4096 @@ fun file ->
  let halves = check_mapped_ranges file in
  Gc.full_major ();
  equal ~msg:"a view alone keeps the record" (option int) (Some 32)
    (offset_in file halves);
  let stored = Bytes.create 32 in
  blit_to_bytes halves stored;
  equal ~msg:"whose bytes are the file's at that offset" bytes
    (Bytes.init 32 (fun i -> Char.chr (32 + i)))
    stored;
  raises_match ~msg:"an ordinary buffer is not a mapped file" invalid_argument
    (fun () -> register_file file (create UInt8 16))

(* A record dies with its mapping: memory mapped later at the same address,
   which the system tends to hand out again, is not taken for the old file. *)
let test_file_range_after_unmap () =
  with_mapped_file 4096 @@ fun first ->
  with_mapped_file 4096 @@ fun second ->
  let[@inline never] map_and_drop file =
    let mapped = map_root file.path in
    register_file file mapped;
    equal ~msg:"mapped" (option int) (Some 0) (offset_in file mapped);
    unsafe_data_ptr mapped
  in
  let[@inline never] unrecorded_at address =
    let mapped = map_root second.path in
    if unsafe_data_ptr mapped = address then
      equal ~msg:"an unrecorded mapping at a recorded address" (option int) None
        (offset_in first mapped)
  in
  for _ = 1 to 8 do
    let address = map_and_drop first in
    Gc.full_major ();
    unrecorded_at address;
    Gc.full_major ()
  done

(* Test suite *)
let () =
  run "Nx_buffer tests"
    [
      group "creation"
        [
          test "create bfloat16" test_create_bfloat16;
          test "create bool" test_create_bool;
          test "create int4" test_create_int4;
          test "create uint4" test_create_uint4;
          test "create float8_e4m3" test_create_float8_e4m3;
          test "create float8_e5m2" test_create_float8_e5m2;
        ];
      group "genarray" [ test "genarray creation" test_genarray_creation ];
      group "properties"
        [
          test "dtype round-trip" test_dtype_roundtrip;
          test "itemsize" test_itemsize;
        ];
      group "semantics"
        [
          test "bfloat16" test_bfloat16_semantics;
          test "float8 e4m3" test_float8_e4m3_semantics;
          test "float8 e5m2" test_float8_e5m2_semantics;
          test "float64 rounds once" test_float64_rounds_once;
          test "int4 clamping" test_int4_clamping;
          test "uint64 roundtrip" test_uint64_roundtrip;
        ];
      group "operations"
        [
          test "blit" test_blit;
          test "fill" test_fill;
          test "int4 bytes blit roundtrip" test_int4_bytes_blit_roundtrip;
          test "int4 bytes blit offsets" test_int4_bytes_blit_offsets;
          test "int4 bytes blit odd offsets raise"
            test_int4_bytes_blit_odd_raises;
        ];
      group "conversions"
        [
          test "bigarray roundtrip" test_bigarray_roundtrip;
          test "genarray roundtrip" test_genarray_roundtrip;
          test "to_bigarray1 rejects extended kinds"
            test_to_bigarray1_extended_raises;
          test "extended genarray bridge" test_genarray_extended_roundtrip;
          test "of_bigarray1 rejects unsupported kinds"
            test_of_bigarray1_unsupported_raises;
        ];
      group "reinterpret"
        [
          test "bit exact at every dtype" test_reinterpret_bit_exact;
          test "aliases its source" test_reinterpret_aliases;
          test "raises" test_reinterpret_raises;
          test "keeps a mapped file alive" test_reinterpret_mapped_lifetime;
        ];
      group "mapped files"
        [
          test "file_range follows views by address" test_file_range;
          test "a record dies with its mapping" test_file_range_after_unmap;
        ];
    ]
