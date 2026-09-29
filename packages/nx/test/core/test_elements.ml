(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Typed access to host buffers: a store reads back as the dtype rounds it and
   lands as the format's bits, a fill stays inside its buffer, and a gather
   keeps every bit of the elements a view reaches. *)

open Windtrap
module B = Nx_device.Buffer
module E = Nx_core.Elements
module S = Nx_dtype.Scalar
module V = Nx_core.View

let bytes b = B.bigarray Bigarray.char b
let size s = Int.max 1 (S.bitsize s / 8)

(* The bits of element [i] of [ba], elements of [s] in storage order. *)
let bits s ba i =
  if S.bitsize s = 4 then
    let byte = Char.code ba.{i / 2} in
    String.make 1
      (Char.chr (if i land 1 = 0 then byte land 0xf else byte lsr 4))
  else String.init (size s) (fun k -> ba.{(i * size s) + k})

let round32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* The [n] bytes of [c], little-endian. *)
let le n c = String.init n (fun k -> Char.chr ((c lsr (8 * k)) land 0xff))

(* A dtype, values of it, what a store of each reads back as, its witness and
   the bits a store of it lands as, when they are not those of a bigarray
   kind. *)
type elt =
  | Elt : {
      dt : ('a, 'b) Nx_dtype.t;
      value : 'a Gen.t;
      stored : 'a -> 'a;
      w : 'a testable;
      code : ('a -> string) option;
    }
      -> elt

let elt ?code dt value stored w = Elt { dt; value; stored; w; code }
let ints lo hi = Gen.int_range lo hi

let narrow dt =
  let s = S.of_dtype dt in
  elt
    ~code:(fun x -> le (size s) (S.encode s x))
    dt Gen.any_float
    (fun x -> S.decode s (S.encode s x))
    float_exact

let complex round =
  let part = Gen.any_float in
  ( Gen.map (fun (re, im) -> { Complex.re; im }) (Gen.pair part part),
    (fun (c : Complex.t) -> { Complex.re = round c.re; im = round c.im }),
    Testable.contramap
      (fun (c : Complex.t) -> (c.re, c.im))
      (pair float_exact float_exact) )

let elts =
  let c64, r64, w64 = complex round32 and c128, r128, w128 = complex Fun.id in
  [
    elt Nx_dtype.float16 Gen.any_float
      (fun x -> S.decode Float16 (S.encode Float16 (round32 x)))
      float_exact;
    elt Nx_dtype.float32 Gen.any_float round32 float_exact;
    elt Nx_dtype.float64 Gen.any_float Fun.id float_exact;
    narrow Nx_dtype.bfloat16;
    narrow Nx_dtype.float8_e4m3;
    narrow Nx_dtype.float8_e5m2;
    elt
      ~code:(fun v -> le 1 (v land 0xf))
      Nx_dtype.int4 (ints (-8) 7) Fun.id int;
    elt ~code:(le 1) Nx_dtype.uint4 (ints 0 15) Fun.id int;
    elt Nx_dtype.int8 (ints (-128) 127) Fun.id int;
    elt Nx_dtype.uint8 (ints 0 255) Fun.id int;
    elt Nx_dtype.int16 (ints (-32768) 32767) Fun.id int;
    elt Nx_dtype.uint16 (ints 0 65535) Fun.id int;
    elt Nx_dtype.int32 Gen.int32 Fun.id int32;
    elt Nx_dtype.uint32 Gen.int32 Fun.id int32;
    elt Nx_dtype.int64 Gen.int64 Fun.id int64;
    elt Nx_dtype.uint64 Gen.int64 Fun.id int64;
    elt Nx_dtype.complex64 c64 r64 w64;
    elt Nx_dtype.complex128 c128 r128 w128;
    elt ~code:(fun b -> le 1 (Bool.to_int b)) Nx_dtype.bool Gen.bool Fun.id bool;
  ]

let stores =
  group "stores"
    (List.map
       (fun (Elt e) ->
         let s = S.of_dtype e.dt in
         prop
           (S.to_string s
          ^ " reads back what it stores, as a store of it rounds, in its \
             format's bits")
           (Gen.array ~size:(Gen.int_range 0 9) e.value)
           (fun xs ->
             let n = Array.length xs in
             let b = E.create e.dt n in
             Array.iteri (E.set e.dt b) xs;
             equal (array e.w) (Array.map e.stored xs)
               (Array.init n (E.get e.dt b));
             Option.iter
               (fun code ->
                 equal ~msg:"bits" (list string)
                   (List.map code (Array.to_list xs))
                   (List.init n (bits s (bytes b))))
               e.code))
       elts)

let fills =
  group "fill"
    (List.map
       (fun (Elt e) ->
         let s = S.of_dtype e.dt in
         prop
           (S.to_string s
          ^ " stores its value as every element, and nothing outside them")
           (Gen.triple e.value (Gen.int_range 0 9) (Gen.int_range 0 3))
           (fun (x, n, k) ->
             let m = n + (2 * k) + 3 in
             let whole = B.create Nx_device.host s m in
             let ba = bytes whole in
             for i = 0 to Bigarray.Array1.dim ba - 1 do
               ba.{i} <- Char.chr (((i * 37) + 11) land 0xff)
             done;
             let first = 2 * k in
             let others () =
               List.filteri
                 (fun i _ -> i < first || i >= first + n)
                 (List.init m (bits s ba))
             in
             let before = others () in
             let b = B.view whole ~offset:(first * S.bitsize s / 8) s n in
             E.fill e.dt b x;
             equal (array e.w)
               (Array.make n (e.stored x))
               (Array.init n (E.get e.dt b));
             equal ~msg:"the elements around" (list string) before (others ())))
       elts)

(* A view of [shape] and [strides] and the offset and length of a buffer it
   reaches the ends of, give or take [slack]. *)
let views =
  let open Gen in
  let* shape = array ~size:(int_range 0 3) (int_range 0 3) in
  let* strides =
    array ~size:(constant (Array.length shape)) (int_range (-3) 3)
  in
  let+ slack = int_range 0 2 in
  let reach sign =
    if Array.mem 0 shape then 0
    else
      Array.fold_left ( + ) 0
        (Array.mapi
           (fun a n ->
             if sign * strides.(a) > 0 then strides.(a) * (n - 1) else 0)
           shape)
  in
  let offset = slack - reach (-1) in
  (V.create ~offset ~strides shape, offset + reach 1 + 1 + slack)

let pp_view ppf (v, n) =
  Format.fprintf ppf "shape %a strides %a offset %d of %d" Nx_test.pp_shape
    (V.shape v) Nx_test.pp_shape (V.strides v) (V.offset v) n

let formats =
  Gen.of_list
    ~pp:(fun ppf s -> Format.pp_print_string ppf (S.to_string s))
    S.[ Bool; Int4; UInt4; UInt8; BFloat16; Float32; Int64; Complex128 ]

let gather =
  prop "gather is the elements a view reaches in C order, each keeping its bits"
    (Gen.triple formats
       (Gen.with_pp pp_view views)
       (Gen.string_of ~size:(Gen.int_range 1 9) Gen.char))
    (fun (s, (v, n), seed) ->
      let src = B.create Nx_device.host s n in
      let ba = bytes src in
      for i = 0 to Bigarray.Array1.dim ba - 1 do
        ba.{i} <- seed.[i mod String.length seed]
      done;
      let g = E.gather src v in
      let expected =
        List.init (V.numel v) (fun k ->
            let idx = Nx_test.unravel (V.shape v) k in
            let p = ref (V.offset v) in
            Array.iteri (fun a i -> p := !p + (i * (V.strides v).(a))) idx;
            bits s ba !p)
      in
      equal (pair string int)
        (S.to_string s, V.numel v)
        (S.to_string (B.dtype g), B.length g);
      equal (list string) expected (List.init (V.numel v) (bits s (bytes g))))

(* A device over host memory, whose buffers are not the host's. *)
let other =
  lazy
    (let alloc n =
       let ba = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n in
       let a = B.host_address (B.of_bigarray ba) in
       Some { Nx_device.host = Some a; device = a; handle = a }
     in
     Nx_device.make ~name:"OTHER" ~arch:"test" ~budget:max_int
       ~memory:{ alloc; free = ignore } ())

let refusals =
  let f32 = E.create Nx_dtype.float32 2 in
  let open Nx_dtype in
  cases ~name:fst "refuse"
    [
      ("a buffer of another format", fun () -> ignore (E.get int32 f32 0));
      ("an index past the last element", fun () -> ignore (E.get float32 f32 2));
      ("an index of -1", fun () -> ignore (E.get float32 f32 (-1)));
      ( "a 4-bit store past the last element",
        fun () -> E.set int4 (E.create int4 3) 3 0 );
      ( "a buffer of another device",
        fun () -> ignore (E.get uint8 (B.create (Lazy.force other) S.UInt8 1) 0)
      );
      ( "a gather past the end",
        fun () ->
          ignore (E.gather f32 (V.create ~offset:1 ~strides:[| 1 |] [| 2 |])) );
      ( "a gather before the start",
        fun () ->
          ignore (E.gather f32 (V.create ~offset:0 ~strides:[| -1 |] [| 2 |]))
      );
      ( "a gather of another device's buffer",
        fun () ->
          ignore
            (E.gather
               (B.create (Lazy.force other) S.UInt8 2)
               (V.create [| 2 |])) );
    ]
    (fun (_, f) -> raises_match Exn.invalid_arg f)

let () =
  exit
    (run "Nx_core.Elements"
       [
         stores;
         fills;
         gather;
         refusals;
         test
           "4-bit stores clamp out of range values (the interfaces are silent)"
           (fun () ->
             let i4 = E.create Nx_dtype.int4 2
             and u4 = E.create Nx_dtype.uint4 2 in
             List.iteri (E.set Nx_dtype.int4 i4) [ -9; 8 ];
             List.iteri (E.set Nx_dtype.uint4 u4) [ -3; 20 ];
             equal (list int) [ -8; 7; 0; 15 ]
               (List.init 2 (E.get Nx_dtype.int4 i4)
               @ List.init 2 (E.get Nx_dtype.uint4 u4)));
       ])
