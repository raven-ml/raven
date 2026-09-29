(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Buffers against arrays of cells. A buffer, the bigarrays and genarrays that
   view it and the buffers made back from those share one array of cells, so a
   write through one must be seen through all. A cell is [None] until written:
   [create] leaves the contents unspecified. *)

open Windtrap
module B = Nx_buffer
module S = Nx_dtype.Scalar

(* Dtypes *)

type 'a codec = (bytes -> int -> 'a -> unit) * (bytes -> int -> 'a)

(* A dtype, values it holds exactly, and its storage representation, which int4
   and uint4 lack: they pack two elements a byte. *)
type dtype =
  | D : {
      dtype : ('a, 'b) Nx_dtype.t;
      exact : 'a testable;
      value : 'a Gen.t;
      codec : 'a codec option;
    }
      -> dtype

let d ?codec dtype exact value =
  D { dtype; exact; codec; value = Gen.with_pp (Testable.pp exact) value }

let f32 : float codec =
  ( (fun b o x -> Bytes.set_int32_ne b o (Int32.bits_of_float x)),
    fun b o -> Int32.float_of_bits (Bytes.get_int32_ne b o) )

let f64 : float codec =
  ( (fun b o x -> Bytes.set_int64_ne b o (Int64.bits_of_float x)),
    fun b o -> Int64.float_of_bits (Bytes.get_int64_ne b o) )

let complex ((put, take) : float codec) w =
  Some
    ( (fun b o (z : Complex.t) ->
        put b o z.re;
        put b (o + w) z.im),
      fun b o -> Complex.{ re = take b o; im = take b (o + w) } )

(* A narrow float drawn as its bits, so NaN and subnormals are drawn too. *)
let narrow dtype s corners =
  let bits = S.bitsize s in
  let get, set =
    if bits = 16 then (Bytes.get_uint16_ne, Bytes.set_uint16_ne)
    else (Bytes.get_uint8, Bytes.set_uint8)
  in
  d dtype float_exact
    ~codec:
      ((fun b o v -> set b o (S.encode s v)), fun b o -> S.decode s (get b o))
    (Gen.map (S.decode s)
       (Gen.frequency
          [
            (4, Gen.int_range 0 ((1 lsl bits) - 1));
            (1, Gen.of_list ~pp:Format.pp_print_int corners);
          ]))

let ints ?codec dtype lo hi = d ?codec dtype int (Gen.int_range lo hi)

let dtypes =
  let open Nx_dtype in
  let f32s = Gen.map Int32.float_of_bits Gen.int32 in
  let cplx g = Gen.map (fun (re, im) -> Complex.{ re; im }) (Gen.pair g g) in
  let i32 = (Bytes.set_int32_ne, Bytes.get_int32_ne)
  and i64 = (Bytes.set_int64_ne, Bytes.get_int64_ne)
  and cx =
    Testable.contramap
      (fun (z : Complex.t) -> (z.re, z.im))
      (pair float_exact float_exact)
  in
  [
    narrow float16 S.Float16 [ 0; 0x8000; 0x7C00; 0xFC00; 0x7E00; 1; 0x7BFF ];
    d float32 float_exact f32s ~codec:f32;
    d float64 float_exact Gen.any_float ~codec:f64;
    narrow bfloat16 S.BFloat16 [ 0; 0x8000; 0x7F80; 0xFF80; 0x7FC0; 1; 0x7F7F ];
    narrow float8_e4m3 S.Float8_e4m3 [ 0; 0x80; 0x7F; 1; 7; 0x7E ];
    narrow float8_e5m2 S.Float8_e5m2 [ 0; 0x80; 0x7C; 0xFC; 0x7F; 1; 0x7B ];
    ints int4 (-8) 7;
    ints uint4 0 15;
    ints int8 (-128) 127 ~codec:(Bytes.set_int8, Bytes.get_int8);
    ints uint8 0 255 ~codec:(Bytes.set_uint8, Bytes.get_uint8);
    ints int16 (-32768) 32767 ~codec:(Bytes.set_int16_ne, Bytes.get_int16_ne);
    ints uint16 0 65535 ~codec:(Bytes.set_uint16_ne, Bytes.get_uint16_ne);
    d int32 Windtrap.int32 Gen.int32 ~codec:i32;
    d uint32 Windtrap.int32 Gen.int32 ~codec:i32;
    d int64 Windtrap.int64 Gen.int64 ~codec:i64;
    d uint64 Windtrap.int64 Gen.int64 ~codec:i64;
    d complex64 cx (cplx f32s) ?codec:(complex f32 4);
    d complex128 cx (cplx Gen.any_float) ?codec:(complex f64 8);
    d bool Windtrap.bool Gen.bool
      ~codec:
        ( (fun b o v -> Bytes.set_uint8 b o (Bool.to_int v)),
          fun b o -> Bytes.get_uint8 b o <> 0 );
  ]

let name (D d) = Nx_dtype.to_string d.dtype

let of_values dtype values =
  let b = B.create dtype (Array.length values) in
  Array.iteri (B.set b) values;
  b

let elements b = Array.init (B.length b) (B.get b)

let bytes_of b =
  let bytes = Bytes.create (B.length b * Nx_dtype.itemsize (B.dtype b)) in
  B.blit_to_bytes b bytes;
  bytes

let refuses = List.iter (fun f -> raises_match Exn.invalid_arg f)

(* The model *)

type 'a ga = { dims : int array; cells : 'a option array }

(* Integers across the edges of buffers of at most 6 elements and bytes of at
   most 10, and the extremes of [int]. *)
let edge =
  Gen.frequency
    [
      (3, Gen.int_range (-2) 12);
      ( 1,
        Gen.of_list ~pp:Format.pp_print_int
          [ min_int; min_int + 1; max_int - 1; max_int ] );
    ]

(* A bytes copy's offsets and length from the documented defaults, [None] when
   one is negative or the copy runs past either end. *)
let span ~src ~dst ~default (src_off, dst_off, len) =
  let s = Option.value src_off ~default:0
  and t = Option.value dst_off ~default:0 in
  let len = Option.value len ~default:(default s t) in
  if
    s < 0 || t < 0 || len < 0 || s > src || t > dst
    || len > src - s
    || len > dst - t
  then None
  else Some (s, t, len)

let copies dtype exact value codec buf =
  match codec with
  | None -> []
  | Some (put, take) ->
      let size = Nx_dtype.itemsize dtype in
      let encode vs =
        let b = Bytes.create (Array.length vs * size) in
        Array.iteri (fun i v -> put b (i * size) v) vs;
        b
      in
      let blank room = Bytes.make (room * size) 'Z' in
      let decode b =
        Array.init (Bytes.length b / size) (fun i -> take b (i * size))
      in
      let offsets =
        let o =
          Gen.option (Gen.frequency [ (2, Gen.int_range 0 6); (1, edge) ])
        in
        Gen.triple o o o
      in
      let into r =
        span ~dst:(Array.length r) ~default:(fun _ t -> Array.length r - t)
      in
      let out_of r =
        span ~src:(Array.length r) ~default:(fun s _ -> Array.length r - s)
      in
      [
        command "blit_from_bytes"
          (buf ^-> offsets
          @-> Gen.array ~size:(Gen.int_range 0 10) value
          @-> returns unit)
          (fun r offs vs ->
            match into r ~src:(Array.length vs) offs with
            | None -> invalid_arg "blit_from_bytes"
            | Some (s, t, len) ->
                for k = 0 to len - 1 do
                  r.(t + k) <- Some vs.(s + k)
                done)
          (fun b (src_off, dst_off, len) vs ->
            B.blit_from_bytes ?src_off ?dst_off ?len (encode vs) b);
        command "blit_to_bytes"
          ~pre:(fun r offs room ->
            match out_of r ~dst:room offs with
            | None -> true
            | Some (s, _, len) ->
                Array.for_all Option.is_some (Array.sub r s len))
          (buf ^-> offsets @-> Gen.int_range 0 10 @-> returns (array exact))
          (fun r offs room ->
            match out_of r ~dst:room offs with
            | None -> invalid_arg "blit_to_bytes"
            | Some (s, t, len) ->
                cover "a copy out" (len > 0);
                let out = decode (blank room) in
                for k = 0 to len - 1 do
                  out.(t + k) <- Option.get r.(s + k)
                done;
                out)
          (fun b (src_off, dst_off, len) room ->
            let bytes = blank room in
            B.blit_to_bytes ?src_off ?dst_off ?len b bytes;
            decode bytes);
      ]

let commands (D d as dt) =
  let seen r read = Array.mapi (fun i c -> Option.map (fun _ -> read i) c) r in
  let cells = array (option d.exact) and name = name dt in
  let pp = Testable.pp cells in
  let buf =
    abstract "b" ~pp ~invariant:(fun r s ->
        equal ~msg:"dtype" string name (Nx_dtype.to_string (B.dtype s));
        equal cells r (seen r (B.get s)))
  in
  let ga =
    abstract "g" ~invariant:(fun r s ->
        equal ~msg:"dims" (array int) r.dims (B.genarray_dims s);
        equal ~msg:"dtype" string name (Nx_dtype.to_string (B.genarray_dtype s));
        equal cells r.cells (seen r.cells (B.get (B.of_genarray s))))
  in
  let all r = List.init (Array.length r) Fun.id in
  let at = among int buf all in
  let written =
    among int buf (fun r -> List.filter (fun i -> r.(i) <> None) (all r))
  in
  let get r i = Option.get r.(i) and put r i v = r.(i) <- Some v in
  let inside r i =
    if i < 0 || i >= Array.length r then invalid_arg "index out of bounds"
  in
  let n = Array.length in
  let shapes r =
    [ [| n r |]; [| 1; n r |]; [| n r; 1 |] ]
    @ (if n r = 1 then [ [||] ] else [])
    @ if n r mod 2 = 0 then [ [| 2; n r / 2 |] ] else []
  in
  let shape =
    among (Testable.make ~pp:(Testable.pp (array int)) ~equal:( == )) buf shapes
  in
  [
    command "create"
      (Gen.int_range 0 6 @-> makes buf)
      (fun k -> Array.make k None)
      (B.create d.dtype);
    command "fill"
      (buf ^-> d.value @-> returns unit)
      (fun r v -> Array.fill r 0 (n r) (Some v))
      B.fill;
    command "get" (buf ^-> written ^-> returns d.exact) get B.get;
    command "unsafe_get" (buf ^-> written ^-> returns d.exact) get B.unsafe_get;
    command "set" (buf ^-> at ^-> d.value @-> returns unit) put B.set;
    command "unsafe_set"
      (buf ^-> at ^-> d.value @-> returns unit)
      put B.unsafe_set;
    command "get at any index"
      ~pre:(fun r i -> i < 0 || i >= n r || r.(i) <> None)
      (buf ^-> edge @-> returns d.exact)
      (fun r i ->
        inside r i;
        get r i)
      B.get;
    command "set at any index"
      (buf ^-> edge @-> d.value @-> returns unit)
      (fun r i v ->
        inside r i;
        put r i v)
      B.set;
    command "blit"
      (buf ^-> buf ^-> returns unit)
      (fun src dst ->
        if n src <> n dst then invalid_arg "different dimensions";
        Array.blit src 0 dst 0 (n src))
      (fun src dst -> B.blit ~src ~dst);
    command "to_genarray"
      (buf ^-> shape ^-> makes ga)
      (fun cells dims -> { dims; cells })
      B.to_genarray;
    command "genarray_create"
      (Gen.array ~size:(Gen.int_range 0 3) (Gen.int_range 0 3) @-> makes ga)
      (fun dims ->
        { dims; cells = Array.make (Array.fold_left ( * ) 1 dims) None })
      (B.genarray_create d.dtype Bigarray.c_layout);
    command "of_genarray" (ga ^-> makes buf) (fun g -> g.cells) B.of_genarray;
    command "genarray_blit"
      ~pre:(fun a b -> a.dims = b.dims)
      (ga ^-> ga ^-> returns unit)
      (fun a b -> Array.blit a.cells 0 b.cells 0 (n a.cells))
      B.genarray_blit;
  ]
  @ (match Nx_dtype.to_bigarray_kind d.dtype with
    | None -> []
    | Some _ ->
        let ba =
          abstract "a" ~pp ~invariant:(fun r s ->
              equal cells r (seen r (Bigarray.Array1.get s)))
        in
        [
          command "to_bigarray1" (buf ^-> makes ba) Fun.id B.to_bigarray1;
          command "of_bigarray1" (ba ^-> makes buf) Fun.id B.of_bigarray1;
          command "Bigarray.Array1.set"
            (ba ^-> among int ba all ^-> d.value @-> returns unit)
            put Bigarray.Array1.set;
        ])
  @ copies d.dtype d.exact d.value d.codec buf

let buffers =
  group "buffers behave like arrays whose views share their cells"
    (List.map (fun dt -> stateful ~count:200 (name dt) (commands dt)) dtypes)

(* Reinterpretation *)

type target = T : ('a, 'b) Nx_dtype.t -> target

let widths =
  Gen.of_list
    ~pp:(fun ppf (T t) -> Format.pp_print_string ppf (Nx_dtype.to_string t))
    Nx_dtype.[ T uint8; T int16; T float32; T int64; T complex128 ]

let rec gcd a b = if b = 0 then a else gcd b (a mod b)

let reinterpret_law (D d as dt) =
  let size = Nx_dtype.itemsize d.dtype in
  let drawn =
    Gen.bind widths (fun (T t as target) ->
        let unit = Nx_dtype.itemsize t / gcd size (Nx_dtype.itemsize t) in
        Gen.map
          (fun vs -> (target, vs))
          (Gen.bind (Gen.int_range 0 3) (fun k ->
               Gen.array ~size:(Gen.constant (k * unit)) d.value)))
  in
  prop
    (name dt ^ " read at every width and back keeps its elements and bytes")
    drawn
    (fun (T t, vs) ->
      let b = of_values d.dtype vs in
      equal ~msg:"bytes" bytes (bytes_of b) (bytes_of (B.reinterpret t b));
      Law.round_trip
        (Testable.contramap elements (array d.exact))
        pass (B.reinterpret t) (B.reinterpret d.dtype) b)

let reinterpretation =
  group "reinterpret"
    (List.filter_map
       (fun (D d as dt) -> Option.map (fun _ -> reinterpret_law dt) d.codec)
       dtypes
    @ [
        (* Two domains reinterpret each fresh buffer at once and one keeps its
           view, which must share the storage and so outlive the buffer. *)
        test "views taken on two domains at once share the storage" (fun () ->
            let rounds = 2000 and current = Atomic.make None in
            let taken = Array.init 2 (fun _ -> Atomic.make 0)
            and kept = ref [] in
            let worker k () =
              for r = 0 to rounds - 1 do
                let rec next () =
                  match Atomic.get current with
                  | Some b when Atomic.get taken.(k) = r -> b
                  | _ ->
                      Domain.cpu_relax ();
                      next ()
                in
                let v = B.reinterpret Nx_dtype.int32 (next ()) in
                if k = 0 then (
                  B.fill v 7l;
                  kept := v :: !kept);
                Atomic.incr taken.(k)
              done
            in
            let workers = List.init 2 (fun k -> Domain.spawn (worker k)) in
            for r = 0 to rounds - 1 do
              Atomic.set current
                (Some
                   (B.of_bigarray1
                      (Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout
                         16)));
              while Atomic.get taken.(0) <= r || Atomic.get taken.(1) <= r do
                Domain.cpu_relax ()
              done
            done;
            Atomic.set current None;
            List.iter Domain.join workers;
            Gc.full_major ();
            let filler =
              List.init 20000 (fun _ ->
                  Bigarray.Array1.init Bigarray.int32 Bigarray.c_layout 16
                    (fun _ -> 0x55l))
            in
            equal int 0
              (List.length (List.filter (fun v -> B.get v 5 <> 7l) !kept));
            ignore (Sys.opaque_identity filler));
        test "shares its source's storage and address" (fun () ->
            let source = of_values Nx_dtype.uint8 (Array.make 8 0) in
            let view = B.reinterpret Nx_dtype.int32 source in
            B.set view 1 0x04030201l;
            equal (array int)
              (if Sys.big_endian then [| 0; 0; 0; 0; 4; 3; 2; 1 |]
               else [| 0; 0; 0; 0; 1; 2; 3; 4 |])
              (elements source);
            equal nativeint (B.unsafe_data_ptr source) (B.unsafe_data_ptr view));
        test
          "refuses a size that is not a multiple, a misaligned address and int4"
          (fun () ->
            let u8 = B.create Nx_dtype.uint8 32 in
            let sub off len =
              B.of_bigarray1 (Bigarray.Array1.sub (B.to_bigarray1 u8) off len)
            in
            refuses
              [
                (fun () -> ignore (B.reinterpret Nx_dtype.int16 (sub 0 3)));
                (fun () -> ignore (B.reinterpret Nx_dtype.bfloat16 (sub 1 8)));
                (fun () -> ignore (B.reinterpret Nx_dtype.int4 u8));
                (fun () ->
                  ignore
                    (B.reinterpret Nx_dtype.uint8 (B.create Nx_dtype.uint4 8)));
              ]);
      ])

(* Int4 and uint4 *)

let packed =
  group "int4 and uint4"
    (List.filter_map
       (fun (D d as dt) ->
         if d.codec <> None then None
         else
           Some
             (prop
                (name dt ^ " round trips through packed bytes")
                (Gen.array ~size:(Gen.int_range 0 9) d.value)
                (fun vs ->
                  let bytes = Bytes.create ((Array.length vs + 1) / 2) in
                  B.blit_to_bytes (of_values d.dtype vs) bytes;
                  let back = B.create d.dtype (Array.length vs) in
                  B.blit_from_bytes bytes back;
                  equal (array d.exact) vs (elements back))))
       dtypes
    @ [
        test "a byte holds two elements, the first in its low nibble" (fun () ->
            equal string "\xe1\x78"
              (Bytes.sub_string
                 (bytes_of (of_values Nx_dtype.int4 [| 1; -2; -8; 7 |]))
                 0 2));
        test
          "copies at even offsets move whole bytes, and an odd tail reaches \
           the end" (fun () ->
            let bytes = Bytes.create 2 in
            B.blit_to_bytes ~src_off:4 ~len:4
              (of_values Nx_dtype.int4 (Array.init 8 (fun i -> i - 4)))
              bytes;
            let dst = of_values Nx_dtype.int4 (Array.make 8 0) in
            B.blit_from_bytes ~dst_off:2 ~len:4 bytes dst;
            equal (array int) [| 0; 0; 0; 1; 2; 3; 0; 0 |] (elements dst);
            let tail = of_values Nx_dtype.uint4 (Array.make 5 0) in
            B.blit_from_bytes ~dst_off:2 ~len:3
              (Bytes.of_string "\x21\x43")
              tail;
            equal (array int) [| 0; 0; 1; 2; 3 |] (elements tail));
        test
          "bytes copies refuse odd int4 offsets and lengths short of the end, \
           and an offset whose sum with the length overflows" (fun () ->
            let buf = B.create Nx_dtype.int4 8 and bytes = Bytes.create 4 in
            let one = B.create Nx_dtype.uint8 1 in
            refuses
              [
                (fun () ->
                  B.blit_from_bytes ~dst_off:(max_int - 1) ~len:2 bytes one);
                (fun () ->
                  B.blit_to_bytes ~dst_off:(max_int - 1) ~len:2
                    (B.create Nx_dtype.uint8 2)
                    bytes);
                (fun () -> B.blit_to_bytes ~src_off:1 ~len:2 buf bytes);
                (fun () -> B.blit_to_bytes ~dst_off:1 ~len:2 buf bytes);
                (fun () -> B.blit_from_bytes ~dst_off:1 ~len:2 bytes buf);
                (fun () -> B.blit_from_bytes ~src_off:1 ~len:2 bytes buf);
                (fun () -> B.blit_from_bytes ~len:3 bytes buf);
              ]);
        test "set clamps to the range (nx_buffer.mli is silent)" (fun () ->
            equal (array int) [| 7; -8 |]
              (elements (of_values Nx_dtype.int4 [| 9; -9 |]));
            equal (array int) [| 15; 0 |]
              (elements (of_values Nx_dtype.uint4 [| 99; -1 |])));
      ])

(* Conversions *)

let conversions =
  group "conversions"
    [
      test
        "to_bigarray1 refuses an extended dtype, of_bigarray1 and of_genarray \
         char, int and nativeint" (fun () ->
          let kind k =
            [
              (fun () ->
                ignore
                  (B.of_bigarray1
                     (Bigarray.Array1.create k Bigarray.c_layout 4)));
              (fun () ->
                ignore
                  (B.of_genarray
                     (Bigarray.Genarray.create k Bigarray.c_layout [| 2; 2 |])));
            ]
          in
          refuses
            (List.filter_map
               (fun (D d) ->
                 if Nx_dtype.to_bigarray_kind d.dtype = None then
                   Some (fun () -> ignore (B.to_bigarray1 (B.create d.dtype 2)))
                 else None)
               dtypes
            @ kind Bigarray.char @ kind Bigarray.int @ kind Bigarray.nativeint));
      test "genarray_change_layout reverses the dims and keeps every dtype"
        (fun () ->
          List.iter
            (fun (D d as dt) ->
              let ga =
                B.genarray_create d.dtype Bigarray.c_layout [| 2; 3; 4 |]
              in
              let f = B.genarray_change_layout ga Bigarray.fortran_layout in
              equal
                (pair (array int) string)
                ([| 4; 3; 2 |], name dt)
                (B.genarray_dims f, Nx_dtype.to_string (B.genarray_dtype f)))
            dtypes);
      test "a stored NaN keeps its sign, as Nx_dtype.Scalar.encode states"
        (fun () ->
          let sign dt =
            let b = bytes_of (of_values dt [| -.Float.nan |]) in
            if Bytes.length b = 2 then Bytes.get_uint16_ne b 0 lsr 15
            else Bytes.get_uint8 b 0 lsr 7
          in
          equal (list int) [ 1; 1; 1 ]
            Nx_dtype.[ sign bfloat16; sign float8_e4m3; sign float8_e5m2 ]);
    ]

(* Mapped files *)

let map path =
  let fd = Unix.openfile path [ Unix.O_RDONLY ] 0 in
  Fun.protect
    ~finally:(fun () -> Unix.close fd)
    (fun () ->
      B.of_bigarray1
        (Bigarray.array1_of_genarray
           (Unix.map_file fd Bigarray.int8_unsigned Bigarray.c_layout false
              [| -1 |])))

let sub b off len =
  B.of_bigarray1 (Bigarray.Array1.sub (B.to_bigarray1 b) off len)

let mapped_file n =
  let path = temp_file ~suffix:".bin" () in
  Out_channel.with_open_bin path (fun oc ->
      Out_channel.output_string oc
        (String.init n (fun i -> Char.chr (i land 0xff))));
  let st = Unix.stat path in
  B.{ path; size = n; mtime = st.st_mtime; inode = st.st_ino }

let offset_in (file : B.file) b =
  Option.map
    (fun (f, o) -> if f = file then o else fail "another file")
    (B.file_range b)

(* Not inlined, so that no slot of the caller's frame holds the mapping once it
   returns. *)
let[@inline never] ranges file =
  let m = map file.B.path in
  B.register_file file m;
  let halves = B.reinterpret Nx_dtype.bfloat16 (sub (sub m 24 64) 8 32) in
  refuses [ (fun () -> B.register_file file m) ];
  equal
    (list (option int))
    [ Some 0; Some 24; Some 32; None ]
    [
      offset_in file m;
      offset_in file (sub m 24 64);
      offset_in file halves;
      offset_in file (B.create Nx_dtype.uint8 64);
    ];
  halves

let[@inline never] map_and_drop file =
  let m = map file.B.path in
  B.register_file file m;
  B.unsafe_data_ptr m

let mapped_files =
  group "mapped files"
    [
      test
        "file_range follows every view by address, and a view alone keeps the \
         mapping" (fun () ->
          let file = mapped_file 4096 in
          let halves = ranges file in
          Gc.full_major ();
          equal (option int) (Some 32) (offset_in file halves);
          equal string
            (String.init 32 (fun i -> Char.chr (32 + i)))
            (Bytes.to_string (bytes_of halves));
          refuses
            [ (fun () -> B.register_file file (B.create Nx_dtype.uint8 16)) ]);
      (* The system tends to hand the same address out again. *)
      test "a record dies with its mapping" (fun () ->
          let first = mapped_file 4096 and second = mapped_file 4096 in
          for _ = 1 to 8 do
            let address = map_and_drop first in
            Gc.full_major ();
            let m = map second.path in
            if B.unsafe_data_ptr m = address then
              equal (option int) None (offset_in first m)
          done);
    ]

let () =
  exit
    (run "nx buffer"
       [ buffers; reinterpretation; packed; conversions; mapped_files ])
