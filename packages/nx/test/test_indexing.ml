(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Gathers and scatters by index tensors. Slicing and [set] are in
   test_values. *)

open Windtrap
open Nx_test

let ints = Ref.witness int32
let positions = Ref.witness int64
let shape = Gen.array ~size:(Gen.int_range 1 3) (Gen.int_range 1 4)
let iota s = Array.init (Ref.numel s) (fun i -> Int32.of_int (i + 1))
let tensor_of s = (Ref.create s (iota s), Nx.create Nx.int32 s (iota s))

let indices_tensor l =
  Nx.create Nx.int64 [| Array.length l |] (Array.map Int64.of_int l)

let far = 1 lsl 32

(* An index along an axis of [n], [n > 0]: from one past each end, or a position
   of the axis moved by a multiple of 2^32, which a 32-bit truncation would
   bring back to that position. *)
let index n =
  let open Gen in
  frequency
    [
      (4, int_range (-2) (n + 1));
      ( 1,
        let+ i = int_range 0 (n - 1) and+ k = of_list [ -2; -1; 1; 2 ] in
        i + (k * far) );
    ]

(* Demands that both kinds of index outside an axis of [n] were drawn. *)
let cover_outside n idx =
  cover "an index just outside its axis"
    (Array.exists (fun k -> (k < 0 || k >= n) && Int.abs k < far / 2) idx);
  cover "an index 2^32 from a position of its axis"
    (Array.exists (fun k -> Int.abs k >= far / 2) idx)

(* A shape, an axis of it, and indices along that axis. *)
let along =
  let open Gen in
  let* s = shape in
  let* axis = int_range 0 (Array.length s - 1) in
  let+ idx = array ~size:(int_range 0 5) (index s.(axis)) in
  (s, axis, idx)

(* A shape, an axis of it, and an index along that axis at each position. *)
let positioned =
  let open Gen in
  let* s = shape in
  let* axis = int_range 0 (Array.length s - 1) in
  let+ idx = array ~size:(constant (Ref.numel s)) (index s.(axis)) in
  (s, axis, idx)

let read r src =
  if Array.for_all2 (fun k d -> k >= 0 && k < d) src r.Ref.shape then
    Ref.get r src
  else 0l

let gathers =
  group "gathers"
    [
      prop "take reads the flattened tensor, and zero out of range"
        (let open Gen in
         let* s = shape in
         let+ idx = array ~size:(int_range 0 6) (index (Ref.numel s)) in
         (s, idx))
        (fun (s, idx) ->
          let r, t = tensor_of s in
          let n = Ref.numel s in
          equal ints
            (Ref.init
               [| Array.length idx |]
               (fun i ->
                 let k = idx.(i.(0)) in
                 if k >= 0 && k < n then r.data.(k) else 0l))
            (Ref.of_nx (Nx.take ~indices:(indices_tensor idx) t)));
      prop
        "take reads an int64 index inside its axis and zero outside it, \
         however far"
        ~examples:[ ([| 4 |], 0, [| far + 1; 2 - far; 1 |]) ]
        along
        (fun (s, axis, idx) ->
          cover_outside s.(axis) idx;
          let r, t = tensor_of s in
          let out = Array.copy s in
          out.(axis) <- Array.length idx;
          equal ints
            (Ref.init out (fun i ->
                 let src = Array.copy i in
                 src.(axis) <- idx.(i.(axis));
                 read r src))
            (Ref.of_nx (Nx.take ~axis ~indices:(indices_tensor idx) t)));
      prop "take_along_axis reads, at each position, the index found there"
        positioned (fun (s, axis, idx) ->
          let r, t = tensor_of s in
          let indices = Nx.create Nx.int64 s (Array.map Int64.of_int idx) in
          equal ints
            (Ref.init s (fun i ->
                 let src = Array.copy i in
                 src.(axis) <- idx.(Ref.ravel s i);
                 read r src))
            (Ref.of_nx (Nx.take_along_axis ~axis ~indices t)));
      prop "take without an axis gives a value of the indices' shape"
        ~examples:
          [ ([| 3 |], [||], [| 1 |]); ([| 2; 2 |], [| 2; 1 |], [| 3; 5 |]) ]
        (let open Gen in
         with_pp (fun ppf (s, at, idx) ->
             let ints a =
               String.concat "; " (Array.to_list (Array.map string_of_int a))
             in
             Format.fprintf ppf "([|%s|], [|%s|], [|%s|])" (ints s) (ints at)
               (ints idx))
         @@ let* s = shape in
            let* at = array ~size:(int_range 0 3) (int_range 0 3) in
            let+ idx =
              array ~size:(constant (Ref.numel at)) (index (Ref.numel s))
            in
            (s, at, idx))
        (fun (s, at, idx) ->
          cover "scalar indices" (at = [||]);
          cover "indices of rank 2 or more" (Array.length at >= 2);
          let r, t = tensor_of s in
          let n = Ref.numel s in
          let indices = Nx.create Nx.int64 at (Array.map Int64.of_int idx) in
          equal ints
            (Ref.init at (fun i ->
                 let k = idx.(Ref.ravel at i) in
                 if k >= 0 && k < n then r.data.(k) else 0l))
            (Ref.of_nx (Nx.take ~indices t)));
      test "take without an axis reads a transposed tensor" (fun () ->
          let r, t = tensor_of [| 2; 3 |] in
          let indices = [| 1; 4 |] in
          equal ints
            (Ref.take ~zero:0l indices (Ref.transpose r))
            (Ref.of_nx
               (Nx.take ~indices:(indices_tensor indices) (Nx.transpose t))));
      test "take_along_axis refuses indices of another rank" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.take_along_axis ~axis:0 ~indices:(indices_tensor [| 0 |])
                (Nx.zeros Nx.int32 [| 2; 2 |])));
    ]

let scatters =
  group "scatters"
    [
      prop
        "scatter writes each value at its index, the last one winning, and \
         drops every update outside its axis, however far; a scalar value is \
         broadcast" (Gen.triple positioned Gen.bool Gen.bool)
        (fun ((s, axis, idx), add, scalar) ->
          let n = s.(axis) in
          cover_outside n idx;
          let r, t = tensor_of s in
          let values =
            Ref.init s (fun i ->
                if scalar then 100l else Int32.of_int (100 * (Ref.ravel s i + 1)))
          in
          let expected = Array.copy r.data in
          for i = 0 to Ref.numel s - 1 do
            let k = idx.(i) in
            if k >= 0 && k < n then begin
              let dst = Ref.unravel s i in
              dst.(axis) <- k;
              let j = Ref.ravel s dst in
              expected.(j) <-
                (if add then Int32.add expected.(j) values.data.(i)
                 else values.data.(i))
            end
          done;
          equal ints (Ref.create s expected)
            (Ref.of_nx
               (Nx.scatter
                  ~mode:(if add then `Add else `Set)
                  ~axis
                  ~indices:(Nx.create Nx.int64 s (Array.map Int64.of_int idx))
                  ~values:
                    (if scalar then Nx.scalar Nx.int32 100l
                     else Nx.create Nx.int32 s values.data)
                  t)));
      test "scatter's additions run from +0, and it keeps an unreached -0"
        (fun () ->
          let v xs = Nx.create Nx.float32 [| Array.length xs |] xs in
          let add ~unique_indices indices values =
            Nx.scatter ~mode:`Add ~unique_indices ~axis:0
              ~indices:(Nx.create Nx.int64 [| Array.length indices |] indices)
              ~values:(v values)
              (v [| -0.; -0.; -0. |])
          in
          List.iter
            (fun unique_indices ->
              let msg = Printf.sprintf "unique_indices = %b" unique_indices in
              equal ~msg (tensor float_exact) (v [| 0.; -0.; 0. |])
                (add ~unique_indices [| 0L; 2L |] [| -0.; -0. |]))
            [ false; true ];
          equal ~msg:"duplicates" (tensor float_exact) (v [| 0.; -0.; -0. |])
            (add ~unique_indices:false [| 0L; 0L |] [| -0.; -0. |]));
      group "scatter Add of integers wraps at their width"
        (List.map
           (fun (Int_dtype d) ->
             let value = int_value ~bits:d.bits ~signed:d.signed in
             let drawn =
               let open Gen in
               let* m = int_range 0 12 in
               let+ updates = array ~size:(constant m) value
               and+ idx = array ~size:(constant m) (int_range (-2) 4)
               and+ into = array ~size:(constant 4) value
               and+ flipped = bool in
               (updates, idx, into, flipped)
             in
             prop d.name drawn (fun (updates, idx, into, flipped) ->
                 let expected = Array.copy into in
                 Array.iteri
                   (fun i k ->
                     if k >= 0 && k < 4 then
                       expected.(k) <-
                         wrap ~bits:d.bits ~signed:d.signed
                           (Int64.add expected.(k) updates.(i)))
                   idx;
                 let m = Array.length updates in
                 (* Flipped operands are the same arrays through negative
                    strides. *)
                 let flip x =
                   if flipped then Nx.flip (Nx.contiguous (Nx.flip x)) else x
                 in
                 let vector a =
                   flip (Nx.create d.dtype [| m |] (Array.map d.of_i64 a))
                 in
                 let indices =
                   flip
                     (Nx.create Nx.int64 [| m |] (Array.map Int64.of_int idx))
                 in
                 equal (array d.exact)
                   (Array.map d.of_i64 expected)
                   (Nx.to_array
                      (Nx.scatter ~mode:`Add ~axis:0 ~indices
                         ~values:(vector updates)
                         (Nx.create d.dtype [| 4 |]
                            (Array.map d.of_i64 into))))))
           int_dtypes);
      test "scatter refuses indices of another rank" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.scatter ~axis:0 ~indices:(indices_tensor [| 0 |])
                ~values:(Nx.zeros Nx.int32 [| 1 |])
                (Nx.zeros Nx.int32 [| 2; 2 |])));
      prop "scatter with unique indices writes as the default does" along
        (fun (s, axis, _) ->
          let _, t = tensor_of s in
          let n = s.(axis) in
          (* Each lane takes the axis positions in reverse, which are unique. *)
          let positions =
            Nx.init Nx.int64 s (fun i -> Int64.of_int (n - 1 - i.(axis)))
          in
          let values = Nx.neg t in
          equal (tensor int32)
            (Nx.scatter ~axis ~indices:positions ~values t)
            (Nx.scatter ~unique_indices:true ~axis ~indices:positions ~values t));
      test "scatter refuses indices whose shape differs off the axis" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.scatter ~axis:0
                ~indices:(Nx.zeros Nx.int64 [| 1; 3 |])
                ~values:(Nx.zeros Nx.int32 [| 1; 3 |])
                (Nx.zeros Nx.int32 [| 2; 2 |])));
    ]

(* Scatters by extremes. A tensor's elements are drawn from a pool of values of
   its dtype, each of distinct bits, and named by their place in the pool, so a
   result compares bit for bit: NaN payloads and the signs of zeros included.
   [value] orders the pool as [maximum] does, NaN as [nan]. *)
type pool =
  | Pool : {
      name : string;
      dtype : ('a, 'b) Nx.dtype;
      values : float array;
      make : int array -> int array -> ('a, 'b) Nx.t;
      cells : ('a, 'b) Nx.t -> int array;
    }
      -> pool

(* The place in [bits] of each of [read]'s patterns. *)
let places bits read =
  Array.map (fun b -> Option.get (Array.find_index (( = ) b) bits)) read

let float_pool name dtype bits values ~to_bits ~of_bits =
  Pool
    {
      name;
      dtype;
      values;
      make = (fun shape c -> of_bits shape (Array.map (Array.get bits) c));
      cells = (fun t -> places bits (to_bits t));
    }

(* Every pool holds -inf, -1, -0, 0, 2.5, inf and three NaNs of distinct
   payloads, one negative. *)
let float_values =
  [| neg_infinity; -1.; -0.; 0.; 2.5; infinity; nan; nan; nan |]

let pools =
  let wide (type b) (dtype : (float, b) Nx.dtype) unsigned bits =
    float_pool (Nx_dtype.to_string dtype) dtype bits float_values
      ~to_bits:(fun t -> Nx.to_array (Nx.bitcast unsigned t))
      ~of_bits:(fun shape b -> Nx.bitcast dtype (Nx.create unsigned shape b))
  in
  let half dtype bits = wide dtype Nx.uint16 bits in
  [
    wide Nx.float64 Nx.uint64
      [|
        0xfff0000000000000L;
        0xbff0000000000000L;
        0x8000000000000000L;
        0L;
        0x4004000000000000L;
        0x7ff0000000000000L;
        0x7ff8000000000001L;
        0x7ff8000000000002L;
        0xfff8000000000003L;
      |];
    wide Nx.float32 Nx.uint32
      [|
        0xff800000l;
        0xbf800000l;
        0x80000000l;
        0l;
        0x40200000l;
        0x7f800000l;
        0x7fc00001l;
        0x7fc00002l;
        0xffc00003l;
      |];
    half Nx.float16
      [| 0xfc00; 0xbc00; 0x8000; 0; 0x4100; 0x7c00; 0x7e01; 0x7e02; 0xfe03 |];
    half Nx.bfloat16
      [| 0xff80; 0xbf80; 0x8000; 0; 0x4020; 0x7f80; 0x7fc1; 0x7fc2; 0xffc3 |];
    (let v = [| Int32.min_int; -1l; 0l; 1l; Int32.max_int |] in
     Pool
       {
         name = "int32";
         dtype = Nx.int32;
         values = Array.map Int32.to_float v;
         make =
           (fun shape c -> Nx.create Nx.int32 shape (Array.map (Array.get v) c));
         cells = (fun t -> places v (Nx.to_array t));
       });
    (let v = [| 0L; 1L; Int64.max_int; Int64.min_int; -1L |] in
     Pool
       {
         name = "uint64";
         dtype = Nx.uint64;
         values = [| 0.; 1.; 0x1p63 -. 1024.; 0x1p63; 0x1p64 |];
         make =
           (fun shape c ->
             Nx.create Nx.uint64 shape (Array.map (Array.get v) c));
         cells = (fun t -> places v (Nx.to_array t));
       });
    (let v = [| false; true |] in
     Pool
       {
         name = "bool";
         dtype = Nx.bool;
         values = [| 0.; 1. |];
         make =
           (fun shape c -> Nx.create Nx.bool shape (Array.map (Array.get v) c));
         cells = (fun t -> places v (Nx.to_array t));
       });
  ]

(* Whether the update [b] replaces the element [a] under [`Max] ([greater]) or
   [`Min]: it is the extreme, -0 below +0 and a NaN beyond every number, and [a]
   is not a NaN. *)
let wins ~greater a b =
  (not (Float.is_nan a))
  && (Float.is_nan b
     || (if greater then b > a else b < a)
     || (b = a && Float.sign_bit a = greater && Float.sign_bit b <> greater))

(* A shape, an axis of it, the element and the update at each position, and an
   index along the axis at each position. *)
let pooled size =
  let open Gen in
  let* s, axis, idx = positioned in
  let cell = int_range 0 (size - 1) in
  let+ into = array ~size:(constant (Ref.numel s)) cell
  and+ updates = array ~size:(constant (Ref.numel s)) cell in
  (s, axis, idx, into, updates)

let scatters_by_extremes =
  let extreme (Pool p) =
    prop
      (p.name
     ^ " scatter Max and Min keep the extreme of each position and its \
        updates, its bits included")
      (Gen.pair (pooled (Array.length p.values)) Gen.bool)
      (fun ((s, axis, idx, into, updates), greater) ->
        let n = s.(axis) in
        let expected = Array.copy into in
        Array.iteri
          (fun i k ->
            if k >= 0 && k < n then begin
              let dst = Ref.unravel s i in
              dst.(axis) <- k;
              let j = Ref.ravel s dst in
              if wins ~greater p.values.(expected.(j)) p.values.(updates.(i))
              then expected.(j) <- updates.(i)
            end)
          idx;
        equal (array int) expected
          (p.cells
             (Nx.scatter
                ~mode:(if greater then `Max else `Min)
                ~axis
                ~indices:(Nx.create Nx.int64 s (Array.map Int64.of_int idx))
                ~values:(p.make s updates) (p.make s into))))
  in
  let unique (Pool p) =
    prop
      (p.name ^ " scatter Max and Min do not depend on unique_indices")
      (Gen.pair (pooled (Array.length p.values)) Gen.bool)
      (fun ((s, axis, _, into, updates), greater) ->
        let n = s.(axis) in
        let positions =
          Nx.init Nx.int64 s (fun i -> Int64.of_int (n - 1 - i.(axis)))
        in
        let scatter unique_indices =
          p.cells
            (Nx.scatter
               ~mode:(if greater then `Max else `Min)
               ~unique_indices ~axis ~indices:positions
               ~values:(p.make s updates) (p.make s into))
        in
        equal (array int) (scatter false) (scatter true))
  in
  (* Without NaN, permuting the updates and their indices alike keeps the
     result. *)
  let order (Pool p) =
    let numbers =
      List.filter
        (fun c -> not (Float.is_nan p.values.(c)))
        (List.init (Array.length p.values) Fun.id)
    in
    let drawn =
      let open Gen in
      let* updates = array ~size:(int_range 0 8) (of_list numbers) in
      let m = Array.length updates in
      let+ idx = array ~size:(constant m) (int_range (-1) 3)
      and+ into = array ~size:(constant 3) (of_list numbers)
      and+ perm = permutation (List.init m Fun.id)
      and+ greater = bool in
      (updates, idx, into, Array.of_list perm, greater)
    in
    prop
      (p.name
     ^ " without NaN, the order of the updates does not change scatter Max and \
        Min") drawn (fun (updates, idx, into, perm, greater) ->
        let m = Array.length updates in
        let scatter updates idx =
          p.cells
            (Nx.scatter
               ~mode:(if greater then `Max else `Min)
               ~axis:0
               ~indices:
                 (Nx.create Nx.int64 [| m |] (Array.map Int64.of_int idx))
               ~values:(p.make [| m |] updates) (p.make [| 3 |] into))
        in
        equal (array int) (scatter updates idx)
          (scatter
             (Array.map (Array.get updates) perm)
             (Array.map (Array.get idx) perm)))
  in
  let f64_bits bits =
    Nx.bitcast Nx.float64 (Nx.create Nx.uint64 [| Array.length bits |] bits)
  in
  group "scatter by extremes"
    (List.map extreme pools @ List.map unique pools @ List.map order pools
    @ [
        test
          "a NaN element keeps its payload, and a number takes the first NaN \
           update's" (fun () ->
            let updates =
              f64_bits [| 0x7ff8000000000002L; 0x7ff8000000000003L |]
            in
            let scatter mode into =
              Nx.to_array
                (Nx.bitcast Nx.uint64
                   (Nx.scatter ~mode ~axis:0
                      ~indices:(Nx.zeros Nx.int64 [| 2 |])
                      ~values:updates (f64_bits [| into |])))
            in
            List.iter
              (fun mode ->
                equal (array int64) [| 0x7ff8000000000001L |]
                  (scatter mode 0x7ff8000000000001L);
                equal (array int64) [| 0x7ff8000000000002L |]
                  (scatter mode (Int64.bits_of_float 1.)))
              [ `Max; `Min ]);
        test "scatter Max of -0 and +0 is +0, and Min is -0" (fun () ->
            let scatter mode into update =
              Nx.to_array
                (Nx.scatter ~mode ~axis:0
                   ~indices:(Nx.zeros Nx.int64 [| 1 |])
                   ~values:(Nx.create Nx.float32 [| 1 |] [| update |])
                   (Nx.create Nx.float32 [| 1 |] [| into |]))
            in
            List.iter
              (fun (into, update) ->
                equal (array float_exact) [| 0. |] (scatter `Max into update);
                equal (array float_exact) [| -0. |] (scatter `Min into update))
              [ (-0., 0.); (0., -0.) ]);
        test "scatter Max and Min refuse complex numbers" (fun () ->
            List.iter
              (fun mode ->
                raises_invalid_arg (fun () ->
                    Nx.scatter ~mode ~axis:0
                      ~indices:(Nx.zeros Nx.int64 [| 1 |])
                      ~values:(Nx.zeros Nx.complex64 [| 1 |])
                      (Nx.zeros Nx.complex64 [| 1 |])))
              [ `Max; `Min ]);
      ])

(* float16, bfloat16 and float8 additions accumulate in float32 and round once
   per position. *)
let narrow_additions =
  let ones dtype k =
    Nx.scatter ~mode:`Add ~axis:0
      ~indices:(Nx.zeros Nx.int64 [| k |])
      ~values:(Nx.ones dtype [| k |]) (Nx.zeros dtype [| 1 |])
  in
  (* Multiples of 1/16 in [-8, 8], exact in every narrow dtype here and summed
     exactly in float32. *)
  let sixteenths n =
    Gen.array ~size:(Gen.constant n)
      (Gen.map (fun k -> float_of_int k /. 16.) (Gen.int_range (-128) 128))
  in
  let sums (type b) name (dtype : (float, b) Nx.dtype) =
    prop
      (name ^ " scatter Add is the float32 sum rounded once")
      (let open Gen in
       let* m = int_range 0 40 in
       let+ updates = sixteenths m
       and+ idx = array ~size:(constant m) (int_range (-1) 3)
       and+ into = sixteenths 3 in
       (updates, idx, into))
      (fun (updates, idx, into) ->
        let m = Array.length updates in
        let indices = Nx.create Nx.int64 [| m |] (Array.map Int64.of_int idx) in
        let scatter dt =
          Nx.scatter ~mode:`Add ~axis:0 ~indices
            ~values:(Nx.create dt [| m |] updates)
            (Nx.create dt [| 3 |] into)
        in
        equal (tensor int)
          (Nx.bitcast Nx.uint16 (Nx.cast dtype (scatter Nx.float32)))
          (Nx.bitcast Nx.uint16 (scatter dtype)))
  in
  group "narrow additions"
    [
      test "4096 float16 ones added into one position give 4096" (fun () ->
          equal (array float_exact) [| 4096. |]
            (Nx.to_array (ones Nx.float16 4096)));
      test "512 bfloat16 ones and 32 float8_e4m3 ones add up exactly" (fun () ->
          equal (array float_exact) [| 512. |]
            (Nx.to_array (ones Nx.bfloat16 512));
          equal (array float_exact) [| 32. |]
            (Nx.to_array (ones Nx.float8_e4m3 32)));
      test
        "a position no update reaches keeps its bits, and a reached -0 plus -0 \
         is +0" (fun () ->
          let into =
            Nx.bitcast Nx.float16
              (Nx.create Nx.uint16 [| 3 |] [| 0x8000; 0x7e05; 0x8000 |])
          in
          let y =
            Nx.scatter ~mode:`Add ~axis:0
              ~indices:(Nx.create Nx.int64 [| 2 |] [| 2L; 2L |])
              ~values:(Nx.full Nx.float16 [| 2 |] (-0.))
              into
          in
          equal (array int) [| 0x8000; 0x7e05; 0 |]
            (Nx.to_array (Nx.bitcast Nx.uint16 y)));
      sums "float16" Nx.float16;
      sums "bfloat16" Nx.bfloat16;
    ]

(* Counts of a dtype, drawn as OCaml ints in [0, 3]. *)
type counted = Counted : string * ('a, 'b) Nx.dtype * (int -> 'a) -> counted

let counted =
  [
    Counted ("bool", Nx.bool, fun c -> c > 0);
    Counted ("int4", Nx.int4, Fun.id);
    Counted ("int8", Nx.int8, Fun.id);
    Counted ("uint8", Nx.uint8, Fun.id);
    Counted ("int32", Nx.int32, Int32.of_int);
    Counted ("int64", Nx.int64, Int64.of_int);
    Counted ("uint64", Nx.uint64, Int64.of_int);
  ]

let positions_of_counts =
  let repeats (Counted (name, dtype, of_int)) =
    let drawn =
      let open Gen in
      let* n =
        frequency
          [
            (4, int_range 0 20);
            (1, int_range 4090 4100);
            (1, int_range 8990 9000);
          ]
      in
      let bound = if name = "bool" then 1 else 3 in
      array ~size:(constant n) (int_range 0 bound)
    in
    prop (name ^ " positions repeats each index by its count") drawn (fun c ->
        let expected =
          List.concat_map
            (fun i -> List.init c.(i) (Fun.const (Int64.of_int i)))
            (List.init (Array.length c) Fun.id)
        in
        equal (array int64) (Array.of_list expected)
          (Nx.to_array
             (Nx.positions
                (Nx.create dtype [| Array.length c |] (Array.map of_int c)))))
  in
  let refuses message c =
    raises
      (Invalid_argument ("positions: " ^ message))
      (fun () -> ignore (Nx.positions c))
  in
  group "positions"
    (List.map repeats counted
    @ [
        test
          "positions refuses a negative count, a uint64 count past int64, a \
           sum past int64, a float and a matrix" (fun () ->
            refuses "count -1 is negative"
              (Nx.create Nx.int32 [| 2 |] [| 2l; -1l |]);
            refuses "a count is past int64's range"
              (Nx.create Nx.uint64 [| 2 |] [| 1L; Int64.min_int |]);
            refuses "the counts sum past int64's range"
              (Nx.create Nx.int64 [| 2 |] [| Int64.max_int; 1L |]);
            refuses "counts of dtype float32, not boolean or integer"
              (Nx.create Nx.float32 [| 1 |] [| 1. |]);
            refuses "counts of shape [2,2], not 1-D"
              (Nx.zeros Nx.bool [| 2; 2 |]));
        test "positions of nothing is empty" (fun () ->
            equal (array int64) [||]
              (Nx.to_array (Nx.positions (Nx.zeros Nx.bool [| 0 |])));
            equal (array int64) [||]
              (Nx.to_array (Nx.positions (Nx.zeros Nx.int32 [| 2 |]))));
        test "a count past the last index fills one run at the end" (fun () ->
            equal (array int64) [| 2L; 2L; 2L; 2L; 2L |]
              (Nx.to_array
                 (Nx.positions (Nx.create Nx.int32 [| 3 |] [| 0l; 0l; 5l |]))));
      ])

let selections =
  group "selections"
    [
      prop "compress keeps the positions of an axis where the condition holds"
        along (fun (s, axis, _) ->
          let r, t = tensor_of s in
          let cond = Array.init s.(axis) (fun i -> i mod 3 <> 1) in
          equal ints
            (Ref.slice
               (List.init axis (fun _ -> Nx.A)
               @ [
                   Nx.L
                     (List.filter
                        (fun i -> cond.(i))
                        (List.init s.(axis) Fun.id));
                 ])
               r)
            (Ref.of_nx
               (Nx.compress ~axis
                  ~condition:(Nx.create Nx.bool [| s.(axis) |] cond)
                  t)));
      prop
        "extract lists, in row-major order, the elements where the condition \
         holds"
        shape (fun s ->
          let r, t = tensor_of s in
          let cond = Array.map (fun v -> Int32.rem v 3l <> 0l) r.data in
          let kept =
            List.filter_map
              (fun (c, v) -> if c then Some v else None)
              (List.combine (Array.to_list cond) (Array.to_list r.data))
          in
          equal ints
            (Ref.create [| List.length kept |] (Array.of_list kept))
            (Ref.of_nx (Nx.extract ~condition:(Nx.create Nx.bool s cond) t)));
      prop
        "nonzero and argwhere list the coordinates of non-zero elements, in \
         row-major order"
        (Gen.array ~size:(Gen.int_range 0 3) (Gen.int_range 0 4))
        (fun s ->
          let r =
            Ref.init s (fun i -> Int32.of_int ((Ref.ravel s i + 1) mod 3))
          in
          let t = Nx.create Nx.int32 s r.data in
          let coords =
            List.filter_map
              (fun i ->
                if r.data.(i) <> 0l then Some (Ref.unravel s i) else None)
              (List.init (Ref.numel s) Fun.id)
          in
          let k = List.length coords and n = Array.length s in
          let rows = Array.of_list coords in
          equal positions
            (Ref.init [| k; n |] (fun i -> Int64.of_int rows.(i.(0)).(i.(1))))
            (Ref.of_nx (Nx.argwhere t));
          Array.iteri
            (fun d axis ->
              equal
                ~msg:(Printf.sprintf "axis %d" d)
                positions
                (Ref.init [| k |] (fun i -> Int64.of_int rows.(i.(0)).(d)))
                (Ref.of_nx axis))
            (Nx.nonzero t));
      prop
        "compress without an axis keeps the flattened positions where the \
         condition holds"
        shape (fun s ->
          let r, t = tensor_of s in
          let cond = Array.init (Ref.numel s) (fun i -> i mod 3 <> 1) in
          equal ints (Ref.compress cond r)
            (Ref.of_nx
               (Nx.compress
                  ~condition:(Nx.create Nx.bool [| Ref.numel s |] cond)
                  t)));
      test "compress and extract without an axis read a transposed tensor"
        (fun () ->
          let r, t = tensor_of [| 2; 3 |] in
          let cond = [| true; false; false; true; true; false |] in
          let condition = Nx.create Nx.bool [| 6 |] cond in
          let expected = Ref.compress cond (Ref.transpose r) in
          equal ints expected
            (Ref.of_nx (Nx.compress ~condition (Nx.transpose t)));
          equal ints expected
            (Ref.of_nx
               (Nx.extract
                  ~condition:(Nx.reshape [| 3; 2 |] condition)
                  (Nx.transpose t))));
      test
        "compress refuses a condition of another length, with or without an \
         axis" (fun () ->
          let t = Nx.zeros Nx.int32 [| 2; 3 |] in
          List.iter
            (fun (axis, n) ->
              raises_invalid_arg (fun () ->
                  Nx.compress ?axis ~condition:(Nx.ones Nx.bool [| n |]) t))
            [ (None, 5); (None, 7); (Some 1, 2); (Some 1, 4) ]);
      test "compress refuses a 2-D condition, even of the right size" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.compress
                ~condition:(Nx.ones Nx.bool [| 2; 3 |])
                (Nx.zeros Nx.int32 [| 2; 3 |])));
      test "compress keeps nothing or everything, along an empty axis too"
        (fun () ->
          let r, t = tensor_of [| 2; 3 |] in
          let all = Nx.ones Nx.bool [| 3 |]
          and none = Nx.zeros Nx.bool [| 3 |] in
          equal ints r (Ref.of_nx (Nx.compress ~axis:1 ~condition:all t));
          equal (array int) [| 2; 0 |]
            (Nx.shape (Nx.compress ~axis:1 ~condition:none t));
          equal (array int) [| 0; 3 |]
            (Nx.shape
               (Nx.compress ~axis:0 ~condition:(Nx.zeros Nx.bool [| 0 |])
                  (Nx.zeros Nx.int32 [| 0; 3 |]))));
      test
        "nonzero of a scalar is empty, and argwhere of a scalar has no column"
        (fun () ->
          equal int 0 (Array.length (Nx.nonzero (Nx.scalar Nx.int32 3l)));
          equal (array int) [| 1; 0 |]
            (Nx.shape (Nx.argwhere (Nx.scalar Nx.int32 3l)));
          equal (array int) [| 0; 0 |]
            (Nx.shape (Nx.argwhere (Nx.scalar Nx.int32 0l))));
      test "nonzero takes NaN as non-zero, and -0 and complex zero as zero"
        (fun () ->
          let x = Nx.create Nx.float64 [| 4 |] [| -0.; Float.nan; 0.; 2. |] in
          equal (array int64) [| 1L; 3L |] (Nx.to_array (Nx.nonzero x).(0));
          let z = Nx.create Nx.complex64 [| 2 |] Complex.[| zero; one |] in
          equal (array int64) [| 1L |] (Nx.to_array (Nx.nonzero z).(0)));
      test "extract flattens a condition of the same size and another shape"
        (fun () ->
          let r, t = tensor_of [| 2; 3 |] in
          let cond = [| true; false; false; true; true; false |] in
          equal ints (Ref.compress cond r)
            (Ref.of_nx
               (Nx.extract ~condition:(Nx.create Nx.bool [| 6 |] cond) t)));
      test "extract refuses a condition of another size" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.extract
                ~condition:(Nx.create Nx.bool [| 3 |] [| true; false; true |])
                (Nx.zeros Nx.int32 [| 2 |])));
    ]

let extremes =
  let x = Nx.create Nx.int32 [| 3 |] [| 1l; 2l; 3l |] in
  let at = Nx.create Nx.int64 [| 3 |] [| Int64.min_int; 1L; Int64.max_int |] in
  group "extreme indices"
    [
      test "Int64.min_int and Int64.max_int read zero" (fun () ->
          let expected = Nx.create Nx.int32 [| 3 |] [| 0l; 2l; 0l |] in
          equal (tensor int32) expected (Nx.take ~indices:at x);
          equal (tensor int32) expected
            (Nx.take_along_axis ~axis:0 ~indices:at x));
      test "updates at Int64.min_int and Int64.max_int are dropped" (fun () ->
          equal (tensor int32)
            (Nx.create Nx.int32 [| 3 |] [| 1l; 9l; 3l |])
            (Nx.scatter ~axis:0 ~indices:at ~values:(Nx.scalar Nx.int32 9l) x));
    ]

(* Each case holds one tensor of 2^31 + 2 bytes, after collecting the ones
   before it. *)
let past_int32 =
  let n = (1 lsl 31) + 2 in
  let zeros () = Nx.broadcast_to [| n |] (Nx.scalar Nx.uint8 0) in
  let at x i = Nx.item [ i ] x in
  group "indices past 2^31"
    [
      slow "take, argmax and argmin reach positions past 2^31" (fun () ->
          Gc.full_major ();
          let x = Nx.pad [| (n - 1, 0) |] 0 (Nx.ones Nx.uint8 [| 1 |]) in
          equal int64 (Int64.of_int (n - 1)) (Nx.item [] (Nx.argmax x));
          equal int64 0L (Nx.item [] (Nx.argmin x));
          equal (array int) [| 1; 0; 0 |]
            (Nx.to_array
               (Nx.take ~indices:(indices_tensor [| n - 1; n; -1 |]) x)));
      slow "scatter writes past 2^31 and drops what a truncation would alias"
        (fun () ->
          Gc.full_major ();
          let y =
            Nx.scatter ~axis:0
              ~indices:(indices_tensor [| n - 1; far + 3 |])
              ~values:(Nx.create Nx.uint8 [| 2 |] [| 7; 9 |])
              (zeros ())
          in
          equal int 7 (at y (n - 1));
          equal int 0 (at y 3));
      slow "a D window starts past 2^31" (fun () ->
          Gc.full_major ();
          let y =
            Nx.set
              [ Nx.D (Nx.scalar Nx.int64 (Int64.of_int (n - 2)), 2) ]
              (Nx.ones Nx.uint8 [| 2 |]) (zeros ())
          in
          equal (array int) [| 0; 1; 1 |]
            (Array.map (at y) [| n - 3; n - 2; n - 1 |]));
    ]

(* [D] reads a window whose start is a run-time scalar, clamped so the window
   fits. *)
let windows =
  let window start len =
    Nx.slice
      [ Nx.D (Nx.scalar Nx.int64 (Int64.of_int start), len) ]
      (Nx.arange Nx.int32 0 5 1)
  in
  let expected first len =
    Nx.init Nx.int32 [| len |] (fun i -> Int32.of_int (first + i.(0)))
  in
  group "dynamic windows"
    [
      cases "a window starts at its start, clamped into the axis"
        ~name:(fun (start, len, _) ->
          Printf.sprintf "start %d, length %d" start len)
        [
          (0, 2, 0);
          (2, 2, 2);
          (4, 2, 3);
          (-3, 2, 0);
          (9, 5, 0);
          (1, 0, 1);
          (far + 1, 2, 3);
          (1 - far, 2, 0);
        ]
        (fun (start, len, first) ->
          equal (tensor int32) (expected first len) (window start len));
      test "a window longer than its axis is refused" (fun () ->
          raises_invalid_arg (fun () -> window 0 6));
      test "set writes a dynamic window" (fun () ->
          let x = Nx.zeros Nx.int32 [| 4 |] in
          equal (tensor int32)
            (Nx.create Nx.int32 [| 4 |] [| 0l; 0l; 7l; 7l |])
            (Nx.set
               [ Nx.D (Nx.scalar Nx.int64 5L, 2) ]
               (Nx.full Nx.int32 [| 2 |] 7l)
               x));
    ]

(* nx.mli leaves the bounds of a range open; [R] clamps them into the axis, as
   Python slices do, and [Ref] does the same for [Rs]. *)
let stepped_ranges =
  group "stepped ranges"
    [
      cases "a stepped range starting outside its axis selects as R does"
        ~name:(fun (s : Nx.index) -> Format.asprintf "%a" pp_index s)
        [ Rs (-10, 3, 1); Rs (-10, 4, 2); Rs (10, 0, -1); Rs (10, 0, -2) ]
        (fun spec ->
          let r, t = tensor_of [| 4 |] in
          equal ints (Ref.slice [ spec ] r) (Ref.of_nx (Nx.slice [ spec ] t)));
    ]

let () =
  exit
    (run "nx indexing"
       [
         gathers;
         scatters;
         scatters_by_extremes;
         narrow_additions;
         extremes;
         positions_of_counts;
         selections;
         windows;
         stepped_ranges;
         past_int32;
       ])
