(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Packed dtypes in compiled calls. A compiled call computes on [bit], [int4]
   and [uint4] as eager does, bit for bit, over every layout of its arguments,
   views at sub-byte offsets and odd lengths included, and writes the bits of a
   new result's last byte past its last element as 0. [int4] and [uint4] compute
   as [int8] and [uint8] reduced modulo 16, under the integer rules of every
   width. They compile under vmap and grad, over several devices, and on Metal
   where the machine has it. *)

open Windtrap
open Nx_test

let host t = Nx.place Nx.Placement.host t

(* Signatures, made at each use: each instance has dtypes of its own. *)
let one () = Nx.Ptree.(tensor @-> returns tensor)
let two () = Nx.Ptree.(tensor @-> tensor @-> returns tensor)

(* A value as the tests compare it: its dtype, shape and elements, read as
   int64, which holds every element these functions return exactly. *)
type seen = { dtype : string; shape : int array; elements : int64 array }

let seen t =
  let t = host t in
  {
    dtype = Nx_dtype.to_string (Nx.dtype t);
    shape = Nx.shape t;
    elements = Nx.to_array (Nx.cast Nx.int64 t);
  }

let pp_seen ppf s =
  Format.fprintf ppf "%s%a [%a]" s.dtype pp_shape s.shape
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       (fun ppf v -> Format.fprintf ppf "%Ld" v))
    (Array.to_list s.elements)

let seen_w = Testable.make ~pp:pp_seen ~equal:( = )

(* [agrees f g] checks that [g], [f] compiled, gives [f]'s value, or raises
   [Invalid_argument] where [f] does. *)
let agrees ?msg f g =
  match f () with
  | expected -> equal ?msg seen_w (seen expected) (seen (g ()))
  | exception Invalid_argument m ->
      raises_match ~msg:m Exn.invalid_arg (fun () -> ignore (g ()))

(* The bytes of the buffer [b], copied to the host. *)
let buffer_bytes b =
  let n = Nx_device.Buffer.nbytes b
  and u8 = Nx_dtype.Scalar.of_dtype Nx.uint8 in
  let host = Nx_device.Buffer.create Nx_device.host u8 n in
  Nx_device.Buffer.copy ~src:(Nx_device.Buffer.view b ~offset:0 u8 n) ~dst:host;
  Nx.to_array
    (Nx.of_shards Nx.Placement.host Nx.uint8
       (Nx_array.View.create [| n |])
       [ host ])

(* The bytes of [t]'s storage, from its first, for a value [t] that owns all of
   it. *)
let bytes t = buffer_bytes (storage t)

(* The bits of each device's last byte of [t] past the elements of its window,
   which a new value writes as 0: [0] where the window ends on a byte. *)
let tails t =
  let bufs, v = Nx.shards t in
  let bits = Nx_dtype.Scalar.(bitsize (of_dtype (Nx.dtype t))) in
  let used = Nx_array.View.numel v * bits mod 8 in
  List.map
    (fun b ->
      let bytes = buffer_bytes b in
      if used = 0 || Array.length bytes = 0 then 0
      else bytes.(Array.length bytes - 1) lsr used)
    bufs

(* Operands *)

(* A tensor of [shape] and [dtype] whose storage starts [offset] elements into a
   flat buffer: an offset within a byte for a packed dtype. *)
let offset dtype shape offset xs =
  let n = Array.length xs in
  let flat =
    Nx.create dtype
      [| n + offset |]
      (Array.append (Array.make offset xs.(0)) xs)
  in
  Nx.reshape shape (Nx.slice [ R (offset, offset + n) ] flat)

let shapes = Gen.array ~size:(Gen.int_range 0 3) (Gen.int_range 0 9)

(* Two operands of one shape under one drawn layout, their elements drawn from
   [value], their storages starting at drawn offsets. *)
let operands ~pp dtype value =
  let open Gen in
  let drawn =
    let* s = shapes in
    let n = Ref.numel s in
    let* xs = array ~size:(constant n) value in
    let* ys = array ~size:(constant n) value in
    let* o1 = int_range 0 9 in
    let* o2 = int_range 0 9 in
    let+ steps = layout in
    let make o xs =
      if n = 0 then lay_out steps (Nx.create dtype s xs)
      else lay_out steps (offset dtype s o xs)
    in
    ((steps, (o1, o2)), make o1 xs, make o2 ys)
  in
  with_pp
    (fun ppf ((steps, (o1, o2)), a, b) ->
      Format.fprintf ppf "from offsets %d and %d, %a:@ %a@ %a" o1 o2 pp_layout
        steps (Ref.pp pp) (Ref.of_nx a) (Ref.pp pp) (Ref.of_nx b))
    drawn

let nibbles ~signed =
  let lo = if signed then -8 else 0 in
  Gen.int_range lo (lo + 15)

(* Families of functions *)

(* A function of two operands of one packed dtype, by name: compiled, it gives
   eager's value. *)
type 'a family = { name : string; check : 'a -> 'a -> unit }

let family name f =
  {
    name;
    check =
      (fun a b -> agrees (fun () -> f a b) (fun () -> Rune.jit (two ()) f a b));
  }

(* Functions of [int4] or [uint4] operands, [dt]. *)
let integers (type b) (dt : (int, b) Nx.dtype) : (int, b) Nx.t family list =
  let along_first f t = if Nx.ndim t = 0 then t else f t in
  [
    family "add, sub, mul and neg" (fun a b ->
        Nx.add (Nx.mul a b) (Nx.neg (Nx.sub a b)));
    family "div and mod_, by zero included" (fun a b ->
        Nx.add (Nx.div a b) (Nx.mod_ a b));
    family "pow" Nx.pow;
    family "maximum, minimum, abs and sign" (fun a b ->
        Nx.maximum (Nx.abs a) (Nx.minimum (Nx.sign b) a));
    family "bitwise and, or, xor and not" (fun a b ->
        Nx.bitwise_xor (Nx.bitwise_and a b) (Nx.bitwise_or (Nx.bitwise_not a) b));
    family "shifts below the width" (fun a b ->
        Nx.add (Nx.lshift a 1) (Nx.rshift b 2));
    family "shifts by the width and past it" (fun a b ->
        Nx.add (Nx.lshift a 4) (Nx.rshift b 5));
    family "equal and less" (fun a b ->
        Nx.logical_or (Nx.equal a b) (Nx.less b a));
    family "where" (fun a b -> Nx.where (Nx.less a b) b a);
    family "a comparison, a quotient and an extreme of results that wrap"
      (fun a b ->
        Nx.where
          (Nx.less (Nx.add a b) a)
          (Nx.div (Nx.mul a b) (Nx.add_s b 1))
          (Nx.maximum (Nx.neg a) (Nx.add a a)));
    family "a sum and a product over every axis" (fun a b ->
        Nx.add (Nx.sum ~keepdims:true a) (Nx.prod ~keepdims:true b));
    family "a sum over the last axis" (fun a _ ->
        along_first (Nx.sum ~axes:[ -1 ]) a);
    family "max and min over every axis" (fun a b ->
        Nx.sub (Nx.max ~keepdims:true a) (Nx.min ~keepdims:true b));
    family "running sums and products" (fun a b ->
        along_first
          (fun a -> Nx.add (Nx.cumsum ~axis:0 a) (Nx.cumprod ~axis:0 b))
          a);
    family "argmax" (fun a _ -> Nx.argmax a);
    family "sort along the last axis" (fun a _ ->
        along_first (fun a -> fst (Nx.sort a)) a);
    family "casts from float32 and int16" (fun a b ->
        Nx.add
          (Nx.cast dt (Nx.mul_s (Nx.cast Nx.float32 a) 1.5))
          (Nx.cast dt (Nx.mul (Nx.cast Nx.int16 b) (Nx.cast Nx.int16 b))));
    family "a cast to int4" (fun a _ -> Nx.cast Nx.int4 a);
    family "a cast to uint4" (fun a _ -> Nx.cast Nx.uint4 a);
    family "a cast to bit" (fun a _ -> Nx.cast Nx.bit a);
    family "a cast to float32" (fun a _ -> Nx.cast Nx.float32 a);
    family "flip, concatenate and contiguous" (fun a b ->
        along_first
          (fun a -> Nx.contiguous (Nx.concatenate ~axis:0 [ Nx.flip a; b ]))
          a);
    family "pad" (fun a _ ->
        Nx.pad (Array.map (fun _ -> (1, 2)) (Nx.shape a)) (-3) a);
    family "take along the first axis" (fun a _ ->
        along_first
          (fun a ->
            let n = Nx.dim 0 a in
            Nx.take ~axis:0
              ~indices:
                (Nx.create Nx.int64 [| 3 |] [| 0L; Int64.of_int (n - 1); 0L |])
              a)
          a);
  ]

(* Functions of [bit] operands. *)
let bits : Nx.bit_t family list =
  let along_last f t = if Nx.ndim t = 0 then t else f t in
  [
    family "logical and, or, xor and not" (fun a b ->
        Nx.logical_xor (Nx.logical_and a b) (Nx.logical_or (Nx.logical_not a) b));
    family "bitwise and, or, xor and not" (fun a b ->
        Nx.bitwise_xor (Nx.bitwise_and a b) (Nx.bitwise_or (Nx.bitwise_not a) b));
    family "maximum and minimum" (fun a b -> Nx.maximum a (Nx.minimum a b));
    family "equal and less" (fun a b ->
        Nx.logical_and (Nx.equal a b) (Nx.less a b));
    family "where between bits" (fun a b ->
        Nx.where (Nx.cast Nx.bool a) b (Nx.logical_not b));
    family "a kept mask as a condition" (fun a b ->
        Nx.where (Nx.cast Nx.bool a) (Nx.cast Nx.int32 b)
          (Nx.full_like (Nx.cast Nx.int32 b) 7l));
    family "all and any over every axis" (fun a b ->
        Nx.logical_and (Nx.all a) (Nx.any b));
    family "max and min over the last axis" (fun a b ->
        along_last
          (fun a ->
            Nx.logical_xor (Nx.max ~axes:[ -1 ] a) (Nx.min ~axes:[ -1 ] b))
          a);
    family "count over every axis" (fun a _ -> Nx.count a);
    family "count over the last axis" (fun a _ ->
        along_last (fun a -> Nx.cast Nx.bit (Nx.count ~axes:[ -1 ] a)) a);
    family "casts to bool, int4, uint8 and float32" (fun a b ->
        Nx.add
          (Nx.cast Nx.float32 (Nx.add (Nx.cast Nx.int4 a) (Nx.cast Nx.int4 b)))
          (Nx.cast Nx.float32
             (Nx.add (Nx.cast Nx.uint8 a)
                (Nx.cast Nx.uint8 (Nx.cast Nx.bool b)))));
    family "a cast from int8" (fun a b ->
        Nx.cast Nx.bit (Nx.sub (Nx.cast Nx.int8 a) (Nx.cast Nx.int8 b)));
    family "flip, concatenate and contiguous" (fun a b ->
        along_last
          (fun a -> Nx.contiguous (Nx.concatenate ~axis:(-1) [ Nx.flip a; b ]))
          a);
    family "pad" (fun a _ ->
        Nx.pad (Array.map (fun _ -> (3, 2)) (Nx.shape a)) true a);
    family "take along the last axis" (fun a _ ->
        along_last
          (fun a ->
            let n = Nx.dim (-1) a in
            Nx.take ~axis:(-1)
              ~indices:
                (Nx.create Nx.int64 [| 3 |] [| Int64.of_int (n - 1); 0L; 1L |])
              a)
          a);
    family "sort and argmax along the last axis" (fun a _ ->
        along_last
          (fun a -> Nx.cast Nx.bit (Nx.argmax ~axis:(-1) (fst (Nx.sort a))))
          a);
  ]

let equals_eager name gen families =
  group name
    (List.map
       (fun f -> prop ~count:12 f.name gen (fun (_, a, b) -> f.check a b))
       families)

let int4_operands =
  operands ~pp:Format.pp_print_int Nx.int4 (nibbles ~signed:true)

let uint4_operands =
  operands ~pp:Format.pp_print_int Nx.uint4 (nibbles ~signed:false)

let bit_operands = operands ~pp:Format.pp_print_bool Nx.bit Gen.bool

let values =
  group "a compiled call computes eager's values"
    [
      equals_eager "int4" int4_operands (integers Nx.int4);
      equals_eager "uint4" uint4_operands (integers Nx.uint4);
      equals_eager "bit" bit_operands bits;
    ]

(* Four bits are a width *)

(* [width dt w gen] states that a function of [dt] operands, compiled, gives the
   function of [w] operands, the byte dtype of [dt]'s sign, reduced to [dt] when
   it returns its operands' dtype. *)
let width (type b c) name (dt : (int, b) Nx.dtype) (w : (int, c) Nx.dtype) gen =
  let law name check = prop ~count:12 name gen (fun (_, a, b) -> check a b) in
  let ring name f g =
    law name (fun a b ->
        agrees
          (fun () -> Nx.cast dt (g (Nx.cast w a) (Nx.cast w b)))
          (fun () -> Rune.jit (two ()) f a b))
  in
  let relation name f g =
    law name (fun a b ->
        agrees
          (fun () -> g (Nx.cast w a) (Nx.cast w b))
          (fun () -> Rune.jit (two ()) f a b))
  in
  let last t = if Nx.ndim t = 0 then [] else [ -1 ] in
  group name
    [
      ring "add" Nx.add Nx.add;
      ring "sub" Nx.sub Nx.sub;
      ring "mul" Nx.mul Nx.mul;
      ring "div" Nx.div Nx.div;
      ring "mod_" Nx.mod_ Nx.mod_;
      ring "pow" Nx.pow Nx.pow;
      ring "maximum" Nx.maximum Nx.maximum;
      ring "minimum" Nx.minimum Nx.minimum;
      ring "bitwise_and" Nx.bitwise_and Nx.bitwise_and;
      ring "bitwise_xor" Nx.bitwise_xor Nx.bitwise_xor;
      ring "neg" (fun a _ -> Nx.neg a) (fun a _ -> Nx.neg a);
      ring "abs" (fun a _ -> Nx.abs a) (fun a _ -> Nx.abs a);
      ring "sign" (fun a _ -> Nx.sign a) (fun a _ -> Nx.sign a);
      ring "lshift by 1, 4 and 9"
        (fun a _ ->
          Nx.add (Nx.lshift a 1) (Nx.add (Nx.lshift a 4) (Nx.lshift a 9)))
        (fun a _ ->
          Nx.add (Nx.lshift a 1) (Nx.add (Nx.lshift a 4) (Nx.lshift a 9)));
      ring "rshift by 1, 4 and 9"
        (fun a _ ->
          Nx.add (Nx.rshift a 1) (Nx.add (Nx.rshift a 4) (Nx.rshift a 9)))
        (fun a _ ->
          Nx.add (Nx.rshift a 1) (Nx.add (Nx.rshift a 4) (Nx.rshift a 9)));
      ring "sum over the last axis"
        (fun a _ -> Nx.sum ~axes:(last a) a)
        (fun a _ -> Nx.sum ~axes:(last a) a);
      ring "prod over the last axis"
        (fun a _ -> Nx.prod ~axes:(last a) a)
        (fun a _ -> Nx.prod ~axes:(last a) a);
      ring "cumsum over every element"
        (fun a _ -> Nx.cumsum a)
        (fun a _ -> Nx.cumsum a);
      ring "cumprod over every element"
        (fun a _ -> Nx.cumprod a)
        (fun a _ -> Nx.cumprod a);
      ring "max over every axis" (fun a _ -> Nx.max a) (fun a _ -> Nx.max a);
      ring "sort over every element"
        (fun a _ -> fst (Nx.sort (Nx.flatten a)))
        (fun a _ -> fst (Nx.sort (Nx.flatten a)));
      ring "where"
        (fun a b -> Nx.where (Nx.less a b) a b)
        (fun a b -> Nx.where (Nx.less a b) a b);
      relation "equal" Nx.equal Nx.equal;
      relation "less" Nx.less Nx.less;
      relation "argmin over every element"
        (fun a _ -> Nx.argmin a)
        (fun a _ -> Nx.argmin a);
      ring "matmul of a matrix by its transpose"
        (fun a b ->
          if Nx.ndim a <> 2 then a else Nx.matmul a (Nx.matrix_transpose b))
        (fun a b ->
          if Nx.ndim a <> 2 then a else Nx.matmul a (Nx.matrix_transpose b));
    ]

let widths =
  group "an int4 or uint4 function is the byte function reduced modulo 16"
    [
      width "int4 through int8" Nx.int4 Nx.int8 int4_operands;
      width "uint4 through uint8" Nx.uint4 Nx.uint8 uint4_operands;
    ]

(* Values the spec states *)

let stated =
  group "stated values"
    [
      test "a cast to bit of 3 and 5 is two true elements" (fun () ->
          let c = Nx.create Nx.int32 [| 2 |] [| 3l; 5l |] in
          equal (tensor bool)
            (Nx.create Nx.bool [| 2 |] [| true; true |])
            (Nx.cast Nx.bool (Rune.jit (one ()) (Nx.cast Nx.bit) c)));
      test "a float cast to int4 holds at 7, and 9 does not wrap to -7"
        (fun () ->
          let x =
            Nx.create Nx.float32 [| 5 |] [| 9.; -9.; 7.9; -8.5; Float.nan |]
          in
          equal (tensor int)
            (Nx.create Nx.int4 [| 5 |] [| 7; -8; 7; -8; 0 |])
            (Rune.jit (one ()) (Nx.cast Nx.int4) x));
      test "a float cast to uint4 holds at 0 and 15" (fun () ->
          let x =
            Nx.create Nx.float32 [| 4 |] [| 16.; -1.; 15.5; Float.infinity |]
          in
          equal (tensor int)
            (Nx.create Nx.uint4 [| 4 |] [| 15; 0; 15; 15 |])
            (Rune.jit (one ()) (Nx.cast Nx.uint4) x));
      test "an int4 literal is reduced modulo 16" (fun () ->
          let q = Nx.create Nx.int4 [| 3 |] [| 1; -2; 7 |] in
          equal (tensor int)
            (Nx.create Nx.int4 [| 3 |] [| -6; 7; 0 |])
            (Rune.jit (one ()) (fun q -> Nx.add_s q 9) q));
      test "a scatter of two bits into one byte keeps the other six" (fun () ->
          let m =
            Nx.cast Nx.bit
              (Nx.create Nx.bool [| 8 |]
                 [| true; false; false; true; true; false; false; true |])
          in
          let set m =
            Nx.scatter ~axis:0
              ~indices:(Nx.create Nx.int64 [| 2 |] [| 1L; 3L |])
              ~values:(Nx.create Nx.bit [| 2 |] [| true; false |])
              m
          in
          let expected =
            [| true; true; false; false; true; false; false; true |]
          in
          equal (tensor bool)
            (Nx.create Nx.bit [| 8 |] expected)
            (Rune.jit (one ()) set m);
          equal ~msg:"consumed" (tensor bool)
            (Nx.create Nx.bit [| 8 |] expected)
            (Rune.jit
               Nx.Ptree.(consumes tensor @@ returns tensor)
               set (Nx.copy m)));
      test
        "a consumed elementwise result of odd length takes its argument's \
         storage and clears its tail" (fun () ->
          (* A bit value over bytes whose bits past its 13 elements are set. *)
          let u8 = Nx.create Nx.uint8 [| 2 |] [| 0xAA; 0xFF |] in
          let b =
            Nx_device.Buffer.view (storage u8) ~offset:0
              (Nx_dtype.Scalar.of_dtype Nx.bit)
              13
          in
          let m =
            Nx.of_shards Nx.Placement.host Nx.bit
              (Nx_array.View.create [| 13 |])
              [ b ]
          in
          let expected = Nx.logical_not m in
          let before = Witness.addresses m in
          let r =
            Rune.jit
              Nx.Ptree.(consumes tensor @@ returns tensor)
              Nx.logical_not m
          in
          equal ~msg:"storage" (list nativeint) before (Witness.addresses r);
          agrees (fun () -> expected) (fun () -> r);
          equal ~msg:"tail" (list int) [ 0 ] (tails r));
      cases ~name:fst "a consumed set or scatter of bits is eager's, in place"
        [
          ("a window at an odd offset", `Window (5, 11));
          ("a window over the first byte", `Window (0, 3));
          ("a window to the end", `Window (17, 21));
          ("two bits of one byte", `Scatter [| 1L; 3L |]);
          ( "bits of several bytes, one twice",
            `Scatter [| 20L; 2L; 9L; 2L; -1L |] );
        ]
        (fun (_, write) ->
          let fresh () =
            Nx.cast Nx.bit (Nx.init Nx.bool [| 21 |] (fun i -> i.(0) mod 3 = 0))
          in
          let f m =
            match write with
            | `Window (lo, hi) ->
                Nx.set [ R (lo, hi) ] (Nx.ones Nx.bit [| hi - lo |]) m
            | `Scatter is ->
                let k = Array.length is in
                Nx.scatter ~axis:0
                  ~indices:(Nx.create Nx.int64 [| k |] is)
                  ~values:
                    (Nx.create Nx.bit [| k |]
                       (Array.init k (fun i -> i mod 2 = 0)))
                  m
          in
          let m = fresh () in
          let before = Witness.addresses m in
          let r = Rune.jit Nx.Ptree.(consumes tensor @@ returns tensor) f m in
          equal ~msg:"storage" (list nativeint) before (Witness.addresses r);
          agrees (fun () -> f (fresh ())) (fun () -> r));
      test "a consumed set and scatter of int4 nibbles are eager's, in place"
        (fun () ->
          let fresh () =
            Nx.init Nx.int4 [| 3; 5 |] (fun i -> (3 * i.(0)) - i.(1))
          in
          let set q =
            Nx.set [ R (1, 2); R (1, 4) ] (Nx.full Nx.int4 [| 1; 3 |] 7) q
          in
          let scatter q =
            Nx.scatter ~axis:1
              ~indices:
                (Nx.create Nx.int64 [| 3; 2 |] [| 0L; 3L; 4L; 4L; 1L; 2L |])
              ~values:(Nx.create Nx.int4 [| 3; 2 |] [| -8; 5; 1; 2; -1; 6 |])
              q
          in
          List.iter
            (fun (msg, f) ->
              let q = fresh () in
              let before = Witness.addresses q in
              let r =
                Rune.jit Nx.Ptree.(consumes tensor @@ returns tensor) f q
              in
              equal ~msg (list nativeint) before (Witness.addresses r);
              agrees ~msg (fun () -> f (fresh ())) (fun () -> r))
            [ ("set", set); ("scatter", scatter) ]);
      test "a window set into bits at an odd offset keeps the bits around it"
        (fun () ->
          let m =
            Nx.cast Nx.bit (Nx.init Nx.bool [| 21 |] (fun i -> i.(0) mod 3 = 0))
          in
          let v = Nx.ones Nx.bit [| 6 |] in
          let f m v = Nx.set [ R (5, 11) ] v m in
          equal (tensor bool) (f m v) (Rune.jit (two ()) f m v));
      test "an int4 scatter that adds wraps modulo 16" (fun () ->
          let q = Nx.create Nx.int4 [| 3 |] [| 7; -8; 0 |] in
          let f q =
            Nx.scatter ~mode:`Add ~axis:0
              ~indices:(Nx.create Nx.int64 [| 3 |] [| 0L; 1L; 0L |])
              ~values:(Nx.create Nx.int4 [| 3 |] [| 1; -1; 1 |])
              q
          in
          equal (tensor int)
            (Nx.create Nx.int4 [| 3 |] [| -7; 7; 0 |])
            (Rune.jit (one ()) f q));
      test "a captured int4 view at an odd offset and a captured bit are read"
        (fun () ->
          let q =
            Nx.slice
              [ R (3, 10) ]
              (Nx.init Nx.int4 [| 11 |] (fun i -> i.(0) - 8))
          in
          let m =
            Nx.slice
              [ R (5, 12) ]
              (Nx.cast Nx.bit
                 (Nx.init Nx.bool [| 13 |] (fun i -> i.(0) mod 3 = 0)))
          in
          let one_bit = Nx.slice [ R (2, 3) ] m in
          let f x =
            Nx.where
              (Nx.cast Nx.bool (Nx.logical_and m one_bit))
              (Nx.add x q) (Nx.neg x)
          in
          let x = Nx.init Nx.int4 [| 7 |] (fun i -> 2 * i.(0)) in
          agrees (fun () -> f x) (fun () -> Rune.jit (one ()) f x));
      test
        "a new bit, int4 or uint4 result writes the bits past its last element \
         as 0" (fun () ->
          let m =
            Nx.cast Nx.bit (Nx.init Nx.bool [| 13 |] (fun i -> i.(0) mod 2 = 0))
          in
          let r = Rune.jit (one ()) Nx.logical_not m in
          equal (array int) [| 0xAA; 0x0A |] (bytes r);
          let q = Nx.create Nx.int4 [| 3 |] [| 1; -1; 7 |] in
          equal (array int) [| 0x1F; 0x09 |]
            (bytes (Rune.jit (one ()) Nx.neg q));
          let u = Nx.create Nx.uint4 [| 1 |] [| 15 |] in
          equal (array int) [| 0x0E |]
            (bytes (Rune.jit (one ()) (fun u -> Nx.add u u) u)));
      test
        "a bitcast of bits along a last axis of 8 reads their bytes, at an \
         offset" (fun () ->
          let m = Nx.init Nx.bool [| 19 |] (fun i -> i.(0) * 5 mod 7 < 3) in
          let b =
            Nx.reshape [| 2; 8 |] (Nx.slice [ R (3, 19) ] (Nx.cast Nx.bit m))
          in
          let f = Nx.bitcast Nx.uint8 in
          equal (tensor int) (f b) (Rune.jit (one ()) f b);
          let u = Nx.create Nx.uint8 [| 3 |] [| 0x5A; 0x81; 0xFF |] in
          equal (tensor bool) (Nx.bitcast Nx.bit u)
            (Rune.jit (one ()) (Nx.bitcast Nx.bit) u));
      test
        "a bitcast between int4, uint4, int8 and int16 reads nibbles low first"
        (fun () ->
          let q =
            Nx.init Nx.int4 [| 3; 4 |] (fun i -> (i.(0) * 4) + i.(1) - 8)
          in
          equal (tensor int) (Nx.bitcast Nx.uint4 q)
            (Rune.jit (one ()) (Nx.bitcast Nx.uint4) q);
          let q2 = Nx.reshape [| 6; 2 |] q in
          equal (tensor int) (Nx.bitcast Nx.int8 q2)
            (Rune.jit (one ()) (Nx.bitcast Nx.int8) q2);
          equal (tensor int) (Nx.bitcast Nx.int16 q)
            (Rune.jit (one ()) (Nx.bitcast Nx.int16) q);
          let w = Nx.create Nx.int16 [| 2 |] [| -2; 0x1234 |] in
          equal (tensor int) (Nx.bitcast Nx.int4 w)
            (Rune.jit (one ()) (Nx.bitcast Nx.int4) w);
          let b =
            Nx.reshape [| 2; 4 |]
              (Nx.cast Nx.bit
                 (Nx.init Nx.bool [| 8 |] (fun i -> i.(0) mod 3 = 0)))
          in
          equal (tensor int) (Nx.bitcast Nx.int4 b)
            (Rune.jit (one ()) (Nx.bitcast Nx.int4) b);
          equal (tensor bool) (Nx.bitcast Nx.bit q)
            (Rune.jit (one ()) (Nx.bitcast Nx.bit) q));
    ]

(* Transformations *)

(* The names of the host kernels that [f ()] runs, once warmed. *)
let kernels f =
  ignore (f ());
  Nx_device.synchronize Nx_device.host;
  let p = Nx_device.Profile.start () in
  ignore (f ());
  Nx_device.synchronize Nx_device.host;
  List.filter_map
    (function Nx_device.Profile.Span s -> Some s.name | _ -> None)
    (Nx_device.Profile.stop p)

(* The elements a kernel's name says it spans: the product of its sizes. *)
let extent name =
  List.fold_left
    (fun acc w ->
      match int_of_string_opt w with Some k -> acc * k | None -> acc)
    1
    (String.split_on_char '_' name)

(* A scan over 64 rows of 1024 int4 weights: each step reads its row from the
   input's bytes, so no kernel spans the 65536 elements of the input. *)
let stacked_rows () =
  let ws =
    Nx.init Nx.int4 [| 64; 1024 |] (fun i -> (((i.(0) * 7) + i.(1)) mod 16) - 8)
  in
  let f ws =
    snd
      (Rune.scan'
         ~f:(fun c w ->
           let c = Nx.add c (Nx.sum (Nx.cast Nx.int32 w)) in
           (c, c))
         ~init:(Nx.zeros Nx.int32 [||]) ws)
  in
  let g = Rune.jit (one ()) f in
  agrees (fun () -> f ws) (fun () -> g ws);
  let widest =
    List.fold_left
      (fun m k -> Int.max m (extent k))
      0
      (kernels (fun () -> g ws))
  in
  less int ~than:65536 widest

let transformations =
  let q =
    Nx.init Nx.int4 [| 3; 7 |] (fun i ->
        (((i.(0) * 7) + (i.(1) * 5)) mod 16) - 8)
  in
  let m =
    Nx.cast Nx.bit
      (Nx.init Nx.bool [| 3; 11 |] (fun i -> (i.(0) + i.(1)) mod 3 = 0))
  in
  let x =
    Nx.init Nx.float32 [| 3; 7 |] (fun i -> float_of_int (i.(0) - i.(1)) /. 4.)
  in
  group "transformations"
    [
      test "a mapped int4 function, compiled, is eager's" (fun () ->
          let f r = Nx.mul (Nx.sum ~keepdims:true r) r in
          agrees
            (fun () -> Rune.vmap' f q)
            (fun () -> Rune.jit (one ()) (Rune.vmap' f) q));
      test "a mapped bit function, compiled, is eager's" (fun () ->
          let f r =
            Nx.logical_xor (Nx.flip r)
              (Nx.any ~keepdims:true r |> Nx.cast Nx.bit
              |> Nx.broadcast_to (Nx.shape r))
          in
          agrees
            (fun () -> Rune.vmap' f m)
            (fun () -> Rune.jit (one ()) (Rune.vmap' f) m);
          agrees
            (fun () -> Rune.vmap' Nx.count m)
            (fun () -> Rune.jit (one ()) (Rune.vmap' Nx.count) m));
      test "a compiled scan carries int4 and bit values through its rows"
        (fun () ->
          let step (c, k) (r, b) =
            let c = Nx.add c r and k = Nx.logical_xor k b in
            ((c, k), Nx.mul c (Nx.cast Nx.int4 k))
          in
          let f q m =
            let (c, k), ys =
              Rune.scan
                Nx.Ptree.(pair tensor tensor)
                Nx.Ptree.(pair tensor tensor)
                Nx.Ptree.tensor ~f:step
                ~init:(Nx.zeros Nx.int4 [| 7 |], Nx.zeros Nx.bit [| 7 |])
                (q, m)
            in
            Nx.add (Nx.add c (Nx.cast Nx.int4 k)) (Nx.sum ~axes:[ 0 ] ys)
          in
          let m = Nx.slice [ A; R (2, 9) ] m in
          agrees (fun () -> f q m) (fun () -> Rune.jit (two ()) f q m));
      test
        "a compiled scan reads each int4 row in place, unpacking no whole input"
        stacked_rows;
      test "a compiled scan returns int4 and bit carries with clear tails"
        (fun () ->
          let f q m =
            fst
              (Rune.scan
                 Nx.Ptree.(pair tensor tensor)
                 Nx.Ptree.(pair tensor tensor)
                 Nx.Ptree.tensor
                 ~f:(fun (c, k) (r, b) ->
                   let c = Nx.add c r and k = Nx.logical_xor k b in
                   ((c, k), c))
                 ~init:(Nx.zeros Nx.int4 [| 7 |], Nx.zeros Nx.bit [| 7 |])
                 (q, m))
          in
          let m = Nx.slice [ A; R (2, 9) ] m in
          let c, k =
            Rune.jit
              Nx.Ptree.(tensor @-> tensor @-> returns (pair tensor tensor))
              f q m
          in
          let c', k' = f q m in
          agrees ~msg:"int4 carry" (fun () -> c') (fun () -> c);
          agrees ~msg:"bit carry" (fun () -> k') (fun () -> k);
          equal ~msg:"int4 tail" (list int) [ 0 ] (tails c);
          equal ~msg:"bit tail" (list int) [ 0 ] (tails k));
      test "a gradient through int4 weights, compiled, is eager's" (fun () ->
          let loss q x =
            Nx.sum (Nx.mul (Nx.mul x x) (Nx.cast Nx.float32 (Nx.add_s q 3)))
          in
          let g x q = Rune.grad' (fun x -> loss q x) x in
          equal (tensor float_exact) (g x q) (Rune.jit (two ()) g x q));
      test "a gradient through a bit mask, compiled, is eager's" (fun () ->
          let mask = Nx.cast Nx.bit (Nx.greater_s x 0.) in
          let loss mask x =
            Nx.sum (Nx.where (Nx.cast Nx.bool mask) (Nx.mul x x) x)
          in
          let g x mask = Rune.grad' (fun x -> loss mask x) x in
          equal (tensor float_exact) (g x mask) (Rune.jit (two ()) g x mask));
      test "an int4 or bit result carries a zero tangent, compiled" (fun () ->
          let f x = Nx.add (Nx.cast Nx.int4 (Nx.mul_s x 8.)) q in
          let y, dy = Rune.jvp' (Rune.jit (one ()) f) x (Nx.ones_like x) in
          agrees (fun () -> f x) (fun () -> y);
          equal (tensor int) (Nx.zeros_like q) dy;
          let g x = Nx.cast Nx.bit (Nx.greater_s x 0.) in
          let y, dy = Rune.jvp' (Rune.jit (one ()) g) x (Nx.ones_like x) in
          agrees (fun () -> g x) (fun () -> y);
          equal (tensor bool) (Nx.zeros Nx.bit (Nx.shape x)) dy);
    ]

(* Devices *)

let d1, d2 = (Nx.Device.cpu 1, Nx.Device.cpu 2)

(* [placements_agree p] checks bit and int4 functions of values placed at
   [p]. *)
let placed_values p =
  let m =
    Nx.cast Nx.bit
      (Nx.init Nx.bool [| 4; 16 |] (fun i -> ((i.(0) * 7) + i.(1)) mod 3 = 0))
  in
  let q = Nx.init Nx.int4 [| 4; 16 |] (fun i -> (i.(0) * 5) - i.(1)) in
  let at t = Nx.place p t in
  let f m =
    Nx.logical_and (Nx.logical_not m)
      (Nx.cast Nx.bit (Nx.add_s (Nx.cast Nx.int8 m) 1))
  in
  let g q = Nx.add (Nx.mul q q) (Nx.sum ~axes:[ 1 ] ~keepdims:true q) in
  agrees ~msg:"bit" (fun () -> f m) (fun () -> Rune.jit (one ()) f (at m));
  agrees ~msg:"int4" (fun () -> g q) (fun () -> Rune.jit (one ()) g (at q));
  agrees ~msg:"cast to bit"
    (fun () -> Nx.cast Nx.bit q)
    (fun () -> Rune.jit (one ()) (Nx.cast Nx.bit) (at q))

let devices =
  group "devices"
    [
      test "values on a device" (fun () -> placed_values (Nx.Placement.on d1));
      test "values copied to two devices" (fun () ->
          placed_values (Nx.Placement.replicated [ d1; d2 ]));
      test "values split along their rows over two devices" (fun () ->
          placed_values (Nx.Placement.sharded ~axis:0 [ d1; d2 ]));
      test "values split along their columns over two devices" (fun () ->
          placed_values (Nx.Placement.sharded ~axis:1 [ d1; d2 ]));
      test
        "a call that captures two dtypes of one storage reads each as its own"
        (fun () ->
          (* A bitcast of a placed value views its storage, so a uint8 value and
             its int8 and uint4 readings share it, two at the same view. *)
          let u =
            Nx.place (Nx.Placement.on d1)
              (Nx.init Nx.uint8 [| 2; 3 |] (fun i -> 120 + (i.(0) * 60) + i.(1)))
          in
          let i8 = Nx.bitcast Nx.int8 u and q = Nx.bitcast Nx.uint4 u in
          let f x =
            Nx.add
              (Nx.add (Nx.cast Nx.int32 u) (Nx.cast Nx.int32 i8))
              (Nx.add x (Nx.sum ~axes:[ 2 ] (Nx.cast Nx.int32 q)))
          in
          let x =
            Nx.place (Nx.Placement.on d1) (Nx.zeros Nx.int32 [| 2; 3 |])
          in
          agrees (fun () -> f x) (fun () -> Rune.jit (one ()) f x));
      cases ~name:fst "a split whose windows end within a byte"
        [
          ("[2; 13] along its rows", ([| 2; 13 |], 0));
          ("[4; 6] along its columns", ([| 4; 6 |], 1));
        ]
        (fun (_, (shape, axis)) ->
          let p = Nx.Placement.sharded ~axis [ d1; d2 ] in
          let m =
            Nx.cast Nx.bit
              (Nx.init Nx.bool shape (fun i -> (i.(0) + (2 * i.(1))) mod 3 = 0))
          in
          let q = Nx.init Nx.int4 shape (fun i -> (i.(0) * 5) - i.(1)) in
          let f m = Nx.logical_not m and g q = Nx.neg q in
          let rm = Rune.jit (one ()) f (Nx.place p m) in
          let rq = Rune.jit (one ()) g (Nx.place p q) in
          agrees ~msg:"bit" (fun () -> f m) (fun () -> rm);
          agrees ~msg:"int4" (fun () -> g q) (fun () -> rq);
          equal ~msg:"bit tails" (list int) [ 0; 0 ] (tails rm);
          equal ~msg:"int4 tails" (list int) [ 0; 0 ] (tails rq));
    ]

(* MXFP4 codes as uint4 *)

(* The float32 bits of the e2m1 codes [q]: a sign bit, two exponent bits and a
   mantissa bit. An exponent of 0 is 0 or 0.5; another, [e], is 2^(e - 1) (1 + m
   / 2). A scale byte [s] is 2^(s - 127). *)
let mxfp4_values codes scales =
  let q = Nx.cast Nx.uint32 codes in
  let k = Nx.scalar_like q in
  let e = Nx.bitwise_and (Nx.rshift q 1) (k 3l)
  and m = Nx.bitwise_and q (k 1l) in
  let sign = Nx.lshift (Nx.bitwise_and q (k 8l)) 28 in
  let magnitude =
    Nx.where (Nx.equal_s e 0l)
      (Nx.mul_s m (Int32.shift_left 126l 23))
      (Nx.bitwise_or (Nx.lshift (Nx.add_s e 126l) 23) (Nx.lshift m 22))
  in
  let v = Nx.bitcast Nx.float32 (Nx.bitwise_or sign magnitude) in
  let scale = Nx.bitcast Nx.float32 (Nx.lshift (Nx.cast Nx.uint32 scales) 23) in
  Nx.mul v scale

let mxfp4 =
  test "MXFP4 codes as uint4 decode to nx.quant's values, compiled" (fun () ->
      let rows = 3 and k = 64 in
      let bytes =
        Nx.init Nx.uint8
          [| rows; k / 2 |]
          (fun i -> ((i.(0) * 37) + (i.(1) * 11)) land 255)
      in
      let scales =
        Nx.init Nx.uint8 [| rows; k / 32 |] (fun i -> 120 + i.(0) + i.(1))
      in
      let codes = Nx.reshape [| rows; k |] (Nx.bitcast Nx.uint4 bytes) in
      let decode codes scales =
        let groups = Nx.reshape [| rows; k / 32; 32 |] codes in
        Nx.reshape [| rows; k |]
          (mxfp4_values groups (Nx.reshape [| rows; k / 32; 1 |] scales))
      in
      let expected =
        Nx_quant.dequant Nx.float32 (Nx_quant.mxfp4 ~scales bytes)
      in
      equal (tensor float_exact) expected (decode codes scales);
      equal (tensor float_exact) expected
        (Rune.jit (two ()) decode codes scales))

(* Metal *)

let metal =
  match Result.to_option (Nx_metal.get 0) with
  | None -> slow "no Metal device" (fun () -> skip ~reason:"no Metal device" ())
  | Some d ->
      group ~tags:[ "slow" ] "metal"
        [
          test "values on Metal" (fun () -> placed_values (Nx.Placement.on d));
          prop ~count:6 "int4 values on Metal" int4_operands (fun (_, a, b) ->
              let f a b = Nx.add (Nx.mul a b) (Nx.div a b) in
              let on t = Nx.place (Nx.Placement.on d) t in
              agrees
                (fun () -> f a b)
                (fun () -> Rune.jit (two ()) f (on a) (on b)));
          prop ~count:6 "bit values on Metal" bit_operands (fun (_, a, b) ->
              let f a b = Nx.logical_xor (Nx.flip a) b in
              let on t = Nx.place (Nx.Placement.on d) t in
              agrees
                (fun () -> f a b)
                (fun () -> Rune.jit (two ()) f (on a) (on b)));
        ]

let () =
  exit
    (run "packed dtypes in compiled calls"
       [ values; widths; stated; transformations; devices; mxfp4; metal ])
