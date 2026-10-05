(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The bit dtype: its storage order, its views at any bit, and its agreement
   with bool, which holds the same values one to a byte. The laws draw masks
   whose views start at every bit of a word and end at word edges, dense,
   transposed, flipped, strided and broadcast, and compare a function of the bit
   mask with the same function of the bool mask. *)

open Windtrap
open Nx_test

let same = tensor bool

(* Masks *)

(* A bit mask and the bool mask of the same values and shape. *)
type mask = { bits : Nx.bit_t; bools : Nx.bool_t }

let pp_mask ppf { bools; bits } =
  let v = view bits in
  Format.fprintf ppf "offset %d, strides %a: %a" (Nx_array.View.offset v)
    pp_shape (Nx_array.View.strides v)
    (Ref.pp Format.pp_print_bool)
    (Ref.of_nx bools)

(* A movement, which applies to a mask of either dtype. *)
type move = { move : 'b. (bool, 'b) Nx.t -> (bool, 'b) Nx.t }

let both { move } { bits; bools } = { bits = move bits; bools = move bools }

(* Lengths at the edges of bytes and words. *)
let length =
  Gen.frequency
    [
      ( 3,
        Gen.of_list ~pp:Format.pp_print_int
          [ 0; 1; 7; 8; 9; 63; 64; 65; 127; 128; 129 ] );
      (2, Gen.int_range 0 300);
    ]

(* Offsets from every bit of a word, and past one. *)
let offset =
  Gen.frequency
    [
      (6, Gen.int_range 1 63);
      (1, Gen.of_list ~pp:Format.pp_print_int [ 0; 64; 65; 127 ]);
    ]

(* [stored n] is a mask of [n] values in a 1-D view at a drawn offset of a
   longer storage, so at any bit of a word. *)
let stored n =
  let open Gen in
  let* off = offset in
  let* extra = int_range 0 70 in
  let+ vs = array ~size:(constant (off + n + extra)) bool in
  let bools = Nx.create Nx.bool [| Array.length vs |] vs in
  both
    { move = (fun t -> Nx.shrink [| (off, off + n) |] t) }
    { bits = Nx.cast Nx.bit bools; bools }

(* How a mask of a shape lies in its storage. *)
type layout = Dense | Transposed | Flipped | Strided | Broadcast

let pp_layout ppf l =
  Format.pp_print_string ppf
    (match l with
    | Dense -> "dense"
    | Transposed -> "transposed"
    | Flipped -> "flipped"
    | Strided -> "every other element"
    | Broadcast -> "broadcast from its first row")

let numel s = Array.fold_left ( * ) 1 s

(* A view of stride 2 over the 1-D [t], of half its elements. *)
let every_other t =
  if Nx.numel t = 0 then t
  else Nx.squeeze ~axes:[ -1 ] (Nx.sliding_window ~window:1 ~step:2 t)

let laid s l =
  let reversed = Array.of_list (List.rev (Array.to_list s)) in
  let first_row = Array.mapi (fun i d -> if i = 0 then 1 else d) s in
  match l with
  | Dense ->
      Gen.map (both { move = (fun t -> Nx.reshape s t) }) (stored (numel s))
  | Transposed ->
      Gen.map
        (both { move = (fun t -> Nx.transpose (Nx.reshape reversed t)) })
        (stored (numel s))
  | Flipped ->
      Gen.map
        (both { move = (fun t -> Nx.flip (Nx.reshape s t)) })
        (stored (numel s))
  | Strided ->
      Gen.map
        (both { move = (fun t -> Nx.reshape s (every_other t)) })
        (stored (2 * numel s))
  | Broadcast ->
      Gen.map
        (both { move = (fun t -> Nx.broadcast_to s (Nx.reshape first_row t)) })
        (stored (numel first_row))

let shape =
  Gen.frequency
    [
      (3, Gen.map (fun n -> [| n |]) length);
      ( 2,
        Gen.(
          let+ r = int_range 0 9 and+ c = int_range 0 70 in
          [| r; c |]) );
      (1, Gen.constant ~pp:pp_shape [||]);
    ]

(* A mask of shape [s] in a drawn layout. *)
let mask_of s =
  let layouts = [ Dense; Transposed; Flipped; Strided; Broadcast ] in
  let drawn =
    Gen.bind (Gen.of_list ~pp:pp_layout layouts) (fun l ->
        Gen.map (fun m -> (l, m)) (laid s l))
  in
  Gen.map snd
    (Gen.with_pp
       (fun ppf (l, m) -> Format.fprintf ppf "%a, %a" pp_layout l pp_mask m)
       drawn)

let mask = Gen.bind shape mask_of

(* Two masks of one shape, each in a layout of its own. *)
let two = Gen.bind shape (fun s -> Gen.pair (mask_of s) (mask_of s))

(* The views a law over masks must meet: one that starts inside a byte, one that
   ends inside a word past the first, and one that is not a single run. *)
let cover_views { bits; _ } =
  let v = view bits and n = Nx.numel bits in
  cover "a view that starts inside a byte" (Nx_array.View.offset v mod 8 <> 0);
  cover "a view that ends inside a word" (n > 64 && n mod 64 <> 0);
  cover "a view that is not one run" (n > 1 && not (Nx.is_c_contiguous bits))

(* Functions *)

(* A function of masks that returns their dtype. *)
type f1 = { name : string; f : 'b. (bool, 'b) Nx.t -> (bool, 'b) Nx.t }

type f2 = {
  name2 : string;
  f2 : 'b. (bool, 'b) Nx.t -> (bool, 'b) Nx.t -> (bool, 'b) Nx.t;
}

let flat t = Nx.reshape [| -1 |] t

let indices n f =
  Nx.create Nx.int64 [| n |] (Array.init n (fun i -> Int64.of_int (f i)))

(* Moves, which nx.cpu computes on bits, and functions computed through bool,
   each over every shape. *)
let unaries =
  [
    { name = "copy"; f = (fun t -> Nx.copy t) };
    { name = "contiguous"; f = (fun t -> Nx.contiguous t) };
    { name = "logical_not"; f = (fun t -> Nx.logical_not t) };
    { name = "bitwise_not"; f = (fun t -> Nx.bitwise_not t) };
    { name = "a copy of its flip"; f = (fun t -> Nx.copy (Nx.flip t)) };
    {
      name = "a copy of its transpose";
      f = (fun t -> Nx.copy (Nx.transpose t));
    };
    {
      name = "concatenate of three of it";
      f = (fun t -> Nx.concatenate ~axis:0 [ flat t; flat t; flat t ]);
    };
    {
      name = "concatenate between parts of 13";
      f =
        (fun t ->
          let thirteen = Nx.full (Nx.dtype t) [| 13 |] true in
          Nx.concatenate ~axis:0 [ thirteen; flat t; thirteen ]);
    };
    {
      name = "concatenate along its last axis";
      f =
        (fun t ->
          if Nx.ndim t < 2 then t
          else Nx.concatenate ~axis:1 [ t; Nx.logical_not t; t ]);
    };
    {
      name = "pad";
      f = (fun t -> Nx.pad (Array.map (fun _ -> (3, 13)) (Nx.shape t)) true t);
    };
    {
      name = "take with indices outside";
      f =
        (fun t ->
          let n = Nx.numel t in
          Nx.take ~indices:(indices (n + 2) (fun i -> n - i)) (flat t));
    };
    {
      name = "take_along_axis";
      f =
        (fun t ->
          if Nx.ndim t < 2 || Nx.dim 1 t = 0 then t
          else
            let r = Nx.dim 0 t and c = Nx.dim 1 t in
            Nx.take_along_axis ~axis:1
              ~indices:
                (Nx.reshape [| r; c |] (indices (r * c) (fun i -> i * 5 mod c)))
              t);
    };
    {
      name = "set of a window";
      f =
        (fun t ->
          let n = Nx.numel t in
          let lo = n / 3 and hi = n - (n / 5) in
          Nx.set
            [ R (lo, hi) ]
            (Nx.logical_not (Nx.slice [ R (0, hi - lo) ] (flat t)))
            (flat t));
    };
    {
      name = "scatter with Set";
      f =
        (fun t ->
          let n = Nx.numel t in
          if n = 0 then flat t
          else
            Nx.scatter ~axis:0
              ~indices:(indices (2 * n) (fun i -> (i * 7) + 3 - n))
              ~values:
                (Nx.logical_not (Nx.concatenate ~axis:0 [ flat t; flat t ]))
              (flat t));
    };
    {
      name = "scatter with Max";
      f =
        (fun t ->
          let n = Nx.numel t in
          if n = 0 then flat t
          else
            Nx.scatter ~mode:`Max ~axis:0
              ~indices:(indices n (fun i -> i * 3 mod n))
              ~values:(Nx.logical_not (flat t))
              (flat t));
    };
    {
      name = "where of its negation";
      f =
        (fun t ->
          Nx.where (Nx.cast Nx.bool (Nx.logical_not t)) (Nx.logical_not t) t);
    };
    { name = "sort"; f = (fun t -> fst (Nx.sort (flat t))) };
    { name = "cummax"; f = (fun t -> Nx.cummax (flat t)) };
    { name = "roll"; f = (fun t -> Nx.roll 5 (flat t)) };
    { name = "tile"; f = (fun t -> Nx.tile [| 3 |] (flat t)) };
    { name = "repeat"; f = (fun t -> Nx.repeat 2 (flat t)) };
    { name = "tril"; f = (fun t -> if Nx.ndim t < 2 then t else Nx.tril t) };
    {
      name = "max along its first axis";
      f = (fun t -> if Nx.ndim t = 0 then t else Nx.max ~axes:[ 0 ] t);
    };
    {
      name = "min along its last axis";
      f = (fun t -> if Nx.ndim t = 0 then t else Nx.min ~axes:[ -1 ] t);
    };
    { name = "fill"; f = (fun t -> Nx.fill true t) };
    {
      name = "a copy of its flattened transpose";
      f = (fun t -> flat (Nx.transpose t));
    };
    {
      name = "every third element";
      f = (fun t -> Nx.slice [ Rs (0, Nx.numel t, 3) ] (flat t));
    };
    {
      name = "sort descending";
      f = (fun t -> fst (Nx.sort ~descending:true (flat t)));
    };
    {
      name = "sort along its last axis";
      f = (fun t -> if Nx.ndim t = 0 then t else fst (Nx.sort ~axis:(-1) t));
    };
    {
      name = "top_k of 5";
      f = (fun t -> fst (Nx.top_k ~k:(min 5 (Nx.numel t)) (flat t)));
    };
    {
      name = "top_k of 40";
      f = (fun t -> fst (Nx.top_k ~k:(min 40 (Nx.numel t)) (flat t)));
    };
    { name = "cummin"; f = (fun t -> Nx.cummin (flat t)) };
    {
      name = "stack with its negation";
      f = (fun t -> Nx.stack [ t; Nx.logical_not t ]);
    };
    {
      name = "diagonal";
      f = (fun t -> if Nx.ndim t < 2 then t else Nx.diagonal t);
    };
    { name = "triu"; f = (fun t -> if Nx.ndim t < 2 then t else Nx.triu t) };
    {
      name = "compress where its flip holds";
      f =
        (fun t ->
          Nx.compress ~condition:(Nx.cast Nx.bool (Nx.flip (flat t))) (flat t));
    };
    {
      name = "extract where it holds";
      f = (fun t -> Nx.extract ~condition:(Nx.cast Nx.bool t) t);
    };
    {
      name = "its halves swapped";
      f =
        (fun t ->
          if Nx.ndim t = 0 || Nx.dim 0 t mod 2 <> 0 then t
          else Nx.concatenate ~axis:0 (List.rev (Nx.split ~axis:0 2 t)));
    };
    {
      name = "take along its last axis with indices outside";
      f =
        (fun t ->
          if Nx.ndim t = 0 then t
          else
            let c = Nx.dim (-1) t in
            Nx.take ~axis:(-1) ~indices:(indices (c + 3) (fun i -> c + 1 - i)) t);
    };
    {
      name = "scatter with Min";
      f =
        (fun t ->
          let n = Nx.numel t in
          if n = 0 then flat t
          else
            Nx.scatter ~mode:`Min ~axis:0
              ~indices:(indices n (fun i -> i * 5 mod n))
              ~values:(Nx.logical_not (flat t))
              (flat t));
    };
    {
      name = "scatter with Add";
      f =
        (fun t ->
          let n = Nx.numel t in
          if n = 0 then flat t
          else
            Nx.scatter ~mode:`Add ~axis:0
              ~indices:(indices n (fun i -> i))
              ~values:(flat t) (flat t));
    };
    {
      name = "set of a window of rows to a broadcast true";
      f =
        (fun t ->
          if Nx.ndim t <> 2 || Nx.dim 0 t < 2 || Nx.dim 1 t < 3 then t
          else
            Nx.set
              [ R (1, Nx.dim 0 t); R (1, Nx.dim 1 t - 1) ]
              (Nx.full (Nx.dtype t) [||] true)
              t);
    };
    {
      name = "set of a window of rows to its negated corner";
      f =
        (fun t ->
          if Nx.ndim t <> 2 || Nx.dim 0 t < 2 || Nx.dim 1 t < 3 then t
          else
            let r = Nx.dim 0 t - 1 and c = Nx.dim 1 t - 2 in
            Nx.set
              [ R (1, r + 1); R (2, c + 2) ]
              (Nx.logical_not (Nx.slice [ R (0, r); R (0, c) ] t))
              t);
    };
    {
      name = "scatter with Set of a broadcast true";
      f =
        (fun t ->
          let n = Nx.numel t in
          if n = 0 then flat t
          else
            Nx.scatter ~axis:0
              ~indices:(indices n (fun i -> (i * 11) + 1 - n))
              ~values:(Nx.full (Nx.dtype t) [||] true)
              (flat t));
    };
    {
      name = "where with a broadcast true branch";
      f =
        (fun t ->
          Nx.where
            (Nx.cast Nx.bool (Nx.logical_not t))
            (Nx.full (Nx.dtype t) [||] true)
            t);
    };
    {
      name = "where of a condition from its flip";
      f =
        (fun t ->
          Nx.where (Nx.cast Nx.bool (Nx.flip t)) (Nx.flip t) (Nx.logical_not t));
    };
  ]

let binaries =
  [
    { name2 = "logical_and"; f2 = (fun a b -> Nx.logical_and a b) };
    { name2 = "logical_or"; f2 = (fun a b -> Nx.logical_or a b) };
    { name2 = "logical_xor"; f2 = (fun a b -> Nx.logical_xor a b) };
    { name2 = "bitwise_and"; f2 = (fun a b -> Nx.bitwise_and a b) };
    { name2 = "bitwise_or"; f2 = (fun a b -> Nx.bitwise_or a b) };
    { name2 = "bitwise_xor"; f2 = (fun a b -> Nx.bitwise_xor a b) };
    { name2 = "maximum"; f2 = (fun a b -> Nx.maximum a b) };
    { name2 = "minimum"; f2 = (fun a b -> Nx.minimum a b) };
    {
      name2 = "logical_and with a broadcast true";
      f2 = (fun a _ -> Nx.logical_and a (Nx.full (Nx.dtype a) [||] true));
    };
    {
      name2 = "logical_or with a broadcast false";
      f2 = (fun a _ -> Nx.logical_or a (Nx.full (Nx.dtype a) [||] false));
    };
  ]

let one_meaning =
  group "one meaning"
    ([
       prop "cast bool (cast bit m) is m" mask (fun { bools; _ } ->
           equal same bools (Nx.cast Nx.bool (Nx.cast Nx.bit bools)));
       prop "a view at any bit holds its bool's values" mask (fun m ->
           cover_views m;
           equal same m.bools (Nx.cast Nx.bool m.bits));
     ]
    @ List.map
        (fun { name; f } ->
          prop (name ^ " of a bit mask is cast bit of it of the bool mask") mask
            (fun m ->
              cover_views m;
              match f m.bools with
              | expected -> equal same expected (Nx.cast Nx.bool (f m.bits))
              | exception Invalid_argument _ ->
                  raises_invalid_arg (fun () -> f m.bits)))
        unaries
    @ List.map
        (fun { name2; f2 } ->
          prop (name2 ^ " of bit masks is cast bit of it of the bool masks") two
            (fun (a, b) ->
              cover_views a;
              cover_views b;
              equal same (f2 a.bools b.bools)
                (Nx.cast Nx.bool (f2 a.bits b.bits))))
        binaries)

(* Functions of masks that return another dtype give a bit mask what they give
   its bool. *)
type g = { gname : string; g : 'b. (bool, 'b) Nx.t -> Nx.packed }

let other_dtypes =
  let p t = Nx.P t in
  [
    { gname = "count"; g = (fun t -> p (Nx.count t)) };
    {
      gname = "count along the last axis";
      g =
        (fun t ->
          if Nx.ndim t = 0 then p (Nx.count t) else p (Nx.count ~axes:[ -1 ] t));
    };
    {
      gname = "count, keeping dims";
      g = (fun t -> p (Nx.count ~keepdims:true t));
    };
    { gname = "any"; g = (fun t -> p (Nx.any t)) };
    { gname = "all"; g = (fun t -> p (Nx.all t)) };
    { gname = "argmax"; g = (fun t -> p (Nx.argmax t)) };
    { gname = "positions"; g = (fun t -> p (Nx.positions (flat t))) };
    {
      gname = "nonzero";
      g =
        (fun t ->
          p (Nx.concatenate ~axis:0 (Array.to_list (Nx.nonzero (flat t)))));
    };
    { gname = "equal to its flip"; g = (fun t -> p (Nx.equal t (Nx.flip t))) };
    {
      gname = "array_equal to its negation";
      g = (fun t -> p (Nx.array_equal t (Nx.logical_not t)));
    };
    { gname = "cast float32"; g = (fun t -> p (Nx.cast Nx.float32 t)) };
    { gname = "cast int8"; g = (fun t -> p (Nx.cast Nx.int8 t)) };
    {
      gname = "cast int4";
      g = (fun t -> p (Nx.cast Nx.int8 (Nx.cast Nx.int4 t)));
    };
    { gname = "cast complex64"; g = (fun t -> p (Nx.cast Nx.complex64 t)) };
    { gname = "order_key uint8"; g = (fun t -> p (Nx.order_key Nx.uint8 t)) };
    {
      gname = "to_array";
      g = (fun t -> p (Nx.create Nx.bool [| Nx.numel t |] (Nx.to_array t)));
    };
    { gname = "argmin"; g = (fun t -> p (Nx.argmin t)) };
    { gname = "argsort"; g = (fun t -> p (Nx.argsort (flat t))) };
    {
      gname = "argsort descending";
      g = (fun t -> p (Nx.argsort ~descending:true (flat t)));
    };
    {
      gname = "top_k's indices";
      g = (fun t -> p (snd (Nx.top_k ~k:(min 7 (Nx.numel t)) (flat t))));
    };
    { gname = "unique's groups"; g = (fun t -> p (Nx.unique (flat t)).ids) };
    { gname = "unique's counts"; g = (fun t -> p (Nx.unique (flat t)).counts) };
    {
      gname = "lexsort of its rows";
      g =
        (fun t ->
          if Nx.ndim t <> 2 then p (Nx.lexsort (flat t)) else p (Nx.lexsort t));
    };
    {
      gname = "searchsorted in its sorted values";
      g =
        (fun t ->
          let s = fst (Nx.sort (flat t)) in
          p
            (Nx.concatenate ~axis:0
               [
                 Nx.searchsorted ~side:`Left s (flat t);
                 Nx.searchsorted ~side:`Right s (flat t);
               ]));
    };
    {
      gname = "not_equal to its flip";
      g = (fun t -> p (Nx.not_equal t (Nx.flip t)));
    };
    { gname = "less than its flip"; g = (fun t -> p (Nx.less t (Nx.flip t))) };
    {
      gname = "greater_equal to its flip";
      g = (fun t -> p (Nx.greater_equal t (Nx.flip t)));
    };
    {
      gname = "array_equal to itself";
      g = (fun t -> p (Nx.array_equal t (Nx.copy t)));
    };
    {
      gname = "count along the first axis, keeping dims";
      g =
        (fun t ->
          if Nx.ndim t = 0 then p (Nx.count t)
          else p (Nx.count ~axes:[ 0 ] ~keepdims:true t));
    };
    { gname = "cast bool"; g = (fun t -> p (Nx.cast Nx.bool t)) };
    {
      gname = "cast uint4";
      g = (fun t -> p (Nx.cast Nx.int8 (Nx.cast Nx.uint4 t)));
    };
    { gname = "cast uint64"; g = (fun t -> p (Nx.cast Nx.uint64 t)) };
    { gname = "cast float16"; g = (fun t -> p (Nx.cast Nx.float16 t)) };
    { gname = "cast bfloat16"; g = (fun t -> p (Nx.cast Nx.bfloat16 t)) };
    { gname = "cast float8_e4m3"; g = (fun t -> p (Nx.cast Nx.float8_e4m3 t)) };
    { gname = "cast float64"; g = (fun t -> p (Nx.cast Nx.float64 t)) };
    { gname = "cast complex128"; g = (fun t -> p (Nx.cast Nx.complex128 t)) };
    {
      gname = "one_hot of its cast";
      g = (fun t -> p (Nx.one_hot ~num_classes:3 (Nx.cast Nx.int8 t)));
    };
  ]

let returning_other_dtypes =
  group "functions to other dtypes"
    (List.map
       (fun { gname; g } ->
         prop (gname ^ " of a bit mask is it of the bool mask") mask (fun m ->
             cover_views m;
             match g m.bools with
             | expected -> equal Stored.packed expected (g m.bits)
             | exception Invalid_argument _ ->
                 raises_invalid_arg (fun () -> g m.bits)))
       other_dtypes)

(* Storage *)

let bools l = Nx.create Nx.bool [| List.length l |] (Array.of_list l)
let bits l = Nx.cast Nx.bit (bools l)
let bytes_of t = Nx.to_array (Nx.bitcast Nx.uint8 (Nx.reshape [| -1; 8 |] t))

(* The bytes that pack [vs], element [i] at bit [i mod 8] of byte [i / 8]. *)
let packed vs =
  Array.init
    ((Array.length vs + 7) / 8)
    (fun k ->
      let b = ref 0 in
      for j = 0 to 7 do
        let i = (8 * k) + j in
        if i < Array.length vs && vs.(i) then b := !b lor (1 lsl j)
      done;
      !b)

let storage =
  group "one bit order"
    [
      test "bitcast uint8 of [true; false x 7] is 1" (fun () ->
          equal (array int) [| 1 |]
            (bytes_of (bits (true :: List.init 7 (fun _ -> false)))));
      test "element i is bit i mod 8 of byte i / 8" (fun () ->
          let m =
            Nx.bitcast Nx.bit (Nx.create Nx.uint8 [| 2 |] [| 0x01; 0x82 |])
          in
          equal (array int) [| 2; 8 |] (Nx.shape m);
          equal (array bool)
            (Array.init 16 (fun i -> i = 0 || i = 9 || i = 15))
            (Nx.to_array (Nx.reshape [| 16 |] m)));
      test "bitcast uint8 of a view at offset 3 reads its bits from 3"
        (fun () ->
          let vs = Array.init 80 (fun i -> i mod 3 = 0 || i mod 7 = 1) in
          let m = Nx.cast Nx.bit (Nx.create Nx.bool [| 80 |] vs) in
          equal (array int)
            (packed (Array.sub vs 3 64))
            (bytes_of (Nx.shrink [| (3, 67) |] m)));
      test "a [h; w] mask holds row i from bit i * w" (fun () ->
          let vs = Array.init 15 (fun i -> i mod 4 = 1) in
          let m = Nx.cast Nx.bit (Nx.create Nx.bool [| 3; 5 |] vs) in
          equal (array int) (packed vs)
            (bytes_of (Nx.pad [| (0, 1) |] false (Nx.reshape [| 15 |] m))));
      test "bitcast uint64 reads 64 elements as one word, the first lowest"
        (fun () ->
          let m = bits (List.init 64 (fun i -> i = 0 || i = 63)) in
          equal (array int64)
            [| Int64.logor 1L Int64.min_int |]
            (Nx.to_array (Nx.bitcast Nx.uint64 m)));
      test "a bitcast to bit and back gives the bytes" (fun () ->
          let b = Nx.create Nx.uint8 [| 3 |] [| 0x5a; 0xff; 0x01 |] in
          equal (array int) [| 0x5a; 0xff; 0x01 |]
            (Nx.to_array (Nx.bitcast Nx.uint8 (Nx.bitcast Nx.bit b))));
      test "nbytes counts bits, rounded up to a byte" (fun () ->
          equal int 2 (Nx.nbytes (Nx.zeros Nx.bit [| 13 |]));
          equal int 1 (Nx.nbytes (Nx.zeros Nx.bit [| 2; 3 |]));
          equal int 0 (Nx.nbytes (Nx.zeros Nx.bit [| 0 |]));
          equal int 2 (Nx.nbytes (Nx.zeros Nx.int4 [| 3 |]));
          equal int 13 (Nx.nbytes (Nx.zeros Nx.bool [| 13 |])));
    ]

(* Casts to bit read every dtype's values as a cast to bool does, NaN payloads,
   signed zeros, subnormals and the 4-bit integers included. *)

let nibbles name dtype lo hi =
  Stored.case name dtype
    (viewed ~pp:Format.pp_print_int dtype (Gen.int_range lo hi))
    (tensor int)

let every_dtype =
  Stored.every
  @ [ nibbles "int4" Nx.int4 (-8) 7; nibbles "uint4" Nx.uint4 0 15 ]

let to_bit (Stored.Case c) =
  prop (c.name ^ " cast to bit holds its cast to bool") c.tensors (fun t ->
      equal same (Nx.cast Nx.bool t) (Nx.cast Nx.bool (Nx.cast Nx.bit t)))

let casts =
  group "casts"
    (List.map to_bit every_dtype
    @ [
        test "zeros of either sign are false, and NaN and infinities true"
          (fun () ->
            let x =
              Nx.create Nx.float32 [| 6 |]
                [|
                  0.; -0.; Float.nan; Float.infinity; Float.neg_infinity; 1e-45;
                |]
            in
            equal (array bool)
              [| false; false; true; true; true; true |]
              (Nx.to_array (Nx.cast Nx.bit x)));
        test "a complex value is true when either part is not zero" (fun () ->
            let z re im = { Complex.re; im } in
            let x =
              Nx.create Nx.complex64 [| 4 |]
                [| z 0. 0.; z (-0.) 0.; z 0. 1.; z Float.nan 0. |]
            in
            equal same (Nx.cast Nx.bool x) (Nx.cast Nx.bool (Nx.cast Nx.bit x)));
        test "an int4 is true unless it is 0, -8 included" (fun () ->
            let x = Nx.create Nx.int4 [| 5 |] [| 0; -8; 7; 16; -1 |] in
            equal (array bool)
              [| false; true; true; false; true |]
              (Nx.to_array (Nx.cast Nx.bit x)));
        test "true casts to 1 and false to 0 in every integer and float dtype"
          (fun () ->
            let m = bits [ true; false; true ] in
            equal (array int) [| 1; 0; 1 |] (Nx.to_array (Nx.cast Nx.int4 m));
            equal (array int) [| 1; 0; 1 |] (Nx.to_array (Nx.cast Nx.uint4 m));
            equal (array int64) [| 1L; 0L; 1L |]
              (Nx.to_array (Nx.cast Nx.uint64 m));
            equal (array float_exact) [| 1.; 0.; 1. |]
              (Nx.to_array (Nx.cast Nx.bfloat16 m)));
      ])

(* Refusals *)

type run = { run : 'b. (bool, 'b) Nx.t -> unit }

(* [Str_split.on sep s] is [s] cut at each [sep]. *)
module Str_split = struct
  let on sep s =
    let n = String.length sep in
    let rec go acc start i =
      if i + n > String.length s then
        List.rev (String.sub s start (String.length s - start) :: acc)
      else if String.sub s i n = sep then
        go (String.sub s start (i - start) :: acc) (i + n) (i + n)
      else go acc start (i + 1)
    in
    go [] 0 0
end

let arithmetic =
  (* The message is bool's, the dtype it names aside. *)
  let as_bit m = String.concat "dtype bit" (Str_split.on "dtype bool" m) in
  let refused name { run = f } =
    test (name ^ " raises as on bool") (fun () ->
        let b = bools [ true; false; true ] in
        let message =
          match f b with
          | _ -> fail "bool computed it"
          | exception Invalid_argument m -> m
        in
        raises
          (Invalid_argument (as_bit message))
          (fun () -> f (Nx.cast Nx.bit b)))
  in
  group "arithmetic"
    [
      refused "add" { run = (fun t -> ignore (Nx.add t t)) };
      refused "sub" { run = (fun t -> ignore (Nx.sub t t)) };
      refused "mul" { run = (fun t -> ignore (Nx.mul t t)) };
      refused "div" { run = (fun t -> ignore (Nx.div t t)) };
      refused "neg" { run = (fun t -> ignore (Nx.neg t)) };
      refused "sum" { run = (fun t -> ignore (Nx.sum t)) };
      refused "cumsum" { run = (fun t -> ignore (Nx.cumsum t)) };
      refused "matmul"
        {
          run =
            (fun t ->
              ignore
                (Nx.matmul (Nx.reshape [| 1; 3 |] t) (Nx.reshape [| 3; 1 |] t)));
        };
      refused "lshift" { run = (fun t -> ignore (Nx.lshift t 1)) };
      refused "fma" { run = (fun t -> ignore (Nx.fma t t t)) };
    ]

(* Views are exact *)

(* The bits of [t]'s last byte past its last element, [t] holding its buffer
   from element 0. *)
let tail t =
  let n = Nx.numel t in
  let b = Nx_device.Buffer.bigarray Bigarray.int8_unsigned (Nx.to_buffer t) in
  if n mod 8 = 0 then 0 else b.{n / 8} lsr (n mod 8)

let fresh =
  [
    ("cast to bit", fun m -> Nx.cast Nx.bit m.bools);
    ("copy", fun m -> Nx.copy m.bits);
    ("logical_and", fun m -> Nx.logical_and m.bits (Nx.logical_not m.bits));
    ("logical_not", fun m -> Nx.logical_not m.bits);
    ( "concatenate",
      fun m -> Nx.concatenate ~axis:0 [ flat m.bits; Nx.ones Nx.bit [| 5 |] ] );
    ( "take",
      fun m -> Nx.take ~indices:(indices 13 (fun i -> i - 2)) (flat m.bits) );
    ( "pad",
      fun m ->
        Nx.pad (Array.map (fun _ -> (1, 2)) (Nx.shape m.bits)) true m.bits );
  ]

let exact =
  group "views are exact"
    ([
       test "a window write keeps the bits of its word around it" (fun () ->
           let m =
             Nx.set
               [ R (3, 10) ]
               (Nx.ones Nx.bit [| 7 |]) (Nx.zeros Nx.bit [| 64 |])
           in
           equal int64 7L (Nx.item [] (Nx.count m));
           equal (array bool)
             (Array.init 64 (fun i -> i >= 3 && i < 10))
             (Nx.to_array m));
       test "parts of 13 concatenate into shared bytes" (fun () ->
           let parts =
             List.init 9 (fun k ->
                 bits (List.init 13 (fun i -> (i + k) mod 3 = 0)))
           in
           equal same
             (Nx.concatenate ~axis:0 (List.map (Nx.cast Nx.bool) parts))
             (Nx.cast Nx.bool (Nx.concatenate ~axis:0 parts)));
       test "rows of 15 padded from 13 write their bytes once" (fun () ->
           let b =
             Nx.cast Nx.bool
               (Nx.reshape [| 1000; 13 |]
                  (Nx.cast Nx.bit
                     (Nx.init Nx.bool [| 13000 |] (fun i -> i.(0) mod 5 = 2))))
           in
           equal same
             (Nx.pad [| (1, 1); (1, 1) |] false b)
             (Nx.cast Nx.bool
                (Nx.pad [| (1, 1); (1, 1) |] false (Nx.cast Nx.bit b))));
       slow
         "parts of 13 around 2^24 bits, which nx.cpu writes on several threads"
         (fun () ->
           let n = (1 lsl 24) + 13 in
           let big = Nx.init Nx.bool [| n |] (fun i -> i.(0) mod 7 < 3) in
           let thirteen = bools (List.init 13 (fun i -> i mod 2 = 0)) in
           let expected = Nx.concatenate ~axis:0 [ thirteen; big; thirteen ] in
           let got =
             Nx.concatenate ~axis:0
               [
                 Nx.cast Nx.bit thirteen;
                 Nx.cast Nx.bit big;
                 Nx.cast Nx.bit thirteen;
               ]
           in
           equal (pair int64 int64)
             (Nx.item [] (Nx.count expected), 0L)
             ( Nx.item [] (Nx.count got),
               Nx.item []
                 (Nx.count (Nx.logical_xor (Nx.cast Nx.bit expected) got)) );
           equal same
             (Nx.pad [| (13, 13) |] true big)
             (Nx.cast Nx.bool (Nx.pad [| (13, 13) |] true (Nx.cast Nx.bit big))));
       test "any over a view ending a mapped file reads no byte past it"
         (fun () ->
           let module B = Nx_device.Buffer in
           let size = 65536 in
           let path = temp_file () in
           let pp = Format.pp_print_string in
           let file = require_ok ~pp (B.create_file path size) in
           let ones = Nx.full Nx.uint8 [| size |] 0 in
           let src =
             Nx.to_buffer
               (Nx.set [ I (size - 1) ] (Nx.scalar Nx.uint8 0x80) ones)
           in
           B.copy ~src ~dst:file;
           let mapped =
             require_ok ~pp
               (B.borrow Nx_device.host (require_ok ~pp (B.of_file path)))
           in
           let m =
             Nx.of_buffer Nx.bit
               [| 8 * size |]
               (B.view mapped ~offset:0 Nx_dtype.Scalar.Bit (8 * size))
           in
           let v = Nx.shrink [| (3, 8 * size) |] m in
           equal bool true (Nx.item [] (Nx.any v));
           equal int64 1L (Nx.item [] (Nx.count v)));
     ]
    @ List.map
        (fun (name, f) ->
          prop (name ^ " writes the bits past its last element as 0") mask
            (fun m ->
              let t = f m in
              cover "a length inside a byte" (Nx.numel t mod 8 <> 0);
              equal int 0 (tail t)))
        fresh)

(* Counting *)

let counting =
  group "count"
    [
      prop "count of a bit mask is the number of its true elements" mask
        (fun m ->
          cover_views m;
          let expected =
            Array.fold_left
              (fun n b -> if b then n + 1 else n)
              0 (Nx.to_array m.bools)
          in
          equal int64 (Int64.of_int expected) (Nx.item [] (Nx.count m.bits)));
      test "count of a bool mask is the sum of its cast" (fun () ->
          let m = bools [ true; false; true; true ] in
          equal int64 3L (Nx.item [] (Nx.count m)));
      test "count of an empty mask is 0" (fun () ->
          equal int64 0L (Nx.item [] (Nx.count (Nx.zeros Nx.bit [| 0 |]))));
      test "count along axes keeps the others" (fun () ->
          let m =
            Nx.cast Nx.bit
              (Nx.init Nx.bool [| 3; 10 |] (fun i -> i.(1) < i.(0) + 2))
          in
          equal (array int64) [| 2L; 3L; 4L |]
            (Nx.to_array (Nx.count ~axes:[ 1 ] m));
          equal (array int) [| 3; 1 |]
            (Nx.shape (Nx.count ~axes:[ 1 ] ~keepdims:true m)));
    ]

(* Elements *)

let elements =
  group "elements"
    [
      prop "init and item agree at every element" (Gen.pair length offset)
        (fun (n, k) ->
          let f i = i.(0) * k mod 3 = 1 in
          let m = Nx.init Nx.bit [| n |] f in
          for i = 0 to n - 1 do
            equal ~msg:(string_of_int i) bool (f [| i |]) (Nx.item [ i ] m)
          done);
      test "a mask prints as its bool does" (fun () ->
          let b = bools [ true; false; true ] in
          let printed t = Format.asprintf "%a" Nx.pp t in
          equal string (printed b) (printed (Nx.cast Nx.bit b)));
      test "to_bigarray refuses a mask, as it refuses bool" (fun () ->
          raises_invalid_arg (fun () -> Nx.to_bigarray (bools [ true ]));
          raises_invalid_arg (fun () -> Nx.to_bigarray (bits [ true ])));
      test "arange bit holds what arange bool holds" (fun () ->
          equal same (Nx.arange Nx.bool 0 2 1)
            (Nx.cast Nx.bool (Nx.arange Nx.bit 0 2 1));
          raises_invalid_arg (fun () -> Nx.arange Nx.bool 0 3 1);
          raises_invalid_arg (fun () -> Nx.arange Nx.bit 0 3 1));
      prop "a view at any bit reads, gets and prints as its bool" mask (fun m ->
          cover_views m;
          equal ~msg:"to_array" (array bool) (Nx.to_array m.bools)
            (Nx.to_array m.bits);
          let printed t = Format.asprintf "%a" Nx.pp t in
          (* An empty tensor prints its dtype's name. *)
          let renamed s = String.concat "bool" (Str_split.on "bit" s) in
          equal ~msg:"printed" string (printed m.bools)
            (renamed (printed m.bits));
          if Nx.ndim m.bits > 0 && Nx.dim 0 m.bits > 0 then
            let last = Nx.dim 0 m.bits - 1 in
            equal ~msg:"get of its last row" same (Nx.get [ last ] m.bools)
              (Nx.cast Nx.bool (Nx.get [ last ] m.bits)));
    ]

(* Packed bytes *)

(* Rows of [k] elements of a mask in any layout. *)
let rows k = Gen.bind (Gen.int_range 0 9) (fun r -> mask_of [| r; k |])

(* The bit patterns of a byte, as [packed] writes them. *)
let byte_bits b = Array.init 8 (fun j -> (b lsr j) land 1 = 1)

let nibble_rows dtype lo hi =
  viewed ~pp:Format.pp_print_int ~layout:row_layout
    ~shape:(Gen.map (fun r -> [| r; 2 |]) (Gen.int_range 0 9))
    dtype (Gen.int_range lo hi)

let low_nibble_first lo_hi =
  Array.map (fun (lo, hi) -> lo land 15 lor ((hi land 15) lsl 4)) lo_hi

let row_pairs t =
  let vs = Nx.to_array t in
  Array.init (Array.length vs / 2) (fun i -> (vs.(2 * i), vs.((2 * i) + 1)))

let bytes =
  group "packed bytes"
    [
      prop "bitcast uint8 of rows of eight bits in any layout packs each row"
        (rows 8) (fun m ->
          cover_views m;
          let vs = Nx.to_array m.bools in
          equal (array int) (packed vs)
            (Nx.to_array (Nx.bitcast Nx.uint8 m.bits)));
      prop "bitcast bit of bytes in any layout reads bit i mod 8 of byte i / 8"
        (viewed ~pp:Format.pp_print_int Nx.uint8 (Gen.int_range 0 255))
        (fun b ->
          let m = Nx.bitcast Nx.bit b in
          equal ~msg:"shape" (array int)
            (Array.append (Nx.shape b) [| 8 |])
            (Nx.shape m);
          equal (array bool)
            (Array.concat (List.map byte_bits (Array.to_list (Nx.to_array b))))
            (Nx.to_array m));
      prop "bitcast uint8 of rows of two int4 in any layout holds the first low"
        (nibble_rows Nx.int4 (-8) 7) (fun q ->
          equal (array int)
            (low_nibble_first (row_pairs q))
            (Nx.to_array (Nx.bitcast Nx.uint8 q)));
      prop
        "bitcast uint8 of rows of two uint4 in any layout holds the first low"
        (nibble_rows Nx.uint4 0 15) (fun q ->
          equal (array int)
            (low_nibble_first (row_pairs q))
            (Nx.to_array (Nx.bitcast Nx.uint8 q)));
      test
        "bitcast int4 of four bits reads them as one nibble, the first lowest"
        (fun () ->
          let m =
            Nx.reshape [| 2; 4 |]
              (bits [ true; false; false; true; false; false; false; true ])
          in
          equal (array int) [| -7; -8 |] (Nx.to_array (Nx.bitcast Nx.int4 m));
          equal (array int) [| 9; 8 |] (Nx.to_array (Nx.bitcast Nx.uint4 m)));
      test "bitcast bit of an int4 gives its four bits, the lowest first"
        (fun () ->
          let q = Nx.create Nx.int4 [| 2 |] [| -7; 6 |] in
          equal (array bool)
            [| true; false; false; true; false; true; true; false |]
            (Nx.to_array (Nx.bitcast Nx.bit q)));
      test
        "bitcast uint8 of rows of eight bits from byte 1 shares their storage"
        (fun () ->
          let m =
            Nx.cast Nx.bit (Nx.init Nx.bool [| 32 |] (fun i -> i.(0) mod 3 = 0))
          in
          let rows = Nx.reshape [| 3; 8 |] (Nx.shrink [| (8, 32) |] m) in
          is_true
            (share_memory
               (Nx_test.storage (Nx.bitcast Nx.uint8 rows))
               (Nx_test.storage m)));
    ]

(* Across buffers and devices *)

let d1 = Devices.d1
and d2 = Devices.d2

(* A sub-byte value under a layout, and the witness of its elements: bytes past
   its last element hold no element, so values compare by their elements. *)
type sub_byte = Sub : string * ('a, 'b) Nx.t Gen.t * 'a testable -> sub_byte

let sub_bytes =
  [
    Sub ("bit", Gen.map (fun m -> m.bits) mask, bool);
    Sub
      ( "int4",
        viewed ~pp:Format.pp_print_int Nx.int4 (Gen.int_range (-8) 7),
        int );
    Sub
      ( "uint4",
        viewed ~pp:Format.pp_print_int Nx.uint4 (Gen.int_range 0 15),
        int );
  ]

let on_host x = Nx.place Nx.Placement.host x

let devices =
  group "buffers and devices"
    (List.concat_map
       (fun (Sub (name, tensors, w)) ->
         [
           prop (name ^ ": of_buffer reads back to_buffer's elements") tensors
             (fun t ->
               equal (tensor w) t
                 (Nx.of_buffer (Nx.dtype t) (Nx.shape t) (Nx.to_buffer t)));
           prop
             (name ^ ": a value placed on devices and back is the value")
             (Runtimes.placed [ d1; d2 ] tensors)
             (fun (t, p) ->
               let x = Nx.place p t in
               equal ~msg:"back on the host" (tensor w) t (on_host x);
               let buffers, v = Nx.shards x in
               equal ~msg:"of_shards over shards" (tensor w) t
                 (on_host (Nx.of_shards p (Nx.dtype x) v buffers)));
           prop (name ^ ": a copy on a device is the host's copy") tensors
             (fun t ->
               equal (tensor w) (Nx.copy t)
                 (on_host (Nx.copy (Nx.place (Nx.Placement.on d1) t))));
         ])
       sub_bytes
    @ [
        prop "logical_and of masks on a device is the host's" two (fun (a, b) ->
            let on = Nx.place (Nx.Placement.on d1) in
            equal same
              (Nx.cast Nx.bool (Nx.logical_and a.bits b.bits))
              (Nx.cast Nx.bool
                 (on_host (Nx.logical_and (on a.bits) (on b.bits)))));
      ])

(* From two domains *)

(* Calls on one mask from two domains at once give what they give one after the
   other: each reads the mask's bits and writes words of its own result. *)

let shared = abstract "m"

let mask_values =
  Gen.with_pp
    (fun ppf vs ->
      Array.iter (fun v -> Format.pp_print_char ppf (if v then '1' else '0')) vs)
    (Gen.array ~size:(Gen.int_range 0 300) Gen.bool)

(* The system's mask starts at bit 3 of its storage. *)
let shared_mask vs =
  let n = Array.length vs in
  let padded =
    Array.init (n + 5) (fun i -> i >= 3 && i < n + 3 && vs.(i - 3))
  in
  Nx.shrink
    [| (3, n + 3) |]
    (Nx.cast Nx.bit (Nx.create Nx.bool [| n + 5 |] padded))

let on_both { name; f } =
  command name
    (shared ^-> returns (array bool))
    (fun b -> Nx.to_array (f b))
    (fun m -> Nx.to_array (f m))

let concurrent_calls =
  command "mask"
    (mask_values @-> makes shared)
    (fun vs -> Nx.create Nx.bool [| Array.length vs |] vs)
    shared_mask
  :: command "count"
       (shared ^-> returns int64)
       (fun b -> Nx.item [] (Nx.count b))
       (fun m -> Nx.item [] (Nx.count m))
  :: List.map on_both
       (List.filter
          (fun { name; _ } ->
            List.mem name
              [
                "copy";
                "logical_not";
                "a copy of its flip";
                "concatenate between parts of 13";
                "pad";
                "take with indices outside";
                "set of a window";
                "scatter with Set";
                "every third element";
              ])
          unaries)

let domains =
  group "domains"
    [
      stateful "calls on one mask give their bool results" concurrent_calls;
      stateful ~tags:[ "slow" ] ~domains:2
        "calls on one mask from two domains at once give their bool results"
        concurrent_calls;
    ]

(* Many workers *)

(* nx.cpu splits a write of tens of millions of elements between workers by the
   64-bit words of the destination. A destination that starts inside a word
   shares its first word with what lies before it. *)

let pattern n =
  Nx.reshape
    [| 8 * n |]
    (Nx.bitcast Nx.bit
       (Nx.init Nx.uint8 [| n |] (fun i -> i.(0) * 37 land 255)))

let differences a b = Nx.item [] (Nx.count (Nx.logical_xor a b))

let workers =
  group "many workers"
    [
      slow "a part of 2^28 bits concatenated at bit 13 holds its bits"
        (fun () ->
          let big = pattern (1 lsl 25) and n = 1 lsl 28 in
          let thirteen = bits (List.init 13 (fun i -> i mod 3 = 0)) in
          let c = Nx.concatenate ~axis:0 [ thirteen; big; thirteen ] in
          equal ~msg:"the first part" same (Nx.cast Nx.bool thirteen)
            (Nx.cast Nx.bool (Nx.shrink [| (0, 13) |] c));
          equal ~msg:"bits that differ" int64 0L
            (differences big (Nx.shrink [| (13, n + 13) |] c));
          equal ~msg:"the last part" same (Nx.cast Nx.bool thirteen)
            (Nx.cast Nx.bool (Nx.shrink [| (n + 13, n + 26) |] c)));
      slow "a window of 2^28 bits set from bit 5 keeps the bits around it"
        (fun () ->
          let n = 1 lsl 28 in
          let src = pattern (1 lsl 25) in
          let t = Nx.set [ R (5, n + 5) ] src (Nx.ones Nx.bit [| n + 70 |]) in
          equal ~msg:"bits that differ" int64 0L
            (differences src (Nx.shrink [| (5, n + 5) |] t));
          equal ~msg:"the ones around it" int64 (Int64.of_int 70)
            (Int64.sub (Nx.item [] (Nx.count t)) (Nx.item [] (Nx.count src))));
      slow
        "int4 parts of 13 around 2^26 elements concatenate as their int8 twins"
        (fun () ->
          let n = (1 lsl 26) + 3 in
          let big = Nx.init Nx.int8 [| n |] (fun i -> (i.(0) * 7 mod 16) - 8) in
          let thirteen = Nx.init Nx.int8 [| 13 |] (fun i -> i.(0) - 6) in
          let parts = [ thirteen; big; thirteen ] in
          let q = Nx.concatenate ~axis:0 (List.map (Nx.cast Nx.int4) parts) in
          equal bool true
            (Nx.item []
               (Nx.array_equal
                  (Nx.concatenate ~axis:0 parts)
                  (Nx.cast Nx.int8 q))));
    ]

let () =
  exit
    (run "nx bit"
       [
         storage;
         one_meaning;
         returning_other_dtypes;
         casts;
         arithmetic;
         exact;
         counting;
         elements;
         bytes;
         devices;
         domains;
         workers;
       ])
