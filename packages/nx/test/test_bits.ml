(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Packed bitmaps, against a model of their bits read from the bytes by Arrow's
   layout: bit [i] is bit [(offset + i) mod 8] of byte [(offset + i) / 8]. The
   bytes are drawn whole, so the bits outside the range are arbitrary. *)

open Windtrap
open Nx_test

type drawn = { bytes : int array; offset : int; length : int }

let pp_drawn ppf d =
  Format.fprintf ppf "%d bits from bit %d of [%a]" d.length d.offset
    (Format.pp_print_seq
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       (fun ppf b -> Format.fprintf ppf "0x%02x" b))
    (Array.to_seq d.bytes)

(* Bitmaps of [length] bits, from bit 0 to 15, with a spare byte or none. *)
let of_length length =
  let open Gen in
  let* offset = int_range 0 15 in
  let* spare = int_range 0 1 in
  let+ bytes =
    array
      ~size:(constant (((offset + length + 7) / 8) + spare))
      (int_range 0 255)
  in
  { bytes; offset; length }

let drawn = Gen.with_pp pp_drawn (Gen.bind (Gen.int_range 0 40) of_length)

(* Two bitmaps of one length, at offsets drawn apart. *)
let two =
  Gen.with_pp
    (fun ppf (a, b) -> Format.fprintf ppf "%a and %a" pp_drawn a pp_drawn b)
    Gen.(
      let* n = int_range 0 40 in
      pair (of_length n) (of_length n))

let model d =
  Array.init d.length (fun i ->
      let k = d.offset + i in
      (d.bytes.(k / 8) lsr (k mod 8)) land 1 = 1)

let bitmap d =
  Nx.Bits.v ~offset:d.offset ~length:d.length
    (Nx.create Nx.uint8 [| Array.length d.bytes |] d.bytes)

let bits b = Nx.to_array (Nx.Bits.to_bool b)
let mask m = Nx.create Nx.bool [| Array.length m |] m
let bools = Gen.array ~size:(Gen.int_range 0 40) Gen.bool

let bytes_of b =
  let bytes, offset = Nx.Bits.bytes b in
  { bytes = Nx.to_array bytes; offset; length = Nx.Bits.length b }

(* Bitmaps *)

let packing =
  group "packing"
    [
      prop "v reads bit offset + i of the bytes, least significant first" drawn
        (fun d -> equal (array bool) (model d) (bits (bitmap d)));
      prop "to_bool reads back what of_bool packs" bools
        (Law.round_trip (array bool)
           (Testable.contramap bits (array bool))
           (fun m -> Nx.Bits.of_bool (mask m))
           bits);
      prop "of_bool packs eight bits a byte from bit 0 of the first byte" bools
        (fun m ->
          let d = bytes_of (Nx.Bits.of_bool (mask m)) in
          equal int 0 d.offset;
          equal int ((Array.length m + 7) / 8) (Array.length d.bytes);
          equal (array bool) m (model d));
      prop
        "bytes is an offset below 8 and the bytes the bits reach, which v \
         reads back"
        drawn (fun d ->
          let b = bitmap d in
          let back = bytes_of b in
          is_true ~msg:"offset in [0, 7]" (back.offset >= 0 && back.offset < 8);
          equal int ((back.offset + d.length + 7) / 8) (Array.length back.bytes);
          equal (array bool) (model d) (model back));
      prop "length is the number of bits" drawn (fun d ->
          equal int d.length (Nx.Bits.length (bitmap d)));
      prop "count is the number of bits set in the range" drawn (fun d ->
          let set = Array.fold_left (fun n b -> if b then n + 1 else n) 0 in
          equal int64
            (Int64.of_int (set (model d)))
            (Nx.item [] (Nx.Bits.count (bitmap d))));
      test "count ignores every bit of a byte outside a range within it"
        (fun () ->
          let b =
            Nx.Bits.v ~offset:3 ~length:2
              (Nx.create Nx.uint8 [| 1 |] [| 0xFF |])
          in
          equal int64 2L (Nx.item [] (Nx.Bits.count b)));
      test "a bitmap of no bit counts none" (fun () ->
          equal int64 0L
            (Nx.item [] (Nx.Bits.count (Nx.Bits.of_bool (mask [||])))));
      cases "v refuses bytes that do not hold the bits" ~name:fst
        [
          ( "bytes of two axes",
            fun () -> Nx.Bits.v ~length:1 (Nx.zeros Nx.uint8 [| 1; 1 |]) );
          ( "a negative offset",
            fun () ->
              Nx.Bits.v ~offset:(-1) ~length:1 (Nx.zeros Nx.uint8 [| 1 |]) );
          ( "a negative length",
            fun () -> Nx.Bits.v ~length:(-1) (Nx.zeros Nx.uint8 [| 1 |]) );
          ( "one bit past the bytes",
            fun () -> Nx.Bits.v ~offset:3 ~length:6 (Nx.zeros Nx.uint8 [| 1 |])
          );
        ]
        (fun (_, f) -> raises_invalid_arg f);
      test "v takes bits that end at the last bit of the bytes" (fun () ->
          equal int 13
            (Nx.Bits.length
               (Nx.Bits.v ~offset:3 ~length:13 (Nx.zeros Nx.uint8 [| 2 |]))));
      test "of_bool refuses a mask of two axes" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.Bits.of_bool (Nx.zeros Nx.bool [| 2; 2 |])));
    ]

(* Logic *)

let logic =
  let pointwise f a b = Array.map2 f (model a) (model b) in
  group "logic"
    [
      prop "logand is the conjunction of the bits" two (fun (a, b) ->
          cover "one offset" (a.offset mod 8 = b.offset mod 8);
          cover "two offsets" (a.offset mod 8 <> b.offset mod 8);
          equal (array bool) (pointwise ( && ) a b)
            (bits (Nx.Bits.logand (bitmap a) (bitmap b))));
      prop "logor is the disjunction of the bits" two (fun (a, b) ->
          equal (array bool) (pointwise ( || ) a b)
            (bits (Nx.Bits.logor (bitmap a) (bitmap b))));
      prop "lognot flips every bit" drawn (fun d ->
          equal (array bool)
            (Array.map not (model d))
            (bits (Nx.Bits.lognot (bitmap d))));
      test "logand and logor refuse bitmaps of different lengths" (fun () ->
          let a = Nx.Bits.of_bool (mask [| true; false |])
          and b = Nx.Bits.of_bool (mask [| true |]) in
          raises_invalid_arg (fun () -> Nx.Bits.logand a b);
          raises_invalid_arg (fun () -> Nx.Bits.logor a b));
    ]

(* Selecting *)

let range =
  Gen.with_pp
    (fun ppf (d, o, n) ->
      Format.fprintf ppf "bits %d to %d of %a" o (o + n) pp_drawn d)
    Gen.(
      let* d = drawn in
      let* o = int_range 0 d.length in
      let+ n = int_range 0 (d.length - o) in
      (d, o, n))

(* Indices across the range and past both ends, and the extremes of int64. *)
let indices n =
  Gen.(
    array ~size:(int_range 0 12)
      (frequency
         [
           (6, map Int64.of_int (int_range (-2) (n + 2)));
           ( 1,
             of_list
               ~pp:(fun ppf -> Format.fprintf ppf "%Ld")
               [
                 Int64.min_int;
                 Int64.succ Int64.min_int;
                 Int64.pred Int64.max_int;
                 Int64.max_int;
               ] );
         ]))

let taken =
  Gen.with_pp
    (fun ppf (d, i) ->
      Format.fprintf ppf "%a at [%a]" pp_drawn d
        (Format.pp_print_seq
           ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
           (fun ppf i -> Format.fprintf ppf "%Ld" i))
        (Array.to_seq i))
    Gen.(
      let* d = drawn in
      pair (constant d) (indices d.length))

let selecting =
  group "selecting"
    [
      prop "sub is the bits of the range" range (fun (d, o, n) ->
          equal (array bool)
            (Array.sub (model d) o n)
            (bits (Nx.Bits.sub (bitmap d) ~offset:o ~length:n)));
      prop "sub shares the bytes" range (fun (d, o, n) ->
          let b = bitmap d in
          assume (n > 0);
          is_true
            (share_memory
               (storage
                  (fst (Nx.Bits.bytes (Nx.Bits.sub b ~offset:o ~length:n))))
               (storage (fst (Nx.Bits.bytes b)))));
      cases "sub refuses a range outside the bits" ~name:fst
        [
          ("a negative offset", (-1, 1));
          ("a negative length", (0, -1));
          ("one bit past the end", (3, 3));
        ]
        (fun (_, (offset, length)) ->
          raises_invalid_arg (fun () ->
              Nx.Bits.sub
                (Nx.Bits.of_bool (mask [| true; true; true; true; true |]))
                ~offset ~length));
      prop
        "take reads the bit at each index, and an unset bit outside the range"
        taken (fun (d, i) ->
          let m = model d in
          let expected =
            Array.map
              (fun k ->
                k >= 0L && k < Int64.of_int d.length && m.(Int64.to_int k))
              i
          in
          equal (array bool) expected
            (bits
               (Nx.Bits.take
                  ~indices:(Nx.create Nx.int64 [| Array.length i |] i)
                  (bitmap d))));
      test "take refuses indices of two axes" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.Bits.take
                ~indices:(Nx.zeros Nx.int64 [| 1; 1 |])
                (Nx.Bits.of_bool (mask [| true |]))));
      prop "concat is the bits one after the other"
        (Gen.with_pp
           (Format.pp_print_list pp_drawn)
           (Gen.list ~size:(Gen.int_range 1 4) drawn))
        (fun ds ->
          equal (array bool)
            (Array.concat (List.map model ds))
            (bits (Nx.Bits.concat (List.map bitmap ds))));
      test "concat refuses no bitmap" (fun () ->
          raises_invalid_arg (fun () -> Nx.Bits.concat []));
    ]

let () = exit (run "nx bits" [ packing; logic; selecting ])
