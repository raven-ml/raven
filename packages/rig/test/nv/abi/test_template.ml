(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Templates against template.mli: words little-endian, a slot's patch filling
   the low bits of the narrowest 1, 2, 4 or 8-byte word that holds its bits and
   keeping the word's other bits, and templates one after another. *)

open Windtrap
open Rig_nv_abi
module S = Rig_nv_abi_support

(* Words *)

(* A word over an operand: 32 bits shifted, or 64 bits with an addition. *)
type word = U32 of int | U64 of int64

let pp_word ppf = function
  | U32 s -> Format.fprintf ppf "u32 %d" s
  | U64 n -> Format.fprintf ppf "u64 0x%Lx" n

let make w v =
  match w with U32 s -> Template.u32 s v | U64 n -> Template.u64 n v

let word =
  Gen.with_pp pp_word
    Gen.(
      one_of
        [ map (fun s -> U32 s) (int_range 0 63); map (fun n -> U64 n) S.u64 ])

(* An unsigned integer an int holds. *)
let value = Gen.int_range 0 max_int

let words =
  group ~timeout:10. "words"
    [
      prop "a word's slot, filled, is the word of its value"
        (Gen.pair word value) (fun (w, v) ->
          equal string (make w (Int v)).bytes
            (Template.fill (fun _ -> Int64.of_int v) (make w (Slot 0))));
      test "u32 is the low 32 bits of the value shifted, little-endian"
        (fun () ->
          equal (list int) [ 0x2345_6789 ]
            (S.words (Template.u32 4 (Int 0x1_2345_6789_0)).bytes));
      test "u64 is its low word, then its high word" (fun () ->
          equal (list int) [ 2; 1 ]
            (S.words (Template.u64 0L (Int 0x1_0000_0002)).bytes));
      test "u64's addition wraps modulo 2^64" (fun () ->
          equal (list int) [ 1; 0 ] (S.words (Template.u64 (-1L) (Int 2)).bytes);
          equal (list int) [ 1; 0 ]
            (S.words
               (Template.fill (fun _ -> 2L) (Template.u64 (-1L) (Slot 0)))));
      cases ~name:fst "a negative known value is refused"
        [
          ("u32", fun () -> Template.u32 0 (Int (-1)));
          ("u64", fun () -> Template.u64 0L (Int min_int));
        ]
        (fun (_, f) ->
          raises_match (Exn.invalid_arg ~substring:"unsigned") (fun () -> f ()));
      test "a fill shifts the sum in with zeros" (fun () ->
          equal (list int) [ 0x3000_0000 ]
            (S.words
               (Template.fill
                  (fun _ -> 0x2ff_ffff_ff00L)
                  (Template.concat
                     [
                       {
                         bytes = "\000\000\000\000";
                         patches =
                           [
                             {
                               at = 0;
                               bits = 32;
                               slot = 0;
                               add = 0x100L;
                               shift = 12;
                             };
                           ];
                       };
                     ]))));
    ]

(* Templates one after another *)

let templates =
  Gen.list ~size:(Gen.int_range 0 8)
    Gen.(
      let+ w = word and+ v = value and+ slot = bool in
      (w, v, slot))

let concat =
  group ~timeout:10. "concat"
    [
      prop "templates one after another fill as their fills one after another"
        templates (fun ts ->
          let value i =
            let _, v, _ = List.nth ts i in
            Int64.of_int v
          in
          let t =
            List.mapi
              (fun i (w, v, slot) ->
                make w (if slot then Template.Slot i else Int v))
              ts
          in
          cover "a slot" (List.exists (fun (_, _, s) -> s) ts);
          equal string
            (String.concat "" (List.map (Template.fill value) t))
            (Template.fill value (Template.concat t)));
    ]

(* Patches, as descriptors hold them *)

let width bits = List.find (fun w -> 8 * w >= bits) [ 1; 2; 4; 8 ]
let mask bits = if bits = 64 then -1L else Int64.(sub (shift_left 1L bits) 1L)

let read b at w =
  let n = ref 0L in
  for i = w - 1 downto 0 do
    n :=
      Int64.(
        logor (shift_left !n 8) (of_int (Char.code (Bytes.get b (at + i)))))
  done;
  !n

let write b at w n =
  for i = 0 to w - 1 do
    Bytes.set b (at + i)
      (Char.chr (Int64.to_int (Int64.shift_right_logical n (8 * i)) land 0xff))
  done

(* [t.bytes] with each patch filled as template.mli describes it. *)
let reference value (t : Template.t) =
  let b = Bytes.of_string t.bytes in
  List.iter
    (fun (p : Template.patch) ->
      let w = width p.bits and m = mask p.bits in
      let x =
        Int64.shift_right_logical (Int64.add (value p.slot) p.add) p.shift
      in
      let kept = Int64.logand (read b p.at w) (Int64.lognot m) in
      write b p.at w (Int64.logor kept (Int64.logand x m)))
    t.patches;
  Bytes.to_string b

(* [cls]'s descriptor of a kernel with two banks, every value a slot. *)
let descriptor cls =
  let k =
    S.kernel
      ~banks:
        [
          { index = 0; offset = 0; bytes = 0x17c };
          { index = 3; offset = 0x200; bytes = 0x40 };
        ]
      ()
  in
  Qmd.make
    (S.launch (S.gpu ~compute_class:cls ()) k)
    {
      program = Slot 0;
      local = Slot 1;
      banks = [ (0, Slot 2); (3, Slot 3) ];
      grid = (Slot 4, Slot 5, Slot 6);
      block = (Slot 7, Slot 8, Slot 9);
      shared = Some (Slot 10);
      chain = Some (Slot 11);
      releases = [ Release (Slot 12, Slot 13); Stamped (Slot 14, Slot 15) ];
    }

let patches =
  group ~timeout:10. "patches"
    [
      cases ~name:S.class_name
        "patches are by increasing offset, their words in the bytes and apart"
        S.classes (fun cls ->
          let t = descriptor cls in
          let _ =
            List.fold_left
              (fun next (p : Template.patch) ->
                let msg = Format.asprintf "%a" S.pp_patch p in
                at_least ~msg int ~than:1 p.bits;
                at_most ~msg int ~than:64 p.bits;
                at_least ~msg int ~than:next p.at;
                at_most ~msg int ~than:(String.length t.bytes)
                  (p.at + width p.bits);
                p.at + width p.bits)
              0 t.patches
          in
          ());
      cases ~name:S.class_name "a patch's field is zero in the bytes" S.classes
        (fun cls ->
          let t = descriptor cls in
          let b = Bytes.of_string t.bytes in
          List.iter
            (fun (p : Template.patch) ->
              equal
                ~msg:(Format.asprintf "%a" S.pp_patch p)
                int64 0L
                (Int64.logand (read b p.at (width p.bits)) (mask p.bits)))
            t.patches);
      prop "fill writes each patch's field, and keeps every other bit"
        (Gen.pair S.compute_class (Gen.list ~size:(Gen.constant 16) S.u64))
        (fun (cls, vs) ->
          let t = descriptor cls in
          let value = List.nth vs in
          equal string (reference value t) (Template.fill value t));
    ]

let () = exit (run "rig_nv_abi.template" [ words; concat; patches ])
