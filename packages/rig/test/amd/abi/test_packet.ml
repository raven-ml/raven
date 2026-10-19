(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Packets as their interpreters read them: sizes in words, little-endian bytes,
   terms over 64 bits, and the holes of a template. *)

open Windtrap
open Rig_amd_abi
module S = Rig_amd_abi_support

let timeout = S.timeout
let id (v : int64) = v
let words = S.words

(* Packets over values a template knows now, or leaves for later. *)

type v = Known of int64 | Later of int64

let value = function Known n | Later n -> n
let known = function Known n -> Some n | Later _ -> None

let pp_v ppf = function
  | Known n -> Format.fprintf ppf "Known 0x%Lx" n
  | Later n -> Format.fprintf ppf "Later 0x%Lx" n

let pp_term ppf (t : v Packet.term) =
  Format.fprintf ppf "{ value = %a; add = 0x%Lx; shift = %d; set = 0x%Lx }"
    pp_v t.value t.add t.shift t.set

let pp_word ppf : v Packet.word -> unit = function
  | Dword n -> Format.fprintf ppf "Dword 0x%x" n
  | W32 t -> Format.fprintf ppf "W32 (%a)" pp_term t
  | W64 t -> Format.fprintf ppf "W64 (%a)" pp_term t

let pp_packet ppf p =
  Format.fprintf ppf "[%a]"
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
       pp_word)
    p

(* 64-bit integers at the edges of their halves and of the type. *)
let u64 =
  Gen.frequency
    [
      (3, Gen.int64);
      ( 1,
        Gen.of_list
          [
            0L;
            1L;
            -1L;
            0xffff_ffffL;
            0x1_0000_0000L;
            0x8000_0000L;
            Int64.min_int;
            Int64.max_int;
          ] );
    ]

let v =
  Gen.(
    frequency
      [ (2, map (fun n -> Known n) u64); (1, map (fun n -> Later n) u64) ])

(* A constant, or the one that leaves the value as it is. *)
let or_none none g = Gen.frequency [ (1, Gen.constant none); (2, g) ]

let term : v Packet.term Gen.t =
  let open Gen in
  let+ value = v
  and+ add = or_none 0L u64
  and+ shift = or_none 0 (int_range 0 63)
  and+ set = or_none 0L u64 in
  { Packet.value; add; shift; set }

let word : v Packet.word Gen.t =
  let open Gen in
  frequency
    [
      (1, map (fun n -> Packet.Dword n) int);
      (2, map (fun t -> Packet.W32 t) term);
      (2, map (fun t -> Packet.W64 t) term);
    ]

let packet = Gen.with_pp pp_packet (Gen.list ~size:(Gen.int_range 0 12) word)

(* The terms' meaning, as the type states it: unsigned 64-bit integers, an
   addition modulo 2^64, a logical shift, then an or. *)
let eval (t : v Packet.term) =
  Int64.logor
    (Int64.shift_right_logical (Int64.add (value t.value) t.add) t.shift)
    t.set

let low n = Int64.to_int n land 0xffff_ffff
let high n = Int64.to_int (Int64.shift_right_logical n 32) land 0xffff_ffff

let reference p =
  List.concat_map
    (function
      | Packet.Dword n -> [ n land 0xffff_ffff ]
      | W32 t -> [ low (eval t) ]
      | W64 t ->
          let n = eval t in
          [ low n; high n ])
    p

let later (t : v Packet.term) =
  match t.value with Later _ -> true | Known _ -> false

let hole : v Packet.word -> bool = function
  | Dword _ -> false
  | W32 t | W64 t -> later t

let encode =
  group ~timeout "encode"
    [
      prop "a packet is four bytes a word of its size" packet (fun p ->
          equal int (4 * Packet.size p) (String.length (Packet.encode value p)));
      prop "words are little-endian, terms unsigned over 64 bits" packet
        (fun p ->
          cover "a 64-bit word"
            (List.exists (function Packet.W64 _ -> true | _ -> false) p);
          equal (list int) (reference p) (words (Packet.encode value p)));
      test "an or keeps bit 63" (fun () ->
          equal (list int) [ 0x100; 0x8000_0000 ]
            (words
               (Packet.encode id
                  [ W64 { (Packet.value 0x100L) with set = Int64.min_int } ])));
      cases ~name:string_of_int "a shift outside [0;63] is refused"
        [ min_int; -1; 64; max_int ] (fun n ->
          raises_match (Exn.invalid_arg ~substring:"Packet.encode") (fun () ->
              Packet.encode id
                [ Dword 0; W32 { (Packet.value 1L) with shift = n } ]));
    ]

(* [b] with each hole's word, encoded with every value, at its index. *)
let fill b holes =
  let b = Bytes.of_string b in
  List.iter
    (fun (i, w) ->
      let s = Packet.encode value [ w ] in
      Bytes.blit_string s 0 b (4 * i) (String.length s))
    holes;
  Bytes.to_string b

(* The index of each word that holds a value for later. *)
let holes p =
  let _, hs =
    List.fold_left
      (fun (i, hs) w ->
        (i + Packet.size [ w ], if hole w then (i, w) :: hs else hs))
      (0, []) p
  in
  List.rev hs

let template =
  group ~timeout "template"
    [
      prop "a template's holes, filled, are the encoding" packet (fun p ->
          let b, hs = Packet.template known p in
          cover "a hole" (hs <> []);
          cover "no hole" (hs = [] && p <> []);
          equal string (Packet.encode value p) (fill b hs));
      prop "the holes are the words that hold a value for later, in order"
        packet (fun p ->
          let _, hs = Packet.template known p in
          equal
            (list (pair int (Testable.make ~pp:pp_word ~equal:( = ))))
            (holes p) hs);
      prop "a hole's words are zero" packet (fun p ->
          let b, hs = Packet.template known p in
          List.iter
            (fun (i, w) ->
              equal (list int)
                (List.init (Packet.size [ w ]) (fun _ -> 0))
                (List.filteri
                   (fun j _ -> j >= i && j < i + Packet.size [ w ])
                   (words b)))
            hs);
      test "a hole's shift is not evaluated" (fun () ->
          let _, holes =
            Packet.template known
              [ W32 { (Packet.value (Later 0L)) with shift = 64 } ]
          in
          equal int 1 (List.length holes));
      cases ~name:string_of_int "a known word's shift outside [0;63] is refused"
        [ -1; 64 ] (fun n ->
          raises_match (Exn.invalid_arg ~substring:"Packet.template") (fun () ->
              Packet.template known
                [ W32 { (Packet.value (Known 1L)) with shift = n } ]));
    ]

let () = exit (run "rig_amd_abi.packet" [ encode; template ])
