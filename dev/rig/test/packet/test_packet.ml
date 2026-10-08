(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Packets as their interpreters read them: sizes in words, little-endian bytes,
   terms over 64 bits, the holes of a template, and a C template filled. *)

open Windtrap
open Rig_packet

let id (v : int64) = v

let words s =
  List.init
    (String.length s / 4)
    (fun i -> Int32.to_int (String.get_int32_le s (4 * i)) land 0xffff_ffff)

(* Printing *)

let rec pp_term pp_v ppf = function
  | Value v -> pp_v ppf v
  | Add (t, n) -> Format.fprintf ppf "Add (%a, 0x%Lx)" (pp_term pp_v) t n
  | Shift (t, n) -> Format.fprintf ppf "Shift (%a, %d)" (pp_term pp_v) t n
  | Or (t, n) -> Format.fprintf ppf "Or (%a, 0x%Lx)" (pp_term pp_v) t n

let pp_word pp_v ppf = function
  | Dword n -> Format.fprintf ppf "Dword 0x%x" n
  | W32 t -> Format.fprintf ppf "W32 (%a)" (pp_term pp_v) t
  | W64 t -> Format.fprintf ppf "W64 (%a)" (pp_term pp_v) t

let pp_packet pp_v ppf p =
  Format.fprintf ppf "[%a]"
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
       (pp_word pp_v))
    p

(* Drawing *)

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

let shift = Gen.(frequency [ (3, int_range 0 63); (1, of_list [ 0; 63 ]) ])
let map2 f a b = Gen.map (fun (a, b) -> f a b) (Gen.pair a b)

(* Terms over [v] with at most [depth] operations. *)
let rec term v depth =
  let open Gen in
  let leaf = map (fun v -> Value v) v in
  if depth = 0 then leaf
  else
    let sub = term v (depth - 1) in
    frequency
      [
        (2, leaf);
        (1, map2 (fun t n -> Add (t, n)) sub u64);
        (1, map2 (fun t n -> Shift (t, n)) sub shift);
        (1, map2 (fun t n -> Or (t, n)) sub u64);
      ]

let word ?(dwords = 1) v =
  let open Gen in
  frequency
    [
      (dwords, map (fun n -> Dword n) int);
      (2, map (fun t -> W32 t) (term v 3));
      (2, map (fun t -> W64 t) (term v 3));
    ]

let is_term = function Dword _ -> false | W32 _ | W64 _ -> true
let is_w64 = function W64 _ -> true | Dword _ | W32 _ -> false

(* The terms' meaning, as the type states it: unsigned 64-bit integers, an
   addition modulo 2^64, a logical shift and an or. *)
let rec eval value = function
  | Value v -> value v
  | Add (t, n) -> Int64.add (eval value t) n
  | Shift (t, n) -> Int64.shift_right_logical (eval value t) n
  | Or (t, n) -> Int64.logor (eval value t) n

let low n = Int64.to_int n land 0xffff_ffff
let high n = Int64.to_int (Int64.shift_right_logical n 32) land 0xffff_ffff

let reference value p =
  List.concat_map
    (function
      | Dword n -> [ n land 0xffff_ffff ]
      | W32 t -> [ low (eval value t) ]
      | W64 t ->
          let n = eval value t in
          [ low n; high n ])
    p

(* Encoding *)

(* Packets over values a template knows now, or leaves for later. *)

type v = Known of int64 | Later of int64

let value = function Known n | Later n -> n
let known = function Known n -> Some n | Later _ -> None

let pp_v ppf = function
  | Known n -> Format.fprintf ppf "Known 0x%Lx" n
  | Later n -> Format.fprintf ppf "Later 0x%Lx" n

let v =
  Gen.(
    frequency
      [ (2, map (fun n -> Known n) u64); (1, map (fun n -> Later n) u64) ])

let packet =
  Gen.with_pp (pp_packet pp_v) (Gen.list ~size:(Gen.int_range 0 12) (word v))

let encode =
  group "encode"
    [
      prop "a packet is four bytes a word of its size" packet (fun p ->
          equal int (4 * size p) (String.length (encode value p)));
      prop "words are little-endian, terms unsigned over 64 bits" packet
        (fun p ->
          cover "a 64-bit word" (List.exists is_w64 p);
          equal (list int) (reference value p) (words (encode value p)));
      test "a 64-bit word is its low word, then its high word" (fun () ->
          equal (list int) [ 2; 1 ]
            (words (encode id [ W64 (Value 0x1_0000_0002L) ])));
      test "an addition wraps modulo 2^64" (fun () ->
          equal (list int) [ 1; 0 ]
            (words (encode id [ W64 (Add (Value (-1L), 2L)) ])));
      test "a shift brings in zeros" (fun () ->
          equal (list int) [ 0xffff_ffff; 1 ]
            (words (encode id [ W64 (Shift (Value (-1L), 31)) ])));
      test "an or keeps bit 63" (fun () ->
          equal (list int) [ 0x100; 0x8000_0000 ]
            (words (encode id [ W64 (Or (Value 0x100L, Int64.min_int)) ])));
      test "a term adds, then shifts, in its order" (fun () ->
          equal (list int) [ 0x3000_0000; 0 ]
            (words
               (encode id
                  [ W64 (Shift (Add (Value 0x2ff_ffff_ff00L, 0x100L), 12)) ])));
      cases ~name:string_of_int "a shift outside [0;63] is refused"
        [ min_int; -1; 64; max_int ] (fun n ->
          raises_match (Exn.invalid_arg ~substring:"Rig_packet.encode")
            (fun () ->
              encode id [ Dword 0; W32 (Add (Shift (Value 1L, n), 1L)) ]));
    ]

(* Templates *)

(* [b] with each hole's word, encoded with every value, at its index. *)
let fill b holes =
  let b = Bytes.of_string b in
  List.iter
    (fun (i, w) ->
      let s = Rig_packet.encode value [ w ] in
      Bytes.blit_string s 0 b (4 * i) (String.length s))
    holes;
  Bytes.to_string b

let rec later = function
  | Value (Later _) -> true
  | Value (Known _) -> false
  | Add (t, _) | Shift (t, _) | Or (t, _) -> later t

let hole = function Dword _ -> false | W32 t | W64 t -> later t

(* The index of each word that holds a value for later. *)
let holes p =
  let _, hs =
    List.fold_left
      (fun (i, hs) w -> (i + size [ w ], if hole w then (i, w) :: hs else hs))
      (0, []) p
  in
  List.rev hs

let template =
  group "template"
    [
      prop "a template's holes, filled, are the encoding" packet (fun p ->
          let b, hs = template known p in
          cover "a hole" (hs <> []);
          cover "no hole" (hs = [] && p <> []);
          equal string (Rig_packet.encode value p) (fill b hs));
      prop "the holes are the words that hold a value for later, in order"
        packet (fun p ->
          let _, hs = template known p in
          equal
            (list (pair int (Testable.make ~pp:(pp_word pp_v) ~equal:( = ))))
            (holes p) hs);
      prop "a hole's words are zero" packet (fun p ->
          let b, hs = template known p in
          List.iter
            (fun (i, w) ->
              equal (list int)
                (List.init (size [ w ]) (fun _ -> 0))
                (List.filteri
                   (fun j _ -> j >= i && j < i + size [ w ])
                   (words b)))
            hs);
      test "a hole's shift is not evaluated" (fun () ->
          let _, holes =
            template known [ W32 (Shift (Value (Later 0L), 64)) ]
          in
          equal int 1 (List.length holes));
      cases ~name:string_of_int "a known word's shift outside [0;63] is refused"
        [ -1; 64 ] (fun n ->
          raises_match (Exn.invalid_arg ~substring:"Rig_packet.template")
            (fun () -> template known [ W32 (Shift (Value (Known 1L), n)) ]));
    ]

(* C templates *)

external at : unit -> nativeint = "test_packet_template"
external c_fill : int64 -> int64 -> int64 -> string = "test_packet_fill"

let slot = at ()
let max_words = 16
let max_holes = 6

(* Packets within a C template's bounds, over its arguments: words drawn in
   turn, a word kept while the packet stays within 16 words and 6 holes, so that
   many reach the bounds. *)
let bounded =
  let fits (p, n, h) w =
    let n' = n + size [ w ] and h' = if is_term w then h + 1 else h in
    if n' > max_words || h' > max_holes then (p, n, h) else (w :: p, n', h')
  in
  let keep ws =
    let p, _, _ = List.fold_left fits ([], 0, 0) ws in
    List.rev p
  in
  Gen.map keep
    (Gen.list ~size:(Gen.int_range 0 24) (word ~dwords:4 (Gen.int_range 0 2)))

let args = Gen.triple u64 u64 u64

let pp_case ppf (p, (a, b, c)) =
  Format.fprintf ppf "%a@ with 0x%Lx, 0x%Lx, 0x%Lx"
    (pp_packet Format.pp_print_int)
    p a b c

let filled = Gen.with_pp pp_case (Gen.pair bounded args)

(* Whether an addition of [t] wraps past 2^64 with the arguments [value]. *)
let rec wraps value = function
  | Value _ -> false
  | Add (t, n) ->
      let x = eval value t in
      wraps value t || Int64.unsigned_compare (Int64.add x n) x < 0
  | Shift (t, _) | Or (t, _) -> wraps value t

let rec has f = function
  | Value _ -> false
  | (Add (t, _) | Shift (t, _) | Or (t, _)) as n -> f n || has f t

let any_term f p =
  List.exists (function Dword _ -> false | W32 t | W64 t -> f t) p

(* The index after the last word that holds a term. *)
let last_hole p =
  let _, last =
    List.fold_left
      (fun (i, last) w ->
        let i' = i + size [ w ] in
        (i', if is_term w then i' else last))
      (0, 0) p
  in
  last

let c_template =
  let refused name p =
    test name (fun () ->
        raises_match (Exn.invalid_arg ~substring:"Rig_packet.load") (fun () ->
            load slot p))
  in
  let value_at n = W32 (Value n) in
  group "C template"
    [
      prop "rig_fill on a loaded template is the encoding" filled
        (fun (p, (a, b, c)) ->
          let value i = [| a; b; c |].(i) in
          cover "16 words" (size p = max_words);
          cover "6 holes" (List.length (List.filter is_term p) = max_holes);
          cover "a hole ending at word 16" (last_hole p = max_words);
          cover "a 64-bit hole" (List.exists is_w64 p);
          cover "a shift by 0"
            (any_term (has (function Shift (_, 0) -> true | _ -> false)) p);
          cover "a shift by 63"
            (any_term (has (function Shift (_, 63) -> true | _ -> false)) p);
          cover "an or"
            (any_term (has (function Or _ -> true | _ -> false)) p);
          cover "an addition that wraps" (any_term (wraps value) p);
          load slot p;
          equal string (Rig_packet.encode value p) (c_fill a b c));
      test "a load replaces the template" (fun () ->
          load slot [ Dword 1; W32 (Value 0); W64 (Value 1) ];
          load slot [ W32 (Value 2) ];
          equal (list int) [ 7 ] (words (c_fill 5L 6L 7L)));
      refused "17 words are refused" (List.init 17 (fun i -> Dword i));
      refused "7 holes are refused" (List.init 7 (fun _ -> value_at 0));
      refused "a term of 4 operations is refused"
        [ W32 (Add (Add (Add (Add (Value 0, 1L), 1L), 1L), 1L)) ];
      cases ~name:string_of_int "an argument outside [0;2] is refused"
        [ -1; 3; max_int ] (fun i ->
          raises_match (Exn.invalid_arg ~substring:"Rig_packet.load") (fun () ->
              load slot [ value_at i ]));
      cases ~name:string_of_int "a shift outside [0;63] is refused" [ -1; 64 ]
        (fun n ->
          raises_match (Exn.invalid_arg ~substring:"Rig_packet.load") (fun () ->
              load slot [ W32 (Shift (Value 0, n)) ]));
    ]

let () = exit (run "rig_packet" [ encode; template; c_template ])
