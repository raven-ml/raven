(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Packets as their interpreters read them: sizes in words, little-endian bytes,
   terms over 64 bits, and the holes of a template. *)

open Windtrap
open Device_amd_abi

let id (v : int64) = v

(* The 32-bit words of [s], each little-endian. *)
let words s =
  List.init
    (String.length s / 4)
    (fun i -> Int32.to_int (String.get_int32_le s (4 * i)) land 0xffff_ffff)

let encode =
  group "encode"
    [
      test "a packet's size counts a 64-bit word twice" (fun () ->
          equal int 4 (Packet.size [ Dword 1; W32 (Value 2L); W64 (Value 3L) ]));
      test "a constant word is the integer's low 32 bits" (fun () ->
          equal (list int) [ 0xffff_ffff ]
            (words (Packet.encode id [ Dword 0x1_ffff_ffff ])));
      test "a 64-bit word is its low word, then its high word" (fun () ->
          equal (list int) [ 2; 1 ]
            (words (Packet.encode id [ W64 (Value 0x1_0000_0002L) ])));
      test "a term adds, then shifts, over 64 bits" (fun () ->
          equal (list int) [ 0x3000_0000; 0 ]
            (words
               (Packet.encode id
                  [ W64 (Shift (Add (Value 0x2ff_ffff_ff00L, 0x100L), 12)) ])));
      test "an or keeps bit 63" (fun () ->
          equal (list int) [ 0x100; 0x8000_0000 ]
            (words
               (Packet.encode id [ W64 (Or (Value 0x100L, Int64.min_int)) ])));
      cases ~name:string_of_int "a shift outside [0;63] is refused" [ -1; 64 ]
        (fun n ->
          raises_match (Exn.invalid_arg ~substring:"Packet.encode") (fun () ->
              Packet.encode id [ W32 (Shift (Value 1L, n)) ]));
    ]

type hole = Known of int64 | Later

let known = function Known n -> Some n | Later -> None

let template =
  group "template"
    [
      test "a hole is a zero word at its index" (fun () ->
          let b, holes =
            Packet.template known
              [ Dword 7; W64 (Value (Known 9L)); W32 (Add (Value Later, 4L)) ]
          in
          equal (list int) [ 7; 9; 0; 0 ] (words b);
          equal (list int) [ 3 ] (List.map fst holes));
      test "a template of known values is the encoding" (fun () ->
          let p : int64 Packet.t =
            [ Dword 1; W64 (Shift (Value 0x1234_5678_9000L, 8)) ]
          in
          equal (pair string int)
            (Packet.encode id p, 0)
            (let b, holes = Packet.template (fun v -> Some v) p in
             (b, List.length holes)));
      test "a hole's shift is not evaluated" (fun () ->
          let _, holes =
            Packet.template known [ W32 (Shift (Value Later, 64)) ]
          in
          equal int 1 (List.length holes));
    ]

let () = exit (run "device_amd_abi.packet" [ encode; template ])
