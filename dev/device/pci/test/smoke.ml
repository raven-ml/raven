(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The library on process memory and a transport over a buffer. The tester owns
   the suite. *)

open Windtrap
open Device_pci

external buffer : int -> int = "smoke_buffer"
external far : unit -> int = "smoke_transport"
external break : unit -> unit = "smoke_break"

(* Every access stores what it reads back, through either kind of window. *)
let accesses w =
  Window.set32 w 8 0xdead_beef;
  equal ~msg:"32 bits" int 0xdead_beef (Window.get32 w 8);
  Window.set64 w 16 0x0102_0304_0506_0708L;
  equal ~msg:"64 bits" int64 0x0102_0304_0506_0708L (Window.get64 w 16);
  equal ~msg:"little-endian" int 0x08 (Window.get8 w 16);
  Window.write w 33 "unaligned";
  equal ~msg:"bulk" string "unaligned" (Window.read w 33 9);
  Window.fill w 64 5 'x';
  equal ~msg:"fill" string "xxxxx" (Window.read w 64 5);
  let s = Window.sub w 8 8 in
  equal ~msg:"sub" int 0xdead_beef (Window.get32 s 0);
  raises_match (Exn.invalid_arg ~substring:"outside") (fun () ->
      ignore (Window.get32 s 8));
  raises_match (Exn.invalid_arg ~substring:"aligned") (fun () ->
      ignore (Window.get32 w 2))

let test_mapped () =
  let w = Window.v (buffer 4096) 4096 in
  equal ~msg:"mapped" bool true (Window.mapped w);
  accesses w;
  equal ~msg:"bigarray" char 'x' (Window.bigarray w).{64}

let test_transport () =
  let tr = Window.transport (far ()) in
  let w = Window.through tr 0 4096 in
  equal ~msg:"not mapped" bool false (Window.mapped w);
  accesses w;
  break ();
  raises ~msg:"a failed transport" (Failure "far: the link broke") (fun () ->
      ignore (Window.get32 w 0))

let () =
  exit
  @@ run "device_pci"
       [
         group "window"
           [
             test "mapped" test_mapped;
             test "through a transport" test_transport;
           ];
       ]
