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

(* Space *)

let test_space () =
  let s = Space.create ~base:0x10000 (1 lsl 20) in
  let a = Option.get (Space.alloc s 0x3000) in
  equal ~msg:"aligned to the size's top bit" int 0 (a land 0x1fff);
  let b = Option.get (Space.alloc s 0x1000) in
  equal ~msg:"disjoint" bool true (b + 0x1000 <= a || a + 0x3000 <= b);
  Space.free s a;
  Space.free s b;
  equal ~msg:"merged: half the space allocates again" bool true
    (Option.is_some (Space.alloc s (1 lsl 18)));
  raises_match (Exn.invalid_arg ~substring:"no range") (fun () ->
      Space.free s 0x10001)

(* Page tables over a GPU memory of entries kept in a table. Entries: the
   address, bit 0 valid, bit 1 a table. *)

let format () =
  let entries = Hashtbl.create 64 in
  let key table i = table + (8 * i) in
  {
    Page_table.levels = [ 12; 21; 30; 39 ];
    bits = 48;
    first = 0;
    get =
      (fun ~level:_ ~table i ->
        Option.value ~default:0L (Hashtbl.find_opt entries (key table i)));
    set = (fun ~level:_ ~table i e -> Hashtbl.replace entries (key table i) e);
    encode =
      (fun ~level:_ ~table _ ~uncached:_ ~snooped:_ ~fragment:_ ~valid pa ->
        Int64.of_int (pa lor (if valid then 1 else 0) lor if table then 2 else 0));
    valid = (fun e -> Int64.logand e 1L = 1L);
    leaf = (fun ~level e -> level = 3 || Int64.logand e 2L = 0L);
    address = (fun e -> Int64.to_int e land lnot 0xfff);
    large = (fun ~level -> level >= 2);
    zero = (fun _ _ -> ());
    flush = (fun () -> ());
  }

let test_page_table () =
  let space = Space.create ~base:(1 lsl 40) (1 lsl 30) in
  let t =
    Page_table.create (format ()) space ~memory:(64 lsl 20) ~boot:(1 lsl 20)
      ~tables:Pool
      ~pages:[ (2 lsl 20, 2 lsl 20); (0x1000, 0x1000) ]
  in
  Page_table.booted t;
  let m = Option.get (Page_table.alloc t (4 lsl 20)) in
  equal ~msg:"two large pages" int 2 (List.length m.pages);
  Page_table.free t m;
  equal ~msg:"all of the main pool again" bool true
    (Option.is_some (Page_table.alloc t (16 lsl 20)));
  equal ~msg:"exhaustion is a value" bool true
    (Option.is_none (Page_table.alloc t (128 lsl 20)))

let () =
  exit
  @@ run "device_pci"
       [
         group "window"
           [
             test "mapped" test_mapped;
             test "through a transport" test_transport;
           ];
         test "space" test_space;
         test "page tables" test_page_table;
       ]
