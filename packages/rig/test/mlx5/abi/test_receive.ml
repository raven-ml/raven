(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Receive entries, against rdma-core's mlx5dv.h and qp.c: one data segment of
   16 bytes (byte count, local key, address, big-endian); a receive of no bytes
   is the segment that ends a scatter list, of byte count 0 and the invalid key
   0x100, as mlx5_post_recv skips a zero-length segment and ends the list. *)

open Windtrap
open Rig_mlx5_abi

let strf = Printf.sprintf
let ring = 4 * Receive.size
let invalid_key = 0x100

let buffer n =
  let b = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n in
  Bigarray.Array1.fill b '\xa5';
  b

let byte b at = Char.code (Bigarray.Array1.get b at)

let be b at n =
  let v = ref 0 in
  for i = 0 to n - 1 do
    v := (!v lsl 8) lor byte b (at + i)
  done;
  !v

let local =
  Gen.(
    let+ address =
      frequency [ (3, int_range 0 (1 lsl 48)); (1, constant max_int) ]
    and+ bytes =
      frequency
        [
          (3, int_range 1 (1 lsl 20));
          (1, constant 0);
          (1, constant 0x7fff_ffff);
        ]
    and+ key = int_range 0 0xffff_ffff in
    { Entry.address; bytes; key })

let placed =
  Gen.with_pp
    (fun ppf ((l : Entry.local), slot) ->
      Format.fprintf ppf "%d bytes 0x%x/%x, slot %d" l.bytes l.address l.key
        slot)
    Gen.(pair local (int_range 0 3))

let layout =
  group "layout"
    [
      test "a receive is one data segment of 16 bytes" (fun () ->
          equal int 16 Receive.size);
      prop "a receive's fields are where mlx5dv.h lays them out" placed
        (fun ((l : Entry.local), slot) ->
          cover "a receive of no bytes" (l.bytes = 0);
          let b = buffer ring in
          let at = slot * Receive.size in
          Receive.write b at l;
          let count, key, address =
            if l.bytes = 0 then (0, invalid_key, 0)
            else (l.bytes, l.key, l.address)
          in
          equal ~msg:"byte count" int count (be b at 4);
          equal ~msg:"key" int key (be b (at + 4) 4);
          equal ~msg:"address" int address (be b (at + 8) 8));
      prop "a receive writes its own segment and no other byte" placed
        (fun (l, slot) ->
          let b = buffer ring in
          Receive.write b (slot * Receive.size) l;
          for i = 0 to ring - 1 do
            if i / Receive.size <> slot then
              equal ~msg:(strf "byte %d" i) int 0xa5 (byte b i)
          done);
    ]

let refusals =
  let l = { Entry.address = 0x1000; bytes = 8; key = 1 } in
  let refused name ?(at = 0) l =
    test name (fun () ->
        let b = buffer ring in
        raises_match (Exn.invalid_arg ~sub:"Receive.write") (fun () ->
            Receive.write b at l);
        for i = 0 to ring - 1 do
          equal ~msg:(strf "byte %d is untouched" i) int 0xa5 (byte b i)
        done)
  in
  group "refusals"
    [
      refused "a segment past the buffer's end" ~at:ring l;
      refused "a segment that starts mid-segment" ~at:8 l;
      refused "a negative byte" ~at:(-16) l;
      refused "a length past 31 bits" { l with bytes = 0x8000_0000 };
      refused "a negative length" { l with bytes = -1 };
      refused "a key past 32 bits" { l with key = 0x1_0000_0000 };
      refused "a negative address" { l with address = -1 };
    ]

let () = exit (run "rig_mlx5_abi.receive" [ layout; refusals ])
