(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Work entries, against the segments of rdma-core's mlx5dv.h: a control segment
   (opcode, index, queue pair and 16-byte units, then the completion flag at
   byte 11), a remote address segment at byte 16, then a data segment or an
   inline segment at byte 32, every field big-endian. A data segment's byte
   count of 0 means 2{^31} bytes to the NIC, so a copy of no bytes carries no
   data segment, as rdma-core's qp.c skips a zero-length one. *)

open Windtrap
open Rig_mlx5_abi

let strf = Printf.sprintf

(* Reading the spec's fields *)

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

let hex b at n =
  String.concat " "
    (List.init (n / 4) (fun w ->
         String.concat ""
           (List.init 4 (fun i -> strf "%02x" (byte b (at + (4 * w) + i))))))

(* The values of the spec *)

let rdma_write = 0x08
let rdma_read = 0x10
let cq_update = 0x08
let inline_seg = 0x8000_0000

(* Generators *)

let key = Gen.int_range 0 0xffff_ffff

let address =
  Gen.frequency [ (3, Gen.int_range 0 (1 lsl 48)); (1, Gen.constant max_int) ]

let length =
  Gen.frequency
    [
      (3, Gen.int_range 1 (1 lsl 20));
      (1, Gen.constant 0);
      (1, Gen.constant 0x7fff_ffff);
    ]

let local =
  Gen.(
    let+ address = address and+ bytes = length and+ key = key in
    { Entry.address; bytes; key })

let remote =
  Gen.(
    let+ address = address and+ key = key in
    { Entry.address; key })

let op =
  Gen.(
    frequency
      [
        (2, map (fun (src, dst) -> Entry.Write { src; dst }) (pair local remote));
        ( 2,
          map
            (fun (data, dst) -> Entry.Write_inline { data; dst })
            (pair (string_of ~size:(int_range 0 Entry.max_inline) char) remote)
        );
        (2, map (fun (src, dst) -> Entry.Read { src; dst }) (pair remote local));
      ])

let entry =
  Gen.(
    let+ op = op and+ signal = bool in
    { Entry.op; signal })

let placed =
  Gen.(
    let+ e = entry
    and+ qp = frequency [ (3, int_range 0 0xff_ffff); (1, constant 0xff_ffff) ]
    and+ index = frequency [ (3, int_range 0 0x1_ffff); (1, constant 0xffff) ]
    and+ slot = int_range 0 3 in
    (e, qp, index, slot))

let pp_entry ppf ((e : Entry.t), qp, index, slot) =
  let op =
    match e.op with
    | Write { src; dst } ->
        strf "write %d bytes 0x%x/%x -> 0x%x/%x" src.bytes src.address src.key
          dst.address dst.key
    | Write_inline { data; dst } ->
        strf "write inline %S -> 0x%x/%x" data dst.address dst.key
    | Read { src; dst } ->
        strf "read %d bytes 0x%x/%x -> 0x%x/%x" dst.bytes src.address src.key
          dst.address dst.key
  in
  Format.fprintf ppf "%s, signal %b, qp 0x%x, index %d, slot %d" op e.signal qp
    index slot

let placed = Gen.with_pp pp_entry placed

(* The entry's fields as the spec reads them at [at]. *)
let check_entry b at ((e : Entry.t), qp, index) =
  let opcode, units =
    match e.op with
    | Write { src = l; _ } | Read { dst = l; _ } ->
        ( (match e.op with Read _ -> rdma_read | _ -> rdma_write),
          if l.bytes = 0 then 2 else 3 )
    | Write_inline { data; _ } ->
        (rdma_write, (36 + String.length data + 15) / 16)
  in
  equal ~msg:"opmod, index, opcode" int
    (((index land 0xffff) lsl 8) lor opcode)
    (be b at 4);
  equal ~msg:"queue pair, units" int ((qp lsl 8) lor units) (be b (at + 4) 4);
  equal ~msg:"signature and stream" int 0 (be b (at + 8) 3);
  equal ~msg:"flags" int (if e.signal then cq_update else 0) (byte b (at + 11));
  equal ~msg:"immediate" int 0 (be b (at + 12) 4);
  let remote (r : Entry.remote) =
    equal ~msg:"remote address" int r.address (be b (at + 16) 8);
    equal ~msg:"remote key" int r.key (be b (at + 24) 4);
    equal ~msg:"remote reserved" int 0 (be b (at + 28) 4)
  in
  let local (l : Entry.local) =
    equal ~msg:"byte count" int l.bytes (be b (at + 32) 4);
    equal ~msg:"local key" int l.key (be b (at + 36) 4);
    equal ~msg:"local address" int l.address (be b (at + 40) 8)
  in
  let rest from =
    for i = from to 63 do
      equal ~msg:(strf "byte %d" i) int 0 (byte b (at + i))
    done
  in
  let data (l : Entry.local) =
    if l.bytes = 0 then rest 32
    else (
      local l;
      rest 48)
  in
  match e.op with
  | Write { src; dst } ->
      remote dst;
      data src
  | Read { src; dst } ->
      remote src;
      data dst
  | Write_inline { data; dst } ->
      let n = String.length data in
      remote dst;
      equal ~msg:"inline byte count" int (n lor inline_seg) (be b (at + 32) 4);
      equal ~msg:"inline data" string data
        (String.init n (fun i -> Bigarray.Array1.get b (at + 36 + i)));
      rest (36 + n)

let ring = 4 * Entry.size

let layout =
  group "layout"
    [
      test "an entry is one basic block of 64 bytes" (fun () ->
          equal int 64 Entry.size);
      test "an inline write carries up to 28 bytes" (fun () ->
          equal int 28 Entry.max_inline);
      prop "an entry's fields are where mlx5dv.h lays them out" placed
        (fun (e, qp, index, slot) ->
          cover "a read" (match e.op with Read _ -> true | _ -> false);
          cover "a copy of no bytes"
            (match e.op with
            | Write { src = l; _ } | Read { dst = l; _ } -> l.bytes = 0
            | _ -> false);
          cover "a full inline write"
            (match e.op with
            | Write_inline { data; _ } -> String.length data = Entry.max_inline
            | _ -> false);
          cover "an index past 16 bits" (index > 0xffff);
          let b = buffer ring in
          Entry.write b (slot * Entry.size) ~qp ~index e;
          check_entry b (slot * Entry.size) (e, qp, index));
      prop "an entry writes its own block and no other byte" placed
        (fun (e, qp, index, slot) ->
          let b = buffer ring in
          Entry.write b (slot * Entry.size) ~qp ~index e;
          for i = 0 to ring - 1 do
            if i / Entry.size <> slot then
              equal ~msg:(strf "byte %d" i) int 0xa5 (byte b i)
          done);
    ]

(* Entries a reader can compare with the spec by eye: the three of a transfer,
   as a sending machine posts them. *)
let transfer =
  test "a transfer's three entries" (fun () ->
      let b = buffer ring in
      let qp = 0x1c3 in
      Entry.write b 0 ~qp ~index:0x12345
        {
          op =
            Write
              {
                src =
                  {
                    address = 0x7f00_1000_0000;
                    bytes = 0x10_0000;
                    key = 0x1234;
                  };
                dst = { address = 0x2_0000_0000; key = 0xabcd };
              };
          signal = false;
        };
      Entry.write b 64 ~qp ~index:0x12346
        {
          op =
            Write_inline
              {
                data = "\x07\x00\x00\x00\x00\x00\x00\x00";
                dst = { address = 0x2_0010_0000; key = 0xabce };
              };
          signal = false;
        };
      Entry.write b 128 ~qp ~index:0x12347
        {
          op =
            Read
              {
                src = { address = 0x2_0010_0000; key = 0xabce };
                dst = { address = 0x7f00_2000_0000; bytes = 8; key = 0x1235 };
              };
          signal = true;
        };
      print_string
        (String.concat "\n" (List.init 12 (fun i -> hex b (16 * i) 16)));
      expect (output ())
      @@ __POS_OF__
           {|
        00234508 0001c303 00000000 00000000
        00000002 00000000 0000abcd 00000000
        00100000 00001234 00007f00 10000000
        00000000 00000000 00000000 00000000
        00234608 0001c303 00000000 00000000
        00000002 00100000 0000abce 00000000
        80000008 07000000 00000000 00000000
        00000000 00000000 00000000 00000000
        00234710 0001c303 00000008 00000000
        00000002 00100000 0000abce 00000000
        00000008 00001235 00007f00 20000000
        00000000 00000000 00000000 00000000
      |})

let refusals =
  let e =
    {
      Entry.op =
        Write
          {
            src = { address = 0; bytes = 8; key = 1 };
            dst = { address = 0; key = 2 };
          };
      signal = false;
    }
  in
  let refused name ?(at = 0) ?(qp = 1) ?(index = 0) e =
    test name (fun () ->
        let b = buffer ring in
        raises_match (Exn.invalid_arg ~substring:"Entry.write") (fun () ->
            Entry.write b at ~qp ~index e);
        for i = 0 to ring - 1 do
          equal ~msg:(strf "byte %d is untouched" i) int 0xa5 (byte b i)
        done)
  in
  let with_op op = { e with op } in
  group "refusals"
    [
      refused "a block past the buffer's end" ~at:ring e;
      refused "a block that starts mid-block" ~at:16 e;
      refused "a negative byte" ~at:(-64) e;
      refused "a queue pair past 24 bits" ~qp:0x100_0000 e;
      refused "a negative queue pair" ~qp:(-1) e;
      refused "a negative index" ~index:(-1) e;
      refused "29 inline bytes"
        (with_op
           (Write_inline
              { data = String.make 29 'x'; dst = { address = 0; key = 0 } }));
      refused "a key past 32 bits"
        (with_op
           (Write
              {
                src = { address = 0; bytes = 8; key = 0x1_0000_0000 };
                dst = { address = 0; key = 0 };
              }));
      refused "a length past 31 bits"
        (with_op
           (Read
              {
                src = { address = 0; key = 0 };
                dst = { address = 0; bytes = 0x8000_0000; key = 0 };
              }));
      refused "a negative remote address"
        (with_op
           (Write
              {
                src = { address = 0; bytes = 8; key = 0 };
                dst = { address = -1; key = 0 };
              }));
    ]

let () = exit (run "rig_mlx5_abi.entry" [ layout; transfer; refusals ])
