(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The AMD rows: each prepares its operands on the GPU, checks its result once,
   and gives the run to time, with the floor kernel that moves its bytes in its
   directions where it is memory-bound. *)

module S = Nx_amd_support

let strf = Printf.sprintf

type work = Bytes of int | Flops of float
type prepared = { run : S.run; work : work; floor : S.run option }
type row = { name : string; make : S.gpu -> prepared }

(* [n] as a row names a size: 64K, 16M. *)
let size n =
  if n >= 1 lsl 20 then strf "%dM" (n lsr 20) else strf "%dK" (n lsr 10)

(* Floors *)

(* The copy floor of a row that reads [ins] bytes from each input and writes
   [out] bytes. *)
let copy_floor g ins out =
  let buffer n = S.buffer g (16 * ((n + 15) / 16)) in
  S.record (S.harness g)
    [ S.floor_copy ~ins:(List.map buffer ins) ~out:(buffer out) ]

let copy n =
  {
    name = strf "floor/copy-%sB" (size n);
    make =
      (fun g ->
        { run = copy_floor g [ n ] n; work = Bytes (2 * n); floor = None });
  }

(* The driver's copy, which a copy floor should not trail. *)
let copy_driver n =
  {
    name = strf "floor/copy-driver-%sB" (size n);
    make =
      (fun g ->
        let run = S.driver_copy ~src:(S.buffer g n) ~dst:(S.buffer g n) in
        { run; work = Bytes (2 * n); floor = None });
  }

let read n =
  {
    name = strf "floor/read-%sB" (size n);
    make =
      (fun g ->
        let run = S.record (S.harness g) [ S.floor_read g (S.buffer g n) ] in
        { run; work = Bytes n; floor = None });
  }

let launch =
  {
    name = "floor/launch";
    make =
      (fun g ->
        let run =
          S.record (S.harness g)
            [ S.launch "empty" ~groups:(1, 1, 1) ~threads:1 [] ]
        in
        { run; work = Bytes 0; floor = None });
  }

(* A peak row: 2 workgroups of 8 waves per compute unit, each wave [rounds]
   rounds of 16 instructions of [flops] each, per lane for an fma. *)
let peak name ~rounds ~flops =
  {
    name = "floor/" ^ name ^ "-peak";
    make =
      (fun g ->
        let blocks = 2 * S.cus g in
        let o = S.buffer g (8 * blocks * 8) in
        let kernel = String.map (fun c -> if c = '-' then '_' else c) name in
        let run =
          S.record (S.harness g)
            [
              S.launch kernel ~groups:(blocks, 1, 1) ~threads:S.threads
                [ A o; W rounds ];
            ]
        in
        let total = float (blocks * 8 * rounds * 16) *. flops in
        { run; work = Flops total; floor = None });
  }

let peaks =
  [
    peak "wmma-bf16" ~rounds:2000 ~flops:8192.;
    peak "fma-f32" ~rounds:40000 ~flops:64.;
  ]

(* Copies and reads in the R9700's 8 MB L2, its 64 MB Infinity Cache and DRAM; a
   copy's working set is twice its size. *)
let floors =
  let copies = [ 1 lsl 18; 1 lsl 21; 1 lsl 24; 1 lsl 28 ] in
  [ launch ]
  @ List.concat_map (fun n -> [ copy n; copy_driver n ]) copies
  @ List.map read [ 1 lsl 22; 1 lsl 25; 1 lsl 28 ]
  @ peaks

let all = floors
