(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Scratch of two GPUs: 48 compute units of 32 slots and 6 engines on GFX11, at
   least 128 bytes a lane in 256-byte wave units; 38 of 32, 4 engines and 8 dies
   on GFX9, in 1024-byte units. The descriptor's words follow ROCr's registers.h
   layouts. *)

open Windtrap
open Device_amd_abi
module S = Device_amd_abi_support

let timeout = S.timeout
let gfx11 = S.gpu ~shader_engines:6 ~compute_units:48 (11, 0, 0)

let gfx9 =
  S.gpu ~target:(9, 4, 2) ~sdma:(4, 4, 2) ~xccs:8 ~compute_units:38 (9, 4, 3)

let words = S.words

let pp_gpu ppf (g : Gpu.t) =
  Format.fprintf ppf "GC %s, %d dies of %d engines, %d CUs of %d slots"
    (S.version g.gc) g.xccs g.shader_engines g.compute_units g.scratch_slots

(* GPUs of every generation the tables define, of any counts. *)
let gpus =
  Gen.with_pp pp_gpu
    (let open Gen in
     let+ gc = of_list S.families
     and+ xccs = int_range 1 8
     and+ shader_engines = int_range 1 8
     and+ compute_units = int_range 1 96
     and+ scratch_slots = int_range 1 32 in
     S.gpu ~xccs ~shader_engines ~compute_units ~scratch_slots gc)

(* Bytes per lane around the 128-byte floor and the granules. *)
let lanes =
  Gen.frequency
    [
      (2, Gen.int_range 0 300);
      ( 1,
        Gen.of_list ~pp:Format.pp_print_int
          [ 0; 1; 127; 128; 129; 131; 132; 143; 144; 145 ] );
      (1, Gen.int_range 0 (1 lsl 16));
    ]

(* A GPU, a base and a size: below the bound, around 2^32 bytes a die, or
   negative. *)
let descriptors =
  let open Gen in
  with_pp
    (fun ppf (g, base, n) ->
      Format.fprintf ppf "%a, base 0x%x, %d bytes" pp_gpu g base n)
    (let* g = gpus in
     let around =
       let+ d =
         frequency
           [ (1, constant (-1)); (1, constant 0); (1, int_range (-2) 2) ]
       and+ r = int_range 0 (g.xccs - 1) in
       (((1 lsl 32) + d) * g.xccs) + r
     in
     let+ base = int_range 0 ((1 lsl 48) - 1)
     and+ n =
       frequency
         [
           (3, int_range 0 ((1 lsl 32) - 1));
           (2, around);
           (1, of_list ~pp:Format.pp_print_int [ min_int; -1 ]);
         ]
     in
     (g, base, n))

let granule (g : Gpu.t) = match g.gc with 9, _, _ -> 16 | _ -> 4
let round_up n m = (n + m - 1) / m * m

let laws =
  group ~timeout "laws"
    [
      prop "a buffer holds a wave's lanes for every slot, unit and die"
        (Gen.pair gpus lanes) (fun ((g : Gpu.t), n) ->
          let share = round_up (Int.max n 128) (granule g) in
          cover "below the floor" (n < 128);
          cover "off the granule" (n > 128 && n mod granule g <> 0);
          equal int
            (share * 64 * g.scratch_slots * g.compute_units * g.xccs)
            (Scratch.size g n));
      prop "a descriptor holds the base and n / xccs bytes, below 2^32 a die"
        descriptors (fun ((g : Gpu.t), base, n) ->
          let share = 1 lsl 32 in
          cover "a share rounded down" (n mod g.xccs <> 0);
          cover "the largest share" (n >= 0 && n / g.xccs = share - 1);
          cover "a share of 2^32" (n / g.xccs = share);
          cover "a negative size" (n < 0);
          if n < 0 || n / g.xccs >= share then
            raises_match (Exn.invalid_arg ~substring:"Scratch.descriptor")
              (fun () -> Scratch.descriptor g ~base n)
          else
            match words (Scratch.descriptor g ~base n) with
            | [ w0; w1; w2; _ ] ->
                equal (triple int int int)
                  (base land 0xffff_ffff, base lsr 32, n / g.xccs)
                  (w0, w1 land 0xffff, w2)
            | ws -> failf "%d words" (List.length ws));
      prop
        "a scratch ring holds a wave's scratch in units and the waves it serves"
        (Gen.pair gpus lanes) (fun ((g : Gpu.t), n) ->
          let r = Option.get (Register.find g "regCOMPUTE_TMPRING_SIZE") in
          let field f w =
            let lo, hi = List.assoc f r.fields in
            (w lsr lo) land ((1 lsl (hi - lo + 1)) - 1)
          in
          let w = Scratch.tmpring g n in
          let unit = match g.gc with 9, _, _ -> 1024 | _ -> 256 in
          let wave =
            Scratch.size g n / g.xccs / g.compute_units / g.scratch_slots
          in
          let size = (wave + unit - 1) / unit in
          let engines =
            match g.gc with 9, _, _ -> 1 | _ -> g.shader_engines
          in
          let waves =
            Int.min
              (Scratch.size g n / g.xccs / (size * unit) / engines)
              (g.compute_units * g.scratch_slots)
          in
          equal (pair int int) (size, waves)
            (field "wavesize" w, field "waves" w));
    ]

let sizes =
  group ~timeout "sizes"
    [
      test "a GC with no scratch ring register is refused" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"Scratch.tmpring") (fun () ->
              Scratch.tmpring { gfx11 with gc = (10, 3, 0) } 0));
    ]

(* Words 3's selects of x, y, z and w, and its added thread id. *)
let selects = 4 lor (5 lsl 3) lor (6 lsl 6) lor (7 lsl 9) lor (1 lsl 23)

let descriptor =
  group ~timeout "descriptor"
    [
      test "a GFX11 descriptor" (fun () ->
          equal (list int)
            [
              0x3456_7800;
              0x12 lor (1 lsl 30);
              0x10_0000;
              selects lor (0x14 lsl 12) lor (2 lsl 28);
            ]
            (words (Scratch.descriptor gfx11 ~base:0x12_3456_7800 0x10_0000)));
      test "a GFX9 descriptor splits the buffer among the dies" (fun () ->
          equal (list int)
            [
              0x3456_7800;
              0x12 lor (1 lsl 31);
              0x2_0000;
              selects lor (4 lsl 12) lor (4 lsl 15) lor (1 lsl 19) lor (3 lsl 21);
            ]
            (words (Scratch.descriptor gfx9 ~base:0x12_3456_7800 0x10_0000)));
      test "a GC with no layout is refused" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"Scratch.descriptor")
            (fun () ->
              Scratch.descriptor { gfx11 with gc = (10, 3, 0) } ~base:0 0));
    ]

let () = exit (run "device_amd_abi.scratch" [ sizes; laws; descriptor ])
