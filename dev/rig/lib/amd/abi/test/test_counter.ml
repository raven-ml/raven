(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Counters of rocprofiler's counter_defs.yaml (gen/headers/counter_defs.yaml)
   and the layout of a run's samples, as counter.mli states it. *)

open Windtrap
open Rig_amd_abi
module S = Rig_amd_abi_support

let timeout = S.timeout

(* An R9700's GC: one die of 4 engines and 64 compute units. *)
let gfx12 =
  S.gpu ~target:(12, 0, 1) ~shader_engines:4 ~compute_units:64 (12, 0, 1)

let gfx11 = S.gpu ~shader_engines:6 ~compute_units:48 (11, 0, 0)

let gfx942 =
  S.gpu ~target:(9, 4, 2) ~sdma:(4, 4, 2) ~xccs:8 ~compute_units:38 (9, 4, 3)

let gfx90a = S.gpu ~target:(9, 0, 10) (9, 4, 3)

let counter =
  Testable.make
    ~pp:(fun ppf (c : Counter.t) ->
      Format.fprintf ppf "{%s %s/%d register %d; %d x %d x %d x %d at %d}"
        c.name c.block c.event c.register c.instances c.engines c.arrays c.wgps
        c.offset)
    ~equal:( = )

let layout =
  Testable.make
    ~pp:(fun ppf -> function
      | Ok (l : Counter.layout) ->
          Format.fprintf ppf "Ok {%a; %d bytes}"
            (Format.pp_print_list (Testable.pp counter))
            l.counters l.bytes
      | Error e -> Format.fprintf ppf "Error %S" e)
    ~equal:( = )

let c name block event register (instances, engines, arrays, wgps) offset =
  {
    Counter.name;
    block;
    event;
    register;
    instances;
    engines;
    arrays;
    wgps;
    offset;
  }

(* A GPU of each processor the table knows, and a profile of its counters. *)
let profiles =
  Gen.with_pp
    (fun ppf (g, names) ->
      Format.fprintf ppf "%s: %s" (Gpu.processor g) (String.concat ", " names))
    (let open Gen in
     let* g = of_list [ gfx12; gfx11; gfx942 ] in
     let names = Array.of_list (Counter.names g) in
     let+ picks =
       list ~size:(int_range 0 12) (int_range 0 (Array.length names - 1))
     in
     (g, List.map (fun i -> names.(i)) picks))

let laws =
  group ~timeout "layout"
    [
      prop "a run's samples are each counter's values, one after another"
        profiles (fun (g, names) ->
          match Counter.layout g names with
          | Error e -> fail e
          | Ok l ->
              let size (c : Counter.t) =
                8 * g.xccs * c.instances * c.engines * c.arrays * c.wgps
              in
              let rec check offset seen = function
                | [] -> equal int ~msg:"bytes" offset l.bytes
                | (c : Counter.t) :: rest ->
                    equal int ~msg:"offset" offset c.offset;
                    equal int ~msg:"register"
                      (List.length (List.filter (( = ) c.block) seen))
                      c.register;
                    check (offset + size c) (c.block :: seen) rest
              in
              equal (list string) ~msg:"names" names
                (List.map (fun (c : Counter.t) -> c.name) l.counters);
              check 0 [] l.counters);
    ]

let cases =
  group ~timeout "cases"
    [
      test "an R9700 counts its engines' work-group processors" (fun () ->
          equal layout
            (Ok
               {
                 counters =
                   [
                     c "GRBM_GUI_ACTIVE" "GRBM" 2 0 (1, 1, 1, 1) 0;
                     c "SQ_WAVES" "SQ" 4 0 (1, 4, 2, 4) 8;
                     c "GL2C_HIT" "GL2C" 41 0 (32, 1, 1, 1) 264;
                     c "SQ_BUSY_CYCLES" "SQ" 3 1 (1, 4, 2, 4) 520;
                   ];
                 bytes = 776;
               })
            (Counter.layout gfx12
               [ "GRBM_GUI_ACTIVE"; "SQ_WAVES"; "GL2C_HIT"; "SQ_BUSY_CYCLES" ]));
      test "a gfx942 counts its SQ per engine, on each of its dies" (fun () ->
          equal layout
            (Ok
               {
                 counters =
                   [
                     c "SQ_WAVES" "SQ" 4 0 (1, 4, 1, 1) 0;
                     c "TCC_HIT" "TCC" 17 0 (16, 1, 1, 1) 256;
                   ];
                 bytes = 256 + (8 * 8 * 16);
               })
            (Counter.layout gfx942 [ "SQ_WAVES"; "TCC_HIT" ]));
      test "a counter the GPU lacks is refused, naming it" (fun () ->
          equal layout (Error "gfx1201 counts no SQ_FOO")
            (Counter.layout gfx12 [ "SQ_WAVES"; "SQ_FOO" ]));
      test "a processor the table lacks counts nothing" (fun () ->
          equal (list string) [] (Counter.names gfx90a);
          equal layout (Error "gfx90a counts no SQ_WAVES")
            (Counter.layout gfx90a [ "SQ_WAVES" ]));
      test "a GPU's names are in increasing order" (fun () ->
          let names = Counter.names gfx12 in
          equal (list string) (List.sort String.compare names) names);
    ]

let () = exit (run "rig_amd_abi.counter" [ laws; cases ])
