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

(* [l] with each element's repeats dropped. *)
let distinct l =
  List.rev
    (List.fold_left
       (fun acc x -> if List.mem x acc then acc else x :: acc)
       [] l)

(* A GPU of each processor the table knows, and a profile of its counters, each
   listed once. *)
let profiles =
  Gen.with_pp
    (fun ppf (g, names) ->
      Format.fprintf ppf "%s: %s" (Gpu.processor g) (String.concat ", " names))
    (let open Gen in
     let* g = of_list [ gfx12; gfx11; gfx942 ] in
     let names = Array.of_list (Counter.names g) in
     let+ picks =
       list ~size:(int_range 0 16) (int_range 0 (Array.length names - 1))
     in
     (g, distinct (List.map (fun i -> names.(i)) picks)))

(* The counter registers of [block] on [g]: its [_LO] registers the GC's facts
   list, from [0]. *)
let slots g block =
  let rec go n =
    match Register.find g (Printf.sprintf "reg%s_PERFCOUNTER%d_LO" block n) with
    | Some _ -> go (n + 1)
    | None -> n
  in
  go 0

(* Whether [names] asks more counters of a block than its registers. *)
let overflows g names =
  let block name =
    match Counter.layout g [ name ] with
    | Ok { counters = [ c ]; _ } -> c.block
    | _ -> fail ("no block for " ^ name)
  in
  let blocks = List.map block names in
  List.exists
    (fun b -> List.length (List.filter (( = ) b) blocks) > slots g b)
    blocks

let laws =
  group ~timeout "layout"
    [
      prop "a profile is refused exactly when a block lacks registers" profiles
        (fun (g, names) ->
          let over = overflows g names in
          cover "fits" (not over);
          cover "overflows" over;
          match Counter.layout g names with
          | Ok l ->
              equal bool ~msg:"overflows" false over;
              List.iter
                (fun (c : Counter.t) ->
                  less int ~msg:c.name ~than:(slots g c.block) c.register)
                l.counters
          | Error e -> equal bool ~msg:e true over);
      prop "a profile listing a counter twice is refused, naming it"
        Gen.(pair profiles nat)
        (fun ((g, names), i) ->
          assume (names <> []);
          let twice = List.nth names (i mod List.length names) in
          equal layout
            (Error (twice ^ " is listed twice"))
            (Counter.layout g (names @ [ twice ])));
      prop "a run's samples are each counter's values, one after another"
        profiles (fun (g, names) ->
          match Counter.layout g names with
          | Error _ -> reject ()
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
      test "a fifth GL2C counter on an R9700 is refused, naming the block"
        (fun () ->
          equal layout (Error "gfx1201 counts at most 4 GL2C counters at once")
            (Counter.layout gfx12
               [
                 "GL2C_HIT";
                 "GL2C_MISS";
                 "GL2C_EA_RDREQ";
                 "GL2C_EA_WRREQ";
                 "GL2C_EA_WRREQ_STALL";
               ]));
      test "a processor the table lacks counts nothing" (fun () ->
          equal (list string) [] (Counter.names gfx90a);
          equal layout (Error "gfx90a counts no SQ_WAVES")
            (Counter.layout gfx90a [ "SQ_WAVES" ]));
      test "a GPU's names are in increasing order" (fun () ->
          let names = Counter.names gfx12 in
          equal (list string) (List.sort String.compare names) names);
    ]

let () = exit (run "rig_amd_abi.counter" [ laws; cases ])
