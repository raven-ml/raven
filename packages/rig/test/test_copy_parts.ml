(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Copy parts as rig_edge.h states them, on the drivers of rig's own suites:
   the memory device's, whose memory is host memory, and Polled's, whose
   Device memory the host does not address. Rig's interface makes no box yet,
   so each part goes to the driver's C functions directly
   (Rig_gpu_support.hand). *)

open Windtrap
module G = Rig_gpu_support
module H = Rig_gpu_support.Host
module P = Rig_support.Polled
module E = Rig_gpu_support.Edge (P)

let timeout = Rig_support.time 60.

let pattern n seed =
  String.init n (fun i -> Char.chr ((seed + (i * 7)) land 255))

(* The memory device's C state over the word at [word], as rig.memory_device
   makes it. These bind rig's private stubs, whose types the compiler cannot
   check here, because nothing public hands the memory device a box or a
   fill: the host copies and sets its memory. They go once a public call,
   such as a copy of boxes in rig.mli, hands the memory device's copy engine
   a box or a fill. *)
external memory_device : nativeint -> nativeint = "caml_rig_memory_new"
external drop_memory_device : nativeint -> unit = "caml_rig_memory_drop"

(* A box of [rows] rows of [width] bytes, each side at its offset and with
   its pitch [width] plus its pad, its memory [slack] bytes past its last
   row; or, with [fill], a fill of the destination's one row. *)
type case = {
  width : int;
  rows : int;
  src : int * int;  (** The source's offset and pad. *)
  dst : int * int;
  slack : int;
  fill : bool;
}

let span ~width ~rows ~pitch =
  if rows = 0 then 0 else ((rows - 1) * pitch) + width

let cases =
  let open Gen in
  let width =
    frequency [ (1, constant 0); (3, int_range 1 40); (1, int_range 41 300) ]
  in
  let rows =
    frequency [ (1, constant 0); (1, constant 1); (3, int_range 2 5) ]
  in
  let pad = frequency [ (1, constant 0); (1, int_range 1 9) ] in
  let side = pair (int_range 0 9) pad in
  with_pp
    (fun ppf c ->
      Format.fprintf ppf
        "%s%d rows of %d bytes, from %d pitch %d into %d pitch %d, slack %d"
        (if c.fill then "a fill: " else "")
        c.rows c.width (fst c.src) (c.width + snd c.src) (fst c.dst)
        (c.width + snd c.dst) c.slack)
    (let+ width = width
     and+ rows = rows
     and+ src = side
     and+ dst = side
     and+ slack = int_range 0 3
     and+ fill = frequency [ (2, constant false); (1, constant true) ] in
     (* A fill has one row. *)
     { width; rows = (if fill then 1 else rows); src; dst; slack; fill })

let covers c =
  let ps = snd c.src and pd = snd c.dst in
  cover "a fill" c.fill;
  cover "a fill of no bytes" (c.fill && c.width = 0);
  cover "a box of no rows" ((not c.fill) && c.rows = 0);
  cover "a box of no width" ((not c.fill) && c.rows > 0 && c.width = 0);
  cover "rows end to end" ((not c.fill) && c.rows > 1 && ps = 0 && pd = 0);
  cover "rows apart" ((not c.fill) && c.rows > 1 && (ps > 0 || pd > 0))

(* The part of [c], with the sides' handles, and what its destination holds
   after it, from what each side held before. *)
let part c ~src ~dst : G.raw =
  let so, ps = c.src and d_o, pd = c.dst in
  {
    dst;
    dst_offset = d_o;
    src = (if c.fill then 0n else src);
    src_offset = (if c.fill then 0 else so);
    width = c.width;
    rows = c.rows;
    src_pitch = (if c.fill then 0 else c.width + ps);
    dst_pitch = c.width + pd;
    local = `None;
    device_memory = false;
    fill = (if c.fill then Some 'f' else None);
  }

let model c ~from ~into =
  let so, ps = c.src and d_o, pd = c.dst in
  let m = Bytes.of_string into in
  if c.fill then Bytes.fill m d_o c.width 'f'
  else
    for r = 0 to c.rows - 1 do
      Bytes.blit_string from (so + (r * (c.width + ps))) m
        (d_o + (r * (c.width + pd)))
        c.width
    done;
  Bytes.to_string m

let sizes c =
  let so, ps = c.src and d_o, pd = c.dst in
  let n off pad =
    off + span ~width:c.width ~rows:c.rows ~pitch:(c.width + pad)
  in
  (n so ps + c.slack, n d_o pd + c.slack)

(* The memory device copies each row of a box into place and fills a fill's
   row, at its hand-over, and writes no other byte. *)
let memory_device_law c =
  covers c;
  let word = H.word () in
  let edge = memory_device word in
  Fun.protect ~finally:(fun () -> drop_memory_device edge) @@ fun () ->
  let ns, nd = sizes c in
  let src = H.pages (Int.max ns 1) and dst = H.pages (Int.max nd 1) in
  let from = pattern ns 1 and into = pattern nd 2 in
  H.write src from;
  H.write dst into;
  let raw =
    part c ~src:(Nativeint.of_int src) ~dst:(Nativeint.of_int dst)
  in
  (match G.hand edge ~v:1 ~queue:1 raw with
  | Handed -> ()
  | Room r -> failf "the room check answered %d" r
  | Failed -> fail "the submit failed");
  equal int ~msg:"the word" 1 (H.get64 (Nativeint.to_int word));
  equal string (model c ~from ~into) (H.read dst nd);
  H.free_pages src (Int.max ns 1);
  H.free_pages dst (Int.max nd 1)

(* Polled copies each row of a box into place and fills a fill's row, in
   memory of either kind, and writes no other byte. *)
let polled_law (c, (sm, dm)) =
  covers c;
  let p = P.make ~host_visible:false () in
  let g = P.bind p ~word:(H.word ()) in
  let e = E.make g in
  Fun.protect ~finally:(fun () -> E.release e) @@ fun () ->
  let ns, nd = sizes c in
  let src = E.alloc e sm (Int.max ns 1) and dst = E.alloc e dm (Int.max nd 1) in
  let from = pattern ns 1 and into = pattern nd 2 in
  E.write e src ~at:0 from;
  E.write e dst ~at:0 into;
  let so, ps = c.src and d_o, pd = c.dst in
  E.run e
    {
      dst = Region dst;
      dst_offset = d_o;
      src = (if c.fill then None else Some (Region src));
      src_offset = (if c.fill then 0 else so);
      width = c.width;
      rows = c.rows;
      src_pitch = (if c.fill then 0 else c.width + ps);
      dst_pitch = c.width + pd;
      fill = (if c.fill then Some 'f' else None);
    };
  equal string (model c ~from ~into) (E.read e dst ~at:0 nd)

let kinds =
  let pp ppf (m : Rig_edge.memory) =
    Format.pp_print_string ppf
      (match m with
      | Device -> "Device"
      | Pinned -> "Pinned"
      | Mapped -> "Mapped")
  in
  let kind = Gen.of_list ~pp [ Rig_edge.Device; Pinned ] in
  Gen.pair kind kind

let () =
  exit
  @@ run "rig.copy_parts"
       [
         group ~timeout "copy parts"
           [
             prop ~count:200
               "the memory device runs boxes and fills as rig_edge.h states"
               cases memory_device_law;
             prop ~count:200 "Polled runs boxes and fills as rig_edge.h states"
               (Gen.pair cases kinds) polled_law;
           ];
       ]
