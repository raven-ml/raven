(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Copy parts as rig_edge.h's struct rig_part states them (RIG_COPY). Rows and
   fills run on the drivers of rig's own suites: the software device's, whose
   memory is host memory, and Polled's, whose Device memory the host does not
   address. rig copies host-addressed memory on the host, so no call of rig.mli
   reaches the software device's rows: each part goes to the driver's C
   functions directly (Rig_gpu_support.hand). The facts rig hands with a part,
   [copy.device_memory] and [copy.local], are read from what Polled ran. *)

open Windtrap
module G = Rig_gpu_support
module H = Rig_gpu_support.Host
module P = Rig_support.Polled
module E = Rig_gpu_support.Edge (P)
module B = Rig.Buffer
module Sub = Rig.Submission
module Support = Rig_support

let pattern n seed =
  String.init n (fun i -> Char.chr ((seed + (i * 7)) land 255))

(* The software device's C state over the word at [word], as Rig.software makes
   it, and its free, as rig.h's section for rig's tests declares them. *)
external software : nativeint -> nativeint = "caml_rig_software_new"
external drop_software : nativeint -> unit = "caml_rig_software_drop"

(* A box of [rows] rows of [width] bytes, each side at its offset and with its
   pitch [width] plus its pad, its memory [slack] bytes past its last row; or,
   with [fill], a fill of the destination's one row. *)
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
        c.rows c.width (fst c.src)
        (c.width + snd c.src)
        (fst c.dst)
        (c.width + snd c.dst)
        c.slack)
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
      Bytes.blit_string from
        (so + (r * (c.width + ps)))
        m
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

(* The software device copies each row of a box into place and fills a fill's
   row, at its hand-over, and writes no other byte. *)
let software_law c =
  covers c;
  let word = H.word () in
  let edge = software word in
  Fun.protect ~finally:(fun () -> drop_software edge) @@ fun () ->
  let ns, nd = sizes c in
  let src = H.pages (Int.max ns 1) and dst = H.pages (Int.max nd 1) in
  let from = pattern ns 1 and into = pattern nd 2 in
  H.write src from;
  H.write dst into;
  let raw = part c ~src:(Nativeint.of_int src) ~dst:(Nativeint.of_int dst) in
  (match G.hand edge ~v:1 ~queue:1 raw with
  | Handed -> ()
  | Room r -> failf "the room check answered %d" r
  | Failed -> fail "the submit failed");
  equal int ~msg:"the word" 1 (H.get64 (Nativeint.to_int word));
  equal string (model c ~from ~into) (H.read dst nd);
  H.free_pages src (Int.max ns 1);
  H.free_pages dst (Int.max nd 1)

(* Polled copies each row of a box into place and fills a fill's row, in memory
   of either kind, and writes no other byte. *)
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

(* Cases the draws reach too rarely for every run: a fill of no bytes, in about
   7% of cases, and a box of rows of no width, in about 10%. *)
let empty_fill =
  { width = 0; rows = 1; src = (0, 0); dst = (3, 0); slack = 1; fill = true }

let empty_rows =
  { width = 0; rows = 3; src = (2, 1); dst = (0, 4); slack = 0; fill = false }

(* copy.device_memory *)

let submit ?(buffers = [||]) s = Rig.submit s ~buffers ~args:[||]

(* [b] borrowed on [d], as a submit on [d] takes it. *)
let on d b = require_some (B.borrow d b)

(* Each end of a copy, by where its allocation lies, and its
   [copy.device_memory] where both ends are of its kind: 1 exactly for this
   device's Device memory. *)
let origins =
  [
    ("device", `Own B.Device, true);
    ("device returned", `Returned, true);
    ("mapped", `Own B.Mapped, false);
    ("pinned", `Own B.Pinned, false);
    ("host", `Host, false);
    ("peer device", `Peer B.Device, false);
    ("peer mapped", `Peer B.Mapped, false);
    ("peer pinned", `Peer B.Pinned, false);
  ]

let serial = Atomic.make 0

(* Copies between views of [length] bytes, at [src_first] and [dst_first] of
   allocations of every origin, borrowed on the device, have
   [copy.device_memory] where both ends' allocations are the device's Device
   memory with bytes. *)
let device_memory_law (length, src_first, dst_first) =
  let name =
    Printf.sprintf "copy-parts:memory-%d" (Atomic.fetch_and_add serial 1)
  in
  let d, p = P.open_ ~host_visible:false name in
  let peer, _ = P.open_ ~host_visible:false (name ^ "-peer") in
  let bytes = length + max src_first dst_first in
  let end_ first (label, origin, device_memory) =
    let root =
      match origin with
      | `Own memory -> B.create ~memory d bytes
      | `Returned -> on d (on peer (B.create d bytes))
      | `Peer memory -> B.create ~memory peer bytes
      | `Host ->
          B.create Rig.host (if bytes = 0 then 0 else max B.paged_from bytes)
    in
    ( label,
      on d (B.view (on d root) ~first ~length),
      device_memory && bytes <> 0 )
  in
  let rows =
    List.concat_map
      (fun (src_label, src, src_device) ->
        List.map
          (fun (dst_label, dst, dst_device) ->
            ( src_label ^ " to " ^ dst_label,
              Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst)),
              src_device && dst_device ))
          (List.map (end_ dst_first) origins))
      (List.map (end_ src_first) origins)
  in
  Rig.Point.wait
    (submit
       (Sub.make d (Array.of_list (List.map (fun (_, part, _) -> part) rows))));
  equal
    (list (pair string bool))
    (List.map (fun (label, _, expected) -> (label, expected)) rows)
    (List.map2
       (fun (label, _, _) (c : P.copy) -> (label, c.device_memory))
       rows (P.copies p));
  Rig.close d;
  Rig.close peer

let device_memory_covers (length, src_first, dst_first) =
  cover "an empty view of an allocation"
    (length = 0 && max src_first dst_first > 0);
  cover "one byte" (length = 1);
  cover "different offsets" (src_first <> dst_first);
  device_memory_law (length, src_first, dst_first)

(* A device of another machine, a Polled one, and its copies of [a] into [b] and
   of memory this process's host addresses, by Buffer.copy and by a submission's
   copies through a buffer or a slot, as it ran them. *)
let far_copies () =
  let machine = Support.machine "copy-parts:machine" in
  let p = P.make ~host_visible:false () in
  let d =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_ ~machine (P.opener "copy-parts:far" p))
  in
  let a = B.create d 8 and b = B.create d 8 in
  let mapped = B.create ~memory:B.Mapped d 8 in
  let pinned = B.create ~memory:B.Pinned d 8 in
  let host = B.create Rig.host 8 in
  List.iter
    (fun (src, dst) -> B.copy ~src ~dst)
    [ (a, b); (host, a); (a, b); (b, host); (mapped, a); (a, b); (a, pinned) ];
  let copy src dst buffers =
    Rig.Point.wait (submit (Sub.make d [| Sub.copy ~src ~dst |]) ~buffers)
  in
  copy (Slot 0) (Buffer a) [| host |];
  copy (Buffer a) (Slot 0) [| host |];
  copy (Buffer host) (Slot 0) [| b |];
  (d, host, P.copies p)

(* The copies [far_copies] makes, in order. *)
let far_labels =
  [
    "device to device";
    "host to device";
    "device to device";
    "device to host";
    "mapped to device";
    "device to device";
    "device to pinned";
    "a host slot to device";
    "device to a host slot";
    "host to a device slot";
  ]

let test_far_device_memory () =
  let d, _, copies = far_copies () in
  equal
    (list (pair string bool))
    (List.combine far_labels
       [ true; false; true; false; false; true; false; false; false; false ])
    (List.combine far_labels
       (List.map (fun (c : P.copy) -> c.device_memory) copies));
  Rig.close d

(* copy.local *)

let test_far_local () =
  let d, _, copies = far_copies () in
  let side = function `None -> "none" | `Src -> "src" | `Dst -> "dst" in
  equal
    (list (pair string string))
    (List.combine far_labels
       [
         "none";
         "src";
         "none";
         "dst";
         "none";
         "none";
         "none";
         "src";
         "dst";
         "src";
       ])
    (List.combine far_labels
       (List.map (fun (c : P.copy) -> side c.local) copies));
  Rig.close d

(* Submission.copy *)

(* rig.mli, Submission.copy: on a driver's device of another machine, one of the
   two ends may instead be memory this process's host addresses, and a slot's
   buffer only if copies alone name the slot. *)
let test_far_refusals () =
  let d, host, _ = far_copies () in
  let other = B.create Rig.host 8 in
  raises_match ~msg:"two local buffers" Exn.invalid_arg (fun () ->
      Sub.make d Sub.[| copy ~src:(Buffer host) ~dst:(Buffer other) |]);
  raises_match ~msg:"a local buffer and a local slot" Exn.invalid_arg (fun () ->
      submit
        (Sub.make d Sub.[| copy ~src:(Buffer host) ~dst:(Slot 0) |])
        ~buffers:[| other |]);
  let fill = Sub.make d [| Support.touching [| Sub.read |] |] in
  raises_match ~msg:"a ref's local buffer" Exn.invalid_arg (fun () ->
      Rig.submit fill ~buffers:[| host |] ~args:(Support.zeros 1));
  Rig.close d

let () =
  exit
  @@ run "rig.copy_parts"
       [
         group "copy parts"
           [
             prop ~count:200 ~examples:[ empty_fill; empty_rows ]
               "the software device runs boxes and fills as rig_edge.h states"
               cases software_law;
             prop ~count:200
               ~examples:
                 [
                   (empty_fill, (Rig_edge.Device, Rig_edge.Device));
                   (empty_rows, (Rig_edge.Pinned, Rig_edge.Device));
                 ]
               "Polled runs boxes and fills as rig_edge.h states"
               (Gen.pair cases kinds) polled_law;
           ];
         group "copy.device_memory"
           [
             test "is 0 for empty allocations" (fun () ->
                 device_memory_law (0, 0, 0));
             prop
               ~examples:[ (0, 1, 2); (1, 0, 0); (17, 3, 7) ]
               "is 1 exactly where both ends' allocations are the device's \
                Device memory, through views and borrows"
               (Gen.triple (Gen.int_range 0 64) (Gen.int_range 0 7)
                  (Gen.int_range 0 7))
               device_memory_covers;
             test "a far device's copies carry it as rig_edge.h states"
               test_far_device_memory;
           ];
         group "copy.local"
           [
             test "a far device's copies name the side of this process's memory"
               test_far_local;
           ];
         group "Submission.copy"
           [
             test
               "a far device refuses two ends of this process's memory, and a \
                ref's"
               test_far_refusals;
           ];
       ]
