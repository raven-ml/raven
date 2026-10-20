(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Queued copies and fills (Buffer.submit_copy, Buffer.submit_fill), boxes, and
   staging memory's halves under them. *)

open Windtrap
module B = Rig.Buffer
module P = Rig_support.Polled
module Support = Rig_support

let timeout = 60.
let half = 32 lsl 20

let bytes b =
  B.read b Bigarray.char (fun ba ->
      String.init (Bigarray.Array1.dim ba) (Bigarray.Array1.get ba))

let filled n c =
  let b = B.create Rig.host n in
  B.write b Bigarray.char (fun a -> Bigarray.Array1.fill a c);
  b

(* [b]'s bytes, through a blocking copy into host memory. *)
let contents b =
  let s = Bytes.create (B.length b) in
  B.blit_to_bytes b 0 s 0 (B.length b);
  Bytes.to_string s

(* A buffer of [n] bytes on [d] holding [s]'s first [n]. *)
let holding d s =
  let b = B.create d (String.length s) in
  B.blit_from_string s 0 b 0 (String.length s);
  b

(* The sleeps of every device of [ps]. *)
let sleeps ps = List.fold_left (fun n p -> n + P.sleeps p) 0 ps

(* The 32 MiB host halves [p]'s device mapped. *)
let halves_mapped p = List.length (List.filter (( = ) half) (P.host_maps p))

(* A device whose memory the host does not address. *)
let hidden ?peers name = P.open_ ~host_visible:false ?peers name

(* Returning once queued *)

(* A copy a device runs returns before the device has run it: its point is
   not reached and the host never slept, and the bytes are there once the
   point is. *)
let test_direct_queued () =
  let d, p = hidden "copy:direct" in
  let src = holding d (String.make 64 'q') and dst = B.create d 64 in
  let before = P.sleeps p in
  let pt = B.submit_copy ~src ~dst [| B.row 64 |] in
  equal ~msg:"sleeps" int before (P.sleeps p);
  equal ~msg:"reached on return" bool false (Rig.Point.reached pt);
  Rig.Point.wait pt;
  equal string (String.make 64 'q') (contents dst)

(* A copy from host memory that starts on no page goes through staging memory:
   the host writes the piece and returns with the device's leg out of it still
   queued. *)
let test_staged_queued () =
  let d, p = hidden "copy:staged" in
  let dst = B.create d 64 in
  let before = P.sleeps p in
  let pt = B.submit_copy ~src:(filled 64 's') ~dst [| B.row 64 |] in
  equal ~msg:"sleeps" int before (P.sleeps p);
  equal ~msg:"reached on return" bool false (Rig.Point.reached pt);
  Rig.Point.wait pt;
  equal string (String.make 64 's') (contents dst)

(* A copy the host makes is done when it returns, and its point, the host's,
   is reached. *)
let test_host_done () =
  let src = filled 64 'h' and dst = filled 64 '\000' in
  let pt = B.submit_copy ~src ~dst [| B.row ~dst_offset:8 32 |] in
  equal ~msg:"the point's device" bool true
    (Option.equal Rig.equal (Some Rig.host) (Rig.Point.device pt));
  equal ~msg:"reached" bool true (Rig.Point.reached pt);
  equal string (String.make 8 '\000' ^ String.make 32 'h' ^ String.make 24 '\000')
    (bytes dst)

(* A queued copy follows the points of its waits: on a device whose queue
   waits for no other device's, its submit waits for them on the host. *)
let test_waits () =
  let d, pd = hidden "copy:waits-src" in
  let e, _ = hidden "copy:waits-dst" in
  let a = holding d (String.make 64 'w') and b = B.create d 64 in
  let first = B.submit_copy ~src:a ~dst:b [| B.row 64 |] in
  let out = B.create e 64 in
  let before = P.sleeps pd in
  let pt = B.submit_copy ~waits:[| first |] ~src:(B.create e 64) ~dst:out
      [| B.row 64 |]
  in
  greater ~msg:"sleeps on the wait's device" int ~than:before (P.sleeps pd);
  equal ~msg:"the wait reached" bool true (Rig.Point.reached first);
  Rig.Point.wait pt

(* A fill a device's copy engine runs returns before it has run. *)
let test_fill_queued () =
  let d, p = hidden "copy:fill" in
  let b = B.create d 64 in
  let before = P.sleeps p in
  let pt = B.submit_fill b 'f' in
  equal ~msg:"sleeps" int before (P.sleeps p);
  equal ~msg:"reached on return" bool false (Rig.Point.reached pt);
  Rig.Point.wait pt;
  equal string (String.make 64 'f') (contents b)

(* Staging memory under queued copies *)

(* Small copies from host memory to several devices through the host's
   staging memory return with no sleep, their pieces one after the other in
   one half, which each device maps once. *)
let test_small_uploads () =
  let devices = List.init 4 (fun i -> hidden (Printf.sprintf "copy:up-%d" i)) in
  let ps = List.map snd devices in
  let before = sleeps ps in
  let copies =
    List.concat_map
      (fun (d, _) ->
        List.map
          (fun c ->
            let dst = B.create d 64 in
            (B.submit_copy ~src:(filled 64 c) ~dst [| B.row 64 |], dst, c))
          [ 'a'; 'b' ])
      devices
  in
  equal ~msg:"sleeps" int before (sleeps ps);
  equal ~msg:"halves mapped by each device" (list int) [ 1; 1; 1; 1 ]
    (List.map halves_mapped ps);
  List.iter (fun (pt, _, _) -> Rig.Point.wait pt) copies;
  equal ~msg:"the copies' bytes" (list string)
    (List.map (fun (_, _, c) -> String.make 64 c) copies)
    (List.map (fun (_, dst, _) -> contents dst) copies)

(* A copy from one device to several others through staging memory, which
   none maps of another, writes a half whose work is done for each, or a new
   one: the source's legs follow none of the destinations' drains, so its
   device runs them all with the destinations' still queued, and no submit
   waits on the host. *)
let test_fan_out () =
  let open_ name =
    P.open_ ~host_visible:false ~peers:false ~waits_on:[ `Host ] name
  in
  let a, pa = open_ "copy:fan-src" in
  let src = holding a (String.make 64 'o') in
  let dsts =
    List.init 3 (fun i ->
        let d, p = open_ (Printf.sprintf "copy:fan-%d" i) in
        (B.create d 64, p))
  in
  let ps = pa :: List.map snd dsts in
  let before = sleeps ps in
  let pts =
    List.map (fun (dst, _) -> B.submit_copy ~src ~dst [| B.row 64 |]) dsts
  in
  equal ~msg:"sleeps" int before (sleeps ps);
  equal ~msg:"the source's legs run alone" int 3 (P.run pa);
  List.iter Rig.Point.wait pts;
  equal ~msg:"the copies' bytes" (list string)
    (List.init 3 (fun _ -> String.make 64 'o'))
    (List.map (fun (dst, _) -> contents dst) dsts)

(* A staged copy of three pieces through a device's own staging memory holds
   two halves, which take the pieces in turn. *)
let test_large_halves () =
  let d, p = P.open_ ~maps_host:false ~host_visible:false "copy:large" in
  let pinned () =
    List.length
      (List.filter (fun (k, n, _) -> k = B.Pinned && n = half) (P.allocs p))
  in
  let n = (2 * half) + 4096 in
  let dst = B.create d n in
  Rig.Point.wait (B.submit_copy ~src:(filled n 'l') ~dst [| B.row n |]);
  equal ~msg:"halves made" int 2 (pinned ());
  let back = B.create Rig.host n in
  B.copy ~src:dst ~dst:back;
  equal ~msg:"bytes that differ" int 0
    (B.read back Bigarray.char (fun ba ->
         let k = ref 0 in
         for i = 0 to n - 1 do
           if Bigarray.Array1.unsafe_get ba i <> 'l' then incr k
         done;
         !k))

(* Boxes *)

(* Copies [boxes] from [src] into [dst] on the host, one after the other. *)
let reference src dst boxes =
  let dst = Bytes.of_string dst in
  List.iter
    (fun (x : B.box) ->
      for i = 0 to (if x.width = 0 then 0 else x.rows) - 1 do
        Bytes.blit_string src
          (x.src_offset + (i * x.src_pitch))
          dst
          (x.dst_offset + (i * x.dst_pitch))
          x.width
      done)
    boxes;
  Bytes.to_string dst

let side = 256
let pattern seed = String.init side (fun i -> Char.chr ((seed + (i * 7)) land 255))

(* A box of the buffers of [side] bytes: no rows or no width now and then,
   pitches equal to the width or above it, and a last row at a side's end. *)
let box_gen =
  let open Gen in
  let* width =
    frequency [ (1, constant 0); (1, constant 1); (6, int_range 1 24) ]
  in
  let* rows =
    frequency [ (1, constant 0); (2, constant 1); (5, int_range 2 5) ]
  in
  let pitch =
    frequency [ (2, constant width); (3, map (( + ) width) (int_range 1 8)) ]
  in
  let* src_pitch = pitch in
  let* dst_pitch = pitch in
  let span pitch =
    if width = 0 || rows = 0 then 0 else ((rows - 1) * pitch) + width
  in
  let offset pitch =
    let most = side - span pitch in
    frequency [ (1, constant most); (4, int_range 0 most) ]
  in
  let+ src_offset = offset src_pitch and+ dst_offset = offset dst_pitch in
  { B.src_offset; dst_offset; width; rows; src_pitch; dst_pitch }

let pp_box ppf (x : B.box) =
  Format.fprintf ppf "{%d->%d %dx%d pitches %d,%d}" x.src_offset x.dst_offset
    x.width x.rows x.src_pitch x.dst_pitch

(* An io library over host memory whose pages it never gives, so the io
   device's reads and writes move every byte. *)
module Store = struct
  type t = unit
  type region = B.t

  exception Fault of string

  let region_key : region Type.Id.t = Type.Id.make ()
  let budget _ = max_int
  let alloc () n = Some (B.create Rig.host (Int.max n (1 lsl 16)))
  let free () _ = ()
  let base r = Support.Reader.host r
  let read () r ~at ~dst ~len = Support.move ~dst ~src:(base r + at) len
  let write () r ~at ~src ~len = Support.move ~dst:(base r + at) ~src len
  let pages () _ = None
  let prefetch () _ ~at:_ ~len:_ = ()
  let stop () = ()
end

(* The routes a copy takes, each with how it makes its two buffers. *)
type route = { name : string; make : unit -> B.t * B.t }

let routes =
  lazy
    (let d, _ = hidden "copy:box-device" in
     let e, _ = hidden ~peers:false "copy:box-peer" in
     let f, _ = hidden ~peers:false "copy:box-peer-2" in
     let own name = P.open_ ~maps_host:false ~peers:false ~host_visible:false name in
     let g, _ = own "copy:box-own" and h, _ = own "copy:box-own-2" in
     let m = Result.get_ok (Rig.memory_device "copy:box-memory") in
     let io =
       require_ok ~pp:Format.pp_print_string
         (Rig.open_io
            { name = "copy:box-io"; io = (module Store); make = (fun () -> Ok ()) })
     in
     let on a b () = (B.create a side, B.create b side) in
     [
       { name = "host"; make = on Rig.host Rig.host };
       { name = "a device's copy engine"; make = on d d };
       { name = "a memory device"; make = on m m };
       { name = "staged into a device"; make = on Rig.host d };
       { name = "staged out of a device"; make = on d Rig.host };
       { name = "staged between two devices"; make = on e f };
       { name = "between two devices' own staging"; make = on g h };
       { name = "out of an io device"; make = on io Rig.host };
       { name = "into an io device"; make = on Rig.host io };
     ])

let case =
  let open Gen in
  let+ route = int_range 0 8
  and+ boxes = list ~size:(int_range 1 4) box_gen
  and+ seed = int_range 0 255 in
  (route, boxes, seed)

let case =
  Gen.with_pp
    (fun ppf (r, boxes, seed) ->
      Format.fprintf ppf "%s, seed %d, %a" (List.nth (Lazy.force routes) r).name
        seed
        (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_box)
        boxes)
    case

(* A queued copy of boxes lands each box after the one before it, on every
   route, as the host's reference copy does. *)
let box_law (r, boxes, seed) =
  let route = List.nth (Lazy.force routes) r in
  cover route.name true;
  cover "a box of no bytes"
    (List.exists (fun (x : B.box) -> x.width = 0 || x.rows = 0) boxes);
  cover "a box of rows that are no run"
    (List.exists (fun (x : B.box) -> x.rows > 1 && x.src_pitch > x.width) boxes);
  cover "a last row at a side's end"
    (List.exists
       (fun (x : B.box) ->
         x.rows > 0 && x.width > 0
         && (x.rows - 1) * x.dst_pitch + x.width + x.dst_offset = side)
       boxes);
  let src, dst = route.make () in
  let s = pattern seed and t = pattern (seed + 101) in
  B.blit_from_string s 0 src 0 side;
  B.blit_from_string t 0 dst 0 side;
  Rig.Point.wait (B.submit_copy ~src ~dst (Array.of_list boxes));
  equal string (reference s t boxes) (contents dst)

(* Many runs in one call, whose destinations overlap, land in order: a byte
   two runs write holds the later run's, and one submission runs them all. *)
let test_many_runs () =
  let d, p = hidden "copy:many" in
  let s = String.init 4096 (fun i -> Char.chr (i land 255)) in
  let src = holding d s and dst = holding d (String.make 4096 '.') in
  let runs =
    List.init 600 (fun i ->
        B.row ~src_offset:(i * 5 mod 4000) ~dst_offset:(i * 3 mod 4000) 17)
  in
  let before = P.submits p in
  let pt = B.submit_copy ~src ~dst (Array.of_list runs) in
  equal ~msg:"submissions" int (before + 1) (P.submits p);
  Rig.Point.wait pt;
  equal string (reference s (String.make 4096 '.') runs) (contents dst)

(* A box larger than a half goes through staging memory in pieces of whole
   rows, three rows of 9 MiB and then the fourth, and a row larger than a half
   in ranges of its bytes. *)
let test_large_box () =
  let d, _ = hidden "copy:large-box" in
  let width = 9 lsl 20 in
  let rows =
    {
      B.src_offset = 7;
      dst_offset = 0;
      width;
      rows = 4;
      src_pitch = width + 9;
      dst_pitch = width;
    }
  in
  let wide = B.row ~src_offset:5 ~dst_offset:(4 * width) (half + 3) in
  let n = (4 * (width + 9)) + half + 16 in
  let byte i = Char.unsafe_chr ((i + (i lsr 20)) land 255) in
  let src = B.create Rig.host n in
  B.write src Bigarray.char (fun ba ->
      for i = 0 to n - 1 do
        Bigarray.Array1.unsafe_set ba i (byte i)
      done);
  let dst = B.create d n in
  Rig.Point.wait (B.submit_copy ~src ~dst [| rows; wide |]);
  let back = B.create Rig.host n in
  B.copy ~src:dst ~dst:back;
  let expect i =
    if i < 4 * width then byte (7 + (i / width * (width + 9)) + (i mod width))
    else byte (5 + i - (4 * width))
  in
  let wrong =
    B.read back Bigarray.char (fun ba ->
        let k = ref 0 in
        for i = 0 to (4 * width) + half + 2 do
          if Bigarray.Array1.unsafe_get ba i <> expect i then incr k
        done;
        !k)
  in
  equal ~msg:"bytes that differ" int 0 wrong

(* The refusals of a box, at the call and with nothing queued. *)
let test_refusals () =
  let d, p = hidden "copy:refusals" in
  let src = B.create d 64 and dst = B.create d 64 in
  let box ?(src_offset = 0) ?(dst_offset = 0) ?(src_pitch = 8)
      ?(dst_pitch = 8) width rows =
    { B.src_offset; dst_offset; width; rows; src_pitch; dst_pitch }
  in
  let before = P.submits p in
  List.iter
    (fun x ->
      raises_match ~msg:(Format.asprintf "%a" pp_box x) Exn.invalid_arg
        (fun () -> B.submit_copy ~src ~dst [| B.row 8; x |]))
    [
      box ~src_pitch:7 8 2;
      box ~dst_pitch:0 8 2;
      box (-1) 1;
      box 8 (-1);
      box ~src_offset:(-1) 8 1;
      box ~src_offset:57 8 1;
      box ~dst_offset:1 8 8;
      box ~src_pitch:max_int 8 2;
      box ~dst_offset:max_int 8 1;
    ];
  equal ~msg:"submissions" int before (P.submits p);
  Rig.Point.wait
    (B.submit_copy ~src ~dst
       [| box ~src_offset:max_int ~src_pitch:max_int 0 3; box 8 8 |])

(* Several domains *)

(* Copies through the host's staging memory from two domains: uploads into
   buffers on two devices that map none of each other's memory, queued
   copies between them, and reads back, each as some order of the calls
   says. *)
let stage_devices =
  lazy
    (Array.init 2 (fun i ->
         fst
           (P.open_ ~host_visible:false ~peers:false ~runs:`Itself
              (Printf.sprintf "copy:domains-%d" i))))

let buf = abstract "b"
let byte = Gen.with_pp Format.pp_print_char (Gen.char_range 'a' 'z')

let domain_commands =
  [
    command "create"
      (Gen.with_pp Format.pp_print_int (Gen.int_range 0 1) @-> makes buf)
      (fun _ -> ref (String.make 64 '\000'))
      (fun i ->
        let b = B.create (Lazy.force stage_devices).(i) 64 in
        B.fill b '\000';
        b);
    command "upload"
      (byte @-> buf ^-> returns unit)
      (fun c r -> r := String.make 64 c)
      (fun c b ->
        ignore (B.submit_copy ~src:(filled 64 c) ~dst:b [| B.row 64 |]
          : Rig.Point.t));
    command "copy"
      (buf ^-> buf ^-> returns unit)
      (fun a b -> b := !a)
      (fun a b ->
        if a != b then
          ignore (B.submit_copy ~src:a ~dst:b [| B.row 64 |] : Rig.Point.t));
    command "read" (buf ^-> returns string) (fun r -> !r) contents;
  ]

let () =
  exit
  @@ run "rig.copy"
       [
         group ~timeout "returning once queued"
           [
             test "a copy a device runs returns before it runs"
               test_direct_queued;
             test "a staged copy returns once the host wrote its piece"
               test_staged_queued;
             test "a copy the host makes returns done, at a point of the host"
               test_host_done;
             test "a queued copy follows its waits" test_waits;
             test "a fill a copy engine runs returns before it runs"
               test_fill_queued;
           ];
         group ~timeout "staging memory"
           [
             test "small uploads to several devices share a half and never sleep"
               test_small_uploads;
             test "a staged fan-out from one device chains no device to another"
               test_fan_out;
             test "a large staged copy holds two halves" test_large_halves;
           ];
         group ~timeout "boxes"
           [
             prop ~count:300
               "a queued copy of boxes lands as the host's copy, on every route"
               case box_law;
             test "many runs in one call land in order, in one submission"
               test_many_runs;
             test "a box larger than a half lands through staging memory"
               test_large_box;
             test "a box that does not fit is refused, queueing nothing"
               test_refusals;
           ];
         group ~timeout "domains"
           [
             stateful ~count:20 ~steps:8 ~domains:2
               "copies through staging memory on two domains answer as some \
                order of them"
               domain_commands;
           ];
       ]
