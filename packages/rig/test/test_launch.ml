(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module Run = Sub.Run
module P = Rig_support.Polled

let timeout = Rig_support.time 60.
let strf = Printf.sprintf

let functions d =
  require_ok ~pp:Format.pp_print_string (Rig.Image.load d "functions")

let fn image f = require_some (Rig.Image.resolve image f)

let launch ?(params = 0) ?(refs = [||]) image kernel =
  Sub.launch ~refs (fn image kernel) ~params

let one run b =
  Run.groups run b 1 1 1;
  Run.threads run b 1 1 1

let submit ?(buffers = [||]) ?waits s run = Rig.submit ?waits s ~run ~buffers

(* Fresh Polled devices: a name opens once. *)
let opened = Atomic.make 0

let polled ?addresses name =
  P.open_ ?addresses (strf "launch:%s-%d" name (Atomic.fetch_and_add opened 1))

let le64 v =
  let b = Bytes.create 8 in
  Bytes.set_int64_le b 0 (Int64.of_int v);
  Bytes.to_string b

let host_bytes b n =
  let s = Bytes.create n in
  B.blit_to_bytes b 0 s 0 n;
  Bytes.to_string s

(* Patching *)

(* A run's buffer, used with [access]: [bytes] bytes, [first] bytes into a
   buffer of Polled's own memory, or into host memory that starts a page,
   borrowed on Polled. *)
type slot = { bytes : int; first : int; borrowed : bool; access : Rig.access }

(* A ref: the parameter word [at] holds [offset] bytes into run buffer
   [slot]. *)
type ref_case = { at : int; slot : int; offset : int }

type launch_case = {
  params : string;  (** The parameter bytes, a whole number of 32-bit words. *)
  refs : ref_case list;
  groups : int * int * int;
  threads : int * int * int;
  shared : int;
}

type case = { slots : slot list; launches : launch_case list }

let pp_case ppf c =
  let pp_slot ppf s =
    Format.fprintf ppf "{%d bytes at %d%s%s}" s.bytes s.first
      (if s.borrowed then ", borrowed" else "")
      (if s.access = Rig.Read_write then ", written" else "")
  in
  let pp_ref ppf r = Format.fprintf ppf "%d->%d+%d" r.at r.slot r.offset in
  let pp_launch ppf l =
    let x, y, z = l.groups and tx, ty, tz = l.threads in
    Format.fprintf ppf
      "{%d params; refs [%a]; groups %dx%dx%d; threads %dx%dx%d; shared %d}"
      (String.length l.params)
      (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_ref)
      l.refs x y z tx ty tz l.shared
  in
  Format.fprintf ppf "slots [%a]; launches [%a]"
    (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_slot)
    c.slots
    (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_launch)
    c.launches

let gen_case =
  let open Gen in
  let* nslots = int_range 1 4 in
  let slot =
    let+ bytes = int_range 8 64
    and+ first = int_range 0 2
    and+ borrowed = bool
    and+ access = of_list [ Rig.Read; Rig.Read_write ] in
    { bytes; first = 8 * first; borrowed; access }
  in
  let* slots = list ~size:(constant nslots) slot in
  let launch_case =
    let* words = int_range 0 24 in
    let params_bytes = 4 * words in
    let* params = string_of ~size:(constant params_bytes) char in
    (* The 8-byte words a ref may name: those inside the parameters. *)
    let ats = List.init (params_bytes / 8) (fun k -> 8 * k) in
    let* named = subsequence ats in
    let ref_at at =
      let+ slot = int_range 0 (nslots - 1) and+ offset = int_range 0 7 in
      { at; slot; offset }
    in
    let* refs =
      List.fold_right
        (fun at acc ->
          let+ r = ref_at at and+ rest = acc in
          r :: rest)
        named (constant [])
    in
    let axes n = triple (int_range 1 n) (int_range 1 n) (int_range 1 n) in
    let+ groups = axes 3 and+ threads = axes 4 and+ shared = int_range 0 64 in
    { params; refs; groups; threads; shared }
  in
  let+ launches = list ~size:(int_range 1 2) launch_case in
  { slots; launches }

let gen_case = Gen.with_pp pp_case gen_case

(* One device for every case: a case's memory returns as it is collected. *)
let patch_device = lazy (polled "patch")

(* The 64 KiB that host memory a Polled device maps starts a page from. *)
let page = 1 lsl 16

let make_slot d s =
  let mem =
    if s.borrowed then require_some (B.borrow d (B.create Rig.host page))
    else B.create d (s.first + s.bytes)
  in
  B.view mem ~first:s.first ~length:s.bytes

(* Stores [launch]'s geometry and parameters into its block [b] of [run], each
   ref's word the offset. *)
let store run b l =
  let x, y, z = l.groups and tx, ty, tz = l.threads in
  Run.groups run b x y z;
  Run.threads run b tx ty tz;
  Run.shared run b l.shared;
  for k = 0 to (String.length l.params / 4) - 1 do
    Run.int32 run b (4 * k)
      (Int32.to_int (String.get_int32_le l.params (4 * k)))
  done;
  List.iter (fun r -> Run.int64 run b r.at r.offset) l.refs

(* What the function reads: the parameters, each ref's word the address of its
   buffer plus the offset. *)
let patched buffers l =
  let p = Bytes.of_string l.params in
  List.iter
    (fun r ->
      Bytes.set_int64_le p r.at
        (Int64.of_int (Rig_support.at buffers.(r.slot) + r.offset)))
    l.refs;
  Bytes.to_string p

let pp_launch ppf (l : P.launch) =
  let x, y, z = l.groups and tx, ty, tz = l.threads in
  Format.fprintf ppf "groups %dx%dx%d threads %dx%dx%d shared %d params %S" x y
    z tx ty tz l.shared l.params

let launch_t = Testable.make ~pp:pp_launch ~equal:( = )

(* Whether a written slot comes before a read one. *)
let rec write_first = function
  | [] -> false
  | { access = Rig.Read; _ } :: rest -> write_first rest
  | { access = Rig.Read_write; _ } :: rest ->
      List.exists (fun s -> s.access = Rig.Read) rest

let patch_law c =
  let d, p = Lazy.force patch_device in
  let image = functions d in
  let buffers = Array.of_list (List.map (make_slot d) c.slots) in
  let use slot : int -> at:int -> Sub.ref =
    match (List.nth c.slots slot).access with
    | Read -> Sub.read
    | Read_write -> Sub.write
  in
  let launches =
    List.map
      (fun l ->
        let refs =
          Array.of_list (List.map (fun r -> use r.slot r.slot ~at:r.at) l.refs)
        in
        launch image "main" ~params:(String.length l.params) ~refs)
      c.launches
  in
  (* Every slot is a ref's: a fill after the launches names each. *)
  let named = Rig_support.touching (Array.of_list (List.mapi (fun i _ -> use i) c.slots)) in
  let s = Sub.make d (Array.of_list (launches @ [ named ])) in
  let run = Run.make () in
  List.iteri (fun i l -> store run (Sub.block s i) l) c.launches;
  ignore (P.launches p);
  let pt = submit ~buffers s run in
  (* Stores after the submit returned change nothing it handed over. *)
  List.iteri
    (fun i l ->
      let b = Sub.block s i in
      Run.groups run b 7 7 7;
      for k = 0 to (String.length l.params / 4) - 1 do
        Run.int32 run b (4 * k) 0x5a5a5a5a
      done)
    c.launches;
  Rig.Point.wait pt;
  let expected =
    List.map
      (fun l ->
        {
          P.groups = l.groups;
          threads = l.threads;
          shared = l.shared;
          params = patched buffers l;
        })
      c.launches
  in
  equal (list launch_t) expected (P.launches p);
  let refs = List.concat_map (fun l -> l.refs) c.launches in
  let last_word l = (String.length l.params / 8 * 8) - 8 in
  cover "a ref at 0" (List.exists (fun r -> r.at = 0) refs);
  cover "a ref at the last word"
    (List.exists
       (fun l -> List.exists (fun r -> r.at = last_word l && r.at > 0) l.refs)
       c.launches);
  cover "a slot two refs name"
    (List.exists
       (fun r -> List.exists (fun r' -> r' != r && r'.slot = r.slot) refs)
       refs);
  cover "a view at an offset" (List.exists (fun s -> s.first > 0) c.slots);
  cover "a borrowed slot" (List.exists (fun s -> s.borrowed) c.slots);
  cover "a written slot before a read one" (write_first c.slots);
  cover "no parameter bytes" (List.exists (fun l -> l.params = "") c.launches);
  cover "two launches, their blocks adjacent" (List.length c.launches = 2)

(* Ordering *)

(* A launch that writes [b] is what a read of [b] waits for. *)
let test_write_seen () =
  let d, _ = polled "write-seen" in
  let image = functions d in
  let out = B.create d 32 in
  let s =
    Sub.make d [| launch image "fill" ~params:16 ~refs:[| Sub.write 0 ~at:0 |] |]
  in
  let run = Run.make () and b = Sub.block s 0 in
  Run.groups run b 2 2 1;
  Run.threads run b 1 1 1;
  Run.int64 run b 0 0;
  Run.int64 run b 8 100;
  ignore (submit ~buffers:[| out |] s run);
  B.wait out Read;
  let host = B.create Rig.host 32 in
  B.copy ~src:out ~dst:host;
  equal string ~msg:"each group's word"
    (String.concat "" (List.map le64 [ 100; 101; 102; 103 ]))
    (host_bytes host 32)

(* A launch that reads [b] after a copy into [b], on the copy queue, reads what
   the copy wrote: the copy's value is the one before the launch's, and a
   value's work starts once the previous value's completed, on every queue. And
   the other way round. *)
let test_device_order () =
  let d, _ = polled "device-order" in
  let image = functions d in
  let data = String.init 64 (fun i -> Char.chr (i + 1)) in
  let src = B.create d 64 and b = B.create d 64 and c = B.create d 64 in
  B.copy ~src:(B.of_string data) ~dst:src;
  let copy_part ~dst src =
    Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst))
  in
  let copy_into_b = Sub.make d [| copy_part ~dst:b src |] in
  let copying =
    Sub.make d
      [|
        launch image "copy" ~params:24
          ~refs:[| Sub.read 0 ~at:0; Sub.write 1 ~at:8 |];
      |]
  in
  let run = Run.make () and blk = Sub.block copying 0 in
  one run blk;
  Run.int64 run blk 16 64;
  ignore (submit copy_into_b (Run.make ()));
  ignore (submit ~buffers:[| b; c |] copying run);
  let out = B.create Rig.host 64 in
  B.copy ~src:c ~dst:out;
  equal string ~msg:"the launch read the copy" data (host_bytes out 64);
  (* A launch on COMPUTE:0 writes [b], then a copy on COPY:0 copies it. *)
  let filling =
    Sub.make d [| launch image "fill" ~params:16 ~refs:[| Sub.write 0 ~at:0 |] |]
  in
  let run = Run.make () and blk = Sub.block filling 0 in
  Run.groups run blk 8 1 1;
  Run.threads run blk 1 1 1;
  Run.int64 run blk 0 0;
  Run.int64 run blk 8 7;
  ignore (submit ~buffers:[| b |] filling run);
  let copy_b = Sub.make d [| copy_part ~dst:c b |] in
  ignore (submit copy_b (Run.make ()));
  B.copy ~src:c ~dst:out;
  equal string ~msg:"the copy read the launch"
    (String.concat "" (List.init 8 (fun i -> le64 (7 + i))))
    (host_bytes out 64)

(* Lifetime *)

(* A submission keeps its launches' images loaded after the caller's parts array
   changed, and its submit launches the function it was made with. *)
let test_parts_changed () =
  let d, _ = polled "parts-changed" in
  let collected = ref false in
  let load () =
    let image = functions d in
    Gc.finalise_last (fun () -> collected := true) image;
    image
  in
  let fill image =
    launch image "fill" ~params:16 ~refs:[| Sub.write 0 ~at:0 |]
  in
  let parts = [| fill (load ()) |] in
  let s = Sub.make d parts in
  parts.(0) <- fill (functions d);
  Gc.full_major ();
  equal ~msg:"the part's image collected" bool false !collected;
  let out = B.create d 8 in
  let run = Run.make () and b = Sub.block s 0 in
  one run b;
  Run.int64 run b 0 0;
  Run.int64 run b 8 7;
  ignore (submit ~buffers:[| out |] s run);
  let host = B.create Rig.host 8 in
  B.copy ~src:out ~dst:host;
  equal string (le64 7) (host_bytes host 8)

(* A launch reads its refs once: a change to their array once it returned
   changes nothing. *)
let test_parts_read_once () =
  let d, p = polled "read-once" in
  let image = functions d in
  let refs = [| Sub.write 0 ~at:0 |] in
  let part = launch image "main" ~params:16 ~refs in
  refs.(0) <- Sub.write 0 ~at:8;
  let s = Sub.make d [| part |] in
  let out = B.create d 8 in
  let run = Run.make () and b = Sub.block s 0 in
  one run b;
  Run.int64 run b 0 0;
  Run.int64 run b 8 7;
  ignore (P.launches p);
  ignore (submit ~buffers:[| out |] s run);
  ignore (P.run p);
  equal ~msg:"the ref the launch was given" (list string)
    [ le64 (Rig_support.at out) ^ le64 7 ]
    (List.map (fun (l : P.launch) -> l.params) (P.launches p))

(* Refusals *)

let refused ~msg f = raises_match ~msg Exn.invalid_arg f

let test_make_refusals () =
  let d, _ = polled "make" and other, _ = polled "make-other" in
  let image = functions d in
  let make ?(image = image) ?(kernel = "main") ?(params = 16) refs =
    Sub.make d [| launch image kernel ~params ~refs |]
  in
  let at at slot = Sub.write slot ~at in
  refused ~msg:"an image of another device" (fun () ->
      make ~image:(functions other) [||]);
  is_none ~msg:"no such function" (Rig.Image.resolve image "absent");
  refused ~msg:"negative parameter bytes" (fun () -> make ~params:(-1) [||]);
  refused ~msg:"a ref at 4" (fun () -> make [| at 4 0 |]);
  refused ~msg:"a ref ending past the parameters" (fun () -> make [| at 16 0 |]);
  refused ~msg:"a ref at -8" (fun () -> make [| at (-8) 0 |]);
  refused ~msg:"a ref whose end overflows" (fun () ->
      make [| at (max_int - 7) 0 |]);
  refused ~msg:"a ref to slot 2 alone" (fun () -> make [| at 0 2 |]);
  refused ~msg:"a ref to slot -1" (fun () -> make [| at 0 (-1) |]);
  refused ~msg:"two refs at 8" (fun () -> make [| at 8 0; at 8 1 |]);
  ignore (make ~params:4096 [| at 0 0; at 4088 1 |]);
  raises ~msg:"fewer bytes than the function reads"
    (Invalid_argument
       "Rig.Submission.make: fill takes 16 bytes of parameters; part 0 gives 8")
    (fun () -> make ~kernel:"fill" ~params:8 [||]);
  raises ~msg:"a ref past the bytes the function reads"
    (Invalid_argument
       "Rig.Submission.make: part 0's ref at 16 lies past the 16 bytes of \
        parameters fill takes")
    (fun () -> make ~kernel:"fill" ~params:24 [| at 16 0 |]);
  ignore (make ~kernel:"fill" ~params:24 [| at 8 0 |]);
  let s = make [||] in
  equal int ~msg:"no value assigned" 0 (Rig.submitted d);
  refused ~msg:"the block of part 1 of 1" (fun () -> Sub.block s 1);
  refused ~msg:"the block of part -1" (fun () -> Sub.block s (-1));
  let copy =
    Sub.(copy ~src:(Buffer (B.create d 8)) ~dst:(Buffer (B.create d 8)))
  in
  let s = Sub.make d [| copy |] in
  refused ~msg:"the block of a copy" (fun () -> Sub.block s 0)

(* A refused submit assigns no value and loses nothing. *)
let refused_submit ~msg d f =
  let before = Rig.submitted d in
  refused ~msg f;
  equal int ~msg:(msg ^ ": no value") before (Rig.submitted d);
  equal (option string) ~msg:(msg ^ ": not lost") None (Rig.lost d)

let test_submit_refusals () =
  let d, _ = polled ~addresses:false "submit" in
  let image = functions d in
  let s =
    Sub.make d [| launch image "main" ~params:8 ~refs:[| Sub.read 0 ~at:0 |] |]
  in
  let b = Sub.block s 0 in
  let buffers = [| B.create ~memory:Pinned d 8 |] in
  refused_submit ~msg:"a run no setter stored into" d (fun () ->
      submit ~buffers s (Run.make ()));
  let geometry ?(groups = (1, 1, 1)) ?(threads = (1, 1, 1)) ?(shared = 0) () =
    let run = Run.make () in
    let x, y, z = groups and tx, ty, tz = threads in
    Run.groups run b x y z;
    Run.threads run b tx ty tz;
    Run.shared run b shared;
    run
  in
  refused_submit ~msg:"a ref to memory with no address" d (fun () ->
      submit ~buffers:[| B.create d 8 |] s (geometry ()));
  refused_submit ~msg:"no groups along y" d (fun () ->
      submit ~buffers s (geometry ~groups:(1, 0, 1) ()));
  refused_submit ~msg:"no threads along x" d (fun () ->
      submit ~buffers s (geometry ~threads:(0, 1, 1) ()));
  refused_submit ~msg:"65536 groups along z" d (fun () ->
      submit ~buffers s (geometry ~groups:(1, 1, 65536) ()));
  refused_submit ~msg:"shared memory past 48 KiB" d (fun () ->
      submit ~buffers s (geometry ~shared:49153 ()));
  let v = submit ~buffers s (geometry ~threads:(1024, 1, 1) ~shared:49152 ()) in
  Rig.Point.wait v

let test_setter_refusals () =
  let d, _ = polled "setters" in
  let image = functions d in
  let s = Sub.make d [| launch image "main" ~params:12 |] in
  let run = Run.make () and b = Sub.block s 0 in
  refused ~msg:"int32 at -1" (fun () -> Run.int32 run b (-1) 0);
  refused ~msg:"int32 at 9 of 12" (fun () -> Run.int32 run b 9 0);
  refused ~msg:"int64 at 8 of 12" (fun () -> Run.int64 run b 8 0);
  refused ~msg:"float64 at max_int" (fun () -> Run.float64 run b max_int 0.);
  refused ~msg:"float32 at 12 of 12" (fun () -> Run.float32 run b 12 0.);
  refused ~msg:"negative groups" (fun () -> Run.groups run b 1 (-1) 1);
  refused ~msg:"2^32 threads" (fun () -> Run.threads run b (1 lsl 32) 1 1);
  raises ~msg:"1025 threads"
    (Invalid_argument
       "Rig.Submission.Run.threads: main runs at most 1024 threads per group; \
        the block sets 5x205x1")
    (fun () -> Run.threads run b 5 205 1);
  raises ~msg:"threads whose product overflows 64 bits"
    (Invalid_argument
       "Rig.Submission.Run.threads: main runs at most 1024 threads per group; \
        the block sets 4294967295x4294967295x2")
    (fun () -> Run.threads run b 0xffff_ffff 0xffff_ffff 2);
  Run.threads run b 32 32 1;
  refused ~msg:"negative shared memory" (fun () -> Run.shared run b (-1));
  Run.int32 run b 8 0;
  Run.int64 run b 4 0;
  Run.float32 run b 8 0.;
  Run.groups run b 0xffff_ffff 1 1

(* What a driver states of its functions: Polled's fill reads 16 bytes of
   parameters and its copy 24, its main states none, and each allows 1024
   threads per group. *)
let test_entry_facts () =
  let d, _ = polled "facts" in
  let image = functions d in
  let params f = Rig.Image.params (fn image f)
  and threads f = Rig.Image.threads (fn image f) in
  equal (option int) ~msg:"fill's parameters" (Some 16) (params "fill");
  equal (option int) ~msg:"copy's parameters" (Some 24) (params "copy");
  equal (option int) ~msg:"main's parameters" None (params "main");
  equal (option int) ~msg:"main's threads" (Some 1024) (threads "main");
  let main = fn image "main" in
  equal string ~msg:"its name" "main" (Rig.Image.name main);
  is_true ~msg:"its image" (Rig.Image.image main == image)

(* Resolving a function asks the driver for it, which can fail: with no
   memory for what its launches need after the rounds of reclamation, resolve
   raises Out_of_memory; where the driver finds the function needs more than
   the device offers, Invalid_argument with its reason, the device live; and
   where the asking faults, the device's loss. A part of a resolved function
   asks nothing more. *)
let test_resolve_failures () =
  let d, p = polled "entry-failures" in
  let image = functions d in
  P.fail_at p 1 (`Refuse max_int);
  raises_match
    (function
      | Rig.Out_of_memory (d', n) -> Rig.equal d d' && n = P.entry_scratch
      | _ -> false)
    (fun () -> Rig.Image.resolve image "main");
  P.fail_at p 1 (`Refuse 0);
  let image = functions d in
  P.before p "entry" (fun () -> invalid_arg "main needs more than 1 TiB");
  raises_match
    (Exn.invalid_arg ~substring:"main needs more than 1 TiB")
    (fun () -> Rig.Image.resolve image "main");
  equal (option string) None (Rig.lost d);
  let main = fn (functions d) "main" and image = functions d in
  P.fail_at p 1 (`Fault "the engine hung");
  ignore
    (Sub.make d
       [|
         Sub.launch main ~params:8;
         Sub.fill ~refs:[| Sub.entry main ~at:0 |] Rig_support.noop ~params:8;
       |]);
  raises_match
    (function
      | Rig.Lost (d', why) -> Rig.equal d d' && why = "the engine hung"
      | _ -> false)
    (fun () -> Rig.Image.resolve image "main")

(* A setter called from a signal handler while a submit holds the run, waiting
   for a producer, raises; the submit then completes with what was stored
   before. *)
let test_setter_in_use () =
  let producer, pp = polled "in-use-producer" in
  let d, p = polled "in-use" in
  let image = functions d in
  let s = Sub.make d [| launch image "main" ~params:8 |] in
  let run = Run.make () and b = Sub.block s 0 in
  one run b;
  Run.int64 run b 0 41;
  let point = submit (Sub.make producer [||]) (Run.make ()) in
  let refused = ref None in
  let handle _ =
    refused :=
      Some
        (match Run.int64 run b 0 42 with
        | () -> "stored"
        | exception Invalid_argument why -> why)
  in
  P.interrupt pp;
  let before = Sys.signal Sys.sigint (Sys.Signal_handle handle) in
  let v =
    Fun.protect
      ~finally:(fun () -> Sys.set_signal Sys.sigint before)
      (fun () -> submit ~waits:[| point |] s run)
  in
  is_some ~msg:"the handler stored" !refused;
  contains ~msg:"its refusal" ~sub:"a submit is using the run"
    (Option.get !refused);
  ignore (P.launches p);
  Rig.Point.wait v;
  equal (list string) ~msg:"the parameters run"
    [ le64 41 ]
    (List.map (fun (l : P.launch) -> l.params) (P.launches p))

(* A run holds one submission's blocks: a store into another's zeroes the run,
   and a submit of a submission whose blocks the run does not hold is refused.
   Two fills whose blocks lie at the same bytes. *)
let test_block_owner () =
  let d, _ = polled "owner" in
  let image = functions d in
  let filling () =
    Sub.make d [| launch image "fill" ~params:16 ~refs:[| Sub.write 0 ~at:0 |] |]
  in
  let s1 = filling () and s2 = filling () in
  let b1 = Sub.block s1 0 and b2 = Sub.block s2 0 and run = Run.make () in
  let fill s =
    let out = B.create d 8 in
    Rig.Point.wait (submit ~buffers:[| out |] s run);
    Int64.to_int (String.get_int64_le (host_bytes out 8) 0)
  in
  one run b1;
  Run.int64 run b1 8 41;
  equal ~msg:"its own stores" int 41 (fill s1);
  Run.int64 run b2 8 42;
  raises_match ~msg:"another's blocks"
    (Exn.invalid_arg ~substring:"another submission's blocks") (fun () ->
      submit ~buffers:[| B.create d 8 |] s1 run);
  one run b1;
  equal ~msg:"bytes not stored since read 0" int 0 (fill s1)

(* Two domains share a run. Each store and submit takes the run whole, or
   refuses it and changes nothing while a call on the other domain holds it, so
   every outcome is one of an order of the calls with some refusals. A submit
   fills a buffer of its own with the parameter it read. A grow stores into the
   last block of a submission of [grown] launches of 4096 parameter bytes, so a
   run's first grow moves its blocks, and two runs' grows of different
   submissions each move them again; it also takes the run for that submission,
   which zeroes the fill's bytes. *)
let grown = 256

(* The model: whether the run holds [filling]'s blocks, the geometry fields
   stored since, and the parameter it holds. *)
type shared = {
  mutable own : bool;
  mutable groups : bool;
  mutable threads : bool;
  mutable stored : int;
}

let shared_t =
  abstract
    ~pp:(fun ppf r ->
      Format.fprintf ppf "own %b, groups %b, threads %b, stores %d" r.own
        r.groups r.threads r.stored)
    "r"

(* The device, [filling], and the last blocks of two submissions to grow runs
   with, made once for every case. *)
let shared =
  lazy
    (let d, _ = polled "shared" in
     let image = functions d in
     let filling =
       Sub.make d [| launch image "fill" ~params:16 ~refs:[| Sub.write 0 ~at:0 |] |]
     in
     let last n =
       let main = launch image "main" ~params:4096 in
       Sub.block (Sub.make d (Array.make n main)) (n - 1)
     in
     (d, filling, [| last grown; last (2 * grown) |]))

(* Whether [e] is a refusal of a run another call holds. *)
let busy = function
  | Invalid_argument why ->
      let sub = "is using the run" in
      let n = String.length sub and m = String.length why in
      m >= n && String.sub why (m - n) n = sub
  | _ -> false

(* The run takes [filling]'s blocks back: every byte reads 0. *)
let take_back r =
  if not r.own then begin
    r.own <- true;
    r.groups <- false;
    r.threads <- false;
    r.stored <- 0
  end

let groups run =
  let _, filling, _ = Lazy.force shared in
  Run.groups run (Sub.block filling 0) 1 1 1

let threads run =
  let _, filling, _ = Lazy.force shared in
  Run.threads run (Sub.block filling 0) 1 1 1

let store run v =
  let _, filling, _ = Lazy.force shared in
  Run.int64 run (Sub.block filling 0) 8 v

let grow run k =
  let _, _, lasts = Lazy.force shared in
  Run.int64 run lasts.(k) 4088 k

let fill run =
  let d, filling, _ = Lazy.force shared in
  let out = B.create d 8 in
  Rig.Point.wait (submit ~buffers:[| out |] filling run);
  Int64.to_int (String.get_int64_le (host_bytes out 8) 0)

(* A call's outcome: [ok] on a result, nothing on a refusal of a held run. *)
let judged ok = function
  | Ok v -> ok v
  | Error e when busy e -> ()
  | Error e -> raise e

let shared_commands =
  [
    command "make"
      (Gen.unit @-> makes shared_t)
      (fun () -> { own = false; groups = false; threads = false; stored = 0 })
      (fun () ->
        (* Makes run in the sequential prefix, before either domain uses it. *)
        ignore (Lazy.force shared);
        Run.make ());
    command "groups"
      (shared_t ^-> judges unit)
      (fun r ->
        judged (fun () ->
            take_back r;
            r.groups <- true))
      groups;
    command "threads"
      (shared_t ^-> judges unit)
      (fun r ->
        judged (fun () ->
            take_back r;
            r.threads <- true))
      threads;
    command "store"
      (shared_t ^-> Gen.int_range 0 1000 @-> judges unit)
      (fun r v ->
        judged (fun () ->
            take_back r;
            r.stored <- v))
      store;
    command "grow"
      (shared_t ^-> Gen.int_range 0 1 @-> judges unit)
      (fun r _ -> judged (fun () -> r.own <- false))
      grow;
    command "fill"
      (shared_t ^-> judges int)
      (fun r outcome ->
        match outcome with
        | Ok v ->
            equal (triple bool bool bool)
              ~msg:"the run holds the fill's block and both geometry fields"
              (true, true, true)
              (r.own, r.groups, r.threads);
            equal int ~msg:"the stored parameter" r.stored v
        | Error e when busy e -> ()
        | Error (Invalid_argument _) when not (r.own && r.groups && r.threads)
          ->
            ()
        | Error e -> raise e)
      fill;
  ]

(* Growth the host refuses *)

(* A child process limits its address space to [room] bytes above what it
   maps, and stores into the last block of a submission whose blocks span 256
   MiB: the store raises Out_of_memory. With the limit lifted, the run takes
   another submission's stores and serves its submit. On Linux only: the limit
   bounds no allocation elsewhere. *)
let room = 64 lsl 20
let spanning = 1 lsl 16

let exe =
  let e = Sys.executable_name in
  if Filename.is_relative e then Filename.concat (Sys.getcwd ()) e else e

(* The bytes the process maps: the [VmSize] of its status. *)
let mapped () =
  let field l = Scanf.sscanf_opt l "VmSize: %d kB" (fun k -> k * 1024) in
  let lines =
    In_channel.with_open_text "/proc/self/status" In_channel.input_lines
  in
  match List.find_map field lines with
  | Some n -> n
  | None -> failwith "no VmSize in /proc/self/status"

let limit n =
  match Rig_support.set_address_space n with
  | 0 -> ()
  | e -> failwith (strf "setrlimit: errno %d" e)

let grow_child () =
  let d, _ = polled "grow" in
  let image = functions d in
  let wide = launch image "main" ~params:4096 in
  let big = Sub.make d (Array.make spanning wide) in
  let small =
    Sub.make d [| launch image "fill" ~params:16 ~refs:[| Sub.write 0 ~at:0 |] |]
  in
  let run = Run.make () in
  limit (mapped () + room);
  let refused =
    match Run.int64 run (Sub.block big (spanning - 1)) 0 1 with
    | () -> "stored"
    | exception Stdlib.Out_of_memory -> "Out_of_memory"
  in
  limit max_int;
  let b = Sub.block small 0 and out = B.create d 8 in
  one run b;
  Run.int64 run b 8 41;
  Rig.Point.wait (submit ~buffers:[| out |] small run);
  Printf.printf "%s\n%Ld\n" refused (String.get_int64_le (host_bytes out 8) 0)

let test_grow_refused () =
  if not (Sys.file_exists "/proc/self/status") then
    skip ~reason:"the limit bounds no allocation off Linux" ();
  let ic = Unix.open_process_args_in exe [| exe; "grow" |] in
  let out = In_channel.input_all ic in
  (match Unix.close_process_in ic with
  | WEXITED 0 -> ()
  | _ -> failf "the child failed: %s" out);
  equal (list string) [ "Out_of_memory"; "41" ]
    (String.split_on_char '\n' (String.trim out))

(* Cost *)

let words f =
  let before = Gc.minor_words () in
  f ();
  int_of_float (Gc.minor_words () -. before)

let test_allocation () =
  let d, _ = polled "words" in
  let image = functions d in
  let refs =
    Array.init 8 (fun k ->
        if k < 4 then Sub.read k ~at:(8 * k) else Sub.write k ~at:(8 * k))
  in
  let s = Sub.make d [| launch image "main" ~params:64 ~refs |] in
  let buffers = Array.init 8 (fun _ -> B.create d 8) in
  let run = Run.make () and b = Sub.block s 0 in
  one run b;
  let first = submit ~buffers s run in
  let n = 100 in
  equal int ~msg:"a submit of a launch" 0
    (words (fun () ->
         for _ = 1 to n do
           ignore (Sys.opaque_identity (Rig.submit s ~run ~buffers))
         done));
  let setter name f =
    equal int ~msg:name 0
      (words (fun () ->
           for i = 1 to n do
             f i
           done))
  in
  setter "groups" (fun i -> Run.groups run b i 1 1);
  setter "threads" (fun i -> Run.threads run b ((i land 7) + 1) 1 1);
  setter "shared" (fun i -> Run.shared run b i);
  setter "int32" (fun i -> Run.int32 run b 0 i);
  setter "int64" (fun i -> Run.int64 run b 8 i);
  setter "float32" (fun i -> Run.float32 run b 16 (float_of_int i *. 0.5));
  setter "float64" (fun i -> Run.float64 run b 24 (float_of_int i *. 0.5));
  let last = submit ~buffers s run in
  Rig.Point.wait first;
  Rig.Point.wait last

let tests =
  [
    group ~timeout "patching"
      [
        prop
          "a launch reads each ref's word as its buffer's address plus the \
           offset, and its other bytes as stored"
          gen_case patch_law;
      ];
    group ~timeout "order"
      [
        test "a read of a buffer waits for the launch that writes it"
          test_write_seen;
        test "a launch and a copy on another queue follow each other"
          test_device_order;
      ];
    group ~timeout "lifetime"
      [
        test "a submission keeps its launches' images after their array changed"
          test_parts_changed;
        test "a launch reads its refs once, whatever changes them after"
          test_parts_read_once;
      ];
    group ~timeout "refusals"
      [
        test "make refuses each launch its interface states" test_make_refusals;
        test "resolve raises what asking for a function raises"
          test_resolve_failures;
        test "submit refuses a run or a block the device cannot launch"
          test_submit_refusals;
        test "a setter refuses bytes outside its block" test_setter_refusals;
        test "an image states the facts its driver knows of a function"
          test_entry_facts;
        test "a setter refuses a run a submit holds" test_setter_in_use;
        test "a run holds the blocks of the submission last stored into"
          test_block_owner;
        test "a store whose growth the host refuses raises, the run usable"
          test_grow_refused;
      ];
    group ~timeout "domains"
      [
        stateful ~domains:2
          "stores and submits on two domains take a shared run or refuse it"
          shared_commands;
      ];
    group ~timeout "cost"
      [
        test "a launch's submit and its setters allocate nothing"
          test_allocation;
      ];
  ]

let () =
  match Sys.argv with
  | [| _; "grow" |] -> grow_child ()
  | _ -> exit (run "rig.launch" tests)
