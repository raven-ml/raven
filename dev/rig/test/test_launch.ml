(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module Run = Sub.Run
module P = Rig_support.Polled

let timeout = 60.
let strf = Printf.sprintf

let functions d =
  require_ok ~pp:Format.pp_print_string (Rig.Image.load d "functions")

let launch ?(queue = "COMPUTE:0") ?(after = [||]) ?(params = 0) ?(refs = [||])
    image kernel =
  { Sub.queue; after; work = Sub.Launch { image; kernel; params; refs } }

let one run b =
  Run.groups run b 1 1 1;
  Run.threads run b 1 1 1

let submit ?(reads = [||]) ?(writes = [||]) ?(waits = [||]) s run =
  Rig.submit s ~run ~reads ~writes ~waits

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

(* A run's buffer: [bytes] bytes, [first] bytes into a buffer of Polled's own
   memory, or into host memory that starts a page, borrowed on Polled. *)
type slot = { bytes : int; first : int; borrowed : bool }

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

type case = { slots : slot list; nreads : int; launches : launch_case list }

let pp_case ppf c =
  let pp_slot ppf s =
    Format.fprintf ppf "{%d bytes at %d%s}" s.bytes s.first
      (if s.borrowed then ", borrowed" else "")
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
  Format.fprintf ppf "slots [%a]; %d reads; launches [%a]"
    (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_slot)
    c.slots c.nreads
    (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_launch)
    c.launches

let gen_case =
  let open Gen in
  let* nslots = int_range 1 4 in
  let* nreads = int_range 0 nslots in
  let slot =
    let+ bytes = int_range 8 64
    and+ first = int_range 0 2
    and+ borrowed = bool in
    { bytes; first = 8 * first; borrowed }
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
  { slots; nreads; launches }

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
        (Int64.of_int (B.address buffers.(r.slot) + r.offset)))
    l.refs;
  Bytes.to_string p

let pp_launch ppf (l : P.launch) =
  let x, y, z = l.groups and tx, ty, tz = l.threads in
  Format.fprintf ppf "groups %dx%dx%d threads %dx%dx%d shared %d params %S" x y
    z tx ty tz l.shared l.params

let launch_t = Testable.make ~pp:pp_launch ~equal:( = )

let patch_law c =
  let d, p = Lazy.force patch_device in
  let image = functions d in
  let buffers = Array.of_list (List.map (make_slot d) c.slots) in
  let parts =
    Array.of_list
      (List.map
         (fun l ->
           let refs =
             Array.of_list
               (List.map (fun r -> { Sub.at = r.at; slot = r.slot }) l.refs)
           in
           launch image "main" ~params:(String.length l.params) ~refs)
         c.launches)
  in
  let nslots = Array.length buffers in
  let s = Sub.make ~reads:c.nreads ~writes:(nslots - c.nreads) d parts in
  let run = Run.make () in
  List.iteri (fun i l -> store run (Sub.block s i) l) c.launches;
  let reads = Array.sub buffers 0 c.nreads
  and writes = Array.sub buffers c.nreads (nslots - c.nreads) in
  ignore (P.launches p);
  let v = Rig.Point.value (submit ~reads ~writes s run) in
  (* Stores after the submit returned change nothing it handed over. *)
  List.iteri
    (fun i l ->
      let b = Sub.block s i in
      Run.groups run b 7 7 7;
      for k = 0 to (String.length l.params / 4) - 1 do
        Run.int32 run b (4 * k) 0x5a5a5a5a
      done)
    c.launches;
  Rig.wait d v;
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
  cover "no parameter bytes" (List.exists (fun l -> l.params = "") c.launches);
  cover "two launches, their blocks adjacent" (List.length c.launches = 2)

(* Ordering *)

(* A launch that writes [b] is what a read of [b] waits for. *)
let test_write_seen () =
  let d, _ = polled "write-seen" in
  let image = functions d in
  let out = B.create d 32 in
  let s =
    Sub.make ~reads:0 ~writes:1 d
      [| launch image "fill" ~params:16 ~refs:[| { at = 0; slot = 0 } |] |]
  in
  let run = Run.make () and b = Sub.block s 0 in
  Run.groups run b 2 2 1;
  Run.threads run b 1 1 1;
  Run.int64 run b 0 0;
  Run.int64 run b 8 100;
  ignore (submit ~writes:[| out |] s run);
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
    { Sub.queue = "COPY:0"; after = [||]; work = Sub.Copy { src; dst } }
  in
  let copy_into_b = Sub.make ~reads:0 ~writes:0 d [| copy_part ~dst:b src |] in
  let copying =
    Sub.make ~reads:1 ~writes:1 d
      [|
        launch image "copy" ~params:24
          ~refs:[| { at = 0; slot = 0 }; { at = 8; slot = 1 } |];
      |]
  in
  let run = Run.make () and blk = Sub.block copying 0 in
  one run blk;
  Run.int64 run blk 16 64;
  ignore (submit copy_into_b (Run.make ()));
  ignore (submit ~reads:[| b |] ~writes:[| c |] copying run);
  let out = B.create Rig.host 64 in
  B.copy ~src:c ~dst:out;
  equal string ~msg:"the launch read the copy" data (host_bytes out 64);
  (* A launch on COMPUTE:0 writes [b], then a copy on COPY:0 copies it. *)
  let filling =
    Sub.make ~reads:0 ~writes:1 d
      [| launch image "fill" ~params:16 ~refs:[| { at = 0; slot = 0 } |] |]
  in
  let run = Run.make () and blk = Sub.block filling 0 in
  Run.groups run blk 8 1 1;
  Run.threads run blk 1 1 1;
  Run.int64 run blk 0 0;
  Run.int64 run blk 8 7;
  ignore (submit ~writes:[| b |] filling run);
  let copy_b = Sub.make ~reads:0 ~writes:0 d [| copy_part ~dst:c b |] in
  ignore (submit copy_b (Run.make ()));
  B.copy ~src:c ~dst:out;
  equal string ~msg:"the copy read the launch"
    (String.concat "" (List.init 8 (fun i -> le64 (7 + i))))
    (host_bytes out 64)

(* Lifetime *)

(* A submission keeps its launches' images loaded after the caller's parts
   array changed, and its submit launches the function it was made with. *)
let test_parts_changed () =
  let d, _ = polled "parts-changed" in
  let collected = ref false in
  let load () =
    let image = functions d in
    Gc.finalise_last (fun () -> collected := true) image;
    image
  in
  let fill image =
    launch image "fill" ~params:16 ~refs:[| { at = 0; slot = 0 } |]
  in
  let parts = [| fill (load ()) |] in
  let s = Sub.make ~reads:0 ~writes:1 d parts in
  parts.(0) <- fill (functions d);
  Gc.full_major ();
  equal ~msg:"the part's image collected" bool false !collected;
  let out = B.create d 8 in
  let run = Run.make () and b = Sub.block s 0 in
  one run b;
  Run.int64 run b 0 0;
  Run.int64 run b 8 7;
  ignore (submit ~writes:[| out |] s run);
  let host = B.create Rig.host 8 in
  B.copy ~src:out ~dst:host;
  equal string (le64 7) (host_bytes host 8)

(* make reads a part's arrays once: refs changed inside make's call of the
   driver, after make checked them and before it compiled them, change
   nothing. *)
let test_parts_read_once () =
  let d, p = polled "read-once" in
  let image = functions d in
  let refs = [| { Sub.at = 0; slot = 0 } |] in
  P.before p "entry" (fun () -> refs.(0) <- { at = 8; slot = 0 });
  let s =
    Sub.make ~reads:0 ~writes:1 d [| launch image "main" ~params:16 ~refs |]
  in
  let out = B.create d 8 in
  let run = Run.make () and b = Sub.block s 0 in
  one run b;
  Run.int64 run b 0 0;
  Run.int64 run b 8 7;
  ignore (P.launches p);
  ignore (submit ~writes:[| out |] s run);
  ignore (P.run p);
  equal ~msg:"the ref make checked" (list string)
    [ le64 (B.address out) ^ le64 7 ]
    (List.map (fun (l : P.launch) -> l.params) (P.launches p))

(* Refusals *)

let refused ~msg f = raises_match ~msg Exn.invalid_arg f

let test_make_refusals () =
  let d, _ = polled "make" and other, _ = polled "make-other" in
  let image = functions d in
  let make ?(image = image) ?(kernel = "main") ?(params = 16) refs =
    Sub.make ~reads:1 ~writes:1 d [| launch image kernel ~params ~refs |]
  in
  let at at slot = { Sub.at; slot } in
  refused ~msg:"an image of another device" (fun () ->
      make ~image:(functions other) [||]);
  refused ~msg:"no such function" (fun () -> make ~kernel:"absent" [||]);
  refused ~msg:"4097 parameter bytes" (fun () -> make ~params:4097 [||]);
  refused ~msg:"negative parameter bytes" (fun () -> make ~params:(-1) [||]);
  refused ~msg:"a ref at 4" (fun () -> make [| at 4 0 |]);
  refused ~msg:"a ref ending past the parameters" (fun () -> make [| at 16 0 |]);
  refused ~msg:"a ref at -8" (fun () -> make [| at (-8) 0 |]);
  refused ~msg:"a ref to slot 2 of 2" (fun () -> make [| at 0 2 |]);
  refused ~msg:"a ref to slot -1" (fun () -> make [| at 0 (-1) |]);
  refused ~msg:"two refs at 8" (fun () -> make [| at 8 0; at 8 1 |]);
  ignore (make ~params:4096 [| at 0 0; at 4088 1 |]);
  let s = make [||] in
  equal int ~msg:"no value assigned" 0 (Rig.submitted d);
  refused ~msg:"the block of part 1 of 1" (fun () -> Sub.block s 1);
  refused ~msg:"the block of part -1" (fun () -> Sub.block s (-1));
  let copy =
    {
      Sub.queue = "COPY:0";
      after = [||];
      work = Sub.Copy { src = B.create d 8; dst = B.create d 8 };
    }
  in
  let s = Sub.make ~reads:0 ~writes:0 d [| copy |] in
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
    Sub.make ~reads:1 ~writes:0 d
      [| launch image "main" ~params:8 ~refs:[| { at = 0; slot = 0 } |] |]
  in
  let b = Sub.block s 0 in
  let reads = [| B.create ~memory:Pinned d 8 |] in
  refused_submit ~msg:"a run no setter stored into" d (fun () ->
      submit ~reads s (Run.make ()));
  let geometry ?(groups = (1, 1, 1)) ?(threads = (1, 1, 1)) ?(shared = 0) () =
    let run = Run.make () in
    let x, y, z = groups and tx, ty, tz = threads in
    Run.groups run b x y z;
    Run.threads run b tx ty tz;
    Run.shared run b shared;
    run
  in
  refused_submit ~msg:"a ref to memory with no address" d (fun () ->
      submit ~reads:[| B.create d 8 |] s (geometry ()));
  refused_submit ~msg:"no groups along y" d (fun () ->
      submit ~reads s (geometry ~groups:(1, 0, 1) ()));
  refused_submit ~msg:"no threads along x" d (fun () ->
      submit ~reads s (geometry ~threads:(0, 1, 1) ()));
  refused_submit ~msg:"1025 threads" d (fun () ->
      submit ~reads s (geometry ~threads:(5, 205, 1) ()));
  refused_submit ~msg:"65536 groups along z" d (fun () ->
      submit ~reads s (geometry ~groups:(1, 1, 65536) ()));
  refused_submit ~msg:"shared memory past 48 KiB" d (fun () ->
      submit ~reads s (geometry ~shared:49153 ()));
  let v = submit ~reads s (geometry ~threads:(1024, 1, 1) ~shared:49152 ()) in
  Rig.Point.wait v

let test_setter_refusals () =
  let d, _ = polled "setters" in
  let image = functions d in
  let s = Sub.make ~reads:0 ~writes:0 d [| launch image "main" ~params:12 |] in
  let run = Run.make () and b = Sub.block s 0 in
  refused ~msg:"int32 at -1" (fun () -> Run.int32 run b (-1) 0);
  refused ~msg:"int32 at 9 of 12" (fun () -> Run.int32 run b 9 0);
  refused ~msg:"int64 at 8 of 12" (fun () -> Run.int64 run b 8 0);
  refused ~msg:"float64 at max_int" (fun () -> Run.float64 run b max_int 0.);
  refused ~msg:"float32 at 12 of 12" (fun () -> Run.float32 run b 12 0.);
  refused ~msg:"negative groups" (fun () -> Run.groups run b 1 (-1) 1);
  refused ~msg:"2^32 threads" (fun () -> Run.threads run b (1 lsl 32) 1 1);
  refused ~msg:"negative shared memory" (fun () -> Run.shared run b (-1));
  Run.int32 run b 8 0;
  Run.int64 run b 4 0;
  Run.float32 run b 8 0.;
  Run.groups run b 0xffff_ffff 1 1

(* A setter called from a signal handler while a submit holds the run, waiting
   for a producer, raises; the submit then completes with what was stored
   before. *)
let test_setter_in_use () =
  let producer, pp = polled "in-use-producer" in
  let d, p = polled "in-use" in
  let image = functions d in
  let s = Sub.make ~reads:0 ~writes:0 d [| launch image "main" ~params:8 |] in
  let run = Run.make () and b = Sub.block s 0 in
  one run b;
  Run.int64 run b 0 41;
  let point =
    submit (Sub.make ~reads:0 ~writes:0 producer [||]) (Run.make ())
  in
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

(* Cost *)

let words f =
  let before = Gc.minor_words () in
  f ();
  int_of_float (Gc.minor_words () -. before)

let test_allocation () =
  let d, _ = polled "words" in
  let image = functions d in
  let refs = Array.init 8 (fun k -> { Sub.at = 8 * k; slot = k }) in
  let s =
    Sub.make ~reads:4 ~writes:4 d [| launch image "main" ~params:64 ~refs |]
  in
  let buffers = Array.init 8 (fun _ -> B.create d 8) in
  let reads = Array.sub buffers 0 4 and writes = Array.sub buffers 4 4 in
  let run = Run.make () and b = Sub.block s 0 in
  one run b;
  ignore (submit ~reads ~writes s run);
  let n = 100 in
  equal int ~msg:"a submit of a launch" 0
    (words (fun () ->
         for _ = 1 to n do
           ignore (Sys.opaque_identity (submit ~reads ~writes s run))
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
  Rig.wait d (Rig.submitted d)

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
        test "make reads its parts' arrays once, whatever changes them during it"
          test_parts_read_once;
      ];
    group ~timeout "refusals"
      [
        test "make refuses each launch its interface states" test_make_refusals;
        test "submit refuses a run or a block the device cannot launch"
          test_submit_refusals;
        test "a setter refuses bytes outside its block" test_setter_refusals;
        test "a setter refuses a run a submit holds" test_setter_in_use;
      ];
    group ~timeout "cost"
      [
        test "a launch's submit and its setters allocate nothing"
          test_allocation;
      ];
  ]

let () = exit (run "rig.launch" tests)
