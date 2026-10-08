(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module C = Rig
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module Support = Rig_support

let timeout = 60.
let memory name = require_ok ~pp:Format.pp_print_string (C.memory_device name)
let lost = function C.Lost _ -> true | _ -> false
let count call p = List.length (List.filter (( = ) call) (P.log p))

let bytes b =
  let ba = B.bigarray Bigarray.char b in
  String.init (Bigarray.Array1.dim ba) (Bigarray.Array1.get ba)

let filled n c =
  let b = B.create C.host n in
  Bigarray.Array1.fill (B.bigarray Bigarray.char b) c;
  b

(* Host buffers *)

let test_create () =
  let b = B.create C.host 40 in
  equal int 40 (B.length b);
  equal bool false (B.is_borrowed b);
  equal bool true (B.spans b);
  let big = B.create C.host (1 lsl 20) in
  equal int 0 (B.address big mod 4096)

let test_refusals () =
  raises_match Exn.invalid_arg (fun () -> B.create C.host (-1));
  let b = B.create C.host 16 in
  raises_match Exn.invalid_arg (fun () -> B.view b ~first:12 ~length:8);
  raises_match Exn.invalid_arg (fun () -> B.view b ~first:(-1) ~length:1)

let test_views () =
  let b = filled 8 'a' in
  let v = B.view b ~first:4 ~length:4 in
  Bigarray.Array1.fill (B.bigarray Bigarray.char v) 'b';
  equal string "aaaabbbb" (bytes b);
  equal bool false (B.spans v);
  equal bool true (B.overlaps b v);
  equal bool false (B.overlaps (B.view b ~first:0 ~length:4) v)

(* Copies *)

let test_copy_host () =
  let src = filled 100 'x' and dst = filled 100 'y' in
  B.copy ~src ~dst;
  equal string (String.make 100 'x') (bytes dst);
  raises_match Exn.invalid_arg (fun () -> B.copy ~src ~dst:(filled 99 'y'));
  raises_match Exn.invalid_arg (fun () ->
      B.copy
        ~src:(B.view src ~first:0 ~length:50)
        ~dst:(B.view src ~first:10 ~length:50))

let test_copy_device () =
  let d = memory "buffer:copy" in
  let on = B.create d 64 in
  B.copy ~src:(filled 64 'q') ~dst:on;
  let back = filled 64 'z' in
  B.copy ~src:on ~dst:back;
  equal string (String.make 64 'q') (bytes back)

(* A copy on a Polled device runs only when a wait reaches its sleep: the copy's
   own wait for its point. *)
let test_copy_waits () =
  let d, p = P.open_ "buffer:copy-polled" in
  let on = B.create d 4096 in
  B.copy ~src:(filled 4096 'w') ~dst:on;
  let back = filled 4096 '0' in
  B.copy ~src:on ~dst:back;
  equal string (String.make 4096 'w') (bytes back);
  equal int 0 (P.queued p)

(* Borrows *)

let test_borrow () =
  let d = memory "buffer:borrow" in
  let h = filled 32 'h' in
  let b = require_some (B.borrow d h) in
  equal bool true (B.is_borrowed b);
  equal bool true (B.overlaps b h);
  equal bool true (C.equal d (B.device b));
  equal int (B.address h) (B.address b)

(* A host buffer a device maps is mapped once, whatever its borrows. *)
let test_borrow_maps_once () =
  let d, p = P.open_ "buffer:map-once" in
  let h = B.create C.host (1 lsl 16) in
  for _ = 1 to 3 do
    ignore (Sys.opaque_identity (B.borrow d h))
  done;
  equal int 1 (List.length (List.filter (( = ) "map_host") (P.log p)))

let test_borrow_small () =
  let d, _ = P.open_ "buffer:small" in
  let paged = B.create C.host (1 lsl 16) in
  let off_page = Bigarray.Array1.sub (B.bigarray Bigarray.char paged) 8 16 in
  is_none (B.borrow d (B.of_bigarray off_page))

(* Waits *)

let test_wait () =
  let d, p = P.open_ "buffer:wait" in
  let n = 1 lsl 16 in
  let h = filled n 'a' and on = B.create d n in
  let part =
    {
      C.Submission.queue = "COPY:0";
      after = [||];
      work = C.Submission.Copy { src = require_some (B.borrow d h); dst = on };
    }
  in
  ignore (C.submit (C.Submission.make ~reads:0 ~writes:0 ~waits:0 d [| part |]));
  equal int 1 (P.queued p);
  B.wait on B.Read;
  equal int 0 (P.queued p)

(* Each [(name, f)] raises an exception [pred] accepts. *)
let cases_of pred l =
  List.iter (fun (name, f) -> raises_match ~msg:name pred f) l

let test_dead () =
  let b = B.create C.host 8 in
  C.Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
      ignore (C.Claim.consume c ~why:"donated" b));
  raises_match (Exn.invalid_arg ~substring:"donated") (fun () ->
      B.wait b B.Read)

(* Every function that reaches a dead buffer's bytes refuses it with the
   consumption's reason; [release] accepts it (claim suite). *)
let test_dead_refused () =
  let d, _ = P.open_ "buffer:dead" in
  let consumed b =
    C.Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
        ignore (C.Claim.consume c ~why:"donated" b))
  in
  let h = B.create C.host (1 lsl 16) and m = B.create d 64 in
  consumed h;
  consumed m;
  let dead = Exn.invalid_arg ~substring:"donated" in
  let other = B.create C.host (1 lsl 16) in
  cases_of dead
    [
      ("copy from", fun () -> B.copy ~src:h ~dst:other);
      ("copy to", fun () -> B.copy ~src:other ~dst:h);
      ("address", fun () -> ignore (B.address h));
      ("handle", fun () -> ignore (B.handle m));
      ("view", fun () -> ignore (B.view h ~first:0 ~length:8));
      ("bigarray", fun () -> ignore (B.bigarray Bigarray.char h));
      ("borrow", fun () -> ignore (B.borrow d h));
      ( "consume",
        fun () ->
          C.Claim.with_ ~read:[ h ] ~donate:[] (fun c ->
              ignore (C.Claim.consume c ~why:"again" h)) );
    ]

(* A view starts its offset into its buffer's memory, at its address plus that
   offset, in its driver's object. *)
let test_offset () =
  let d, _ = P.open_ "buffer:offset" in
  let b = B.create d 64 in
  let v = B.view b ~first:16 ~length:8 in
  equal int (B.offset b + 16) (B.offset v);
  equal int (B.address b + 16) (B.address v);
  equal nativeint (B.handle b) (B.handle v);
  raises_match Exn.invalid_arg (fun () -> B.handle (B.create C.host 8))

(* A borrow collected while its memory lives keeps its mapping: a borrow made
   again maps nothing. *)
let test_borrow_remade () =
  let d, p = P.open_ "buffer:remade" in
  let h = B.create C.host (1 lsl 16) in
  ignore (Sys.opaque_identity (B.borrow d h));
  Gc.full_major ();
  Gc.full_major ();
  ignore (B.create ~memory:Pinned d 8);
  ignore (require_some (B.borrow d h));
  equal int 1 (List.length (List.filter (( = ) "map_host") (P.log p)))

let test_lost_memory () =
  let d, p = P.open_ "buffer:lost" in
  let on = B.create d 8 in
  let s = C.Submission.make ~reads:0 ~writes:1 ~waits:0 d [||] in
  C.Submission.write s 0 on;
  P.fail p;
  raises_match lost (fun () -> C.submit s);
  raises_match lost (fun () -> C.Claim.read on);
  raises_match lost (fun () -> B.wait on B.Read)

(* Words *)

(* The fewest minor and major words of ten calls of [f], after a warm-up call,
   so that a finaliser running inside one call cannot inflate the count. *)
let words f =
  ignore (Sys.opaque_identity (f ()));
  let minor = ref max_int and major = ref max_int in
  for _ = 1 to 10 do
    let m0, _, j0 = Gc.counters () in
    ignore (Sys.opaque_identity (f ()));
    let m1, _, j1 = Gc.counters () in
    minor := Int.min !minor (int_of_float (m1 -. m0));
    major := Int.min !major (int_of_float (j1 -. j0))
  done;
  (!minor, !major)

(* Host buffers cost at most these words, [(minor, major)]: programs pin their
   own allocation counts on them. *)
let at_most_words (minor, major) f =
  let m, j = words f in
  at_most ~msg:"minor words" int ~than:minor m;
  at_most ~msg:"major words" int ~than:major j

let f32 = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 16
let b16 = B.create C.host 64

(* Once every point is reached, a wait allocates nothing. *)
let test_wait_words () =
  let d, _ = P.open_ "buffer:wait-words" in
  let b = B.create d 64 in
  let s = Sub.make ~reads:0 ~writes:1 ~waits:0 d [||] in
  Sub.write s 0 b;
  ignore (C.submit s);
  B.wait b B.Read_write;
  let before = Gc.minor_words () in
  for _ = 1 to 100 do
    B.wait b B.Read_write
  done;
  equal int 0 (int_of_float (Gc.minor_words () -. before) / 100)

let test_create_words () = at_most_words (61, 0) (fun () -> B.create C.host 64)

let test_create_large_words () =
  at_most_words (54, 7) (fun () -> B.create C.host (1 lsl 20))

let test_create_empty_words () =
  at_most_words (32, 0) (fun () -> B.create C.host 0)

(* Borrows of memory that dies *)

(* Memory of another machine's driver device copies only directly, by the
   source's device: with this machine's memory, its own machine's io memory, a
   device the source does not map, or on a device that runs no copy, the copy
   raises. *)
let test_copy_machines () =
  let far = Support.machine "elsewhere" in
  let gpu ?copies name =
    require_ok ~pp:Format.pp_print_string
      (C.open_
         (module P)
         ~machine:"elsewhere" ~name
         (fun () -> Ok (P.make ?copies ~host_visible:false ~peers:false ())))
  in
  let g = gpu "buffer:far-gpu" and g' = gpu "buffer:far-other" in
  let still = gpu ~copies:false "buffer:far-still" in
  List.iter
    (fun (msg, src, dst) ->
      raises_match ~msg Exn.invalid_arg (fun () -> B.copy ~src ~dst))
    [
      ("from this machine's host", B.create C.host 64, B.create g 64);
      ("into its machine's host", B.create g 64, B.create far 64);
      ("from its machine's host", B.create far 64, B.create g 64);
      ("into a device it does not map", B.create g 64, B.create g' 64);
      ("within a device that runs no copy", B.create still 64, B.create still 64);
    ]

(* An empty buffer of a driver's device names no memory: its address and its
   handle are 0. *)
let test_empty_address () =
  let d, _ = P.open_ "buffer:empty" in
  let b = B.create d 0 in
  equal ~msg:"address" int 0 (B.address b);
  equal ~msg:"handle" nativeint 0n (B.handle b)

let test_borrow_own () =
  let d, _ = P.open_ "buffer:own" in
  let b = B.create d 64 in
  match B.borrow d b with
  | Some b' -> equal bool true (b == b')
  | None -> failf "no borrow of its own memory"

(* A mapping lasts while its memory lives, and is released with it once the work
   the mapper was handed until then is done. *)
let test_mapping_released () =
  let d, p = P.open_ "buffer:released" in
  let read_borrow () =
    let h = B.create C.host (1 lsl 16) in
    let s = Sub.make ~reads:1 ~writes:0 ~waits:0 d [||] in
    Sub.read s 0 (require_some (B.borrow d h));
    ignore (C.submit s)
  in
  read_borrow ();
  let unmaps () = count "unmap" p in
  let drain () =
    Gc.full_major ();
    Gc.full_major ();
    ignore (B.create C.host 8);
    ignore (B.create ~memory:Pinned d 8)
  in
  drain ();
  equal ~msg:"while its work is unrun" int 0 (unmaps ());
  ignore (P.run p);
  drain ();
  equal ~msg:"once it ran" int 1 (unmaps ())

(* A borrow of a borrow maps the memory under it. *)
let test_borrow_of_borrow () =
  let d, _ = P.open_ "buffer:first" in
  let e, pe = P.open_ "buffer:second" in
  let h = B.create C.host (1 lsl 16) in
  let b = require_some (B.borrow e (require_some (B.borrow d h))) in
  equal (list int) [ 1 lsl 16 ] (P.host_maps pe);
  equal int (B.address h) (B.address b);
  equal bool true (B.overlaps b h)

let test_borrow_peer () =
  let d, _ = P.open_ ~host_visible:false "buffer:owner" in
  let e, pe = P.open_ ~host_visible:false "buffer:peer" in
  let m = B.create d 64 in
  let b = require_some (B.borrow e m) in
  equal int 1 (count "map_peer" pe);
  equal bool true (C.equal e (B.device b));
  equal bool true (B.overlaps b m)

let test_borrow_spans () =
  let d, _ = P.open_ "buffer:spans" in
  let h = B.create C.host (1 lsl 17) in
  equal bool true (B.spans (require_some (B.borrow d h)));
  let part = B.view h ~first:0 ~length:(1 lsl 16) in
  equal bool false (B.spans (require_some (B.borrow d part)))

let test_overlaps () =
  let ba = Bigarray.Array1.create Bigarray.char Bigarray.c_layout 16 in
  let whole = B.of_bigarray ba in
  let part = B.of_bigarray (Bigarray.Array1.sub ba 8 8) in
  equal bool true (B.overlaps whole part);
  equal bool false (B.overlaps whole (B.view whole ~first:4 ~length:0))

(* Routes of copies *)

let roundtrip ~src ~dst =
  Bigarray.Array1.fill (B.bigarray Bigarray.char src) 'r';
  B.copy ~src ~dst;
  let back = filled (B.length src) '0' in
  B.copy ~src:dst ~dst:back;
  equal string (String.make (B.length src) 'r') (bytes back)

(* Between host memory and memory the host does not address, the device copies:
   from a mapping of host memory that starts on a page. *)
let test_copy_mapped_host () =
  let d, p = P.open_ ~host_visible:false "buffer:direct" in
  let src = B.create C.host (1 lsl 16) in
  let dst = B.create d (1 lsl 16) in
  roundtrip ~src ~dst;
  equal ~msg:"copies the device ran" int 2 (P.submits p);
  equal ~msg:"host memory mapped" (list int)
    [ 1 lsl 16; 1 lsl 16 ]
    (List.filter (( = ) (1 lsl 16)) (P.host_maps p))

let staging = 32 lsl 20

(* Host memory and a device's memory that the host does not address, filled from
   host memory the device maps. *)
let on d c =
  let b = B.create d 64 in
  B.copy ~src:(B.view (filled (1 lsl 16) c) ~first:0 ~length:64) ~dst:b;
  b

let contents b =
  let back = B.create C.host (1 lsl 16) in
  let back = B.view back ~first:0 ~length:(B.length b) in
  B.copy ~src:b ~dst:back;
  bytes back

(* Host memory off a page goes through the host's staging memory, whose halves
   of 32 MiB the device maps once. *)
let test_copy_staged () =
  let d, p = P.open_ ~host_visible:false "buffer:staged" in
  let paged = B.create C.host (1 lsl 16) in
  let off_page = Bigarray.Array1.sub (B.bigarray Bigarray.char paged) 8 64 in
  Bigarray.Array1.fill off_page 's';
  let dst = B.create d 64 in
  B.copy ~src:(B.of_bigarray off_page) ~dst;
  equal string (String.make 64 's') (contents dst);
  equal ~msg:"staging slots mapped" int 1
    (List.length (List.filter (( = ) staging) (P.host_maps p)))

(* Between devices that map none of each other's memory, the bytes go through
   the staging memory. *)
let test_copy_unmapped () =
  let d, pd = P.open_ ~host_visible:false ~peers:false "buffer:unmapped-src" in
  let e, pe = P.open_ ~host_visible:false ~peers:false "buffer:unmapped-dst" in
  let src = on d 'u' and dst = B.create e 64 in
  B.copy ~src ~dst;
  equal string (String.make 64 'u') (contents dst);
  let slots p = List.length (List.filter (( = ) staging) (P.host_maps p)) in
  equal ~msg:"slots each device mapped" (pair int int) (1, 1)
    (slots pd, slots pe)

(* Between two devices of one driver, the source's device copies, mapping the
   destination. *)
let test_copy_peer () =
  let d, pd = P.open_ ~host_visible:false "buffer:peer-src" in
  let e, pe = P.open_ ~host_visible:false "buffer:peer-dst" in
  let src = on d 'p' and dst = B.create e 64 in
  let before = (P.submits pd, P.submits pe) in
  B.copy ~src ~dst;
  equal (pair int int) (fst before + 1, snd before) (P.submits pd, P.submits pe);
  equal int 1 (count "map_peer" pd);
  equal string (String.make 64 'p') (contents dst)

(* A device that runs no copy has memory the host addresses: the host copies. *)
let test_copy_no_queue () =
  let d, p = P.open_ ~copies:false "buffer:no-queue" in
  roundtrip ~src:(filled 64 'x') ~dst:(B.create d 64);
  equal int 0 (P.submits p)

(* A device lost while it used the staging memory leaves other devices' copies
   through it working. *)
(* A driver's device's borrow of host memory is host memory: the host copies
   into and out of it, on a device that runs no copy too. *)
let test_copy_borrowed_host () =
  let d, _ = P.open_ ~copies:false "buffer:borrowed-host" in
  let h = filled (1 lsl 16) 'h' in
  let b = require_some (B.borrow d h) in
  let out = B.create C.host (1 lsl 16) in
  B.copy ~src:b ~dst:out;
  equal ~msg:"out of the borrow" string (String.make (1 lsl 16) 'h') (bytes out);
  B.copy ~src:(filled (1 lsl 16) 'i') ~dst:b;
  equal ~msg:"into the borrow" string (String.make (1 lsl 16) 'i') (bytes h)

(* A copy larger than a staging slot goes through its halves in turn: every byte
   lands where it belongs, across the halves' edges and a half's reuse. *)
let test_copy_staged_large () =
  let open_ name = P.open_ ~host_visible:false ~peers:false name in
  let d, _ = open_ "buffer:staged-large-src" in
  let e, _ = open_ "buffer:staged-large-dst" in
  let n = (2 * staging) + 4096 in
  let byte i = Char.unsafe_chr ((i + ((i lsr 16) * 13)) land 255) in
  let h = B.create C.host n in
  let ba = B.bigarray Bigarray.char h in
  for i = 0 to n - 1 do
    Bigarray.Array1.unsafe_set ba i (byte i)
  done;
  let src = B.create d n and dst = B.create e n in
  B.copy ~src:h ~dst:src;
  B.copy ~src ~dst;
  let back = B.create C.host n in
  B.copy ~src:dst ~dst:back;
  let got = B.bigarray Bigarray.char back in
  let at = [ 0; 4095; staging - 1; staging; staging + 1; 2 * staging; n - 1 ] in
  equal (list char) (List.map byte at) (List.map (Bigarray.Array1.get got) at);
  let wrong = ref 0 in
  for i = 0 to n - 1 do
    if Bigarray.Array1.unsafe_get got i <> byte i then incr wrong
  done;
  equal ~msg:"bytes that differ" int 0 !wrong

(* The two staging slots are taken in turn: two copies waiting on a device hold
   both, a third waits for one, and every copy lands once the device runs. *)
let test_staging_turns () =
  let open_ name = P.open_ ~host_visible:false ~peers:false name in
  let d, pd = open_ "buffer:turns-src" in
  let e, _ = open_ "buffer:turns-dst" in
  let srcs = List.map (on d) [ 'a'; 'b'; 'c' ] in
  let dsts = List.map (fun _ -> B.create e 64) srcs in
  P.gate pd;
  let copier src dst = Thread.create (fun () -> B.copy ~src ~dst) () in
  let first =
    List.map2 copier
      [ List.nth srcs 0; List.nth srcs 1 ]
      [ List.nth dsts 0; List.nth dsts 1 ]
  in
  Support.await "two copies in the slots" (fun () -> P.sleepers pd = 2);
  let waiting = Support.waiting () in
  let third = copier (List.nth srcs 2) (List.nth dsts 2) in
  Support.await "the third copy waiting for a slot" (fun () ->
      Support.waiting () > waiting);
  equal ~msg:"copies waiting on the device" int 2 (P.sleepers pd);
  P.open_gate pd;
  List.iter Thread.join (third :: first);
  equal (list string)
    [ String.make 64 'a'; String.make 64 'b'; String.make 64 'c' ]
    (List.map contents dsts)

let test_staging_after_loss () =
  let open_ name = P.open_ ~host_visible:false ~peers:false name in
  let d, p = open_ "buffer:staging-lost" in
  let e, _ = open_ "buffer:staging-dst" in
  let f, _ = open_ "buffer:staging-src" in
  let src = on d 'l' and dst = B.create e 64 in
  P.fail p;
  raises_match lost (fun () -> B.copy ~src ~dst);
  B.copy ~src:(on f 'k') ~dst;
  equal string (String.make 64 'k') (contents dst)

(* Bigarrays *)

type kind = Kind : string * ('a, 'b) Bigarray.kind * (int -> 'a) -> kind

let kinds =
  Bigarray.
    [
      Kind ("float16", float16, float_of_int);
      Kind ("float32", float32, float_of_int);
      Kind ("float64", float64, float_of_int);
      Kind ("int8_signed", int8_signed, Fun.id);
      Kind ("int8_unsigned", int8_unsigned, Fun.id);
      Kind ("char", char, Char.chr);
      Kind ("int16_signed", int16_signed, Fun.id);
      Kind ("int16_unsigned", int16_unsigned, Fun.id);
      Kind ("int32", int32, Int32.of_int);
      Kind ("int64", int64, Int64.of_int);
      Kind ("int", int, Fun.id);
      Kind ("nativeint", nativeint, Nativeint.of_int);
      Kind
        ( "complex32",
          complex32,
          fun i -> { Complex.re = float_of_int i; im = 0. } );
      Kind
        ( "complex64",
          complex64,
          fun i -> { Complex.re = float_of_int i; im = 0. } );
    ]

(* A bigarray's buffer is its bytes, and [bigarray] reads them back over the
   same memory. *)
let test_kind (Kind (_, k, v)) =
  List.iter
    (fun n ->
      let ba = Bigarray.Array1.create k Bigarray.c_layout n in
      let b = B.of_bigarray ba in
      equal ~msg:"borrowed" bool true (B.is_borrowed b);
      equal ~msg:"bytes" int (n * Bigarray.kind_size_in_bytes k) (B.length b);
      let back = B.bigarray k b in
      for i = 0 to n - 1 do
        ba.{i} <- v i
      done;
      for i = 0 to n - 1 do
        if back.{i} <> v i then failf "element %d of %d differs" i n
      done)
    [ 0; 1; 17 ]

let test_bigarray_refusals () =
  let b = B.create C.host 8 in
  raises_match Exn.invalid_arg (fun () ->
      B.bigarray Bigarray.float32 (B.view b ~first:0 ~length:6));
  raises_match Exn.invalid_arg (fun () ->
      B.bigarray Bigarray.float32 (B.view b ~first:2 ~length:4));
  (* A complex element starts at a multiple of one component's size. *)
  let c = B.create C.host 32 in
  equal int 1
    (Bigarray.Array1.dim
       (B.bigarray Bigarray.complex64 (B.view c ~first:8 ~length:16)));
  raises_match Exn.invalid_arg (fun () ->
      B.bigarray Bigarray.complex64 (B.view c ~first:4 ~length:16));
  let d, _ = P.open_ "buffer:not-host" in
  raises_match Exn.invalid_arg (fun () ->
      B.bigarray Bigarray.char (B.create d 8))

(* A bigarray of a buffer keeps its memory once the buffer is collected. *)
let test_bigarray_keeps () =
  let n = 1 lsl 17 in
  let view =
    (fun () ->
      let ba = B.bigarray Bigarray.char (B.create C.host n) in
      Bigarray.Array1.fill ba 'v';
      ba)
      ()
  in
  Gc.full_major ();
  Gc.full_major ();
  Bigarray.Array1.fill (B.bigarray Bigarray.char (B.create C.host n)) 'w';
  equal char 'v' view.{n - 1}

(* A pinned buffer on [d] filled with [c], seen through a borrow on the host. *)
let viewed d n c =
  let m = B.create ~memory:Pinned d n in
  let view = B.bigarray Bigarray.char (require_some (B.borrow C.host m)) in
  Bigarray.Array1.fill view c;
  (B.address m, view)

(* Collects, then drains [d], as an allocation on it does first. *)
let collect d =
  Gc.full_major ();
  Gc.full_major ();
  ignore (Sys.opaque_identity (B.create ~memory:Pinned d 8))

(* A bigarray over a borrow on the host keeps the device memory it reads, as
   does an array made from it: no buffer reuses that memory while one is
   reachable. Collected, it returns to its device's cache. *)
let test_bigarray_keeps_borrowed () =
  let d, _ = P.open_ "buffer:viewed" in
  let n = 4096 in
  let at, view = viewed d n 'v' in
  let view = ref (Some view) in
  let sub = ref (Some (Bigarray.Array1.sub (Option.get !view) 0 8)) in
  view := None;
  collect d;
  let other = B.create ~memory:Pinned d n in
  Bigarray.Array1.fill
    (B.bigarray Bigarray.char (require_some (B.borrow C.host other)))
    'w';
  equal ~msg:"another buffer's memory" bool false (B.address other = at);
  equal ~msg:"read through the array" char 'v' (Option.get !sub).{7};
  sub := None;
  collect d;
  let again = B.create ~memory:Pinned d n in
  equal ~msg:"once no array reads it" int at (B.address again);
  ignore (Sys.opaque_identity other)

(* The core holds a share of the storage of every bigarray over a borrow: once
   collected, the views dropped while its memory lives leave that share and free
   nothing. *)
let test_bigarray_shares () =
  let d, _ = P.open_ "buffer:shared" in
  let m = B.create ~memory:Pinned d 4096 in
  let on_host = require_some (B.borrow C.host m) in
  let collected_shares () =
    Gc.full_major ();
    Gc.full_major ();
    let view = B.bigarray Bigarray.char on_host in
    Gc.full_major ();
    (view, Rig_support.shares view)
  in
  let first () =
    let view, alone = collected_shares () in
    Bigarray.Array1.fill view 's';
    let sub = Bigarray.Array1.sub view 0 8 in
    [ alone; Rig_support.shares sub ]
  in
  equal ~msg:"the core and a view, then its sub" (list int) [ 2; 3 ] (first ());
  let view, alone = collected_shares () in
  equal ~msg:"the core and a new view" int 2 alone;
  equal ~msg:"the bytes" char 's' view.{4095};
  ignore (Sys.opaque_identity m)

let tests =
  [
    group ~timeout "host buffers"
      [
        test "a buffer holds its elements' bytes" test_create;
        test "a buffer refuses bounds it cannot hold" test_refusals;
        test "a view shares its buffer's memory" test_views;
      ];
    group ~timeout "copies"
      [
        test "a copy between host buffers moves their bytes" test_copy_host;
        test "a copy to and from a memory device moves the bytes"
          test_copy_device;
        test "a copy returns once the device's work ran" test_copy_waits;
      ];
    group ~timeout "borrows"
      [
        test "a borrow is the memory it maps" test_borrow;
        test "a memory a device borrows is mapped once" test_borrow_maps_once;
        test "a borrow collected and made again maps nothing" test_borrow_remade;
        test "a view lies its offset into its buffer's memory" test_offset;
        test "host memory off a page does not borrow on a driver's device"
          test_borrow_small;
        test "a copy between machines with no device to copy raises"
          test_copy_machines;
        test "an empty buffer of a driver's device has address and handle 0"
          test_empty_address;
        test "a borrow on its own device is the buffer" test_borrow_own;
        test "a mapping is released once its memory died and its work ran"
          test_mapping_released;
        test "a borrow of a borrow maps the memory under it"
          test_borrow_of_borrow;
        test "a device borrows another device's memory of its driver"
          test_borrow_peer;
        test "a borrow of all of a memory spans it, of part of it does not"
          test_borrow_spans;
        test "bigarrays over the same bytes overlap, no bytes overlap nothing"
          test_overlaps;
      ];
    group ~timeout "copy routes"
      [
        test "a device copies from host memory it maps" test_copy_mapped_host;
        test "host memory off a page is copied through the staging memory"
          test_copy_staged;
        test
          "devices that map none of each other's memory copy through the \
           staging memory"
          test_copy_unmapped;
        test "the source's device copies to a device of its driver"
          test_copy_peer;
        test "the host copies for a device that runs no copy" test_copy_no_queue;
        test "the host copies into and out of a device's borrow of host memory"
          test_copy_borrowed_host;
        test "a copy larger than a staging slot lands every byte"
          test_copy_staged_large;
        test "the staging slots are taken in turn" test_staging_turns;
        test "a loss with the staging memory leaves other copies working"
          test_staging_after_loss;
      ];
    cases ~timeout
      ~name:(fun (Kind (n, _, _)) -> n)
      "a bigarray's buffer is its bytes" kinds test_kind;
    group ~timeout "bigarrays"
      [
        test "a bigarray refuses kinds, sizes and alignments it cannot read"
          test_bigarray_refusals;
        test "a bigarray keeps its memory once its buffer is collected"
          test_bigarray_keeps;
        test "a bigarray over a borrow keeps the device memory it reads"
          test_bigarray_keeps_borrowed;
        test "a bigarray's storage keeps the core's share" test_bigarray_shares;
      ];
    group ~timeout "waits"
      [
        test "a wait returns once the work that wrote the memory ran" test_wait;
        test "a dead buffer's wait raises its reason" test_dead;
        test "every function reaching a dead buffer's bytes refuses it"
          test_dead_refused;
        test "memory a lost device wrote raises Lost" test_lost_memory;
      ];
  ]

let words =
  group ~timeout "words"
    [
      test "a host buffer of 64 bytes costs at most 61 words" test_create_words;
      test "a wait whose points are reached allocates nothing" test_wait_words;
      test "a host buffer of 1 MiB costs at most 54 and 7 major words"
        test_create_large_words;
      test "a host buffer of no bytes costs at most 32 words"
        test_create_empty_words;
      test "a bigarray's buffer costs at most 54 words" (fun () ->
          at_most_words (54, 0) (fun () -> B.of_bigarray f32));
      test "a view costs at most 19 words" (fun () ->
          at_most_words (19, 0) (fun () -> B.view b16 ~first:16 ~length:16));
      test "a buffer's bigarray costs at most 27 words" (fun () ->
          at_most_words (27, 0) (fun () -> B.bigarray Bigarray.float32 b16));
    ]

let () = exit (run "rig.buffer" (tests @ [ words ]))
