(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module Support = Rig_support

let submit ?(buffers = [||]) ?waits s =
  Rig.submit ?waits s ~run:(Sub.Run.make ()) ~buffers

let timeout = 60.
let memory name = Rig.memory_device name
let lost = function Rig.Lost _ -> true | _ -> false
let count call p = List.length (List.filter (( = ) call) (P.log p))

let bytes b =
  B.read b Bigarray.char (fun ba ->
      String.init (Bigarray.Array1.dim ba) (Bigarray.Array1.get ba))

let filled n c =
  let b = B.create Rig.host n in
  B.write b Bigarray.char (fun a -> Bigarray.Array1.fill a c);
  b

(* A device that maps no host memory, whose copies to and from the host go
   through its own staging memory. *)
let staging_device name = P.open_ ~maps_host:false ~host_visible:false name

(* Host buffers *)

let test_create () =
  let b = B.create Rig.host 40 in
  equal int 40 (B.length b);
  equal bool false (B.is_borrowed b);
  equal bool true (B.spans b);
  let big = B.create Rig.host (1 lsl 20) in
  equal bool true (B.aligned big ~at:0 4096)

let test_refusals () =
  raises_match Exn.invalid_arg (fun () -> B.create Rig.host (-1));
  let b = B.create Rig.host 16 in
  raises_match Exn.invalid_arg (fun () -> B.view b ~first:12 ~length:8);
  raises_match Exn.invalid_arg (fun () -> B.view b ~first:(-1) ~length:1)

let test_views () =
  let b = filled 8 'a' in
  let v = B.view b ~first:4 ~length:4 in
  B.write v Bigarray.char (fun a -> Bigarray.Array1.fill a 'b');
  equal string "aaaabbbb" (bytes b);
  equal bool false (B.spans v);
  equal bool true (B.overlaps b v);
  equal bool false (B.overlaps (B.view b ~first:0 ~length:4) v)

(* A host buffer of 64 KiB or more starts on a page: an address [at] bytes
   past a view into it, inside the view or outside it, lies at a multiple of a
   power of two up to the page iff its offset from the page is one. *)
let aligned_case =
  let open Gen in
  let+ first = int_range 0 4096
  and+ at = int_range (-4096) 12288
  and+ k = int_range 0 12 in
  (first, at, k)

let aligned_case =
  Gen.with_pp
    (fun ppf (first, at, k) ->
      Format.fprintf ppf "view from %d, byte %d, 2^%d" first at k)
    aligned_case

let aligned_law (first, at, k) =
  let b = B.view (B.create Rig.host (1 lsl 16)) ~first ~length:8192 in
  let n = 1 lsl k in
  cover "aligned" ((first + at) mod n = 0);
  cover "not aligned" ((first + at) mod n <> 0);
  cover "outside the view" (at < 0 || at >= 8192);
  equal bool ((first + at) mod n = 0) (B.aligned b ~at n)

let test_aligned_refusals () =
  let b = B.create Rig.host 8 in
  raises_match ~msg:"0" Exn.invalid_arg (fun () -> B.aligned b ~at:0 0);
  raises_match ~msg:"3" Exn.invalid_arg (fun () -> B.aligned b ~at:0 3);
  raises_match ~msg:"io memory" Exn.invalid_arg (fun () ->
      B.aligned (B.create (Support.io "buffer:aligned-io") 8) ~at:0 16)

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
  equal bool true (Rig.equal d (B.device b));
  equal int (Support.Reader.host h) (Support.Reader.host b)

(* A host buffer a device maps is mapped once, whatever its borrows. *)
let test_borrow_maps_once () =
  let d, p = P.open_ "buffer:map-once" in
  let h = B.create Rig.host (1 lsl 16) in
  for _ = 1 to 3 do
    ignore (Sys.opaque_identity (B.borrow d h))
  done;
  equal int 1 (List.length (List.filter (( = ) "map_host") (P.log p)))

(* Two buffers that are not [==], made fresh, and whether they are on one
   device. *)
type devices = One_device | Two_devices

let distinct =
  let two f = (f (), f ()) in
  let viewed () =
    let b = B.create Rig.host 16 in
    (B.view b ~first:0 ~length:8, B.view b ~first:0 ~length:8)
  in
  let borrowed () =
    let h = B.create Rig.host 16 in
    (h, require_some (B.borrow (memory "buffer:compare-borrow") h))
  in
  let bigarray () =
    let a = Bigarray.Array1.create Bigarray.char Bigarray.c_layout 4 in
    Bigarray.Array1.fill a 'b';
    B.of_bigarray a
  in
  [
    ("two host buffers", One_device, fun () ->
        two (fun () -> B.create Rig.host 16));
    ("two empty host buffers", One_device, fun () ->
        two (fun () -> B.create Rig.host 0));
    ("a buffer and its view", One_device, fun () ->
        let b = B.create Rig.host 16 in
        (b, B.view b ~first:0 ~length:16));
    ("two views of one window", One_device, viewed);
    ("two buffers of a device", One_device, fun () ->
        let d = memory "buffer:compare" in
        two (fun () -> B.create d 64));
    ("buffers of two devices", Two_devices, fun () ->
        (B.create Rig.host 64, B.create (memory "buffer:compare-other") 64));
    ("a buffer and its borrow", Two_devices, borrowed);
    ("two strings' buffers", One_device, fun () ->
        two (fun () -> B.of_string "abc"));
    ("two bigarrays' buffers", One_device, fun () -> two bigarray);
  ]

let test_polymorphic (_, devices, make) =
  let a, b = make () in
  let sign x = Int.compare x 0 in
  equal ~msg:"compare is an order" int
    (sign (compare a b))
    (-sign (compare b a));
  equal ~msg:"compare of lists" int (sign (compare a b))
    (sign (compare [ a ] [ b ]));
  raises_match ~msg:"= on itself" Exn.invalid_arg (fun () -> a = a);
  match devices with
  | One_device -> raises_match ~msg:"=" Exn.invalid_arg (fun () -> a = b)
  | Two_devices -> equal ~msg:"=" bool false (a = b)

let test_borrow_small () =
  let d, _ = P.open_ "buffer:small" in
  let paged = B.create Rig.host (1 lsl 16) in
  let off_page = Bigarray.Array1.sub (B.export paged Bigarray.char) 8 16 in
  is_none (B.borrow d (B.of_bigarray off_page))

(* Waits *)

let test_wait () =
  let d, p = P.open_ "buffer:wait" in
  let n = 1 lsl 16 in
  let h = filled n 'a' and on = B.create d n in
  let part =
    let src = require_some (B.borrow d h) in
    Rig.Submission.(copy ~src:(Buffer src) ~dst:(Buffer on))
  in
  ignore (submit (Rig.Submission.make d [| part |]));
  equal int 1 (P.queued p);
  B.wait on Rig.Read;
  equal int 0 (P.queued p)

(* Host access follows the device's work on the memory with no wait by hand: a
   read follows the queued copy that writes it, and a write follows the queued
   copy that reads it. *)
let test_access_orders () =
  let d, p = P.open_ "buffer:access-orders" in
  let n = 1 lsl 16 in
  let h = filled n 'a' and src = B.create d n and dst = B.create d n in
  B.write src Bigarray.char (fun a -> Bigarray.Array1.fill a 'x');
  let on_d = require_some (B.borrow d h) in
  let copy ~src ~dst =
    Rig.Submission.(make d [| copy ~src:(Buffer src) ~dst:(Buffer dst) |])
  in
  ignore (submit (copy ~src ~dst:on_d));
  equal ~msg:"the write is queued" int 1 (P.queued p);
  equal ~msg:"a read sees it" string (String.make n 'x') (bytes h);
  ignore (submit (copy ~src:on_d ~dst));
  equal ~msg:"the read is queued" int 1 (P.queued p);
  B.write h Bigarray.char (fun a -> Bigarray.Array1.fill a 'w');
  equal ~msg:"the copy read the bytes before the write" string
    (String.make n 'x') (bytes dst);
  equal ~msg:"the host's write" string (String.make n 'w') (bytes h)

(* An export waits for the memory's work and puts it outside the claims: it is
   never donated again. Only host memory exports. *)
let test_export () =
  let d, p = P.open_ "buffer:export" in
  let n = 1 lsl 16 in
  let h = filled n 'a' and src = B.create d n in
  B.write src Bigarray.char (fun a -> Bigarray.Array1.fill a 'x');
  let dst = require_some (B.borrow d h) in
  let copy = Rig.Submission.(copy ~src:(Buffer src) ~dst:(Buffer dst)) in
  ignore (submit (Rig.Submission.make d [| copy |]));
  let a = B.export h Bigarray.char in
  equal ~msg:"the work ran" int 0 (P.queued p);
  equal ~msg:"its bytes" char 'x' a.{n - 1};
  is_none ~msg:"never donated" (Rig.Claim.take ~why:"donated" h);
  raises_match (Exn.invalid_arg ~substring:"not the host's") (fun () ->
      ignore (B.export src Bigarray.char));
  raises_match (Exn.invalid_arg ~substring:"not the host's") (fun () ->
      ignore (B.export (require_some (B.borrow Rig.host src)) Bigarray.char))

(* Each [(name, f)] raises an exception [pred] accepts. *)
let cases_of pred l =
  List.iter (fun (name, f) -> raises_match ~msg:name pred f) l

let test_dead () =
  let b = B.create Rig.host 8 in
  Rig.Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
      ignore (Rig.Claim.consume c ~why:"donated" b));
  raises_match (Exn.invalid_arg ~substring:"donated") (fun () ->
      B.wait b Rig.Read)

(* Every function that reaches a dead buffer's bytes refuses it with the
   consumption's reason; [release] accepts it (claim suite). *)
let test_dead_refused () =
  let d, _ = P.open_ "buffer:dead" in
  let consumed b =
    Rig.Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
        ignore (Rig.Claim.consume c ~why:"donated" b))
  in
  let h = B.create Rig.host (1 lsl 16) and m = B.create d 64 in
  consumed h;
  consumed m;
  let dead = Exn.invalid_arg ~substring:"donated" in
  let other = B.create Rig.host (1 lsl 16) in
  cases_of dead
    [
      ("copy from", fun () -> B.copy ~src:h ~dst:other);
      ("copy to", fun () -> B.copy ~src:other ~dst:h);
      ("aligned", fun () -> ignore (B.aligned h ~at:0 8));
      ("handle", fun () -> ignore (B.handle m));
      ("view", fun () -> ignore (B.view h ~first:0 ~length:8));
      ("read", fun () -> B.read h Bigarray.char ignore);
      ("write", fun () -> B.write h Bigarray.char ignore);
      ("export", fun () -> ignore (B.export h Bigarray.char));
      ("blit_from_string", fun () -> B.blit_from_string "s" 0 h 0 1);
      ("blit_to_bytes", fun () -> B.blit_to_bytes h 0 (Bytes.create 1) 0 1);
      ("fill", fun () -> B.fill h 'f');
      ("borrow", fun () -> ignore (B.borrow d h));
      ( "consume",
        fun () ->
          Rig.Claim.with_ ~read:[ h ] ~donate:[] (fun c ->
              ignore (Rig.Claim.consume c ~why:"again" h)) );
    ]

(* A view starts its offset into its buffer's memory, at its address plus that
   offset, in its driver's object. *)
let test_offset () =
  let d, _ = P.open_ "buffer:offset" in
  let b = B.create d 64 in
  let v = B.view b ~first:16 ~length:8 in
  equal int (B.offset b + 16) (B.offset v);
  let s = Sub.make ~fixed:[ (b, Rig.Read) ] d [||] in
  equal int (Sub.address s b + 16) (Sub.address s v);
  equal nativeint (B.handle b) (B.handle v);
  raises_match Exn.invalid_arg (fun () -> B.handle (B.create Rig.host 8))

(* A borrow collected while its memory lives keeps its mapping: a borrow made
   again maps nothing. *)
let test_borrow_remade () =
  let d, p = P.open_ "buffer:remade" in
  let h = B.create Rig.host (1 lsl 16) in
  ignore (Sys.opaque_identity (B.borrow d h));
  Gc.full_major ();
  Gc.full_major ();
  ignore (B.create ~memory:Pinned d 8);
  ignore (require_some (B.borrow d h));
  equal int 1 (List.length (List.filter (( = ) "map_host") (P.log p)))

let test_lost_memory () =
  let d, p = P.open_ "buffer:lost" in
  let on = B.create d 8 in
  let s = Rig.Submission.make d [| Support.touching [| Rig.Submission.write |] |] in
  P.fail p;
  raises_match lost (fun () -> submit s ~buffers:[| on |]);
  raises_match lost (fun () -> B.read on Bigarray.char ignore);
  raises_match lost (fun () -> B.wait on Rig.Read)

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
let b16 = B.create Rig.host 64

(* Once every point is reached, a wait allocates nothing. *)
let test_wait_words () =
  let d, _ = P.open_ "buffer:wait-words" in
  let b = B.create d 64 in
  let s = Sub.make d [| Support.touching [| Sub.write |] |] in
  ignore (submit s ~buffers:[| b |]);
  B.wait b Rig.Read_write;
  let before = Gc.minor_words () in
  for _ = 1 to 100 do
    B.wait b Rig.Read_write
  done;
  equal int 0 (int_of_float (Gc.minor_words () -. before) / 100)

(* Behind a transport, where the host reads a device's word through its
   driver, a wait for reached work allocates nothing: a buffer's wait, and a
   submit that waits for the point. *)
let test_wait_words_transport () =
  let d, _ = P.open_ ~transport:true "buffer:wait-words-transport" in
  let b = B.create d 64 in
  let s = Sub.make d [| Support.touching [| Sub.write |] |] in
  let p = submit s ~buffers:[| b |] in
  B.wait b Rig.Read_write;
  let waiter = Sub.make (memory "buffer:wait-words-waiter") [||] in
  let run = Sub.Run.make () in
  (* Passed as an option made once: a supplied optional argument may allocate
     its [Some] at each call. *)
  let waits = Some [| p |] in
  let wait () = B.wait b Rig.Read_write in
  let submit () =
    ignore (Rig.submit waiter ?waits ~run ~buffers:[||] : Rig.Point.t)
  in
  let per_call f =
    f ();
    let before = Gc.minor_words () in
    for _ = 1 to 100 do
      f ()
    done;
    int_of_float (Gc.minor_words () -. before) / 100
  in
  equal ~msg:"a buffer's wait" int 0 (per_call wait);
  equal ~msg:"a submit's wait" int 0 (per_call submit)

let test_create_words () =
  at_most_words (41, 0) (fun () -> B.create Rig.host 64)

let test_create_large_words () =
  at_most_words (32, 9) (fun () -> B.create Rig.host (1 lsl 20))

let test_create_empty_words () =
  at_most_words (32, 0) (fun () -> B.create Rig.host 0)

(* Borrows of memory that dies *)

(* Memory of another machine's driver device copies only directly: with this
   process's memory by its own device, with its machine's memory by the source's
   device. Into a device the source does not map, or on a device that runs no
   copy, the copy raises. *)
let test_copy_machines () =
  let far = Support.machine "elsewhere" in
  let gpu ?copies name =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_ ~machine:far
         {
           name;
           driver = (module P);
           make =
             (fun () -> Ok (P.make ?copies ~host_visible:false ~peers:false ()));
         })
  in
  let g = gpu "buffer:far-gpu" and g' = gpu "buffer:far-other" in
  let still = gpu ~copies:false "buffer:far-still" in
  List.iter
    (fun (msg, src, dst) ->
      raises_match ~msg Exn.invalid_arg (fun () -> B.copy ~src ~dst))
    [
      ("into its machine's host", B.create g 64, B.create far 64);
      ("from its machine's host", B.create far 64, B.create g 64);
      ("into a device it does not map", B.create g 64, B.create g' 64);
      ("within a device that runs no copy", B.create still 64, B.create still 64);
      ( "from this process's memory on a device that runs no copy",
        B.create Rig.host 64,
        B.create still 64 );
    ]

(* A device of another machine copies this process's memory as it is, in one
   copy each way. *)
let test_copy_across () =
  let machine = Support.machine "across" in
  let g, p =
    let p = P.make ~host_visible:false ~peers:false () in
    ( require_ok ~pp:Format.pp_print_string
        (Rig.open_ ~machine
           {
             name = "buffer:across";
             driver = (module P);
             make = (fun () -> Ok p);
           }),
      p )
  in
  let src = B.of_string "across machines" in
  let far = B.create g (B.length src) in
  let back = B.create Rig.host (B.length src) in
  B.copy ~src ~dst:far;
  B.copy ~src:far ~dst:back;
  equal string "across machines"
    (B.read back Bigarray.char (fun ba ->
       String.init (B.length back) (Bigarray.Array1.get ba)));
  let within = B.create g (B.length src) in
  B.copy ~src:far ~dst:within;
  equal ~msg:"no host maps" (list int) [] (P.host_maps p);
  equal ~msg:"local sides"
    (list
       (Testable.structural ~pp:(fun ppf s ->
            Format.pp_print_string ppf
              (match s with `None -> "none" | `Src -> "src" | `Dst -> "dst"))))
    [ `Src; `Dst; `None ] (P.copy_sides p)

(* An empty buffer of a driver's device names no memory: its address and its
   handle are 0. *)
let test_empty_address () =
  let d, _ = P.open_ "buffer:empty" in
  let b = B.create d 0 in
  let s = Sub.make ~fixed:[ (b, Rig.Read) ] d [||] in
  equal ~msg:"address" int 0 (Sub.address s b);
  equal ~msg:"handle" nativeint 0n (B.handle b)

(* Polled that notes each host mapping rig asks of it off a 4 KiB boundary,
   which is off every page. *)
let off_page = ref []

module Paged = struct
  include P

  let map_host d p n =
    if p mod 4096 <> 0 then off_page := p :: !off_page;
    P.map_host d p n
end

(* rig asks a driver to map host memory only from the start of a page: memory
   of a memory device that starts elsewhere is none that a driver's device
   borrows. *)
let test_map_host_paged () =
  let d =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_
         {
           name = "buffer:paged";
           driver = (module Paged);
           make = (fun () -> Ok (P.make ()));
         })
  in
  let m = memory "buffer:paged-memory" in
  let rec off () =
    let b = B.create m 64 in
    if Support.at b mod 4096 <> 0 then b else off ()
  in
  let b = off () in
  equal ~msg:"reaches" bool true (Rig.reaches d m);
  equal ~msg:"borrow" bool true (Option.is_none (B.borrow d b));
  equal ~msg:"mappings off a page" (list int) [] !off_page

(* A device that maps no host memory reaches none: it shares none with the host,
   and a borrow of host memory that starts on a page is [None], with no mapping
   asked of its driver. *)
let test_maps_no_host () =
  let d, p = P.open_ ~maps_host:false "buffer:no-host-maps" in
  let h = B.create Rig.host (64 * 1024) in
  equal ~msg:"reaches" bool false (Rig.reaches d Rig.host);
  equal ~msg:"shares" bool false (Rig.shares_host_memory d);
  equal ~msg:"borrow" bool true (Option.is_none (B.borrow d h));
  equal ~msg:"maps" (list int) [] (P.host_maps p)

(* A device that maps no host memory copies host bytes into its own memory and
   back, its own [Pinned] memory standing in for the host's staging, and maps no
   host memory for it. *)
let test_staged_no_host () =
  let d, p = staging_device "buffer:staged-no-host" in
  let m = B.create d 64 in
  let into = B.create Rig.host 64 and back = B.create Rig.host 64 in
  B.write into Bigarray.char (fun a -> Bigarray.Array1.fill a 's');
  B.copy ~src:into ~dst:m;
  B.copy ~src:m ~dst:back;
  equal ~msg:"the bytes back" string (String.make 64 's')
    (B.read back Bigarray.char (fun ba ->
       String.init 64 (Bigarray.Array1.get ba)));
  equal ~msg:"host maps" (list int) [] (P.host_maps p)

(* Empty host memory that starts on a page borrows on a driver's device with no
   mapping: the borrow names no memory. *)
let test_empty_borrow () =
  let d, p = P.open_ "buffer:empty-borrow" in
  let paged = B.export (B.create Rig.host (64 * 1024)) Bigarray.char in
  let empty = B.of_bigarray (Bigarray.Array1.sub paged 0 0) in
  let b = require_some (B.borrow d empty) in
  let s = Sub.make ~fixed:[ (b, Rig.Read) ] d [||] in
  equal ~msg:"address" int 0 (Sub.address s b);
  equal ~msg:"maps" (list int) [] (P.host_maps p)

(* A borrow, on a device that runs no copy, of a peer's memory the host does not
   address copies by the memory's own device: in and out. *)
let test_copyless_borrow () =
  let owner, _ = P.open_ ~host_visible:false "buffer:copyless-owner" in
  let still, _ = P.open_ ~copies:false "buffer:copyless" in
  let m = B.create owner 64 in
  let on_still = require_some (B.borrow still m) in
  let into = B.create Rig.host 64 and back = B.create Rig.host 64 in
  B.write into Bigarray.char (fun a -> Bigarray.Array1.fill a 'c');
  B.copy ~src:into ~dst:on_still;
  B.copy ~src:on_still ~dst:back;
  equal string (String.make 64 'c')
    (B.read back Bigarray.char (fun ba ->
       String.init 64 (Bigarray.Array1.get ba)))

(* A borrow, on a device that maps no host memory, of a peer's memory the host
   does not address copies in and out, staged through the borrowing device's own
   memory. *)
let test_mapless_borrow () =
  let owner, _ = P.open_ ~host_visible:false "buffer:mapless-owner" in
  let mapless, _ = P.open_ ~maps_host:false "buffer:mapless" in
  let m = B.create owner 1 in
  let on_mapless = require_some (B.borrow mapless m) in
  let into = B.create Rig.host 1 and back = B.create Rig.host 1 in
  B.write into Bigarray.char (fun a -> Bigarray.Array1.fill a 'c');
  B.copy ~src:into ~dst:on_mapless;
  B.copy ~src:on_mapless ~dst:back;
  equal char 'c' (B.read back Bigarray.char (fun a -> Bigarray.Array1.get a 0))

(* A borrow of a lost device's buffer raises its loss, empty or not. *)
let test_borrow_lost () =
  let d, p = P.open_ "buffer:borrow-lost" in
  let e, _ = P.open_ "buffer:borrow-lost-on" in
  let m = B.create d 64 and empty = B.create d 0 in
  P.fail p;
  (try ignore (submit (Sub.make d [||])) with Rig.Lost _ -> ());
  raises_match ~msg:"64 bytes" lost (fun () -> B.borrow e m);
  raises_match ~msg:"empty" lost (fun () -> B.borrow e empty)

(* A device that maps no host memory maps a peer's memory the host addresses,
   through its driver's peer mapping. *)
let test_peer_no_host () =
  let owner, _ = P.open_ "buffer:peer-visible" in
  let d, p = P.open_ ~maps_host:false "buffer:peer-no-host" in
  let m = B.create owner 64 in
  equal ~msg:"borrowed" bool true (Option.is_some (B.borrow d m));
  equal ~msg:"host maps" (list int) [] (P.host_maps p)

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
    let h = B.create Rig.host (1 lsl 16) in
    let s = Sub.make d [| Support.touching [| Sub.read |] |] in
    ignore (submit s ~buffers:[| require_some (B.borrow d h) |])
  in
  read_borrow ();
  let unmaps () = count "unmap" p in
  let drain () =
    Gc.full_major ();
    Gc.full_major ();
    ignore (B.create Rig.host 8);
    ignore (B.create ~memory:Pinned d 8)
  in
  drain ();
  equal ~msg:"while its work is unrun" int 0 (unmaps ());
  ignore (P.run p);
  drain ();
  equal ~msg:"once it ran" int 1 (unmaps ())

(* Borrowed bytes return to their keeper only once every device's mapping of
   them is released: a device whose unmap waits for its other work keeps them
   from the collector until it ran. *)
let test_bytes_after_unmap () =
  let d, p = P.open_ "buffer:bytes-after-unmap" in
  let collected = ref false in
  (fun () ->
    let paged = B.export (B.create Rig.host (1 lsl 16)) Bigarray.char in
    let ba = Bigarray.Array1.sub paged 0 (1 lsl 16) in
    Gc.finalise (fun _ -> collected := true) ba;
    let h = B.of_bigarray ba in
    let s = Sub.make d [| Support.touching [| Sub.read |] |] in
    ignore (submit s ~buffers:[| require_some (B.borrow d h) |]);
    ignore (P.run p))
    ();
  ignore (submit (Sub.make d [||]));
  let drain () =
    Gc.full_major ();
    Gc.full_major ();
    ignore (B.create Rig.host 8);
    ignore (B.create ~memory:Pinned d 8);
    Gc.full_major ();
    Gc.full_major ()
  in
  drain ();
  equal ~msg:"while d's other work is unrun: unmaps, bytes collected"
    (pair int bool) (0, false)
    (count "unmap" p, !collected);
  ignore (P.run p);
  drain ();
  equal ~msg:"once it ran" (pair int bool) (1, true)
    (count "unmap" p, !collected)

(* A borrow of a borrow maps the memory under it. *)
let test_borrow_of_borrow () =
  let d, _ = P.open_ "buffer:first" in
  let e, pe = P.open_ "buffer:second" in
  let h = B.create Rig.host (1 lsl 16) in
  let b = require_some (B.borrow e (require_some (B.borrow d h))) in
  equal (list int) [ 1 lsl 16 ] (P.host_maps pe);
  equal int (Support.Reader.host h) (Support.Reader.host b);
  equal bool true (B.overlaps b h)

let test_borrow_peer () =
  let d, _ = P.open_ ~host_visible:false "buffer:owner" in
  let e, pe = P.open_ ~host_visible:false "buffer:peer" in
  let m = B.create d 64 in
  let b = require_some (B.borrow e m) in
  equal int 1 (count "map_peer" pe);
  equal bool true (Rig.equal e (B.device b));
  equal bool true (B.overlaps b m)

let test_borrow_spans () =
  let d, _ = P.open_ "buffer:spans" in
  let h = B.create Rig.host (1 lsl 17) in
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
  B.write src Bigarray.char (fun a -> Bigarray.Array1.fill a 'r');
  B.copy ~src ~dst;
  let back = filled (B.length src) '0' in
  B.copy ~src:dst ~dst:back;
  equal string (String.make (B.length src) 'r') (bytes back)

(* Between host memory and memory the host does not address, the device copies:
   from a mapping of host memory that starts on a page. *)
let test_copy_mapped_host () =
  let d, p = P.open_ ~host_visible:false "buffer:direct" in
  let src = B.create Rig.host (1 lsl 16) in
  let dst = B.create d (1 lsl 16) in
  roundtrip ~src ~dst;
  equal ~msg:"copies the device ran" int 2 (P.submits p);
  equal ~msg:"host memory mapped" (list int)
    [ 1 lsl 16; 1 lsl 16 ]
    (List.filter (( = ) (1 lsl 16)) (P.host_maps p))

(* Host memory and a device's memory that the host does not address, filled from
   host memory the device maps. *)
let on d c =
  let b = B.create d 64 in
  B.copy ~src:(B.view (filled (1 lsl 16) c) ~first:0 ~length:64) ~dst:b;
  b

let contents b =
  let back = B.create Rig.host (1 lsl 16) in
  let back = B.view back ~first:0 ~length:(B.length b) in
  B.copy ~src:b ~dst:back;
  bytes back

(* Host memory off a page goes through the host's staging memory, which the
   device maps once: a second copy maps nothing. *)
let test_copy_staged () =
  let d, p = P.open_ ~host_visible:false "buffer:staged" in
  let paged = B.create Rig.host (1 lsl 16) in
  let off_page = Bigarray.Array1.sub (B.export paged Bigarray.char) 8 64 in
  Bigarray.Array1.fill off_page 's';
  let dst = B.create d 64 in
  B.copy ~src:(B.of_bigarray off_page) ~dst;
  equal string (String.make 64 's') (contents dst);
  let maps = List.length (P.host_maps p) in
  B.copy ~src:(B.of_bigarray off_page) ~dst;
  equal ~msg:"maps of a second copy" int maps (List.length (P.host_maps p))

(* Between devices that map none of each other's memory, the bytes go through
   the staging memory, which each device maps once. *)
let test_copy_unmapped () =
  let d, pd = P.open_ ~host_visible:false ~peers:false "buffer:unmapped-src" in
  let e, pe = P.open_ ~host_visible:false ~peers:false "buffer:unmapped-dst" in
  let src = on d 'u' and dst = B.create e 64 in
  B.copy ~src ~dst;
  equal string (String.make 64 'u') (contents dst);
  let maps () = (List.length (P.host_maps pd), List.length (P.host_maps pe)) in
  let before = maps () in
  B.copy ~src ~dst;
  equal ~msg:"maps of a second copy" (pair int int) before (maps ())

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
  let out = B.create Rig.host (1 lsl 16) in
  B.copy ~src:b ~dst:out;
  equal ~msg:"out of the borrow" string (String.make (1 lsl 16) 'h') (bytes out);
  B.copy ~src:(filled (1 lsl 16) 'i') ~dst:b;
  equal ~msg:"into the borrow" string (String.make (1 lsl 16) 'i') (bytes h)

(* A copy larger than the staging memory goes through it in pieces: every byte
   lands where it belongs. *)
let test_copy_staged_large () =
  let open_ name = P.open_ ~host_visible:false ~peers:false name in
  let d, _ = open_ "buffer:staged-large-src" in
  let e, _ = open_ "buffer:staged-large-dst" in
  let n = (3 * (64 lsl 20)) + 4096 in
  let byte i =
    Char.unsafe_chr ((i + ((i lsr 16) * 13) + (i lsr 24)) land 255)
  in
  let h = B.create Rig.host n in
  B.write h Bigarray.char (fun ba ->
      for i = 0 to n - 1 do
        Bigarray.Array1.unsafe_set ba i (byte i)
      done);
  let src = B.create d n and dst = B.create e n in
  B.copy ~src:h ~dst:src;
  B.copy ~src ~dst;
  let back = B.create Rig.host n in
  B.copy ~src:dst ~dst:back;
  let wrong =
    B.read back Bigarray.char (fun got ->
        let wrong = ref 0 in
        for i = 0 to n - 1 do
          if Bigarray.Array1.unsafe_get got i <> byte i then incr wrong
        done;
        !wrong)
  in
  equal ~msg:"bytes that differ" int 0 wrong

(* Copies through the staging memory take it in turn: copies waiting on a device
   hold it, and every copy lands once the device runs. *)
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
  Support.await "a copy waiting" (fun () -> P.sleepers pd >= 1);
  let third = copier (List.nth srcs 2) (List.nth dsts 2) in
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

(* An io device whose memory reads succeed [reads] times, then raise
   [Sys_error], as a file cut short does. *)
module Cut = struct
  type t = int Atomic.t
  type region = unit

  exception Fault of string

  let region_key : region Type.Id.t = Type.Id.make ()
  let budget _ = max_int
  let alloc _ _ = Some ()
  let free _ () = ()

  let read reads () ~at:_ ~dst:_ ~len:_ =
    if Atomic.fetch_and_add reads (-1) <= 0 then raise (Sys_error "cut short")

  let write _ () ~at:_ ~src:_ ~len:_ = ()
  let pages _ () = None
  let prefetch _ () ~at:_ ~len:_ = ()
  let stop _ = ()
end

let cuts = Atomic.make 0

let cut reads =
  let name = Printf.sprintf "buffer:cut-%d" (Atomic.fetch_and_add cuts 1) in
  match
    Rig.open_io
      { name; io = (module Cut); make = (fun () -> Ok (Atomic.make reads)) }
  with
  | Ok d -> d
  | Error e -> failwith e

(* Windtrap's [raises] lets [Sys.Break] through, as it ends a run. *)
let breaks f =
  Sys.catch_break true;
  Fun.protect
    ~finally:(fun () -> Sys.catch_break false)
    (fun () -> match f () with _ -> false | exception Sys.Break -> true)

(* A copy through the staging memory gives its slot back on every exit. A copy
   of two pieces from an io device to a device the host does not address queues
   the first piece's leg out of its slot, then its second read fails; the wait
   for that leg is interrupted. After two such copies, as many as the staging
   memory has slots, a third copy runs. Each broken copy released its claim on
   its destination. *)
let test_staging_gives_back () =
  let d, p = P.open_ ~host_visible:false "buffer:staging-gives-back" in
  let n = (32 * 1024 * 1024) + 8 in
  let copy reads dst = B.copy ~src:(B.create (cut reads) n) ~dst in
  for i = 1 to 2 do
    P.interrupt p;
    let dst = B.create d n in
    equal
      ~msg:(Printf.sprintf "copy %d broke" i)
      bool true
      (breaks (fun () -> copy 1 dst));
    is_some
      ~msg:(Printf.sprintf "copy %d's destination donated after it" i)
      (Rig.Claim.take ~why:"donated" dst)
  done;
  let copied = Atomic.make false in
  let third =
    Domain.spawn (fun () ->
        copy 2 (B.create d n);
        Atomic.set copied true)
  in
  Support.await "the third copy" (fun () -> Atomic.get copied);
  Domain.join third

(* A transfer through staging memory holds a read claim on its buffers until it
   returns, so a donation on another domain meanwhile gets a read claim only.
   [donate_during p b transfer donor] runs [transfer] on another domain until
   it waits at [p]'s gate for a leg, with legs still to run, then donates [b]:
   a donation that holds [b]'s memory exclusive gives [donor] the consumed
   buffer to write in place. It is that buffer if the donation held the
   memory, and how [transfer] ended. *)
let donate_during p b transfer donor =
  P.gate p;
  let ended =
    Domain.spawn (fun () ->
        match transfer () with
        | () -> "returned"
        | exception e -> Printexc.to_string e)
  in
  Support.await "the transfer waiting for a leg" (fun () -> P.sleepers p > 0);
  let donated = Rig.Claim.take ~why:"donated" b in
  Option.iter donor donated;
  P.open_gate p;
  (donated, Domain.join ended)

(* Three pieces of staging memory, whose halves hold 32 MiB each: the host's
   leg of the third reads the source after the device ran the first's leg out
   of the same half. *)
let three_pieces = (2 * (32 lsl 20)) + 4096

let differ c b =
  let back = B.create Rig.host (B.length b) in
  B.copy ~src:b ~dst:back;
  B.read back Bigarray.char (fun got ->
      let wrong = ref 0 in
      for i = 0 to Bigarray.Array1.dim got - 1 do
        if Bigarray.Array1.unsafe_get got i <> c then incr wrong
      done;
      !wrong)

(* The host's leg reads the source, which a donation would write. *)
let test_staged_source_claimed () =
  let d, p = staging_device "buffer:staged-source" in
  let src = filled three_pieces 's' and dst = B.create d three_pieces in
  let donated, ended =
    donate_during p src
      (fun () -> B.copy ~src ~dst)
      (fun b -> B.fill b 'X')
  in
  equal ~msg:"the copy" string "returned" ended;
  equal ~msg:"bytes not the source's" int 0 (differ 's' dst);
  equal ~msg:"donated during the copy" bool false (Option.is_some donated)

(* The host's leg writes the destination, which a donation would write too. *)
let test_staged_destination_claimed () =
  let d, p = staging_device "buffer:staged-destination" in
  let src = B.create d 64 in
  B.fill src 's';
  let dst = filled 64 'd' in
  let donated, ended =
    donate_during p dst
      (fun () -> B.copy ~src ~dst)
      (fun b -> B.fill b 'X')
  in
  Option.iter
    (fun b -> equal ~msg:"bytes not the donor's" int 0 (differ 'X' b))
    donated;
  equal ~msg:"the copy" string "returned" ended;
  equal ~msg:"bytes not the source's" int 0 (differ 's' dst);
  equal ~msg:"donated during the copy" bool false (Option.is_some donated)

(* A blit's legs into the buffer run after the donation, which consumed it. *)
let test_staged_blit_claimed () =
  let d, p = staging_device "buffer:staged-blit" in
  let s = String.make three_pieces 's' and b = B.create d three_pieces in
  let donated, ended =
    donate_during p b
      (fun () -> B.blit_from_string s 0 b 0 three_pieces)
      ignore
  in
  equal ~msg:"the blit" string "returned" ended;
  equal ~msg:"bytes not the string's" int 0 (differ 's' b);
  equal ~msg:"donated during the blit" bool false (Option.is_some donated)

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

(* A bigarray's buffer is its bytes, and [export] reads them back over the
   same memory. *)
let test_kind (Kind (_, k, v)) =
  List.iter
    (fun n ->
      let ba = Bigarray.Array1.create k Bigarray.c_layout n in
      let b = B.of_bigarray ba in
      equal ~msg:"borrowed" bool true (B.is_borrowed b);
      equal ~msg:"bytes" int (n * Bigarray.kind_size_in_bytes k) (B.length b);
      let back = B.export b k in
      for i = 0 to n - 1 do
        ba.{i} <- v i
      done;
      for i = 0 to n - 1 do
        if back.{i} <> v i then failf "element %d of %d differs" i n
      done)
    [ 0; 1; 17 ]

let test_bigarray_refusals () =
  let b = B.create Rig.host 8 in
  raises_match Exn.invalid_arg (fun () ->
      B.read (B.view b ~first:0 ~length:6) Bigarray.float32 ignore);
  raises_match Exn.invalid_arg (fun () ->
      B.read (B.view b ~first:2 ~length:4) Bigarray.float32 ignore);
  (* A complex element starts at a multiple of one component's size. *)
  let c = B.create Rig.host 32 in
  equal int 1
    (B.read (B.view c ~first:8 ~length:16) Bigarray.complex64
       Bigarray.Array1.dim);
  raises_match Exn.invalid_arg (fun () ->
      B.read (B.view c ~first:4 ~length:16) Bigarray.complex64 ignore)

(* Strings *)

(* A blit: the bytes of the buffer under it, those of a view [first] bytes into
   it, its source or destination, and its ranges, valid or not. *)
type blit = {
  size : int;
  first : int;
  length : int;
  outside : string;
  at : int;
  inside : int;
  n : int;
}

let pp_blit ppf c =
  Format.fprintf ppf
    "a view [%d,+%d] of %d bytes, %S, %d bytes from %d to byte %d of the view"
    c.first c.length c.size c.outside c.n c.at c.inside

(* A valid range, often moved by a byte or two past either end. *)
let blit =
  let open Gen in
  let nudge =
    bind bool (fun out ->
        if out then int_range (-2) 2 else constant ~pp:Format.pp_print_int 0)
  in
  let ranges outside length =
    bind
      (int_range 0 (Int.min (String.length outside) length))
      (fun n ->
        map
          (fun ((at, inside), (dat, din, dn)) ->
            (at + dat, inside + din, n + dn))
          (pair
             (pair
                (int_range 0 (String.length outside - n))
                (int_range 0 (length - n)))
             (triple nudge nudge nudge)))
  in
  with_pp pp_blit
    (bind (int_range 0 40) (fun size ->
         bind (int_range 0 size) (fun first ->
             bind
               (int_range 0 (size - first))
               (fun length ->
                 bind
                   (string_of ~size:(int_range 0 24) char)
                   (fun outside ->
                     map
                       (fun (at, inside, n) ->
                         { size; first; length; outside; at; inside; n })
                       (ranges outside length))))))

let valid c =
  c.at >= 0 && c.n >= 0
  && c.at + c.n <= String.length c.outside
  && c.inside >= 0
  && c.inside + c.n <= c.length

let cover_blit c =
  cover "a blit of bytes" (valid c && c.n > 0);
  cover "a blit of no bytes" (valid c && c.n = 0);
  cover "a range outside" (not (valid c));
  cover "a view into its buffer" (c.first > 0 && valid c && c.n > 0)

(* Where a blit's buffer lies: host memory, or memory of a device that the host
   does not address, which the blit reaches through staging memory. *)
type place = Host_memory | Unaddressed

let unaddressed = lazy (fst (P.open_ ~host_visible:false "buffer:blits"))

(* A buffer of [place] holding [s], made and read back by copies. *)
let holding place s =
  match place with
  | Host_memory -> B.of_string s
  | Unaddressed ->
      let b = B.create (Lazy.force unaddressed) (String.length s) in
      B.copy ~src:(B.of_string s) ~dst:b;
      b

let letters n = String.init n (fun i -> Char.chr (97 + (i mod 26)))

(* [blit_from_string] writes the view's bytes the string gives and no other, and
   raises, changing nothing, on a range outside either. *)
let law_from_string place c =
  cover_blit c;
  let model = Bytes.of_string (letters c.size) in
  let b = holding place (Bytes.to_string model) in
  let v = B.view b ~first:c.first ~length:c.length in
  (match B.blit_from_string c.outside c.at v c.inside c.n with
  | () ->
      equal ~msg:"valid" bool true (valid c);
      Bytes.blit_string c.outside c.at model (c.first + c.inside) c.n
  | exception Invalid_argument _ -> equal ~msg:"valid" bool false (valid c));
  equal string (Bytes.to_string model) (contents b)

(* [blit_to_bytes] reads the view's bytes into the bytes' range and writes no
   other, and raises, changing nothing, on a range outside either. *)
let law_to_bytes place c =
  cover_blit c;
  let whole = letters c.size in
  let v = B.view (holding place whole) ~first:c.first ~length:c.length in
  let model = Bytes.of_string c.outside and got = Bytes.of_string c.outside in
  let valid =
    c.inside >= 0 && c.n >= 0
    && c.inside + c.n <= c.length
    && c.at >= 0
    && c.at + c.n <= Bytes.length got
  in
  (match B.blit_to_bytes v c.inside got c.at c.n with
  | () ->
      equal ~msg:"valid" bool true valid;
      Bytes.blit_string whole (c.first + c.inside) model c.at c.n
  | exception Invalid_argument _ -> equal ~msg:"valid" bool false valid);
  equal string (Bytes.to_string model) (Bytes.to_string got)

(* [fill] writes every byte of the view and no other. *)
let law_fill place c =
  cover "a fill of bytes" (c.length > 0);
  cover "a fill of no bytes" (c.length = 0);
  cover "a view into its buffer" (c.first > 0 && c.length > 0);
  let model = Bytes.of_string (letters c.size) in
  let b = holding place (Bytes.to_string model) in
  B.fill (B.view b ~first:c.first ~length:c.length) 'z';
  Bytes.fill model c.first c.length 'z';
  equal string (Bytes.to_string model) (contents b)

(* Blits and a fill of three pieces of the staging memory keep every byte in
   its place: each piece's bytes differ from the others'. *)
let test_blits_staged_large () =
  let d, _ = P.open_ ~host_visible:false "buffer:blits-large" in
  let n = (2 * (32 lsl 20)) + 3 in
  let data = String.init n (fun i -> Char.unsafe_chr ((i + (i / 251)) land 255)) in
  let b = B.create d n in
  let differ s got =
    let wrong = ref 0 in
    for i = 0 to n - 1 do
      if String.unsafe_get s i <> Bytes.unsafe_get got i then incr wrong
    done;
    !wrong
  in
  let got = Bytes.create n in
  B.blit_from_string data 0 b 0 n;
  B.blit_to_bytes b 0 got 0 n;
  equal ~msg:"bytes a blit there and back changed" int 0 (differ data got);
  B.fill b 'q';
  B.blit_to_bytes b 0 got 0 n;
  equal ~msg:"bytes the fill missed" int 0 (differ (String.make n 'q') got)

(* Host access *)

(* The host reads and writes the memory of a device that it addresses as the
   memory itself, without a borrow: what a borrow on the host reads, byte for
   byte, and what it writes. *)
let test_host_addressed () =
  let d, _ = P.open_ "buffer:host-addressed" in
  let m = B.create d 64 in
  let on_host = require_some (B.borrow Rig.host m) in
  B.blit_from_string (String.make 64 'b') 0 m 0 64;
  equal ~msg:"a blit, through a borrow" string (String.make 64 'b')
    (bytes on_host);
  B.write m Bigarray.char (fun a -> Bigarray.Array1.fill a 'a');
  equal ~msg:"a bigarray, through a borrow" string (String.make 64 'a')
    (bytes on_host);
  B.write on_host Bigarray.char (fun a -> Bigarray.Array1.fill a 'h');
  let got = Bytes.create 64 in
  B.blit_to_bytes m 0 got 0 64;
  equal ~msg:"a borrow's write, by a blit" string (String.make 64 'h')
    (Bytes.to_string got);
  equal ~msg:"a borrow's write, by a bigarray" string (String.make 64 'h')
    (bytes m)

(* Memory the host does not address: a device's that it reaches only by a
   mapping, an io device's, whose pages a borrow asks for, and another
   machine's, though its driver gives it a host address. *)
let unaddressed_memory name =
  let d, _ = P.open_ ~host_visible:false ("buffer:" ^ name) in
  let machine = Support.machine name in
  let far =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_ ~machine
         {
           name = "buffer:" ^ name ^ "-far";
           driver = (module P);
           make = (fun () -> Ok (P.make ()));
         })
  in
  [
    ("a device's", B.create d 8);
    ("an io device's", B.create (Support.io ("buffer:" ^ name ^ "-io")) 8);
    ("another machine's", B.create far 8);
  ]

(* The host refuses to read or write in place memory it does not address. *)
let test_host_unaddressed () =
  List.iter
    (fun (whose, m) ->
      let refused what f =
        raises_match ~msg:(what ^ " " ^ whose) Exn.invalid_arg f
      in
      refused "a read of" (fun () -> B.read m Bigarray.char ignore);
      refused "a write of" (fun () -> B.write m Bigarray.char ignore))
    (unaddressed_memory "unaddressed")

(* The blits and a fill reach memory the host does not address. An io device's
   is test_io's: this one reads and writes nothing. *)
let test_blits_unaddressed () =
  List.iter
    (fun (whose, m) ->
      let got = Bytes.create 8 in
      B.blit_from_string "abcdefgh" 0 m 0 8;
      B.blit_to_bytes m 0 got 0 8;
      equal ~msg:("blits into and out of " ^ whose) string "abcdefgh"
        (Bytes.to_string got);
      B.fill (B.view m ~first:2 ~length:4) 'f';
      B.blit_to_bytes m 0 got 0 8;
      equal ~msg:("a fill of " ^ whose) string "abffffgh" (Bytes.to_string got))
    (List.filter
       (fun (whose, _) -> whose <> "an io device's")
       (unaddressed_memory "unaddressed-blits"))

(* A blit from a string waits for every use of the buffer, its own device's
   included; a blit to bytes for its last write only. *)
let test_blit_waits () =
  let d, p = P.open_ "buffer:blit-waits" in
  let h = B.of_string (String.make (1 lsl 16) 'a') in
  let on_d = require_some (B.borrow d h) in
  ignore (submit (Sub.make d [| Support.touching [| Sub.read |] |]) ~buffers:[| on_d |]);
  let got = Bytes.create 4 in
  B.blit_to_bytes h 0 got 0 4;
  equal ~msg:"a read left queued" int 1 (P.queued p);
  equal ~msg:"bytes" string "aaaa" (Bytes.to_string got);
  B.blit_from_string "bbbb" 0 h 0 4;
  equal ~msg:"the read ran first" int 0 (P.queued p);
  ignore
    (submit (Sub.make d [| Support.touching [| Sub.write |] |]) ~buffers:[| on_d |]);
  B.blit_to_bytes h 0 got 0 4;
  equal ~msg:"the write ran first" int 0 (P.queued p)

(* A host access that waits keeps its buffer until it returns: in another
   domain, [use] gets the only reference to memory of [d] that queued work
   writes, and sleeps on that work at the gate. The work then runs and the
   memory is collected and its device's cache freed, while the use sleeps:
   memory the access still walks the stamps of, or blits through, does not
   return. *)
let host_accesses =
  [
    ("a wait for every use", fun b -> B.wait b Rig.Read_write);
    ("a blit to bytes", fun b -> B.blit_to_bytes b 0 (Bytes.create 8) 0 8);
    ("a blit from a string", fun b -> B.blit_from_string "abcdefgh" 0 b 0 8);
  ]

let test_wait_keeps (name, use) =
  let d, p = P.open_ ("buffer:wait-keeps-" ^ name) in
  P.gate p;
  let user =
    Domain.spawn (fun () ->
        let b = B.create d 64 in
        let s = Sub.make d [| Support.touching [| Sub.write |] |] in
        ignore (submit s ~buffers:[| b |]);
        use b)
  in
  Support.await "the use asleep" (fun () -> P.sleepers p = 1);
  ignore (P.run p);
  Gc.full_major ();
  Gc.full_major ();
  ignore (B.create ~memory:Pinned d 8);
  Rig.free_cache d;
  equal ~msg:"frees while the use sleeps" int 0 (count "free" p);
  P.open_gate p;
  Domain.join user

(* A bigarray of a buffer keeps its memory once the buffer is collected. *)
let test_bigarray_keeps () =
  List.iter
    (fun n ->
      let view =
        (fun () ->
          let ba = B.export (B.create Rig.host n) Bigarray.char in
          Bigarray.Array1.fill ba 'v';
          ba)
          ()
      in
      Gc.full_major ();
      Gc.full_major ();
      B.write (B.create Rig.host n) Bigarray.char (fun a ->
          Bigarray.Array1.fill a 'w');
      equal ~msg:(Printf.sprintf "%d bytes" n) char 'v' view.{n - 1})
    [ 16; 1 lsl 17 ]

(* A pinned buffer on [d] filled with [c], seen through a borrow on the host. *)
let viewed d n c =
  let m = B.create ~memory:Pinned d n in
  let view = B.write (require_some (B.borrow Rig.host m)) Bigarray.char Fun.id in
  Bigarray.Array1.fill view c;
  (Support.at m, view)

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
  B.write (require_some (B.borrow Rig.host other)) Bigarray.char (fun a ->
      Bigarray.Array1.fill a 'w');
  equal ~msg:"another buffer's memory" bool false (Support.at other = at);
  equal ~msg:"read through the array" char 'v' (Option.get !sub).{7};
  sub := None;
  collect d;
  let again = B.create ~memory:Pinned d n in
  equal ~msg:"once no array reads it" int at (Support.at again);
  ignore (Sys.opaque_identity other)

(* Rig holds a share of the storage of every bigarray over a borrow: once
   collected, the views dropped while its memory lives leave that share and free
   nothing. *)
let test_bigarray_shares () =
  let d, _ = P.open_ "buffer:shared" in
  let m = B.create ~memory:Pinned d 4096 in
  let on_host = require_some (B.borrow Rig.host m) in
  let collected_shares () =
    Gc.full_major ();
    Gc.full_major ();
    let view = B.read on_host Bigarray.char Fun.id in
    Gc.full_major ();
    (view, Rig_support.shares view)
  in
  let first () =
    let view, alone = collected_shares () in
    Bigarray.Array1.fill view 's';
    let sub = Bigarray.Array1.sub view 0 8 in
    [ alone; Rig_support.shares sub ]
  in
  equal ~msg:"rig and a view, then its sub" (list int) [ 2; 3 ] (first ());
  let view, alone = collected_shares () in
  equal ~msg:"rig and a new view" int 2 alone;
  equal ~msg:"the bytes" char 's' view.{4095};
  ignore (Sys.opaque_identity m)

let tests =
  [
    group ~timeout "host buffers"
      [
        test "a buffer holds its elements' bytes" test_create;
        test "a buffer refuses bounds it cannot hold" test_refusals;
        test "a view shares its buffer's memory" test_views;
        prop "an address is aligned to a power of two, in or out of the buffer"
          aligned_case aligned_law;
        test "aligned refuses what is no power of two, and io memory"
          test_aligned_refusals;
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
        test "another machine's device copies this process's memory as it is"
          test_copy_across;
        test "an empty buffer of a driver's device has address and handle 0"
          test_empty_address;
        test "a borrow of a peer's memory on a device that runs no copy copies"
          test_copyless_borrow;
        test "a borrow of a lost device's buffer raises its loss"
          test_borrow_lost;
        test "a device that maps no host memory maps a peer's visible memory"
          test_peer_no_host;
        test "empty host memory borrows on a device with no mapping"
          test_empty_borrow;
        test "a device that maps no host memory borrows none" test_maps_no_host;
        test "a driver maps host memory only from the start of a page"
          test_map_host_paged;
        test
          "a device that maps no host memory copies to and from the host \
           through its own memory"
          test_staged_no_host;
        test
          "a borrow of a peer's hidden memory on a device that maps no host \
           memory copies in and out"
          test_mapless_borrow;
        test "a borrow on its own device is the buffer" test_borrow_own;
        test "a mapping is released once its memory died and its work ran"
          test_mapping_released;
        test
          "borrowed bytes return to their keeper once every mapping of them is \
           released"
          test_bytes_after_unmap;
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
        test "a copy larger than the staging memory lands every byte"
          test_copy_staged_large;
        test "copies take the staging memory in turn" test_staging_turns;
        test "a loss with the staging memory leaves other copies working"
          test_staging_after_loss;
        test "a staged copy gives its slot back on every exit"
          test_staging_gives_back;
        test "a staged copy holds its source against a donation"
          test_staged_source_claimed;
        test "a staged copy holds its destination against a donation"
          test_staged_destination_claimed;
        test "a staged blit holds its buffer against a donation"
          test_staged_blit_claimed;
      ];
    cases ~timeout
      ~name:(fun (n, _, _) -> n)
      "polymorphic comparison ends, and equality refuses one device's buffers"
      distinct test_polymorphic;
    cases ~timeout
      ~name:(fun (Kind (n, _, _)) -> n)
      "a bigarray's buffer is its bytes" kinds test_kind;
    group ~timeout "strings"
      [
        prop "a blit from a string writes its range of the buffer and no other"
          blit (law_from_string Host_memory);
        prop "a blit to bytes reads its range of the buffer into its range" blit
          (law_to_bytes Host_memory);
        prop "a fill writes its view of the buffer and no other" blit
          (law_fill Host_memory);
        prop
          "a blit from a string writes its range of memory the host does not \
           address"
          blit (law_from_string Unaddressed);
        prop
          "a blit to bytes reads its range of memory the host does not address"
          blit (law_to_bytes Unaddressed);
        prop "a fill writes its view of memory the host does not address" blit
          (law_fill Unaddressed);
        test "blits and a fill through several staging pieces keep every byte"
          test_blits_staged_large;
        test "a blit waits as a host access of its kind does" test_blit_waits;
      ];
    group ~timeout "host access"
      [
        test "the host accesses the device memory it addresses directly"
          test_host_addressed;
        test "the host refuses in place access to memory it does not address"
          test_host_unaddressed;
        test "the blits and a fill reach memory the host does not address"
          test_blits_unaddressed;
      ];
    group ~timeout "bigarrays"
      [
        test "a bigarray refuses kinds, sizes and alignments it cannot read"
          test_bigarray_refusals;
        test "a bigarray keeps its memory once its buffer is collected"
          test_bigarray_keeps;
        test "a bigarray over a borrow keeps the device memory it reads"
          test_bigarray_keeps_borrowed;
        test "a bigarray's storage keeps rig's share" test_bigarray_shares;
      ];
    group ~timeout "waits"
      [
        test "a wait returns once the work that wrote the memory ran" test_wait;
        test "a read and a write follow the memory's work" test_access_orders;
        test "an export waits and leaves the claims for good" test_export;
        test "a dead buffer's wait raises its reason" test_dead;
        test "every function reaching a dead buffer's bytes refuses it"
          test_dead_refused;
        test "memory a lost device wrote raises Lost" test_lost_memory;
        cases ~name:fst "a host access keeps its buffer through its wait"
          host_accesses test_wait_keeps;
      ];
  ]

let words =
  group ~timeout "words"
    [
      test "a host buffer of 64 bytes costs at most 41 words" test_create_words;
      test "a wait whose points are reached allocates nothing" test_wait_words;
      test "behind a transport a wait for reached work allocates nothing"
        test_wait_words_transport;
      test "a host buffer of 1 MiB costs at most 32 and 9 major words"
        test_create_large_words;
      test "a host buffer of no bytes costs at most 32 words"
        test_create_empty_words;
      test "a bigarray's buffer costs at most 54 words" (fun () ->
          at_most_words (54, 0) (fun () -> B.of_bigarray f32));
      test "a view costs at most 19 words" (fun () ->
          at_most_words (19, 0) (fun () -> B.view b16 ~first:16 ~length:16));
      test "a read costs at most 27 words" (fun () ->
          at_most_words (27, 0) (fun () -> B.read b16 Bigarray.float32 ignore));
    ]

let () = exit (run "rig.buffer" (tests @ [ words ]))
