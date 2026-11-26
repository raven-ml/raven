(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Buffer › Copying: the routes copies take, staging memory, boxes, the copies,
   the blits and the fills. Polled logs what its driver runs. *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module Support = Rig_support

(* The most bytes of an allocation of staging memory (rig.mli, Copying). *)
let half = 32 lsl 20

(* Three pieces of staging memory: two allocations of 32 MiB, and the third
   piece through one of them used again. *)
let three_pieces = (2 * half) + 4096
let submit ?(buffers = [||]) ?(args = [||]) s = Rig.submit s ~buffers ~args

(* The values of a submission's one fill, which reads none. *)
let one = Support.zeros 1
let lost = function Rig.Lost _ -> true | _ -> false
let count call p = List.length (List.filter (( = ) call) (P.log p))
let memory name = Result.get_ok (Rig.software name)

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
  Bytes.unsafe_to_string s

(* A buffer on [d] holding [s]. *)
let holding d s =
  let b = B.create d (String.length s) in
  B.blit_from_string s 0 b 0 (String.length s);
  b

(* The first byte at which [got] differs from [want], or [None]. The compare
   runs first, so a pass costs no loop over the bytes. *)
let first_difference want got =
  if String.equal want got then None
  else
    let n = Int.min (String.length want) (String.length got) in
    let rec go i = if i = n || want.[i] <> got.[i] then i else go (i + 1) in
    Some (go 0)

let lands ?(msg = "the bytes") want got =
  equal
    ~msg:(msg ^ ": first byte that differs")
    (option int) None
    (first_difference want got)

(* [n] bytes in which each 64 KiB block differs from the next, and each 16 MiB
   from the next, so a piece that lands at the wrong place, or out of the wrong
   allocation of 32 MiB, shows. *)
let pattern n =
  String.init n (fun i ->
      Char.unsafe_chr ((i + ((i lsr 16) * 13) + (i lsr 24)) land 255))

(* The sleeps of every device of [ps]. *)
let sleeps ps = List.fold_left (fun n p -> n + P.sleeps p) 0 ps

(* The host memory [p]'s device mapped: staging memory alone, where the copies
   name no borrow. *)
let staging_mapped p = List.length (P.host_maps p)

(* The allocations of [Pinned] memory [p]'s driver made. *)
let pinned p =
  List.length (List.filter (fun (k, _, ok) -> k = B.Pinned && ok) (P.allocs p))

(* A device whose memory the host does not address. *)
let hidden ?peers name = P.open_ ~host_visible:false ?peers name

(* A device that maps no host memory, whose copies to and from the host go
   through its own staging memory. *)
let staging_device name = P.open_ ~maps_host:false ~host_visible:false name

(* 64 bytes of [c] on [d], whose memory the host does not address. *)
let on d c = holding d (String.make 64 c)

(* [n] bytes [c] of host memory that starts off a page, which a device that maps
   host memory reaches only through staging memory. *)
let off_page n c =
  let paged =
    B.export (filled (Int.max B.paged_from (64 + n)) c) Bigarray.char
  in
  B.of_bigarray (Bigarray.Array1.sub paged 64 n)

(* [Out_of_memory] for [d] of an allocation of staging memory, of at most 32 MiB
   (rig.mli, Copying). *)
let staging_refused d = function
  | Rig.Out_of_memory (d', n) -> Rig.equal d d' && n > 0 && n <= half
  | _ -> false

let with_budget d n f =
  let budget = Rig.budget d in
  Rig.set_budget d n;
  Fun.protect ~finally:(fun () -> Rig.set_budget d budget) f

(* Routes *)

(* Bytes into [dst] and back out of it: the bytes back. *)
let round_trip_bytes ~src ~dst =
  B.write src Bigarray.char (fun a -> Bigarray.Array1.fill a 'r');
  B.copy ~src ~dst;
  let back = filled (B.length src) '0' in
  B.copy ~src:dst ~dst:back;
  bytes back

let test_host_copy () =
  let src = filled 100 'x' and dst = filled 100 'y' in
  B.copy ~src ~dst;
  equal string (String.make 100 'x') (bytes dst)

(* A software device's memory is memory the host addresses: the host copies,
   both ways. *)
let test_software_copy () =
  let d = memory "copy:software" in
  equal string (String.make 64 'r')
    (round_trip_bytes ~src:(B.create Rig.host 64) ~dst:(B.create d 64))

(* A device that runs no copy has memory the host addresses: the host copies,
   and the device runs nothing. *)
let test_no_copy_engine () =
  let d, p = P.open_ ~copies:false "copy:no-queue" in
  equal ~msg:"the bytes back" string (String.make 64 'r')
    (round_trip_bytes ~src:(B.create Rig.host 64) ~dst:(B.create d 64));
  equal ~msg:"copies the device ran" int 0 (P.submits p)

(* A driver's device's borrow of host memory is host memory: the host copies
   into and out of it, on a device that runs no copy too. *)
let test_borrowed_host () =
  let d, _ = P.open_ ~copies:false "copy:borrowed-host" in
  let h = filled B.paged_from 'h' in
  let b = require_some (B.borrow d h) in
  let out = B.create Rig.host B.paged_from in
  B.copy ~src:b ~dst:out;
  equal ~msg:"out of the borrow" string
    (String.make B.paged_from 'h')
    (bytes out);
  B.copy ~src:(filled B.paged_from 'i') ~dst:b;
  equal ~msg:"into the borrow" string (String.make B.paged_from 'i') (bytes h)

(* Polled whose queues are named ["Q0"], ["Q1"]. *)
module Renamed = struct
  include P

  let facts d =
    let f = P.facts d in
    let name i (q : Rig_edge.queue) =
      { q with name = Printf.sprintf "Q%d" i }
    in
    { f with queues = List.mapi name f.queues }
end

(* Polled whose queue named "COPY:0" runs no copy, and which has no copy
   engine. *)
module Copyless = struct
  include P

  let facts d =
    let f = P.facts d in
    let fills name = { Rig_edge.name; runs = [ Fill ] } in
    let queues = [ fills "COMPUTE:0"; fills "COPY:0" ] in
    { f with queues; copy_engine = None }
end

(* Polled with its first queue alone, which runs fills, copies and launches, and
   is its copy engine. *)
module One_queue = struct
  include P

  let facts d =
    let f = P.facts d in
    { f with queues = [ List.hd f.queues ]; copy_engine = Some 0 }
end

(* The driver [D] whose copy engine is [engine]. *)
let engine (module D : Rig_edge.Driver with type t = P.t) engine =
  let module E = struct
    include D

    let facts d = { (D.facts d) with copy_engine = engine }
  end in
  (module E : Rig_edge.Driver with type t = P.t)

let open_as (module D : Rig_edge.Driver with type t = P.t) name p =
  require_ok ~pp:Format.pp_print_string
    (Rig.open_
       { name; driver = (module D); make = (fun ~word -> Ok (P.bind p ~word)) })

(* Bytes into memory of [d] the host does not address and back, which [d]'s copy
   engine runs: the copies [p] ran and the bytes back. *)
let engine_round_trip d p =
  let got = round_trip_bytes ~src:(B.create Rig.host 64) ~dst:(B.create d 64) in
  (P.submits p, got)

(* rig's copies go to the copy engine the facts name, whatever the queues' names
   and order: a queue named "COPY:0" may run none, and a device without a copy
   engine takes no copy part. *)
let test_copy_engine () =
  let p = P.make ~host_visible:false () in
  let d = open_as (module Renamed) "copy:renamed" p in
  equal ~msg:"renamed: copies run, bytes back" (pair int string)
    (2, String.make 64 'r')
    (engine_round_trip d p);
  let p = P.make ~host_visible:false () in
  let d = open_as (module One_queue) "copy:one-queue" p in
  equal ~msg:"one queue: copies run, bytes back" (pair int string)
    (2, String.make 64 'r')
    (engine_round_trip d p);
  let p = P.make ~host_visible:false () in
  let d = open_as (engine (module P) (Some 0)) "copy:first-engine" p in
  let (), events =
    Rig.Profile.take (fun () -> ignore (engine_round_trip d p))
  in
  let lanes =
    List.filter_map
      (function
        | Rig.Profile.Work s when Rig.equal s.device d -> Some s.lane
        | _ -> None)
      events
  in
  equal ~msg:"the first queue runs the copies" (list string)
    [ "COMPUTE:0"; "COMPUTE:0" ]
    lanes;
  let c = open_as (module Copyless) "copy:copyless" (P.make ()) in
  let src = B.create c 8 and dst = B.create c 8 in
  raises_match (Exn.invalid_arg ~sub:"runs no copies") (fun () ->
      Sub.(make c [| copy ~src:(Buffer src) ~dst:(Buffer dst) |]))

(* Between two devices of one driver, the source's device copies, mapping the
   destination. *)
let test_peer () =
  let d, pd = hidden "copy:peer-src" in
  let e, pe = hidden "copy:peer-dst" in
  let src = on d 'p' and dst = B.create e 64 in
  let before = (P.submits pd, P.submits pe) in
  B.copy ~src ~dst;
  equal ~msg:"copies each device ran" (pair int int)
    (fst before + 1, snd before)
    (P.submits pd, P.submits pe);
  equal ~msg:"peer mappings" int 1 (count "map_peer" pd);
  equal ~msg:"the bytes" string (String.make 64 'p') (contents dst)

(* A borrow, on a device that runs no copy, of a peer's memory the host does not
   address copies by the memory's own device: in and out. *)
let test_copyless_borrow () =
  let owner, _ = hidden "copy:copyless-owner" in
  let still, _ = P.open_ ~copies:false "copy:copyless" in
  let on_still = require_some (B.borrow still (B.create owner 64)) in
  equal string (String.make 64 'r')
    (round_trip_bytes ~src:(B.create Rig.host 64) ~dst:on_still)

(* Memory of another machine's driver device copies only directly: with this
   process's memory by its own device, with its machine's memory by the source's
   device. Into a device the source does not map, or on a device that runs no
   copy, the copy raises. *)
let test_machines_refused () =
  let far = Support.machine "elsewhere" in
  let gpu ?copies name =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_ ~machine:far
         {
           name;
           driver = (module P);
           make =
             (fun ~word ->
               Ok
                 (P.bind ~word
                    (P.make ?copies ~host_visible:false ~peers:false ())));
         })
  in
  let g = gpu "copy:far-gpu" and g' = gpu "copy:far-other" in
  let still = gpu ~copies:false "copy:far-still" in
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
   copy each way, each naming this process's side, and within its machine one
   naming none. *)
let test_machines () =
  let machine = Support.machine "across" in
  let p = P.make ~host_visible:false ~peers:false () in
  let g =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_ ~machine
         {
           name = "copy:across";
           driver = (module P);
           make = (fun ~word -> Ok (P.bind p ~word));
         })
  in
  let src = B.of_string "across machines" in
  let far = B.create g (B.length src) in
  let back = B.create Rig.host (B.length src) in
  B.copy ~src ~dst:far;
  B.copy ~src:far ~dst:back;
  equal ~msg:"the bytes back" string "across machines" (bytes back);
  B.copy ~src:far ~dst:(B.create g (B.length src));
  equal ~msg:"no host maps" (list int) [] (P.host_maps p);
  let side = function `None -> "none" | `Src -> "src" | `Dst -> "dst" in
  equal ~msg:"this process's side of each copy" (list string)
    [ "src"; "dst"; "none" ]
    (List.map (fun (c : P.copy) -> side c.local) (P.copies p))

(* A copy from a borrow of host memory on the device that copies runs directly
   through the borrow's mapping, made at the borrow: the copy maps nothing. One
   from the host buffer itself goes through staging memory while the borrow
   lives: a copy runs directly only on what it names. *)
let test_borrow_direct () =
  let d, p = hidden "copy:borrowed-direct" in
  let n = B.paged_from in
  let h = filled n 'b' in
  let on_d = require_some (B.borrow d h) in
  let dst = B.create d n in
  equal ~msg:"mapped at the borrow" (list int) [ n ] (P.host_maps p);
  B.copy ~src:on_d ~dst;
  equal ~msg:"a copy from the borrow: one submit" int 1 (P.submits p);
  equal ~msg:"and no mapping" (list int) [ n ] (P.host_maps p);
  let back = filled n '0' in
  B.copy ~src:dst ~dst:(require_some (B.borrow d back));
  equal ~msg:"the bytes back through a borrow" string (String.make n 'b')
    (bytes back);
  B.copy ~src:h ~dst;
  equal ~msg:"a copy from the host buffer, through staging memory" int 3
    (List.length (P.host_maps p));
  not_equal ~msg:"its mapping is the buffer's" int (Support.at h)
    (List.nth (P.host_mapped_at p) 2);
  equal ~msg:"its leg on the device" int 3 (P.submits p);
  ignore (Sys.opaque_identity on_d)

(* Staging memory *)

(* Between host memory and memory the host does not address, the device copies
   through staging memory, which it maps: a copy maps no memory of the host's
   that it does not name, even memory that starts on a page. Both ways go
   through one allocation of staging memory, which the device maps once. The
   host's staging memory starts empty. *)
let test_host_staged () =
  Rig.free_cache Rig.host;
  let d, p = hidden "copy:direct" in
  let src = B.create Rig.host B.paged_from in
  ignore (round_trip_bytes ~src ~dst:(B.create d B.paged_from));
  equal ~msg:"copies the device ran" int 2 (P.submits p);
  equal ~msg:"host memory mapped" int 1 (List.length (P.host_maps p));
  not_equal ~msg:"the mapping is the source's" int (Support.at src)
    (List.hd (P.host_mapped_at p))

(* Host memory off a page goes through the host's staging memory, which the
   device maps once: a second copy maps nothing. *)
let test_off_page_staged () =
  let d, p = hidden "copy:staged" in
  let src = off_page 64 's' in
  let dst = B.create d 64 in
  B.copy ~src ~dst;
  equal ~msg:"the bytes" string (String.make 64 's') (contents dst);
  let maps = List.length (P.host_maps p) in
  B.copy ~src ~dst;
  equal ~msg:"maps of a second copy" int maps (List.length (P.host_maps p))

(* Between devices that map none of each other's memory, the bytes go through
   the staging memory, which each device maps once. *)
let test_unmapped_peers () =
  let d, pd = hidden ~peers:false "copy:unmapped-src" in
  let e, pe = hidden ~peers:false "copy:unmapped-dst" in
  let src = on d 'u' and dst = B.create e 64 in
  B.copy ~src ~dst;
  equal ~msg:"the bytes" string (String.make 64 'u') (contents dst);
  let maps () = (List.length (P.host_maps pd), List.length (P.host_maps pe)) in
  let before = maps () in
  B.copy ~src ~dst;
  equal ~msg:"maps of a second copy" (pair int int) before (maps ())

(* A copy larger than an allocation of staging memory goes through it in pieces:
   three pieces through two allocations, the last through one used again, land
   every byte where it belongs. *)
let test_staged_large () =
  let d, _ = hidden ~peers:false "copy:staged-large-src" in
  let e, _ = hidden ~peers:false "copy:staged-large-dst" in
  let want = pattern three_pieces in
  let src = holding d want and dst = B.create e three_pieces in
  B.copy ~src ~dst;
  lands want (contents dst)

(* A device that maps no host memory copies host bytes into its own memory and
   back, its own [Pinned] memory standing in for the host's staging, and maps no
   host memory for it. *)
let test_own_staging () =
  let d, p = staging_device "copy:staged-no-host" in
  let m = B.create d 64 in
  equal ~msg:"the bytes back" string (String.make 64 'r')
    (round_trip_bytes ~src:(B.create Rig.host 64) ~dst:m);
  equal ~msg:"host maps" (list int) [] (P.host_maps p)

(* A borrow, on a device that maps no host memory, of a peer's memory the host
   does not address copies in and out, staged through the borrowing device's own
   memory. *)
let test_own_staging_borrow () =
  let owner, _ = hidden "copy:mapless-owner" in
  let mapless, _ = P.open_ ~maps_host:false "copy:mapless" in
  let on_mapless = require_some (B.borrow mapless (B.create owner 1)) in
  equal string "r" (round_trip_bytes ~src:(B.create Rig.host 1) ~dst:on_mapless)

(* Two devices that map no host memory and no peer's, and whose memory the host
   does not address, stage through their own memory each: a copy between them of
   three pieces lands byte for byte, and neither device maps host memory for
   it. *)
let test_own_staging_pair () =
  let open_ name =
    P.open_ ~maps_host:false ~peers:false ~host_visible:false name
  in
  let d, pd = open_ "copy:mapless-pair-src" in
  let e, pe = open_ "copy:mapless-pair-dst" in
  let want = pattern three_pieces in
  let src = holding d want and dst = B.create e three_pieces in
  B.copy ~src ~dst;
  lands ~msg:"the bytes" want (contents dst);
  equal ~msg:"host maps" (list int) [] (P.host_maps pd @ P.host_maps pe)

(* Copies between two such devices both ways run concurrently, each through
   allocations of each device's staging memory. Two copies each way, with both
   devices' sleeps held: all four wait at the gates at once, none for an
   allocation. Every copy lands once the gates open. *)
let test_own_staging_both_ways () =
  let open_ name =
    P.open_ ~maps_host:false ~peers:false ~host_visible:false name
  in
  let d, pd = open_ "copy:mapless-slots-a" in
  let e, pe = open_ "copy:mapless-slots-b" in
  let letter k = Char.chr (Char.code 'a' + k) in
  let copies =
    List.init 4 (fun k ->
        let from, into = if k < 2 then (d, e) else (e, d) in
        (on from (letter k), B.create into 64))
  in
  P.gate pd;
  P.gate pe;
  let copier (src, dst) = Thread.create (fun () -> B.copy ~src ~dst) () in
  let copiers = List.map copier copies in
  Support.await "four copies at the gates" (fun () ->
      P.sleepers pd + P.sleepers pe = 4);
  P.open_gate pd;
  P.open_gate pe;
  List.iter Thread.join copiers;
  equal (list string)
    (List.init 4 (fun k -> String.make 64 (letter k)))
    (List.map (fun (_, dst) -> contents dst) copies)

(* A device that maps no host memory stages through allocations of its own
   [Pinned] memory, of at most 32 MiB each, made at the first piece that needs
   them: a blit of one byte makes one, a second blit of one byte reuses it, and
   a blit of two pieces, which holds two at once, makes at most two more. *)
let test_made_at_need () =
  let d, p = staging_device "copy:staging-at-need" in
  let sizes () =
    List.filter_map
      (fun (k, n, ok) -> if k = B.Pinned && ok then Some n else None)
      (P.allocs p)
  in
  B.blit_from_string "x" 0 (B.create d 1) 0 1;
  equal ~msg:"one piece" int 1 (List.length (sizes ()));
  B.blit_from_string "z" 0 (B.create d 1) 0 1;
  equal ~msg:"one piece again" int 1 (List.length (sizes ()));
  let n = half + 1 in
  B.blit_from_string (String.make n 'y') 0 (B.create d n) 0 n;
  let made = List.length (sizes ()) in
  greater ~msg:"two pieces" int ~than:1 made;
  less ~msg:"two pieces" int ~than:4 made;
  List.iter (fun n -> at_most ~msg:"bytes" int ~than:half n) (sizes ())

(* A device that maps no host memory, and whose own memory the host addresses,
   stages through that memory, in its budget. A copy between host memory and its
   borrow of a peer's memory that the host does not address stages, so a budget
   of 0 refuses it, but for a copy into [Mapped] memory, which the host writes
   through its window. *)
type way = Out_of | Into

let staged_borrows =
  [
    ("out of Device memory", B.Device, Out_of);
    ("into Device memory", B.Device, Into);
    ("out of Mapped memory", B.Mapped, Out_of);
    ("into Mapped memory", B.Mapped, Into);
  ]

let test_staged_borrow (row, kind, way) =
  let slug = String.map (fun c -> if c = ' ' then '-' else c) row in
  let name = "copy:staged-" ^ slug in
  let owner, _ = hidden (name ^ "-owner") in
  let d, _ = P.open_ ~maps_host:false name in
  let on_d = require_some (B.borrow d (B.create ~memory:kind owner 1)) in
  let host = B.of_string "w" in
  let copy () =
    match way with
    | Out_of -> B.copy ~src:on_d ~dst:host
    | Into -> B.copy ~src:host ~dst:on_d
  in
  match (kind, way) with
  | B.Mapped, Into ->
      with_budget d 0 copy;
      equal string "w" (contents on_d)
  | _ -> raises_match (staging_refused d) (fun () -> with_budget d 0 copy)

(* A copy whose staging memory a host budget of 0 refuses raises Out_of_memory
   for the host, and gives back what it held: a copy after several refusals
   still finds staging memory. *)
let test_refused_gives_back () =
  let d, _ = hidden ~peers:false "copy:staging-src" in
  let e, _ = hidden ~peers:false "copy:staging-dst" in
  let src = on d 'g' and dst = B.create e 64 in
  with_budget Rig.host 0 (fun () ->
      for _ = 1 to 3 do
        raises_match (staging_refused Rig.host) (fun () -> B.copy ~src ~dst)
      done);
  B.copy ~src ~dst;
  equal string (String.make 64 'g') (contents dst)

(* Small copies from host memory to several devices through the host's staging
   memory return with no sleep, their pieces one after the other in one
   allocation, which each device maps once. The host's staging memory starts
   empty. *)
let test_small_uploads () =
  Rig.free_cache Rig.host;
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
  equal ~msg:"allocations each device mapped" (list int) [ 1; 1; 1; 1 ]
    (List.map staging_mapped ps);
  List.iter (fun (pt, _, _) -> Rig.Point.wait pt) copies;
  equal ~msg:"the copies' bytes" (list string)
    (List.map (fun (_, _, c) -> String.make 64 c) copies)
    (List.map (fun (_, dst, _) -> contents dst) copies)

(* Copies from host memory off a page that the host writes into staging memory
   put their pieces one after the other in one allocation, and wait for no work
   there: a copy gives its allocation back once the host wrote its piece, and
   while it waits for the device's leg, a second copy writes after it in the
   same allocation, which the device maps once. The host's staging memory starts
   empty. *)
let test_packed () =
  Rig.free_cache Rig.host;
  let d, p = hidden "copy:staging-packed" in
  let dsts = [ B.create d 64; B.create d 64 ] in
  P.gate p;
  let copier c dst =
    Thread.create (fun () -> B.copy ~src:(off_page 64 c) ~dst) ()
  in
  let first = copier 'a' (List.nth dsts 0) in
  Support.await "the first copy waiting" (fun () -> P.sleepers p = 1);
  let second = copier 'b' (List.nth dsts 1) in
  Support.await "both copies waiting" (fun () -> P.sleepers p = 2);
  P.open_gate p;
  Thread.join first;
  Thread.join second;
  equal ~msg:"allocations mapped" int 1 (staging_mapped p);
  equal ~msg:"the copies' bytes" (list string)
    [ String.make 64 'a'; String.make 64 'b' ]
    (List.map contents dsts)

(* A piece the host writes is never packed behind a device's queued write into
   the same allocation: the drain of an upload to [g] follows no work of [s],
   whose copy to [d] wrote the allocation the upload could share, so [s]'s loss
   leaves [g] working and the upload lands. *)
let test_no_chain () =
  let open_ ?answer name =
    P.open_ ~host_visible:false ~peers:false ~waits_on:[ `Host ] ?answer name
  in
  let s, ps = open_ ~answer:`Unknown "copy:chain-s" in
  let d, _ = open_ "copy:chain-d" and g, pg = open_ "copy:chain-g" in
  let src = on s 's' in
  ignore (B.submit_copy ~src ~dst:(B.create d 64) [| B.row 64 |] : Rig.Point.t);
  let dst = B.create g 64 in
  let pt = B.submit_copy ~src:(filled 64 'g') ~dst [| B.row 64 |] in
  P.fault ps "the test's fault";
  (try B.wait src Rig.Read_write with Rig.Lost _ -> ());
  equal ~msg:"the source's device" (option string) (Some "the test's fault")
    (Rig.lost s);
  ignore (P.run pg : int);
  equal ~msg:"the upload's device" (option string) None (Rig.lost g);
  equal ~msg:"the upload reached" bool true (Rig.Point.reached pt);
  equal ~msg:"the upload's bytes" string (String.make 64 'g') (contents dst)

(* A copy from one device to several others through staging memory, which none
   maps of another, writes an allocation whose work is done for each, or a new
   one: the source's legs follow none of the destinations' drains, so its device
   runs them all with the destinations' still queued, and no submit waits on the
   host. *)
let test_fan_out () =
  let open_ name =
    P.open_ ~host_visible:false ~peers:false ~waits_on:[ `Host ] name
  in
  let a, pa = open_ "copy:fan-src" in
  let src = on a 'o' in
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

(* A staged copy of three pieces through a device's own staging memory holds two
   allocations, which take the pieces in turn. *)
let test_two_allocations () =
  let d, p = staging_device "copy:large" in
  let want = String.make three_pieces 'l' in
  let dst = B.create d three_pieces in
  Rig.Point.wait
    (B.submit_copy ~src:(B.of_string want) ~dst [| B.row three_pieces |]);
  equal ~msg:"allocations made" int 2 (pinned p);
  lands ~msg:"the bytes" want (contents dst)

(* Three copies at once from [d], whose memory the host does not address, to
   host memory off a page, held at [p]'s gate until all three wait there for
   [d]'s legs into staging memory: each makes an allocation, since a device
   writes only an allocation whose work is done. Is the copies' destinations. *)
let staging_burst d p =
  let srcs = List.map (on d) [ 'a'; 'b'; 'c' ] in
  let dsts = List.map (fun _ -> off_page 64 '\000') srcs in
  P.gate p;
  let copier src dst = Thread.create (fun () -> B.copy ~src ~dst) () in
  let copiers = List.map2 copier srcs dsts in
  Support.await "three copies waiting" (fun () -> P.sleepers p = 3);
  P.open_gate p;
  List.iter Thread.join copiers;
  dsts

let burst_landed dsts =
  equal ~msg:"the copies' bytes" (list string)
    (List.map (String.make 64) [ 'a'; 'b'; 'c' ])
    (List.map bytes dsts)

(* Staging memory keeps what concurrent copies made for later copies, through
   collections, and returns it at [Rig.free_cache]: three copies at once hold
   three allocations, of at most 64 MiB for each copy. *)
let test_kept () =
  let d, p = staging_device "copy:staging-kept" in
  let dsts = staging_burst d p in
  let made = P.allocated p B.Pinned in
  equal ~msg:"staging allocations" int 3 (pinned p);
  at_most ~msg:"staging bytes" int ~than:(3 * (64 lsl 20)) made;
  Gc.full_major ();
  Gc.full_major ();
  ignore (B.create d 0);
  equal ~msg:"pinned bytes once they returned" int made (P.allocated p B.Pinned);
  Rig.free_cache d;
  equal ~msg:"pinned bytes after free_cache" int 0 (P.allocated p B.Pinned);
  burst_landed dsts

(* Under pressure, staging memory no copy holds returns: after a burst of three
   copies through a device whose [Pinned] memory counts in the host's budget, a
   [Pinned] buffer of 64 MiB on it, which a budget short of it by the burst's
   staging memory over what the host holds admits once that is gone, is made,
   and is all the [Pinned] memory left. The host's own cache is empty first. *)
let test_pressure () =
  (* A host allocation a budget of 0 refuses runs every round of reclamation
     first, so nothing earlier tests left in the host's budget is still there to
     return in place of the burst's staging memory. *)
  with_budget Rig.host 0 (fun () ->
      raises_match ~msg:"a refusal under a budget of 0"
        (function Rig.Out_of_memory _ -> true | _ -> false)
        (fun () -> B.create Rig.host 64));
  let d, p = staging_device "copy:staging-pressure" in
  let dsts = staging_burst d p in
  let staged = P.allocated p B.Pinned in
  let n = 64 lsl 20 in
  Gc.full_major ();
  Rig.free_cache Rig.host;
  let room =
    Rig.allocated Rig.host + Rig.allocation ~memory:Pinned d n - staged
  in
  let b = with_budget Rig.host room (fun () -> B.create ~memory:Pinned d n) in
  equal ~msg:"pinned bytes" int n (P.allocated p B.Pinned);
  ignore (Sys.opaque_identity b);
  burst_landed dsts

(* Copies several at a time over and over keep their staging memory. Two domains
   copy at once from [d], whose memory the host does not address, to host memory
   off a page, each held at [p]'s gate until both wait there for [d]'s leg into
   an allocation of the host's: each takes one, since a device writes only an
   allocation whose work is done. Eight copies in turn then put their bytes in
   one, passing the other by, and the next pair takes both again. [d] maps each
   allocation once, at its first use: after the first round, no round maps a new
   one. The host's staging memory starts empty. *)
let test_rounds_keep () =
  Rig.free_cache Rig.host;
  let d, p = hidden "copy:staging-pairs" in
  let pair () =
    let srcs = [ on d 'a'; on d 'b' ] in
    let dsts = List.map (fun _ -> off_page 64 '\000') srcs in
    P.gate p;
    let copier src dst = Domain.spawn (fun () -> B.copy ~src ~dst) in
    let copiers = List.map2 copier srcs dsts in
    Support.await "two copies waiting" (fun () -> P.sleepers p = 2);
    P.open_gate p;
    List.iter Domain.join copiers;
    equal ~msg:"the pair's bytes" (list string)
      [ String.make 64 'a'; String.make 64 'b' ]
      (List.map bytes dsts)
  in
  let round () =
    pair ();
    String.iter
      (fun c -> B.copy ~src:(off_page 64 c) ~dst:(B.create d 64))
      "stuvwxyz";
    ignore (B.create Rig.host 0)
  in
  round ();
  let first = staging_mapped p in
  for _ = 1 to 3 do
    round ()
  done;
  equal ~msg:"allocations mapped after the first round" int first
    (staging_mapped p)

(* A copy the host writes waits for the work of an allocation of 32 MiB no copy
   holds, rather than make another, where no such allocation's work is done.
   Copies made allocations and [Rig.free_cache] returned them; then one copy of
   32 MiB from host memory off a page into [d], whose legs [d] runs only once
   the host sleeps at its gate, makes an allocation, and a second copy on
   another domain waits there for that allocation's work, mapping nothing more.
   The first copy's bytes land before the second writes the allocation. *)
let test_waits_for_allocation () =
  let d, p = hidden "copy:staging-waits" in
  Rig.free_cache Rig.host;
  burst_landed (staging_burst d p);
  Rig.free_cache Rig.host;
  let before = staging_mapped p in
  let first = B.create d half and second = B.create d half in
  P.gate p;
  let pt = B.submit_copy ~src:(off_page half 'f') ~dst:first [| B.row half |] in
  let returned = Atomic.make false in
  let copier =
    Domain.spawn (fun () ->
        let src = off_page half 's' in
        let pt = B.submit_copy ~src ~dst:second [| B.row half |] in
        Atomic.set returned true;
        pt)
  in
  Support.await "the second copy waiting or returned" (fun () ->
      P.sleepers p = 1 || Atomic.get returned);
  let waiting = P.sleepers p and mapped = staging_mapped p - before in
  P.open_gate p;
  let pt' = Domain.join copier in
  Rig.Point.wait pt;
  Rig.Point.wait pt';
  equal ~msg:"copies waiting at the gate" int 1 waiting;
  equal ~msg:"allocations mapped" int 1 mapped;
  lands ~msg:"the first copy's bytes" (String.make half 'f') (contents first);
  lands ~msg:"the second copy's bytes" (String.make half 's') (contents second)

(* Copies the host writes ahead of their legs make allocations until one waits,
   and the staging memory then holds an allocation of 32 MiB and at most 64 MiB
   in all: copies of 256 KiB from host memory off a page into [d], whose legs
   [d] runs only once the host sleeps at its gate, go on until one sleeps there.
   The host's staging memory starts empty. *)
let test_grows () =
  let n = 256 lsl 10 in
  let d, p = hidden "copy:staging-grows" in
  Rig.free_cache Rig.host;
  let src = off_page n 'g' and dst = B.create d n in
  let stop = Atomic.make false in
  P.gate p;
  let writer =
    Domain.spawn (fun () ->
        let last = ref None in
        while not (Atomic.get stop) do
          last := Some (B.submit_copy ~src ~dst [| B.row n |])
        done;
        !last)
  in
  Support.await "a copy waiting" (fun () -> P.sleepers p = 1);
  let mapped = P.host_maps p in
  Atomic.set stop true;
  P.open_gate p;
  Option.iter Rig.Point.wait (Domain.join writer);
  mem ~msg:"allocations mapped" int half mapped;
  at_most ~msg:"bytes mapped" int ~than:(64 lsl 20)
    (List.fold_left ( + ) 0 mapped);
  lands ~msg:"the bytes" (String.make n 'g') (contents dst)

(* Copies through one staging memory run concurrently, each through allocations
   of its own: three copies wait on a device at once, and every copy lands once
   the device runs. *)
let test_concurrent () =
  let d, pd = hidden ~peers:false "copy:turns-src" in
  let e, _ = hidden ~peers:false "copy:turns-dst" in
  let srcs = List.map (on d) [ 'a'; 'b'; 'c' ] in
  let dsts = List.map (fun _ -> B.create e 64) srcs in
  P.gate pd;
  let copier src dst = Thread.create (fun () -> B.copy ~src ~dst) () in
  let copiers = List.map2 copier srcs dsts in
  Support.await "three copies waiting" (fun () -> P.sleepers pd = 3);
  P.open_gate pd;
  List.iter Thread.join copiers;
  equal (list string)
    [ String.make 64 'a'; String.make 64 'b'; String.make 64 'c' ]
    (List.map contents dsts)

(* A device lost while it used the staging memory leaves other devices' copies
   through it working. *)
let test_after_loss () =
  let d, p = hidden ~peers:false "copy:staging-lost" in
  let e, _ = hidden ~peers:false "copy:staging-dst-lost" in
  let f, _ = hidden ~peers:false "copy:staging-src-lost" in
  let src = on d 'l' and dst = B.create e 64 in
  P.fail p;
  raises_match ~msg:"the lost device's copy" lost (fun () -> B.copy ~src ~dst);
  B.copy ~src:(on f 'k') ~dst;
  equal ~msg:"another device's copy" string (String.make 64 'k') (contents dst)

(* An io library whose region's reads succeed as many times as it counts, then
   raise [Sys_error], as a file cut short does. *)
module Cut = struct
  type region = int Atomic.t

  let free _ = ()

  let read reads ~at:_ ~dst:_ ~len:_ =
    if Atomic.fetch_and_add reads (-1) <= 0 then raise (Sys_error "cut short")

  let write _ ~at:_ ~src:_ ~len:_ = ()
  let pages _ = None
  let prefetch _ ~at:_ ~len:_ = ()
end

(* [n] bytes of io memory whose reads succeed [reads] times. *)
let cut reads n =
  B.of_io (Support.io "copy:cut")
    (module Cut)
    (Atomic.make reads) ~access:Read_write n

(* How [f ()] ended: windtrap's [raises] lets [Sys.Break] through, as it ends a
   run. *)
let ending f =
  Sys.catch_break true;
  Fun.protect
    ~finally:(fun () -> Sys.catch_break false)
    (fun () ->
      match f () with () -> "returned" | exception Sys.Break -> "broke")

(* A copy through the staging memory gives its allocations back on every exit. A
   copy of three pieces from an io device to a device that maps no host memory
   queues the legs out of both its allocations, then waits for the first one's
   leg before it reads the third piece into it, and that wait is interrupted.
   Two such copies, then a third that lands, all go through the allocations the
   first made. Each broken copy released its claim on its destination. *)
let test_gives_back () =
  let d, p = staging_device "copy:staging-gives-back" in
  let n = (2 * half) + 8 in
  let copy reads dst = B.copy ~src:(cut reads n) ~dst in
  for i = 1 to 2 do
    P.interrupt p;
    let dst = B.create d n in
    equal
      ~msg:(Printf.sprintf "copy %d" i)
      string "broke"
      (ending (fun () -> copy 2 dst));
    ignore
      (require_some
         ~msg:(Printf.sprintf "copy %d's destination taken after it" i)
         (Support.donated dst))
  done;
  let copied = Atomic.make false in
  let third =
    Domain.spawn (fun () ->
        copy 3 (B.create d n);
        Atomic.set copied true)
  in
  Support.await "the third copy" (fun () -> Atomic.get copied);
  Domain.join third;
  equal ~msg:"staging allocations made" int 2
    (List.length (List.filter (fun (k, _, _) -> k = B.Pinned) (P.allocs p)))

(* An io library whose region's reads each run the next of the functions it
   holds, as a callback inside a read does. *)
module Hooked = struct
  type region = (unit -> unit) list Atomic.t

  let free _ = ()

  let rec read hooks ~at ~dst ~len =
    match Atomic.get hooks with
    | [] -> ()
    | f :: rest as all ->
        if Atomic.compare_and_set hooks all rest then f ()
        else read hooks ~at ~dst ~len

  let write _ ~at:_ ~src:_ ~len:_ = ()
  let pages _ = None
  let prefetch _ ~at:_ ~len:_ = ()
end

(* [n] bytes of io memory whose reads run [hooks]. *)
let hooked name hooks n =
  B.of_io (Support.io name) (module Hooked) hooks ~access:Read_write n

(* A staged copy from a callback inside a copy inside a copy, on the thread that
   holds two allocations of the staging memory, makes a third and lands, and so
   do both outer copies. *)
let test_nested () =
  let d, _ = hidden "copy:staging-held" in
  let hooks = Atomic.make [] in
  let src = hooked "copy:staging-held-io" hooks 64 in
  let first = B.create d 64 and second = B.create d 64 in
  let third = B.create d 64 in
  let copied = ref [] in
  let copy name dst () =
    B.copy ~src ~dst;
    copied := name :: !copied
  in
  Atomic.set hooks [ copy "second" second; copy "third" third ];
  copy "first" first ();
  equal ~msg:"the copies that landed, in order" (list string)
    [ "third"; "second"; "first" ]
    (List.rev !copied)

(* Two threads copy from an io device to one device that maps no host memory,
   and each read of the io device copies to that device again once both reads
   have started: each copy is inside its read, and the copies from the reads
   start while both outer copies are under way. Every copy lands. *)
let test_nested_threads () =
  let d, _ = staging_device "copy:staging-nested" in
  let hooks = Atomic.make [] in
  let src = hooked "copy:staging-nested-io" hooks 64 in
  let inside = Atomic.make 0 and landed = Atomic.make 0 in
  let copy () =
    B.copy ~src ~dst:(B.create d 64);
    Atomic.incr landed
  in
  let read () =
    Atomic.incr inside;
    Support.await "both reads started" (fun () -> Atomic.get inside = 2);
    copy ()
  in
  Atomic.set hooks [ read; read ];
  let copiers = List.init 2 (fun _ -> Thread.create copy ()) in
  Support.await "the fourth copy" (fun () -> Atomic.get landed = 4);
  List.iter Thread.join copiers

(* Copies through the host's staging memory from two domains. rig orders two
   domains' work on shared memory only where the program orders the calls
   (rig.mli, Order), so each call copies between buffers of its own: it uploads
   64 bytes of [c] into a new buffer on one of two devices that map none of each
   other's memory, queues a copy of it into a new buffer on the other, and reads
   that back. The calls share only the staging memory their pieces go through,
   so every order gives each call its own bytes, and a call that reads others'
   bytes or fails shows a race in it. *)
let stage_devices =
  lazy
    (Array.init 2 (fun i ->
         fst
           (P.open_ ~host_visible:false ~peers:false ~runs:`Itself
              (Printf.sprintf "copy:domains-%d" i))))

(* The devices, opened by the first call of either domain: a lazy value forced
   on two domains at once raises [Lazy.Undefined]. *)
let opening = Mutex.create ()
let devices () = Mutex.protect opening (fun () -> Lazy.force stage_devices)
let byte = Gen.with_pp Format.pp_print_char (Gen.char_range 'a' 'z')
let device = Gen.with_pp Format.pp_print_int (Gen.int_range 0 1)

let domain_round_trip c i =
  let devices = devices () in
  let a = B.create devices.(i) 64 and b = B.create devices.(1 - i) 64 in
  ignore (B.submit_copy ~src:(filled 64 c) ~dst:a [| B.row 64 |] : Rig.Point.t);
  ignore (B.submit_copy ~src:a ~dst:b [| B.row 64 |] : Rig.Point.t);
  contents b

let domain_commands =
  [
    command "round trip"
      (byte @-> device @-> returns string)
      (fun c _ -> String.make 64 c)
      domain_round_trip;
  ]

(* Read claims *)

(* A transfer through staging memory holds a read claim on its buffers until it
   returns, so a take on another domain meanwhile is refused. [take_during p b
   transfer taker] runs [transfer] on another domain until it waits at [p]'s
   gate for a leg, with legs still to run, then takes [b]: a take that succeeds
   gives [taker] the buffer to write in place. It is whether the take succeeded,
   and how [transfer] ended. *)
let take_during p b transfer taker =
  P.gate p;
  let ended =
    Domain.spawn (fun () ->
        match transfer () with
        | () -> "returned"
        | exception e -> Printexc.to_string e)
  in
  Support.await "the transfer waiting for a leg" (fun () -> P.sleepers p > 0);
  let got = Support.donated b in
  Option.iter taker got;
  P.open_gate p;
  let taken = if Option.is_some got then "taken" else "refused" in
  (taken, Domain.join ended)

(* The host's leg of the third piece reads the source, which a taker would
   write, after the device ran the first's leg out of the same allocation. *)
let test_source_claimed () =
  let d, p = staging_device "copy:staged-source" in
  let want = String.make three_pieces 's' in
  let src = B.of_string want and dst = B.create d three_pieces in
  let taken, ended =
    take_during p src (fun () -> B.copy ~src ~dst) (fun b -> B.fill b 'X')
  in
  equal ~msg:"a take during the copy" string "refused" taken;
  equal ~msg:"the copy" string "returned" ended;
  lands ~msg:"the source's bytes" want (contents dst)

(* The host's leg writes the destination, which a taker would write too. *)
let test_destination_claimed () =
  let d, p = staging_device "copy:staged-destination" in
  let src = B.create d 64 in
  B.fill src 's';
  let dst = filled 64 'd' in
  let taken, ended =
    take_during p dst (fun () -> B.copy ~src ~dst) (fun b -> B.fill b 'X')
  in
  equal ~msg:"a take during the copy" string "refused" taken;
  equal ~msg:"the copy" string "returned" ended;
  equal ~msg:"the source's bytes" string (String.make 64 's') (contents dst)

(* A blit's legs into the buffer hold it from a take until they ran. *)
let test_blit_claimed () =
  let d, p = staging_device "copy:staged-blit" in
  let s = String.make three_pieces 's' and b = B.create d three_pieces in
  let taken, ended =
    take_during p b (fun () -> B.blit_from_string s 0 b 0 three_pieces) ignore
  in
  equal ~msg:"a take during the blit" string "refused" taken;
  equal ~msg:"the blit" string "returned" ended;
  lands ~msg:"the string's bytes" s (contents b)

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

let side_pattern seed =
  String.init side (fun i -> Char.chr ((seed + (i * 7)) land 255))

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

(* An io library over host memory whose pages it never gives, so the io device's
   reads and writes move every byte. *)
module Store = struct
  type region = B.t

  let free _ = ()
  let base r = Support.Reader.host r

  let read r ~at ~dst ~len =
    Array.iteri
      (fun i at -> Support.move ~dst:dst.(i) ~src:(base r + at) len.(i))
      at

  let write r ~at ~src ~len = Support.move ~dst:(base r + at) ~src len
  let pages _ = None
  let prefetch _ ~at:_ ~len:_ = ()
end

(* A buffer of [io] over [n] bytes of a new region of [Store]. *)
let stored io n =
  let r = B.create Rig.host (Int.max n B.paged_from) in
  B.of_io io (module Store) r ~access:Read_write n

(* The routes a copy takes, each with how it makes its two buffers. *)
type route = { name : string; make : unit -> B.t * B.t }

let routes =
  lazy
    (let d, _ = hidden "copy:box-device" in
     let e, _ = hidden ~peers:false "copy:box-peer" in
     let f, _ = hidden ~peers:false "copy:box-peer-2" in
     let own name =
       P.open_ ~maps_host:false ~peers:false ~host_visible:false name
     in
     let g, _ = own "copy:box-own" and h, _ = own "copy:box-own-2" in
     let m = memory "copy:box-memory" in
     let io = Support.io "copy:box-io" in
     let on a b () = (B.create a side, B.create b side) in
     [
       { name = "host"; make = on Rig.host Rig.host };
       { name = "a device's copy engine"; make = on d d };
       { name = "a software device"; make = on m m };
       { name = "staged into a device"; make = on Rig.host d };
       { name = "staged out of a device"; make = on d Rig.host };
       { name = "staged between two devices"; make = on e f };
       { name = "between two devices' own staging"; make = on g h };
       {
         name = "out of an io device";
         make = (fun () -> (stored io side, B.create Rig.host side));
       };
       {
         name = "into an io device";
         make = (fun () -> (B.create Rig.host side, stored io side));
       };
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
         && ((x.rows - 1) * x.dst_pitch) + x.width + x.dst_offset = side)
       boxes);
  let src, dst = route.make () in
  let s = side_pattern seed and t = side_pattern (seed + 101) in
  B.blit_from_string s 0 src 0 side;
  B.blit_from_string t 0 dst 0 side;
  Rig.Point.wait (B.submit_copy ~src ~dst (Array.of_list boxes));
  equal string (reference s t boxes) (contents dst)

(* A box's fields at and past each bound of fitting, against the definition:
   [submit_copy] refuses, with nothing queued, every box that does not fit 64
   bytes on either side, and copies every one that does. *)
let fits (x : B.box) =
  (* [offset + (rows - 1) * pitch + width <= 64] where the box has bytes, in
     terms that cannot overflow: a pitch of at least a positive width is
     positive. *)
  let last_row_within offset pitch =
    x.width = 0 || x.rows = 0
    || x.width <= 64
       && offset <= 64 - x.width
       && x.rows - 1 <= (64 - x.width - offset) / pitch
  in
  x.src_offset >= 0 && x.dst_offset >= 0 && x.width >= 0 && x.rows >= 0
  && x.src_pitch >= x.width && x.dst_pitch >= x.width
  && last_row_within x.src_offset x.src_pitch
  && last_row_within x.dst_offset x.dst_pitch

let fit_box =
  let open Gen in
  let field = frequency [ (6, int_range (-1) 66); (1, constant max_int) ] in
  let small = frequency [ (6, int_range (-1) 9); (1, constant max_int) ] in
  let+ src_offset = field
  and+ dst_offset = field
  and+ width = small
  and+ rows = small
  and+ src_pitch = small
  and+ dst_pitch = small in
  { B.src_offset; dst_offset; width; rows; src_pitch; dst_pitch }

let fit_box = Gen.with_pp pp_box fit_box
let box = Testable.make ~pp:pp_box ~equal:( = )
let refusals = lazy (hidden "copy:refusals")

let fit_law x =
  cover "a box that fits" (fits x);
  cover "a box of bytes that fits" (fits x && x.width > 0 && x.rows > 0);
  cover "a box that does not fit" (not (fits x));
  cover "a field at max_int"
    (List.mem max_int
       [ x.src_offset; x.dst_offset; x.width; x.rows; x.src_pitch; x.dst_pitch ]);
  let d, p = Lazy.force refusals in
  let src = B.create d 64 and dst = B.create d 64 in
  let before = P.submits p in
  match B.submit_copy ~src ~dst [| B.row 8; x |] with
  | pt ->
      Rig.Point.wait pt;
      satisfies ~claim:"fits 64 bytes" box fits x
  | exception Invalid_argument _ ->
      satisfies ~claim:"does not fit 64 bytes" box (fun x -> not (fits x)) x;
      equal ~msg:"submissions queued" int before (P.submits p)

(* The boxes past one bound each, and the boxes of no bytes whose fields would
   overflow, from the old table of refusals. *)
let fit_examples =
  let box ?(src_offset = 0) ?(dst_offset = 0) ?(src_pitch = 8) ?(dst_pitch = 8)
      width rows =
    { B.src_offset; dst_offset; width; rows; src_pitch; dst_pitch }
  in
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
    box ~src_offset:max_int ~src_pitch:max_int 0 3;
    box 8 8;
  ]

(* submit_copy *)

(* A copy a device runs returns before the device has run it: its point is not
   reached and the host never slept, and the bytes are there once the point
   is. *)
let test_direct_queued () =
  let d, p = hidden "copy:direct-queued" in
  let src = on d 'q' and dst = B.create d 64 in
  let before = P.sleeps p in
  let pt = B.submit_copy ~src ~dst [| B.row 64 |] in
  equal ~msg:"sleeps" int before (P.sleeps p);
  equal ~msg:"reached on return" bool false (Rig.Point.reached pt);
  Rig.Point.wait pt;
  equal ~msg:"the bytes" string (String.make 64 'q') (contents dst)

(* A copy from host memory through staging memory returns once the host wrote
   its piece, with the device's leg out of it still queued. *)
let test_staged_queued () =
  let d, p = hidden "copy:staged-queued" in
  let dst = B.create d 64 in
  let before = P.sleeps p in
  let pt = B.submit_copy ~src:(filled 64 's') ~dst [| B.row 64 |] in
  equal ~msg:"sleeps" int before (P.sleeps p);
  equal ~msg:"reached on return" bool false (Rig.Point.reached pt);
  Rig.Point.wait pt;
  equal ~msg:"the bytes" string (String.make 64 's') (contents dst)

(* A copy the host makes is done when it returns, and its point, the host's, is
   reached. *)
let test_host_done () =
  let src = filled 64 'h' and dst = filled 64 '\000' in
  let pt = B.submit_copy ~src ~dst [| B.row ~dst_offset:8 32 |] in
  equal ~msg:"the point's device"
    (option (Testable.make ~pp:Rig.pp ~equal:Rig.equal))
    (Some Rig.host) (Rig.Point.device pt);
  equal ~msg:"reached" bool true (Rig.Point.reached pt);
  equal ~msg:"the bytes" string
    (String.make 8 '\000' ^ String.make 32 'h' ^ String.make 24 '\000')
    (bytes dst)

(* A copy by a device into a host buffer passes through staging memory, so the
   host waits for its bytes: [submit_copy] returns once they are there. Into a
   borrow of the buffer it returns once the copy is queued. *)
let test_into_host_waits () =
  let d, p = hidden "copy:into-host" in
  let n = B.paged_from in
  let src = holding d (String.make n 'w') in
  let h = filled n '0' in
  ignore (B.submit_copy ~src ~dst:h [| B.row n |] : Rig.Point.t);
  equal ~msg:"a host buffer: reached at the return" bool true
    (B.reached h Read_write);
  equal ~msg:"a host buffer: the bytes" string (String.make n 'w') (bytes h);
  let pt =
    B.submit_copy ~src ~dst:(require_some (B.borrow d h)) [| B.row n |]
  in
  equal ~msg:"a borrow: queued, unreached, at the return" (pair int bool)
    (1, false)
    (P.queued p, Rig.Point.reached pt);
  ignore (P.run p);
  Rig.Point.wait pt

(* A queued copy follows the points of its waits, also of work on memory it does
   not name: on a device whose queue waits for no other device's, its submit
   waits for them on the host, so the waited copy is done once it returns,
   though Polled runs work only when the host sleeps on it. *)
let test_waits () =
  let d, pd = hidden "copy:waits-src" in
  let e, _ = hidden "copy:waits-dst" in
  let first =
    B.submit_copy ~src:(on d 'w') ~dst:(B.create d 64) [| B.row 64 |]
  in
  let before = P.sleeps pd in
  let pt =
    B.submit_copy ~waits:[| first |] ~src:(B.create e 64) ~dst:(B.create e 64)
      [| B.row 64 |]
  in
  greater ~msg:"sleeps on the wait's device" int ~than:before (P.sleeps pd);
  equal ~msg:"the wait reached at the return" bool true
    (Rig.Point.reached first);
  Rig.Point.wait pt

(* Many runs in one call, whose destinations overlap, land in order: a byte two
   runs write holds the later run's, and one submission runs them all. *)
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
  equal ~msg:"the bytes" string
    (reference s (String.make 4096 '.') runs)
    (contents dst)

(* A copy of more boxes than the driver takes in one submission, or of more
   rows, lands as one would, in several submissions queued in order, and its
   point is the last one's: reached once every byte is there. *)
let test_split () =
  let d, p = P.open_ ~host_visible:false ~capacity:4 "copy:split" in
  let s = String.init 4096 (fun i -> Char.chr (i land 255)) in
  let blank = String.make 4096 '.' in
  let lands what boxes ~submissions =
    let src = holding d s and dst = holding d blank in
    let before = P.submits p in
    let pt = B.submit_copy ~src ~dst (Array.of_list boxes) in
    equal ~msg:(what ^ ": submissions") int (before + submissions) (P.submits p);
    equal ~msg:(what ^ ": reached on return") bool false (Rig.Point.reached pt);
    Rig.Point.wait pt;
    equal ~msg:(what ^ ": queued") int 0 (P.queued p);
    equal ~msg:what string (reference s blank boxes) (contents dst)
  in
  (* Ten runs whose destinations overlap, so that a later box's byte wins, at
     most four per submission: halved to 5 and 5, then to 2, 3, 2 and 3. *)
  let runs =
    List.init 10 (fun i -> B.row ~src_offset:(i * 37) ~dst_offset:(i * 11) 64)
  in
  lands "boxes" runs ~submissions:4;
  P.limit_rows p 30;
  let box =
    {
      B.src_offset = 1;
      dst_offset = 3;
      width = 7;
      rows = 100;
      src_pitch = 40;
      dst_pitch = 33;
    }
  in
  (* 100 rows halve to 50, then 25 each: four submissions. *)
  lands "rows" [ box ] ~submissions:4;
  P.limit_rows p 0

(* A box larger than an allocation of staging memory goes through it in pieces
   of whole rows, three rows of 9 MiB and then the fourth, and a row larger than
   an allocation in ranges of its bytes. *)
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
  let s = pattern n in
  let dst = B.create d n in
  Rig.Point.wait (B.submit_copy ~src:(B.of_string s) ~dst [| rows; wide |]);
  let copied = (4 * width) + half + 3 in
  let want =
    String.init copied (fun i ->
        if i < 4 * width then s.[7 + (i / width * (width + 9)) + (i mod width)]
        else s.[5 + i - (4 * width)])
  in
  lands want (String.sub (contents dst) 0 copied)

(* A copy refuses buffers that share a byte, also for a box of no bytes. *)
let test_shared_byte () =
  let b = B.create Rig.host 100 in
  raises_match Exn.invalid_arg (fun () ->
      B.submit_copy
        ~src:(B.view b ~first:0 ~length:50)
        ~dst:(B.view b ~first:10 ~length:50)
        [| B.row 0 |])

(* copy *)

(* A copy on a Polled device runs only when a wait reaches its sleep: the copy's
   own wait for its point, so the copy returns with nothing queued. *)
let test_copy_waits () =
  let d, p = P.open_ "copy:copy-polled" in
  let on = B.create d 4096 in
  B.copy ~src:(filled 4096 'w') ~dst:on;
  let back = filled 4096 '0' in
  B.copy ~src:on ~dst:back;
  equal ~msg:"the bytes back" string (String.make 4096 'w') (bytes back);
  equal ~msg:"work queued" int 0 (P.queued p)

let test_copy_lengths () =
  raises_match Exn.invalid_arg (fun () ->
      B.copy ~src:(filled 100 'x') ~dst:(filled 99 'y'))

let test_copy_shared_byte () =
  let b = filled 100 'x' in
  raises_match Exn.invalid_arg (fun () ->
      B.copy
        ~src:(B.view b ~first:0 ~length:50)
        ~dst:(B.view b ~first:10 ~length:50))

(* A copy between host buffers allocates nothing, warm: 1,000 copies take no
   word of the minor heap. rig.mli promises it in native code, which this
   suite is compiled to. *)
let test_host_allocates_nothing () =
  let src = filled 4096 'a' and dst = filled 4096 'b' in
  B.copy ~src ~dst;
  let before = Gc.minor_words () in
  for _ = 1 to 1000 do
    B.copy ~src ~dst
  done;
  equal ~msg:"words" int 0 (int_of_float (Gc.minor_words () -. before))

(* The blits and the fills *)

(* A blit: the bytes of the buffer under it, those of a view [first] bytes into
   it, its source or destination, and its ranges, valid or not. *)
type place = Host_memory | Unaddressed | Far

let place_name = function
  | Host_memory -> "host memory"
  | Unaddressed -> "memory the host does not address"
  | Far -> "another machine's memory"

type blit = {
  place : place;
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
    "%s: a view [%d,+%d] of %d bytes, %S, %d bytes from %d to byte %d of the \
     view"
    (place_name c.place) c.first c.length c.size c.outside c.n c.at c.inside

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
  let place =
    of_list
      ~pp:(fun ppf p -> Format.pp_print_string ppf (place_name p))
      [ Host_memory; Unaddressed; Far ]
  in
  with_pp pp_blit
    (bind place (fun place ->
         bind (int_range 0 40) (fun size ->
             bind (int_range 0 size) (fun first ->
                 bind
                   (int_range 0 (size - first))
                   (fun length ->
                     bind
                       (string_of ~size:(int_range 0 24) char)
                       (fun outside ->
                         map
                           (fun (at, inside, n) ->
                             {
                               place;
                               size;
                               first;
                               length;
                               outside;
                               at;
                               inside;
                               n;
                             })
                           (ranges outside length)))))))

(* Per place, a blit of no bytes into a view and a fill of an empty view. *)
let blit_examples =
  List.concat_map
    (fun place ->
      [
        {
          place;
          size = 8;
          first = 2;
          length = 4;
          outside = "ab";
          at = 1;
          inside = 2;
          n = 0;
        };
        {
          place;
          size = 8;
          first = 3;
          length = 0;
          outside = "";
          at = 0;
          inside = 0;
          n = 0;
        };
      ])
    [ Host_memory; Unaddressed; Far ]

let valid c =
  c.at >= 0 && c.n >= 0
  && c.at + c.n <= String.length c.outside
  && c.inside >= 0
  && c.inside + c.n <= c.length

let cover_blit c =
  cover (place_name c.place) true;
  cover "a blit of bytes" (valid c && c.n > 0);
  cover "a blit of no bytes" (valid c && c.n = 0);
  cover (place_name c.place ^ ": a blit of no bytes") (valid c && c.n = 0);
  cover "a range outside" (not (valid c));
  cover "a view into its buffer" (c.first > 0 && valid c && c.n > 0)

(* The device whose memory the host does not address, and another machine's
   device whose memory its driver gives a host address. *)
let unaddressed = lazy (fst (hidden "copy:blits"))

let far =
  lazy
    (let machine = Support.machine "copy-blits" in
     require_ok ~pp:Format.pp_print_string
       (Rig.open_ ~machine
          {
            name = "copy:blits-far";
            driver = (module P);
            make = (fun ~word -> Ok (P.bind (P.make ()) ~word));
          }))

(* A buffer of [place] holding [s], made and read back by copies. *)
let placed place s =
  let on d =
    let b = B.create (Lazy.force d) (String.length s) in
    B.copy ~src:(B.of_string s) ~dst:b;
    b
  in
  match place with
  | Host_memory -> B.of_string s
  | Unaddressed -> on unaddressed
  | Far -> on far

(* [b]'s bytes, copied into host memory. *)
let copied_out b =
  let back = B.create Rig.host (B.length b) in
  B.copy ~src:b ~dst:back;
  bytes back

let letters n = String.init n (fun i -> Char.chr (97 + (i mod 26)))

(* [blit_from_string] writes the view's bytes the string gives and no other, and
   raises, changing nothing, on a range outside either. *)
let law_from_string c =
  cover_blit c;
  let model = Bytes.of_string (letters c.size) in
  let b = placed c.place (Bytes.to_string model) in
  let v = B.view b ~first:c.first ~length:c.length in
  let outcome =
    match B.blit_from_string c.outside c.at v c.inside c.n with
    | () ->
        Bytes.blit_string c.outside c.at model (c.first + c.inside) c.n;
        "copied"
    | exception Invalid_argument _ -> "refused"
  in
  equal ~msg:"the outcome" string
    (if valid c then "copied" else "refused")
    outcome;
  equal ~msg:"the buffer's bytes" string (Bytes.to_string model) (copied_out b)

(* [blit_to_bytes] reads the view's bytes into the bytes' range and writes no
   other, and raises, changing nothing, on a range outside either. *)
let law_to_bytes c =
  cover_blit c;
  let whole = letters c.size in
  let v = B.view (placed c.place whole) ~first:c.first ~length:c.length in
  let model = Bytes.of_string c.outside and got = Bytes.of_string c.outside in
  let outcome =
    match B.blit_to_bytes v c.inside got c.at c.n with
    | () ->
        Bytes.blit_string whole (c.first + c.inside) model c.at c.n;
        "copied"
    | exception Invalid_argument _ -> "refused"
  in
  equal ~msg:"the outcome" string
    (if valid c then "copied" else "refused")
    outcome;
  equal ~msg:"the bytes" string (Bytes.to_string model) (Bytes.to_string got)

(* [fill] writes every byte of the view and no other. *)
let law_fill c =
  cover (place_name c.place) true;
  cover "a fill of bytes" (c.length > 0);
  cover "a fill of no bytes" (c.length = 0);
  cover (place_name c.place ^ ": a fill of no bytes") (c.length = 0);
  cover "a view into its buffer" (c.first > 0 && c.length > 0);
  let model = Bytes.of_string (letters c.size) in
  let b = placed c.place (Bytes.to_string model) in
  B.fill (B.view b ~first:c.first ~length:c.length) 'z';
  Bytes.fill model c.first c.length 'z';
  equal string (Bytes.to_string model) (copied_out b)

(* Blits and a fill of three pieces of the staging memory keep every byte in its
   place: each piece's bytes differ from the others'. *)
let test_blits_staged_large () =
  let d, _ = hidden "copy:blits-large" in
  let n = (2 * half) + 3 in
  let data = pattern n in
  let b = B.create d n in
  B.blit_from_string data 0 b 0 n;
  lands ~msg:"there and back" data (contents b);
  B.fill b 'q';
  lands ~msg:"the fill" (String.make n 'q') (contents b)

(* A blit from a string waits for every use of the buffer, its own device's
   included; a blit to bytes for its last write only. *)
let test_blit_waits () =
  let d, p = P.open_ "copy:blit-waits" in
  let h = B.of_string (String.make B.paged_from 'a') in
  let on_d = require_some (B.borrow d h) in
  let touch access = Sub.make d [| Support.touching [| access |] |] in
  ignore (submit ~args:one (touch Sub.read) ~buffers:[| on_d |]);
  let got = Bytes.create 4 in
  B.blit_to_bytes h 0 got 0 4;
  equal ~msg:"a read left queued" int 1 (P.queued p);
  equal ~msg:"bytes" string "aaaa" (Bytes.to_string got);
  B.blit_from_string "bbbb" 0 h 0 4;
  equal ~msg:"the read ran first" int 0 (P.queued p);
  ignore (submit ~args:one (touch Sub.write) ~buffers:[| on_d |]);
  B.blit_to_bytes h 0 got 0 4;
  equal ~msg:"the write ran first" int 0 (P.queued p)

(* submit_fill *)

(* A fill a device's copy engine runs returns before it has run. *)
let test_fill_queued () =
  let d, p = hidden "copy:fill" in
  let b = B.create d 64 in
  let before = P.sleeps p in
  let pt = B.submit_fill b 'f' in
  equal ~msg:"sleeps" int before (P.sleeps p);
  equal ~msg:"reached on return" bool false (Rig.Point.reached pt);
  Rig.Point.wait pt;
  equal ~msg:"the bytes" string (String.make 64 'f') (contents b)

(* A fill follows the points of its waits, also of work on memory it does not
   name, as a copy does. *)
let test_fill_waits () =
  let d, pd = hidden "copy:fill-waits-src" in
  let e, _ = hidden "copy:fill-waits" in
  let first =
    B.submit_copy ~src:(on d 'w') ~dst:(B.create d 64) [| B.row 64 |]
  in
  let before = P.sleeps pd in
  let pt = B.submit_fill ~waits:[| first |] (B.create e 64) 'f' in
  greater ~msg:"sleeps on the wait's device" int ~than:before (P.sleeps pd);
  equal ~msg:"the wait reached at the return" bool true
    (Rig.Point.reached first);
  Rig.Point.wait pt

(* A fill of memory the host does not address is one fill of its byte on the
   device's copy engine, which reads no memory, and stages nothing: Polled runs
   one copy part, a fill of one row of the view's bytes. An empty fill runs
   none. *)
let test_fill_on_engine () =
  let d, p = hidden "copy:fill-engine" in
  let b = B.create d 4099 in
  B.fill (B.view b ~first:3 ~length:4096) 'f';
  B.fill (B.view b ~first:0 ~length:0) 'e';
  let side = function `None -> "none" | `Src -> "src" | `Dst -> "dst" in
  let fill =
    Testable.structural ~pp:(fun ppf (local, device_memory, w, r, c) ->
        Format.fprintf ppf
          "{ local = %s; device_memory = %b; width = %d; rows = %d; fill = %a }"
          local device_memory w r
          (Format.pp_print_option Format.pp_print_char)
          c)
  in
  (* rig_edge.h: a fill's [device_memory] is its one buffer's, here the device's
     Device memory, and it names no memory of this process's. *)
  equal ~msg:"the copy parts Polled ran" (list fill)
    [ ("none", true, 4096, 1, Some 'f') ]
    (List.map
       (fun (c : P.copy) ->
         (side c.local, c.device_memory, c.width, c.rows, c.fill))
       (P.copies p));
  equal ~msg:"the view's bytes" string (String.make 4096 'f')
    (String.sub (contents b) 3 4096);
  Rig.close d

let () =
  exit
  @@ run "rig.copy"
       [
         group "routes"
           [
             test "the host copies between host buffers" test_host_copy;
             test "the host copies to and from a software device"
               test_software_copy;
             test "the host copies for a device that runs no copy"
               test_no_copy_engine;
             test
               "the host copies into and out of a device's borrow of host \
                memory"
               test_borrowed_host;
             test "rig's copies go to the copy engine the facts name"
               test_copy_engine;
             test "the source's device copies to a device of its driver"
               test_peer;
             test
               "a borrow of a peer's memory on a device that runs no copy \
                copies by the peer"
               test_copyless_borrow;
             test
               "another machine's device copies this process's memory as it is"
               test_machines;
             test "a copy between machines with no device to copy raises"
               test_machines_refused;
             test "a copy names a borrow to run directly" test_borrow_direct;
           ];
         group "staging memory"
           [
             test
               "a device copies host memory through staging memory, mapping \
                none"
               test_host_staged;
             test "host memory off a page is copied through staging memory"
               test_off_page_staged;
             test
               "devices that map none of each other's memory copy through \
                staging memory"
               test_unmapped_peers;
             test "a copy larger than an allocation lands every byte"
               test_staged_large;
             test
               "a device that maps no host memory copies to and from the host \
                through its own memory"
               test_own_staging;
             test
               "a borrow of a peer's hidden memory on a device that maps no \
                host memory copies in and out"
               test_own_staging_borrow;
             test
               "a copy between two devices that map no host memory goes \
                through both their staging memory"
               test_own_staging_pair;
             test
               "copies between two devices that map no host memory run \
                concurrently both ways"
               test_own_staging_both_ways;
             test "an allocation is made at the first piece that needs it"
               test_made_at_need;
             cases
               "a device that maps no host memory stages a borrow's copy in \
                its budget, but into Mapped memory"
               ~id:(fun (row, _, _) -> row)
               staged_borrows test_staged_borrow;
             test
               "a copy refused staging memory raises Out_of_memory for the \
                host and gives back what it held"
               test_refused_gives_back;
             test
               "small uploads to several devices share one allocation and \
                never wait"
               test_small_uploads;
             test
               "copies the host stages share an allocation and wait for no work"
               test_packed;
             test "an upload packs behind no device's queued write"
               test_no_chain;
             test
               "a staged fan-out from one device takes an allocation whose \
                work is done for each"
               test_fan_out;
             test "a staged copy of more than 32 MiB holds two allocations"
               test_two_allocations;
             test "staging memory stays until free_cache" test_kept;
             test "staging memory no copy holds returns under pressure"
               test_pressure;
             test "copies several at a time over and over keep their staging"
               test_rounds_keep;
             test
               "a copy the host writes waits for the work of an allocation of \
                32 MiB"
               test_waits_for_allocation;
             test
               "copies the host writes ahead grow the staging memory to 32 MiB \
                and at most 64"
               test_grows;
             test "copies through one staging memory run concurrently"
               test_concurrent;
             test "a loss with the staging memory leaves other copies working"
               test_after_loss;
             test "a staged copy gives its allocations back on every exit"
               test_gives_back;
             test "a staged copy inside two staged copies on one thread lands"
               test_nested;
             test
               "copies from io reads inside staged copies on two threads land"
               test_nested_threads;
             stateful ~count:20 ~steps:0 ~domains:2
               "copies of their own through one staging memory on two domains \
                answer as some order of them"
               domain_commands;
           ];
         group "read claims"
           [
             test "a staged copy holds its source against a take"
               test_source_claimed;
             test "a staged copy holds its destination against a take"
               test_destination_claimed;
             test "a staged blit holds its buffer against a take"
               test_blit_claimed;
           ];
         group "box"
           [
             prop ~count:300 ~examples:fit_examples
               "submit_copy copies a box iff it fits, queueing nothing \
                otherwise"
               fit_box fit_law;
           ];
         group "submit_copy"
           [
             prop ~count:300
               "a copy of boxes lands as the host's copy, on every route" case
               box_law;
             test "a copy a device runs returns before it runs"
               test_direct_queued;
             test "a staged copy returns once the host wrote its piece"
               test_staged_queued;
             test "a copy the host makes returns done, at a point of the host"
               test_host_done;
             test "a copy into a host buffer returns once its bytes are there"
               test_into_host_waits;
             test "a copy follows its waits" test_waits;
             test "many runs in one call land in order, in one submission"
               test_many_runs;
             test
               "a copy larger than a submission the driver takes lands in \
                several"
               test_split;
             test "a box larger than an allocation lands through staging memory"
               test_large_box;
             test "a copy refuses buffers that share a byte" test_shared_byte;
           ];
         group "copy"
           [
             test "a copy returns once the device's work ran" test_copy_waits;
             test "a copy refuses buffers that differ in length"
               test_copy_lengths;
             test "a copy refuses buffers that share a byte"
               test_copy_shared_byte;
             test "a copy between host buffers allocates nothing"
               test_host_allocates_nothing;
           ];
         group "blit_from_string"
           [
             prop ~count:300 ~examples:blit_examples
               "a blit from a string writes its range of the buffer and no \
                other"
               blit law_from_string;
             test "a blit waits as a host access of its kind does"
               test_blit_waits;
             test
               "blits and a fill through several staging pieces keep every byte"
               test_blits_staged_large;
           ];
         group "blit_to_bytes"
           [
             prop ~count:300 ~examples:blit_examples
               "a blit to bytes reads its range of the buffer into its range"
               blit law_to_bytes;
           ];
         group "submit_fill"
           [
             test "a fill a copy engine runs returns before it runs"
               test_fill_queued;
             test "a fill follows its waits" test_fill_waits;
             test
               "a fill of memory the host does not address is one fill on its \
                copy engine"
               test_fill_on_engine;
           ];
         group "fill"
           [
             prop ~count:300 ~examples:blit_examples
               "a fill writes its view of the buffer and no other" blit law_fill;
           ];
       ]
