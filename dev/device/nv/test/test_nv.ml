(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module N = Device_nv
module A = Device_nv_abi
module S = Device_nv_support

let strf = Printf.sprintf
let second = 1_000_000_000
let host = S.host
let address = S.address

let copy g ?after (dst, o) (src, o') n =
  N.part g ~queue:"COPY:0" ?after (`Copy ((dst, o), (src, o'), n))

let compute g ?after ws = N.part g ~queue:"COMPUTE:0" ?after (`Words ws)

(* [f ()], the host word [w] set to [x] after it whether it returns or raises,
   so that work held on [w] ends. *)
let releasing w x f = Fun.protect ~finally:(fun () -> S.set64 (host w) x) f

let room_answer =
  Testable.make
    ~pp:(fun ppf r ->
      Format.pp_print_string ppf
        (match r with
        | `Fits -> "`Fits"
        | `Later -> "`Later"
        | `Never -> "`Never"))
    ~equal:( = )

(* Paths *)

(* A path whose RM is of release [release] and whose every function fails the
   test: [make] must refuse it before calling any. *)
let path release : unit N.path =
  let called what = fail (what ^ " was called") in
  let rm =
    {
      N.release;
      client = 0;
      alloc = (fun ~parent:_ _ _ -> called "rm.alloc");
      control = (fun _ _ _ -> called "rm.control");
      free = (fun ~parent:_ _ -> called "rm.free");
    }
  in
  {
    N.key = Type.Id.make ();
    index = 0;
    rm;
    device = 0;
    subdevice = 0;
    vaspace = 0;
    gpu =
      {
        channel_class = 0;
        compute_class = 0;
        copy_class = 0;
        sm_version = 0;
        gpcs = 1;
        tpcs_per_gpc = 1;
        sms_per_tpc = 1;
        warps_per_sm = 1;
      };
    budget = 0;
    doorbell = 0;
    alloc = (fun _ _ -> called "alloc");
    map_host = (fun _ _ -> called "map_host");
    reaches = (fun _ -> called "reaches");
    map_peer = (fun _ -> called "map_peer");
    free = (fun _ -> called "free");
    register = (fun _ -> called "register");
    unregister = (fun _ -> called "unregister");
  }

(* A path that refuses its [n]th call (an RM object, a control, memory or a
   registration), [0] for none, over host pages for the memory the host
   addresses: what the device holds of it at any time. *)
module Fake = struct
  type t = {
    refuse : int;
    mutable calls : int;
    mutable objects : (int * int) list; (* object, parent *)
    mutable memory : (int * int option * int) list;
    mutable registered : int list;
    mutable wrong : string list; (* what the device gave back wrongly *)
    mutable next : int;
  }

  (* The path's own objects. *)
  let device = 1
  let subdevice = 2
  let vaspace = 3

  let refused f =
    f.calls <- f.calls + 1;
    f.calls = f.refuse

  let fresh f =
    f.next <- f.next + 1;
    f.next

  let rec under f h parent =
    h = parent
    ||
    match List.assoc_opt h f.objects with
    | Some p -> under f p parent
    | None -> false

  let rm f =
    {
      N.release = 615;
      client = 4;
      alloc =
        (fun ~parent _ _ ->
          if refused f then Error "refused"
          else
            let h = fresh f in
            f.objects <- (h, parent) :: f.objects;
            Ok h);
      control = (fun _ _ _ -> if refused f then Error "refused" else Ok ());
      free =
        (fun ~parent:_ h ->
          if not (List.mem_assoc h f.objects) then
            f.wrong <- Printf.sprintf "freed object %d" h :: f.wrong;
          f.objects <- List.filter (fun (o, _) -> not (under f o h)) f.objects;
          Ok ());
    }

  let alloc f kind n =
    if refused f then None
    else
      let bytes = (n + S.page - 1) / S.page * S.page in
      let host = if kind = `Gpu then None else Some (S.pages bytes) in
      let address = fresh f * (1 lsl 24) in
      f.memory <- (address, host, bytes) :: f.memory;
      Some { N.address; host; handle = fresh f; data = () }

  let free f (m : unit N.memory) =
    match List.find_opt (fun (a, _, _) -> a = m.address) f.memory with
    | None -> f.wrong <- Printf.sprintf "freed memory 0x%x" m.address :: f.wrong
    | Some ((_, host, bytes) as x) ->
        f.memory <- List.filter (fun y -> y != x) f.memory;
        Option.iter (fun a -> S.free_pages a bytes) host

  let path f : unit N.path =
    {
      N.key = Type.Id.make ();
      index = 0;
      rm = rm f;
      device;
      subdevice;
      vaspace;
      gpu =
        {
          channel_class = 0xc86f;
          compute_class = 0xc9c0;
          copy_class = 0xc7b5;
          sm_version = 0x809;
          gpcs = 11;
          tpcs_per_gpc = 6;
          sms_per_tpc = 2;
          warps_per_sm = 48;
        };
      budget = 1 lsl 30;
      doorbell = S.pages S.page;
      alloc = alloc f;
      map_host = (fun _ _ -> None);
      reaches = (fun _ -> false);
      map_peer = (fun _ -> None);
      free = free f;
      register =
        (fun c ->
          if refused f then Error "refused"
          else (
            f.registered <- c :: f.registered;
            Ok ()));
      unregister =
        (fun c ->
          if not (List.mem c f.registered) then
            f.wrong <- Printf.sprintf "unregistered %d" c :: f.wrong;
          f.registered <- List.filter (( <> ) c) f.registered;
          Ok ());
    }

  let make refuse =
    {
      refuse;
      calls = 0;
      objects = [];
      memory = [];
      registered = [];
      wrong = [];
      next = 16;
    }
end

(* Whatever call the path refuses, make gives back all it took; a device made
   and stopped keeps only its timeline word. *)
let refused_makes () =
  let rec each n =
    let f = Fake.make n in
    let at = strf "refusing call %d" n in
    (match N.make (Fake.path f) with
    | Error e when f.calls < n -> failf "make refused nothing: %s" e
    | Error _ ->
        equal (list (pair int int)) ~msg:(at ^ ": RM objects") [] f.objects;
        equal int ~msg:(at ^ ": memory") 0 (List.length f.memory);
        equal (list int) ~msg:(at ^ ": registrations") [] f.registered
    | Ok g ->
        N.stop g;
        equal (list (pair int int)) ~msg:(at ^ ": RM objects") [] f.objects;
        equal (list int) ~msg:(at ^ ": memory")
          [ address (N.word g) ]
          (List.map (fun (a, _, _) -> a) f.memory);
        equal (list int) ~msg:(at ^ ": registrations") [] f.registered);
    equal (list string) ~msg:(at ^ ": given back wrongly") [] f.wrong;
    if f.calls >= n then each (n + 1)
  in
  each 1

(* A cubin is placed over a region the caller allocated: neither [image] nor
   [lay] asks the path for anything. *)
let image_asks_nothing () =
  let f = Fake.make 0 in
  let g = require_ok (N.make (Fake.path f)) in
  let bin = S.fixture "kernels_sm89.cubin" in
  let n, lay =
    match require_ok (N.image g bin) with
    | `Place (n, lay) -> (n, lay)
    | `Loaded _ -> fail "an image with nothing to place"
  in
  let r = require_some (N.alloc g `Device n) in
  let calls = f.calls and memory = List.length f.memory in
  let i, _ = lay r in
  equal int ~msg:"the path's calls" calls f.calls;
  equal int ~msg:"the path's memory" memory (List.length f.memory);
  N.unload g i;
  N.free g r;
  N.stop g

let paths =
  group ~timeout:10. "paths"
    [
      test "an NVIDIA display controller is a GPU" (fun () ->
          equal (list bool)
            [ true; true; false; false ]
            (List.map
               (fun (vendor, class_) -> N.is_gpu ~vendor ~class_)
               [
                 (0x10de, 0x030000);
                 (0x10de, 0x030200);
                 (0x10de, 0x040300);
                 (0x1002, 0x030000);
               ]));
      test "make refuses an RM of an unknown release" (fun () ->
          let e = require_error (N.make (path 999)) in
          contains ~sub:"999" e);
      test
        "make refused at any call gives back what it took, and a stopped \
         device keeps only its word"
        refused_makes;
      test "image and lay ask the path for nothing" image_asks_nothing;
    ]

(* Facts *)

let digits s = String.for_all (function '0' .. '9' -> true | _ -> false) s

let facts () =
  S.with_gpu @@ fun g ->
  let arch = N.arch g in
  starts_with ~affix:"sm_" arch;
  equal bool
    ~msg:(strf "%s ends in digits" arch)
    true
    (String.length arch > 3
    && digits (String.sub arch 3 (String.length arch - 3)));
  greater int ~msg:"budget" ~than:0 (N.budget g);
  equal (list string) [ "COMPUTE:0"; "COPY:0" ] (N.queues g);
  equal bool ~msg:"completion is the store" true (N.completion g = `Store);
  equal (list bool) [ true; false; true ]
    (List.map (N.waits_on g) [ `Store; `Object; `Host ]);
  equal bool ~msg:"submit returns" true (N.blocks g = `Returns);
  let w = N.word g in
  equal nativeint ~msg:"the word's handle is its address"
    (Nativeint.of_int (address w))
    (N.handle w);
  equal int ~msg:"the word starts at 0" 0 (S.get64 (host w));
  equal int ~msg:"signaled reads the word" 0 (N.signaled g)

let capability () =
  S.with_gpu @@ fun g ->
  let c = N.capability g in
  equal bool ~msg:"the key is the ABI's" true
    (Option.is_some (Type.Id.provably_equal N.capability_key A.Gpu.key));
  mem int ~msg:"compute class" c.compute_class [ 0xc7c0; 0xc9c0; 0xcec0 ];
  List.iter
    (fun (what, n) -> greater int ~msg:what ~than:0 n)
    [
      ("gpcs", c.gpcs);
      ("tpcs_per_gpc", c.tpcs_per_gpc);
      ("sms_per_tpc", c.sms_per_tpc);
      ("warps_per_sm", c.warps_per_sm);
    ];
  at_least int ~msg:"shared window" ~than:(1 lsl 40) c.shared_window;
  at_least int ~msg:"local window" ~than:(1 lsl 40) c.local_window;
  equal (result unit string) ~msg:"local 0" (Ok ()) (c.local 0)

let facts =
  group ~timeout:60. "facts"
    [
      test "states a GPU's facts" facts;
      test "declares the GPU to compiled code" capability;
    ]

(* Memory *)

let kinds = [ `Device; `Pinned; `Mapped ]

let pp_kind ppf k =
  Format.pp_print_string ppf
    (match k with
    | `Device -> "`Device"
    | `Pinned -> "`Pinned"
    | `Mapped -> "`Mapped")

let kind = Gen.of_list ~pp:pp_kind kinds

(* Sizes up to past 8 MiB, where GPU memory takes 2 MiB pages. *)
let size =
  Gen.of_list ~pp:Format.pp_print_int
    [ 1; 7; 4096; (2 lsl 20) + 7; (8 lsl 20) + 3 ]

let offset = Gen.of_list ~pp:Format.pp_print_int [ 0; 1; 4095 ]

(* host -> a -> b -> host through Copy parts. [dst] holds another pattern
   before, so that a copy that did not run shows. *)
let round_trip (ka, kb, n, (oa, ob)) =
  S.with_gpu @@ fun g ->
  let src = require_some (N.alloc g `Pinned n) in
  let dst = require_some (N.alloc g `Pinned n) in
  let a = require_some (N.alloc g ka (n + oa)) in
  let b = require_some (N.alloc g kb (n + ob)) in
  equal bool ~msg:"host of a" (ka <> `Device) (Option.is_some (N.host a));
  equal bool ~msg:"host of b" (kb <> `Device) (Option.is_some (N.host b));
  less int ~msg:"address of a" ~than:(1 lsl 40) (address a);
  let seed = n + oa + ob in
  S.pattern (host src) n seed;
  S.pattern (host dst) n (seed + 1);
  let ps =
    [|
      copy g (a, oa) (src, 0) n;
      copy g ~after:[| 0 |] (b, ob) (a, oa) n;
      copy g ~after:[| 1 |] (dst, 0) (b, ob) n;
    |]
  in
  S.wait g (S.submit g ps);
  equal int ~msg:"the first byte that differs" (-1)
    (S.mismatch (host dst) n seed);
  List.iter (N.free g) [ src; dst; a; b ]

(* A copy of more bytes than the copy engine moves at once, out to GPU memory at
   an odd offset and back. *)
let long_copy () =
  S.with_gpu @@ fun g ->
  let n = A.Method.max_copy + 4099 in
  let h = require_some (N.alloc g `Pinned n) in
  let d = require_some (N.alloc g `Device (n + 1)) in
  S.pattern (host h) n 3;
  S.wait g (S.submit g [| copy g (d, 1) (h, 0) n |]);
  S.pattern (host h) n 4;
  S.wait g (S.submit g [| copy g (h, 0) (d, 1) n |]);
  equal int ~msg:"the first byte that differs" (-1) (S.mismatch (host h) n 3);
  List.iter (N.free g) [ h; d ]

(* The host and the GPU see each other's stores to Mapped memory across one
   submission, round after round. *)
let mapped () =
  S.with_gpu @@ fun g ->
  let n = 4096 in
  let m = require_some (N.alloc g `Mapped n) in
  let p = require_some (N.alloc g `Pinned n) in
  for round = 1 to 100 do
    S.pattern (host m) n (2 * round);
    S.wait g (S.submit g [| copy g (p, 0) (m, 0) n |]);
    equal int
      ~msg:(strf "round %d: the GPU reads" round)
      (-1)
      (S.mismatch (host p) n (2 * round));
    S.pattern (host p) n ((2 * round) + 1);
    S.wait g (S.submit g [| copy g (m, 0) (p, 0) n |]);
    equal int
      ~msg:(strf "round %d: the host reads" round)
      (-1)
      (S.mismatch (host m) n ((2 * round) + 1))
  done;
  List.iter (N.free g) [ m; p ]

let refusal () =
  S.with_gpu @@ fun g ->
  for _ = 1 to 100 do
    is_none ~msg:"twice the budget" (N.alloc g `Device (2 * N.budget g))
  done;
  let r = require_some ~msg:"64 MiB after" (N.alloc g `Device (64 lsl 20)) in
  N.free g r

let memory =
  group ~timeout:120. "memory"
    [
      prop ~count:30 "copies through any two kinds of memory are the identity"
        (Gen.quad kind kind size (Gen.pair offset offset))
        round_trip;
      test "a copy longer than the copy engine's is the identity" long_copy;
      test "host and GPU stores to Mapped memory reach each other" mapped;
      test
        "an allocation past the GPU's memory is None, and gives back what it \
         took"
        refusal;
    ]

(* Work *)

(* A wait on a host word holds a submission's work on [queues] until the host
   writes the word: COMPUTE:0 releases a value into a host buffer, COPY:0 copies
   GPU memory to one. *)
let held queues g ~start ~wait ~below ~release =
  let w = require_some (N.alloc g `Pinned 8) in
  let src = require_some (N.alloc g `Pinned 64) in
  let mid = require_some (N.alloc g `Device 64) in
  let marked = require_some (N.alloc g `Pinned 8) in
  let copied = require_some (N.alloc g `Pinned 64) in
  let l = S.launches g in
  S.pattern (host src) 64 1;
  S.wait g (S.submit g [| copy g (mid, 0) (src, 0) 64 |]);
  let base = S.last g in
  S.set64 (host marked) 0;
  S.write (host copied) (String.make 64 '\000');
  let ps =
    List.map
      (function
        | `Compute -> compute g (S.release l (address marked) 7)
        | `Copy -> copy g (copied, 0) (mid, 0) 64)
      queues
  in
  let held () =
    S.still ~msg:"the word" int base (fun () -> N.signaled g) ~ms:20;
    equal int ~msg:"held on COMPUTE:0" 0 (S.get64 (host marked));
    equal string ~msg:"held on COPY:0" (String.make 64 '\000')
      (S.read (host copied) 64)
  in
  S.set64 (host w) start;
  releasing w release (fun () ->
      let v =
        S.submit g ~waits:[| (`Word, address w, wait) |] (Array.of_list ps)
      in
      held ();
      S.set64 (host w) below;
      held ();
      S.set64 (host w) release;
      S.wait g v);
  if List.mem `Compute queues then
    equal int ~msg:"released on COMPUTE:0" 7 (S.get64 (host marked));
  if List.mem `Copy queues then
    equal int ~msg:"copied on COPY:0" (-1) (S.mismatch (host copied) 64 1);
  S.free_launches g l;
  List.iter (N.free g) [ w; src; mid; marked; copied ]

let pp_queues ppf qs =
  Format.pp_print_string ppf
    (String.concat " and "
       (List.map (function `Compute -> "COMPUTE:0" | `Copy -> "COPY:0") qs))

(* A Word wait on a word of host memory wherever the process placed it, such as
   above 2^40, the widest address a channel's semaphore names. *)
let high_word () =
  S.with_gpu @@ fun g ->
  let p = S.pages S.page in
  at_least int ~msg:"the word's host address" ~than:(1 lsl 40) p;
  S.set64 p 0;
  let w = require_some (N.map_host g p 8) in
  let b = require_some (N.alloc g `Pinned 16) in
  S.set64 (host b) 0;
  let l = S.launches g in
  S.watchdog "a wait on a high host word" (fun () ->
      let v =
        S.submit g
          ~waits:[| (`Word, address w, 1) |]
          [| compute g (S.release l (address b) 9) |]
      in
      S.still ~msg:"the word" int (v - 1) (fun () -> N.signaled g) ~ms:20;
      S.set64 p 1;
      S.wait g v);
  equal int ~msg:"released" 9 (S.get64 (host b));
  S.free_launches g l;
  List.iter (N.free g) [ w; b ];
  S.free_pages p S.page

let waits =
  [
    test "a Word wait holds work on a host word above 2^40" high_word;
    cases
      ~name:
        (Format.asprintf
           "a Word wait holds work on %a across the 64-bit wrap (sampled)"
           pp_queues)
      "foreign waits"
      [ [ `Compute ]; [ `Copy ]; [ `Compute; `Copy ] ]
      (fun qs -> S.with_gpu (held qs ~start:(-3) ~wait:2 ~below:1 ~release:2));
    cases ~name:(strf "%d satisfied waits complete on both channels")
      "batches" [ 255; 256 ] (fun n ->
        S.with_gpu @@ fun g ->
        let w = require_some (N.alloc g `Pinned 8) in
        let b = require_some (N.alloc g `Pinned 16) in
        S.set64 (host w) 1;
        let l = S.launches g in
        let ps =
          [|
            copy g (b, 8) (b, 0) 8;
            compute g ~after:[| 0 |] (S.release l (address b) 9);
          |]
        in
        S.wait g (S.submit g ~waits:(Array.make n (`Word, address w, 1)) ps);
        equal int ~msg:"released" 9 (S.get64 (host b));
        S.free_launches g l);
  ]

(* A part runs after the parts of the other channel its [after] names: a kernel
   reads what a copy wrote, and a copy reads what a late kernel wrote. *)
let joins () =
  S.with_gpu @@ fun g ->
  let k = S.kernels g in
  let l = S.launches g in
  let n = 4096 in
  let src = require_some (N.alloc g `Pinned n) in
  let mid = require_some (N.alloc g `Device n) in
  let dst = require_some (N.alloc g `Pinned n) in
  let late ns d s =
    compute g ~after:[| 0 |]
      (S.launch l k "copy_after" ~blocks:1 [ ns; address d; address s; n ])
  in
  S.pattern (host src) n 1;
  S.pattern (host dst) n 0;
  S.wait g (S.submit g [| copy g (mid, 0) (src, 0) n; late 0 dst mid |]);
  equal int ~msg:"the kernel read the copy" (-1) (S.mismatch (host dst) n 1);
  S.pattern (host src) n 2;
  let ps =
    [|
      compute g
        (S.launch l k "copy_after" ~blocks:1
           [ 100_000; address mid; address src; n ]);
      copy g ~after:[| 0 |] (dst, 0) (mid, 0) n;
    |]
  in
  S.wait g (S.submit g ps);
  equal int ~msg:"the copy read the kernel's" (-1) (S.mismatch (host dst) n 2);
  S.free_launches g l;
  S.unload g k;
  List.iter (N.free g) [ src; mid; dst ]

(* Two kernels' parts on COMPUTE:0: the second copies what the first, started
   100 us late, wrote. *)
let compute_order () =
  S.with_gpu @@ fun g ->
  let k = S.kernels g in
  let l = S.launches g in
  let n = 4096 in
  let src = require_some (N.alloc g `Pinned n) in
  let mid = require_some (N.alloc g `Device n) in
  let dst = require_some (N.alloc g `Pinned n) in
  S.pattern (host src) n 1;
  S.pattern (host dst) n 0;
  let late ns d s =
    compute g
      (S.launch l k "copy_after" ~blocks:1 [ ns; address d; address s; n ])
  in
  S.wait g (S.submit g [| late 100_000 mid src; late 0 dst mid |]);
  equal int ~msg:"the second read the first's" (-1) (S.mismatch (host dst) n 1);
  S.free_launches g l;
  S.unload g k;
  List.iter (N.free g) [ src; mid; dst ]

let launch () =
  S.with_gpu @@ fun g ->
  let k = S.kernels g in
  let l = S.launches g in
  let n = 1000 in
  let out = require_some (N.alloc g `Pinned (4 * n)) in
  let ws = S.launch l k "double_index" ~blocks:4 [ address out; n ] in
  S.wait g (S.submit g [| compute g ws |]);
  equal (list int)
    (List.init n (fun i -> 2 * i))
    (List.init n (S.get32 (host out)));
  S.free_launches g l;
  S.unload g k;
  N.free g out

let misuse () =
  S.with_gpu @@ fun g ->
  let r = require_some (N.alloc g `Device 64) in
  let part ?(queue = "COMPUTE:0") w = ignore (N.part g ~queue w) in
  let raises name f = raises_match ~msg:name Exn.invalid_arg f in
  let submit ?(waits = [||]) ?(v = S.last g + 1) ps =
    N.submit g ~v ~waits ~handles:[||] ps
  in
  raises "a fill" (fun () -> part (`Fill (0n, 0n, 0, 0)));
  raises "odd words" (fun () -> part (`Words [| 0 |]));
  raises "no such queue" (fun () -> part ~queue:"COPY:1" (`Words [||]));
  raises "a copy on COMPUTE:0" (fun () -> part (`Copy ((r, 0), (r, 32), 8)));
  raises "a copy past its region" (fun () ->
      part ~queue:"COPY:0" (`Copy ((r, 0), (r, 40), 32)));
  raises "a negative offset" (fun () ->
      part ~queue:"COPY:0" (`Copy ((r, -1), (r, 32), 8)));
  raises "a negative after" (fun () ->
      N.part g ~queue:"COMPUTE:0" ~after:[| -1 |] (`Words [||]));
  let p = compute g ~after:[| 0 |] [||] in
  raises "an after at its own index" (fun () -> submit [| p |]);
  raises "a value other than the next" (fun () -> submit ~v:(S.last g + 2) [||]);
  raises "an Object wait" (fun () -> submit ~waits:[| (`Object, 0, 1) |] [||]);
  let w = require_some (N.alloc g `Pinned 8) in
  raises "257 waits" (fun () ->
      submit ~waits:(Array.make 257 (`Word, address w, 0)) [||]);
  raises "alloc of 0 bytes" (fun () -> N.alloc g `Device 0);
  raises "map_host of 0 bytes" (fun () -> N.map_host g (host w) 0);
  raises "map_peer of one device" (fun () -> N.map_peer g g r);
  raises "free of the word" (fun () -> N.free g (N.word g));
  N.free g r;
  raises "free twice" (fun () -> N.free g r);
  raises "a copy of a freed region" (fun () ->
      part ~queue:"COPY:0" (`Copy ((r, 0), (r, 32), 8)));
  let a = S.pages S.page in
  let m = require_some (N.map_host g a 64) in
  N.free g m;
  raises "free of a mapping twice" (fun () -> N.free g m);
  raises "a copy of a freed mapping" (fun () ->
      part ~queue:"COPY:0" (`Copy ((m, 0), (m, 32), 8)));
  S.free_pages a S.page;
  N.free g w

(* A device's regions, parts and images, used on the device opened after it
   stopped. *)
let another_device () =
  let a = S.gpu () in
  let r = require_some (N.alloc a `Pinned 64) in
  let p = S.pages S.page in
  let m = require_some (N.map_host a p 64) in
  let part = copy a (r, 32) (r, 0) 8 in
  let k = S.kernels a in
  S.stop a;
  S.with_gpu @@ fun b ->
  let raises name f = raises_match ~msg:name Exn.invalid_arg f in
  raises "a copy of its region" (fun () ->
      N.part b ~queue:"COPY:0" (`Copy ((r, 0), (r, 32), 8)));
  raises "free of its region" (fun () -> N.free b r);
  raises "free of its mapping" (fun () -> N.free b m);
  raises "submit of its part" (fun () ->
      N.submit b ~v:1 ~waits:[||] ~handles:[||] [| part |]);
  raises "unload of its image" (fun () -> N.unload b (S.image k));
  List.iter (N.free a) [ r; m; S.code k ];
  S.free_pages p S.page

let room_c () =
  S.with_gpu @@ fun g ->
  let room = S.room g in
  equal (list int)
    [ 0; 0; 0; 2; 2; 2; 2; 2; 2; 2 ]
    [
      room ~queue:0 ~words:0 [||];
      room ~queue:0 ~words:2 [||];
      room ~queue:1 ~words:0 ~copy:true [||];
      room ~queue:0 ~words:0 ~fill:true [||];
      room ~queue:0 ~words:0 ~units:1 [||];
      room ~queue:0 ~words:0 ~bytes:1 [||];
      room ~queue:0 ~words:1 [||];
      room ~queue:0 ~words:0 ~copy:true [||];
      room ~queue:2 ~words:0 [||];
      room ~queue:0 ~words:0 [| 0 |];
    ]

let work =
  group ~timeout:60. "work"
    ([
       test "a kernel scheduled from a ring entry computes" launch;
       test "a part runs after the parts of the other channel it names" joins;
       test "parts on COMPUTE:0 run in array order" compute_order;
       test "misuse raises" misuse;
       test "a region of another device raises" another_device;
       test "the C room refuses what part refuses" room_c;
     ]
    @ waits)

(* Room *)

(* Work held behind a wait on a host word fills the rings: room answers Later,
   and Fits once the work is reached. *)
let later () =
  S.with_gpu @@ fun g ->
  let w = require_some (N.alloc g `Pinned 8) in
  S.set64 (host w) 0;
  releasing w 1 (fun () ->
      ignore (S.submit g ~waits:[| (`Word, address w, 1) |] [||]);
      let rec fill n =
        if n > 100_000 then fail "room never answered Later";
        match N.room g [||] with
        | `Later -> ()
        | `Never -> fail "room is Never"
        | `Fits ->
            let v = S.last g + 1 in
            equal S.answer `Ok (N.submit g ~v ~waits:[||] ~handles:[||] [||]);
            S.given g v;
            fill (n + 1)
      in
      fill 0);
  S.wait g (S.last g);
  equal room_answer ~msg:"room once reached" `Fits (N.room g [||]);
  N.free g w

let never () =
  S.with_gpu @@ fun g ->
  let ring = Array.make (2 * 16_384) 0 in
  equal room_answer ~msg:"entries past an empty ring" `Never
    (N.room g [| compute g ring |])

let too_many () =
  S.with_gpu @@ fun g ->
  let parts = Array.make 65_536 (compute g [||]) in
  equal room_answer ~msg:"65,536 parts" `Never (N.room g parts)

(* 40,000 submissions, empty ones on COMPUTE:0 between one-byte copies on
   COPY:0, wrap both channels' rings and segments; every byte arrives. *)
let wraps () =
  S.with_gpu @@ fun g ->
  let n = 20_000 in
  let src = require_some (N.alloc g `Pinned n) in
  let dst = require_some (N.alloc g `Pinned n) in
  S.pattern (host src) n 5;
  S.pattern (host dst) n 6;
  for i = 0 to n - 1 do
    ignore (S.submit g [||]);
    ignore (S.submit g [| copy g (dst, i) (src, i) 1 |])
  done;
  S.wait g (S.last g);
  equal int ~msg:"the word" (2 * n) (N.signaled g);
  equal int ~msg:"the first byte that differs" (-1) (S.mismatch (host dst) n 5);
  List.iter (N.free g) [ src; dst ]

let room =
  group ~timeout:60. "room"
    [
      test "room is Later while the rings hold unreached work, then Fits" later;
      test "room is Never past the empty rings" never;
      xfail
        ~reason:
          "room raises Invalid_argument for 65,536 parts, where its doc \
           answers Never"
        (test "room is Never for more than 65,535 parts" too_many);
      test "40,000 submissions wrap both channels and every copy arrives" wraps;
    ]

(* Local memory *)

(* The 32-bit word i of the stack kernels' output. *)
let stacked i = (512 * i) + 130816

let local () =
  S.with_gpu @@ fun g ->
  let k = S.kernels g in
  let l = S.launches g in
  let n = 1024 in
  let out = require_some (N.alloc g `Pinned (4 * n)) in
  let run () =
    S.write (host out) (String.make (4 * n) '\000');
    S.wait g
      (S.submit g
         [| compute g (S.launch l k "stack" ~blocks:4 [ address out; n ]) |]);
    S.reset l;
    equal (list int) (List.init n stacked) (List.init n (S.get32 (host out)))
  in
  run ();
  let { A.Gpu.local; _ } = N.capability g in
  equal (result unit string) ~msg:"a smaller one" (Ok ()) (local 1024);
  run ();
  S.free_launches g l;
  S.unload g k;
  N.free g out

(* A kernel holds its local memory while the device grows it and schedules a
   kernel on the new one: both compute right. *)
let growth () =
  S.with_gpu @@ fun g ->
  let k = S.kernels g in
  let l = S.launches g in
  let n = 1024 in
  let flag = require_some (N.alloc g `Pinned 8) in
  let first = require_some (N.alloc g `Pinned (4 * n)) in
  let next = require_some (N.alloc g `Pinned (4 * n)) in
  S.set64 (host flag) 0;
  releasing flag 1 (fun () ->
      let held =
        S.launch l k "stack_held" ~blocks:4
          [ address flag; 10 * second; address first; n ]
      in
      ignore (S.submit g [| compute g held |]);
      let { A.Gpu.local; _ } = N.capability g in
      equal (result unit string) ~msg:"grown" (Ok ()) (local 4096);
      let v =
        S.submit g
          [| compute g (S.launch l k "stack" ~blocks:4 [ address next; n ]) |]
      in
      S.still ~msg:"the word while held" int (v - 2)
        (fun () -> N.signaled g)
        ~ms:20;
      S.set64 (host flag) 1;
      S.wait g v);
  let want = List.init n stacked in
  equal (list int) ~msg:"the held kernel" want
    (List.init n (S.get32 (host first)));
  equal (list int) ~msg:"the next kernel" want
    (List.init n (S.get32 (host next)));
  S.free_launches g l;
  S.unload g k;
  List.iter (N.free g) [ flag; first; next ]

let local =
  group ~timeout:60. "local memory"
    [
      test "local memory serves kernels that keep 2 KiB a thread" local;
      test "a growth leaves a running kernel's local memory intact" growth;
    ]

(* Images *)

let images () =
  S.with_gpu @@ fun g ->
  let bin = S.fixture "kernels_sm89.cubin" in
  let size = A.Cubin.size (require_ok (A.Cubin.of_string bin)) in
  (match N.image g bin with
  | Ok (`Place (n, lay)) ->
      equal int ~msg:"the region's bytes" size n;
      let r = require_some (N.alloc g `Device n) in
      let i, bytes = lay r in
      equal int ~msg:"the image's bytes" size (String.length bytes);
      N.unload g i;
      N.free g r
  | Ok (`Loaded _) -> fail "an image with nothing to place"
  | Error e -> fail e);
  let k = S.kernels g in
  equal bool ~msg:"double_index" true
    (Option.is_some (N.entry (S.image k) "double_index"));
  equal (option int) ~msg:"a missing kernel" None
    (N.entry (S.image k) "missing");
  is_error ~msg:"not a cubin" (N.image g "not a cubin");
  S.unload g k;
  raises_match ~msg:"entry after unload" Exn.invalid_arg (fun () ->
      N.entry (S.image k) "empty");
  raises_match ~msg:"unload twice" Exn.invalid_arg (fun () ->
      N.unload g (S.image k))

(* A cubin loads where another one's code ran and was unloaded: its launch runs
   its own code. *)
let reloaded () =
  S.with_gpu @@ fun g ->
  let l = S.launches g in
  let n = 1000 in
  let out = require_some (N.alloc g `Pinned (4 * n)) in
  let run k factor =
    S.write (host out) (String.make (4 * n) '\000');
    S.wait g
      (S.submit g
         [| compute g (S.launch l k "index" ~blocks:4 [ address out; n ]) |]);
    S.reset l;
    equal (list int) ~msg:(strf "%d i" factor)
      (List.init n (fun i -> factor * i))
      (List.init n (S.get32 (host out)))
  in
  let twice = S.kernels ~file:"twice_sm89.cubin" g in
  run twice 2;
  let at = address (S.code twice) in
  S.unload g twice;
  let thrice = S.kernels ~file:"thrice_sm89.cubin" g in
  equal int ~msg:"the second loads at the first one's address" at
    (address (S.code thrice));
  run thrice 3;
  S.free_launches g l;
  S.unload g thrice;
  N.free g out

let images =
  group ~timeout:60. "images"
    [
      test "load, find their kernels and unload" images;
      test "code loaded where other code ran runs as loaded" reloaded;
    ]

(* Timeline and loss *)

let long_work () =
  S.with_gpu @@ fun g ->
  let w = require_some (N.alloc g `Pinned 8) in
  S.set64 (host w) 0;
  releasing w 1 (fun () ->
      let v = S.submit g ~waits:[| (`Word, address w, 1) |] [||] in
      let t0 = Sys.time () in
      for _ = 1 to 20 do
        N.sleep g ~seen:0 ~still_ms:50;
        equal int ~msg:"the work still waits" 0 (N.signaled g)
      done;
      less float_exact ~msg:"CPU seconds over at least 1 s of sleeps" ~than:0.5
        (Sys.time () -. t0);
      S.set64 (host w) 1;
      N.sleep g ~seen:0 ~still_ms:60_000;
      S.wait g v);
  N.sleep g ~seen:0 ~still_ms:60_000;
  N.free g w

(* One domain sleeps on the word while another submits: each sleep returns as
   the word moves. *)
let sleep_aside () =
  S.with_gpu @@ fun g ->
  let n = 1000 in
  let sleeper =
    Domain.spawn (fun () ->
        while N.signaled g < n do
          N.sleep g ~seen:(N.signaled g) ~still_ms:1000
        done)
  in
  for _ = 1 to n do
    ignore (S.submit g [||])
  done;
  Domain.join sleeper;
  equal int ~msg:"the word" n (N.signaled g)

let stop_idle () =
  let g = S.gpu () in
  let r = require_some (N.alloc g `Device 64) in
  let a = S.pages S.page in
  let m = require_some (N.map_host g a 64) in
  let v = S.submit g [||] in
  S.wait g v;
  S.stop g;
  equal int ~msg:"the word" v (N.signaled g);
  N.free g r;
  N.free g m;
  S.free_pages a S.page;
  let { A.Gpu.local; _ } = N.capability g in
  is_error ~msg:"local after stop" (local 1024)

(* A channel waits on a host word nobody writes: stop is Stopped, the word holds
   the last value, and the waiting work never runs. *)
let stop_waiting () =
  let g = S.gpu () in
  let w = require_some (N.alloc g `Pinned 8) in
  let marked = require_some (N.alloc g `Pinned 8) in
  let l = S.launches g in
  S.set64 (host w) 0;
  S.set64 (host marked) 0;
  let v =
    S.submit g
      ~waits:[| (`Word, address w, 1) |]
      [| compute g (S.release l (address marked) 7) |]
  in
  S.still ~msg:"the word" int 0 (fun () -> N.signaled g) ~ms:20;
  S.watchdog "stop" (fun () -> S.stop g);
  equal int ~msg:"the word" v (N.signaled g);
  S.set64 (host w) 1;
  S.still ~msg:"the waiting work" int 0
    (fun () -> S.get64 (host marked))
    ~ms:200;
  S.free_launches g l;
  List.iter (N.free g) [ w; marked ];
  S.with_gpu (fun g' -> S.wait g' (S.submit g' [||]))

(* A kernel runs with a release queued behind it: stop is Stopped, the word
   holds the last value, and the queued work never runs. *)
let stop_running () =
  let g = S.gpu () in
  let k = S.kernels g in
  let l = S.launches g in
  let flag = require_some (N.alloc g `Pinned 8) in
  let marked = require_some (N.alloc g `Pinned 8) in
  S.set64 (host flag) 0;
  S.set64 (host marked) 0;
  releasing flag 1 (fun () ->
      let spin = S.launch l k "spin" ~blocks:1 [ address flag; 10 * second ] in
      let v1 = S.submit g [| compute g spin |] in
      let v2 = S.submit g [| compute g (S.release l (address marked) 7) |] in
      S.still ~msg:"the word" int (v1 - 1) (fun () -> N.signaled g) ~ms:20;
      S.watchdog "stop" (fun () -> S.stop g);
      equal int ~msg:"the word" v2 (N.signaled g));
  S.still ~msg:"the queued work" int 0 (fun () -> S.get64 (host marked)) ~ms:200;
  S.free_launches g l;
  List.iter (N.free g) [ flag; marked ]

let timeline =
  group ~timeout:60. "timeline"
    [
      test "long work is no fault, and a stale seen returns at once" long_work;
      test "sleep returns as the word moves while another domain submits"
        sleep_aside;
      test "stop of an idle device leaves the word at the last value" stop_idle;
      test
        "stop of a device waiting on a word that never moves is Stopped \
         (sampled)"
        stop_waiting;
      test
        "stop of a running device is Stopped, its queued work never runs \
         (sampled)"
        stop_running;
    ]

(* Two GPUs *)

let two_gpus () =
  if Device_nv_nvidia.count () < 2 then
    skip ~reason:"the machine has fewer than two NVIDIA GPUs" ();
  let a = S.gpu () in
  Fun.protect ~finally:(fun () -> S.stop a) @@ fun () ->
  let b = require_ok (Device_nv_nvidia.open_ 1) in
  Fun.protect ~finally:(fun () -> N.stop b) @@ fun () ->
  let h = require_some (N.alloc b `Pinned 64) in
  let d = require_some (N.alloc b `Device 64) in
  let ph = require_some ~msg:"host memory maps" (N.map_peer a b h) in
  equal bool ~msg:"Device memory maps iff peer" (N.peer a b)
    (match N.map_peer a b d with
    | Some pd ->
        N.free a pd;
        true
    | None -> false);
  N.free a ph;
  N.free b h;
  N.free b d

let two =
  group ~timeout:60. "two GPUs" [ test "map each other's memory" two_gpus ]

(* The shared device: the stateful tests' programs use one device, opened by the
   first and stopped when the run ends. *)

let shared = fixture ~teardown:S.stop S.gpu

(* map_host: the registry. The path maps whole pages: a range inside the pages
   of a mapped one shares its mapping, a range that shares some of their pages
   is refused. *)

module Registry = struct
  type entry = { lo : int; hi : int; mutable maps : int } (* pages [lo, hi) *)
  type t = { mutable entries : entry list; mutable regions : entry list }

  let arena = 4
  let pages (a, n) = (a / S.page, ((a + n - 1) / S.page) + 1)
  let make () = { entries = []; regions = [] }

  let map m (a, n) =
    let lo, hi = pages (a, n) in
    let inside e = e.lo <= lo && hi <= e.hi in
    match List.find_opt inside m.entries with
    | Some e ->
        cover "a range of a mapped one's pages, elsewhere"
          (lo <> e.lo || hi <> e.hi);
        cover "a range of the same pages as a mapped one"
          (lo = e.lo && hi = e.hi);
        cover "a range under one page that another range maps" (hi - lo = 1);
        e.maps <- e.maps + 1;
        m.regions <- m.regions @ [ e ];
        true
    | None when List.exists (fun e -> lo < e.hi && e.lo < hi) m.entries ->
        cover "a range that shares only some of a mapped one's pages" true;
        false
    | None ->
        let e = { lo; hi; maps = 1 } in
        m.entries <- e :: m.entries;
        m.regions <- m.regions @ [ e ];
        true

  let free m i =
    let e = List.nth m.regions i in
    m.regions <- List.filteri (fun j _ -> j <> i) m.regions;
    e.maps <- e.maps - 1;
    if e.maps = 0 then begin
      cover "the last free of a range" true;
      m.entries <- List.filter (fun e' -> e' != e) m.entries
    end

  (* The system *)

  type sys = {
    g : N.t;
    base : int;
    scratch : N.region;
    lock : Mutex.t;
    mutable live : (N.region * int * int) list;
  }

  let start () =
    let g = shared () in
    {
      g;
      base = S.pages (arena * S.page);
      scratch = Option.get (N.alloc g `Pinned (arena * S.page));
      lock = Mutex.create ();
      live = [];
    }

  let release s =
    List.iter (fun (r, _, _) -> N.free s.g r) s.live;
    N.free s.g s.scratch;
    S.free_pages s.base (arena * S.page)

  let map_sys s (a, n) =
    match N.map_host s.g (s.base + a) n with
    | Some r ->
        Mutex.protect s.lock (fun () -> s.live <- s.live @ [ (r, a, n) ]);
        true
    | None -> false

  let free_sys s i =
    let r, _, _ = List.nth s.live i in
    N.free s.g r;
    Mutex.protect s.lock (fun () ->
        s.live <- List.filteri (fun j _ -> j <> i) s.live)

  (* Each region shares its bytes both ways: the GPU reads what the host wrote,
     and the host reads what the GPU wrote. *)
  let invariant _ s =
    List.iteri
      (fun i (r, a, n) ->
        let at = s.base + a in
        S.pattern at n (2 * i);
        S.wait s.g (S.submit s.g [| copy s.g (s.scratch, 0) (r, 0) n |]);
        equal int
          ~msg:(strf "the GPU reads [%d, %d)" a (a + n))
          (-1)
          (S.mismatch (host s.scratch) n (2 * i));
        S.pattern (host s.scratch) n ((2 * i) + 1);
        S.wait s.g (S.submit s.g [| copy s.g (r, 0) (s.scratch, 0) n |]);
        equal int
          ~msg:(strf "the host reads [%d, %d)" a (a + n))
          (-1)
          (S.mismatch at n ((2 * i) + 1)))
      s.live

  let range =
    let point = [ 0; 8; 2048; 4088; 4096; 4104; 8192; 12280 ] in
    let length = [ 8; 64; 2048; 4096; 4104; 8192 ] in
    let scale x = x * S.page / 4096 in
    Gen.with_pp
      (fun ppf (a, n) -> Format.fprintf ppf "(%d, %d)" a n)
      (Gen.frequency
         [
           ( 8,
             Gen.such_that
               (fun (a, n) -> a + n <= arena * S.page)
               (Gen.map
                  (fun (a, n) -> (scale a, scale n))
                  (Gen.pair (Gen.of_list point) (Gen.of_list length))) );
           (* One range drawn often, so that ranges of the same pages are drawn
              too. *)
           (2, Gen.constant (0, S.page));
         ])
end

let registry =
  abstract "r" ~release:Registry.release ~invariant:Registry.invariant

let mapped_regions =
  among int registry (fun m ->
      List.init (List.length m.Registry.regions) Fun.id)

let registry_commands =
  [
    command "start" (Gen.unit @-> makes registry) Registry.make Registry.start;
    command "map_host"
      (registry ^-> Registry.range @-> returns bool)
      Registry.map Registry.map_sys;
    command "free"
      (registry ^-> mapped_regions ^-> returns unit)
      Registry.free Registry.free_sys;
  ]

(* Submissions: the order of values *)

module Order = struct
  (* A part copies [n] bytes of buffer [src] at [so] to buffer [dst] at [do_]:
     on COPY:0 iff [copy], else by a kernel on COMPUTE:0 that starts [delay]
     nanoseconds late, so that work run out of order shows in the buffers. It
     runs after the earlier parts [after] lists. *)
  type part = {
    delay : int;
    copy : bool;
    src : int;
    so : int;
    dst : int;
    do_ : int;
    n : int;
    after : int list;
  }

  let buffers = 3
  let size = 65536

  let pp_part ppf p =
    Format.fprintf ppf "{%s%s %d@%d -> %d@%d n=%d after=[%s]}"
      (if p.copy then "COPY" else "COMPUTE")
      (if p.delay > 0 then strf " +%dns" p.delay else "")
      p.src p.so p.dst p.do_ p.n
      (String.concat ";" (List.map string_of_int p.after))

  let pp_one ppf ps =
    Format.fprintf ppf "[%a]"
      (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_part)
      ps

  let pp ppf subs =
    Format.pp_print_list ~pp_sep:Format.pp_print_space pp_one ppf subs

  let initial b =
    Bytes.init size (fun i -> Char.chr (((b * 61) + (i * 7)) land 255))

  type t = { bufs : Bytes.t array; mutable last : bool option }

  let make () = { bufs = Array.init buffers initial; last = None }

  (* Part [i] runs before part [j], [i < j]: both on one queue, or [j] runs
     after a part that runs after [i]. *)
  let rec before ps i j =
    let pi = List.nth ps i and pj = List.nth ps j in
    pi.copy = pj.copy
    || List.exists (fun k -> k = i || (k > i && before ps i k)) pj.after

  let overlap (b, o) (b', o') n n' = b = b' && o < o' + n' && o' < o + n

  let conflict p q =
    overlap (p.dst, p.do_) (q.dst, q.do_) p.n q.n
    || overlap (p.src, p.so) (q.dst, q.do_) p.n q.n
    || overlap (p.dst, p.do_) (q.src, q.so) p.n q.n

  (* The result is one: every pair of parts that conflict is ordered, and no
     part copies over its own source. *)
  let determined_one ps =
    List.for_all
      (fun p -> not (overlap (p.src, p.so) (p.dst, p.do_) p.n p.n))
      ps
    &&
    let n = List.length ps in
    List.for_all
      (fun j ->
        List.for_all
          (fun i ->
            before ps i j || not (conflict (List.nth ps i) (List.nth ps j)))
          (List.init j Fun.id))
      (List.init n Fun.id)

  let determined _ subs = List.for_all determined_one subs

  let run_one m ps =
    let channels = List.sort_uniq compare (List.map (fun p -> p.copy) ps) in
    let releaser =
      match List.rev ps with p :: _ -> Some p.copy | [] -> None
    in
    cover "a submission of no parts after one released on COPY:0"
      (ps = [] && m.last = Some true);
    cover "a submission on the channel that released the last"
      (releaser <> None && m.last = releaser);
    cover "a switch of channel"
      (releaser <> None && m.last <> None && m.last <> releaser);
    cover "two channels in one submission" (List.length channels = 2);
    cover "an after from COPY:0 to COMPUTE:0"
      (List.exists
         (fun p ->
           (not p.copy) && List.exists (fun k -> (List.nth ps k).copy) p.after)
         ps);
    cover "an after from COMPUTE:0 to COPY:0"
      (List.exists
         (fun p ->
           p.copy && List.exists (fun k -> not (List.nth ps k).copy) p.after)
         ps);
    List.iter
      (fun p -> Bytes.blit m.bufs.(p.src) p.so m.bufs.(p.dst) p.do_ p.n)
      ps;
    m.last <- Some (Option.value releaser ~default:false)

  let run m subs =
    cover "several values in flight" (List.length subs > 1);
    List.iter (run_one m) subs

  (* The system *)

  type sys = {
    g : N.t;
    regions : N.region array;
    staging : N.region;
    kernels : S.kernels;
    launches : S.launches;
  }

  let start () =
    let g = shared () in
    let staging = Option.get (N.alloc g `Pinned size) in
    let regions =
      Array.init buffers (fun b ->
          let r = Option.get (N.alloc g `Device size) in
          S.write (host staging) (Bytes.to_string (initial b));
          S.wait g (S.submit g [| copy g (r, 0) (staging, 0) size |]);
          r)
    in
    { g; regions; staging; kernels = S.kernels g; launches = S.launches g }

  let release s =
    Array.iter (N.free s.g) s.regions;
    N.free s.g s.staging;
    S.free_launches s.g s.launches;
    S.unload s.g s.kernels

  (* Submits [subs] back to back, then reads the word until it holds the last
     value: each read is at least the one before. *)
  let run_sys s subs =
    let part p =
      let after = Array.of_list p.after in
      let dst = s.regions.(p.dst) and src = s.regions.(p.src) in
      if p.copy then copy s.g ~after (dst, p.do_) (src, p.so) p.n
      else
        compute s.g ~after
          (S.launch s.launches s.kernels "copy_after" ~blocks:1
             [ p.delay; address dst + p.do_; address src + p.so; p.n ])
    in
    let first = S.last s.g + 1 in
    List.iter
      (fun ps -> ignore (S.submit s.g (Array.of_list (List.map part ps))))
      subs;
    let last = S.last s.g in
    let rec watch seen =
      let w = N.signaled s.g in
      at_least int ~msg:"the word" ~than:seen w;
      at_most int ~msg:"the word" ~than:last w;
      if w < last then watch w
    in
    watch (first - 1);
    S.still ~msg:"the word" int last (fun () -> N.signaled s.g) ~ms:1;
    S.reset s.launches

  let invariant m s =
    Array.iteri
      (fun b r ->
        S.wait s.g (S.submit s.g [| copy s.g (s.staging, 0) (r, 0) size |]);
        equal string ~msg:(strf "buffer %d" b)
          (Bytes.to_string m.bufs.(b))
          (S.read (host s.staging) size))
      s.regions

  let parts =
    let open Gen in
    let part =
      let+ delay = frequency [ (2, constant 0); (1, constant 50_000) ]
      and+ copy = bool
      and+ src = int_range 0 (buffers - 1)
      and+ dst = int_range 0 (buffers - 1)
      and+ n = one_of [ int_range 1 64; int_range 1 16384 ]
      and+ so = int_range 0 (size - 16384)
      and+ do_ = int_range 0 (size - 16384)
      and+ after = list ~size:(int_range 0 2) (int_range 0 2) in
      { delay = (if copy then 0 else delay); copy; src; so; dst; do_; n; after }
    in
    let submission =
      let+ ps = list ~size:(int_range 0 3) part in
      List.mapi
        (fun i p ->
          let after = List.filter (fun k -> k < i) p.after in
          { p with after = List.sort_uniq compare after })
        ps
    in
    list ~size:(int_range 1 4) submission
end

let order = abstract "o" ~invariant:Order.invariant ~release:Order.release

let order_commands =
  [
    command "start" (Gen.unit @-> makes order) Order.make Order.start;
    command "submit" ~pre:Order.determined
      (order ^-> Gen.with_pp Order.pp Order.parts @-> returns unit)
      Order.run Order.run_sys;
  ]

(* Ending from two domains: whatever the order, an allocation's first [free], a
   mapping's first [free] and an image's first [unload] return, and every later
   one raises. *)

type ended = { mutable live : bool }

let end_model m =
  if not m.live then invalid_arg "ended";
  m.live <- false

let ends_once ~make ~finish ~release name =
  let v =
    abstract name ~release:(fun x ->
        try release x with Invalid_argument _ -> ())
  in
  [
    command "make" (Gen.unit @-> makes v) (fun () -> { live = true }) make;
    command "end" (v ^-> returns unit) end_model finish;
  ]

(* Each value carries the shared device: the fixture is read on the test's
   domain only. *)
let allocation_commands =
  ends_once "a"
    ~make:(fun () ->
      let g = shared () in
      (g, Option.get (N.alloc g `Device 64)))
    ~finish:(fun (g, r) -> N.free g r)
    ~release:(fun (g, r) -> N.free g r)

(* A mapping of its own page, freed once the run ends. *)
let mapping_commands =
  ends_once "m"
    ~make:(fun () ->
      let g = shared () and p = S.pages S.page in
      (g, p, Option.get (N.map_host g p 64)))
    ~finish:(fun (g, _, r) -> N.free g r)
    ~release:(fun (g, p, r) ->
      Fun.protect
        ~finally:(fun () -> S.free_pages p S.page)
        (fun () -> N.free g r))

(* An image over a code region of its own, freed once the run ends. *)
let image_commands =
  ends_once "i"
    ~make:(fun () ->
      let g = shared () in
      match N.image g (S.fixture "kernels_sm89.cubin") with
      | Ok (`Place (n, lay)) ->
          let r = Option.get (N.alloc g `Device n) in
          (g, fst (lay r), r)
      | Ok (`Loaded _) -> fail "an image with nothing to place"
      | Error e -> fail e)
    ~finish:(fun (g, i, _) -> N.unload g i)
    ~release:(fun (g, i, r) ->
      Fun.protect ~finally:(fun () -> N.free g r) (fun () -> N.unload g i))

let stateful =
  group ~timeout:300. "stateful"
    [
      stateful ~count:100 ~steps:20 "map_host shares a host range both ways"
        registry_commands;
      stateful ~count:100 ~steps:20
        "values complete in order and the word never moves backwards (sampled)"
        order_commands;
      stateful ~count:30 ~domains:2
        "an allocation freed from two domains is freed once" allocation_commands;
      stateful ~count:30 ~domains:2
        "a mapping freed from two domains is freed once" mapping_commands;
      stateful ~count:30 ~domains:2
        "an image unloaded from two domains is unloaded once" image_commands;
    ]

let () =
  exit
    (run "device_nv"
       [
         paths;
         facts;
         memory;
         work;
         room;
         local;
         images;
         timeline;
         two;
         stateful;
       ])
