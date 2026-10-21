(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Rig_pci
open Rig_pci_support

let strf = Printf.sprintf

(* Bus addresses *)

(* Linux names a function [pci_name]: "%04x:%02x:%02x.%d" of its domain, bus,
   device and function. *)
let spelled =
  cases "a bus address is spelled as Linux names the function"
    ~name:(fun (_, s) -> s)
    [
      ((0, 3, 0, 0), "0000:03:00.0");
      ((0, 0xff, 0x1f, 7), "0000:ff:1f.7");
      ((0xabcd, 0x0a, 1, 1), "abcd:0a:01.1");
      ((0x10000, 0, 2, 0), "10000:00:02.0");
    ]
    (fun ((domain, bus, device, fn), s) ->
      equal string s (Machine.address ~domain ~bus ~device ~fn))

(* Domains around 0x10000 order differently as numbers and as text. *)
let numbers =
  let domain =
    Gen.frequency
      [
        (2, Gen.int_range 0 2);
        (1, Gen.int_range 0xfff0 0xffff);
        (1, Gen.int_range 0x10000 0x10002);
      ]
  in
  Gen.quad domain (Gen.int_range 0 0xff) (Gen.int_range 0 0x1f)
    (Gen.int_range 0 7)

let address (domain, bus, device, fn) = Machine.address ~domain ~bus ~device ~fn

let bus_order =
  prop "bus order is the order of the numbers" (Gen.pair numbers numbers)
    (fun (a, b) ->
      let (da, _, _, _), (db, _, _, _) = (a, b) in
      cover "a four-digit and a five-digit domain"
        (da < 0x10000 <> (db < 0x10000));
      cover "one domain" (da = db);
      equal int
        (Int.compare (compare a b) 0)
        (Int.compare (Machine.compare_address (address a) (address b)) 0))

let not_addresses =
  cases "a string that is no bus address is refused on either side"
    ~name:(strf "%S")
    [ ""; "0000:03:00"; "0000:03:00."; "0000:03:00.0.1"; "0000:3g:00.0"; "x" ]
    (fun s ->
      let ok = "0000:00:00.0" in
      raises_match (Exn.invalid_arg ~substring:"") (fun () ->
          Machine.compare_address s ok);
      raises_match (Exn.invalid_arg ~substring:"") (fun () ->
          Machine.compare_address ok s))

let addresses =
  group ~timeout:patience "bus addresses" [ spelled; bus_order; not_addresses ]

(* Machines *)

let test_this () =
  equal ~msg:"name" (option string) None (Machine.name Machine.this);
  equal ~msg:"failed" (option string) None (Machine.failed Machine.this)

let test_page () =
  let p = Machine.page Machine.this in
  at_least ~msg:"this machine's" int ~than:4096 p;
  equal ~msg:"a power of two" int 0 (p land (p - 1))

let machines =
  group ~timeout:patience "machines"
    [
      test "this machine has no name and has not failed" test_this;
      test "this machine's page size is a power of two" test_page;
    ]

(* Functions *)

let pp_id ppf (d : Machine.id) =
  Format.fprintf ppf "%s %04x:%04x class %06x" d.bus d.vendor d.device d.class_

let id = Testable.make ~pp:pp_id ~equal:( = )

(* [listed m] is [m]'s functions, which the test requires it lists. *)
let listed m = require_ok ~pp:Format.pp_print_string (Machine.functions m)

let test_no_sysfs () =
  if on_linux then skip ~reason:"this machine has /sys/bus/pci" ();
  equal (result (list id) string) (Ok []) (Machine.functions Machine.this)

(* A machine's functions, from their files, in hexadecimal: the class code
   whole, its subclass and programming interface kept, so that a VGA controller
   and a 3D controller of one base class stay apart. *)
let test_tree () =
  let gpu = Tree.gpu "0000:c3:00.0" in
  let nv =
    {
      (Tree.gpu "10000:21:00.0") with
      vendor = 0x10de;
      device = 0x2684;
      class_ = 0x030200;
    }
  in
  let audio = { (Tree.gpu "0000:03:00.1") with class_ = 0x040300; bars = [] } in
  let m = Machine.at (Tree.make [ gpu; nv; audio ]) in
  equal (list id)
    [
      {
        bus = "0000:03:00.1";
        vendor = 0x1002;
        device = 0x744c;
        class_ = 0x040300;
      };
      {
        bus = "0000:c3:00.0";
        vendor = 0x1002;
        device = 0x744c;
        class_ = 0x030000;
      };
      {
        bus = "10000:21:00.0";
        vendor = 0x10de;
        device = 0x2684;
        class_ = 0x030200;
      };
    ]
    (listed m)

(* A function whose files cannot be read, as while the kernel removes it, is
   left out. *)
let test_tree_unreadable () =
  let root = Tree.make [ Tree.gpu "0000:03:00.0"; Tree.gpu "0000:43:00.0" ] in
  let vendor = "sys/bus/pci/devices/0000:43:00.0/vendor" in
  Out_channel.with_open_bin (Filename.concat root vendor) (fun oc ->
      output_string oc "zz\n");
  equal (list string) [ "0000:03:00.0" ]
    (List.map
       (fun (d : Machine.id) -> d.bus)
       (listed (Machine.at root)))

(* Whatever order the files list them in, with domains of four and five
   digits. *)
let listed_in_bus_order =
  prop "a machine lists its functions in bus order" ~count:30
    (Gen.list ~size:(Gen.int_range 0 12) numbers)
    (fun ns ->
      let ns = List.sort_uniq compare ns in
      cover "a four- and a five-digit domain"
        (List.exists (fun (d, _, _, _) -> d < 0x10000) ns
        && List.exists (fun (d, _, _, _) -> d >= 0x10000) ns);
      let root = Tree.make (List.map (fun n -> Tree.gpu (address n)) ns) in
      equal (list string) (List.map address ns)
        (List.map
           (fun (d : Machine.id) -> d.bus)
           (listed (Machine.at root))))

(* A machine whose functions' directory cannot be read says why, naming the
   directory. The test makes it unreadable, which a process that reads every
   file cannot test. *)
let test_tree_unlisted () =
  let root = Tree.make [ Tree.gpu "0000:03:00.0" ] in
  let devices = Filename.concat root "sys/bus/pci/devices" in
  Unix.chmod devices 0;
  Fun.protect ~finally:(fun () -> Unix.chmod devices 0o755) @@ fun () ->
  (match Sys.readdir devices with
  | _ -> skip ~reason:"the process reads unreadable directories" ()
  | exception Sys_error _ -> ());
  let why =
    require_error
      ~pp:(Format.pp_print_list pp_id)
      (Machine.functions (Machine.at root))
  in
  contains ~sub:devices why

let test_tree_empty () =
  equal (list id) [] (listed (Machine.at (Tree.make [])))

(* Read only: this machine lists the functions its kernel shows. *)
let test_sysfs () =
  if not on_linux then skip ~reason:"this machine has no /sys/bus/pci" ();
  let buses = Array.to_list (Sys.readdir "/sys/bus/pci/devices") in
  equal (list string)
    (List.sort Machine.compare_address buses)
    (List.map (fun (d : Machine.id) -> d.bus) (listed Machine.this))

let functions =
  group ~timeout:patience "functions"
    [
      listed_in_bus_order;
      test "this machine without /sys/bus/pci has none" test_no_sysfs;
      test "a machine's functions are its files', in bus order" test_tree;
      test "a function whose files cannot be read is left out"
        test_tree_unreadable;
      test "a machine without functions lists none" test_tree_empty;
      test "a machine whose functions cannot be listed says why"
        test_tree_unlisted;
      test "this machine's are those of /sys/bus/pci, in bus order" test_sysfs;
    ]

(* Reservations *)

let test_reserve_this () =
  let n = 4 lsl 20 in
  let reserve () = Machine.reserve Machine.this ~base:free_base n in
  if not on_linux then
    contains ~msg:"off Linux" ~sub:"needs Linux" (require_error (reserve ()))
  else begin
    require_ok (reserve ());
    require_ok ~msg:"again" (reserve ());
    let page = Machine.page Machine.this in
    let used = memory (4 * page) in
    let base = (used + page - 1) / page * page in
    contains ~sub:"in use"
      (require_error (Machine.reserve Machine.this ~base page))
  end

let reservations =
  group ~timeout:patience "reservations"
    [
      test
        "this machine reserves a range once and refuses one in use, on Linux \
         alone"
        test_reserve_this;
    ]

(* Transports *)

(* A machine of 16 KiB pages that a transport reaches, whose one function
   records the operations it is asked, oldest first. Its DMA memory is new
   windows of the transport, and none past 1 MiB. Its functions are listed out
   of bus order. *)
let far_page = 16 * kib
let far_bus = "0000:03:00.0"

type recorder = { m : Machine.t; tr : int; calls : string list ref }

(* A call's bound on its runs, where the function states one. *)
let bound below = if below = max_int then "" else strf " below 0x%x" below

let recorder () =
  let tr = far 0 far_page in
  let calls = ref [] in
  let log fmt = Printf.ksprintf (fun s -> calls := s :: !calls) fmt in
  let next = ref 0 in
  let fn : Machine.fn =
    {
      addressing = Iommu;
      inherited = false;
      config =
        (fun off n ->
          log "config %d %d" off n;
          0);
      set_config = (fun off n _ -> log "set_config %d %d" off n);
      bar =
        (fun i ->
          log "bar %d" i;
          None);
      map =
        (fun ~combine:_ i _ _ ->
          log "map %d" i;
          Error "no BAR");
      unmap = (fun _ -> log "unmap");
      interrupt =
        (fun ms ->
          log "interrupt %d" ms;
          false);
      reset =
        (fun () ->
          log "reset";
          Ok ());
      forget =
        (fun () ->
          log "forget";
          Ok ());
      alloc_dma =
        (fun ~contiguous:_ ~va:_ ~below n ->
          log "alloc_dma %d%s" n (bound below);
          if n > mib then Ok None
          else begin
            incr next;
            let w =
              Window.through (Window.unsafe_transport tr) (!next * mib) n
            in
            Ok (Some (w, [ (Window.address w, n) ]))
          end);
      free_dma = (fun w -> log "free_dma 0x%x" (Window.address w));
      pin =
        (fun ~below a n ->
          log "pin 0x%x %d%s" a n (bound below);
          Ok (Some [ (a, n) ]));
      unpin = (fun a n -> log "unpin 0x%x %d" a n);
      release = (fun () -> log "release");
    }
  in
  let id bus =
    { Machine.bus; vendor = 0x1002; device = 0x744c; class_ = 0x030000 }
  in
  let m =
    Machine.make ~name:"far"
      {
        transport = Window.unsafe_transport tr;
        page = far_page;
        functions =
          (fun () ->
            Ok (List.map id [ "0000:43:00.0"; far_bus; "0000:0a:00.0" ]));
        take =
          (fun bus ->
            log "take %s" bus;
            Ok fn);
        reserve =
          (fun ~base n ->
            log "reserve 0x%x %d" base n;
            Ok ());
      }
  in
  { m; tr; calls }

(* The operations [r]'s function was asked since the last call. *)
let asked r =
  let calls = List.rev !(r.calls) in
  r.calls := [];
  calls

let misuse msg f = raises_match ~msg (Exn.invalid_arg ~substring:"") f

let test_far_machine () =
  let r = recorder () in
  equal ~msg:"name" (option string) (Some "far") (Machine.name r.m);
  equal ~msg:"page" int far_page (Machine.page r.m);
  equal ~msg:"functions, in bus order" (list string)
    [ far_bus; "0000:0a:00.0"; "0000:43:00.0" ]
    (List.map (fun (d : Machine.id) -> d.bus) (listed r.m));
  equal ~msg:"failed" (option string) None (Machine.failed r.m);
  break r.tr;
  let why = Some "far: the link broke" in
  equal ~msg:"failed once broken" (option string) why (Machine.failed r.m);
  equal ~msg:"failed still" (option string) why (Machine.failed r.m);
  equal ~msg:"a take" (result pass string) (Error "far: the link broke")
    (Result.map ignore (Function.take r.m far_bus));
  equal ~msg:"asked" (list string) [] (asked r)

let test_far_function () =
  let r = recorder () in
  let base = free_base in
  require_ok (Machine.reserve r.m ~base (4 * far_page));
  let f = require_ok (Function.take r.m far_bus) in
  ignore (asked r);
  let w, _ = require_some (require_ok (Function.alloc_dma f 1)) in
  ignore (require_ok (Function.alloc_dma ~va:base f (far_page + 1)));
  misuse "a va off a page" (fun () -> Function.alloc_dma ~va:(base + 1) f 1);
  misuse "a va unreserved" (fun () ->
      Function.alloc_dma ~va:(base + (4 * far_page)) f 1);
  misuse "past the range" (fun () ->
      Function.alloc_dma ~va:(base + (3 * far_page)) f (far_page + 1));
  misuse "a pin off a page" (fun () -> Function.pin f (base + 1) far_page);
  ignore (require_some (require_ok (Function.pin f base far_page)));
  ignore (require_some (require_ok (Function.pin f base far_page)));
  Function.unpin f base far_page;
  Function.unpin f base far_page;
  misuse "an unpin past the pins" (fun () -> Function.unpin f base far_page);
  Function.set_dma_bits f 60;
  ignore (require_some (require_ok (Function.pin f base far_page)));
  Function.unpin f base far_page;
  Function.free_dma f w;
  misuse "a free_dma twice" (fun () -> Function.free_dma f w);
  equal (list string)
    [
      strf "alloc_dma %d" far_page;
      strf "alloc_dma %d" (2 * far_page);
      strf "pin 0x%x %d" base far_page;
      strf "pin 0x%x %d" base far_page;
      strf "unpin 0x%x %d" base far_page;
      strf "unpin 0x%x %d" base far_page;
      strf "pin 0x%x %d below 0x%x" base far_page (1 lsl 60);
      strf "unpin 0x%x %d" base far_page;
      strf "free_dma 0x%x" (Window.address w);
    ]
    (asked r);
  Function.release f

(* The largest size whose rounding up to the page does not overflow. *)
let test_far_largest () =
  let r = recorder () in
  let f = require_ok (Function.take r.m far_bus) in
  ignore (asked r);
  let n = max_int - far_page + 1 in
  equal ~msg:"no memory for it"
    (result (option pass) string)
    (Ok None)
    (Result.map (Option.map ignore) (Function.alloc_dma f n));
  misuse "one more byte" (fun () -> Function.alloc_dma f (n + 1));
  equal (list string) [ strf "alloc_dma %d" n ] (asked r);
  Function.release f

let test_far_released () =
  let r = recorder () in
  let base = free_base in
  require_ok (Machine.reserve r.m ~base far_page);
  let f = require_ok (Function.take r.m far_bus) in
  let w, _ = require_some (require_ok (Function.alloc_dma f 1)) in
  ignore (require_some (require_ok (Function.pin f base far_page)));
  ignore (asked r);
  Function.release f;
  Function.release f;
  List.iter
    (fun (msg, use) -> misuse msg use)
    [
      ("failed", fun () -> ignore (Function.failed f : string option));
      ("config8", fun () -> ignore (Function.config8 f 0 : int));
      ("set_config8", fun () -> Function.set_config8 f 0 0);
      ("bar", fun () -> ignore (Function.bar f 0 : (int * int) option));
      ("map", fun () -> ignore (Function.map f 0 : _ result));
      ("interrupt", fun () -> ignore (Function.interrupt f 0 : bool));
      ("reset", fun () -> ignore (Function.reset f : _ result));
      ("alloc_dma", fun () -> ignore (Function.alloc_dma f 1 : _ result));
      ("pin", fun () -> ignore (Function.pin f base far_page : _ result));
      ("set_bus_master", fun () -> Function.set_bus_master f false);
    ];
  equal ~msg:"a wait" (result unit string) (Ok ())
    (Function.wait f ~us:0 "a wait" (fun () -> true));
  Function.free_dma f w;
  Function.unpin f base far_page;
  equal (list string)
    [
      "release";
      strf "free_dma 0x%x" (Window.address w);
      strf "unpin 0x%x %d" base far_page;
    ]
    (asked r)

let transports =
  group ~timeout:patience "transports"
    [
      test
        "a transport's machine has its name and page, lists its functions in \
         bus order, and stays failed once its transport failed"
        test_far_machine;
      test
        "a function asks its transport only what it accepts, sizes rounded up \
         to the page, each pin and unpin"
        test_far_function;
      test "the largest DMA memory whose size rounds up is asked of the machine"
        test_far_largest;
      test
        "a released function asks its transport only free_dma, unpin and one \
         release"
        test_far_released;
    ]

(* Routes

   A GPU at 0000:03:00.0 and a network adapter at 0000:04:00.0, placed as each
   case says, on a machine whose configuration files hold 4096 bytes, so that
   ACS reads. Linux's p2pdma (drivers/pci/p2pdma.c) decides what the bus
   carries: a bridge above both, none of whose devices on the way redirects
   peer requests, or else a root complex it lists or an AMD CPU from Zen on. *)

let gpu_bus = "0000:03:00.0"
let nic_bus = "0000:04:00.0"

let nic ?host ?bridges () =
  {
    (Tree.gpu ?host ?bridges nic_bus) with
    vendor = 0x14e4;
    device = 0x1760;
    class_ = 0x020000;
  }

(* The GPU and the adapter under root ports 00:01.0 and 00:02.0 of one host
   bridge, or under host bridges pci0000:00 and pci0000:40. *)
let two_ports () =
  [
    Tree.gpu ~bridges:[ "0000:00:01.0" ] gpu_bus;
    nic ~bridges:[ "0000:00:02.0" ] ();
  ]

let two_hosts () =
  [
    Tree.gpu ~host:"pci0000:00" ~bridges:[ "0000:00:01.0" ] gpu_bus;
    nic ~host:"pci0000:40" ~bridges:[ "0000:40:01.0" ] ();
  ]

(* A host bridge's first device, at 00.0 of its root bus, Intel's [device]:
   0x2020 is a Skylake-E root complex Linux lists for every host bridge. *)
let listed h root =
  let d = strf "sys/devices/pci0000:%s/0000:%s:00.0/" h h in
  Tree.add root (d ^ "vendor") "0x8086\n";
  Tree.add root (d ^ "device") "0x2020\n";
  Tree.add root (d ^ "class") "0x060000\n"

(* The shared root port's ACS, the first extended capability, set to redirect
   peer requests. *)
let acs_redirect root =
  let b = Bytes.make 4096 '\000' in
  Bytes.set_int32_le b 0x100 (Int32.of_int (0x0d lor (1 lsl 16)));
  Bytes.set_uint16_le b 0x106 0x04;
  Tree.add root "sys/devices/pci0000:00/0000:00:01.0/config" (Bytes.to_string b)

let one_port () = [ Tree.gpu gpu_bus; nic () ]

let pp_distance ppf d =
  Format.pp_print_string ppf
    (match d with
    | Machine.Switch -> "Switch"
    | Bridge -> "Bridge"
    | System -> "System")

let distance = Testable.make ~pp:pp_distance ~equal:( = )

let placements =
  [
    ("under one root port", one_port, ignore, true, Machine.Switch);
    ( "under two root ports of an unlisted host bridge",
      two_ports,
      ignore,
      false,
      Bridge );
    ( "under two root ports of a listed host bridge",
      two_ports,
      listed "00",
      true,
      Bridge );
    ("under two unlisted host bridges", two_hosts, ignore, false, System);
    ( "under two listed host bridges",
      two_hosts,
      (fun root ->
        listed "00" root;
        listed "40" root),
      true,
      System );
    ( "under one root port whose ACS redirects",
      one_port,
      acs_redirect,
      false,
      Switch );
  ]

let test_placement (_, fns, setup, carried, apart) =
  let root = Tree.make ~privileged:true (fns ()) in
  setup root;
  let m = Machine.at root in
  equal ~msg:"carried" bool carried
    (Machine.forwards m ~provider:gpu_bus ~client:nic_bus);
  equal ~msg:"distance" (option distance) (Some apart)
    (Machine.distance m gpu_bus nic_bus);
  equal ~msg:"distance, the other way" (option distance) (Some apart)
    (Machine.distance m nic_bus gpu_bus)

(* A function that is no bus address, names no function, or is another
   machine's is unread. *)
let unread =
  let here bus () = (Machine.at (Tree.make (one_port ())), bus) in
  [
    ("no bus address", here "04:00.0");
    ("a path for a bus", here "../../..");
    ("no function of the machine", here "0000:05:00.0");
    ("a machine a transport reaches", fun () -> ((recorder ()).m, nic_bus));
  ]

let test_unread (_, make) =
  let m, bus = make () in
  equal ~msg:"carried" bool false
    (Machine.forwards m ~provider:gpu_bus ~client:bus);
  equal ~msg:"carried, the other way" bool false
    (Machine.forwards m ~provider:bus ~client:gpu_bus);
  equal ~msg:"distance" (option distance) None (Machine.distance m gpu_bus bus)

let routes =
  group ~timeout:patience "routes"
    [
      cases "forwards and distance follow Linux's p2pdma and the bridges"
        ~name:(fun (n, _, _, _, _) -> n)
        placements test_placement;
      cases "an unread function is carried nowhere and has no distance"
        ~name:fst unread test_unread;
    ]

let () =
  exit
  @@ run "rig_pci.machine"
       [ addresses; machines; functions; reservations; transports; routes ]
