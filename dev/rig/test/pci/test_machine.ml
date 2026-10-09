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

let id =
  Testable.make
    ~pp:(fun ppf (d : Machine.id) ->
      Format.fprintf ppf "%s %04x:%04x class %06x" d.bus d.vendor d.device
        d.class_)
    ~equal:( = )

let test_no_sysfs () =
  if on_linux then skip ~reason:"this machine has /sys/bus/pci" ();
  equal (list id) [] (Machine.functions Machine.this)

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
    (Machine.functions m)

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
       (Machine.functions (Machine.at root)))

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
           (Machine.functions (Machine.at root))))

let test_tree_empty () =
  equal (list id) [] (Machine.functions (Machine.at (Tree.make [])))

(* Read only: this machine lists the functions its kernel shows. *)
let test_sysfs () =
  if not on_linux then skip ~reason:"this machine has no /sys/bus/pci" ();
  let buses = Array.to_list (Sys.readdir "/sys/bus/pci/devices") in
  equal (list string)
    (List.sort Machine.compare_address buses)
    (List.map (fun (d : Machine.id) -> d.bus) (Machine.functions Machine.this))

let functions =
  group ~timeout:patience "functions"
    [
      listed_in_bus_order;
      test "this machine without /sys/bus/pci has none" test_no_sysfs;
      test "a machine's functions are its files', in bus order" test_tree;
      test "a function whose files cannot be read is left out"
        test_tree_unreadable;
      test "a machine without functions lists none" test_tree_empty;
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

let () =
  exit @@ run "rig_pci.machine" [ addresses; machines; functions; reservations ]
