(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Device_pci

external far : int -> int -> int = "test_far"
external break : int -> unit = "test_far_break"
external now_ns : unit -> int = "test_now_ns"

let ms_since t0 = (now_ns () - t0) / 1_000_000

(* A machine a transport reaches: [far] is its C transport, [calls] the
   operations it was asked for, newest first. *)
type fake = { far : int; machine : Machine.t; calls : string list ref }

let fake ?(page = 16384) ?(ids = []) ?(name = "far:7000") () =
  let far = far 0 4096 and calls = ref [] in
  let ask call x =
    calls := call :: !calls;
    x
  in
  let machine =
    Machine.make ~name
      {
        transport = Window.transport far;
        page;
        functions = (fun () -> ask "functions" ids);
        take = (fun ~lock:_ _ -> ask "take" (Error "far:7000: taken"));
        reserve =
          (fun ~base n ->
            if base = 0 then failwith "far:7000: the range is in use";
            ask (Printf.sprintf "reserve 0x%x %d" base n) ());
      }
  in
  { far; machine; calls }

let on_linux = Sys.file_exists "/sys/bus/pci/devices"

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
    ~name:(Printf.sprintf "%S")
    [ ""; "0000:03:00"; "0000:03:00."; "0000:03:00.0.1"; "0000:3g:00.0"; "x" ]
    (fun s ->
      let ok = "0000:00:00.0" in
      raises_match (Exn.invalid_arg ~substring:"") (fun () ->
          Machine.compare_address s ok);
      raises_match (Exn.invalid_arg ~substring:"") (fun () ->
          Machine.compare_address ok s))

let addresses = group "bus addresses" [ spelled; bus_order; not_addresses ]

(* Machines *)

let test_this () =
  equal ~msg:"name" (option string) None (Machine.name Machine.this);
  equal ~msg:"failed" (option string) None (Machine.failed Machine.this)

(* 7e635faa9: another machine's name is its transport's. *)
let test_named () =
  let f = fake ~name:"host:7000" () in
  equal (option string) (Some "host:7000") (Machine.name f.machine)

let test_page () =
  let f = fake ~page:65536 () in
  equal ~msg:"another machine's" int 65536 (Machine.page f.machine);
  let p = Machine.page Machine.this in
  at_least ~msg:"this machine's" int ~than:4096 p;
  equal ~msg:"a power of two" int 0 (p land (p - 1))

let test_failed () =
  let f = fake () in
  equal ~msg:"before" (option string) None (Machine.failed f.machine);
  break f.far;
  let why = Some "far: the link broke" in
  equal ~msg:"once its transport failed" (option string) why
    (Machine.failed f.machine);
  equal ~msg:"and after" (option string) why (Machine.failed f.machine);
  equal ~msg:"this machine" (option string) None (Machine.failed Machine.this)

let machines =
  group "machines"
    [
      test "this machine has no name and has not failed" test_this;
      test "another machine is named as its transport names it" test_named;
      test "a machine's page size is its transport's, this one's a power of two"
        test_page;
      test "a machine has failed exactly once its transport has, and stays so"
        test_failed;
    ]

(* Functions *)

let id_of n =
  { Machine.bus = address n; vendor = 0x1002; device = 0x744c; class_ = 3 }

let id =
  Testable.make
    ~pp:(fun ppf (d : Machine.id) ->
      Format.fprintf ppf "%s %04x:%04x class %02x" d.bus d.vendor d.device
        d.class_)
    ~equal:( = )

(* d53e1085e: whatever order the transport lists them in. *)
let listed_in_bus_order =
  xfail ~reason:"the transport's order is kept"
  @@ prop "a machine lists its functions in bus order"
       (Gen.list ~size:(Gen.int_range 0 12) numbers)
       (fun ns ->
         let ns = List.sort_uniq compare ns in
         let shuffled =
           List.map snd
             (List.sort compare
                (List.mapi (fun i n -> (i * 7919 mod 13, n)) ns))
         in
         cover "a four- and a five-digit domain"
           (List.exists (fun (d, _, _, _) -> d < 0x10000) ns
           && List.exists (fun (d, _, _, _) -> d >= 0x10000) ns);
         let f = fake ~ids:(List.map id_of shuffled) () in
         equal (list id) (List.map id_of ns) (Machine.functions f.machine))

(* 07387544f *)
let test_listing_asks () =
  let ids = List.map id_of [ (0, 1, 0, 0); (0, 2, 0, 0) ] in
  let f = fake ~ids () in
  equal ~msg:"functions" (list id) ids (Machine.functions f.machine);
  equal ~msg:"what the machine was asked" (list string) [ "functions" ]
    !(f.calls)

let test_no_sysfs () =
  if on_linux then skip ~reason:"this machine has /sys/bus/pci" ();
  equal (list id) [] (Machine.functions Machine.this)

(* Linux's sysfs: a directory per function, its identity in hexadecimal text
   files. The class file holds base class, subclass and interface. *)
let sysfs_id bus =
  let hex file =
    In_channel.with_open_text
      (Filename.concat (Filename.concat "/sys/bus/pci/devices" bus) file)
      In_channel.input_all
    |> String.trim |> int_of_string
  in
  {
    Machine.bus;
    vendor = hex "vendor";
    device = hex "device";
    class_ = hex "class" lsr 16;
  }

let test_sysfs () =
  if not on_linux then skip ~reason:"this machine has no /sys/bus/pci" ();
  let buses = Array.to_list (Sys.readdir "/sys/bus/pci/devices") in
  let want = List.map sysfs_id (List.sort Machine.compare_address buses) in
  equal (list id) want (Machine.functions Machine.this)

let functions =
  group "functions"
    [
      listed_in_bus_order;
      test "listing a machine's functions asks it nothing else"
        test_listing_asks;
      test "this machine without /sys/bus/pci has none" test_no_sysfs;
      test "this machine's are those of /sys/bus/pci, in bus order" test_sysfs;
    ]

(* Reservations *)

let test_reserve () =
  let f = fake () in
  Machine.reserve f.machine ~base:0x2000_0000_0000 (1 lsl 30);
  equal ~msg:"asked" (list string)
    [ "reserve 0x200000000000 1073741824" ]
    !(f.calls);
  raises ~msg:"refused" (Failure "far:7000: the range is in use") (fun () ->
      Machine.reserve f.machine ~base:0 4096)

(* A range of this process's addresses far from what the runtime maps. *)
let free_base = 0x6f00_0000_0000

external memory : int -> int = "test_memory"

let test_reserve_this () =
  let n = 4 lsl 20 in
  (match Machine.reserve Machine.this ~base:free_base n with
  | () -> ()
  | exception Failure why -> skip ~reason:why ());
  Machine.reserve Machine.this ~base:free_base n;
  let page = Machine.page Machine.this in
  let used = memory (4 * page) in
  let base = (used + page - 1) / page * page in
  raises_match (Exn.failure ~substring:"") (fun () ->
      Machine.reserve Machine.this ~base page)

let reservations =
  group "reservations"
    [
      test "another machine's reservation is its transport's" test_reserve;
      test "this machine reserves a range once and refuses one in use"
        test_reserve_this;
    ]

(* Waits *)

let counter () =
  let n = ref 0 in
  ( n,
    fun k () ->
      incr n;
      !n >= k )

let test_at_once () =
  let n, f = counter () in
  equal ~msg:"result" bool true (Machine.wait Machine.this ~ms:10_000 (f 1));
  equal ~msg:"calls" int 1 !n

let until_true =
  prop "a wait calls its condition until it holds, and no more"
    (Gen.int_range 1 200) (fun k ->
      let n, f = counter () in
      let t0 = now_ns () in
      equal ~msg:"result" bool true (Machine.wait Machine.this ~ms:10_000 (f k));
      equal ~msg:"calls" int k !n;
      less ~msg:"ms, without waiting out its time" int ~than:5_000 (ms_since t0))

let test_times_out () =
  let n, f = counter () in
  equal ~msg:"result" bool false (Machine.wait Machine.this ~ms:30 (f max_int));
  at_least ~msg:"calls" int ~than:2 !n

(* Ten waits, so that one cut short shows however the machine is loaded. *)
let test_full_time () =
  for _ = 1 to 10 do
    let t0 = now_ns () in
    ignore (Machine.wait Machine.this ~ms:2 (fun () -> false) : bool);
    at_least ~msg:"ns waited" int ~than:2_000_000 (now_ns () - t0)
  done

let test_zero () =
  let n, f = counter () in
  equal ~msg:"result" bool true (Machine.wait Machine.this ~ms:0 (f 1));
  equal ~msg:"calls" int 1 !n

let test_failed_wait () =
  let f = fake () in
  break f.far;
  let why = Option.get (Machine.failed f.machine) in
  raises (Failure why) (fun () ->
      Machine.wait f.machine ~ms:10_000 (fun () -> false))

let test_fails_during () =
  let f = fake () in
  let n = ref 0 in
  let cond () =
    incr n;
    if !n = 3 then break f.far;
    false
  in
  let t0 = now_ns () in
  raises (Failure "far: the link broke") (fun () ->
      Machine.wait f.machine ~ms:10_000 cond);
  less ~msg:"ms, without waiting out its time" int ~than:5_000 (ms_since t0)

let waits =
  group "waits"
    [
      test "a wait whose condition holds at once is true after one call"
        test_at_once;
      until_true;
      test "a wait whose condition never holds is false, asked more than once"
        test_times_out;
      xfail ~reason:"a wait ends up to a millisecond early"
        (test "a wait whose condition never holds lasts its whole time"
           test_full_time);
      test "a wait of 0 ms asks its condition once (unstated)" test_zero;
      test "a wait on a failed machine raises its reason" test_failed_wait;
      test "a machine that fails during a wait ends it with its reason"
        test_fails_during;
    ]

let () =
  exit
  @@ run "device_pci machine"
       [ addresses; machines; functions; reservations; waits ]
