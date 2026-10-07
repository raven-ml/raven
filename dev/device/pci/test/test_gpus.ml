(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Device_pci
open Device_pci_support

(* A fake machine

   A transport's machine whose takes lock its functions, as this machine's takes
   do: a function taken and not released refuses another take. It records each
   function taken, and calls [released] once a function is given back. *)

type fake = {
  mutable taken : string list; (* buses, newest first *)
  mutable refusal : string option; (* why every take is refused *)
  mutable released : string -> unit;
  lock : Mutex.t;
}

let fake_fn fake tr bus =
  let gone = ref false in
  let release () =
    if not !gone then begin
      gone := true;
      Mutex.protect fake.lock (fun () ->
          fake.taken <- List.filter (fun b -> b <> bus) fake.taken);
      fake.released bus
    end
  in
  {
    Machine.addressing = Physical;
    config8 = (fun _ -> 0);
    config16 = (fun _ -> 0);
    config32 = (fun _ -> 0);
    set_config8 = (fun _ _ -> ());
    set_config16 = (fun _ _ -> ());
    set_config32 = (fun _ _ -> ());
    bar = (fun i -> if i = 0 then Some (0, 4096) else None);
    map = (fun _ off n -> Ok (Window.through tr off n));
    unmap = (fun _ -> ());
    interrupt = (fun _ -> false);
    reset = (fun () -> Ok ());
    alloc_dma =
      (fun ~contiguous:_ ~va:_ n -> Ok (Window.through tr 0 n, [ (0, n) ]));
    free_dma = (fun _ -> ());
    pin = (fun a n -> Ok [ (a, n) ]);
    unpin = (fun _ _ -> ());
    release;
  }

let machine ?(name = "far:1") ids =
  let fake =
    { taken = []; refusal = None; released = ignore; lock = Mutex.create () }
  in
  let tr = Window.unsafe_transport (far 0 4096) in
  let take bus =
    Mutex.protect fake.lock @@ fun () ->
    match fake.refusal with
    | Some why -> Error why
    | None ->
        if not (List.exists (fun (id : Machine.id) -> id.bus = bus) ids) then
          Error (bus ^ " is no function of " ^ name)
        else if List.mem bus fake.taken then Error (bus ^ " is locked")
        else begin
          fake.taken <- bus :: fake.taken;
          Ok (fake_fn fake tr bus)
        end
  in
  let ops =
    {
      Machine.transport = tr;
      page = 4096;
      functions = (fun () -> ids);
      take;
      reserve = (fun ~base:_ _ -> Ok ());
    }
  in
  (Machine.make ~name ops, fake)

let taken fake = Mutex.protect fake.lock (fun () -> fake.taken)
let taken_w = list string

(* The vendor's GPUs are its display controllers. Its audio functions and
   another vendor's display controllers are no GPUs of it. *)

let vendor = 0x1002

let id ?(vendor = vendor) ?(class_ = 0x03) bus =
  { Machine.bus; vendor; device = 0x73bf; class_ }

let is_gpu (id : Machine.id) = id.vendor = vendor && id.class_ = 0x03
let gpus () = Gpus.make ~memory_bar:0 is_gpu
let gpu_buses = [ "0000:03:00.0"; "0000:43:00.0"; "0000:c3:00.0" ]

let functions =
  [
    id "0000:03:00.0";
    id ~class_:0x04 "0000:03:00.1";
    id ~vendor:0x10de "0000:21:00.0";
    id "0000:43:00.0";
    id "0000:c3:00.0";
  ]

let three () =
  let m, fake = machine functions in
  (gpus (), m, fake)

(* Opening with a driver [d] that starts nothing. *)

let ok () = Ok ()
let pci g m i d = Gpus.open_pci g m i (fun _ _ -> d ())
let kernel g m i d = Gpus.open_kernel g m i (fun _ -> d ())
let reset g m i d = Gpus.reset g m i (fun _ -> d ())
let hold g m i = require_ok (Gpus.open_pci g m i (fun h _ -> Ok h))
let hold_fn g m i = require_ok (Gpus.open_pci g m i (fun h fn -> Ok (h, fn)))

(* [unopened open_] is the message of [open_ driver], an [Error] that never
   called [driver]. *)
let unopened ?__POS__ open_ =
  let calls = ref 0 in
  let r =
    open_ (fun () ->
        incr calls;
        Ok ())
  in
  equal ?__POS__ ~msg:"calls of the driver" int 0 !calls;
  require_error ?__POS__ r

(* [names n why] asserts that the number [n] is a word of [why]. *)
let names n why =
  let digits c = if c >= '0' && c <= '9' then c else ' ' in
  let words = String.split_on_char ' ' (String.map digits why) in
  satisfies
    ~claim:(Printf.sprintf "names the number %d" n)
    string
    (fun _ -> List.mem (string_of_int n) words)
    why

(* Numbering *)

(* Functions in bus order whose spellings sort otherwise: a domain of five
   digits after ones of four. *)
let pool =
  [
    id "0000:00:01.0";
    id ~class_:0x04 "0000:00:01.1";
    id "0000:0a:00.0";
    id ~vendor:0x10de "0000:10:00.0";
    id "0000:10:00.1";
    id "0001:00:00.0";
    id "2000:00:00.0";
    id ~vendor:0x10de "10000:00:00.0";
    id "10000:00:1f.7";
  ]

let pp_id ppf (id : Machine.id) =
  Format.fprintf ppf "%s/%04x/%02x" id.bus id.vendor id.class_

let pp_ids = Format.pp_print_list ~pp_sep:Format.pp_print_space pp_id

let gpus_of ids =
  List.filter_map
    (fun (id : Machine.id) -> if is_gpu id then Some id.bus else None)
    ids

let test_in_order ids =
  let m, _ = machine ids in
  equal (list string) (gpus_of ids) (Gpus.buses (gpus ()) m)

(* A subsequence of [pool], and the order a transport lists it in. *)
let any_order =
  Gen.with_pp
    (fun ppf (_, listed) -> pp_ids ppf listed)
    (Gen.bind (Gen.subsequence pool) (fun ids ->
         Gen.map (fun listed -> (ids, listed)) (Gen.permutation ids)))

let test_any_order (ids, listed) =
  cover "listed out of bus order" (gpus_of ids <> gpus_of listed);
  let m, _ = machine listed in
  equal (list string) (gpus_of ids) (Gpus.buses (gpus ()) m)

let test_ith () =
  let g, m, _ = three () in
  let open_ i bus =
    let msg = Printf.sprintf "GPU %d" i in
    let h =
      require_ok ~msg
        (Gpus.open_pci g m i (fun h fn ->
             equal ~msg string bus (Function.bus fn);
             equal ~msg (option string) (Machine.name m)
               (Machine.name (Function.machine fn));
             Ok h))
    in
    equal ~msg string bus (Gpus.bus h)
  in
  List.iteri open_ gpu_buses

let test_reset_ith () =
  let g, m, _ = three () in
  let reset i bus =
    let seen = ref [] in
    require_ok
      (Gpus.reset g m i (fun fn ->
           seen := Function.bus fn :: !seen;
           Ok ()));
    equal ~msg:(Printf.sprintf "GPU %d" i) (list string) [ bus ] !seen
  in
  List.iteri reset gpu_buses

let no_gpu =
  List.concat_map
    (fun (name, f) -> List.map (fun i -> (name, f, i)) [ 3; 4; 7; max_int ])
    [ ("open_pci", pci); ("reset", reset) ]

let test_no_gpu (_, f, i) =
  let g, m, _ = three () in
  names 3 (unopened (f g m i))

let test_none () =
  let m, _ = machine [ id ~class_:0x04 "0000:03:00.1" ] in
  let g = gpus () in
  equal (list string) [] (Gpus.buses g m);
  names 0 (unopened (pci g m 0));
  names 0 (unopened (reset g m 0))

let negative =
  let far () = fst (machine functions) in
  List.concat_map
    (fun (name, f) -> List.map (fun i -> (name, f, i)) [ -1; min_int ])
    [
      ("open_pci", fun i -> ignore (pci (gpus ()) (far ()) i ok));
      ( "open_kernel of another machine",
        fun i -> ignore (kernel (gpus ()) (far ()) i ok) );
      ( "open_kernel of this machine",
        fun i -> ignore (kernel (gpus ()) Machine.this i ok) );
      ("reset", fun i -> ignore (reset (gpus ()) (far ()) i ok));
      ("detach", fun i -> ignore (Gpus.detach (gpus ()) Machine.this i));
      ("attach", fun i -> ignore (Gpus.attach (gpus ()) Machine.this i));
    ]

let numbering =
  group ~timeout:patience "numbering"
    [
      prop "a vendor's GPUs are the functions it recognizes, in bus order"
        (Gen.with_pp pp_ids (Gen.subsequence pool))
        test_in_order;
      prop
        "a vendor's GPUs are in bus order whatever order the transport lists \
         them in"
        any_order test_any_order;
      test "GPU i is the ith bus address, its function taken there" test_ith;
      test "a reset of GPU i takes the ith bus address's function"
        test_reset_ith;
      cases "no GPU i is an Error naming how many there are"
        ~name:(fun (f, _, i) -> Printf.sprintf "%s %d" f i)
        no_gpu test_no_gpu;
      test "a machine without the vendor's GPUs has none to open" test_none;
      cases "a negative GPU number raises Invalid_argument"
        ~name:(fun (f, _, i) -> Printf.sprintf "%s %d" f i)
        negative
        (fun (_, f, i) ->
          raises_match (Exn.invalid_arg ?substring:None) (fun () -> f i));
    ]

(* Opening over PCI *)

let test_result () =
  let g, m, _ = three () in
  equal (result int string) (Ok 42) (Gpus.open_pci g m 0 (fun _ _ -> Ok 42));
  equal (result int string) (Error "the GPU did not start")
    (Gpus.open_pci g m 1 (fun _ _ -> Error "the GPU did not start"))

(* [given_back g m fake] asserts that GPU 0 and its function were given back. *)
let given_back g m fake =
  equal ~msg:"functions taken" taken_w [] (taken fake);
  ignore (hold g m 0)

let test_error_gives_back () =
  let g, m, fake = three () in
  ignore (Gpus.open_pci g m 0 (fun _ _ -> Error "the GPU did not start"));
  given_back g m fake

let passed =
  List.concat_map
    (fun (op, run) ->
      List.map
        (fun (name, e) -> (op ^ " " ^ name, run, e))
        [
          ("Invalid_argument", Invalid_argument "a bug");
          ("Not_found", Not_found);
          ("Failure", Failure "int_of_string");
          ("Sys_error", Sys_error "/dev/kfd: No such file");
          ("Unix_error", Unix.Unix_error (ENOENT, "open", "/dev/kfd"));
        ])
    [
      ("open_pci", fun g m e -> Gpus.open_pci g m 0 (fun _ _ -> raise e));
      ("reset", fun g m e -> Gpus.reset g m 0 (fun _ -> raise e));
    ]

let test_passed (_, run, e) =
  let g, m, fake = three () in
  raises e (fun () -> run g m e);
  given_back g m fake

let test_held () =
  let g, m, _ = three () in
  ignore (hold g m 0);
  ignore (unopened (pci g m 0))

let test_take_refused () =
  let g, m, fake = three () in
  fake.refusal <- Some "0000:03:00.0 is held by process 4242";
  equal string "0000:03:00.0 is held by process 4242" (unopened (pci g m 0))

let test_per_machine () =
  let g = gpus () in
  let m1, _ = machine ~name:"far:1" functions in
  let m2, _ = machine ~name:"far:2" functions in
  let h1 = hold g m1 0 in
  let h2 = hold g m2 0 in
  Gpus.lose h1;
  Gpus.release h2;
  ignore (hold g m2 0);
  ignore (unopened (pci g m1 0))

let test_kernel_elsewhere () =
  let g, m, _ = three () in
  ignore (unopened (kernel g m 0))

let opening =
  group ~timeout:patience "opening over PCI"
    [
      test "an open is the driver's result" test_result;
      test "an open the driver refuses gives the GPU and its function back"
        test_error_gives_back;
      cases
        "other exceptions from the driver pass through, and give the GPU back"
        ~name:(fun (n, _, _) -> n)
        passed test_passed;
      test "a GPU held is refused without calling the driver" test_held;
      test "a function that cannot be taken is refused with the take's reason"
        test_take_refused;
      test "the same bus on two machines is two GPUs" test_per_machine;
      test "another machine's GPUs are refused to the kernel driver"
        test_kernel_elsewhere;
    ]

(* Giving back *)

let test_release () =
  let g, m, fake = three () in
  let h, fn = hold_fn g m 0 in
  Gpus.release h;
  equal ~msg:"released" bool true (Function.released fn);
  given_back g m fake

let test_lose () =
  let g, m, fake = three () in
  let h, fn = hold_fn g m 0 in
  ignore (hold g m 1);
  Gpus.lose h;
  equal ~msg:"released" bool true (Function.released fn);
  equal ~msg:"functions taken" taken_w [ "0000:43:00.0" ] (taken fake);
  ignore (unopened (pci g m 0));
  ignore (hold g m 2);
  require_ok (reset g m 0 ok);
  ignore (hold g m 0)

let twice =
  [
    ("release", Gpus.release, "release", Gpus.release);
    ("release", Gpus.release, "lose", Gpus.lose);
    ("lose", Gpus.lose, "release", Gpus.release);
    ("lose", Gpus.lose, "lose", Gpus.lose);
  ]

let test_twice (_, first, _, again) =
  let g, m, fake = three () in
  let h = hold g m 0 in
  first h;
  let other = hold g m 1 in
  raises_match (Exn.invalid_arg ?substring:None) (fun () -> again h);
  equal ~msg:"functions taken" taken_w [ "0000:43:00.0" ] (taken fake);
  Gpus.release other

let giving_back =
  group ~timeout:patience "giving back"
    [
      test "release gives the function back and the GPU opens again"
        test_release;
      test "a GPU lost over PCI opens again only after a reset" test_lose;
      cases
        "a hold given back twice raises Invalid_argument and changes nothing"
        ~name:(fun (a, _, b, _) -> a ^ " then " ^ b)
        twice test_twice;
    ]

(* Resets *)

let test_reset () =
  let g, m, fake = three () in
  let seen = ref [] in
  let reset_gpu fn =
    equal ~msg:"taken during the reset" taken_w [ "0000:43:00.0" ] (taken fake);
    equal (option string) (Machine.name m) (Machine.name (Function.machine fn));
    seen := fn :: !seen;
    Ok ()
  in
  require_ok (Gpus.reset g m 1 reset_gpu);
  let fn = require_some (List.nth_opt !seen 0) in
  equal ~msg:"calls" int 1 (List.length !seen);
  equal ~msg:"released" bool true (Function.released fn);
  equal ~msg:"functions taken" taken_w [] (taken fake)

let test_reset_held () =
  let g, m, _ = three () in
  ignore (hold g m 0);
  ignore (unopened (reset g m 0))

let test_reset_failure () =
  let g, m, fake = three () in
  let why =
    require_error
      (Gpus.reset g m 0 (fun _ -> Error "the GPU did not come back"))
  in
  contains ~sub:"the GPU did not come back" why;
  equal ~msg:"functions taken" taken_w [] (taken fake)

let test_reset_take () =
  let g, m, fake = three () in
  fake.refusal <- Some "0000:03:00.0 is held by process 4242";
  ignore (unopened (reset g m 0))

let resets =
  group ~timeout:patience "resets"
    [
      test "a reset takes the function, resets it and releases it" test_reset;
      test "a reset of a GPU held is refused" test_reset_held;
      test
        "a vendor's reset that fails is the reset's Error, the function \
         released"
        test_reset_failure;
      test "a reset whose function cannot be taken is refused" test_reset_take;
    ]

(* This machine *)

(* Taking a GPU of this machine is a hardware opt-in: such a test runs only when
   DEVICE_PCI_TEST_GPU_LOCK names the machine's GPU lock, which it holds while
   it runs, so that it never takes a device another user drives. *)

let test_this_none () =
  let g = Gpus.make ~memory_bar:0 (fun _ -> false) in
  let this = Machine.this in
  equal (list string) [] (Gpus.buses g this);
  names 0 (unopened (kernel g this 0));
  names 0 (unopened (pci g this 0));
  names 0 (unopened (reset g this 0));
  ignore (require_error (Gpus.detach g this 0));
  ignore (require_error (Gpus.attach g this 0))

(* This machine's display controllers: the kernel driver opens nothing, so
   holding one through it changes nothing. *)
let this_gpus () =
  let g = Gpus.make ~memory_bar:0 (fun id -> id.class_ = 0x03) in
  if Gpus.buses g Machine.this = [] then
    skip ~reason:"this machine lists no GPU" ();
  g

let kernel_hold g = require_ok (Gpus.open_kernel g Machine.this 0 Result.ok)

let test_kernel_fixes () =
  let g = this_gpus () in
  Gpus.release (kernel_hold g);
  ignore (unopened (pci g Machine.this 0))

let test_kernel_held () =
  let g = this_gpus () in
  let h = kernel_hold g in
  ignore (unopened (kernel g Machine.this 0));
  ignore (unopened (reset g Machine.this 0));
  Gpus.release h;
  Gpus.release (kernel_hold g)

let test_kernel_lost () =
  let g = this_gpus () in
  Gpus.lose (kernel_hold g);
  Gpus.release (kernel_hold g)

let test_kernel_far () =
  let g = this_gpus () in
  Gpus.release (kernel_hold g);
  let m, _ = machine functions in
  Gpus.release (hold g m 0)

let test_far_leaves () =
  let g = this_gpus () in
  let m, _ = machine functions in
  Gpus.release (hold g m 0);
  Gpus.release (kernel_hold g)

(* A GPU this machine lets the process take: one bound to vfio-pci or to no
   driver. Taking it changes nothing. *)
let test_pci_fixes () =
  let g = this_gpus () in
  with_gpu_lock @@ fun () ->
  let n = List.length (Gpus.buses g Machine.this) in
  let rec first i =
    if i = n then skip ~reason:"this machine has no function to take" ();
    match Gpus.open_pci g Machine.this i (fun h _ -> Ok h) with
    | Ok h -> (i, h)
    | Error _ -> first (i + 1)
  in
  let i, h = first 0 in
  Gpus.release h;
  ignore (unopened (kernel g Machine.this i))

let this_machine =
  group ~timeout:patience "this machine"
    [
      test "this machine without the vendor's GPUs has none to open or change"
        test_this_none;
      test
        "a first open through the kernel driver refuses PCI opens of this \
         machine from then on"
        test_kernel_fixes;
      test "a GPU the kernel driver holds is refused to opens and resets"
        test_kernel_held;
      test "a GPU lost through its kernel driver opens again without a reset"
        test_kernel_lost;
      test "the kernel driver leaves other machines' GPUs open over PCI"
        test_kernel_far;
      test
        "an open of another machine's GPU leaves this machine's interface open"
        test_far_leaves;
      test
        "a first open over PCI refuses the kernel driver this machine's GPUs \
         from then on"
        test_pci_fixes;
    ]

(* Serialized opens and changes *)

(* A window in which a domain that is not held back would have run, sampled:
   nothing in the library signals that a domain waits on it. *)
let sample () = Unix.sleepf 0.05

(* While GPU 0's driver starts, another domain opens or resets GPU 1. Its driver
   must not run before GPU 0's returns. GPU 0's driver waits until the other
   domain is about to call Gpus, then samples a window in which a driver that is
   not held back runs inside it. *)
let test_one_at_a_time (_, other) =
  let g, m, _ = three () in
  let inside = Atomic.make false in
  let about = Atomic.make false in
  let ran = Atomic.make false in
  let overlapped = Atomic.make false in
  let driver () =
    Atomic.set overlapped (Atomic.get inside);
    Atomic.set ran true;
    Ok ()
  in
  let start _ _ =
    Atomic.set inside true;
    let d =
      Domain.spawn (fun () ->
          Atomic.set about true;
          other g m 1 driver)
    in
    equal ~msg:"the other domain reached Gpus" bool true
      (poll (fun () -> Atomic.get about));
    sample ();
    Atomic.set inside false;
    Ok d
  in
  let d = require_ok (Gpus.open_pci g m 0 start) in
  require_ok (Domain.join d);
  equal ~msg:"the other driver ran" bool true (Atomic.get ran);
  equal ~msg:"the other driver ran inside GPU 0's" bool false
    (Atomic.get overlapped)

let others = [ ("an open", pci); ("a reset", reset) ]

(* While [lose] gives GPU 0's function back, another domain opens GPU 0: the
   release waits until the opener is about to call Gpus, then samples a window
   in which an open that is not held back sees the GPU free. *)
let test_lose_race () =
  let g, m, fake = three () in
  let h = hold g m 0 in
  let opener = ref None in
  let about = Atomic.make false in
  let open_ () =
    Atomic.set about true;
    pci g m 0 ok
  in
  fake.released <-
    (fun _ ->
      fake.released <- ignore;
      opener := Some (Domain.spawn open_);
      equal ~msg:"the opener reached Gpus" bool true
        (poll (fun () -> Atomic.get about));
      sample ());
  Gpus.lose h;
  let d = require_some !opener in
  ignore (require_error ~msg:"the open while it was lost" (Domain.join d))

(* While GPU 0's driver starts, the main domain gives GPU 1 back. The driver
   waits for it, so a give-back held back by the start would never return: the
   driver gives up after the hang guard and the test fails. *)
let test_give_back_waits (_, give_back) =
  let g, m, _ = three () in
  let h1 = hold g m 1 in
  let inside = Atomic.make false and back = Atomic.make false in
  let start _ _ =
    Atomic.set inside true;
    Ok (poll (fun () -> Atomic.get back))
  in
  let d = Domain.spawn (fun () -> Gpus.open_pci g m 0 start) in
  equal ~msg:"GPU 0's driver started" bool true
    (poll (fun () -> Atomic.get inside));
  give_back h1;
  Atomic.set back true;
  equal ~msg:"given back while GPU 0's driver started" bool true
    (require_ok (Domain.join d))

let give_backs = [ ("release", Gpus.release); ("lose", Gpus.lose) ]

(* A model *)

type state = Free | Held | Lost
type vendor_ref = { states : state array }
type hold_ref = { v : vendor_ref; i : int; mutable back : bool }
type vendor_sys = { g : Gpus.t; m : Machine.t; fake : fake }

(* An open refused without starting the driver, and one whose driver failed. *)
exception Refused
exception Driver_failed

type start = Starts | Fails | Raises_invalid

let pp_start ppf s =
  Format.pp_print_string ppf
    (match s with
    | Starts -> "starts"
    | Fails -> "fails"
    | Raises_invalid -> "raises-invalid")

let starts = Gen.of_list ~pp:pp_start [ Starts; Fails; Raises_invalid ]
let indices l = Gen.of_list ~pp:Format.pp_print_int l

let two_gpus =
  [ id "0000:03:00.0"; id ~class_:0x04 "0000:03:00.1"; id "0000:43:00.0" ]

let two_buses = [| "0000:03:00.0"; "0000:43:00.0" |]

let held_buses v =
  List.filteri (fun i _ -> v.states.(i) = Held) (Array.to_list two_buses)

let vendor_t =
  abstract "v" ~invariant:(fun v s ->
      equal ~msg:"functions taken" (slist string compare) (held_buses v)
        (taken s.fake))

let hold_t = abstract "h"
let vendor_ref () = { states = Array.make 2 Free }

let vendor_sys () =
  let m, fake = machine two_gpus in
  { g = gpus (); m; fake }

let check_index i = if i < 0 then invalid_arg "a negative GPU number"

let cover_lost v i =
  if i >= 0 && i < 2 then cover "a lost GPU is opened" (v.states.(i) = Lost)

let open_ref v start i =
  check_index i;
  cover_lost v i;
  if i >= 2 || v.states.(i) <> Free then raise Refused;
  match start with
  | Starts ->
      v.states.(i) <- Held;
      { v; i; back = false }
  | Fails -> raise Driver_failed
  | Raises_invalid -> invalid_arg "a driver bug"

let open_sys s start i =
  let started = ref false in
  let driver h fn =
    started := true;
    equal ~msg:"its function" string (Gpus.bus h) (Function.bus fn);
    equal ~msg:"GPU i" string two_buses.(i) (Gpus.bus h);
    match start with
    | Starts -> Ok h
    | Fails -> Error "the GPU did not start"
    | Raises_invalid -> invalid_arg "a driver bug"
  in
  match Gpus.open_pci s.g s.m i driver with
  | Ok h -> h
  | Error _ -> if !started then raise Driver_failed else raise Refused

let try_open_ref v i =
  check_index i;
  cover_lost v i;
  let free = i < 2 && v.states.(i) = Free in
  if free then v.states.(i) <- Held;
  free

let try_open_sys s i = Result.is_ok (pci s.g s.m i ok)

let give_back_ref state h =
  if h.back then invalid_arg "given back already";
  h.back <- true;
  h.v.states.(h.i) <- state

let reset_ref v i =
  check_index i;
  let free = i < 2 && v.states.(i) <> Held in
  if free then cover "a lost GPU is reset" (v.states.(i) = Lost);
  if free then v.states.(i) <- Free;
  free

let reset_sys s i =
  let reset fn =
    equal ~msg:"GPU i" string two_buses.(i) (Function.bus fn);
    Ok ()
  in
  Result.is_ok (Gpus.reset s.g s.m i reset)

(* Two domains contend for GPU 0 alone, so that they meet. *)
let commands index =
  [
    command "vendor" (Gen.unit @-> makes vendor_t) vendor_ref vendor_sys;
    command "open_pci"
      (vendor_t ^-> starts @-> index @-> makes hold_t)
      open_ref open_sys;
    command "open_pci, kept"
      (vendor_t ^-> index @-> returns bool)
      try_open_ref try_open_sys;
    command "release"
      (hold_t ^-> returns unit)
      (give_back_ref Free) Gpus.release;
    command "lose" (hold_t ^-> returns unit) (give_back_ref Lost) Gpus.lose;
    command "reset" (vendor_t ^-> index @-> returns bool) reset_ref reset_sys;
  ]

let serialized =
  group ~timeout:patience "opens and changes"
    [
      stateful "opens, gives back and resets behave as the model" ~count:300
        ~steps:30
        (commands (indices [ 0; 1; 2; -1 ]));
      stateful "from two domains, as some order of the calls" ~domains:2
        ~count:100
        (commands (indices [ 0 ]));
      cases "opens and resets run their drivers one at a time (sampled)"
        ~name:fst others test_one_at_a_time;
      test "a GPU lost while another domain opens it stays lost (sampled)"
        test_lose_race;
      cases "giving a GPU back waits for no driver's start" ~name:fst give_backs
        test_give_back_waits;
    ]

(* Changes on a host's files

   A host in a fixture tree keeps what a change writes as plain files and acts
   on none of it: an unbound driver stays linked, a removed function stays
   listed. Each case states what a change writes there and how it answers for
   the state that remains. Changes lock the function's file, which needs
   Linux. *)

let gpu_bus = "0000:03:00.0"

let audio bus =
  { (Host.gpu ~driver:"snd_hda_intel" bus) with class_ = 0x04; bars = [] }

let needs_flock () =
  if not on_linux then skip ~reason:"flock on a function's file needs Linux" ()

(* The trimmed contents of [file] under the host's [sys/bus/pci]. *)
let pci_file root file =
  Filename.concat root ("sys/bus/pci/" ^ file)
  |> (fun f -> In_channel.with_open_bin f In_channel.input_all)
  |> String.trim

let devices bus file = Printf.sprintf "devices/%s/%s" bus file

let changes =
  let gpu = Host.gpu gpu_bus in
  [
    ( "detach leaves a GPU a process can take as it is",
      [ gpu ],
      `Detach,
      None,
      [ (devices gpu_bus "enable", "1"); (devices gpu_bus "remove", "") ] );
    ( "detach enables a disabled GPU",
      [ Host.gpu ~enabled:false gpu_bus ],
      `Detach,
      None,
      [ (devices gpu_bus "enable", "1") ] );
    ( "detach unbinds the kernel driver, refused while it stays bound",
      [ Host.gpu ~driver:"amdgpu" gpu_bus ],
      `Detach,
      Some "the driver amdgpu stays bound to 0000:03:00.0",
      [ ("drivers/amdgpu/unbind", gpu_bus) ] );
    ( "detach removes the other functions of the GPU's device, refused while \
       they stay",
      [ gpu; audio "0000:03:00.1" ],
      `Detach,
      Some "0000:03:00.0 still shares its device with 0000:03:00.1",
      [ (devices "0000:03:00.1" "remove", "1") ] );
    ( "detach leaves a GPU bound to vfio-pci behind an IOMMU",
      [ Host.gpu ~driver:"vfio-pci" ~group:"12" gpu_bus ],
      `Detach,
      None,
      [ ("drivers/vfio-pci/unbind", "") ] );
    ( "detach refuses a GPU whose addresses an IOMMU translates",
      [ Host.gpu ~group:"12" gpu_bus ],
      `Detach,
      Some "the IOMMU translates the addresses 0000:03:00.0 reaches",
      [] );
    ( "attach probes the drivers for an unbound GPU, refused when none takes it",
      [ gpu ],
      `Attach,
      Some "no kernel driver took 0000:03:00.0",
      [
        (devices gpu_bus "enable", "0");
        ("rescan", "1");
        ("drivers_probe", gpu_bus);
      ] );
    ( "attach leaves a GPU bound to its kernel driver",
      [ Host.gpu ~driver:"amdgpu" gpu_bus ],
      `Attach,
      None,
      [ ("rescan", ""); ("drivers_probe", "") ] );
    ( "attach refuses a GPU bound to vfio-pci, naming the command that unbinds \
       it",
      [ Host.gpu ~driver:"vfio-pci" gpu_bus ],
      `Attach,
      Some "sudo driverctl unset-override 0000:03:00.0",
      [ ("rescan", "") ] );
  ]

let test_change (_, fns, change, refusal, files) =
  needs_flock ();
  let root = Host.make fns in
  let m = Machine.at root in
  let change =
    match change with `Detach -> Gpus.detach | `Attach -> Gpus.attach
  in
  (match (refusal, change (gpus ()) m 0) with
  | None, r -> require_ok r
  | Some sub, r -> contains ~sub (require_error r));
  List.iter
    (fun (file, want) -> equal ~msg:file string want (pci_file root file))
    files

let test_change_held () =
  needs_flock ();
  let g = gpus () and m = Machine.at (Host.make [ Host.gpu gpu_bus ]) in
  let h = hold g m 0 in
  contains ~msg:"detach" ~sub:"0000:03:00.0 is open in this process"
    (require_error (Gpus.detach g m 0));
  contains ~msg:"attach" ~sub:"0000:03:00.0 is open in this process"
    (require_error (Gpus.attach g m 0));
  Gpus.release h

let test_change_unwritable () =
  needs_flock ();
  if Unix.geteuid () = 0 then
    skip ~reason:"root writes a file whatever its mode" ();
  let root = Host.make [ Host.gpu ~enabled:false gpu_bus ] in
  let enable =
    Filename.concat root ("sys/bus/pci/" ^ devices gpu_bus "enable")
  in
  Unix.chmod enable 0o444;
  let why = require_error (Gpus.detach (gpus ()) (Machine.at root) 0) in
  contains ~msg:"names the file" ~sub:enable why;
  contains ~msg:"names the privilege" ~sub:"run as root" why

let test_change_transport () =
  let m, _ = machine functions in
  List.iter
    (fun (msg, change) ->
      contains ~msg ~sub:"far:1 is reached through a transport"
        (require_error (change (gpus ()) m 0)))
    [ ("detach", Gpus.detach); ("attach", Gpus.attach) ]

let host_changes =
  group ~timeout:patience "changes on a host's files"
    [
      cases "each change writes what it says and answers for what remains"
        ~name:(fun (n, _, _, _, _) -> n)
        changes test_change;
      test "a GPU the process holds is refused" test_change_held;
      test "a file the process may not write is refused, naming it"
        test_change_unwritable;
      test "another machine reached through a transport is refused"
        test_change_transport;
    ]

let () =
  exit
  @@ run "device_pci Gpus"
       [
         numbering;
         opening;
         giving_back;
         resets;
         host_changes;
         this_machine;
         serialized;
       ]
