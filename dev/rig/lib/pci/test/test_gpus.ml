(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Rig_pci
open Rig_pci_support

let strf = Printf.sprintf

(* Errors *)

let err_driver_bug = "a driver bug"

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
    map = (fun ~combine:_ _ off n -> Ok (Window.through tr off n));
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

let id ?(vendor = vendor) ?(class_ = 0x030000) bus =
  { Machine.bus; vendor; device = 0x73bf; class_ }

let is_gpu (id : Machine.id) = id.vendor = vendor && id.class_ lsr 16 = 0x03

(* Its kernel driver serves a GPU through character devices that [/sys/bus/pci]
   lists. *)
let no_nodes ~read:_ _ = []

(* The resets the vendor ran, unless a test gives its own reset. *)
let vendor_resets = Atomic.make 0

let counted _ =
  Atomic.incr vendor_resets;
  Ok ()

let gpus ?(reset = counted) ?(nodes = no_nodes) () =
  Gpus.make ~memory_bar:0 ~nodes ~reset is_gpu

let gpu_buses = [ "0000:03:00.0"; "0000:43:00.0"; "0000:c3:00.0" ]

let functions =
  [
    id "0000:03:00.0";
    id ~class_:0x040300 "0000:03:00.1";
    id ~vendor:0x10de "0000:21:00.0";
    id "0000:43:00.0";
    id "0000:c3:00.0";
  ]

let three () =
  let m, fake = machine functions in
  (gpus (), m, fake)

(* Opening with a driver [d] that starts nothing. *)

let ok () = Ok ()
let open_ g m i d = Gpus.open_ g m i ~at_exit:ignore (fun _ _ -> d ())

(* [reset g m i d] resets GPU [i], calling [d] once per reset the vendor ran. *)
let reset g m i d =
  let before = Atomic.get vendor_resets in
  let r = Gpus.reset g m i in
  for _ = before + 1 to Atomic.get vendor_resets do
    ignore (d ())
  done;
  r

let hold g m i = require_ok (Gpus.open_ g m i ~at_exit:ignore (fun h _ -> Ok h))

let hold_fn g m i =
  require_ok (Gpus.open_ g m i ~at_exit:ignore (fun h fn -> Ok (h, fn)))

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
(* The refusal of a GPU on a machine that has none says so. *)
let has_none why = contains ~sub:"the machine has none" why

let names n why =
  let digits c = if c >= '0' && c <= '9' then c else ' ' in
  let words = String.split_on_char ' ' (String.map digits why) in
  satisfies
    ~claim:(strf "names the number %d" n)
    string
    (fun _ -> List.mem (string_of_int n) words)
    why

(* Numbering *)

(* Functions in bus order whose spellings sort otherwise: a domain of five
   digits after ones of four. *)
let pool =
  [
    id "0000:00:01.0";
    id ~class_:0x040300 "0000:00:01.1";
    id "0000:0a:00.0";
    id ~vendor:0x10de "0000:10:00.0";
    id "0000:10:00.1";
    id "0001:00:00.0";
    id "2000:00:00.0";
    id ~vendor:0x10de "10000:00:00.0";
    id "10000:00:1f.7";
  ]

let pp_id ppf (id : Machine.id) =
  Format.fprintf ppf "%s/%04x/%06x" id.bus id.vendor id.class_

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
  let open_ith i bus =
    let msg = strf "GPU %d" i in
    let h =
      require_ok ~msg
        (Gpus.open_ g m i ~at_exit:ignore (fun h fn ->
             equal ~msg string bus (Function.bus fn);
             equal ~msg (option string) (Machine.name m)
               (Machine.name (Function.machine fn));
             Ok h))
    in
    equal ~msg string bus (Gpus.bus h)
  in
  List.iteri open_ith gpu_buses

let test_reset_ith () =
  let m, _ = machine functions in
  let seen = ref [] in
  let g =
    gpus
      ~reset:(fun fn ->
        seen := Function.bus fn :: !seen;
        Ok ())
      ()
  in
  let reset i bus =
    seen := [];
    require_ok (Gpus.reset g m i);
    equal ~msg:(strf "GPU %d" i) (list string) [ bus ] !seen
  in
  List.iteri reset gpu_buses

let no_gpu =
  List.concat_map
    (fun (name, f) -> List.map (fun i -> (name, f, i)) [ 3; 4; 7; max_int ])
    [ ("open_", open_); ("reset", reset) ]

let test_no_gpu (_, f, i) =
  let g, m, _ = three () in
  names 3 (unopened (f g m i))

let test_none () =
  let m, _ = machine [ id ~class_:0x040300 "0000:03:00.1" ] in
  let g = gpus () in
  equal (list string) [] (Gpus.buses g m);
  has_none (unopened (open_ g m 0));
  has_none (unopened (reset g m 0))

let negative =
  let far () = fst (machine functions) in
  List.concat_map
    (fun (name, f) -> List.map (fun i -> (name, f, i)) [ -1; min_int ])
    [
      ("open_", fun i -> ignore (open_ (gpus ()) (far ()) i ok));
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
        ~name:(fun (f, _, i) -> strf "%s %d" f i)
        no_gpu test_no_gpu;
      test "a machine without the vendor's GPUs has none to open" test_none;
      cases "a negative GPU number raises Invalid_argument"
        ~name:(fun (f, _, i) -> strf "%s %d" f i)
        negative
        (fun (_, f, i) ->
          raises_match (Exn.invalid_arg ?substring:None) (fun () -> f i));
    ]

(* Opening *)

let test_result () =
  let g, m, _ = three () in
  equal (result int string) (Ok 42)
    (Gpus.open_ g m 0 ~at_exit:ignore (fun _ _ -> Ok 42));
  equal (result int string) (Error "the GPU did not start")
    (Gpus.open_ g m 1 ~at_exit:ignore (fun _ _ -> Error "the GPU did not start"))

(* [given_back g m fake] asserts that GPU 0 and its function were given back. *)
let given_back g m fake =
  equal ~msg:"functions taken" taken_w [] (taken fake);
  ignore (hold g m 0)

let test_error_gives_back () =
  let g, m, fake = three () in
  ignore
    (Gpus.open_ g m 0 ~at_exit:ignore (fun _ _ -> Error "the GPU did not start"));
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
      ( "open_",
        fun g m e -> Gpus.open_ g m 0 ~at_exit:ignore (fun _ _ -> raise e) );
      ("reset", fun g m _ -> Gpus.reset g m 0);
    ]

let test_passed (_, run, e) =
  let m, fake = machine functions in
  let g = gpus ~reset:(fun _ -> raise e) () in
  raises e (fun () -> run g m e);
  given_back g m fake

let test_held () =
  let g, m, _ = three () in
  ignore (hold g m 0);
  ignore (unopened (open_ g m 0))

let test_take_refused () =
  let g, m, fake = three () in
  fake.refusal <- Some "0000:03:00.0 is held by process 4242";
  equal string "0000:03:00.0 is held by process 4242" (unopened (open_ g m 0))

let test_per_machine () =
  let g = gpus () in
  let m1, _ = machine ~name:"far:1" functions in
  let m2, _ = machine ~name:"far:2" functions in
  let h1 = hold g m1 0 in
  let h2 = hold g m2 0 in
  Gpus.lose h1;
  Gpus.release h2;
  ignore (hold g m2 0);
  ignore (unopened (open_ g m1 0))

let opening =
  group ~timeout:patience "opening"
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
  ignore (unopened (open_ g m 0));
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
      test "a lost GPU opens again only after a reset" test_lose;
      cases
        "a hold given back twice raises Invalid_argument and changes nothing"
        ~name:(fun (a, _, b, _) -> a ^ " then " ^ b)
        twice test_twice;
    ]

(* Resets *)

let test_reset () =
  let m, fake = machine functions in
  let seen = ref [] in
  let reset_gpu fn =
    equal ~msg:"taken during the reset" taken_w [ "0000:43:00.0" ] (taken fake);
    equal (option string) (Machine.name m) (Machine.name (Function.machine fn));
    seen := fn :: !seen;
    Ok ()
  in
  require_ok (Gpus.reset (gpus ~reset:reset_gpu ()) m 1);
  let fn = require_some (List.nth_opt !seen 0) in
  equal ~msg:"calls" int 1 (List.length !seen);
  equal ~msg:"released" bool true (Function.released fn);
  equal ~msg:"functions taken" taken_w [] (taken fake)

let test_reset_held () =
  let g, m, _ = three () in
  ignore (hold g m 0);
  ignore (unopened (reset g m 0))

let test_reset_failure () =
  let m, fake = machine functions in
  let g = gpus ~reset:(fun _ -> Error "the GPU did not come back") () in
  let why = require_error (Gpus.reset g m 0) in
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

let test_this_none () =
  let g =
    Gpus.make ~memory_bar:0 ~nodes:no_nodes ~reset:counted (fun _ -> false)
  in
  let this = Machine.this in
  equal (list string) [] (Gpus.buses g this);
  has_none (unopened (open_ g this 0));
  has_none (unopened (reset g this 0));
  ignore (require_error (Gpus.detach g this 0));
  ignore (require_error (Gpus.attach g this 0))

let this_machine =
  group ~timeout:patience "this machine"
    [
      test "this machine without the vendor's GPUs has none to open or change"
        test_this_none;
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
  let d = require_ok (Gpus.open_ g m 0 ~at_exit:ignore start) in
  require_ok (Domain.join d);
  equal ~msg:"the other driver ran" bool true (Atomic.get ran);
  equal ~msg:"the other driver ran inside GPU 0's" bool false
    (Atomic.get overlapped)

let others = [ ("an open", open_); ("a reset", reset) ]

(* While [lose] gives GPU 0's function back, another domain opens GPU 0: the
   release waits until the opener is about to call Gpus, then samples a window
   in which an open that is not held back sees the GPU free. *)
let test_lose_race () =
  let g, m, fake = three () in
  let h = hold g m 0 in
  let opener = ref None in
  let about = Atomic.make false in
  let try_open () =
    Atomic.set about true;
    open_ g m 0 ok
  in
  fake.released <-
    (fun _ ->
      fake.released <- ignore;
      opener := Some (Domain.spawn try_open);
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
  let d = Domain.spawn (fun () -> Gpus.open_ g m 0 ~at_exit:ignore start) in
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

(* An open that raised [Invalid_argument] for a driver that gave its GPU back
   and answered [Ok]. *)
exception Misused

(* How the driver's start ends. A driver that wrote to the GPU before it failed
   gives it back inside the open: lost, or released. Giving it back and
   answering [Ok] is a driver bug. *)
type start =
  | Starts
  | Fails
  | Raises_invalid
  | Loses_and_fails
  | Releases_and_fails
  | Loses_and_starts

let pp_start ppf s =
  Format.pp_print_string ppf
    (match s with
    | Starts -> "starts"
    | Fails -> "fails"
    | Raises_invalid -> "raises-invalid"
    | Loses_and_fails -> "loses-and-fails"
    | Releases_and_fails -> "releases-and-fails"
    | Loses_and_starts -> "loses-and-starts")

let starts =
  Gen.of_list ~pp:pp_start
    [
      Starts;
      Fails;
      Raises_invalid;
      Loses_and_fails;
      Releases_and_fails;
      Loses_and_starts;
    ]

let indices l = Gen.of_list ~pp:Format.pp_print_int l

let two_gpus =
  [ id "0000:03:00.0"; id ~class_:0x040300 "0000:03:00.1"; id "0000:43:00.0" ]

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
  | Fails | Releases_and_fails -> raise Driver_failed
  | Raises_invalid -> invalid_arg err_driver_bug
  | Loses_and_fails ->
      v.states.(i) <- Lost;
      raise Driver_failed
  | Loses_and_starts ->
      v.states.(i) <- Lost;
      raise Misused

let open_sys s start i =
  let started = ref false in
  let driver h fn =
    started := true;
    equal ~msg:"its function" string (Gpus.bus h) (Function.bus fn);
    equal ~msg:"GPU i" string two_buses.(i) (Gpus.bus h);
    match start with
    | Starts -> Ok h
    | Fails -> Error "the GPU did not start"
    | Raises_invalid -> invalid_arg err_driver_bug
    | Loses_and_fails ->
        Gpus.lose h;
        Error "the GPU did not start"
    | Releases_and_fails ->
        Gpus.release h;
        Error "the GPU did not start"
    | Loses_and_starts ->
        Gpus.lose h;
        Ok h
  in
  match Gpus.open_ s.g s.m i ~at_exit:ignore driver with
  | Ok h -> h
  | Error _ -> if !started then raise Driver_failed else raise Refused
  | exception Invalid_argument _ when start = Loses_and_starts && !started ->
      raise Misused

let try_open_ref v i =
  check_index i;
  cover_lost v i;
  let free = i < 2 && v.states.(i) = Free in
  if free then v.states.(i) <- Held;
  free

let try_open_sys s i = Result.is_ok (open_ s.g s.m i ok)

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

let reset_sys s i = Result.is_ok (Gpus.reset s.g s.m i)

(* Two domains contend for GPU 0 alone, so that they meet. *)
let commands index =
  [
    command "vendor" (Gen.unit @-> makes vendor_t) vendor_ref vendor_sys;
    command "open_"
      (vendor_t ^-> starts @-> index @-> makes hold_t)
      open_ref open_sys;
    command "open_, kept"
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

(* Changes on a machine's files

   A machine in a fixture tree keeps what a change writes as plain files and
   acts on none of it: an unbound driver stays linked, a removed function stays
   listed. Each case states what a change writes there and how it answers for
   the state that remains. Changes lock the function's file, which needs
   Linux. *)

let gpu_bus = "0000:03:00.0"

let audio bus =
  { (Tree.gpu ~driver:"snd_hda_intel" bus) with class_ = 0x040300; bars = [] }

let needs_flock () =
  if not on_linux then skip ~reason:"flock on a function's file needs Linux" ()

(* The trimmed contents of [file] under the machine's [sys/bus/pci]. *)
let pci_file root file =
  Filename.concat root ("sys/bus/pci/" ^ file)
  |> (fun f -> In_channel.with_open_bin f In_channel.input_all)
  |> String.trim

let devices bus file = strf "devices/%s/%s" bus file
let sys file = "sys/bus/pci/" ^ file

(* A function's command register, its bit that lets the function reach memory
   and its bit that lets it master the bus (PCI Express Base Specification,
   7.5.1.1.3). *)
let command = 0x04
let memory_space = 0x2
let bus_master = 0x4
let intx_disable = 0x400

let changes =
  let gpu = Tree.gpu gpu_bus in
  [
    ( "detach leaves a GPU a process can take as it is, but for kernel drivers",
      [ gpu ],
      `Detach,
      None,
      [
        (devices gpu_bus "enable", "1");
        (devices gpu_bus "remove", "");
        (devices gpu_bus "driver_override", "none");
      ] );
    ( "detach enables a disabled GPU",
      [ Tree.gpu ~enabled:false gpu_bus ],
      `Detach,
      None,
      [ (devices gpu_bus "enable", "1") ] );
    ( "detach unbinds the kernel driver, refused while it stays bound",
      [ Tree.gpu ~driver:"amdgpu" gpu_bus ],
      `Detach,
      Some "the driver amdgpu stays bound to 0000:03:00.0",
      [
        ("drivers/amdgpu/unbind", gpu_bus);
        (devices gpu_bus "driver_override", "none");
      ] );
    ( "detach removes the other functions of the GPU's device, refused while \
       they stay",
      [ gpu; audio "0000:03:00.1" ],
      `Detach,
      Some "0000:03:00.0 still shares its device with 0000:03:00.1",
      [ (devices "0000:03:00.1" "remove", "1") ] );
    ( "detach leaves a GPU bound to vfio-pci behind an IOMMU",
      [ Tree.gpu ~driver:"vfio-pci" ~group:"12" gpu_bus ],
      `Detach,
      None,
      [
        ("drivers/vfio-pci/unbind", "");
        (devices gpu_bus "driver_override", "(null)");
      ] );
    ( "detach refuses a GPU whose addresses an IOMMU translates, writing nothing",
      [ Tree.gpu ~group:"12" gpu_bus ],
      `Detach,
      Some "the IOMMU translates the addresses 0000:03:00.0 reaches",
      [ (devices gpu_bus "driver_override", "(null)") ] );
    ( "detach refuses a GPU bound to its kernel driver behind a translating \
       IOMMU, writing nothing",
      [ Tree.gpu ~driver:"amdgpu" ~group:"12" gpu_bus ],
      `Detach,
      Some "the IOMMU translates the addresses 0000:03:00.0 reaches",
      [
        ("drivers/amdgpu/unbind", "");
        (devices gpu_bus "driver_override", "(null)");
      ] );
    ( "detach refuses a GPU under a locked-down kernel, writing nothing",
      [ Tree.gpu ~driver:"amdgpu" gpu_bus ],
      `Detach_locked_down,
      Some "the kernel is locked down",
      [
        ("drivers/amdgpu/unbind", "");
        (devices gpu_bus "driver_override", "(null)");
      ] );
    ( "attach probes the drivers for an unbound GPU, refused when none takes it",
      [ gpu ],
      `Attach,
      Some "no kernel driver took 0000:03:00.0",
      [
        (devices gpu_bus "enable", "0");
        (devices gpu_bus "driver_override", "");
        ("rescan", "1");
        ("drivers_probe", gpu_bus);
      ] );
    ( "attach leaves a GPU bound to its kernel driver",
      [ Tree.gpu ~driver:"amdgpu" gpu_bus ],
      `Attach,
      None,
      [
        ("rescan", "");
        ("drivers_probe", "");
        (devices gpu_bus "driver_override", "(null)");
      ] );
    ( "attach leaves a GPU on vfio-pci whose function cannot be taken as it was",
      [ Tree.gpu ~driver:"vfio-pci" gpu_bus ],
      `Attach,
      Some "0000:03:00.0",
      [
        ("drivers/vfio-pci/unbind", "");
        (devices gpu_bus "driver_override", "(null)");
        ("rescan", "");
        ("drivers_probe", "");
      ] );
  ]

let test_change (_, fns, change, refusal, files) =
  needs_flock ();
  let lockdown =
    match change with
    | `Detach_locked_down -> Some "none [integrity] confidentiality"
    | `Detach | `Attach -> None
  in
  let root = Tree.make ?lockdown fns in
  let m = Machine.at root in
  let change =
    match change with
    | `Detach | `Detach_locked_down -> Gpus.detach
    | `Attach -> Gpus.attach
  in
  (match (refusal, change (gpus ()) m 0) with
  | None, r -> require_ok r
  | Some sub, r -> contains ~sub (require_error r));
  List.iter
    (fun (file, want) -> equal ~msg:file string want (pci_file root file))
    files

(* Turns the fixture GPU's bus mastering on, as a driver that wrote to it may
   leave it. *)
let mastering root =
  let config =
    Filename.concat root ("sys/bus/pci/" ^ devices gpu_bus "config")
  in
  let fd = Unix.openfile config [ O_WRONLY ] 0 in
  Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
  ignore (Unix.lseek fd command SEEK_SET);
  let b = Bytes.create 2 in
  Bytes.set_uint16_le b 0 (memory_space lor bus_master);
  ignore (Unix.write fd b 0 2)

(* The vendor's reset sees the GPU taken, its bus mastering off, before the bus
   is rescanned and its drivers probed; the GPU is free after. *)
let test_attach_resets () =
  needs_flock ();
  let root = Tree.make [ Tree.gpu gpu_bus ] in
  mastering root;
  Tree.add root (sys (devices gpu_bus "driver_override")) "none\n";
  let seen = ref [] in
  let reset fn =
    seen :=
      ( Function.config16 fn command land lnot intx_disable,
        pci_file root (devices gpu_bus "driver_override"),
        pci_file root "rescan" ^ pci_file root "drivers_probe" )
      :: !seen;
    Ok ()
  in
  let m = Machine.at root in
  contains ~sub:"no kernel driver took 0000:03:00.0"
    (require_error (Gpus.attach (gpus ~reset ()) m 0));
  equal ~msg:"what the reset saw"
    (list (triple hex string string))
    [ (memory_space, "none", "") ]
    !seen;
  equal ~msg:"no driver kept off after" string ""
    (pci_file root (devices gpu_bus "driver_override"));
  equal ~msg:"drivers probed after" string gpu_bus
    (pci_file root "drivers_probe");
  Tree.add root (sys (devices gpu_bus "enable")) "1\n";
  Function.release (require_ok (Function.take m gpu_bus))

let test_attach_refused () =
  needs_flock ();
  let root = Tree.make [ Tree.gpu gpu_bus ] in
  Tree.add root (sys (devices gpu_bus "driver_override")) "none\n";
  let reset _ = Error "the GPU did not come back" in
  contains ~sub:"the GPU did not come back"
    (require_error (Gpus.attach (gpus ~reset ()) (Machine.at root) 0));
  List.iter
    (fun (file, want) -> equal ~msg:file string want (pci_file root file))
    [
      (devices gpu_bus "enable", "1");
      (devices gpu_bus "driver_override", "none");
      ("rescan", "");
      ("drivers_probe", "");
    ]

let test_attach_bound () =
  needs_flock ();
  let root = Tree.make [ Tree.gpu ~driver:"amdgpu" gpu_bus ] in
  let before = Atomic.get vendor_resets in
  require_ok (Gpus.attach (gpus ()) (Machine.at root) 0);
  equal ~msg:"resets" int 0 (Atomic.get vendor_resets - before)

(* A GPU this process lost opens again once attach has reset it. *)
let test_attach_lost () =
  needs_flock ();
  let root = Tree.make [ Tree.gpu gpu_bus ] in
  let g = gpus () and m = Machine.at root in
  Gpus.lose (hold g m 0);
  ignore (unopened (open_ g m 0));
  contains ~sub:"no kernel driver took 0000:03:00.0"
    (require_error (Gpus.attach g m 0));
  Tree.add root (sys (devices gpu_bus "enable")) "1\n";
  Gpus.release (hold g m 0)

let test_change_held () =
  needs_flock ();
  let g = gpus () and m = Machine.at (Tree.make [ Tree.gpu gpu_bus ]) in
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
  let root = Tree.make [ Tree.gpu ~enabled:false gpu_bus ] in
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

(* Open devices

   Unbinding a kernel driver waits until no process holds one of its devices
   open. The fixture's descriptors are links in its [proc/self/fd], to this
   machine's [/dev/null] and [/dev/zero]; a GPU's devices are its [dev] files,
   holding [/dev/null]'s number or another, and the vendor's nodes. *)

(* The vendor's nodes as NVIDIA's driver gives them: the GPU's minor in [/proc],
   its node [/dev/nvidiaN]. *)
let nvidia_nodes ~read bus =
  match read (strf "proc/driver/nvidia/gpus/%s/information" bus) with
  | None -> []
  | Some info ->
      List.filter_map
        (fun line ->
          match String.split_on_char ':' line with
          | [ "Device Minor"; n ] -> Some ("dev/nvidia" ^ String.trim n)
          | _ -> None)
        (String.split_on_char '\n' info)

let render = devices gpu_bus "drm/renderD128/dev"

let open_devices =
  let null root = Tree.add root (sys render) (Tree.device_number "/dev/null") in
  let fd n target root = Tree.link root ("proc/self/fd/" ^ n) target in
  let nvidia root =
    Tree.add root
      (strf "proc/driver/nvidia/gpus/%s/information" gpu_bus)
      "Model: a GPU\nDevice Minor: \t 0\n";
    Tree.link root "dev/nvidia0" "/dev/null"
  in
  let all fs root = List.iter (fun f -> f root) fs in
  [
    ( "a DRM node under the GPU's directory is refused",
      all [ null; fd "7" "/dev/null" ],
      no_nodes,
      `Refused "/dev/null" );
    ( "a node the vendor names is refused",
      all [ nvidia; fd "9" "/dev/null" ],
      nvidia_nodes,
      `Refused "/dev/null" );
    ( "other devices and files open are no hold",
      all
        [
          null;
          nvidia;
          fd "3" "/dev/zero";
          fd "4" "../../../sys/bus/pci/rescan";
          fd "5" "../../../closed";
        ],
      no_nodes,
      `Detached );
    ( "a node the vendor names that is absent is no hold",
      fd "9" "/dev/null",
      nvidia_nodes,
      `Detached );
    ( "an unreadable proc/self/fd is an Error",
      (fun root -> Unix.rmdir (Filename.concat root "proc/self/fd")),
      no_nodes,
      `Error "proc/self/fd" );
    ( "a dev file that holds no number is an Error",
      (fun root -> Tree.add root (sys render) "renderD128\n"),
      no_nodes,
      `Error "is no device number" );
  ]

let test_open_device (_, setup, nodes, want) =
  needs_flock ();
  let root = Tree.make [ Tree.gpu gpu_bus ] in
  setup root;
  let g = gpus ~nodes () in
  match (want, Gpus.detach g (Machine.at root) 0) with
  | `Detached, r -> require_ok r
  | `Refused file, r ->
      let why = require_error r in
      contains ~sub:"0000:03:00.0 is open in this process" why;
      contains ~msg:"names the device" ~sub:file why
  | `Error sub, r -> contains ~sub (require_error r)

let tree_changes =
  group ~timeout:patience "changes on a machine's files"
    [
      cases "each change writes what it says and answers for what remains"
        ~name:(fun (n, _, _, _, _) -> n)
        changes test_change;
      test "a GPU the process holds is refused" test_change_held;
      test
        "attach resets the GPU, its bus mastering off, before the bus is \
         rescanned"
        test_attach_resets;
      test "attach leaves the GPU detached when its reset fails"
        test_attach_refused;
      test "attach leaves a GPU bound to its kernel driver unreset"
        test_attach_bound;
      test "a GPU lost opens again after attach reset it" test_attach_lost;
      cases "detach refuses a GPU whose device this process holds open"
        ~name:(fun (n, _, _, _) -> n)
        open_devices test_open_device;
      test "a file the process may not write is refused, naming it"
        test_change_unwritable;
      test "another machine reached through a transport is refused"
        test_change_transport;
    ]

(* Exit

   A process that exits holding GPUs stops them. The suite's executable, run
   with [exit_holding], is such a process; the test reads what it printed. *)

let exit_holding = "--exit-holding"

(* Holds GPUs 0 and 1 of a fake machine, whose stops raise and print, holds and
   gives back GPU 2, and forks a child that exits, before it exits. Given a
   fixture tree [root], it also holds the tree's GPU with its bus mastering on,
   and prints at exit whether it still is. *)
let exit_holding_gpus root =
  let g, m, _ = three () in
  let open_ i at_exit =
    Result.get_ok (Gpus.open_ g m i ~at_exit (fun h _ -> Ok h))
  in
  let stopped h = print_endline ("stopped " ^ Gpus.bus h) in
  ignore (open_ 0 (fun _ -> failwith "a driver bug"));
  ignore (open_ 1 stopped);
  Gpus.release (open_ 2 stopped);
  if root <> "-" then begin
    let mastering fn =
      print_endline
        (if Function.config16 fn command land bus_master <> 0 then "mastering"
         else "not mastering")
    in
    let start _ fn =
      Function.set_config16 fn command bus_master;
      Ok fn
    in
    let tree = Machine.at root in
    ignore
      (Result.get_ok (Gpus.open_ (gpus ()) tree 0 ~at_exit:mastering start))
  end;
  (match Unix.fork () with
  | 0 -> exit 0
  | child -> ignore (Unix.waitpid [] child));
  exit 0

(* The lines [exit_holding_gpus root] printed on standard output, and what it
   printed on standard error. *)
let exiting root =
  let exe = Sys.executable_name in
  let out_r, out_w = Unix.pipe ~cloexec:true () in
  let err_r, err_w = Unix.pipe ~cloexec:true () in
  let pid =
    Unix.create_process exe [| exe; exit_holding; root |] Unix.stdin out_w err_w
  in
  Unix.close out_w;
  Unix.close err_w;
  let exited () = fst (Unix.waitpid [ WNOHANG ] pid) <> 0 in
  if not (poll exited) then begin
    Unix.kill pid Sys.sigkill;
    failf "the process holding GPUs did not exit"
  end;
  let read fd = In_channel.input_all (Unix.in_channel_of_descr fd) in
  let out = read out_r and err = read err_r in
  Unix.close out_r;
  Unix.close err_r;
  (List.filter (( <> ) "") (String.split_on_char '\n' out), err)

let test_exit () =
  let out, err = exiting "-" in
  equal ~msg:"stopped, once" (list string) [ "stopped 0000:43:00.0" ] out;
  contains ~msg:"the stop that raised"
    ~sub:"stopping 0000:03:00.0 at exit: Failure(\"a driver bug\")" err

let test_exit_order () =
  needs_flock ();
  let out, _ = exiting (Tree.make [ Tree.gpu gpu_bus ]) in
  equal (slist string compare) [ "mastering"; "stopped 0000:43:00.0" ] out

let exits =
  group ~timeout:patience "exit"
    [
      test
        "a process that exits stops each GPU it holds once, past a stop that \
         raises, and its forked child stops none"
        test_exit;
      test "a GPU stops at exit before its bus mastering is turned off"
        test_exit_order;
    ]

let () =
  match Sys.argv with
  | [| _; arg; root |] when arg = exit_holding -> exit_holding_gpus root
  | _ ->
      hold_gpu ();
      exit
      @@ run "rig_pci.gpus"
           [
             numbering;
             opening;
             giving_back;
             resets;
             tree_changes;
             exits;
             this_machine;
             serialized;
           ]
