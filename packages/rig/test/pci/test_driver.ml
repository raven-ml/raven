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

(* The vendor's GPUs are its display controllers. Its audio functions and
   another vendor's display controllers are no GPUs of it. *)

let vendor = 0x1002

let id ?(vendor = vendor) ?(class_ = 0x030000) bus =
  { Machine.bus; vendor; device = 0x73bf; class_ }

let is_gpu (id : Machine.id) = id.vendor = vendor && id.class_ lsr 16 = 0x03

(* Its kernel driver serves a GPU through character devices that [/sys/bus/pci]
   lists. *)
let no_nodes ~root:_ _ = []

(* The resets the vendor ran, unless a test gives its own reset. *)
let vendor_resets = Atomic.make 0

let counted _ =
  Atomic.incr vendor_resets;
  Ok ()

(* Its kernel driver lets go of a GPU once no process holds its devices. *)
let released ~root:_ _ = None

(* How long detach waits for a kernel driver to let go: long enough for the
   tests that let go while it waits. *)
let teardown_ms = 2000

(* The vendor's name, which names its GPUs. *)
let name = "AMD-PCI"

let gpus ?(reset = counted) ?(nodes = no_nodes) ?(unreleased = released)
    ?(teardown_ms = teardown_ms) ?(siblings = Driver.Remove) () =
  Driver.make ~name ~memory_bar:(Some 0) ~siblings ~nodes ~unreleased
    ~teardown_ms ~reset is_gpu

let pp_progress ppf (d : Driver.progress) =
  match d with
  | Detached -> Format.pp_print_string ppf "Detached"
  | Unreleased why -> Format.fprintf ppf "Unreleased %S" why
  | Part_way why -> Format.fprintf ppf "Part_way %S" why

let progress = Testable.make ~pp:pp_progress ~equal:( = )

(* [r]'s reason of a detach that stopped part-way. *)
let part_way r =
  match require_ok ~pp:Format.pp_print_string r with
  | Driver.Part_way why -> why
  | d -> failf "detach went %a, not part-way" pp_progress d

let gpu_buses = [ "0000:03:00.0"; "0000:43:00.0"; "0000:c3:00.0" ]
let gpu_bus = "0000:03:00.0"

(* Takes lock a function's file, which needs flock. *)
let needs_flock () =
  if not flocks then skip ~reason:"locking a function's file needs flock" ()

let audio bus =
  { (Tree.gpu ~driver:"snd_hda_intel" bus) with class_ = 0x040300; bars = [] }

let other_vendor bus = { (Tree.gpu bus) with vendor = 0x10de }

(* The vendor's three GPUs among other functions. *)
let functions =
  [
    Tree.gpu "0000:03:00.0";
    audio "0000:04:00.1";
    other_vendor "0000:21:00.0";
    Tree.gpu "0000:43:00.0";
    Tree.gpu "0000:c3:00.0";
  ]

(* The vendor's GPUs on a new tree of [functions]: the GPUs, the machine and the
   tree's root. *)
let three () =
  let root = Tree.make functions in
  (gpus (), Machine.at root, root)

(* [held root bus] is [true] iff a take holds the function at [bus] of the tree
   [root]: its configuration file is locked. *)
let held root bus =
  Tree.flocked (Filename.concat root ("sys/bus/pci/devices/" ^ bus ^ "/config"))

(* Opening with a driver [d] that starts nothing. *)

let ok () = Ok ()

(* [kept h] gives [h] a clean stop and keeps it. *)
let kept h =
  Driver.set_stop h (fun () -> `Clean);
  Ok h

let open_ g m i d =
  Driver.open_ g m i (fun h _ ->
      let r = d () in
      if Result.is_ok r then ignore (kept h);
      r)

(* [reset g m i d] resets GPU [i], calling [d] once per reset the vendor ran. *)
let reset g m i d =
  let before = Atomic.get vendor_resets in
  let r = Driver.reset g m i in
  for _ = before + 1 to Atomic.get vendor_resets do
    ignore (d ())
  done;
  r

let hold g m i = require_ok (Driver.open_ g m i (fun h _ -> kept h))

(* [stopping g m i answer] holds GPU [i], whose vendor's stop answers
   [answer]. *)
let stopping g m i answer =
  require_ok
    (Driver.open_ g m i (fun h _ ->
         Driver.set_stop h (fun () -> answer);
         Ok h))

let stop h = ignore (Driver.stop h : [ `Stopped | `Unknown ])

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
let has_none why = contains ~sub:(name ^ " drives none here") why

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
  let fn (id : Machine.id) =
    { (Tree.gpu id.bus) with vendor = id.vendor; class_ = id.class_ }
  in
  let m = Machine.at (Tree.make (List.map fn ids)) in
  equal
    (result (list string) string)
    (Ok (gpus_of ids))
    (Driver.buses (gpus ()) m)

let test_ith () =
  needs_flock ();
  let g, m, _ = three () in
  let open_ith i bus =
    let msg = strf "GPU %d" i in
    stop
      (require_ok ~msg
         (Driver.open_ g m i (fun h fn ->
              equal ~msg string bus (Function.bus fn);
              equal ~msg (option string) (Machine.name m)
                (Machine.name (Function.machine fn));
              kept h)))
  in
  List.iteri open_ith gpu_buses

let test_reset_ith () =
  needs_flock ();
  let _, m, _ = three () in
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
    require_ok (Driver.reset g m i);
    equal ~msg:(strf "GPU %d" i) (list string) [ bus ] !seen
  in
  List.iteri reset gpu_buses

let no_gpu =
  List.concat_map
    (fun (name, f) -> List.map (fun i -> (name, f, i)) [ 3; 4; 7; max_int ])
    [ ("open_", open_); ("reset", reset) ]

let test_no_gpu (_, f, i) =
  let g, m, _ = three () in
  let why = unopened (f g m i) in
  starts_with ~msg:"names the GPU" ~affix:(Driver.name g i) why;
  names 3 why

let test_none () =
  let m = Machine.at (Tree.make [ audio "0000:03:00.1" ]) in
  let g = gpus () in
  equal (result (list string) string) (Ok []) (Driver.buses g m);
  has_none (unopened (open_ g m 0));
  has_none (unopened (reset g m 0))

let negative =
  let far () =
    let _, m, _ = three () in
    m
  in
  List.concat_map
    (fun (name, f) -> List.map (fun i -> (name, f, i)) [ -1; min_int ])
    [
      ("name", fun i -> ignore (Driver.name (gpus ()) i));
      ("open_", fun i -> ignore (open_ (gpus ()) (far ()) i ok));
      ("reset", fun i -> ignore (reset (gpus ()) (far ()) i ok));
      ("detach", fun i -> ignore (Driver.detach (gpus ()) Machine.this i));
      ("attach", fun i -> ignore (Driver.attach (gpus ()) Machine.this i));
    ]

(* GPU 0 bears the vendor's name, and GPU i that name and its number. *)
let test_name =
  cases ~name:(strf "GPU %d") "a GPU is named after its vendor and number"
    [ 0; 1; 12; max_int ] (fun i ->
      let want = if i = 0 then name else strf "%s:%d" name i in
      equal string want (Driver.name (gpus ()) i))

let numbering =
  group ~timeout:patience "numbering"
    [
      test_name;
      prop "a vendor's GPUs are the functions it recognizes, in bus order"
        (Gen.with_pp pp_ids (Gen.subsequence pool))
        test_in_order;
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

(* The driver's Error, as every Error about GPU i, starts with its name. *)
let test_result () =
  needs_flock ();
  let g, m, _ = three () in
  equal (result int string) (Ok 42)
    (Driver.open_ g m 0 (fun h _ -> Result.map (fun _ -> 42) (kept h)));
  let why =
    require_error (Driver.open_ g m 1 (fun _ _ -> Error "the GPU did not start"))
  in
  starts_with ~msg:"names the GPU" ~affix:(Driver.name g 1) why;
  ends_with ~msg:"the driver's" ~affix:"the GPU did not start" why

(* [given_back g m root] asserts that GPU 0's function was given back, and that
   the GPU opens again. *)
let given_back g m root =
  equal ~msg:"its function held" bool false (held root "0000:03:00.0");
  stop (hold g m 0)

let test_error_gives_back () =
  needs_flock ();
  let g, m, root = three () in
  ignore (Driver.open_ g m 0 (fun _ _ -> Error "the GPU did not start"));
  given_back g m root

let passed =
  List.concat_map
    (fun (op, run) ->
      List.map
        (fun (name, e) -> (op ^ " " ^ name, run, e, op = "reset"))
        [
          ("Invalid_argument", Invalid_argument "a bug");
          ("Not_found", Not_found);
          ("Failure", Failure "int_of_string");
          ("Sys_error", Sys_error "/dev/kfd: No such file");
          ("Unix_error", Unix.Unix_error (ENOENT, "open", "/dev/kfd"));
        ])
    [
      ("open_", fun g m e -> Driver.open_ g m 0 (fun _ _ -> raise e));
      ("reset", fun g m _ -> Driver.reset g m 0);
    ]

(* A driver's start that raises before its stop releases the GPU as found, and
   the next open does not renew it; a reset that raises loses the GPU, and the
   next open renews it, raising again. *)
let test_passed (_, run, e, lost) =
  needs_flock ();
  let _, m, root = three () in
  let g = gpus ~reset:(fun _ -> raise e) () in
  raises e (fun () -> run g m e);
  equal ~msg:"its function held" bool false (held root "0000:03:00.0");
  if lost then
    raises ~msg:"renewed by the next open" e (fun () -> open_ g m 0 ok)
  else stop (hold g m 0)

let test_held () =
  needs_flock ();
  let g, m, _ = three () in
  ignore (hold g m 0);
  ignore (unopened (open_ g m 0))

(* A GPU bound to a kernel driver cannot be taken. *)
let refusing () = Machine.at (Tree.make [ Tree.gpu ~driver:"amdgpu" gpu_bus ])

let test_take_refused () =
  needs_flock ();
  let g = gpus () in
  let why = unopened (open_ g (refusing ()) 0) in
  starts_with ~msg:"names the GPU" ~affix:(Driver.name g 0) why;
  contains ~msg:"the take's" ~sub:"0000:03:00.0 is bound to the driver amdgpu"
    why

(* The GPU lost on one machine is renewed there alone. *)
let test_per_machine () =
  needs_flock ();
  let g = gpus () in
  let _, m1, _ = three () and _, m2, _ = three () in
  let h1 = stopping g m1 0 `Lost in
  let h2 = stopping g m2 0 `Clean in
  stop h1;
  stop h2;
  let before = Atomic.get vendor_resets in
  stop (hold g m2 0);
  equal ~msg:"resets of the GPU released" int before (Atomic.get vendor_resets);
  stop (hold g m1 0);
  equal ~msg:"resets of the GPU lost" int (before + 1)
    (Atomic.get vendor_resets)

let opening =
  group ~timeout:patience "opening"
    [
      test "an open is the driver's result" test_result;
      test "an open the driver refuses gives the GPU and its function back"
        test_error_gives_back;
      cases
        "exceptions from the driver or the vendor's reset pass through, and \
         give the GPU back"
        ~name:(fun (n, _, _, _) -> n)
        passed test_passed;
      test "a GPU held is refused without calling the driver" test_held;
      test "a function that cannot be taken is refused with the take's reason"
        test_take_refused;
      test "the same bus on two machines is two GPUs" test_per_machine;
    ]

(* Stopping *)

let stopped =
  Testable.make
    ~pp:(fun ppf s ->
      Format.pp_print_string ppf
        (match s with `Stopped -> "`Stopped" | `Unknown -> "`Unknown"))
    ~equal:( = )

let pp_answer ppf a =
  Format.pp_print_string ppf
    (match a with
    | `Clean -> "`Clean"
    | `Lost -> "`Lost"
    | `Unknown -> "`Unknown")

let answers = [ `Clean; `Lost; `Unknown ]

(* Resets *)

(* The vendor's reset sees the function taken; it is released after. *)
let test_reset () =
  needs_flock ();
  let _, m, root = three () in
  let seen = ref [] in
  let reset_gpu fn =
    let bus = Function.bus fn in
    seen := (bus, Function.released fn, held root bus) :: !seen;
    Ok ()
  in
  require_ok (Driver.reset (gpus ~reset:reset_gpu ()) m 1);
  equal ~msg:"what the reset saw: bus, released, held"
    (list (triple string bool bool))
    [ ("0000:43:00.0", false, true) ]
    !seen;
  equal ~msg:"held after" bool false (held root "0000:43:00.0")

let test_reset_held () =
  needs_flock ();
  let g, m, _ = three () in
  ignore (hold g m 0);
  ignore (unopened (reset g m 0))

let test_reset_failure () =
  needs_flock ();
  let _, m, root = three () in
  let g = gpus ~reset:(fun _ -> Error "the GPU did not come back") () in
  let why = require_error (Driver.reset g m 0) in
  contains ~sub:"the GPU did not come back" why;
  equal ~msg:"held after" bool false (held root "0000:03:00.0")

let test_reset_take () =
  needs_flock ();
  ignore (unopened (reset (gpus ()) (refusing ()) 0))

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
    Driver.make ~name ~memory_bar:(Some 0) ~siblings:Remove
      ~nodes:no_nodes ~unreleased:released
      ~teardown_ms ~reset:counted (fun _ -> false)
  in
  let this = Machine.this in
  equal (result (list string) string) (Ok []) (Driver.buses g this);
  has_none (unopened (open_ g this 0));
  has_none (unopened (reset g this 0));
  ignore (require_error (Driver.detach g this 0));
  ignore (require_error (Driver.attach g this 0))

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
   domain is about to call Driver, then samples a window in which a driver that is
   not held back runs inside it. *)
let test_one_at_a_time (_, other) =
  needs_flock ();
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
  let start h _ =
    Atomic.set inside true;
    let d =
      Domain.spawn (fun () ->
          Atomic.set about true;
          other g m 1 driver)
    in
    equal ~msg:"the other domain reached Driver" bool true
      (poll (fun () -> Atomic.get about));
    sample ();
    Atomic.set inside false;
    Result.map (fun _ -> d) (kept h)
  in
  let d = require_ok (Driver.open_ g m 0 start) in
  require_ok (Domain.join d);
  equal ~msg:"the other driver ran" bool true (Atomic.get ran);
  equal ~msg:"the other driver ran inside GPU 0's" bool false
    (Atomic.get overlapped)

let others = [ ("an open", open_); ("a reset", reset) ]

(* While GPU 0's driver starts, the main domain stops GPU 1. The driver waits
   for it, so a stop held back by the start would never return: the driver gives
   up after the hang guard and the test fails. *)
let test_stop_waits answer =
  needs_flock ();
  let g, m, _ = three () in
  let h1 = stopping g m 1 answer in
  let inside = Atomic.make false and back = Atomic.make false in
  let start h _ =
    Atomic.set inside true;
    Result.map (fun _ -> poll (fun () -> Atomic.get back)) (kept h)
  in
  let d = Domain.spawn (fun () -> Driver.open_ g m 0 start) in
  equal ~msg:"GPU 0's driver started" bool true
    (poll (fun () -> Atomic.get inside));
  stop h1;
  Atomic.set back true;
  equal ~msg:"stopped while GPU 0's driver started" bool true
    (require_ok (Domain.join d))

(* A model

   The reference knows which GPUs are free, held or lost, and counts the
   vendor's resets and stops. *)

type state = Free | Held | Lost
type answer = [ `Clean | `Lost | `Unknown ]

type vendor_ref = {
  states : state array;
  mutable resets : int;
  mutable stops : int;
}

type hold_ref = {
  v : vendor_ref;
  i : int;
  answer : answer;
  mutable stopped : [ `Stopped | `Unknown ] option;
}

type vendor_sys = {
  g : Driver.t;
  m : Machine.t;
  resets : int Atomic.t;
  stops : int Atomic.t;
  holds : Driver.hold list Atomic.t;  (** Every hold an open made. *)
  configs : Unix.file_descr array;
      (** Each GPU's configuration file, open, so that the check after every
          call makes no open. *)
}

(* [keep s h] records [h], which the vendor's release stops. *)
let rec keep s h =
  let hs = Atomic.get s.holds in
  if not (Atomic.compare_and_set s.holds hs (h :: hs)) then keep s h

(* An open refused without starting the driver, and one whose driver failed. *)
exception Refused
exception Driver_failed

(* How the driver's start ends: before or after it gave the hold its stop. *)
type start = Starts | Fails | Raises | Fails_started | Raises_started

let pp_start ppf s =
  Format.pp_print_string ppf
    (match s with
    | Starts -> "starts"
    | Fails -> "fails"
    | Raises -> "raises"
    | Fails_started -> "fails-started"
    | Raises_started -> "raises-started")

let starts =
  Gen.of_list ~pp:pp_start
    [ Starts; Fails; Raises; Fails_started; Raises_started ]

let answer_gen = Gen.of_list ~pp:pp_answer answers
let indices l = Gen.of_list ~pp:Format.pp_print_int l

let two_gpus =
  [ Tree.gpu "0000:03:00.0"; audio "0000:04:00.1"; Tree.gpu "0000:43:00.0" ]

let two_buses = [| "0000:03:00.0"; "0000:43:00.0" |]

let held_buses v =
  List.filteri (fun i _ -> v.states.(i) = Held) (Array.to_list two_buses)

(* A program's GPUs are stopped when it ends, so that no take outlives it. *)
let vendor_t =
  abstract "v"
    ~release:(fun s ->
      List.iter stop (Atomic.get s.holds);
      Array.iter Unix.close s.configs)
    ~invariant:(fun v s ->
      let taken =
        List.filteri
          (fun i _ -> Tree.flocked_fd s.configs.(i))
          (Array.to_list two_buses)
      in
      equal ~msg:"functions taken" (list string) (held_buses v) taken)

let hold_t = abstract "h"
let vendor_ref () = { states = Array.make 2 Free; resets = 0; stops = 0 }

let vendor_sys () =
  let root = Tree.make two_gpus in
  let resets = Atomic.make 0 and stops = Atomic.make 0 in
  let reset _ =
    Atomic.incr resets;
    Ok ()
  in
  let config bus =
    Unix.openfile
      (Filename.concat root ("sys/bus/pci/devices/" ^ bus ^ "/config"))
      [ O_RDONLY; O_CLOEXEC ] 0
  in
  {
    g = gpus ~reset ();
    m = Machine.at root;
    resets;
    stops;
    holds = Atomic.make [];
    configs = Array.map config two_buses;
  }

let check_index i = if i < 0 then invalid_arg "a negative GPU number"

(* A lost GPU is renewed by the open that finds it free. *)
let renew_ref v i =
  if v.states.(i) = Lost then begin
    cover "a lost GPU is renewed" true;
    v.resets <- v.resets + 1;
    v.states.(i) <- Free
  end

let open_ref v start answer i =
  check_index i;
  if i >= 2 || v.states.(i) = Held then raise Refused;
  renew_ref v i;
  let started () =
    v.stops <- v.stops + 1;
    v.states.(i) <- Lost
  in
  match start with
  | Starts ->
      v.states.(i) <- Held;
      { v; i; answer; stopped = None }
  | Fails -> raise Driver_failed
  | Raises -> invalid_arg err_driver_bug
  | Fails_started ->
      started ();
      raise Driver_failed
  | Raises_started ->
      started ();
      invalid_arg err_driver_bug

let open_sys s start answer i =
  let begun = ref false in
  let driver h _ =
    begun := true;
    let set () =
      Driver.set_stop h (fun () ->
          Atomic.incr s.stops;
          answer)
    in
    match start with
    | Starts ->
        set ();
        Ok h
    | Fails -> Error "the GPU did not start"
    | Raises -> invalid_arg err_driver_bug
    | Fails_started ->
        set ();
        Error "the GPU did not start"
    | Raises_started ->
        set ();
        invalid_arg err_driver_bug
  in
  match Driver.open_ s.g s.m i driver with
  | Ok h ->
      keep s h;
      h
  | Error _ -> if !begun then raise Driver_failed else raise Refused

let try_open_ref v i =
  check_index i;
  let free = i < 2 && v.states.(i) <> Held in
  if free then begin
    renew_ref v i;
    v.states.(i) <- Held
  end;
  free

let try_open_sys s i =
  match Driver.open_ s.g s.m i (fun h _ -> kept h) with
  | Ok h ->
      keep s h;
      true
  | Error _ -> false

let stop_ref h =
  match h.stopped with
  | Some s ->
      cover "a hold stopped again" true;
      s
  | None ->
      let v = h.v in
      v.stops <- v.stops + 1;
      v.states.(h.i) <- (if h.answer = `Clean then Free else Lost);
      let s = if h.answer = `Unknown then `Unknown else `Stopped in
      h.stopped <- Some s;
      s

let reset_ref v i =
  check_index i;
  let free = i < 2 && v.states.(i) <> Held in
  if free then begin
    cover "a lost GPU is reset" (v.states.(i) = Lost);
    v.resets <- v.resets + 1;
    v.states.(i) <- Free
  end;
  free

let reset_sys s i = Result.is_ok (Driver.reset s.g s.m i)

(* Two domains contend for GPU 0 alone, so that they meet. *)
let commands index =
  [
    command "vendor" (Gen.unit @-> makes vendor_t) vendor_ref vendor_sys;
    command "open_"
      (vendor_t ^-> starts @-> answer_gen @-> index @-> makes hold_t)
      open_ref open_sys;
    command "open_, kept"
      (vendor_t ^-> index @-> returns bool)
      try_open_ref try_open_sys;
    command "stop" (hold_t ^-> returns stopped) stop_ref Driver.stop;
    command "reset" (vendor_t ^-> index @-> returns bool) reset_ref reset_sys;
    command "resets"
      (vendor_t ^-> returns int)
      (fun v -> v.resets)
      (fun s -> Atomic.get s.resets);
    command "stops"
      (vendor_t ^-> returns int)
      (fun v -> v.stops)
      (fun s -> Atomic.get s.stops);
  ]

(* [flocking name t] is [t] where takes lock a function's file. *)
let flocking name t =
  if flocks then t
  else
    test name (fun () ->
        skip ~reason:"locking a function's file needs flock" ())

let model_name = "opens, stops and resets behave as the model"

let two_domains_name =
  "from two domains, as some order of the calls: a stop calls the vendor's \
   once and answers the same twice"

let serialized =
  group ~timeout:patience "opens and changes"
    [
      (* Each vendor a program makes is a tree of its own, which the file
         system clones, so a program's cost grows with its length twice: in
         its vendors and in the checks of each after every call. Programs of
         20 calls reach every cover in all but about 1 run in 10^7. *)
      flocking model_name
        (stateful model_name ~count:300 ~steps:20
           (commands (indices [ 0; 1; 2; -1 ])));
      (* 60 programs of 50 runs each, each run a new tree, with a limit of three
         patiences: a run hands off between domains, which waits for a time
         slice when the processors are busy. On the Mac the hand-offs, in
         windtrap's workers, take most of the run and pass the limit, so it
         runs on Linux. *)
      (if on_linux then Fun.id
       else fun _ ->
         test two_domains_name (fun () ->
             skip ~reason:"windtrap's hand-offs between domains pass its bound"
               ()))
      @@ flocking two_domains_name
      @@ stateful two_domains_name ~timeout:(3. *. patience) ~domains:2
           ~count:60
           (commands (indices [ 0 ]));
      cases "opens and resets run their drivers one at a time (sampled)"
        ~name:fst others test_one_at_a_time;
      cases "a stop waits for no driver's start"
        ~name:(Format.asprintf "%a" pp_answer)
        answers test_stop_waits;
    ]

(* Changes on a machine's files

   A machine in a fixture tree keeps what a change writes as plain files and
   acts on none of it: an unbound driver stays linked, a removed function stays
   listed. Each case states what a change writes there and how it answers for
   the state that remains. Changes lock the function's file, which needs
   Linux. *)

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

(* A function of another device on vfio-pci, which makes the tree's
   vfio-pci driver. *)
let vfio_elsewhere = Tree.gpu ~driver:"vfio-pci" ~group:"13" "0000:43:00.0"

let changes =
  let gpu = Tree.gpu gpu_bus in
  [
    ( "detach leaves a GPU a process can take as it is, but for kernel drivers",
      [ gpu ],
      `Detach,
      `Done,
      [
        (devices gpu_bus "enable", "1");
        (devices gpu_bus "remove", "");
        (devices gpu_bus "driver_override", "none");
      ] );
    ( "detach enables a disabled GPU",
      [ Tree.gpu ~enabled:false gpu_bus ],
      `Detach,
      `Done,
      [ (devices gpu_bus "enable", "1") ] );
    ( "detach unbinds the kernel driver, refused while it stays bound",
      [ Tree.gpu ~driver:"amdgpu" gpu_bus ],
      `Detach,
      `Part_way "the driver amdgpu stays bound to 0000:03:00.0",
      [
        ("drivers/amdgpu/unbind", gpu_bus);
        (devices gpu_bus "driver_override", "none");
      ] );
    ( "detach removes the other functions of the GPU's device, refused while \
       they stay",
      [ gpu; audio "0000:03:00.1" ],
      `Detach,
      `Part_way "0000:03:00.0 still shares its device with 0000:03:00.1",
      [ (devices "0000:03:00.1" "remove", "1") ] );
    ( "detach leaves a GPU bound to vfio-pci behind an IOMMU",
      [ Tree.gpu ~driver:"vfio-pci" ~group:"12" gpu_bus ],
      `Detach,
      `Done,
      [
        ("drivers/vfio-pci/unbind", "");
        (devices gpu_bus "driver_override", "(null)");
      ] );
    ( "detach refuses a GPU whose addresses an IOMMU translates without \
       vfio-pci, writing nothing",
      [ Tree.gpu ~group:"12" gpu_bus ],
      `Detach,
      `Refused "through vfio-pci, which is not loaded",
      [ (devices gpu_bus "driver_override", "(null)") ] );
    ( "detach refuses a GPU bound to its kernel driver behind a translating \
       IOMMU without vfio-pci, writing nothing",
      [ Tree.gpu ~driver:"amdgpu" ~group:"12" gpu_bus ],
      `Detach,
      `Refused "through vfio-pci, which is not loaded",
      [
        ("drivers/amdgpu/unbind", "");
        (devices gpu_bus "driver_override", "(null)");
      ] );
    ( "detach refuses a GPU whose IOMMU group holds another device's function \
       on another driver, writing nothing",
      [
        Tree.gpu ~group:"12" gpu_bus;
        Tree.gpu ~driver:"amdgpu" ~group:"12" "0000:04:00.0";
        vfio_elsewhere;
      ],
      `Detach,
      `Refused
        "IOMMU group 12 of 0000:03:00.0 also holds 0000:04:00.0 (amdgpu)",
      [ (devices gpu_bus "driver_override", "(null)") ] );
    ( "detach keeps the GPU and its other functions for vfio-pci behind a \
       translating IOMMU, and unbinds their drivers, the other functions' \
       first, refused while one stays bound",
      [
        Tree.gpu ~driver:"amdgpu" ~group:"12" gpu_bus;
        { (audio "0000:03:00.1") with group = Some "12" };
        vfio_elsewhere;
      ],
      `Detach,
      `Part_way "the driver snd_hda_intel stays bound to 0000:03:00.1",
      [
        ("drivers/snd_hda_intel/unbind", "0000:03:00.1");
        ("drivers/amdgpu/unbind", "");
        (devices gpu_bus "driver_override", "vfio-pci");
        (devices "0000:03:00.1" "driver_override", "vfio-pci");
        (devices "0000:03:00.1" "remove", "");
      ] );
    ( "detach binds an unbound GPU and its other functions to vfio-pci behind \
       a translating IOMMU, refused while vfio-pci does not take them",
      [
        Tree.gpu ~group:"12" gpu_bus;
        { (Tree.gpu ~group:"12" "0000:03:00.1") with class_ = 0x040300 };
        vfio_elsewhere;
      ],
      `Detach,
      `Part_way "vfio-pci did not take 0000:03:00.0 at its probe",
      [
        (devices gpu_bus "driver_override", "vfio-pci");
        (devices "0000:03:00.1" "driver_override", "vfio-pci");
        ("drivers_probe", gpu_bus);
      ] );
    ( "detach refuses a GPU under a locked-down kernel, writing nothing",
      [ Tree.gpu ~driver:"amdgpu" gpu_bus ],
      `Detach_locked_down,
      `Refused "the kernel is locked down",
      [
        ("drivers/amdgpu/unbind", "");
        (devices gpu_bus "driver_override", "(null)");
      ] );
    ( "attach probes the drivers for an unbound GPU, refused when none takes it",
      [ gpu ],
      `Attach,
      `Refused "no kernel driver took 0000:03:00.0",
      [
        (devices gpu_bus "enable", "0");
        (devices gpu_bus "driver_override", "");
        ("rescan", "1");
        ("drivers_probe", gpu_bus);
      ] );
    ( "attach leaves a GPU bound to its kernel driver",
      [ Tree.gpu ~driver:"amdgpu" gpu_bus ],
      `Attach,
      `Done,
      [
        ("rescan", "");
        ("drivers_probe", "");
        (devices gpu_bus "driver_override", "(null)");
      ] );
    ( "attach unbinds vfio-pci from the other functions of the GPU's device, \
       refused while it stays bound",
      [
        Tree.gpu ~driver:"amdgpu" gpu_bus;
        { (audio "0000:03:00.1") with driver = Some "vfio-pci" };
      ],
      `Attach,
      `Refused "the driver vfio-pci stays bound to 0000:03:00.1",
      [
        ("drivers/vfio-pci/unbind", "0000:03:00.1");
        (devices "0000:03:00.1" "driver_override", "vfio-pci");
        ("drivers_probe", "");
      ] );
    ( "attach leaves a GPU on vfio-pci whose function cannot be taken as it was",
      [ Tree.gpu ~driver:"vfio-pci" gpu_bus ],
      `Attach,
      `Refused "0000:03:00.0",
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
  let r =
    match change with
    | `Detach | `Detach_locked_down -> Driver.detach (gpus ()) m 0
    | `Attach ->
        Result.map (fun () -> Driver.Detached) (Driver.attach (gpus ()) m 0)
  in
  (match refusal with
  | `Done -> equal progress Detached (require_ok r)
  | `Part_way sub -> contains ~sub (part_way r)
  | `Refused sub -> contains ~sub (require_error r));
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

(* The command register as the fixture's configuration file holds it. *)
let command_in root =
  let config =
    Filename.concat root ("sys/bus/pci/" ^ devices gpu_bus "config")
  in
  String.get_uint16_le
    (In_channel.with_open_bin config In_channel.input_all)
    command

(* A kernel driver's unbind may leave the GPU mastering the bus: Linux clears
   the bit only once the function's enable count reaches zero, and an enable
   through sysfs counts too. Without an IOMMU such a GPU may write any host
   memory, so detach turns it off, and leaves the rest of the register. *)
let test_detach_stops_mastering () =
  needs_flock ();
  let root = Tree.make [ Tree.gpu gpu_bus ] in
  mastering root;
  equal progress Detached
    (require_ok (Driver.detach (gpus ()) (Machine.at root) 0));
  equal hex memory_space (command_in root)

(* A detach that stops part-way may leave the GPU bound to its kernel driver
   with the override set: attach clears it and brings back the functions the
   detach removed, so that no driver stays kept off the GPU. *)
let test_attach_part_way () =
  needs_flock ();
  let root = Tree.make [ Tree.gpu ~driver:"amdgpu" gpu_bus ] in
  let g = gpus () and m = Machine.at root in
  contains ~sub:"stays bound" (part_way (Driver.detach g m 0));
  equal ~msg:"kept off" string "none"
    (pci_file root (devices gpu_bus "driver_override"));
  require_ok ~pp:Format.pp_print_string (Driver.attach g m 0);
  equal ~msg:"the override" string ""
    (pci_file root (devices gpu_bus "driver_override"));
  equal ~msg:"rescan" string "1" (pci_file root "rescan")

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
    (require_error (Driver.attach (gpus ~reset ()) m 0));
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
    (require_error (Driver.attach (gpus ~reset ()) (Machine.at root) 0));
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
  require_ok (Driver.attach (gpus ()) (Machine.at root) 0);
  equal ~msg:"resets" int 0 (Atomic.get vendor_resets - before)

(* A GPU this process lost and attach reset opens again without a reset. *)
let test_attach_lost () =
  needs_flock ();
  let root = Tree.make [ Tree.gpu gpu_bus ] in
  let g = gpus () and m = Machine.at root in
  stop (stopping g m 0 `Lost);
  let before = Atomic.get vendor_resets in
  contains ~sub:"no kernel driver took 0000:03:00.0"
    (require_error (Driver.attach g m 0));
  equal ~msg:"reset by attach" int 1 (Atomic.get vendor_resets - before);
  Tree.add root (sys (devices gpu_bus "enable")) "1\n";
  stop (hold g m 0);
  equal ~msg:"not by the open" int 1 (Atomic.get vendor_resets - before)

let test_change_held () =
  needs_flock ();
  let g = gpus () and m = Machine.at (Tree.make [ Tree.gpu gpu_bus ]) in
  let h = hold g m 0 in
  contains ~msg:"detach" ~sub:"0000:03:00.0 is open in this process"
    (require_error (Driver.detach g m 0));
  contains ~msg:"attach" ~sub:"0000:03:00.0 is open in this process"
    (require_error (Driver.attach g m 0));
  stop h

let test_change_unwritable () =
  needs_flock ();
  if Unix.geteuid () = 0 then
    skip ~reason:"root writes a file whatever its mode" ();
  let root = Tree.make [ Tree.gpu ~enabled:false gpu_bus ] in
  let enable =
    Filename.concat root ("sys/bus/pci/" ^ devices gpu_bus "enable")
  in
  Unix.chmod enable 0o444;
  let why = part_way (Driver.detach (gpus ()) (Machine.at root) 0) in
  contains ~msg:"names the file" ~sub:enable why;
  contains ~msg:"names the privilege" ~sub:"run as root" why

(* Open devices

   Unbinding a kernel driver waits until no process holds one of its devices
   open. The fixture's descriptors are links in its [proc/self/fd], to this
   machine's [/dev/null] and [/dev/zero]; a GPU's devices are its [dev] files,
   holding [/dev/null]'s number or another, and the vendor's nodes. *)

(* The vendor's nodes as NVIDIA's driver gives them: the GPU's minor in [/proc],
   its node [/dev/nvidiaN]. *)
let nvidia_nodes ~root bus =
  match
    In_channel.with_open_bin
      (Filename.concat root (strf "proc/driver/nvidia/gpus/%s/information" bus))
      In_channel.input_all
  with
  | exception Sys_error _ -> []
  | info ->
      List.filter_map
        (fun line ->
          match String.split_on_char ':' line with
          | [ "Device Minor"; n ] -> Some ("dev/nvidia" ^ String.trim n)
          | _ -> None)
        (String.split_on_char '\n' info)

let render = devices gpu_bus "drm/renderD128/dev"

(* The files of the GPU's DRM device that debugfs lists, as [(pid, command)]. *)
let clients root files =
  Tree.add root
    (strf "sys/kernel/debug/dri/%s/clients" gpu_bus)
    (String.concat ""
       (strf "%20s %5s %3s master a %5s %10s\n" "command" "tgid" "dev" "uid"
          "magic"
       :: List.map
            (fun (pid, command) ->
              strf "%20s %5d %3d   %c    %c %5d %10u\n" command pid 128 'n' 'y'
                1000 0)
            files))

let open_devices =
  let null root =
    Tree.add root (sys render) (Tree.device_number "/dev/null");
    clients root []
  in
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
  match (want, Driver.detach g (Machine.at root) 0) with
  | `Detached, r -> equal progress Detached (require_ok r)
  | `Refused file, r ->
      let why = require_error r in
      contains ~sub:"0000:03:00.0 is open in this process" why;
      contains ~msg:"names the device" ~sub:file why
  | `Error sub, r -> contains ~sub (require_error r)

(* The kernel driver lets go

   Detach refuses at once a file a process holds, and waits up to [teardown_ms]
   for those the kernel holds: files of the GPU's DRM device debugfs lists with
   no descriptor or mapping in [/proc], and, of an unbound GPU, the vendor's
   [unreleased]. Another process's descriptors and mappings are links in the
   fixture's [proc/PID/fd] and [proc/PID/map_files]. *)

let override root =
  In_channel.with_open_text
    (Filename.concat root ("sys/bus/pci/" ^ devices gpu_bus "driver_override"))
    In_channel.input_all

let held_tree () =
  let root = Tree.make [ Tree.gpu gpu_bus ] in
  Tree.add root (sys render) (Tree.device_number "/dev/null");
  clients root [];
  root

(* A device another process holds open or mapped is refused at once, with the
   machine untouched, naming the process and the device. *)
let test_held_elsewhere () =
  needs_flock ();
  let g = gpus ~teardown_ms:60_000 () in
  List.iter
    (fun (msg, link) ->
      let root = held_tree () in
      Tree.link root link "/dev/null";
      Tree.link root "proc/4242/fd/8" "/dev/zero";
      let before = override root in
      let why = require_error (Driver.detach g (Machine.at root) 0) in
      contains ~msg ~sub:"process 4242 holds /dev/null" why;
      equal ~msg string before (override root))
    [
      ("an open descriptor", "proc/4242/fd/7");
      ("a mapping", "proc/4242/map_files/7f00-7f01");
    ]

(* A daemon that holds a GPU's device, as nvidia-persistenced holds
   /dev/nvidiaN, is named by its command, to be stopped first. *)
let test_held_by_daemon () =
  needs_flock ();
  let root = held_tree () in
  Tree.link root "proc/4242/fd/7" "/dev/null";
  Tree.add root "proc/4242/comm" "nvidia-persistenced\n";
  let g = gpus ~teardown_ms:60_000 () in
  let why = require_error (Driver.detach g (Machine.at root) 0) in
  contains ~sub:"process 4242 (nvidia-persistenced) holds /dev/null" why;
  contains ~sub:"stop it first" why

(* Detach waits until the kernel lets go of the last file of the GPU's DRM
   device, as a compute runtime's deferred release does, then detaches. *)
let test_let_go_drm () =
  needs_flock ();
  let root = held_tree () in
  clients root [ (4242, "<unknown>") ];
  let closer =
    Domain.spawn (fun () ->
        Unix.sleepf 0.2;
        clients root [])
  in
  let r = Driver.detach (gpus ()) (Machine.at root) 0 in
  Domain.join closer;
  equal progress Detached (require_ok r);
  equal ~msg:"detached" string "none" (String.trim (override root))

(* A file of the GPU's DRM device the kernel keeps past the bound refuses
   detach, naming the process it holds it for, with the machine untouched. *)
let test_kernel_holds () =
  needs_flock ();
  let root = held_tree () in
  clients root [ (4242, "a compositor") ];
  let before = override root in
  let why =
    require_error (Driver.detach (gpus ~teardown_ms:200 ()) (Machine.at root) 0)
  in
  contains ~msg:"names the process" ~sub:"process 4242 (a compositor)" why;
  equal ~msg:"nothing written" string before (override root)

(* A file of the GPU's DRM device this process opened is refused at once: detach
   would wait for itself. *)
let test_own_drm () =
  needs_flock ();
  let root = held_tree () in
  clients root [ (Unix.getpid (), "this test") ];
  let g = gpus ~teardown_ms:60_000 () in
  contains ~sub:"0000:03:00.0 is open in this process"
    (require_error (Driver.detach g (Machine.at root) 0))

(* A GPU with a DRM device whose files debugfs does not list is refused, since
   detach cannot know them, naming the list. *)
let test_no_debugfs () =
  needs_flock ();
  let root = held_tree () in
  let list = strf "sys/kernel/debug/dri/%s/clients" gpu_bus in
  Sys.remove (Filename.concat root list);
  let before = override root in
  contains ~sub:list (require_error (Driver.detach (gpus ()) (Machine.at root) 0));
  equal ~msg:"nothing written" string before (override root)

(* A process's [/proc] entry that cannot be read for a reason other than its end
   is an Error naming it, never a silent "nothing held". *)
let test_unreadable_proc () =
  needs_flock ();
  if Unix.geteuid () = 0 then
    skip ~reason:"root reads a directory whatever its mode" ();
  let root = held_tree () in
  Tree.link root "proc/4242/fd/3" "/dev/zero";
  let fd = Filename.concat root "proc/4242/fd" in
  Unix.chmod fd 0o000;
  let r = Driver.detach (gpus ()) (Machine.at root) 0 in
  Unix.chmod fd 0o755;
  contains ~sub:"proc/4242/fd" (require_error r)

(* A mapped file whose target the process may not read, as without
   CAP_SYS_ADMIN, is an Error naming the entry: [/proc] cannot be read. *)
let test_unreadable_mapping () =
  needs_flock ();
  if Unix.geteuid () = 0 then
    skip ~reason:"root reads a directory whatever its mode" ();
  let root = held_tree () in
  Tree.link root "proc/4242/map_files/7f00-7f01" "/dev/zero";
  let dir = Filename.concat root "proc/4242/map_files" in
  Unix.chmod dir 0o444;
  let r = Driver.detach (gpus ()) (Machine.at root) 0 in
  Unix.chmod dir 0o755;
  let why = require_error r in
  contains ~sub:"proc/4242/map_files/7f00-7f01" why

(* The vendor's [unreleased] is asked of an unbound GPU: detach waits until it
   lets go, and refuses past the bound, changing nothing. *)
let test_unreleased () =
  needs_flock ();
  let asked = Atomic.make 0 in
  let root = held_tree () in
  let twice ~root:_ _ =
    if Atomic.fetch_and_add asked 1 < 2 then Some "its release pending"
    else None
  in
  equal progress Detached
    (require_ok (Driver.detach (gpus ~unreleased:twice ()) (Machine.at root) 0));
  at_least ~msg:"asked until it let go" int ~than:3 (Atomic.get asked);
  let root = held_tree () in
  let original = override root in
  let always ~root:_ _ = Some "its release pending" in
  let why =
    require_error
      (Driver.detach
         (gpus ~unreleased:always ~teardown_ms:200 ())
         (Machine.at root) 0)
  in
  contains ~msg:"the vendor's reason" ~sub:"its release pending" why;
  equal ~msg:"nothing written" string original (override root)

(* An unbound GPU its kernel driver has not let go of, still writing to it, is
   refused an open; it opens once the driver let go. *)
let test_open_unreleased () =
  needs_flock ();
  let root = Tree.make [ Tree.gpu gpu_bus ] in
  let pending = Atomic.make true in
  let unreleased ~root:_ _ =
    if Atomic.get pending then Some "its release pending" else None
  in
  let g = gpus ~unreleased () and m = Machine.at root in
  contains ~sub:"its release pending" (unopened (open_ g m 0));
  Atomic.set pending false;
  stop (hold g m 0)

(* The vendor's [unreleased] is asked under the lock the take holds, so that no
   change comes between the answer and the take. *)
let test_unreleased_locked () =
  needs_flock ();
  let root = Tree.make [ Tree.gpu gpu_bus ] in
  let asked = ref [] in
  let unreleased ~root:_ _ =
    asked := held root gpu_bus :: !asked;
    None
  in
  let g = gpus ~unreleased () and m = Machine.at root in
  stop (hold g m 0);
  equal ~msg:"locked when asked" (list bool) [ true ] !asked

(* An unbound GPU its kernel driver has not let go of is refused a reset and an
   attach, as an open: the driver's release would write to it after the reset.
   The vendor resets nothing. *)
let test_reset_unreleased () =
  needs_flock ();
  let root = Tree.make [ Tree.gpu gpu_bus ] in
  let unreleased ~root:_ _ = Some "its release pending" in
  let g = gpus ~unreleased () and m = Machine.at root in
  let before = Atomic.get vendor_resets in
  contains ~msg:"reset" ~sub:"its release pending"
    (require_error (Driver.reset g m 0));
  contains ~msg:"attach" ~sub:"its release pending"
    (require_error (Driver.attach g m 0));
  equal ~msg:"resets" int 0 (Atomic.get vendor_resets - before)

(* A GPU bound to vfio-pci stays as it is: nothing is waited for. *)
let test_vfio_unwaited () =
  needs_flock ();
  let root = Tree.make [ Tree.gpu ~driver:"vfio-pci" ~group:"7" gpu_bus ] in
  let always ~root:_ _ = Some "its release pending" in
  equal progress Detached
    (require_ok
       (Driver.detach
          (gpus ~unreleased:always ~teardown_ms:60_000 ())
          (Machine.at root) 0))

(* A vendor that keeps a device's other functions, such as a NIC's other
   ports, has detach refuse a device that has any where Linux resets the
   function through its bus, naming each with its driver, and write nothing;
   one alone on its device detaches, and so does one with a reset of its
   own, its other functions left as they were. *)
(* Behind a translating IOMMU, attach gives back what a detach that stopped
   part-way left: a GPU still bound to its driver, whose override and its
   other function's are cleared; and an unbound GPU, which it binds to
   vfio-pci for the reset first. *)
let test_attach_vfio_part_way () =
  needs_flock ();
  let g = gpus () in
  let sibling = "0000:03:00.1" in
  let bound =
    Tree.make
      [
        Tree.gpu ~driver:"amdgpu" ~group:"12" gpu_bus;
        { (audio sibling) with group = Some "12" };
        vfio_elsewhere;
      ]
  in
  contains ~sub:"snd_hda_intel stays bound"
    (part_way (Driver.detach g (Machine.at bound) 0));
  require_ok ~msg:"attach of a GPU still bound"
    (Driver.attach g (Machine.at bound) 0);
  equal ~msg:"the GPU's override" string ""
    (pci_file bound (devices gpu_bus "driver_override"));
  equal ~msg:"the other function's override" string ""
    (pci_file bound (devices sibling "driver_override"));
  let unbound =
    Tree.make
      [
        Tree.gpu ~group:"12" gpu_bus;
        { (Tree.gpu ~group:"12" sibling) with class_ = 0x040300 };
        vfio_elsewhere;
      ]
  in
  contains ~sub:"vfio-pci did not take"
    (part_way (Driver.detach g (Machine.at unbound) 0));
  Tree.add unbound
    (Filename.concat "sys/bus/pci" (devices gpu_bus "driver_override"))
    "";
  let why = require_error (Driver.attach g (Machine.at unbound) 0) in
  contains ~msg:"attach binds the GPU to vfio-pci first"
    ~sub:"vfio-pci did not take 0000:03:00.0" why;
  contains ~msg:"where to find why" ~sub:"the kernel's log (dmesg)" why;
  contains ~msg:"how to proceed" ~sub:"reboot" why;
  equal ~msg:"its override" string "vfio-pci"
    (pci_file unbound (devices gpu_bus "driver_override"));
  equal ~msg:"its probe" string gpu_bus (pci_file unbound "drivers_probe")

let test_keep_siblings () =
  needs_flock ();
  let root = Tree.make [ Tree.gpu gpu_bus; audio "0000:03:00.1" ] in
  let g = gpus ~siblings:Driver.Keep () in
  let why = require_error (Driver.detach g (Machine.at root) 0) in
  contains ~sub:"0000:03:00.1 (bound to snd_hda_intel)" why;
  contains ~sub:"which detach leaves" why;
  contains ~sub:"remove or detach them first" why;
  equal ~msg:"the override" string "(null)"
    (pci_file root (devices gpu_bus "driver_override"));
  equal ~msg:"the other function" string ""
    (pci_file root (devices "0000:03:00.1" "remove"));
  let alone = Tree.make [ Tree.gpu gpu_bus ] in
  equal progress Detached
    (require_ok (Driver.detach g (Machine.at alone) 0));
  let own =
    Tree.make
      [ Tree.gpu ~reset:(Some "flr bus") gpu_bus; audio "0000:03:00.1" ]
  in
  equal progress Detached
    (require_ok ~msg:"a reset of its own" (Driver.detach g (Machine.at own) 0));
  equal ~msg:"the other function, kept" string ""
    (pci_file own (devices "0000:03:00.1" "remove"))

(* A device's other functions under a kernel

   A GPU's kernel driver may read the data of its device's other functions'
   drivers as its own: NVIDIA's reads an audio function's as a sound card. So
   with [Remove], no other function of the GPU's device is on vfio-pci while
   the GPU's kernel driver is bound, during its probe and its unbind too. The
   tree's kernel acts on what a change writes and records, at each bind and
   unbind, the drivers of the device's other functions. *)

let vfio = "vfio-pci"
let audio_bus = "0000:03:00.1"
let own_drivers = [ (gpu_bus, "amdgpu"); (audio_bus, "snd_hda_intel") ]
let on_kernel_driver = function Some d -> d <> vfio | None -> false

(* The binds and unbinds of [events] that break the rule: the GPU's kernel
   driver's while another function is on vfio-pci, and another function's
   bind to vfio-pci while the GPU's kernel driver is bound. *)
let broken events =
  List.filter
    (fun (e : Tree.event) ->
      match e.change with
      | (Bind d | Unbind d) when e.fn = gpu_bus ->
          on_kernel_driver (Some d)
          && List.exists (fun (_, d) -> d = Some vfio) e.others
      | Bind d ->
          d = vfio
          && on_kernel_driver
               (Option.join (List.assoc_opt gpu_bus e.others))
      | Unbind _ -> false)
    events

let pp_event ppf (e : Tree.event) =
  let change, d =
    match e.change with Bind d -> ("bind", d) | Unbind d -> ("unbind", d)
  in
  let pp_other ppf (f, d) =
    Format.fprintf ppf "%s on %s" f (Option.value d ~default:"no driver")
  in
  Format.fprintf ppf "%s %s %s, %a" change d e.fn
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ", ")
       pp_other)
    e.others

let event = Testable.make ~pp:pp_event ~equal:( = )

(* [drivers root] is the drivers the tree [root] binds to the GPU and its
   audio function. *)
let drivers root = (Tree.driver root gpu_bus, Tree.driver root audio_bus)

(* An attach that stopped part-way, or an administrator, may leave the audio
   function on vfio-pci with the GPU unbound. Attach takes it off vfio-pci
   before it probes the GPU's kernel driver, and gives it back to its own
   after. *)
let test_attach_siblings () =
  needs_flock ();
  let root =
    Tree.make
      [
        Tree.gpu ~reset:(Some "flr") gpu_bus;
        { (audio audio_bus) with driver = Some vfio };
      ]
  in
  let k = Tree.kernel ~drivers:own_drivers root in
  require_ok ~pp:Format.pp_print_string
    (Driver.attach (gpus ()) (Tree.machine k) 0);
  equal ~msg:"broken" (list event) [] (broken (Tree.events k));
  equal ~msg:"drivers after"
    (pair (option string) (option string))
    (Some "amdgpu", Some "snd_hda_intel")
    (drivers root)

(* An attach that fails at the GPU's probe, its kernel driver not loaded,
   leaves the audio function unbound, its override naming vfio-pci. Loading
   the module binds the GPU, and the next attach gives the audio function
   back. *)
let test_attach_siblings_again () =
  needs_flock ();
  let root =
    Tree.make
      [
        Tree.gpu ~reset:(Some "flr") gpu_bus;
        { (audio audio_bus) with driver = Some vfio };
      ]
  in
  let k = Tree.kernel ~unloaded:[ "amdgpu" ] ~drivers:own_drivers root in
  let g = gpus () and m = Tree.machine k in
  contains ~sub:"no kernel driver took 0000:03:00.0"
    (require_error (Driver.attach g m 0));
  equal ~msg:"drivers after the failure"
    (pair (option string) (option string))
    (None, None) (drivers root);
  equal ~msg:"the audio function's override" string "vfio-pci"
    (pci_file root (devices audio_bus "driver_override"));
  Tree.load k "amdgpu";
  require_ok ~pp:Format.pp_print_string (Driver.attach g m 0);
  equal ~msg:"broken" (list event) [] (broken (Tree.events k));
  equal ~msg:"drivers after"
    (pair (option string) (option string))
    (Some "amdgpu", Some "snd_hda_intel")
    (drivers root)

let tree_changes =
  group ~timeout:patience "changes on a machine's files"
    [
      cases "each change writes what it says and answers for what remains"
        ~name:(fun (n, _, _, _, _) -> n)
        changes test_change;
      test "a GPU the process holds is refused" test_change_held;
      test "detach keeps a device's other functions when the vendor says so"
        test_keep_siblings;
      test
        "attach takes the GPU's other functions off vfio-pci before it \
         probes the GPU"
        test_attach_siblings;
      test
        "an attach that fails at the GPU's probe leaves its other functions \
         off vfio-pci for the attach after its module loads"
        test_attach_siblings_again;
      test
        "attach gives back a GPU that a detach behind a translating IOMMU left \
         part-way"
        test_attach_vfio_part_way;
      test "detach leaves the GPU's bus mastering off"
        test_detach_stops_mastering;
      test "attach clears what a detach that stopped part-way left"
        test_attach_part_way;
      test
        "attach resets the GPU, its bus mastering off, before the bus is \
         rescanned"
        test_attach_resets;
      test "attach leaves the GPU detached when its reset fails"
        test_attach_refused;
      test "attach leaves a GPU bound to its kernel driver unreset"
        test_attach_bound;
      test "a GPU lost opens again without a reset once attach reset it"
        test_attach_lost;
      cases "detach refuses a GPU whose device this process holds open"
        ~name:(fun (n, _, _, _) -> n)
        open_devices test_open_device;
      test
        "detach refuses at once a GPU another process holds open or mapped, \
         changing nothing"
        test_held_elsewhere;
      test "detach names the command of a daemon that holds the GPU's device"
        test_held_by_daemon;
      test "detach waits for the last file of the GPU's DRM device to go"
        test_let_go_drm;
      test
        "detach refuses a file of the DRM device the kernel keeps past its \
         bound, changing nothing"
        test_kernel_holds;
      test "detach refuses at once a file of the DRM device this process opened"
        test_own_drm;
      test "detach refuses a DRM device whose files debugfs does not list"
        test_no_debugfs;
      test "a process's /proc entry that cannot be read is an Error"
        test_unreadable_proc;
      test "a mapping whose target cannot be read is an Error naming it"
        test_unreadable_mapping;
      test "detach waits for the vendor's driver to let go of an unbound GPU"
        test_unreleased;
      test "detach waits for nothing on a GPU bound to vfio-pci"
        test_vfio_unwaited;
      test
        "an unbound GPU its kernel driver has not let go of is refused an open"
        test_open_unreleased;
      test "the vendor is asked whether the GPU is released under the take's lock"
        test_unreleased_locked;
      test
        "an unbound GPU its kernel driver has not let go of is refused a reset \
         and an attach"
        test_reset_unreleased;
      test "a file the process may not write is refused, naming it"
        test_change_unwritable;
    ]

(* Holds on a machine's files

   A vendor whose reset counts its calls opens the GPUs of a fixture tree, whose
   functions it takes physically, which needs Linux. *)

type tree = {
  g : Driver.t;
  m : Machine.t;
  root : string;
  resets : int Atomic.t;
  answer : (unit -> (unit, string) result) ref;
      (** What the vendor's reset does once counted. *)
}

let tree ?(gpus_on = [ gpu_bus ]) () =
  needs_flock ();
  let root = Tree.make (List.map Tree.gpu gpus_on) in
  let resets = Atomic.make 0 and answer = ref ok in
  let reset _ =
    Atomic.incr resets;
    !answer ()
  in
  { g = gpus ~reset (); m = Machine.at root; root; resets; answer }

let resets_of t = Atomic.get t.resets

(* [opened t f] opens GPU 0 of [t] with [f], and is [f]'s result with the resets
   the vendor ran before [f] started, if it did. *)
let opened t f =
  let before = ref None in
  let r =
    Driver.open_ t.g t.m 0 (fun h fn ->
        before := Some (resets_of t);
        f h fn)
  in
  (r, !before)

let reset_before = option int

(* Each request about GPU 1, which the process holds, is an Error that starts
   with its name. *)
let test_named_errors (_, request) =
  let t = tree ~gpus_on:[ gpu_bus; "0000:43:00.0" ] () in
  let h = hold t.g t.m 1 in
  starts_with ~affix:(Driver.name t.g 1 ^ ":") (require_error (request t));
  stop h

let named_requests =
  [
    ("open_", fun t -> open_ t.g t.m 1 ok);
    ("reset", fun t -> Driver.reset t.g t.m 1);
    ("detach", fun t -> Result.map ignore (Driver.detach t.g t.m 1));
    ("attach", fun t -> Driver.attach t.g t.m 1);
  ]

let test_set_stop_twice () =
  let t = tree () in
  let h =
    require_ok
      (Driver.open_ t.g t.m 0 (fun h _ ->
           Driver.set_stop h (fun () -> `Clean);
           raises_match (Exn.invalid_arg ~substring:"") (fun () ->
               Driver.set_stop h (fun () -> `Lost));
           Ok h))
  in
  stop h;
  raises_match ~msg:"a stop once stopped" (Exn.invalid_arg ~substring:"")
    (fun () -> Driver.set_stop h (fun () -> `Clean));
  let r, before = opened t (fun h _ -> kept h) in
  stop (require_ok r);
  equal ~msg:"the first stop kept: resets before the next start" reset_before
    (Some 0) before

(* The vendor's stop runs once; a second stop answers what the first did. *)
let test_stop_once answer =
  let t = tree () in
  let calls = Atomic.make 0 in
  let h, fn =
    require_ok
      (Driver.open_ t.g t.m 0 (fun h fn ->
           Driver.set_stop h (fun () ->
               Atomic.incr calls;
               answer);
           Ok (h, fn)))
  in
  let want = if answer = `Unknown then `Unknown else `Stopped in
  equal ~msg:"the first stop" stopped want (Driver.stop h);
  equal ~msg:"the second" stopped want (Driver.stop h);
  equal ~msg:"calls of the vendor's stop" int 1 (Atomic.get calls);
  equal ~msg:"the function released" bool true (Function.released fn)

(* A stop of a second GPU from a callback at any allocation of the first's
   stop. Inside the lock of the GPUs' holds, which the first's give-back
   holds, the second's give-back is refused, and its next stop gives the GPU
   back without running its vendor's stop again; outside every lock it stops
   the GPU. Each vendor's stop runs once, and both functions are released. *)
let test_stop_in_stop () =
  let rec from n =
    let t = tree ~gpus_on:(Array.to_list two_buses) () in
    let stops = [| 0; 0 |] in
    let open_ i =
      require_ok
        (Driver.open_ t.g t.m i (fun h fn ->
             Driver.set_stop h (fun () ->
                 stops.(i) <- stops.(i) + 1;
                 `Clean);
             Ok (h, fn)))
    in
    let h0, f0 = open_ 0 in
    let h1, f1 = open_ 1 in
    let call () =
      match Driver.stop h1 with
      | _ -> ()
      | exception Invalid_argument why
        when String.starts_with ~prefix:"Rig_pci: a lock taken again" why ->
          ()
    in
    let reached =
      Rig_support.interrupted ~landing:(Call call) n (fun () -> stop h0)
    in
    stop h1;
    let at what = strf "%s, called at %d" what n in
    equal ~msg:(at "the vendor's stops") (array int) [| 1; 1 |] stops;
    equal ~msg:(at "the functions released") (pair bool bool) (true, true)
      (Function.released f0, Function.released f1);
    if reached then from (n + 1)
  in
  from 1

type failure = Answers_error | Raises_exit

let failures = [ ("answers Error", Answers_error); ("raises", Raises_exit) ]

let fail_start how =
  match how with
  | Answers_error -> Error "the GPU did not start"
  | Raises_exit -> raise Exit

let run_failing how f =
  match how with
  | Answers_error -> ignore (require_error (f ()))
  | Raises_exit -> raises Exit (fun () -> f ())

(* A start that fails before the hold has a stop wrote nothing to the GPU: it is
   released as found. *)
let test_fails_before (_, how) =
  let t = tree () in
  let fn = ref None in
  run_failing how (fun () ->
      fst
        (opened t (fun _ f ->
             fn := Some f;
             fail_start how)));
  equal ~msg:"its function released" bool true
    (Function.released (require_some !fn));
  let r, before = opened t (fun h _ -> kept h) in
  stop (require_ok r);
  equal ~msg:"resets before the next open's start" reset_before (Some 0) before

(* A start that fails once the hold has its stop may have written to the GPU:
   the open stops it, and the GPU is lost, renewed before the next start. *)
let test_fails_after (_, how) =
  let t = tree () in
  let calls = Atomic.make 0 in
  run_failing how (fun () ->
      fst
        (opened t (fun h _ ->
             Driver.set_stop h (fun () ->
                 Atomic.incr calls;
                 `Clean);
             fail_start how)));
  equal ~msg:"the vendor's stop" int 1 (Atomic.get calls);
  let r, before = opened t (fun h _ -> kept h) in
  stop (require_ok r);
  equal ~msg:"resets before the next open's start" reset_before (Some 1) before

let test_stop_answer answer =
  let t = tree () in
  let r, _ =
    opened t (fun h _ ->
        Driver.set_stop h (fun () -> answer);
        Ok h)
  in
  stop (require_ok r);
  let r, before = opened t (fun h _ -> kept h) in
  stop (require_ok r);
  equal ~msg:"resets before the next open's start" reset_before
    (Some (if answer = `Clean then 0 else 1))
    before

(* A renew whose reset fails loses the GPU, whatever the driver answers after
   it. *)
let test_renew_error (_, answers) =
  let t = tree () in
  (t.answer := fun () -> Error "the GPU is stuck");
  let r, _ =
    opened t (fun h _ ->
        contains ~msg:"the reset's reason" ~sub:"the GPU is stuck"
          (require_error (Driver.renew h));
        if answers then Result.map Option.some (kept h)
        else Error "the GPU did not start")
  in
  if answers then stop (require_some (require_ok r))
  else ignore (require_error r);
  t.answer := ok;
  let r, before = opened t (fun h _ -> kept h) in
  stop (require_ok r);
  equal ~msg:"resets before the next open's start" reset_before (Some 2) before

(* A process that died leaving GPU 0 reaching memory left a memory file under
   the machine's dev/hugepages that no process locks, with a marker beside it
   that names the GPU among the functions reaching it. *)
let dead_file t =
  let name = "dev/hugepages/rig-pci-999999-1" in
  Tree.add t.root name "";
  Tree.add t.root (name ^ ".reach-" ^ gpu_bus) "";
  Filename.concat t.root name

let test_dead_left () =
  let t = tree () in
  let file = dead_file t in
  let r, before = opened t (fun h _ -> kept h) in
  stop (require_ok r);
  equal ~msg:"resets before the start" reset_before (Some 1) before;
  equal ~msg:"the memory given back" bool false (Sys.file_exists file);
  let r, before = opened t (fun h _ -> kept h) in
  stop (require_ok r);
  equal ~msg:"resets before the next open's start" reset_before (Some 1) before

(* A renewal that fails or raises leaves the GPU lost, the dead process's memory
   kept. *)
let test_dead_left_stuck () =
  let t = tree () in
  let file = dead_file t in
  (t.answer := fun () -> Error "the GPU is stuck");
  let r, before = opened t (fun h _ -> kept h) in
  let why = require_error r in
  starts_with ~msg:"names the GPU" ~affix:(Driver.name t.g 0) why;
  contains ~msg:"the reset's reason" ~sub:"the GPU is stuck" why;
  equal ~msg:"the driver" reset_before None before;
  equal ~msg:"the memory kept" bool true (Sys.file_exists file);
  t.answer := ok;
  let r, before = opened t (fun h _ -> kept h) in
  stop (require_ok r);
  equal ~msg:"resets before the next open's start" reset_before (Some 2) before

let test_dead_left_raising () =
  let t = tree () in
  ignore (dead_file t);
  (t.answer := fun () -> raise Exit);
  raises Exit (fun () -> open_ t.g t.m 0 ok);
  t.answer := ok;
  let r, before = opened t (fun h _ -> kept h) in
  stop (require_ok r);
  equal ~msg:"resets before the next open's start" reset_before (Some 2) before

(* A start that answers Ok without a stop is a driver bug: the open releases the
   GPU and raises. *)
let test_no_stop () =
  let t = tree () in
  let fn = ref None in
  raises_match (Exn.invalid_arg ~substring:"") (fun () ->
      opened t (fun _ f ->
          fn := Some f;
          Ok ()));
  equal ~msg:"released" bool true (Function.released (require_some !fn));
  let r, before = opened t (fun h _ -> kept h) in
  stop (require_ok r);
  equal ~msg:"resets before the next open's start" reset_before (Some 0) before

(* An exception from the vendor's reset in a renew passes through the open, and
   the GPU is lost. *)
let test_renew_raises () =
  let t = tree () in
  (t.answer := fun () -> raise Exit);
  raises Exit (fun () ->
      opened t (fun h _ ->
          ignore (Driver.renew h : (unit, string) result);
          kept h));
  t.answer := ok;
  let r, before = opened t (fun h _ -> kept h) in
  stop (require_ok r);
  equal ~msg:"resets before the next open's start" reset_before (Some 2) before

(* While the vendor's stop of a GPU it lost runs, the GPU is still held: another
   domain's open is refused, and the open after the stop renews it. *)
let test_open_while_stopping () =
  let t = tree () in
  let during = ref None in
  let r, _ =
    opened t (fun h _ ->
        Driver.set_stop h (fun () ->
            let other =
              Domain.spawn (fun () -> fst (opened t (fun h _ -> kept h)))
            in
            during := Some (Domain.join other);
            `Lost);
        Ok h)
  in
  stop (require_ok r);
  contains ~msg:"the open during the stop" ~sub:"is open in this process"
    (require_error (require_some !during));
  let r, before = opened t (fun h _ -> kept h) in
  stop (require_ok r);
  equal ~msg:"resets before the next open's start" reset_before (Some 1) before

let tree_holds =
  group ~timeout:patience "holds on a machine's files"
    [
      cases "each Error about a GPU starts with its name" ~name:fst
        named_requests test_named_errors;
      test "a hold's stop is set once" test_set_stop_twice;
      test
        "an open from another domain while a lost GPU's stop runs is refused, \
         and the open after renews it"
        test_open_while_stopping;
      test "a start that answers Ok without a stop raises, the GPU released"
        test_no_stop;
      test "a renew whose reset raises passes it through, the GPU lost"
        test_renew_raises;
      cases "stop calls the vendor's stop once and answers the same twice"
        ~name:(Format.asprintf "%a" pp_answer)
        answers test_stop_once;
      test
        "a stop from a callback inside another's give-back gives its GPU \
         back at its next stop, stopping it once"
        test_stop_in_stop;
      cases
        "an open whose start fails before set_stop releases the GPU, unrenewed"
        ~name:fst failures test_fails_before;
      cases
        "an open whose start fails after set_stop loses the GPU, renewed \
         before the next start"
        ~name:fst failures test_fails_after;
      cases "a stop that answers `Lost or `Unknown has the next open renew it"
        ~name:(Format.asprintf "%a" pp_answer)
        answers test_stop_answer;
      cases "a renew that fails loses the GPU whatever the start answers"
        ~name:fst
        [ ("the start answers Ok", true); ("the start answers Error", false) ]
        test_renew_error;
      test
        "a GPU a dead process left reaching memory is renewed before the \
         start, and once"
        test_dead_left;
      test
        "a GPU a dead process left whose renewal fails is lost, its memory kept"
        test_dead_left_stuck;
      test
        "a GPU a dead process left whose reset raises passes it through, lost"
        test_dead_left_raising;
    ]

(* Exit

   A process that exits holding GPUs stops them. The suite's executable, run
   with [exit_holding], is such a process; the test reads what it printed. *)

let exit_holding = "--exit-holding"

(* Opens the three GPUs of the tree [root], each starting with its bus mastering
   on and a stop that prints its bus and whether it still masters the bus: GPU
   0's stop raises instead, and GPU 2 is stopped before the exit. It forks a
   child that exits, then exits. *)
let exit_holding_gpus root =
  let g = gpus () and m = Machine.at root in
  let open_ i stop =
    let start h fn =
      Driver.set_stop h (stop fn);
      Function.set_bus_master fn true;
      Ok h
    in
    Result.get_ok (Driver.open_ g m i start)
  in
  let prints fn () =
    let on = Function.config16 fn command land bus_master <> 0 in
    print_endline
      (strf "stopped %s %s" (Function.bus fn)
         (if on then "mastering" else "not mastering"));
    `Clean
  in
  ignore (open_ 0 (fun _ () -> failwith "a driver bug"));
  ignore (open_ 1 prints);
  stop (open_ 2 prints);
  (match Unix.fork () with
  | 0 -> exit 0
  | child -> ignore (Unix.waitpid [] child));
  exit 0

(* Opens GPU 1 of the tree [root] with a stop that prints, as
   [exit_holding_gpus] does, then exits from the start of GPU 2's open, which
   holds the vendor's lock and no lock a stop takes. *)
let exit_inside = "--exit-inside"

let exit_inside_lock root =
  let g = gpus () and m = Machine.at root in
  let prints fn () =
    print_endline ("stopped " ^ Function.bus fn);
    `Clean
  in
  let start h fn =
    Driver.set_stop h (prints fn);
    Ok h
  in
  ignore (Result.get_ok (Driver.open_ g m 1 start));
  ignore (Driver.open_ g m 2 (fun _ _ -> exit 0))

(* The lines the suite's executable run with [mode] on the tree [root] printed
   on standard output, and what it printed on standard error. *)
let exiting ?(mode = exit_holding) root =
  let exe = Sys.executable_name in
  let out_r, out_w = Unix.pipe ~cloexec:true () in
  let err_r, err_w = Unix.pipe ~cloexec:true () in
  let pid =
    Unix.create_process exe [| exe; mode; root |] Unix.stdin out_w err_w
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
  needs_flock ();
  let root = Tree.make (List.map Tree.gpu gpu_buses) in
  let out, err = exiting root in
  equal ~msg:"stopped, once each, still mastering" (list string)
    [ "stopped 0000:c3:00.0 mastering"; "stopped 0000:43:00.0 mastering" ]
    out;
  contains ~msg:"the stop that raised"
    ~sub:"stopping 0000:03:00.0 at exit: Failure(\"a driver bug\")" err

(* An exit inside a lock that no stop takes stops every GPU. *)
let test_exit_inside () =
  needs_flock ();
  let root = Tree.make (List.map Tree.gpu gpu_buses) in
  let out, err = exiting ~mode:exit_inside root in
  equal ~msg:"stopped" (list string) [ "stopped 0000:43:00.0" ] out;
  let at_exit l =
    let k = String.length " at exit: " and n = String.length l in
    let rec go i =
      i + k <= n && (String.sub l i k = " at exit: " || go (i + 1))
    in
    go 0
  in
  equal ~msg:"stops that raised at exit" (list string) []
    (List.filter at_exit (String.split_on_char '\n' err))

let exits =
  group ~timeout:patience "exit"
    [
      test
        "a process that exits stops each GPU it holds once, before its bus \
         mastering is turned off, past a stop that raises, and its forked \
         child stops none"
        test_exit;
      test
        "a process that exits inside a lock that no stop takes stops every \
         GPU"
        test_exit_inside;
    ]

let () =
  match Sys.argv with
  | [| _; arg; root |] when arg = exit_holding -> exit_holding_gpus root
  | [| _; arg; root |] when arg = exit_inside -> exit_inside_lock root
  | _ ->
      exit
      @@ run "rig_pci.driver"
           [
             numbering;
             opening;
             resets;
             tree_changes;
             tree_holds;
             exits;
             this_machine;
             serialized;
           ]
