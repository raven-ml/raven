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
    ?(teardown_ms = teardown_ms) () =
  Gpus.make ~name ~memory_bar:0 ~nodes ~unreleased ~teardown_ms ~reset is_gpu

let gpu_buses = [ "0000:03:00.0"; "0000:43:00.0"; "0000:c3:00.0" ]
let gpu_bus = "0000:03:00.0"

(* Takes lock a function's file, which needs Linux. *)
let needs_flock () =
  if not on_linux then skip ~reason:"flock on a function's file needs Linux" ()

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
  Gpus.set_stop h (fun () -> `Clean);
  Ok h

let open_ g m i d =
  Gpus.open_ g m i (fun h _ ->
      let r = d () in
      if Result.is_ok r then ignore (kept h);
      r)

(* [reset g m i d] resets GPU [i], calling [d] once per reset the vendor ran. *)
let reset g m i d =
  let before = Atomic.get vendor_resets in
  let r = Gpus.reset g m i in
  for _ = before + 1 to Atomic.get vendor_resets do
    ignore (d ())
  done;
  r

let hold g m i = require_ok (Gpus.open_ g m i (fun h _ -> kept h))

(* [stopping g m i answer] holds GPU [i], whose vendor's stop answers
   [answer]. *)
let stopping g m i answer =
  require_ok
    (Gpus.open_ g m i (fun h _ ->
         Gpus.set_stop h (fun () -> answer);
         Ok h))

let stop h = ignore (Gpus.stop h : [ `Stopped | `Unknown ])

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
  let fn (id : Machine.id) =
    { (Tree.gpu id.bus) with vendor = id.vendor; class_ = id.class_ }
  in
  let m = Machine.at (Tree.make (List.map fn ids)) in
  equal (list string) (gpus_of ids) (Gpus.buses (gpus ()) m)

let test_ith () =
  needs_flock ();
  let g, m, _ = three () in
  let open_ith i bus =
    let msg = strf "GPU %d" i in
    stop
      (require_ok ~msg
         (Gpus.open_ g m i (fun h fn ->
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
  let why = unopened (f g m i) in
  starts_with ~msg:"names the GPU" ~affix:(Gpus.name g i) why;
  names 3 why

let test_none () =
  let m = Machine.at (Tree.make [ audio "0000:03:00.1" ]) in
  let g = gpus () in
  equal (list string) [] (Gpus.buses g m);
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
      ("name", fun i -> ignore (Gpus.name (gpus ()) i));
      ("open_", fun i -> ignore (open_ (gpus ()) (far ()) i ok));
      ("reset", fun i -> ignore (reset (gpus ()) (far ()) i ok));
      ("detach", fun i -> ignore (Gpus.detach (gpus ()) Machine.this i));
      ("attach", fun i -> ignore (Gpus.attach (gpus ()) Machine.this i));
    ]

(* GPU 0 bears the vendor's name, and GPU i that name and its number. *)
let test_name =
  cases ~name:(strf "GPU %d") "a GPU is named after its vendor and number"
    [ 0; 1; 12; max_int ] (fun i ->
      let want = if i = 0 then name else strf "%s:%d" name i in
      equal string want (Gpus.name (gpus ()) i))

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
    (Gpus.open_ g m 0 (fun h _ -> Result.map (fun _ -> 42) (kept h)));
  let why =
    require_error (Gpus.open_ g m 1 (fun _ _ -> Error "the GPU did not start"))
  in
  starts_with ~msg:"names the GPU" ~affix:(Gpus.name g 1) why;
  ends_with ~msg:"the driver's" ~affix:"the GPU did not start" why

(* [given_back g m root] asserts that GPU 0's function was given back, and that
   the GPU opens again. *)
let given_back g m root =
  equal ~msg:"its function held" bool false (held root "0000:03:00.0");
  stop (hold g m 0)

let test_error_gives_back () =
  needs_flock ();
  let g, m, root = three () in
  ignore (Gpus.open_ g m 0 (fun _ _ -> Error "the GPU did not start"));
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
      ("open_", fun g m e -> Gpus.open_ g m 0 (fun _ _ -> raise e));
      ("reset", fun g m _ -> Gpus.reset g m 0);
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
  starts_with ~msg:"names the GPU" ~affix:(Gpus.name g 0) why;
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
  require_ok (Gpus.reset (gpus ~reset:reset_gpu ()) m 1);
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
  let why = require_error (Gpus.reset g m 0) in
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
    Gpus.make ~name ~memory_bar:0 ~nodes:no_nodes ~unreleased:released
      ~teardown_ms ~reset:counted (fun _ -> false)
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
    equal ~msg:"the other domain reached Gpus" bool true
      (poll (fun () -> Atomic.get about));
    sample ();
    Atomic.set inside false;
    Result.map (fun _ -> d) (kept h)
  in
  let d = require_ok (Gpus.open_ g m 0 start) in
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
  let d = Domain.spawn (fun () -> Gpus.open_ g m 0 start) in
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
  g : Gpus.t;
  m : Machine.t;
  root : string;
  resets : int Atomic.t;
  stops : int Atomic.t;
  holds : Gpus.hold list Atomic.t;  (** Every hold an open made. *)
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
    ~release:(fun s -> List.iter stop (Atomic.get s.holds))
    ~invariant:(fun v s ->
      equal ~msg:"functions taken" (list string) (held_buses v)
        (List.filter (held s.root) (Array.to_list two_buses)))

let hold_t = abstract "h"
let vendor_ref () = { states = Array.make 2 Free; resets = 0; stops = 0 }

let vendor_sys () =
  let root = Tree.make two_gpus in
  let resets = Atomic.make 0 and stops = Atomic.make 0 in
  let reset _ =
    Atomic.incr resets;
    Ok ()
  in
  {
    g = gpus ~reset ();
    m = Machine.at root;
    root;
    resets;
    stops;
    holds = Atomic.make [];
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
      Gpus.set_stop h (fun () ->
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
  match Gpus.open_ s.g s.m i driver with
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
  match Gpus.open_ s.g s.m i (fun h _ -> kept h) with
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

let reset_sys s i = Result.is_ok (Gpus.reset s.g s.m i)

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
    command "stop" (hold_t ^-> returns stopped) stop_ref Gpus.stop;
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

(* [on_linux_only t] is [t] where takes lock a function's file. *)
let on_linux_only name t =
  if on_linux then t
  else
    test name (fun () ->
        skip ~reason:"flock on a function's file needs Linux" ())

let model_name = "opens, stops and resets behave as the model"

let two_domains_name =
  "from two domains, as some order of the calls: a stop calls the vendor's \
   once and answers the same twice"

let serialized =
  group ~timeout:patience "opens and changes"
    [
      on_linux_only model_name
        (stateful model_name ~count:300 ~steps:30
           (commands (indices [ 0; 1; 2; -1 ])));
      (* 60 programs of 50 runs each, each run a new tree, with a limit of three
         patiences: a run hands off between domains, which waits for a time
         slice when the processors are busy. *)
      on_linux_only two_domains_name
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
  require_ok (Gpus.detach (gpus ()) (Machine.at root) 0);
  equal hex memory_space (command_in root)

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

(* A GPU this process lost and attach reset opens again without a reset. *)
let test_attach_lost () =
  needs_flock ();
  let root = Tree.make [ Tree.gpu gpu_bus ] in
  let g = gpus () and m = Machine.at root in
  stop (stopping g m 0 `Lost);
  let before = Atomic.get vendor_resets in
  contains ~sub:"no kernel driver took 0000:03:00.0"
    (require_error (Gpus.attach g m 0));
  equal ~msg:"reset by attach" int 1 (Atomic.get vendor_resets - before);
  Tree.add root (sys (devices gpu_bus "enable")) "1\n";
  stop (hold g m 0);
  equal ~msg:"not by the open" int 1 (Atomic.get vendor_resets - before)

let test_change_held () =
  needs_flock ();
  let g = gpus () and m = Machine.at (Tree.make [ Tree.gpu gpu_bus ]) in
  let h = hold g m 0 in
  contains ~msg:"detach" ~sub:"0000:03:00.0 is open in this process"
    (require_error (Gpus.detach g m 0));
  contains ~msg:"attach" ~sub:"0000:03:00.0 is open in this process"
    (require_error (Gpus.attach g m 0));
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
  let why = require_error (Gpus.detach (gpus ()) (Machine.at root) 0) in
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
  match (want, Gpus.detach g (Machine.at root) 0) with
  | `Detached, r -> require_ok r
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
      let why = require_error (Gpus.detach g (Machine.at root) 0) in
      contains ~msg ~sub:"process 4242 holds /dev/null" why;
      equal ~msg string before (override root))
    [
      ("an open descriptor", "proc/4242/fd/7");
      ("a mapping", "proc/4242/map_files/7f00-7f01");
    ]

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
  let r = Gpus.detach (gpus ()) (Machine.at root) 0 in
  Domain.join closer;
  require_ok r;
  equal ~msg:"detached" string "none" (String.trim (override root))

(* A file of the GPU's DRM device the kernel keeps past the bound refuses
   detach, naming the process it holds it for, with the machine untouched. *)
let test_kernel_holds () =
  needs_flock ();
  let root = held_tree () in
  clients root [ (4242, "a compositor") ];
  let before = override root in
  let why =
    require_error (Gpus.detach (gpus ~teardown_ms:200 ()) (Machine.at root) 0)
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
    (require_error (Gpus.detach g (Machine.at root) 0))

(* A GPU with a DRM device whose files debugfs does not list is refused, since
   detach cannot know them, naming the list. *)
let test_no_debugfs () =
  needs_flock ();
  let root = held_tree () in
  let list = strf "sys/kernel/debug/dri/%s/clients" gpu_bus in
  Sys.remove (Filename.concat root list);
  let before = override root in
  contains ~sub:list (require_error (Gpus.detach (gpus ()) (Machine.at root) 0));
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
  let r = Gpus.detach (gpus ()) (Machine.at root) 0 in
  Unix.chmod fd 0o755;
  contains ~sub:"proc/4242/fd" (require_error r)

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
  require_ok (Gpus.detach (gpus ~unreleased:twice ()) (Machine.at root) 0);
  at_least ~msg:"asked until it let go" int ~than:3 (Atomic.get asked);
  let root = held_tree () in
  let original = override root in
  let always ~root:_ _ = Some "its release pending" in
  let why =
    require_error
      (Gpus.detach
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
    (require_error (Gpus.reset g m 0));
  contains ~msg:"attach" ~sub:"its release pending"
    (require_error (Gpus.attach g m 0));
  equal ~msg:"resets" int 0 (Atomic.get vendor_resets - before)

(* A GPU bound to vfio-pci stays as it is: nothing is waited for. *)
let test_vfio_unwaited () =
  needs_flock ();
  let root = Tree.make [ Tree.gpu ~driver:"vfio-pci" ~group:"7" gpu_bus ] in
  let always ~root:_ _ = Some "its release pending" in
  require_ok
    (Gpus.detach
       (gpus ~unreleased:always ~teardown_ms:60_000 ())
       (Machine.at root) 0)

let tree_changes =
  group ~timeout:patience "changes on a machine's files"
    [
      cases "each change writes what it says and answers for what remains"
        ~name:(fun (n, _, _, _, _) -> n)
        changes test_change;
      test "a GPU the process holds is refused" test_change_held;
      test "detach leaves the GPU's bus mastering off"
        test_detach_stops_mastering;
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
  g : Gpus.t;
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
    Gpus.open_ t.g t.m 0 (fun h fn ->
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
  starts_with ~affix:(Gpus.name t.g 1 ^ ":") (require_error (request t));
  stop h

let named_requests =
  [
    ("open_", fun t -> open_ t.g t.m 1 ok);
    ("reset", fun t -> Gpus.reset t.g t.m 1);
    ("detach", fun t -> Gpus.detach t.g t.m 1);
    ("attach", fun t -> Gpus.attach t.g t.m 1);
  ]

let test_set_stop_twice () =
  let t = tree () in
  let h =
    require_ok
      (Gpus.open_ t.g t.m 0 (fun h _ ->
           Gpus.set_stop h (fun () -> `Clean);
           raises_match (Exn.invalid_arg ~substring:"") (fun () ->
               Gpus.set_stop h (fun () -> `Lost));
           Ok h))
  in
  stop h;
  raises_match ~msg:"a stop once stopped" (Exn.invalid_arg ~substring:"")
    (fun () -> Gpus.set_stop h (fun () -> `Clean));
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
      (Gpus.open_ t.g t.m 0 (fun h fn ->
           Gpus.set_stop h (fun () ->
               Atomic.incr calls;
               answer);
           Ok (h, fn)))
  in
  let want = if answer = `Unknown then `Unknown else `Stopped in
  equal ~msg:"the first stop" stopped want (Gpus.stop h);
  equal ~msg:"the second" stopped want (Gpus.stop h);
  equal ~msg:"calls of the vendor's stop" int 1 (Atomic.get calls);
  equal ~msg:"the function released" bool true (Function.released fn)

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
             Gpus.set_stop h (fun () ->
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
        Gpus.set_stop h (fun () -> answer);
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
          (require_error (Gpus.renew h));
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
   the machine's dev/hugepages that no process locks, whose list of the
   functions reaching it names the GPU. *)
let dead_file t =
  let name = "dev/hugepages/rig-pci-999999-1" in
  Tree.add t.root name "";
  Tree.add t.root (name ^ ".reach") (gpu_bus ^ "\n");
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
  starts_with ~msg:"names the GPU" ~affix:(Gpus.name t.g 0) why;
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
          ignore (Gpus.renew h : (unit, string) result);
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
        Gpus.set_stop h (fun () ->
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
      Gpus.set_stop h (stop fn);
      Function.set_bus_master fn true;
      Ok h
    in
    Result.get_ok (Gpus.open_ g m i start)
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
  needs_flock ();
  let root = Tree.make (List.map Tree.gpu gpu_buses) in
  let out, err = exiting root in
  equal ~msg:"stopped, once each, still mastering" (list string)
    [ "stopped 0000:c3:00.0 mastering"; "stopped 0000:43:00.0 mastering" ]
    out;
  contains ~msg:"the stop that raised"
    ~sub:"stopping 0000:03:00.0 at exit: Failure(\"a driver bug\")" err

let exits =
  group ~timeout:patience "exit"
    [
      test
        "a process that exits stops each GPU it holds once, before its bus \
         mastering is turned off, past a stop that raises, and its forked \
         child stops none"
        test_exit;
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
             resets;
             tree_changes;
             tree_holds;
             exits;
             this_machine;
             serialized;
           ]
