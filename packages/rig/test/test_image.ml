(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Polled loads "code:N": N bytes of code ('c' each) in its memory, its function
   "main" at their start; and "functions", which it places nowhere. Its "fill"
   reads 16 bytes of parameters, its "copy" 24, its "main" states none, and each
   runs at most 1024 threads per group (rig_support.mli). *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module Image = Rig.Image
module P = Rig_support.Polled
module Support = Rig_support

let submit ?(buffers = [||]) ?(args = [||]) ?waits s =
  Rig.submit ?waits s ~buffers ~args

let count call p = List.length (List.filter (( = ) call) (P.log p))
let load d binary = require_ok ~pp:Format.pp_print_string (Image.load d binary)
let fn image f = require_some (Image.resolve image f)
let lost d = function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false
let device = Testable.make ~pp:Rig.pp ~equal:Rig.equal

(* Images are told apart by identity. One prints as its device and the entry of
   its [main], which every Polled binary has and each load places anew ("Each
   call loads [binary] again"), so two images of one device print apart. *)
let image =
  let pp ppf i =
    let main =
      match Image.resolve i "main" with
      | Some f -> Printf.sprintf ", main at %#x" (Image.entry f)
      | None | (exception _) -> ""
    in
    Format.fprintf ppf "an image on %a%s" Rig.pp (Image.device i) main
  in
  Testable.make ~equal:( == ) ~pp

(* Loses [d] by a submit its driver fails to take. *)
let lose d p =
  P.fail p;
  raises_match (lost d) (fun () -> submit (Sub.make d [||]))

(* The bytes at the start of "code:N" once placed: eight 'c's. *)
let placed_code = 0x6363636363636363

(* Work submitted after a load sees its code at [main]'s entry. *)
let sees_code d pd image =
  Rig.Point.wait (submit (Sub.make d [||]));
  equal ~msg:"no work left queued" int 0 (P.queued pd);
  equal ~msg:"the code placed" int placed_code
    (Support.load (Image.entry (fn image "main")))

(* Image.load *)

(* A copy takes a value after the previous work's, and the load returns before
   it ran, whether the copy reads device memory the host addresses or pinned
   memory. *)
let test_load_copied host_visible =
  let d, pd =
    P.open_ ~host_visible (Printf.sprintf "image:copied-%b" host_visible)
  in
  let previous = submit (Sub.make d [||]) in
  let i = load d "code:64" in
  less ~msg:"the copy's value follows the previous work's" int
    ~than:(Rig.submitted d) (Rig.Point.value previous);
  less ~msg:"the load returns before the copy ran" int ~than:(Rig.submitted d)
    (Rig.signaled d);
  sees_code d pd i

(* With no copy engine the host writes the code once the previous work is
   done. *)
let test_load_after_work () =
  let d, pd = P.open_ ~copies:false "image:after-work" in
  let previous = submit (Sub.make d [||]) in
  let i = load d "code:64" in
  at_least ~msg:"previous work completed" int ~than:(Rig.Point.value previous)
    (Rig.signaled d);
  sees_code d pd i

let test_load_unreached () =
  let d, pd = P.open_ ~copies:false ~host_visible:false "image:unreached" in
  ignore (submit (Sub.make d [||]));
  raises_match Exn.invalid_arg (fun () -> Image.load d "code:64");
  let later = submit (Sub.make d [||]) in
  Rig.Point.wait later;
  equal ~msg:"later work completed" int (Rig.Point.value later) (Rig.signaled d);
  equal ~msg:"no work left queued" int 0 (P.queued pd)

(* "Each call loads [binary] again": on Polled, which places code in its memory,
   two loads are two images with two copies of the code. Host images of one
   object share their code instead (host/test_host.ml). *)
let test_load_again () =
  let d, p = P.open_ "image:again" in
  let a = load d "code:64" and b = load d "code:64" in
  equal ~msg:"images loaded" int 2 (P.images p);
  not_equal ~msg:"main's entries" int
    (Image.entry (fn a "main"))
    (Image.entry (fn b "main"))

let test_device () =
  let d, _ = P.open_ "image:device" in
  equal device d (Image.device (load d "functions"))

let test_budget () =
  let d, _ = P.open_ "image:budget" in
  Rig.set_budget d 4096;
  raises_match
    (function
      | Rig.Out_of_memory (d', n) -> Rig.equal d d' && n >= 8192 | _ -> false)
    (fun () -> Image.load d "code:8192")

(* Polled's reason is "not a polled binary". *)
let test_refused () =
  let d, _ = P.open_ "image:refused" in
  let base = Rig.submitted d in
  let why = require_error (Image.load d "not code") in
  starts_with ~msg:"the device's name first" ~prefix:"image:refused" why;
  ends_with ~msg:"the driver's reason last" ~suffix:"not a polled binary" why;
  equal ~msg:"the device's next value" int (base + 1)
    (Rig.Point.value (submit (Sub.make d [||])))

let test_no_code () =
  raises_match Exn.invalid_arg (fun () ->
      Image.load (Rig_support.io "image:no-code") "code:64")

let test_load_lost () =
  let d, p = P.open_ "image:load-lost" in
  lose d p;
  raises_match (lost d) (fun () -> Image.load d "functions")

(* A load whose upload fails loses the device. *)
let test_upload_fails () =
  let d, p = P.open_ "image:upload" in
  P.fail p;
  raises_match (lost d) (fun () -> Image.load d "code:64")

(* Collects, then frees [d]'s cache, which first drains [d]: a lost device's
   memory collected after its stop returns so. *)
let collect d =
  Gc.full_major ();
  Rig.free_cache d

(* A load whose upload loses the device gives its code memory back only once the
   device counts as stopped: its work may still run until then. The upload's
   staging memory is pinned on a device whose memory the host does not address,
   so the device memory held is the code's. *)
let test_upload_frees_after_stop () =
  let d, p =
    P.open_ ~host_visible:false ~answer:`Unknown "image:upload-unknown"
  in
  P.fail p;
  raises_match (lost d) (fun () -> Image.load d "code:64");
  Rig.close d;
  collect d;
  equal ~msg:"before the word drained" int 64 (P.allocated p Device);
  P.set_word p (Rig.submitted d);
  collect d;
  equal ~msg:"once it did" int 0 (P.allocated p Device)

(* Unloading *)

(* An unreachable image is unloaded once the work its device was handed until
   then is done, and its code's memory then returns to its device. *)
let test_unload () =
  let d, pd = P.open_ "image:unload" in
  let code =
    (fun () ->
      let p = load d "code:64" in
      ignore (submit (Sub.make d [||]));
      Image.entry (fn p "main"))
      ()
  in
  let drain () =
    Gc.full_major ();
    Gc.full_major ();
    ignore (B.create d 0)
  in
  drain ();
  equal ~msg:"while its work is unrun" int 0 (count "unload" pd);
  ignore (P.run pd);
  drain ();
  equal ~msg:"once it ran" int 1 (count "unload" pd);
  Rig.free_cache d;
  mem ~msg:"its code's memory is freed" int code (List.map fst (P.frees pd))

let rec after_stop = function
  | "stop" :: rest -> rest
  | _ :: rest -> after_stop rest
  | [] -> []

(* An image still loaded when its device stops, at its close, is unloaded after
   the stop, once unreachable: the stop releases nothing this library made. *)
let test_unload_after_stop () =
  let d, p = P.open_ "image:unload-after-stop" in
  let i = ref (Some (load d "code:64")) in
  let code = Image.entry (fn (Option.get !i) "main") in
  lose d p;
  raises_match (lost d) (fun () -> Image.resolve (Option.get !i) "main");
  Rig.close d;
  equal ~msg:"by the stop" int 0 (count "unload" p);
  i := None;
  collect d;
  equal ~msg:"after the stop" (list string) [ "unload" ]
    (List.filter (( = ) "unload") (after_stop (P.log p)));
  mem ~msg:"its code's memory is freed" int code (List.map fst (P.frees p))

(* A lost device whose work may still run unloads nothing until its word reads
   its last value. *)
let test_unload_after_unknown () =
  let d, p = P.open_ ~answer:`Unknown "image:unload-unknown" in
  let i = ref (Some (load d "code:64")) in
  ignore (submit (Sub.make d [||]));
  lose d p;
  Rig.close d;
  i := None;
  collect d;
  equal ~msg:"with the word short" int 0 (count "unload" p);
  P.set_word p (Rig.submitted d);
  collect d;
  equal ~msg:"once the word drained" int 1 (count "unload" p)

let drain d = ignore (Sys.opaque_identity (B.create d 0))

let collect_on d =
  for _ = 1 to 3 do
    Gc.full_major ();
    drain d
  done

(* A function alone keeps its image loaded; once it goes too, the image
   unloads. *)
let test_func_keeps () =
  let d, p = P.open_ "image:func-keeps" in
  let main = ref (Some (fn (load d "functions") "main")) in
  ignore (P.run p);
  collect_on d;
  equal ~msg:"while its function is held" int 0 (count "unload" p);
  ignore (Sys.opaque_identity !main);
  main := None;
  collect_on d;
  equal ~msg:"once it goes" int 1 (count "unload" p)

(* Code can follow an image with no code in the release queue. Pressure must
   find those bytes and finish the work that holds them before allocating. *)
let test_pending_code_pressure () =
  let d, p = P.open_ ~budget:4096 ~copies:false "image:pending-code-pressure" in
  let first = ref (Some (load d "functions")) in
  let second = ref (Some (load d "code:4096")) in
  ignore (submit (Sub.make d [||]));
  first := None;
  collect_on d;
  second := None;
  collect_on d;
  equal ~msg:"both images wait" int 0 (count "unload" p);
  let b = B.create d 4096 in
  equal ~msg:"the replacement fits" int 4096 (B.length b);
  equal ~msg:"the work completed" int 0 (P.queued p);
  equal ~msg:"both images unloaded" int 2 (count "unload" p)

(* A native unload holds no release lock. A concurrent collection can append
   another image for a later value while that unload is still running. *)
let test_append_during_unload () =
  let d, p = P.open_ "image:append-during-unload" in
  let first = ref (Some (load d "functions")) in
  let second = ref (Some (load d "functions")) in
  ignore (submit (Sub.make d [||]));
  first := None;
  collect_on d;
  ignore (P.run p);
  let entered = Atomic.make false and finish = Atomic.make false in
  P.before p "unload" (fun () ->
      Atomic.set entered true;
      Support.await "the concurrent append" (fun () -> Atomic.get finish));
  let worker = Domain.spawn (fun () -> drain d) in
  Fun.protect
    ~finally:(fun () ->
      Atomic.set finish true;
      Domain.join worker)
    (fun () ->
      Support.await "the first unload" (fun () -> Atomic.get entered);
      ignore (submit (Sub.make d [||]));
      second := None;
      collect_on d;
      equal ~msg:"only the first image unloads" int 1 (count "unload" p);
      equal ~msg:"later work stays queued" int 1 (P.queued p));
  ignore (P.run p);
  drain d;
  equal ~msg:"the later image unloads once" int 2 (count "unload" p)

(* The first unload discovers a fault after newer work was handed over. Other
   releases already taken from the queue must wait again for that work. *)
let test_unloads_beside_loss () =
  let d, p = P.open_ ~answer:`Unknown "image:unloads-beside-loss" in
  let images = Array.init 2 (fun _ -> ref (Some (load d "functions"))) in
  ignore (submit (Sub.make d [||]));
  Array.iter (fun image -> image := None) images;
  collect_on d;
  ignore (P.run p);
  P.before p "unload" (fun () ->
      ignore (submit (Sub.make d [||]));
      P.fault p "the engine hung");
  drain d;
  equal ~msg:"the fault is recorded" (option string) (Some "the engine hung")
    (Rig.lost d);
  equal ~msg:"later unloads wait" int 1 (count "unload" p);
  Rig.close d;
  collect d;
  equal ~msg:"the unknown stop keeps them" int 1 (count "unload" p);
  P.set_word p (Rig.submitted d);
  collect d;
  equal ~msg:"the final release retries once stopped" int 2 (count "unload" p)

(* Two domains drop images and drain their device while its work waits to run.
   One domain's drain, which found the first image collected, is held before its
   first step ({!Rig.on_drain}) while the other drops the second and collects
   it, and with [drains] drains too. No drain unloads an image before the work
   ran; then each collected image unloads once, and the live one stays. *)
let test_parallel_drops drains =
  let d, p =
    P.open_ (if drains then "image:drops-drains" else "image:drops-collects")
  in
  let first = ref (Some (load d "functions")) in
  let second = ref (Some (load d "functions")) in
  let live = load d "functions" in
  ignore (submit (Sub.make d [||]));
  first := None;
  Gc.full_major ();
  let entered = Atomic.make false and go = Atomic.make false in
  Rig.on_drain (fun () ->
      Atomic.set entered true;
      Support.await "the other domain's drop" (fun () -> Atomic.get go));
  let held = Domain.spawn (fun () -> drain d) in
  Fun.protect
    ~finally:(fun () ->
      Atomic.set go true;
      Domain.join held)
    (fun () ->
      Support.await "the held drain" (fun () -> Atomic.get entered);
      second := None;
      Gc.full_major ();
      if drains then drain d;
      equal ~msg:"unloads while a drain is held" int 0 (count "unload" p));
  equal ~msg:"unloads once both drains returned" int 0 (count "unload" p);
  ignore (P.run p);
  drain d;
  equal ~msg:"unloads once the work ran" int 2 (count "unload" p);
  ignore (Sys.opaque_identity live)

(* Image.resolve *)

let test_resolve_absent () =
  let d, _ = P.open_ "image:absent" in
  is_none
    ~pp:(fun ppf f -> Format.pp_print_string ppf (Image.name f))
    (Image.resolve (load d "functions") "other")

(* An image stays loaded while its driver makes one of its functions: a
   collection and a drain at the start of the driver's entry call, the image's
   last use, unload nothing. *)
let test_entry_keeps () =
  let d, pd = P.open_ "image:entry-keeps" in
  let unloads = ref (-1) in
  P.before pd "entry" (fun () ->
      Gc.full_major ();
      Gc.full_major ();
      ignore (B.create d 0);
      unloads := count "unload" pd);
  ignore (Image.resolve (load d "code:64") "main");
  equal ~msg:"unloads during the entry" int 0 !unloads

let test_entry_lost () =
  let d, p = P.open_ "image:entry-lost" in
  let i = load d "code:64" in
  lose d p;
  raises_match (lost d) (fun () -> Image.resolve i "main")

(* A function the device has no memory for is asked for again after each round
   of reclamation, and raises Out_of_memory for the bytes its launches need
   after the last, the device live. *)
let test_entry_no_memory () =
  let d, pd = P.open_ "image:entry-no-memory" in
  let i = load d "code:64" in
  P.fail_at pd 1 (`Refuse max_int);
  raises_match
    (function
      | Rig.Out_of_memory (d', n) -> Rig.equal d d' && n = P.entry_scratch
      | _ -> false)
    (fun () -> Image.resolve i "main");
  P.fail_at pd 1 (`Refuse 0);
  at_least ~msg:"times asked" int ~than:2 (count "entry" pd);
  equal ~msg:"the device's loss" (option string) None (Rig.lost d)

(* A function the device had no memory for at first is found once a round of
   reclamation is done. *)
let test_entry_reclaims () =
  let d, pd = P.open_ "image:entry-reclaims" in
  let i = load d "code:64" in
  P.before pd "entry" (fun () -> P.fail_at pd 1 (`Refuse 1));
  equal ~msg:"the function found" (option string) (Some "main")
    (Option.map Image.name (Image.resolve i "main"));
  equal ~msg:"times asked" int 2 (count "entry" pd)

(* The driver finds the function needs more than the device offers. *)
let test_entry_refused () =
  let d, p = P.open_ "image:entry-refused" in
  let i = load d "functions" in
  P.before p "entry" (fun () -> invalid_arg "main needs more than 1 TiB");
  raises_match (Exn.invalid_arg ~sub:"main needs more than 1 TiB") (fun () ->
      Image.resolve i "main");
  equal ~msg:"the device's loss" (option string) None (Rig.lost d)

(* The fault lands at the device's next fallible call: parts of a function
   already resolved ask the driver nothing, so it is the resolve's. *)
let test_entry_faults () =
  let d, p = P.open_ "image:entry-faults" in
  let main = fn (load d "functions") "main" and i = load d "functions" in
  P.fail_at p 1 (`Fault "the engine hung");
  ignore
    (Sub.make d
       [|
         Sub.launch main ~params:8;
         Sub.fill ~refs:[| Sub.entry main ~at:0 |] Rig_support.noop ~params:8;
       |]);
  raises_match
    (function
      | Rig.Lost (d', why) -> Rig.equal d d' && why = "the engine hung"
      | _ -> false)
    (fun () -> Image.resolve i "main")

(* Image.prepare *)

(* Polled makes nothing slowly: its prepare answers at once. *)
let test_prepare () =
  let d, p = P.open_ "image:prepare" in
  let i = load d "functions" in
  Image.prepare i "fill";
  equal ~msg:"asked before the resolve" int 0 (count "entry" p);
  equal ~msg:"fill's parameters" (option int) (Some 16)
    (Image.params (fn i "fill"));
  equal ~msg:"times asked" int 1 (count "entry" p)

let test_prepare_absent () =
  let d, _ = P.open_ "image:prepare-absent" in
  let i = load d "functions" in
  Image.prepare i "other";
  is_none
    ~pp:(fun ppf f -> Format.pp_print_string ppf (Image.name f))
    (Image.resolve i "other");
  equal ~msg:"the device's loss" (option string) None (Rig.lost d)

let test_prepare_lost () =
  let d, p = P.open_ "image:prepare-lost" in
  let i = load d "functions" in
  lose d p;
  raises_match (lost d) (fun () -> Image.prepare i "main")

(* Image.func *)

let test_func_names () =
  let d, _ = P.open_ "image:func-names" in
  let i = load d "functions" in
  let main = fn i "main" in
  equal ~msg:"its name" string "main" (Image.name main);
  equal ~msg:"its image" image i (Image.image main)

(* Polled states fill's 16 bytes, copy's 24 and none for main, and 1024 threads
   for each. *)
let test_func_facts () =
  let d, _ = P.open_ "image:func-facts" in
  let i = load d "functions" in
  let params f = Image.params (fn i f) and threads f = Image.threads (fn i f) in
  equal (option int) ~msg:"fill's parameters" (Some 16) (params "fill");
  equal (option int) ~msg:"copy's parameters" (Some 24) (params "copy");
  equal (option int) ~msg:"main's parameters" None (params "main");
  equal (option int) ~msg:"main's threads" (Some 1024) (threads "main")

let tests =
  [
    group "Image.load"
      [
        cases "a load returns before its code's copy runs"
          ~id:(fun host_visible ->
            if host_visible then "from memory the host addresses"
            else "from pinned memory")
          [ true; false ] test_load_copied;
        test "a load places code after earlier work" test_load_after_work;
        test "a load raises where neither copies nor the host reach the code"
          test_load_unreached;
        test "each load on a device that places code loads the binary again"
          test_load_again;
        test "code counts in its device's budget" test_budget;
        test "a binary its driver refuses is an error naming the device"
          test_refused;
        test "an io device refuses a binary" test_no_code;
        test "a load on a lost device raises Lost" test_load_lost;
        test "a load whose upload fails loses the device" test_upload_fails;
        test "a failed upload frees its code once the device counts as stopped"
          test_upload_frees_after_stop;
      ];
    group "unloading"
      [
        test "an unreachable image unloads once its device's work is done"
          test_unload;
        test "a function keeps its image loaded" test_func_keeps;
        test "a lost device's image is unloaded after its stop"
          test_unload_after_stop;
        test "a lost device's image waits for its word to drain"
          test_unload_after_unknown;
        test "allocation pressure waits for queued code memory"
          test_pending_code_pressure;
        test
          "an image collected while another unloads unloads once its own work \
           is done"
          test_append_during_unload;
        test "images collected before a loss unload once the device has stopped"
          test_unloads_beside_loss;
        cases "parallel drops and drains keep images until work completes"
          ~id:(fun drains ->
            if drains then "both domains drain" else "one domain drains")
          [ true; false ] test_parallel_drops;
      ];
    group "Image.device"
      [ test "an image is on the device it was loaded on" test_device ];
    group "Image.resolve"
      [
        test "an image has no function it does not define" test_resolve_absent;
        test "an image stays loaded while its driver makes a function"
          test_entry_keeps;
        test "a resolve on a lost device raises Lost" test_entry_lost;
        test "a function with no memory raises Out_of_memory after reclaiming"
          test_entry_no_memory;
        test "a function with no memory is found once reclaiming frees some"
          test_entry_reclaims;
        test "a function the device cannot run raises Invalid_argument"
          test_entry_refused;
        test "a fault while asking for a function loses the device"
          test_entry_faults;
      ];
    group "Image.prepare"
      [
        test "on a device that makes nothing slowly, the first resolve asks"
          test_prepare;
        test "prepare of a function the image lacks does nothing"
          test_prepare_absent;
        test "prepare on a lost device raises Lost" test_prepare_lost;
      ];
    group "Image.func"
      [
        test "a function has its name and image" test_func_names;
        test "a function's parameters and threads are what its driver states"
          test_func_facts;
      ];
  ]

let () = exit (run "rig.image" tests)
