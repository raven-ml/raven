(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* rig.mli's Submitting and [submit]. Polled devices run their queue only when
   the test runs it or a wait sleeps, so a submit there hands its work over and
   returns; a software device runs it before [submit] returns. *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module Support = Rig_support

let strf = Printf.sprintf
let memory name = Result.get_ok (Rig.software name)

(* [b] borrowed on [d], as a submit on [d] takes it. *)
let on d b = require_some (B.borrow d b)

(* A submission of no part, or of one fill that does nothing and uses a slot per
   element of [uses] ({!Support.touching}). *)
let empty ?(uses = [||]) d =
  if uses = [||] then Sub.make d [||]
  else Sub.make d [| Support.touching uses |]

let submit ?(buffers = [||]) ?(args = [||]) ?waits s =
  Rig.submit ?waits s ~buffers ~args

(* The values of a submission's one fill, which reads none. *)
let one = Support.zeros 1

(* Submits [empty ~uses d] over [buffers]. *)
let touch ?waits uses d buffers =
  submit ?waits (empty ~uses d) ~buffers ~args:one

let page_bytes = B.paged_from

(* A part that adds 1 to the 64-bit word in a submit's slot 0. *)
let bump = Sub.fill ~refs:[| Sub.write 0 ~at:0 |] Support.bump ~params:8

(* A part that holds one unit of a Polled queue. *)
let idle = Sub.fill Support.noop ~params:0
let word b = Support.load (Support.Reader.host b)
let lost = function Rig.Lost _ -> true | _ -> false
let lost_on d = function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false
let count call p = List.length (List.filter (( = ) call) (P.log p))
let names = Atomic.make 0
let fresh what = strf "submit:%s-%d" what (Atomic.fetch_and_add names 1)

let functions d =
  require_ok ~pp:Format.pp_print_string (Rig.Image.load d "functions")

let fn image f = require_some (Rig.Image.resolve image f)

let launch ?(params = 0) ?(refs = [||]) image kernel =
  Sub.launch ~refs (fn image kernel) ~params

(* The values of a launch of one group of one thread. *)
let single ?(params = Bytes.empty) () =
  { Sub.groups = (1, 1, 1); threads = (1, 1, 1); shared = 0; params }

(* The bytes of the 64-bit words [vs], little-endian. *)
let words vs =
  let b = Bytes.create (8 * List.length vs) in
  List.iteri (fun i v -> Bytes.set_int64_le b (8 * i) (Int64.of_int v)) vs;
  b

let host_bytes b n =
  let s = Bytes.create n in
  B.blit_to_bytes b 0 s 0 n;
  Bytes.to_string s

let pp_launch ppf (l : P.launch) =
  let x, y, z = l.groups and tx, ty, tz = l.threads in
  Format.fprintf ppf "groups %dx%dx%d threads %dx%dx%d shared %d params %S" x y
    z tx ty tz l.shared l.params

let launch_t = Testable.make ~pp:pp_launch ~equal:( = )
let params_of launches = List.map (fun (l : P.launch) -> l.params) launches

(* The 64-bit word each launch read first in its parameters. *)
let first_words launches =
  List.map
    (fun p -> Int64.to_int (String.get_int64_le p 0))
    (params_of launches)

(* Waits *)

(* A buffer only the submit's array reaches, while the submit waits on the host
   for another device, is not collected: a buffer another domain makes
   meanwhile, after a full collection and a drain, gets other memory. *)
let test_run_keeps () =
  let producer, pp = P.open_ "submit:keeps-producer" in
  let d, _ = P.open_ "submit:keeps" in
  let n = 4096 in
  let a = submit (empty producer) in
  P.gate pp;
  let other =
    Domain.spawn (fun () ->
        Support.await "the submit waiting on the producer" (fun () ->
            P.sleepers pp = 1);
        Gc.full_major ();
        let b = B.create d n in
        P.open_gate pp;
        b)
  in
  let at = ref 0 in
  let point =
    touch [| Sub.write |] d
      (let w = B.create d n in
       at := Support.Reader.host w;
       [| w |])
      ~waits:[| a |]
  in
  let b = Domain.join other in
  not_equal ~msg:"another domain's buffer's address" int !at
    (Support.Reader.host b);
  Rig.Point.wait point

(* A Polled device that waits on the words of producers of [completion] waits
   for a producer in its queue: the submit hands it over without waiting, and
   the wait names the producer's word. *)
let in_queue ~completion =
  let name = match completion with `Host -> "host" | `Store -> "store" in
  let producer, pp = P.open_ ~completion ("submit:iq-producer-" ^ name) in
  let consumer, cp =
    P.open_
      ~waits_on:[ (completion :> [ `Store | `Host ]) ]
      ("submit:iq-consumer-" ^ name)
  in
  let a = submit (empty producer) in
  let b = submit (empty consumer) ~waits:[| a |] in
  equal (list (pair int int)) [ (P.word_at pp, 1) ] (P.last_waits cp);
  equal int 1 (P.queued pp);
  equal int 1 (P.queued cp);
  equal int 0 (P.run cp);
  Rig.Point.wait a;
  equal int 1 (P.run cp);
  equal int (Rig.Point.value b) (Rig.signaled consumer)

(* A device's room check is given the number of waits its queue gets in the
   submit after it, which is the room they take: every unreached producer. *)
let test_room_waits () =
  let producers =
    List.init 12 (fun i ->
        P.open_ (Printf.sprintf "submit:room-waits-producer-%d" i))
  in
  let points = List.map (fun (d, _) -> submit (empty d)) producers in
  let consumer, cp = P.open_ ~waits_on:[ `Host ] "submit:room-waits" in
  ignore (submit (empty consumer) ~waits:(Array.of_list points));
  equal ~msg:"the room's waits" int 12 (P.room_waits cp);
  List.iter (fun (_, p) -> ignore (P.run p)) producers

(* A producer whose word the device cannot map, as a device that maps no host
   memory, is waited for on the host: the submit returns with the producer's
   work done and no wait in the queue. *)
let test_unmapped_wait () =
  let producer, pp = P.open_ "submit:unmapped-producer" in
  let consumer, cp =
    P.open_ ~peers:false ~maps_host:false ~waits_on:[ `Host ]
      "submit:unmapped-consumer"
  in
  let a = submit (empty producer) in
  ignore (submit (empty consumer) ~waits:[| a |]);
  equal ~msg:"waits in the queue" int 0 (List.length (P.last_waits cp));
  equal ~msg:"the producer's work" int 0 (P.queued pp);
  equal int (Rig.Point.value a) (Rig.signaled producer)

let test_in_queue () = in_queue ~completion:`Host
let test_in_queue_store () = in_queue ~completion:`Store

(* Room *)

(* Submissions of one [idle] part and of none, each with its values, and the
   value a submit of one takes. *)
let one_part d = (Sub.make d [| idle |], Support.zeros 1)
let no_part d = (empty d, [||])
let value (s, args) = Rig.Point.value (submit s ~args)

(* A thread of the test's domain running [f], whose outcome [join] gives. *)
let spawn f =
  let r = ref (Error Exit) in
  let t = Thread.create (fun () -> r := try Ok (f ()) with e -> Error e) () in
  fun () ->
    Thread.join t;
    !r

let pp_outcome ppf = function
  | Ok v -> Format.fprintf ppf "Ok %d" v
  | Error e -> Format.fprintf ppf "Error %s" (Printexc.to_string e)

let outcome = Testable.make ~pp:pp_outcome ~equal:( = )

(* Room alone bounds a submission: one the device's queues hold runs, and one
   they never hold raises. *)
let test_never_fits () =
  let d, _ = P.open_ ~capacity:4 "submit:never-fits" in
  let four = Sub.make d (Array.make 4 idle) in
  Rig.Point.wait (submit four ~args:(Support.zeros 4));
  let five = Sub.make d (Array.make 5 idle) in
  raises_match ~msg:"more than the queue holds"
    (Exn.invalid_arg ~sub:"never fit") (fun () ->
      submit five ~args:(Support.zeros 5))

(* A full queue answers Later: the submit waits for one more value. *)
let test_room () =
  let d, p = P.open_ ~capacity:1 "submit:room" in
  let base = Rig.submitted d in
  let s = Sub.make d [| idle |] in
  ignore (submit s ~args:one);
  ignore (submit s ~args:one);
  equal int 1 (P.queued p);
  equal int (base + 1) (Rig.signaled d)

(* A submit that waits for room waits with the turn released: a submission with
   no parts, which fits, is handed over meanwhile. *)
let test_later_releases () =
  let d, p = P.open_ ~capacity:1 "submit:later" in
  let base = Rig.submitted d in
  equal int (base + 1) (value (one_part d));
  P.gate p;
  let waiting = spawn (fun () -> value (one_part d)) in
  Support.await "a sleep at the gate" (fun () -> P.sleepers p = 1);
  equal int (base + 2) (value (no_part d));
  P.open_gate p;
  equal outcome (Ok (base + 3)) (waiting ())

(* A fault found while a submit waits for room loses the device, and the submit
   commits nothing. *)
let test_fault_waiting () =
  let d, p = P.open_ ~capacity:1 "submit:fault-room" in
  let base = Rig.submitted d in
  equal int (base + 1) (value (one_part d));
  P.gate p;
  let waiting = spawn (fun () -> value (one_part d)) in
  Support.await "a sleep at the gate" (fun () -> P.sleepers p = 1);
  P.fault p "the engine hung";
  P.open_gate p;
  let why =
    match waiting () with
    | Error (Rig.Lost (d', why)) when Rig.equal d d' -> why
    | Error e -> failf "the waiting submit raised %s" (Printexc.to_string e)
    | Ok v -> failf "the waiting submit returned %d" v
  in
  equal ~msg:"the waiting submit's loss" string "the engine hung" why;
  equal ~msg:"values assigned" int (base + 1) (Rig.submitted d)

(* Polled refuses a launch of [main] with no group as work it cannot run:
   [submit] raises and assigns no value, and the next submit takes the next
   value. *)
let test_never () =
  let d, _ = P.open_ "submit:never" in
  let base = Rig.submitted d in
  let never = Sub.make d [| launch (functions d) "main" |] in
  raises_match Exn.invalid_arg (fun () -> submit never ~args:one);
  equal int base (Rig.submitted d);
  equal int (base + 1) (value (one_part d))

(* Submit *)

let device = Testable.make ~pp:Rig.pp ~equal:Rig.equal

let test_values () =
  let d = memory "submit:values" in
  let base = Rig.submitted d in
  let s = empty d in
  let values = List.init 5 (fun _ -> Rig.Point.value (submit s)) in
  equal (list int) (List.init 5 (fun i -> base + i + 1)) values;
  equal int (base + 5) (Rig.submitted d);
  equal int (base + 5) (Rig.signaled d);
  equal (option device) (Some d) (Rig.Point.device (submit s))

let test_fill () =
  let d = memory "submit:fill" in
  let arg = B.create Rig.host 8 in
  Support.store (Support.Reader.host arg) 0;
  let s = Sub.make d [| bump; bump |] in
  let buffers = [| on d arg |] and args = Support.zeros 2 in
  ignore (submit ~buffers ~args s);
  ignore (submit ~buffers ~args s);
  equal int 4 (word arg)

(* A submit that raises keeps nothing: the next submit of the submission waits
   for none of its points. *)
let test_cleared_on_raise () =
  let d = memory "submit:cleared" in
  let base = Rig.submitted d in
  let producer, pp = P.open_ "submit:cleared-producer" in
  let point = submit (empty producer) in
  P.fail pp;
  (try ignore (submit (empty producer)) with Rig.Lost _ -> ());
  let s = empty ~uses:[| Sub.read |] d and b = on d (B.create Rig.host 8) in
  raises_match
    (function Rig.Lost _ -> true | _ -> false)
    (fun () -> submit s ~buffers:[| b |] ~args:one ~waits:[| point |]);
  equal int (base + 1) (Rig.Point.value (submit s ~buffers:[| b |] ~args:one))

(* The device and image that each domain test's submission launches [fill] on,
   made once for every case. *)
let pool_device =
  lazy
    (let d, _ = P.open_ (fresh "pool") in
     (d, functions d))

let filling_on (d, image) =
  Sub.make d [| launch image "fill" ~params:16 ~refs:[| Sub.write 0 ~at:0 |] |]

(* The values of [fill] that store [v]. *)
let fill_values v = single ~params:(words [ 0; v ]) ()

(* Submits [s], a submission of [filling_on], with [args], and reads back the
   word its launch wrote. *)
let fill_with s args =
  let d, _ = Lazy.force pool_device in
  let out = B.create d 8 in
  Rig.Point.wait (submit ~buffers:[| out |] ~args s);
  Int64.to_int (String.get_int64_le (host_bytes out 8) 0)

(* Two domains submit one submission, each with values of its own, and each
   reads back its own parameter. *)
let sub_t = abstract "s"

let own_commands =
  [
    command "make"
      (Gen.unit @-> makes sub_t)
      Fun.id
      (fun () -> filling_on (Lazy.force pool_device));
    command "submit"
      (sub_t ^-> Gen.int_range 1 1000 @-> returns int)
      (fun () v -> v)
      (fun s v -> fill_with s [| fill_values v |]);
  ]

(* Two domains share one array of values, whose element a store replaces,
   and submit one submission with it. A submit reads the element once and
   checks what it read, so each reads back the parameter of the last store
   before it in some order of the calls, and raises where that store put
   values the launch refuses: more bytes than its 16. *)
(* The model: the parameter of the last store, [None] where it put values
   the launch refuses. *)
type passed = { mutable stored : int option }
type passing = { sub : Sub.t; args : Sub.args array }

let passing_t =
  abstract
    ~pp:(fun ppf m ->
      match m.stored with
      | Some v -> Format.fprintf ppf "stored %d" v
      | None -> Format.pp_print_string ppf "refused")
    "p"

let too_long = single ~params:(Bytes.make 24 '\000') ()

let passing_commands =
  [
    command "make"
      (Gen.unit @-> makes passing_t)
      (fun () -> { stored = Some 0 })
      (fun () ->
        let sub = filling_on (Lazy.force pool_device) in
        { sub; args = [| fill_values 0 |] });
    command "store"
      (passing_t ^-> Gen.int_range 1 1000 @-> returns unit)
      (fun m v -> m.stored <- Some v)
      (fun t v -> t.args.(0) <- fill_values v);
    command "spoil"
      (passing_t ^-> returns unit)
      (fun m -> m.stored <- None)
      (fun t -> t.args.(0) <- too_long);
    command "submit"
      (passing_t ^-> judges int)
      (fun m -> function
        | Ok got ->
            equal (option int) ~msg:"the last stored parameter" m.stored
              (Some got)
        | Error (Invalid_argument _) when m.stored = None -> ()
        | Error e -> raise e)
      (fun t -> fill_with t.sub t.args);
  ]

(* A submission of [n] launches of [fill] on the domain tests' device. *)
let fills_on n =
  let d, image = Lazy.force pool_device in
  let fill = launch image "fill" ~params:16 ~refs:[| Sub.write 0 ~at:0 |] in
  Sub.make d (Array.make n fill)

(* Submits [s], [n] launches of [fills_on], launch [i] storing [v + i] at word
   [i], and reads the words back. *)
let fills_with s n v =
  let d, _ = Lazy.force pool_device in
  let out = B.create d (8 * n) in
  let args =
    Array.init n (fun i -> single ~params:(words [ 8 * i; v + i ]) ())
  in
  Rig.Point.wait (submit ~buffers:[| out |] ~args s);
  let b = host_bytes out (8 * n) in
  List.init n (fun i -> Int64.to_int (String.get_int64_le b (8 * i)))

(* "each block 32 bytes and twice its parameters rounded up to a multiple of
   16": the bytes of the blocks of [n] launches of [fill], which takes 16. *)
let fill_blocks n = n * (32 + (2 * 16))

(* A free_cache of the host frees the state rig keeps for later submits, and a
   submit after it runs. *)
let test_state_freed () =
  let n = 256 in
  let s = fills_on n in
  let from v = List.init n (( + ) v) in
  equal (list int) ~msg:"a submit before" (from 1) (fills_with s n 1);
  at_least int ~msg:"the state kept after it" ~than:(fill_blocks n)
    (Rig.submit_state ());
  Rig.free_cache Rig.host;
  equal int ~msg:"the state kept after free_cache" 0 (Rig.submit_state ());
  equal (list int) ~msg:"a submit after it" (from 7) (fills_with s n 7)

(* "as many sets as submits ... were ever under way at once, each as large as
   the largest submission it served": submits one at a time keep one set. Once
   it served the large submission, the small one's submit takes it and adds
   nothing; a set per submission would add the small one's. *)
let test_state_sets () =
  Rig.free_cache Rig.host;
  let small = fills_on 1 and large = fills_on 64 in
  ignore (fills_with small 1 1);
  let small_alone = Rig.submit_state () in
  Rig.free_cache Rig.host;
  ignore (fills_with large 64 1);
  let large_alone = Rig.submit_state () in
  ignore (fills_with small 1 1);
  equal ~msg:"the large one, then the small one" int large_alone
    (Rig.submit_state ());
  at_least ~msg:"the large one alone" int ~than:(fill_blocks 64) large_alone;
  less ~msg:"the small one alone" int ~than:large_alone small_alone;
  Rig.free_cache Rig.host

(* A free_cache of the host while a submit of another domain waits in its
   device's room, its values copied into its state, leaves that state: the
   driver reads the values the submit was given. *)
let test_state_held () =
  let d, p = P.open_ ~may_block:true (fresh "state-held") in
  let s = Sub.make d [| launch (functions d) "main" ~params:64 |] in
  let params = words (List.init 8 (( + ) 100)) in
  ignore (P.launches p);
  P.gate_room p;
  let held = Domain.spawn (fun () -> submit ~args:[| single ~params () |] s) in
  Rig_support.await "a submit in its room" (fun () -> P.in_room p = 1);
  Rig.free_cache Rig.host;
  P.open_room p;
  Rig.Point.wait (Domain.join held);
  let given =
    {
      P.groups = (1, 1, 1);
      threads = (1, 1, 1);
      shared = 0;
      params = Bytes.to_string params;
    }
  in
  equal (list launch_t) ~msg:"the launch its driver read" [ given ]
    (P.launches p)

(* Two domains submit submissions of 1 to 64 launches while either frees the
   host's cache, which may take a set of state as a submit would: each submit
   reads back its own values. *)
let fills_t = abstract ~pp:Format.pp_print_int "s"

let freeing_commands =
  [
    command "make"
      (Gen.int_range 1 64 @-> makes fills_t)
      Fun.id
      (fun n -> (fills_on n, n));
    command "submit"
      (fills_t ^-> Gen.int_range 1 1000 @-> returns (list int))
      (fun n v -> List.init n (( + ) v))
      (fun (s, n) v -> fills_with s n v);
    command "free_cache"
      (Gen.unit @-> returns unit)
      Fun.id
      (fun () -> Rig.free_cache Rig.host);
  ]

(* Two domains submitting one submission: the submits take turns, each takes the
   device's next value, and every fill runs. A program counts values from
   [base], its device's value when it began. *)
type shared = {
  dev : Rig.t;
  base : int;
  arg : B.t;
  buffers : B.t array;
  sub : Sub.t;
}

type shared_model = { mutable values : int }

(* Devices go back to a pool when their program ends. *)
let shared_pool = Mutex.create ()
let shared_free = ref []
let shared_opened = Atomic.make 0

let make_shared () =
  let d =
    match
      Mutex.protect shared_pool (fun () ->
          match !shared_free with
          | d :: rest ->
              shared_free := rest;
              Some d
          | [] -> None)
    with
    | Some d -> d
    | None ->
        memory
          (Printf.sprintf "submit:shared-%d"
             (Atomic.fetch_and_add shared_opened 1))
  in
  let arg = B.create Rig.host 8 in
  Support.store (Support.Reader.host arg) 0;
  {
    dev = d;
    base = Rig.submitted d;
    arg;
    buffers = [| on d arg |];
    sub = Sub.make d [| bump; bump |];
  }

(* Every submit ran both its fills. *)
let release_shared t =
  equal ~msg:"fills" int (2 * (Rig.submitted t.dev - t.base)) (word t.arg);
  Mutex.protect shared_pool (fun () -> shared_free := t.dev :: !shared_free)

let shared =
  abstract
    ~pp:(fun ppf r -> Format.fprintf ppf "values %d" r.values)
    ~release:release_shared "s"

let judge_submit r = function
  | Ok v ->
      equal ~msg:"value" int (r.values + 1) v;
      r.values <- v
  | Error e -> raise e

let shared_commands =
  [
    command "make"
      (Gen.unit @-> makes shared)
      (fun () -> { values = 0 })
      make_shared;
    command "submit"
      (shared ^-> judges int)
      judge_submit
      (fun t ->
        let args = Support.zeros 2 in
        Rig.Point.value (submit ~args ~buffers:t.buffers t.sub) - t.base);
  ]

(* Two domains copying on one device at once, each between buffers of its own:
   through one submission whose copy's ends are its two slots, and through
   Buffer.copy, which shares what it prepares for the device between domains.
   Each copy's destination ends with its source's word. The host addresses no
   memory of the device, so its copy engine runs each copy between two of its
   buffers. *)
let slot_copier =
  let copier =
    lazy
      (let d, _ = P.open_ ~host_visible:false "submit:slot-copier" in
       (d, Sub.make d Sub.[| copy ~src:(Slot 0) ~dst:(Slot 1) |]))
  in
  (* Two domains may make the first call at once, and a lazy value forced from
     two domains at once raises [CamlinternalLazy.Undefined]. *)
  let lock = Mutex.create () in
  fun () -> Mutex.protect lock (fun () -> Lazy.force copier)

(* A buffer of [d] holding [v], and one to copy it into. *)
let copy_ends d v =
  let src = B.create d 8 and dst = B.create d 8 in
  let h = B.create Rig.host 8 in
  Support.store (Support.Reader.host h) v;
  B.copy ~src:h ~dst:src;
  (src, dst)

(* The word of [b], a buffer of a device. *)
let device_word b =
  let h = B.create Rig.host 8 in
  B.copy ~src:b ~dst:h;
  word h

let copy_commands =
  [
    command "submit"
      (Gen.int @-> returns int)
      Fun.id
      (fun v ->
        let d, s = slot_copier () in
        let src, dst = copy_ends d v in
        Rig.Point.wait (Rig.submit s ~buffers:[| src; dst |] ~args:[||]);
        device_word dst);
    command "Buffer.copy"
      (Gen.int @-> returns int)
      Fun.id
      (fun v ->
        let d, _ = slot_copier () in
        let src, dst = copy_ends d v in
        B.copy ~src ~dst;
        device_word dst);
  ]

(* While a submit of a submission waits on the host for a producer, a submit of
   the same submission on another domain, with its own values, completes. *)
let test_run_beside_wait () =
  let producer, pp = P.open_ "submit:beside-producer" in
  let d, p = P.open_ "submit:beside" in
  let base = Rig.submitted d in
  let s = Sub.make d [| launch (functions d) "main" ~params:8 |] in
  let point = submit (empty producer) in
  let values v = [| single ~params:(words [ v ]) () |] in
  let first = values 41 and second = values 42 in
  P.gate pp;
  let waiting =
    Domain.spawn (fun () -> submit s ~args:first ~waits:[| point |])
  in
  let finished = ref None in
  Fun.protect
    ~finally:(fun () ->
      P.open_gate pp;
      finished := Some (Domain.join waiting))
    (fun () ->
      Support.await "a sleep at the gate" (fun () -> P.sleepers pp = 1);
      equal ~msg:"the submit beside the wait" int (base + 1)
        (Rig.Point.value (submit s ~args:second)));
  let finished = Option.get !finished in
  equal ~msg:"the waiting submit" int (base + 2) (Rig.Point.value finished);
  Rig.Point.wait finished;
  equal ~msg:"each launch reads its submit's parameter" (list int) [ 42; 41 ]
    (first_words (P.launches p))

(* A submit made inside another's wait of the same submission, as a signal
   handler runs, completes with its own values: each launch reads its submit's
   parameter. *)
let test_run_reentrant () =
  let producer, pp = P.open_ "submit:reentrant-producer" in
  let d, p = P.open_ "submit:reentrant" in
  let base = Rig.submitted d in
  let s = Sub.make d [| launch (functions d) "main" ~params:8 |] in
  let point = submit (empty producer) in
  let values v = [| single ~params:(words [ v ]) () |] in
  let outer_args = values 41 and inner_args = values 42 in
  let inner = ref None in
  let handle _ = inner := Some (Rig.Point.value (submit s ~args:inner_args)) in
  P.interrupt pp;
  let before = Sys.signal Sys.sigint (Sys.Signal_handle handle) in
  let outer =
    Fun.protect
      ~finally:(fun () -> Sys.set_signal Sys.sigint before)
      (fun () -> submit s ~args:outer_args ~waits:[| point |])
  in
  equal ~msg:"the submit inside the wait" (option int) (Some (base + 1)) !inner;
  equal ~msg:"the waiting submit" int (base + 2) (Rig.Point.value outer);
  Rig.Point.wait outer;
  equal ~msg:"each launch reads its submit's parameter" (list int) [ 42; 41 ]
    (first_words (P.launches p))

(* Values rewritten from a signal handler while a submit of them waits for a
   producer: the submit completes, and its launch reads each byte as the values
   held it before or after. *)
let test_rewritten_values () =
  let producer, pp = P.open_ "submit:rewritten-producer" in
  let d, p = P.open_ "submit:rewritten" in
  let s = Sub.make d [| launch (functions d) "main" ~params:8 |] in
  let params = Bytes.make 8 'a' in
  let args = [| single ~params () |] in
  let point = submit (empty producer) in
  let rewrites = ref 0 in
  let handle _ =
    Bytes.fill params 0 8 'b';
    incr rewrites
  in
  ignore (P.launches p);
  P.interrupt pp;
  let before = Sys.signal Sys.sigint (Sys.Signal_handle handle) in
  let v =
    Fun.protect
      ~finally:(fun () -> Sys.set_signal Sys.sigint before)
      (fun () -> submit ~waits:[| point |] ~args s)
  in
  equal ~msg:"the handler's runs" int 1 !rewrites;
  Rig.Point.wait v;
  let neither l = String.exists (fun c -> c <> 'a' && c <> 'b') l in
  equal ~msg:"launches reading a byte neither before nor after" (list string) []
    (List.filter neither (params_of (P.launches p)))

(* A linked program's run [n] uses copy [n mod 2]: it writes run [n]'s arguments
   into the copy's values, which run [n - 2] passed, and submits the copy's
   prepared submission. Its fill stores [n] where the arguments say, into run
   [n]'s output. The host goes on to the next run without waiting for the
   output. *)
type replay = {
  d : Rig.t;
  p : P.t;
  subs : Sub.t array; (* Each copy's submission. *)
  values : Sub.args array array; (* Each copy's values. *)
  outs : (int, B.t) Hashtbl.t;
  points : (int, int) Hashtbl.t;
  mutable runs : int;
}

let poke = Sub.fill ~refs:[| Sub.write 0 ~at:0 |] Support.poke ~params:16
let replays = Atomic.make 0

let open_replay () =
  let name =
    Printf.sprintf "submit:replay-%d" (Atomic.fetch_and_add replays 1)
  in
  let d, p = P.open_ name in
  let values _ = [| { Support.zero with params = Bytes.make 16 '\000' } |] in
  {
    d;
    p;
    subs = Array.init 2 (fun _ -> Sub.make d [| poke |]);
    values = Array.init 2 values;
    outs = Hashtbl.create 8;
    points = Hashtbl.create 8;
    runs = 0;
  }

let run_replay t =
  let n = t.runs in
  let k = n mod 2 in
  cover "a copy reused before its last run ran"
    (n >= 2 && Rig.signaled t.d < Hashtbl.find t.points (n - 2));
  let out = B.create t.d 8 in
  B.wait out Rig.Read_write;
  Support.store (Support.Reader.host out) (-1);
  let args = t.values.(k) in
  Bytes.set_int64_le args.(0).params 0 0L;
  Bytes.set_int64_le args.(0).params 8 (Int64.of_int n);
  let v = Rig.Point.value (Rig.submit t.subs.(k) ~buffers:[| out |] ~args) in
  Hashtbl.replace t.outs n out;
  Hashtbl.replace t.points n v;
  t.runs <- n + 1

let output t n =
  let out = Hashtbl.find t.outs n in
  B.wait out Rig.Read;
  Support.load (Support.Reader.host out)

(* A program's device closes when it ends, so no later collection walks it. *)
let replay = abstract ~release:(fun t -> Rig.close t.d) "r"
let run_index = among int replay (fun runs -> List.init !runs Fun.id)
let run_command = command "run" (replay ^-> returns unit) incr run_replay

(* Runs are drawn twice as often as the rest, so the host runs ahead. *)
let replay_commands =
  [
    command "open" (Gen.unit @-> makes replay) (fun () -> ref 0) open_replay;
    run_command;
    run_command;
    command "device runs"
      (replay ^-> returns unit)
      (fun _ -> ())
      (fun t -> ignore (P.run t.p));
    command "output"
      (replay ^-> run_index ^-> returns int)
      (fun _ n -> n)
      output;
  ]

(* Growth *)

(* A device whose queue holds 4 parts and grows. *)
let growing name = P.open_ ~capacity:4 ~grows:max_int name

(* A submission of [n] parts that each add 1 to the host word [arg] borrows on
   [d], with its values and its buffers. *)
let bumps d arg n =
  (Sub.make d (Array.make n bump), Support.zeros n, [| on d arg |])

(* A submission of more parts than the queue holds runs, as one value, once the
   queue grew to hold it; one the queue holds needs no growth. *)
let test_grows n =
  let d, p = growing (Printf.sprintf "submit:grows-%d" n) in
  let base = Rig.submitted d in
  let arg = B.create Rig.host page_bytes in
  Support.store (Support.Reader.host arg) 0;
  let s, args, buffers = bumps d arg n in
  let pt = submit ~args ~buffers s in
  Rig.Point.wait pt;
  equal ~msg:"its value" int (base + 1) (Rig.Point.value pt);
  equal ~msg:"the parts that ran" int n (word arg);
  equal ~msg:"growths" int (if n > 4 then 1 else 0) (count "grow" p)

(* A growth the device has no memory for is asked again after each round of
   reclamation, and raises Out_of_memory after the last, having assigned no
   value; once memory returns, the submission runs. *)
let test_grow_no_memory () =
  let d, p = growing "submit:grow-no-memory" in
  let base = Rig.submitted d in
  let arg = B.create Rig.host page_bytes in
  Support.store (Support.Reader.host arg) 0;
  let s, args, buffers = bumps d arg 5 in
  P.fail_at p 1 (`Refuse max_int);
  raises_match
    (function
      | Rig.Out_of_memory (d', n) -> Rig.equal d d' && n = 4096 | _ -> false)
    (fun () -> submit ~args ~buffers s);
  P.fail_at p 1 (`Refuse 0);
  at_least ~msg:"times asked" int ~than:2 (count "grow" p);
  equal ~msg:"values assigned" int base (Rig.submitted d);
  Rig.Point.wait (submit ~args ~buffers s);
  equal ~msg:"the parts that ran" int 5 (word arg)

(* A growth the device had no memory for at first is made once a round of
   reclamation is done. *)
let test_grow_reclaims () =
  let d, p = growing "submit:grow-reclaims" in
  let arg = B.create Rig.host page_bytes in
  Support.store (Support.Reader.host arg) 0;
  let s, args, buffers = bumps d arg 5 in
  P.before p "grow" (fun () -> P.fail_at p 1 (`Refuse 1));
  Rig.Point.wait (submit ~args ~buffers s);
  equal ~msg:"times asked" int 2 (count "grow" p);
  equal ~msg:"the parts that ran" int 5 (word arg)

(* A growth past the memory the device has raises Out_of_memory and leaves no
   want behind: a smaller growth after it is made at its first ask. *)
let test_grow_past_memory () =
  let d, p = P.open_ ~capacity:4 ~grows:8 "submit:grow-past-memory" in
  let base = Rig.submitted d in
  let arg = B.create Rig.host page_bytes in
  Support.store (Support.Reader.host arg) 0;
  let large, args, buffers = bumps d arg 9 in
  raises_match
    (function
      | Rig.Out_of_memory (d', n) -> Rig.equal d d' && n = 4096 | _ -> false)
    (fun () -> submit ~args ~buffers large);
  let asked = count "grow" p in
  at_least ~msg:"times asked" int ~than:2 asked;
  equal ~msg:"values assigned" int base (Rig.submitted d);
  let small, args, buffers = bumps d arg 5 in
  Rig.Point.wait (submit ~args ~buffers small);
  equal ~msg:"times asked after" int (asked + 1) (count "grow" p);
  equal ~msg:"the parts that ran" int 5 (word arg)

(* A growth short of memory for one submission fails no other: while domain A's
   growth past the memory runs, domain B's smaller growth is made at its first
   ask and B's submission runs; A raises Out_of_memory. *)
let test_grow_short_other () =
  let d, p = P.open_ ~capacity:4 ~grows:8 "submit:grow-short-other" in
  let base = Rig.submitted d in
  let arg = B.create Rig.host page_bytes in
  Support.store (Support.Reader.host arg) 0;
  let large, args, buffers = bumps d arg 9 in
  let small, small_args, small_buffers = bumps d arg 5 in
  let other = ref None and asked = ref 0 in
  P.before p "grow" (fun () ->
      let submits () =
        Rig.Point.value (submit ~args:small_args ~buffers:small_buffers small)
      in
      other := Some (Domain.join (Domain.spawn submits));
      asked := count "grow" p);
  raises_match
    (function
      | Rig.Out_of_memory (d', n) -> Rig.equal d d' && n = 4096 | _ -> false)
    (fun () -> submit ~args ~buffers large);
  equal ~msg:"the other domain's value" (option int) (Some (base + 1)) !other;
  equal ~msg:"growths asked until B's ran" int 2 !asked;
  B.wait arg Rig.Read;
  equal ~msg:"the parts that ran" int 5 (word arg)

(* While a growth runs, holding no turn, a submission of another domain that
   needs a growth too takes the device's next value; the first submission takes
   the one after, and every part runs once. *)
let test_grow_interleaved () =
  let d, p = growing "submit:grow-interleaved" in
  let base = Rig.submitted d in
  let arg = B.create Rig.host page_bytes in
  Support.store (Support.Reader.host arg) 0;
  let first, args, buffers = bumps d arg 9 in
  let second, other_args, other_buffers = bumps d arg 5 in
  let other = ref None in
  P.before p "grow" (fun () ->
      let submits () =
        Rig.Point.value (submit ~args:other_args ~buffers:other_buffers second)
      in
      other := Some (Domain.join (Domain.spawn submits)));
  let v = Rig.Point.value (submit ~args ~buffers first) in
  equal ~msg:"the other domain's value" (option int) (Some (base + 1)) !other;
  equal ~msg:"its value" int (base + 2) v;
  B.wait arg Rig.Read;
  equal ~msg:"the parts that ran" int 14 (word arg);
  equal ~msg:"growths" int 2 (count "grow" p)

(* A free_cache of the device gives back what its queue grew into: the next
   submission the queue held only once grown grows it again. *)
let test_grow_trimmed () =
  let d, p = growing "submit:grow-trimmed" in
  let arg = B.create Rig.host page_bytes in
  Support.store (Support.Reader.host arg) 0;
  let s, args, buffers = bumps d arg 5 in
  Rig.Point.wait (submit ~args ~buffers s);
  Rig.free_cache d;
  equal ~msg:"trims" int 1 (P.trims p);
  Rig.Point.wait (submit ~args ~buffers s);
  equal ~msg:"growths" int 2 (count "grow" p);
  equal ~msg:"the parts that ran" int 10 (word arg)

(* A free_cache of another domain while a submit grows the queue trims nothing,
   so the growth stays for that submit's room; one after the submit trims. *)
let test_grow_untrimmed () =
  let d, p = growing "submit:grow-untrimmed" in
  let arg = B.create Rig.host page_bytes in
  Support.store (Support.Reader.host arg) 0;
  let s, args, buffers = bumps d arg 5 in
  let during = ref (-1) in
  P.before p "grow" (fun () ->
      Domain.join (Domain.spawn (fun () -> Rig.free_cache d));
      during := P.trims p);
  Rig.Point.wait (submit ~args ~buffers s);
  equal ~msg:"trims while it grew" int 0 !during;
  equal ~msg:"growths" int 1 (count "grow" p);
  Rig.free_cache d;
  equal ~msg:"trims after" int 1 (P.trims p);
  equal ~msg:"the parts that ran" int 5 (word arg)

(* A free_cache shrinks a device's record of queue waits to the waits of its
   unreached work, whether or not its driver has a trim: [finished] waits whose
   work ran, then [live] waits on a producer's unreached value. The record still
   spreads a loss along what it kept: the producer's loss loses the consumer
   once a live wait, or one submitted [later], after the free_cache, names that
   value. *)
let record_trimmed (trim, finished, live, later) =
  let name = fresh "record-producer" in
  let producer, pp = P.open_ name in
  let consumer, _ =
    P.open_ ~trim ~waits_on:[ `Host ] (fresh "record-consumer")
  in
  let waits n a =
    List.fold_left
      (fun _ _ -> Some (submit (empty consumer) ~waits:[| a |]))
      None (List.init n Fun.id)
  in
  let a = submit (empty producer) in
  Option.iter Rig.Point.wait (waits finished a);
  let b = submit (empty producer) in
  ignore (waits live b : Rig.Point.t option);
  let room = Rig.wait_room consumer in
  cover "a driver with no trim" (not trim);
  cover "finished waits alone" (finished > 0 && live = 0);
  cover "room past the live waits" (live > 0 && room > live);
  cover "a wait after an empty record" (finished > 0 && live = 0 && later);
  Rig.free_cache consumer;
  at_most int ~msg:"the record's room" ~than:live (Rig.wait_room consumer);
  if later then ignore (waits 1 b : Rig.Point.t option);
  P.fail pp;
  raises_match (lost_on producer) (fun () -> submit (empty producer));
  equal (option string) ~msg:"the consumer's loss"
    (if live > 0 || later then Some (name ^ " lost") else None)
    (Rig.lost consumer);
  Rig.close consumer;
  Rig.close producer

(* Each covered case of [record_trimmed] draws in a sixth of the cases or
   more. *)
let record_cases =
  let waits ~none =
    Gen.frequency [ (none, Gen.constant 0); (3, Gen.int_range 1 20) ]
  in
  let pp ppf (trim, finished, live, later) =
    Format.fprintf ppf "trim %b, finished %d, live %d, later %b" trim finished
      live later
  in
  Gen.with_pp pp (Gen.quad Gen.bool (waits ~none:1) (waits ~none:3) Gen.bool)

(* A free_cache of another domain while a submit grows the queue leaves the
   record of queue waits as it was; one after the submit shrinks it. *)
let test_record_untrimmed () =
  let producer, _ = P.open_ "submit:record-untrimmed-producer" in
  let d, p =
    P.open_ ~capacity:4 ~grows:max_int ~waits_on:[ `Host ]
      "submit:record-untrimmed"
  in
  Rig.Point.wait (submit (empty d) ~waits:[| submit (empty producer) |]);
  let room = Rig.wait_room d in
  greater ~msg:"the record's room before it grew" int ~than:0 room;
  let during = ref (-1) in
  P.before p "grow" (fun () ->
      Domain.join (Domain.spawn (fun () -> Rig.free_cache d));
      during := Rig.wait_room d);
  let arg = B.create Rig.host page_bytes in
  Support.store (Support.Reader.host arg) 0;
  let s, args, buffers = bumps d arg 5 in
  Rig.Point.wait (submit ~args ~buffers s);
  equal ~msg:"the record's room while it grew" int room !during;
  Rig.free_cache d;
  equal ~msg:"the record's room after" int 0 (Rig.wait_room d);
  equal ~msg:"the parts that ran" int 5 (word arg)

(* A growth that faults loses the device, and the submission assigns no
   value. *)
let test_grow_fault () =
  let d, p = growing "submit:grow-fault" in
  let base = Rig.submitted d in
  let arg = B.create Rig.host page_bytes in
  let s, args, buffers = bumps d arg 5 in
  P.before p "grow" (fun () -> P.fail_at p 1 (`Fault "grow fault"));
  raises_match lost (fun () -> submit ~args ~buffers s);
  equal ~msg:"values assigned" int base (Rig.submitted d);
  equal (option string) ~msg:"the loss" (Some "grow fault") (Rig.lost d)

(* A retire that faults at a free_cache loses the device with the fault's text,
   and the free_cache returns. *)
let test_retire_fault () =
  let d, p = P.open_ "submit:retire-fault" in
  P.before p "retire" (fun () -> P.fail_at p 1 (`Fault "retire fault"));
  Rig.free_cache d;
  equal (option string) ~msg:"the loss" (Some "retire fault") (Rig.lost d)

(* Another exception that a retire raises loses the device and passes out of the
   free_cache. *)
let test_retire_raises () =
  let d, p = P.open_ "submit:retire-raises" in
  P.before p "retire" (fun () -> raise Exit);
  raises ~msg:"the free_cache" Exit (fun () -> Rig.free_cache d);
  equal (option string) ~msg:"the loss" (Some "Stdlib.Exit") (Rig.lost d)

(* A lost device's free_cache calls no retire. *)
let test_retire_lost () =
  let d, p = P.open_ "submit:retire-lost" in
  P.before p "alloc" (fun () -> P.fail_at p 1 (`Fault "alloc fault"));
  raises_match lost (fun () -> B.create d 8);
  Rig.free_cache d;
  equal ~msg:"retires" int 0 (count "retire" p)

(* A child process limits its address space to [room] bytes above what it maps,
   and submits a submission whose blocks span more than 256 MiB: the submit
   raises Out_of_memory. With the limit lifted, a submit of another submission
   runs. On Linux only: the limit bounds no allocation elsewhere. *)
let room = 64 lsl 20
let spanning = 1 lsl 16

let exe =
  let e = Sys.executable_name in
  if Filename.is_relative e then Filename.concat (Sys.getcwd ()) e else e

(* The bytes the process maps: the [VmSize] of its status. *)
let mapped () =
  let field l = Scanf.sscanf_opt l "VmSize: %d kB" (fun k -> k * 1024) in
  let lines =
    In_channel.with_open_text "/proc/self/status" In_channel.input_lines
  in
  match List.find_map field lines with
  | Some n -> n
  | None -> failwith "no VmSize in /proc/self/status"

let limit n =
  match Rig_support.set_address_space n with
  | 0 -> ()
  | e -> failwith (strf "setrlimit: errno %d" e)

let grow_child () =
  let d, _ = P.open_ "submit:grow-child" in
  let image = functions d in
  let wide = launch image "main" ~params:4096 in
  let big = Sub.make d (Array.make spanning wide) in
  let big_args = Array.make spanning (single ()) in
  let small =
    Sub.make d
      [| launch image "fill" ~params:16 ~refs:[| Sub.write 0 ~at:0 |] |]
  in
  limit (mapped () + room);
  let refused =
    match submit big ~args:big_args with
    | _ -> "submitted"
    | exception Stdlib.Out_of_memory -> "Out_of_memory"
  in
  limit max_int;
  let out = B.create d 8 in
  Rig.Point.wait (submit ~buffers:[| out |] ~args:[| fill_values 41 |] small);
  Printf.printf "%s\n%Ld\n" refused (String.get_int64_le (host_bytes out 8) 0)

let test_grow_refused () =
  if not (Sys.file_exists "/proc/self/status") then
    skip ~reason:"the limit bounds no allocation off Linux" ();
  let ic = Unix.open_process_args_in exe [| exe; "grow" |] in
  let out = In_channel.input_all ic in
  (match Unix.close_process_in ic with
  | WEXITED 0 -> ()
  | _ -> failf "the child failed: %s" out);
  equal (list string) [ "Out_of_memory"; "41" ]
    (String.split_on_char '\n' (String.trim out))

(* Cost *)

(* A submit that waits for nothing allocates nothing, over a slot, values, a
   copy's slots and a copy from a buffer. *)
let test_allocation () =
  let d = memory "submit:words" in
  let bytes_of_submits s buffers args =
    ignore (Rig.submit s ~buffers ~args);
    Support.allocated_bytes (fun () ->
        for _ = 1 to 100 do
          ignore (Sys.opaque_identity (Rig.submit s ~buffers ~args))
        done)
  in
  let fixed = B.create d 8 in
  let values = [| { Support.zero with params = Bytes.make 16 'v' } |] in
  List.iter
    (fun (msg, s, buffers, args) ->
      equal ~msg int 0 (bytes_of_submits s buffers args))
    [
      ( "a fill's slot",
        empty ~uses:[| Sub.read |] d,
        [| on d (B.create Rig.host 8) |],
        one );
      ( "a fill's values",
        Sub.make d [| Sub.fill Support.noop ~params:16 |],
        [||],
        values );
      ( "a copy's slots",
        Sub.make d Sub.[| copy ~src:(Slot 0) ~dst:(Slot 1) |],
        [| B.create d 8; B.create d 8 |],
        [||] );
      ( "a copy from a buffer into a slot",
        Sub.make d Sub.[| copy ~src:(Buffer fixed) ~dst:(Slot 0) |],
        [| B.create d 8 |],
        [||] );
    ]

(* A launch's submit allocates nothing, nor does one after its values were
   rewritten in place. *)
let test_launch_allocation () =
  let d, _ = P.open_ "submit:launch-words" in
  let image = functions d in
  let refs =
    Array.init 8 (fun k ->
        if k < 4 then Sub.read k ~at:(8 * k) else Sub.write k ~at:(8 * k))
  in
  let s = Sub.make d [| launch image "main" ~params:72 ~refs |] in
  let buffers = Array.init 8 (fun _ -> B.create d 8) in
  let params = Bytes.make 72 '\000' in
  let args = [| single ~params () |] in
  let first = submit ~buffers ~args s in
  let n = 100 in
  let allocates msg f =
    equal int ~msg 0
      (Rig_support.allocated_bytes (fun () ->
           for i = 1 to n do
             f i
           done))
  in
  allocates "a submit of a launch" (fun _ ->
      ignore (Sys.opaque_identity (Rig.submit s ~buffers ~args)));
  allocates "a rewrite of its values in place and a submit" (fun i ->
      Bytes.set_int64_le params 64 (Int64.of_int i);
      ignore (Sys.opaque_identity (Rig.submit s ~buffers ~args)));
  let last = submit ~buffers ~args s in
  Rig.Point.wait first;
  Rig.Point.wait last

let tests =
  [
    group "waits"
      [
        test "a queue that waits on host words waits in the queue" test_in_queue;
        test "a queue that waits on stores waits on the producer's word"
          test_in_queue_store;
        test "a producer whose word the device cannot map is waited on the host"
          test_unmapped_wait;
        test "a room check is given the waits its submit is given"
          test_room_waits;
        test "a submit keeps its buffers while it waits on the host"
          test_run_keeps;
      ];
    group "room"
      [
        test "a full queue's submit waits for room" test_room;
        test "a submit waiting for room lets a submission that fits through"
          test_later_releases;
        test "a fault while a submit waits for room commits nothing"
          test_fault_waiting;
        test "a submission runs if its queues hold it, and raises if never"
          test_never_fits;
      ];
    group "submit"
      [
        test "each submit's value is one more than the last" test_values;
        test "each submit runs every fill of its submission" test_fill;
        stateful ~domains:2
          "two domains' submits of one submission each read their own values"
          own_commands;
        stateful ~domains:2
          "submits of one submission from two domains take the device's next \
           values"
          shared_commands;
        stateful ~steps:0 ~domains:2
          "two domains copy at once through one submission, each between \
           buffers of its own"
          copy_commands;
        test "a submit beside another's wait completes with its own values"
          test_run_beside_wait;
        test "a submit inside another's wait completes with its own values"
          test_run_reentrant;
        stateful ~domains:2
          "a submit reads the element a store left in its values, or refuses it"
          passing_commands;
        test "values rewritten during their submit give it some bytes of each"
          test_rewritten_values;
        stateful ~steps:12
          "values rewritten for a later submit change nothing of earlier work \
           still queued"
          replay_commands;
        test "a free_cache of the host frees the state kept for submits"
          test_state_freed;
        test
          "submits one at a time keep one set of state, as large as the \
           largest submission"
          test_state_sets;
        test
          "a free_cache of the host while a submit is in its room leaves its \
           values"
          test_state_held;
        stateful ~count:30 ~domains:2
          "submits on two domains read their own values while the host's cache \
           is freed"
          freeing_commands;
        cases "a submission runs once its queue grew to hold it"
          ~id:(Printf.sprintf "%d parts on a queue of 4 that grows")
          [ 1; 4; 5; 9; 300 ] test_grows;
        test "a growth without memory raises Out_of_memory after reclamation"
          test_grow_no_memory;
        test "a growth short at first is made after a round of reclamation"
          test_grow_reclaims;
        test
          "after a growth raises Out_of_memory, a smaller one is made at its \
           first ask"
          test_grow_past_memory;
        test "a submission of another domain runs while a growth runs"
          test_grow_interleaved;
        test "a growth short for one submission fails no other"
          test_grow_short_other;
        test "a growth that faults loses the device" test_grow_fault;
        test "a submit that does not wait allocates nothing" test_allocation;
        test "a launch's submit and a rewrite of its values allocate nothing"
          test_launch_allocation;
        test "a submit its driver cannot run assigns no value" test_never;
        test "a submit that raises keeps nothing" test_cleared_on_raise;
        test "a submit whose blocks the host has no memory for raises"
          test_grow_refused;
      ];
    group "free_cache"
      [
        test "free_cache gives back what the queue grew into" test_grow_trimmed;
        test "a free_cache while a submit grows the queue leaves its growth"
          test_grow_untrimmed;
        prop
          ~examples:[ (false, 1, 0, true) ]
          "free_cache shrinks the record of queue waits to the live ones, \
           which a loss still spreads along"
          record_cases record_trimmed;
        test "a free_cache while a submit grows leaves the record of waits"
          test_record_untrimmed;
        test "a retire that faults at free_cache loses the device"
          test_retire_fault;
        test "a retire's other exception loses the device and passes out"
          test_retire_raises;
        test "a lost device's free_cache calls no retire" test_retire_lost;
      ];
  ]

let () =
  match Sys.argv with
  | [| _; "grow" |] -> grow_child ()
  | _ -> exit (run "rig.submit" tests)
