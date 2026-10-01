open Windtrap
open Tolk
open Helpers

(* The host's target, as the engine gives it. *)
let host_target = Tolk_engine.target Nx_device.host

exception Failed of int

let with_parallel p f = context [ B (parallel, p) ] f
let domain () = (Domain.self () :> int)

(* A call that never returns fails its test instead of the run. *)
let group name tests = group ~timeout:30. name tests

(* Synchronisation *)

(* [await what cond] returns once [cond ()] holds, and fails, naming [what],
   after ten seconds. *)
let await what cond =
  let deadline = Unix.gettimeofday () +. 10. in
  while not (cond ()) do
    if Unix.gettimeofday () > deadline then
      failf "waited ten seconds for %s" what;
    Unix.sleepf 0.0002
  done

(* [meeting n] returns a function that returns once [n] calls have reached it:
   applications that call it all run at once, on [n] domains. *)
let meeting n =
  let arrived = Atomic.make 0 in
  fun () ->
    Atomic.incr arrived;
    await (Printf.sprintf "%d applications at once" n) (fun () ->
        Atomic.get arrived >= n)

(* [spawn_all fs] runs each of [fs] on a domain of its own, all at once, and is
   their results. *)
let spawn_all fs =
  let meet = meeting (List.length fs) in
  List.map
    (fun f ->
      Domain.spawn (fun () ->
          meet ();
          f ()))
    fs
  |> List.map Domain.join

(* The number of applications running at once, and its peak. *)
type gauge = { live : int Atomic.t; peak : int Atomic.t }

let gauge () = { live = Atomic.make 0; peak = Atomic.make 0 }

let rec raise_peak g n =
  let seen = Atomic.get g.peak in
  if n > seen && not (Atomic.compare_and_set g.peak seen n) then raise_peak g n

let within g f =
  raise_peak g (1 + Atomic.fetch_and_add g.live 1);
  Fun.protect ~finally:(fun () -> Atomic.decr g.live) f

(* The law *)

(* A function to map: [x] goes to [a * x + b], written, after a delay that
   varies with [x], so that applications end out of order. *)
type fn = { a : int; b : int; spins : int }

let pp_fn ppf { a; b; spins } =
  Format.fprintf ppf "fun x -> %d * x + %d, after %d spins per x mod 8" a b
    spins

let fn =
  Gen.with_pp pp_fn
    Gen.(
      let+ a = small_int and+ b = small_int and+ spins = int_range 0 200 in
      { a; b; spins })

let apply fn x =
  for _ = 1 to x land 7 * fn.spins do
    Domain.cpu_relax ()
  done;
  string_of_int ((fn.a * x) + fn.b)

let parallels = Gen.of_list ~pp:Format.pp_print_int [ -1; 0; 1; 2; 3; 8 ]

let is_list_map (p, fn, l) =
  cover "several domains may apply f" (p > 1 && List.length l > 1);
  equal (list string)
    (List.map (apply fn) l)
    (with_parallel p (fun () -> Worker.map (apply fn) l))

let applies_each_element_once p () =
  let n = 200 in
  let applied = Array.init n (fun _ -> Atomic.make 0) in
  ignore
    (with_parallel p (fun () ->
         Worker.map (fun i -> Atomic.incr applied.(i)) (List.init n Fun.id)));
  Array.iteri
    (fun i count ->
      equal ~msg:(Printf.sprintf "element %d" i) int 1 (Atomic.get count))
    applied

let keeps_order_of_late_finishers () =
  let l = List.init 8 Fun.id in
  let f i =
    Unix.sleepf (0.002 *. Float.of_int (8 - i));
    i * 10
  in
  equal (list int) (List.map f l) (with_parallel 4 (fun () -> Worker.map f l))

let law =
  group "law"
    [
      prop "map f l is List.map f l"
        ~examples:
          [
            (4, { a = 1; b = 0; spins = 0 }, []);
            (4, { a = 2; b = 1; spins = 0 }, [ 7 ]);
          ]
        Gen.(triple parallels fn (list small_int))
        is_list_map;
      cases ~name:(Printf.sprintf "PARALLEL=%d")
        "applies f once to each element under" [ 0; 1; 4 ] (fun p ->
          applies_each_element_once p ());
      test "keeps the order of l when later elements finish first"
        keeps_order_of_late_finishers;
    ]

(* Domains *)

(* [spreads n] maps [n] applications that meet under PARALLEL=[n]: they run on
   [n] domains, the caller's among them. *)
let spreads n =
  let meet = meeting n in
  let on =
    with_parallel n (fun () ->
        Worker.map
          (fun () ->
            meet ();
            domain ())
          (List.init n ignore))
  in
  equal int n (List.length (List.sort_uniq Int.compare on));
  mem int (domain ()) on

let stays_on_caller p =
  let on =
    with_parallel p (fun () -> Worker.map domain (List.init 16 ignore))
  in
  List.iter (equal int (domain ())) on

let runs_at_most p () =
  let g = gauge () in
  ignore
    (with_parallel p (fun () ->
         Worker.map
           (fun () -> within g (fun () -> Unix.sleepf 0.001))
           (List.init 40 ignore)));
  at_most int ~than:p (Atomic.get g.peak)

(* Callers on domains of their own, outside the budget, each call map at once:
   the other domains of all calls are at most [p - 1]. *)
let shares_budget () =
  let p = 3 and callers = 3 in
  let g = gauge () in
  let call () =
    with_parallel p (fun () ->
        Worker.map
          (fun i ->
            within g (fun () ->
                Unix.sleepf 0.005;
                i))
          (List.init 6 Fun.id))
  in
  let results = spawn_all (List.init callers (fun _ -> call)) in
  List.iter (equal (list int) (List.init 6 Fun.id)) results;
  at_most int ~than:(callers + p - 1) (Atomic.get g.peak)

(* [while_held p n f] is [f ()], run while a call on another caller, under
   PARALLEL=[p], holds [n] applications running at once. *)
let while_held p n f =
  let meet = meeting n in
  let holding = Atomic.make 0 and released = Atomic.make false in
  let hold () =
    meet ();
    Atomic.incr holding;
    await "the release" (fun () -> Atomic.get released)
  in
  let holder =
    Domain.spawn (fun () ->
        with_parallel p (fun () -> Worker.map hold (List.init n ignore)))
  in
  Fun.protect
    ~finally:(fun () ->
      Atomic.set released true;
      ignore (Domain.join holder))
    (fun () ->
      await "the holding call" (fun () -> Atomic.get holding = n);
      f ())

(* Under PARALLEL=3, a call of two elements holds one of the two other domains,
   and a second call of two elements takes the last. *)
let holds_what_it_needs () =
  while_held 3 2 (fun () ->
      let meet = meeting 2 in
      let on =
        with_parallel 3 (fun () ->
            Worker.map
              (fun () ->
                meet ();
                domain ())
              [ (); () ])
      in
      equal int 2 (List.length (List.sort_uniq Int.compare on)))

let gives_domains_back () =
  (match
     with_parallel 3 (fun () ->
         Worker.map (fun i -> if i = 0 then raise (Failed 0)) [ 0; 1; 2; 3 ])
   with
  | _ -> fail "the call did not raise"
  | exception Failed 0 -> ());
  ignore (with_parallel 3 (fun () -> Worker.map Fun.id [ 1; 2; 3 ]));
  spreads 3

(* Each application lasts, so that the domains a call spawns are all alive at
   once. *)
let beyond_runtime_limit () =
  let l = List.init 300 Fun.id in
  let f i =
    Unix.sleepf 0.01;
    i + 1
  in
  equal (list int) (List.map succ l)
    (with_parallel 300 (fun () -> Worker.map f l))

let domains =
  group "domains"
    [
      test "applies f on PARALLEL domains at once, the caller's among them"
        (fun () -> spreads 3);
      cases
        ~name:(Printf.sprintf "PARALLEL=%d")
        "applies every element on the calling domain under" [ -1; 0; 1 ]
        stays_on_caller;
      cases ~name:(Printf.sprintf "PARALLEL=%d")
        "runs at most PARALLEL applications at once under" [ 1; 2; 4 ] (fun p ->
          runs_at_most p ());
      test "shares PARALLEL - 1 domains between concurrent calls" shares_budget;
      test "works on the calling domain alone when no domain is free" (fun () ->
          while_held 2 2 (fun () -> stays_on_caller 2));
      test "holds at most one domain fewer than its elements"
        holds_what_it_needs;
      test "gives its domains back when it returns or raises" gives_domains_back;
      test "computes List.map with PARALLEL above the runtime's domain limit"
        beyond_runtime_limit;
    ]

(* Nesting *)

let nested_while_outer_holds_all () =
  let meet = meeting 2 in
  let inner () =
    meet ();
    let outer = domain () in
    let on = Worker.map (fun _ -> domain ()) (List.init 5 Fun.id) in
    List.for_all (fun d -> d = outer) on
  in
  equal (list bool) [ true; true ]
    (with_parallel 2 (fun () -> Worker.map inner [ (); () ]))

let nested_is_list_map () =
  let g = gauge () in
  let l = List.init 8 Fun.id in
  let f i = Worker.map (fun j -> within g (fun () -> (i * 8) + j)) l in
  equal
    (list (list int))
    (List.map f l)
    (with_parallel 4 (fun () -> Worker.map f l));
  at_most int ~than:4 (Atomic.get g.peak)

let nesting =
  group "nesting"
    [
      test "a call from f while the outer call holds every domain stays on f's"
        nested_while_outer_holds_all;
      test "calls from f compute List.map and share the budget"
        nested_is_list_map;
    ]

(* Failure *)

let fail_at i = raise (Failed i)
let fail_line = __LINE__ - 1

(* [n] elements, of which [failing] raise; a failing element waits longer the
   lower its index, so that later failures tend to come first. *)
let failure_case =
  Gen.(
    let* p, n = pair parallels (int_range 1 24) in
    let+ first = int_range 0 (n - 1)
    and+ others = subsequence ~pp:Format.pp_print_int (List.init n Fun.id) in
    (p, n, first :: others))

let raises_lowest (p, n, failing) =
  let first_raised = Atomic.make None in
  let f i =
    if List.mem i failing then begin
      Unix.sleepf (0.0002 *. Float.of_int (n - i));
      ignore (Atomic.compare_and_set first_raised None (Some i));
      fail_at i
    end
    else i
  in
  let lowest = List.fold_left min n failing in
  raises (Failed lowest) (fun () ->
      with_parallel p (fun () -> Worker.map f (List.init n Fun.id)));
  cover "a later element raised first"
    (match Atomic.get first_raised with Some i -> i > lowest | None -> false)

let raise_site bt =
  match Printexc.backtrace_slots bt with
  | None -> None
  | Some slots ->
      Array.find_map Printexc.Slot.location slots
      |> Option.map (fun (l : Printexc.location) ->
          (Filename.basename l.filename, l.line_number))

let keeps_backtrace ~on_caller () =
  let caller = domain () and meet = meeting 2 in
  let f i =
    meet ();
    if Bool.equal (domain () = caller) on_caller then fail_at i else i
  in
  match with_parallel 2 (fun () -> Worker.map f [ 0; 1 ]) with
  | _ -> fail "the call did not raise"
  | exception Failed _ ->
      let bt = Printexc.get_raw_backtrace () in
      equal
        (option (pair string int))
        (Some ("test_worker.ml", fail_line))
        (raise_site bt)

let stops_after_failure p () =
  let n = 2000 and applied = Atomic.make 0 in
  let f i =
    if i = 0 then fail_at 0;
    Atomic.incr applied;
    Unix.sleepf 0.0002
  in
  raises (Failed 0) (fun () ->
      with_parallel p (fun () -> Worker.map f (List.init n Fun.id)));
  less int ~than:(n - 1) (Atomic.get applied)

let joins_before_raising () =
  let g = gauge () and meet = meeting 4 in
  let f i =
    within g (fun () ->
        meet ();
        if i = 0 then fail_at 0 else Unix.sleepf 0.05)
  in
  raises (Failed 0) (fun () ->
      with_parallel 4 (fun () -> Worker.map f [ 0; 1; 2; 3 ]));
  equal ~msg:"applications running once the call raised" int 0
    (Atomic.get g.live)

let failure =
  group "failure"
    [
      prop "raises the exception of the lowest failing element" failure_case
        raises_lowest;
      test "raises with the backtrace of an application on another domain"
        (keeps_backtrace ~on_caller:false);
      test "raises with the backtrace of an application on the calling domain"
        (keeps_backtrace ~on_caller:true);
      cases ~name:(Printf.sprintf "PARALLEL=%d")
        "stops applying f once an element raised, under" [ 0; 4 ] (fun p ->
          stops_after_failure p ());
      test "raises only after the applications it started have ended"
        joins_before_raising;
    ]

(* Settings *)

(* The values of BEAM, DEFAULT_FLOAT and CHECK_OOB. *)
let seen () =
  ( Context_var.value beam,
    Context_var.value default_float,
    Context_var.value check_oob )

let seen_w = triple int string bool

let under (b, d, c) f =
  context [ B (beam, b); B (default_float, d); B (check_oob, c) ] f

let sees_caller_settings () =
  let s = (3, "half", true) in
  let meet = meeting 3 in
  let on =
    under s (fun () ->
        with_parallel 3 (fun () ->
            Worker.map
              (fun () ->
                meet ();
                seen ())
              [ (); (); () ]))
  in
  List.iter (equal seen_w s) on

(* Two callers on domains of their own, each under its own settings, map at once
   with domains to spare. *)
let each_caller_passes_its_own () =
  let call s () =
    under s (fun () ->
        let meet = meeting 2 in
        ( s,
          with_parallel 5 (fun () ->
              Worker.map
                (fun () ->
                  meet ();
                  seen ())
                [ (); () ]) ))
  in
  spawn_all [ call (1, "float16", true); call (2, "float64", false) ]
  |> List.iter (fun (s, on) -> List.iter (equal seen_w s) on)

let binding_stays_in_its_application () =
  let before = Context_var.value beam and meet = meeting 3 in
  let f i =
    context
      [ B (beam, 10 + i) ]
      (fun () ->
        meet ();
        Context_var.value beam)
  in
  equal (list int) [ 10; 11; 12 ]
    (with_parallel 3 (fun () -> Worker.map f [ 0; 1; 2 ]));
  equal ~msg:"the caller's value" int before (Context_var.value beam)

(* Two applications on two domains bind CHECK_OOB apart and build nodes under
   SPEC=2, whose construction check binds CHECK_OOB itself. *)
let check_oob_race () =
  let fresh = Atomic.make (1 lsl 40) and meet = meeting 2 in
  let before = Context_var.value check_oob in
  let build own =
    context
      [ B (spec, 2); B (check_oob, own) ]
      (fun () ->
        meet ();
        let strays = ref 0 in
        for _ = 1 to 2000 do
          let k = Atomic.fetch_and_add fresh 1 in
          ignore (Ops.const (`Int (Bigint.of_int k)));
          if Context_var.value check_oob <> own then incr strays
        done;
        !strays)
  in
  equal (list int) [ 0; 0 ]
    (with_parallel 2 (fun () -> Worker.map build [ true; false ]));
  equal ~msg:"the caller's CHECK_OOB" bool before (Context_var.value check_oob)

let settings =
  group "settings"
    [
      test "every application sees the caller's settings" sees_caller_settings;
      test "concurrent callers each pass their own settings"
        each_caller_passes_its_own;
      test "a setting bound in an application is seen by it alone"
        binding_stays_in_its_application;
      test "applications building nodes keep their own CHECK_OOB" check_oob_race;
    ]

(* Compiling *)

module Compiler = Renderer.Compiler

(* The Clang kernels of the Compiler_cpu suite, each a linear program. *)
let kernels =
  lazy
    (Ops.src (Golden.sink "../../runtime/support/compiler_cpu/kernels.golden"))

(* Binaries compare as bytes and print as digests. *)
let binaries =
  Testable.contramap
    (List.map (fun b -> Digest.to_hex (Digest.string b)))
    (list string)

let compiles_as_serial () =
  let renderer = Cstyle.clang host_target in
  let clang = Compiler_cpu.clang host_target.arch in
  let sources =
    List.map (fun k -> renderer.render (Ops.src k)) (Lazy.force kernels)
  in
  let compile src = Compiler.compile clang src in
  equal binaries (List.map compile sources)
    (with_parallel 3 (fun () -> Worker.map compile sources))

let compiling =
  group "compiling"
    [
      test "compiles Clang kernels to the binaries of a serial compilation"
        compiles_as_serial;
    ]

let () =
  exit
    (run "Tolk.Worker" [ law; domains; nesting; failure; settings; compiling ])
