open Windtrap
open Tolk

(* The host's target, as the engine gives it. *)
let host_target () = Run.target Rig.host

exception Failed of int

let with_parallel p f = Setting.context [ B (Setting.parallel, p) ] f
let domain () = (Domain.self () :> int)

(* Synchronisation *)

(* A count of events that tests wait on: [bump c] records one, and
   [until c n] returns once [c] has recorded [n]. A wait reads no clock:
   WINDTRAP_TIMEOUT fails a test whose event never comes. *)
type count = { mutable n : int; lock : Mutex.t; changed : Condition.t }

let count () = { n = 0; lock = Mutex.create (); changed = Condition.create () }

let bump c =
  Mutex.protect c.lock (fun () ->
      c.n <- c.n + 1;
      Condition.broadcast c.changed)

let until c n =
  Mutex.protect c.lock (fun () ->
      while c.n < n do
        Condition.wait c.changed c.lock
      done)

(* [meeting n] returns a function that returns once [n] calls have reached it:
   applications that call it all run at once, on [n] domains. *)
let meeting n =
  let arrived = count () in
  fun () ->
    bump arrived;
    until arrived n

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

(* Four applications meet, then each ends once the next has ended: they end
   last element first. *)
let keeps_order_of_late_finishers () =
  let n = 4 in
  let meet = meeting n and ended = Array.init n (fun _ -> count ()) in
  let f i =
    meet ();
    if i < n - 1 then until ended.(i + 1) 1;
    bump ended.(i);
    i * 10
  in
  equal (list int)
    (List.init n (fun i -> i * 10))
    (with_parallel n (fun () -> Worker.map f (List.init n Fun.id)))

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
      cases "applies f once to each element under"
        ~id:(Printf.sprintf "PARALLEL=%d") [ 0; 1; 4 ] (fun p ->
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

(* The domains a call spawns beside its own: [f ()], and how many [f] spawned
   on this domain. A domain runs one application at a time, so a call that
   spawns at most [p - 1] runs at most [p] applications at once. *)
let spawning f =
  let before = Worker.spawns () in
  let r = f () in
  (r, Worker.spawns () - before)

let runs_at_most p () =
  let _, spawned =
    spawning (fun () ->
        with_parallel p (fun () -> Worker.map domain (List.init 40 ignore)))
  in
  at_most int ~than:(p - 1) spawned

(* [while_held p n f] is [f ()], run while a call on another caller, under
   PARALLEL=[p], holds [n] applications running at once. *)
let while_held p n f =
  let meet = meeting n in
  let holding = count () and released = count () in
  let hold () =
    meet ();
    bump holding;
    until released 1
  in
  let holder =
    Domain.spawn (fun () ->
        with_parallel p (fun () -> Worker.map hold (List.init n ignore)))
  in
  Fun.protect
    ~finally:(fun () ->
      bump released;
      ignore (Domain.join holder))
    (fun () ->
      until holding n;
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

(* A child whose runtime holds at most four domains maps 300 elements under
   PARALLEL=300, and prints whether the call gave List.map's result and the
   domains it spawned. Each spawned application waits for the caller's first,
   which starts once the call has spawned what it could: the three spawned
   domains stay alive, and every later spawn is refused. The limit is the
   child's alone. *)
let beyond_limit () =
  let caller = domain () and started = count () in
  let f i =
    if domain () = caller then bump started else until started 1;
    i + 1
  in
  let l = List.init 300 Fun.id in
  let r, spawned = spawning (fun () -> Worker.map f l) in
  Printf.printf "%b, %d spawned" (r = List.map succ l) spawned

let parts = [ ("beyond the limit", beyond_limit) ]

let beyond_runtime_limit () =
  match
    Disk_cache.child
      ~env:[ ("OCAMLRUNPARAM", "d=4"); ("PARALLEL", "300") ]
      ~cachedb:(Disk_cache.fresh ()) "beyond the limit"
  with
  | Ok out -> equal string "true, 3 spawned" out
  | Error err -> failf "the child failed: %s" err

(* By system number, every signal OCaml names that the system has, but those a
   fault raises and the two no thread can block. A handler that ran on a domain
   the call spawned, before [Domain.spawn] had made the domain's DLS, would end
   the process. *)
let faults =
  List.map Sys.signal_to_int
    Sys.[ sigsegv; sigbus; sigfpe; sigill; sigtrap; sigabrt; sigsys ]

let maskable =
  List.init 30 (fun i -> Sys.signal_to_int (-i - 1))
  |> List.filter (fun s ->
      s > 0
      && not
           (List.mem s
              (Sys.signal_to_int Sys.sigkill
              :: Sys.signal_to_int Sys.sigstop
              :: faults)))
  |> List.sort_uniq Int.compare

(* The calling thread's mask, by system number. *)
let thread_mask () =
  Unix.sigprocmask Unix.SIG_BLOCK []
  |> List.map Sys.signal_to_int |> List.sort Int.compare

(* [run mask] applies [mask] to two elements, one on the calling domain and one
   on a domain the call spawns, and is their results. The spawned domain blocks
   the signals, and the calling thread's mask is as it was. *)
let blocks_signals run =
  if Sys.win32 then skip ~reason:"Windows has no signal masks" ();
  let caller = domain () and meet = meeting 2 in
  let mask () =
    meet ();
    if domain () = caller then None else Some (thread_mask ())
  in
  let before = thread_mask () in
  let results = with_parallel 2 (fun () -> run mask) in
  equal ~msg:"the caller's mask" (list int) before (thread_mask ());
  match List.filter_map Fun.id results with
  | [ blocked ] ->
      List.iter (fun s -> mem int s blocked) maskable;
      equal ~msg:"faults blocked" (list int) []
        (List.filter (fun s -> List.mem s blocked) faults)
  | masks -> failf "%d applications on a spawned domain" (List.length masks)

exception Interrupted

(* A signal comes while the call joins the domain it spawned, which blocks the
   signal: the call raises the handler's exception once the domain's application
   has ended, and gives the domain back. The spawned domain waits for the caller
   to reach its join. *)
let signal_while_joining () =
  if Sys.win32 then skip ~reason:"Windows has no signal masks" ();
  let caller = domain () and meet = meeting 2 in
  let returned = count () and ended = Atomic.make false in
  (* The sleeps aim the signal at the caller's join and keep the spawned
     application running past it, so that a call that raised before the join
     would see it unended. No event can hold them: the handler runs once the
     join returns, and the join waits for this application. A sleep can only
     miss a wrong call, never fail a right one. *)
  let f () =
    meet ();
    if domain () = caller then bump returned
    else begin
      until returned 1;
      Unix.sleepf 0.05;
      Unix.kill (Unix.getpid ()) Sys.sigusr1;
      Unix.sleepf 0.05;
      Atomic.set ended true
    end
  in
  let interrupt = Sys.Signal_handle (fun _ -> raise Interrupted) in
  let previous = Sys.signal Sys.sigusr1 interrupt in
  Fun.protect
    ~finally:(fun () -> Sys.set_signal Sys.sigusr1 previous)
    (fun () ->
      raises Interrupted (fun () ->
          with_parallel 2 (fun () -> Worker.map f [ (); () ])));
  equal ~msg:"the spawned application ended" bool true (Atomic.get ended);
  spreads 2

let domains =
  group "domains"
    [
      test "applies f on PARALLEL domains at once, the caller's among them"
        (fun () -> spreads 3);
      cases "applies every element on the calling domain under"
        ~id:(Printf.sprintf "PARALLEL=%d")
        [ -1; 0; 1 ] stays_on_caller;
      cases "runs at most PARALLEL applications at once under"
        ~id:(Printf.sprintf "PARALLEL=%d") [ 1; 2; 4 ] (fun p ->
          runs_at_most p ());
      test "works on the calling domain alone when no domain is free" (fun () ->
          while_held 2 2 (fun () -> stays_on_caller 2));
      test "holds at most one domain fewer than its elements"
        holds_what_it_needs;
      test "gives its domains back when it returns or raises" gives_domains_back;
      test "computes List.map with PARALLEL above the runtime's domain limit"
        beyond_runtime_limit;
      test "runs no signal handler on the domains it spawns" (fun () ->
          blocks_signals (fun mask -> Worker.map mask [ (); () ]));
      test "raises a signal's exception during its joins once they end"
        signal_while_joining;
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

(* [n] elements, of which [failing] raise. *)
let failure_case =
  Gen.(
    let* p, n = pair parallels (int_range 1 24) in
    let+ first = int_range 0 (n - 1)
    and+ others = subsequence ~pp:Format.pp_print_int (List.init n Fun.id) in
    (p, n, first :: others))

let raises_lowest (p, n, failing) =
  let f i = if List.mem i failing then fail_at i else i in
  let lowest = List.fold_left min n failing in
  raises (Failed lowest) (fun () ->
      with_parallel p (fun () -> Worker.map f (List.init n Fun.id)))

(* Two applications meet, and element 0 raises once element 1 has. *)
let raises_lowest_after_a_later () =
  let meet = meeting 2 and raised = count () in
  let f i =
    meet ();
    if i = 0 then until raised 1 else bump raised;
    fail_at i
  in
  raises (Failed 0) (fun () ->
      with_parallel 2 (fun () -> Worker.map f [ 0; 1 ]))

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

let joins_before_raising () =
  let g = gauge () and meet = meeting 4 in
  (* The other applications sleep past the raise, as in
     [signal_while_joining]: a call that raised before joining them would see
     them running. *)
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
      test
        "raises the lowest failing element's exception when a later element \
         raised first"
        raises_lowest_after_a_later;
      test "raises with the backtrace of an application on another domain"
        (keeps_backtrace ~on_caller:false);
      test "raises with the backtrace of an application on the calling domain"
        (keeps_backtrace ~on_caller:true);
      test "raises only after the applications it started have ended"
        joins_before_raising;
    ]

(* Settings *)

(* The values of BEAM, DEFAULT_FLOAT and CHECK_OOB. *)
let seen () =
  ( Setting.value Setting.beam,
    Setting.value Setting.default_float,
    Setting.value Setting.check_oob )

let seen_w = triple int string bool

let under (b, d, c) f =
  Setting.context
    [
      B (Setting.beam, b); B (Setting.default_float, d); B (Setting.check_oob, c);
    ]
    f

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
  let before = Setting.value Setting.beam and meet = meeting 3 in
  let f i =
    Setting.context
      [ B (Setting.beam, 10 + i) ]
      (fun () ->
        meet ();
        Setting.value Setting.beam)
  in
  equal (list int) [ 10; 11; 12 ]
    (with_parallel 3 (fun () -> Worker.map f [ 0; 1; 2 ]));
  equal ~msg:"the caller's value" int before (Setting.value Setting.beam)

(* Two applications on two domains bind CHECK_OOB apart and build nodes, each
   within a binding of its own that negates it. *)
let check_oob_race () =
  let fresh = Atomic.make (1 lsl 40) and meet = meeting 2 in
  let before = Setting.value Setting.check_oob in
  let build own =
    Setting.context
      [ B (Setting.check_oob, own) ]
      (fun () ->
        meet ();
        let strays = ref 0 in
        for _ = 1 to 2000 do
          let k = Atomic.fetch_and_add fresh 1 in
          Setting.context
            [ B (Setting.check_oob, not own) ]
            (fun () -> ignore (Ops.const (`Int (Bigint.of_int k))));
          if Setting.value Setting.check_oob <> own then incr strays
        done;
        !strays)
  in
  equal (list int) [ 0; 0 ]
    (with_parallel 2 (fun () -> Worker.map build [ true; false ]));
  equal ~msg:"the caller's CHECK_OOB" bool before
    (Setting.value Setting.check_oob)

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

(* Streaming *)

(* The elements [iter] handed to [g], with their images, in order. *)
let streamed p f l =
  let seen = ref [] in
  with_parallel p (fun () ->
      Worker.iter f (fun x y -> seen := (x, y) :: !seen) l);
  List.rev !seen

(* [g] takes each element once, with its image: in [l]'s order on one domain,
   in the order the applications end on several. *)
let hands_each_once (p, fn, l) =
  cover "several domains may apply f" (p > 1 && List.length l > 1);
  let expected = List.map (fun x -> (x, apply fn x)) l in
  let got = streamed p (apply fn) l in
  if p <= 1 then equal (list (pair int string)) expected got
  else
    equal
      (list (pair int string))
      (List.sort compare expected) (List.sort compare got)

(* The last element's application waits for [g]'s first: [g] reaches an element
   once its own image is computed, whatever the later ones. *)
let consumes_while_computing p () =
  let consumed = count () in
  let n = 6 in
  let f i =
    if i = n - 1 then until consumed 1;
    i
  in
  let g _ _ = bump consumed in
  with_parallel p (fun () -> Worker.iter f g (List.init n Fun.id))

let consumes_on_caller p =
  let caller = domain () and on = ref [] in
  with_parallel p (fun () ->
      Worker.iter Fun.id
        (fun _ () -> on := domain () :: !on)
        (List.init 16 ignore));
  List.iter (equal int caller) !on

let computes_at_most p () =
  let (), spawned =
    spawning (fun () ->
        with_parallel p (fun () ->
            Worker.iter domain (fun () _ -> ()) (List.init 40 ignore)))
  in
  at_most int ~than:(p - 1) spawned

(* [n] elements of which [f] raises on [failing] and [g] on [g_failing]: [g] is
   applied to the elements before the first of either, and the call raises the
   first one's exception. *)
let streamed_failure_case =
  let ints = Format.(pp_print_list ~pp_sep:pp_print_space pp_print_int) in
  let pp ppf (p, n, failing, g_failing) =
    Format.fprintf ppf "PARALLEL=%d, %d elements, f raises on [%a], g on [%a]" p
      n ints failing ints g_failing
  in
  Gen.(
    with_pp pp
      (let* p, n = pair parallels (int_range 1 24) in
       let+ failing = subsequence ~pp:Format.pp_print_int (List.init n Fun.id)
       and+ g_failing =
         subsequence ~pp:Format.pp_print_int (List.init n Fun.id)
       in
       (p, n, failing, g_failing)))

exception G_failed of int

let raises_first (p, n, failing, g_failing) =
  let consumed = ref [] in
  let f i = if List.mem i failing then fail_at i else i in
  let g i _ =
    if List.mem i g_failing then raise (G_failed i);
    consumed := i :: !consumed
  in
  let first l = List.fold_left min n l in
  let f_first = first failing and g_first = first g_failing in
  let stop = min f_first g_first in
  let raised =
    match with_parallel p (fun () -> Worker.iter f g (List.init n Fun.id)) with
    | () -> None
    | exception Failed i -> Some (`F i)
    | exception G_failed i -> Some (`G i)
  in
  let consumed = List.rev !consumed in
  (* On one domain, [g]'s order is [List.iter]'s; on several, the order the
     applications end, so the call raises one of the failures and [g] took
     elements that fail neither. *)
  (if p <= 1 then begin
     (match raised with
     | None -> equal ~msg:"no element raised" int n stop
     | Some (`F i) ->
         equal ~msg:"f raised" int f_first i;
         at_most int ~than:g_first i
     | Some (`G i) ->
         equal ~msg:"g raised" int g_first i;
         less int ~than:f_first i);
     equal ~msg:"consumed" (list int) (List.init stop Fun.id) consumed
   end
   else begin
     (match raised with
     | None -> equal ~msg:"no element raised" int n stop
     | Some (`F i) ->
         equal ~msg:"f raised on a failing element" (list int) [ i ]
           (List.filter (( = ) i) failing)
     | Some (`G i) ->
         equal ~msg:"g raised on a failing element" (list int) [ i ]
           (List.filter (( = ) i) g_failing));
     equal ~msg:"consumed elements that fail neither" (list int) []
       (List.filter (fun i -> List.mem i failing || List.mem i g_failing) consumed);
     equal ~msg:"consumed each once" (list int) (List.sort_uniq compare consumed)
       (List.sort compare consumed)
   end);
  cover "f raised first" (f_first < g_first);
  cover "f and g raised on one element" (f_first = g_first && f_first < n);
  cover "g raised first" (g_first < f_first);
  cover "several domains" (p > 1)
let joins_before_raising_from_g () =
  let g_live = gauge () and started = count () in
  (* The later applications sleep past the raise, as in
     [joins_before_raising]. *)
  let f i =
    within g_live (fun () ->
        bump started;
        if i > 0 then Unix.sleepf 0.05)
  in
  let g i () =
    if i = 0 then begin
      until started 3;
      raise (G_failed 0)
    end
  in
  raises (G_failed 0) (fun () ->
      with_parallel 4 (fun () -> Worker.iter f g [ 0; 1; 2; 3 ]));
  equal ~msg:"applications running once the call raised" int 0
    (Atomic.get g_live.live)

let iter_sees_caller_settings () =
  let s = (3, "half", true) in
  let on = ref [] in
  under s (fun () ->
      with_parallel 3 (fun () ->
          Worker.iter seen (fun () v -> on := v :: !on) (List.init 8 ignore)));
  List.iter (equal seen_w s) !on

let streaming =
  group "streaming"
    [
      prop
        "iter f g l applies g x (f x) to each element once, in l's order on \
         one domain"
        ~examples:[ (4, { a = 1; b = 0; spins = 0 }, []) ]
        Gen.(triple parallels fn (list small_int))
        hands_each_once;
      cases "applies g to an element while f applies to later ones, under"
        ~id:(Printf.sprintf "PARALLEL=%d") [ 0; 1; 2; 4 ] (fun p ->
          consumes_while_computing p ());
      cases "applies g on the calling domain under"
        ~id:(Printf.sprintf "PARALLEL=%d")
        [ 0; 1; 4 ] consumes_on_caller;
      cases "runs at most PARALLEL applications of f at once under"
        ~id:(Printf.sprintf "PARALLEL=%d") [ 1; 2; 4 ] (fun p ->
          computes_at_most p ());
      prop
        "raises an exception of f or g, the first of l's on one domain, and \
         applies g to elements that raise neither"
        streamed_failure_case raises_first;
      test
        "raises from g only after the applications of f it started have ended"
        joins_before_raising_from_g;
      test "every application of f sees the caller's settings"
        iter_sees_caller_settings;
      test "runs no signal handler on the domains it spawns" (fun () ->
          blocks_signals (fun mask ->
              let masks = ref [] in
              Worker.iter mask (fun () m -> masks := m :: !masks) [ (); () ];
              !masks));
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
  let host_target = host_target () in
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
  Disk_cache.play parts;
  exit
    (run "Tolk.Worker"
       [ law; domains; nesting; failure; settings; streaming; compiling ])
