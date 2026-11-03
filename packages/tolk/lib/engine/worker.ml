(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Domains live for one call: an idle domain would still join every minor
   collection. A new domain starts with its spawner's settings. *)
let spawned = Atomic.make 0

let rec reserve wanted =
  let n = Atomic.get spawned in
  let free = Setting.value Setting.parallel - 1 - n in
  let k = max 0 (min wanted free) in
  if k = 0 || Atomic.compare_and_set spawned n (n + k) then k
  else reserve wanted

let release k = ignore (Atomic.fetch_and_add spawned (-k))
let spawns_made = Atomic.make 0
let spawns () = Atomic.get spawns_made

(* Signals

   A signal's handler runs at the next poll point of any domain whose thread
   does not block the signal, and a domain takes its spawner's mask. One spawned
   with signals unblocked could run it at its first poll, before [Domain.spawn]
   has made the domain's DLS, and crash the process; later, it would fail an
   application with the handler's exception, such as [Sys.Break]. So a call
   spawns its domains with every signal but a fault's blocked ([block]), and
   they keep that mask: handlers run on the program's own threads. The calling
   thread blocks them too, but in [body] and in its waits for the domains, where
   it has its own mask ([restore]): a handler's exception then cuts nothing
   short that must run whole, and a signal's default action, such as a
   SIGTERM's, takes effect at once. *)
external block : unit -> (int64[@unboxed])
  = "tolk_worker_block_byte" "tolk_worker_block"
[@@noalloc]

external restore : (int64[@unboxed]) -> unit
  = "tolk_worker_restore_byte" "tolk_worker_restore"

external reraise : exn -> 'a = "%reraise"

(* [settle outer wanted work body] spawns the domains and joins them, with the
   calling thread's signals blocked, but in [body] and in the joins, where its
   mask is [outer]: what [body] ended with, the first exception of a handler
   that cut a join short, and the exceptions the domains' [work] raised. *)
let settle outer wanted work body =
  let domains = ref [] in
  (* A domain's [work] raises only an exception such as [Out_of_memory]: it
     comes back as the domain's result, so that a join raises only a
     handler's. *)
  let guarded () =
    match work () with
    | () -> None
    | exception e -> Some (e, Printexc.get_raw_backtrace ())
  in
  let rec spawn k =
    if k > 0 then
      match Domain.spawn guarded with
      | d ->
          Atomic.incr spawns_made;
          domains := d :: !domains;
          spawn (k - 1)
      | exception Failure _ -> release k
      | exception e ->
          release k;
          reraise e
  in
  (* [f ()] with the caller's mask, then signals blocked again however it ends.
     The mask is restored inside the [match], so that a handler's exception at
     any poll point after it is [f]'s. *)
  let unblocked f =
    match
      restore outer;
      f ()
    with
    | r ->
        ignore (block ());
        Ok r
    | exception e ->
        ignore (block ());
        Error (e, Printexc.get_raw_backtrace ())
  in
  let outcome =
    match spawn (reserve wanted) with
    | () -> unblocked body
    | exception e -> Error (e, Printexc.get_raw_backtrace ())
  in
  (* A join that a handler's exception cuts short is made again, and the first
     such exception is kept. *)
  let interrupted = ref None in
  let rec join d =
    match unblocked (fun () -> Domain.join d) with
    | Ok r -> r
    | Error failure ->
        if Option.is_none !interrupted then interrupted := Some failure;
        join d
  in
  let failed = List.filter_map join !domains in
  release (List.length !domains);
  (outcome, !interrupted, failed)

(* [alongside wanted work body] where a domain may be spawned. An exception
   that is not a handler's, from a finaliser, a memory profiler's callback or
   [Out_of_memory] at an allocation while signals are blocked, still gives the
   caller its mask back, but may leave domains unjoined. *)
let spawning wanted work body =
  let outer = block () in
  let ended =
    match settle outer wanted work body with
    | ended -> ended
    | exception e ->
        restore outer;
        reraise e
  in
  restore outer;
  match ended with
  | _, Some (e, bt), _ | Error (e, bt), None, _ | Ok _, None, (e, bt) :: _ ->
      Printexc.raise_with_backtrace e bt
  | Ok r, None, [] -> r

(* [body ()] on the calling domain while up to [wanted] other domains run
   [work], returning once they all have. The runtime refuses a domain past its
   limit, which other domains of the program may have reached: the call then
   does with fewer. *)
let alongside wanted work body =
  if wanted <= 0 || Setting.value Setting.parallel <= 1 then body ()
  else spawning wanted work body

let map f l =
  let tasks = Array.of_list l in
  let count = Array.length tasks in
  let results = Array.make count None and next = Atomic.make 0 in
  (* The raising application of least index. Indices are claimed in order, so
     every one below it has been claimed and runs to its end. *)
  let failure = Atomic.make None in
  let rec fail i e bt =
    match Atomic.get failure with
    | Some (j, _, _) when (j < i) [@mutate off "an index fails at most once"] ->
        ()
    | seen ->
        if not (Atomic.compare_and_set failure seen (Some (i, e, bt))) then
          fail i e bt
  in
  let rec work () =
    let i = Atomic.fetch_and_add next 1 in
    let runs =
      match Atomic.get failure with
      | Some (j, _, _) -> i < j
      | None -> i < count
    in
    if runs then begin
      (match f tasks.(i) with
      | y -> results.(i) <- Some y
      | exception e -> fail i e (Printexc.get_raw_backtrace ()));
      work ()
    end
  in
  alongside (count - 1) work work;
  match Atomic.get failure with
  | Some (_, e, bt) -> Printexc.raise_with_backtrace e bt
  | None -> List.init count (fun i -> Option.get results.(i))

let iter f g l =
  let tasks = Array.of_list l in
  let count = Array.length tasks in
  let next = Atomic.make 0 and stopped = Atomic.make false in
  let ready = Mutex.create () and filled = Condition.create () in
  let finished = Queue.create () in
  (* The next index no domain has claimed, unless an application raised or the
     consumer stopped: then none starts. *)
  let claim () =
    if Atomic.get stopped then None
    else
      let i = Atomic.fetch_and_add next 1 in
      if i < count then Some i else None
  in
  let apply i =
    match f tasks.(i) with
    | y -> Ok y
    | exception e ->
        let bt = Printexc.get_raw_backtrace () in
        Atomic.set stopped true;
        Error (e, bt)
  in
  let rec work () =
    match claim () with
    | None -> ()
    | Some i ->
        let r = apply i in
        Mutex.protect ready (fun () ->
            Queue.push (i, r) finished;
            Condition.broadcast filled);
        work ()
  in
  (* The next application to end: one another domain ended, or else one the
     consumer runs itself, rather than wait, while an index is unclaimed. *)
  let take () =
    match Mutex.protect ready (fun () -> Queue.take_opt finished) with
    | Some r -> r
    | None -> (
        match claim () with
        | Some i -> (i, apply i)
        | None ->
            Mutex.protect ready (fun () ->
                while Queue.is_empty finished do
                  Condition.wait filled ready
                done;
                Queue.take finished))
  in
  let consume () =
    for _ = 1 to count do
      match take () with
      | i, Ok y -> g tasks.(i) y
      | _, Error (e, bt) -> Printexc.raise_with_backtrace e bt
    done
  in
  alongside (count - 1) work (fun () ->
      Fun.protect ~finally:(fun () -> Atomic.set stopped true) consume)
