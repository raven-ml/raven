(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let ( let* ) = Result.bind

(* A vendor's GPUs. [mutex] serializes opens, resets and changes, drivers
   included, and guards [exits], whether the exit hook is registered. Only opens
   add holds and only renewals clear lost GPUs, so what they check stays true
   while they run. [holds] guards the GPUs held, from the start of their open,
   and those lost, which their next open renews; it is held briefly, so that
   stopping a GPU waits for no driver. A GPU is named by its machine and bus
   address. *)
type siblings = Remove | Keep

type t = {
  name : string;
  noun : string;
  memory_bar : int option;
  siblings : siblings;
  nodes : root:string -> string -> string list;
  unreleased : root:string -> string -> string option;
  teardown_ms : int;
  reset : Function.t -> (unit, string) result;
  is_gpu : Machine.id -> bool;
  mutex : Lock.t;
  mutable exits : bool;
  holds : Lock.t;
  mutable held : hold list;
  mutable spent : (Machine.t * string) list;
}

(* [pid] is the process that opened it: a child of fork inherits the hold, and
   it stays the parent's. [lock] makes the first [stop] the only one to call
   the vendor's: [stopped] is its answer, and [lost] whether it loses the GPU.
   [returned] is whether a stop released the function, which the next stop
   does if the first's release raised. A renewal that failed leaves the GPU
   lost whatever stops it. *)
and hold = {
  gpus : t;
  machine : Machine.t;
  bus : string;
  fn : Function.t;
  pid : int;
  lock : Lock.t;
  mutable vendor_stop : (unit -> [ `Clean | `Lost | `Unknown ]) option;
  mutable unrenewed : bool;
  mutable stopped : [ `Stopped | `Unknown ] option;
  mutable lost : bool;
  mutable returned : bool;
}

let make ~name ~noun ~memory_bar ~siblings ~nodes ~unreleased ~teardown_ms
    ~reset is_gpu =
  {
    name;
    noun;
    memory_bar;
    siblings;
    nodes;
    unreleased;
    teardown_ms;
    reset;
    is_gpu;
    mutex = Lock.create ();
    exits = false;
    holds = Lock.create ();
    held = [];
    spent = [];
  }

(* Work no kernel driver bounds is hung once it has not advanced for 30 s. *)
let hang_ms = 30_000
let index g fn i =
  if i < 0 then invalid_argf "Gpus.%s: %s %d is negative" fn g.noun i

let name g i =
  index g "name" i;
  if i = 0 then g.name else strf "%s:%d" g.name i

let named g i r = Result.map_error (fun why -> strf "%s: %s" (name g i) why) r

let buses g m =
  List.filter_map
    (fun (id : Machine.id) -> if g.is_gpu id then Some id.bus else None)
    (Machine.functions m)

(* Checks *)

(* The bus of GPU [i] of [m], which the process does not hold. *)
let gpu g m i =
  let all = buses g m in
  match List.nth_opt all i with
  | None when all = [] -> Error (strf "no such %s; the machine has none" g.noun)
  | None ->
      Error (strf "no such %s; the machine has %d" g.noun (List.length all))
  | Some bus
    when Lock.protect g.holds (fun () ->
             List.exists (fun h -> h.machine == m && h.bus = bus) g.held) ->
      Error (bus ^ " is open in this process")
  | Some bus -> Ok bus

(* Renewals *)

let lost g m bus =
  Lock.protect g.holds (fun () ->
      List.exists (fun (m', b) -> m' == m && b = bus) g.spent)

(* Records whether the GPU at [bus] of [m] is lost, under [holds]. *)
let mark g m bus ~lost =
  let others = List.filter (fun (m', b) -> not (m' == m && b = bus)) g.spent in
  g.spent <- (if lost then (m, bus) :: others else others)

(* Resets the GPU of the taken function [fn] as its vendor does: it reaches none
   of the memory processes that died left then. One whose reset failed or raised
   is lost, and renewed by its next open. *)
let renew_taken g m bus fn =
  let reset () =
    Function.set_bus_master fn false;
    let* () = g.reset fn in
    Function.forget fn
  in
  match reset () with
  | r ->
      Lock.protect g.holds (fun () -> mark g m bus ~lost:(Result.is_error r));
      r
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      Lock.protect g.holds (fun () -> mark g m bus ~lost:true);
      Printexc.raise_with_backtrace e bt

let renew h =
  match renew_taken h.gpus h.machine h.bus h.fn with
  | Ok () -> Ok ()
  | Error _ as e ->
      h.unrenewed <- true;
      e
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      h.unrenewed <- true;
      Printexc.raise_with_backtrace e bt

(* Stopping *)

(* Gives [h]'s function back, under [holds]: no open or reset sees the GPU free
   with its function still taken, or lost before it is marked. *)
let give_back h ~lost =
  let g = h.gpus in
  Lock.protect g.holds @@ fun () ->
  g.held <- List.filter (fun h' -> h' != h) g.held;
  if lost then mark g h.machine h.bus ~lost:true;
  Function.release h.fn

(* Releases [h]'s function once its stop recorded its answer. A release that
   raised, as one from a callback inside [holds] on the same thread does, is
   made by the next stop. *)
let give_once h =
  if not h.returned then begin
    give_back h ~lost:h.lost;
    h.returned <- true
  end

(* The first call stops the GPU. A failed open's bracket releases a GPU no write
   reached as it found it, and loses one its vendor started, whatever the
   vendor's stop answers. A vendor's stop that raises leaves the GPU lost. The
   answer is recorded with no allocation between the vendor's stop and its
   store, so no asynchronous exception has a later stop call it again. *)
let finish h ~failed =
  Lock.protect h.lock @@ fun () ->
  match h.stopped with
  | Some s ->
      give_once h;
      s
  | None ->
      let ended =
        match h.vendor_stop with
        | None -> `Clean
        | Some stop -> (
            match stop () with
            | s -> s
            | exception e ->
                h.lost <- true;
                h.stopped <- Some `Unknown;
                let bt = Printexc.get_raw_backtrace () in
                give_once h;
                Printexc.raise_with_backtrace e bt)
      in
      let started = Option.is_some h.vendor_stop in
      h.lost <- h.unrenewed || ended <> `Clean || (failed && started);
      let s =
        match ended with `Unknown -> `Unknown | `Clean | `Lost -> `Stopped
      in
      (h.stopped <-
         match s with `Unknown -> Some `Unknown | `Stopped -> Some `Stopped);
      give_once h;
      s

let stop h = finish h ~failed:false

let set_stop h stop =
  Lock.protect h.lock @@ fun () ->
  if Option.is_some h.stopped then
    invalid_argf "Gpus.set_stop: %s was stopped" h.bus;
  if Option.is_some h.vendor_stop then
    invalid_argf "Gpus.set_stop: %s has a stop already" h.bus;
  h.vendor_stop <- Some stop

(* At exit the process stops the GPUs it holds. The hook is registered at the
   first open, after the one that turns physical takes' bus mastering off, which
   registers when the library loads: hooks run newest first, so a GPU stops
   while its function still reaches memory. The holds are read without their
   lock, which a child of fork may find held by a thread it does not have; a
   stop that gives its GPU back takes it. An exit from a callback inside a lock
   may hold a lock a stop takes: that take raises, and is printed as any
   stop's exception, and the other GPUs stop. *)
let stop_held g =
  let pid = Unix.getpid () in
  let stop h =
    if h.pid = pid then
      try ignore (stop h)
      with e ->
        prerr_endline
          (strf "stopping %s at exit: %s" h.bus (Printexc.to_string e))
  in
  List.iter stop g.held

let register_exit g =
  if not g.exits then begin
    g.exits <- true;
    Stdlib.at_exit (fun () -> stop_held g)
  end

(* Opening *)

(* An unbound GPU whose kernel driver has not let go of it yet is not taken, to
   open, reset or attach it: the driver's release, which writes to it, would
   race the take's writes. *)
let released g l =
  let bus = Sysfs.bus l in
  if Sysfs.driver l <> None then Ok ()
  else
    match g.unreleased ~root:(Sysfs.root (Sysfs.files l)) bus with
    | None -> Ok ()
    | Some why ->
        Error
          (strf
             "its kernel driver has not let go of %s yet, and writes to it when \
              it does: %s"
             bus why)

let take_locked g m l =
  let* () = released g l in
  Function.take_locked m l

(* The check and the take hold one lock, so that no change comes between them.
   Through a transport the far machine's take locks the function. *)
let take g m bus =
  match Machine.files m with
  | None -> Function.take m bus
  | Some files -> Sysfs.locked files bus (fun l -> take_locked g m l)

(* A GPU this process lost, or one a process that died left reaching memory, is
   renewed before the driver starts; one whose renewal fails is lost. *)
let open_ g m i f =
  index g "open_" i;
  named g i @@ Lock.protect g.mutex
  @@ fun () ->
  let* bus = gpu g m i in
  let* fn = take g m bus in
  let h =
    {
      gpus = g;
      machine = m;
      bus;
      fn;
      pid = Unix.getpid ();
      lock = Lock.create ();
      vendor_stop = None;
      unrenewed = false;
      stopped = None;
      lost = false;
      returned = false;
    }
  in
  Lock.protect g.holds (fun () -> g.held <- h :: g.held);
  (* A vendor's stop that raises here replaces [f]'s answer: the GPU is lost
     either way. *)
  let failed () = ignore (finish h ~failed:true) in
  match
    let* () =
      if lost g m bus || Function.inherited fn then renew h else Ok ()
    in
    f h fn
  with
  | Ok _ when Option.is_none h.vendor_stop ->
      failed ();
      invalid_argf "Gpus.open_: the driver of %s answered Ok with no stop" bus
  | Ok _ as r ->
      register_exit g;
      r
  | Error _ as e ->
      failed ();
      e
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      failed ();
      Printexc.raise_with_backtrace e bt

(* Changes to the machine *)

(* [f l] for GPU [i] of [m], which no process takes while it runs: [l] is the
   lock a take holds. *)
let change g fn m i (f : 's. 's Sysfs.lock -> 'a) =
  index g fn i;
  named g i @@ Lock.protect g.mutex
  @@ fun () ->
  match Machine.files m with
  | None ->
      Error
        (strf "%s is reached through a transport; change its %ss there"
           (Option.value (Machine.name m) ~default:"the machine")
           g.noun)
  | Some files ->
      let* bus = gpu g m i in
      Sysfs.locked files bus (fun l -> Fail.result (fun () -> f l))

(* What of this process holds the GPU at [bus]: a device open or mapped, or a
   file of its DRM device the kernel still holds for it. *)
let own files bus nodes =
  match Sysfs.held files bus nodes with
  | Some _ as file -> file
  | None ->
      let me = Unix.getpid () in
      if List.exists (fun (pid, _) -> pid = me) (Sysfs.drm_clients files bus)
      then Some "a file of its DRM device"
      else None

(* A file of the GPU at [bus] another process holds open or mapped: no bound
   holds its wait, so it is refused at once. *)
let refuse_held files bus nodes =
  Option.iter
    (fun (pid, file) ->
      Fail.fail "process %d holds %s open, a file of %s" pid file bus)
    (Sysfs.held_elsewhere files bus nodes)

(* Why the kernel driver has not let go of the GPU at [bus], once no process
   holds a file of it: a file of its DRM device the kernel still holds for a
   process, or, unbound, the vendor's reason. *)
let kernel_holds g files bus ~bound =
  let me = Unix.getpid () in
  match
    List.find_opt (fun (pid, _) -> pid <> me) (Sysfs.drm_clients files bus)
  with
  | Some (0, command) ->
      Some
        (strf
           "a process of another PID namespace (%s) holds a file of its DRM \
            device"
           command)
  | Some (pid, command) ->
      Some
        (strf "the kernel holds a file of its DRM device for process %d (%s)"
           pid command)
  | None when bound -> None
  | None -> g.unreleased ~root:(Sysfs.root files) bus

(* Waits up to [g.teardown_ms] until the kernel lets go, reading every [poll_s]
   and refusing a process that opens a file meanwhile. *)
let poll_s = 0.1

let let_go g files bus nodes ~bound =
  let deadline = Unix.gettimeofday () +. (float g.teardown_ms /. 1000.) in
  let rec go () =
    refuse_held files bus nodes;
    match kernel_holds g files bus ~bound with
    | None -> ()
    | Some why when Unix.gettimeofday () >= deadline ->
        Fail.fail "%s after %d ms: %s" bus g.teardown_ms why
    | Some _ ->
        Unix.sleepf poll_s;
        go ()
  in
  go ()

type progress = Detached | Unreleased of string | Part_way of string

(* A kernel driver lets go of a GPU, writing to it as it does, once the last
   file of its devices goes: a DRM driver's unbind returns first. So detach
   unbinds a driver only once no file remains, refusing at once a file a process
   holds, this one's included, which it would wait for itself, and the driver
   lets go inside the unbind; the vendor confirms it after. A GPU on vfio-pci
   stays as it is, and an unbound one waits for the vendor. A failure before
   [keep_off]'s write leaves the machine as it was, and one after it
   part-way. *)
let detach g m i =
  change g "detach" m i (fun l ->
      let files = Sysfs.files l and bus = Sysfs.bus l in
      let nodes = g.nodes ~root:(Sysfs.root files) bus in
      Option.iter
        (Fail.fail "%s is open in this process, through %s" bus)
        (own files bus nodes);
      let bound =
        match Sysfs.driver l with
        | Some d when d = Sysfs.vfio_pci -> false
        | driver ->
            Option.iter (Fail.fail "%s") (Sysfs.refusal l);
            let bound = Option.is_some driver in
            let_go g files bus nodes ~bound;
            bound
      in
      let remove = g.siblings = Remove in
      Sysfs.keep_off l ~remove;
      let finish () =
        Sysfs.detach l ~remove;
        match
          if bound then g.unreleased ~root:(Sysfs.root files) bus else None
        with
        | Some why -> Unreleased why
        | None ->
            Option.iter (Sysfs.resize l) g.memory_bar;
            Detached
      in
      match Fail.result finish with Ok d -> d | Error why -> Part_way why)

(* Resets *)

(* Renews the GPU of [m] whose function [taken] is, releasing it whatever the
   reset answers. *)
let reset_taken g m taken =
  let* fn = taken in
  Fun.protect ~finally:(fun () -> Function.release fn) @@ fun () ->
  renew_taken g m (Function.bus fn) fn

let reset g m i =
  index g "reset" i;
  named g i @@ Lock.protect g.mutex
  @@ fun () ->
  let* bus = gpu g m i in
  reset_taken g m (take g m bus)

(* A kernel driver's probe expects the GPU as its vendor's reset leaves it,
   whatever ran on it before, in this process or another: a GPU on no kernel
   driver, or on vfio-pci, is reset first, through a take, while the process can
   take it. The take holds the change's lock, so no other comes between.

   On Linux 6.12, an AMD GPU given back to amdgpu after a detach serves its
   render node, but KFD refuses every process until the amdgpu module reloads:
   the unbind left KFD locked. *)
let attach g m i =
  change g "attach" m i (fun l ->
      match Sysfs.driver l with
      | Some d when d <> Sysfs.vfio_pci -> Sysfs.attach l
      | _ ->
          Result.iter_error (Fail.fail "%s")
            (reset_taken g m (take_locked g m l));
          Sysfs.attach l)
