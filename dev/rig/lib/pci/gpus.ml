(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let ( let* ) = Result.bind

(* A vendor's GPUs. [mutex] serializes opens, resets and changes, drivers
   included, and guards [exits], whether the exit hook is registered. Only opens
   add holds and only resets clear lost GPUs, so what they check stays true
   while they run. [holds] guards the GPUs held, from the start of their open,
   with how each stops at exit, and those lost, which open again only after a
   reset; it is held briefly, so that giving a GPU back waits for no driver. A
   GPU is named by its machine and bus address. *)
type t = {
  memory_bar : int;
  nodes : root:string -> string -> string list;
  unreleased : root:string -> string -> string option;
  teardown_ms : int;
  reset : Function.t -> (unit, string) result;
  is_gpu : Machine.id -> bool;
  mutex : Mutex.t;
  mutable exits : bool;
  holds : Mutex.t;
  mutable held : hold list;
  mutable spent : (Machine.t * string) list;
}

(* [pid] is the process that opened it: a child of fork inherits the hold, and
   it stays the parent's. [stop] is what the exit hook calls: nothing while the
   open runs. *)
and hold = {
  gpus : t;
  machine : Machine.t;
  bus : string;
  fn : Function.t;
  pid : int;
  mutable stop : unit -> unit;
}

let make ~memory_bar ~nodes ~unreleased ~teardown_ms ~reset is_gpu =
  {
    memory_bar;
    nodes;
    unreleased;
    teardown_ms;
    reset;
    is_gpu;
    mutex = Mutex.create ();
    exits = false;
    holds = Mutex.create ();
    held = [];
    spent = [];
  }

let buses g m =
  List.filter_map
    (fun (id : Machine.id) -> if g.is_gpu id then Some id.bus else None)
    (Machine.functions m)

let bus h = h.bus

(* Checks *)

let index fn i = if i < 0 then invalid_argf "Gpus.%s: GPU %d is negative" fn i

(* The bus of GPU [i] of [m], which the process does not hold. *)
let gpu g m i =
  let all = buses g m in
  match List.nth_opt all i with
  | None when all = [] -> Error "no such GPU; the machine has none"
  | None -> Error (strf "no such GPU; the machine has %d" (List.length all))
  | Some bus
    when Mutex.protect g.holds (fun () ->
             List.exists (fun h -> h.machine == m && h.bus = bus) g.held) ->
      Error (bus ^ " is open in this process")
  | Some bus -> Ok bus

(* Opening *)

let hold g m bus fn =
  { gpus = g; machine = m; bus; fn; pid = Unix.getpid (); stop = ignore }

let lost g m bus =
  Mutex.protect g.holds (fun () ->
      List.exists (fun (m', b) -> m' == m && b = bus) g.spent)

(* At exit the process stops the GPUs it holds. The hook is registered at the
   first open, after the one that turns physical takes' bus mastering off, which
   registers when the library loads: hooks run newest first, so a GPU stops
   while its function still reaches memory. The holds are read without their
   lock, which a child of fork may find held by a thread it does not have; a
   stop that gives its GPU back takes it. *)
let stop_held g =
  let pid = Unix.getpid () in
  let stop h =
    if h.pid = pid then
      try h.stop ()
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

(* [drop g h] gives back [h]'s function, under [holds]: no open or reset sees
   the GPU free with its function still taken. *)
let drop g h =
  g.held <- List.filter (fun h' -> h' != h) g.held;
  Function.release h.fn

(* The driver may give the GPU back inside the open: the open then gives back
   nothing more. *)
let open_ g m i ~at_exit f =
  index "open_" i;
  Mutex.protect g.mutex @@ fun () ->
  let* bus = gpu g m i in
  let* () =
    if lost g m bus then Error (bus ^ " was lost; reset the GPU first")
    else Ok ()
  in
  let* fn = Function.take m bus in
  let h = hold g m bus fn in
  Mutex.protect g.holds (fun () -> g.held <- h :: g.held);
  let give_back () =
    Mutex.protect g.holds (fun () -> if List.memq h g.held then drop g h)
  in
  match f h fn with
  | Ok v ->
      register_exit g;
      Mutex.protect g.holds (fun () ->
          if not (List.memq h g.held) then
            invalid_argf
              "Gpus.open_: the driver gave %s back and its open answered Ok" bus;
          h.stop <- (fun () -> at_exit v));
      Ok v
  | Error _ as e ->
      give_back ();
      e
  | exception e ->
      give_back ();
      raise e

type ending = Released | Lost

(* One step against opens and resets: none sees the GPU free with its function
   still taken, or lost before it is spent. *)
let give_back ending h =
  let g = h.gpus in
  Mutex.protect g.holds @@ fun () ->
  if not (List.memq h g.held) then
    invalid_argf "Gpus.%s: %s was given back already"
      (match ending with Released -> "release" | Lost -> "lose")
      h.bus;
  if ending = Lost then g.spent <- (h.machine, h.bus) :: g.spent;
  drop g h

let release h = give_back Released h
let lose h = give_back Lost h

(* Changes to the machine *)

(* [f files bus] for GPU [i] of [m], whose files are [files], which no process
   takes while it runs: the lock a take holds is held around [f]. *)
let change g fn m i f =
  index fn i;
  Mutex.protect g.mutex @@ fun () ->
  match Machine.files m with
  | None ->
      Error
        (strf "%s is reached through a transport; change its GPUs there"
           (Option.value (Machine.name m) ~default:"the machine"))
  | Some files ->
      let* bus = gpu g m i in
      Local.locked files bus (fun () -> Fail.result (fun () -> f files bus))

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

(* A kernel driver lets go of a GPU, writing to it as it does, once the last
   file of its devices goes: a DRM driver's unbind returns first. So detach
   unbinds a driver only once no file remains, refusing at once a file a process
   holds, this one's included, which it would wait for itself, and the driver
   lets go inside the unbind; the vendor confirms it after. A GPU on vfio-pci
   stays as it is, and an unbound one waits for the vendor. *)
let detach g m i =
  change g "detach" m i (fun files bus ->
      let nodes = g.nodes ~root:(Sysfs.root files) bus in
      Option.iter
        (Fail.fail "%s is open in this process, through %s" bus)
        (own files bus nodes);
      (match Sysfs.driver files bus with
      | Some d when d = Sysfs.vfio_pci -> Sysfs.detach files bus
      | driver ->
          Option.iter (Fail.fail "%s") (Sysfs.refusal files bus);
          let bound = Option.is_some driver in
          let_go g files bus nodes ~bound;
          Sysfs.detach files bus;
          if bound then
            Option.iter
              (Fail.fail
                 "%s is detached, but its kernel driver has not let go of it: \
                  %s"
                 bus)
              (g.unreleased ~root:(Sysfs.root files) bus));
      Sysfs.resize files bus g.memory_bar)

(* Resets *)

(* Takes the function of the GPU at [bus] on [m], turns its bus mastering off
   and resets it as its vendor does, releasing it whatever the reset answers. A
   GPU lost and reset opens again. *)
let reset_gpu g m bus =
  let* fn = Function.take m bus in
  let r =
    Fun.protect ~finally:(fun () -> Function.release fn) @@ fun () ->
    let c = Function.config16 fn Local.command in
    Function.set_config16 fn Local.command (c land lnot Local.bus_master);
    let r = g.reset fn in
    (* Still taken, so no process holds the function: every file naming it was
       left by one that died, and the GPU reaches none of it now. *)
    if Result.is_ok r then
      Option.iter
        (fun files -> Sysmem.forget ~root:(Sysfs.root files) ~bus)
        (Machine.files m);
    r
  in
  if Result.is_ok r then
    Mutex.protect g.holds (fun () ->
        g.spent <- List.filter (fun (m', b) -> not (m' == m && b = bus)) g.spent);
  r

let reset g m i =
  index "reset" i;
  Mutex.protect g.mutex @@ fun () ->
  let* bus = gpu g m i in
  reset_gpu g m bus

(* A kernel driver's probe expects the GPU as its vendor's reset leaves it,
   whatever ran on it before, in this process or another: a GPU on no kernel
   driver, or on vfio-pci, is reset first, through a take, while the process can
   take it. The take shares the change's lock, so no other comes between. *)
let attach g m i =
  change g "attach" m i (fun files bus ->
      match Sysfs.driver files bus with
      | Some d when d <> Sysfs.vfio_pci -> ()
      | _ ->
          Result.iter_error (Fail.fail "%s") (reset_gpu g m bus);
          Sysfs.attach files bus)
