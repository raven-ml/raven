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
   while they run. [holds] guards the GPUs held, each with how it stops at exit,
   and those lost, which open again only after a reset; it is held briefly, so
   that giving a GPU back waits for no driver. A GPU is named by its machine and
   bus address. *)
type t = {
  memory_bar : int;
  is_gpu : Machine.id -> bool;
  mutex : Mutex.t;
  mutable exits : bool;
  holds : Mutex.t;
  mutable held : (hold * (unit -> unit)) list;
  mutable spent : (Machine.t * string) list;
}

(* [pid] is the process that opened it: a child of fork inherits the hold, and
   it stays the parent's. *)
and hold = {
  gpus : t;
  machine : Machine.t;
  bus : string;
  fn : Function.t;
  pid : int;
}

let make ~memory_bar is_gpu =
  {
    memory_bar;
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
             List.exists (fun (h, _) -> h.machine == m && h.bus = bus) g.held)
    ->
      Error (bus ^ " is open in this process")
  | Some bus -> Ok bus

(* Opening *)

let hold g m bus fn = { gpus = g; machine = m; bus; fn; pid = Unix.getpid () }

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
  let stop (h, at_exit) =
    if h.pid = pid then
      try at_exit ()
      with e ->
        prerr_endline
          (strf "stopping %s at exit: %s" h.bus (Printexc.to_string e))
  in
  List.iter stop g.held

let keep g h at_exit =
  if not g.exits then begin
    g.exits <- true;
    Stdlib.at_exit (fun () -> stop_held g)
  end;
  Mutex.protect g.holds (fun () -> g.held <- (h, at_exit) :: g.held)

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
  match f h fn with
  | Ok v ->
      keep g h (fun () -> at_exit v);
      Ok v
  | Error _ as e ->
      Function.release fn;
      e
  | exception e ->
      Function.release fn;
      raise e

type ending = Released | Lost

(* One step against opens and resets: none sees the GPU free with its function
   still taken, or lost before it is spent. *)
let give_back ending h =
  let g = h.gpus in
  Mutex.protect g.holds @@ fun () ->
  if not (List.exists (fun (h', _) -> h' == h) g.held) then
    invalid_argf "Gpus.%s: %s was given back already"
      (match ending with Released -> "release" | Lost -> "lose")
      h.bus;
  g.held <- List.filter (fun (h', _) -> h' != h) g.held;
  if ending = Lost then g.spent <- (h.machine, h.bus) :: g.spent;
  Function.release h.fn

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

let detach g m i =
  change g "detach" m i (fun files bus ->
      Sysfs.detach files bus;
      Sysfs.resize files bus g.memory_bar)

let attach g m i = change g "attach" m i Sysfs.attach

let reset g m i f =
  index "reset" i;
  Mutex.protect g.mutex @@ fun () ->
  let* bus = gpu g m i in
  let* fn = Function.take m bus in
  let r =
    Fun.protect ~finally:(fun () -> Function.release fn) (fun () -> f fn)
  in
  if Result.is_ok r then
    Mutex.protect g.holds (fun () ->
        g.spent <- List.filter (fun (m', b) -> not (m' == m && b = bus)) g.spent);
  r
