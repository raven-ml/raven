(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type interface = Kernel | Pci

(* A vendor's GPUs and the process's state for them, under [mutex]: the
   interface this machine's GPUs are reached through, fixed by the first
   successful open; the GPUs held; and those lost over PCI, which open again
   only after a reset. A GPU is named by its machine and bus address. *)
type t = {
  name : string;
  lock : string;
  memory_bar : int;
  is_gpu : Machine.id -> bool;
  mutex : Mutex.t;
  mutable interface : interface option;
  mutable held : hold list;
  mutable spent : (Machine.t * string) list;
}

and hold = {
  gpus : t;
  machine : Machine.t;
  bus : string;
  fn : Function.t option;
  mutable given_back : bool;
}

let make ~name ~lock ~memory_bar is_gpu =
  {
    name;
    lock;
    memory_bar;
    is_gpu;
    mutex = Mutex.create ();
    interface = None;
    held = [];
    spent = [];
  }

let buses g m =
  List.filter_map
    (fun (id : Machine.id) -> if g.is_gpu id then Some id.bus else None)
    (Machine.functions m)

let bus h = h.bus

(* Checks *)

let linux () =
  match
    In_channel.with_open_text "/proc/sys/kernel/ostype" In_channel.input_all
  with
  | s -> String.trim s = "Linux"
  | exception Sys_error _ -> false

let index fn i =
  if i < 0 then invalid_arg (Printf.sprintf "Gpus.%s: GPU %d" fn i)

let ( let* ) = Result.bind

(* The bus of GPU [i] of [m], which the process does not hold. *)
let gpu g m i =
  let all = buses g m in
  match List.nth_opt all i with
  | None ->
      Error
        (Printf.sprintf "no GPU %d; there are %d %s GPUs" i (List.length all)
           g.name)
  | Some bus when List.exists (fun h -> h.machine == m && h.bus = bus) g.held ->
      Error (bus ^ " is open in this process")
  | Some bus -> Ok bus

(* [f ()], with the failures the world causes as [Error]s. *)
let caught f =
  match f () with
  | r -> r
  | exception (Failure why | Sys_error why) -> Error why
  | exception Unix.Unix_error (e, fn, arg) ->
      Error (Printf.sprintf "%s %s: %s" fn arg (Unix.error_message e))

let interface_name = function Kernel -> "their kernel driver" | Pci -> "PCI"

let refuse_interface g wanted =
  match g.interface with
  | Some c when c <> wanted ->
      Error
        (Printf.sprintf "this process reaches %s GPUs through %s, not %s" g.name
           (interface_name c) (interface_name wanted))
  | _ -> Ok ()

(* Opening *)

let hold g m bus fn = { gpus = g; machine = m; bus; fn; given_back = false }

let keep g m interface h =
  if m == Machine.this then g.interface <- Some interface;
  g.held <- h :: g.held

let open_kernel g m i f =
  index "open_kernel" i;
  Mutex.protect g.mutex @@ fun () ->
  let* () =
    if m != Machine.this then
      Error
        (Printf.sprintf "another machine's %s GPUs are reached over PCI" g.name)
    else if not (linux ()) then
      Error (Printf.sprintf "%s GPUs need Linux" g.name)
    else Ok ()
  in
  let* () = refuse_interface g Kernel in
  let* bus = gpu g m i in
  let h = hold g m bus None in
  let* v = caught (fun () -> f h) in
  keep g m Kernel h;
  Ok v

let open_pci g m i f =
  index "open_pci" i;
  Mutex.protect g.mutex @@ fun () ->
  let* () = if m == Machine.this then refuse_interface g Pci else Ok () in
  let* bus = gpu g m i in
  let* () =
    if List.exists (fun (m', b) -> m' == m && b = bus) g.spent then
      Error (bus ^ " was lost; over PCI only a reset recovers it")
    else Ok ()
  in
  let* fn = Function.take m ~lock:g.lock bus in
  let h = hold g m bus (Some fn) in
  match caught (fun () -> f h fn) with
  | Ok v ->
      keep g m Pci h;
      Ok v
  | Error _ as e ->
      Function.release fn;
      e
  | exception e ->
      Function.release fn;
      raise e

let give_back fn h =
  let g = h.gpus in
  Mutex.protect g.mutex @@ fun () ->
  if h.given_back then
    invalid_arg (Printf.sprintf "Gpus.%s: %s was given back already" fn h.bus);
  h.given_back <- true;
  g.held <- List.filter (fun h' -> h' != h) g.held;
  Option.iter Function.release h.fn

let release h = give_back "release" h

let lose h =
  give_back "lose" h;
  if Option.is_some h.fn then
    Mutex.protect h.gpus.mutex (fun () ->
        h.gpus.spent <- (h.machine, h.bus) :: h.gpus.spent)

(* Changes to the machine *)

(* [f bus] for GPU [i] of this machine, which no process holds while it runs:
   the library's own lock on it is taken around [f]. *)
let change g fn i f =
  index fn i;
  Mutex.protect g.mutex @@ fun () ->
  let* () =
    if linux () then Ok ()
    else Error (Printf.sprintf "%s GPUs need Linux" g.name)
  in
  let* bus = gpu g Machine.this i in
  let* lock = Local.lock bus "nx" in
  Fun.protect
    ~finally:(fun () -> Local.close lock)
    (fun () -> caught (fun () -> Ok (f bus)))

let detach g i =
  change g "detach" i (fun bus ->
      Sysfs.detach bus;
      Sysfs.resize bus g.memory_bar)

let attach g i = change g "attach" i Sysfs.attach

let reset g m i f =
  index "reset" i;
  Mutex.protect g.mutex @@ fun () ->
  let* bus = gpu g m i in
  let* fn = Function.take m ~lock:g.lock bus in
  let r =
    Fun.protect
      ~finally:(fun () -> Function.release fn)
      (fun () -> caught (fun () -> Ok (f fn)))
  in
  if Result.is_ok r then
    g.spent <- List.filter (fun (m', b) -> not (m' == m && b = bus)) g.spent;
  r
