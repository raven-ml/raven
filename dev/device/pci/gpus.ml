(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let ( let* ) = Result.bind

type interface = Kernel | Pci

(* A vendor's GPUs. [mutex] serializes opens, resets and changes, drivers
   included, and guards [interface], how this machine's GPUs are reached, fixed
   by the first successful open. Only opens add holds and only resets clear lost
   GPUs, so what they check stays true while they run. [holds] guards the GPUs
   held and those lost over PCI, which open again only after a reset; it is held
   briefly, so that giving a GPU back waits for no driver. A GPU is named by its
   machine and bus address. *)
type t = {
  memory_bar : int;
  is_gpu : Machine.id -> bool;
  mutex : Mutex.t;
  mutable interface : interface option;
  holds : Mutex.t;
  mutable held : hold list;
  mutable spent : (Machine.t * string) list;
}

and hold = {
  gpus : t;
  machine : Machine.t;
  bus : string;
  fn : Function.t option;
}

let make ~memory_bar is_gpu =
  {
    memory_bar;
    is_gpu;
    mutex = Mutex.create ();
    interface = None;
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
  | None -> Error (strf "no such GPU; the machine has %d" (List.length all))
  | Some bus
    when Mutex.protect g.holds (fun () ->
             List.exists (fun h -> h.machine == m && h.bus = bus) g.held) ->
      Error (bus ^ " is open in this process")
  | Some bus -> Ok bus

let interface_name = function
  | Kernel -> "through their kernel driver"
  | Pci -> "over PCI"

let refuse_interface g wanted =
  match g.interface with
  | Some c when c <> wanted ->
      Error
        (strf
           "this process reaches these GPUs %s; open this one that way, or \
            from another process"
           (interface_name c))
  | _ -> Ok ()

(* Opening *)

let hold g m bus fn = { gpus = g; machine = m; bus; fn }

let lost g m bus =
  Mutex.protect g.holds (fun () ->
      List.exists (fun (m', b) -> m' == m && b = bus) g.spent)

let keep g m interface h =
  if m == Machine.this then g.interface <- Some interface;
  Mutex.protect g.holds (fun () -> g.held <- h :: g.held)

let open_kernel g m i f =
  index "open_kernel" i;
  Mutex.protect g.mutex @@ fun () ->
  let* () =
    if m == Machine.this then Ok ()
    else Error "another machine's GPUs are reached over PCI only"
  in
  let* () = refuse_interface g Kernel in
  let* bus = gpu g m i in
  let h = hold g m bus None in
  let* v = f h in
  keep g m Kernel h;
  Ok v

let open_pci g m i f =
  index "open_pci" i;
  Mutex.protect g.mutex @@ fun () ->
  let* () = if m == Machine.this then refuse_interface g Pci else Ok () in
  let* bus = gpu g m i in
  let* () =
    if lost g m bus then Error (bus ^ " was lost; reset the GPU first")
    else Ok ()
  in
  let* fn = Function.take m bus in
  let h = hold g m bus (Some fn) in
  match f h fn with
  | Ok v ->
      keep g m Pci h;
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
  if not (List.memq h g.held) then
    invalid_argf "Gpus.%s: %s was given back already"
      (match ending with Released -> "release" | Lost -> "lose")
      h.bus;
  g.held <- List.filter (fun h' -> h' != h) g.held;
  if ending = Lost && Option.is_some h.fn then
    g.spent <- (h.machine, h.bus) :: g.spent;
  Option.iter Function.release h.fn

let release h = give_back Released h
let lose h = give_back Lost h

(* Changes to the machine *)

(* [f host bus] for GPU [i] of [m], whose files are [host], which no process
   takes while it runs: the lock a take holds is held around [f]. *)
let change g fn m i f =
  index fn i;
  Mutex.protect g.mutex @@ fun () ->
  match Machine.host m with
  | None ->
      Error
        (strf "%s is reached through a transport; change its GPUs there"
           (Option.value (Machine.name m) ~default:"the machine"))
  | Some host ->
      let* bus = gpu g m i in
      Local.locked host bus (fun () -> Fail.result (fun () -> f host bus))

let detach g m i =
  change g "detach" m i (fun host bus ->
      Sysfs.detach host bus;
      Sysfs.resize host bus g.memory_bar)

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
