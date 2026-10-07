(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What the process holds of a function, so that misuse is refused before its
   machine is asked: its BAR windows, DMA memory and pins. Windows are values,
   so each table binds a window or a pinned range once for each time it is live.
   The tables are used from any domain under [lock]. *)
type t = {
  machine : Machine.t;
  bus : string;
  fn : Machine.fn;
  mutable released : bool;
  lock : Mutex.t;
  maps : (Window.t, unit) Hashtbl.t;
  dmas : (Window.t, unit) Hashtbl.t;
  pins : (int * int, unit) Hashtbl.t; (* (address, bytes) *)
}

(* A bus is parsed before it names a file of the machine's. *)
let take machine bus =
  let taken =
    match Machine.failed machine with
    | Some why -> Error why
    | None when Option.is_none (Address.numbers bus) ->
        Error (Printf.sprintf "%S is no PCI bus address" bus)
    | None -> ( try Machine.take machine bus with Failure why -> Error why)
  in
  Result.map
    (fun fn ->
      {
        machine;
        bus;
        fn;
        released = false;
        lock = Mutex.create ();
        maps = Hashtbl.create 16;
        dmas = Hashtbl.create 16;
        pins = Hashtbl.create 16;
      })
    taken

let machine f = f.machine
let bus f = f.bus
let addressing f = f.fn.addressing
let released f = f.released

let live f fn =
  if f.released then
    invalid_arg (Printf.sprintf "Function.%s: %s is released" fn f.bus)

let release f =
  if not f.released then begin
    f.released <- true;
    let maps =
      Mutex.protect f.lock (fun () ->
          let ws = List.of_seq (Hashtbl.to_seq_keys f.maps) in
          Hashtbl.reset f.maps;
          ws)
    in
    List.iter f.fn.unmap maps;
    f.fn.release ()
  end

(* Configuration space *)

(* The configuration space of a PCI Express function, in bytes. *)
let config_size = 4096

let in_config f fn off n =
  live f fn;
  if n <> 1 && n <> 2 && n <> 4 then
    invalid_arg (Printf.sprintf "Function.%s: %d bytes" fn n);
  if off < 0 || off > config_size - n then
    invalid_arg
      (Printf.sprintf "Function.%s: byte %d outside configuration space" fn off)

let config f off n =
  in_config f "config" off n;
  f.fn.config off n

let set_config f off n x =
  in_config f "set_config" off n;
  f.fn.set_config off n x

(* BARs *)

let index f fn i =
  live f fn;
  if i < 0 then invalid_arg (Printf.sprintf "Function.%s: BAR %d" fn i)

let bar f i =
  index f "bar" i;
  f.fn.bar i

let map ?(off = 0) ?length f i =
  index f "map" i;
  let size =
    match f.fn.bar i with
    | Some (_, size) -> size
    | None ->
        invalid_arg (Printf.sprintf "Function.map: %s has no BAR %d" f.bus i)
  in
  let length = Option.value length ~default:(size - off) in
  if off < 0 || length < 0 || off > size - length then
    invalid_arg
      (Printf.sprintf "Function.map: %d bytes at %d outside BAR %d of %d bytes"
         length off i size);
  let w = f.fn.map i off length in
  Mutex.protect f.lock (fun () -> Hashtbl.add f.maps w ());
  w

(* Removes one binding of the live window [w] from [table], or refuses [w]. *)
let forget f table fn w =
  Mutex.protect f.lock @@ fun () ->
  if not (Hashtbl.mem table w) then
    invalid_arg (Printf.sprintf "Function.%s: no such window of %s" fn f.bus);
  Hashtbl.remove table w

let unmap f w =
  forget f f.maps "unmap" w;
  f.fn.unmap w

(* Interrupts and reset *)

let interrupt f ms =
  live f "interrupt";
  if ms < 0 then invalid_arg (Printf.sprintf "Function.interrupt: %d ms" ms);
  f.fn.interrupt ms

(* A function answers again once its vendor ID reads other than all ones. *)
let absent = 0xffff
let reset_ms = 1000

let reset f =
  live f "reset";
  f.fn.reset ();
  let answers () = f.fn.config 0 2 <> absent in
  if not (Machine.wait f.machine ~ms:reset_ms answers) then
    failwith (Printf.sprintf "%s does not answer after its reset" f.bus)

(* System memory *)

(* Contiguous memory is at most a huge page, of 2 MiB. *)
let huge = 2 lsl 20

let on_page f fn a =
  if a mod Machine.page f.machine <> 0 then
    invalid_arg (Printf.sprintf "Function.%s: 0x%x is not on a page" fn a)

let alloc_dma ?(contiguous = false) ?va f n =
  live f "alloc_dma";
  let page = Machine.page f.machine in
  if n <= 0 || n > max_int - page then
    invalid_arg (Printf.sprintf "Function.alloc_dma: %d bytes" n);
  let n = (n + page - 1) / page * page in
  if contiguous && n > huge then
    invalid_arg "Function.alloc_dma: contiguous memory is at most 2 MiB";
  Option.iter (on_page f "alloc_dma") va;
  (match va with
  | Some va
    when contiguous && n > page
         && f.fn.addressing = Machine.Physical
         && va mod huge <> 0 ->
      invalid_arg (Printf.sprintf "Function.alloc_dma: 0x%x is not on 2 MiB" va)
  | _ -> ());
  let ((w, _) as dma) = f.fn.alloc_dma ~contiguous ~va n in
  Mutex.protect f.lock (fun () -> Hashtbl.add f.dmas w ());
  dma

let free_dma f w =
  forget f f.dmas "free_dma" w;
  f.fn.free_dma w

let pin f a n =
  live f "pin";
  on_page f "pin" a;
  if n <= 0 then invalid_arg (Printf.sprintf "Function.pin: %d bytes" n);
  let runs = f.fn.pin a n in
  Mutex.protect f.lock (fun () -> Hashtbl.add f.pins (a, n) ());
  runs

let unpin f a n =
  Mutex.protect f.lock (fun () ->
      if not (Hashtbl.mem f.pins (a, n)) then
        invalid_arg
          (Printf.sprintf "Function.unpin: 0x%x is not pinned for %s" a f.bus);
      Hashtbl.remove f.pins (a, n));
  f.fn.unpin a n
