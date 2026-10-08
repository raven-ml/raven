(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let ( let* ) = Result.bind

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
  maps : (Window.t, int * bool) Hashtbl.t; (* to its BAR and [combine] *)
  dmas : (Window.t, unit) Hashtbl.t;
  pins : (int * int, unit) Hashtbl.t; (* (address, bytes) *)
}

(* A bus is parsed before it names a file of the machine's. *)
let take machine bus =
  let taken =
    match Machine.failed machine with
    | Some why -> Error why
    | None when Option.is_none (Bus_address.numbers bus) ->
        Error (strf "%S is no PCI bus address, expected DDDD:BB:DD.F" bus)
    | None -> Machine.take machine bus
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
let live f fn = if f.released then Fail.err_released fn f.bus

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
  if off < 0 || off > config_size - n then
    invalid_argf
      "Function.%s: %d bytes at %d outside the %d bytes of configuration space"
      fn n off config_size

let config8 f off =
  in_config f "config8" off 1;
  f.fn.config8 off

let config16 f off =
  in_config f "config16" off 2;
  f.fn.config16 off

let config32 f off =
  in_config f "config32" off 4;
  f.fn.config32 off

let set_config8 f off x =
  in_config f "set_config8" off 1;
  f.fn.set_config8 off x

let set_config16 f off x =
  in_config f "set_config16" off 2;
  f.fn.set_config16 off x

let set_config32 f off x =
  in_config f "set_config32" off 4;
  f.fn.set_config32 off x

(* BARs *)

let index f fn i =
  live f fn;
  if i < 0 then invalid_argf "Function.%s: BAR %d is negative" fn i

let bar f i =
  index f "bar" i;
  f.fn.bar i

let map ?(combine = false) ?(off = 0) ?length f i =
  index f "map" i;
  let size =
    match f.fn.bar i with
    | Some (_, size) -> size
    | None -> invalid_argf "Function.map: %s has no BAR %d" f.bus i
  in
  let length = Option.value length ~default:(size - off) in
  if off < 0 || length < 0 || off > size - length then
    invalid_argf "Function.map: %d bytes at %d outside BAR %d of %d bytes"
      length off i size;
  (* The processor maps a BAR's addresses one way at a time: x86's PAT refuses a
     second mapping of another type, or makes both uncached. Only the owner
     changes [maps], so it reads them without the lock. *)
  Hashtbl.iter
    (fun _ (j, c) ->
      if j = i && c <> combine then
        invalid_argf
          "Function.map: a live window maps BAR %d of %s with combine:%b" i
          f.bus c)
    f.maps;
  let* w = f.fn.map ~combine i off length in
  Mutex.protect f.lock (fun () -> Hashtbl.add f.maps w (i, combine));
  Ok w

(* Removes one binding of the live window [w] from [table], or refuses [w]. *)
let forget f table fn w =
  Mutex.protect f.lock @@ fun () ->
  if not (Hashtbl.mem table w) then
    invalid_argf "Function.%s: no such window of %s" fn f.bus;
  Hashtbl.remove table w

let unmap f w =
  forget f f.maps "unmap" w;
  f.fn.unmap w

(* Interrupts and reset *)

let interrupt f ms =
  live f "interrupt";
  if ms < 0 then invalid_argf "Function.interrupt: %d ms is negative" ms;
  f.fn.interrupt ms

(* A function answers again once its vendor ID reads other than all ones. *)
let absent = 0xffff
let reset_ms = 1000

let failed f =
  live f "failed";
  match Machine.failed f.machine with
  | Some _ as why -> why
  | None when f.fn.config16 0 = absent ->
      Some (strf "%s left the bus: its vendor ID reads 0xffff" f.bus)
  | None -> None

let reset f =
  live f "reset";
  let* () = f.fn.reset () in
  let answers () = f.fn.config16 0 <> absent in
  if Machine.wait f.machine ~ms:reset_ms answers then Ok ()
  else
    match Machine.failed f.machine with
    | Some why -> Error why
    | None ->
        Error (strf "%s does not answer %d ms after its reset" f.bus reset_ms)

(* System memory *)

(* Contiguous memory is at most a huge page, of 2 MiB. *)
let huge = 2 lsl 20

let on_page f fn a =
  if a mod Machine.page f.machine <> 0 then
    invalid_argf "Function.%s: 0x%x is not on a %d-byte page" fn a
      (Machine.page f.machine)

let alloc_dma ?(contiguous = false) ?va f n =
  live f "alloc_dma";
  let page = Machine.page f.machine in
  if n <= 0 || n > max_int - page then
    invalid_argf "Function.alloc_dma: %d bytes, expected 1 to %d" n
      (max_int - page);
  let n = (n + page - 1) / page * page in
  if contiguous && n > huge then
    invalid_argf
      "Function.alloc_dma: %d bytes of contiguous memory, expected at most 2 \
       MiB"
      n;
  (* Reached physically, contiguous memory larger than a page is a huge page,
     which maps whole at [va]. *)
  let huge_page =
    contiguous && n > page && f.fn.addressing = Machine.Physical
  in
  let mapped = if huge_page then huge else n in
  (match va with
  | None -> ()
  | Some va ->
      on_page f "alloc_dma" va;
      if huge_page && va mod huge <> 0 then
        invalid_argf
          "Function.alloc_dma: 0x%x is not on 2 MiB, which a huge page needs" va;
      if not (Machine.reserved f.machine va mapped) then
        invalid_argf
          "Function.alloc_dma: %d bytes at 0x%x lie in no range \
           Machine.reserve reserved"
          mapped va);
  let* ((w, _) as dma) = f.fn.alloc_dma ~contiguous ~va n in
  Mutex.protect f.lock (fun () -> Hashtbl.add f.dmas w ());
  Ok dma

let free_dma f w =
  forget f f.dmas "free_dma" w;
  f.fn.free_dma w

let pin f a n =
  live f "pin";
  on_page f "pin" a;
  if n <= 0 then invalid_argf "Function.pin: %d bytes, expected more than 0" n;
  let* runs = f.fn.pin a n in
  Mutex.protect f.lock (fun () -> Hashtbl.add f.pins (a, n) ());
  Ok runs

let unpin f a n =
  Mutex.protect f.lock (fun () ->
      if not (Hashtbl.mem f.pins (a, n)) then
        invalid_argf "Function.unpin: 0x%x is not pinned for %s" a f.bus;
      Hashtbl.remove f.pins (a, n));
  f.fn.unpin a n
