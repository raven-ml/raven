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
   The tables, [released] and [users], the pins and allocations from other
   domains that run, are used from any domain under [lock]; [idle] signals the
   last user's end. *)
type t = {
  machine : Machine.t;
  bus : string;
  fn : Machine.fn;
  released : bool Atomic.t; (* set under [lock] *)
  mutable users : int;
  lock : Mutex.t;
  idle : Condition.t;
  maps : (int * bool) Tables.Window.t; (* to its BAR and [combine] *)
  dmas : unit Tables.Window.t;
  pins : unit Tables.Range.t; (* (address, bytes) *)
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
        released = Atomic.make false;
        users = 0;
        lock = Mutex.create ();
        idle = Condition.create ();
        maps = Tables.Window.create 16;
        dmas = Tables.Window.create 16;
        pins = Tables.Range.create 16;
      })
    taken

let machine f = f.machine
let bus f = f.bus
let addressing f = f.fn.addressing
let released f = Atomic.get f.released
let live f fn = if released f then Fail.err_released fn f.bus

(* A pin or an allocation, which any domain may make while the owner releases
   the function, runs between [enter] and [leave]: the release waits until none
   runs, and none starts once it began, so the machine sees none after the
   release. [leave_locked] holds [lock], as the table a pin or an allocation
   records it in needs. *)
let enter f fn =
  Mutex.lock f.lock;
  if Atomic.get f.released then begin
    Mutex.unlock f.lock;
    Fail.err_released fn f.bus
  end;
  f.users <- f.users + 1;
  Mutex.unlock f.lock

let leave_locked f =
  f.users <- f.users - 1;
  if f.users = 0 && Atomic.get f.released then Condition.broadcast f.idle

let leave f =
  Mutex.lock f.lock;
  leave_locked f;
  Mutex.unlock f.lock

let release f =
  let maps =
    Mutex.protect f.lock (fun () ->
        if Atomic.get f.released then None
        else begin
          Atomic.set f.released true;
          while f.users > 0 do
            Condition.wait f.idle f.lock
          done;
          let ws = List.of_seq (Tables.Window.to_seq_keys f.maps) in
          Tables.Window.reset f.maps;
          Some ws
        end)
  in
  Option.iter
    (fun maps ->
      List.iter f.fn.unmap maps;
      f.fn.release ())
    maps

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

let map ?combine ?(off = 0) ?length f i =
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
     second mapping of another type, or makes both uncached. A window asked no
     way takes the way of the BAR's live windows. Only the owner changes [maps],
     so it reads them without the lock; a GPU holds a handful of windows, so the
     walk costs less than the mapping it precedes. *)
  let live =
    Tables.Window.fold
      (fun _ (j, c) w -> if j = i then Some c else w)
      f.maps None
  in
  let combine =
    match (combine, live) with
    | Some c, Some c' when c <> c' ->
        invalid_argf
          "Function.map: a live window maps BAR %d of %s with combine:%b" i
          f.bus c'
    | Some c, _ | None, Some c -> c
    | None, None -> false
  in
  let* w = f.fn.map ~combine i off length in
  Mutex.protect f.lock (fun () -> Tables.Window.add f.maps w (i, combine));
  Ok w

(* Removes one binding of the live window [w] from [table], or refuses [w]. *)
let forget f table fn w =
  Mutex.protect f.lock @@ fun () ->
  if not (Tables.Window.mem table w) then
    invalid_argf "Function.%s: no such window of %s" fn f.bus;
  Tables.Window.remove table w

let unmap f w =
  forget f f.maps "unmap" w;
  f.fn.unmap w

(* Interrupts and reset *)

let interrupt f ms =
  live f "interrupt";
  if ms < 0 then invalid_argf "Function.interrupt: %d ms is negative" ms;
  f.fn.interrupt ms

(* A function answers again once its vendor ID reads other than all ones, within
   1 s of its reset: software waits 100 ms, and the function may have
   configuration requests retried until 1 s has passed (PCI Express Base
   Specification, 6.6.1). *)
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

(* Contiguous memory is at most a huge page. *)
let huge = Sysmem.huge

let on_page f fn a =
  if a mod Machine.page f.machine <> 0 then
    invalid_argf "Function.%s: 0x%x is not on a %d-byte page" fn a
      (Machine.page f.machine)

let alloc_dma_entered ~contiguous ?va f n =
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
  (* Reached physically, memory lies in huge pages, each mapped whole on the 2
     MiB block of addresses that holds it; contiguous memory larger than a page
     is one of them. *)
  let physical = f.fn.addressing = Machine.Physical in
  (match va with
  | None -> ()
  | Some va ->
      on_page f "alloc_dma" va;
      if physical && contiguous && n > page && va mod huge <> 0 then
        invalid_argf
          "Function.alloc_dma: 0x%x is not on 2 MiB, which a huge page needs" va;
      let lo, hi =
        if physical then (va / huge * huge, (va + n + huge - 1) / huge * huge)
        else (va, va + n)
      in
      if not (Machine.reserved f.machine lo (hi - lo)) then
        invalid_argf
          "Function.alloc_dma: %d bytes at 0x%x lie in no range \
           Machine.reserve reserved%s"
          n va
          (if physical then ", whole with the 2 MiB blocks that hold them"
           else ""));
  f.fn.alloc_dma ~contiguous ~va n

let alloc_dma ?(contiguous = false) ?va f n =
  enter f "alloc_dma";
  match alloc_dma_entered ~contiguous ?va f n with
  | Ok (Some (w, _)) as dma ->
      Mutex.lock f.lock;
      Tables.Window.add f.dmas w ();
      leave_locked f;
      Mutex.unlock f.lock;
      dma
  | (Ok None | Error _) as r ->
      leave f;
      r
  | exception e ->
      leave f;
      raise e

let free_dma f w =
  forget f f.dmas "free_dma" w;
  f.fn.free_dma w

let pin_entered f a n =
  on_page f "pin" a;
  if n <= 0 then invalid_argf "Function.pin: %d bytes, expected more than 0" n;
  f.fn.pin a n

let pin f a n =
  enter f "pin";
  match pin_entered f a n with
  | Ok _ as runs ->
      Mutex.lock f.lock;
      Tables.Range.add f.pins (a, n) ();
      leave_locked f;
      Mutex.unlock f.lock;
      runs
  | Error _ as e ->
      leave f;
      e
  | exception e ->
      leave f;
      raise e

let unpin f a n =
  Mutex.protect f.lock (fun () ->
      if not (Tables.Range.mem f.pins (a, n)) then
        invalid_argf "Function.unpin: 0x%x is not pinned for %s" a f.bus;
      Tables.Range.remove f.pins (a, n));
  f.fn.unpin a n
