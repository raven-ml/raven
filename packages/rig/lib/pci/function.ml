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
   The tables are used from any domain under [lock]; [users] counts the pins,
   allocations and reads at a wait's bound that run. *)
type t = {
  machine : Machine.t;
  bus : string;
  fn : Machine.fn;
  inherited : bool Atomic.t; (* until [forget] *)
  released : bool Atomic.t;
  dma_end : int Atomic.t;
      (* the end of the addresses it reaches, [2^n] of [set_dma_bits], read
         from any domain *)
  users : int Atomic.t;
  lock : Lock.t;
  maps : (int * bool) Tables.Window.t; (* to its BAR and [combine] *)
  dmas : unit Tables.Window.t;
  pins : unit Tables.Range.t; (* (address, bytes) *)
}

(* A function reaches 64 bits of address until its driver says otherwise. *)
let full_bits = 64

let make machine bus taken =
  Result.map
    (fun fn ->
      {
        machine;
        bus;
        fn;
        inherited = Atomic.make fn.inherited;
        released = Atomic.make false;
        dma_end = Atomic.make max_int;
        users = Atomic.make 0;
        lock = Lock.create ();
        maps = Tables.Window.create 16;
        dmas = Tables.Window.create 16;
        pins = Tables.Range.create 16;
      })
    taken

let take machine bus =
  make machine bus
  @@
  match Machine.failed machine with
  | Some why -> Error why
  | None -> Machine.take machine bus

let take_locked machine l =
  make machine (Sysfs.bus l) (Machine.take_locked machine l)

let machine f = f.machine
let bus f = f.bus
let addressing f = f.fn.addressing
let released f = Atomic.get f.released
let live f fn = if released f then Fail.err_released fn f.bus

(* A pin, an allocation or a read at a wait's bound, which any domain may make
   while the owner releases the function, counts itself in [users] from
   [enter] to [leave]. Each is one atomic operation, which allocates nothing,
   so no asynchronous exception splits it ([Atomic.incr] and [Atomic.decr]
   allocate the location they add to first), and a use's handler starts as
   [enter] returns and calls [leave] first. A use counts itself before it
   reads [released], and the release sets [released] before it reads the
   count, so the release sees every use that started, and none starts after
   it. *)
let count f n = ignore (Atomic.fetch_and_add f.users n)

let enter f fn =
  count f 1;
  if Atomic.get f.released then begin
    count f (-1);
    Fail.err_released fn f.bus
  end

let leave f = count f (-1)

external reraise : exn -> 'a = "%reraise"

(* The release naps until no use runs. A wake-up from [leave] would take a
   lock or a call that an asynchronous exception can end before it wakes the
   release; a use is one call of the machine, so the naps are few. *)
let nap_s = 0.0001

let release f =
  if not (Atomic.exchange f.released true) then begin
    while Atomic.get f.users > 0 do
      Unix.sleepf nap_s
    done;
    let maps =
      Lock.protect f.lock (fun () ->
          let ws = List.of_seq (Tables.Window.to_seq_keys f.maps) in
          Tables.Window.reset f.maps;
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
  f.fn.config off 1

let config16 f off =
  in_config f "config16" off 2;
  f.fn.config off 2

let config32 f off =
  in_config f "config32" off 4;
  f.fn.config off 4

let set_config8 f off x =
  in_config f "set_config8" off 1;
  f.fn.set_config off 1 x

let set_config16 f off x =
  in_config f "set_config16" off 2;
  f.fn.set_config off 2 x

let set_config32 f off x =
  in_config f "set_config32" off 4;
  f.fn.set_config off 4 x

let set_bus_master f on =
  live f "set_bus_master";
  let c = f.fn.config Local.command 2 in
  f.fn.set_config Local.command 2
    (if on then c lor Local.bus_master else c land lnot Local.bus_master)

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
  Lock.protect f.lock (fun () -> Tables.Window.add f.maps w (i, combine));
  Ok w

(* Removes one binding of the live window [w] from [table], or refuses [w]. *)
let unlist f table fn w =
  Lock.protect f.lock @@ fun () ->
  if not (Tables.Window.mem table w) then
    invalid_argf "Function.%s: no such window of %s" fn f.bus;
  Tables.Window.remove table w

let unmap f w =
  unlist f f.maps "unmap" w;
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
  | None when f.fn.config 0 2 = absent ->
      Some (strf "%s left the bus: its vendor ID reads 0xffff" f.bus)
  | None -> None

(* The failure at a wait's bound. A driver's sleep may race its device's stop:
   the vendor ID is read as a user, so that a release waits for the read, and a
   release that came first answers as the machine does. *)
let failed_at_bound f =
  match Machine.failed f.machine with
  | Some _ as why -> why
  | None ->
      count f 1;
      if Atomic.get f.released then begin
        leave f;
        None
      end
      else
        match f.fn.config 0 2 = absent with
        | left ->
            leave f;
            if left then
              Some (strf "%s left the bus: its vendor ID reads 0xffff" f.bus)
            else None
        | exception e ->
            leave f;
            reraise e

let wait f ~us what cond =
  if us < 0 then invalid_argf "Function.wait: %d us is negative" us;
  if Machine.wait f.machine ~us cond then Ok ()
  else
    match failed_at_bound f with
    | Some why -> Error (strf "%s: %s" what why)
    | None -> Error (strf "%s did not complete in %d us" what us)

let inherited f = Atomic.get f.inherited

let forget f =
  live f "forget";
  let* () = f.fn.forget () in
  Atomic.set f.inherited false;
  Ok ()

let reset f =
  live f "reset";
  let* () = f.fn.reset () in
  let answers () = f.fn.config 0 2 <> absent in
  if Machine.wait f.machine ~us:(reset_ms * 1000) answers then Ok ()
  else
    match Machine.failed f.machine with
    | Some why -> Error why
    | None ->
        Error (strf "%s does not answer %d ms after its reset" f.bus reset_ms)

(* System memory *)

(* Contiguous memory is at most a huge page. *)
let huge = Sysmem.huge

(* Past 61 bits, the end of the addresses is past every address an OCaml int
   holds. *)
let set_dma_bits f n =
  live f "set_dma_bits";
  if n < 1 || n > full_bits then
    invalid_argf "Function.set_dma_bits: %d bits, expected 1 to %d" n
      full_bits;
  Atomic.set f.dma_end (if n < Sys.int_size - 1 then 1 lsl n else max_int)

(* Whether a run of [runs] ends past [e]. *)
let beyond e runs = List.exists (fun (a, n) -> a > e - n) runs

let on_page f fn a =
  if a mod Machine.page f.machine <> 0 then
    invalid_argf "Function.%s: 0x%x is not on a %d-byte page" fn a
      (Machine.page f.machine)

let alloc_dma_entered ~contiguous ?va f n =
  let page = Machine.page f.machine in
  (* The largest [n] whose rounding up to [page] stays at most [max_int]. *)
  let most = max_int - page + 1 in
  if n <= 0 || n > most then
    invalid_argf "Function.alloc_dma: %d bytes, expected 1 to %d" n most;
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
  let dma =
    let below = Atomic.get f.dma_end in
    match f.fn.alloc_dma ~contiguous ~va ~below n with
    | Ok (Some (w, runs)) when beyond below runs ->
        f.fn.free_dma w;
        Ok None
    | dma -> dma
  in
  (* The allocation is in the table before its use ends, so a release, which
     waits for its uses, finds it. *)
  (match dma with
  | Ok (Some (w, _)) ->
      Lock.protect f.lock (fun () -> Tables.Window.add f.dmas w ())
  | Ok None | Error _ -> ());
  dma

let alloc_dma ?(contiguous = false) ?va f n =
  enter f "alloc_dma";
  match alloc_dma_entered ~contiguous ?va f n with
  | r ->
      leave f;
      r
  | exception e ->
      leave f;
      reraise e

let free_dma f w =
  unlist f f.dmas "free_dma" w;
  f.fn.free_dma w

let pin_entered f a n =
  on_page f "pin" a;
  if n <= 0 then invalid_argf "Function.pin: %d bytes, expected more than 0" n;
  let runs =
    let below = Atomic.get f.dma_end in
    match f.fn.pin ~below a n with
    | Ok (Some runs) when beyond below runs ->
        f.fn.unpin a n;
        Ok None
    | runs -> runs
  in
  (* As [alloc_dma_entered]'s. *)
  (match runs with
  | Ok (Some _) ->
      Lock.protect f.lock (fun () -> Tables.Range.add f.pins (a, n) ())
  | Ok None | Error _ -> ());
  runs

let pin f a n =
  enter f "pin";
  match pin_entered f a n with
  | r ->
      leave f;
      r
  | exception e ->
      leave f;
      reraise e

let unpin f a n =
  Lock.protect f.lock (fun () ->
      if not (Tables.Range.mem f.pins (a, n)) then
        invalid_argf "Function.unpin: 0x%x is not pinned for %s" a f.bus;
      Tables.Range.remove f.pins (a, n));
  f.fn.unpin a n
