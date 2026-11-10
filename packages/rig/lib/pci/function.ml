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
   The tables are used from any domain under [lock]. [state] is one word:
   twice the uses that run, plus 1 once a release has begun. *)
type t = {
  machine : Machine.t;
  bus : string;
  mutable fn : Machine.fn; (* [vacant] until its take returns *)
  inherited : bool Atomic.t; (* until [forget] *)
  releasing : bool Atomic.t; (* once a release was called *)
  dma_end : int Atomic.t;
      (* the end of the addresses it reaches, [2^n] of [set_dma_bits], read
         from any domain *)
  state : int Atomic.t;
  kept : bool Atomic.t; (* [keep] *)
  quiet : bool Atomic.t;
      (* once its give-back returned with the bus mastering off *)
  lock : Lock.t;
  maps : (int * bool) Tables.Window.t; (* to its BAR and [combine] *)
  dmas : unit Tables.Window.t;
  pins : unit Tables.Range.t; (* (address, bytes) *)
  giving : Lock.t; (* held by the one call that gives it back *)
  space : Space.holder; (* the addresses its GPU's memory lies at *)
  given : bool Atomic.t;
      (* once the machine's release or [keep]'s returned, read from any
         domain *)
  lent : int Atomic.t; (* [lend]s not [unlend]ed *)
  writers : writers Atomic.t;
      (* the functions an [unlend_held] named, which may still write its
         memory *)
  parked : bool Atomic.t;
      (* once its give-back stopped its DMA and kept its take for lent
         memory, until [unpark] gives the take back *)
}

(* A list each of whose cells is made before the section that links it,
   which then allocates nothing. *)
and writers = No_writer | Writer of { w : t; mutable rest : writers }

(* A function reaches 64 bits of address until its driver says otherwise. *)
let full_bits = 64

let rec update a f =
  let v = Atomic.get a in
  if not (Atomic.compare_and_set a v (f v)) then update a f

(* The functions the process took whose release has not finished, with the
   process that took each. A function is listed before its take ([vacant]),
   so that a release begins with no allocation, as an exception handler's
   must, and an exception at any of its steps leaves it listed for
   [finish_releases]. It leaves the list once given back. A list in an
   atomic, so that a child of fork meets no held lock; the child finishes
   none of its parent's. *)
let unfinished = Atomic.make []

(* What a function holds before its take returns: nothing, which its release
   gives back. Only a release reaches it, since a take returns the function
   only once it holds the machine's. *)
let vacant_fn =
  let no _ = invalid_arg "Function: the function is not taken" in
  {
    Machine.addressing = Physical;
    inherited = false;
    config = (fun _ -> no);
    set_config = (fun _ _ -> no);
    bar = no;
    map = (fun ~combine:_ _ _ -> no);
    unmap = no;
    interrupt = no;
    reset = no;
    forget = no;
    alloc_dma = (fun ~contiguous:_ ~va:_ ~below:_ -> no);
    free_dma = no;
    pin = (fun ~below:_ _ -> no);
    unpin = (fun _ -> no);
    release = (fun () -> true);
  }

let vacant machine bus =
  let f =
    {
      machine;
      bus;
      fn = vacant_fn;
      inherited = Atomic.make false;
      releasing = Atomic.make false;
      dma_end = Atomic.make max_int;
      state = Atomic.make 0;
      kept = Atomic.make false;
      quiet = Atomic.make false;
      lock = Lock.create ();
      maps = Tables.Window.create 16;
      dmas = Tables.Window.create 16;
      pins = Tables.Range.create 16;
      giving = Lock.create ();
      space = Space.holder ();
      given = Atomic.make false;
      lent = Atomic.make 0;
      writers = Atomic.make No_writer;
      parked = Atomic.make false;
    }
  in
  update unfinished (List.cons (Unix.getpid (), f));
  f

(* The functions [keep] keeps, by machine and bus, which the process takes no
   more. A list in an atomic, which only grows: a child of fork meets no held
   lock, and keeps its parent's list, whose functions its parent holds. *)
let given_out = Atomic.make []

let rec give_out f =
  let l = Atomic.get given_out in
  if not (Atomic.compare_and_set given_out l ((f.machine, f.bus) :: l)) then
    give_out f

(* Lent memory

   A GPU whose memory another GPU maps ([Memory.map_peer]) lends it: its
   release stops its DMA and keeps its take, so that no open resets it and
   gives the memory out while the other GPU may write it. The take goes back
   once nothing lent remains: at the last [unlend], at a take of its bus, or
   at a finish of the releases. [lenders] lists, by process, the functions
   that lent and are not yet given back whole, so that a take finds a parked
   one of its bus. *)
let lenders = Atomic.make []

let unlist f = update lenders (List.filter (fun (_, f') -> f' != f))

(* Whether [f] lends memory that another function may still write: a
   mapping is out, or a function that unmapped it is not released and quiet
   yet. *)
let rec writing = function
  | No_writer -> false
  | Writer { w; rest } ->
      Atomic.get w.state land 1 = 0
      || (not (Atomic.get w.quiet))
      || writing rest

let lends f = Atomic.get f.lent > 0 || writing (Atomic.get f.writers)

(* The machine's release of [f]'s take, which gives back what the take
   holds, then, once the function masters the bus no more, the addresses its
   memory lay at: the memory goes first, since another GPU's memory there
   would be mapped over it. *)
let give_take f =
  let quiet = f.fn.release () in
  if quiet then begin
    Space.release f.space;
    Atomic.set f.quiet true
  end;
  quiet

(* Gives back the take that [f]'s parked give-back kept, once: of two calls,
   the one that clears [parked] gives it back. An exception that ends the
   machine's release parks [f] again, for the next call. *)
let unpark f =
  if Atomic.compare_and_set f.parked true false then
    match give_take f with
    | (_ : bool) -> unlist f
    | exception e ->
        let bt = Printexc.get_raw_backtrace () in
        Atomic.set f.parked true;
        Printexc.raise_with_backtrace e bt

let settle_parked f = if Atomic.get f.parked && not (lends f) then unpark f

let refused machine bus =
  let me = Unix.getpid () in
  let parked (pid, f) =
    pid = me && f.machine == machine && f.bus = bus
    && (settle_parked f;
        Atomic.get f.parked)
  in
  if List.exists (fun (m, b) -> m == machine && b = bus) (Atomic.get given_out)
  then
    Some
      (Error
         (strf
            "%s: its memory was given to another PCI function, which may still \
             write it, until the process exits"
            bus))
  else if List.exists parked (Atomic.get lenders) then
    Some
      (Error
         (strf
            "%s: another GPU maps memory of its last open and may still write \
             it; the open waits for that GPU's free of it"
            bus))
  else None

let takeable machine bus =
  match refused machine bus with Some (Error why) -> Error why | _ -> Ok ()

(* Stores the machine's take in [f] and is [ok], [Ok f] allocated before the
   take, so that nothing between the machine's take returning and the store
   allocates or polls: an exception before leaves [f] vacant. *)
let fill f ok = function
  | Ok fn ->
      f.fn <- fn;
      Atomic.set f.inherited fn.inherited;
      ok
  | Error why -> Error why

let vacant_only fn f =
  if f.fn != vacant_fn || Atomic.get f.releasing then
    invalid_argf "Function.%s: %s is taken or released" fn f.bus

let take_into f =
  vacant_only "take_into" f;
  match refused f.machine f.bus with
  | Some e -> e
  | None ->
      let ok = Ok f in
      fill f ok
        (match Machine.failed f.machine with
        | Some why -> Error why
        | None -> Machine.take f.machine f.bus)

let take_locked_into f l =
  vacant_only "take_locked_into" f;
  if Sysfs.bus l <> f.bus then
    invalid_argf "Function.take_locked_into: the lock is of %s, not %s"
      (Sysfs.bus l) f.bus;
  match refused f.machine f.bus with
  | Some e -> e
  | None ->
      let ok = Ok f in
      fill f ok (Machine.take_locked f.machine l)

let machine f = f.machine
let bus f = f.bus
let holder f = f.space
let addressing f = f.fn.addressing
let released f = Atomic.get f.state land 1 = 1
let live f fn = if released f then Fail.err_released fn f.bus

external reraise : exn -> 'a = "%reraise"

(* Gives [f] back, recording each step that gives something back as the step
   returns, with nothing between that allocates or polls, so that an
   exception that ends it part-way leaves the rest to the next call, and no
   step runs twice. The machine's release gives back what the take holds,
   its windows, memory and pins. A kept or lent function's release stops its
   DMA and gives back nothing more: its take stays the process's until the
   exit, or until it lends nothing, gives it back. The function is quiet once
   its bus mastering reads off: a write the function refused leaves it
   writing the memory it reached. *)
let give_back f =
  if not (Atomic.get f.given) then begin
    let kept = Atomic.get f.kept in
    let lent = (not kept) && lends f in
    if kept || lent then begin
      f.fn.set_config Local.command 2
        (f.fn.config Local.command 2 land lnot Local.bus_master);
      if not (Local.masters (f.fn.config Local.command 2)) then
        Atomic.set f.quiet true
    end
    else ignore (give_take f : bool);
    if lent then Atomic.set f.parked true;
    Atomic.set f.given true;
    (* A last [unlend] that ran before [parked] was set left the take to
       this. *)
    if lent then settle_parked f else unlist f
  end

(* Gives [f] back under [giving], which the caller took, giving it back
   first on both exits. [f] leaves the list once given back. *)
let give_locked f =
  match give_back f with
  | () ->
      Lock.give f.giving;
      update unfinished (List.filter (fun (_, f') -> f' != f))
  | exception e ->
      Lock.give f.giving;
      reraise e

(* Gives [f] back, unless a call that holds [giving], on another thread or,
   below a callback, on this one, is giving it back: a finish or a use that
   left after the give-back's. *)
let give f = if Lock.try_take f.giving then give_locked f

(* A use, which any domain may make while the owner releases the function,
   counts itself in [state] from [start] to [leave], and the release sets
   [state]'s bit. Each is one atomic operation on the one word, which allocates
   nothing, so no asynchronous exception splits it ([Atomic.incr] and
   [Atomic.decr] allocate the location they add to first), and a use's handler
   starts as [start] returns and calls [leave] first. The word orders them: a
   use that finds the bit set as it starts does nothing, and every other use is
   counted when the bit is set. The word goes to "released, no use" as the
   release finds no use or as the last use that ran leaves, and that call gives
   [f] back. A use refused after the bit sees the same value as it leaves, and
   so may a finish: [giving] keeps two of them from giving back at once, and
   [given] keeps anything from being given back twice. No release waits for a
   use. A use also counts among the locks its thread holds ([rig_hold] in
   [rig_edge.h]), so that a call of rig's or a driver's from a callback inside
   it is refused, and no finish waits for a use of its own thread. *)

(* Counts a use, and is whether it started: [false] once the release's bit
   is set, the use then to be left at once. *)
let start f =
  Rig_edge_primitives.hold ();
  Atomic.fetch_and_add f.state 2 land 1 = 0

(* The use that leaves the word at "released, no use" gives [f] back. A use
   that found the bit set leaves so too, after the last use that ran or
   after the give-back; [giving] then keeps it from a second, and [given]
   from giving anything back twice. The compiler polls at the entry of a
   function whose tail call may reach a function defined after it; [give]
   is defined before, so nothing polls before the count. *)
let leave f =
  let s = Atomic.fetch_and_add f.state (-2) in
  Rig_edge_primitives.unhold ();
  if s = 3 then give f

let enter f fn =
  if not (start f) then begin
    leave f;
    Fail.err_released fn f.bus
  end

(* As [enter], refused with [false] rather than raising, so that a caller
   whose next step installs its handler leaves nothing between the count
   and it. *)
let use f =
  if start f then true
  else begin
    leave f;
    false
  end

let unuse = leave

(* [releasing] elects the one release that sets the bit, by one exchange,
   so that the bit is added once. A loop of compare-and-set would poll at
   its back edge, where a signal could land before the bit is set and leave
   [f] unreleased and held. The release that finds no use gives [f] back
   before it returns, waiting for a finish that took [giving] first, since
   that finish gives it back. Nothing allocates or polls before the bit is
   set: the compiler polls at the entry of a function whose tail call may
   reach a function defined after it, and [give_locked] is defined before. *)
let release f =
  if not (Atomic.exchange f.releasing true) then
    if Atomic.fetch_and_add f.state 1 = 0 then begin
      Lock.take f.giving;
      give_locked f
    end

(* A take that fails leaves its function vacant, given back by its
   release. *)
let given_back f = function
  | Ok _ as r -> r
  | Error _ as e ->
      release f;
      e

let take machine bus =
  let f = vacant machine bus in
  match take_into f with
  | r -> given_back f r
  | exception e ->
      release f;
      reraise e

let take_locked machine l =
  let f = vacant machine (Sysfs.bus l) in
  match take_locked_into f l with
  | r -> given_back f r
  | exception e ->
      release f;
      reraise e

(* A finish waits for the uses that run, then for a give-back that runs on
   another thread, so that the function is given back as it returns. A use
   makes a few calls of the machine, which call nothing of rig's and return
   (Machine's transports), so the naps are few; a wake-up from [leave] would
   take a lock or a call that an asynchronous exception can end before it
   wakes the finish. A give-back takes only locks below [giving], none of
   which a finish holds, since its caller holds none but its driver's. A
   release elected and not yet counted in the word is waited for too. The
   wait for uses naps with the thread's signals unblocked ({!Machine.nap}),
   so that a signal's handler runs there: an exception it raises ends the
   finish before it takes [giving], and leaves [f] listed for the next. The
   wait for a give-back is the take of [giving], which leaves the mask as it
   is, as every lock's take does: inside a mask, signals stay blocked. *)
let nap_s = 0.0001

let finish f =
  if Atomic.get f.releasing then begin
    while Atomic.get f.state <> 1 do
      Machine.nap nap_s
    done;
    Lock.take f.giving;
    give_locked f
  end

(* Finishes every listed function of this process, then raises the first
   exception a finish raised, so that one that raises leaves the others
   finished. The walk allocates nothing, so that an exception in one finish
   ends no other. [Finished] stands for no exception yet. *)
exception Finished

let rec finish_from me first = function
  | [] -> if first != Finished then raise first
  | (pid, f) :: rest -> (
      if pid <> me then finish_from me first rest
      else
        match finish f with
        | () -> finish_from me first rest
        | exception e ->
            finish_from me (if first == Finished then e else first) rest)

let finish_held () =
  let me = Unix.getpid () in
  finish_from me Finished (Atomic.get unfinished);
  List.iter
    (fun (pid, f) -> if pid = me then settle_parked f)
    (Atomic.get lenders)

let finish_releases () =
  Lock.check "Function.finish_releases";
  finish_held ()

let reaching f = released f && not (Atomic.get f.quiet)

(* A lend counts before it checks the release's bit, and the give-back reads
   the count after the bit is set: either the lend sees the bit and takes its
   count back, or the give-back sees the count and parks. [f] is listed at
   its first lend, before the count, so that a parked [f] is always listed. *)
let lend f =
  if not (List.exists (fun (_, f') -> f' == f) (Atomic.get lenders)) then
    update lenders (List.cons (Unix.getpid (), f));
  Atomic.incr f.lent;
  if released f then begin
    ignore (Atomic.fetch_and_add f.lent (-1) : int);
    false
  end
  else true

(* A refused lend may have been the count a release waited for, and listed
   a function given back already. *)
let lend_refused f =
  settle_parked f;
  if Atomic.get f.given && not (Atomic.get f.parked) then unlist f

type writer = writers

let writer w = Writer { w; rest = No_writer }

(* The lend's count and the writer's link change with nothing that allocates
   or polls: the caller's section, which ends the mapping, makes them one
   step. The lone writer of [f] at a time is its caller's, under that
   section's lock, and a reader of [writers] meets the cell whole. *)
let unlend_held f w =
  (match w with
  | Some (Writer c as cell) ->
      c.rest <- Atomic.get f.writers;
      Atomic.set f.writers cell
  | Some No_writer | None -> ());
  ignore (Atomic.fetch_and_add f.lent (-1) : int)

let settle = settle_parked

let unlend f =
  unlend_held f None;
  settle f

(* [kept] is set once [f] is given out, so that a keep an exception ended
   gives it out again. A function given out twice is listed twice. *)
let keep f =
  live f "keep";
  if not (Atomic.get f.kept) then begin
    give_out f;
    Atomic.set f.kept true
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

(* Valid until [f] is given back, which no use outlives: a use's turn off,
   made once a release has begun, reaches the function. *)
let set_bus_master f on =
  if Atomic.get f.given then Fail.err_released "set_bus_master" f.bus;
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
  live f "unmap";
  unlist f f.maps "unmap" w;
  f.fn.unmap w

(* A use's unmap, which a release begun meanwhile does not refuse: the
   give-back, which follows every use, lists the windows left. *)
let unmap_used f w =
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
   the vendor ID is read as a use, so that a release leaves the give-back to
   the read, and a release that came first answers as the machine does. *)
let failed_at_bound f =
  match Machine.failed f.machine with
  | Some _ as why -> why
  | None ->
      if not (start f) then begin
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

let set_own_length = Sysmem.set_own_length
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

(* Reached physically, memory lies in huge pages, each mapped whole on the 2
   MiB block of addresses that holds it; contiguous memory larger than a page
   is one of them. *)
let placed f ~va n =
  let lo, hi =
    if f.fn.addressing = Machine.Physical then
      (va / huge * huge, (va + n + huge - 1) / huge * huge)
    else (va, va + n)
  in
  Machine.reserved f.machine lo (hi - lo)

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
  let physical = f.fn.addressing = Machine.Physical in
  (match va with
  | None -> ()
  | Some va ->
      on_page f "alloc_dma" va;
      if physical && contiguous && n > page && va mod huge <> 0 then
        invalid_argf
          "Function.alloc_dma: 0x%x is not on 2 MiB, which a huge page needs" va;
      if not (placed f ~va n) then
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
  (* The allocation is in the table before its use ends, so a give-back,
     which follows its uses, finds it. *)
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

(* A free and an unpin are uses, so that no give-back runs while they give
   memory back. One refused, once a release has begun, leaves the memory to
   the give-back, which gives back all the take holds, but for a kept
   function, whose take stays until the exit: once its bus mastering reads
   off, the free gives back its own memory, which the machine counts apart
   from what another function was given. *)
let free_dma f w =
  unlist f f.dmas "free_dma" w;
  if use f then
    match f.fn.free_dma w with
    | () -> leave f
    | exception e ->
        leave f;
        reraise e
  else if Atomic.get f.kept && Atomic.get f.quiet then f.fn.free_dma w

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
  if use f then
    match f.fn.unpin a n with
    | () -> leave f
    | exception e ->
        leave f;
        reraise e
  else if Atomic.get f.kept && Atomic.get f.quiet then f.fn.unpin a n
