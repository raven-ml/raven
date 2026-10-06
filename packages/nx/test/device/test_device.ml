(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Devices over fake drivers whose memory is host memory, so that a test reads
   what a device holds at its addresses. Programs of creates, views, borrows,
   copies, drops and budgets run against a model of every buffer's bytes and
   every device's memory and counters; the laws, the refusals, the timeline and
   the failure rules are tests of their own. *)

open Windtrap
module B = Nx_device.Buffer
module S = Nx_dtype.Scalar
module Driver = Nx_device.Driver
module Region = Driver.Region

type chars =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

external memmove : nativeint -> nativeint -> int -> unit
  = "test_nx_device_memmove"

external store_signal : nativeint -> int -> unit = "test_nx_device_signal"
external c_host : B.t -> nativeint = "test_nx_device_buffer_host"
external c_live : B.t -> bool = "test_nx_device_buffer_live"

let host = Nx_device.host
let chars n : chars = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n

(* Byte loops: the tests move hundreds of megabytes through these, where the
   init functions of Bigarray and String call a closure per byte. *)
let string_of (ba : chars) =
  let b = Bytes.create (Bigarray.Array1.dim ba) in
  for i = 0 to Bytes.length b - 1 do
    Bytes.unsafe_set b i (Bigarray.Array1.unsafe_get ba i)
  done;
  Bytes.unsafe_to_string b

let of_string s =
  let ba = chars (String.length s) in
  for i = 0 to String.length s - 1 do
    Bigarray.Array1.unsafe_set ba i (String.unsafe_get s i)
  done;
  B.of_bigarray ba

(* The [n] bytes at the host address [a]. *)
let peek a n =
  let ba = chars n in
  memmove (B.address (B.of_bigarray ba)) a n;
  ba

let read b =
  let ba = chars (B.nbytes b) in
  B.copy ~src:b ~dst:(B.of_bigarray ba);
  string_of ba

let write b s = B.copy ~src:(of_string s) ~dst:b

(* [d]'s borrow of [b], which it maps. *)
let borrow d b = match B.borrow d b with Ok b -> b | Error why -> failwith why

(* [b] consumed with [why] by a donation, as a compiled call consumes it. *)
let consume ~why b =
  B.Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c -> B.Claim.consume c ~why b)

(* The file at [path], opened, or created with [n] bytes. *)
let of_file path =
  match B.of_file path with Ok b -> b | Error why -> failwith why

let create_file path n =
  match B.create_file path n with Ok b -> b | Error why -> failwith why

(* The function [name] of [binary], which [d] loads. *)
let program d ~binary ~name =
  match Nx_device.Program.load d ~binary ~name with
  | Ok p -> p
  | Error why -> failwith why

(* A driver's load of binaries whose every function has the handle [h]. *)
let loads h ~binary:_ =
  Ok { Driver.code = None; entry = (fun _ -> Ok h); unload = ignore }

let pattern seed n =
  let b = Bytes.create n in
  for i = 0 to n - 1 do
    Bytes.unsafe_set b i
      (Char.unsafe_chr (((seed * 31) + (i * 7) + (i / 5)) land 0xff))
  done;
  Bytes.unsafe_to_string b

(* A machine has one device of a name, and the tests make many devices of one
   kind: [unique kind] names one of them, and [kind_of] reads its kind back. *)
let made = Atomic.make 0
let unique kind = Printf.sprintf "%s#%d" kind (Atomic.fetch_and_add made 1)

let kind_of name =
  match String.index_opt name '#' with
  | Some i -> String.sub name 0 i
  | None -> name

(* [masked s] is [s] with the number of every device it names removed: what a
   message or a profile says of devices, by their kinds. *)
let masked s =
  let b = Buffer.create (String.length s) in
  let n = String.length s in
  let i = ref 0 in
  while !i < n do
    if s.[!i] = '#' then begin
      incr i;
      while !i < n && s.[!i] >= '0' && s.[!i] <= '9' do
        incr i
      done
    end
    else begin
      Buffer.add_char b s.[!i];
      incr i
    end
  done;
  Buffer.contents b

let stats = Nx_device.stats
let allocated d = Nx_device.Stats.allocated (stats d)
let cached d = Nx_device.Stats.cached (stats d)
let pp_format ppf s = Format.pp_print_string ppf (S.to_string s)
let pp_device ppf d = Format.pp_print_string ppf (Nx_device.name d)
let arg pp = Testable.make ~pp ~equal:( == )
let ints = Gen.of_list ~pp:Format.pp_print_int
let seeds = Gen.int_range 0 255

let out_of_memory d n = function
  | Nx_device.Out_of_memory (d', n') -> d' == d && n' = n
  | _ -> false

let lost d why = function
  | Nx_device.Lost (d', why') -> d' == d && why' = why
  | _ -> false

(* Host buffers of this many bytes start on a page on every platform. *)
let page = 1 lsl 16

(* Runs [f] and collects what it allocated. *)
let dropped f =
  ignore (Sys.opaque_identity (f ()));
  Gc.full_major ()

(* Fake drivers *)

(* A driver whose memory is host memory aligned to 16 bytes and never to a page.
   It counts its bytes, its calls and its live mappings but those of the host's
   staging memory, and the memory each of its regions is. A far driver's own
   memory is not addressed by the host, except through a window of [window]
   bytes of mapped memory when it has one, and its copy queue runs copies and
   timestamps when the host waits for them, as a GPU runs behind the host:
   [stalled] stops it, [broken] makes it refuse to enqueue, and a device named
   PEER... copies into the memory of the others. Its timestamps are the host
   clock, or, with a clock of its own, ticks of it two hours ahead. *)
type driver = {
  blocks : (nativeint, chars * int) Hashtbl.t;
  memories : (nativeint, B.memory) Hashtbl.t;
  mutable window : int;
  mutable held : int;
  mutable frees : int;
  mutable refuse : bool;
  mutable mapped : int;
  mutable staging : nativeint list;
  mutable queued : int;
  mutable staged : int;
  mutable stalled : bool;
  mutable broken : bool;
}

type fake = { dev : Nx_device.t; drv : driver }

let staging_bytes = 128 lsl 20
let transfers name = String.starts_with ~prefix:"PEER" name
let ahead = 7_200_000_000_000

let fake ?(name = "NEAR") ?(budget = max_int) ?(far = false) ?(maps = far)
    ?window ?signal ?load ?peer ?synchronized ?report ?sleep ?finalize ?clock
    ?resolve ?room ?reaches () =
  let drv =
    {
      blocks = Hashtbl.create 8;
      memories = Hashtbl.create 8;
      window = Option.value window ~default:0;
      held = 0;
      frees = 0;
      refuse = false;
      mapped = 0;
      staging = [];
      queued = 0;
      staged = 0;
      stalled = false;
      broken = false;
    }
  in
  let alloc ~addressed memory n =
    if drv.refuse then None
    else
      let ba = chars (n + 31) in
      let a = B.address (B.of_bigarray ba) in
      let a = Nativeint.(logand (add a 15n) (lognot 15n)) in
      (* Never on a page, so that no other device maps it by chance. *)
      let a = if Nativeint.rem a 4096n = 0n then Nativeint.add a 16n else a in
      Hashtbl.replace drv.blocks a (ba, n);
      Hashtbl.replace drv.memories a memory;
      drv.held <- drv.held + n;
      let host = if addressed then Some a else None in
      Some (Region.v ?host ~handle:a a n)
  in
  let free r =
    let a = Region.address r in
    drv.held <- drv.held - snd (Hashtbl.find drv.blocks a);
    drv.frees <- drv.frees + 1;
    Hashtbl.remove drv.blocks a;
    Hashtbl.remove drv.memories a
  in
  let in_window n =
    if n > drv.window then None
    else (
      drv.window <- drv.window - n;
      alloc ~addressed:true Mapped n)
  in
  let out_of_window r =
    drv.window <- drv.window + snd (Hashtbl.find drv.blocks (Region.address r));
    free r
  in
  let map a n =
    if n = staging_bytes then drv.staging <- a :: drv.staging
    else drv.mapped <- drv.mapped + 1;
    Ok (Region.v ~host:a ~handle:1n a n)
  in
  let staged a =
    let n = Nativeint.of_int staging_bytes in
    List.exists (fun s -> a >= s && a < Nativeint.add s n) drv.staging
  in
  let queue = Queue.create () and signaled = ref 0 in
  let copy ~dst ~src n ~signal:v =
    if drv.broken then failwith "enqueue refused";
    Queue.push
      ( v,
        fun () ->
          memmove dst src n;
          drv.queued <- drv.queued + 1;
          if staged dst || staged src then drv.staged <- drv.staged + 1 )
      queue
  in
  let ticks () =
    match clock with
    | Some (Driver.Device_clock { hz }) ->
        (Nx_device.Profile.now () + ahead) / (1_000_000_000 / hz)
    | Some Driver.Host_clock | None -> Nx_device.Profile.now ()
  in
  let stamp ~slot ~signal:v =
    if drv.broken then failwith "enqueue refused";
    Queue.push
      (v, fun () -> store_signal (Nativeint.add slot 8n) (ticks ()))
      queue
  in
  let rec run_to v =
    if (not drv.stalled) && !signaled < v && not (Queue.is_empty queue) then begin
      let v', f = Queue.pop queue in
      f ();
      signaled := v';
      run_to v
    end
  in
  let queue_signal ~timeline:_ =
    {
      Driver.signaled = (fun () -> !signaled);
      wait =
        (fun v ~ms:_ ->
          run_to v;
          (* A stalled queue is one its driver declares hung. *)
          if !signaled < v && drv.stalled then failwith "hang detected";
          !signaled >= v);
    }
  in
  let queue ~timeline:_ =
    let transfer d =
      if transfers name && transfers (Nx_device.name d) then Some copy else None
    in
    {
      Driver.copy;
      transfer;
      stamp;
      clock = Option.value clock ~default:Driver.Host_clock;
    }
  in
  let unmap _ = drv.mapped <- drv.mapped - 1 in
  let mapping = Driver.Pages { map; unmap } in
  let memory : Driver.memory =
    if far then
      Device_local
        {
          memory = { alloc = alloc ~addressed:false Device; free };
          host_memory = { alloc = alloc ~addressed:true Pinned; free };
          mapped =
            Option.map
              (fun w -> ({ Driver.alloc = in_window; free = out_of_window }, w))
              window;
          mapping;
          queue;
        }
    else
      Host_visible
        {
          memory = { alloc = alloc ~addressed:true Device; free };
          mapping = (if maps then Some mapping else None);
        }
  in
  let completion : Driver.completion =
    match (far, signal, sleep) with
    | true, _, _ -> Signal queue_signal
    | false, Some s, _ -> Signal (fun ~timeline:_ -> s)
    | false, None, Some sleep -> Sleep (fun ~timeline:_ -> sleep)
    | false, None, None -> Poll
  in
  let dev =
    Driver.device ~name:(unique name) ~arch:"test" ~budget ~completion ?load
      ?peer ?synchronized ?report ?finalize ?resolve ?room ?reaches memory
  in
  { dev; drv }

let far ?(name = "FAR") ?budget ?window ?clock ?peer ?reaches () =
  fake ~name ?budget ~far:true ?window ?clock ?peer ?reaches ()

(* The memory of [d]'s driver that holds [b]. *)
let memory_of f b =
  Hashtbl.find f.drv.memories (Region.address (Region.of_buffer b))

(* A signal whose waits answer [wait ms]. *)
let signal ?(signaled = 0) wait =
  { Driver.signaled = (fun () -> signaled); wait = (fun _ ~ms -> wait ms) }

(* A signal that its driver declares hung at the first wait, counting the waits
   for it. *)
let hung waits =
  signal (fun _ ->
      incr waits;
      failwith "hang detected")

(* The sleep of a driver that declares its device hung the first time a wait
   sleeps, 200 ms after its signal word stopped moving. *)
let hangs ~still:_ _ = failwith "hang detected"

(* A signal that a gate opens: until then the device has signaled nothing, and a
   wait for it fails. *)
let gate opened =
  {
    Driver.signaled = (fun () -> if !opened then max_int else 0);
    wait = (fun _ ~ms:_ -> !opened);
  }

(* Submits work of [d] alone that touches [touches], and is [f] of the value the
   work signals. *)
let submit ?(touches = []) d f =
  Nx_device.submit [ d ] ~touches (fun s -> f (Nx_device.Submission.value s d))

(* Signals [v] on [d]'s timeline after 50 ms from another domain, as the device
   would, and records that it did. *)
let signal_later d v =
  let signaled = Atomic.make false in
  let word = B.address (Nx_device.signal_word d) in
  let domain =
    Domain.spawn (fun () ->
        Unix.sleepf 0.05;
        Atomic.set signaled true;
        store_signal word v)
  in
  (signaled, domain)

(* Memory *)

(* Each buffer's bytes, [-1] where nothing wrote them, in memory shared with its
   views and borrows, and each device's budget, counters, owned memory and
   borrows. *)
module Model = struct
  exception Dropped
  exception No_memory
  exception Refused

  (* A cell is a byte that something wrote, or -1. Out of the heap, so that the
     collections that drops force do not scan it. *)
  type cells =
    (int, Bigarray.int16_signed_elt, Bigarray.c_layout) Bigarray.Array1.t

  (* Every cell that something wrote lies in [lo, hi), so that a check of a
     large memory reads only the cells written. *)
  type memory = {
    cells : cells;
    on_page : bool;
    mutable lo : int;
    mutable hi : int;
  }

  let wrote m off n =
    if n > 0 then begin
      m.lo <- Int.min m.lo off;
      m.hi <- Int.max m.hi (off + n)
    end

  (* The buffers over some memory, the bytes owned, whether they count in the
     device's budget, and for a borrow the host memory it maps. *)
  type holding = {
    owned : int;
    budgeted : bool;
    mutable holders : int;
    maps : memory option;
  }

  type device = {
    name : string;
    mutable budget : int;
    mutable bytes_in : int;
    mutable bytes_out : int;
    mutable owns : holding list;
    mutable borrows : holding list;
  }

  type buffer = {
    memory : memory;
    off : int;
    dtype : S.t;
    length : int;
    device : device option;
    holding : holding;
    borrowed : bool;
    addressed : bool;
    mutable dropped : bool;
  }

  let nbytes s n = ((n * S.bitsize s) + 7) / 8
  let size r = nbytes r.dtype r.length
  let element s = Int.max 1 (S.bitsize s / 8)

  (* Whether [n] elements of [s] take more than [max_int] bytes. *)
  let too_big s n = S.bitsize s >= 8 && n > max_int / (S.bitsize s / 8)

  let device name budget =
    { name; budget; bytes_in = 0; bytes_out = 0; owns = []; borrows = [] }

  let live d =
    List.fold_left
      (fun n h -> if h.holders > 0 then n + h.owned else n)
      0 d.owns

  let budgeted d =
    List.fold_left
      (fun n h -> if h.holders > 0 && h.budgeted then n + h.owned else n)
      0 d.owns

  let mappings d =
    let add seen h =
      match h.maps with
      | Some m when h.holders > 0 && not (List.memq m seen) -> m :: seen
      | _ -> seen
    in
    List.length (List.fold_left add [] d.borrows)

  let name r = match r.device with None -> "CPU" | Some d -> d.name
  let alive r = if r.dropped then raise Dropped

  let fresh ?(on_page = false) ?device ?(owned = 0) ?(budgeted = true)
      ~addressed s n =
    if n < 0 || too_big s n then invalid_arg "create";
    let holding = { owned; budgeted; holders = 1; maps = None } in
    Option.iter (fun d -> d.owns <- holding :: d.owns) device;
    let cells = Bigarray.(Array1.create int16_signed c_layout (nbytes s n)) in
    Bigarray.Array1.fill cells (-1);
    let memory = { cells; on_page; lo = Bigarray.Array1.dim cells; hi = 0 } in
    let borrowed = false and dropped = false in
    {
      memory;
      off = 0;
      dtype = s;
      length = n;
      device;
      holding;
      borrowed;
      addressed;
      dropped;
    }

  let create_host big s n =
    if big then fresh ~on_page:true ~addressed:true s (page * 8 / S.bitsize s)
    else fresh ~addressed:true s n

  (* A far device's memory is addressed by the host when it is pinned memory,
     which its mapped memory is, having no window. Pinned memory is the host's,
     and counts in no budget. *)
  let create d (memory : B.memory) s n =
    let bytes = if n < 0 || too_big s n then 0 else nbytes s n in
    let counts = d.name = "NEAR" || memory = Device in
    if bytes > 0 && counts && budgeted d + bytes > d.budget then raise No_memory;
    fresh ~device:d ~owned:bytes ~budgeted:counts
      ~addressed:(d.name = "NEAR" || memory <> Device)
      s n

  let view r (offset, s, n) =
    alive r;
    if
      offset < 0 || n < 0 || too_big s n
      || offset > size r - nbytes s n
      || (r.off + offset) mod element s <> 0
    then invalid_arg "view";
    r.holding.holders <- r.holding.holders + 1;
    { r with off = r.off + offset; dtype = s; length = n; dropped = false }

  (* A host buffer that create made of fewer than 64 KiB is refused, wherever it
     starts. *)
  (* A buffer on [d] borrows as itself, a buffer of no bytes always borrows, a
     borrow on another device borrows the host memory under it, and another
     device's own memory is refused: the fake drivers map no peer. *)
  let rec borrow d r =
    alive r;
    match r.device with
    | Some d' when d' == d ->
        r.holding.holders <- r.holding.holders + 1;
        { r with dropped = false }
    | _ when size r = 0 -> map d { r with off = 0 } (* over no memory *)
    | Some _ when not r.borrowed -> raise Refused
    | Some _ | None ->
        if d.name = "NEAR" || not r.memory.on_page then raise Refused;
        map d r

  and map d r =
    let maps = if size r > 0 then Some r.memory else None in
    let mapped = mappings d in
    let holding = { owned = 0; budgeted = false; holders = 1; maps } in
    cover "a second borrow of mapped memory"
      (maps <> None
      && mappings { d with borrows = holding :: d.borrows } = mapped);
    d.borrows <- holding :: d.borrows;
    let device = Some d in
    {
      r with
      device;
      holding;
      borrowed = true;
      addressed = true;
      dropped = false;
    }

  (* The device whose memory [r] is: a borrow's is its host's. *)
  let holder r = if r.borrowed then None else r.device

  let fill r s =
    wrote r.memory r.off (String.length s);
    for i = 0 to String.length s - 1 do
      r.memory.cells.{r.off + i} <- Char.code (String.unsafe_get s i)
    done

  let write seed r =
    alive r;
    fill r (pattern seed (size r));
    Option.iter (fun d -> d.bytes_in <- d.bytes_in + size r) (holder r)

  (* A copy reads all of its source first: a borrow and the memory it borrows
     are two buffers, which may overlap. *)
  let copy src dst =
    alive src;
    alive dst;
    let n = size src in
    if n <> size dst then invalid_arg "copy";
    let overlap = src.off < dst.off + n && dst.off < src.off + n in
    if n > 0 && src.memory == dst.memory && overlap then invalid_arg "copy";
    let sub r = Bigarray.Array1.sub r.memory.cells r.off n in
    let cells = Bigarray.(Array1.create int16_signed c_layout n) in
    Bigarray.Array1.blit (sub src) cells;
    Bigarray.Array1.blit cells (sub dst);
    wrote dst.memory dst.off n;
    let between =
      match (holder src, holder dst) with
      | Some a, Some b -> a != b
      | a, b -> Option.is_some a || Option.is_some b
    in
    cover "a copy between devices" (n > 0 && between);
    cover "a copy within one memory" (n > 0 && src.memory == dst.memory);
    if between then begin
      Option.iter (fun s -> s.bytes_out <- s.bytes_out + n) (holder src);
      Option.iter (fun d -> d.bytes_in <- d.bytes_in + n) (holder dst)
    end

  (* The [n] bytes at [a] in a buffer of [src] bytes and at [b] in one of [dst]
     bytes, clamped into both. *)
  let spans src dst (a, b, n) =
    let a = Int.min a src and b = Int.min b dst in
    (a, b, Int.min n (Int.min (src - a) (dst - b)))

  let copy_bytes src dst spec =
    alive src;
    alive dst;
    let a, b, n = spans (size src) (size dst) spec in
    let bytes r o = { r with off = r.off + o; dtype = S.UInt8; length = n } in
    copy (bytes src a) (bytes dst b)

  (* Whether [ba] holds the bytes of [r] that something wrote. *)
  let holds r (ba : chars) =
    let n = size r and m = r.memory and off = r.off in
    if off + n > Bigarray.Array1.dim m.cells || n > Bigarray.Array1.dim ba then
      invalid_arg "holds";
    let last = Int.min (off + n) m.hi in
    let i = ref (Int.max off m.lo) in
    while
      !i < last
      &&
      let c = Bigarray.Array1.unsafe_get m.cells !i in
      c < 0 || Char.code (Bigarray.Array1.unsafe_get ba (!i - off)) = c
    do
      incr i
    done;
    !i >= last

  (* [s] with the bytes of [r] that nothing wrote masked. *)
  let masked r s =
    String.mapi (fun i c -> if r.memory.cells.{r.off + i} < 0 then '?' else c) s

  let contents r =
    String.init (size r) (fun i ->
        let c = r.memory.cells.{r.off + i} in
        if c < 0 then '?' else Char.chr c)

  let drop r =
    if not r.dropped then begin
      r.dropped <- true;
      r.holding.holders <- r.holding.holders - 1
    end

  let set_budget d n =
    if n < 0 then invalid_arg "set_budget";
    d.budget <- n

  let pp ppf r =
    Format.fprintf ppf "%d %s on %s%s" r.length (S.to_string r.dtype) (name r)
      (if r.dropped then ", dropped" else "")
end

(* A fake device with the driver bytes it held once made, and a buffer that a
   drop makes unreachable. *)
type device = { fake : fake; baseline : int }
type buffer = { mutable b : B.t option }

let device fake = { fake; baseline = fake.drv.held }
let get s = match s.b with Some b -> b | None -> raise Model.Dropped
let some b = { b = Some b }

let device_invariant (r : Model.device) { fake; baseline } =
  let st = stats fake.dev in
  let allocated = Nx_device.Stats.allocated st in
  let cached = Nx_device.Stats.cached st in
  equal ~msg:"allocated is the live buffers' bytes" int (Model.live r) allocated;
  equal ~msg:"the driver holds the live buffers and the cache" int
    (fake.drv.held - baseline) (allocated + cached);
  (* A far device's pinned memory and its cache count in no budget. *)
  if cached > 0 && r.name = "NEAR" then
    at_most ~msg:"the cache keeps within the budget" int ~than:r.budget
      (allocated + cached);
  equal ~msg:"budget" int r.budget (Nx_device.budget fake.dev);
  equal ~msg:"bytes in" int r.bytes_in (Nx_device.Stats.bytes_in st);
  equal ~msg:"bytes out" int r.bytes_out (Nx_device.Stats.bytes_out st);
  equal ~msg:"one mapping per host memory borrowed" int (Model.mappings r)
    fake.drv.mapped

let buffer_invariant (r : Model.buffer) s =
  if not r.dropped then begin
    let b = get s and n = Model.size r in
    equal ~msg:"format" string (S.to_string r.dtype) (S.to_string (B.dtype b));
    equal ~msg:"length" int r.length (B.length b);
    equal ~msg:"bytes" int n (B.nbytes b);
    equal ~msg:"borrowed" bool r.borrowed (B.is_borrowed b);
    equal ~msg:"device" string (Model.name r)
      (kind_of (Nx_device.name (B.device b)));
    if n > 0 then begin
      let held = peek (B.address b) n in
      if not (Model.holds r held) then
        equal ~msg:"what the device holds" string (Model.contents r)
          (Model.masked r (string_of held));
      (match Region.host_address (Region.of_buffer b) with
      | None -> is_false ~msg:"addressed by the host" r.addressed
      | Some a ->
          is_true ~msg:"addressed by the host" r.addressed;
          equal ~msg:"the host address, read in C" nativeint
            (Nativeint.add a (Nativeint.of_int (B.offset b)))
            (c_host b));
      let bigarray () = B.bigarray Bigarray.char b in
      if r.device = None && bigarray () <> held then
        equal ~msg:"its bigarray" string (string_of held)
          (string_of (bigarray ()))
    end
  end

let dev = abstract "d" ~invariant:device_invariant
let buf = abstract "b" ~pp:Model.pp ~invariant:buffer_invariant

let pp_budget ppf n =
  if n = max_int then Format.pp_print_string ppf "max_int"
  else Format.pp_print_int ppf n

(* Mostly false: a device's host memory. *)
let pp_memory ppf (m : B.memory) =
  Format.pp_print_string ppf
    (match m with
    | Device -> "Device"
    | Pinned -> "Pinned"
    | Mapped -> "Mapped")

(* Mostly the device's own memory. *)
let memories =
  Gen.frequency
    [
      (3, Gen.constant ~pp:pp_memory B.Device);
      (1, Gen.of_list ~pp:pp_memory [ B.Pinned; Mapped ]);
    ]

(* Mostly room for every buffer. *)
let budgets =
  Gen.frequency
    [
      (3, Gen.constant ~pp:pp_budget max_int);
      (1, Gen.of_list ~pp:pp_budget [ 0; 16; 64; 256 ]);
    ]

(* Every storage format. *)
let formats =
  Gen.of_list ~pp:pp_format
    (S.[ Bool; Bit; UInt8; Int8; Int4; UInt4; Int16; UInt16; Int32; UInt32 ]
    @ S.[ Int64; UInt64; Float8_e4m3; Float8_e5m2; Float8_e4m3fnuz ]
    @ S.[ Float8_e5m2fnuz; Float16; BFloat16; Float32; Float64 ]
    @ S.[ Complex64; Complex128 ])

let counts = ints [ 0; 1; 2; 3; 5; 16; 17; -1 ]

(* Budgets at the edges of what a device holds. *)
let edges =
  among (arg pp_budget) dev (fun r ->
      let live = Model.live r in
      [ live; live + 1; live + 16; Int.max 0 (live - 1); 0; max_int; -1 ])

(* Views of a buffer: itself, its bytes, and windows of every width that fit,
   overflow, cross its end or are not aligned. *)
let windows =
  let pp ppf (o, s, n) = Format.fprintf ppf "%d %a at %d" n pp_format s o in
  among (arg pp) buf (fun (r : Model.buffer) ->
      let n = Model.size r in
      S.
        [
          (0, r.dtype, r.length);
          (0, UInt8, n);
          (1, UInt8, n - 1);
          (n, UInt8, 0);
          (0, Int4, (2 * n) - 1);
          (1, UInt4, 1);
          (0, Bit, (8 * n) - 3);
          (1, Bit, 9);
          (2, Int16, 1);
          (1, Int16, 1);
          (4, Float32, (n - 4) / 4);
          (8, Complex128, 1);
          (0, Complex128, n / 16);
          (-1, UInt8, 1);
          (0, UInt8, -1);
          (0, UInt8, n + 1);
          (0, Float64, (max_int / 8) + 1);
        ])

(* Spans of bytes: where in the source, where in the destination, how many. *)
let spans =
  let at = Gen.int_range 0 3 in
  Gen.with_pp
    (fun ppf (a, b, n) -> Format.fprintf ppf "(%d bytes from %d to %d)" n a b)
    (Gen.triple at at (Gen.int_range 0 64))

(* A create the device cannot serve raises Out_of_memory with the device and the
   bytes. *)
let created d s n f =
  match f () with
  | b -> some b
  | exception (Nx_device.Out_of_memory _ as e) ->
      is_true (out_of_memory d (Model.nbytes s n) e);
      raise Model.No_memory

(* Listed twice in the commands, so that programs copy often. *)
let copy =
  command "copy"
    (buf ^-> buf ^-> spans @-> returns unit)
    Model.copy_bytes
    (fun src dst spans ->
      let src = get src and dst = get dst in
      let a, b, n = Model.spans (B.nbytes src) (B.nbytes dst) spans in
      B.copy
        ~src:(B.view src ~offset:a S.UInt8 n)
        ~dst:(B.view dst ~offset:b S.UInt8 n))

let commands =
  [
    command "near"
      (budgets @-> makes dev)
      (Model.device "NEAR")
      (fun budget -> device (fake ~budget ()));
    command "far"
      (budgets @-> Gen.bool @-> makes dev)
      (fun budget peer -> Model.device (if peer then "PEER" else "FAR") budget)
      (fun budget peer ->
        device (far ~name:(if peer then "PEER" else "FAR") ~budget ()));
    command "create on the host"
      (Gen.bool @-> formats @-> counts @-> makes buf)
      Model.create_host
      (fun big s n ->
        some (B.create host s (if big then page * 8 / S.bitsize s else n)));
    command "create"
      (dev ^-> memories @-> formats @-> counts @-> makes buf)
      Model.create
      (fun d memory s n ->
        created d.fake.dev s n (fun () -> B.create ~memory d.fake.dev s n));
    command "view"
      (buf ^-> windows ^-> makes buf)
      Model.view
      (fun s (offset, f, n) -> some (B.view (get s) ~offset f n));
    command "write"
      (seeds @-> buf ^-> returns unit)
      Model.write
      (fun seed s ->
        let b = get s in
        write b (pattern seed (B.nbytes b)));
    copy;
    copy;
    command "drop"
      (buf ^-> returns unit)
      Model.drop
      (fun s ->
        s.b <- None;
        Gc.full_major ());
    command "set_budget"
      (dev ^-> edges ^-> returns unit)
      Model.set_budget
      (fun d n -> Nx_device.set_budget d.fake.dev n);
    command "free_cache"
      (dev ^-> returns int)
      (Fun.const 0)
      (fun d ->
        Nx_device.free_cache d.fake.dev;
        cached d.fake.dev);
  ]
  @
  (* Listed twice, so that memory is often borrowed again. *)
  let borrow =
    command "borrow"
      (dev ^-> buf ^-> makes buf)
      Model.borrow
      (fun d s ->
        match B.borrow d.fake.dev (get s) with
        | Ok b -> some b
        | Error _ -> raise Model.Refused)
  in
  [ borrow; borrow ]

let memory_kind = Testable.make ~pp:pp_memory ~equal:( = )
let all_memories = [ B.Device; Pinned; Mapped ]

let memories =
  group "memories"
    [
      test "every memory of a device the host addresses is its own" (fun () ->
          List.iter
            (fun m ->
              let f = fake () in
              let msg = Format.asprintf "%a" pp_memory m in
              dropped (fun () -> B.create ~memory:m f.dev S.UInt8 64);
              let held = f.drv.held in
              let b = B.create f.dev S.UInt8 64 in
              equal ~msg memory_kind Device (memory_of f b);
              equal ~msg:(msg ^ ", whose cache serves it") int held f.drv.held)
            all_memories);
      test "pinned memory counts in no budget" (fun () ->
          let f = far ~budget:64 () in
          let pinned = B.create ~memory:Pinned f.dev S.UInt8 64 in
          let own = B.create f.dev S.UInt8 64 in
          equal ~msg:"the device's own memory, beside pinned memory" memory_kind
            Device (memory_of f own);
          equal ~msg:"both allocated" int 128 (allocated f.dev);
          ignore (Sys.opaque_identity pinned));
      test
        "mapped memory that the window or the device's own memory cannot hold \
         is pinned memory, and the cache stays" (fun () ->
          let f = far ~budget:96 ~window:64 () in
          dropped (fun () -> B.create f.dev S.UInt8 16);
          let own = B.create f.dev S.UInt8 64 in
          let pinned =
            List.map
              (fun (msg, n) ->
                let b = B.create ~memory:Mapped f.dev S.UInt8 n in
                equal ~msg memory_kind Pinned (memory_of f b);
                b)
              [
                ("beyond the device's own memory", 48);
                ("beyond the window", 80);
                ("beyond the budget", 128);
              ]
          in
          equal ~msg:"the cache" int 16 (cached f.dev);
          ignore (Sys.opaque_identity (own, pinned)));
      test "a device with memory of its own and a window gives each memory"
        (fun () ->
          let f = far ~window:1024 () in
          List.iter
            (fun m ->
              equal
                ~msg:(Format.asprintf "%a" pp_memory m)
                memory_kind m
                (memory_of f (B.create ~memory:m f.dev S.UInt8 64)))
            all_memories);
      test "mapped memory is pinned memory on a device with no window"
        (fun () ->
          let f = far () in
          equal memory_kind Pinned
            (memory_of f (B.create ~memory:Mapped f.dev S.UInt8 64)));
      test "mapped memory is pinned memory once the window has no room"
        (fun () ->
          let f = far ~window:100 () in
          let first = B.create ~memory:Mapped f.dev S.UInt8 64 in
          let second = B.create ~memory:Mapped f.dev S.UInt8 64 in
          equal
            (pair memory_kind memory_kind)
            (Mapped, Pinned)
            (memory_of f first, memory_of f second));
      test
        "the host addresses mapped memory, which copies reach without staging"
        (fun () ->
          let f = far ~window:1024 () in
          let b = B.create ~memory:Mapped f.dev S.UInt8 64 in
          let own = B.create f.dev S.UInt8 64 in
          is_true ~msg:"a host address"
            (Region.host_address (Region.of_buffer b) <> None);
          let staged = f.drv.staged in
          write b (pattern 1 64);
          B.copy ~src:b ~dst:own;
          write b (pattern 2 64);
          B.copy ~src:own ~dst:b;
          equal ~msg:"no staged copy" int staged f.drv.staged;
          equal ~msg:"its bytes, there and back" string (pattern 1 64) (read b));
      test "freed mapped memory is cached as mapped memory alone" (fun () ->
          let f = far ~window:64 () in
          dropped (fun () -> B.create ~memory:Mapped f.dev S.UInt8 64);
          let pinned = B.create ~memory:Pinned f.dev S.UInt8 64 in
          let mapped = B.create ~memory:Mapped f.dev S.UInt8 64 in
          equal
            (pair memory_kind memory_kind)
            (Pinned, Mapped)
            (memory_of f pinned, memory_of f mapped));
      test
        "another device borrows mapped memory through its peer mapping, as the \
         device's own" (fun () ->
          let peered = ref 0 in
          let peer _ r =
            incr peered;
            Ok (r, ignore)
          in
          let o = far ~name:"OWNER" ~window:1024 () in
          let a = far ~name:"MAPPER" ~peer () in
          let b = B.create ~memory:Mapped o.dev S.UInt8 64 in
          let mappings = a.drv.mapped in
          let on_a = borrow a.dev b in
          equal ~msg:"a peer mapping" int 1 !peered;
          equal ~msg:"no host mapping" int mappings a.drv.mapped;
          ignore (Sys.opaque_identity on_a));
      test
        "a mapped request the window refuses first frees cached mapped memory"
        (fun () ->
          let f = far ~window:64 () in
          dropped (fun () ->
              ( B.create ~memory:Mapped f.dev S.UInt8 32,
                B.create f.dev S.UInt8 32 ));
          let frees = f.drv.frees in
          equal memory_kind Mapped
            (memory_of f (B.create ~memory:Mapped f.dev S.UInt8 64));
          equal ~msg:"the cached mapped region alone is freed" int (frees + 1)
            f.drv.frees);
      test
        "the host borrows mapped memory over its host address, as pinned memory"
        (fun () ->
          List.iter
            (fun (window, m) ->
              let f = far ?window () in
              let b = B.create ~memory:Mapped f.dev S.UInt8 64 in
              let msg = Format.asprintf "%a" pp_memory m in
              equal ~msg memory_kind m (memory_of f b);
              write b (pattern 4 64);
              match B.borrow host b with
              | Error why -> failf "%s: %s" msg why
              | Ok h ->
                  equal
                    ~msg:(msg ^ ", over its host address")
                    bool true
                    (Some (B.address h)
                    = Region.host_address (Region.of_buffer b));
                  equal ~msg:(msg ^ ", its bytes") string (pattern 4 64)
                    Bigarray.Array1.(
                      let ba = B.bigarray Bigarray.char h in
                      String.init (dim ba) (fun i -> unsafe_get ba i)))
            [ (Some 1024, B.Mapped); (None, Pinned) ]);
      test "mapped memory counts in its device's budget" (fun () ->
          let f = far ~budget:100 ~window:1000 () in
          let b = B.create ~memory:Mapped f.dev S.UInt8 80 in
          equal ~msg:"allocated" int 80 (allocated f.dev);
          raises_match (out_of_memory f.dev 40) (fun () ->
              B.create f.dev S.UInt8 40);
          ignore (Sys.opaque_identity b));
    ]

(* Memory that finaliser closures keep, two deep, returns to an allocation that
   needs it: each closure releases its hold a cycle after its value dies, which
   one collection does not cover. *)
let test_finaliser_chain () =
  let d = (fake ~name:"CHAIN" ~budget:4096 ()).dev in
  (fun () ->
    let b = B.create d S.UInt8 4096 in
    let inner = ref 0 and outer = ref 0 in
    Gc.finalise (fun _ -> ignore (Sys.opaque_identity b)) inner;
    Gc.finalise (fun _ -> ignore (Sys.opaque_identity inner)) outer)
    ();
  equal int 4096 (B.nbytes (B.create d S.UInt8 4096))

(* Allocating many budgets of a device's memory in dropped buffers, a sixteenth
   of its budget each, never collects by force: the collector is paced by the
   device's memory, and finds the dropped buffers before the budget runs out. *)
let test_paced () =
  let budget = 64 lsl 20 in
  let d = (fake ~name:"PACED" ~budget ()).dev in
  let forced () = (Gc.quick_stat ()).forced_major_collections in
  let before = forced () in
  for _ = 1 to 20 * 16 do
    ignore (Sys.opaque_identity (B.create d S.UInt8 (budget / 16)))
  done;
  equal int 0 (forced () - before)

(* The host's cache of collected buffers shrinks to its bound as major cycles
   end, whichever domain runs them: here while the domain that made the buffers
   is blocked. *)
let test_measured_while_blocked () =
  let held = List.init 48 (fun _ -> B.create host S.UInt8 (4 lsl 20)) in
  Gc.full_major ();
  Gc.full_major ();
  ignore (Sys.opaque_identity held);
  let worker =
    Domain.spawn (fun () ->
        for _ = 1 to 4 do
          Gc.full_major ()
        done;
        cached host)
  in
  at_most int ~than:(32 lsl 20) (Domain.join worker)

(* A domain blocked in a lock runs no OCaml code, its finalisers included. *)
let test_dropped_by_blocked_domain () =
  let d = (fake ()).dev in
  let lock = Mutex.create () and dropped = Atomic.make false in
  Mutex.lock lock;
  let blocked =
    Domain.spawn (fun () ->
        ignore (Sys.opaque_identity (B.create d S.UInt8 4096));
        Atomic.set dropped true;
        Mutex.lock lock;
        Mutex.unlock lock)
  in
  while not (Atomic.get dropped) do
    Domain.cpu_relax ()
  done;
  Gc.full_major ();
  Gc.full_major ();
  let held = allocated d in
  Mutex.unlock lock;
  Domain.join blocked;
  equal int 0 held

let memory =
  group "memory"
    [
      stateful ~count:300 ~steps:50
        "buffers hold what was written through them and their views, and \
         devices hold their live buffers and cache within their budget and \
         count what is copied (nx_device.mli is silent on the alignment of a \
         new buffer: 16 bytes assumed)"
        commands;
      test "a buffer dropped by a domain that then blocks returns to its device"
        test_dropped_by_blocked_domain;
      test
        "memory that finaliser closures hold, two deep, returns to an \
         allocation that needs it"
        test_finaliser_chain;
      test "many budgets of dropped buffers of a device never collect by force"
        test_paced;
      test
        "the host's cache shrinks as cycles end on any domain, the one that \
         made its buffers blocked"
        test_measured_while_blocked;
      test
        "four domains create, write and read buffers of one device at once, \
         and all their memory returns" (fun () ->
          let d = (fake ()).dev in
          let work k () =
            for i = 1 to 200 do
              let n = 1 + (((i * 37) + k) mod 512) in
              let b = B.create d S.UInt8 n and s = pattern (i + k) n in
              write b s;
              equal string s (read b)
            done;
            Gc.full_major ()
          in
          List.iter Domain.join (List.init 4 (fun k -> Domain.spawn (work k)));
          Gc.full_major ();
          equal int 0 (allocated d));
      test
        "host memory counts while its buffer lives, returns when it is \
         collected, and keeps within the host's budget" (fun () ->
          Gc.full_major ();
          let s0 = stats host in
          let grown () = Nx_device.Stats.(allocated (diff s0 (stats host))) in
          dropped (fun () ->
              let b = B.create host S.UInt8 1000 in
              equal ~msg:"counted" int 1000 (grown ());
              b);
          equal ~msg:"returned" int 0 (grown ());
          Fun.protect ~finally:(fun () -> Nx_device.set_budget host max_int)
          @@ fun () ->
          (* A refused allocation gets back the memory that earlier tests' idle
             devices still borrow, so that only live buffers count. *)
          Nx_device.set_budget host (allocated host);
          (try ignore (B.create host S.UInt8 1)
           with Nx_device.Out_of_memory _ -> ());
          Nx_device.set_budget host (allocated host + 1000);
          let a = B.create host S.UInt8 600 in
          raises_match (out_of_memory host 600) (fun () ->
              B.create host S.UInt8 600);
          ignore (Sys.opaque_identity a));
      test
        "the bigarray of a host buffer on a page is one that bigarray \
         functions handle and the collector frees" (fun () ->
          List.iter
            (fun n ->
              let freed = ref false in
              (fun () ->
                let ba = B.bigarray Bigarray.char (B.create host S.UInt8 n) in
                Gc.finalise_last (fun () -> freed := true) ba;
                let s = pattern n n in
                String.iteri (Bigarray.Array1.set ba) s;
                let tail = Bigarray.Array1.sub ba 3 (n - 3) in
                Bigarray.Array1.blit tail (Bigarray.Array1.sub ba 0 (n - 3));
                let moved = String.sub s 3 (n - 3) ^ String.sub s (n - 3) 3 in
                equal ~msg:"sub and blit" string moved (string_of ba);
                let back : chars =
                  Marshal.from_string (Marshal.to_string ba []) 0
                in
                equal ~msg:"marshalled" string moved (string_of back);
                ignore (Sys.opaque_identity tail))
                ();
              Gc.full_major ();
              Gc.full_major ();
              is_true ~msg:"collected" !freed)
            [ page; (4 * page) + 3 ]);
      test
        "a collected host buffer's memory is kept for the next buffer of its \
         size, unless a view of it lives, until the cache is freed" (fun () ->
          let n = (4 * page) + 4093 in
          let made () = B.address (B.create host S.UInt8 n) in
          let first = made () in
          Gc.full_major ();
          Gc.full_major ();
          is_true ~msg:"kept" (cached host >= n);
          equal ~msg:"reused" nativeint first (made ());
          Gc.full_major ();
          Nx_device.free_cache host;
          equal ~msg:"given back" int 0 (cached host);
          let a, view =
            (fun () ->
              let b = B.create host S.UInt8 n in
              write b (pattern 7 n);
              (B.address b, B.bigarray Bigarray.char b))
              ()
          in
          Gc.full_major ();
          Gc.full_major ();
          let b = B.create host S.UInt8 n in
          is_false ~msg:"not the viewed memory"
            (Nativeint.equal a (B.address b));
          write b (pattern 9 n);
          equal ~msg:"the view keeps its bytes" string (pattern 7 n)
            (string_of view));
      test
        "the host's cache gives back a dropped working set within a cycle, \
         with no buffer freed after it" (fun () ->
          let mib = 1 lsl 20 in
          (* 160 buffers of about 1 MiB, of distinct sizes, held live through a
             cycle and then dropped: the cache may keep a cycle's share of them,
             more than 32 MiB, until the next cycle ends. *)
          dropped (fun () ->
              let held =
                List.init 160 (fun i ->
                    B.create host S.UInt8 (mib + (i * page)))
              in
              Gc.full_major ();
              held);
          Gc.full_major ();
          Gc.full_major ();
          is_true ~msg:"at most the floor" (cached host <= 32 * mib);
          Nx_device.free_cache host);
      test
        "a buffer of up to max_int bytes that its device cannot allocate \
         raises Out_of_memory" (fun () ->
          let small = (fake ~budget:1000 ()).dev and n = (1 lsl 58) + 1 in
          List.iter
            (fun (d, s, n, bytes) ->
              raises_match (out_of_memory d bytes) (fun () -> B.create d s n))
            [
              (host, S.Float64, n, 8 * n);
              (host, S.Float64, max_int / 8, max_int / 8 * 8);
              (host, S.Int4, max_int, (max_int / 2) + 1);
              (small, S.Float64, n, 8 * n);
            ]);
      test "an allocation over the budget raises at once, and keeps the cache"
        (fun () ->
          let f = fake ~budget:1000 () in
          dropped (fun () -> B.create f.dev S.UInt8 500);
          raises_match (out_of_memory f.dev 2000) (fun () ->
              B.create f.dev S.UInt8 2000);
          equal (pair int int) (500, 0) (cached f.dev, f.drv.frees));
      test "an allocation the driver refuses releases the cache, then raises"
        (fun () ->
          let f = fake () in
          dropped (fun () -> B.create f.dev S.UInt8 100);
          f.drv.refuse <- true;
          raises_match (out_of_memory f.dev 200) (fun () ->
              B.create f.dev S.UInt8 200);
          equal (pair int int) (0, 1) (cached f.dev, f.drv.frees));
      test "an allocation over the budget collects unreachable buffers first"
        (fun () ->
          let f = fake ~budget:1000 () in
          let unreachable = ref (Some (B.create f.dev S.UInt8 600)) in
          ignore (Sys.opaque_identity !unreachable);
          unreachable := None;
          equal int 600 (B.nbytes (B.create f.dev S.UInt8 600)));
    ]

(* Pools *)

(* A far device's memories as pools: its own memory under its budget, the window
   that holds its mapped memory and loaded code within its own, and pinned
   memory, the host's, under no ceiling. Live buffers count in the pools of the
   memory they got, a loaded binary's code in the window, or in pinned memory on
   a device without one, and the cache holds released regions by size and
   memory. Which cached regions a refused allocation or a budget releases is
   unspecified: after every call the cache is read from what the driver holds,
   and the change is judged. *)
module Pools = struct
  exception No_memory

  type image = { binary : string; code : int; mutable holders : int }

  type device = {
    window : int option;
    mutable budget : int;
    mutable buffers : buffer list;
    mutable images : image list;
    mutable cache : (int * B.memory) list;
    mutable releases : bool; (* the call may release cached memory *)
    mutable empties : bool; (* the call releases all of it *)
  }

  and buffer = {
    owner : device;
    bytes : int;
    memory : B.memory;
    mutable live : bool;
  }

  type program = { image : image; mutable held : bool }

  let device budget window =
    {
      window;
      budget;
      buffers = [];
      images = [];
      cache = [];
      releases = false;
      empties = false;
    }

  (* Whether memory [m] counts in the device's own memory, and in its window. *)
  let own d (m : B.memory) = m = Device || (m = Mapped && d.window <> None)
  let windowed d (m : B.memory) = m = Mapped && d.window <> None
  let pinned d m = not (own d m)

  (* A binary named ["h..."] has code the host addresses, which lies in mapped
     memory; other code lies in the device's own memory. *)
  let hosted binary = binary.[0] = 'h'

  let code_memory d i : B.memory =
    if not (hosted i.binary) then Device
    else if d.window = None then Pinned
    else Mapped

  let used d counts =
    let buffers n b = if b.live && counts b.memory then n + b.bytes else n in
    let code n i =
      if i.holders > 0 && counts (code_memory d i) then n + i.code else n
    in
    List.fold_left code (List.fold_left buffers 0 d.buffers) d.images

  let cached counts cache =
    List.fold_left (fun n (b, m) -> if counts m then n + b else n) 0 cache

  let own_room d cache = d.budget - used d (own d) - cached (own d) cache

  let in_window d cache n =
    match d.window with
    | None -> false
    | Some w -> n <= w - used d (windowed d) - cached (windowed d) cache

  (* The memory a request of [kind] gets beside [cache]: mapped memory that the
     window or the device's own memory cannot hold is pinned memory. *)
  let placed d (kind : B.memory) n cache : B.memory option =
    match kind with
    | Pinned -> Some Pinned
    | Device -> if n <= own_room d cache then Some Device else None
    | Mapped ->
        if n <= own_room d cache && in_window d cache n then Some Mapped
        else Some Pinned

  (* The memory a request of [kind] asks for: on a device with no window, mapped
     memory is pinned memory. *)
  let asked d (kind : B.memory) : B.memory =
    if kind = Mapped && d.window = None then Pinned else kind

  let rec remove x = function
    | [] -> []
    | y :: l -> if x = y then l else y :: remove x l

  let made d bytes memory =
    let b = { owner = d; bytes; memory; live = true } in
    d.buffers <- b :: d.buffers;
    b

  (* [n] bytes of [memory], from a cached region of that size and memory if
     there is one. *)
  let reused d n memory =
    let key = (n, memory) in
    if List.mem key d.cache then begin
      cover "cached memory serves a request" true;
      d.cache <- remove key d.cache
    end;
    made d n memory

  (* A request of the device's own memory over the budget is refused at once,
     keeping the cache; a cached region of its size and memory serves it; and
     one the pools refuse as asked releases cached memory when releasing all of
     it lets them serve it as asked. Mapped memory they cannot serve as asked
     even then is pinned memory, cached pinned memory included, and keeps the
     rest of the cache. *)
  let create d kind n =
    let kind = asked d kind in
    if kind = B.Device && n > d.budget then raise No_memory;
    if List.mem (n, kind) d.cache then reused d n kind
    else
      match placed d kind n d.cache with
      | Some m when m = kind -> made d n m
      | Some _ | None -> (
          match placed d kind n [] with
          | Some m when m = kind ->
              cover "a request the cache crowds out is served" true;
              d.releases <- true;
              made d n m
          | Some m ->
              cover "mapped memory the pools cannot hold is pinned memory" true;
              cover "the pinned fallback keeps a cache" (d.cache <> []);
              reused d n m
          | None ->
              d.releases <- true;
              d.empties <- true;
              raise No_memory)

  let drop b =
    if b.live then begin
      b.live <- false;
      b.owner.cache <- (b.bytes, b.memory) :: b.owner.cache
    end

  let code_of binary =
    int_of_string (String.sub binary 1 (String.length binary - 1))

  (* A binary loads once while a program of it is held, and its code counts
     whatever room is left. *)
  let load d binary =
    let image =
      match List.find_opt (fun i -> i.binary = binary) d.images with
      | Some i -> i
      | None ->
          let i = { binary; code = code_of binary; holders = 0 } in
          d.images <- i :: d.images;
          i
    in
    image.holders <- image.holders + 1;
    cover "loaded code leaves no room" (own_room d d.cache < 0);
    { image; held = true }

  let unload p =
    if p.held then begin
      p.held <- false;
      p.image.holders <- p.image.holders - 1
    end

  let set_budget d n =
    if n < 0 then invalid_arg "set_budget";
    d.budget <- n

  let free_cache d =
    d.releases <- true;
    d.empties <- true

  (* [l] less the elements of [l'], if it holds them all. *)
  let rec without l l' =
    match l' with
    | [] -> Some l
    | x :: l' -> if List.mem x l then without (remove x l) l' else None

  let pp_region ppf (n, m) = Format.fprintf ppf "%d %a" n pp_memory m
  let regions = slist (Testable.make ~pp:pp_region ~equal:( = )) compare

  (* The cache that [held], what the driver holds for buffers, leaves beside the
     live buffers, judged against the cache before the call, then kept. *)
  let observe d held =
    let live =
      List.filter_map
        (fun b -> if b.live then Some (b.bytes, b.memory) else None)
        d.buffers
    in
    let cache =
      match without held live with
      | Some cache -> cache
      | None -> failf "the driver holds every live buffer"
    in
    if without d.cache cache = None then
      equal ~msg:"only released buffers enter the cache" regions d.cache cache;
    if cache <> [] && own_room d cache < 0 then
      equal ~msg:"the cache keeps within the budget, or is empty" regions []
        cache;
    cover "the cache is released to keep within the budget"
      (own_room d d.cache < 0 && cache <> d.cache);
    if (not d.releases) && own_room d d.cache >= 0 then
      equal ~msg:"cached memory stays until a release" regions d.cache cache;
    if d.empties then equal ~msg:"all cached memory released" regions [] cache;
    d.cache <- cache;
    d.releases <- false;
    d.empties <- false
end

(* A far device that loads binaries whose code is as many bytes as the number
   their name ends with, which the host addresses when their name starts with
   ['h'], and the regions its driver held once made: its timeline's. *)
type pool_device = { pools : fake; timeline : (int * B.memory) list }
type pool_buffer = { on : fake; mutable pb : B.t option }
type loaded = { by : fake; mutable p : Nx_device.Program.t option }

(* The regions [f]'s driver holds, by size and memory. *)
let held f =
  Hashtbl.fold
    (fun a (_, n) l -> (n, Hashtbl.find f.drv.memories a) :: l)
    f.drv.blocks []

let code_loads = ref 0

let load_code ~binary =
  incr code_loads;
  let at = Nativeint.of_int (0x10000 * !code_loads) in
  let host = if Pools.hosted binary then Some at else None in
  let code = Region.v ?host at (Pools.code_of binary) in
  Ok { Driver.code = Some code; entry = (fun _ -> Ok 1n); unload = ignore }

let pools_invariant (r : Pools.device) { pools; timeline } =
  let st = stats pools.dev in
  (match Pools.without (held pools) timeline with
  | Some held -> Pools.observe r held
  | None -> fail "the driver holds the timeline");
  equal ~msg:"allocated: live buffers and loaded code" int
    (Pools.used r (Pools.own r) + Pools.used r (Pools.pinned r))
    (Nx_device.Stats.allocated st);
  equal ~msg:"cached" int
    (Pools.cached (Fun.const true) r.cache)
    (Nx_device.Stats.cached st);
  equal ~msg:"retained" int 0 (Nx_device.Stats.retained st);
  equal ~msg:"budget" int r.budget (Nx_device.budget pools.dev)

let pool_buffer_invariant (r : Pools.buffer) s =
  match s.pb with
  | None -> ()
  | Some b ->
      equal ~msg:"bytes" int r.bytes (B.nbytes b);
      equal ~msg:"its memory" memory_kind r.memory (memory_of s.on b)

let pdev = abstract "d" ~invariant:pools_invariant
let pbuf = abstract "b" ~invariant:pool_buffer_invariant

let prog =
  abstract "p" ~invariant:(fun (r : Pools.program) s ->
      match s.p with
      | None -> ()
      | Some p ->
          equal ~msg:"its code" (option int) (Some r.image.code)
            (Option.map B.nbytes (Nx_device.Program.code p)))

(* [d]'s unreachable buffers and programs, collected and released. *)
let collected d =
  Gc.full_major ();
  ignore (stats d)

let pool_commands =
  let windows =
    let pp ppf = function
      | None -> Format.pp_print_string ppf "no window"
      | Some w -> Format.fprintf ppf "window %d" w
    in
    Gen.of_list ~pp [ None; Some 32; Some 96 ]
  in
  let requests =
    Gen.frequency
      [
        (2, Gen.constant ~pp:pp_memory B.Device);
        (2, Gen.constant ~pp:pp_memory B.Mapped);
        (1, Gen.constant ~pp:pp_memory B.Pinned);
      ]
  in
  let edges =
    among (arg pp_budget) pdev (fun r ->
        let own = Pools.used r (Pools.own r) in
        [ own; own + 16; own + 64; r.budget / 2; 0; max_int; -1 ])
  in
  let drop =
    command "drop"
      (pbuf ^-> returns unit)
      Pools.drop
      (fun s ->
        s.pb <- None;
        collected s.on.dev)
  in
  [
    command "device"
      (ints [ 64; 128; 256 ] @-> windows @-> makes pdev)
      Pools.device
      (fun budget window ->
        let pools =
          fake ~name:"POOLS" ~far:true ~budget ?window ~load:load_code ()
        in
        { pools; timeline = held pools });
    command "create"
      (pdev ^-> requests @-> ints [ 16; 32; 64; 96 ] @-> makes pbuf)
      Pools.create
      (fun d memory n ->
        match B.create ~memory d.pools.dev S.UInt8 n with
        | b -> { on = d.pools; pb = Some b }
        | exception (Nx_device.Out_of_memory _ as e) ->
            is_true (out_of_memory d.pools.dev n e);
            raise Pools.No_memory);
    drop;
    drop;
    command "load"
      (pdev
      ^-> Gen.of_list ~pp:Format.pp_print_string [ "a16"; "h48"; "c160"; "h96" ]
      @-> makes prog)
      Pools.load
      (fun d binary ->
        { by = d.pools; p = Some (program d.pools.dev ~binary ~name:"f") });
    command "unload"
      (prog ^-> returns unit)
      Pools.unload
      (fun s ->
        s.p <- None;
        collected s.by.dev);
    command "set_budget"
      (pdev ^-> edges ^-> returns unit)
      Pools.set_budget
      (fun d n -> Nx_device.set_budget d.pools.dev n);
    command "free_cache"
      (pdev ^-> returns unit)
      Pools.free_cache
      (fun d -> Nx_device.free_cache d.pools.dev);
  ]

(* Last uses *)

(* A device's buffers, its work on them, and a reader's, another device's work
   that touches them. Released memory enters the cache once the reader's work on
   it is done, and a create of its size reuses it at once, the device's own work
   being ordered by its queue. A buffer that the reader's work touches waits, on
   the host, for the latest of the device's work that touched its memory, under
   any buffer, and a free of the cache for at most all the device submitted. The
   cache holds one memory per size, so that which one a create reuses is
   known. *)
module Uses = struct
  type memory = {
    size : int;
    mutable touched : int; (* the device's latest work on it *)
    mutable read : int; (* the reader's latest work on it *)
  }

  type device = {
    mutable submitted : int;
    mutable signaled : int;
    mutable reads : int; (* the reader's submitted work *)
    mutable reads_done : int;
    mutable live : memory list;
    mutable retiring : memory list;
    mutable cache : memory list;
  }

  type buffer = { device : device; memory : memory; mutable dropped : bool }

  let device () =
    {
      submitted = 0;
      signaled = 0;
      reads = 0;
      reads_done = 0;
      live = [];
      retiring = [];
      cache = [];
    }

  let sizes l = List.fold_left (fun n m -> n + m.size) 0 l
  let sized n = List.find_opt (fun m -> m.size = n)

  let create d n =
    let memory =
      match sized n d.cache with
      | Some m ->
          d.cache <- List.filter (( != ) m) d.cache;
          cover "a create reuses memory whose work is not done"
            (m.touched > d.signaled);
          m
      | None -> { size = n; touched = 0; read = 0 }
    in
    d.live <- memory :: d.live;
    { device = d; memory; dropped = false }

  let work b =
    let d = b.device in
    d.submitted <- d.submitted + 1;
    b.memory.touched <- d.submitted

  (* The device's work that the host waited for, judged against the work that
     may use [ms]: at least the latest that touched them, and at most [upto]. *)
  let waited d ms ~upto outcome =
    let touched = List.fold_left (fun v m -> Int.max v m.touched) 0 ms in
    match outcome with
    | Error e -> raise e
    | Ok None ->
        at_most ~msg:"no wait: its latest work was done" int ~than:d.signaled
          touched
    | Ok (Some v) ->
        at_least ~msg:"a wait for its latest work" int ~than:touched v;
        at_most ~msg:"a wait for no later work" int ~than:upto v;
        d.signaled <- Int.max d.signaled v

  let read b outcome =
    let d = b.device in
    waited d [ b.memory ] ~upto:b.memory.touched outcome;
    d.reads <- d.reads + 1;
    b.memory.read <- d.reads

  (* Memory the reader's work touched waits for it before entering the cache. *)
  let retire d =
    let fresh, still =
      List.partition (fun m -> m.read <= d.reads_done) d.retiring
    in
    d.retiring <- still;
    d.cache <- fresh @ d.cache

  let complete d owner =
    if owner then d.signaled <- d.submitted
    else begin
      d.reads_done <- d.reads;
      retire d
    end

  let drop b =
    let d = b.device in
    if not b.dropped then begin
      b.dropped <- true;
      d.live <- List.filter (( != ) b.memory) d.live;
      d.retiring <- b.memory :: d.retiring;
      retire d
    end

  let free_cache d outcome =
    waited d d.cache ~upto:d.submitted outcome;
    d.cache <- []

  (* Whether dropping [b] keeps one released memory per size. *)
  let alone b =
    let d = b.device and n = b.memory.size in
    sized n d.cache = None && sized n d.retiring = None
end

(* A device whose waits are recorded, and a reader that addresses its signal
   word. *)
type uses = {
  owner : fake;
  reader : fake;
  signaled : int ref;
  waits : int list ref;
  baseline : int;
}

let uses () =
  let signaled = ref 0 and waits = ref [] in
  let signal =
    {
      Driver.signaled = (fun () -> !signaled);
      wait =
        (fun v ~ms:_ ->
          waits := v :: !waits;
          signaled := Int.max !signaled v;
          true);
    }
  in
  let owner = fake ~name:"OWNER" ~maps:true ~signal () in
  let reader = fake ~name:"READER" ~maps:true () in
  { owner; reader; signaled; waits; baseline = owner.drv.held }

(* The latest of the device's work that [f] waited for on the host. *)
let waits_in u f =
  let before = !(u.waits) in
  f ();
  let rec since = function
    | l when l == before -> []
    | v :: l -> v :: since l
    | [] -> []
  in
  match since !(u.waits) with
  | [] -> None
  | vs -> Some (List.fold_left Int.max 0 vs)

let uses_invariant (r : Uses.device) u =
  let st = stats u.owner.dev in
  let live = Uses.sizes r.live and retiring = Uses.sizes r.retiring in
  let cached = Uses.sizes r.cache in
  equal ~msg:"allocated: live buffers and memory not yet returned" int
    (live + retiring)
    (Nx_device.Stats.allocated st);
  equal ~msg:"cached" int cached (Nx_device.Stats.cached st);
  equal ~msg:"the driver holds the buffers, released or not" int
    (live + retiring + cached)
    (u.owner.drv.held - u.baseline);
  equal ~msg:"signaled" int r.signaled (Nx_device.signaled u.owner.dev)

type use_buffer = { uses : uses; mutable ub : B.t option }

(* The reader's work completes as the program ends, or the device would wait for
   it at exit. *)
let udev =
  abstract "d" ~invariant:uses_invariant ~release:(fun u ->
      store_signal
        (B.address (Nx_device.signal_word u.reader.dev))
        (Nx_device.submitted u.reader.dev))

let ubuf = abstract "b"
let use_get s = match s.ub with Some b -> b | None -> raise Model.Dropped

let uses_commands =
  let waited = judges (option int) in
  [
    command "devices" (Gen.unit @-> makes udev) Uses.device uses;
    command "create"
      (udev ^-> ints [ 16; 32; 48 ] @-> makes ubuf)
      Uses.create
      (fun u n -> { uses = u; ub = Some (B.create u.owner.dev S.UInt8 n) });
    (* The device's work waits on the host for the reader's unfinished work on
       any of its memory, which completes only by a call. *)
    command "work"
      ~pre:(fun (b : Uses.buffer) ->
        (not b.dropped) && b.device.reads_done = b.device.reads)
      (ubuf ^-> returns unit)
      Uses.work
      (fun s -> ignore (submit s.uses.owner.dev ~touches:[ use_get s ] Fun.id));
    command "read"
      ~pre:(fun (b : Uses.buffer) -> not b.dropped)
      (ubuf ^-> waited) Uses.read
      (fun s ->
        let b = use_get s in
        waits_in s.uses (fun () ->
            ignore (submit s.uses.reader.dev ~touches:[ b ] Fun.id)));
    command "complete"
      (udev ^-> Gen.bool @-> returns unit)
      Uses.complete
      (fun u owner ->
        if owner then u.signaled := Nx_device.submitted u.owner.dev
        else
          store_signal
            (B.address (Nx_device.signal_word u.reader.dev))
            (Nx_device.submitted u.reader.dev));
    command "drop"
      ~pre:(fun (b : Uses.buffer) -> (not b.dropped) && Uses.alone b)
      (ubuf ^-> returns unit)
      Uses.drop
      (fun s ->
        s.ub <- None;
        collected s.uses.owner.dev);
    command "free_cache" (udev ^-> waited) Uses.free_cache (fun u ->
        waits_in u (fun () -> Nx_device.free_cache u.owner.dev));
  ]

let pools =
  group "pools"
    [
      stateful ~count:100 ~steps:40
        "buffers and loaded code count in the pools of their memory, loaded \
         code even beyond the budget, mapped memory that the window or the \
         device's own memory cannot hold is pinned memory, a request over the \
         budget keeps the cache, and the cache keeps within the budget"
        pool_commands;
      stateful ~count:200 ~steps:40
        "released memory is cached once other devices' work on it is done and \
         reused at once, and a reused buffer waits for the latest work that \
         touched its memory"
        uses_commands;
    ]

(* Buffers *)

let near = fake ()
let far_one = far ()

(* A buffer of [n] elements of [s], [k] elements into a larger one on the host,
   a device the host addresses, one it does not, or that one's host memory. *)
let placed =
  let place =
    Gen.of_list
      ~pp:(fun ppf (d, memory) ->
        Format.fprintf ppf "on %a%s" pp_device d
          (if memory = B.Pinned then "'s host memory" else ""))
      [
        (host, B.Device);
        (near.dev, Device);
        (far_one.dev, Device);
        (far_one.dev, Pinned);
      ]
  in
  Gen.quad place formats (ints [ 0; 1; 2; 3; 17; 1000 ]) (ints [ 0; 1; 3 ])

let inside ((d, memory), s, n, k) =
  let o = k * Model.element s in
  B.view (B.create ~memory d S.UInt8 (o + Model.nbytes s n + 5)) ~offset:o s n

type kind = Kind : ('a, 'b) Bigarray.kind * S.t -> kind

let kinds =
  let k name kind s = (name, Kind (kind, s)) in
  Gen.of_list
    ~pp:(fun ppf (name, _) -> Format.pp_print_string ppf name)
    Bigarray.
      [
        k "float16" float16 S.Float16;
        k "float32" float32 S.Float32;
        k "float64" float64 S.Float64;
        k "int8_signed" int8_signed S.Int8;
        k "int8_unsigned" int8_unsigned S.UInt8;
        k "char" char S.UInt8;
        k "int16_signed" int16_signed S.Int16;
        k "int16_unsigned" int16_unsigned S.UInt16;
        k "int32" int32 S.Int32;
        k "int64" int64 S.Int64;
        k "complex32" complex32 S.Complex64;
        k "complex64" complex64 S.Complex128;
      ]

let laws =
  group "laws"
    [
      prop
        "a copy there and back is the identity, and counts its bytes on both \
         sides"
        (Gen.pair placed seeds) (fun ((((d, _), _, _, _) as spec), seed) ->
          let b = inside spec in
          let n = B.nbytes b and h0 = stats host and d0 = stats d in
          let there s =
            write b s;
            b
          in
          Law.round_trip string
            (arg (fun ppf _ -> Format.fprintf ppf "b"))
            there read (pattern seed n);
          let h = Nx_device.Stats.diff h0 (stats host)
          and d = Nx_device.Stats.diff d0 (stats d) in
          let moved = if Nx_device.equal (B.device b) host then 0 else n in
          equal
            ~msg:"out of the host, into and out of the device, into the host"
            (list int)
            [ moved; moved; moved; moved ]
            Nx_device.Stats.[ bytes_out h; bytes_in d; bytes_out d; bytes_in h ]);
      prop
        "of_bigarray aliases its bigarray, a borrowed host buffer of its \
         kind's format, and bigarray is its inverse"
        (Gen.pair kinds (ints [ 0; 1; 3; 17 ]))
        (fun ((_, Kind (k, s)), n) ->
          let ba = Bigarray.Array1.create k Bigarray.c_layout n in
          let b = B.of_bigarray ba in
          equal string (S.to_string s) (S.to_string (B.dtype b));
          equal (list int)
            [ n; Bigarray.Array1.size_in_bytes ba ]
            [ B.length b; B.nbytes b ];
          is_true (B.is_borrowed b && Nx_device.equal (B.device b) host);
          let other = Bigarray.Array1.create k Bigarray.c_layout n in
          let bytes = pattern n (B.nbytes b) in
          write b bytes;
          Bigarray.Array1.blit ba other;
          equal ~msg:"a write through the buffer" string bytes
            (read (B.of_bigarray other));
          write (B.of_bigarray other) (pattern (n + 1) (B.nbytes b));
          Bigarray.Array1.blit other ba;
          equal ~msg:"a write through the bigarray" string
            (pattern (n + 1) (B.nbytes b))
            (read b);
          let back = B.bigarray k b in
          equal ~msg:"bigarray gives the elements back" string (read b)
            (read (B.of_bigarray back));
          is_true ~msg:"over the same memory"
            (n = 0
            || Nativeint.equal (B.address b) (B.address (B.of_bigarray back))));
      prop
        "a buffer lies in its region: its views share the region, at their \
         offsets, and its address is the region's plus its offset"
        (Gen.pair placed (ints [ 0; 1; 3 ]))
        (fun (spec, k) ->
          let b = inside spec in
          let r = Region.of_buffer b and k = Int.min k (B.nbytes b) in
          let v = B.view b ~offset:k S.UInt8 (B.nbytes b - k) in
          is_true ~msg:"one region" (Region.of_buffer v == r);
          equal ~msg:"the view's offset" int (B.offset b + k) (B.offset v);
          equal ~msg:"the address" nativeint
            (Nativeint.add (Region.address r) (Nativeint.of_int (B.offset b)))
            (B.address b);
          equal ~msg:"spans" bool
            (B.offset b = 0 && B.nbytes b = Region.nbytes r)
            (B.spans b);
          equal ~msg:"a buffer overlaps itself and its views" (pair bool bool)
            (B.nbytes b > 0, B.nbytes v > 0)
            (B.overlaps b b, B.overlaps b v));
    ]

(* Two domains take bigarrays of the same fresh buffer at once, and one keeps
   its bigarray: it shares the buffer's storage, so it outlives the buffer
   whichever domain made the storage's proxy. *)
let test_concurrent_views () =
  let rounds = 2000 in
  let current = Atomic.make None
  and taken = Array.init 2 (fun _ -> Atomic.make 0) in
  let kept = ref [] in
  let worker k () =
    for r = 0 to rounds - 1 do
      let rec next () =
        match Atomic.get current with
        | Some b when Atomic.get taken.(k) = r -> b
        | _ ->
            Domain.cpu_relax ();
            next ()
      in
      let v = B.bigarray Bigarray.int32 (next ()) in
      if k = 0 then begin
        Bigarray.Array1.fill v 7l;
        kept := v :: !kept
      end;
      Atomic.incr taken.(k)
    done
  in
  let workers = List.init 2 (fun k -> Domain.spawn (worker k)) in
  for r = 0 to rounds - 1 do
    Atomic.set current (Some (B.create host S.Int32 16));
    while Atomic.get taken.(0) <= r || Atomic.get taken.(1) <= r do
      Domain.cpu_relax ()
    done
  done;
  Atomic.set current None;
  List.iter Domain.join workers;
  Gc.full_major ();
  Gc.full_major ();
  let filler =
    List.init 20000 (fun _ ->
        let a = Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout 16 in
        Bigarray.Array1.fill a 0x55l;
        a)
  in
  equal ~msg:"kept bigarrays that lost what was written" int 0
    (List.length (List.filter (fun v -> v.{5} <> 7l) !kept));
  ignore (Sys.opaque_identity filler)

(* A mapping whose last borrow was released waits for the work on it: a borrow
   made meanwhile takes it again, and keeps it mapped. *)
let test_borrow_again () =
  let opened = ref false in
  let g = fake ~maps:true ~signal:(gate opened) () in
  let hb = B.create host S.UInt8 page in
  dropped (fun () ->
      let bm = borrow g.dev hb in
      ignore (submit g.dev ~touches:[ bm ] Fun.id));
  ignore (stats g.dev);
  let again = borrow g.dev hb in
  opened := true;
  ignore (stats g.dev);
  equal ~msg:"mapped while borrowed again" int 1 g.drv.mapped;
  write (B.view again ~offset:0 S.UInt8 4) "abcd";
  equal ~msg:"through it" string "abcd" (read (B.view hb ~offset:0 S.UInt8 4));
  ignore (Sys.opaque_identity again);
  dropped ignore;
  ignore (stats g.dev);
  ignore (stats g.dev);
  equal ~msg:"unmapped once released" int 0 g.drv.mapped

(* A borrow's release waits for its device's next operation. A device that runs
   none still gives back the memory of another that needs it: the allocation
   that is refused drains the devices that map that memory. *)
let test_idle_mapper () =
  let g = fake ~name:"IDLE" ~maps:true () in
  let n = 1 lsl 20 and budget = Nx_device.budget host in
  Fun.protect ~finally:(fun () -> Nx_device.set_budget host budget) @@ fun () ->
  Gc.full_major ();
  Nx_device.set_budget host (allocated host + n + (n / 2));
  dropped (fun () ->
      let hb = B.create host S.UInt8 n in
      ignore (Sys.opaque_identity (borrow g.dev hb)));
  let b = B.create host S.UInt8 n in
  equal ~msg:"unmapped" int 0 g.drv.mapped;
  ignore (Sys.opaque_identity b)

let test_borrow_lifetime () =
  let collected = ref false and opened = ref false in
  let g = fake ~maps:true ~signal:(gate opened) () in
  (fun () ->
    let hb = B.create host S.UInt8 (1 lsl 20) in
    Gc.finalise_last (fun () -> collected := true) hb;
    let bm = borrow g.dev hb in
    ignore (submit g.dev ~touches:[ bm ] Fun.id);
    ignore (Sys.opaque_identity bm))
    ();
  Gc.full_major ();
  ignore (stats g.dev);
  Gc.full_major ();
  Gc.full_major ();
  is_false ~msg:"kept while the work is unsignaled" !collected;
  opened := true;
  ignore (stats g.dev);
  Gc.full_major ();
  Gc.full_major ();
  is_true ~msg:"collected once it is signaled" !collected

(* Which devices' queues run a copy, and how many of their copies go through the
   host's staging memory. *)
let routes =
  let a = far () and b = far () in
  let p = far ~name:"PEER-A" () and q = far ~name:"PEER-B" () in
  let fakes = [ a; b; p; q ] in
  let mapped = B.create host S.UInt8 page in
  let borrow = borrow a.dev mapped in
  let small () = B.create host S.UInt8 8 in
  let on f () = B.create f.dev S.UInt8 8 in
  let pinned f () = B.create ~memory:Pinned f.dev S.UInt8 8 in
  let of_mapped () = B.view mapped ~offset:8 S.UInt8 8 in
  let queued () = List.map (fun f -> f.drv.queued) fakes in
  let staged () = List.fold_left (fun n f -> n + f.drv.staged) 0 fakes in
  let mappings () = List.map (fun f -> f.drv.mapped) fakes in
  let row name src dst queued staged = (name, src, dst, (queued, staged)) in
  cases
    ~name:(fun (name, _, _, _) -> name)
    "a copy runs"
    [
      row "from host memory into a device's, staged by the device" small (on a)
        [ 1; 0; 0; 0 ] 1;
      row "from a device's memory into host memory, staged by the device" (on a)
        small [ 1; 0; 0; 0 ] 1;
      row "from a device's memory into its host memory, directly" (on a)
        (pinned a) [ 1; 0; 0; 0 ] 0;
      row "from host memory the device maps, directly" of_mapped (on a)
        [ 1; 0; 0; 0 ] 0;
      row "from another device's host memory, mapped for the copy alone"
        (pinned b) (on a) [ 1; 0; 0; 0 ] 0;
      row "within a device, directly" (on a) (on a) [ 1; 0; 0; 0 ] 0;
      row "between devices that cannot reach each other, through staging" (on a)
        (on b) [ 1; 1; 0; 0 ] 2;
      row "between devices that can, by the source alone" (on p) (on q)
        [ 0; 0; 1; 0 ] 0;
    ]
    (fun (_, src, dst, expected) ->
      let src = src () and dst = dst () in
      write src "abcdefgh";
      let q0 = queued () and s0 = staged () and m0 = mappings () in
      B.copy ~src ~dst;
      equal ~msg:"copies by each device, and staged ones"
        (pair (list int) int)
        expected
        (List.map2 ( - ) (queued ()) q0, staged () - s0);
      equal ~msg:"mappings" (list int) m0 (mappings ());
      equal string "abcdefgh" (read dst);
      ignore (Sys.opaque_identity borrow))

(* Two and a half staging slots, whose bytes differ from slot to slot. *)
let test_staged_slots () =
  let f = far () and g = far () in
  let slot = 64 lsl 20 in
  let n = (2 * slot) + (slot / 2) + 12345 in
  let bytes =
    let b = Bytes.create n in
    for i = 0 to n - 1 do
      Bytes.unsafe_set b i
        (Char.unsafe_chr (((i * 7) + (i lsr 16) + (i / slot * 13)) land 0xff))
    done;
    Bytes.unsafe_to_string b
  in
  let differing s =
    let d = ref 0 in
    for i = 0 to n - 1 do
      if s.[i] <> bytes.[i] then incr d
    done;
    !d
  in
  let dev = B.view (B.create f.dev S.UInt8 (n + 1000)) ~offset:1000 S.UInt8 n in
  write dev bytes;
  equal ~msg:"bytes that differ after a round trip" int 0 (differing (read dev));
  let on_g = B.create g.dev S.UInt8 n in
  B.copy ~src:dev ~dst:on_g;
  equal ~msg:"bytes that differ on another device" int 0 (differing (read on_g));
  let tail = B.create f.dev S.UInt8 100 in
  B.copy ~src:(B.view dev ~offset:(2 * slot) S.UInt8 100) ~dst:tail;
  equal ~msg:"from an offset" string
    (String.sub bytes (2 * slot) 100)
    (read tail)

(* The disk *)

let disk = Nx_device.disk

(* A file holding [contents]. *)
let file_of contents =
  let path = temp_file () in
  Out_channel.with_open_bin path (fun oc -> output_string oc contents);
  path

(* The bytes of the file at [path], read with the system's reads. *)
let contents path = In_channel.with_open_bin path In_channel.input_all

let transferred d =
  let s = stats d in
  Nx_device.Stats.(bytes_in s, bytes_out s)

(* A copy between a file and host memory, both ways, of more bytes than one read
   or write request moves, from and to an odd offset. *)
let test_file_round_trip () =
  let n = (5 lsl 20) + 12345 and at = 4097 in
  let bytes = pattern 3 n in
  let path = temp_file () in
  let file = create_file path (at + n + 3) in
  write (B.view file ~offset:at S.UInt8 n) bytes;
  let on_disk = contents path in
  equal ~msg:"the file" int (at + n + 3) (String.length on_disk);
  equal ~msg:"bytes written" string bytes (String.sub on_disk at n);
  equal ~msg:"around them" string
    (String.make at '\000' ^ String.make 3 '\000')
    (String.sub on_disk 0 at ^ String.sub on_disk (at + n) 3);
  equal ~msg:"bytes read" string bytes
    (read (B.view (of_file path) ~offset:at S.UInt8 n))

(* Through a device whose memory the host does not address: the file's bytes are
   read into staging slots that the device copies, and a device's written from
   them. *)
let test_file_staged () =
  let f = far () in
  let slot = 64 lsl 20 in
  let n = (2 * slot) + (slot / 2) + 12345 and at = 1001 in
  let bytes = pattern 5 n in
  let file =
    B.view (create_file (temp_file ()) (at + n)) ~offset:at S.UInt8 n
  in
  let dev = B.create f.dev S.UInt8 n in
  write dev bytes;
  let staged = f.drv.staged in
  B.copy ~src:dev ~dst:file;
  equal ~msg:"slots written from the device" int 3 (f.drv.staged - staged);
  let back = B.create f.dev S.UInt8 n in
  B.copy ~src:file ~dst:back;
  equal ~msg:"slots read into the device" int 6 (f.drv.staged - staged);
  let other = create_file (temp_file ()) n in
  B.copy ~src:file ~dst:other;
  is_true ~msg:"bytes through the device and back" (read back = bytes);
  is_true ~msg:"bytes from file to file" (read other = bytes)

let test_file_closes () =
  let path = file_of "old" in
  let b = of_file path in
  let replacement = file_of "new" in
  Sys.rename replacement path;
  equal ~msg:"the file opened" string "old" (read b);
  equal ~msg:"the file now at its path" string "new" (read (of_file path))

(* The disk keeps at most this many descriptors open. *)
let max_descriptors = 64

(* Opens and reads [max_descriptors] other files, which closes every descriptor
   opened before. *)
let evict () =
  List.iter
    (fun p -> ignore (read (of_file p)))
    (List.init max_descriptors (fun _ -> file_of "x"))

(* A file used before 63 others keeps its descriptor, and reads the file it
   opened through it after its path names another; a file used before 64 others
   reopens the path, which now names another file. *)
let test_file_bound () =
  let after_others n =
    let path = file_of "old" in
    let b = of_file path in
    ignore (read b);
    List.iter
      (fun p -> ignore (read (of_file p)))
      (List.init n (fun _ -> file_of "x"));
    Sys.rename (file_of "new") path;
    (path, b)
  in
  let _, b = after_others (max_descriptors - 1) in
  equal ~msg:"open after 63 others" string "old" (read b);
  let path, b = after_others max_descriptors in
  raises_match (Exn.sys_error ~substring:path) (fun () -> read b)

let test_file_descriptors () =
  if Sys.win32 then skip ~reason:"no /dev/fd to count descriptors" ();
  let descriptors () = Array.length (Sys.readdir "/dev/fd") in
  let before = descriptors () in
  let files =
    List.init 200 (fun i -> (i, of_file (file_of (string_of_int i))))
  in
  at_most int ~than:(before + max_descriptors) (descriptors ());
  List.iter
    (fun (i, b) ->
      equal ~msg:(string_of_int i) string (string_of_int i) (read b))
    files

let test_file_reopened () =
  let path = file_of "old" in
  let b = of_file path in
  evict ();
  equal ~msg:"reopened, the same file" string "old" (read b);
  let written = create_file (temp_file ()) 3 in
  write written "abc";
  evict ();
  write (B.view written ~offset:1 S.UInt8 1) "z";
  equal ~msg:"its own writes do not change it" string "azc" (read written);
  evict ();
  Sys.rename (file_of "new") path;
  raises_match (Exn.sys_error ~substring:path) (fun () -> read b)

(* A borrow of a file's bytes is its mapping: the host and a device that maps
   host memory read the file's bytes in place, a write through it stays in the
   process, and the mapping outlives the file's buffers while it is borrowed. *)
let test_file_borrows () =
  let sharing = fake ~name:"SHARING" ~maps:true () in
  let bytes = pattern 7 (3 * page) in
  let path = file_of bytes in
  let read_before = Nx_device.Stats.bytes_out (stats disk) in
  let on_host, on_device =
    let file = of_file path in
    let window = B.view file ~offset:page S.UInt8 page in
    (borrow host window, borrow sharing.dev window)
  in
  Gc.full_major ();
  Nx_device.synchronize disk;
  equal ~msg:"the host's" string (String.sub bytes page page) (read on_host);
  equal ~msg:"a device's" string (String.sub bytes page page) (read on_device);
  equal ~msg:"bytes read" int read_before
    (Nx_device.Stats.bytes_out (stats disk));
  let pages = B.bigarray Bigarray.char on_host in
  pages.{0} <- 'z';
  equal ~msg:"a write, through the device's" char 'z' (read on_device).[0];
  equal ~msg:"and not in the file" string bytes (contents path);
  equal ~msg:"the borrows of the pages overlap, the file does not"
    (pair bool bool) (true, false)
    (B.overlaps on_host on_device, B.overlaps (of_file path) on_host)

let disks =
  group "disk"
    [
      test "DISK has no processor and a budget of max_int" (fun () ->
          equal (triple string string int) ("DISK", "", max_int)
            (Nx_device.name disk, Nx_device.arch disk, Nx_device.budget disk));
      test "a file is its bytes on DISK, borrowed, which a copy reads"
        (fun () ->
          let b = of_file (file_of "hello world") in
          equal
            (quad string bool int string)
            ("DISK", true, 11, "hello world")
            (Nx_device.name (B.device b), B.is_borrowed b, B.length b, read b);
          equal ~msg:"a view" string "world"
            (read (B.view b ~offset:6 S.UInt8 5)));
      test "a new file is zero until a copy writes it, at any offset" (fun () ->
          let path = temp_file () in
          let b = create_file path 10 in
          equal ~msg:"zeros" string (String.make 10 '\000') (read b);
          write (B.view b ~offset:3 S.UInt8 5) "abcde";
          equal string "\000\000\000abcde\000\000" (contents path);
          equal ~msg:"read again" string (contents path) (read (of_file path)));
      test "a view of a file's bytes is of any format, at any byte" (fun () ->
          let bytes = pattern 9 64 in
          let f = B.view (of_file (file_of bytes)) ~offset:3 S.Float32 4 in
          let h = B.create host S.Float32 4 in
          B.copy ~src:f ~dst:h;
          equal string (String.sub bytes 3 16)
            (read (B.view h ~offset:0 S.UInt8 16)));
      test "a copy of a file's bytes to or from memory moves every byte"
        test_file_round_trip;
      test
        "a copy through a device that the host does not address goes through \
         staging, and file to file too"
        test_file_staged;
      test
        "a read counts in DISK's bytes_out and a write in its bytes_in, \
         allocating nothing" (fun () ->
          let b = create_file (temp_file ()) 8 in
          let (i0, o0), (hi0, ho0) = (transferred disk, transferred host) in
          write b "abcdefgh";
          ignore (read (B.view b ~offset:2 S.UInt8 3));
          let (i1, o1), (hi1, ho1) = (transferred disk, transferred host) in
          equal (list int) [ 8; 3; 3; 8; 0 ]
            [ i1 - i0; o1 - o0; hi1 - hi0; ho1 - ho0; allocated disk ]);
      test "a file is read through the descriptor it was opened with"
        test_file_closes;
      test
        "the disk keeps the descriptors of the 64 files it used last, and \
         closes the least recently used"
        test_file_bound;
      test "the descriptors open stay bounded, whatever the buffers held"
        test_file_descriptors;
      test
        "a file whose descriptor was closed is reopened, and refused once its \
         path names another file"
        test_file_reopened;
      test
        "a borrow of a file's bytes is its pages, copy-on-write, kept while \
         borrowed"
        test_file_borrows;
      test "a read past the end of a file truncated since, naming it" (fun () ->
          let path = file_of (String.make 10 'x') in
          let b = of_file path in
          Unix.truncate path 4;
          raises_match (Exn.sys_error ~substring:path) (fun () ->
              read (B.view b ~offset:2 S.UInt8 5)));
    ]

(* Another device's memory, which [peer] maps where it lies: the fake drivers'
   memory is host memory. *)
let test_peer_borrows () =
  let calls = ref 0 in
  let peer _ r =
    incr calls;
    Ok (r, ignore)
  in
  let a = far ~name:"MAPPER" ~peer () and o = far ~name:"OWNER" () in
  let b = B.create o.dev S.UInt8 64 in
  write b (pattern 3 64);
  let v = B.view b ~offset:16 S.UInt8 8 in
  let on_a = borrow a.dev v and again = borrow a.dev b in
  equal ~msg:"read through the borrow" string
    (String.sub (pattern 3 64) 16 8)
    (read on_a);
  equal ~msg:"one mapping of the region" int 1 !calls;
  is_true ~msg:"borrowed on the mapper"
    (B.is_borrowed on_a && Nx_device.equal (B.device on_a) a.dev);
  is_true ~msg:"over the owner's memory"
    (B.overlaps on_a b && B.overlaps on_a again
    && not (B.overlaps on_a (B.view b ~offset:0 S.UInt8 16)));
  is_true ~msg:"a buffer on its own device borrows as itself"
    (borrow o.dev b == b);
  (match B.borrow (far ()).dev b with
  | Ok _ -> fail "borrowed without a peer"
  | Error why ->
      contains ~msg:"no peer" ~sub:"cannot address OWNER memory" (masked why));
  let refusing = far ~peer:(fun _ _ -> Error "no route") () in
  (match B.borrow refusing.dev b with
  | Ok _ -> fail "borrowed a refused region"
  | Error why -> contains ~msg:"the driver's reason" ~sub:"no route" why);
  ignore (Sys.opaque_identity (on_a, again))

(* Another device's mapping of a device's memory outlives its borrows, and is
   unmapped when the memory is released, once the mapper's work submitted until
   then is done, listed or not. *)
let test_peer_mappings () =
  let opened = ref false and maps = ref 0 and unmaps = ref 0 in
  let peer _ r =
    incr maps;
    Ok (r, fun () -> incr unmaps)
  in
  let a = fake ~name:"MAPPER" ~peer ~signal:(gate opened) () in
  let o = far ~name:"OWNER" () in
  (* A borrow's record keeps its memory until the mapper releases it, and a
     collected token's finaliser runs in the collection after. *)
  let settle () =
    for _ = 1 to 4 do
      Gc.full_major ();
      ignore (stats a.dev);
      ignore (stats o.dev)
    done
  in
  dropped (fun () ->
      let b = B.create o.dev S.UInt8 64 in
      ignore (Sys.opaque_identity (borrow a.dev b));
      settle ();
      equal ~msg:"kept once its borrows are unreachable" (pair int int) (1, 0)
        (!maps, !unmaps);
      ignore (Sys.opaque_identity (borrow a.dev b));
      equal ~msg:"shared by the next borrow" int 1 !maps;
      b);
  ignore (submit a.dev Fun.id);
  settle ();
  equal ~msg:"kept while the mapper's work runs" int 0 !unmaps;
  opened := true;
  ignore (stats o.dev);
  equal ~msg:"unmapped once the memory is released and that work is done" int 1
    !unmaps

(* A copy from one device into another's memory writes through the source's
   mapping of it, made once and unmapped when that memory is released. *)
let test_transfer_mappings () =
  let maps = ref 0 and unmaps = ref 0 in
  let peer _ r =
    incr maps;
    Ok (r, fun () -> incr unmaps)
  in
  let p = far ~name:"PEER-A" ~peer () and q = far ~name:"PEER-B" () in
  let src = B.create p.dev S.UInt8 64 in
  write src (pattern 7 64);
  dropped (fun () ->
      let dst = B.create q.dev S.UInt8 64 in
      B.copy ~src ~dst;
      B.copy ~src ~dst;
      equal ~msg:"copied" string (pattern 7 64) (read dst);
      equal ~msg:"mapped once" (pair int int) (1, 0) (!maps, !unmaps);
      dst);
  for _ = 1 to 4 do
    Gc.full_major ();
    ignore (stats q.dev)
  done;
  equal ~msg:"unmapped once the memory is released" int 1 !unmaps

(* A copy into a borrow of another device's memory maps the memory under it,
   which the borrow's own region is not. *)
let test_transfer_into_borrow () =
  let asked = ref [] in
  let peer owner r =
    asked := Nx_device.name owner :: !asked;
    Ok (r, ignore)
  in
  let p = far ~name:"PEER-A" ~peer ()
  and q = far ~name:"PEER-B" ~peer:(fun _ r -> Ok (r, ignore)) ()
  and c = far ~name:"PEER-C" () in
  let memory = B.create c.dev S.UInt8 64 in
  let dst = B.view (borrow q.dev memory) ~offset:8 S.UInt8 32 in
  let src = B.create p.dev S.UInt8 32 in
  write src (pattern 9 32);
  B.copy ~src ~dst;
  equal ~msg:"mapped the borrowed memory's owner" (list string) [ "PEER-C" ]
    (List.map masked !asked);
  equal ~msg:"written where the borrow lies" string (pattern 9 32)
    (read (B.view memory ~offset:8 S.UInt8 32))

(* A copy whose source cannot map the destination goes through the host, and
   loses no device. *)
let test_transfer_refused () =
  let p = far ~name:"PEER-A" ~peer:(fun _ _ -> Error "no window") ()
  and q = far ~name:"PEER-B" () in
  let src = B.create p.dev S.UInt8 64 and dst = B.create q.dev S.UInt8 64 in
  write src (pattern 4 64);
  let staged = p.drv.staged + q.drv.staged in
  B.copy ~src ~dst;
  equal ~msg:"copied" string (pattern 4 64) (read dst);
  less ~msg:"through staging" int ~than:(p.drv.staged + q.drv.staged) staged;
  ignore (B.create p.dev S.UInt8 8)

(* A borrow of a borrow maps the memory under the first. *)
let test_borrow_of_borrow () =
  let a = fake ~name:"FIRST" ~maps:true () and c = far () in
  let hb = B.create host S.UInt8 page in
  write hb (pattern 5 page);
  let on_a = borrow a.dev hb in
  let on_c = borrow c.dev on_a in
  equal ~msg:"the host's bytes" string (pattern 5 page) (read on_c);
  is_true ~msg:"one memory" (B.overlaps on_c hb && B.overlaps on_c on_a);
  is_false ~msg:"a buffer of no bytes overlaps nothing"
    (B.overlaps (B.view hb ~offset:0 S.UInt8 0) hb)

let test_bigarray_overlaps () =
  let ba = chars 64 in
  let b = B.of_bigarray ba
  and b' = B.of_bigarray (Bigarray.Array1.sub ba 32 32) in
  equal (pair bool bool) (true, false)
    ( B.overlaps b b',
      B.overlaps (B.of_bigarray (Bigarray.Array1.sub ba 0 32)) b' )

(* Devices over the host's memory, as test devices are: system memory, which one
   maps for the other through its mapping of host memory. *)
let test_system_borrows () =
  let cpu name =
    Driver.device ~name ~arch:"test" ~budget:max_int
      (Host_visible { memory = Driver.host_memory; mapping = Some Identity })
  in
  let c1 = cpu "CPU:1" and c2 = cpu "CPU:2" in
  let b = B.create c2 S.UInt8 page in
  write b (pattern 9 page);
  let on_c1 = borrow c1 b in
  equal ~msg:"read through the borrow" string (pattern 9 page) (read on_c1);
  is_true ~msg:"over the same memory" (B.overlaps on_c1 b);
  let on_host = borrow host b in
  is_true ~msg:"the host's borrow, over the device's memory"
    (Nx_device.equal (B.device on_host) host && B.overlaps on_host b);
  equal ~msg:"read in place" string (pattern 9 page)
    (string_of (B.bigarray Bigarray.char on_host));
  (B.bigarray Bigarray.char on_host).{0} <- 'z';
  equal ~msg:"a write through it" char 'z' (read b).[0];
  (match B.borrow host (B.create far_one.dev S.UInt8 page) with
  | Ok _ -> fail "the host borrowed a Device_local device's memory"
  | Error why -> contains ~msg:"refused" ~sub:"cannot address" why);
  (* The identity maps no page: memory that starts anywhere borrows. *)
  let small = B.create c2 S.UInt8 8 in
  write small "abcdefgh";
  let small_on_c1 = borrow c1 small in
  equal ~msg:"a small buffer, at its host address" (pair string nativeint)
    ("abcdefgh", B.address small)
    (read small_on_c1, B.address small_on_c1);
  let word = borrow c1 (Nx_device.signal_word c2) in
  equal ~msg:"another device's signal word, at its host address" nativeint
    (B.address (Nx_device.signal_word c2))
    (B.address word);
  equal ~msg:"a small host buffer, through the identity" string "wxyz"
    (let hb = B.create host S.UInt8 4 in
     write hb "wxyz";
     read (borrow c1 hb));
  let unaddressed =
    Driver.device ~name:"UNADDRESSED" ~arch:"test" ~budget:max_int
      (Host_visible
         {
           memory = { alloc = (fun n -> Some (Region.v 16n n)); free = ignore };
           mapping = None;
         })
  in
  raises_match ~msg:"a Host_visible region without a host address"
    (Exn.invalid_arg ~substring:"does not address") (fun () ->
      B.create unaddressed S.UInt8 8)

let borrows =
  group "borrows and overlaps"
    [
      test
        "system memory of another device borrows through the mapping of host \
         memory, the identity's at any address"
        test_system_borrows;
      test "another device's memory borrows through its driver's peer mapping"
        test_peer_borrows;
      test
        "another device's mapping of memory outlives its borrows, until the \
         memory is released and the mapper's work is done"
        test_peer_mappings;
      test
        "a copy into another device's memory maps it once, until it is released"
        test_transfer_mappings;
      test "a copy into a borrow of another device's memory maps its owner's"
        test_transfer_into_borrow;
      test
        "a copy whose source cannot map the destination goes through the host"
        test_transfer_refused;
      test "a borrow of a borrow maps the memory under it" test_borrow_of_borrow;
      test "two bigarrays over the same bytes overlap" test_bigarray_overlaps;
    ]

let buffers =
  group "buffers"
    [
      test
        "a consumed buffer's handles are dead, the buffer consume returns is \
         live over the same bytes" (fun () ->
          let b = of_string "abcdefgh" in
          let before = B.view b ~offset:2 S.UInt8 4 in
          let c = consume ~why:"taken" b in
          let dead f =
            raises (Invalid_argument "taken") (fun () -> ignore (f ()))
          in
          dead (fun () -> B.address b);
          dead (fun () -> B.address before);
          dead (fun () -> B.bigarray Bigarray.char b);
          dead (fun () -> B.view b ~offset:0 S.UInt8 1);
          dead (fun () -> B.copy ~src:before ~dst:(B.create host S.UInt8 4));
          dead (fun () -> B.copy ~src:(of_string "abcdefgh") ~dst:b);
          dead (fun () -> consume ~why:"again" b);
          dead (fun () -> B.Claim.read b);
          equal (pair string string) ("abcdefgh", "cdef")
            (read c, read (B.view c ~offset:2 S.UInt8 4));
          equal bool true (B.is_borrowed c));
      test
        "a buffer consumed twice: every dead handle names the last \
         consumption, and the memory stays owned" (fun () ->
          let b = B.create host S.Float32 4 in
          let c = consume ~why:"first" b in
          let d = consume ~why:"second" c in
          let dead why b =
            raises (Invalid_argument why) (fun () -> ignore (B.address b))
          in
          dead "second" b;
          dead "second" c;
          equal (pair int bool) (4, false) (B.length d, B.is_borrowed d));
      test
        "a buffer consumed twice keeps its memory while the last buffer \
         consume returned lives" (fun () ->
          let d = (fake ()).dev in
          let before = allocated d in
          let consumed_twice () =
            consume ~why:"second"
              (consume ~why:"first" (B.create d S.UInt8 4096))
          in
          let last = Sys.opaque_identity (consumed_twice ()) in
          Gc.full_major ();
          equal int (before + 4096) (allocated d);
          ignore (Sys.opaque_identity last));
      test
        "a borrow of all of a memory spans it, through a mapping of whole \
         pages, and a borrow of part of it does not" (fun () ->
          let file = create_file (temp_file ()) 16 in
          let on_host = borrow host file in
          let whole = borrow far_one.dev on_host in
          let part = borrow far_one.dev (B.view on_host ~offset:8 S.UInt8 8) in
          equal (list bool)
            [ true; true; true; false ]
            (List.map B.spans [ file; on_host; whole; part ]));
      test "only a buffer that spans its memory can be consumed" (fun () ->
          let b = B.create host S.UInt8 8 in
          let window = B.view b ~offset:0 S.UInt8 4 in
          equal (pair bool bool) (true, false) (B.spans b, B.spans window);
          raises_match Exn.invalid_arg (fun () ->
              ignore (consume ~why:"window" window));
          equal bool true (B.spans (consume ~why:"whole" b)));
      test
        "of_bigarray takes elements at multiples of their size, of one \
         component for complex kinds, and refuses others" (fun () ->
          let fd = Unix.openfile (temp_file ()) [ Unix.O_RDWR ] 0 in
          Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
          ignore (Unix.write_substring fd (String.make 32 '\000') 0 32);
          let at pos k =
            B.of_bigarray
              (Bigarray.array1_of_genarray
                 (Unix.map_file fd ~pos k Bigarray.c_layout false [| 3 |]))
          in
          equal (list int) [ 3; 3 ]
            [
              B.length (at 8L Bigarray.complex64);
              B.length (at 3L Bigarray.char);
            ];
          raises_match Exn.invalid_arg (fun () -> at 2L Bigarray.float32);
          raises_match Exn.invalid_arg (fun () -> at 4L Bigarray.complex64));
      test
        "a vendor's region is a borrowed buffer of up to max_int bytes that no \
         budget counts" (fun () ->
          let f = fake () and n = (1 lsl 58) + 1 in
          let r = Region.v ~handle:16n 16n (8 * n) in
          let b = Driver.buffer f.dev r S.Float64 n in
          equal
            (quad bool nativeint int int)
            (true, 16n, 8 * n, 0)
            (B.is_borrowed b, B.address b, B.nbytes b, allocated f.dev));
      test
        "a bigarray of a buffer keeps its memory after the buffer is collected"
        (fun () ->
          let v =
            let v = B.bigarray Bigarray.int32 (B.create host S.Int32 1000) in
            Bigarray.Array1.fill v 7l;
            v
          in
          Gc.full_major ();
          let fills =
            List.init 100 (fun _ ->
                let b = B.create host S.Int32 1000 in
                Bigarray.Array1.fill (B.bigarray Bigarray.int32 b) 0x55l;
                b)
          in
          Gc.full_major ();
          ignore (Sys.opaque_identity fills);
          equal (array int32) (Array.make 1000 7l)
            (Array.init 1000 (Bigarray.Array1.get v)));
      test "bigarrays taken on two domains at once share their buffer's memory"
        test_concurrent_views;
      test
        "a borrow keeps its host memory until its device's work on it is \
         signaled"
        test_borrow_lifetime;
      test
        "a borrow made while its released mapping waits for work takes the \
         mapping again"
        test_borrow_again;
      test
        "memory an idle device borrowed returns to an allocation that needs it"
        test_idle_mapper;
      test "the host's staging memory is one, which each device maps once"
        (fun () ->
          let a = far () and b = far () in
          List.iter
            (fun f -> write (B.create f.dev S.UInt8 3) "abc")
            [ a; a; b ];
          equal (list int) [ 1; 1 ]
            [ List.length a.drv.staging; List.length b.drv.staging ];
          equal (list nativeint) a.drv.staging b.drv.staging);
      routes;
      test
        "a copy of several staging slots round trips through a queue that runs \
         behind the host, into a view at an offset and between devices"
        test_staged_slots;
      test "copies between two devices in opposite directions do not deadlock"
        (fun () ->
          let a = near.dev and b = (fake ()).dev in
          let ba = B.create a S.UInt8 64 and bb = B.create b S.UInt8 64 in
          let copies src dst () =
            for _ = 1 to 2000 do
              B.copy ~src ~dst
            done
          in
          let d = Domain.spawn (copies ba bb) in
          copies bb ba ();
          Domain.join d;
          equal int (2000 * 64) (Nx_device.Stats.bytes_out (stats b)));
    ]

(* Claims. A test that takes a claim releases it, so a failure leaves no buffer
   claimed for the next. *)

module Claim = B.Claim

let busy = Exn.invalid_arg ~substring:"in use"
let unbalanced = Exn.invalid_arg ~substring:"unbalanced claim"

let test_claim_counts () =
  let b = B.create host S.UInt8 8 in
  Claim.read b;
  Claim.read b;
  equal ~msg:"two readers" bool false (Claim.try_exclusive b);
  Claim.release b;
  equal ~msg:"one reader, the caller" bool true (Claim.try_exclusive b);
  raises_match ~msg:"a read while exclusive" busy (fun () -> Claim.read b);
  raises_match ~msg:"a read of a view while exclusive" busy (fun () ->
      Claim.read (B.view b ~offset:4 S.UInt8 4));
  Claim.finish b;
  Claim.read b;
  Claim.release b;
  Claim.release b;
  equal ~msg:"free again" bool true
    (Claim.read b;
     let x = Claim.try_exclusive b in
     Claim.finish b;
     Claim.release b;
     x)

let test_unbalanced_release () =
  let b = B.create host S.UInt8 8 in
  raises_match ~msg:"no claim" unbalanced (fun () -> Claim.release b);
  Claim.read b;
  ignore (Claim.try_exclusive b : bool);
  raises_match ~msg:"an exclusive claim" unbalanced (fun () -> Claim.release b);
  Claim.finish b;
  Claim.release b;
  (* Neither refused release wrote: one read is the only claim. *)
  Claim.read b;
  equal bool true (Claim.try_exclusive b);
  Claim.finish b;
  Claim.release b

(* A borrow and the memory it maps share one count (L12). *)
let test_claims_through_borrows () =
  let cpu =
    Driver.device ~name:(unique "CPU:1") ~arch:"test" ~budget:max_int
      (Host_visible { memory = Driver.host_memory; mapping = Some Identity })
  in
  let b = B.create host S.UInt8 page in
  let on_cpu = borrow cpu b in
  Claim.read on_cpu;
  Claim.read b;
  equal ~msg:"a read through the borrow" bool false (Claim.try_exclusive b);
  Claim.release on_cpu;
  equal ~msg:"the borrow's read released" bool true (Claim.try_exclusive b);
  raises_match ~msg:"a read through the borrow while exclusive" busy (fun () ->
      Claim.read on_cpu);
  Claim.finish b;
  Claim.release b

let test_exported () =
  let b = B.create host S.UInt8 8 in
  Claim.export b;
  Claim.read b;
  equal ~msg:"exported" bool false (Claim.try_exclusive b);
  Claim.release b;
  let imported = of_string "abcdefgh" in
  Claim.read imported;
  equal ~msg:"over a bigarray" bool false (Claim.try_exclusive imported);
  Claim.release imported

let test_stale_claims () =
  let b = B.create host S.UInt8 8 in
  let view = B.view b ~offset:2 S.UInt8 4 in
  let c = consume ~why:"taken" b in
  let stale f = raises (Invalid_argument "taken") f in
  stale (fun () -> Claim.read b);
  stale (fun () -> Claim.read view);
  stale (fun () -> ignore (consume ~why:"again" view));
  equal ~msg:"the C check" bool false (c_live b);
  equal ~msg:"the C check, of the new handle" bool true (c_live c);
  (* A release accepts a stale handle: it acts on the memory. *)
  Claim.read c;
  Claim.release b;
  equal ~msg:"released through the stale handle" bool true
    (Claim.read c;
     let x = Claim.try_exclusive c in
     Claim.finish b;
     Claim.release c;
     x)

let test_bracket () =
  let a = B.create host S.UInt8 8 and b = B.create host S.UInt8 8 in
  let outcome =
    Claim.with_ ~read:[ a ] ~donate:[ [ b ] ] (fun c ->
        raises_match ~msg:"a read of the donation" busy (fun () -> Claim.read b);
        Claim.read a;
        Claim.release a;
        (Claim.exclusive c b, Claim.exclusive c a))
  in
  equal ~msg:"the donation, exclusive" (pair bool bool) (true, false) outcome;
  raises_match ~msg:"donating what it reads" Exn.invalid_arg (fun () ->
      Claim.with_
        ~read:[ B.view a ~offset:4 S.UInt8 4 ]
        ~donate:[ [ a ] ]
        (fun _ -> fail "ran"));
  raises_match ~msg:"donating one memory twice" Exn.invalid_arg (fun () ->
      Claim.with_ ~read:[] ~donate:[ [ a ]; [ a ] ] (fun _ -> fail "ran"));
  raises (Failure "f") (fun () ->
      Claim.with_ ~read:[ a ] ~donate:[ [ b ] ] (fun _ -> failwith "f"));
  Claim.read b;
  equal ~msg:"every claim released when f raised" bool true
    (Claim.try_exclusive b);
  Claim.finish b;
  Claim.release b;
  Claim.read a;
  ignore (Claim.try_exclusive a : bool);
  raises_match ~msg:"a buffer exclusive elsewhere" busy (fun () ->
      Claim.with_ ~read:[ a ] ~donate:[] (fun _ -> fail "ran"));
  Claim.finish a;
  Claim.release a

(* A value over several devices is exclusive only if each of its shards is. *)
let test_bracket_shards () =
  let s1 = B.create host S.UInt8 8 and s2 = B.create host S.UInt8 8 in
  Claim.read s2;
  Claim.with_ ~read:[]
    ~donate:[ [ s1; s2 ] ]
    (fun c ->
      equal ~msg:"neither shard exclusive" (pair bool bool) (false, false)
        (Claim.exclusive c s1, Claim.exclusive c s2);
      Claim.read s1;
      Claim.release s1);
  Claim.release s2

(* A window of its memory is never exclusive and never consumed. *)
let test_bracket_window () =
  let b = B.create host S.UInt8 8 in
  let window = B.view b ~offset:0 S.UInt8 4 in
  Claim.with_ ~read:[] ~donate:[ [ window ] ] (fun c ->
      equal ~msg:"not exclusive" bool false (Claim.exclusive c window);
      Claim.read b;
      Claim.release b;
      raises_match ~msg:"not consumed" Exn.invalid_arg (fun () ->
          ignore (Claim.consume c ~why:"window" window)));
  is_true ~msg:"the memory lives" (c_live b && c_live window)

(* With a read claim only, consumption kills the handles and the caller copies:
   the memory is not the caller's to write. *)
let test_consume_read_only () =
  let b = B.create host S.UInt8 8 in
  let outside = B.view b ~offset:0 S.UInt8 8 in
  Claim.read outside;
  let c =
    Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
        equal ~msg:"not exclusive" bool false (Claim.exclusive c b);
        Claim.consume c ~why:"copied" b)
  in
  raises (Invalid_argument "copied") (fun () -> ignore (B.address b));
  Claim.release outside;
  is_true ~msg:"the new handle" (c_live c);
  raises_match ~msg:"a buffer the bracket does not hold" Exn.invalid_arg
    (fun () ->
      Claim.with_ ~read:[] ~donate:[] (fun k ->
          ignore (Claim.consume k ~why:"stray" c)))

(* Views of two memories, each read or donated, and whether the bracket over
   them raises: exactly when a donated view shares a byte with another view. *)
let views =
  let view =
    Gen.triple (Gen.int_range 0 1) (Gen.int_range 0 63) Gen.bool
    |> Gen.map (fun (m, offset, donated) -> (m, offset, 64 - offset, donated))
  in
  let view = Gen.pair view (Gen.int_range 0 64) in
  Gen.list ~size:(Gen.int_range 0 8) view
  |> Gen.map
       (List.map (fun ((m, offset, room, donated), n) ->
            (m, offset, Int.min n room, donated)))
  |> Gen.with_pp (fun ppf vs ->
      List.iter
        (fun (m, o, n, d) ->
          Format.fprintf ppf "%s %d[%d, +%d] "
            (if d then "donate" else "read")
            m o n)
        vs)

let test_bracket_overlaps vs =
  let memories = [| B.create host S.UInt8 64; B.create host S.UInt8 64 |] in
  let views =
    List.map
      (fun (m, o, n, d) -> (B.view memories.(m) ~offset:o S.UInt8 n, d))
      vs
  in
  let read = List.filter_map (fun (b, d) -> if d then None else Some b) views in
  let donated =
    List.filter_map (fun (b, d) -> if d then Some b else None) views
  in
  let overlap =
    List.exists
      (fun (i, (b, d)) ->
        d
        && List.exists
             (fun (j, (b', _)) -> i <> j && B.overlaps b b')
             (List.mapi (fun j v -> (j, v)) views))
      (List.mapi (fun i v -> (i, v)) views)
  in
  let raised =
    match
      Claim.with_ ~read ~donate:(List.map (fun b -> [ b ]) donated) ignore
    with
    | () -> false
    | exception Invalid_argument _ -> true
  in
  equal bool overlap raised

let claims =
  group "claims"
    [
      prop "the bracket refuses exactly the donations that overlap another view"
        views test_bracket_overlaps;
      test
        "reads share, one reader becomes exclusive, and exclusive \
         refuses             reads"
        test_claim_counts;
      test "a release with no read claim raises and changes nothing"
        test_unbalanced_release;
      test
        "a read through a borrow and one through the memory it maps \
         count             together"
        test_claims_through_borrows;
      test "an exported or imported memory is never exclusive" test_exported;
      test
        "a consumed buffer's claims raise its consumption, and a \
         release             accepts it"
        test_stale_claims;
      test "the bracket claims before its function and releases after"
        test_bracket;
      test "a donation over shards is exclusive only if every shard is"
        test_bracket_shards;
      test "a donated window is held for reading and never consumed"
        test_bracket_window;
      test
        "a donation consumed under a read claim dies without being             \
         exclusive"
        test_consume_read_only;
    ]

let refusals =
  let memory = { Driver.alloc = (fun _ -> None); free = ignore } in
  let make ?(budget = 0) memory =
    Driver.device ~name:"X" ~arch:"x" ~budget memory
  in
  let unaddressed = Region.v ~handle:16n 16n 16 in
  let external_ s n = Driver.buffer near.dev unaddressed s n in
  let queue ?(clock = Driver.Host_clock) ~timeline:_ () =
    {
      Driver.copy = (fun ~dst:_ ~src:_ _ ~signal:_ -> ());
      transfer = (fun _ -> None);
      stamp = (fun ~slot:_ ~signal:_ -> ());
      clock;
    }
  in
  let local ?clock host_memory : Driver.memory =
    Device_local
      {
        memory;
        host_memory;
        mapped = None;
        mapping = Pages { map = (fun _ _ -> Error ""); unmap = ignore };
        queue = (fun ~timeline -> queue ?clock ~timeline ());
      }
  in
  let off_page () =
    let ba = B.bigarray Bigarray.char (B.create host S.UInt8 page) in
    B.of_bigarray (Bigarray.Array1.sub ba 1 8)
  in
  let refusing () =
    let mapping =
      Driver.Pages { map = (fun _ _ -> Error "locked"); unmap = ignore }
    in
    make (Host_visible { memory; mapping = Some mapping })
  in
  let rejecting () = (fake ~load:(fun ~binary:_ -> Error "rejected") ()).dev in
  let raise_ ?(exn = Exn.invalid_arg ?substring:None) name f =
    (name, fun () -> raises_match exn (fun () -> ignore (f ())))
  in
  let error ~sub name f =
    ( name,
      fun () ->
        match f () with
        | Ok _ -> fail "accepted"
        | Error why -> contains ~msg:"the reason" ~sub (masked why) )
  in
  let host_visible =
    Driver.Host_visible { memory = Driver.host_memory; mapping = None }
  in
  let named name memory = Driver.device ~name ~arch:"x" ~budget:0 memory in
  let exists = Exn.invalid_arg ~substring:"exists on its machine" in
  cases ~name:fst "refuse"
    [
      ( "a second device of a name on its machine, the host's and the disk's \
         included",
        fun () ->
          let name = unique "ONCE" in
          ignore (named name host_visible);
          raises_match exists (fun () -> named name host_visible);
          raises_match exists (fun () -> named "CPU" host_visible);
          raises_match exists (fun () -> named "DISK" host_visible) );
      ( "nothing of a device that fails to be made, its name included",
        fun () ->
          let name = unique "AGAIN" in
          raises_match (Exn.failure ~substring:"timeline") (fun () ->
              named name (local memory));
          ignore (named name host_visible) );
      raise_ "a device with a negative budget" (fun () ->
          make ~budget:(-1) (Host_visible { memory; mapping = None }));
      raise_ "a device with a clock of 0 Hz" (fun () ->
          make (local ~clock:(Device_clock { hz = 0 }) Driver.host_memory));
      raise_ "host memory that the host does not address" (fun () ->
          make (local { memory with alloc = (fun _ -> Some unaddressed) }));
      raise_ ~exn:(Exn.failure ~substring:"timeline")
        "host memory with none for the timeline" (fun () -> make (local memory));
      raise_ "a buffer of more than max_int bytes" (fun () ->
          B.create host S.Float64 ((max_int / 8) + 1));
      raise_ "a vendor's region on the host" (fun () ->
          Driver.buffer host (Region.v ~host:0n 0n 0) S.UInt8 0);
      raise_ "a vendor's region of -1 elements" (fun () ->
          external_ S.UInt8 (-1));
      raise_ "a vendor's region of more than max_int bytes" (fun () ->
          external_ S.Float64 ((max_int / 8) + 1));
      raise_ "a vendor's region too small for its elements" (fun () ->
          external_ S.Float64 3);
      raise_ "a region of -1 bytes" (fun () -> Region.v 0n (-1));
      raise_ "a bigarray of Int" (fun () ->
          B.of_bigarray
            (Bigarray.Array1.create Bigarray.int Bigarray.c_layout 1));
      raise_ "a bigarray of Nativeint" (fun () ->
          B.of_bigarray
            (Bigarray.Array1.create Bigarray.nativeint Bigarray.c_layout 1));
      raise_ "a bigarray of Int over a buffer" (fun () ->
          B.bigarray Bigarray.int (B.create host S.Int64 1));
      raise_ "a bigarray over a device's buffer" (fun () ->
          B.bigarray Bigarray.char (B.create near.dev S.UInt8 1));
      raise_ "a bigarray over part of an element" (fun () ->
          B.bigarray Bigarray.float32 (B.create host S.UInt8 3));
      raise_ "a bigarray over bytes not aligned to its elements" (fun () ->
          B.bigarray Bigarray.int16_signed
            (B.view (B.create host S.UInt8 4) ~offset:1 S.UInt8 2));
      error ~sub:"start on a page"
        "a borrow of memory that does not start on a page" (fun () ->
          B.borrow far_one.dev (off_page ()));
      error ~sub:"locked" "a borrow its driver refuses, with its reason"
        (fun () -> B.borrow (refusing ()) (B.create host S.UInt8 page));
      error ~sub:"cannot address host memory"
        "a borrow by a device that maps no host memory" (fun () ->
          B.borrow near.dev (B.create host S.UInt8 page));
      raise_
        "a copy of memory the host does not address, by a device without a \
         copy queue" (fun () ->
          B.copy ~src:(external_ S.UInt8 4) ~dst:(B.create host S.UInt8 4));
      raise_ "a copy between buffers of different sizes" (fun () ->
          B.copy ~src:(B.create host S.UInt8 4) ~dst:(B.create host S.UInt8 3));
      error ~sub:"NEAR: the device loads no programs"
        "a program on a device that loads none" (fun () ->
          Nx_device.Program.load (fake ()).dev ~binary:"lib" ~name:"f");
      error ~sub:"NEAR: rejected"
        "a program its driver rejects, with its message" (fun () ->
          Nx_device.Program.load (rejecting ()) ~binary:"lib" ~name:"f");
      raise_ "a buffer of DISK made by create" (fun () ->
          B.create Nx_device.disk S.UInt8 1);
      raise_ "a vendor's region on DISK" (fun () ->
          Driver.buffer Nx_device.disk unaddressed S.UInt8 1);
      raise_ "a new file of -1 bytes" (fun () ->
          B.create_file (temp_file ()) (-1));
      error ~sub:"missing" "a file that does not exist, naming it" (fun () ->
          B.of_file (Filename.concat (temp_dir ()) "missing"));
      error ~sub:"not a regular file" "a directory" (fun () ->
          B.of_file (temp_dir ()));
      raise_ "a copy into a file opened for reading" (fun () ->
          B.copy ~src:(of_string "a") ~dst:(of_file (file_of "b")));
      raise_ "a bigarray over a file's bytes" (fun () ->
          B.bigarray Bigarray.char (of_file (file_of "a")));
      error ~sub:"does not share the host's memory"
        "a borrow of a file's bytes by a device apart from the host" (fun () ->
          B.borrow far_one.dev (of_file (file_of "a")));
      error ~sub:"not aligned"
        "a borrow of a file's bytes not aligned to their elements" (fun () ->
          B.borrow host
            (B.view (of_file (file_of "abcdef")) ~offset:1 S.Int16 2));
    ]
    (fun (_, check) -> check ())

(* A driver that counts its loads and unloads, whose binaries' code lies in
   [code] bytes at a fake address, and which refuses memory for the code of
   [device] while [room ()] is [false]. A function's handle is the binary's load
   count times 100, plus its name's length. *)
type loader = {
  loaded : (string * string) list ref; (* (binary, function), latest first *)
  unloaded : string list ref;
  refused : int ref;
  device : Nx_device.t ref;
}

let loader ?(code = 0) ?(room = fun () -> true) () =
  let l =
    { loaded = ref []; unloaded = ref []; refused = ref 0; device = ref host }
  in
  let images = ref 0 in
  let load ~binary =
    if not (room ()) then begin
      incr l.refused;
      raise (Nx_device.Out_of_memory (!(l.device), code))
    end;
    incr images;
    let image = !images in
    let entry name =
      l.loaded := (binary, name) :: !(l.loaded);
      Ok (Nativeint.of_int ((image * 100) + String.length name))
    in
    Ok
      {
        Driver.code =
          (if code = 0 then None
           else Some (Region.v (Nativeint.of_int (0x10000 * image)) code));
        entry;
        unload = (fun () -> l.unloaded := binary :: !(l.unloaded));
      }
  in
  (l, load)

let test_loaded_once () =
  let l, load = loader () in
  let d = (fake ~load ()).dev in
  let p = program d ~binary:"lib" ~name:"f" in
  let again = program d ~binary:"lib" ~name:"f" in
  let g = program d ~binary:"lib" ~name:"g" in
  equal (list (pair string string)) [ ("lib", "g"); ("lib", "f") ] !(l.loaded);
  equal ~msg:"the same function" nativeint
    (Nx_device.Program.handle p)
    (Nx_device.Program.handle again);
  equal ~msg:"of the same image" (pair nativeint nativeint) (101n, 101n)
    (Nx_device.Program.handle p, Nx_device.Program.handle g);
  equal (pair string bool) ("f", true)
    Nx_device.Program.(name p, Nx_device.equal d (device p));
  ignore (Sys.opaque_identity (p, again, g))

let test_loaded_by_bytes () =
  let l, load = loader () in
  let d = (fake ~load ()).dev in
  let p = program d ~binary:"liba" ~name:"f" in
  let q = program d ~binary:"libb" ~name:"f" in
  let r = program d ~binary:(String.concat "" [ "lib"; "a" ]) ~name:"f" in
  equal ~msg:"another binary of the same length is another image"
    (pair nativeint nativeint) (101n, 201n)
    Nx_device.Program.(handle p, handle q);
  equal ~msg:"the same bytes are the same image" nativeint 101n
    (Nx_device.Program.handle r);
  equal (list (pair string string)) [ ("libb", "f"); ("liba", "f") ] !(l.loaded);
  ignore (Sys.opaque_identity (p, q, r))

(* Finding a function of a loaded binary reads none of the binary: a thousand
   finds in a loaded 16 MiB binary take less than ten reads of it, which
   [Digest.string] stands for. *)
let test_found_unread () =
  let _, load = loader () in
  let d = (fake ~load ()).dev in
  let binary = pattern 1 (16 lsl 20) in
  let p = program d ~binary ~name:"f" in
  let time f =
    let t = Unix.gettimeofday () in
    f ();
    Unix.gettimeofday () -. t
  in
  let read =
    time (fun () -> ignore (Sys.opaque_identity (Digest.string binary)))
  in
  let finds =
    time (fun () ->
        for i = 1 to 1000 do
          ignore
            (Sys.opaque_identity (program d ~binary ~name:(Int.to_string i)))
        done)
  in
  less float_exact ~than:(10. *. read) finds;
  ignore (Sys.opaque_identity p)

let test_unloaded () =
  let opened = ref false in
  let l, load = loader () in
  let d = (fake ~load ~signal:(gate opened) ()).dev in
  ignore (Sys.opaque_identity (program d ~binary:"lib" ~name:"f"));
  ignore (submit d Fun.id);
  collected d;
  equal ~msg:"kept while its device's work runs" (list string) [] !(l.unloaded);
  opened := true;
  collected d;
  equal ~msg:"unloaded once that work is done" (list string) [ "lib" ]
    !(l.unloaded);
  equal ~msg:"loaded anew" nativeint 201n
    (Nx_device.Program.handle (program d ~binary:"lib" ~name:"f"))

let test_code_keeps () =
  let l, load = loader ~code:64 () in
  let d = (fake ~load ()).dev in
  let code =
    Option.get (Nx_device.Program.code (program d ~binary:"lib" ~name:"f"))
  in
  equal ~msg:"the code" (pair nativeint int) (0x10000n, 64)
    (B.address code, B.nbytes code);
  collected d;
  equal ~msg:"kept by its code" (list string) [] !(l.unloaded);
  equal ~msg:"found again" nativeint 101n
    (Nx_device.Program.handle (program d ~binary:"lib" ~name:"f"));
  ignore (Sys.opaque_identity code);
  collected d;
  equal ~msg:"unloaded once the code is unreachable" (list string) [ "lib" ]
    !(l.unloaded);
  equal ~msg:"no code without its driver's" (option nativeint) None
    (Option.map B.address
       (Nx_device.Program.code
          (program (fake ~load:(loads 1n) ()).dev ~binary:"lib" ~name:"f")))

let test_load_collects () =
  let live = ref 0 in
  let l, load = loader ~room:(fun () -> !live = 0) () in
  let load ~binary =
    Result.map
      (fun (i : Driver.image) ->
        incr live;
        {
          i with
          unload =
            (fun () ->
              decr live;
              i.unload ());
        })
      (load ~binary)
  in
  let d = (fake ~load ()).dev in
  l.device := d;
  let first = ref (Some (program d ~binary:"a" ~name:"f")) in
  Gc.full_major ();
  first := None;
  let p = program d ~binary:"b" ~name:"f" in
  equal ~msg:"refused, then loaded"
    (pair bool (list string))
    (true, [ "a" ])
    (!(l.refused) > 0, !(l.unloaded));
  let before = !(l.refused) in
  raises_match
    (function Nx_device.Out_of_memory (d', _) -> d' == d | _ -> false)
    (fun () -> Nx_device.Program.load d ~binary:"c" ~name:"f");
  equal ~msg:"tried again four times" int 5 (!(l.refused) - before);
  ignore (Sys.opaque_identity (first, p))

let programs =
  group "programs"
    [
      test
        "a device runs on the host when the host addresses its memory and it \
         loads no programs" (fun () ->
          is_true ~msg:"host" (Nx_device.runs_on_host host);
          is_true ~msg:"near" (Nx_device.runs_on_host (fake ()).dev);
          is_false ~msg:"near, loading"
            (Nx_device.runs_on_host (fake ~load:(loads 1n) ()).dev);
          is_false ~msg:"far" (Nx_device.runs_on_host (fake ~far:true ()).dev);
          is_false ~msg:"disk" (Nx_device.runs_on_host Nx_device.disk));
      test
        "a binary loads once while a program of it is reachable, and each of \
         its functions once"
        test_loaded_once;
      test
        "an unreachable binary is unloaded once its device's work is done, and \
         loads anew"
        test_unloaded;
      test "a buffer of a binary's code keeps it loaded" test_code_keeps;
      test "a binary is found by its bytes" test_loaded_by_bytes;
      test "a function of a loaded binary is found without reading it"
        test_found_unread;
      test "a binary's code counts in its device's memory while it is loaded"
        (fun () ->
          let _, load = loader ~code:64 () in
          let d = (fake ~load ()).dev in
          let before = allocated d in
          let p = program d ~binary:"lib" ~name:"f" in
          equal ~msg:"loaded" int (before + 64) (allocated d);
          ignore (Sys.opaque_identity p);
          collected d;
          equal ~msg:"unloaded" int before (allocated d));
      test "a host word a program keeps reads and writes through its bigarray"
        (fun () ->
          let p =
            program (fake ~load:(loads 1n) ()).dev ~binary:"lib" ~name:"f"
          in
          let word = Nx_device.Program.keep p (B.create host S.Int64 1) in
          let a = B.bigarray Bigarray.int64 word in
          a.{0} <- 42L;
          equal int64 42L (B.bigarray Bigarray.int64 word).{0});
      test
        "a load refused memory collects unreachable binaries and tries again, \
         then raises Out_of_memory"
        test_load_collects;
      test "a loader that faults loses its device" (fun () ->
          let load ~binary:_ = failwith "context lost" in
          let d = (fake ~name:"LOADER" ~load ()).dev in
          let faulted = lost d "context lost" in
          raises_match faulted (fun () ->
              Nx_device.Program.load d ~binary:"lib" ~name:"f");
          raises_match faulted (fun () -> B.create d S.UInt8 1));
    ]

(* Timeline *)

let word t = Int64.to_int (B.bigarray Bigarray.int64 t).{0}

let timeline =
  group "timeline"
    [
      test
        "starts at 0, gives each submission the next value, and records none \
         that raised" (fun () ->
          let d = (fake ()).dev in
          equal (pair int int) (0, 0)
            (Nx_device.submitted d, Nx_device.signaled d);
          equal int 1 (submit d Fun.id);
          raises (Failure "encode") (fun () ->
              submit d (fun _ -> failwith "encode"));
          let t = Nx_device.signal_word d in
          equal ~msg:"one UInt64 of the host" (triple string string int)
            ("CPU", "uint64", 1)
            (Nx_device.name (B.device t), S.to_string (B.dtype t), B.length t);
          equal ~msg:"not signaled by the submission" int 0 (word t);
          store_signal (B.address t) 1;
          equal ~msg:"signaled" int 1 (Nx_device.signaled d));
      test "is the host memory of a device that allocates some" (fun () ->
          let f = far () in
          is_true
            (Nx_device.equal f.dev (B.device (Nx_device.signal_word f.dev))));
      test
        "of a device with its own signal reports through it, and leaves the \
         signal word at 0" (fun () ->
          let d = (fake ~signal:(signal ~signaled:5 (fun _ -> true)) ()).dev in
          ignore (submit d Fun.id);
          Nx_device.synchronize d;
          equal int 5 (Nx_device.signaled d);
          equal int 0 (word (Nx_device.signal_word d)));
      cases ~name:fst "waits for work"
        [
          ("of its device in synchronize", fun a _ -> ([], a));
          ( "of another device that touched the host, in its synchronize",
            fun _ _ -> ([ B.create host S.UInt8 8 ], host) );
          ( "of another device that touched it, in its synchronize",
            fun _ b -> ([ B.create b S.UInt8 8 ], b) );
        ]
        (fun (_, pick) ->
          let a = (fake ~name:"A" ()).dev and b = (fake ~name:"B" ()).dev in
          let touches, waiter = pick a b in
          let signaled, domain = submit a ~touches (signal_later a) in
          Nx_device.synchronize waiter;
          is_true ~msg:"signaled" (Atomic.get signaled);
          Domain.join domain);
      test "a copy waits for its devices' work first" (fun () ->
          let d = (fake ()).dev in
          let b = B.create d S.UInt8 1 in
          let signaled, domain = submit d (signal_later d) in
          write b "x";
          is_true ~msg:"signaled" (Atomic.get signaled);
          Domain.join domain);
      test "the synchronized hook ends each synchronization, a copy's included"
        (fun () ->
          let calls = ref 0 in
          let d = (fake ~synchronized:(fun () -> incr calls) ()).dev in
          Nx_device.synchronize d;
          write (B.create d S.UInt8 1) "x";
          equal int 2 !calls);
      test
        "a wait asks a device that signals in its own way again while it has \
         not signaled, each time for at most 200 ms" (fun () ->
          let asked = ref [] in
          let wait ms =
            asked := ms :: !asked;
            List.length !asked > 3
          in
          let d = (fake ~signal:(signal wait) ()).dev in
          submit d ignore;
          Nx_device.synchronize d;
          equal ~msg:"asked until it signaled" int 4 (List.length !asked);
          List.iter (fun ms -> at_most int ~than:200 ms) !asked;
          equal ~msg:"not lost" (option string) None (Nx_device.lost d));
      test
        "a wait without a signal lasts until the signal word arrives, however \
         long it stays still" (fun () ->
          let d = (fake ~name:"SLOW" ()).dev in
          let word = B.address (Nx_device.signal_word d) in
          ignore (submit d Fun.id);
          let late =
            Domain.spawn (fun () ->
                Unix.sleepf 0.5;
                store_signal word 1)
          in
          Nx_device.synchronize d;
          Domain.join late;
          equal ~msg:"not lost" (option string) None (Nx_device.lost d));
      test
        "Ctrl-C interrupts a wait for work that never signals, and loses no \
         device" (fun () ->
          let d = (fake ~name:"STUCK" ()).dev in
          ignore (submit d Fun.id);
          let pid = Unix.getpid () in
          Sys.catch_break true;
          Fun.protect
            ~finally:(fun () -> Sys.catch_break false)
            (fun () ->
              let kill =
                Unix.create_process "/bin/sh"
                  [|
                    "/bin/sh";
                    "-c";
                    Printf.sprintf "sleep 0.3; kill -INT %d" pid;
                  |]
                  Unix.stdin Unix.stdout Unix.stderr
              in
              (* The test runner raises [Sys.Break] again past any matcher. *)
              let interrupted =
                match Nx_device.synchronize d with
                | () -> false
                | exception Sys.Break -> true
              in
              ignore (Unix.waitpid [] kill);
              equal ~msg:"interrupted" bool true interrupted);
          equal ~msg:"not lost" (option string) None (Nx_device.lost d);
          store_signal (B.address (Nx_device.signal_word d)) 1;
          Nx_device.synchronize d);
      test "a submission runs once its device's queues have room" (fun () ->
          let asked = ref 0 in
          let room () =
            incr asked;
            !asked > 3
          in
          let d = (fake ~name:"ROOMY" ~room ()).dev in
          equal int 1 (submit d Fun.id);
          equal ~msg:"asked until there was room" int 4 !asked;
          store_signal (B.address (Nx_device.signal_word d)) 1);
      test
        "a submission waits for room however long its device's queues stay \
         full, its driver sleeping meanwhile" (fun () ->
          let t0 = Unix.gettimeofday () and sleeps = ref [] in
          let room () = Unix.gettimeofday () -. t0 > 0.5 in
          let sleep ~still ms = sleeps := (still, ms) :: !sleeps in
          let d = (fake ~name:"FULL" ~room ~sleep ()).dev in
          equal int 1 (submit d Fun.id);
          is_true ~msg:"slept" (!sleeps <> []);
          List.iter (fun (still, _) -> at_least int ~than:200 still) !sleeps;
          store_signal (B.address (Nx_device.signal_word d)) 1);
      test
        "a fault reported while a submission waits for room loses the device, \
         and the submission commits nothing" (fun () ->
          let ran = ref false in
          let sleep ~still:_ _ = failwith "page fault" in
          let d = (fake ~name:"FULL" ~sleep ~room:(fun () -> false) ()).dev in
          raises_match (lost d "page fault") (fun () ->
              submit d (fun _ -> ran := true));
          is_false ~msg:"no work enqueued" !ran;
          equal int 0 (Nx_device.submitted d));
    ]

(* Submissions *)

(* Signals each device's submitted work, as the device would. *)
let settle ds =
  List.iter
    (fun d ->
      store_signal (B.address (Nx_device.signal_word d)) (Nx_device.submitted d))
    ds

let named = List.map (fun (d, v) -> (masked (Nx_device.name d), v))

let waits_of ds ~touches =
  Nx_device.submit ds ~touches Nx_device.Submission.waits

let test_waits () =
  let a = (fake ~name:"A" ~maps:true ()).dev
  and b = (fake ~name:"B" ~maps:true ()).dev
  and c = (fake ~name:"C" ~maps:true ()).dev in
  let on_b = B.create b S.UInt8 8 in
  let waits () = named (waits_of [ a ] ~touches:[ on_b ]) in
  equal ~msg:"nothing pending: values signaled already"
    (list (pair string int))
    [ ("A", 0); ("B", 0) ]
    (waits ());
  settle [ a ];
  ignore (submit b ~touches:[ on_b ] Fun.id);
  equal ~msg:"the latest work on the memory"
    (list (pair string int))
    [ ("A", 1); ("B", 1) ]
    (waits ());
  settle [ a; b ];
  let signaled, domain = submit c ~touches:[ on_b ] (signal_later c) in
  equal ~msg:"the same devices" (list string) [ "A"; "B" ]
    (List.map fst (waits ()));
  is_true ~msg:"work of a device outside them, waited for on the host"
    (Atomic.get signaled);
  Domain.join domain;
  equal ~msg:"never the host" (list string) [ "A" ]
    (List.map fst (named (waits_of [ a ] ~touches:[ B.create host S.UInt8 8 ])));
  settle [ a ]

let test_host_waits () =
  let own_waits = ref 0 in
  let own =
    (fake ~name:"OWN" ~maps:true
       ~signal:
         (signal (fun _ ->
              incr own_waits;
              true))
       ())
      .dev
  in
  let a = (fake ~name:"A" ~maps:true ()).dev
  and plain = (fake ~name:"PLAIN" ()).dev in
  let on_own = B.create own S.UInt8 8 and on_a = B.create a S.UInt8 8 in
  ignore (submit own ~touches:[ on_own ] Fun.id);
  equal ~msg:"its own earlier work is its own to order" int 0 !own_waits;
  equal ~msg:"a device that signals in its own way is left out" (list string)
    [ "A" ]
    (List.map fst (named (waits_of [ a ] ~touches:[ on_own ])));
  equal ~msg:"and its work is waited for on the host" int 1 !own_waits;
  settle [ a ];
  let signaled, domain = submit a ~touches:[ on_a ] (signal_later a) in
  equal
    ~msg:
      "a device whose signal word the submission's devices cannot address is \
       left out"
    (list string) [ "PLAIN" ]
    (List.map fst (named (waits_of [ plain ] ~touches:[ on_a ])));
  is_true ~msg:"and its work is waited for on the host" (Atomic.get signaled);
  Domain.join domain;
  settle [ plain ]

let test_wait () =
  let a = (fake ~name:"A" ()).dev and b = (fake ~name:"B" ()).dev in
  let signaled, domain = submit a (signal_later a) in
  Nx_device.submit [ a ] ~touches:[] (fun s ->
      Nx_device.Submission.wait s a (Nx_device.submitted a);
      is_true ~msg:"the previous work completed" (Atomic.get signaled);
      raises_match ~msg:"a value not submitted" Exn.invalid_arg (fun () ->
          Nx_device.Submission.wait s a (Nx_device.Submission.value s a));
      raises_match ~msg:"a device not taken" Exn.invalid_arg (fun () ->
          Nx_device.Submission.wait s b 0));
  Domain.join domain;
  settle [ a ];
  let stuck = (fake ~name:"STUCK" ~sleep:hangs ()).dev in
  ignore (submit stuck Fun.id);
  raises_match ~msg:"a value that does not arrive" (lost stuck "hang detected")
    (fun () ->
      Nx_device.submit [ stuck ] ~touches:[] (fun s ->
          Nx_device.Submission.wait s stuck 1));
  equal ~msg:"nothing committed" int 1 (Nx_device.submitted stuck)

let test_values () =
  let a = (fake ~name:"A" ()).dev
  and b = (fake ~name:"B" ()).dev
  and c = (fake ~name:"C" ()).dev in
  ignore (submit a Fun.id);
  settle [ a ];
  let hb = B.create host S.UInt8 8 in
  let values =
    Nx_device.submit [ b; a; b ] ~touches:[ hb ] (fun s ->
        raises_match ~msg:"of a device outside it" Exn.invalid_arg (fun () ->
            Nx_device.Submission.value s c);
        List.map (Nx_device.Submission.value s) [ a; b ])
  in
  equal ~msg:"one more than each device's submitted value" (list int) [ 2; 1 ]
    values;
  equal ~msg:"committed" (list int) [ 2; 1 ]
    (List.map Nx_device.submitted [ a; b ]);
  let from_a, on_a = signal_later a 2 and from_b, on_b = signal_later b 1 in
  Nx_device.synchronize host;
  is_true ~msg:"the host waits for the work of both"
    (Atomic.get from_a && Atomic.get from_b);
  List.iter Domain.join [ on_a; on_b ]

let test_submit_refusals () =
  let a = (fake ()).dev in
  let refused ~msg ds touches =
    raises_match ~msg Exn.invalid_arg (fun () ->
        Nx_device.submit ds ~touches ignore)
  in
  refused ~msg:"no device" [] [];
  refused ~msg:"the host" [ host ] [];
  refused ~msg:"the disk" [ Nx_device.disk ] [];
  refused ~msg:"a buffer on the disk" [ a ] [ create_file (temp_file ()) 8 ];
  let b = B.create a S.UInt8 8 in
  ignore (consume ~why:"consumed" b);
  refused ~msg:"a dead buffer" [ a ] [ b ];
  equal ~msg:"nothing submitted" int 0 (Nx_device.submitted a)

let test_copied () =
  let a = (fake ~name:"A" ()).dev and b = (fake ~name:"B" ()).dev in
  let ids = B.create host S.UInt8 32 in
  let host0 = Nx_device.stats host and a0 = Nx_device.stats a in
  Nx_device.submit [ a ] ~touches:[ ids ] (fun s ->
      Nx_device.Submission.copied s ~src:host ~dst:a 32;
      Nx_device.Submission.copied s ~src:a ~dst:a 8;
      raises_match ~msg:"a device not taken" Exn.invalid_arg (fun () ->
          Nx_device.Submission.copied s ~src:b ~dst:a 8);
      raises_match ~msg:"a negative count" Exn.invalid_arg (fun () ->
          Nx_device.Submission.copied s ~src:host ~dst:a (-1)));
  let moved d d0 f = f (Nx_device.Stats.diff d0 (Nx_device.stats d)) in
  equal ~msg:"into the device" int 32 (moved a a0 Nx_device.Stats.bytes_in);
  equal ~msg:"out of the host" int 32
    (moved host host0 Nx_device.Stats.bytes_out);
  settle [ a ]

(* The minor words a submission to one device allocates on the calling domain,
   touching three buffers of the host and three of the device, the one before it
   having left its work pending on their memory. *)
let submission_words () =
  let d = (fake ~maps:true ()).dev in
  let touches =
    List.init 3 (fun _ -> B.create host S.UInt8 8)
    @ List.init 3 (fun _ -> B.create d S.UInt8 8)
  in
  let run () = ignore (submit d ~touches Fun.id) in
  run ();
  settle [ d ];
  let before = Gc.minor_words () in
  run ();
  let words = Gc.minor_words () -. before in
  settle [ d ];
  Float.to_int words

let submissions =
  group "submissions"
    [
      test "a submission touching six buffers allocates at most 300 minor words"
        (fun () -> at_most int ~than:300 (submission_words ()));
      test
        "wait for one pair per device of the submission and of its buffers, \
         whatever is pending, and for the rest on the host"
        test_waits;
      test
        "wait on the host for a device that signals in its own way, or whose \
         signal word they cannot address"
        test_host_waits;
      test
        "wait on the host for a value of a device they took, until it signals"
        test_wait;
      test
        "give each device the value after its submitted one, and commit them \
         together"
        test_values;
      test "refuse no device, a host, the disk, and disk or dead buffers"
        test_submit_refusals;
      test
        "count the bytes their work copies between devices, and none within one"
        test_copied;
    ]

(* Failures *)

let test_hang () =
  let waits = ref 0 in
  let d = (fake ~name:"HUNG" ~signal:(hung waits) ()).dev in
  let b = B.create d S.UInt8 8 in
  ignore (submit d Fun.id);
  (match Nx_device.synchronize d with
  | () -> fail "synchronized a hung device"
  | exception e ->
      equal ~msg:"printed" string "HUNG lost: hang detected"
        (masked (Printexc.to_string e)));
  let hung = lost d "hang detected" in
  List.iter (raises_match hung)
    [
      (fun () -> Nx_device.synchronize d);
      (fun () -> ignore (B.create d S.UInt8 1));
      (fun () -> ignore (B.borrow d b));
      (fun () -> ignore (submit d Fun.id));
      (fun () -> ignore (read b));
      (fun () -> Nx_device.set_budget d 0);
      (fun () -> Nx_device.free_cache d);
    ];
  equal ~msg:"waits" int 1 !waits;
  equal ~msg:"why it is lost" (option string) (Some "hang detected")
    (Nx_device.lost d);
  equal ~msg:"name and arch" (pair string string) ("HUNG", "test")
    (masked (Nx_device.name d), Nx_device.arch d);
  equal ~msg:"budget, submitted, signaled" (triple int int int) (max_int, 1, 0)
    (Nx_device.budget d, Nx_device.submitted d, Nx_device.signaled d);
  equal ~msg:"its signal word" int 1 (B.length (Nx_device.signal_word d));
  Gc.full_major ();
  equal ~msg:"allocated for good" int 8 (allocated d)

let test_scope () =
  let waits = ref 0 in
  let gpu = (fake ~name:"GPU" ~maps:true ~signal:(hung waits) ()).dev in
  let shared = B.view (B.create host S.UInt8 page) ~offset:0 S.UInt8 8 in
  let mapped = borrow gpu shared in
  ignore (submit gpu ~touches:[ mapped ] Fun.id);
  let failed = lost gpu "hang detected" in
  raises_match failed (fun () -> Nx_device.synchronize gpu);
  Nx_device.synchronize host;
  let a = B.create host S.UInt8 8 and other = B.create near.dev S.UInt8 8 in
  write a "12345678";
  B.copy ~src:a ~dst:other;
  dropped (fun () -> B.create host S.UInt8 1000);
  equal ~msg:"memory out of its reach" (pair string string)
    ("12345678", "12345678")
    (read a, read other);
  List.iter (raises_match failed)
    [
      (fun () -> B.copy ~src:shared ~dst:a);
      (fun () -> B.copy ~src:a ~dst:shared);
      (fun () ->
        B.copy
          ~src:(B.view shared ~offset:4 S.UInt8 4)
          ~dst:(B.view a ~offset:0 S.UInt8 4));
      (fun () -> ignore (B.bigarray Bigarray.char shared));
      (fun () -> ignore (read mapped));
      (fun () -> B.Claim.read shared);
      (fun () -> B.Claim.export (B.view shared ~offset:4 S.UInt8 4));
      (fun () -> B.Claim.with_ ~read:[ a ] ~donate:[ [ mapped ] ] ignore);
    ];
  B.Claim.with_ ~read:[ a; other ] ~donate:[] ignore;
  equal ~msg:"claims of memory out of its reach, all released" bool true
    (B.Claim.try_exclusive
       (B.Claim.read a;
        a));
  B.Claim.finish a;
  B.Claim.release a;
  equal ~msg:"a healthy device" (option string) None (Nx_device.lost near.dev);
  equal ~msg:"a view of it" int 4 (B.length (B.view shared ~offset:0 S.UInt8 4));
  equal ~msg:"waits" int 1 !waits

(* A signal that arrives until [hung] is set, and that its driver then declares
   hung. *)
let until hung =
  signal (fun _ -> if !hung then failwith "hang detected" else true)

let test_unmapped () =
  let hung = ref false in
  let gpu = (fake ~name:"LATE" ~maps:true ~signal:(until hung) ()).dev in
  let shared = B.view (B.create host S.UInt8 page) ~offset:0 S.UInt8 4 in
  dropped (fun () -> borrow gpu shared);
  ignore (stats gpu);
  hung := true;
  ignore (submit gpu Fun.id);
  raises_match (lost gpu "hang detected") (fun () -> Nx_device.synchronize gpu);
  write shared "abcd";
  equal string "abcd" (read shared)

let test_cut_short () =
  let hung = ref false in
  let gpu = (fake ~name:"CUT" ~maps:true ~signal:(until hung) ()).dev in
  let shared = B.create host S.UInt8 page in
  dropped (fun () ->
      let b = borrow gpu shared in
      submit gpu ~touches:[ b ] (fun _ -> Sys.opaque_identity b));
  hung := true;
  let failed = lost gpu "hang detected" in
  raises_match failed (fun () -> Nx_device.synchronize gpu);
  raises_match failed (fun () ->
      B.copy ~src:shared ~dst:(B.create host S.UInt8 page))

(* The address of a buffer of [n] bytes of [d] that [f] makes and drops. *)
let dropped_at d n f =
  let a = ref 0n in
  dropped (fun () ->
      let b = B.create d S.UInt8 n in
      a := B.address b;
      f b);
  !a

let test_foreign_stamps () =
  let owner = (fake ~name:"OWNER" ()).dev and opened = ref false in
  let reader = (fake ~name:"READER" ~signal:(gate opened) ()).dev in
  let first =
    dropped_at owner 4096 (fun b ->
        ignore (submit reader ~touches:[ b ] Fun.id))
  in
  let held = B.create owner S.UInt8 4096 in
  equal ~msg:"not reused while the reader's work is unsignaled" bool false
    (B.address held = first);
  opened := true;
  equal ~msg:"reused once it is signaled" nativeint first
    (B.address (B.create owner S.UInt8 4096));
  ignore (Sys.opaque_identity held)

let test_own_stamps () =
  let opened = ref false and waits = ref 0 in
  let signal =
    {
      Driver.signaled = (fun () -> if !opened then max_int else 0);
      wait =
        (fun _ ~ms:_ ->
          incr waits;
          opened := true;
          true);
    }
  in
  let f = fake ~name:"OWNER" ~signal () in
  let touched b = ignore (submit f.dev ~touches:[ b ] Fun.id) in
  let first = dropped_at f.dev 4096 touched in
  let reused = B.create f.dev S.UInt8 4096 in
  equal ~msg:"reused at once by the device whose work touched it" nativeint
    first (B.address reused);
  ignore (dropped_at f.dev 8192 touched);
  ignore (stats f.dev);
  equal ~msg:"cached, nothing waited for" (pair int int) (8192, 0)
    (cached f.dev, !waits);
  Nx_device.free_cache f.dev;
  equal ~msg:"freed once the device's work is done" (pair int int) (1, 1)
    (!waits, f.drv.frees);
  ignore (Sys.opaque_identity reused)

let test_lost_stamps () =
  let owner = fake ~name:"OWNER" ~budget:8192 () and hung = ref false in
  let reader = (fake ~name:"READER" ~signal:(until hung) ()).dev in
  ignore
    (dropped_at owner.dev 4096 (fun b ->
         ignore (submit reader ~touches:[ b ] Fun.id)));
  hung := true;
  raises_match (lost reader "hang detected") (fun () ->
      Nx_device.synchronize reader);
  let b = B.create owner.dev S.UInt8 4096 in
  equal ~msg:"retained, and the owner allocates" (pair int int) (4096, 4096)
    (Nx_device.Stats.retained (stats owner.dev), allocated owner.dev);
  raises_match (out_of_memory owner.dev 1) (fun () ->
      B.create owner.dev S.UInt8 1);
  ignore (Sys.opaque_identity b)

let test_hung_transfer () =
  let a = far ~name:"PEER-HUNG" () and c = far ~name:"PEER-DEST" () in
  let src = B.create a.dev S.UInt8 4 in
  (fun () ->
    let dst = B.create c.dev S.UInt8 4 in
    a.drv.stalled <- true;
    let hung = lost a.dev "hang detected" in
    raises_match hung (fun () -> B.copy ~src ~dst);
    raises_match hung (fun () -> read dst);
    let healthy = B.create c.dev S.UInt8 3 in
    write healthy "abc";
    equal ~msg:"the destination device" string "abc" (read healthy))
    ();
  Gc.full_major ();
  equal ~msg:"retained, and cached" (pair int int) (4, 3)
    (Nx_device.Stats.retained (stats c.dev), cached c.dev)

let test_retained () =
  let f = fake ~name:"D" ~budget:1000 ~signal:(hung (ref 0)) () in
  (* The work does not list the buffer: a free to the driver still waits for it,
     as for all of the device's work submitted before the release. *)
  dropped (fun () ->
      let b = B.create f.dev S.UInt8 600 in
      ignore (submit f.dev Fun.id);
      b);
  raises_match (lost f.dev "hang detected") (fun () ->
      Nx_device.free_cache f.dev);
  equal ~msg:"retained, cached and freed" (triple int int int) (600, 0, 0)
    (Nx_device.Stats.retained (stats f.dev), cached f.dev, f.drv.frees)

(* Memory of [n] bytes on 16 bytes, kept in [keep] until freed. *)
let block keep ~addressed n =
  let ba = chars (n + 15) in
  let a = B.address (B.of_bigarray ba) in
  let a = Nativeint.(logand (add a 15n) (lognot 15n)) in
  Hashtbl.replace keep a ba;
  let host = if addressed then Some a else None in
  Some (Region.v ?host ~handle:a a n)

(* A device of host memory whose driver callback [what] raises [Failure]: a
   fault, which loses the device. *)
let faulty what =
  let fault name = if name = what then failwith (name ^ " fault") in
  let keep = Hashtbl.create 4 in
  let memory =
    {
      Driver.alloc =
        (fun n ->
          fault "alloc";
          block keep ~addressed:true n);
      free =
        (fun r ->
          fault "free";
          Hashtbl.remove keep (Region.address r));
    }
  in
  let mapping =
    Driver.Pages
      {
        map =
          (fun a n ->
            fault "map";
            Ok (Region.v ~host:a a n));
        unmap = (fun _ -> fault "unmap");
      }
  in
  Driver.device ~name:(unique "FAULTY") ~arch:"test" ~budget:max_int
    ~peer:(fun _ r ->
      fault "peer";
      Ok (r, ignore))
    ~dma:(fun _ ->
      fault "dma";
      Error "undescribed")
    ~resolve:(fun _ -> fault "resolve")
    (Host_visible { memory; mapping = Some mapping })

let test_faulting_callbacks () =
  let lost_to what run =
    let d = faulty what in
    let fault = lost d (what ^ " fault") in
    raises_match ~msg:what fault (fun () -> run d);
    raises_match ~msg:(what ^ ", then") fault (fun () ->
        Nx_device.synchronize d);
    d
  in
  ignore (lost_to "alloc" (fun d -> B.create d S.UInt8 8));
  let d =
    lost_to "free" (fun d ->
        dropped (fun () -> B.create d S.UInt8 8);
        Nx_device.free_cache d)
  in
  equal ~msg:"the memory a faulting free did not free is retained" int 8
    (Nx_device.Stats.retained (stats d));
  (* The host memory a lost device was unmapping stays allocated. *)
  let kept = Weak.create 1 in
  ignore
    (lost_to "unmap" (fun d ->
         (fun () ->
           let hb = B.create host S.UInt8 page in
           Weak.set kept 0 (Some hb);
           ignore (Sys.opaque_identity (borrow d hb)))
           ();
         Gc.full_major ();
         B.create d S.UInt8 8));
  Gc.full_major ();
  is_true ~msg:"the borrowed host memory is kept" (Weak.check kept 0);
  ignore (lost_to "map" (fun d -> B.borrow d (B.create host S.UInt8 page)));
  ignore (lost_to "peer" (fun d -> B.borrow d (B.create far_one.dev S.UInt8 8)));
  ignore (lost_to "dma" (fun d -> Driver.dma (B.create d S.UInt8 8)));
  ignore
  @@ lost_to "resolve" (fun d ->
      let p = Nx_device.Profile.start () in
      Fun.protect
        ~finally:(fun () -> ignore (Nx_device.Profile.stop p))
        (fun () ->
          let v =
            Nx_device.submit [ d ] ~touches:[] (fun s ->
                Nx_device.Submission.record s d ~lane:"l" ~name:"n"
                  (B.create host S.UInt64 4);
                Nx_device.Submission.value s d)
          in
          store_signal (B.address (Nx_device.signal_word d)) v;
          Nx_device.synchronize d))

(* A free that faults partway: the memory freed before it is gone, and the
   memory from it on is retained, kept for the life of the process. *)
let test_fault_midway () =
  let keep = Hashtbl.create 4 and frees = ref 0 in
  let memory =
    {
      Driver.alloc = block keep ~addressed:true;
      free =
        (fun r ->
          incr frees;
          if !frees = 2 then failwith "free fault";
          Hashtbl.remove keep (Region.address r));
    }
  in
  let d =
    Driver.device ~name:"MIDWAY" ~arch:"test" ~budget:max_int
      (Host_visible { memory; mapping = None })
  in
  dropped (fun () -> List.init 3 (fun _ -> B.create d S.UInt8 8));
  raises_match (lost d "free fault") (fun () -> Nx_device.free_cache d);
  equal ~msg:"retained, and held by the driver" (pair int int) (16, 2)
    (Nx_device.Stats.retained (stats d), Hashtbl.length keep)

(* A device of a fixed name over host memory, whose driver reports the fault
   [reset] at a synchronization once [faulted] is set: each call is an open of
   the same hardware. *)
let reopen_test () =
  let name = unique "REOPENED" and faulted = ref false in
  let open_ () =
    Driver.device ~name ~arch:"test" ~budget:max_int
      ~synchronized:(fun () -> if !faulted then failwith "reset")
      (Host_visible { memory = Driver.host_memory; mapping = Some Identity })
  in
  let d = open_ () in
  raises_match (Exn.invalid_arg ~substring:"exists on its machine") (fun () ->
      ignore (open_ ()));
  let b = B.create d S.UInt8 8 in
  write b "12345678";
  faulted := true;
  let reset = lost d "reset" in
  raises_match reset (fun () -> Nx_device.synchronize d);
  faulted := false;
  let d' = open_ () in
  equal ~msg:"a new device" bool false (Nx_device.equal d d');
  equal ~msg:"ordered after the lost one" int (-1) (Nx_device.compare d d');
  equal ~msg:"of the same name" string (Nx_device.name d) (Nx_device.name d');
  equal ~msg:"not lost" (option string) None (Nx_device.lost d');
  let b' = B.create d' S.UInt8 8 in
  write b' "abcdefgh";
  equal ~msg:"the fresh device's memory" string "abcdefgh" (read b');
  List.iter (raises_match reset)
    [
      (fun () -> ignore (read b));
      (fun () -> B.Claim.read b);
      (fun () -> ignore (B.create d S.UInt8 1));
      (fun () -> ignore (B.create d S.UInt8 0));
    ];
  raises_match (Exn.invalid_arg ~substring:"exists on its machine") (fun () ->
      ignore (open_ ()))

let failures =
  group "failures"
    [
      test
        "a device whose work does not signal in time is lost for good, and \
         answers what does not take it"
        test_hang;
      test "a fault its driver reports loses a device with the driver's message"
        (fun () ->
          let signal = signal (fun _ -> failwith "page fault") in
          let d = (fake ~name:"FAULTY" ~signal ()).dev in
          ignore (submit d Fun.id);
          let fault = lost d "page fault" in
          raises_match fault (fun () -> Nx_device.synchronize d);
          raises_match fault (fun () -> B.create d S.UInt8 1));
      test "a driver error while enqueueing a copy loses the device" (fun () ->
          let f = far ~name:"REFUSING" () in
          let b = B.create f.dev S.UInt8 4 in
          f.drv.broken <- true;
          let refused = lost f.dev "enqueue refused" in
          raises_match refused (fun () -> write b "abcd");
          raises_match refused (fun () -> B.create f.dev S.UInt8 1));
      test "a loss reaches the memory the lost device can reach, and no other"
        test_scope;
      test
        "a lost device's name makes a fresh device, unequal and after it, \
         while its memory stays lost"
        reopen_test;
      test "a borrow unmapped before a loss is out of its reach" test_unmapped;
      test "a borrow whose unmapping a loss cut short stays in its reach"
        test_cut_short;
      test
        "a transfer that cannot be waited for leaves its destination in the \
         source's reach, retained"
        test_hung_transfer;
      test "memory that a hung wait could not free is retained" test_retained;
      test
        "memory another device's work touched returns to its owner once that \
         work is signaled, and not before"
        test_foreign_stamps;
      test
        "memory its own device's work touched is reused at once, and freed to \
         the driver once that work is done"
        test_own_stamps;
      test
        "memory a lost device's unfinished work touched is retained, and its \
         owner allocates on"
        test_lost_stamps;
      test "a driver callback that raises Failure loses its device"
        test_faulting_callbacks;
      test "a free that faults midway retains the memory from it on"
        test_fault_midway;
    ]

(* Sleep and finalize *)

(* A device that sleeps on its interrupts sleeps only once its signal word has
   stayed still for 200 ms, told how long it has, and a sleep that raises once
   the word stayed still past its driver's limit loses the device. The work
   signals from the third sleep. *)
let test_sleep () =
  let sleeps = ref [] and signal_work = ref ignore in
  let sleep ~still ms =
    sleeps := (still, ms) :: !sleeps;
    if List.length !sleeps = 3 then !signal_work ();
    Unix.sleepf 0.01
  in
  let d = (fake ~name:"SLEEPY" ~sleep ()).dev in
  let v = submit d Fun.id in
  let word = B.address (Nx_device.signal_word d) in
  (signal_work := fun () -> store_signal word v);
  Nx_device.synchronize d;
  equal ~msg:"slept until signaled" int 3 (List.length !sleeps);
  List.iter
    (fun (still, ms) ->
      equal ~msg:"for 200 ms each" int 200 ms;
      at_least ~msg:"still for 200 ms" int ~than:200 still)
    !sleeps;
  let stills = List.rev_map fst !sleeps in
  equal ~msg:"how long, growing" (list int) (List.sort compare stills) stills;
  let limit ~still _ =
    Unix.sleepf 0.01;
    if still >= 400 then failwith "hang detected"
  in
  let still = (fake ~name:"STILL" ~sleep:limit ()).dev in
  ignore (submit still Fun.id);
  let t0 = Unix.gettimeofday () in
  raises_match (lost still "hang detected") (fun () ->
      Nx_device.synchronize still);
  at_least ~msg:"seconds waited" (float 0.01) ~than:0.4
    (Unix.gettimeofday () -. t0)

(* At exit every device finalizes, told whether it failed: a healthy one after
   synchronizing, a failed one without, and one that hangs at exit once its
   synchronization failed. The exit happens in a child process of this test. *)
let finalize_child () =
  let say name ~failed =
    Printf.printf "%s finalized, failed %b\n" name failed
  in
  let healthy = (fake ~name:"HEALTHY" ~finalize:(say "HEALTHY") ()).dev in
  ignore (submit healthy Fun.id);
  store_signal (B.address (Nx_device.signal_word healthy)) 1;
  let broken =
    (fake ~name:"BROKEN" ~sleep:hangs ~finalize:(say "BROKEN") ()).dev
  in
  ignore (submit broken Fun.id);
  (try Nx_device.synchronize broken with Nx_device.Lost _ -> ());
  let hanging =
    (fake ~name:"HANGING" ~sleep:hangs ~finalize:(say "HANGING") ()).dev
  in
  ignore (submit hanging Fun.id);
  ignore (fake ~name:"RAISING" ~finalize:(fun ~failed:_ -> failwith "boom") ());
  exit 0

let test_finalize () =
  let env =
    Array.append [| "NX_DEVICE_FINALIZE_CHILD=1" |] (Unix.environment ())
  in
  let ic, oc, ec =
    Unix.open_process_args_full Sys.executable_name [| Sys.executable_name |]
      env
  in
  close_out oc;
  let out = In_channel.input_all ic and err = In_channel.input_all ec in
  ignore (Unix.close_process_full (ic, oc, ec));
  equal ~msg:"every device finalizes" (list string)
    [
      "BROKEN finalized, failed true";
      "HANGING finalized, failed true";
      "HEALTHY finalized, failed false";
    ]
    (List.sort compare (String.split_on_char '\n' (String.trim out)));
  contains ~msg:"a failed exit synchronization is reported"
    ~sub:"HANGING synchronization failed" (masked err);
  contains ~msg:"a raising finalize is reported" ~sub:"boom" err

(* [with_host_word f] is [f word a] for [word] a region of the host's heap whose
   first word holds 0, and [a] its host address. *)
let with_host_word f =
  match Driver.host_memory.alloc 8 with
  | None -> fail "no host memory"
  | Some r ->
      let a = Option.get (Region.host_address r) in
      store_signal a 0;
      Fun.protect
        ~finally:(fun () -> Driver.host_memory.free r)
        (fun () -> f r a)

let sleep_for ~timeline:_ ~still:_ ms = Unix.sleepf (Float.of_int ms /. 1000.)

(* [ready_after s] is a condition that holds [s] seconds from now. *)
let ready_after s =
  let t = Unix.gettimeofday () +. s in
  fun () -> Unix.gettimeofday () >= t

(* A driver's sleep that declares a hang once the timeline stayed still for [ms]
   milliseconds. *)
let hangs_after ms ~timeline:_ ~still _ =
  if still >= ms then failwith "hang detected";
  Unix.sleepf 0.01

let driver_wait =
  group "Driver.wait"
    [
      test "returns once the condition holds" (fun () ->
          with_host_word (fun timeline _ ->
              let ready = ready_after 0.05 in
              Driver.wait ~sleep:sleep_for ~timeline ready;
              is_true ~msg:"held" (ready ())));
      test
        "sleeps once the timeline stayed still for 200 ms, told how long, at \
         most 200 ms each" (fun () ->
          with_host_word (fun timeline _ ->
              let sleeps = ref [] in
              let sleep ~timeline:t ~still ms =
                sleeps := (t == timeline, still, ms) :: !sleeps;
                sleep_for ~timeline ~still ms
              in
              Driver.wait ~sleep ~timeline (ready_after 0.6);
              is_true ~msg:"slept" (!sleeps <> []);
              List.iter
                (fun (on_timeline, still, ms) ->
                  equal ~msg:"given the timeline" bool true on_timeline;
                  at_least ~msg:"still" int ~than:200 still;
                  at_most int ~than:200 ms)
                !sleeps));
      test
        "counts how long the timeline stayed still from its last move, however \
         long the condition takes" (fun () ->
          with_host_word (fun timeline a ->
              (* The timeline moves at every check, for a second. *)
              let moves = ref 0 and held = ready_after 1.0 in
              let moving () =
                incr moves;
                store_signal a !moves;
                held ()
              in
              let sleeps = ref 0 in
              let sleep ~timeline:_ ~still:_ _ = incr sleeps in
              Driver.wait ~sleep ~timeline moving;
              equal ~msg:"no sleep while the timeline moves" int 0 !sleeps;
              raises_match (Exn.failure ~substring:"hang detected") (fun () ->
                  Driver.wait ~sleep:(hangs_after 400) ~timeline (fun () ->
                      false))));
      test "refuses a timeline the host does not address" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Driver.wait ~sleep:sleep_for ~timeline:(Region.v 0x1000n 8)
                (fun () -> true)));
      test "refuses a timeline of fewer than 8 bytes" (fun () ->
          with_host_word (fun _ a ->
              raises_match Exn.invalid_arg (fun () ->
                  Driver.wait ~sleep:sleep_for ~timeline:(Region.v ~host:a a 4)
                    (fun () -> true))));
      test "refuses a host that is no host" (fun () ->
          with_host_word (fun timeline _ ->
              let d = (fake ~name:"NOHOST" ()).dev in
              raises_match Exn.invalid_arg (fun () ->
                  Driver.wait ~host:d ~sleep:sleep_for ~timeline (fun () ->
                      true))));
    ]

let hooks =
  group "sleep and finalize"
    [
      test
        "a device sleeps on its interrupts once its signal word stays still, \
         and its driver alone declares a hang"
        test_sleep;
      test "a fault found asleep loses the device with the driver's message"
        (fun () ->
          let sleep ~still:_ _ = failwith "page fault at 0x1000" in
          let d = (fake ~name:"FAULTED" ~sleep ()).dev in
          ignore (submit d Fun.id);
          let fault = lost d "page fault at 0x1000" in
          raises_match fault (fun () -> Nx_device.synchronize d);
          raises_match fault (fun () -> B.create d S.UInt8 1));
      test "finalize runs at exit on every device, told whether it failed"
        test_finalize;
    ]

let devices =
  group "devices"
    [
      test
        "the host is CPU, of the machine's instruction set, with a budget of \
         max_int" (fun () ->
          equal string "CPU" (Nx_device.name host);
          mem string (Nx_device.arch host) [ "arm64"; "x86_64" ];
          equal int max_int (Nx_device.budget host));
      test "each make is a device of its own, equal only to itself" (fun () ->
          let a = (fake ()).dev and b = (fake ()).dev in
          equal (list bool) [ true; false; false ]
            [ Nx_device.equal a a; Nx_device.equal a b; Nx_device.equal a host ]);
      programs;
    ]

(* Profiles *)

module P = Nx_device.Profile

(* The events of a profile taken around [f]. *)
let profiled f =
  let p = P.start () in
  match f () with
  | () -> P.stop p
  | exception e ->
      ignore (P.stop p);
      raise e

type span = { on : string; lane : string; what : string; t0 : int; t1 : int }

let spans =
  List.filter_map (function
    | P.Span s ->
        Some
          {
            on = masked (Nx_device.name s.device);
            lane = s.lane;
            what = masked s.name;
            t0 = s.start;
            t1 = s.stop;
          }
    | P.Allocation _ | P.Load _ | P.Counters _ | P.Trace _ | P.Overwritten _ ->
        None)

let span_ =
  Testable.make
    ~pp:(fun ppf s ->
      Format.fprintf ppf "%s on %s of %s, %d to %d" s.what s.lane s.on s.t0 s.t1)
    ~equal:( = )

let where = triple string string string
let placed s = (s.on, s.lane, s.what)

(* [inner] runs within [outer], [slack] nanoseconds either side allowed. *)
let within ?(slack = 0) ~outer inner =
  is_true
    ~msg:
      (Printf.sprintf "%s [%d, %d] within %s [%d, %d]" inner.what inner.t0
         inner.t1 outer.what outer.t0 outer.t1)
    (inner.t0 <= inner.t1
    && outer.t0 - slack <= inner.t0
    && inner.t1 <= outer.t1 + slack)

let main_lane = Printf.sprintf "domain %d" (Domain.self () :> int)

(* Work on [d], recorded as the spans [names] of [stamps], that a domain does
   after 50 ms: it writes [t0] and [t1] into the stamp words of [stamps]'s two
   slots, then signals once [d]'s earlier work has, as work completes in
   order. *)
let stamped d stamps names (t0, t1) =
  let timeline = Nx_device.signal_word d in
  let word = B.address timeline
  and signal = B.bigarray Bigarray.int64 timeline in
  let ba = B.bigarray Bigarray.int64 stamps in
  Nx_device.submit [ d ] ~touches:[] (fun s ->
      List.iter
        (fun (lane, name) -> Nx_device.Submission.record s d ~lane ~name stamps)
        names;
      let v = Nx_device.Submission.value s d in
      Domain.spawn (fun () ->
          Unix.sleepf 0.05;
          ba.{1} <- Int64.of_int t0;
          ba.{3} <- Int64.of_int t1;
          while Int64.to_int signal.{0} < v - 1 do
            Domain.cpu_relax ()
          done;
          store_signal word v))

let test_sessions () =
  is_false (P.enabled ());
  let p = P.start () in
  is_true (P.enabled ());
  raises_match Exn.invalid_arg P.start;
  equal (list span_) [] (spans (P.stop p));
  is_false (P.enabled ());
  raises_match ~msg:"a profile stopped already" Exn.invalid_arg (fun () ->
      P.stop p);
  let p' = P.start () in
  raises_match ~msg:"another profile" Exn.invalid_arg (fun () -> P.stop p);
  is_true ~msg:"still taken" (P.enabled ());
  equal (list span_) [] (spans (P.stop p'))

let test_host_spans () =
  let other = ref "" in
  let events =
    profiled (fun () ->
        P.span "outer" (fun () -> P.span "inner" ignore);
        raises (Failure "raised") (fun () ->
            P.span "raising" (fun () -> failwith "raised"));
        Domain.join
          (Domain.spawn (fun () ->
               other := Printf.sprintf "domain %d" (Domain.self () :> int);
               P.span "elsewhere" ignore)))
  in
  match spans events with
  | [ outer; inner; raising; elsewhere ] ->
      equal (list where)
        [
          ("CPU", main_lane, "outer");
          ("CPU", main_lane, "inner");
          ("CPU", main_lane, "raising");
          ("CPU", !other, "elsewhere");
        ]
        (List.map placed [ outer; inner; raising; elsewhere ]);
      not_equal string main_lane !other;
      within ~outer inner;
      is_true ~msg:"in order" (outer.t1 <= raising.t0)
  | l -> fail (Printf.sprintf "%d spans" (List.length l))

let test_off () =
  let d = (fake ()).dev in
  let stamps = B.create host S.UInt64 4 in
  let f () = () in
  let words loop =
    let before = Gc.minor_words () in
    for _ = 1 to 1000 do
      loop ()
    done;
    Gc.minor_words () -. before
  in
  let idle = words ignore in
  let v =
    Nx_device.submit [ d ] ~touches:[] (fun s ->
        equal ~msg:"words allocated" float_exact idle
          (words (fun () ->
               P.span "x" f;
               Nx_device.Submission.record s d ~lane:"compute" ~name:"x" stamps;
               ignore (Sys.opaque_identity (P.enabled ()))));
        Nx_device.Submission.value s d)
  in
  store_signal (B.address (Nx_device.signal_word d)) v;
  let events = profiled (fun () -> Nx_device.synchronize d) in
  equal ~msg:"spans recorded before" (list span_) [] (spans events)

let test_submitted () =
  let d = (fake ~name:"GPU" ()).dev in
  let stamps = B.create host S.UInt64 4 and again = B.create host S.UInt64 4 in
  let events =
    profiled (fun () ->
        let first =
          stamped d stamps
            [ ("compute", "replaced"); ("compute", "kernel") ]
            (100, 250)
        in
        let second = stamped d again [ ("copy", "next") ] (300, 420) in
        Nx_device.synchronize d;
        List.iter Domain.join [ first; second ])
  in
  equal (list span_)
    [
      { on = "GPU"; lane = "compute"; what = "kernel"; t0 = 100; t1 = 250 };
      { on = "GPU"; lane = "copy"; what = "next"; t0 = 300; t1 = 420 };
    ]
    (spans events)

let test_resolve () =
  let log = ref [] in
  let resolve a =
    log := "resolve" :: !log;
    store_signal (Nativeint.add a 8n) 7;
    store_signal (Nativeint.add a 24n) 9
  in
  let d =
    (fake ~resolve ~synchronized:(fun () -> log := "synchronized" :: !log) ())
      .dev
  in
  let stamps = B.create host S.UInt64 4 in
  let events =
    profiled (fun () ->
        Nx_device.submit [ d ] ~touches:[] (fun s ->
            Nx_device.Submission.record s d ~lane:"compute" ~name:"k" stamps);
        store_signal (B.address (Nx_device.signal_word d)) 1;
        Nx_device.synchronize d)
  in
  equal
    (list (pair int int))
    [ (7, 9) ]
    (List.map (fun s -> (s.t0, s.t1)) (spans events));
  equal (list string) [ "resolve"; "synchronized" ] (List.rev !log)

(* The spans named [name] of [events]: the host's, and those of copy queues. *)
let copies events name =
  List.partition
    (fun s -> s.on = "CPU")
    (List.filter (fun s -> s.what = name) (spans events))

let test_copies ?clock ?(slack = 0) () =
  let f = far ?clock () in
  let small = B.create host S.UInt8 100 and big = B.create host S.UInt8 page in
  let on_far = B.create f.dev S.UInt8 100
  and big_far = B.create f.dev S.UInt8 page in
  let borrowed = borrow f.dev big in
  let events =
    profiled (fun () ->
        B.copy ~src:small ~dst:on_far;
        B.copy ~src:big ~dst:big_far;
        B.copy ~src:on_far ~dst:small;
        B.copy ~src:small ~dst:(B.create host S.UInt8 100))
  in
  ignore (Sys.opaque_identity borrowed);
  let host_in, far_in = copies events "CPU -> FAR" in
  let host_out, far_out = copies events "FAR -> CPU" in
  let host_host, _ = copies events "CPU -> CPU" in
  equal ~msg:"host spans" (list where)
    [
      ("CPU", main_lane, "CPU -> FAR");
      ("CPU", main_lane, "CPU -> FAR");
      ("CPU", main_lane, "FAR -> CPU");
      ("CPU", main_lane, "CPU -> CPU");
    ]
    (List.map placed (host_in @ host_out @ host_host));
  equal ~msg:"copy queue spans" (list where)
    [
      ("FAR", "copy", "CPU -> FAR");
      ("FAR", "copy", "CPU -> FAR");
      ("FAR", "copy", "FAR -> CPU");
    ]
    (List.map placed (far_in @ far_out));
  List.iter2
    (fun outer s -> within ~slack ~outer s)
    (host_in @ host_out) (far_in @ far_out)

let test_bounce () =
  let a = far ~name:"A" () and b = far ~name:"B" () in
  let src = B.create a.dev S.UInt8 64 and dst = B.create b.dev S.UInt8 64 in
  let events = profiled (fun () -> B.copy ~src ~dst) in
  match spans events with
  | [ host_span; on_a; on_b ] ->
      equal (list where)
        [
          ("CPU", main_lane, "A -> B");
          ("A", "copy", "A -> B");
          ("B", "copy", "A -> B");
        ]
        (List.map placed [ host_span; on_a; on_b ]);
      within ~outer:host_span on_a;
      within ~outer:host_span on_b
  | l -> fail (Printf.sprintf "%d spans" (List.length l))

let test_memory_events () =
  let d = (fake ~name:"MEM" ()).dev in
  let samples events =
    List.filter_map
      (function
        | P.Allocation m when Nx_device.equal m.device d -> Some m.allocated
        | _ -> None)
      events
  in
  let events =
    profiled (fun () ->
        dropped (fun () -> B.create d S.UInt8 100);
        ignore (stats d))
  in
  equal (list int) [ 100; 0 ] (samples events)

let test_program_events () =
  let d = (fake ~load:(loads 42n) ()).dev in
  let events =
    profiled (fun () ->
        ignore (program d ~binary:"lib" ~name:"k");
        ignore (program d ~binary:"lib" ~name:"k"))
  in
  match events with
  | [ P.Load p ] ->
      equal
        (triple string string nativeint)
        ("k", "lib", 42n)
        ( Nx_device.Program.name p.program,
          p.binary,
          Nx_device.Program.handle p.program );
      is_true (Nx_device.equal d (Nx_device.Program.device p.program))
  | l -> fail (Printf.sprintf "%d events" (List.length l))

let test_failed_spans () =
  let d = (fake ~name:"HUNG" ~signal:(hung (ref 0)) ()).dev in
  let stamps = B.create host S.UInt64 4 in
  let events =
    profiled (fun () ->
        Nx_device.submit [ d ] ~touches:[] (fun s ->
            Nx_device.Submission.record s d ~lane:"compute" ~name:"lost" stamps);
        P.span "kept" ignore)
  in
  equal (list where)
    [ ("CPU", main_lane, "kept") ]
    (List.map placed (spans events));
  raises_match (lost d "hang detected") (fun () -> Nx_device.synchronize d)

(* JSON *)

type json =
  | Null
  | Bool of bool
  | Num of float
  | Str of string
  | Arr of json list
  | Obj of (string * json) list

(* Strict JSON, as RFC 8259 has it. *)
let parse s =
  let i = ref 0 in
  let peek () = if !i < String.length s then s.[!i] else '\000' in
  let error what = failwith (Printf.sprintf "JSON: %s at byte %d" what !i) in
  let rec ws () =
    match peek () with
    | ' ' | '\n' | '\t' | '\r' ->
        incr i;
        ws ()
    | _ -> ()
  in
  let expect c =
    if peek () = c then incr i else error (Printf.sprintf "no %c" c)
  in
  let literal word v =
    if
      String.length s - !i >= String.length word
      && String.sub s !i (String.length word) = word
    then (
      i := !i + String.length word;
      v)
    else error "a bad literal"
  in
  let str () =
    expect '"';
    let b = Buffer.create 16 in
    let rec go () =
      match peek () with
      | '"' -> incr i
      | '\\' ->
          incr i;
          let c = peek () in
          incr i;
          (match c with
          | '"' | '\\' | '/' -> Buffer.add_char b c
          | 'b' -> Buffer.add_char b '\b'
          | 'f' -> Buffer.add_char b '\012'
          | 'n' -> Buffer.add_char b '\n'
          | 'r' -> Buffer.add_char b '\r'
          | 't' -> Buffer.add_char b '\t'
          | 'u' ->
              let code = int_of_string ("0x" ^ String.sub s !i 4) in
              i := !i + 4;
              Buffer.add_utf_8_uchar b (Uchar.of_int code)
          | _ -> error "a bad escape");
          go ()
      | c when Char.code c < 0x20 -> error "a control character"
      | c ->
          Buffer.add_char b c;
          incr i;
          go ()
    in
    go ();
    Buffer.contents b
  in
  let num () =
    let j = !i in
    while
      match peek () with
      | '0' .. '9' | '-' | '+' | '.' | 'e' | 'E' -> true
      | _ -> false
    do
      incr i
    done;
    match float_of_string_opt (String.sub s j (!i - j)) with
    | Some f when !i > j -> Num f
    | _ -> error "a bad number"
  in
  let rec value () =
    ws ();
    match peek () with
    | '{' ->
        incr i;
        ws ();
        if peek () = '}' then (
          incr i;
          Obj [])
        else members []
    | '[' ->
        incr i;
        ws ();
        if peek () = ']' then (
          incr i;
          Arr [])
        else elements []
    | '"' -> Str (str ())
    | 't' -> literal "true" (Bool true)
    | 'f' -> literal "false" (Bool false)
    | 'n' -> literal "null" Null
    | _ -> num ()
  and members acc =
    ws ();
    let k = str () in
    ws ();
    expect ':';
    let v = value () in
    ws ();
    match peek () with
    | ',' ->
        incr i;
        members ((k, v) :: acc)
    | '}' ->
        incr i;
        Obj (List.rev ((k, v) :: acc))
    | _ -> error "no , or }"
  and elements acc =
    let v = value () in
    ws ();
    match peek () with
    | ',' ->
        incr i;
        elements (v :: acc)
    | ']' ->
        incr i;
        Arr (List.rev (v :: acc))
    | _ -> error "no , or ]"
  in
  let v = value () in
  ws ();
  if !i <> String.length s then error "trailing bytes";
  v

let field k = function
  | Obj kvs -> (
      match List.assoc_opt k kvs with
      | Some v -> v
      | None -> failwith ("no field " ^ k))
  | _ -> failwith ("no object for " ^ k)

let num k e =
  match field k e with Num f -> f | _ -> failwith (k ^ " is no number")

let str k e =
  match field k e with Str s -> s | _ -> failwith (k ^ " is no string")

let written events =
  let path = Filename.temp_file "profile" ".json" in
  Fun.protect
    ~finally:(fun () -> Sys.remove path)
    (fun () ->
      Out_channel.with_open_bin path (fun oc -> P.output_chrome_trace oc events);
      parse (In_channel.with_open_bin path In_channel.input_all))

let test_output () =
  let odd = "a \"quote\", a \\, a\nnewline, \001, \xff and \xc3\xa9" in
  let f = far () in
  let d = (fake ~name:"P" ~load:(loads 0x1234n) ()).dev in
  let events =
    profiled (fun () ->
        P.span "outer" (fun () ->
            P.span odd ignore;
            B.copy ~src:(B.create host S.UInt8 8)
              ~dst:(B.create f.dev S.UInt8 8));
        ignore (program d ~binary:"b" ~name:"k");
        dropped (fun () -> B.create d S.UInt8 5);
        ignore (stats d))
  in
  let trace =
    match field "traceEvents" (written events) with
    | Arr l -> l
    | _ -> fail "no array"
  in
  let ph p = List.filter (fun e -> str "ph" e = p) trace in
  let meta what =
    List.filter_map
      (fun e ->
        if str "name" e = what then
          Some
            ( (int_of_float (num "pid" e), int_of_float (num "tid" e)),
              str "name" (field "args" e) )
        else None)
      (ph "M")
  in
  let processes = meta "process_name" and threads = meta "thread_name" in
  let named (pid, tid) =
    (masked (List.assoc (pid, 0) processes), List.assoc (pid, tid) threads)
  in
  let complete =
    List.map
      (fun e ->
        let key = (int_of_float (num "pid" e), int_of_float (num "tid" e)) in
        let lo = num "ts" e in
        (named key, masked (str "name" e), lo, lo +. num "dur" e))
      (ph "X")
  in
  equal ~msg:"spans"
    (list (pair (pair string string) string))
    [
      (("CPU", main_lane), "outer");
      ( ("CPU", main_lane),
        "a \"quote\", a \\, a\nnewline, \001, \u{FFFD} and \xc3\xa9" );
      (("CPU", main_lane), "CPU -> FAR");
      (("FAR", "copy"), "CPU -> FAR");
    ]
    (List.map (fun (k, name, _, _) -> (k, name)) complete);
  let times =
    List.map
      (fun e -> num "ts" e)
      (List.filter (fun e -> str "ph" e <> "M") trace)
  in
  equal ~msg:"in time order" (list float_exact)
    (List.sort Float.compare times)
    times;
  equal ~msg:"from the earliest" float_exact 0. (List.hd times);
  List.iter
    (fun (k, name, lo, hi) ->
      List.iter
        (fun (k', name', lo', hi') ->
          if k = k' && name <> name' then
            is_true
              ~msg:(name ^ " and " ^ name' ^ " nest or are apart")
              (hi <= lo' || hi' <= lo
              || (lo <= lo' && hi' <= hi)
              || (lo' <= lo && hi <= hi')))
        complete)
    complete;
  (match ph "i" with
  | [ e ] ->
      equal
        (triple string string string)
        ("k", "p", "0x1234")
        (str "name" e, str "s" e, str "handle" (field "args" e))
  | l -> fail (Printf.sprintf "%d instants" (List.length l)));
  let counters =
    List.filter_map
      (fun e ->
        let pid = int_of_float (num "pid" e) in
        if masked (List.assoc (pid, 0) processes) = "P" then
          Some (str "name" e, num "allocated" (field "args" e))
        else None)
      (ph "C")
  in
  equal ~msg:"memory of P"
    (list (pair string float_exact))
    [ ("memory", 5.); ("memory", 0.) ]
    counters;
  equal ~msg:"nothing" (list string) []
    (match field "traceEvents" (written []) with
    | Arr l -> List.map (str "ph") l
    | _ -> [ "?" ])

(* A device whose runs of [k] count [i] and [i + 1] in two units of the [i]th
   counter the profile asked for when they ran, and trace if it asked for
   traces, reported at each synchronization with the times each run was given,
   after the runs it lost, and logged in [log]. *)
let counting log =
  let runs = ref [] and lost = ref 0 and d = ref None in
  let report () =
    log := "report" :: !log;
    let device = Option.get !d in
    let overwritten =
      if !lost = 0 then []
      else [ P.Overwritten { device; time = P.now (); runs = !lost } ]
    in
    let counted =
      List.concat
        (List.rev_map
           (fun ((start, stop), (counters, traced)) ->
             (if counters = [] then []
              else
                [
                  P.Counters
                    {
                      device;
                      name = "k";
                      start;
                      stop;
                      counters =
                        List.mapi (fun i c -> (c, [| i; i + 1 |])) counters;
                    };
                ])
             @
             if traced then
               [
                 P.Trace
                   { device; name = "k"; start; stop; part = 0; data = "trace" };
               ]
             else [])
           !runs)
    in
    runs := [];
    lost := 0;
    overwritten @ counted
  in
  let dev = (fake ~name:"COUNTING" ~report ()).dev in
  d := Some dev;
  let run times =
    runs :=
      List.rev_map (fun t -> (t, (P.counters (), P.traced ()))) times @ !runs
  in
  (dev, run, fun n -> lost := !lost + n)

let counted =
  List.filter_map (function
    | P.Counters c ->
        Some
          ( masked (Nx_device.name c.device),
            c.name,
            (c.start, c.stop),
            c.counters )
    | P.Span _ | P.Allocation _ | P.Load _ | P.Trace _ | P.Overwritten _ -> None)

let counts =
  list
    (Testable.make
       ~pp:(fun ppf (on, name, (t0, t1), _) ->
         Format.fprintf ppf "%s %s %d-%d" on name t0 t1)
       ~equal:( = ))

let test_counters () =
  let log = ref [] in
  let d, run, _ = counting log in
  raises_match ~msg:"a counter asked twice" Exn.invalid_arg (fun () ->
      P.start ~counters:[ "A"; "B"; "A" ] ());
  is_false ~msg:"no profile taken" (P.enabled ());
  equal ~msg:"no profile" (list string) [] (P.counters ());
  let p = P.start ~counters:[ "A"; "B" ] () in
  equal ~msg:"asked" (list string) [ "A"; "B" ] (P.counters ());
  run [ (10, 20); (30, 40) ];
  Nx_device.synchronize d;
  run [ (50, 60) ];
  let events = P.stop p in
  equal ~msg:"no profile after" (list string) [] (P.counters ());
  let ab = [ ("A", [| 0; 1 |]); ("B", [| 1; 2 |]) ] in
  equal
    ~msg:
      "each run, timed by itself, read at a synchronization and when the \
       profile stops"
    counts
    [
      ("COUNTING", "k", (10, 20), ab);
      ("COUNTING", "k", (30, 40), ab);
      ("COUNTING", "k", (50, 60), ab);
    ]
    (counted events);
  equal ~msg:"reported twice" (list string) [ "report"; "report" ] !log;
  log := [];
  let events = profiled (fun () -> Nx_device.synchronize d) in
  equal ~msg:"none without counters asked" counts [] (counted events);
  equal ~msg:"not asked" (list string) [] !log

let test_traces () =
  let log = ref [] in
  let d, run, _ = counting log in
  is_false ~msg:"no profile" (P.traced ());
  let p = P.start ~trace:true () in
  is_true ~msg:"asked" (P.traced ());
  run [ (1_000, 2_000) ];
  Nx_device.synchronize d;
  run [ (3_000, 3_500) ];
  let events = P.stop p in
  let traces =
    List.filter_map
      (function
        | P.Trace t -> Some (t.name, t.start, t.stop, t.data) | _ -> None)
      events
  in
  equal ~msg:"each run's trace, read at a synchronization and when it stops"
    (list
       (Testable.make
          ~pp:(fun ppf (n, a, b, _) -> Format.fprintf ppf "%s %d-%d" n a b)
          ~equal:( = )))
    [ ("k", 1_000, 2_000, "trace"); ("k", 3_000, 3_500, "trace") ]
    traces;
  equal ~msg:"no counters asked" counts [] (counted events);
  equal ~msg:"reported twice" (list string) [ "report"; "report" ] !log;
  let trace =
    match field "traceEvents" (written events) with
    | Arr l -> l
    | _ -> fail "no array"
  in
  equal ~msg:"an instant of each trace, with its part and bytes"
    (list (pair float_exact float_exact))
    [ (0., 5.); (0., 5.) ]
    (List.map
       (fun e ->
         let args = field "args" e in
         (num "part" args, num "bytes" args))
       (List.filter (fun e -> str "ph" e = "i") trace))

(* Counters are their run's, whatever runs are not counted between them, and
   runs lost before they were read are an event of their own. *)
let test_counted_runs () =
  let log = ref [] in
  let d, run, lose = counting log in
  let stamps = B.create host S.UInt64 4 in
  let p = P.start ~counters:[ "A" ] () in
  run [ (1_000, 2_000) ];
  let uncounted = stamped d stamps [ ("compute", "k") ] (3_000, 4_500) in
  run [ (5_000, 6_500) ];
  lose 2;
  Nx_device.synchronize d;
  Domain.join uncounted;
  let trace =
    match field "traceEvents" (written (P.stop p)) with
    | Arr l -> l
    | _ -> fail "no array"
  in
  let ph p = List.filter (fun e -> str "ph" e = p) trace in
  let lane e =
    let pid = num "pid" e and tid = num "tid" e in
    List.find_map
      (fun m ->
        if
          str "name" m = "thread_name" && num "pid" m = pid && num "tid" m = tid
        then Some (str "name" (field "args" m))
        else None)
      (ph "M")
  in
  let complete =
    List.map
      (fun e ->
        ( Option.value ~default:"" (lane e),
          num "ts" e,
          num "dur" e,
          match field "args" e with
          | exception Failure _ -> None
          | args -> Some (num "A" args) ))
      (ph "X")
  in
  equal ~msg:"the counted runs at their own times, the span uncounted"
    (list
       (Testable.make
          ~pp:(fun ppf (l, t, d, a) ->
            Format.fprintf ppf "%s %g+%g %s" l t d
              (Option.fold ~none:"-" ~some:string_of_float a))
          ~equal:( = )))
    [
      ("counters", 0., 1., Some 1.);
      ("compute", 2., 1.5, None);
      ("counters", 4., 1.5, Some 1.);
    ]
    complete;
  equal ~msg:"the runs lost" (list float_exact) [ 2. ]
    (List.map
       (fun e -> num "runs" (field "args" e))
       (List.filter (fun e -> str "name" e = "overwritten") (ph "i")))

let test_record () =
  let a = (fake ~name:"A" ()).dev and b = (fake ~name:"B" ()).dev in
  let stamps = B.create host S.UInt64 4 in
  let events =
    profiled (fun () ->
        Nx_device.submit [ a ] ~touches:[] (fun s ->
            raises_match ~msg:"a device outside the submission" Exn.invalid_arg
              (fun () ->
                Nx_device.Submission.record s b ~lane:"l" ~name:"n" stamps);
            raises_match ~msg:"stamps of one slot" Exn.invalid_arg (fun () ->
                Nx_device.Submission.record s a ~lane:"l" ~name:"n"
                  (B.create host S.UInt64 2)));
        raises (Failure "encode") (fun () ->
            Nx_device.submit [ a ] ~touches:[] (fun s ->
                Nx_device.Submission.record s a ~lane:"l" ~name:"dropped" stamps;
                failwith "encode"));
        settle [ a ];
        Nx_device.synchronize a)
  in
  equal ~msg:"no span of a submission that raised" (list span_) []
    (spans events)

let profiles =
  group "profiles"
    [
      test "are taken one at a time, and stopped by their holder" test_sessions;
      test
        "host spans nest on the lane of their domain, and one records a \
         function that raises"
        test_host_spans;
      test "record nothing and allocate nothing while no profile is taken"
        test_off;
      test
        "read a submitter's stamps at the next synchronization, a later record \
         of the same stamps replacing the earlier"
        test_submitted;
      test "let the device resolve stamps before they are read" test_resolve;
      test
        "refuse a span of a device outside its submission or of one slot, and \
         keep none of a submission that raised"
        test_record;
      test
        "make each copy a span of the host, and each a copy queue ran a span \
         of its copy lane inside it"
        (test_copies ?clock:None ~slack:0);
      test "calibrate the spans of a device's own clock onto the host's"
        (test_copies
           ~clock:(Driver.Device_clock { hz = 1_000_000 })
           ~slack:1_000_000);
      test "time both devices of a bounce" test_bounce;
      test "sample a device's allocated memory at each change"
        test_memory_events;
      test "record the first load of a program" test_program_events;
      test "leave out the unread spans of a device that fails" test_failed_spans;
      test
        "read the counters a profile asks for at each synchronization and when \
         it stops, in the order of the runs"
        test_counters;
      test
        "write Chrome's trace event format: named processes and threads, \
         escaped names, time order, nesting"
        test_output;
      test "show each run's counters at the run's own times, and the runs lost"
        test_counted_runs;
      test "read the traces a profile asks for, and show where they are"
        test_traces;
    ]

(* Machines *)

(* Another machine, whose memory stands in for its own: the process reaches it
   only through the host's [io], which counts its calls and fails once the
   machine is [down]. *)
type machine = {
  mhost : Nx_device.t;
  down : bool ref;
  reads : int ref;
  writes : int ref;
  copies : int ref;
}

let machine ?(name = "far:1") ?(most = max_int) ?programs () =
  let down = ref false
  and reads = ref 0
  and writes = ref 0
  and copies = ref 0 in
  let keep = Hashtbl.create 8 in
  let reach count f =
    if !down then failwith (name ^ ": connection lost");
    incr count;
    f ()
  in
  let io =
    {
      Driver.read =
        (fun ~src ~dst n -> reach reads (fun () -> memmove dst src n));
      write = (fun ~dst ~src n -> reach writes (fun () -> memmove dst src n));
      copy = (fun ~dst ~src n -> reach copies (fun () -> memmove dst src n));
    }
  in
  let memory =
    {
      Driver.alloc =
        (fun n -> if n > most then None else block keep ~addressed:true n);
      free = (fun m -> Hashtbl.remove keep (Region.address m));
    }
  in
  let mhost = Driver.host ~address:name ~arch:"test" ?programs ~memory io in
  { mhost; down; reads; writes; copies }

(* A GPU of [m] whose memory its host does not address. Its copies run at once
   and signal its timeline, which only its machine's host addresses. *)
let remote_gpu ?(name = "GPU") m =
  let keep = Hashtbl.create 8 in
  let free r = Hashtbl.remove keep (Region.address r) in
  let queue ~timeline =
    let word = Option.get (Region.host_address timeline) in
    let copy ~dst ~src n ~signal:v =
      if !(m.down) then failwith "the machine is down";
      memmove dst src n;
      store_signal word v
    in
    let stamp ~slot ~signal:v =
      store_signal (Nativeint.add slot 8n) v;
      store_signal word v
    in
    { Driver.copy; transfer = (fun _ -> None); stamp; clock = Host_clock }
  in
  let map a n = Ok (Region.v ~host:a ~handle:a a n) in
  Driver.device ~host:m.mhost ~name ~arch:"test" ~budget:max_int
    (Device_local
       {
         memory = { alloc = block keep ~addressed:false; free };
         host_memory = { alloc = block keep ~addressed:true; free };
         mapped = None;
         mapping = Pages { map; unmap = ignore };
         queue;
       })

let calls (m : machine) = !(m.reads) + !(m.writes) + !(m.copies)

let test_machines () =
  let m = machine () in
  let gpu = remote_gpu m in
  let near = fake () in
  equal ~msg:"a device of another machine" (arg pp_device) m.mhost
    (Nx_device.host_of gpu);
  equal ~msg:"a host is its own" (arg pp_device) m.mhost
    (Nx_device.host_of m.mhost);
  equal ~msg:"this machine's" (arg pp_device) host (Nx_device.host_of near.dev);
  equal ~msg:"the host" (arg pp_device) host (Nx_device.host_of host);
  equal ~msg:"names composed by the runtime" (list string)
    [ "CPU@far:1"; "GPU@far:1"; "GPU:2@far:1" ]
    [
      Nx_device.name m.mhost;
      Nx_device.name gpu;
      Driver.name ~host:m.mhost "GPU:2";
    ];
  raises_match (Exn.invalid_arg ~substring:"not a host") (fun () ->
      Driver.device ~host:gpu ~name:"X" ~arch:"test" ~budget:1
        (Host_visible
           {
             memory = { alloc = (fun _ -> None); free = ignore };
             mapping = None;
           }));
  let b = B.create m.mhost S.UInt8 16 in
  equal ~msg:"its address is its host's" (option nativeint)
    (Some (B.address b))
    (Region.host_address (Region.of_buffer b));
  raises_match (Exn.invalid_arg ~substring:"not CPU") (fun () ->
      B.bigarray Bigarray.char b);
  raises_match (Exn.invalid_arg ~substring:"not on GPU@far:1's machine")
    (fun () -> B.borrow gpu (B.create host S.UInt8 page))

let test_host_copies () =
  let m = machine () in
  let b = B.create m.mhost S.UInt8 100 and c = B.create m.mhost S.UInt8 100 in
  let w = !(m.writes) and r = !(m.reads) in
  write b (pattern 1 100);
  is_true ~msg:"written through io" (!(m.writes) > w);
  B.copy ~src:b ~dst:c;
  is_true ~msg:"a copy there is the host's" (!(m.copies) > 0);
  equal ~msg:"read back through io" string (pattern 1 100) (read c);
  is_true ~msg:"read through io" (!(m.reads) > r);
  let s = Nx_device.Stats.diff (stats m.mhost) (stats m.mhost) in
  equal ~msg:"stats answer" int 0 (Nx_device.Stats.allocated s)

let test_remote_gpu () =
  let m = machine () in
  let gpu = remote_gpu m in
  let on = B.create gpu S.UInt8 5000 and back = B.create gpu S.UInt8 5000 in
  write on (pattern 2 5000);
  B.copy ~src:on ~dst:back;
  equal ~msg:"from this machine and back, through the far host" string
    (pattern 2 5000) (read back);
  let hb = B.create m.mhost S.UInt8 5000 in
  B.copy ~src:back ~dst:hb;
  equal ~msg:"into its host's memory" string (pattern 2 5000) (read hb);
  let pinned = B.create ~memory:Pinned gpu S.UInt8 5000 in
  write pinned (pattern 3 5000);
  B.copy ~src:pinned ~dst:on;
  equal ~msg:"from its host memory" string (pattern 3 5000) (read on);
  let r = !(m.reads) in
  ignore (submit gpu Fun.id);
  (* The work never signals: the wait polls the far word until the connection
     fails. *)
  let down =
    Domain.spawn (fun () ->
        Unix.sleepf 0.1;
        m.down := true)
  in
  raises_match (lost gpu "far:1: connection lost") (fun () ->
      Nx_device.synchronize gpu);
  Domain.join down;
  is_true ~msg:"the signal word was read through io" (!(m.reads) > r)

let test_between_machines () =
  let m = machine () and m' = machine ~name:"far:2" () in
  let g = remote_gpu m and g' = remote_gpu m' in
  let local = (far ()).dev in
  let slot = 64 lsl 20 in
  let n = slot + 777 in
  let bytes = pattern 5 n in
  let on_local = B.create local S.UInt8 n in
  write on_local bytes;
  let on_g = B.create g S.UInt8 n in
  B.copy ~src:on_local ~dst:on_g;
  let on_g' = B.create g' S.UInt8 n in
  B.copy ~src:on_g ~dst:on_g';
  let hb = B.create m'.mhost S.UInt8 n in
  B.copy ~src:on_g' ~dst:hb;
  let back = B.create local S.UInt8 n in
  B.copy ~src:hb ~dst:back;
  is_true ~msg:"the bytes crossed three machines and came back"
    (String.equal bytes (read back));
  equal ~msg:"bytes out of the first" int n
    (Nx_device.Stats.bytes_out (stats g))

let test_machine_down () =
  let m = machine () in
  let gpu = remote_gpu m in
  let on = B.create gpu S.UInt8 64 and hb = B.create m.mhost S.UInt8 64 in
  let near = B.create host S.UInt8 64 and here = B.create host S.UInt8 20 in
  m.down := true;
  let down = "far:1: connection lost" in
  raises_match (lost m.mhost down) (fun () -> B.copy ~src:near ~dst:hb);
  raises_match (lost m.mhost down) (fun () -> B.create m.mhost S.UInt8 8);
  raises_match (lost m.mhost down) (fun () -> B.copy ~src:near ~dst:on);
  (* The GPU is lost once an operation of its own meets the machine, which a
     submission does not: the runtime publishes no submitted value. *)
  let c = calls m in
  ignore (submit gpu Fun.id);
  equal ~msg:"a submission reaches no machine" int c (calls m);
  raises_match (lost gpu down) (fun () -> Nx_device.synchronize gpu);
  let c = calls m in
  raises_match (lost gpu down) (fun () -> Nx_device.synchronize gpu);
  equal ~msg:"a lost device reaches its machine no more" int c (calls m);
  write here "this machine goes on";
  equal ~msg:"this machine goes on" string "this machine goes on" (read here)

let test_no_staging () =
  let m = machine ~most:(1 lsl 20) () in
  let gpu = remote_gpu m in
  raises_match (out_of_memory m.mhost staging_bytes) (fun () ->
      B.copy ~src:(B.create host S.UInt8 64) ~dst:(B.create gpu S.UInt8 64))

(* A link that carries copies between machines, counting them. *)
let test_links () =
  let m = machine () in
  let gpu = remote_gpu m in
  let moved = ref 0 and broken = ref false in
  let rec nic =
    lazy
      (Driver.device ~name:"NIC" ~arch:"test" ~budget:0
         ~link:(fun ~src ~dst ->
           if
             Nx_device.host_of (B.device src)
             != Nx_device.host_of (B.device dst)
           then
             Some
               {
                 Driver.through = [ Lazy.force nic ];
                 move =
                   (fun ~src ~dst ->
                     if !broken then failwith "NIC: retries exhausted";
                     incr moved;
                     ignore (src, dst));
               }
           else None)
         (Host_visible
            {
              memory = { alloc = (fun _ -> None); free = ignore };
              mapping = None;
            }))
  in
  let nic = Lazy.force nic in
  let local = (far ()).dev in
  let a = B.create local S.UInt8 32 and b = B.create gpu S.UInt8 32 in
  let moves () = !(m.writes) + !(m.copies) in
  let c = moves () in
  B.copy ~src:a ~dst:b;
  equal ~msg:"carried by the link" int 1 !moved;
  equal ~msg:"not through the hosts" int c (moves ());
  B.copy ~src:a ~dst:(B.create local S.UInt8 32);
  equal ~msg:"not within a machine" int 1 !moved;
  broken := true;
  let exhausted = lost nic "NIC: retries exhausted" in
  raises_match exhausted (fun () -> B.copy ~src:a ~dst:b);
  raises_match exhausted (fun () -> B.create nic S.UInt8 1);
  raises_match exhausted (fun () ->
      B.copy ~src:(B.create gpu S.UInt8 32) ~dst:b);
  B.copy ~src:(B.create local S.UInt8 32) ~dst:a

(* A device that describes its memory, and what runs when it frees it. *)
let test_dma () =
  let keep = Hashtbl.create 4 in
  let d =
    Driver.device ~name:"DMA" ~arch:"test" ~budget:max_int
      ~dma:(fun r ->
        if Hashtbl.length keep > 2 then Error "no window left"
        else
          Ok
            {
              Driver.bus = "0000:03:00.0";
              pages = [ (0x1000, Nativeint.to_int (Region.address r)) ];
            })
      (Host_visible
         {
           memory =
             {
               alloc = block keep ~addressed:true;
               free = (fun r -> Hashtbl.remove keep (Region.address r));
             };
           mapping = None;
         })
  in
  let b = B.create d S.UInt8 4096 in
  let dma =
    require_ok ~pp:Format.pp_print_string
      (Driver.dma (B.view b ~offset:64 S.UInt8 8))
  in
  equal ~msg:"the memory the buffer lies in"
    (pair string (list (pair int int)))
    ("0000:03:00.0", [ (0x1000, Nativeint.to_int (B.address b)) ])
    (dma.bus, dma.pages);
  (match Driver.dma (B.create (fake ()).dev S.UInt8 8) with
  | Ok _ -> fail "described"
  | Error why -> contains ~msg:"undescribed" ~sub:"does not describe" why);
  ignore (Sys.opaque_identity b)

let test_depends () =
  let opened = ref false in
  let f = fake ~signal:(gate opened) () in
  let runs = ref 0 in
  raises_match (Exn.invalid_arg ~substring:"allocated") (fun () ->
      Driver.depends (B.create host S.UInt8 8) ignore);
  let first =
    dropped_at f.dev 4096 (fun b ->
        Driver.depends (B.view b ~offset:64 S.UInt8 8) (fun () -> incr runs);
        ignore (submit f.dev ~touches:[ b ] Fun.id))
  in
  ignore (stats f.dev);
  equal ~msg:"not run while its own device's work is unsignaled" int 0 !runs;
  let held = B.create f.dev S.UInt8 4096 in
  equal ~msg:"not reused before it runs" bool false (B.address held = first);
  opened := true;
  ignore (stats f.dev);
  equal ~msg:"run once that work is signaled" int 1 !runs;
  equal ~msg:"then reused" nativeint first
    (B.address (B.create f.dev S.UInt8 4096));
  ignore
    (dropped_at f.dev 8192 (fun b ->
         Driver.depends b (fun () -> failwith "the mapper is gone")));
  Nx_device.free_cache f.dev;
  equal ~msg:"memory whose dependant raised is retained" int 8192
    (Nx_device.Stats.retained (stats f.dev));
  ignore (Sys.opaque_identity held)

let test_remote_programs () =
  let calls = ref [] and unloaded = ref 0 in
  let load ~binary ~entry =
    ignore binary;
    Ok (Nativeint.of_int (String.length entry), fun () -> incr unloaded)
  in
  let call h bufs vals = calls := (h, bufs, vals) :: !calls in
  let m = machine ~programs:{ load; call } () in
  let p = program m.mhost ~binary:"elf" ~name:"submit" in
  let b = B.create m.mhost S.UInt8 24 in
  Nx_device.Program.call p [| B.view b ~offset:8 S.UInt8 16 |] [| 7 |];
  (match !calls with
  | [ (h, [| (a, n) |], [| 7 |]) ] ->
      equal ~msg:"the handle" nativeint 6n h;
      equal ~msg:"the buffer's size" int 16 n;
      is_true ~msg:"an address there" (a <> 0n)
  | _ -> fail "one call");
  raises_match (Exn.invalid_arg ~substring:"does not address") (fun () ->
      Nx_device.Program.call p [| B.create host S.UInt8 8 |] [||]);
  dropped (fun () -> p);
  ignore (stats m.mhost);
  equal ~msg:"an unreachable program is unloaded there" int 1 !unloaded

let test_reach () =
  let reaches = Nx_device.reaches in
  let visible = (fake ~name:"VISIBLE" ~maps:true ()).dev in
  let unmapped = (fake ~name:"UNMAPPED" ()).dev in
  let f1 = (far ~name:"F1" ()).dev in
  let f2 = (far ~name:"F2" ~reaches:(fun d -> d == f1) ()).dev in
  let m = machine () in
  let gpu = remote_gpu m in
  is_true ~msg:"its own memory" (reaches f1 f1);
  is_true ~msg:"the host's, mapped" (reaches f1 host && reaches visible host);
  is_false ~msg:"the host's, unmapped" (reaches unmapped host);
  is_true ~msg:"the host reaches memory it addresses" (reaches host visible);
  is_false ~msg:"the host does not reach a GPU's own" (reaches host f1);
  is_true ~msg:"memory the host addresses, through a mapping"
    (reaches f1 visible);
  is_false ~msg:"a peer its driver does not map" (reaches f1 f2);
  is_true ~msg:"a peer its driver maps" (reaches f2 f1);
  is_false ~msg:"the disk"
    (reaches f1 Nx_device.disk || reaches Nx_device.disk host);
  is_true ~msg:"its machine's host" (reaches gpu m.mhost);
  is_false ~msg:"another machine's" (reaches gpu host || reaches f1 gpu)

let machines =
  group "machines"
    [
      test "which devices' memory a device's work reaches" test_reach;
      test "devices of another machine and their host" test_machines;
      test "another machine's host copies through its io" test_host_copies;
      test "a GPU of another machine" test_remote_gpu;
      test "copies between three machines, in chunks" test_between_machines;
      test "a machine that goes down loses its devices alone" test_machine_down;
      test "a host without memory for its staging raises Out_of_memory"
        test_no_staging;
      test "links carry copies between machines" test_links;
      test "memory described to other functions" test_dma;
      test
        "what depends on memory runs once it is released and all work on it is \
         done, its own device's too"
        test_depends;
      test "programs of another machine's host" test_remote_programs;
    ]

(* Reaching memory for a device's work: a borrow where the device maps it, and a
   staged buffer for host memory under 64 KiB, which a device that maps whole
   pages does not. The device's work is a domain that reads or writes the staged
   memory at its address, then signals. *)

(* The [n] bytes at the address [a]. *)
let bytes_at a n =
  let ba = chars n in
  memmove (B.address (B.of_bigarray ba)) a n;
  List.init n (fun i -> Char.code ba.{i})

(* A host buffer of [n] bytes holding [bytes]. *)
let host_bytes bytes =
  let ba = chars (List.length bytes) in
  List.iteri (fun i b -> ba.{i} <- Char.chr b) bytes;
  let b = B.create host S.UInt8 (List.length bytes) in
  B.copy ~src:(B.of_bigarray ba) ~dst:b;
  b

let contents b =
  let n = B.nbytes b in
  let ba = chars n in
  B.copy ~src:b ~dst:(B.of_bigarray ba);
  List.init n (fun i -> Char.code ba.{i})

(* Whether [s] holds [sub]. *)
let mentions s sub =
  let n = String.length sub in
  let rec at i =
    i + n <= String.length s && (String.sub s i n = sub || at (i + 1))
  in
  at 0

(* A device's signal whose first wait holds until [release], as a run whose work
   completed but whose domain has not yet gone on. A wait made meanwhile finds
   the work done. *)
let held_signal () =
  let signaled = Atomic.make 0 and held = Atomic.make false in
  let opened = Atomic.make false in
  let lock = Mutex.create () and cond = Condition.create () in
  let rec reach v =
    let s = Atomic.get signaled in
    if v > s && not (Atomic.compare_and_set signaled s v) then reach v
  in
  let wait v ~ms:_ =
    if (not (Atomic.get opened)) && not (Atomic.exchange held true) then
      Mutex.protect lock (fun () ->
          while not (Atomic.get opened) do
            Condition.wait cond lock
          done);
    reach v;
    true
  in
  let release () =
    Mutex.protect lock (fun () ->
        Atomic.set opened true;
        Condition.broadcast cond)
  in
  ({ Driver.signaled = (fun () -> Atomic.get signaled); wait }, held, release)

let reached d b access =
  match B.reach d b access with Ok r -> r | Error why -> failwith why

(* [d]'s work: after 50 ms, [f], then the signal of [v]. *)
let work_later d v f =
  let word = B.address (Nx_device.signal_word d) in
  Domain.spawn (fun () ->
      Unix.sleepf 0.05;
      f ();
      store_signal word v)

(* Host buffers that devices reach for their work, borrowed when they map the
   memory and staged otherwise, and far devices' buffers that copies fill and
   drain through the host's staging memory, while an adversary loses devices
   between calls. A lost device's loss reaches the host memory it borrowed, and
   no other host memory; the host's staging memory that a lost device mapped is
   replaced at its next use, which the devices then map anew. *)
module Stage = struct
  exception Lost
  exception Refused

  type device = {
    far : bool;
    mutable lost : bool;
    mutable mappings : int; (* of the host's staging memory *)
  }

  (* The host's staging memory, and the far devices that mapped it. It is the
     process's, so its model is too. *)
  type staging = { mutable mappers : device list }

  let current = ref { mappers = [] }

  type host = {
    size : int;
    mutable bytes : string;
    mutable borrowers : device list;
  }

  type reached = {
    original : host;
    device : device;
    access : B.access;
    staged : bool;
  }

  type on_device = { owner : device; mutable contents : string }

  let device far = { far; lost = false; mappings = 0 }
  let big h = h.size >= page
  let alive d = if d.lost then raise Lost

  (* Raises [Lost] if a lost device borrowed [h]'s memory. *)
  let reachable h =
    cover "a lost device borrowed a host buffer"
      (List.exists (fun d -> d.lost) h.borrowers);
    List.iter alive h.borrowers

  let host n seed = { size = n; bytes = pattern seed n; borrowers = [] }

  let write h seed =
    reachable h;
    h.bytes <- pattern seed h.size

  let read h =
    reachable h;
    Digest.to_hex (Digest.string h.bytes)

  (* Memory a device maps is borrowed, and fewer than 64 KiB of host memory,
     which starts on no page, is staged. *)
  let reach d h access =
    alive d;
    let staged = not (big h) in
    if not staged then h.borrowers <- d :: h.borrowers;
    { original = h; device = d; access; staged }

  (* The work reads the bytes the original held at the submission, and writes
     [seed]'s pattern when it may, which the original holds once it returns. *)
  let run r seed =
    alive r.device;
    reachable r.original;
    let seen = Digest.to_hex (Digest.string r.original.bytes) in
    cover "a staged buffer is copied back" (r.staged && r.access = Read_write);
    cover "the work writes a borrow" ((not r.staged) && r.access = Read_write);
    if r.access = B.Read_write then
      r.original.bytes <- pattern seed r.original.size;
    seen

  (* A copy of [d] through the host's staging memory, which [d] maps at its
     first such copy, and again once a lost device's mapping made it
     replaced. *)
  let staged_copy d =
    let replaced = List.exists (fun m -> m.lost) !current.mappers in
    cover "the host's staging memory is replaced" replaced;
    if replaced then current := { mappers = [] };
    if not (List.memq d !current.mappers) then begin
      !current.mappers <- d :: !current.mappers;
      d.mappings <- d.mappings + 1
    end

  let blit src dst =
    let n = Int.min (String.length src) (String.length dst) in
    String.sub src 0 n ^ String.sub dst n (String.length dst - n)

  let create d h =
    alive d;
    reachable h;
    staged_copy d;
    { owner = d; contents = h.bytes }

  let copy_in h b =
    alive b.owner;
    reachable h;
    staged_copy b.owner;
    b.contents <- blit h.bytes b.contents

  let copy_out b h =
    alive b.owner;
    reachable h;
    staged_copy b.owner;
    h.bytes <- blit b.contents h.bytes

  let lose d =
    cover "a device that mapped the staging memory is lost" (d.mappings > 0);
    d.lost <- true;
    raise Lost
end

(* A device whose driver faults at its next synchronization once [faulted]. *)
type stage_device = { stage : fake; faulted : bool ref }

let stage_device far =
  let faulted = ref false in
  let synchronized () = if !faulted then failwith "faulted" in
  let stage =
    if far then fake ~name:"FAR" ~far:true ~synchronized ()
    else
      (* Its work completes once the host waits for it. *)
      let signaled = ref 0 in
      let signal =
        {
          Driver.signaled = (fun () -> !signaled);
          wait =
            (fun v ~ms:_ ->
              signaled := Int.max !signaled v;
              true);
        }
      in
      fake ~name:"NEAR" ~maps:true ~signal ~synchronized ()
  in
  { stage; faulted }

type staged_buffer = { sb : B.t }
type reach = { rb : B.t; writes : bool }

let digest_at b n =
  Digest.to_hex (Digest.string (string_of (peek (B.address b) n)))

let stage_dev =
  abstract "d" ~invariant:(fun (r : Stage.device) s ->
      if r.far then
        equal ~msg:"the host's staging memories it mapped" int r.mappings
          (List.length s.stage.drv.staging))

let host_buf =
  abstract "h" ~invariant:(fun (r : Stage.host) s ->
      equal ~msg:"its bytes" string r.bytes
        (string_of (peek (B.address s.sb) r.size)))

let reached_buf =
  abstract "r" ~invariant:(fun (r : Stage.reached) s ->
      equal ~msg:"staged" bool r.staged (B.is_staged s.rb);
      equal ~msg:"bytes" int r.original.size (B.nbytes s.rb))

let device_buf =
  abstract "b" ~invariant:(fun (r : Stage.on_device) s ->
      equal ~msg:"its bytes" string r.contents
        (string_of (peek (B.address s.sb) (String.length r.contents))))

(* The first [n] bytes of [b], or all of it. *)
let prefix b n = B.view b ~offset:0 S.UInt8 (Int.min n (B.nbytes b))

let stage_commands =
  let accesses =
    let pp ppf (a : B.access) =
      Format.pp_print_string ppf
        (match a with Read -> "Read" | Read_write -> "Read_write")
    in
    Gen.of_list ~pp B.[ Read; Read_write ]
  in
  let sizes =
    Gen.frequency
      [
        (2, ints [ 1; 3; 16; 64 ]);
        (1, Gen.constant ~pp:Format.pp_print_int page);
      ]
  in
  let small (h : Stage.host) = not (Stage.big h) in
  let copied ~src ~dst =
    let n = Int.min (B.nbytes src) (B.nbytes dst) in
    B.copy ~src:(prefix src n) ~dst:(prefix dst n)
  in
  [
    command "near"
      (Gen.unit @-> makes stage_dev)
      (fun () -> Stage.device false)
      (fun () -> stage_device false);
    command "far"
      (Gen.unit @-> makes stage_dev)
      (fun () -> Stage.device true)
      (fun () -> stage_device true);
    command "host"
      (sizes @-> seeds @-> makes host_buf)
      Stage.host
      (fun n seed ->
        let sb = B.create host S.UInt8 n in
        write sb (pattern seed n);
        { sb });
    command "write"
      (host_buf ^-> seeds @-> returns unit)
      Stage.write
      (fun s seed -> write s.sb (pattern seed (B.nbytes s.sb)));
    command "read"
      (host_buf ^-> returns string)
      Stage.read
      (fun s -> Digest.to_hex (Digest.string (read s.sb)));
    command "reach"
      (stage_dev ^-> host_buf ^-> accesses @-> makes reached_buf)
      Stage.reach
      (fun d s access ->
        match B.reach d.stage.dev s.sb access with
        | Ok rb -> { rb; writes = access = B.Read_write }
        | Error _ -> raise Stage.Refused);
    command "run"
      ~pre:(fun (r : Stage.reached) _ -> not r.device.far)
      (reached_buf ^-> seeds @-> returns string)
      Stage.run
      (fun { rb; writes } seed ->
        let n = B.nbytes rb in
        let written = of_string (pattern seed n) in
        submit (B.device rb) ~touches:[ rb ] (fun _ ->
            let seen = digest_at rb n in
            if writes then memmove (B.address rb) (B.address written) n;
            seen));
    command "copy to a device"
      ~pre:(fun (d : Stage.device) h -> d.far && small h)
      (stage_dev ^-> host_buf ^-> makes device_buf)
      Stage.create
      (fun d s ->
        let sb = B.create d.stage.dev S.UInt8 (B.nbytes s.sb) in
        B.copy ~src:s.sb ~dst:sb;
        { sb });
    command "copy in"
      ~pre:(fun h _ -> small h)
      (host_buf ^-> device_buf ^-> returns unit)
      Stage.copy_in
      (fun h b -> copied ~src:h.sb ~dst:b.sb);
    command "copy out"
      ~pre:(fun _ h -> small h)
      (device_buf ^-> host_buf ^-> returns unit)
      Stage.copy_out
      (fun b h -> copied ~src:b.sb ~dst:h.sb);
    command "lose"
      (stage_dev ^-> returns unit)
      Stage.lose
      (fun d ->
        d.faulted := true;
        Nx_device.synchronize d.stage.dev);
  ]

let staging =
  group "reach"
    [
      stateful ~count:200 ~steps:30
        "devices reach host buffers by borrowing or staging them, the work \
         reads and writes them through either, and a lost device reaches the \
         host memory it borrowed alone, its mapping of the host's staging \
         memory replaced"
        stage_commands;
      test "memory the device maps is borrowed, and its own is itself"
        (fun () ->
          let d = (fake ~maps:true ()).dev in
          let big = B.create host S.UInt8 page in
          let r = reached d big B.Read in
          is_false ~msg:"borrowed, not staged" (B.is_staged r);
          is_true ~msg:"the borrow's memory"
            (Nativeint.equal
               (B.address (Result.get_ok (B.borrow d big)))
               (B.address r));
          let own = B.create d S.UInt8 8 in
          is_true ~msg:"its own" (reached d own B.Read_write == own));
      test
        "a staged buffer read by the work holds the bytes of its submission, \
         which returns before the work completes" (fun () ->
          let d = (fake ~maps:true ()).dev in
          let b = host_bytes [ 1; 2; 3; 4 ] in
          let r = reached d b B.Read in
          is_true ~msg:"staged" (B.is_staged r);
          let v = submit d ~touches:[ r ] Fun.id in
          is_true ~msg:"returned before the work" (Nx_device.signaled d < v);
          B.copy ~src:(host_bytes [ 9; 9; 9; 9 ]) ~dst:b;
          equal (list int) ~msg:"what the work reads" [ 1; 2; 3; 4 ]
            (bytes_at (B.address r) 4);
          store_signal (B.address (Nx_device.signal_word d)) v);
      test
        "a staged buffer the work writes is copied back once the work \
         completed, the bytes it did not write kept" (fun () ->
          let d = (fake ~maps:true ()).dev in
          let b = host_bytes [ 1; 2; 3; 4; 5; 6; 7; 8 ] in
          let r = reached d b B.Read_write in
          let written = B.of_bigarray (chars 4) in
          B.copy ~src:(host_bytes [ 9; 9; 9; 9 ]) ~dst:written;
          let domain =
            submit d ~touches:[ r ] (fun v ->
                work_later d v (fun () ->
                    memmove (B.address r) (B.address written) 4))
          in
          is_true ~msg:"completed" (Nx_device.signaled d = Nx_device.submitted d);
          equal (list int) [ 9; 9; 9; 9; 5; 6; 7; 8 ] (contents b);
          Domain.join domain);
      test
        "two runs over different sources, the first in flight: each reads its \
         own, and the first's memory is not reused under it" (fun () ->
          let d = (fake ~maps:true ()).dev in
          let first () =
            let r = reached d (host_bytes [ 1; 2; 3; 4 ]) B.Read in
            ignore (submit d ~touches:[ r ] Fun.id);
            B.address r
          in
          let a1 = first () in
          Gc.full_major ();
          let r2 = reached d (host_bytes [ 5; 6; 7; 8 ]) B.Read in
          let v2 = submit d ~touches:[ r2 ] Fun.id in
          equal (list int) ~msg:"the first run's bytes" [ 1; 2; 3; 4 ]
            (bytes_at a1 4);
          equal (list int) ~msg:"the second run's bytes" [ 5; 6; 7; 8 ]
            (bytes_at (B.address r2) 4);
          store_signal (B.address (Nx_device.signal_word d)) v2);
      test
        "a staged copy fills a slot of the host's staging memory once the work \
         that queued a use of it is done" (fun () ->
          let near = (fake ~maps:true ()).dev and gpu = (far ()).dev in
          let staging = Nx_device.staging host in
          B.copy
            ~src:(host_bytes [ 7; 7; 7; 7 ])
            ~dst:(B.view staging ~offset:0 S.UInt8 4);
          let on_near = Result.get_ok (B.borrow near staging) in
          let seen = chars 4 in
          let domain =
            submit near ~touches:[ on_near ] (fun v ->
                work_later near v (fun () ->
                    memmove
                      (B.address (B.of_bigarray seen))
                      (B.address on_near) 4))
          in
          let dst = B.create gpu S.UInt8 4 in
          B.copy ~src:(host_bytes [ 1; 2; 3; 4 ]) ~dst;
          Domain.join domain;
          equal (list int) ~msg:"what the queued use read" [ 7; 7; 7; 7 ]
            (List.init 4 (fun i -> Char.code seen.{i}));
          equal (list int) ~msg:"what the copy landed" [ 1; 2; 3; 4 ]
            (contents dst));
      test
        "a device lost while it maps the host's staging memory leaves the \
         other devices' staged copies working" (fun () ->
          let broken = far ~name:"BROKEN" () and gpu = (far ()).dev in
          let b = B.create broken.dev S.UInt8 4 in
          B.copy ~src:(host_bytes [ 1; 1; 1; 1 ]) ~dst:b;
          broken.drv.broken <- true;
          raises_match (lost broken.dev "enqueue refused") (fun () ->
              B.copy ~src:(host_bytes [ 2; 2; 2; 2 ]) ~dst:b);
          let dst = B.create gpu S.UInt8 4 in
          B.copy ~src:(host_bytes [ 3; 4; 5; 6 ]) ~dst;
          equal (list int) [ 3; 4; 5; 6 ] (contents dst));
      test
        "two runs from two domains over one staged buffer that the work writes \
         copy back in turn" (fun () ->
          let signal, held, release = held_signal () in
          let d = (fake ~maps:true ~signal ()).dev in
          let b = host_bytes [ 1; 1; 1; 1 ] in
          let r = reached d b B.Read_write in
          let nines = B.of_bigarray (chars 4) in
          B.copy ~src:(host_bytes [ 9; 9; 9; 9 ]) ~dst:nines;
          (* The first run writes nines and is held after its work, before it
             copies them back. *)
          let first =
            Domain.spawn (fun () ->
                submit d ~touches:[ r ] (fun _ ->
                    memmove (B.address r) (B.address nines) 4))
          in
          while not (Atomic.get held) do
            Domain.cpu_relax ()
          done;
          let second =
            Domain.spawn (fun () -> submit d ~touches:[ r ] ignore)
          in
          Unix.sleepf 0.05;
          release ();
          Domain.join first;
          Domain.join second;
          equal (list int) [ 9; 9; 9; 9 ] (contents b));
      test
        "host memory of 64 KiB or more the device does not map is refused, \
         naming its size" (fun () ->
          let d = (fake ~maps:true ()).dev in
          let ba = chars (2 * page) in
          let off_page = B.of_bigarray (Bigarray.Array1.sub ba 1 page) in
          match B.reach d off_page B.Read with
          | Ok _ -> fail "reached"
          | Error why -> is_true ~msg:why (mentions why (string_of_int page)));
    ]

let () =
  if Sys.getenv_opt "NX_DEVICE_FINALIZE_CHILD" = Some "1" then finalize_child ();
  exit
    (run "nx.device"
       [
         devices;
         memory;
         memories;
         pools;
         laws;
         borrows;
         buffers;
         claims;
         disks;
         refusals;
         timeline;
         submissions;
         failures;
         hooks;
         driver_wait;
         profiles;
         machines;
         staging;
       ])
