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

let host = Nx_device.host
let chars n : chars = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n

let string_of (ba : chars) =
  String.init (Bigarray.Array1.dim ba) (Bigarray.Array1.get ba)

let of_string s =
  B.of_bigarray
    (Bigarray.Array1.init Bigarray.char Bigarray.c_layout (String.length s)
       (String.get s))

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

let pattern seed n =
  String.init n (fun i ->
      Char.chr (((seed * 31) + (i * 7) + (i / 5)) land 0xff))

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
   staging memory. A far driver's own memory is not addressed by the host, and
   its copy queue runs copies and timestamps when the host waits for them, as a
   GPU runs behind the host: [stalled] stops it, [broken] makes it refuse to
   enqueue, and a device named PEER... copies into the memory of the others. Its
   timestamps are the host clock, or, with a clock of its own, ticks of it two
   hours ahead. *)
type driver = {
  blocks : (nativeint, chars * int) Hashtbl.t;
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
    ?signal ?load ?peer ?timeout_ms ?synchronized ?sleep ?finalize ?clock
    ?resolve ?room () =
  let drv =
    {
      blocks = Hashtbl.create 8;
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
  let alloc ~addressed n =
    if drv.refuse then None
    else
      let ba = chars (n + 31) in
      let a = B.address (B.of_bigarray ba) in
      let a = Nativeint.(logand (add a 15n) (lognot 15n)) in
      (* Never on a page, so that no other device maps it by chance. *)
      let a = if Nativeint.rem a 4096n = 0n then Nativeint.add a 16n else a in
      Hashtbl.replace drv.blocks a (ba, n);
      drv.held <- drv.held + n;
      let host = if addressed then Some a else None in
      Some (Region.v ?host ~handle:a a n)
  in
  let free r =
    let a = Region.address r in
    drv.held <- drv.held - snd (Hashtbl.find drv.blocks a);
    drv.frees <- drv.frees + 1;
    Hashtbl.remove drv.blocks a
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
        (fun v ~timeout_ms:_ ->
          run_to v;
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
          memory = { alloc = alloc ~addressed:false; free };
          host_memory = { alloc = alloc ~addressed:true; free };
          mapping;
          queue;
        }
    else
      Host_visible
        {
          memory = { alloc = alloc ~addressed:true; free };
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
    Driver.device ~name ~arch:"test" ~budget ~completion ?load ?peer
      ?synchronized ?finalize ?resolve ?room memory
  in
  Option.iter (Nx_device.set_timeout dev) timeout_ms;
  { dev; drv }

let far ?(name = "FAR") ?budget ?clock ?peer () =
  fake ~name ?budget ~far:true ?clock ?peer ()

(* A signal whose waits answer [wait timeout_ms]. *)
let signal ?(signaled = 0) wait =
  {
    Driver.signaled = (fun () -> signaled);
    wait = (fun _ ~timeout_ms -> wait timeout_ms);
  }

(* A signal that never arrives, counting the waits for it. *)
let never waits =
  signal (fun _ ->
      incr waits;
      false)

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

  type memory = { cells : int array; on_page : bool }

  (* The buffers over some memory, the bytes owned, and for a borrow the host
     memory it maps. *)
  type holding = { owned : int; mutable holders : int; maps : memory option }

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

  let mappings d =
    let add seen h =
      match h.maps with
      | Some m when h.holders > 0 && not (List.memq m seen) -> m :: seen
      | _ -> seen
    in
    List.length (List.fold_left add [] d.borrows)

  let name r = match r.device with None -> "CPU" | Some d -> d.name
  let alive r = if r.dropped then raise Dropped

  let fresh ?(on_page = false) ?device ?(owned = 0) ~addressed s n =
    if n < 0 || too_big s n then invalid_arg "create";
    let holding = { owned; holders = 1; maps = None } in
    Option.iter (fun d -> d.owns <- holding :: d.owns) device;
    let memory = { cells = Array.make (nbytes s n) (-1); on_page } in
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

  (* A far device's memory is addressed by the host when it is host memory. *)
  let create d pinned s n =
    let bytes = if n < 0 || too_big s n then 0 else nbytes s n in
    if bytes > 0 && live d + bytes > d.budget then raise No_memory;
    fresh ~device:d ~owned:bytes ~addressed:(d.name = "NEAR" || pinned) s n

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
    let holding = { owned = 0; holders = 1; maps } in
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
    String.iteri (fun i c -> r.memory.cells.(r.off + i) <- Char.code c) s

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
    let cells = Array.sub src.memory.cells src.off n in
    Array.blit cells 0 dst.memory.cells dst.off n;
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
    let rec from i =
      i = size r
      ||
      let c = r.memory.cells.(r.off + i) in
      (c < 0 || Char.code ba.{i} = c) && from (i + 1)
    in
    from 0

  (* [s] with the bytes of [r] that nothing wrote masked. *)
  let masked r s =
    String.mapi (fun i c -> if r.memory.cells.(r.off + i) < 0 then '?' else c) s

  let contents r =
    String.init (size r) (fun i ->
        let c = r.memory.cells.(r.off + i) in
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
  if cached > 0 then
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
    equal ~msg:"device" string (Model.name r) (Nx_device.name (B.device b));
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
let seldom =
  Gen.frequency
    [ (3, Gen.constant ~pp:Format.pp_print_bool false); (1, Gen.bool) ]

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
    (S.[ Bool; UInt8; Int8; Int4; UInt4; Int16; UInt16; Int32; UInt32 ]
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
      (dev ^-> seldom @-> formats @-> counts @-> makes buf)
      Model.create
      (fun d pinned s n ->
        created d.fake.dev s n (fun () -> B.create ~pinned d.fake.dev s n));
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

let memory =
  group "memory"
    [
      stateful ~count:300 ~steps:50
        "buffers hold what was written through them and their views, and \
         devices hold their live buffers and cache within their budget and \
         count what is copied (nx_device.mli is silent on the alignment of a \
         new buffer: 16 bytes assumed)"
        commands;
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
          Nx_device.set_budget host (allocated host + 1000);
          let a = B.create host S.UInt8 600 in
          raises_match (out_of_memory host 600) (fun () ->
              B.create host S.UInt8 600);
          ignore (Sys.opaque_identity a));
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

(* Buffers *)

let near = fake ()
let far_one = far ()

(* A buffer of [n] elements of [s], [k] elements into a larger one on the host,
   a device the host addresses, one it does not, or that one's host memory. *)
let placed =
  let place =
    Gen.of_list
      ~pp:(fun ppf (d, pinned) ->
        Format.fprintf ppf "on %a%s" pp_device d
          (if pinned then "'s host memory" else ""))
      [
        (host, false);
        (near.dev, false);
        (far_one.dev, false);
        (far_one.dev, true);
      ]
  in
  Gen.quad place formats (ints [ 0; 1; 2; 3; 17; 1000 ]) (ints [ 0; 1; 3 ])

let inside ((d, pinned), s, n, k) =
  let o = k * Model.element s in
  B.view (B.create ~pinned d S.UInt8 (o + Model.nbytes s n + 5)) ~offset:o s n

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

let test_borrow_lifetime () =
  let collected = ref false and during = ref true and armed = ref false in
  let wait _ =
    if !armed then begin
      armed := false;
      Gc.full_major ();
      Gc.full_major ();
      during := !collected
    end;
    true
  in
  let g = fake ~maps:true ~signal:(signal ~signaled:max_int wait) () in
  (fun () ->
    let hb = B.create host S.UInt8 (1 lsl 20) in
    Gc.finalise_last (fun () -> collected := true) hb;
    let bm = borrow g.dev hb in
    ignore (submit g.dev ~touches:[ bm ] Fun.id);
    ignore (Sys.opaque_identity bm))
    ();
  Gc.full_major ();
  armed := true;
  ignore (stats g.dev);
  is_false ~msg:"collected during the wait" !during;
  Gc.full_major ();
  Gc.full_major ();
  is_true ~msg:"collected after it" !collected

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
  let pinned f () = B.create ~pinned:true f.dev S.UInt8 8 in
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
    String.init n (fun i ->
        Char.chr (((i * 7) + (i lsr 16) + (i / slot * 13)) land 0xff))
  in
  let differing s =
    let d = ref 0 in
    String.iteri (fun i c -> if c <> bytes.[i] then incr d) s;
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
  equal ~msg:"the file now at its path" string "new" (read (of_file path));
  if not Sys.win32 then begin
    let descriptors () = Array.length (Sys.readdir "/dev/fd") in
    Gc.full_major ();
    Nx_device.synchronize disk;
    let before = descriptors () in
    let opened = List.init 20 (fun _ -> of_file path) in
    equal ~msg:"open" int (before + 20) (descriptors ());
    ignore (Sys.opaque_identity opened);
    Gc.full_major ();
    Nx_device.synchronize disk;
    equal ~msg:"closed" int before (descriptors ())
  end

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
      test
        "a file is read through the descriptor it was opened with, which \
         closes once its buffers are collected"
        test_file_closes;
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
    Ok r
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
  | Error why -> contains ~msg:"no peer" ~sub:"cannot address OWNER memory" why);
  let refusing = far ~peer:(fun _ _ -> Error "no route") () in
  (match B.borrow refusing.dev b with
  | Ok _ -> fail "borrowed a refused region"
  | Error why -> contains ~msg:"the driver's reason" ~sub:"no route" why);
  ignore (Sys.opaque_identity (on_a, again))

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
          let c = B.consume ~why:"taken" b in
          let dead f =
            raises (Invalid_argument "taken") (fun () -> ignore (f ()))
          in
          dead (fun () -> B.address b);
          dead (fun () -> B.address before);
          dead (fun () -> B.bigarray Bigarray.char b);
          dead (fun () -> B.view b ~offset:0 S.UInt8 1);
          dead (fun () -> B.copy ~src:before ~dst:(B.create host S.UInt8 4));
          dead (fun () -> B.copy ~src:(of_string "abcdefgh") ~dst:b);
          dead (fun () -> B.consume ~why:"again" b);
          equal (pair string string) ("abcdefgh", "cdef")
            (read c, read (B.view c ~offset:2 S.UInt8 4));
          equal bool true (B.is_borrowed c));
      test
        "a buffer consumed twice: each dead handle names the consumption that \
         killed it, and the memory stays owned" (fun () ->
          let b = B.create host S.Float32 4 in
          let c = B.consume ~why:"first" b in
          let d = B.consume ~why:"second" c in
          let dead why b =
            raises (Invalid_argument why) (fun () -> ignore (B.address b))
          in
          dead "first" b;
          dead "second" c;
          equal (pair int bool) (4, false) (B.length d, B.is_borrowed d));
      test "only a buffer that spans its memory can be consumed" (fun () ->
          let b = B.create host S.UInt8 8 in
          let window = B.view b ~offset:0 S.UInt8 4 in
          equal (pair bool bool) (true, false) (B.spans b, B.spans window);
          raises_match Exn.invalid_arg (fun () ->
              ignore (B.consume ~why:"window" window));
          equal bool true (B.spans (B.consume ~why:"whole" b)));
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
        "a borrow keeps its host memory until its device's work is done, \
         although the collector runs during the wait"
        test_borrow_lifetime;
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
  let rejecting () =
    (fake ~load:(fun ~binary:_ ~entry:_ -> Error "rejected") ()).dev
  in
  let raise_ ?(exn = Exn.invalid_arg ?substring:None) name f =
    (name, fun () -> raises_match exn (fun () -> ignore (f ())))
  in
  let error ~sub name f =
    ( name,
      fun () ->
        match f () with
        | Ok _ -> fail "accepted"
        | Error why -> contains ~msg:"the reason" ~sub why )
  in
  cases ~name:fst "refuse"
    [
      raise_ "a device with a negative budget" (fun () ->
          make ~budget:(-1) (Host_visible { memory; mapping = None }));
      raise_ "a device with a clock of 0 Hz" (fun () ->
          make (local ~clock:(Device_clock { hz = 0 }) Driver.host_memory));
      raise_ "host memory that the host does not address" (fun () ->
          make (local { memory with alloc = (fun _ -> Some unaddressed) }));
      raise_ ~exn:(Exn.failure ~substring:"timeline")
        "host memory with none for the timeline" (fun () -> make (local memory));
      raise_ "a timeout of 0 ms" (fun () -> Nx_device.set_timeout near.dev 0);
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

let programs =
  group "programs"
    [
      test "a function of a binary loads once, on its device" (fun () ->
          let loads = ref [] in
          let load ~binary ~entry =
            loads := (binary, entry) :: !loads;
            Ok (Nativeint.of_int (List.length !loads))
          in
          let d = (fake ~load ()).dev in
          let p = program d ~binary:"lib" ~name:"f" in
          is_true ~msg:"loaded again" (program d ~binary:"lib" ~name:"f" == p);
          ignore (program d ~binary:"lib" ~name:"g");
          equal
            (list (pair string string))
            [ ("lib", "g"); ("lib", "f") ]
            !loads;
          equal
            (triple string bool nativeint)
            ("f", true, 1n)
            Nx_device.Program.(name p, Nx_device.equal d (device p), handle p));
      test "a loader that faults loses its device" (fun () ->
          let load ~binary:_ ~entry:_ = failwith "context lost" in
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
        "the timeout starts at the default of 30 s, and set_timeout sets it \
         for the waits after it" (fun () ->
          let seen = ref 0 in
          let wait timeout_ms =
            seen := timeout_ms;
            true
          in
          let d = (fake ~signal:(signal wait) ()).dev in
          equal (pair int int) (30_000, 30_000)
            (Nx_device.timeout d, Driver.default_timeout);
          Nx_device.set_timeout d 7;
          submit d ignore;
          Nx_device.synchronize d;
          equal int 7 !seen);
      test
        "a wait without a signal restarts its timeout whenever the signal word \
         moves" (fun () ->
          let d = (fake ~name:"SLOW" ~timeout_ms:200 ()).dev in
          let word = B.address (Nx_device.signal_word d) in
          List.iter (fun _ -> ignore (submit d Fun.id)) [ 1; 2; 3 ];
          let progress =
            Domain.spawn (fun () ->
                for k = 1 to 3 do
                  Unix.sleepf 0.12;
                  store_signal word k
                done)
          in
          let t0 = Unix.gettimeofday () in
          Nx_device.synchronize d;
          greater ~msg:"seconds waited" (float 0.01) ~than:0.3
            (Unix.gettimeofday () -. t0);
          Domain.join progress;
          let stuck = (fake ~name:"STUCK" ~timeout_ms:200 ()).dev in
          ignore (submit stuck Fun.id);
          raises_match (lost stuck "hang detected") (fun () ->
              Nx_device.synchronize stuck));
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
        "a device whose queues stay full through its timeout is lost, and the \
         submission commits nothing" (fun () ->
          let ran = ref false in
          let d =
            (fake ~name:"FULL" ~timeout_ms:50 ~room:(fun () -> false) ()).dev
          in
          raises_match (lost d "no room in its queues") (fun () ->
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

let named = List.map (fun (d, v) -> (Nx_device.name d, v))

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
  let stuck = (fake ~name:"STUCK" ~timeout_ms:50 ()).dev in
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
  ignore (B.consume ~why:"consumed" b);
  refused ~msg:"a dead buffer" [ a ] [ b ];
  equal ~msg:"nothing submitted" int 0 (Nx_device.submitted a)

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
        "wait on the host for a value of a device they took, within its timeout"
        test_wait;
      test
        "give each device the value after its submitted one, and commit them \
         together"
        test_values;
      test "refuse no device, a host, the disk, and disk or dead buffers"
        test_submit_refusals;
    ]

(* Failures *)

let test_hang () =
  let waits = ref 0 in
  let d = (fake ~name:"HUNG" ~signal:(never waits) ()).dev in
  let b = B.create d S.UInt8 8 in
  ignore (submit d Fun.id);
  (match Nx_device.synchronize d with
  | () -> fail "synchronized a hung device"
  | exception e ->
      equal ~msg:"printed" string "HUNG: hang detected" (Printexc.to_string e));
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
  equal ~msg:"name and arch" (pair string string) ("HUNG", "test")
    (Nx_device.name d, Nx_device.arch d);
  equal ~msg:"budget, submitted, signaled" (triple int int int) (max_int, 1, 0)
    (Nx_device.budget d, Nx_device.submitted d, Nx_device.signaled d);
  equal ~msg:"its signal word" int 1 (B.length (Nx_device.signal_word d));
  Gc.full_major ();
  equal ~msg:"allocated for good" int 8 (allocated d)

let test_scope () =
  let waits = ref 0 in
  let gpu = (fake ~name:"GPU" ~maps:true ~signal:(never waits) ()).dev in
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
    ];
  equal ~msg:"a view of it" int 4 (B.length (B.view shared ~offset:0 S.UInt8 4));
  equal ~msg:"waits" int 1 !waits

(* A signal that arrives until [hung] is set. *)
let until hung = signal (fun _ -> not !hung)

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
  let f = fake ~name:"D" ~budget:1000 ~signal:(never (ref 0)) () in
  dropped (fun () -> B.create f.dev S.UInt8 600);
  ignore (submit f.dev Fun.id);
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
  Driver.device ~name:"FAULTY" ~arch:"test" ~budget:max_int
    ~peer:(fun _ r ->
      fault "peer";
      Ok r)
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
      test "a borrow unmapped before a loss is out of its reach" test_unmapped;
      test "a borrow whose unmapping a loss cut short stays in its reach"
        test_cut_short;
      test
        "a transfer that cannot be waited for leaves its destination in the \
         source's reach, retained"
        test_hung_transfer;
      test "memory that a hung wait could not free is retained" test_retained;
      test "a driver callback that raises Failure loses its device"
        test_faulting_callbacks;
      test "a free that faults midway retains the memory from it on"
        test_fault_midway;
    ]

(* Sleep and finalize *)

(* A device that sleeps on its interrupts sleeps only once its signal word has
   stayed still for 200 ms, and once more, briefly, before it is declared
   hung. *)
let test_sleep () =
  let sleeps = ref [] in
  let sleep ms = sleeps := ms :: !sleeps in
  let d = (fake ~name:"SLEEPY" ~sleep ()).dev in
  let v = submit d Fun.id in
  let word = B.address (Nx_device.signal_word d) in
  let late =
    Domain.spawn (fun () ->
        Unix.sleepf 0.7;
        store_signal word v)
  in
  Nx_device.synchronize d;
  Domain.join late;
  is_true ~msg:"slept while still" (List.length !sleeps >= 2);
  is_true ~msg:"for 200 ms each" (List.for_all (fun ms -> ms = 200) !sleeps);
  sleeps := [];
  let busy = (fake ~name:"BUSY" ~sleep ()).dev in
  let word = B.address (Nx_device.signal_word busy) in
  let v =
    List.fold_left (fun _ _ -> submit busy Fun.id) 0 (List.init 5 Fun.id)
  in
  let progress =
    Domain.spawn (fun () ->
        for k = 1 to v do
          Unix.sleepf 0.1;
          store_signal word k
        done)
  in
  Nx_device.synchronize busy;
  Domain.join progress;
  equal ~msg:"no sleep while the word moves" (list int) [] !sleeps;
  let still = (fake ~name:"STILL" ~sleep ~timeout_ms:500 ()).dev in
  ignore (submit still Fun.id);
  raises_match (lost still "hang detected") (fun () ->
      Nx_device.synchronize still);
  equal ~msg:"a last brief sleep before the hang" int 1 (List.hd !sleeps)

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
    (fake ~name:"BROKEN" ~timeout_ms:50 ~finalize:(say "BROKEN") ()).dev
  in
  ignore (submit broken Fun.id);
  (try Nx_device.synchronize broken with Nx_device.Lost _ -> ());
  let hanging =
    (fake ~name:"HANGING" ~timeout_ms:50 ~finalize:(say "HANGING") ()).dev
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
    ~sub:"HANGING synchronization failed" err;
  contains ~msg:"a raising finalize is reported" ~sub:"boom" err

let hooks =
  group "sleep and finalize"
    [
      test
        "a device sleeps on its interrupts once its signal word stays still, \
         and once more before a hang"
        test_sleep;
      test "a fault found asleep loses the device with the driver's message"
        (fun () ->
          let sleep _ = failwith "page fault at 0x1000" in
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
            on = Nx_device.name s.device;
            lane = s.lane;
            what = s.name;
            t0 = s.start;
            t1 = s.stop;
          }
    | P.Allocation _ | P.Load _ -> None)

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
  let d = (fake ~load:(fun ~binary:_ ~entry:_ -> Ok 42n) ()).dev in
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
  let d = (fake ~name:"HUNG" ~timeout_ms:50 ~signal:(never (ref 0)) ()).dev in
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
  let d =
    (fake ~name:"P" ~load:(fun ~binary:_ ~entry:_ -> Ok 0x1234n) ()).dev
  in
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
    (List.assoc (pid, 0) processes, List.assoc (pid, tid) threads)
  in
  let complete =
    List.map
      (fun e ->
        let key = (int_of_float (num "pid" e), int_of_float (num "tid" e)) in
        let lo = num "ts" e in
        (named key, str "name" e, lo, lo +. num "dur" e))
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
        if List.assoc (pid, 0) processes = "P" then
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
        "write Chrome's trace event format: named processes and threads, \
         escaped names, time order, nesting"
        test_output;
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
  let pinned = B.create ~pinned:true gpu S.UInt8 5000 in
  write pinned (pattern 3 5000);
  B.copy ~src:pinned ~dst:on;
  equal ~msg:"from its host memory" string (pattern 3 5000) (read on);
  let r = !(m.reads) in
  ignore (submit gpu Fun.id);
  (* The work never signals: the wait polls the far word until it times out. *)
  Nx_device.set_timeout gpu 50;
  raises_match (lost gpu "hang detected") (fun () -> Nx_device.synchronize gpu);
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
  let freed = ref 0 in
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
               free =
                 (fun r ->
                   incr freed;
                   Hashtbl.remove keep (Region.address r));
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
  let runs = ref 0 in
  Driver.on_free b (fun () -> incr runs);
  raises_match (Exn.invalid_arg ~substring:"allocated") (fun () ->
      Driver.on_free (B.create host S.UInt8 8) ignore);
  dropped (fun () -> b);
  ignore (stats d);
  equal ~msg:"cached memory stays mapped" int 0 !runs;
  Nx_device.free_cache d;
  equal ~msg:"unmapped before it is freed" (pair int int) (1, 1) (!runs, !freed);
  let b = B.create d S.UInt8 4096 in
  Driver.on_free b (fun () -> failwith "the mapper is gone");
  dropped (fun () -> b);
  Nx_device.free_cache d;
  equal ~msg:"memory a mapper could not unmap is retained" (pair int int)
    (4096, 1)
    (Nx_device.Stats.retained (stats d), !freed)

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

let machines =
  group "machines"
    [
      test "devices of another machine and their host" test_machines;
      test "another machine's host copies through its io" test_host_copies;
      test "a GPU of another machine" test_remote_gpu;
      test "copies between three machines, in chunks" test_between_machines;
      test "a machine that goes down loses its devices alone" test_machine_down;
      test "a host without memory for its staging raises Out_of_memory"
        test_no_staging;
      test "links carry copies between machines" test_links;
      test "memory described to other functions, and unmapped at free" test_dma;
      test "programs of another machine's host" test_remote_programs;
    ]

let () =
  if Sys.getenv_opt "NX_DEVICE_FINALIZE_CHILD" = Some "1" then finalize_child ();
  exit
    (run "nx.device"
       [
         devices;
         memory;
         laws;
         borrows;
         buffers;
         disks;
         refusals;
         timeline;
         submissions;
         failures;
         hooks;
         profiles;
         machines;
       ])
