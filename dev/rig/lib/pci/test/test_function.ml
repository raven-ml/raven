(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Rig_pci
open Rig_pci_support

let strf = Printf.sprintf
let config_size = 4096

exception In_use

(* A fake machine

   A transport's machine that keeps the contract of each operation as this
   machine's does, and records each call it gets. A call that breaks an
   operation's contract is misuse the library had to refuse before asking the
   machine: the fake records it in [wrong] and fails. Each BAR range, DMA
   allocation and pin is reached at a fresh address, as a server that maps them
   in its own process gives. Operations take a lock, since pins and DMA memory
   may be asked from any domain. *)

type fn_fake = {
  addressing : Machine.addressing;
  page : int;
  config : Bytes.t;
  mutable maps : (int * int) list; (* live BAR windows, (address, length) *)
  mutable ways : (int * bool) list; (* their BARs and [combine], a multiset *)
  mutable dmas : (int * int) list;
  mutable pins : (int * int) list; (* a multiset *)
  mutable released : bool;
  mutable before_pin : unit -> unit; (* runs before a pin reaches the fake *)
  mutable calls : string list; (* newest first *)
  mutable wrong : string list;
  lock : Mutex.t;
}

and machine_fake = {
  far : int; (* its C transport *)
  tr : Window.transport;
  buses : string list;
  m_page : int;
  m_addressing : Machine.addressing;
  mutable held : string list;
  mutable taken : fn_fake list; (* newest first *)
  mutable next : int;
  m_lock : Mutex.t;
}

let transport () = Window.unsafe_transport (far 0 4096)

(* BAR 0 is 64-bit, so index 1 is its upper half. *)
let bars = [| Some (0xe000_0000, 64 * 1024); None; Some (0xf000_0000, 4096) |]
let bar_of i = if i >= 0 && i < Array.length bars then bars.(i) else None

let fresh m n =
  Mutex.protect m.m_lock @@ fun () ->
  let a = m.next in
  m.next <- m.next + round_up (max n 1) (2 * mib);
  a

(* A run per page, or one. *)
let runs f a n ~one =
  let n = round_up n f.page in
  if one || f.addressing = Iommu then [ (0x1_0000_0000 + a, n) ]
  else List.init (n / f.page) (fun i -> (a + (i * f.page), f.page))

let disjoint (a, n) (b, k) = a + n <= b || b + k <= a

let remove x l =
  let rec go = function
    | [] -> None
    | y :: l when y = x -> Some l
    | y :: l -> Option.map (fun l -> y :: l) (go l)
  in
  go l

let fake_fn m bus =
  let f =
    {
      addressing = m.m_addressing;
      page = m.m_page;
      config = Bytes.init config_size (fun i -> Char.chr (i land 0xff));
      maps = [];
      ways = [];
      dmas = [];
      pins = [];
      released = false;
      before_pin = ignore;
      calls = [];
      wrong = [];
      lock = Mutex.create ();
    }
  in
  let call ?(after_release = false) name check run =
    Mutex.protect f.lock @@ fun () ->
    f.calls <- name :: f.calls;
    let misuse =
      if f.released && not after_release then Some "after release" else check ()
    in
    match misuse with
    | Some why ->
        f.wrong <- strf "%s: %s" name why :: f.wrong;
        failwith ("misuse reached the machine: " ^ name)
    | None -> run ()
  in
  let in_config off n =
    if off < 0 || off > config_size - n then Some "offset" else None
  in
  let config n off =
    call
      (strf "config%d" (8 * n))
      (fun () -> in_config off n)
      (fun () ->
        let x = ref 0 in
        for i = n - 1 downto 0 do
          x := (!x lsl 8) lor Bytes.get_uint8 f.config (off + i)
        done;
        !x)
  in
  let set_config n off x =
    call
      (strf "set_config%d" (8 * n))
      (fun () -> in_config off n)
      (fun () ->
        for i = 0 to n - 1 do
          Bytes.set_uint8 f.config (off + i) ((x lsr (8 * i)) land 0xff)
        done)
  in
  let window w = (Window.address w, Window.length w) in
  let ops =
    {
      Machine.addressing = f.addressing;
      config8 = config 1;
      config16 = config 2;
      config32 = config 4;
      set_config8 = set_config 1;
      set_config16 = set_config 2;
      set_config32 = set_config 4;
      bar =
        (fun i ->
          call "bar"
            (fun () -> if i < 0 then Some "index" else None)
            (fun () -> bar_of i));
      map =
        (fun ~combine i off n ->
          call "map"
            (fun () ->
              match bar_of i with
              | _ when List.mem (i, not combine) f.ways ->
                  Some "a BAR mapped the other way"
              | Some (_, size) when off >= 0 && n >= 0 && off <= size - n ->
                  None
              | _ -> Some "bytes outside the BAR")
            (fun () ->
              let a = fresh m n in
              f.maps <- (a, n) :: f.maps;
              f.ways <- (i, combine) :: f.ways;
              Ok (Window.through m.tr a n)));
      unmap =
        (fun w ->
          call "unmap"
            (fun () ->
              if List.mem (window w) f.maps then None else Some "no window")
            (fun () ->
              let rec drop k = function
                | x :: l when x = window w -> (k, l)
                | x :: l ->
                    let k, l = drop (k + 1) l in
                    (k, x :: l)
                | [] -> (k, [])
              in
              let k, maps = drop 0 f.maps in
              f.maps <- maps;
              f.ways <- List.filteri (fun j _ -> j <> k) f.ways));
      interrupt = (fun _ -> call "interrupt" (fun () -> None) (fun () -> false));
      reset = (fun () -> call "reset" (fun () -> None) (fun () -> Ok ()));
      alloc_dma =
        (fun ~contiguous ~va n ->
          let bytes = round_up n f.page in
          call "alloc_dma"
            (fun () ->
              match va with
              | _ when n mod f.page <> 0 -> Some "a size off the page"
              | Some v when v mod f.page <> 0 -> Some "va off a page"
              | _ when contiguous && bytes > 2 * mib -> Some "too large"
              | Some v
                when contiguous && f.addressing = Physical && bytes > f.page
                     && v mod (2 * mib) <> 0 ->
                  Some "va off 2 MiB"
              | _ -> None)
            (fun () ->
              let a = match va with Some v -> v | None -> fresh m bytes in
              let used = List.concat_map (fun g -> g.dmas) m.taken in
              if List.exists (fun r -> not (disjoint r (a, bytes))) used then
                Error "far:1: the addresses are in use"
              else begin
                f.dmas <- (a, bytes) :: f.dmas;
                Ok
                  (Some
                     ( Window.through m.tr a bytes,
                       runs f a bytes ~one:contiguous ))
              end));
      free_dma =
        (fun w ->
          call ~after_release:true "free_dma"
            (fun () ->
              if List.mem (window w) f.dmas then None else Some "no memory")
            (fun () -> f.dmas <- Option.get (remove (window w) f.dmas)));
      pin =
        (fun a n ->
          f.before_pin ();
          call "pin"
            (fun () -> if a mod f.page <> 0 then Some "off a page" else None)
            (fun () ->
              f.pins <- (a, n) :: f.pins;
              Ok (runs f a n ~one:false)));
      unpin =
        (fun a n ->
          call ~after_release:true "unpin"
            (fun () ->
              if List.mem (a, n) f.pins then None else Some "not pinned")
            (fun () -> f.pins <- Option.get (remove (a, n) f.pins)));
      release =
        (fun () ->
          call ~after_release:true "release"
            (fun () -> if f.released then Some "released twice" else None)
            (fun () ->
              if not f.released then begin
                f.released <- true;
                f.maps <- [];
                f.ways <- [];
                Mutex.protect m.m_lock (fun () ->
                    m.held <- List.filter (( <> ) bus) m.held)
              end));
    }
  in
  (f, ops)

let bus1 = "0000:01:00.0"
let bus2 = "0000:02:00.0"
let absent = "0000:03:00.0"

(* Each machine reaches its BAR ranges and memory at its own 4 GiB, below the
   addresses DMA memory is asked at, so that no two machines' windows are equal;
   [test_other_machine] makes them equal. *)
let machines = Atomic.make 0
let next_base () = (1 + (Atomic.fetch_and_add machines 1 mod 0x7000)) lsl 32

(* Each machine reserves [reserved] bytes from [va_base], where DMA memory is
   asked. *)
let va_base = 0x7f00_0000_0000
let reserved = 8 * mib

let fake_machine ?(base = next_base ()) ?(page = 4096)
    ?(addressing = Machine.Physical) () =
  let far = far 0 4096 in
  let m =
    {
      far;
      tr = Window.unsafe_transport far;
      buses = [ bus1; bus2 ];
      m_page = page;
      m_addressing = addressing;
      held = [];
      taken = [];
      next = base;
      m_lock = Mutex.create ();
    }
  in
  let take bus =
    Mutex.protect m.m_lock @@ fun () ->
    if not (List.mem bus m.buses) then
      Error (bus ^ " is no PCI function of far:1")
    else if List.mem bus m.held then Error (bus ^ " is held")
    else begin
      let f, ops = fake_fn m bus in
      m.held <- bus :: m.held;
      m.taken <- f :: m.taken;
      Ok ops
    end
  in
  let machine =
    Machine.make ~name:"far:1"
      {
        transport = m.tr;
        page;
        functions = (fun () -> []);
        take;
        reserve = (fun ~base:_ _ -> Ok ());
      }
  in
  require_ok (Machine.reserve machine ~base:va_base reserved);
  (machine, m)

let take_fake ?base ?page ?addressing () =
  let machine, m = fake_machine ?base ?page ?addressing () in
  let f = Result.get_ok (Function.take machine bus1) in
  (machine, f, List.hd m.taken)

(* The requests of a fake machine, which refuses none within its contract. *)
external c_combines : Window.t -> bool = "rig_pci_test_combines"

let map ?combine ?off ?length f i =
  require_ok (Function.map ?combine ?off ?length f i)

let alloc_dma ?contiguous ?va f n =
  require_some (require_ok (Function.alloc_dma ?contiguous ?va f n))

let pin f a n = require_ok (Function.pin f a n)

let addressing =
  Testable.make
    ~pp:(fun ppf a ->
      Format.pp_print_string ppf
        (match a with Machine.Physical -> "Physical" | Iommu -> "Iommu"))
    ~equal:( = )

let runs_w = list (pair int int)

(* Taking *)

let test_take_asks () =
  let calls = ref [] in
  let machine =
    Machine.make ~name:"far:1"
      {
        transport = transport ();
        page = 4096;
        functions = (fun () -> []);
        take =
          (fun bus ->
            calls := bus :: !calls;
            Error "far:1: run `driverctl set-override 0000:01:00.0 vfio-pci`");
        reserve = (fun ~base:_ _ -> Ok ());
      }
  in
  equal ~msg:"the machine's refusal" (result pass string)
    (Error "far:1: run `driverctl set-override 0000:01:00.0 vfio-pci`")
    (Function.take machine bus1);
  equal ~msg:"what the machine was asked" (list string) [ bus1 ] !calls

let test_taken () =
  List.iter
    (fun a ->
      let machine, f, fake = take_fake ~addressing:a () in
      equal ~msg:"machine" (option string) (Machine.name machine)
        (Machine.name (Function.machine f));
      equal ~msg:"bus" string bus1 (Function.bus f);
      equal ~msg:"addressing" addressing a (Function.addressing f);
      equal ~msg:"released" bool false (Function.released f);
      equal ~msg:"what the machine was asked" (list string) [] fake.calls)
    [ Machine.Physical; Iommu ]

(* A refused take holds nothing, and a release lets the function be taken
   again. *)
let test_refused () =
  let machine, m = fake_machine () in
  equal ~msg:"absent" bool true (Result.is_error (Function.take machine absent));
  let f = Result.get_ok (Function.take machine bus1) in
  equal ~msg:"held" (result pass string)
    (Error (bus1 ^ " is held"))
    (Function.take machine bus1);
  Function.release f;
  let g = Result.get_ok (Function.take machine bus1) in
  equal ~msg:"taken again once released" string bus1 (Function.bus g);
  equal ~msg:"functions made" int 2 (List.length m.taken)

(* A bus that is no bus address reaches no file, here or on another machine:
   these name sysfs's directory, its parent, or a path through it. *)
let not_buses =
  [
    "";
    ".";
    "..";
    "0000:01:00.0/..";
    "../../../etc";
    "0000:01:00.0\000";
    "0000:01:00";
  ]

let test_take_no_bus () =
  let calls = ref [] in
  let machine =
    Machine.make ~name:"far:1"
      {
        transport = transport ();
        page = 4096;
        functions = (fun () -> []);
        take =
          (fun bus ->
            calls := bus :: !calls;
            Error "asked");
        reserve = (fun ~base:_ _ -> Ok ());
      }
  in
  List.iter
    (fun bus ->
      equal ~msg:(String.escaped bus) (result pass string)
        (Error (strf "%S is no PCI bus address, expected DDDD:BB:DD.F" bus))
        (Function.take machine bus))
    not_buses;
  equal ~msg:"what the machine was asked" (list string) [] !calls

let test_take_failed () =
  let machine, m = fake_machine () in
  break m.far;
  equal ~msg:"a failed machine" (result pass string)
    (Error "far: the link broke")
    (Function.take machine bus1);
  equal ~msg:"functions made" int 0 (List.length m.taken)

(* After release, only free_dma and unpin reach the machine. *)
let test_released () =
  let _, f, fake = take_fake () in
  let d, _ = alloc_dma f 4096 in
  ignore (pin f 0x5000_0000 4096 : (int * int) list);
  Function.release f;
  List.iter
    (fun (name, use) ->
      raises_match ~msg:name (Exn.invalid_arg ~substring:"released") use)
    [
      ("failed", fun () -> ignore (Function.failed f : string option));
      ("config16", fun () -> ignore (Function.config16 f 0 : int));
      ("set_config16", fun () -> Function.set_config16 f 0 0);
      ("bar", fun () -> ignore (Function.bar f 0 : (int * int) option));
      ("map", fun () -> ignore (Function.map f 0 : _ result));
      ("interrupt", fun () -> ignore (Function.interrupt f 0 : bool));
      ("reset", fun () -> ignore (Function.reset f : _ result));
      ("alloc_dma", fun () -> ignore (Function.alloc_dma f 4096 : _ result));
      ("pin", fun () -> ignore (Function.pin f 0x5000_0000 4096 : _ result));
    ];
  Function.free_dma f d;
  Function.unpin f 0x5000_0000 4096;
  Function.release f;
  equal ~msg:"what the machine was asked" (list string)
    [ "unpin"; "free_dma"; "release"; "pin"; "alloc_dma" ]
    fake.calls

let taking =
  group ~timeout:patience "taking"
    [
      test "a take asks the machine for its bus, and keeps its refusal"
        test_take_asks;
      test "a taken function is on its machine at its bus, as the machine says"
        test_taken;
      test "a function another holder has is refused until released"
        test_refused;
      test "a string that is no bus address is refused before the machine"
        test_take_no_bus;
      test "a failed machine refuses a take" test_take_failed;
      test "a released function refuses all but free_dma and unpin"
        test_released;
    ]

(* Pass-through: a function's windows and runs are its machine's. *)
let test_machine_values () =
  let _, f, fake = take_fake ~page:4096 () in
  let w = map f 0 ~off:256 ~length:512 in
  equal ~msg:"a BAR window" (pair int int) (List.hd fake.maps)
    (Window.address w, Window.length w);
  equal ~msg:"its BAR" (option (pair int int)) bars.(0) (Function.bar f 0);
  let d, runs = alloc_dma f (3 * 4096) in
  equal ~msg:"DMA memory" (pair int int) (List.hd fake.dmas)
    (Window.address d, Window.length d);
  equal ~msg:"its runs" runs_w
    (List.init 3 (fun i -> (Window.address d + (i * 4096), 4096)))
    runs;
  equal ~msg:"pinned runs" runs_w
    [ (0x7000_0000, 4096); (0x7000_1000, 4096) ]
    (pin f 0x7000_0000 8192)

let test_defaults () =
  let _, f, fake = take_fake () in
  let size = 64 * 1024 in
  let lengths =
    List.map Window.length [ map f 0; map f 0 ~off:256; map f 0 ~length:16 ]
  in
  equal ~msg:"lengths" (list int) [ size; size - 256; 16 ] lengths;
  equal ~msg:"windows the machine mapped" int 3 (List.length fake.maps)

(* Two machines at the same address give equal windows but for their
   transports. *)
let test_other_machine () =
  let _, f, fake = take_fake ~base:(1 lsl 40) () in
  let _, g, _ = take_fake ~base:(1 lsl 40) () in
  let w, _ = alloc_dma f 4096 in
  let w', _ = alloc_dma g 4096 in
  equal ~msg:"one address" int (Window.address w) (Window.address w');
  raises_match ~msg:"DMA memory" (Exn.invalid_arg ~substring:"") (fun () ->
      Function.free_dma f w');
  let b = map f 0 and b' = map g 0 in
  equal ~msg:"one address" int (Window.address b) (Window.address b');
  raises_match ~msg:"a BAR window" (Exn.invalid_arg ~substring:"") (fun () ->
      Function.unmap f b');
  equal ~msg:"its memory and window stay" (pair int int) (1, 1)
    (List.length fake.dmas, List.length fake.maps)

(* Bytes [0, 0) at [size] lie in the BAR, as a window's [sub] takes them. *)
let test_empty () =
  let _, f, _ = take_fake () in
  let size = 64 * 1024 in
  equal ~msg:"at the start" int 0 (Window.length (map f 0 ~length:0));
  equal ~msg:"at the end" int 0 (Window.length (map f 0 ~off:size))

(* After the reset, the function's vendor ID is read until it answers. *)
let test_reset () =
  let _, f, fake = take_fake () in
  require_ok (Function.reset f);
  equal ~msg:"asked" (list string) [ "config16"; "reset" ] fake.calls

(* A vendor ID of all ones is a function that does not answer. *)
let test_reset_silent () =
  let _, f, _ = take_fake () in
  Function.set_config16 f 0 0xffff;
  equal (result unit string)
    (Error (bus1 ^ " does not answer 1000 ms after its reset"))
    (Function.reset f)

(* A pin from another domain runs while the owner releases the function: the
   release waits for it, so no pin reaches the machine after the release, and a
   pin that starts after the release begins is refused. The pin is held inside
   the function, before the machine; the release is sampled while it waits. *)
let test_release_waits_pin () =
  let _, f, fake = take_fake () in
  let inside = Atomic.make false and go = Atomic.make false in
  fake.before_pin <-
    (fun () ->
      Atomic.set inside true;
      ignore (poll (fun () -> Atomic.get go)));
  let pinning = Domain.spawn (fun () -> Function.pin f 0 4096) in
  equal ~msg:"the pin is inside" bool true (poll (fun () -> Atomic.get inside));
  fake.before_pin <- ignore;
  let releasing = Domain.spawn (fun () -> Function.release f) in
  Unix.sleepf 0.05;
  equal ~msg:"the release waits for the pin" bool false
    (Mutex.protect fake.lock (fun () -> fake.released));
  Atomic.set go true;
  ignore (require_ok (Domain.join pinning));
  Domain.join releasing;
  equal ~msg:"misuse that reached the machine" (list string) []
    (Mutex.protect fake.lock (fun () -> fake.wrong));
  equal ~msg:"released" bool true fake.released;
  raises_match (Exn.invalid_arg ~substring:"released") (fun () ->
      Function.pin f 0 4096)

let uses =
  group ~timeout:patience "uses"
    [
      test "a function's windows, BARs and runs are its machine's"
        test_machine_values;
      test
        "a release waits for a pin from another domain, and refuses later ones \
         (sampled)"
        test_release_waits_pin;
      test "a BAR window is the rest of the BAR from its offset by default"
        test_defaults;
      test "a reset asks the machine, then waits for the function to answer"
        test_reset;
      test "a function that does not answer after its reset fails"
        test_reset_silent;
      test "a window of no bytes inside a BAR is mapped" test_empty;
      test "another machine's window at the same address is refused"
        test_other_machine;
    ]

(* Misuse at the bounds, each refused before the machine is asked. *)
let refused name ?page ?addressing use =
  test name (fun () ->
      let _, f, fake = take_fake ?page ?addressing () in
      raises_match (Exn.invalid_arg ~substring:"") (fun () -> use f);
      equal ~msg:"what the machine was asked" (list string) [] fake.calls)

let misuse_refused =
  group ~timeout:patience "misuse is refused before the machine is asked"
    [
      refused "a BAR index below zero" (fun f -> Function.bar f (-1));
      refused "the least BAR index" (fun f -> Function.bar f min_int);
      refused "a map of a BAR index below zero" (fun f -> map f (-1));
      refused "a pin off a page" ~page:16384 (fun f ->
          Function.pin f (0x5000_0000 + 4096) 4096);
      refused "DMA memory at an address off a page" ~page:16384 (fun f ->
          Function.alloc_dma ~va:(va_base + 4096) f 16384);
      refused "contiguous DMA memory above 2 MiB" (fun f ->
          Function.alloc_dma ~contiguous:true f ((2 * mib) + 1));
      refused "a huge page at an address off 2 MiB" ~page:16384
        ~addressing:Physical (fun f ->
          Function.alloc_dma ~contiguous:true ~va:(va_base + 16384) f 32768);
      refused "DMA memory of no bytes" (fun f -> alloc_dma f 0);
      refused "DMA memory of more bytes than an int holds" (fun f ->
          alloc_dma f max_int);
      refused "a pin of no bytes" (fun f -> Function.pin f 0x5000_0000 0);
      refused "configuration space below its first byte" (fun f ->
          Function.config8 f (-1));
      refused "configuration space past its 4096 bytes" (fun f ->
          Function.set_config32 f 4094 0);
      refused "an interrupt wait below zero" (fun f ->
          Function.interrupt f (-1));
      refused "DMA memory at an address no reservation holds" (fun f ->
          Function.alloc_dma ~va:(va_base + reserved) f 4096);
      refused "DMA memory that ends past its reservation" (fun f ->
          Function.alloc_dma ~va:(va_base + reserved - 4096) f 8192);
    ]

(* Sequences against a model

   The reference is a model of what the interface promises: which windows are
   live, which ranges are pinned, which functions are held. The fake machine
   checks the other side: what reached it is never misuse, and its live windows,
   memory and pins are the model's. *)

type m_ref = {
  r_page : int;
  r_addressing : Machine.addressing;
  mutable r_held : string list;
  mutable r_fns : f_ref list;
}

and f_ref = {
  rm : m_ref;
  r_bus : string;
  r_config : Bytes.t;
  mutable r_released : bool;
  mutable r_maps : w_ref list;
  mutable r_dmas : w_ref list;
  mutable r_pins : (int * int) list;
}

(* [at] is the address asked for DMA memory. *)
and w_ref = {
  owner : f_ref;
  kind : [ `Bar of int * bool | `Dma ];  (** A BAR's index and [combine]. *)
  len : int;
  at : int option;
}

let sorted l = List.sort_uniq compare l
let machine_t = abstract "m"

let fn_t =
  abstract "f" ~invariant:(fun r ((f : Function.t), (fake : fn_fake)) ->
      equal ~msg:"released" bool r.r_released (Function.released f);
      equal ~msg:"misuse that reached the machine" (list string) [] fake.wrong;
      equal ~msg:"the machine's BAR windows" (slist int compare)
        (List.map (fun w -> w.len) r.r_maps)
        (List.map snd fake.maps);
      equal ~msg:"the machine's DMA memory" (slist int compare)
        (List.map (fun w -> w.len) r.r_dmas)
        (List.map snd fake.dmas);
      if not r.r_released then
        equal ~msg:"the machine's pins" runs_w (sorted r.r_pins)
          (sorted fake.pins))

let win_t =
  abstract "w" ~invariant:(fun r w ->
      equal ~msg:"length" int r.len (Window.length w))

let pp_int ppf = Format.fprintf ppf "%#x"
let ints l = Gen.of_list ~pp:pp_int l

let pp_opt ppf = function
  | None -> Format.pp_print_string ppf "None"
  | Some x -> pp_int ppf x

let opt_ints l = Gen.of_list ~pp:pp_opt (None :: List.map Option.some l)

let make_ref (page, a) =
  { r_page = page; r_addressing = a; r_held = []; r_fns = [] }

let make_sys (page, a) = fake_machine ~page ~addressing:a ()

exception Refused

let take_ref m bus =
  if List.mem bus m.r_held || bus = absent then raise Refused;
  m.r_held <- bus :: m.r_held;
  let f =
    {
      rm = m;
      r_bus = bus;
      r_config = Bytes.init config_size (fun i -> Char.chr (i land 0xff));
      r_released = false;
      r_maps = [];
      r_dmas = [];
      r_pins = [];
    }
  in
  m.r_fns <- f :: m.r_fns;
  f

let take_sys ((machine : Machine.t), (m : machine_fake)) bus =
  match Function.take machine bus with
  | Ok f -> (f, List.hd m.taken)
  | Error _ -> raise Refused

let release_ref f =
  if not f.r_released then begin
    f.r_released <- true;
    f.r_maps <- [];
    f.rm.r_held <- List.filter (( <> ) f.r_bus) f.rm.r_held
  end

(* Each refusal is labelled, so a run that never reaches one fails. *)
let misuse label =
  cover label true;
  invalid_arg label

let live f = if f.r_released then misuse "a released function"

(* Without [combine] a window maps its BAR as the live windows of the BAR do,
   uncached where there is none. *)
let map_ref f combine i off len =
  live f;
  if i < 0 then misuse "a BAR index below zero";
  match bar_of i with
  | None -> misuse "no such BAR"
  | Some (_, size) ->
      let off = Option.value off ~default:0 in
      let len = Option.value len ~default:(size - off) in
      if off < 0 || len < 0 || off > size - len then
        misuse "bytes outside the BAR";
      let way c = List.exists (fun w -> w.kind = `Bar (i, c)) f.r_maps in
      let combine = match combine with Some c -> c | None -> way true in
      if way (not combine) then misuse "a BAR mapped the other way";
      let w = { owner = f; kind = `Bar (i, combine); len; at = None } in
      f.r_maps <- w :: f.r_maps;
      w

let map_sys (f, _) combine i off len = map ?combine ?off ?length:len f i
let without w l = List.filter (fun x -> x != w) l

(* Windows are values: a window equal to a live one, as DMA memory asked twice
   at one address of one machine gives, names it. *)
let live_one w l =
  List.find_opt
    (fun x ->
      x == w
      || w.at <> None && x.at = w.at && x.len = w.len
         && x.owner.rm == w.owner.rm)
    l

let unmap_ref f w =
  if not (w.owner == f && List.memq w f.r_maps) then
    misuse "a window that is not the function's live BAR window";
  f.r_maps <- without w f.r_maps

let alloc_at f contiguous va n =
  live f;
  let page = f.rm.r_page in
  let bytes = round_up n page in
  (match va with
  | Some v when v mod page <> 0 -> misuse "an address off a page"
  | _ when contiguous && bytes > 2 * mib -> misuse "contiguous above 2 MiB"
  | Some v
    when contiguous
         && f.rm.r_addressing = Physical
         && bytes > page
         && v mod (2 * mib) <> 0 ->
      misuse "a huge page off 2 MiB"
  | Some v ->
      let huge = contiguous && f.rm.r_addressing = Physical && bytes > page in
      let mapped = if huge then 2 * mib else bytes in
      if v + mapped > va_base + reserved then
        misuse "an address no reservation holds"
  | _ -> ());
  let used =
    List.concat_map
      (fun g ->
        List.filter_map
          (fun w -> Option.map (fun a -> (a, w.len)) w.at)
          g.r_dmas)
      f.rm.r_fns
  in
  (match va with
  | Some a when List.exists (fun r -> not (disjoint r (a, bytes))) used ->
      cover "addresses in use" true;
      raise In_use
  | _ -> ());
  let w = { owner = f; kind = `Dma; len = bytes; at = va } in
  f.r_dmas <- w :: f.r_dmas;
  w

(* Reserved addresses: the fake maps memory at any [va]. *)
let alloc_sys (f, _) contiguous va n =
  let va = Option.map (fun v -> va_base + v) va in
  match Function.alloc_dma ~contiguous ?va f n with
  | Ok (Some (w, _)) -> w
  | Ok None | Error _ -> raise In_use

let alloc_ref f contiguous va n =
  alloc_at f contiguous (Option.map (fun v -> va_base + v) va) n

let free_ref f w =
  match live_one w f.r_dmas with
  | None -> misuse "memory that is not the function's live DMA memory"
  | Some w -> f.r_dmas <- without w f.r_dmas

let pin_ref f a n =
  live f;
  if a mod f.rm.r_page <> 0 then misuse "a pin off a page";
  f.r_pins <- (a, n) :: f.r_pins

let unpin_ref f a n =
  match remove (a, n) f.r_pins with
  | None -> misuse "a range not pinned"
  | Some l ->
      cover "a range pinned twice, unpinned once" (List.mem (a, n) l);
      f.r_pins <- l

let pin_base = 0x5000_0000

let config_ref f off n =
  live f;
  let x = ref 0 in
  for i = n - 1 downto 0 do
    x := (!x lsl 8) lor Bytes.get_uint8 f.r_config (off + i)
  done;
  !x

let set_config_ref f off n x =
  live f;
  for i = 0 to n - 1 do
    Bytes.set_uint8 f.r_config (off + i) ((x lsr (8 * i)) land 0xff)
  done

(* The access of width [n] bytes. *)
let config_sys f off = function
  | 1 -> Function.config8 f off
  | 2 -> Function.config16 f off
  | _ -> Function.config32 f off

let set_config_sys f off n x =
  match n with
  | 1 -> Function.set_config8 f off x
  | 2 -> Function.set_config16 f off x
  | _ -> Function.set_config32 f off x

let bar_ref f i =
  live f;
  if i < 0 then misuse "a BAR index below zero";
  bar_of i

let page_gen = ints [ 4096; 16384 ]

let addressing_gen =
  Gen.of_list
    ~pp:(fun ppf a ->
      Format.pp_print_string ppf
        (match a with Machine.Physical -> "Physical" | Iommu -> "Iommu"))
    [ Machine.Physical; Iommu ]

let machine_cmd =
  command "machine"
    (Gen.pair page_gen addressing_gen @-> makes machine_t)
    make_ref make_sys

let take_cmd =
  command "take"
    (machine_t
    ^-> Gen.of_list ~pp:Format.pp_print_string [ bus1; bus2; absent ]
    @-> makes fn_t)
    take_ref take_sys

let fsys g (f, _) = g f

(* Inputs at each bound: empty windows, addresses off a 16 KiB page, contiguous
   memory above 2 MiB, huge pages off 2 MiB and memory past the reservation
   among them. *)
let size = 64 * 1024
let offs = [ -1; 0; 1; 16; size - 1; size; size + 1; max_int ]
let map_lens = [ -1; 0; 1; 16; 4096; size - 1; size + 1; max_int ]
let lens = [ 1; 4095; 4096; 4097; 16385; (2 * mib) - 1; 2 * mib; (2 * mib) + 1 ]
let vas = [ 0; 4096; 16384; 2 * mib; 4 * mib; reserved - 4096; reserved ]
let pin_addrs = ints (List.map (( + ) pin_base) [ 0; 4096; 16384 ])
let pin_lens = ints [ 1; 16384 ]
let pinned = among (pair int int) fn_t (fun f -> sorted f.r_pins)

let dma_cmds =
  [
    command "alloc_dma"
      (fn_t ^-> Gen.bool @-> opt_ints vas @-> ints lens @-> makes win_t)
      alloc_ref alloc_sys;
    command "free_dma"
      (fn_t ^-> win_t ^-> returns unit)
      free_ref (fsys Function.free_dma);
    command "pin"
      (fn_t ^-> pin_addrs @-> pin_lens @-> returns unit)
      pin_ref
      (fun (f, _) a n -> ignore (pin f a n : (int * int) list));
    command "unpin"
      (fn_t ^-> pin_addrs @-> pin_lens @-> returns unit)
      unpin_ref (fsys Function.unpin);
    command "unpin a pinned range"
      (fn_t ^-> pinned ^-> returns unit)
      (fun f (a, n) -> unpin_ref f a n)
      (fun (f, _) (a, n) -> Function.unpin f a n);
  ]

let commands =
  [
    machine_cmd;
    take_cmd;
    command "release"
      (fn_t ^-> returns unit)
      release_ref (fsys Function.release);
    command "released"
      (fn_t ^-> returns bool)
      (fun f -> f.r_released)
      (fsys Function.released);
    command "bar"
      (fn_t
      ^-> ints [ min_int; -1; 0; 1; 2; 5; 6 ]
      @-> returns (option (pair int int)))
      bar_ref (fsys Function.bar);
    command "map"
      (fn_t ^-> Gen.option Gen.bool
      @-> ints [ -1; 0; 1; 2; 6 ]
      @-> opt_ints offs @-> opt_ints map_lens @-> makes win_t)
      map_ref map_sys;
    command "unmap"
      (fn_t ^-> win_t ^-> returns unit)
      unmap_ref (fsys Function.unmap);
    command "pin a pinned range again, then unpin it"
      (fn_t ^-> pinned ^-> returns unit)
      (fun f (a, n) ->
        pin_ref f a n;
        unpin_ref f a n)
      (fun (f, _) (a, n) ->
        ignore (pin f a n : (int * int) list);
        Function.unpin f a n);
    command "config"
      (fn_t
      ^-> ints [ 0; 4; 60; 64; 256; 4092 ]
      @-> ints [ 1; 2; 4 ]
      @-> returns int)
      config_ref (fsys config_sys);
    command "set_config"
      (fn_t
      ^-> ints [ 0; 4; 64; 4092 ]
      @-> ints [ 1; 2; 4 ]
      @-> ints [ 0; 0xff; 0x1234; 0xdead_beef; -1; max_int; min_int ]
      @-> returns unit)
      set_config_ref (fsys set_config_sys);
  ]
  @ dma_cmds

let sequences =
  stateful "takes, uses and releases behave as the model" ~count:300 ~steps:30
    commands

(* Pins and DMA memory from two domains at once. A pin lost by a race shows when
   the suffix unpins it. The invariant runs only before the parallel calls, so
   misuse that reached the machine is read by a command. *)
let parallel =
  stateful "pins and DMA memory are counted the same from two domains"
    ~domains:2 ~count:100
    ([
       machine_cmd;
       take_cmd;
       command "misuse that reached the machine"
         (fn_t ^-> returns (list string))
         (fun _ -> [])
         (fun (_, fake) -> Mutex.protect fake.lock (fun () -> fake.wrong));
     ]
    @ dma_cmds)

let model = group ~timeout:patience "against a model" [ sequences; parallel ]

(* This machine *)

let sysfs bus file =
  Filename.concat (Filename.concat "/sys/bus/pci/devices" bus) file

(* What a take must not change: the function's driver and whether it is
   enabled. *)
let state bus =
  let driver =
    match Unix.readlink (sysfs bus "driver") with
    | l -> Some (Filename.basename l)
    | exception Unix.Unix_error _ -> None
  in
  let enabled =
    match
      In_channel.with_open_text (sysfs bus "enable") In_channel.input_all
    with
    | s -> Some (String.trim s)
    | exception Sys_error _ -> None
  in
  (driver, enabled)

let test_refused_here () =
  if on_linux then skip ~reason:"this machine has /sys/bus/pci" ();
  match Function.take Machine.this "0000:00:00.0" with
  | Ok _ -> fail "a function taken on a machine without PCI functions"
  | Error why -> contains ~msg:"names the bus" ~sub:"0000:00:00.0" why

(* Listing and taking change nothing on the machine. Only GPUs are taken:
   another class of function may be another user's device. *)
let test_changes_nothing () =
  if not on_linux then skip ~reason:"this machine has no /sys/bus/pci" ();
  hold_gpu ();
  let buses = List.map (fun (d : Machine.id) -> d.bus) (this_gpus ()) in
  let before = List.map state buses in
  List.iter
    (fun bus ->
      match Function.take Machine.this bus with
      | Ok f -> Function.release f
      | Error _ -> ())
    buses;
  let state_w = pair (option string) (option string) in
  equal
    (list (pair string state_w))
    (List.combine buses before)
    (List.combine buses (List.map state buses))

let takeable () =
  this_gpus ()
  |> List.find_map (fun (d : Machine.id) ->
      match Function.take Machine.this d.bus with
      | Ok f ->
          Function.release f;
          Some d
      | Error _ -> None)

(* One process holds a function at a time, this one included. *)
let test_held_here () =
  if not on_linux then skip ~reason:"this machine has no /sys/bus/pci" ();
  hold_gpu ();
  let d =
    match takeable () with
    | Some d -> d
    | None -> skip ~reason:"no function this process may take" ()
  in
  let f = Result.get_ok (Function.take Machine.this d.bus) in
  is_error ~msg:"held" (Function.take Machine.this d.bus);
  Function.release f;
  let g = Result.get_ok (Function.take Machine.this d.bus) in
  Function.release g

let vfio_function () =
  if not (Sys.file_exists "/dev/vfio") then skip ~reason:"no /dev/vfio" ();
  this_gpus ()
  |> List.find_map (fun (d : Machine.id) ->
      match Function.take Machine.this d.bus with
      | Ok f when Function.addressing f = Iommu -> Some (d, f)
      | Ok f ->
          Function.release f;
          None
      | Error _ -> None)
  |> function
  | Some x -> x
  | None -> skip ~reason:"no function this process may take behind an IOMMU" ()

(* Behind an IOMMU, a function needs no root. *)
let test_vfio () =
  hold_gpu ();
  let d, f = vfio_function () in
  Fun.protect ~finally:(fun () -> Function.release f) @@ fun () ->
  equal ~msg:"its vendor" int d.vendor (Function.config16 f 0);
  equal ~msg:"its device" int d.device (Function.config16 f 2);
  let page = Machine.page Machine.this in
  let allocs = List.map (fun n -> alloc_dma f n) [ 1; 3 * page; 2 * mib ] in
  List.iter
    (fun (w, runs) ->
      equal ~msg:"one run of its bytes" (list int)
        [ Window.length w ]
        (List.map snd runs))
    allocs;
  let runs = List.sort compare (List.concat_map snd allocs) in
  List.iter2
    (fun (a, n) (b, _) ->
      at_most ~msg:"device addresses apart" int ~than:b (a + n))
    (List.filteri (fun i _ -> i < List.length runs - 1) runs)
    (List.tl runs);
  let w, _ = List.hd allocs in
  let pinned = pin f (Window.address w) (Window.length w) in
  equal ~msg:"a pin is one run" int 1 (List.length pinned);
  Function.unpin f (Window.address w) (Window.length w);
  List.iter (fun (w, _) -> Function.free_dma f w) allocs

(* Failing at any access *)

(* A machine of one function whose BAR 0 is [base, base + 4096) of the far
   machine [far], as a transport reaches it. *)
let far_function far =
  let tr = Window.unsafe_transport far and base = 0x10_0000 in
  let fn =
    {
      Machine.addressing = Iommu;
      config8 = (fun _ -> 0);
      config16 = (fun off -> if off = 0 then 0x1002 else 0);
      config32 = (fun _ -> 0);
      set_config8 = (fun _ _ -> ());
      set_config16 = (fun _ _ -> ());
      set_config32 = (fun _ _ -> ());
      bar = (fun i -> if i = 0 then Some (base, 4096) else None);
      map = (fun ~combine:_ _ off n -> Ok (Window.through tr (base + off) n));
      unmap = ignore;
      interrupt = (fun _ -> false);
      reset = (fun () -> Ok ());
      alloc_dma = (fun ~contiguous:_ ~va:_ _ -> Error "far:1: no memory");
      free_dma = ignore;
      pin = (fun _ _ -> Error "far:1: no memory");
      unpin = (fun _ _ -> ());
      release = ignore;
    }
  in
  let m =
    Machine.make ~name:"far:1"
      {
        transport = tr;
        page = 4096;
        functions = (fun () -> []);
        take = (fun _ -> Ok fn);
        reserve = (fun ~base:_ _ -> Ok ());
      }
  in
  (m, require_ok (Function.take m bus1))

let data = "0123456789abcdef"

(* A driver's step with the checks it owes: a command, a wait for the device's
   ready bit, a copy out of 16 bytes, then [Function.failed] before the bytes
   leave. Three accesses reach the transport. *)
let step m f w =
  Window.set32 w 0 1;
  let ready () = Window.get32 w 4 land 1 = 1 in
  if not (Machine.wait m ~ms:1000 ready) then
    Error (Option.value (Function.failed f) ~default:"the device is not ready")
  else
    let s = Window.read w 8 16 in
    match Function.failed f with Some why -> Error why | None -> Ok s

let accesses = 3

(* Whichever access the transport fails at, the step ends in the machine's
   reason, never in bytes: a read through it gives all ones, and the checks
   catch them. *)
let fails_at_any_access =
  prop "a transport failing at access k ends a step in Error, never in bytes"
    (Gen.int_range 0 (accesses + 2))
    (fun k ->
      let far = far 0x10_0000 4096 in
      let m, f = far_function far in
      let w = require_ok (Function.map f 0) in
      Window.set32 w 4 1;
      Window.write w 8 data;
      break_at far k;
      cover "fails before the copy is checked" (k < accesses);
      cover "never fails" (k >= accesses);
      let want =
        if k < accesses then Error "far: the link broke" else Ok data
      in
      equal (result string string) want (step m f w))

let test_failed_function () =
  let far = far 0x10_0000 4096 in
  let m, f = far_function far in
  equal ~msg:"live" (option string) None (Function.failed f);
  break far;
  equal ~msg:"its machine failed" (option string) (Some "far: the link broke")
    (Function.failed f);
  let w = require_ok (Function.map f 0) in
  equal ~msg:"a read gives all ones" int 0xffff_ffff (Window.get32 w 4);
  Window.set32 w 4 0;
  equal ~msg:"a write is dropped" string (String.make 16 '\xff')
    (Window.read w 8 16);
  equal ~msg:"a wait is false" bool false
    (Machine.wait m ~ms:1000 (fun () -> true));
  equal ~msg:"a reset is refused" (result unit string)
    (Error "far: the link broke") (Function.reset f)

(* A function whose vendor ID reads all ones left the bus. *)
let test_left_bus () =
  let _, f, _ = take_fake () in
  equal ~msg:"live" (option string) None (Function.failed f);
  Function.set_config16 f 0 0xffff;
  equal (option string)
    (Some (bus1 ^ " left the bus: its vendor ID reads 0xffff"))
    (Function.failed f)

let failures =
  group ~timeout:patience "failures"
    [
      fails_at_any_access;
      test "a function of a failed machine is failed, its accesses all ones"
        test_failed_function;
      test "a function whose vendor ID reads 0xffff left the bus" test_left_bus;
    ]

(* A machine's files *)

(* [take_on fns bus] takes [bus] on a machine whose functions are [fns]. *)
let take_on ?lockdown ?groups ?noiommu fns bus =
  Function.take (Machine.at (Tree.make ?lockdown ?groups ?noiommu fns)) bus

let audio bus =
  { (Tree.gpu ~driver:"snd_hda_intel" bus) with class_ = 0x040300; bars = [] }

(* Each refusal names the function and its cause, and what cures it where a
   detach does. *)
let refusals =
  [
    ( "a bus the machine lacks",
      [ Tree.gpu "0000:03:00.0" ],
      "0000:04:00.0",
      [],
      [ "0000:04:00.0 is no PCI function" ] );
    ( "a driver other than vfio-pci, without an IOMMU",
      [ Tree.gpu ~driver:"amdgpu" "0000:03:00.0" ],
      "0000:03:00.0",
      [],
      [ "0000:03:00.0 is bound to the driver amdgpu"; "detach the GPU" ] );
    ( "a driver other than vfio-pci, behind an IOMMU",
      [ Tree.gpu ~driver:"amdgpu" ~group:"12" "0000:03:00.0" ],
      "0000:03:00.0",
      [],
      [ "bound to the driver amdgpu"; "vfio-pci" ] );
    ( "no driver behind a translating IOMMU",
      [ Tree.gpu ~group:"12" "0000:03:00.0" ],
      "0000:03:00.0",
      [],
      [ "the IOMMU translates the addresses 0000:03:00.0 reaches"; "iommu=pt" ]
    );
    ( "a device shared with another function",
      [ Tree.gpu "0000:03:00.0"; audio "0000:03:00.1" ],
      "0000:03:00.0",
      [],
      [ "0000:03:00.0 shares its device with 0000:03:00.1"; "detach the GPU" ]
    );
    ( "a device shared with another function, bound to vfio-pci without an IOMMU",
      [ Tree.gpu ~driver:"vfio-pci" "0000:03:00.0"; audio "0000:03:00.1" ],
      "0000:03:00.0",
      [],
      [ "0000:03:00.0 shares its device with 0000:03:00.1" ] );
    ( "a function on vfio-pci in no IOMMU group",
      [ Tree.gpu ~driver:"vfio-pci" "0000:03:00.0" ],
      "0000:03:00.0",
      [ "flock" ],
      [ "0000:03:00.0 is bound to vfio-pci but in no IOMMU group" ] );
    ( "a disabled function",
      [ Tree.gpu ~enabled:false "0000:03:00.0" ],
      "0000:03:00.0",
      [],
      [ "0000:03:00.0 is disabled"; "detach the GPU" ] );
    ( "a configuration file the process may not write",
      [ Tree.gpu "0000:03:00.0" ],
      "0000:03:00.0",
      [ "read-only" ],
      [ "taking 0000:03:00.0 needs write access"; "run as root" ] );
    ( "a locked-down kernel",
      [ Tree.gpu "0000:03:00.0" ],
      "0000:03:00.0",
      [ "lockdown" ],
      [ "the kernel is locked down"; "0000:03:00.0" ] );
    ( "BARs that cannot be read",
      [ Tree.gpu "0000:03:00.0" ],
      "0000:03:00.0",
      [ "unreadable:sys/bus/pci/devices/0000:03:00.0/resource" ],
      [ "reading "; "0000:03:00.0/resource: " ] );
    ( "an IOMMU group whose type cannot be read",
      [ Tree.gpu ~group:"12" "0000:03:00.0" ],
      "0000:03:00.0",
      [ "identity"; "unreadable:sys/kernel/iommu_groups/12/type" ],
      [ "reading "; "iommu_groups/12/type: " ] );
  ]

let test_refusal (_, fns, bus, opts, subs) =
  if List.mem "flock" opts && not on_linux then
    skip ~reason:"flock on a function's file needs Linux" ();
  let lockdown =
    if List.mem "lockdown" opts then Some "none [integrity] confidentiality"
    else None
  in
  let groups =
    if List.mem "identity" opts then [ ("12", "identity") ] else []
  in
  let root = Tree.make ?lockdown ~groups fns in
  let chmod file mode =
    if Unix.geteuid () = 0 then
      skip ~reason:"root opens a file whatever its mode" ();
    Unix.chmod (Filename.concat root file) mode
  in
  List.iter
    (fun opt ->
      match String.split_on_char ':' opt with
      | [ "read-only" ] ->
          chmod ("sys/bus/pci/devices/" ^ bus ^ "/config") 0o444
      | "unreadable" :: path -> chmod (String.concat ":" path) 0o000
      | _ -> ())
    opts;
  let why = require_error (Function.take (Machine.at root) bus) in
  List.iter (fun sub -> contains ~sub why) subs

(* An identity IOMMU passes physical addresses through, as does VFIO's no-IOMMU
   mode: a function under either is taken physically. On Linux the take locks
   its configuration file; elsewhere flock is refused. *)
let test_physical () =
  if not on_linux then skip ~reason:"flock on a function's file needs Linux" ();
  List.iter
    (fun (msg, groups, noiommu, group) ->
      let fn = Tree.gpu ?group "0000:03:00.0" in
      let m = Machine.at (Tree.make ~groups ~noiommu [ fn ]) in
      let f = require_ok ~msg (Function.take m fn.bus) in
      equal ~msg addressing Physical (Function.addressing f);
      equal ~msg:"vendor" int 0x1002 (Function.config16 f 0);
      equal ~msg:"device" int 0x744c (Function.config16 f 2);
      let bars = List.init 7 (Function.bar f) in
      equal ~msg:"BARs"
        (list (option (pair hex int)))
        [
          Some (0x7c_0000_0000, 256 * mib);
          None;
          Some (0xfc00_0000, 2 * mib);
          None;
          Some (0xe000, 256);
          Some (0xfcc0_0000, mib);
          None;
        ]
        bars;
      equal ~msg:"past the 64 bytes Linux shows" hex 0xffff_ffff
        (Function.config32 f 64);
      equal ~msg:"no interrupt to wait for" bool false
        (Function.interrupt f max_int);
      contains ~msg:"a second take" ~sub:"0000:03:00.0 is taken already"
        (require_error (Function.take m fn.bus));
      Function.release f;
      Function.release
        (require_ok ~msg:"a take once released" (Function.take m fn.bus)))
    [
      ("no IOMMU", [], [], None);
      ("an identity IOMMU", [ ("12", "identity") ], [], Some "12");
      ("VFIO's no-IOMMU mode", [], [ "12" ], Some "12");
    ]

(* Bound to vfio-pci, a function is taken through VFIO, which takes its IOMMU
   group whole: neither what shares its device nor whether it is enabled refuses
   it. The machine has none of VFIO's files, so the take is refused naming the
   first one it opens. In VFIO's no-IOMMU mode the take locks the function's
   file first. *)
let through_vfio =
  [
    ("behind a translating IOMMU, beside its audio", [], [], true, true);
    ("behind an identity IOMMU", [ ("12", "identity") ], [], false, true);
    ("behind an IOMMU, disabled", [], [], false, false);
    ("in VFIO's no-IOMMU mode", [], [ "12" ], false, true);
  ]

let test_through_vfio (_, groups, noiommu, beside, enabled) =
  if noiommu <> [] && not on_linux then
    skip ~reason:"flock on a function's file needs Linux" ();
  let fn = Tree.gpu ~driver:"vfio-pci" ~group:"12" ~enabled "0000:03:00.0" in
  let fns = if beside then [ fn; audio "0000:03:00.1" ] else [ fn ] in
  contains ~sub:"dev/vfio/vfio does not exist"
    (require_error (take_on ~groups ~noiommu fns fn.bus))

(* The command register and two of its bits (PCI Express Base Specification,
   7.5.1.1.3): the function answers at its memory BARs, and it masters the bus,
   reaching system memory by DMA. *)
let command = 0x04
let memory_space = 0x2
let bus_master = 0x4

(* The command register's bit that keeps the function from signalling legacy
   interrupts, its INTx line (PCI Express Base Specification, 7.5.1.1.3). *)
let intx_disable = 0x400

(* [take_mastering m bus] takes [bus] on [m] and turns its bus mastering on. *)
let take_mastering m bus =
  let f = require_ok (Function.take m bus) in
  Function.set_config16 f command (memory_space lor bus_master);
  f

let test_release_stops_dma () =
  if not on_linux then skip ~reason:"flock on a function's file needs Linux" ();
  let fn = Tree.gpu "0000:03:00.0" in
  let m = Machine.at (Tree.make [ fn ]) in
  Function.release (take_mastering m fn.bus);
  let f = require_ok (Function.take m fn.bus) in
  equal hex (memory_space lor intx_disable) (Function.config16 f command);
  Function.release f

(* The command register as the fixture's configuration file holds it. *)
let config_file root bus =
  Filename.concat root (strf "sys/bus/pci/devices/%s/config" bus)

let command_in root bus =
  let s =
    In_channel.with_open_bin (config_file root bus) In_channel.input_all
  in
  String.get_uint16_le s command

let set_command root bus v =
  let fd = Unix.openfile (config_file root bus) [ O_WRONLY ] 0 in
  Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
  let b = Bytes.create 2 in
  Bytes.set_uint16_le b 0 v;
  ignore (Unix.lseek fd command SEEK_SET);
  ignore (Unix.write fd b 0 2)

(* A function taken physically, with no interrupt route through VFIO, signals no
   legacy interrupt while taken: nothing handles it, and bus mastering does not
   gate it. Release gives its INTx back as it found it. *)
let test_intx (_, before) =
  if not on_linux then skip ~reason:"flock on a function's file needs Linux" ();
  let fn = Tree.gpu "0000:03:00.0" in
  let root = Tree.make [ fn ] in
  let m = Machine.at root in
  set_command root fn.bus (memory_space lor before);
  let f = require_ok (Function.take m fn.bus) in
  equal ~msg:"taken" hex
    (memory_space lor intx_disable)
    (Function.config16 f command);
  Function.release f;
  equal ~msg:"released" hex (memory_space lor before) (command_in root fn.bus)

let intx_states = [ ("INTx on", 0); ("INTx off", intx_disable) ]

(* The test's executable, run with [exiting], is [exit_mastering]'s process. *)
let exiting = "--exit-mastering"

(* Takes [bus] of the machine at [root] with its bus mastering on, forks a child
   that exits, and exits, with 0 iff the child's exit left the bus mastering
   on. *)
let exit_mastering root bus =
  let f = take_mastering (Machine.at root) bus in
  (match Unix.fork () with
  | 0 -> exit 0
  | child -> ignore (Unix.waitpid [] child));
  exit (if Function.config16 f command land bus_master <> 0 then 0 else 1)

let test_exit_stops_dma () =
  if not on_linux then skip ~reason:"flock on a function's file needs Linux" ();
  let fn = Tree.gpu "0000:03:00.0" in
  let root = Tree.make [ fn ] in
  let exe = Sys.executable_name in
  let pid =
    Unix.create_process exe
      [| exe; exiting; root; fn.bus |]
      Unix.stdin Unix.stdout Unix.stderr
  in
  let status = ref None in
  let exited () =
    match Unix.waitpid [ WNOHANG ] pid with
    | 0, _ -> false
    | _, s ->
        status := Some s;
        true
  in
  if not (poll exited) then begin
    Unix.kill pid Sys.sigkill;
    failf "the process holding %s did not exit" fn.bus
  end;
  let code = match !status with Some (WEXITED c) -> c | _ -> -1 in
  equal ~msg:"its child's exit left it mastering the bus" int 0 code;
  equal ~msg:"its own exit stopped it" hex 0
    (command_in root fn.bus land bus_master)

(* Linux offers a prefetchable BAR combining through [resourceN_wc]: BAR 0 of
   the fixture's GPU is prefetchable, BAR 5 is not. *)
let test_combining () =
  if not on_linux then skip ~reason:"flock on a function's file needs Linux" ();
  let fn = Tree.gpu "0000:03:00.0" in
  let f = require_ok (Function.take (Machine.at (Tree.make [ fn ])) fn.bus) in
  Fun.protect ~finally:(fun () -> Function.release f) @@ fun () ->
  let combines ?combine i =
    let w = map ?combine ~length:4096 f i in
    equal ~msg:"its bytes" int 0 (Window.get32 w 0);
    (w, c_combines w)
  in
  let w, c = combines ~combine:true 0 in
  equal ~msg:"a prefetchable BAR, asked" bool true c;
  raises_match ~msg:"the other way while it lives" Exn.invalid_arg (fun () ->
      Function.map ~combine:false f 0);
  let w', c = combines 0 in
  equal ~msg:"not asked, as the live window" bool true c;
  Function.unmap f w';
  Function.unmap f w;
  let w, c = combines 0 in
  equal ~msg:"a prefetchable BAR, not asked" bool false c;
  Function.unmap f w;
  let _, c = combines ~combine:true 5 in
  equal ~msg:"a BAR that is not prefetchable, asked" bool false c

let tree_files =
  group ~timeout:patience "a machine's files"
    [
      cases "a take is refused, naming the function and the cause"
        ~name:(fun (n, _, _, _, _) -> n)
        refusals test_refusal;
      test
        "a function alone and enabled, under no translating IOMMU, is taken \
         physically by one take at a time, its BARs as its registers and \
         resource file say, all ones past 64 bytes, without interrupts"
        test_physical;
      cases
        "a function bound to vfio-pci is taken through VFIO, whatever shares \
         its device or whether it is enabled"
        ~name:(fun (n, _, _, _, _) -> n)
        through_vfio test_through_vfio;
      test
        "a prefetchable BAR of a function taken physically combines where \
         asked, one way at a time"
        test_combining;
      test "a function taken physically stops mastering the bus when released"
        test_release_stops_dma;
      cases
        "a function taken physically has its INTx off while taken, as found \
         after"
        ~name:fst intx_states test_intx;
      test
        "a function taken physically stops mastering the bus when its process \
         exits, and a child that process forked exits without stopping it"
        test_exit_stops_dma;
    ]

(* Locked system memory

   Functions of a machine in a fixture tree, taken physically, reach this
   process's memory as GPUs of this machine do, at the physical addresses the
   tree's page map gives. Locking memory needs the locked-memory limit: a test
   the machine refuses it skips with the reason. *)

(* [with_fixtures n f] is [f root fns], [fns] the [n] functions of a fixture
   machine at [root], taken. *)
let with_fixtures n f =
  if not on_linux then skip ~reason:"flock on a function's file needs Linux" ();
  let buses = List.init n (fun i -> strf "0000:%02x:00.0" (3 + i)) in
  let root = Tree.make (List.map Tree.gpu buses) in
  let m = Machine.at root in
  let fns = List.map (fun bus -> require_ok (Function.take m bus)) buses in
  Fun.protect
    ~finally:(fun () -> List.iter Function.release fns)
    (fun () -> f root fns)

let with_fixture f = with_fixtures 1 (fun root fns -> f root (List.hd fns))
let huge = 2 * mib

(* Frames the fixture's page map gives the 2 MiB blocks of addresses that hold
   the [n] bytes at [a], one block of frames each, from a frame no other block
   of a test uses; and the physical addresses of the pages of the [n] bytes. *)
let frames root a n =
  let page = Machine.page Machine.this in
  let lo = a / huge * huge and hi = round_up (a + n) huge in
  let first block = 0x10_0000 + (block / huge land 0xfff * (huge / page)) in
  let rec give block =
    if block < hi then begin
      Tree.pagemap root ~page block
        (List.init (huge / page) (fun i -> first block + i));
      give (block + huge)
    end
  in
  give lo;
  List.init
    ((n + page - 1) / page)
    (fun i ->
      let p = a + (i * page) in
      (first (p / huge * huge) * page) + (p mod huge))

(* The runs of pages [pas]: those that follow each other merged. *)
let runs_of pas =
  let page = Machine.page Machine.this in
  List.fold_left
    (fun acc pa ->
      match acc with
      | (a, n) :: rest when a + n = pa -> (a, n + page) :: rest
      | _ -> (pa, page) :: acc)
    [] pas
  |> List.rev

let granted = function Ok x -> x | Error why -> skip ~reason:why ()

let given r =
  match granted r with
  | Some x -> x
  | None -> skip ~reason:"the machine has no free memory" ()

(* Memory reached physically lies in huge pages: its runs are those of the
   blocks of frames that hold it. *)
let test_dma () =
  with_fixture @@ fun root f ->
  granted (Machine.reserve (Function.machine f) ~base:free_base (8 * mib));
  let va = free_base + mib in
  let pas = frames root va (3 * mib) in
  let w, runs = given (Function.alloc_dma ~va f (3 * mib)) in
  equal ~msg:"a run per block of frames"
    (list (pair hex int))
    (runs_of pas) runs;
  equal ~msg:"zeroed" string
    (String.make (3 * mib) '\000')
    (Window.read w 0 (3 * mib));
  Function.free_dma f w

(* The process's pages go back to the system when it dies: a function taken
   physically, which would keep writing them, is refused them. *)
let test_pin_physical () =
  with_fixture @@ fun root f ->
  let page = Machine.page Machine.this in
  let a = round_up (memory (2 * page)) page in
  ignore (frames root a page);
  contains ~sub:"without an IOMMU" (require_error (Function.pin f a page))

let test_contiguous () =
  with_fixture @@ fun root f ->
  granted (Machine.reserve (Function.machine f) ~base:free_base (8 * mib));
  let va = free_base + (2 * mib) in
  ignore (frames root va (2 * mib));
  let w, runs = given (Function.alloc_dma ~contiguous:true ~va f (300 * kib)) in
  equal ~msg:"at the address asked" hex va (Window.address w);
  equal ~msg:"one run" int 1 (List.length runs);
  Function.free_dma f w

(* Memory that outlives the process

   A function taken physically keeps writing memory after its process dies: its
   memory lies in huge pages of files under the machine's [dev/hugepages], which
   keep their pages until no function reaches them. *)

let memory_files root =
  Sys.readdir (Filename.concat root "dev/hugepages")
  |> Array.to_list
  |> List.filter (fun f ->
      String.starts_with ~prefix:"rig-pci-" f
      && not (String.ends_with ~suffix:".reach" f))

let memory_file root =
  match memory_files root with
  | [ f ] -> Filename.concat root ("dev/hugepages/" ^ f)
  | fs -> failf "%d memory files" (List.length fs)

let reserved f =
  granted (Machine.reserve (Function.machine f) ~base:free_base (8 * mib))

let test_memory_file () =
  with_fixture @@ fun root f ->
  reserved f;
  let va = free_base + (4 * mib) in
  ignore (frames root va mib);
  let w, _ = given (Function.alloc_dma ~va f mib) in
  equal ~msg:"a file of its own" int 1 (List.length (memory_files root));
  Function.free_dma f w;
  equal ~msg:"kept while the function is held" int 1
    (List.length (memory_files root));
  Function.release f;
  equal ~msg:"gone once released" (list string) [] (memory_files root)

(* A function is named by its machine and bus: the release of one leaves the
   memory of a function at the same bus on another machine. *)
let test_memory_machines () =
  with_fixture @@ fun root f ->
  with_fixture @@ fun root' f' ->
  reserved f;
  reserved f';
  let va = free_base + (4 * mib) and va' = free_base + (6 * mib) in
  ignore (frames root va mib);
  ignore (frames root' va' mib);
  let w, _ = given (Function.alloc_dma ~va f mib) in
  let w', _ = given (Function.alloc_dma ~va:va' f' mib) in
  equal ~msg:"one bus on both machines" string (Function.bus f)
    (Function.bus f');
  let kept = memory_files root in
  Function.release f';
  equal ~msg:"the other machine's memory stays" (list string) kept
    (memory_files root);
  Function.free_dma f' w';
  Function.free_dma f w

(* The test's executable, run with [holding], is [hold_memory]'s process. *)
let holding = "--hold-memory"

(* The code [hold_memory] exits with when the machine refuses it the function's
   memory. *)
let refused_code = 3

(* Takes the fixture's function at [bus] and allocates its memory at [va], whose
   frames the tree's page map gives, which it never frees. With [how] ["die"] it
   then dies by SIGKILL, which runs no exit function; with ["wait"] it waits for
   its standard input to close and exits. ["released-die"] and ["released-wait"]
   release the function first. *)
let hold_memory how root bus va =
  let ok = function Ok x -> x | Error _ -> exit refused_code in
  let f = ok (Function.take (Machine.at root) bus) in
  ok (Machine.reserve (Function.machine f) ~base:free_base (8 * mib));
  if Option.is_none (ok (Function.alloc_dma ~va f mib)) then exit refused_code;
  if String.starts_with ~prefix:"released-" how then Function.release f;
  match how with
  | "die" | "released-die" -> Unix.kill (Unix.getpid ()) Sys.sigkill
  | _ ->
      print_endline "holding";
      ignore (In_channel.input_all stdin);
      exit 0

(* [holder how root bus va] starts [hold_memory]: its pid, the pipe on its
   standard input, and its first line of output, which waits until it holds. *)
let holder how root bus va =
  let exe = Sys.executable_name in
  let input, feed = Unix.pipe ~cloexec:true () in
  let said, output = Unix.pipe ~cloexec:true () in
  let pid =
    Unix.create_process exe
      [| exe; holding; how; root; bus; string_of_int va |]
      input output Unix.stderr
  in
  Unix.close input;
  Unix.close output;
  (pid, feed, said)

let wait_exit pid =
  let status = ref None in
  let exited () =
    match Unix.waitpid [ WNOHANG ] pid with
    | 0, _ -> false
    | _, s ->
        status := Some s;
        true
  in
  if not (poll exited) then begin
    Unix.kill pid Sys.sigkill;
    failf "the process %d did not exit" pid
  end;
  match !status with
  | Some (WEXITED c) when c = refused_code ->
      skip ~reason:"the machine refused the function's memory" ()
  | Some s -> s
  | None -> assert false

let fixture_gpus () =
  Gpus.make ~memory_bar:0
    ~nodes:(fun ~read:_ _ -> [])
    ~reset:(fun _ -> Ok ())
    (fun (id : Machine.id) -> id.class_ lsr 16 = 0x03)

(* What a process that died left stays through another take and release, which
   does not reset the GPU, and goes once its GPU is reset. *)
let test_death () =
  if not on_linux then skip ~reason:"flock on a function's file needs Linux" ();
  let fn = Tree.gpu "0000:03:00.0" in
  let root = Tree.make [ fn ] in
  let va = free_base + (6 * mib) in
  ignore (frames root va mib);
  let pid, feed, said = holder "die" root fn.bus va in
  Unix.close feed;
  Unix.close said;
  (match wait_exit pid with
  | WSIGNALED s when s = Sys.sigkill -> ()
  | _ -> fail "the process holding the function did not die by SIGKILL");
  let left = memory_files root in
  equal ~msg:"left by the dead process" int 1 (List.length left);
  let m = Machine.at root in
  Function.release (require_ok (Function.take m fn.bus));
  equal ~msg:"kept through a take and a release" (list string) left
    (memory_files root);
  require_ok (Gpus.reset (fixture_gpus ()) m 0);
  equal ~msg:"gone once the GPU is reset" (list string) [] (memory_files root)

(* A reset of a GPU another process holds is refused, its memory untouched; the
   holder's exit gives the memory back. *)
let test_reset_held () =
  if not on_linux then skip ~reason:"flock on a function's file needs Linux" ();
  let fn = Tree.gpu "0000:03:00.0" in
  let root = Tree.make [ fn ] in
  let va = free_base + (6 * mib) in
  ignore (frames root va mib);
  let pid, feed, said = holder "wait" root fn.bus va in
  let line = In_channel.input_line (Unix.in_channel_of_descr said) in
  if line <> Some "holding" then begin
    Unix.close feed;
    ignore (wait_exit pid);
    fail "the holder did not hold the function"
  end;
  let held = memory_files root in
  equal ~msg:"the holder's memory" int 1 (List.length held);
  ignore (require_error (Gpus.reset (fixture_gpus ()) (Machine.at root) 0));
  equal ~msg:"kept while held" (list string) held (memory_files root);
  Unix.close feed;
  (match wait_exit pid with
  | WEXITED 0 -> ()
  | _ -> fail "the holder did not exit");
  equal ~msg:"given back at the holder's exit" (list string) []
    (memory_files root)

(* A released function's memory stays, and a later take of the machine allocates
   beside it in its huge page. *)
let test_released_block () =
  with_fixture @@ fun root f ->
  reserved f;
  let va = free_base + (2 * mib) and page = Machine.page Machine.this in
  ignore (frames root va (2 * page));
  let w, _ = given (Function.alloc_dma ~va f page) in
  Function.release f;
  equal ~msg:"kept while it holds memory" int 1
    (List.length (memory_files root));
  let f' = require_ok (Function.take (Function.machine f) (Function.bus f)) in
  let w', _ = given (Function.alloc_dma ~va:(va + page) f' page) in
  Function.free_dma f w;
  Function.free_dma f' w';
  Function.release f';
  equal ~msg:"gone once it lists no function and holds nothing" (list string) []
    (memory_files root)

(* A reset deletes the file of a process that died after it released its
   function, and keeps that of a process that lives. *)
let test_released_holder how () =
  if not on_linux then skip ~reason:"flock on a function's file needs Linux" ();
  let fn = Tree.gpu "0000:03:00.0" in
  let root = Tree.make [ fn ] in
  let va = free_base + (6 * mib) in
  ignore (frames root va mib);
  let pid, feed, said = holder how root fn.bus va in
  let reset () =
    require_ok (Gpus.reset (fixture_gpus ()) (Machine.at root) 0)
  in
  match how with
  | "released-die" ->
      Unix.close feed;
      Unix.close said;
      (match wait_exit pid with
      | WSIGNALED s when s = Sys.sigkill -> ()
      | _ -> fail "the holder did not die by SIGKILL");
      equal ~msg:"left by the dead process" int 1
        (List.length (memory_files root));
      reset ();
      equal ~msg:"gone at the reset" (list string) [] (memory_files root)
  | _ ->
      if In_channel.input_line (Unix.in_channel_of_descr said) <> Some "holding"
      then begin
        Unix.close feed;
        ignore (wait_exit pid);
        fail "the holder did not hold the function"
      end;
      let held = memory_files root in
      equal ~msg:"the holder's memory" int 1 (List.length held);
      reset ();
      equal ~msg:"kept while its process lives" (list string) held
        (memory_files root);
      Unix.close feed;
      (match wait_exit pid with
      | WEXITED 0 -> ()
      | _ -> fail "the holder did not exit");
      equal ~msg:"gone at its process's exit" (list string) []
        (memory_files root)

(* A huge page is one block: frames that are not, as a file system that is no
   hugetlbfs gives, are refused, naming the mount. *)
let test_scattered () =
  with_fixture @@ fun root f ->
  reserved f;
  let va = free_base + (2 * mib) and page = Machine.page Machine.this in
  Tree.pagemap root ~page va
    (List.init (2 * mib / page) (fun i -> 0x30_0000 + (2 * i)));
  contains ~sub:"hugetlbfs"
    (require_error (Function.alloc_dma ~va f (300 * kib)))

(* Memory whose addresses share a 2 MiB block shares its huge page, which goes
   back once neither holds it. *)
let test_shared_page () =
  with_fixture @@ fun root f ->
  reserved f;
  let va = free_base + (2 * mib) in
  let pas = frames root va (128 * kib) in
  let a, ra = given (Function.alloc_dma ~va f (64 * kib)) in
  let b, rb = given (Function.alloc_dma ~va:(va + (64 * kib)) f (64 * kib)) in
  let pa = List.hd pas in
  equal ~msg:"the first at its frames"
    (list (pair hex int))
    [ (pa, 64 * kib) ]
    ra;
  equal ~msg:"the second after it, in the same huge page"
    (list (pair hex int))
    [ (pa + (64 * kib), 64 * kib) ]
    rb;
  let file = memory_file root in
  equal ~msg:"one huge page" int huge (Tree.stored file);
  Function.free_dma f a;
  equal ~msg:"kept while the second holds it" int huge (Tree.stored file);
  Function.free_dma f b;
  equal ~msg:"given back once neither does" int 0 (Tree.stored file)

(* Memory handed out again in a huge page that stayed is zeroed. *)
let test_reused_zeroed () =
  with_fixture @@ fun root f ->
  reserved f;
  let va = free_base + (2 * mib) and page = Machine.page Machine.this in
  ignore (frames root va (2 * page));
  let a, _ = given (Function.alloc_dma ~va f page) in
  let b, _ = given (Function.alloc_dma ~va:(va + page) f page) in
  Window.write a 0 (String.make page 'x');
  Function.free_dma f a;
  let a, _ = given (Function.alloc_dma ~va f page) in
  equal ~msg:"zeroed" string (String.make page '\000') (Window.read a 0 page);
  Function.free_dma f a;
  Function.free_dma f b

(* Addresses whose 2 MiB block another machine's memory holds are refused. *)
let test_block_taken () =
  with_fixture @@ fun root f ->
  with_fixture @@ fun root' f' ->
  reserved f;
  reserved f';
  let va = free_base + (2 * mib) and page = Machine.page Machine.this in
  ignore (frames root va (2 * page));
  ignore (frames root' va (2 * page));
  let w, _ = given (Function.alloc_dma ~va f page) in
  ignore (require_error (Function.alloc_dma ~va:(va + page) f' page));
  Function.free_dma f w

(* The huge page around memory at a reserved address must be reserved whole. *)
let test_block_reserved () =
  with_fixture @@ fun root f ->
  let page = Machine.page Machine.this in
  let base = free_base + (16 * mib) + page in
  granted (Machine.reserve (Function.machine f) ~base (4 * mib));
  ignore (frames root base page);
  raises_match (Exn.invalid_arg ~substring:"2 MiB") (fun () ->
      Function.alloc_dma ~va:base f page)

let system_memory =
  group ~timeout:patience "system memory"
    [
      test "DMA memory, zeroed, at the frames of the huge pages that hold it"
        test_dma;
      test "a function taken physically is refused the process's pages"
        test_pin_physical;
      test "contiguous memory is one run at the reserved address asked"
        test_contiguous;
      test
        "a physical take's memory lies in a file of its own, gone once released"
        test_memory_file;
      test "a release leaves the memory of another machine's function"
        test_memory_machines;
      test
        "memory a killed process left stays through a release and goes at its \
         GPU's reset (SIGKILL in a child)"
        test_death;
      test
        "a reset of a GPU another process holds is refused, its memory kept (a \
         child holds)"
        test_reset_held;
      test
        "a released function's memory stays, and a later take shares its page"
        test_released_block;
      test
        "a reset deletes the file of a process that died after releasing its \
         function (SIGKILL in a child)"
        (test_released_holder "released-die");
      test
        "a reset keeps the file of a living process that released its function \
         (a child holds)"
        (test_released_holder "released-wait");
      test "memory whose frames are not one block is refused" test_scattered;
      test "memory in one 2 MiB block shares a huge page, gone once both are"
        test_shared_page;
      test "memory handed out again in a huge page is zeroed" test_reused_zeroed;
      test "a 2 MiB block another machine's memory holds is refused"
        test_block_taken;
      test "the 2 MiB block around memory must be reserved" test_block_reserved;
    ]

let this_machine =
  group ~timeout:patience "this machine"
    [
      test "this machine without /sys/bus/pci refuses a take, naming the bus"
        test_refused_here;
      test "listing functions and taking GPUs change nothing on this machine"
        test_changes_nothing;
      test "a GPU taken here refuses a second take until released"
        test_held_here;
      test "a GPU behind an IOMMU reaches its memory at one run apart" test_vfio;
    ]

let () =
  match Sys.argv with
  | [| _; arg; root; bus |] when arg = exiting -> exit_mastering root bus
  | [| _; arg; how; root; bus; va |] when arg = holding ->
      hold_memory how root bus (int_of_string va)
  | _ ->
      hold_gpu ();
      exit
      @@ run "rig_pci.function"
           [
             taking;
             uses;
             misuse_refused;
             model;
             failures;
             tree_files;
             system_memory;
             this_machine;
           ]
