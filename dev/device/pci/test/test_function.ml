(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Device_pci
open Device_pci_support

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
  mutable dmas : (int * int) list;
  mutable pins : (int * int) list; (* a multiset *)
  mutable released : bool;
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
      dmas = [];
      pins = [];
      released = false;
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
        f.wrong <- Printf.sprintf "%s: %s" name why :: f.wrong;
        failwith ("misuse reached the machine: " ^ name)
    | None -> run ()
  in
  let width n = if List.mem n [ 1; 2; 4 ] then None else Some "width" in
  let in_config off n =
    if off < 0 || off > config_size - n then Some "offset" else width n
  in
  let window w = (Window.address w, Window.length w) in
  let ops =
    {
      Machine.addressing = f.addressing;
      config =
        (fun off n ->
          call "config"
            (fun () -> in_config off n)
            (fun () ->
              let x = ref 0 in
              for i = n - 1 downto 0 do
                x := (!x lsl 8) lor Bytes.get_uint8 f.config (off + i)
              done;
              !x));
      set_config =
        (fun off n x ->
          call "set_config"
            (fun () -> in_config off n)
            (fun () ->
              for i = 0 to n - 1 do
                Bytes.set_uint8 f.config (off + i) ((x lsr (8 * i)) land 0xff)
              done));
      bar =
        (fun i ->
          call "bar"
            (fun () -> if i < 0 then Some "index" else None)
            (fun () -> bar_of i));
      map =
        (fun i off n ->
          call "map"
            (fun () ->
              match bar_of i with
              | Some (_, size) when off >= 0 && n >= 0 && off <= size - n ->
                  None
              | _ -> Some "bytes outside the BAR")
            (fun () ->
              let a = fresh m n in
              f.maps <- (a, n) :: f.maps;
              Ok (Window.through m.tr a n)));
      unmap =
        (fun w ->
          call "unmap"
            (fun () ->
              if List.mem (window w) f.maps then None else Some "no window")
            (fun () -> f.maps <- Option.get (remove (window w) f.maps)));
      interrupt = (fun _ -> call "interrupt" (fun () -> None) (fun () -> false));
      reset = (fun () -> call "reset" (fun () -> None) (fun () -> Ok ()));
      alloc_dma =
        (fun ~contiguous ~va n ->
          let bytes = round_up n f.page in
          call "alloc_dma"
            (fun () ->
              match va with
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
                Ok (Window.through m.tr a bytes, runs f a bytes ~one:contiguous)
              end));
      free_dma =
        (fun w ->
          call ~after_release:true "free_dma"
            (fun () ->
              if List.mem (window w) f.dmas then None else Some "no memory")
            (fun () -> f.dmas <- Option.get (remove (window w) f.dmas)));
      pin =
        (fun a n ->
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
            (fun () -> None)
            (fun () ->
              if not f.released then begin
                f.released <- true;
                f.maps <- [];
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
  (machine, m)

let take_fake ?base ?page ?addressing () =
  let machine, m = fake_machine ?base ?page ?addressing () in
  let f = Result.get_ok (Function.take machine bus1) in
  (machine, f, List.hd m.taken)

(* The requests of a fake machine, which refuses none within its contract. *)
let map ?off ?length f i = require_ok (Function.map ?off ?length f i)

let alloc_dma ?contiguous ?va f n =
  require_ok (Function.alloc_dma ?contiguous ?va f n)

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
        (Error (Printf.sprintf "%S is no PCI bus address" bus))
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
  equal ~msg:"asked" (list string) [ "config"; "reset" ] fake.calls

(* A vendor ID of all ones is a function that does not answer. *)
let test_reset_silent () =
  let _, f, _ = take_fake () in
  Function.set_config16 f 0 0xffff;
  equal (result unit string)
    (Error (bus1 ^ " does not answer 1000 ms after its reset"))
    (Function.reset f)

let uses =
  group ~timeout:patience "uses"
    [
      test "a function's windows, BARs and runs are its machine's"
        test_machine_values;
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

let va_base = 0x7f00_0000_0000

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
  kind : [ `Bar | `Dma ];
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

let map_ref f i off len =
  live f;
  if i < 0 then misuse "a BAR index below zero";
  match bar_of i with
  | None -> misuse "no such BAR"
  | Some (_, size) ->
      let off = Option.value off ~default:0 in
      let len = Option.value len ~default:(size - off) in
      if off < 0 || len < 0 || off > size - len then
        misuse "bytes outside the BAR";
      let w = { owner = f; kind = `Bar; len; at = None } in
      f.r_maps <- w :: f.r_maps;
      w

let map_sys (f, _) i off len = map ?off ?length:len f i
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
  if not (w.kind = `Bar && w.owner == f && List.memq w f.r_maps) then
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
  | Ok (w, _) -> w
  | Error _ -> raise In_use

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
   memory above 2 MiB and huge pages off 2 MiB among them. *)
let size = 64 * 1024
let offs = [ -1; 0; 1; 16; size - 1; size; size + 1; max_int ]
let map_lens = [ -1; 0; 1; 16; 4096; size - 1; size + 1; max_int ]
let lens = [ 1; 4095; 4096; 4097; 16385; (2 * mib) - 1; 2 * mib; (2 * mib) + 1 ]
let vas = [ 0; 4096; 16384; 2 * mib; 4 * mib ]
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
      (fn_t
      ^-> ints [ -1; 0; 1; 2; 6 ]
      @-> opt_ints offs @-> opt_ints map_lens @-> makes win_t)
      map_ref map_sys;
    command "unmap"
      (fn_t ^-> win_t ^-> returns unit)
      unmap_ref (fsys Function.unmap);
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

(* Taking a GPU of this machine is a hardware opt-in: such a test runs only when
   DEVICE_PCI_TEST_GPU_LOCK names the machine's GPU lock, which it holds while
   it runs, so that it never takes a device another user drives. *)

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
  with_gpu_lock @@ fun () ->
  let buses = List.map (fun (d : Machine.id) -> d.bus) (host_gpus ()) in
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
  host_gpus ()
  |> List.find_map (fun (d : Machine.id) ->
      match Function.take Machine.this d.bus with
      | Ok f ->
          Function.release f;
          Some d
      | Error _ -> None)

(* One process holds a function at a time, this one included. *)
let test_held_here () =
  if not on_linux then skip ~reason:"this machine has no /sys/bus/pci" ();
  with_gpu_lock @@ fun () ->
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
  host_gpus ()
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
  with_gpu_lock @@ fun () ->
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
      config = (fun off _ -> if off = 0 then 0x1002 else 0);
      set_config = (fun _ _ _ -> ());
      bar = (fun i -> if i = 0 then Some (base, 4096) else None);
      map = (fun _ off n -> Ok (Window.through tr (base + off) n));
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
    (Machine.wait m ~ms:1000 (fun () -> true))

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

(* A host's files *)

(* [take_on fns bus] takes [bus] on a host whose functions are [fns]. *)
let take_on ?lockdown ?groups ?noiommu fns bus =
  Function.take (Machine.at (Host.make ?lockdown ?groups ?noiommu fns)) bus

let audio bus =
  { (Host.gpu ~driver:"snd_hda_intel" bus) with class_ = 0x04; bars = [] }

(* Each refusal names the function and its cause, and what cures it where a
   detach does. *)
let refusals =
  [
    ( "a bus the host lacks",
      [ Host.gpu "0000:03:00.0" ],
      "0000:04:00.0",
      [],
      [ "0000:04:00.0 is no PCI function" ] );
    ( "a driver other than vfio-pci, without an IOMMU",
      [ Host.gpu ~driver:"amdgpu" "0000:03:00.0" ],
      "0000:03:00.0",
      [],
      [ "0000:03:00.0 is bound to the driver amdgpu"; "detach the GPU" ] );
    ( "a driver other than vfio-pci, behind an IOMMU",
      [ Host.gpu ~driver:"amdgpu" ~group:"12" "0000:03:00.0" ],
      "0000:03:00.0",
      [],
      [ "bound to the driver amdgpu"; "vfio-pci" ] );
    ( "no driver behind a translating IOMMU",
      [ Host.gpu ~group:"12" "0000:03:00.0" ],
      "0000:03:00.0",
      [],
      [ "the IOMMU translates the addresses 0000:03:00.0 reaches"; "iommu=pt" ]
    );
    ( "a device shared with another function",
      [ Host.gpu "0000:03:00.0"; audio "0000:03:00.1" ],
      "0000:03:00.0",
      [],
      [ "0000:03:00.0 shares its device with 0000:03:00.1"; "detach the GPU" ]
    );
    ( "a disabled function",
      [ Host.gpu ~enabled:false "0000:03:00.0" ],
      "0000:03:00.0",
      [],
      [ "0000:03:00.0 is disabled"; "detach the GPU" ] );
    ( "a locked-down kernel",
      [ Host.gpu "0000:03:00.0" ],
      "0000:03:00.0",
      [ "lockdown" ],
      [ "the kernel is locked down"; "0000:03:00.0" ] );
  ]

let test_refusal (_, fns, bus, opts, subs) =
  let lockdown =
    if List.mem "lockdown" opts then Some "none [integrity] confidentiality"
    else None
  in
  let why = require_error (take_on ?lockdown fns bus) in
  List.iter (fun sub -> contains ~sub why) subs

(* An identity IOMMU passes physical addresses through, as does VFIO's no-IOMMU
   mode: a function under either is taken physically. On Linux the take locks
   its configuration file; elsewhere flock is refused. *)
let test_physical () =
  if not on_linux then skip ~reason:"flock on a function's file needs Linux" ();
  List.iter
    (fun (msg, groups, noiommu, group) ->
      let fn = Host.gpu ?group "0000:03:00.0" in
      let f = require_ok ~msg (take_on ~groups ~noiommu [ fn ] fn.bus) in
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
      Function.release f)
    [
      ("no IOMMU", [], [], None);
      ("an identity IOMMU", [ ("12", "identity") ], [], Some "12");
      ("VFIO's no-IOMMU mode", [], [ "12" ], Some "12");
    ]

(* The command register and two of its bits (PCI Express Base Specification,
   7.5.1.1.3): the function answers at its memory BARs, and it masters the bus,
   reaching system memory by DMA. *)
let command = 0x04
let memory_space = 0x2
let bus_master = 0x4

(* [take_mastering m bus] takes [bus] on [m] and turns its bus mastering on. *)
let take_mastering m bus =
  let f = require_ok (Function.take m bus) in
  Function.set_config16 f command (memory_space lor bus_master);
  f

let test_release_stops_dma () =
  if not on_linux then skip ~reason:"flock on a function's file needs Linux" ();
  let fn = Host.gpu "0000:03:00.0" in
  let m = Machine.at (Host.make [ fn ]) in
  Function.release (take_mastering m fn.bus);
  let f = require_ok (Function.take m fn.bus) in
  equal hex memory_space (Function.config16 f command);
  Function.release f

(* The test's executable, run with [exiting], is [exit_mastering]'s process. *)
let exiting = "--exit-mastering"

(* Takes [bus] of the host at [root] with its bus mastering on, forks a child
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
  let fn = Host.gpu "0000:03:00.0" in
  let root = Host.make [ fn ] in
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
  let f = require_ok (Function.take (Machine.at root) fn.bus) in
  equal ~msg:"its own exit stopped it" hex memory_space
    (Function.config16 f command);
  Function.release f

let host_files =
  group ~timeout:patience "a host's files"
    [
      cases "a take is refused, naming the function and the cause"
        ~name:(fun (n, _, _, _, _) -> n)
        refusals test_refusal;
      test
        "a function alone and enabled, under no translating IOMMU, is taken \
         physically, its BARs as its registers and resource file say"
        test_physical;
      test "a function taken physically stops mastering the bus when released"
        test_release_stops_dma;
      test
        "a function taken physically stops mastering the bus when its process \
         exits, and a child that process forked exits without stopping it"
        test_exit_stops_dma;
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
  | _ ->
      exit
      @@ run "device_pci Function"
           [
             taking;
             uses;
             misuse_refused;
             model;
             failures;
             host_files;
             this_machine;
           ]
