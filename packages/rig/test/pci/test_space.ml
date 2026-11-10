(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Virtual address spaces against the list of the ranges they handed out. *)

open Windtrap
open Rig_pci
open Rig_pci_support

(* The model *)

type model = {
  base : int;
  length : int;
  mutable live : (int * int) list;  (** (address, addresses), newest first. *)
  mutable freed : int list;  (** Addresses freed and not handed out again. *)
  mutable handed : int;  (** Ranges handed out. *)
  mutable turns : (int * int) list;
      (** The hand-out that each live address came from. *)
}

let create (base, n) =
  if base < 0 || n < 0 || n > max_int - base then invalid_arg "Space.create";
  { base; length = n; live = []; freed = []; handed = 0; turns = [] }

(* The alignment [alloc] promises: the size's and [align]'s. *)
let alignment ?(align = 4096) n = max align (pow2_floor n)
let largest_gap m = largest_gap m.base (m.base + m.length) m.live

(* An allocation is aligned, inside the space and apart from every live range.
   [None] is accepted only where the fit bound does not promise a range: no free
   range of [2 * (max n 16 + a)] addresses. [covered] registers the outcomes the
   generator reaches: on one domain only, since on two the scheduler decides
   which interleaving, and so which outcome, a case meets. *)
let alloc_judge ~covered align n m got =
  let cover name b = if covered then cover name b in
  let refused =
    n <= 0 || match align with Some a -> not (is_pow2 a) | None -> false
  in
  let gap = largest_gap m in
  let a = if refused then 0 else alignment ?align n in
  let none = match got with Ok None -> true | _ -> false in
  cover "a refused request" refused;
  cover "no addresses left" none;
  cover "a request right at the bound"
    ((not refused) && fits ~gap n a && not (fits ~gap (n + 1) a));
  cover "aligned to the size beyond the default"
    ((not refused) && Option.is_none align && a > 4096 && not none);
  cover "addresses handed out again after a free"
    (match got with Ok (Some x) -> List.mem x m.freed | _ -> false);
  match got with
  | Error (Invalid_argument _) when refused -> ()
  | Error e -> raise e
  | Ok r when refused ->
      failf "%a for %d addresses aligned to %a, which raise Invalid_argument"
        (Format.pp_print_option pp_hex)
        r n
        (Format.pp_print_option pp_hex)
        align
  | Ok None ->
      if fits ~gap (max n 16) a then
        failf "None for %d addresses aligned to 0x%x with %d free in a row" n a
          gap
  | Ok (Some x) ->
      equal ~msg:"aligned" hex 0 (x land (a - 1));
      at_least ~msg:"from the base" hex ~than:m.base x;
      at_most ~msg:"no more than the space holds" int ~than:m.length n;
      at_most ~msg:"inside the space" hex ~than:(m.length - n) (x - m.base);
      List.iter
        (fun (b, k) ->
          if not (x + n <= b || b + k <= x) then
            failf "[0x%x, +%d) overlaps [0x%x, +%d)" x n b k)
        m.live;
      m.live <- (x, n) :: m.live;
      m.freed <- List.filter (( <> ) x) m.freed;
      m.handed <- m.handed + 1;
      m.turns <- (x, m.handed) :: m.turns

let free m x =
  m.live <- List.remove_assoc x m.live;
  m.turns <- List.remove_assoc x m.turns;
  m.freed <- x :: m.freed

(* The holds the tests' holder took, by space and address, while live:
   either domain changes them. *)
let holder = Space.holder ()
let held : (Space.t * Space.hold) list Atomic.t = Atomic.make []

let rec update f =
  let l = Atomic.get held in
  if not (Atomic.compare_and_set held l (f l)) then update f

let alloc_held ?align s n =
  Option.map
    (fun x ->
      update (List.cons (s, x));
      Space.address x)
    (Space.alloc ?align s holder n)

(* The hold of the live range at [a] of [s]. *)
let hold_at s a =
  snd
    (List.find
       (fun (s', x) -> s' == s && Space.address x = a)
       (Atomic.get held))

(* Frees the hold of [s] at [a], taken off [held] first, so that one of two
   frees of it on two domains finds it. *)
let free_at s a =
  let rec take () =
    let l = Atomic.get held in
    match List.find_opt (fun (s', x) -> s' == s && Space.address x = a) l with
    | None -> None
    | Some (_, x) as found ->
        if Atomic.compare_and_set held l (List.filter (fun (_, y) -> y != x) l)
        then Option.map snd found
        else take ()
  in
  Option.iter Space.free (take ())

(* Commands *)

let space =
  abstract "s" ~invariant:(fun m s ->
      equal ~msg:"base" hex m.base (Space.base s);
      equal ~msg:"length" int m.length (Space.length s);
      equal ~msg:"ranges taken" int (List.length m.live) (Space.taken s))

let live = among hex space (fun m -> List.map fst m.live)

(* Remember a live range's hold. It survives a free, so parallel calls can free
   it again or after its address has been handed out anew, which frees
   nothing. The model remembers which hand-out it came from. *)
let allocation = abstract "a" ~pp:(fun ppf (_, a, _) -> pp_hex ppf a)

(* The sizes right at the fit bound of a free range of [g] addresses: [n] that
   fits with its alignment [a] where [n + 1] would not, [n = g / 2 - a]. *)
let at_bound g =
  List.filter_map
    (fun k ->
      let a = 4096 lsl k in
      let n = (g / 2) - a in
      if n >= 1 && alignment n = a then Some n else None)
    (List.init 40 Fun.id)

(* Sizes at the bound the largest free range sets, and at the space's length. *)
let edges =
  among int space (fun m ->
      let g = largest_gap m in
      [ g / 4; (g / 4) + 1; g / 2; m.length; m.length + 1 ] @ at_bound g)

let bases =
  Gen.of_list ~pp:pp_hex
    [ 0; 0; 4096; 0x1234; 1 lsl 40; 0x2000_0000_0000; -1; min_int ]

let lengths =
  Gen.of_list ~pp:pp_hex
    [
      0;
      24;
      4096;
      0x3000;
      0x1_0000;
      0x1_0123;
      1 lsl 20;
      1 lsl 20;
      1 lsl 24;
      1 lsl 44;
      max_int;
      -1;
    ]

let sizes =
  Gen.frequency
    [
      (5, Gen.int_range 1 0x1_0000);
      (3, Gen.map (fun k -> k * 4096) (Gen.int_range 1 256));
      (1, Gen.int_range 1 (1 lsl 22));
      ( 2,
        Gen.of_list ~pp:pp_hex
          [
            0;
            -1;
            min_int;
            1;
            4095;
            4096;
            4097;
            1 lsl 20;
            (1 lsl 20) + 1;
            max_int;
          ] );
    ]

let aligns =
  Gen.of_list
    ~pp:
      (Format.pp_print_option
         ~none:(fun ppf () -> Format.fprintf ppf "_")
         pp_hex)
    [
      None;
      None;
      None;
      None;
      Some 1;
      Some 4096;
      Some 0x1_0000;
      Some (2 lsl 20);
      Some (1 lsl 30);
      Some (1 lsl 61);
      Some 0;
      Some 3;
      Some 0x3000;
      Some (-4096);
      Some min_int;
    ]

(* Spaces that hold ranges, for the test on two domains: a create runs in a
   program's prefix only, so a refused one tests nothing there, and a space with
   no room leaves no range for [remember] to take. *)
let roomy =
  Gen.pair
    (Gen.of_list ~pp:pp_hex [ 0; 4096; 1 lsl 40; 0x2000_0000_0000 ])
    (Gen.of_list ~pp:pp_hex [ 0x3000; 0x1_0000; 0x1_0123; 1 lsl 20; 1 lsl 44 ])

let alloc align n s = alloc_held ?align s n

let commands ~covered ~spaces =
  let alloc_judge = alloc_judge ~covered in
  [
    command "create"
      (spaces @-> makes space)
      create
      (fun (base, n) -> Space.create ~base n);
    command "alloc"
      (aligns @-> sizes @-> space ^-> judges (option hex))
      alloc_judge alloc;
    command "alloc"
      (space ^-> edges ^-> judges (option hex))
      (fun m n got -> alloc_judge None n m got)
      (fun s n -> alloc_held s n);
    command "free" (space ^-> live ^-> returns unit) free free_at;
    command "remember"
      (space ^-> live ^-> makes allocation)
      (fun m a -> (m, a, List.assoc a m.turns))
      (fun s a -> (s, a, hold_at s a));
    command "free a remembered hold"
      (allocation ^-> returns unit)
      (fun (m, a, turn) ->
        if List.assoc_opt a m.turns = Some turn then free m a)
      (fun (_, _, x) -> Space.free x);
  ]

(* Creating a space allocates its handle and nothing else, whatever its length,
   so a vendor whose GPUs a program never drives costs it nothing. *)
let test_lazy () =
  let words f =
    let b0 = Gc.allocated_bytes () in
    let b1 = Gc.allocated_bytes () in
    ignore (Sys.opaque_identity (f ()));
    let b2 = Gc.allocated_bytes () in
    int_of_float (b2 -. b1 -. (b1 -. b0)) / (Sys.word_size / 8)
  in
  let small = words (fun () -> Space.create ~base:0 4096) in
  let large = words (fun () -> Space.create ~base:(1 lsl 40) (1 lsl 44)) in
  at_most ~msg:"words of a space" int ~than:32 large;
  equal ~msg:"whatever its length" int small large

(* Once every range is freed the whole space is one free range again: a quarter
   of it, aligned to a quarter, needs all of it. *)
let test_whole () =
  let s = Space.create ~base:(1 lsl 40) (1 lsl 24) in
  let rec take acc =
    match Space.alloc s holder 0x3000 with
    | Some x -> take (x :: acc)
    | None -> acc
  in
  let all = take [] in
  greater ~msg:"ranges taken" int ~than:16 (List.length all);
  List.iter Space.free (List.rev all);
  is_some ~msg:"a quarter" (Space.alloc s holder (1 lsl 22))

(* A range two holders hold goes back with the second's give-back, and a
   share of a hold given back holds nothing, as a release on another thread
   leaves it. *)
let test_shared () =
  let s = Space.create ~base:(1 lsl 40) (1 lsl 24) in
  let h = Space.holder () and h' = Space.holder () in
  let x = Option.get (Space.alloc s h 0x3000) in
  let y = require_some ~msg:"a share" (Space.share h' x) in
  equal ~msg:"the shared range" hex (Space.address x) (Space.address y);
  Space.release h;
  equal ~msg:"held by the second holder" int 1 (Space.taken s);
  is_none ~msg:"a share of a hold given back" (Space.share h' x);
  Space.free y;
  equal ~msg:"back with the second" int 0 (Space.taken s);
  is_none ~msg:"a share once both went" (Space.share h' y)

(* Interrupts *)

(* An exception at any allocation of an alloc or a free, the first making the
   allocator, leaves the space's lock free: the next alloc and free run, and the
   thread holds what it held before. The interrupted call may leave its ranges
   taken. *)
let test_interrupted () =
  let before = Rig_edge_primitives.holds () in
  let rec from n =
    let s = Space.create ~base:(1 lsl 40) (1 lsl 30) in
    let calls () = Option.iter Space.free (Space.alloc s holder 0x3000) in
    let raised = Rig_support.interrupted n calls in
    equal
      ~msg:(Printf.sprintf "locks held, interrupted at %d" n)
      int before (Rig_edge_primitives.holds ());
    calls ();
    if raised then from (n + 1)
  in
  from 1

(* A space of 256 pages whose first 127 pages are taken one at a time and the
   odd ones freed again: an alloc of two pages then carves the gap below its
   aligned start and the tail above it, a free of a taken page joins both
   neighbours, and its blocks are as many as make the allocator's table grow
   at the alloc's first new block. Is the space and its taken pages. *)
let fragmented () =
  let s = Space.create ~base:(1 lsl 40) (256 * 4096) in
  let pages = List.init 127 (fun _ -> Option.get (Space.alloc s holder 4096)) in
  let freed, taken =
    List.partition (fun x -> (Space.address x lsr 12) land 1 = 1) pages
  in
  List.iter Space.free freed;
  (s, taken)

(* Whether [s], a space of 256 pages, is whole: half of it, aligned to half,
   is served only from a free block of all 256 pages. The fit bound of
   tlsf.mli does not promise this; the allocator's search does, since it
   looks for a block of the request plus its alignment less one, 256 pages
   less a byte, rounded up to its class, which starts at 256 pages. *)
let whole s = Space.alloc s holder (128 * 4096)

(* An exception at any allocation of an alloc or a free loses no range: once
   the ranges that the calls hold are freed, the interrupted one's if it
   returned, the space is whole again. A free interrupted before it returned
   may have freed its range or not, and a second free of it frees nothing
   more. *)
let test_lost () =
  let rec from n =
    let s, taken = fragmented () in
    let got = ref None in
    let reached =
      Rig_support.interrupted n (fun () -> got := Space.alloc s holder 0x2000)
    in
    List.iter Space.free taken;
    Option.iter Space.free !got;
    let at what = Printf.sprintf "whole, %s interrupted at %d" what n in
    is_some ~msg:(at "an alloc") (whole s);
    let s, taken = fragmented () in
    let x = List.nth taken 10 in
    let reached' = Rig_support.interrupted n (fun () -> Space.free x) in
    List.iter Space.free taken;
    is_some ~msg:(at "a free") (whole s);
    if reached || reached' then from (n + 1)
  in
  from 1

(* A callback at any allocation of an alloc calls alloc on the same space.
   Inside the space's lock, where the thread holds one more lock, the
   callback's alloc takes the lock its thread holds and raises
   [Invalid_argument], having taken nothing; outside it, it runs. Once both
   calls' ranges are freed the space is whole. *)
let test_reentered () =
  let seen = Rig_support.seen () in
  let rec from n =
    let s, taken = fragmented () in
    let got = ref None and again = ref None in
    let landing, landed =
      Rig_support.landing seen "Rig_pci: a lock taken again" (fun () ->
          again := Space.alloc s holder 4096)
    in
    let reached =
      Rig_support.interrupted ~landing n (fun () ->
          got := Space.alloc s holder 0x2000)
    in
    let at what = Printf.sprintf "%s, called at %d" what n in
    ignore (Rig_support.answered (at "the callback's alloc") landed
      : bool option);
    List.iter Space.free taken;
    List.iter (Option.iter Space.free) [ !got; !again ];
    is_some ~msg:(at "whole") (whole s);
    if reached then from (n + 1)
  in
  from 1;
  Rig_support.covered seen

let () =
  exit
  @@ run "rig_pci.space"
       [
         group ~timeout:patience "ranges"
           [
             stateful "allocations stay apart and within the fit bound"
               ~count:300
               (commands ~covered:true ~spaces:(Gen.pair bases lengths));
             stateful "two domains allocate and free at once" ~domains:2
               ~count:50
               (commands ~covered:false ~spaces:roomy);
             test "creating a space allocates nothing but its handle" test_lazy;
             test "freeing every range makes the space whole" test_whole;
             test
               "a range goes back with its last holder, and a share of a hold \
                given back is None"
               test_shared;
             test "an exception at any allocation of a call leaves the lock \
                   free"
               test_interrupted;
             test "an exception at any allocation of a call loses no range"
               test_lost;
             test
               "a callback's alloc inside the space's lock raises \
                Invalid_argument, having taken nothing, and one outside it \
                runs"
               test_reentered;
           ];
       ]
