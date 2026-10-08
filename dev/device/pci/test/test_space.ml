(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Virtual address spaces against the list of the ranges they handed out. *)

open Windtrap
open Device_pci
open Device_pci_support

(* The model *)

type model = {
  base : int;
  length : int;
  mutable live : (int * int) list;  (** (address, addresses), newest first. *)
  mutable freed : int list;  (** Addresses freed and not handed out again. *)
}

let create (base, n) =
  if base < 0 || n < 0 || n > max_int - base then invalid_arg "Space.create";
  { base; length = n; live = []; freed = [] }

(* The alignment [alloc] promises: the size's and [align]'s. *)
let alignment ?(align = 4096) n = max align (pow2_floor n)
let largest_gap m = largest_gap m.base (m.base + m.length) m.live

(* An allocation is aligned, inside the space and apart from every live range.
   [None] is accepted only where the fit bound does not promise a range: no free
   range of [2 * (n + a)] addresses. *)
let alloc_judge align n m got =
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
      if fits ~gap n a then
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
      m.freed <- List.filter (( <> ) x) m.freed

let free m x =
  if not (List.mem_assoc x m.live) then invalid_arg "Space.free";
  m.live <- List.remove_assoc x m.live;
  m.freed <- x :: m.freed

(* Commands *)

let space =
  abstract "s" ~invariant:(fun m s ->
      equal ~msg:"base" hex m.base (Space.base s);
      equal ~msg:"length" int m.length (Space.length s))

let live = among hex space (fun m -> List.map fst m.live)

(* Addresses [free] must refuse, beside live ones: inside or just after a live
   range, freed already, around the space. *)
let starts =
  among hex space (fun m ->
      List.concat_map (fun (a, n) -> [ a; a + 1; a + n ]) m.live
      @ m.freed
      @ [ m.base - 1; m.base + m.length ])

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

let alloc align n s = Space.alloc ?align s n

let commands =
  [
    command "create"
      (Gen.pair bases lengths @-> makes space)
      create
      (fun (base, n) -> Space.create ~base n);
    command "alloc"
      (aligns @-> sizes @-> space ^-> judges (option hex))
      alloc_judge alloc;
    command "alloc"
      (space ^-> edges ^-> judges (option hex))
      (fun m n got -> alloc_judge None n m got)
      (fun s n -> Space.alloc s n);
    command "free" (space ^-> live ^-> returns unit) free Space.free;
    command "free" (space ^-> starts ^-> returns unit) free Space.free;
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
    match Space.alloc s 0x3000 with Some a -> take (a :: acc) | None -> acc
  in
  let all = take [] in
  greater ~msg:"ranges taken" int ~than:16 (List.length all);
  List.iter (Space.free s) (List.rev all);
  is_some ~msg:"a quarter" (Space.alloc s (1 lsl 22))

let () =
  exit
  @@ run "device_pci Space"
       [
         group ~timeout:patience "ranges"
           [
             stateful "allocations stay apart and within the fit bound"
               ~count:300 commands;
             stateful "two domains allocate at once" ~domains:2 ~count:50
               commands;
             test "creating a space allocates nothing but its handle" test_lazy;
             test "freeing every range makes the space whole" test_whole;
           ];
       ]
