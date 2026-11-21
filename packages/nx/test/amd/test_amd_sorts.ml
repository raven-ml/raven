(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.amd's sorts and numberings. Their plans, on any machine: no launch
   in the slow band, and a numbering kept within 64 MiB, whatever its
   launches. Then on the machine's AMD GPU 0, skipping without one: each
   path nx_amd.mli names gives nx.cpu's bits, whose order test_sort and
   test_unique check against a reference: slices sorted in LDS, selected
   then sorted, and by radix passes of one part a slice, of several, and
   of several tiles a part, of every item width, with the positions
   holding the items, and with passes skipped where every key shares a
   byte; groups that take several units in turn; workspace and tickets of
   a call's own; numberings of many rows and of more than 8 columns. Then
   the workspace nx_amd.mli states they take. *)

open Windtrap
open Elements
module Support = Nx_kernels_support
module Plan = Nx_amd.Plan

let amd () =
  match Support.amd () with
  | Some b -> b
  | None -> skip ~reason:"no AMD GPU nx.amd computes on" ()

(* The longest slice sorted in LDS: 4096 elements, 2048 of complex128. *)
let cap (D.Any dt) = if D.bits dt = 128 then 2048 else 4096

(* The items of a radix pass's work-item: 16 of keys of 32 bits or fewer,
   8 of elements of 64 bits or fewer, 5 of wider ones; a tile is 256
   work-items' items. *)
let tile (D.Any dt) =
  256 * if D.bits dt <= 32 then 16 else if D.bits dt <= 64 then 8 else 5

let tiles d n = (n + tile d - 1) / tile d

(* The bounds a case plans under: the parts a pass cuts a slice into, the
   groups of a launch, and whether its workspace and tickets are the
   call's own. *)
type bounds = { parts : int; groups : int; own : bool }

let pp_bounds ppf b =
  Format.fprintf ppf "%d parts, %d groups%s" b.parts b.groups
    (if b.own then ", own workspace and tickets" else "")

let library = { parts = 256; groups = 1024; own = false }

(* Bounds under which a slice of a few tiles takes several parts of
   several tiles each, in the call's own workspace and tickets. *)
let tiled = { parts = 5; groups = 3; own = true }

let small =
  [
    { parts = 3; groups = 2; own = false };
    { parts = 1; groups = 1; own = true };
    tiled;
  ]

let bounds =
  Gen.frequency
    ((3, Gen.constant library) :: List.map (fun b -> (1, Gen.constant b)) small)

(* Runs [f] under [b], then restores the library's bounds. *)
let under b f =
  Fun.protect
    ~finally:(fun () ->
      Nx_amd.Sort.bound_sorts ~parts:256 ~groups:1024;
      Nx_amd.bound_fresh ~workspace:Nx_amd.Plan.kept_workspace
        ~tickets:Nx_amd.Plan.kept_tickets)
    (fun () ->
      Nx_amd.Sort.bound_sorts ~parts:b.parts ~groups:b.groups;
      if b.own then Nx_amd.bound_fresh ~workspace:0 ~tickets:0;
      f ())

(* Plans *)

(* An array of [dt] and shape [s] that one element on the host
   broadcasts: a plan reads layouts alone. *)
let broadcast dt s =
  let one = A.create Rig.host dt (Iarray.of_array (Array.map (fun _ -> 1) s)) in
  Option.get (A.move (M.Broadcast (Iarray.of_array s)) one)

let planned = function
  | Ok (Some c) -> c
  | Ok None -> failf "no plan"
  | Error a -> failf "refused: %a" Nx_array_support.pp_answer a

(* No launch of [c] lies in the slow band. *)
let out_of_band (c : Plan.t) =
  Array.iter
    (fun (l : Plan.launch) ->
      let g, _, _ = l.groups in
      let msg = Printf.sprintf "%s, %d groups" (Nx_amd.Kernels.name l.kernel) g in
      equal ~msg bool false (Plan.in_band l.kernel g))
    c.launches

type band = { d : D.any; outer : int; n : int; down : bool }

let pp_band ppf c =
  Format.fprintf ppf "sort%s of %a [%d; %d]"
    (if c.down then " descending" else "")
    pp_dtype c.d c.outer c.n

(* Radix sorts of [outer] slices whose parts fill 256 or 512 groups, and
   any others. *)
let band_gen =
  let open Gen in
  let* d =
    of_list ~pp:pp_dtype
      D.[ Any Float32; Any Float64; Any Complex128; Any Uint8; Any Float16 ]
  in
  let* outer = of_list ~pp:Format.pp_print_int [ 1; 2; 4; 8 ] in
  let* n =
    frequency
      [
        (2, map (fun g -> g * tile d / outer) (of_list ~pp:Format.pp_print_int [ 256; 512 ]));
        (1, int_range 1 (1 lsl 21));
      ]
  in
  let+ down = bool in
  { d; outer; n; down }

(* sort.mli: no launch's groups lie in the slow band. *)
let law_sort_band c =
  let (D.Any dt) = c.d in
  let parts = Int.min (tiles c.d c.n) (256 / c.outer) in
  cover "by radix, parts in 256 groups"
    (c.n > cap c.d && c.outer * parts = 256);
  let s = [| c.outer; c.n |] in
  let values = broadcast dt s and positions = broadcast D.Int64 s in
  out_of_band
    (planned
       (Nx_amd.Sort.sort ~axis:1 ~descending:c.down ~values ~positions
          (broadcast dt s)))

let pp_numbering ppf (n, w) = Format.fprintf ppf "unique of [%d; %d]" n w

(* Numberings of no row, a few or many, of up to 12 columns. *)
let numbering_gen =
  let open Gen in
  let* n =
    frequency
      [ (2, constant 0); (2, int_range 1 300); (4, int_range 4097 (1 lsl 21)) ]
  in
  let+ w = int_range 0 12 in
  (n, w)

(* nx_amd.mli: a numbering is kept iff its bytes are at most 64 MiB,
   whatever its launches; sort.mli: no launch's groups lie in the slow
   band. *)
let law_unique_plan (n, w) =
  cover "more than 8 columns" (w > 8);
  cover "no row, more than 8 columns" (n = 0 && w > 8);
  cover "no column" (w = 0);
  let index s = broadcast D.Int64 s in
  let c =
    planned
      (Nx_amd.Sort.unique ~ids:(index [| n |]) ~first:(index [| n |])
         ~counts:(index [| n |]) ~count:(index [||]) (index [| n; w |]))
  in
  let launches = Array.length c.launches in
  cover "at most 8 launches" (launches <= 8);
  cover "more than 8 launches" (launches > 8);
  let msg = Printf.sprintf "kept, %d launches, %d bytes" launches c.bytes in
  equal ~msg bool (c.bytes <= Nx_amd.Plan.kept_submissions) c.kept;
  out_of_band c

(* Sorts *)

type case = {
  axis : int;
  descending : bool;
  k : int option;
  x : A.any;
  narrow : bool;
  views : string list;
  bounds : bounds;
}

let pp_case ppf c =
  Format.fprintf ppf "sort%s along %d%s of %a %a%s (%s), %a"
    (if c.descending then " descending" else "")
    c.axis
    (match c.k with None -> "" | Some k -> Printf.sprintf " keeping %d" k)
    pp_dtype (dtype_of c.x) pp_ints (shape_of c.x)
    (if c.narrow then ", narrow" else "")
    (String.concat ", " c.views)
    pp_bounds c.bounds

(* An operand of [d] drawn from [seed]. A narrow one takes its elements
   from four that differ in their first byte alone, so that a radix sort
   skips the passes of the bytes every key shares and runs another. Any
   other from a few elements one time in two, so that ties are common,
   else at random. *)
let drawn d s v ~narrow seed =
  let rs = Random.State.make [| seed |] in
  if narrow then
    let e0 = element d rs in
    let pool =
      Array.init 4 (fun _ ->
          let e = element d rs in
          String.mapi (fun i c -> if i = 0 then e.[0] else c) e0)
    in
    operand d s v (fun _ -> pool.(Random.State.int rs 4))
  else
    let pool = Array.init 5 (fun _ -> element d rs) in
    operand d s v (fun _ ->
        if Random.State.bool rs then pool.(Random.State.int rs 5)
        else element d rs)

(* A shape of slices of [n] along the last axis or the first. *)
let laid n =
  let open Gen in
  let* outer = frequency [ (2, constant 1); (1, int_range 2 5) ] in
  let+ last = bool in
  if outer = 1 then (0, [| n |])
  else if last then (1, [| outer; n |])
  else (0, [| n; outer |])

let case_of d laid k ~narrow =
  let open Gen in
  let axis, s = laid in
  let* descending = bool in
  let* v = view_of (Array.length s) in
  let* bounds = bounds in
  let+ seed = int in
  let x = drawn d s v ~narrow seed in
  { axis; descending; k; x; narrow; views = view_names v; bounds }

(* Slices sorted in LDS, at its edge, past it keeping at most a quarter of
   it or more, one slice or a few, along the last axis or the first. *)
let any_gen =
  let open Gen in
  let* d = of_list ~pp:pp_dtype D.all in
  let cap = cap d in
  let* n =
    frequency
      [
        (2, int_range 1 300);
        (1, of_list ~pp:Format.pp_print_int [ cap - 1; cap; cap + 1 ]);
        (3, int_range (cap + 1) 20_000);
      ]
  in
  let* laid = laid n in
  let* k =
    frequency
      [
        (2, constant None);
        (2, map Option.some (int_range 1 (Int.min n (cap / 4))));
        (1, map Option.some (int_range 1 n));
      ]
  in
  let* narrow = frequency [ (3, constant false); (1, constant true) ] in
  case_of d laid k ~narrow

(* Radix sorts of 8-, 12- and 24-byte items, half of them narrow, a
   quarter under each bound. *)
let radix_gen =
  let open Gen in
  let* d = of_list ~pp:pp_dtype D.[ Any Float32; Any Int32; Any Complex128 ] in
  let cap = cap d in
  let* n = int_range (cap + 1) 20_000 in
  let* laid = laid n in
  let* k =
    frequency
      [ (3, constant None); (1, map Option.some (int_range ((cap / 4) + 1) n)) ]
  in
  let* narrow = bool in
  let* c = case_of d laid k ~narrow in
  let+ bounds = of_list ~pp:pp_bounds (library :: small) in
  { c with bounds }

(* Radix sorts under [tiled] of slices side by side along the last axis,
   every element kept, of 4-, 8- and 24-byte items, half of them narrow:
   each of two to four slices of more tiles than the parts it is cut
   into, so each draw passes several tiles a part, and at 4-byte keys the
   positions hold the items. *)
let tiled_gen =
  let open Gen in
  let* d = of_list ~pp:pp_dtype D.[ Any Float32; Any Int32; Any Complex128 ] in
  let* outer = int_range 2 4 in
  let parts = (tiled.parts + outer - 1) / outer in
  let* n = int_range ((parts * tile d) + 1) 20_000 in
  let* narrow = bool in
  let+ c = case_of d (1, [| outer; n |]) None ~narrow in
  { c with bounds = tiled }

let cases =
  Gen.with_pp pp_case
    (Gen.frequency [ (3, any_gen); (2, radix_gen); (1, tiled_gen) ])

let sort_on (b : Support.backend) c =
  let module K = (val b.kernels) in
  let (A.Any x) = c.x in
  let y = Array.copy (shape_of c.x) in
  Option.iter (fun k -> y.(c.axis) <- k) c.k;
  let y = Iarray.of_array y in
  let x = Support.on b x in
  let values = A.create b.device (A.dtype x) y in
  let positions = A.create b.device D.Int64 y in
  Rig.Buffer.fill (A.buffer values) '\xA5';
  Rig.Buffer.fill (A.buffer positions) '\xA5';
  match K.sort ~axis:c.axis ~descending:c.descending ~values ~positions x with
  | A.Done -> [| A.Any values; A.Any positions |]
  | r -> failf "%s answered %a" b.name Nx_array_support.pp_answer r

let law_sort c =
  let b = amd () in
  let d = dtype_of c.x and s = shape_of c.x in
  let n = s.(c.axis) in
  let keep = Option.value c.k ~default:n in
  let slices = Array.fold_left ( * ) 1 s / n in
  let radix = n > cap d && keep > cap d / 4 in
  let tiles = tiles d n in
  let most = Int.max 1 ((c.bounds.parts + slices - 1) / slices) in
  let parts = Int.min tiles most in
  let several_tiles = radix && tiles > parts && parts > 1 in
  let (D.Any dt) = d in
  (* The positions hold 8-byte items: 4-byte keys, every element kept, a
     slice's positions side by side. *)
  let reuse = radix && D.bits dt = 32 && keep = n && (slices = 1 || c.axis = 1) in
  cover "in LDS" (n <= cap d);
  cover "in LDS, several slices a group" (n <= cap d / 2 && slices > 1);
  cover "selected" (n > cap d && keep <= cap d / 4);
  cover "selected, several slices" (n > cap d && keep <= cap d / 4 && slices > 1);
  cover "by radix, one part a slice" (radix && parts = 1);
  cover "by radix, several parts a slice" (radix && parts > 1);
  cover "by radix, several tiles a part" several_tiles;
  cover "by radix, several tiles a part, own tickets" (several_tiles && c.bounds.own);
  cover "by radix, several slices" (radix && slices > 1);
  cover "by radix, 24-byte items" (radix && D.bits dt = 128);
  cover "by radix, 24-byte items, several tiles a part, own tickets"
    (radix && D.bits dt = 128 && several_tiles && c.bounds.own);
  cover "by radix, positions hold the items" reuse;
  cover "by radix, positions hold the items, several tiles a part, own tickets"
    (reuse && several_tiles && c.bounds.own);
  cover "by radix, a pass skipped, another run" (radix && c.narrow && D.bits dt >= 16);
  cover "by radix, narrow, several tiles a part, own tickets"
    (radix && c.narrow && D.bits dt >= 16 && several_tiles && c.bounds.own);
  cover "groups that take several units" (c.bounds.groups < 4 && slices > 1);
  cover "own workspace and tickets" c.bounds.own;
  cover "sub-byte" (D.bits dt < 8);
  cover "16-byte keys" (D.bits dt = 128);
  cover "8-byte keys by radix" (D.bits dt = 64 && radix);
  cover "descending" c.descending;
  under c.bounds (fun () -> agree_bits Support.cpu b (fun b -> sort_on b c))

(* Numberings *)

type ucase = { keys : A.any; m : int; uviews : string list; ubounds : bounds }

let pp_ucase ppf c =
  Format.fprintf ppf "unique of %a keeping %d (%s), %a" pp_ints
    (shape_of c.keys) c.m
    (String.concat ", " c.uviews)
    pp_bounds c.ubounds

(* Rows of [w] words, each word from a few values or at random; no row
   one time in six. *)
let ucase_gen =
  let open Gen in
  let* n =
    frequency
      [
        (2, int_range 1 300);
        (2, int_range 4097 20_000);
        (1, constant 1);
        (1, constant 0);
      ]
  in
  let* w = frequency [ (3, int_range 1 3); (1, constant 0); (1, int_range 9 11) ] in
  let* m = frequency [ (2, int_range 0 (n + 2)); (1, constant n) ] in
  let* v = view_of 2 in
  let* ubounds = bounds in
  let+ seed = int in
  let rs = Random.State.make [| seed |] in
  let pool = Array.init 3 (fun _ -> Random.State.bits64 rs) in
  let keys =
    int64s [| n; w |] v (fun () ->
        if Random.State.int rs 4 = 0 then Random.State.bits64 rs
        else pool.(Random.State.int rs 3))
  in
  { keys; m; uviews = view_names v; ubounds }

let ucases = Gen.with_pp pp_ucase ucase_gen

let unique_on (b : Support.backend) c =
  let module K = (val b.kernels) in
  let s = shape_of c.keys in
  let keys = Support.on b (A.expect D.Int64 c.keys) in
  let make n = A.create b.device D.Int64 (Iarray.of_array n) in
  let ids = make [| s.(0) |] and first = make [| c.m |] in
  let counts = make [| c.m |] and count = make [||] in
  List.iter
    (fun a -> Rig.Buffer.fill (A.buffer a) '\xA5')
    [ ids; first; counts; count ];
  match K.unique ~ids ~first ~counts ~count keys with
  | A.Done -> [| A.Any ids; A.Any first; A.Any counts; A.Any count |]
  | r -> failf "%s answered %a" b.name Nx_array_support.pp_answer r

let law_unique c =
  let b = amd () in
  let s = shape_of c.keys in
  cover "no row" (s.(0) = 0);
  cover "no column" (s.(1) = 0 && s.(0) > 0);
  cover "rows of several parts" (s.(0) > 4096 && s.(1) > 0);
  cover "more than 8 columns" (s.(1) > 8);
  cover "groups that take several units" (c.ubounds.groups < 4 && s.(0) > 4096);
  cover "own workspace and tickets" c.ubounds.own;
  under c.ubounds (fun () -> agree_bits Support.cpu b (fun b -> unique_on b c))

(* Memory *)

(* nx_amd.mli: a sort takes workspace of at most 66 bytes an element and
   512 KiB, and a numbering at most 65 bytes a row, 16 KiB a column and
   544 KiB. Each sorts by radix, a float32 sort with 8-byte items, a
   complex128 sort with 24-byte items and 16-byte keys. Each workspace is
   past the 64 MiB the device keeps, a buffer of the call's own, which the
   bound counts in the driver's units ({!Rig.allocation}); its tickets, a
   ticket a slice and one, the device keeps. *)
let test_sort_workspace (type v s) (dt : (v, s) D.t) n () =
  let b = amd () in
  let module K = (val b.kernels) in
  let x = A.create b.device dt [| n |] in
  let values = A.create b.device dt [| n |] in
  let positions = A.create b.device D.Int64 [| n |] in
  Kernels_cases.loop_held b ~calls:4
    ~bound:(Rig.allocation b.device ((66 * n) + (512 * 1024)))
    (fun () ->
      equal answer A.Done (K.sort ~axis:0 ~descending:false ~values ~positions x);
      Rig.Buffer.wait (A.buffer values) Read_write)

let test_unique_workspace () =
  let b = amd () in
  let module K = (val b.kernels) in
  let n = 1 lsl 20 and w = 3 and d = b.device in
  let index s = A.create d D.Int64 s in
  let keys = index [| n; w |] in
  let ids = index [| n |] and first = index [| n |] and counts = index [| n |] in
  let count = index [||] in
  Kernels_cases.loop_held b ~calls:4
    ~bound:(Rig.allocation d ((65 * n) + (16 * 1024 * w) + (544 * 1024)))
    (fun () ->
      equal answer A.Done (K.unique ~ids ~first ~counts ~count keys);
      Rig.Buffer.wait (A.buffer count) Read_write)

let () =
  exit
    (Windtrap.run "nx.amd sorts"
       [
         group "plans"
           [
             prop "of sorts lie out of the slow band"
               (Gen.with_pp pp_band band_gen)
               law_sort_band;
             prop ~count:300
               "of numberings are kept within 64 MiB, out of the slow band"
               (Gen.with_pp pp_numbering numbering_gen)
               law_unique_plan;
           ];
         group "sorts"
           [
             prop ~count:300 "give nx.cpu's values and positions on every path"
               cases law_sort;
             test "of float32 hold 66 bytes an element"
               (test_sort_workspace D.Float32 (1 lsl 24));
             test "of complex128 hold 66 bytes an element"
               (test_sort_workspace D.Complex128 (1 lsl 20));
           ];
         group "numberings"
           [
             prop ~count:200 "give nx.cpu's numbering on every path" ucases
               law_unique;
             test "hold 65 bytes a row and 16 KiB a column" test_unique_workspace;
           ];
       ])
