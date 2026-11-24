(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.metal's gathers, assemblies and folds, beside the laws the shared suites
   hold: their bits against nx.cpu's through every unit a move takes and in
   launches cut by the budget, a fold's ranges of taps, and elements narrower
   than a byte that share a word with another array's. test_index's gather laws
   do not run on nx.metal until it computes scatters; law_gather stands in for
   them. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module K = Nx_metal.Kernels
module Plan = Nx_metal.Plan
module Support = Nx_kernels_support
module S = Nx_metal_support
open Elements
open Index_cases

let backend () =
  match Support.metal () with
  | Some b -> b
  | None -> skip ~reason:"no Metal GPU nx.metal computes on" ()

let some = function Some y -> [| y |] | None -> [||]

(* The unit of each move or fill launch [ls] holds, in bits. *)
let units ls =
  List.filter_map
    (fun (t : Plan.traced) ->
      if t.kernel <> K.Move && t.kernel <> K.Move_fill then None
      else
        let how = Int64.to_int (Bytes.get_int64_le t.words (8 * K.move_how)) in
        Some (1 lsl ((how lsr K.how_unit) land 31)))
    ls

let moves ls = List.length (units ls)

(* The fill launches among [ls]. *)
let fills ls =
  List.length (List.filter (fun (t : Plan.traced) -> t.kernel = K.Move_fill) ls)

(* [f ()] with each launch's time bounded by [s]. *)
let bounded s f =
  Plan.bound_budget s;
  Fun.protect ~finally:(fun () -> Plan.bound_budget Plan.budget) f

(* [f ()] with every launch its own part: each launch boundary, also one inside
   a range gather's levels or a fold's ranges, is a cut. *)
let parted f = S.parted ~launches:1 ~seconds:Plan.part_budget f

(* Whether [c] draws a position outside its axis. *)
let outside c =
  let d = (shape_of c.x).(c.axis) in
  Array.exists (fun p -> p < 0L || p >= Int64.of_int d) (positions c.idx)

let gathers = Gen.with_pp pp_gather (gather_gen ())

(* A gather has nx.cpu's bits: its moved elements' and zero bits outside the
   axis, at every dtype, through any views. *)
let law_gather c =
  let metal = backend () in
  let (D.Any dt) = dtype_of c.x in
  let call b = some (gather_on b ~axis:c.axis c.idx c.x) in
  cover "an element of 1 or 4 bits" (D.bits dt < 8);
  cover "a position outside the axis" (outside c);
  cover "a long axis" (Array.exists (fun e -> e > 5) (shape_of c.x));
  cover "no element" (total (shape_of c.idx) = 0);
  cover "a source of a negative stride"
    (let (A.Any x) = c.x in
     Iarray.exists (fun s -> s < 0) (L.strides (A.layout x)));
  agree_bits Support.cpu metal call

(* At a budget a few units fill, a gather runs as many launches, which give the
   bits of one. *)
let law_cut c =
  let metal = backend () in
  let call b = some (gather_on b ~axis:c.axis c.idx c.x) in
  let (), ls =
    Plan.traced (fun () ->
        bounded 5e-6 (fun () -> agree_bits Support.cpu metal call))
  in
  cover "several launches" (moves ls > 1)

(* The widest gather the generator draws, 40 positions along the first axis of
   [3; 3000] float32 elements: 120,000 elements, which a few units of 5 us do
   not hold, so that each run reaches law_cut's cover. Drawn, a gather takes
   several launches about one time in twenty. *)
let cut_example =
  let rs = Random.State.make [| 1 |] in
  let d = D.Any D.Float32 in
  {
    axis = 0;
    idx = int64s [| 40; 3000 |] (plain 2) (fun () -> position rs 3);
    x = operand d [| 3; 3000 |] (plain 2) (fun _ -> element d rs);
    gviews = [];
  }

(* Rows: a take of [m] rows of [w] elements of [dt] from [n], the positions
   stepping 0 along each row, moves [unit] bits a thread. *)
let rows (dt, w, unit) =
  let metal = backend () in
  let n = 6 and m = 9 in
  let rs = Random.State.make [| w |] in
  let x = operand dt [| n; w |] (plain 2) (fun _ -> element dt rs) in
  let p = Array.init m (fun i -> Int64.of_int ((i * 5 mod (n + 3)) - 1)) in
  let at = ref 0 in
  let idx =
    int64s [| m; w |]
      { (plain 2) with broadcast = [ 1 ] }
      (fun () ->
        let v = p.(!at) in
        incr at;
        v)
  in
  let call b = some (gather_on b ~axis:0 idx x) in
  let (), ls = Plan.traced (fun () -> agree_bits Support.cpu metal call) in
  equal (list int) [ unit ] (units ls)

let row_cases =
  [
    (D.Any D.Float32, 64, 128);
    (D.Any D.Bfloat16, 64, 128);
    (D.Any D.Float32, 3, 32);
    (D.Any D.Float32, 2, 64);
    (D.Any D.Float64, 1, 64);
    (D.Any D.Int8, 3, 8);
    (D.Any D.Uint8, 6, 16);
    (D.Any D.Complex128, 8, 128);
    (D.Any D.Int4, 64, 4);
  ]

(* A gather of one int8 element into the first byte of a buffer of 16, aligned
   at both ends: its unit is a byte, which leaves the other 15. *)
let test_one_byte () =
  let metal = backend () in
  let init = A.of_array D.Int8 [| 16 |] (Array.init 16 (fun i -> i + 1)) in
  let whole = A.buffer (Support.on metal init) in
  let dst = A.v D.Int8 (L.contiguous [| 1 |]) whole in
  let x = Support.on metal (A.of_array D.Int8 [| 16 |] (Array.make 16 (-9))) in
  let idx = Support.on metal (A.of_array D.Int64 [| 1 |] [| 3L |]) in
  let module Kn = (val metal.kernels) in
  equal answer A.Done (Kn.gather ~axis:0 ~dst idx x);
  equal (array int)
    (Array.init 16 (fun i -> if i = 0 then -9 else i + 1))
    (A.to_array (Support.host (A.v D.Int8 (L.contiguous [| 16 |]) whole)))

(* A gather of float16 rows from a source 2 bytes into its buffer: no unit past
   2 bytes lies on both ends' addresses, so each unit is an element, which gives
   nx.cpu's bits. An element lies on a multiple of its bytes, so no unit is
   narrower than it. *)
let test_unaligned_source () =
  let metal = backend () in
  let call (b : Support.backend) =
    let rs = Random.State.make [| 3 |] in
    let bytes =
      A.of_array D.Uint8 [| 34 |]
        (Array.init 34 (fun _ -> Random.State.int rs 256))
    in
    let whole = A.buffer (Support.on b bytes) in
    let x =
      A.v D.Float16
        (L.contiguous [| 4; 4 |])
        (Rig.Buffer.view whole ~first:2 ~length:32)
    in
    let idx = Support.on b (A.of_array D.Int64 [| 3; 1 |] [| 2L; 0L; 3L |]) in
    let idx = A.v D.Int64 (L.v ~strides:[| 1; 0 |] [| 3; 4 |]) (A.buffer idx) in
    let dst = A.create b.device D.Float16 (Iarray.of_list [ 3; 4 ]) in
    let module Kn = (val b.kernels) in
    equal answer A.Done (Kn.gather ~axis:0 ~dst idx x);
    [| A.Any dst |]
  in
  let (), ls =
    Plan.traced (fun () ->
        let want = call Support.cpu in
        let got = call metal in
        equal (array string) (elements want.(0)) (elements got.(0)))
  in
  equal (list int) [ 16 ] (units ls)

(* Elements narrower than a byte *)

(* [n] int4 elements of the buffer [b] from its element [first]. *)
let int4s b ~first n = A.v D.Int4 (L.v ~offset:first ~strides:[| 1 |] [| n |]) b

(* The [n] bytes of the buffer [b]. *)
let bytes_of b n =
  A.to_array (Support.host (A.v D.Uint8 (L.contiguous [| n |]) b))

(* A gather into int4 elements 1 to 6 of a buffer of 8: the elements around
   them, in the words the destination shares, keep their bits. *)
let test_gather_shared () =
  let metal = backend () in
  let init = A.of_array D.Int4 [| 8 |] (Array.init 8 (fun i -> i - 4)) in
  let whole = A.buffer (Support.on metal init) in
  let was = bytes_of whole 4 in
  let dst = int4s whole ~first:1 6 in
  let x = Support.on metal (A.of_array D.Int4 [| 3 |] [| 5; -6; 7 |]) in
  let idx =
    Support.on metal (A.of_array D.Int64 [| 6 |] [| 2L; 0L; 3L; -1L; 1L; 2L |])
  in
  let module Kn = (val metal.kernels) in
  equal answer A.Done (Kn.gather ~axis:0 ~dst idx x);
  equal (array int) [| 7; 5; 0; 0; -6; 7 |] (A.to_array (Support.host dst));
  let now = bytes_of whole 4 in
  equal int (was.(0) land 0xF) (now.(0) land 0xF);
  equal int (was.(3) lsr 4) (now.(3) lsr 4)

(* x and an int4 destination in one buffer of 4 bytes, x its first 2 and dst its
   last 2, sharing a word but no byte: a gather's stores into dst's elements
   leave x's bytes as they were, which its threads read while others store. *)
let test_gather_beside () =
  let metal = backend () in
  let init = A.of_array D.Int4 [| 8 |] [| 1; -2; 3; -4; 5; -6; 7; -8 |] in
  let whole = A.buffer (Support.on metal init) in
  let view first = Rig.Buffer.view whole ~first ~length:2 in
  let x = A.v D.Int4 (L.contiguous [| 4 |]) (view 0) in
  let dst = A.v D.Int4 (L.contiguous [| 4 |]) (view 2) in
  let idx =
    Support.on metal (A.of_array D.Int64 [| 4 |] [| 3L; 2L; 1L; 0L |])
  in
  let module Kn = (val metal.kernels) in
  equal answer A.Done (Kn.gather ~axis:0 ~dst idx x);
  equal (array int) [| -4; 3; -2; 1 |] (A.to_array (Support.host dst));
  equal (array int) [| 1; -2; 3; -4 |] (A.to_array (Support.host x))

(* An assembly into int4 elements 1 to 6 of a buffer of 8, of a fill and a
   piece: the elements around them keep their bits. *)
let test_assemble_shared () =
  let metal = backend () in
  let init = A.of_array D.Int4 [| 8 |] (Array.init 8 (fun i -> i - 4)) in
  let whole = A.buffer (Support.on metal init) in
  let was = bytes_of whole 4 in
  let dst = int4s whole ~first:1 6 in
  let piece = Support.on metal (A.of_array D.Int4 [| 2 |] [| -7; 6 |]) in
  let region = Iarray.of_list [ { M.start = 2; count = 2; step = 1 } ] in
  let module Kn = (val metal.kernels) in
  equal answer A.Done
    (Kn.assemble ~fill:3 ~dst (Iarray.of_list [ (region, piece) ]));
  equal (array int) [| 3; 3; -7; 6; 3; 3 |] (A.to_array (Support.host dst));
  let now = bytes_of whole 4 in
  equal int (was.(0) land 0xF) (now.(0) land 0xF);
  equal int (was.(3) lsr 4) (now.(3) lsr 4)

(* Range gathers *)

type ranges = {
  lead : int;
  k : int;
  n : int;
  cells : int;
  size : int;
  rdt : D.any;
  rseed : int;
}

let pp_ranges ppf c =
  let (D.Any dt) = c.rdt in
  Format.fprintf ppf
    "%d rows of %d runs over %d of %d cells of %s, %d each (seed %d)" c.lead c.k
    c.n c.cells (D.name dt) c.size c.rseed

let ranges_gen =
  let open Gen in
  let+ lead = int_range 1 3
  and+ k =
    frequency [ (1, int_range 1 4); (2, int_range 5 16); (3, int_range 17 90) ]
  and+ n = int_range 0 12
  and+ cells = int_range 1 3
  and+ size = int_range 0 24
  and+ rdt =
    of_list ~pp:pp_dtype
      [ D.Any D.Float32; D.Any D.Int8; D.Any D.Int4; D.Any D.Complex128 ]
  and+ rseed = int in
  { lead; k; n; cells; size; rdt; rseed }

(* [c]'s starts and ends, a row of [k] each: ends mostly running sums of lengths
   0 to 3, else any, which need not ascend; starts around [0, n). *)
let ranges_data c =
  let rs = Random.State.make [| c.rseed |] in
  let sums = Random.State.bool rs in
  let ends =
    Array.init c.lead (fun _ ->
        let at = ref 0 in
        Array.init c.k (fun _ ->
            if sums then begin
              at := !at + Random.State.int rs 4;
              !at
            end
            else Random.State.int rs (c.size + 8) - 4))
  in
  let starts =
    Array.init c.lead (fun _ ->
        Array.init c.k (fun _ -> Random.State.int rs (c.n + 4) - 2))
  in
  (starts, ends, sums)

(* The source position of position [p] of a row by the contract: the start of
   its first run whose end exceeds [p], plus [p] less the end before; [None]
   past every end. *)
let source starts ends p =
  let k = Array.length ends in
  let rec find j =
    if j = k then None else if ends.(j) > p then Some j else find (j + 1)
  in
  Option.map
    (fun j -> starts.(j) + p - if j = 0 then 0 else ends.(j - 1))
    (find 0)

let ranges_on (b : Support.backend) c =
  let starts, ends, _ = ranges_data c in
  let rs = Random.State.make [| c.rseed + 1 |] in
  let s = Iarray.of_list [ c.lead; c.k ] in
  let ints a =
    Support.on b
      (A.of_array D.Int64 s
         (Array.map Int64.of_int (Array.concat (Array.to_list a))))
  in
  let (A.Any x) =
    operand c.rdt [| c.lead; c.n; c.cells |] (plain 3) (fun _ ->
        element c.rdt rs)
  in
  let x = Support.on b x in
  let dst =
    A.create b.device (A.dtype x) (Iarray.of_list [ c.lead; c.size; c.cells ])
  in
  let module Kn = (val b.kernels) in
  match Kn.gather_ranges ~dst ~starts:(ints starts) ~ends:(ints ends) x with
  | A.Done -> [| A.Any dst |]
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* The launches of [k] among [ls]. *)
let count_of k (ls : Plan.traced list) =
  List.length (List.filter (fun (t : Plan.traced) -> t.kernel = k) ls)

(* Levels of 4 runs: a row of more than 4 runs takes two levels, of more than 16
   three, which give nx.cpu's bits. *)
let law_levels c =
  let metal = backend () in
  let starts, ends, sums = ranges_data c in
  let rows = List.init c.lead Fun.id in
  let positions f =
    List.exists (fun r -> List.exists (f r) (List.init c.size Fun.id)) rows
  in
  cover "one level" (c.k <= 4);
  cover "two levels" (c.k > 4 && c.k <= 16);
  cover "three levels or more" (c.k > 16);
  cover "a position past every end"
    (positions (fun r p -> source starts.(r) ends.(r) p = None));
  cover "a source position outside the axis"
    (positions (fun r p ->
         match source starts.(r) ends.(r) p with
         | Some q -> q < 0 || q >= c.n
         | None -> false));
  cover "ends that do not ascend, at two levels or more"
    ((not sums) && c.k > 4
    && Array.exists
         (fun e ->
           Array.exists Fun.id
             (Array.init (c.k - 1) (fun j -> e.(j + 1) < e.(j))))
         ends);
  Nx_metal.bound_index ~runs:4 ();
  Fun.protect
    ~finally:(fun () -> Nx_metal.bound_index ())
    (fun () -> agree_bits Support.cpu metal (fun b -> ranges_on b c))

(* Positions in chunks: a range gather whose positions take at most [region]
   bytes at a time, of 64 bits where its source axis has [2^narrow] elements or
   more, else 32. *)
type chunked = { ranges : ranges; region : int; narrow : int }

let pp_chunked ppf c =
  Format.fprintf ppf "%a, regions of %d bytes, 64 bits from 2^%d" pp_ranges
    c.ranges c.region c.narrow

let chunked_gen =
  let open Gen in
  let+ ranges = ranges_gen
  and+ region = int_range 8 96
  and+ narrow = int_range 1 5 in
  { ranges; region; narrow }

(* Chunks of positions at either width give nx.cpu's bits, the positions'
   launches of a chunk running after the gather of the one before. *)
let law_chunks c =
  let metal = backend () in
  let r = c.ranges in
  let wide = r.n >= 1 lsl c.narrow in
  let width = if wide then 8 else 4 in
  let chunk = c.region / width and count = r.lead * r.size in
  cover "64-bit positions" wide;
  cover "32-bit positions" (not wide);
  cover "several chunks, 64-bit positions" (wide && count > chunk);
  cover "several chunks, 32-bit positions" ((not wide) && count > chunk);
  cover "a chunk that ends inside a row" (count > chunk && chunk mod r.size <> 0);
  Nx_metal.bound_index ~region:c.region ~narrow:c.narrow ();
  let (), ls =
    Fun.protect
      ~finally:(fun () -> Nx_metal.bound_index ())
      (fun () ->
        Plan.traced (fun () ->
            agree_bits Support.cpu metal (fun b -> ranges_on b r)))
  in
  let (D.Any dt) = r.rdt in
  let per = match units ls with u :: _ -> r.cells * D.bits dt / u | [] -> 0 in
  cover "several chunks, cells of several units" (count > chunk && per > 1);
  if count > 0 then
    equal ~msg:"positions launches" int
      ((count + chunk - 1) / chunk)
      (count_of K.Runs_positions ls)

(* A run whose source positions start at 2^32 + 1, past a source of 4 elements,
   reads zero bits through 32-bit positions, which would read 1 to 3 if the
   position were cut to 32 bits. *)
let test_far_start () =
  let metal = backend () in
  let module Kn = (val metal.kernels) in
  let ints a = Support.on metal (A.of_array D.Int64 [| 1 |] a) in
  let x =
    Support.on metal (A.of_array D.Float32 [| 4 |] [| 1.; 2.; 3.; 4. |])
  in
  let dst = A.create metal.device D.Float32 (Iarray.of_list [ 3 ]) in
  equal answer A.Done
    (Kn.gather_ranges ~dst ~starts:(ints [| 4294967297L |])
       ~ends:(ints [| 3L |]) x);
  equal (array float_exact) [| 0.; 0.; 0. |] (A.to_array (Support.host dst))

(* Assemblies of a box *)

(* A box of a [rows] by [cols] destination of [dt] from [r0], [c0], of [h] by
   [w], the fill elsewhere: its edges along the rows cut units of 16 bytes where
   [c0] or [c0 + w] times the element's bytes is not a multiple of 16. *)
type box = {
  bdt : D.any;
  rows : int;
  cols : int;
  r0 : int;
  c0 : int;
  h : int;
  w : int;
  bseed : int;
}

let pp_box ppf b =
  let (D.Any dt) = b.bdt in
  Format.fprintf ppf "%s %dx%d, box %dx%d at (%d, %d) (seed %d)" (D.name dt)
    b.rows b.cols b.h b.w b.r0 b.c0 b.bseed

let box_gen =
  let open Gen in
  let* bdt =
    of_list ~pp:pp_dtype [ D.Any D.Int8; D.Any D.Int16; D.Any D.Float32 ]
  in
  let* rows = int_range 1 9 in
  let* cols = int_range 1 40 in
  let* r0 = int_range 0 (rows - 1) in
  let* c0 = int_range 0 (cols - 1) in
  let+ h = int_range 1 (rows - r0)
  and+ w = int_range 1 (cols - c0)
  and+ bseed = int in
  { bdt; rows; cols; r0; c0; h; w; bseed }

let assemble_box (b : Support.backend) c =
  let rs = Random.State.make [| c.bseed |] in
  let (A.Any piece) =
    operand c.bdt [| c.h; c.w |] (plain 2) (fun _ -> element c.bdt rs)
  in
  let piece = Support.on b piece in
  let dt = A.dtype piece in
  let dst = A.create b.device dt (Iarray.of_list [ c.rows; c.cols ]) in
  let fill = A.get (Support.host piece) (Iarray.of_list [ 0; 0 ]) in
  let region =
    Iarray.of_list
      [
        { M.start = c.r0; count = c.h; step = 1 };
        { M.start = c.c0; count = c.w; step = 1 };
      ]
  in
  let module Kn = (val b.kernels) in
  match Kn.assemble ~fill ~dst (Iarray.of_list [ (region, piece) ]) with
  | A.Done -> [| A.Any dst |]
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* A box assembled over a fill has nx.cpu's bits: the fill writes the rows above
   and below the box, and the parts of the box's rows beside it, whose ends cut
   16-byte units where the box's edges do. *)
let law_box c =
  let metal = backend () in
  let (D.Any dt) = c.bdt in
  let bytes = D.bits dt / 8 in
  let cuts at = at * bytes mod 16 <> 0 in
  let edge = c.h < c.rows || c.w < c.cols in
  List.iter
    (fun n ->
      cover
        (Printf.sprintf "a box edge inside a unit, %d-byte elements" n)
        (bytes = n && edge && (cuts c.c0 || cuts (c.c0 + c.w))))
    [ 1; 2; 4 ];
  cover "rows above and below the box, and beside it on both sides"
    (c.r0 > 0 && c.r0 + c.h < c.rows && c.c0 > 0 && c.c0 + c.w < c.cols);
  agree_bits Support.cpu metal (fun b -> assemble_box b c)

(* Complements at rank 3 *)

(* A destination of [dims] of [cdt], a box from [lo] of [len] an axis over the
   fill and, where [later] is [Some (lo', len')], a second piece inside the box:
   the fill writes the box's complement, before and after the box along each
   axis within the box's ranges on the axes before it. *)
type complement = {
  cdt : D.any;
  dims : int array;
  lo : int array;
  len : int array;
  later : (int array * int array) option;
  cseed : int;
}

let pp_complement ppf c =
  let (D.Any dt) = c.cdt in
  let ints a = String.concat "x" (Array.to_list (Array.map string_of_int a)) in
  Format.fprintf ppf "%s %s, box %s at %s%s (seed %d)" (D.name dt) (ints c.dims)
    (ints c.len) (ints c.lo)
    (match c.later with
    | Some (lo, len) -> Printf.sprintf ", then %s at %s" (ints len) (ints lo)
    | None -> "")
    c.cseed

let complement_gen =
  let open Gen in
  let range d =
    let* lo = int_range 0 (d - 1) in
    let+ len = int_range 1 (d - lo) in
    (lo, len)
  in
  let box dims =
    let+ a = range dims.(0) and+ b = range dims.(1) and+ c = range dims.(2) in
    ([| fst a; fst b; fst c |], [| snd a; snd b; snd c |])
  in
  let* cdt =
    of_list ~pp:pp_dtype
      [ D.Any D.Bit; D.Any D.Int4; D.Any D.Int8; D.Any D.Float32 ]
  in
  let* dims =
    let+ a = int_range 1 4 and+ b = int_range 1 5 and+ c = int_range 1 40 in
    [| a; b; c |]
  in
  let* lo, len = box dims in
  let+ later = option (box len) and+ cseed = int in
  let later = Option.map (fun (l, n) -> (Array.map2 ( + ) lo l, n)) later in
  { cdt; dims; lo; len; later; cseed }

let region lo len =
  Iarray.of_list
    (List.init (Array.length lo) (fun j ->
         { M.start = lo.(j); count = len.(j); step = 1 }))

let assemble_complement (b : Support.backend) c =
  let rs = Random.State.make [| c.cseed |] in
  let piece len =
    let (A.Any p) = operand c.cdt len (plain 3) (fun _ -> element c.cdt rs) in
    A.Any (Support.on b p)
  in
  let (A.Any first) = piece c.len in
  let dt = A.dtype first in
  let dst = A.create b.device dt (Iarray.of_array c.dims) in
  let fill = A.get (Support.host first) (Iarray.of_list [ 0; 0; 0 ]) in
  let later =
    match c.later with
    | None -> []
    | Some (lo, len) -> [ (region lo len, A.expect dt (piece len)) ]
  in
  let pieces = Iarray.of_list ((region c.lo c.len, first) :: later) in
  let module Kn = (val b.kernels) in
  match Kn.assemble ~fill ~dst pieces with
  | A.Done -> [| A.Any dst |]
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* A box at rank 3 over a fill, and a later piece over it, have nx.cpu's bits:
   the fill's slabs along the third axis start inside a word where elements are
   narrower than a byte, and the later piece overwrites the box the fill
   skipped. *)
let law_complement c =
  let metal = backend () in
  let (D.Any dt) = c.cdt in
  let short j = c.len.(j) < c.dims.(j) in
  let after = c.lo.(2) + c.len.(2) in
  let first_after =
    (((c.lo.(0) * c.dims.(1)) + c.lo.(1)) * c.dims.(2)) + after
  in
  cover "slabs along the third axis within the box's ranges on the others"
    (short 2 && (short 0 || short 1));
  cover "slabs along every axis" (short 0 && short 1 && short 2);
  List.iter
    (fun n ->
      cover
        (Printf.sprintf "%d-bit elements, a slab from inside a word" n)
        (D.bits dt = n && after < c.dims.(2) && first_after * n mod 32 <> 0))
    [ 1; 4 ];
  cover "a later piece inside the box" (Option.is_some c.later);
  agree_bits Support.cpu metal (fun b -> assemble_complement b c)

(* Folds in ranges of taps *)

(* A fold into [shape], two axes, each padded by [lo] and [hi] and windowed by
   [size], [step] and [dil], under a launch budget of [budget] seconds, in 64
   bits where a position, a tap number or a step reaches [2^taps], of elements
   of [fdt] from [fseed]. *)
type tapped = {
  fshape : int array;
  flo : int array;
  fhi : int array;
  size : int array;
  step : int array;
  dil : int array;
  fdt : D.any;
  budget : float;
  taps : int;
  fseed : int;
}

let pp_tapped ppf c =
  let (D.Any dt) = c.fdt in
  let ints a = String.concat "," (Array.to_list (Array.map string_of_int a)) in
  Format.fprintf ppf
    "%s into %s, lo %s hi %s, windows %s step %s dilation %s, budget %g, 64 \
     bits from 2^%d (seed %d)"
    (D.name dt) (ints c.fshape) (ints c.flo) (ints c.fhi) (ints c.size)
    (ints c.step) (ints c.dil) c.budget c.taps c.fseed

(* Each axis's padded extent reaches its window's taps and up to 12 positions
   past them, so that several windows read an element, or falls up to 2 short of
   them, so that x has no element. *)
let tapped_gen =
  let open Gen in
  let axis most =
    let* n = int_range 1 8 in
    let* size = int_range 1 most in
    let* step = int_range 1 3 in
    let* dil = int_range 1 3 in
    let* lo = int_range (-2) 3 in
    let reach = ((size - 1) * dil) + 1 in
    let+ extra = int_range (-Int.min 2 reach) 12 in
    (n, lo, reach + extra - lo - n, size, step, dil)
  in
  let* n0, lo0, hi0, s0, st0, d0 = axis 12 in
  let* n1, lo1, hi1, s1, st1, d1 = axis 40 in
  let+ fdt =
    of_list ~pp:pp_dtype [ D.Any D.Float16; D.Any D.Int32; D.Any D.Int4 ]
  and+ budget = of_list ~pp:Format.pp_print_float [ 3e-5; 6e-5; 2e-4 ]
  and+ taps = of_list ~pp:Format.pp_print_int [ 31; 4; 1 ]
  and+ fseed = int in
  (* A tap of a fold in 64 bits weighs about twice one in 32. *)
  let budget = if taps < 31 then budget *. 2. else budget in
  {
    fshape = [| n0; n1 |];
    flo = [| lo0; lo1 |];
    fhi = [| hi0; hi1 |];
    size = [| s0; s1 |];
    step = [| st0; st1 |];
    dil = [| d0; d1 |];
    fdt;
    budget;
    taps;
    fseed;
  }

let pad_of c =
  {
    Nx_kernel.lo = Iarray.of_array c.flo;
    hi = Iarray.of_array c.fhi;
    interior = Iarray.of_list [ 0; 0 ];
    windows =
      Iarray.init 2 (fun a ->
          {
            M.axis = a;
            size = c.size.(a);
            step = c.step.(a);
            dilation = c.dil.(a);
          });
  }

(* The tap numbers that read element [i] of [c]'s result, x of [xs]. *)
let taps_of c xs i =
  let along a =
    List.filter_map
      (fun j ->
        let v = c.flo.(a) + i.(a) - (j * c.dil.(a)) in
        if v >= 0 && v mod c.step.(a) = 0 && v / c.step.(a) < xs.(a) then Some j
        else None)
      (List.init c.size.(a) Fun.id)
  in
  List.concat_map
    (fun j0 -> List.map (fun j1 -> (j0 * c.size.(1)) + j1) (along 1))
    (along 0)

(* The fold by [pad] into [shape] of [x] on [b]. *)
let fold_into (b : Support.backend) pad shape x =
  let module Kn = (val b.kernels) in
  let (A.Any x) = x in
  let x = Support.on b x in
  let dst = A.create b.device (A.dtype x) (Iarray.of_array shape) in
  match Kn.fold pad ~dst x with
  | A.Done -> [| A.Any dst |]
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

let fold_tapped b c x = fold_into b (pad_of c) c.fshape x

let fold_kernels =
  [ K.Fold_0; K.Fold_1; K.Fold_2; K.Fold_3; K.Fold_n; K.Fold_w ]

(* Whether fold launch [t] divides in 64 bits: its how word's wide bit. *)
let divides_wide (t : Plan.traced) =
  let how = Int64.to_int (Bytes.get_int64_le t.words (8 * K.fold_how)) in
  (how lsr K.how_wide) land 1 = 1

(* The tap ranges of the fold launches among [ls]: their first tap numbers, each
   once. *)
let ranges_of ls =
  List.sort_uniq compare
    (List.filter_map
       (fun (t : Plan.traced) ->
         if List.mem t.kernel fold_kernels then
           Some (Int64.to_int (Bytes.get_int64_le t.words (8 * K.fold_t0)))
         else None)
       ls)

(* The fold [c] under its budget, checked against nx.cpu's, any NaN for a NaN:
   [x]'s shape, the launches and each element's tap numbers; [None] where [c]'s
   padded shape has no fit. *)
let run_tapped c =
  let metal = backend () in
  match Nx_kernel.Shape.padded (pad_of c) c.fshape with
  | Error _ -> None
  | Ok xs ->
      let rs = Random.State.make [| c.fseed |] in
      let x = operand c.fdt xs (plain 4) (fun _ -> element c.fdt rs) in
      Nx_metal.bound_index ~taps:c.taps ();
      let (), ls =
        Fun.protect
          ~finally:(fun () -> Nx_metal.bound_index ())
          (fun () ->
            Plan.traced (fun () ->
                bounded c.budget (fun () ->
                    agree Support.cpu metal (fun b -> fold_tapped b c x))))
      in
      let ts =
        List.init (total c.fshape) (fun k -> taps_of c xs (index_of c.fshape k))
      in
      Some (xs, ls, ts)

(* The number of the range of [starts] that tap [t] lies in, from 1. *)
let range_of starts t = List.length (List.filter (fun s -> s <= t) starts)

(* A fold cut into ranges of tap numbers by a small budget gives nx.cpu's bits,
   any NaN for a NaN, in 32 or 64 bits: an element begun in one range adds the
   next range's taps onto the sum it left, one first read in a later range
   starts there. *)
let law_tapped c =
  match run_tapped c with
  | None -> assume false
  | Some (xs, ls, ts) ->
      let starts = ranges_of ls in
      let range = range_of starts in
      cover "several ranges" (List.length starts > 1);
      cover "a window of one tap" (Array.mem 1 c.size);
      (* A window whose dilation divides its step finds an element's first tap
         at once; any other scans the taps one apart up to it. *)
      let apart a =
        let rec gcd x y = if y = 0 then x else gcd y (x mod y) in
        c.step.(a) / gcd c.step.(a) c.dil.(a)
      in
      let window f = List.exists (fun a -> c.size.(a) > 1 && f a) [ 0; 1 ] in
      cover "a window whose first tap comes at once, taps apart"
        (window (fun a -> c.step.(a) mod c.dil.(a) = 0 && apart a > 1));
      cover "a window whose first tap a scan finds, taps apart"
        (window (fun a -> c.step.(a) mod c.dil.(a) <> 0 && apart a > 1));
      cover "a window longer than its axis, x of no element" (Array.mem 0 xs);
      cover "64-bit positions and taps"
        (List.exists (fun (t : Plan.traced) -> t.kernel = K.Fold_w) ls);
      cover "64-bit positions and taps, 32-bit divisions"
        (List.exists
           (fun (t : Plan.traced) ->
             t.kernel = K.Fold_w && not (divides_wide t))
           ls);
      cover "32-bit positions and taps"
        (List.exists (fun (t : Plan.traced) -> t.kernel = K.Fold_2) ls);
      cover "an element whose taps lie in several ranges"
        (List.exists
           (fun t ->
             List.length (List.sort_uniq compare (List.map range t)) > 1)
           ts);
      cover "an element first read past the first range"
        (List.exists (function t :: _ -> range t > 1 | [] -> false) ts);
      cover "a range from inside a row of the second window's taps"
        (List.exists (fun s -> s mod c.size.(1) <> 0) starts)

(* A fold of 9 elements under taps = 4: its element indices reach 2^3, so it
   holds them in 64 bits, and no padded position or step reaches 2^4, so it
   divides in 32. *)
let narrow_divisions =
  {
    fshape = [| 3; 3 |];
    flo = [| 0; 0 |];
    fhi = [| 0; 0 |];
    size = [| 2; 2 |];
    step = [| 1; 1 |];
    dil = [| 1; 1 |];
    fdt = D.Any D.Float16;
    budget = 1e-4 *. 2.;
    taps = 4;
    fseed = 1;
  }

(* A fold whose first window has one tap: drawn, one case in nine has such a
   window. *)
let one_tap =
  {
    fshape = [| 4; 5 |];
    flo = [| 0; 0 |];
    fhi = [| 0; 0 |];
    size = [| 1; 2 |];
    step = [| 1; 1 |];
    dil = [| 1; 1 |];
    fdt = D.Any D.Int32;
    budget = 1e-4;
    taps = 31;
    fseed = 3;
  }

(* A fold of windows of 2 and 3 taps, its indices in 32 bits, so that each run
   reaches the kernel of two windowed axes. *)
let two_windows = { one_tap with size = [| 2; 3 |]; fseed = 4 }

(* A fold of 2,000 elements under windows of 1,000 taps on the second axis,
   under the least budget the generator draws: the middle elements read 1,000
   taps, a chain tens of times the budget's, so the taps run in several ranges,
   and the last element is read by tap 999 alone, past the first range. *)
let late_taps = 1000

let late_first =
  {
    fshape = [| 1; 2 * late_taps |];
    flo = [| 0; 0 |];
    fhi = [| 0; 0 |];
    size = [| 1; late_taps |];
    step = [| 1; 1 |];
    dil = [| 1; 1 |];
    fdt = D.Any D.Float16;
    budget = 3e-5;
    taps = 31;
    fseed = 2;
  }

(* late_first's last element is read by its last tap alone, which lies past the
   first range: law_tapped's example reaches its cover on every seed. *)
let test_late_first () =
  match run_tapped late_first with
  | None -> fail "late_first has no padded shape"
  | Some (_, ls, ts) ->
      let last = List.nth ts (List.length ts - 1) in
      equal (list int) [ late_taps - 1 ] last;
      greater ~msg:"range" int ~than:1 (range_of (ranges_of ls) (late_taps - 1))

(* A fold of -0 elements whose last element's one tap lies in the last of
   several ranges: every element is -0, the last as its tap alone, with no +0
   before it. *)
let test_negative_zero () =
  let metal = backend () in
  let module Kn = (val metal.kernels) in
  let n = 400 and size = 300 in
  let pad =
    {
      Nx_kernel.lo = Iarray.of_list [ 0 ];
      hi = Iarray.of_list [ 0 ];
      interior = Iarray.of_list [ 0 ];
      windows = Iarray.of_list [ { M.axis = 0; size; step = 1; dilation = 1 } ];
    }
  in
  let w = n - size + 1 in
  let x =
    Support.on metal
      (A.of_array D.Float32 [| w; size |] (Array.make (w * size) (-0.)))
  in
  let dst = A.create metal.device D.Float32 (Iarray.of_list [ n ]) in
  let answer', ls =
    Plan.traced (fun () -> bounded 2e-5 (fun () -> Kn.fold pad ~dst x))
  in
  equal answer A.Done answer';
  greater ~msg:"tap ranges" int ~than:1 (List.length (ranges_of ls));
  equal (array string) (Array.make n (f32 (-0.))) (elements (A.Any dst))

(* A fold of 200 float16 elements under one window of 40 taps at step 20, whose
   dilation of 1 divides its step, so that the taps reading an element are those
   congruent to it modulo 20, cut into ranges of a few taps by a small budget.
   An element at a multiple of 20 past 40 is read by taps 0 and 20 alone, so a
   range from a tap in 21 to 39 finds none of its taps past the last: it keeps
   the sum the ranges before it left. Float16 sums in float32 hold no subnormal,
   which the GPU flushes. *)
let test_past_last () =
  let metal = backend () in
  let pad =
    {
      Nx_kernel.lo = Iarray.of_list [ 0 ];
      hi = Iarray.of_list [ 0 ];
      interior = Iarray.of_list [ 0 ];
      windows =
        Iarray.of_list [ { M.axis = 0; size = 40; step = 20; dilation = 1 } ];
    }
  in
  let shape = [| 200 |] in
  let xs = Result.get_ok (Nx_kernel.Shape.padded pad shape) in
  let rs = Random.State.make [| 3 |] in
  let x =
    operand (D.Any D.Float16) xs (plain 2) (fun _ ->
        element (D.Any D.Float16) rs)
  in
  let (), ls =
    Plan.traced (fun () ->
        bounded 1e-5 (fun () ->
            agree Support.cpu metal (fun b -> fold_into b pad shape x)))
  in
  let starts = ranges_of ls in
  let late = List.filter (fun s -> s >= 21 && s <= 39) starts in
  greater ~msg:"ranges from a tap in 21 to 39" int ~than:0 (List.length late)

(* Folds of 2 elements whose padded positions and steps pass 2^31 divide in 64
   bits and give nx.cpu's bits: by an interior of 2^31 under two windows of 2
   taps a step of 2^31 apart, and by a lo past 2^31 under one window of 2 taps a
   step apart, which leaves the first element no tap reads. *)
let test_wide () =
  let big = 1 lsl 31 in
  let fold ~lo ~interior (w : M.window) xs =
    let pad =
      {
        Nx_kernel.lo = Iarray.of_list [ lo ];
        hi = Iarray.of_list [ 0 ];
        interior = Iarray.of_list [ interior ];
        windows = Iarray.of_list [ w ];
      }
    in
    let x =
      A.Any
        (A.of_array D.Float32 xs
           (Array.init (Iarray.fold_left ( * ) 1 xs) (fun i ->
                Float.of_int (i + 1))))
    in
    let (), ls =
      Plan.traced (fun () ->
          agree_bits Support.cpu (backend ()) (fun b ->
              fold_into b pad [| 2 |] x))
    in
    let wide (t : Plan.traced) =
      List.mem t.kernel fold_kernels && divides_wide t
    in
    greater ~msg:"wide fold launches" int ~than:0
      (List.length (List.filter wide ls))
  in
  fold ~lo:0 ~interior:big
    { M.axis = 0; size = 2; step = big; dilation = 1 }
    [| 2; 2 |];
  fold ~lo:(big + 5) ~interior:0
    { M.axis = 0; size = 2; step = big + 6; dilation = big + 6 }
    [| 1; 2 |]

(* Folds of windows apart *)

(* A fold axis: [dst]'s extent, the padding, and a window of [size] taps [dil]
   apart at [step], or none. *)
type apart_axis = {
  n : int;
  alo : int;
  ahi : int;
  win : (int * int * int) option;
}

(* A fold whose windows do not overlap, of elements of [adt] from [aseed]
   through [aview]. *)
type apart = {
  aaxes : apart_axis array;
  adt : D.any;
  aview : view;
  aseed : int;
}

let pp_apart ppf c =
  let (D.Any dt) = c.adt in
  let axis ppf a =
    Format.fprintf ppf "%d lo %d hi %d" a.n a.alo a.ahi;
    Option.iter
      (fun (size, dil, step) ->
        Format.fprintf ppf " window %d dilation %d step %d" size dil step)
      a.win
  in
  Format.fprintf ppf "%s into [%a] %s (seed %d)" (D.name dt)
    (Format.pp_print_seq ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ") axis)
    (Array.to_seq c.aaxes)
    (String.concat ", " (view_names c.aview))
    c.aseed

(* An axis whose windows tile their positions, reach within their step with or
   without gaps, or one of no window. A tiling axis's padding leaves no element
   of [dst] unread, but where its [dst] is too short for it; the others' cut
   windows at either end and leave elements no window reads, or no window at
   all. *)
let apart_axis =
  let open Gen in
  let tile =
    let* size = int_range 1 4 in
    let* nwin = int_range 1 4 in
    let* alo = int_range 0 3 in
    let+ ahi = int_range 0 3 in
    let n = Int.max 1 ((nwin * size) - alo - ahi) in
    { n; alo; ahi = (nwin * size) - alo - n; win = Some (size, 1, size) }
  in
  let spaced =
    let* size = int_range 1 4 in
    let* dil = int_range 1 3 in
    let reach = ((size - 1) * dil) + 1 in
    let* step = int_range reach (reach + 2) in
    let* nw = int_range 0 4 in
    let* extra = int_range 0 (step - 1) in
    let padded = (nw * step) + extra in
    let* alo = int_range (-2) 3 in
    let+ n = int_range 1 (padded + 3) in
    { n; alo; ahi = padded - alo - n; win = Some (size, dil, step) }
  in
  let plain =
    let* n = int_range 2 5 in
    let* alo = int_range (-1) 1 in
    let+ ahi = int_range (-1) 1 in
    { n; alo; ahi; win = None }
  in
  frequency [ (2, tile); (2, spaced); (1, plain) ]

let apart_gen =
  let open Gen in
  let* r = int_range 2 3 in
  let* aaxes = map Array.of_list (list ~size:(constant r) apart_axis) in
  let w =
    Array.fold_left
      (fun w a -> w + Option.fold ~none:0 ~some:(fun _ -> 1) a.win)
      0 aaxes
  in
  let+ adt =
    of_list ~pp:pp_dtype [ D.Any D.Float32; D.Any D.Bfloat16; D.Any D.Int4 ]
  and+ aview = view_of (r + w)
  and+ aseed = int in
  { aaxes; adt; aview; aseed }

let apart_pad c =
  let r = Array.length c.aaxes in
  let windows =
    List.filter_map
      (fun a ->
        Option.map
          (fun (size, dilation, step) -> { M.axis = a; size; step; dilation })
          c.aaxes.(a).win)
      (List.init r Fun.id)
  in
  {
    Nx_kernel.lo = Iarray.init r (fun a -> c.aaxes.(a).alo);
    hi = Iarray.init r (fun a -> c.aaxes.(a).ahi);
    interior = Iarray.init r (fun _ -> 0);
    windows = Iarray.of_list windows;
  }

(* The indices of [dst] along axis [a] that the taps of [nwin] windows read, as
   (window, tap, index), in order of window then tap. *)
let reads a nwin =
  let size, dil, step = Option.value a.win ~default:(1, 1, 1) in
  List.concat_map
    (fun q ->
      List.filter_map
        (fun j ->
          let i = (q * step) + (j * dil) - a.alo in
          if i >= 0 && i < a.n then Some (q, j, i) else None)
        (List.init size Fun.id))
    (List.init nwin Fun.id)

(* A fold of [6; 3] float32 under windows of one tap at step 2 along its first
   axis: rows 0, 2 and 4 are a tap's each, the others +0. *)
let one_tap_apart =
  {
    aaxes =
      [|
        { n = 6; alo = 0; ahi = 0; win = Some (1, 1, 2) };
        { n = 3; alo = 0; ahi = 0; win = None };
      |];
    adt = D.Any D.Float32;
    aview = plain 3;
    aseed = 1;
  }

(* A fold whose windows do not overlap has nx.cpu's bits, each element of [dst]
   one element of [x] or [+0], and runs no fold kernel. A NaN agrees with any
   NaN: a fold's element is a sum, whose NaN has an unspecified payload (Prog's
   NaN rule), and nx.cpu's quiets a signalling NaN of one term that the moves
   keep. *)
let law_apart c =
  let metal = backend () in
  let shape = Array.map (fun a -> a.n) c.aaxes in
  let pad = apart_pad c in
  match Nx_kernel.Shape.padded pad shape with
  | Error _ -> assume false
  | Ok xs ->
      let rs = Random.State.make [| c.aseed |] in
      let x = operand c.adt xs c.aview (fun _ -> element c.adt rs) in
      let (), ls =
        Plan.traced (fun () ->
            agree Support.cpu metal (fun b -> fold_into b pad shape x))
      in
      let folds =
        List.filter (fun (t : Plan.traced) -> List.mem t.kernel fold_kernels) ls
      in
      equal ~msg:"fold launches" int 0 (List.length folds);
      let r = Array.length shape in
      let axes =
        List.init r (fun a -> (c.aaxes.(a), reads c.aaxes.(a) xs.(a)))
      in
      let windowed f =
        List.exists
          (fun (a, rd) ->
            match a.win with
            | Some (size, _, _) when size > 1 -> f a size rd
            | _ -> false)
          axes
      in
      let tiles =
        List.for_all
          (fun (a, _) ->
            match a.win with
            | None -> true
            | Some (size, dil, step) -> step = size && (dil = 1 || size = 1))
          axes
      in
      let fills = fills ls in
      cover "no fill" (fills = 0);
      cover "a fill around a box" (fills > 0 && tiles);
      cover "a fill of every element" (fills > 0 && not tiles);
      cover "an element before every window"
        (List.exists (fun (a, _) -> a.alo < 0) axes);
      cover "a window cut at the first element"
        (windowed (fun _ _ rd ->
             match rd with (_, j, _) :: _ -> j > 0 | [] -> false));
      cover "a window cut at the last element"
        (windowed (fun _ size rd ->
             match List.rev rd with
             | (_, j, _) :: _ -> j < size - 1
             | [] -> false));
      cover "a window of one tap at step 2 or more"
        (List.exists
           (fun (a, _) ->
             match a.win with Some (1, _, step) -> step >= 2 | _ -> false)
           axes);
      cover "a dilated window within its step"
        (windowed (fun a _ _ ->
             match a.win with Some (_, dil, _) -> dil > 1 | None -> false));
      cover "gaps between windows"
        (List.exists
           (fun (a, _) ->
             match a.win with
             | Some (size, dil, step) -> step > ((size - 1) * dil) + 1
             | None -> false)
           axes);
      cover "a 4-bit dtype" (c.adt = D.Any D.Int4);
      cover "x of no element" (total xs = 0);
      cover "x permuted" (not (identity c.aview.perm))

(* A fold of 16x16 patches at step 16 of float32 moves 16 bytes a thread, in one
   launch, with no fill, and has nx.cpu's bits, any NaN for a NaN. *)
let test_patches () =
  let metal = backend () in
  let pad =
    {
      Nx_kernel.lo = Iarray.init 4 (fun _ -> 0);
      hi = Iarray.init 4 (fun _ -> 0);
      interior = Iarray.init 4 (fun _ -> 0);
      windows =
        Iarray.of_list
          [
            { M.axis = 2; size = 16; step = 16; dilation = 1 };
            { M.axis = 3; size = 16; step = 16; dilation = 1 };
          ];
    }
  in
  let shape = [| 2; 3; 32; 48 |] in
  let xs = Result.get_ok (Nx_kernel.Shape.padded pad shape) in
  let rs = Random.State.make [| 1 |] in
  let x =
    operand (D.Any D.Float32) xs (plain 6) (fun _ ->
        element (D.Any D.Float32) rs)
  in
  let (), ls =
    Plan.traced (fun () ->
        agree Support.cpu metal (fun b -> fold_into b pad shape x))
  in
  equal (list int) [ 128 ] (units ls)

(* A fold of 2-tap windows at step 2 along 4 axes of 4 elements, padded by one
   at each end, cuts the first and last window of each axis: 3 boxes an axis, 81
   moves. Under a launch budget their dispatches pass, the fold kernel computes
   it. Both give nx.cpu's bits, any NaN for a NaN. *)
let test_many_cuts () =
  let metal = backend () in
  let r = 4 in
  let pad =
    {
      Nx_kernel.lo = Iarray.init r (fun _ -> 1);
      hi = Iarray.init r (fun _ -> 1);
      interior = Iarray.init r (fun _ -> 0);
      windows =
        Iarray.init r (fun a ->
            { M.axis = a; size = 2; step = 2; dilation = 1 });
    }
  in
  let shape = Array.make r 4 in
  let xs = Result.get_ok (Nx_kernel.Shape.padded pad shape) in
  let rs = Random.State.make [| 1 |] in
  let x =
    operand (D.Any D.Float32) xs
      (plain (2 * r))
      (fun _ -> element (D.Any D.Float32) rs)
  in
  let run () =
    snd
      (Plan.traced (fun () ->
           agree Support.cpu metal (fun b -> fold_into b pad shape x)))
  in
  let folds ls =
    List.length
      (List.filter (fun (t : Plan.traced) -> List.mem t.kernel fold_kernels) ls)
  in
  let ls = run () in
  equal ~msg:"moves" int 81 (moves ls);
  equal ~msg:"fold launches" int 0 (folds ls);
  let ls = bounded 1e-4 run in
  equal ~msg:"moves" int 0 (moves ls);
  greater ~msg:"fold launches" int ~than:0 (folds ls)

(* Cuts *)

(* Under a budget just past the longest chain of one unit's launch, a range
   gather of 4 rows of 200,000 runs at levels of 4 cuts its runs, carries and
   positions, and an assembly of a box of 16 MiB its fill and its piece, into
   launches, which give nx.cpu's bits. *)
let test_cuts () =
  let metal = backend () in
  let ranges =
    {
      lead = 4;
      k = 200_000;
      n = 100;
      cells = 1;
      size = 20_000;
      rdt = D.Any D.Float32;
      rseed = 1;
    }
  and box =
    {
      bdt = D.Any D.Float32;
      rows = 1024;
      cols = 4096;
      r0 = 3;
      c0 = 5;
      h = 1000;
      w = 4000;
      bseed = 1;
    }
  in
  Nx_metal.bound_index ~runs:4 ();
  let (), ls =
    Fun.protect
      ~finally:(fun () -> Nx_metal.bound_index ())
      (fun () ->
        Plan.traced (fun () ->
            bounded 2e-5 (fun () ->
                agree_bits Support.cpu metal (fun b -> ranges_on b ranges);
                agree_bits Support.cpu metal (fun b -> assemble_box b box))))
  in
  List.iter
    (fun (what, k) -> greater ~msg:what int ~than:1 (count_of k ls))
    [
      ("runs launches", K.Runs);
      ("carry launches", K.Runs_carry);
      ("positions launches", K.Runs_positions);
      ("move launches", K.Move);
    ];
  greater ~msg:"fill launches" int ~than:1 (fills ls)

(* Declines *)

(* Under a budget no unit's launch fits, every call declines. *)
let test_declines () =
  let metal = backend () in
  let module Kn = (val metal.kernels) in
  let x =
    Support.on metal (A.of_array D.Float32 [| 4 |] [| 1.; 2.; 3.; 4. |])
  in
  let idx =
    Support.on metal (A.of_array D.Int64 [| 4 |] [| 3L; 2L; 1L; 0L |])
  in
  let ints a = Support.on metal (A.of_array D.Int64 [| 2 |] a) in
  let dst () = A.create metal.device D.Float32 (Iarray.of_list [ 4 ]) in
  let region = Iarray.of_list [ { M.start = 1; count = 2; step = 1 } ] in
  let piece = Support.on metal (A.of_array D.Float32 [| 2 |] [| 5.; 6. |]) in
  bounded 1e-15 (fun () ->
      equal answer A.Declined (Kn.gather ~axis:0 ~dst:(dst ()) idx x);
      equal answer A.Declined
        (Kn.gather_ranges ~dst:(dst ())
           ~starts:(ints [| 0L; 2L |])
           ~ends:(ints [| 2L; 4L |])
           x);
      equal answer A.Declined
        (Kn.assemble ~fill:0. ~dst:(dst ())
           (Iarray.of_list [ (region, piece) ])))

let () =
  exit
    (run "nx_metal.moves"
       [
         group "gathers"
           [
             prop "a gather has nx.cpu's bits" gathers law_gather;
             prop ~examples:[ cut_example ]
               "a gather cut into launches has nx.cpu's bits" gathers law_cut;
             prop ~examples:[ cut_example ]
               "a gather cut into launches, a part each, has nx.cpu's bits"
               gathers (fun c -> parted (fun () -> law_cut c));
             cases "a take of rows moves rows in the widest unit"
               ~id:(fun (D.Any dt, w, u) ->
                 Printf.sprintf "%s rows of %d, units of %d bits" (D.name dt) w
                   u)
               row_cases rows;
             test "a gather of one byte writes one byte" test_one_byte;
             test "a gather from a source on 2 bytes moves elements"
               test_unaligned_source;
             test
               "a gather into int4 elements keeps the elements beside them in \
                their words"
               test_gather_shared;
             test
               "a gather beside its source in one word keeps the source's \
                elements"
               test_gather_beside;
           ];
         group "assemblies"
           [
             test
               "an assembly into int4 elements keeps the elements beside them \
                in their words"
               test_assemble_shared;
             prop "a box over a fill has nx.cpu's bits"
               (Gen.with_pp pp_box box_gen)
               law_box;
             prop "a box at rank 3 and a piece over it have nx.cpu's bits"
               (Gen.with_pp pp_complement complement_gen)
               law_complement;
             prop
               "a box at rank 3 and a piece over it, a part a launch, have \
                nx.cpu's bits" (Gen.with_pp pp_complement complement_gen)
               (fun c -> parted (fun () -> law_complement c));
           ];
         group "range gathers"
           [
             prop "runs in levels give nx.cpu's bits"
               (Gen.with_pp pp_ranges ranges_gen)
               law_levels;
             prop "runs in levels, a part a launch, give nx.cpu's bits"
               (Gen.with_pp pp_ranges ranges_gen) (fun c ->
                 parted (fun () -> law_levels c));
             prop "positions in chunks give nx.cpu's bits"
               (Gen.with_pp pp_chunked chunked_gen)
               law_chunks;
             prop "positions in chunks, a part a launch, give nx.cpu's bits"
               (Gen.with_pp pp_chunked chunked_gen) (fun c ->
                 parted (fun () -> law_chunks c));
             test "a source position past 2^32 reads zero bits" test_far_start;
           ];
         group "folds"
           [
             prop "a fold in ranges of taps agrees with nx.cpu"
               ~examples:[ narrow_divisions; one_tap; two_windows; late_first ]
               (Gen.with_pp pp_tapped tapped_gen)
               law_tapped;
             prop
               "a fold in ranges of taps, a part a launch, agrees with nx.cpu"
               ~examples:[ narrow_divisions; one_tap; two_windows; late_first ]
               (Gen.with_pp pp_tapped tapped_gen) (fun c ->
                 parted (fun () -> law_tapped c));
             test "a fold's element first read past the first range agrees"
               test_late_first;
             test "a fold of -0 taps in ranges gives -0" test_negative_zero;
             test "a fold past 2^31 divides in 64 bits" test_wide;
             test
               "a fold's range from a tap past every tap reading an element \
                keeps its sum"
               test_past_last;
             prop "a fold of windows apart moves elements with nx.cpu's bits"
               ~examples:[ one_tap_apart ]
               (Gen.with_pp pp_apart apart_gen)
               law_apart;
             test "a fold of 16x16 patches moves 16 bytes a thread" test_patches;
             test
               "a fold of windows apart cut along many axes moves its boxes, \
                or runs the fold kernel past a launch's dispatches"
               test_many_cuts;
           ];
         test "launches cut by the budget give nx.cpu's bits" test_cuts;
         test "a call no unit's launch fits declines" test_declines;
       ])
